"""Frozen-model composition for one real P1 prospective issue.

The network boundary is implemented in :mod:`seismoflux.p1_b0_r30.production`.
This module accepts an already sealed, count-first ComCat acquisition and joins
it to the exact frozen local history.  It then calls the same D1 spatial KDE and
complete-cell alarm selection used by the accepted P1-0C scientific path.

No function in this module opens a path, accesses the network, reads a future
target, or appends the public record ledger.  Those side effects belong to the
small operational runner and remain independently auditable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pyproj import Geod

from seismoflux.d1_replay.spatial import (
    D1SpatialDomain,
    build_causal_background_components,
    build_d1_spatial_domain_from_bytes,
)
from seismoflux.data.common import canonical_json_bytes
from seismoflux.data.parquet import schema_sha256, table_content_sha256
from seismoflux.p1_b0_r30.preflight import (
    EXPECTED_COMMON_MC,
    EXPECTED_OPERATIONAL_CELL_COUNT,
    EXPECTED_STUDY_AREA_SHA256,
    EXPECTED_SUPPORT_ID,
    EXPECTED_SUPPORT_MANIFEST_SHA256,
    AlarmSelection,
    SupportWaterLevel,
    parse_support_water_level,
    select_complete_alarm_prefix,
)
from seismoflux.p1_b0_r30.production import (
    ComCatEvent,
    ComCatIssueInputAcquisition,
    P1IssueSchedule,
)
from seismoflux.stage2s.catalog import (
    ArrowFieldContract,
    CatalogByteContract,
    Stage2SEarthquakeCatalog,
    parse_catalog_bytes,
    parse_frozen_catalog_bytes,
)

FloatArray: TypeAlias = NDArray[np.float64]

P1_ALPHA: Final = 0.25
P1_AREA_CAP_KM2: Final = 600_000.0
P1_BANDWIDTH_KM: Final = 75.0
P1_LOCAL_CUTOFF_UTC: Final = datetime(2026, 7, 9, 4, 25, 56, tzinfo=UTC)
P1_CATALOG_START_UTC: Final = datetime(1970, 1, 1, tzinfo=UTC)
P1_SOURCE_BOUNDARY_MANIFEST_SHA256: Final = (
    "b158b4945e0089eff179847fc484dfad47ee5342afb10f5275236108c83fafa8"
)
P1_MODEL_MANIFEST_SHA256: Final = "128cbea397e22eb6ad72d5657abb8f720d3d7866349173d7d0ca31615b3c3316"

_WGS84: Final = Geod(ellps="WGS84")
_UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _epoch_us(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    delta = normalized - _UTC_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _parquet_bytes(table: pa.Table) -> bytes:
    """Return a deterministic, uncompressed Parquet preimage for one table."""

    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        data_page_version="1.0",
    )
    payload: bytes = sink.getvalue().to_pybytes()
    return payload


@dataclass(frozen=True, slots=True)
class SourceBoundaryMatch:
    """One deterministic local-anchor/ComCat cutover match."""

    local_event_id: str
    comcat_event_id: str
    absolute_time_difference_seconds: float
    WGS84_distance_km: float
    absolute_magnitude_difference: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "local_event_id": self.local_event_id,
            "comcat_event_id": self.comcat_event_id,
            "absolute_time_difference_seconds": self.absolute_time_difference_seconds,
            "WGS84_distance_km": self.WGS84_distance_km,
            "absolute_magnitude_difference": self.absolute_magnitude_difference,
        }


def deduplicate_local_comcat_boundary(
    local_catalog: Stage2SEarthquakeCatalog,
    comcat_events: tuple[ComCatEvent, ...],
) -> tuple[tuple[ComCatEvent, ...], tuple[SourceBoundaryMatch, ...]]:
    """Apply the frozen deterministic one-to-one source-cutover matching rule."""

    if not isinstance(local_catalog, Stage2SEarthquakeCatalog):
        raise TypeError("local_catalog must be a verified Stage2SEarthquakeCatalog")
    if any(event.origin_time_utc <= P1_LOCAL_CUTOFF_UTC for event in comcat_events):
        raise ValueError("ComCat model input must be strictly after the local cutoff")

    cutoff_us = _epoch_us(P1_LOCAL_CUTOFF_UTC)
    candidate_local_indices = np.flatnonzero(
        local_catalog.origin_time_us >= cutoff_us - 300 * 1_000_000
    )
    candidates: list[tuple[float, float, float, bytes, bytes, int, int]] = []
    for candidate_local_index in candidate_local_indices:
        local_index = int(candidate_local_index)
        local_origin_us = int(local_catalog.origin_time_us[local_index])
        for comcat_index, event in enumerate(comcat_events):
            time_difference = abs(_epoch_us(event.origin_time_utc) - local_origin_us) / 1_000_000.0
            if time_difference > 300.0:
                continue
            _, _, distance_m = _WGS84.inv(
                float(local_catalog.longitude[local_index]),
                float(local_catalog.latitude[local_index]),
                event.longitude,
                event.latitude,
            )
            distance_km = abs(float(distance_m)) / 1_000.0
            magnitude_difference = abs(
                float(local_catalog.magnitude[local_index]) - event.magnitude
            )
            if distance_km <= 50.0 and magnitude_difference <= 0.5:
                candidates.append(
                    (
                        time_difference,
                        distance_km,
                        magnitude_difference,
                        event.event_id.encode("utf-8"),
                        local_catalog.event_ids[local_index].encode("utf-8"),
                        local_index,
                        comcat_index,
                    )
                )

    matched_local: set[int] = set()
    matched_comcat: set[int] = set()
    matches: list[SourceBoundaryMatch] = []
    for (
        time_difference,
        distance_km,
        magnitude_difference,
        _,
        _,
        local_index,
        comcat_index,
    ) in sorted(candidates):
        if local_index in matched_local or comcat_index in matched_comcat:
            continue
        matched_local.add(local_index)
        matched_comcat.add(comcat_index)
        matches.append(
            SourceBoundaryMatch(
                local_event_id=local_catalog.event_ids[local_index],
                comcat_event_id=comcat_events[comcat_index].event_id,
                absolute_time_difference_seconds=time_difference,
                WGS84_distance_km=distance_km,
                absolute_magnitude_difference=magnitude_difference,
            )
        )

    retained = tuple(
        event for index, event in enumerate(comcat_events) if index not in matched_comcat
    )
    return retained, tuple(matches)


def _comcat_table(
    events: tuple[ComCatEvent, ...],
    *,
    domain: D1SpatialDomain,
    schema: pa.Schema,
) -> pa.Table:
    rows: list[dict[str, object]] = []
    for event in events:
        if event.provider_updated_at_utc > event.observed_at_utc:
            raise ValueError("ComCat provider revision cannot postdate the sealed observation")
        inside_index = domain.locator.locate_lonlat(event.longitude, event.latitude)
        rows.append(
            {
                "event_id": f"comcat:{event.event_id}",
                "origin_time_utc": event.origin_time_utc,
                # ComCat GeoJSON exposes the provider's update time but not a
                # historical first-seen feed.  Using update time is conservative:
                # an update after Q is excluded by the causal selector.
                "available_at": event.provider_updated_at_utc,
                "origin_time_local": event.origin_time_utc,
                "longitude": event.longitude,
                "latitude": event.latitude,
                "depth_km": event.depth_km,
                "magnitude": event.magnitude,
                "magnitude_type": None,
                "place": None,
                "catalog_sources": [event.source_id],
                "inside_study_area": inside_index is not None,
                "dedup_confidence": "comcat_associated_id_component",
                "anchor_source_record_id": event.event_id,
                "quality_flags": ["available_at_conservative_provider_updated_at"],
            }
        )
    return pa.Table.from_pylist(rows, schema=schema)


@dataclass(frozen=True, slots=True)
class ProductionCatalogArtifact:
    """Exact derived local+ComCat model-input catalogue and cutover audit."""

    catalog: Stage2SEarthquakeCatalog
    parquet_bytes: bytes
    retained_comcat_events: tuple[ComCatEvent, ...]
    cutover_matches: tuple[SourceBoundaryMatch, ...]

    @property
    def parquet_sha256(self) -> str:
        return _sha256(self.parquet_bytes)

    def audit_mapping(self) -> dict[str, object]:
        return {
            "combined_catalog_sha256": self.parquet_sha256,
            "combined_catalog_row_count": self.catalog.row_count,
            "retained_comcat_event_count": len(self.retained_comcat_events),
            "cutover_match_count": len(self.cutover_matches),
            "cutover_matches": [item.as_mapping() for item in self.cutover_matches],
            "available_at_semantics": "ComCat_provider_updated_at_conservative_lte_Q",
        }


def build_production_catalog(
    *,
    local_catalog_bytes: bytes,
    study_area_bytes: bytes,
    acquisition: ComCatIssueInputAcquisition,
) -> tuple[ProductionCatalogArtifact, D1SpatialDomain]:
    """Derive the one shared causal B0/R30 model-input catalogue."""

    if acquisition.status != "available" or acquisition.query_snapshot is None:
        raise ValueError("an unavailable count-first acquisition cannot produce a forecast")
    if _sha256(study_area_bytes) != EXPECTED_STUDY_AREA_SHA256:
        raise ValueError("study-area SHA-256 differs from the frozen P1 domain")
    local = parse_frozen_catalog_bytes(local_catalog_bytes)
    domain = build_d1_spatial_domain_from_bytes(study_area_bytes)
    if domain.operational_grid.cell_count != EXPECTED_OPERATIONAL_CELL_COUNT:
        raise ValueError("operational grid differs from the frozen P1 domain")

    snapshot = acquisition.query_snapshot
    schedule_q = snapshot.query_end_inclusive_utc
    if any(event.origin_time_utc > schedule_q for event in snapshot.events):
        raise ValueError("source snapshot contains an origin after Q")
    if any(event.provider_updated_at_utc > schedule_q for event in snapshot.events):
        raise ValueError("source snapshot exposes a provider revision unavailable at Q")

    retained, matches = deduplicate_local_comcat_boundary(local, snapshot.events)
    comcat_table = _comcat_table(retained, domain=domain, schema=local.table.schema)
    combined = pa.concat_tables([local.table, comcat_table], promote_options="none")
    order = pc.sort_indices(
        combined,
        sort_keys=[("origin_time_utc", "ascending"), ("event_id", "ascending")],
    )
    combined = pc.take(combined, order)
    if not isinstance(combined, pa.Table):
        raise TypeError("sorted combined catalogue must remain a PyArrow table")
    combined.validate(full=True)
    payload = _parquet_bytes(combined)
    with pa.BufferReader(payload) as reader:
        persisted = pq.read_table(reader, use_threads=False)
    persisted.validate(full=True)
    contract = CatalogByteContract(
        row_count=persisted.num_rows,
        file_sha256=_sha256(payload),
        content_sha256=table_content_sha256(persisted),
        schema_sha256=schema_sha256(persisted.schema),
        fields=tuple(ArrowFieldContract.from_arrow(field) for field in persisted.schema),
    )
    parsed = parse_catalog_bytes(payload, contract=contract)
    return (
        ProductionCatalogArtifact(
            catalog=parsed,
            parquet_bytes=payload,
            retained_comcat_events=retained,
            cutover_matches=matches,
        ),
        domain,
    )


@dataclass(frozen=True, slots=True)
class ProductionForecastBundle:
    """One future-blind, same-snapshot B0 versus B0_R30 forecast bundle."""

    schedule: P1IssueSchedule
    acquisition: ComCatIssueInputAcquisition
    catalog_artifact: ProductionCatalogArtifact
    support: SupportWaterLevel
    domain: D1SpatialDomain
    b0_mass: FloatArray
    challenger_mass: FloatArray
    b0_source_count: int
    recent_source_count: int
    b0_alarm: AlarmSelection
    challenger_alarm: AlarmSelection

    def __post_init__(self) -> None:
        if type(self.b0_source_count) is not int or self.b0_source_count < 1:
            raise ValueError("B0_source_count must be a positive integer")
        if type(self.recent_source_count) is not int or not (
            0 <= self.recent_source_count <= self.b0_source_count
        ):
            raise ValueError("R30_source_count must be an integer in [0, B0_source_count]")

    @property
    def source_snapshot_sha256(self) -> str:
        return _sha256(canonical_json_bytes(self.acquisition.as_mapping()))

    @property
    def source_request_sha256(self) -> str:
        query = self.acquisition.query_snapshot
        if query is None:
            raise ValueError("available acquisition lost its query snapshot")
        return _sha256(
            canonical_json_bytes(
                {
                    "count_request_url": self.acquisition.count_snapshot.request_url,
                    "query_request_url": query.request_url,
                }
            )
        )

    def forecast_mapping(self, *, code_commit: str) -> dict[str, object]:
        if len(code_commit) != 40 or any(
            character not in "0123456789abcdef" for character in code_commit
        ):
            raise ValueError("code_commit must be a lowercase 40-character Git SHA")
        grid = self.domain.operational_grid
        cells = [
            {
                "cell_id": cell_id,
                "row": int(row),
                "column": int(column),
                "area_km2": float(area),
                "support_status": self.support.status_for_25km_cell(
                    row=int(row), column=int(column)
                ),
            }
            for cell_id, row, column, area in zip(
                grid.cell_ids,
                grid.rows,
                grid.columns,
                grid.clipped_area_km2,
                strict=True,
            )
        ]
        return {
            "issue_id": self.schedule.issue_id,
            "scheduled_issue_time_utc": _utc_text(self.schedule.scheduled_issue_time_utc),
            "query_cutoff_utc": _utc_text(self.schedule.query_cutoff_utc),
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "source_request_sha256": self.source_request_sha256,
            "support_manifest_sha256": self.support.manifest_sha256,
            "code_commit": code_commit,
            "B0_source_count": self.b0_source_count,
            "R30_source_count": self.recent_source_count,
            "grid": {"cell_size_km": 25.0, "cells": cells},
            "models": {
                "B0": {
                    "normalized_cell_mass": self.b0_mass.tolist(),
                    "alarm_cell_ids": list(self.b0_alarm.selected_cell_ids),
                    "actual_alarm_area_km2": self.b0_alarm.actual_area_km2,
                    "next_complete_cell_area_km2": self.b0_alarm.next_complete_cell_area_km2,
                },
                "B0_R30": {
                    "normalized_cell_mass": self.challenger_mass.tolist(),
                    "alarm_cell_ids": list(self.challenger_alarm.selected_cell_ids),
                    "actual_alarm_area_km2": self.challenger_alarm.actual_area_km2,
                    "next_complete_cell_area_km2": (
                        self.challenger_alarm.next_complete_cell_area_km2
                    ),
                },
            },
        }

    def record_forecasts(self) -> list[dict[str, object]]:
        grid = self.domain.operational_grid
        forecasts: list[dict[str, object]] = []
        for model_id, mass, alarm in (
            ("B0", self.b0_mass, self.b0_alarm),
            ("B0_R30", self.challenger_mass, self.challenger_alarm),
        ):
            grid_sha = _sha256(
                canonical_json_bytes(
                    {
                        "domain": "seismoflux.p1.relative-intensity-grid.v1",
                        "model_id": model_id,
                        "cell_ids": list(grid.cell_ids),
                        "normalized_cell_mass_hex": [float(value).hex() for value in mass],
                    }
                )
            )
            forecasts.append(
                {
                    "model_id": model_id,
                    "relative_intensity_grid_sha256": grid_sha,
                    "alarm_mask_sha256": alarm.selected_mask_sha256,
                    "alarm_ranking_sha256": alarm.ranking_sha256,
                    "actual_alarm_area_km2": alarm.actual_area_km2,
                }
            )
        return forecasts

    def audit_mapping(self) -> dict[str, object]:
        query = self.acquisition.query_snapshot
        if query is None:
            raise ValueError("available acquisition lost its query snapshot")
        return {
            "issue_id": self.schedule.issue_id,
            "scheduled_issue_time_utc": _utc_text(self.schedule.scheduled_issue_time_utc),
            "query_cutoff_utc": _utc_text(self.schedule.query_cutoff_utc),
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "count_snapshot_sha256": self.acquisition.count_snapshot.snapshot_sha256,
            "query_snapshot_sha256": query.snapshot_sha256,
            "catalog": self.catalog_artifact.audit_mapping(),
            "B0_source_count": self.b0_source_count,
            "R30_source_count": self.recent_source_count,
            "challenger_formula": "0.75*B0+0.25*R30",
            "bandwidth_km": P1_BANDWIDTH_KM,
            "B0_actual_area_km2": self.b0_alarm.actual_area_km2,
            "B0_R30_actual_area_km2": self.challenger_alarm.actual_area_km2,
            "actual_area_difference_km2": (
                self.b0_alarm.actual_area_km2 - self.challenger_alarm.actual_area_km2
            ),
            "future_outcomes_absent": True,
            "value_semantics": "relative_intensity_not_absolute_probability",
        }


def build_production_forecast(
    *,
    schedule: P1IssueSchedule,
    acquisition: ComCatIssueInputAcquisition,
    local_catalog_bytes: bytes,
    study_area_bytes: bytes,
    support_manifest_bytes: bytes,
) -> ProductionForecastBundle:
    """Build one frozen real P1 forecast without opening a target or path."""

    if acquisition.status != "available" or acquisition.query_snapshot is None:
        raise ValueError("source snapshot is unavailable; this issue must not generate a forecast")
    if acquisition.query_snapshot.query_end_inclusive_utc != schedule.query_cutoff_utc:
        raise ValueError("ComCat snapshot is not bound to this issue Q")
    if acquisition.query_snapshot.fetch_completed_at_utc >= schedule.scheduled_issue_time_utc:
        raise ValueError("ComCat source fetch did not complete before T")

    support = parse_support_water_level(support_manifest_bytes)
    if support.manifest_sha256 != EXPECTED_SUPPORT_MANIFEST_SHA256:
        raise ValueError("support manifest differs from the frozen P1 identity")
    if support.support_id != EXPECTED_SUPPORT_ID or support.common_mc != EXPECTED_COMMON_MC:
        raise ValueError("support water level differs from the frozen P1 contract")
    artifact, domain = build_production_catalog(
        local_catalog_bytes=local_catalog_bytes,
        study_area_bytes=study_area_bytes,
        acquisition=acquisition,
    )
    background = build_causal_background_components(
        artifact.catalog,
        schedule.query_cutoff_utc,
        domain,
    )
    b0_mass = background.b0_mass_25km
    challenger_mass = background.mass_for_alpha(P1_ALPHA)
    grid = domain.operational_grid
    b0_alarm = select_complete_alarm_prefix(
        b0_mass,
        grid,
        model_id="B0",
        area_cap_km2=P1_AREA_CAP_KM2,
    )
    challenger_alarm = select_complete_alarm_prefix(
        challenger_mass,
        grid,
        model_id="B0_R30",
        area_cap_km2=b0_alarm.actual_area_km2,
    )
    area_difference = b0_alarm.actual_area_km2 - challenger_alarm.actual_area_km2
    next_area = challenger_alarm.next_complete_cell_area_km2
    if area_difference < 0.0 or area_difference >= 625.0:
        raise ValueError("paired alarm areas violate the frozen fairness bound")
    if next_area is not None and area_difference >= next_area:
        raise ValueError("challenger left enough area for its next complete cell")
    if background.audit.recent_30d_source_count == 0 and not np.array_equal(
        b0_mass, challenger_mass
    ):
        raise ValueError("empty R30 must fall back bitwise exactly to B0")
    return ProductionForecastBundle(
        schedule=schedule,
        acquisition=acquisition,
        catalog_artifact=artifact,
        support=support,
        domain=domain,
        b0_mass=b0_mass,
        challenger_mass=challenger_mass,
        b0_source_count=background.audit.b0_source_count,
        recent_source_count=background.audit.recent_30d_source_count,
        b0_alarm=b0_alarm,
        challenger_alarm=challenger_alarm,
    )


def validate_washout(schedule: P1IssueSchedule) -> None:
    """Require 60 days between the local cutover and the first real issue Q."""

    if schedule.query_cutoff_utc - timedelta(days=60) <= P1_LOCAL_CUTOFF_UTC:
        raise ValueError("the issue does not satisfy the frozen 60-day same-source washout")


__all__ = [
    "P1_ALPHA",
    "P1_AREA_CAP_KM2",
    "P1_BANDWIDTH_KM",
    "P1_MODEL_MANIFEST_SHA256",
    "P1_SOURCE_BOUNDARY_MANIFEST_SHA256",
    "ProductionCatalogArtifact",
    "ProductionForecastBundle",
    "SourceBoundaryMatch",
    "build_production_catalog",
    "build_production_forecast",
    "deduplicate_local_comcat_boundary",
    "validate_washout",
]
