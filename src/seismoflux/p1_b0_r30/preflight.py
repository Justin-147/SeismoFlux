"""Target-blind real-history adapter rehearsal for P1 ``B0`` versus ``B0_R30``.

The adapter reads only three explicitly supplied frozen byte payloads: the
historical earthquake catalogue, the study-area GeoJSON, and the local-support
manifest.  It never locates a newer file, accesses the network, reads a future
outcome, or creates a real prospective issue.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.d1_replay.spatial import (
    D1SpatialDomain,
    build_causal_background_components,
    build_d1_spatial_domain_from_bytes,
)
from seismoflux.data.common import canonical_json_bytes
from seismoflux.stage2s.catalog import (
    Stage2SEarthquakeCatalog,
    parse_frozen_catalog_bytes,
)
from seismoflux.stage2s.contracts import MASS_SUM_ABSOLUTE_TOLERANCE, SpatialGrid

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ModelId: TypeAlias = Literal["B0", "B0_R30"]
SupportStatus: TypeAlias = Literal["supported", "indeterminate", "unsupported"]

PREFLIGHT_ISSUE_ID: Final = "p1-preflight-20260909T160000Z"
PREFLIGHT_SCHEDULED_TIME_UTC: Final = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
PREFLIGHT_QUERY_CUTOFF_UTC: Final = datetime(2026, 9, 9, 15, 45, tzinfo=UTC)
PREFLIGHT_AREA_CAP_KM2: Final = 600_000.0
PREFLIGHT_ALPHA: Final = 0.25
PREFLIGHT_BANDWIDTH_KM: Final = 75.0

EXPECTED_STUDY_AREA_SHA256: Final = (
    "5e5dcf012e080882161c95bf592a1ee39a0f0fdad7114bcff58d645aeb30bb02"
)
EXPECTED_SUPPORT_MANIFEST_SHA256: Final = (
    "632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d"
)
EXPECTED_SUPPORT_ID: Final = "local-support-f6816ab6c6581306"
EXPECTED_COMMON_MC: Final = 4.0
EXPECTED_SUPPORT_FIT_END_UTC: Final = "2023-06-30T16:00:00.000000Z"
EXPECTED_CATALOG_LAST_ORIGIN_UTC: Final = "2026-07-09T04:25:56Z"
EXPECTED_CATALOG_LAST_AVAILABLE_UTC: Final = "2026-07-09T04:25:56Z"
EXPECTED_CATALOG_ROW_COUNT: Final = 40_898
EXPECTED_B0_SOURCE_COUNT: Final = 5_991
EXPECTED_RECENT_SOURCE_COUNT: Final = 0
EXPECTED_OPERATIONAL_CELL_COUNT: Final = 15_697

_UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_FORBIDDEN_START_KEY_TOKENS: Final = (
    "truth",
    "target",
    "cluster",
    "hit",
    "miss",
    "recall",
    "gain",
    "review",
    "outcome",
    "score",
    "effect",
)
_ALLOWED_START_INTEGRITY_KEYS: Final = frozenset({"future_outcomes_absent"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _epoch_us_text(value: int) -> str:
    return _utc_text(_UTC_EPOCH + timedelta(microseconds=int(value)))


def _readonly_float(values: object, *, name: str, expected_size: int) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or result.size != expected_size:
        raise ValueError(f"{name} must be a one-dimensional grid-length vector")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values")
    if not math.isclose(
        math.fsum(float(item) for item in result),
        1.0,
        rel_tol=0.0,
        abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{name} must sum to one")
    result.setflags(write=False)
    return result


def _readonly_int(values: object, *, name: str) -> IntArray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if result.ndim != 1 or np.any(result < 0):
        raise ValueError(f"{name} must be a one-dimensional non-negative vector")
    result.setflags(write=False)
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{label} must be an array")
    return cast(Sequence[object], value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def reject_future_outcome_fields(value: object, *, path: str = "$") -> None:
    """Fail closed if a start-time payload contains an outcome/evaluation field."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{path} contains a non-string key")
            normalized = raw_key.strip().casefold().replace("-", "_")
            token = next(
                (item for item in _FORBIDDEN_START_KEY_TOKENS if item in normalized),
                None,
            )
            if token is not None and normalized not in _ALLOWED_START_INTEGRITY_KEYS:
                raise ValueError(
                    f"future outcome field is forbidden at start time: {path}.{raw_key}"
                )
            reject_future_outcome_fields(child, path=f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            reject_future_outcome_fields(child, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class SupportWaterLevel:
    manifest_sha256: str
    support_id: str
    common_mc: float
    fit_end_utc: str
    cell_count: int
    supported_cell_count: int
    indeterminate_cell_count: int
    unsupported_cell_count: int
    retained_area_fraction: float
    retained_area_km2: float
    cell_statuses: tuple[tuple[int, int, SupportStatus], ...]

    def as_mapping(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "support_id": self.support_id,
            "common_mc": self.common_mc,
            "fit_end_utc": self.fit_end_utc,
            "cell_count": self.cell_count,
            "supported_cell_count": self.supported_cell_count,
            "indeterminate_cell_count": self.indeterminate_cell_count,
            "unsupported_cell_count": self.unsupported_cell_count,
            "retained_area_fraction": self.retained_area_fraction,
            "retained_area_km2": self.retained_area_km2,
            "local_mc_scope": "each_frozen_local_unit_only_no_threshold_raise_elsewhere",
        }

    def status_by_base_row_column(self) -> dict[tuple[int, int], SupportStatus]:
        return {(row, column): status for row, column, status in self.cell_statuses}

    def status_for_25km_cell(self, *, row: int, column: int) -> SupportStatus:
        """Map one 25 km index to its 500 km support parent with floor division."""

        if type(row) is not int or type(column) is not int:
            raise TypeError("25 km row and column must be integers")
        try:
            return self.status_by_base_row_column()[(row // 20, column // 20)]
        except KeyError as error:
            raise ValueError("a 25 km cell has no frozen 500 km support parent") from error


def parse_support_water_level(payload: bytes) -> SupportWaterLevel:
    """Verify and summarize the exact frozen final-validation support bytes."""

    if type(payload) is not bytes:
        raise TypeError("support manifest payload must be immutable bytes")
    digest = _sha256(payload)
    if digest != EXPECTED_SUPPORT_MANIFEST_SHA256:
        raise ValueError("support manifest SHA-256 mismatch")
    try:
        root = _mapping(json.loads(payload), label="support manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("support manifest bytes are not valid UTF-8 JSON") from error
    snapshots = _sequence(root.get("snapshots"), label="support snapshots")
    selected = [
        _mapping(item, label="support snapshot")
        for item in snapshots
        if _mapping(item, label="support snapshot").get("snapshot_id") == "final_validation"
    ]
    if len(selected) != 1:
        raise ValueError("support manifest must contain exactly one final_validation snapshot")
    support = _mapping(selected[0].get("support"), label="final support")
    support_id = _text(support.get("support_id"), label="support_id")
    common_mc = _number(support.get("common_mc"), label="common_mc")
    fit_end = _text(support.get("fit_end_utc"), label="fit_end_utc")
    if support_id != EXPECTED_SUPPORT_ID or common_mc != EXPECTED_COMMON_MC:
        raise ValueError("final support identity or common Mc differs from the frozen P1 contract")
    if fit_end != EXPECTED_SUPPORT_FIT_END_UTC:
        raise ValueError("support calibration cutoff differs from the frozen P1 contract")
    cells = tuple(
        _mapping(item, label="support cell")
        for item in _sequence(support.get("cells"), label="support cells")
    )
    statuses = tuple(_text(cell.get("status"), label="support cell status") for cell in cells)
    allowed_statuses = {"supported", "indeterminate", "unsupported"}
    if not cells or any(status not in allowed_statuses for status in statuses):
        raise ValueError("support cells contain an invalid status")
    for cell, status in zip(cells, statuses, strict=True):
        applied_mc = cell.get("applied_mc")
        if status == "unsupported":
            if applied_mc is not None:
                raise ValueError("unsupported support cells must not apply a magnitude threshold")
        elif _number(applied_mc, label="support cell applied_mc") != common_mc:
            raise ValueError("retained support cells must use the frozen common Mc")
    counts = {status: statuses.count(status) for status in allowed_statuses}
    fixed_cells = tuple(
        _mapping(item, label="fixed support cell")
        for item in _sequence(root.get("fixed_cells"), label="fixed support cells")
    )
    fixed_by_id: dict[str, tuple[int, int]] = {}
    for fixed in fixed_cells:
        cell_id = _text(fixed.get("cell_id"), label="fixed support cell id")
        row = fixed.get("row")
        column = fixed.get("column")
        if type(row) is not int or type(column) is not int:
            raise ValueError("fixed support rows and columns must be integers")
        fixed_by_id[cell_id] = (row, column)
    if len(fixed_by_id) != len(fixed_cells) or set(fixed_by_id) != {
        _text(cell.get("cell_id"), label="support cell id") for cell in cells
    }:
        raise ValueError("fixed support cells and final support statuses do not align")
    cell_statuses = tuple(
        sorted(
            (
                *fixed_by_id[_text(cell.get("cell_id"), label="support cell id")],
                cast(SupportStatus, status),
            )
            for cell, status in zip(cells, statuses, strict=True)
        )
    )
    retained_fraction = _number(
        support.get("retained_area_fraction"), label="retained_area_fraction"
    )
    retained_area_km2 = _number(support.get("retained_area_m2"), label="retained_area_m2") / 1e6
    if (
        len(cells) != 61
        or counts["supported"] != 52
        or counts["indeterminate"] != 9
        or counts["unsupported"] != 0
        or retained_fraction != 1.0
    ):
        raise ValueError("final support cell summary differs from the frozen P1 declaration")
    return SupportWaterLevel(
        manifest_sha256=digest,
        support_id=support_id,
        common_mc=common_mc,
        fit_end_utc=fit_end,
        cell_count=len(cells),
        supported_cell_count=counts["supported"],
        indeterminate_cell_count=counts["indeterminate"],
        unsupported_cell_count=counts["unsupported"],
        retained_area_fraction=retained_fraction,
        retained_area_km2=retained_area_km2,
        cell_statuses=cell_statuses,
    )


@dataclass(frozen=True, slots=True)
class AlarmSelection:
    model_id: ModelId
    area_cap_km2: float
    actual_area_km2: float
    next_complete_cell_area_km2: float | None
    ranked_indices: IntArray
    selected_indices: IntArray
    selected_cell_ids: tuple[str, ...]
    ranking_sha256: str
    selected_mask_sha256: str

    def __post_init__(self) -> None:
        if self.model_id not in {"B0", "B0_R30"}:
            raise ValueError("alarm selection model_id must be B0 or B0_R30")
        cap = float(self.area_cap_km2)
        actual = float(self.actual_area_km2)
        if not math.isfinite(cap) or cap < 0.0 or not 0.0 <= actual <= cap:
            raise ValueError("alarm area and cap must be finite and ordered")
        ranking = _readonly_int(self.ranked_indices, name="ranked_indices")
        selected = _readonly_int(self.selected_indices, name="selected_indices")
        if selected.size > ranking.size or len(set(int(item) for item in ranking)) != ranking.size:
            raise ValueError("alarm ranking must be unique and contain the selected prefix")
        if selected.tobytes() != ranking[: selected.size].tobytes():
            raise ValueError("selected alarm cells must be the complete leading ranking prefix")
        identifiers = tuple(self.selected_cell_ids)
        if len(identifiers) != selected.size or len(set(identifiers)) != len(identifiers):
            raise ValueError("selected cell identifiers must align uniquely with selected indices")
        object.__setattr__(self, "area_cap_km2", cap)
        object.__setattr__(self, "actual_area_km2", actual)
        object.__setattr__(self, "ranked_indices", ranking)
        object.__setattr__(self, "selected_indices", selected)
        object.__setattr__(self, "selected_cell_ids", identifiers)

    def as_mapping(self) -> dict[str, object]:
        return {
            "area_cap_km2": self.area_cap_km2,
            "actual_area_km2": self.actual_area_km2,
            "next_complete_cell_area_km2": self.next_complete_cell_area_km2,
            "selected_cell_count": int(self.selected_indices.size),
            "selected_cell_ids": list(self.selected_cell_ids),
            "ranking_sha256": self.ranking_sha256,
            "selected_mask_sha256": self.selected_mask_sha256,
        }


def select_complete_alarm_prefix(
    mass: object,
    grid: SpatialGrid,
    *,
    model_id: ModelId,
    area_cap_km2: float,
) -> AlarmSelection:
    """Select the frozen no-skip complete-cell prefix for an arbitrary fair cap."""

    if not isinstance(grid, SpatialGrid) or grid.cell_size_km != 25.0:
        raise TypeError("P1 alarm selection requires the frozen 25 km SpatialGrid")
    values = _readonly_float(mass, name=f"{model_id}_mass", expected_size=grid.cell_count)
    cap = float(area_cap_km2)
    if not math.isfinite(cap) or cap < 0.0:
        raise ValueError("area_cap_km2 must be finite and non-negative")
    intensity = values / grid.clipped_area_km2
    ranking_tuple = tuple(
        sorted(
            range(grid.cell_count),
            key=lambda index: (
                -float(intensity[index]),
                int(grid.rows[index]),
                int(grid.columns[index]),
                grid.cell_ids[index].encode("utf-8"),
            ),
        )
    )
    selected: list[int] = []
    selected_areas: list[float] = []
    next_area: float | None = None
    for index in ranking_tuple:
        area = float(grid.clipped_area_km2[index])
        if math.fsum((*selected_areas, area)) > cap:
            next_area = area
            break
        selected.append(index)
        selected_areas.append(area)
    ranking = np.asarray(ranking_tuple, dtype=np.int64)
    selected_indices = np.asarray(selected, dtype=np.int64)
    selected_ids = tuple(grid.cell_ids[index] for index in selected)
    return AlarmSelection(
        model_id=model_id,
        area_cap_km2=cap,
        actual_area_km2=math.fsum(selected_areas),
        next_complete_cell_area_km2=next_area,
        ranked_indices=ranking,
        selected_indices=selected_indices,
        selected_cell_ids=selected_ids,
        ranking_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "seismoflux.p1.real-history-ranking.v1",
                    "model_id": model_id,
                    "cell_ids": [grid.cell_ids[index] for index in ranking_tuple],
                }
            )
        ).hexdigest(),
        selected_mask_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "seismoflux.p1.real-history-alarm-mask.v1",
                    "model_id": model_id,
                    "cell_ids": list(selected_ids),
                    "actual_area_km2_hex": math.fsum(selected_areas).hex(),
                }
            )
        ).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class RealHistoryPreflight:
    catalog: Stage2SEarthquakeCatalog
    support: SupportWaterLevel
    study_area_sha256: str
    domain: D1SpatialDomain
    b0_mass: FloatArray
    challenger_mass: FloatArray
    b0_source_count: int
    recent_source_count: int
    b0_alarm: AlarmSelection
    challenger_alarm: AlarmSelection

    def __post_init__(self) -> None:
        grid = self.domain.operational_grid
        b0 = _readonly_float(self.b0_mass, name="b0_mass", expected_size=grid.cell_count)
        challenger = _readonly_float(
            self.challenger_mass,
            name="challenger_mass",
            expected_size=grid.cell_count,
        )
        if self.recent_source_count == 0 and b0.tobytes() != challenger.tobytes():
            raise ValueError("an empty R30 window must fall back bitwise exactly to B0")
        area_difference = self.b0_alarm.actual_area_km2 - self.challenger_alarm.actual_area_km2
        if area_difference < 0.0 or area_difference >= 625.0:
            raise ValueError("challenger alarm area violates the frozen paired fairness rule")
        next_area = self.challenger_alarm.next_complete_cell_area_km2
        if next_area is not None and area_difference >= next_area:
            raise ValueError("challenger left enough area to admit its next complete cell")
        object.__setattr__(self, "b0_mass", b0)
        object.__setattr__(self, "challenger_mass", challenger)

    @property
    def actual_area_difference_km2(self) -> float:
        return self.b0_alarm.actual_area_km2 - self.challenger_alarm.actual_area_km2

    def as_mapping(self) -> dict[str, object]:
        grid = self.domain.operational_grid
        b0_density = self.b0_mass / grid.clipped_area_km2
        challenger_density = self.challenger_mass / grid.clipped_area_km2
        shared_max = max(float(np.max(b0_density)), float(np.max(challenger_density)))
        if not math.isfinite(shared_max) or shared_max <= 0.0:
            raise ValueError("shared relative-intensity scale must have a positive maximum")
        b0_relative = b0_density / shared_max
        challenger_relative = challenger_density / shared_max
        catalog_max_origin = int(np.max(self.catalog.origin_time_us))
        catalog_max_available = int(np.max(self.catalog.available_at_us))
        mapping: dict[str, object] = {
            "schema_version": 1,
            "stage_id": "P1-0C",
            "artifact_role": "historical_adapter_rehearsal_not_real_prospective_issue",
            "real_issue_authorized": False,
            "future_outcomes_absent": True,
            "issue_id": PREFLIGHT_ISSUE_ID,
            "scheduled_time_utc": _utc_text(PREFLIGHT_SCHEDULED_TIME_UTC),
            "query_cutoff_utc": _utc_text(PREFLIGHT_QUERY_CUTOFF_UTC),
            "catalog_water_level": {
                "row_count": self.catalog.row_count,
                "file_sha256": self.catalog.identity.file_sha256,
                "content_sha256": self.catalog.identity.content_sha256,
                "schema_sha256": self.catalog.identity.schema_sha256,
                "maximum_origin_time_utc": _epoch_us_text(catalog_max_origin),
                "maximum_available_at_utc": _epoch_us_text(catalog_max_available),
                "historical_available_at_assumption": (
                    "frozen_local_history_uses_origin_time_as_available_at_only_for_rehearsal"
                ),
            },
            "support_water_level": self.support.as_mapping(),
            "spatial_domain": {
                "study_area_sha256": self.study_area_sha256,
                "grid_id": grid.grid_id,
                "cell_size_km": grid.cell_size_km,
                "cell_count": grid.cell_count,
                "cells": [
                    {
                        "cell_id": cell_id,
                        "row": int(row),
                        "column": int(column),
                        "x_km": float(xy[0]),
                        "y_km": float(xy[1]),
                        "clipped_area_km2": float(area),
                    }
                    for cell_id, row, column, xy, area in zip(
                        grid.cell_ids,
                        grid.rows,
                        grid.columns,
                        grid.query_xy_km,
                        grid.clipped_area_km2,
                        strict=True,
                    )
                ],
            },
            "method": {
                "kernel": "gaussian_KDE",
                "bandwidth_km": PREFLIGHT_BANDWIDTH_KM,
                "B0_event_rule": "1970_plus_M4_plus_inside_origin_and_available_lte_Q",
                "recent_event_rule": "ComCat_M4_plus_in_open_closed_Q_minus_30d_to_Q",
                "challenger_formula": "0.75*B0+0.25*R30",
                "empty_recent_action": "exact_B0_fallback",
                "value_semantics": "relative_intensity_not_absolute_probability",
            },
            "components": {
                "B0_source_count": self.b0_source_count,
                "R30_source_count": self.recent_source_count,
                "empty_recent_fallback_to_B0": self.recent_source_count == 0,
            },
            "shared_color_scale": {
                "minimum": 0.0,
                "maximum": 1.0,
                "definition": "both_models_density_per_km2_divided_by_one_shared_maximum",
            },
            "models": {
                "B0": {
                    "normalized_cell_mass": self.b0_mass.tolist(),
                    "relative_intensity": b0_relative.tolist(),
                    "mass_sha256": hashlib.sha256(self.b0_mass.tobytes()).hexdigest(),
                    "alarm": self.b0_alarm.as_mapping(),
                },
                "B0_R30": {
                    "normalized_cell_mass": self.challenger_mass.tolist(),
                    "relative_intensity": challenger_relative.tolist(),
                    "mass_sha256": hashlib.sha256(self.challenger_mass.tobytes()).hexdigest(),
                    "alarm": self.challenger_alarm.as_mapping(),
                },
            },
            "paired_area_fairness": {
                "B0_actual_area_km2": self.b0_alarm.actual_area_km2,
                "B0_R30_actual_area_km2": self.challenger_alarm.actual_area_km2,
                "actual_area_difference_km2": self.actual_area_difference_km2,
                "challenger_next_complete_cell_area_km2": (
                    self.challenger_alarm.next_complete_cell_area_km2
                ),
                "status": "passed",
            },
            "scientific_value": {
                "category": "necessary_enabler",
                "direct_prediction_improvement": "none_new_in_this_rehearsal",
                "meaning": (
                    "frozen_real_history_reaches_the_same_start_time_map_path_without_future_outcomes"
                ),
            },
        }
        reject_future_outcome_fields(mapping)
        return mapping

    def as_rendering_mapping(self) -> dict[str, object]:
        """Return the strict, start-time-only mapping consumed by visual renderers."""

        grid = self.domain.operational_grid
        rendered_cells: list[dict[str, object]] = []
        for cell_id, row, column, area in zip(
            grid.cell_ids,
            grid.rows,
            grid.columns,
            grid.clipped_area_km2,
            strict=True,
        ):
            support_status = self.support.status_for_25km_cell(
                row=int(row),
                column=int(column),
            )
            rendered_cells.append(
                {
                    "cell_id": cell_id,
                    "row": int(row),
                    "column": int(column),
                    "area_km2": float(area),
                    "support_status": support_status,
                }
            )
        catalog_max_origin = _epoch_us_text(int(np.max(self.catalog.origin_time_us)))
        catalog_max_available = _epoch_us_text(int(np.max(self.catalog.available_at_us)))
        mapping: dict[str, object] = {
            "rehearsal_id": "p1-0c-real-history-adapter-20260909",
            "scheduled_issue_time_utc": _utc_text(PREFLIGHT_SCHEDULED_TIME_UTC),
            "query_cutoff_utc": _utc_text(PREFLIGHT_QUERY_CUTOFF_UTC),
            "catalog": {
                "path": ("data/processed/stage1/debc98054172a4a1/earthquake_event.parquet"),
                "sha256": self.catalog.identity.file_sha256,
                "row_count": self.catalog.row_count,
                "eligible_B0_event_count": self.b0_source_count,
                "R30_event_count": self.recent_source_count,
                "origin_time_max_utc": catalog_max_origin,
                "available_at_max_utc": catalog_max_available,
            },
            "support": {
                "support_id": self.support.support_id,
                "manifest_sha256": self.support.manifest_sha256,
                "fit_end_utc": EXPECTED_SUPPORT_FIT_END_UTC.replace(".000000", ""),
                "common_mc": self.support.common_mc,
                "fixed_cell_count": self.support.cell_count,
                "supported_cell_count": self.support.supported_cell_count,
                "indeterminate_cell_count": self.support.indeterminate_cell_count,
                "unsupported_cell_count": self.support.unsupported_cell_count,
                "retained_area_km2": self.support.retained_area_km2,
            },
            "grid": {"cell_size_km": 25.0, "cells": rendered_cells},
            "models": {
                "B0": {
                    "normalized_cell_mass": self.b0_mass.tolist(),
                    "alarm_cell_ids": list(self.b0_alarm.selected_cell_ids),
                    "actual_alarm_area_km2": self.b0_alarm.actual_area_km2,
                    "next_complete_cell_area_km2": (self.b0_alarm.next_complete_cell_area_km2),
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
        reject_future_outcome_fields(mapping)
        return mapping


def build_real_history_preflight(
    *,
    catalog_bytes: bytes,
    study_area_bytes: bytes,
    support_manifest_bytes: bytes,
) -> RealHistoryPreflight:
    """Run the one frozen P1-0C real-history start-time adapter rehearsal."""

    if type(study_area_bytes) is not bytes:
        raise TypeError("study area payload must be immutable bytes")
    study_area_sha256 = _sha256(study_area_bytes)
    if study_area_sha256 != EXPECTED_STUDY_AREA_SHA256:
        raise ValueError("study area SHA-256 mismatch")
    support = parse_support_water_level(support_manifest_bytes)
    catalog = parse_frozen_catalog_bytes(catalog_bytes)
    if catalog.row_count != EXPECTED_CATALOG_ROW_COUNT:
        raise ValueError("frozen catalog row count differs from the P1-0C declaration")
    max_origin = _epoch_us_text(int(np.max(catalog.origin_time_us)))
    max_available = _epoch_us_text(int(np.max(catalog.available_at_us)))
    if (
        max_origin != EXPECTED_CATALOG_LAST_ORIGIN_UTC
        or max_available != EXPECTED_CATALOG_LAST_AVAILABLE_UTC
    ):
        raise ValueError("frozen catalog water level differs from the P1-0C declaration")

    domain = build_d1_spatial_domain_from_bytes(study_area_bytes)
    if domain.operational_grid.cell_count != EXPECTED_OPERATIONAL_CELL_COUNT:
        raise ValueError("operational grid cell count differs from the frozen P1 domain")
    background = build_causal_background_components(
        catalog,
        PREFLIGHT_QUERY_CUTOFF_UTC,
        domain,
    )
    if (
        background.audit.b0_source_count != EXPECTED_B0_SOURCE_COUNT
        or background.audit.recent_30d_source_count != EXPECTED_RECENT_SOURCE_COUNT
    ):
        raise ValueError("causal source counts differ from the frozen P1-0C declaration")
    b0_mass = background.b0_mass_25km
    challenger_mass = background.mass_for_alpha(PREFLIGHT_ALPHA)
    grid = domain.operational_grid
    b0_alarm = select_complete_alarm_prefix(
        b0_mass,
        grid,
        model_id="B0",
        area_cap_km2=PREFLIGHT_AREA_CAP_KM2,
    )
    challenger_alarm = select_complete_alarm_prefix(
        challenger_mass,
        grid,
        model_id="B0_R30",
        area_cap_km2=b0_alarm.actual_area_km2,
    )
    return RealHistoryPreflight(
        catalog=catalog,
        support=support,
        study_area_sha256=study_area_sha256,
        domain=domain,
        b0_mass=b0_mass,
        challenger_mass=challenger_mass,
        b0_source_count=background.audit.b0_source_count,
        recent_source_count=background.audit.recent_30d_source_count,
        b0_alarm=b0_alarm,
        challenger_alarm=challenger_alarm,
    )


__all__ = [
    "EXPECTED_B0_SOURCE_COUNT",
    "EXPECTED_CATALOG_ROW_COUNT",
    "EXPECTED_OPERATIONAL_CELL_COUNT",
    "EXPECTED_RECENT_SOURCE_COUNT",
    "EXPECTED_STUDY_AREA_SHA256",
    "EXPECTED_SUPPORT_ID",
    "EXPECTED_SUPPORT_MANIFEST_SHA256",
    "PREFLIGHT_ALPHA",
    "PREFLIGHT_AREA_CAP_KM2",
    "PREFLIGHT_BANDWIDTH_KM",
    "PREFLIGHT_ISSUE_ID",
    "PREFLIGHT_QUERY_CUTOFF_UTC",
    "PREFLIGHT_SCHEDULED_TIME_UTC",
    "AlarmSelection",
    "RealHistoryPreflight",
    "SupportWaterLevel",
    "build_real_history_preflight",
    "parse_support_water_level",
    "reject_future_outcome_fields",
    "select_complete_alarm_prefix",
]
