"""Typed, in-memory scientific preimages for the P1 prospective record chain.

The six public record types remain unchanged.  This module treats their hashes as
references to immutable scientific preimages and independently rebuilds the path

``raw catalogue bytes -> events -> forecast/masks -> clusters -> scores -> review``.

It intentionally contains no filesystem, network, ledger, or real-data access.
The small catalogue byte format is an explicit synthetic fixture format used to
close the scientific dependency graph before a real source adapter is authorized.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.core import (
    AlarmPrefix,
    DualModelForecast,
    GridCell,
    RelativeIntensitySurface,
    ScoreSummary,
    SyntheticEvent,
    TargetCluster,
    build_dual_model_forecast,
    cluster_target_events,
    ordered_cluster_registry_sha256,
    score_clusters,
)
from seismoflux.p1_b0_r30.records import (
    validate_record_against_schema,
    validate_record_chain,
)

Horizon: TypeAlias = Literal[30, 90]
Model: TypeAlias = Literal["B0", "B0_R30"]
SnapshotRole: TypeAlias = Literal["forecast", "truth"]

_SNAPSHOT_FORMAT = "seismoflux.p1.synthetic-catalog-snapshot.v1"
_GRID_DOMAIN = "seismoflux.p1.synthetic-grid.v1"
_EVENT_FIELDS = {
    "event_id",
    "origin_time_utc",
    "available_at_utc",
    "x_km",
    "y_km",
    "magnitude",
    "source_id",
    "longitude",
    "latitude",
}
_SNAPSHOT_FIELDS = {
    "schema_version",
    "format",
    "role",
    "issue_id",
    "scheduled_issue_time_utc",
    "query_cutoff_utc",
    "horizon_days",
    "truth_fetched_at_utc",
    "grid_sha256",
    "events",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return _utc(value, label="timestamp").isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(UTC)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid UTC timestamp") from exc


def grid_preimage_sha256(grid: tuple[GridCell, ...]) -> str:
    """Return the deterministic identity of the exact ordered synthetic grid."""

    if not grid or len({cell.cell_id for cell in grid}) != len(grid):
        raise ValueError("grid preimage must contain unique cells")
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "domain": _GRID_DOMAIN,
                "cells": [cell.as_mapping() for cell in grid],
            }
        )
    )


def source_snapshot_sha256(raw_source_bytes: bytes) -> str:
    """Hash the exact immutable source bytes, not a caller-supplied digest."""

    if not isinstance(raw_source_bytes, bytes):
        raise TypeError("raw source preimage must be bytes")
    return _sha256_bytes(raw_source_bytes)


def build_catalog_snapshot_bytes(
    *,
    role: SnapshotRole,
    issue_id: str,
    scheduled_issue_time_utc: datetime,
    grid: tuple[GridCell, ...],
    events: tuple[SyntheticEvent, ...],
    horizon_days: Horizon | None = None,
    truth_fetched_at_utc: datetime | None = None,
) -> bytes:
    """Build canonical bytes for the small, explicitly synthetic catalogue fixture."""

    scheduled = _utc(scheduled_issue_time_utc, label="scheduled_issue_time_utc")
    if role == "forecast":
        if horizon_days is not None or truth_fetched_at_utc is not None:
            raise ValueError("forecast snapshot may not carry truth horizon or fetch time")
        query_cutoff: str | None = _utc_text(scheduled - timedelta(minutes=15))
        truth_fetched: str | None = None
    elif role == "truth":
        if horizon_days not in {30, 90} or truth_fetched_at_utc is None:
            raise ValueError("truth snapshot requires a 30/90-day horizon and fetch time")
        query_cutoff = None
        truth_fetched = _utc_text(truth_fetched_at_utc)
    else:
        raise ValueError("snapshot role must be forecast or truth")
    ordered_events = tuple(
        sorted(events, key=lambda event: (event.origin_time_utc, event.event_id.encode("utf-8")))
    )
    if len({event.event_id for event in ordered_events}) != len(ordered_events):
        raise ValueError("source snapshot event IDs must be unique")
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "format": _SNAPSHOT_FORMAT,
            "role": role,
            "issue_id": issue_id,
            "scheduled_issue_time_utc": _utc_text(scheduled),
            "query_cutoff_utc": query_cutoff,
            "horizon_days": horizon_days,
            "truth_fetched_at_utc": truth_fetched,
            "grid_sha256": grid_preimage_sha256(grid),
            "events": [event.as_mapping() for event in ordered_events],
        }
    )


def _event_from_mapping(raw: object) -> SyntheticEvent:
    if not isinstance(raw, dict) or set(raw) != _EVENT_FIELDS:
        raise ValueError("source event fields differ from the frozen fixture format")
    event_id = raw["event_id"]
    source_id = raw["source_id"]
    if not isinstance(event_id, str) or source_id not in {
        "synthetic_history",
        "synthetic_ComCat",
    }:
        raise ValueError("source event identity or source role is invalid")
    numeric_names = ("x_km", "y_km", "magnitude", "longitude", "latitude")
    values = tuple(raw[name] for name in numeric_names)
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise ValueError("source event numeric fields must be numbers")
    return SyntheticEvent(
        event_id=event_id,
        origin_time_utc=_parse_utc(raw["origin_time_utc"], label="event origin_time_utc"),
        available_at_utc=_parse_utc(raw["available_at_utc"], label="event available_at_utc"),
        x_km=float(raw["x_km"]),
        y_km=float(raw["y_km"]),
        magnitude=float(raw["magnitude"]),
        source_id=cast(Literal["synthetic_history", "synthetic_ComCat"], source_id),
        longitude=float(raw["longitude"]),
        latitude=float(raw["latitude"]),
    )


def _decode_catalog_snapshot(
    raw_source_bytes: bytes,
    *,
    role: SnapshotRole,
    issue_id: str,
    scheduled_issue_time_utc: datetime,
    grid: tuple[GridCell, ...],
    horizon_days: Horizon | None,
    truth_fetched_at_utc: datetime | None,
) -> tuple[SyntheticEvent, ...]:
    if not isinstance(raw_source_bytes, bytes):
        raise TypeError("raw source preimage must be bytes")
    try:
        decoded = json.loads(raw_source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw source preimage is not valid UTF-8 canonical JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != _SNAPSHOT_FIELDS:
        raise ValueError("source snapshot fields differ from the frozen fixture format")
    if canonical_json_bytes(decoded) != raw_source_bytes:
        raise ValueError("source snapshot bytes are not in canonical form")
    scheduled = _utc(scheduled_issue_time_utc, label="scheduled_issue_time_utc")
    expected_query = _utc_text(scheduled - timedelta(minutes=15)) if role == "forecast" else None
    expected_truth_fetched = (
        _utc_text(truth_fetched_at_utc) if truth_fetched_at_utc is not None else None
    )
    expected_context = {
        "schema_version": 1,
        "format": _SNAPSHOT_FORMAT,
        "role": role,
        "issue_id": issue_id,
        "scheduled_issue_time_utc": _utc_text(scheduled),
        "query_cutoff_utc": expected_query,
        "horizon_days": horizon_days,
        "truth_fetched_at_utc": expected_truth_fetched,
        "grid_sha256": grid_preimage_sha256(grid),
    }
    for name, expected in expected_context.items():
        if decoded.get(name) != expected:
            raise ValueError(f"source snapshot {name} does not bind the record context")
    raw_events = decoded["events"]
    if not isinstance(raw_events, list):
        raise ValueError("source snapshot events must be a list")
    events = tuple(_event_from_mapping(raw) for raw in raw_events)
    expected_order = tuple(
        sorted(events, key=lambda event: (event.origin_time_utc, event.event_id.encode("utf-8")))
    )
    if events != expected_order or len({event.event_id for event in events}) != len(events):
        raise ValueError("source snapshot events must use the unique frozen stable order")
    return events


@dataclass(frozen=True, slots=True)
class ModelArtifactPreimage:
    """Saved surface/ranking/mask rows for one model, checked against recomputation."""

    model_id: Model
    active_event_count: int
    relative_intensity: tuple[float, ...]
    ranked_cell_ids: tuple[str, ...]
    selected_cell_ids: tuple[str, ...]

    @classmethod
    def from_parts(
        cls,
        surface: RelativeIntensitySurface,
        alarm: AlarmPrefix,
    ) -> ModelArtifactPreimage:
        if surface.model_id not in {"B0", "B0_R30"} or alarm.model_id != surface.model_id:
            raise ValueError("model artifact requires matching B0 or B0_R30 parts")
        return cls(
            model_id=cast(Model, surface.model_id),
            active_event_count=surface.active_event_count,
            relative_intensity=tuple(float(value) for value in surface.relative_intensity),
            ranked_cell_ids=alarm.ranked_cell_ids,
            selected_cell_ids=alarm.selected_cell_ids,
        )


@dataclass(frozen=True, slots=True)
class ForecastScientificPreimage:
    """Raw source bytes plus the saved forecast artifacts for one issued forecast."""

    issue_id: str
    raw_source_bytes: bytes
    grid: tuple[GridCell, ...]
    models: tuple[ModelArtifactPreimage, ModelArtifactPreimage]

    def __post_init__(self) -> None:
        if not self.issue_id or self.issue_id.strip() != self.issue_id:
            raise ValueError("forecast preimage issue_id must be non-empty and stripped")
        if not isinstance(self.raw_source_bytes, bytes):
            raise TypeError("forecast source preimage must be bytes")
        if tuple(model.model_id for model in self.models) != ("B0", "B0_R30"):
            raise ValueError("forecast preimage must contain ordered B0 and B0_R30 artifacts")

    @classmethod
    def from_forecast(
        cls,
        *,
        raw_source_bytes: bytes,
        forecast: DualModelForecast,
    ) -> ForecastScientificPreimage:
        return cls(
            issue_id=forecast.issue_id,
            raw_source_bytes=raw_source_bytes,
            grid=forecast.grid,
            models=(
                ModelArtifactPreimage.from_parts(forecast.B0, forecast.B0_alarm),
                ModelArtifactPreimage.from_parts(forecast.B0_R30, forecast.B0_R30_alarm),
            ),
        )


@dataclass(frozen=True, slots=True)
class ClusterArtifactPreimage:
    """Saved member and representative rows for one target cluster."""

    cluster_id: str
    member_event_ids: tuple[str, ...]
    representative: SyntheticEvent

    def __post_init__(self) -> None:
        if not self.cluster_id or self.cluster_id.strip() != self.cluster_id:
            raise ValueError("cluster artifact ID must be non-empty and stripped")
        if (
            not self.member_event_ids
            or len(set(self.member_event_ids)) != len(self.member_event_ids)
            or self.representative.event_id not in self.member_event_ids
        ):
            raise ValueError("cluster artifact members and representative are inconsistent")

    @classmethod
    def from_cluster(cls, cluster: TargetCluster) -> ClusterArtifactPreimage:
        return cls(
            cluster_id=cluster.cluster_id,
            member_event_ids=cluster.member_event_ids,
            representative=cluster.representative,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "member_event_ids": list(self.member_event_ids),
            "representative_event": self.representative.as_mapping(),
        }


def cluster_assignment_sha256(clusters: Sequence[ClusterArtifactPreimage]) -> str:
    """Hash the exact saved cluster-member/representative preimage."""

    return _sha256_bytes(canonical_json_bytes([cluster.as_mapping() for cluster in clusters]))


@dataclass(frozen=True, slots=True)
class TruthScientificPreimage:
    """Raw truth-source bytes and the saved cluster assignment for one exposure."""

    issue_id: str
    horizon_days: Horizon
    raw_source_bytes: bytes
    clusters: tuple[ClusterArtifactPreimage, ...]

    def __post_init__(self) -> None:
        if not self.issue_id or self.issue_id.strip() != self.issue_id:
            raise ValueError("truth preimage issue_id must be non-empty and stripped")
        if self.horizon_days not in {30, 90}:
            raise ValueError("truth preimage horizon must be 30 or 90 days")
        if not isinstance(self.raw_source_bytes, bytes):
            raise TypeError("truth source preimage must be bytes")
        member_ids = [
            event_id for cluster in self.clusters for event_id in cluster.member_event_ids
        ]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("truth cluster artifacts may not reuse one member event")

    @classmethod
    def from_clusters(
        cls,
        *,
        issue_id: str,
        horizon_days: Horizon,
        raw_source_bytes: bytes,
        clusters: tuple[TargetCluster, ...],
    ) -> TruthScientificPreimage:
        return cls(
            issue_id=issue_id,
            horizon_days=horizon_days,
            raw_source_bytes=raw_source_bytes,
            clusters=tuple(ClusterArtifactPreimage.from_cluster(cluster) for cluster in clusters),
        )

    @property
    def cluster_assignment_sha256(self) -> str:
        return cluster_assignment_sha256(self.clusters)


@dataclass(frozen=True, slots=True)
class ScientificPreimageStore:
    """Exact typed sidecars for every forecast and every available mature truth."""

    forecasts_by_issue_id: Mapping[str, ForecastScientificPreimage]
    truths_by_issue_horizon: Mapping[tuple[str, int], TruthScientificPreimage]

    def __post_init__(self) -> None:
        forecasts = dict(self.forecasts_by_issue_id)
        truths = dict(self.truths_by_issue_horizon)
        if any(key != value.issue_id for key, value in forecasts.items()):
            raise ValueError("forecast preimage map key differs from its issue identity")
        if any(key != (value.issue_id, value.horizon_days) for key, value in truths.items()):
            raise ValueError("truth preimage map key differs from its issue/horizon identity")
        object.__setattr__(self, "forecasts_by_issue_id", MappingProxyType(forecasts))
        object.__setattr__(self, "truths_by_issue_horizon", MappingProxyType(truths))


@dataclass(frozen=True, slots=True)
class MatureTruthRecomputation:
    """Exact raw-to-score result for one mature synthetic truth snapshot."""

    events: tuple[SyntheticEvent, ...]
    clusters: tuple[TargetCluster, ...]
    scores: ScoreSummary
    cluster_assignment_sha256: str
    ordered_cluster_registry_sha256: str


def recompute_mature_truth_snapshot(
    raw_source_bytes: bytes,
    forecast: DualModelForecast,
    *,
    horizon_days: Horizon,
    truth_fetched_at_utc: datetime,
) -> MatureTruthRecomputation:
    """Rebuild events, clusters and paired scores from exact mature source bytes."""

    if not isinstance(forecast, DualModelForecast):
        raise TypeError("forecast must be a DualModelForecast")
    if horizon_days not in {30, 90}:
        raise ValueError("truth horizon must be 30 or 90 days")
    fetched = _utc(truth_fetched_at_utc, label="truth_fetched_at_utc")
    events = _decode_catalog_snapshot(
        raw_source_bytes,
        role="truth",
        issue_id=forecast.issue_id,
        scheduled_issue_time_utc=forecast.scheduled_issue_time_utc,
        grid=forecast.grid,
        horizon_days=horizon_days,
        truth_fetched_at_utc=fetched,
    )
    if any(event.available_at_utc > fetched for event in events):
        raise ValueError("truth source snapshot contains a revision unavailable at truth fetch")
    clusters = cluster_target_events(
        events,
        issue_id=forecast.issue_id,
        issue_time_utc=forecast.scheduled_issue_time_utc,
        horizon_days=horizon_days,
        truth_fetched_at_utc=fetched,
        grid=forecast.grid,
    )
    scores = score_clusters(forecast, clusters, horizon_days=horizon_days)
    artifacts = tuple(ClusterArtifactPreimage.from_cluster(cluster) for cluster in clusters)
    return MatureTruthRecomputation(
        events=events,
        clusters=clusters,
        scores=scores,
        cluster_assignment_sha256=cluster_assignment_sha256(artifacts),
        ordered_cluster_registry_sha256=ordered_cluster_registry_sha256(scores.scores),
    )


def _record_type(record: Mapping[str, object]) -> object:
    return record.get("record_type")


def _record_issue_id(record: Mapping[str, object], *, label: str) -> str:
    value = record.get("issue_id")
    if not isinstance(value, str):
        raise ValueError(f"{label} issue_id must be a string")
    return value


def _model_records(record: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = record.get("forecasts")
    if not isinstance(raw, list):
        raise ValueError("forecast record must contain the two model rows")
    rows = {row.get("model_id"): row for row in raw if isinstance(row, Mapping)}
    if set(rows) != {"B0", "B0_R30"}:
        raise ValueError("forecast record model rows are incomplete")
    return cast(dict[str, Mapping[str, object]], rows)


def _same_float(left: object, right: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, int | float)
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-9)
    )


def _validate_forecast_preimage(
    record: Mapping[str, object],
    preimage: ForecastScientificPreimage,
) -> DualModelForecast:
    issue_id = _record_issue_id(record, label="forecast")
    if preimage.issue_id != issue_id:
        raise ValueError("forecast preimage belongs to another issue")
    if record.get("source_snapshot_sha256") != source_snapshot_sha256(preimage.raw_source_bytes):
        raise ValueError("forecast raw source bytes differ from the sealed source snapshot")
    scheduled = _parse_utc(record.get("scheduled_issue_time_utc"), label="scheduled_issue_time_utc")
    query_cutoff = _parse_utc(record.get("query_cutoff_utc"), label="query_cutoff_utc")
    events = _decode_catalog_snapshot(
        preimage.raw_source_bytes,
        role="forecast",
        issue_id=issue_id,
        scheduled_issue_time_utc=scheduled,
        grid=preimage.grid,
        horizon_days=None,
        truth_fetched_at_utc=None,
    )
    if any(
        event.origin_time_utc > query_cutoff or event.available_at_utc > query_cutoff
        for event in events
    ):
        raise ValueError("forecast source snapshot contains a post-Q event or revision")
    recomputed = build_dual_model_forecast(
        events,
        preimage.grid,
        issue_id=issue_id,
        scheduled_issue_time_utc=scheduled,
    )
    expected_parts = (
        (recomputed.B0, recomputed.B0_alarm),
        (recomputed.B0_R30, recomputed.B0_R30_alarm),
    )
    rows = _model_records(record)
    for saved, (surface, alarm) in zip(preimage.models, expected_parts, strict=True):
        expected_saved = ModelArtifactPreimage.from_parts(surface, alarm)
        if saved != expected_saved:
            raise ValueError(
                "saved forecast surface, ranking, or alarm mask differs from recomputation"
            )
        row = rows[saved.model_id]
        if row.get("relative_intensity_grid_sha256") != surface.sha256:
            raise ValueError("forecast relative-intensity hash differs from recomputation")
        if row.get("alarm_ranking_sha256") != alarm.ranking_sha256:
            raise ValueError("forecast alarm ranking hash differs from recomputation")
        if row.get("alarm_mask_sha256") != alarm.mask_sha256:
            raise ValueError("forecast alarm mask hash differs from recomputation")
        if not _same_float(row.get("actual_alarm_area_km2"), alarm.actual_area_km2):
            raise ValueError("forecast alarm area differs from recomputation")
    next_area = recomputed.B0_R30_alarm.next_complete_cell_area_km2
    if next_area is None:
        raise ValueError("recomputed challenger alarm lacks its next complete cell")
    scalar_checks = (
        ("B0_reference_area_km2", recomputed.B0_reference_area_km2),
        ("B0_R30_next_complete_cell_area_km2", next_area),
        ("actual_area_difference_km2", recomputed.actual_area_difference_km2),
    )
    if any(not _same_float(record.get(name), expected) for name, expected in scalar_checks):
        raise ValueError("forecast paired-area fields differ from recomputation")
    return recomputed


def _truth_key(record: Mapping[str, object]) -> tuple[str, int]:
    issue_id = _record_issue_id(record, label="truth")
    horizon = record.get("horizon_days")
    if type(horizon) is not int or horizon not in {30, 90}:
        raise ValueError("truth horizon must be 30 or 90 days")
    return issue_id, horizon


def _validate_truth_preimage(
    record: Mapping[str, object],
    preimage: TruthScientificPreimage,
    forecast: DualModelForecast,
) -> tuple[ScoreSummary, set[str]]:
    issue_id, raw_horizon = _truth_key(record)
    horizon = cast(Horizon, raw_horizon)
    if (preimage.issue_id, preimage.horizon_days) != (issue_id, horizon):
        raise ValueError("truth preimage belongs to another issue or horizon")
    if record.get("source_snapshot_sha256") != source_snapshot_sha256(preimage.raw_source_bytes):
        raise ValueError("truth raw source bytes differ from the sealed source snapshot")
    fetched = _parse_utc(record.get("truth_fetched_at_utc"), label="truth_fetched_at_utc")
    recomputed = recompute_mature_truth_snapshot(
        preimage.raw_source_bytes,
        forecast,
        horizon_days=horizon,
        truth_fetched_at_utc=fetched,
    )
    expected_clusters = tuple(
        ClusterArtifactPreimage.from_cluster(cluster) for cluster in recomputed.clusters
    )
    if preimage.clusters != expected_clusters:
        raise ValueError("saved cluster members or representative differ from recomputation")
    if record.get("cluster_assignment_sha256") != preimage.cluster_assignment_sha256:
        raise ValueError("truth cluster-assignment hash differs from its exact preimage")
    target_count = sum(len(cluster.member_event_ids) for cluster in recomputed.clusters)
    if record.get("target_event_count") != target_count:
        raise ValueError("truth target-event count differs from recomputation")
    if record.get("independent_cluster_count") != len(recomputed.clusters):
        raise ValueError("truth independent-cluster count differs from recomputation")
    if record.get("exposure_cluster_registry_sha256") != (
        recomputed.ordered_cluster_registry_sha256
    ):
        raise ValueError("truth score registry differs from raw-source mechanical recomputation")
    member_ids = {
        event_id for cluster in recomputed.clusters for event_id in cluster.member_event_ids
    }
    return recomputed.scores, member_ids


def validate_scientific_record_chain(
    records: Sequence[Mapping[str, object]],
    schema: Mapping[str, object],
    *,
    preimages: ScientificPreimageStore,
) -> None:
    """Validate the public chain only after rebuilding every scientific result.

    Missing or surplus sidecars fail closed.  Caller-authored score rows are never
    accepted as evidence: the only registry passed to the existing chain validator
    is created here from the raw forecast/truth preimages.
    """

    if not isinstance(preimages, ScientificPreimageStore):
        raise TypeError("preimages must be a ScientificPreimageStore")
    for record in records:
        validate_record_against_schema(record, schema)
    forecast_records: dict[str, Mapping[str, object]] = {}
    mature_truth_records: dict[tuple[str, int], Mapping[str, object]] = {}
    for record in records:
        if _record_type(record) == "ForecastIssueRecord":
            issue_id = _record_issue_id(record, label="forecast")
            if issue_id in forecast_records:
                raise ValueError("scientific chain contains a duplicate forecast issue")
            forecast_records[issue_id] = record
        elif _record_type(record) == "TruthSnapshotRecord" and record.get("status") == (
            "mature_truth"
        ):
            key = _truth_key(record)
            if key in mature_truth_records:
                raise ValueError("scientific chain contains a duplicate mature truth")
            mature_truth_records[key] = record
    if set(preimages.forecasts_by_issue_id) != set(forecast_records):
        raise ValueError("forecast scientific preimages must match every and only forecast issue")
    if set(preimages.truths_by_issue_horizon) != set(mature_truth_records):
        raise ValueError("truth scientific preimages must match every and only mature truth")

    recomputed_forecasts = {
        issue_id: _validate_forecast_preimage(
            record,
            preimages.forecasts_by_issue_id[issue_id],
        )
        for issue_id, record in forecast_records.items()
    }
    score_registries: dict[str, list[dict[str, object]]] = {}
    seen_members_by_horizon: dict[int, set[str]] = {30: set(), 90: set()}
    for key, record in mature_truth_records.items():
        issue_id, horizon = key
        forecast = recomputed_forecasts.get(issue_id)
        if forecast is None:
            raise ValueError("mature truth lacks its exact recomputed forecast")
        scores, member_ids = _validate_truth_preimage(
            record,
            preimages.truths_by_issue_horizon[key],
            forecast,
        )
        if seen_members_by_horizon[horizon] & member_ids:
            raise ValueError("one truth event appears in multiple selected exposures")
        seen_members_by_horizon[horizon].update(member_ids)
        registry_sha = ordered_cluster_registry_sha256(scores.scores)
        rows = [score.as_mapping() for score in scores.scores]
        previous = score_registries.get(registry_sha)
        if previous is not None and previous != rows:
            raise ValueError("one score-registry hash resolves to different preimages")
        score_registries[registry_sha] = rows
    validate_record_chain(
        records,
        schema,
        score_registries_by_sha256=score_registries,
    )


__all__ = [
    "ClusterArtifactPreimage",
    "ForecastScientificPreimage",
    "MatureTruthRecomputation",
    "ModelArtifactPreimage",
    "ScientificPreimageStore",
    "TruthScientificPreimage",
    "build_catalog_snapshot_bytes",
    "cluster_assignment_sha256",
    "grid_preimage_sha256",
    "recompute_mature_truth_snapshot",
    "source_snapshot_sha256",
    "validate_scientific_record_chain",
]
