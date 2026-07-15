"""Self-contained local spatial explorer for stage-4 forecast fields.

Prospective forecast fields, retrospective forecast fields, retrospective
target overlays, and display-only context are serialized into four physically
separate JSON payloads.
The default target-blind view never parses the target-overlay payload.  This
module performs no file or target-catalogue read; callers provide already
authorized in-memory arrays.
"""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import shape

from seismoflux.anomaly_increment.contracts import (
    FloatArray,
    canonical_mapping_sha256,
    readonly_float_vector,
)

MagnitudeBin = Literal["M5_6", "M6_plus"]
ModelVariant = Literal["background_no_increment", "coverage_only", "snapshot", "dynamic"]
ForecastStatus = Literal[
    "retrospective_evaluation",
    "retrospective_generated_target_blind_shadow",
    "genuine_prospective_archive",
]
EvidenceStatus = Literal[
    "passed",
    "failed",
    "evidence_insufficient",
    "not_evaluated_target_blind",
]
DisplayRole = Literal["primary", "previous_context"]
ContextLayerKind = Literal[
    "national_boundary",
    "provincial_boundary",
    "fault",
    "tectonic_block_boundary",
]

_FORBIDDEN_PROSPECTIVE_KEYS = (
    "covered_by",
    "event_id",
    "hit",
    "information_gain",
    "recall",
    "score",
    "target",
)
_LOWERCASE_SHA256 = frozenset("0123456789abcdef")


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in _LOWERCASE_SHA256 for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _unique_ids(
    values: Sequence[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    result = tuple(_identifier(value, label=label) for value in values)
    if (not allow_empty and not result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique" + ("" if allow_empty else " and non-empty"))
    return result


def _positive_finite(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return resolved


def _nonnegative_finite(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be non-negative and finite")
    return resolved


def _coordinates(longitude: object, latitude: object) -> tuple[FloatArray, FloatArray]:
    lon = readonly_float_vector("longitude", longitude, allow_empty=True)
    lat = readonly_float_vector("latitude", latitude, allow_empty=True)
    if lon.shape != lat.shape:
        raise ValueError("longitude and latitude must align")
    if np.any((lon < -180.0) | (lon > 180.0)) or np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("dashboard coordinates are outside geographic bounds")
    return lon, lat


def _boolean_vector(value: object, *, size: int) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_) or raw.shape != (size,):
        raise TypeError(f"alarm_selected must be a boolean vector with shape ({size},)")
    output = np.array(raw, dtype=np.bool_, copy=True)
    output.setflags(write=False)
    return output


def _status_label(status: ForecastStatus) -> str:
    if status == "retrospective_evaluation":
        return "回溯评估（含独立目标叠加）"
    if status == "retrospective_generated_target_blind_shadow":
        return "回溯生成的目标盲影子预测"
    return "真正前瞻归档"


def _evidence_label(status: EvidenceStatus) -> str:
    return {
        "passed": "通过",
        "failed": "未通过",
        "evidence_insufficient": "证据不足",
        "not_evaluated_target_blind": "目标盲帧不携带回溯证据",
    }[status]


@dataclass(frozen=True, slots=True)
class DisplayContextLayer:
    """Display-only geometry that is forbidden from model or candidate generation."""

    layer_id: str
    label: str
    layer_kind: ContextLayerKind
    geometry_geojson: Mapping[str, object]
    source_content_sha256: str
    role: Literal["display_only_forbidden_from_model_or_candidate_generation"] = (
        "display_only_forbidden_from_model_or_candidate_generation"
    )

    def __post_init__(self) -> None:
        _identifier(self.layer_id, label="display context layer_id")
        _identifier(self.label, label="display context label")
        _sha256(self.source_content_sha256, label="display context source_content_sha256")
        if self.layer_kind not in {
            "national_boundary",
            "provincial_boundary",
            "fault",
            "tectonic_block_boundary",
        }:
            raise ValueError("display context layer kind changed")
        if self.role != "display_only_forbidden_from_model_or_candidate_generation":
            raise ValueError("display context escaped its display-only role")
        object.__setattr__(
            self,
            "geometry_geojson",
            _validated_context_geometry(self.geometry_geojson),
        )


@dataclass(frozen=True, slots=True)
class DisplayStudyArea:
    """Frozen target-independent study-area geometry used only for display/clip."""

    study_area_id: str
    geometry_geojson: Mapping[str, object]
    source_content_sha256: str
    role: Literal["frozen_target_independent_display_clip"] = (
        "frozen_target_independent_display_clip"
    )

    def __post_init__(self) -> None:
        _identifier(self.study_area_id, label="display study_area_id")
        _sha256(self.source_content_sha256, label="display study-area source_content_sha256")
        if self.role != "frozen_target_independent_display_clip":
            raise ValueError("study-area display geometry escaped its target-independent role")
        object.__setattr__(
            self,
            "geometry_geojson",
            _validated_study_area(self.geometry_geojson),
        )

    @property
    def geometry_content_sha256(self) -> str:
        return canonical_mapping_sha256(dict(self.geometry_geojson))


@dataclass(frozen=True, slots=True)
class ForecastGridFrame:
    """One target-free conditional-relative-intensity surface and alarm prefix."""

    issue_date: str
    magnitude_bin: MagnitudeBin
    horizon_days: int
    model_variant: ModelVariant
    model_version: str
    training_sample_size: int
    grid_id: str
    cell_size_km: float
    alarm_threshold_rank_percentile: float
    selected_alarm_area_km2: float
    displayed_domain_area_km2: float
    cell_ids: tuple[str, ...]
    longitude: FloatArray
    latitude: FloatArray
    relative_strength: FloatArray
    rank_percentile: FloatArray
    alarm_selected: NDArray[np.bool_]
    forecast_status: ForecastStatus
    evidence_status: EvidenceStatus
    display_role: DisplayRole
    knowledge_cutoff_utc: datetime
    source_model_fingerprint_sha256: str
    source_result_fingerprint_sha256: str | None

    def __post_init__(self) -> None:
        for label in ("issue_date", "model_version", "grid_id"):
            value = _identifier(getattr(self, label), label=label)
            object.__setattr__(self, label, value)
        try:
            issue_date = date.fromisoformat(self.issue_date)
        except ValueError as exc:
            raise ValueError("forecast issue_date must be an ISO calendar date") from exc
        cutoff = self.knowledge_cutoff_utc
        if not isinstance(cutoff, datetime) or cutoff.tzinfo is None:
            raise ValueError("knowledge_cutoff_utc must be a timezone-aware datetime")
        if cutoff.utcoffset() != timedelta(0):
            raise ValueError("knowledge_cutoff_utc must already be UTC")
        issue_time_utc = datetime.combine(
            issue_date,
            time.min,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ).astimezone(UTC)
        if cutoff > issue_time_utc:
            raise ValueError("knowledge cutoff cannot pass the UTC+8 issue time")
        object.__setattr__(self, "knowledge_cutoff_utc", cutoff.astimezone(UTC))
        _sha256(
            self.source_model_fingerprint_sha256,
            label="source_model_fingerprint_sha256",
        )
        if self.forecast_status not in {
            "retrospective_evaluation",
            "retrospective_generated_target_blind_shadow",
            "genuine_prospective_archive",
        }:
            raise ValueError("forecast frame status changed")
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("forecast frame magnitude bin changed")
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("forecast frame horizon changed")
        if self.model_variant not in {
            "background_no_increment",
            "coverage_only",
            "snapshot",
            "dynamic",
        }:
            raise ValueError("forecast frame model variant changed")
        if self.evidence_status not in {
            "passed",
            "failed",
            "evidence_insufficient",
            "not_evaluated_target_blind",
        }:
            raise ValueError("forecast frame evidence status changed")
        if self.display_role not in {"primary", "previous_context"}:
            raise ValueError("forecast frame display role changed")
        if self.forecast_status == "retrospective_evaluation":
            if self.evidence_status == "not_evaluated_target_blind":
                raise ValueError("retrospective frame requires real evaluation evidence status")
            if self.source_result_fingerprint_sha256 is None:
                raise ValueError("retrospective frame requires a result fingerprint")
            _sha256(
                self.source_result_fingerprint_sha256,
                label="source_result_fingerprint_sha256",
            )
        else:
            if self.evidence_status != "not_evaluated_target_blind":
                raise ValueError("target-blind frame cannot carry target-derived evidence status")
            if self.source_result_fingerprint_sha256 is not None:
                raise ValueError(
                    "target-blind frame cannot carry a target-derived result fingerprint"
                )
        if (
            isinstance(self.training_sample_size, bool)
            or not isinstance(self.training_sample_size, int)
            or self.training_sample_size < 0
        ):
            raise ValueError("training_sample_size must be a non-negative integer")
        cell_size_km = _positive_finite(self.cell_size_km, label="cell_size_km")
        alarm_area = _nonnegative_finite(
            self.selected_alarm_area_km2,
            label="selected_alarm_area_km2",
        )
        domain_area = _positive_finite(
            self.displayed_domain_area_km2,
            label="displayed_domain_area_km2",
        )
        if alarm_area > domain_area + 1.0e-9:
            raise ValueError("selected alarm area cannot exceed the displayed domain")
        threshold = _nonnegative_finite(
            self.alarm_threshold_rank_percentile,
            label="alarm_threshold_rank_percentile",
        )
        if threshold > 100.0:
            raise ValueError("alarm threshold rank percentile must lie in [0, 100]")

        cell_ids = _unique_ids(self.cell_ids, label="forecast frame cell IDs")
        lon, lat = _coordinates(self.longitude, self.latitude)
        strength = readonly_float_vector("relative_strength", self.relative_strength)
        rank = readonly_float_vector("rank_percentile", self.rank_percentile)
        selected = _boolean_vector(self.alarm_selected, size=len(cell_ids))
        if any(values.shape != (len(cell_ids),) for values in (lon, lat, strength, rank)):
            raise ValueError("forecast frame arrays must have one value per cell")
        if np.any(strength < 0.0):
            raise ValueError("conditional relative strength must be non-negative")
        if np.any((rank < 0.0) | (rank > 100.0)):
            raise ValueError("rank percentiles must lie in [0, 100]")
        if np.any(selected) and alarm_area <= 0.0:
            raise ValueError("a non-empty alarm prefix requires positive selected area")
        if not np.any(selected) and alarm_area != 0.0:
            raise ValueError("an empty alarm prefix requires zero selected area")
        if np.any(selected) and np.any(rank[selected] < threshold):
            raise ValueError("selected alarm cells cannot fall below the declared rank threshold")
        if (
            np.any(selected)
            and np.any(~selected)
            and float(np.min(rank[selected])) < float(np.max(rank[~selected]))
        ):
            raise ValueError("alarm selection must be a highest-rank deterministic prefix")
        object.__setattr__(self, "cell_size_km", cell_size_km)
        object.__setattr__(self, "alarm_threshold_rank_percentile", threshold)
        object.__setattr__(self, "selected_alarm_area_km2", alarm_area)
        object.__setattr__(self, "displayed_domain_area_km2", domain_area)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "longitude", lon)
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "relative_strength", strength)
        object.__setattr__(self, "rank_percentile", rank)
        object.__setattr__(self, "alarm_selected", selected)


@dataclass(frozen=True, slots=True)
class RetrospectiveTargetFrame:
    """Target-bearing overlay kept outside every forecast payload."""

    issue_date: str
    magnitude_bin: MagnitudeBin
    horizon_days: int
    event_ids: tuple[str, ...]
    longitude: FloatArray
    latitude: FloatArray
    covered_by_600000_km2: tuple[bool, ...]
    hit_event_ids_at_600000_km2: tuple[str, ...]
    source_catalog_content_sha256: str
    source_result_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.issue_date, label="retrospective issue_date")
        _sha256(
            self.source_catalog_content_sha256,
            label="source_catalog_content_sha256",
        )
        _sha256(
            self.source_result_fingerprint_sha256,
            label="source_result_fingerprint_sha256",
        )
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("retrospective magnitude bin changed")
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("retrospective horizon changed")
        event_ids = _unique_ids(
            self.event_ids,
            label="retrospective target event_ids",
            allow_empty=True,
        )
        hits = _unique_ids(
            self.hit_event_ids_at_600000_km2,
            label="retrospective hit event_ids",
            allow_empty=True,
        )
        if not set(hits) <= set(event_ids):
            raise ValueError("retrospective hit IDs must be target overlay events")
        lon, lat = _coordinates(self.longitude, self.latitude)
        covered = tuple(self.covered_by_600000_km2)
        if lon.size != len(event_ids) or len(covered) != len(event_ids):
            raise ValueError("retrospective target overlay arrays must align")
        if any(not isinstance(value, bool) for value in covered):
            raise ValueError("retrospective target coverage flags must be booleans")
        if covered != tuple(event_id in set(hits) for event_id in event_ids):
            raise ValueError("retrospective target flags differ from the real hit event IDs")
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "longitude", lon)
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "covered_by_600000_km2", covered)
        object.__setattr__(self, "hit_event_ids_at_600000_km2", hits)


def _frame_identity(frame: ForecastGridFrame) -> tuple[str, MagnitudeBin, int, ModelVariant, str]:
    return (
        frame.issue_date,
        frame.magnitude_bin,
        frame.horizon_days,
        frame.model_variant,
        frame.model_version,
    )


def _frame_mapping(frame: ForecastGridFrame) -> dict[str, object]:
    payload: dict[str, object] = {
        "alarm_selected": [bool(value) for value in frame.alarm_selected],
        "alarm_threshold_rank_percentile": frame.alarm_threshold_rank_percentile,
        "cell_ids": list(frame.cell_ids),
        "cell_size_km": frame.cell_size_km,
        "displayed_domain_area_km2": frame.displayed_domain_area_km2,
        "evidence_status": frame.evidence_status,
        "evidence_status_label": _evidence_label(frame.evidence_status),
        "display_role": frame.display_role,
        "forecast_status": frame.forecast_status,
        "forecast_status_label": _status_label(frame.forecast_status),
        "grid_id": frame.grid_id,
        "horizon_days": frame.horizon_days,
        "issue_date": frame.issue_date,
        "knowledge_cutoff_utc": frame.knowledge_cutoff_utc.isoformat().replace("+00:00", "Z"),
        "latitude": [round(float(value), 5) for value in frame.latitude],
        "longitude": [round(float(value), 5) for value in frame.longitude],
        "magnitude_bin": frame.magnitude_bin,
        "model_variant": frame.model_variant,
        "model_version": frame.model_version,
        "rank_percentile": [round(float(value), 4) for value in frame.rank_percentile],
        "relative_strength": [round(float(value), 8) for value in frame.relative_strength],
        "selected_alarm_area_km2": frame.selected_alarm_area_km2,
        "source_model_fingerprint_sha256": frame.source_model_fingerprint_sha256,
        "training_sample_size": frame.training_sample_size,
    }
    if frame.source_result_fingerprint_sha256 is not None:
        payload["source_result_fingerprint_sha256"] = frame.source_result_fingerprint_sha256
    return payload


def _mapping_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(
            [str(key) for key in value]
            + [nested for item in value.values() for nested in _mapping_keys(item)]
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(nested for item in value for nested in _mapping_keys(item))
    return ()


def _assert_target_free_forecast_payload(value: object) -> None:
    for key in _mapping_keys(value):
        folded = key.casefold()
        if any(token in folded for token in _FORBIDDEN_PROSPECTIVE_KEYS):
            raise ValueError(f"forecast spatial payload contains target-derived field: {key}")


def _position(value: object) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) < 2:
        raise ValueError("study-area GeoJSON positions must contain longitude and latitude")
    lon, lat = value[0], value[1]
    if (
        isinstance(lon, bool)
        or isinstance(lat, bool)
        or not isinstance(lon, int | float)
        or not isinstance(lat, int | float)
        or not math.isfinite(float(lon))
        or not math.isfinite(float(lat))
        or not -180.0 <= float(lon) <= 180.0
        or not -90.0 <= float(lat) <= 90.0
    ):
        raise ValueError("study-area GeoJSON contains an invalid geographic position")
    return [float(lon), float(lat)]


def _ring(value: object) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("study-area GeoJSON rings must be sequences")
    positions = [_position(item) for item in value]
    if len(positions) < 4 or positions[0] != positions[-1]:
        raise ValueError("study-area GeoJSON rings must be closed with at least four positions")
    return positions


def _polygon(value: object) -> list[list[list[float]]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise ValueError("study-area GeoJSON polygons must contain at least one ring")
    return [_ring(item) for item in value]


def _line(value: object) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("display context lines must be coordinate sequences")
    positions = [_position(item) for item in value]
    if len(positions) < 2:
        raise ValueError("display context lines require at least two positions")
    return positions


def _validated_context_geometry(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("display context geometry must be a GeoJSON mapping")
    geometry_type = value.get("type")
    coordinates = value.get("coordinates")
    if geometry_type == "LineString":
        normalized: object = _line(coordinates)
    elif geometry_type == "MultiLineString":
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, str | bytes)
            or not coordinates
        ):
            raise ValueError("display context MultiLineString must contain lines")
        normalized = [_line(item) for item in coordinates]
    elif geometry_type == "Polygon":
        normalized = _polygon(coordinates)
    elif geometry_type == "MultiPolygon":
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, str | bytes)
            or not coordinates
        ):
            raise ValueError("display context MultiPolygon must contain polygons")
        normalized = [_polygon(item) for item in coordinates]
    else:
        raise ValueError(
            "display context geometry must be LineString, MultiLineString, Polygon, or MultiPolygon"
        )
    return {"coordinates": normalized, "type": geometry_type}


def _validated_study_area(value: Mapping[str, object]) -> dict[str, object]:
    geometry_type = value.get("type")
    coordinates = value.get("coordinates")
    if geometry_type == "Polygon":
        normalized: object = _polygon(coordinates)
    elif geometry_type == "MultiPolygon":
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, str | bytes)
            or not coordinates
        ):
            raise ValueError("study-area MultiPolygon must contain at least one polygon")
        normalized = [_polygon(item) for item in coordinates]
    else:
        raise ValueError("study-area GeoJSON must be Polygon or MultiPolygon geometry")
    result = {"coordinates": normalized, "type": geometry_type}
    geometry = shape(result)
    if geometry.is_empty or not geometry.is_valid or geometry.area <= 0.0:
        raise ValueError("study-area GeoJSON must be valid and positive-area")
    return result


def _script_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _validated_frame_mappings(
    frames: tuple[ForecastGridFrame, ...],
    *,
    label: str,
) -> list[dict[str, object]]:
    if not frames:
        raise ValueError(f"spatial dashboard requires {label} fields")
    identities = tuple(_frame_identity(frame) for frame in frames)
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} forecast frames contain duplicate filter identities")
    model_fingerprints = {item.source_model_fingerprint_sha256 for item in frames}
    if len(model_fingerprints) != 1:
        raise ValueError(f"{label} forecast frames mix fitted models")
    result_fingerprints = {item.source_result_fingerprint_sha256 for item in frames}
    if label == "retrospective":
        if None in result_fingerprints or len(result_fingerprints) != 1:
            raise ValueError("retrospective forecast frames mix scoring results")
    elif result_fingerprints != {None}:
        raise ValueError("target-blind forecast frames carry a scoring-result fingerprint")
    return [_frame_mapping(frame) for frame in frames]


def build_local_spatial_dashboard_html(
    *,
    study_area: DisplayStudyArea,
    retrospective_fields: tuple[ForecastGridFrame, ...],
    prospective_fields: tuple[ForecastGridFrame, ...],
    retrospective_targets: tuple[RetrospectiveTargetFrame, ...],
    display_context_layers: tuple[DisplayContextLayer, ...] = (),
) -> str:
    """Render a local zoomable map from already-authorized in-memory payloads."""

    if any(item.forecast_status != "retrospective_evaluation" for item in retrospective_fields):
        raise ValueError("retrospective dashboard fields require retrospective status")
    prospective_statuses = {item.forecast_status for item in prospective_fields}
    if not prospective_statuses or prospective_statuses == {"retrospective_evaluation"}:
        raise ValueError("target-blind dashboard fields require a forecast status")
    if len(prospective_statuses) != 1 or "retrospective_evaluation" in prospective_statuses:
        raise ValueError("target-blind dashboard fields mix archive statuses")

    if not isinstance(study_area, DisplayStudyArea):
        raise TypeError("spatial dashboard requires a frozen target-independent study area")
    retro_forecasts = _validated_frame_mappings(retrospective_fields, label="retrospective")
    prospective = _validated_frame_mappings(prospective_fields, label="target-blind")
    model_fingerprints = {
        item.source_model_fingerprint_sha256
        for item in (*retrospective_fields, *prospective_fields)
    }
    if len(model_fingerprints) != 1:
        raise ValueError("retrospective and target-blind fields use different fitted models")
    retrospective_result_fingerprints = {
        item.source_result_fingerprint_sha256 for item in retrospective_fields
    }
    if None in retrospective_result_fingerprints or len(retrospective_result_fingerprints) != 1:
        raise ValueError("retrospective fields do not bind exactly one scoring result")
    result_fingerprint = next(iter(retrospective_result_fingerprints))
    prospective_payload = {"fields": prospective}
    retrospective_fields_payload = {"fields": retro_forecasts}
    _assert_target_free_forecast_payload(prospective_payload)
    _assert_target_free_forecast_payload(retrospective_fields_payload)

    context_layers = tuple(display_context_layers)
    layer_ids = tuple(item.layer_id for item in context_layers)
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("display context layer IDs must be unique")
    context_payload = {
        "layers": [
            {
                "geometry": dict(item.geometry_geojson),
                "label": item.label,
                "layer_id": item.layer_id,
                "layer_kind": item.layer_kind,
                "role": item.role,
                "source_content_sha256": item.source_content_sha256,
            }
            for item in context_layers
        ],
        "status_label": (
            "上下文层未提供"
            if not context_layers
            else "显示专用上下文：" + "、".join(item.label for item in context_layers)
        ),
        "study_area": dict(study_area.geometry_geojson),
        "study_area_geometry_sha256": study_area.geometry_content_sha256,
        "study_area_id": study_area.study_area_id,
        "study_area_role": study_area.role,
        "study_area_source_content_sha256": study_area.source_content_sha256,
    }

    targets = tuple(retrospective_targets)
    target_keys = tuple(
        (item.issue_date, item.magnitude_bin, item.horizon_days) for item in targets
    )
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("retrospective target overlays contain duplicate filter identities")
    retrospective_keys = {
        (item.issue_date, item.magnitude_bin, item.horizon_days) for item in retrospective_fields
    }
    if not set(target_keys).issubset(retrospective_keys):
        raise ValueError("retrospective target overlay has no matching forecast frame")
    if any(item.source_result_fingerprint_sha256 != result_fingerprint for item in targets):
        raise ValueError("retrospective targets use another scoring result")
    catalog_hashes = {item.source_catalog_content_sha256 for item in targets}
    if len(catalog_hashes) > 1:
        raise ValueError("retrospective targets mix authorized catalogues")
    target_payload = {
        "targets": [
            {
                "covered_by_600000_km2": list(item.covered_by_600000_km2),
                "event_ids": list(item.event_ids),
                "hit_event_ids_at_600000_km2": list(item.hit_event_ids_at_600000_km2),
                "horizon_days": item.horizon_days,
                "issue_date": item.issue_date,
                "latitude": [round(float(value), 5) for value in item.latitude],
                "longitude": [round(float(value), 5) for value in item.longitude],
                "magnitude_bin": item.magnitude_bin,
                "source_catalog_content_sha256": item.source_catalog_content_sha256,
                "source_result_fingerprint_sha256": item.source_result_fingerprint_sha256,
            }
            for item in targets
        ]
    }
    prospective_json = _script_json(prospective_payload)
    retrospective_fields_json = _script_json(retrospective_fields_payload)
    target_json = _script_json(target_payload)
    context_json = _script_json(context_payload)
    prospective_label = _status_label(next(iter(prospective_statuses)))

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SeismoFlux 条件相对强度空间图</title><style>
:root{{color-scheme:light dark;--bg:#f7f8fb;--fg:#172033;--muted:#5d6878;--card:#fff;--line:#657181;--focus:#2469a6;--alarm:#172033;--previous:#7b2cbf;--target-covered:#172033;--target-open:#fff;--context:#75634d;--map-outside:#fff}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111723;--fg:#edf1f7;--muted:#b5bfcc;--card:#1b2433;--line:#91a0b4;--focus:#75b8ee;--alarm:#fff;--previous:#c77dff;--target-covered:#fff;--target-open:#111723;--context:#d2b990;--map-outside:#111723}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1280px;margin:auto;padding:18px}}
h1{{font-size:1.45rem;font-weight:500;margin:0 0 4px}}.muted{{color:var(--muted)}}.controls{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:16px 0}}
label{{display:grid;gap:4px;font-size:.82rem;color:var(--muted)}}select{{width:100%;padding:8px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:5px}}
.map{{position:relative;background:var(--map-outside);border:1px solid var(--line);border-radius:8px;overflow:hidden}}canvas{{display:block;width:100%;height:auto;touch-action:none;background:var(--map-outside)}}
.legend,.metadata{{display:flex;align-items:center;gap:8px 16px;flex-wrap:wrap;margin-top:10px;font-size:.86rem;color:var(--muted)}}.ramp{{width:180px;height:12px;background:linear-gradient(90deg,#2459a6,#6bb7d6,#f2d35c,#d73027);border:1px solid var(--line)}}
.alarm-key{{width:12px;height:12px;background:#f2d35c;border:2px solid var(--alarm);display:inline-block}}.detail{{position:absolute;left:10px;bottom:10px;background:color-mix(in srgb,var(--card) 92%,transparent);padding:7px 9px;border-radius:5px;font-size:.82rem;max-width:calc(100% - 20px)}}[hidden]{{display:none!important}}
@media(max-width:520px){{main{{padding:12px}}h1{{font-size:1.2rem}}.controls{{grid-template-columns:1fr 1fr}}.detail{{position:static;max-width:none;border-radius:0}}}}
</style></head><body><main data-classification="local_restricted_target_bearing_spatial_visualization"><h1>条件相对强度：空间回溯与目标盲预测</h1><div class="muted">颜色只表示同一期内的条件相对强弱与顺位，不能解释为绝对发生率估计。滚轮缩放，拖动平移。</div>
<div class="controls"><label>模式<select id="mode"><option value="prospective" selected>{prospective_label}</option><option value="retrospective">回溯评估</option></select></label><label>起报日<select id="issue"></select></label><label>震级档<select id="magnitude"></select></label><label>窗口<select id="horizon"></select></label><label>模型<select id="variant"></select></label><label>版本<select id="version"></select></label></div>
<div class="map"><canvas id="map" width="1200" height="690" role="img" aria-label="条件相对强度与报警格空间图"></canvas><div id="detail" class="detail"></div></div>
<div class="legend"><span>低顺位</span><span class="ramp" aria-hidden="true"></span><span>高顺位</span><span style="color:var(--previous)">▧ 上一期报警格（紫色虚线）</span><span><span class="alarm-key" aria-hidden="true"></span> 当前报警格（深色实线）</span><span id="targetLegend" hidden>● 回溯覆盖目标　○ 回溯未覆盖目标</span><span>— 显示专用上下文</span></div>
<div class="metadata"><span id="alarmMeta"></span><span id="peakMeta"></span><span id="previousMeta"></span><span id="gridMeta"></span><span id="sample"></span><span id="evidenceMeta"></span><span id="statusMeta"></span><span id="contextMeta"></span></div>
<p class="muted" id="interpretation">解读：优先查看受控报警面积内的高顺位格；限制：条件相对强度不能替代绝对发生率估计。</p>
<script type="application/json" id="prospective-spatial-payload">{prospective_json}</script>
<script type="application/json" id="retrospective-spatial-fields-payload">{retrospective_fields_json}</script>
<script type="application/json" id="retrospective-target-overlay-payload">{target_json}</script>
<script type="application/json" id="display-context-payload">{context_json}</script>
<script>(()=>{{"use strict";
const pros=JSON.parse(document.getElementById("prospective-spatial-payload").textContent),displayContext=JSON.parse(document.getElementById("display-context-payload").textContent);let retroFieldsCache=null,targetCache=null;
function retrospectiveFields(){{if(retroFieldsCache===null)retroFieldsCache=JSON.parse(document.getElementById("retrospective-spatial-fields-payload").textContent);return retroFieldsCache}}
function retrospectiveTargets(){{if(targetCache===null)targetCache=JSON.parse(document.getElementById("retrospective-target-overlay-payload").textContent);return targetCache}}
const $=id=>document.getElementById(id);const controls={{mode:$("mode"),issue:$("issue"),magnitude:$("magnitude"),horizon:$("horizon"),variant:$("variant"),version:$("version")}};const canvas=$("map"),ctx=canvas.getContext("2d");let zoom=1,panX=0,panY=0,drag=null;
function options(control,values){{const old=control.value;const unique=[...new Set(values.map(String))];control.replaceChildren(...unique.map(v=>{{const o=document.createElement("option");o.value=v;o.textContent=v;return o}}));if(unique.includes(old))control.value=old}}
function positions(g){{const out=[];const walk=v=>{{if(Array.isArray(v)&&v.length>=2&&typeof v[0]==="number")out.push(v);else if(Array.isArray(v))v.forEach(walk);else if(v&&typeof v==="object")Object.values(v).forEach(walk)}};walk(g.coordinates);return out}}
const outline=positions(displayContext.study_area);if(outline.length===0)throw new Error("研究区没有坐标");const xs=outline.map(p=>p[0]),ys=outline.map(p=>p[1]);const bounds={{minX:Math.min(...xs),maxX:Math.max(...xs),minY:Math.min(...ys),maxY:Math.max(...ys)}};$("contextMeta").textContent=displayContext.status_label;
function allModeFields(){{return controls.mode.value==="prospective"?pros.fields:retrospectiveFields().fields}}
function activeFields(){{return allModeFields().filter(x=>x.display_role==="primary")}}
function sync(){{const fields=activeFields();options(controls.issue,fields.map(x=>x.issue_date));let rows=fields.filter(x=>x.issue_date===controls.issue.value);options(controls.magnitude,rows.map(x=>x.magnitude_bin));rows=rows.filter(x=>x.magnitude_bin===controls.magnitude.value);options(controls.horizon,rows.map(x=>x.horizon_days));rows=rows.filter(x=>String(x.horizon_days)===controls.horizon.value);options(controls.variant,rows.map(x=>x.model_variant));rows=rows.filter(x=>x.model_variant===controls.variant.value);options(controls.version,rows.map(x=>x.model_version));$("targetLegend").hidden=controls.mode.value!=="retrospective";draw()}}
function frame(){{return activeFields().find(x=>x.issue_date===controls.issue.value&&x.magnitude_bin===controls.magnitude.value&&String(x.horizon_days)===controls.horizon.value&&x.model_variant===controls.variant.value&&x.model_version===controls.version.value)}}
function previousFrame(f){{const candidates=allModeFields().filter(x=>x.issue_date<f.issue_date&&x.magnitude_bin===f.magnitude_bin&&String(x.horizon_days)===String(f.horizon_days)&&x.model_variant===f.model_variant&&x.model_version===f.model_version).sort((a,b)=>a.issue_date.localeCompare(b.issue_date));return candidates.length?candidates[candidates.length-1]:null}}
function project(lon,lat){{const pad=28,dx=Math.max(bounds.maxX-bounds.minX,1e-9),dy=Math.max(bounds.maxY-bounds.minY,1e-9),s=Math.min((canvas.width-2*pad)/dx,(canvas.height-2*pad)/dy)*zoom,cx=(bounds.minX+bounds.maxX)/2,cy=(bounds.minY+bounds.maxY)/2;return[canvas.width/2+(lon-cx)*s+panX,canvas.height/2-(lat-cy)*s+panY]}}
function color(p){{const t=Math.max(0,Math.min(1,p/100)),stops=[[36,89,166],[107,183,214],[242,211,92],[215,48,39]],q=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(q)),a=q-i;return`rgb(${{Math.round(stops[i][0]*(1-a)+stops[i+1][0]*a)}},${{Math.round(stops[i][1]*(1-a)+stops[i+1][1]*a)}},${{Math.round(stops[i][2]*(1-a)+stops[i+1][2]*a)}})`}}
function polygons(g){{return g.type==="Polygon"?[g.coordinates]:g.coordinates}}
function traceStudyArea(){{ctx.beginPath();for(const polygon of polygons(displayContext.study_area))for(const ring of polygon)ring.forEach((p,i)=>{{const q=project(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)}}),ctx.closePath()}}
function contextPaths(g){{if(g.type==="LineString")return[g.coordinates];if(g.type==="MultiLineString")return g.coordinates;if(g.type==="Polygon")return g.coordinates;return g.coordinates.flat()}}
function drawContextLayers(){{ctx.save();ctx.strokeStyle=css("--context");ctx.lineWidth=1;for(const layer of displayContext.layers)for(const path of contextPaths(layer.geometry)){{ctx.beginPath();path.forEach((p,i)=>{{const q=project(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)}});ctx.stroke()}}ctx.restore()}}
function css(name){{return getComputedStyle(document.documentElement).getPropertyValue(name).trim()}}
function drawOutline(){{traceStudyArea();ctx.strokeStyle=css("--line");ctx.lineWidth=1.5;ctx.stroke()}}
function draw(){{const f=frame();ctx.fillStyle=css("--map-outside");ctx.fillRect(0,0,canvas.width,canvas.height);if(!f){{$("detail").textContent="当前筛选无可用帧";return}}const previous=previousFrame(f);ctx.save();traceStudyArea();ctx.clip("evenodd");drawContextLayers();if(previous){{ctx.strokeStyle=css("--previous");ctx.lineWidth=1.5;ctx.setLineDash([4,3]);for(let i=0;i<previous.longitude.length;i++)if(previous.alarm_selected[i]){{const p=project(previous.longitude[i],previous.latitude[i]),size=9;ctx.strokeRect(p[0]-size/2,p[1]-size/2,size,size)}}ctx.setLineDash([])}}for(let i=0;i<f.longitude.length;i++){{const p=project(f.longitude[i],f.latitude[i]),size=f.alarm_selected[i]?7:4;ctx.fillStyle=color(f.rank_percentile[i]);ctx.fillRect(p[0]-size/2,p[1]-size/2,size,size);if(f.alarm_selected[i]){{ctx.strokeStyle=css("--alarm");ctx.lineWidth=2;ctx.strokeRect(p[0]-size/2,p[1]-size/2,size,size)}}}}if(controls.mode.value==="retrospective"){{const t=retrospectiveTargets().targets.find(x=>x.issue_date===f.issue_date&&x.magnitude_bin===f.magnitude_bin&&String(x.horizon_days)===String(f.horizon_days));if(t)for(let i=0;i<t.longitude.length;i++){{const p=project(t.longitude[i],t.latitude[i]);ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.fillStyle=t.covered_by_600000_km2[i]?css("--target-covered"):css("--target-open");ctx.fill();ctx.strokeStyle=css("--target-covered");ctx.lineWidth=1.5;ctx.stroke()}}}}ctx.restore();drawOutline();$("alarmMeta").textContent=`报警面积 ${{Number(f.selected_alarm_area_km2).toLocaleString()}} / 展示域 ${{Number(f.displayed_domain_area_km2).toLocaleString()}} km² · 顺位阈值 ≥ ${{Number(f.alarm_threshold_rank_percentile).toFixed(1)}}`;$("peakMeta").textContent=`相对强度峰值 ${{f.relative_strength.reduce((a,b)=>Math.max(a,b),0).toFixed(2)}}`;$("previousMeta").textContent=previous?`上一期 ${{previous.issue_date}}`:`上一期：无上一期`;$("gridMeta").textContent=`网格 ${{f.grid_id}} · ${{Number(f.cell_size_km).toFixed(1)}} km`;$("sample").textContent=`样本 ${{f.training_sample_size}} · 模型 ${{f.model_version}}`;$("evidenceMeta").textContent=`证据等级 ${{f.evidence_status_label}}`;$("statusMeta").textContent=`${{f.forecast_status_label}} · 知识截止 ${{f.knowledge_cutoff_utc}}`;$("detail").textContent=`${{f.issue_date}} · ${{f.horizon_days}}天 · ${{f.magnitude_bin}} · ${{f.model_variant}} · 条件相对强度`}}
Object.values(controls).forEach(c=>c.addEventListener("change",sync));canvas.addEventListener("wheel",e=>{{e.preventDefault();zoom=Math.max(.75,Math.min(8,zoom*(e.deltaY<0?1.15:.87)));draw()}},{{passive:false}});canvas.addEventListener("pointerdown",e=>{{drag=[e.clientX,e.clientY,panX,panY];canvas.setPointerCapture(e.pointerId)}});canvas.addEventListener("pointermove",e=>{{if(!drag)return;panX=drag[2]+e.clientX-drag[0];panY=drag[3]+e.clientY-drag[1];draw()}});for(const eventName of ["pointerup","pointercancel"])canvas.addEventListener(eventName,()=>{{drag=null}});sync()
}})();</script></main></body></html>"""
    if 'charset="utf-8"' not in document.casefold():
        raise AssertionError("spatial dashboard must declare UTF-8")
    if len(document.encode("utf-8")) > 64 * 1024 * 1024:
        raise ValueError("spatial dashboard exceeds the 64 MiB local payload cap")
    return document


__all__ = [
    "ContextLayerKind",
    "DisplayContextLayer",
    "DisplayRole",
    "DisplayStudyArea",
    "EvidenceStatus",
    "ForecastGridFrame",
    "ForecastStatus",
    "MagnitudeBin",
    "ModelVariant",
    "RetrospectiveTargetFrame",
    "build_local_spatial_dashboard_html",
]
