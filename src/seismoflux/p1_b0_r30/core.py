"""Target-blind scientific primitives for the P1 ``B0`` versus ``B0_R30`` rehearsal.

The module accepts only caller-supplied, explicitly synthetic events.  It does not
locate catalogues, access the network, or interpret relative intensity as an
absolute earthquake probability.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pyproj import Geod

from seismoflux.data.common import canonical_json_bytes

FloatArray: TypeAlias = NDArray[np.float64]
ModelId: TypeAlias = Literal["B0", "R30", "B0_R30"]
IssueStatus: TypeAlias = Literal["on_time", "missed_issue"]
ReviewTrigger: TypeAlias = Literal["cluster_10", "cluster_20", "cluster_30", "time_36_months"]
ReviewDecision: TypeAlias = Literal[
    "continue_accumulation",
    "confirm_strong_prospective_improvement",
    "report_uncertain_at_final_review",
    "stop_B0_R30_retain_B0",
    "report_evidence_insufficient_at_final_review",
]

BANDWIDTH_KM = 75.0
RECENT_WINDOW_DAYS = 30
PRIMARY_HORIZON_DAYS: Literal[30] = 30
SECONDARY_HORIZON_DAYS: Literal[90] = 90
CLUSTER_MAX_TIME_DAYS = 30
CLUSTER_MAX_DISTANCE_KM = 75.0
GRID_CELL_SIZE_KM = 25.0
GRID_CELL_AREA_KM2 = 625.0
MAXIMUM_ALARM_AREA_KM2 = 600_000.0
MIXING_ALPHA = 0.25
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 147
BOOTSTRAP_LOWER_QUANTILE = 0.0083333333333333
BOOTSTRAP_UPPER_QUANTILE = 0.9916666666666667
HISTORICAL_START_UTC = datetime(1970, 1, 1, tzinfo=UTC)
LOCAL_CATALOG_CUTOFF_UTC = datetime(2026, 7, 9, 4, 25, 56, tzinfo=UTC)
_WGS84 = Geod(ellps="WGS84")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _readonly_vector(value: object, *, name: str, allow_zero_sum: bool = False) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    total = float(np.sum(result))
    if allow_zero_sum:
        if total not in (0.0, 1.0) and not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError(f"{name} must sum to zero or one")
    elif not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GridCell:
    """One frozen square synthetic grid cell."""

    cell_id: str
    row: int
    column: int
    x_km: float
    y_km: float
    area_km2: float = GRID_CELL_AREA_KM2

    def __post_init__(self) -> None:
        if not self.cell_id or self.cell_id.strip() != self.cell_id:
            raise ValueError("cell_id must be a non-empty stripped string")
        if self.row < 0 or self.column < 0:
            raise ValueError("grid row and column must be non-negative")
        for name in ("x_km", "y_km", "area_km2"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.area_km2 <= 0.0 or self.area_km2 > GRID_CELL_AREA_KM2:
            raise ValueError("cell area must be in (0, 625] km2")

    def as_mapping(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "column": self.column,
            "x_km": self.x_km,
            "y_km": self.y_km,
            "area_km2": self.area_km2,
        }


@dataclass(frozen=True, slots=True)
class SyntheticEvent:
    """An explicitly synthetic event; real source identifiers are rejected."""

    event_id: str
    origin_time_utc: datetime
    available_at_utc: datetime
    x_km: float
    y_km: float
    magnitude: float
    source_id: Literal["synthetic_history", "synthetic_ComCat"]
    longitude: float | None = None
    latitude: float | None = None

    def __post_init__(self) -> None:
        if not self.event_id or self.event_id.strip() != self.event_id:
            raise ValueError("event_id must be a non-empty stripped string")
        if self.source_id not in {"synthetic_history", "synthetic_ComCat"}:
            raise ValueError("P1-0B accepts only explicitly synthetic source IDs")
        origin = _utc(self.origin_time_utc, label="origin_time_utc")
        available = _utc(self.available_at_utc, label="available_at_utc")
        if available < origin:
            raise ValueError("an event cannot be available before its origin time")
        object.__setattr__(self, "origin_time_utc", origin)
        object.__setattr__(self, "available_at_utc", available)
        for name in ("x_km", "y_km", "magnitude"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if (self.longitude is None) != (self.latitude is None):
            raise ValueError("longitude and latitude must be supplied together")
        if self.longitude is None or self.latitude is None:
            latitude = 25.0 + self.y_km / 111.32
            longitude = 80.0 + self.x_km / (111.32 * math.cos(math.radians(latitude)))
        else:
            longitude = float(self.longitude)
            latitude = float(self.latitude)
        if (
            not math.isfinite(longitude)
            or not math.isfinite(latitude)
            or not -180.0 <= longitude <= 180.0
            or not -90.0 <= latitude <= 90.0
        ):
            raise ValueError("longitude/latitude must be finite WGS84 coordinates")
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "latitude", latitude)

    def as_mapping(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "origin_time_utc": _utc_text(self.origin_time_utc),
            "available_at_utc": _utc_text(self.available_at_utc),
            "x_km": self.x_km,
            "y_km": self.y_km,
            "magnitude": self.magnitude,
            "source_id": self.source_id,
            "longitude": self.longitude,
            "latitude": self.latitude,
        }


@dataclass(frozen=True, slots=True)
class RelativeIntensitySurface:
    """A normalized grid mass, explicitly not an absolute probability surface."""

    model_id: ModelId
    relative_intensity: FloatArray
    active_event_count: int

    def __post_init__(self) -> None:
        if self.model_id not in {"B0", "R30", "B0_R30"}:
            raise ValueError("unsupported model_id")
        if self.active_event_count < 0:
            raise ValueError("active_event_count must be non-negative")
        values = _readonly_vector(
            self.relative_intensity,
            name="relative_intensity",
            allow_zero_sum=self.model_id == "R30" and self.active_event_count == 0,
        )
        if self.active_event_count == 0 and np.any(values != 0.0):
            raise ValueError("an inactive R30 surface must be exactly zero")
        if self.model_id != "R30" and self.active_event_count == 0:
            raise ValueError("B0 and B0_R30 require at least one active historical event")
        object.__setattr__(self, "relative_intensity", values)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "active_event_count": self.active_event_count,
                "model_id": self.model_id,
                "relative_intensity": self.relative_intensity.tolist(),
                "value_semantics": "relative_intensity_not_absolute_probability",
            }
        )


@dataclass(frozen=True, slots=True)
class AlarmPrefix:
    """A complete-cell prefix in the model's unmodified ranking."""

    model_id: Literal["B0", "B0_R30"]
    ranked_cell_ids: tuple[str, ...]
    selected_cell_ids: tuple[str, ...]
    actual_area_km2: float
    area_cap_km2: float
    next_complete_cell_area_km2: float | None

    def __post_init__(self) -> None:
        if self.model_id not in {"B0", "B0_R30"}:
            raise ValueError("alarm model must be B0 or B0_R30")
        if len(self.ranked_cell_ids) != len(set(self.ranked_cell_ids)):
            raise ValueError("ranked cells must be unique")
        expected = self.ranked_cell_ids[: len(self.selected_cell_ids)]
        if self.selected_cell_ids != expected:
            raise ValueError("alarm cells must be an unbroken prefix of the frozen ranking")
        if self.actual_area_km2 < 0.0 or self.actual_area_km2 > self.area_cap_km2 + 1e-9:
            raise ValueError("actual alarm area exceeds its cap")
        if self.next_complete_cell_area_km2 is not None and self.next_complete_cell_area_km2 <= 0:
            raise ValueError("next complete-cell area must be positive")

    @property
    def ranking_sha256(self) -> str:
        return _canonical_sha256(
            {"model_id": self.model_id, "ranked_cell_ids": list(self.ranked_cell_ids)}
        )

    @property
    def mask_sha256(self) -> str:
        return _canonical_sha256(
            {"model_id": self.model_id, "selected_cell_ids": list(self.selected_cell_ids)}
        )


@dataclass(frozen=True, slots=True)
class DualModelForecast:
    """The frozen same-snapshot dual-model map and paired alarm prefixes."""

    issue_id: str
    scheduled_issue_time_utc: datetime
    query_cutoff_utc: datetime
    grid: tuple[GridCell, ...]
    B0: RelativeIntensitySurface
    R30: RelativeIntensitySurface
    B0_R30: RelativeIntensitySurface
    B0_alarm: AlarmPrefix
    B0_R30_alarm: AlarmPrefix
    recent_fallback_to_B0: bool

    def __post_init__(self) -> None:
        if not self.issue_id or self.issue_id.strip() != self.issue_id:
            raise ValueError("issue_id must be a non-empty stripped string")
        scheduled = _utc(self.scheduled_issue_time_utc, label="scheduled_issue_time_utc")
        cutoff = _utc(self.query_cutoff_utc, label="query_cutoff_utc")
        if cutoff != scheduled - timedelta(minutes=15):
            raise ValueError("query cutoff must equal scheduled issue time minus 15 minutes")
        object.__setattr__(self, "scheduled_issue_time_utc", scheduled)
        object.__setattr__(self, "query_cutoff_utc", cutoff)
        size = len(self.grid)
        if size == 0 or len({cell.cell_id for cell in self.grid}) != size:
            raise ValueError("grid must contain unique cells")
        if any(
            surface.relative_intensity.size != size for surface in (self.B0, self.R30, self.B0_R30)
        ):
            raise ValueError("every surface must align one-to-one with the frozen grid")
        if (self.B0.model_id, self.R30.model_id, self.B0_R30.model_id) != (
            "B0",
            "R30",
            "B0_R30",
        ):
            raise ValueError("forecast surface fields must carry their exact frozen model IDs")
        if self.B0_R30.active_event_count != self.B0.active_event_count:
            raise ValueError("B0_R30 data water level must equal B0")
        if self.R30.active_event_count > self.B0.active_event_count:
            raise ValueError("R30 events must be a subset of B0 events")
        if (self.R30.active_event_count == 0) != self.recent_fallback_to_B0:
            raise ValueError("R30 emptiness and fallback state must agree exactly")
        if self.recent_fallback_to_B0:
            if not np.array_equal(self.B0.relative_intensity, self.B0_R30.relative_intensity):
                raise ValueError("empty R30 must fall back exactly to B0")
        else:
            expected_mixed = (
                1.0 - MIXING_ALPHA
            ) * self.B0.relative_intensity + MIXING_ALPHA * self.R30.relative_intensity
            expected_mixed /= float(np.sum(expected_mixed))
            if not np.allclose(
                self.B0_R30.relative_intensity,
                expected_mixed,
                rtol=1e-15,
                atol=0.0,
            ):
                raise ValueError("B0_R30 must equal the frozen 0.75*B0 + 0.25*R30 mixture")
        grid_by_id = {cell.cell_id: cell for cell in self.grid}
        expected_grid_ids = set(grid_by_id)
        for surface, alarm, expected_model_id in (
            (self.B0, self.B0_alarm, "B0"),
            (self.B0_R30, self.B0_R30_alarm, "B0_R30"),
        ):
            if alarm.model_id != expected_model_id:
                raise ValueError("alarm fields must carry their exact frozen model IDs")
            if set(alarm.ranked_cell_ids) != expected_grid_ids or len(alarm.ranked_cell_ids) != len(
                self.grid
            ):
                raise ValueError("alarm ranking must contain every frozen grid cell exactly once")
            expected_ranking = tuple(
                self.grid[index].cell_id
                for index in sorted(
                    range(size),
                    key=lambda index: (
                        -surface.relative_intensity[index] / self.grid[index].area_km2,
                        self.grid[index].row,
                        self.grid[index].column,
                        self.grid[index].cell_id.encode("utf-8"),
                    ),
                )
            )
            if alarm.ranked_cell_ids != expected_ranking:
                raise ValueError(
                    "alarm ranking must be derived from its relative-intensity surface"
                )
            selected_area = sum(grid_by_id[cell_id].area_km2 for cell_id in alarm.selected_cell_ids)
            if not math.isclose(selected_area, alarm.actual_area_km2, abs_tol=1e-9):
                raise ValueError("alarm actual area does not equal its selected complete cells")
            next_index = len(alarm.selected_cell_ids)
            expected_next_area = (
                grid_by_id[alarm.ranked_cell_ids[next_index]].area_km2
                if next_index < len(alarm.ranked_cell_ids)
                else None
            )
            if alarm.next_complete_cell_area_km2 != expected_next_area:
                raise ValueError("alarm next complete-cell area is inconsistent with its ranking")
            if (
                expected_next_area is not None
                and alarm.actual_area_km2 + expected_next_area <= alarm.area_cap_km2 + 1e-9
            ):
                raise ValueError("alarm must use the largest complete-cell prefix within its cap")
        if not math.isclose(self.B0_alarm.area_cap_km2, MAXIMUM_ALARM_AREA_KM2, abs_tol=1e-9):
            raise ValueError("B0 alarm must use the frozen 600,000 km2 initial cap")
        if not math.isclose(
            self.B0_R30_alarm.area_cap_km2,
            self.B0_alarm.actual_area_km2,
            abs_tol=1e-9,
        ):
            raise ValueError("B0_R30 cap must equal the B0 actual reference area")
        difference = self.B0_alarm.actual_area_km2 - self.B0_R30_alarm.actual_area_km2
        next_area = self.B0_R30_alarm.next_complete_cell_area_km2
        if difference < -1e-9:
            raise ValueError("B0_R30 may never use more area than B0")
        if next_area is None:
            raise ValueError("paired fairness requires an unselected next challenger cell")
        if not difference < next_area - 1e-9 and not math.isclose(difference, 0.0):
            raise ValueError("alarm-area difference is not smaller than the next complete cell")
        if not difference < GRID_CELL_AREA_KM2 - 1e-9 and not math.isclose(difference, 0.0):
            raise ValueError("alarm-area difference must be strictly less than 625 km2")

    @property
    def B0_reference_area_km2(self) -> float:
        return self.B0_alarm.actual_area_km2

    @property
    def actual_area_difference_km2(self) -> float:
        return self.B0_alarm.actual_area_km2 - self.B0_R30_alarm.actual_area_km2


@dataclass(frozen=True, slots=True)
class IssueCandidate:
    issue_id: str
    scheduled_issue_time_utc: datetime
    status: IssueStatus

    def __post_init__(self) -> None:
        if not self.issue_id:
            raise ValueError("issue_id must be non-empty")
        if self.status not in {"on_time", "missed_issue"}:
            raise ValueError("invalid issue status")
        object.__setattr__(
            self,
            "scheduled_issue_time_utc",
            _utc(self.scheduled_issue_time_utc, label="scheduled_issue_time_utc"),
        )


@dataclass(frozen=True, slots=True)
class TargetCluster:
    """One within-exposure 30-day/75-km connected component."""

    issue_id: str
    horizon_days: Literal[30, 90]
    cluster_id: str
    member_event_ids: tuple[str, ...]
    representative: SyntheticEvent

    def __post_init__(self) -> None:
        if self.horizon_days not in {PRIMARY_HORIZON_DAYS, SECONDARY_HORIZON_DAYS}:
            raise ValueError("target horizon must be 30 or 90 days")
        if not self.member_event_ids or len(self.member_event_ids) != len(
            set(self.member_event_ids)
        ):
            raise ValueError("cluster members must be non-empty and unique")
        if self.representative.event_id not in self.member_event_ids:
            raise ValueError("cluster representative must be a member")


@dataclass(frozen=True, slots=True)
class ClusterScore:
    issue_id: str
    cluster_id: str
    representative_origin_time_utc: datetime
    representative_event_id: str
    B0_hit: bool
    B0_R30_hit: bool

    def __post_init__(self) -> None:
        if not self.issue_id or self.issue_id.strip() != self.issue_id:
            raise ValueError("score issue_id must be a non-empty stripped string")
        if not self.cluster_id or self.cluster_id.strip() != self.cluster_id:
            raise ValueError("score cluster_id must be a non-empty stripped string")
        if (
            not self.representative_event_id
            or self.representative_event_id.strip() != self.representative_event_id
        ):
            raise ValueError("score representative_event_id must be a non-empty stripped string")
        if type(self.B0_hit) is not bool or type(self.B0_R30_hit) is not bool:
            raise ValueError("paired score hit flags must be booleans")
        object.__setattr__(
            self,
            "representative_origin_time_utc",
            _utc(self.representative_origin_time_utc, label="representative_origin_time_utc"),
        )

    @property
    def paired_difference(self) -> int:
        return int(self.B0_R30_hit) - int(self.B0_hit)

    def as_mapping(self) -> dict[str, object]:
        return {
            "issue_id": self.issue_id,
            "cluster_id": self.cluster_id,
            "representative_origin_time_utc": _utc_text(self.representative_origin_time_utc),
            "representative_event_id": self.representative_event_id,
            "B0_hit": self.B0_hit,
            "B0_R30_hit": self.B0_R30_hit,
        }


def ordered_cluster_registry_sha256(scores: tuple[ClusterScore, ...]) -> str:
    """Bind the full ordered paired-score registry, including both hit flags."""

    return _canonical_sha256(
        {
            "domain": "seismoflux.p1.ordered-cluster-registry.v2",
            "ordered_scores": [score.as_mapping() for score in scores],
        }
    )


def selected_cluster_prefix_sha256(scores: tuple[ClusterScore, ...]) -> str:
    """Bind one immutable score prefix used by a sequential look."""

    return _canonical_sha256(
        {
            "domain": "seismoflux.p1.selected-cluster-prefix.v2",
            "ordered_scores": [score.as_mapping() for score in scores],
        }
    )


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    horizon_days: Literal[30, 90]
    scores: tuple[ClusterScore, ...]

    def __post_init__(self) -> None:
        if self.horizon_days not in {PRIMARY_HORIZON_DAYS, SECONDARY_HORIZON_DAYS}:
            raise ValueError("score horizon must be 30 or 90 days")
        identities = tuple((score.issue_id, score.cluster_id) for score in self.scores)
        if len(identities) != len(set(identities)):
            raise ValueError("paired cluster scores must be globally unique")
        expected = tuple(
            sorted(
                self.scores,
                key=lambda score: (
                    score.representative_origin_time_utc,
                    score.representative_event_id.encode("utf-8"),
                    score.issue_id.encode("utf-8"),
                    score.cluster_id.encode("utf-8"),
                ),
            )
        )
        if self.scores != expected:
            raise ValueError("paired cluster scores must use the frozen global stable order")

    @property
    def cluster_count(self) -> int:
        return len(self.scores)

    @property
    def B0_hit_clusters(self) -> int:
        return sum(score.B0_hit for score in self.scores)

    @property
    def B0_R30_hit_clusters(self) -> int:
        return sum(score.B0_R30_hit for score in self.scores)

    @property
    def B0_recall(self) -> float | None:
        return None if not self.scores else self.B0_hit_clusters / len(self.scores)

    @property
    def B0_R30_recall(self) -> float | None:
        return None if not self.scores else self.B0_R30_hit_clusters / len(self.scores)

    @property
    def recall_gain_percentage_points(self) -> float | None:
        if not self.scores:
            return None
        return 100.0 * (self.B0_R30_hit_clusters - self.B0_hit_clusters) / len(self.scores)


@dataclass(frozen=True, slots=True)
class SequentialReview:
    """One preregistered 30-day paired-cluster scientific look."""

    review_trigger: ReviewTrigger
    look_sequence: int
    prior_completed_look_count: int
    cumulative_cluster_count: int
    ordered_cluster_registry_sha256: str
    selected_cluster_prefix_sha256: str
    elapsed_months: float
    B0_hit_clusters: int
    B0_R30_hit_clusters: int
    recall_gain_percentage_points: float | None
    sequentially_adjusted_interval_lower: float | None
    sequentially_adjusted_interval_upper: float | None
    decision: ReviewDecision

    def as_mapping(self) -> dict[str, object]:
        return {
            "horizon_days": PRIMARY_HORIZON_DAYS,
            "review_trigger": self.review_trigger,
            "look_sequence": self.look_sequence,
            "prior_completed_look_count": self.prior_completed_look_count,
            "cumulative_cluster_count": self.cumulative_cluster_count,
            "ordered_cluster_registry_sha256": self.ordered_cluster_registry_sha256,
            "selected_cluster_prefix_sha256": self.selected_cluster_prefix_sha256,
            "elapsed_months": self.elapsed_months,
            "B0_hit_clusters": self.B0_hit_clusters,
            "B0_R30_hit_clusters": self.B0_R30_hit_clusters,
            "recall_gain_percentage_points": self.recall_gain_percentage_points,
            "sequentially_adjusted_interval_lower": self.sequentially_adjusted_interval_lower,
            "sequentially_adjusted_interval_upper": self.sequentially_adjusted_interval_upper,
            "decision": self.decision,
        }


def make_equal_area_grid(*, rows: int = 34, columns: int = 34) -> tuple[GridCell, ...]:
    """Build a deterministic 25-km square synthetic grid."""

    if rows <= 0 or columns <= 0:
        raise ValueError("grid rows and columns must be positive")
    return tuple(
        GridCell(
            cell_id=f"r{row:02d}c{column:02d}",
            row=row,
            column=column,
            x_km=(column + 0.5) * GRID_CELL_SIZE_KM,
            y_km=(row + 0.5) * GRID_CELL_SIZE_KM,
        )
        for row in range(rows)
        for column in range(columns)
    )


def _cell_contains_point(cell: GridCell, *, x_km: float, y_km: float) -> bool:
    half_side = math.sqrt(cell.area_km2) / 2.0
    return (
        cell.x_km - half_side <= x_km < cell.x_km + half_side
        and cell.y_km - half_side <= y_km < cell.y_km + half_side
    )


def _point_is_inside_frozen_grid(grid: tuple[GridCell, ...], *, x_km: float, y_km: float) -> bool:
    return sum(_cell_contains_point(cell, x_km=x_km, y_km=y_km) for cell in grid) == 1


def gaussian_kde_relative_intensity(
    events: tuple[SyntheticEvent, ...],
    grid: tuple[GridCell, ...],
    *,
    model_id: ModelId,
    bandwidth_km: float = BANDWIDTH_KM,
) -> RelativeIntensitySurface:
    """Evaluate an equal-event-weight Gaussian KDE and normalize over the frozen grid."""

    if not grid:
        raise ValueError("grid must not be empty")
    if not math.isfinite(bandwidth_km) or bandwidth_km <= 0.0:
        raise ValueError("bandwidth_km must be finite and positive")
    if not events:
        if model_id != "R30":
            raise ValueError("only R30 may be empty")
        return RelativeIntensitySurface(model_id, np.zeros(len(grid), dtype=np.float64), 0)
    grid_xy = np.array([(cell.x_km, cell.y_km) for cell in grid], dtype=np.float64)
    event_xy = np.array([(event.x_km, event.y_km) for event in events], dtype=np.float64)
    squared_distance = np.sum((grid_xy[:, None, :] - event_xy[None, :, :]) ** 2, axis=2)
    density = np.mean(np.exp(-0.5 * squared_distance / bandwidth_km**2), axis=1)
    unnormalized_mass = density * np.array([cell.area_km2 for cell in grid], dtype=np.float64)
    total = float(np.sum(unnormalized_mass))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Gaussian KDE produced no finite mass on the frozen grid")
    return RelativeIntensitySurface(model_id, unnormalized_mass / total, len(events))


def deduplicate_source_boundary(
    events: tuple[SyntheticEvent, ...],
) -> tuple[SyntheticEvent, ...]:
    """Apply the frozen local-anchor, deterministic one-to-one cutover deduplication."""

    history = tuple(event for event in events if event.source_id == "synthetic_history")
    comcat = tuple(event for event in events if event.source_id == "synthetic_ComCat")
    candidates: list[tuple[float, float, float, bytes, bytes, str, str]] = []
    for anchor in history:
        for source in comcat:
            time_difference = abs((source.origin_time_utc - anchor.origin_time_utc).total_seconds())
            if time_difference > 300.0:
                continue
            _, _, distance_m = _WGS84.inv(
                anchor.longitude,
                anchor.latitude,
                source.longitude,
                source.latitude,
            )
            distance_km = abs(float(distance_m)) / 1_000.0
            magnitude_difference = abs(source.magnitude - anchor.magnitude)
            if distance_km <= 50.0 and magnitude_difference <= 0.5:
                candidates.append(
                    (
                        time_difference,
                        distance_km,
                        magnitude_difference,
                        source.event_id.encode("utf-8"),
                        anchor.event_id.encode("utf-8"),
                        anchor.event_id,
                        source.event_id,
                    )
                )
    matched_history: set[str] = set()
    matched_comcat: set[str] = set()
    for *_, history_id, comcat_id in sorted(candidates):
        if history_id not in matched_history and comcat_id not in matched_comcat:
            matched_history.add(history_id)
            matched_comcat.add(comcat_id)
    return tuple(
        sorted(
            (event for event in events if event.event_id not in matched_comcat),
            key=lambda event: (event.origin_time_utc, event.event_id.encode("utf-8")),
        )
    )


def _select_alarm_prefix(
    surface: RelativeIntensitySurface,
    grid: tuple[GridCell, ...],
    *,
    area_cap_km2: float,
) -> AlarmPrefix:
    if surface.model_id not in {"B0", "B0_R30"}:
        raise ValueError("only B0 and B0_R30 produce scored alarm prefixes")
    if not math.isfinite(area_cap_km2) or area_cap_km2 < 0.0:
        raise ValueError("area cap must be finite and non-negative")
    if len(grid) != surface.relative_intensity.size:
        raise ValueError("surface does not align with grid")
    ranking = tuple(
        sorted(
            range(len(grid)),
            key=lambda index: (
                -surface.relative_intensity[index] / grid[index].area_km2,
                grid[index].row,
                grid[index].column,
                grid[index].cell_id.encode("utf-8"),
            ),
        )
    )
    selected_indices: list[int] = []
    area = 0.0
    for index in ranking:
        candidate_area = area + grid[index].area_km2
        if candidate_area > area_cap_km2 + 1e-9:
            break
        selected_indices.append(index)
        area = candidate_area
    next_area = (
        grid[ranking[len(selected_indices)]].area_km2
        if len(selected_indices) < len(ranking)
        else None
    )
    model_id: Literal["B0", "B0_R30"] = "B0" if surface.model_id == "B0" else "B0_R30"
    return AlarmPrefix(
        model_id=model_id,
        ranked_cell_ids=tuple(grid[index].cell_id for index in ranking),
        selected_cell_ids=tuple(grid[index].cell_id for index in selected_indices),
        actual_area_km2=area,
        area_cap_km2=area_cap_km2,
        next_complete_cell_area_km2=next_area,
    )


def build_dual_model_forecast(
    events: tuple[SyntheticEvent, ...],
    grid: tuple[GridCell, ...],
    *,
    issue_id: str,
    scheduled_issue_time_utc: datetime,
) -> DualModelForecast:
    """Build B0 and frozen ``0.75*B0 + 0.25*R30`` from one causal snapshot."""

    if not issue_id or issue_id.strip() != issue_id:
        raise ValueError("issue_id must be a non-empty stripped string")
    scheduled = _utc(scheduled_issue_time_utc, label="scheduled_issue_time_utc")
    cutoff = scheduled - timedelta(minutes=15)
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("synthetic event IDs must be unique")
    if any(
        event.source_id == "synthetic_history" and event.origin_time_utc > LOCAL_CATALOG_CUTOFF_UTC
        for event in events
    ):
        raise ValueError("synthetic_history may not extend beyond the frozen local cutoff")
    if any(
        event.source_id == "synthetic_ComCat" and event.origin_time_utc <= LOCAL_CATALOG_CUTOFF_UTC
        for event in events
    ):
        raise ValueError("synthetic_ComCat model input must be strictly after the local cutoff")
    causally_eligible = tuple(
        event
        for event in events
        if event.magnitude >= 4.0
        and event.origin_time_utc >= HISTORICAL_START_UTC
        and event.origin_time_utc <= cutoff
        and event.available_at_utc <= cutoff
        and _point_is_inside_frozen_grid(grid, x_km=event.x_km, y_km=event.y_km)
    )
    eligible = deduplicate_source_boundary(causally_eligible)
    if not eligible:
        raise ValueError("B0 requires at least one causally available synthetic M4+ event")
    recent_start = cutoff - timedelta(days=RECENT_WINDOW_DAYS)
    recent = tuple(
        event
        for event in eligible
        if event.source_id == "synthetic_ComCat" and recent_start < event.origin_time_utc <= cutoff
    )
    B0 = gaussian_kde_relative_intensity(eligible, grid, model_id="B0")
    R30 = gaussian_kde_relative_intensity(recent, grid, model_id="R30")
    if recent:
        mixed = (1.0 - MIXING_ALPHA) * B0.relative_intensity + MIXING_ALPHA * R30.relative_intensity
        mixed /= float(np.sum(mixed))
        B0_R30 = RelativeIntensitySurface("B0_R30", mixed, len(eligible))
        fallback = False
    else:
        B0_R30 = RelativeIntensitySurface(
            "B0_R30", np.array(B0.relative_intensity, copy=True), len(eligible)
        )
        fallback = True
    B0_alarm = _select_alarm_prefix(B0, grid, area_cap_km2=MAXIMUM_ALARM_AREA_KM2)
    challenger_alarm = _select_alarm_prefix(
        B0_R30,
        grid,
        area_cap_km2=B0_alarm.actual_area_km2,
    )
    return DualModelForecast(
        issue_id=issue_id,
        scheduled_issue_time_utc=scheduled,
        query_cutoff_utc=cutoff,
        grid=grid,
        B0=B0,
        R30=R30,
        B0_R30=B0_R30,
        B0_alarm=B0_alarm,
        B0_R30_alarm=challenger_alarm,
        recent_fallback_to_B0=fallback,
    )


def select_guarded_issues(
    issues: tuple[IssueCandidate, ...], *, horizon_days: Literal[30, 90]
) -> tuple[IssueCandidate, ...]:
    """Select non-overlapping on-time exposures with the frozen extra 30-day guard gap."""

    if horizon_days not in {PRIMARY_HORIZON_DAYS, SECONDARY_HORIZON_DAYS}:
        raise ValueError("horizon_days must be 30 or 90")
    on_time = sorted(
        (issue for issue in issues if issue.status == "on_time"),
        key=lambda issue: (issue.scheduled_issue_time_utc, issue.issue_id.encode("utf-8")),
    )
    if len({issue.issue_id for issue in issues}) != len(issues):
        raise ValueError("issue IDs must be unique")
    selected: list[IssueCandidate] = []
    gap = timedelta(days=horizon_days + 30)
    for issue in on_time:
        if (
            not selected
            or issue.scheduled_issue_time_utc >= selected[-1].scheduled_issue_time_utc + gap
        ):
            selected.append(issue)
    return tuple(selected)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            parent = self.find(parent)
            self.parent[value] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def cluster_target_events(
    events: tuple[SyntheticEvent, ...],
    *,
    issue_id: str,
    issue_time_utc: datetime,
    horizon_days: Literal[30, 90],
    truth_fetched_at_utc: datetime,
    grid: tuple[GridCell, ...],
) -> tuple[TargetCluster, ...]:
    """Form one mature exposure's 30-day/75-km WGS84 connected components."""

    if horizon_days not in {PRIMARY_HORIZON_DAYS, SECONDARY_HORIZON_DAYS}:
        raise ValueError("horizon_days must be 30 or 90")
    start = _utc(issue_time_utc, label="issue_time_utc")
    end = start + timedelta(days=horizon_days)
    mature_after = end + timedelta(days=30)
    fetched_at = _utc(truth_fetched_at_utc, label="truth_fetched_at_utc")
    if fetched_at < mature_after:
        raise ValueError("truth snapshot cannot be read before T plus horizon plus 30 days")
    if not grid:
        raise ValueError("truth clustering requires the frozen study/support grid")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("target event IDs must be unique before clustering")
    if any(event.source_id != "synthetic_ComCat" for event in events):
        raise ValueError("future truth must come only from synthetic_ComCat")
    eligible = tuple(
        sorted(
            (
                event
                for event in events
                if 5.0 <= event.magnitude < 6.0
                and start < event.origin_time_utc <= end
                and event.available_at_utc <= fetched_at
                and _point_is_inside_frozen_grid(grid, x_km=event.x_km, y_km=event.y_km)
            ),
            key=lambda event: (event.origin_time_utc, event.event_id.encode("utf-8")),
        )
    )
    disjoint = _DisjointSet(len(eligible))
    for left in range(len(eligible)):
        for right in range(left + 1, len(eligible)):
            time_days = (
                abs(
                    (
                        eligible[right].origin_time_utc - eligible[left].origin_time_utc
                    ).total_seconds()
                )
                / 86_400.0
            )
            _, _, distance_m = _WGS84.inv(
                eligible[left].longitude,
                eligible[left].latitude,
                eligible[right].longitude,
                eligible[right].latitude,
            )
            distance = abs(float(distance_m)) / 1_000.0
            if time_days <= CLUSTER_MAX_TIME_DAYS and distance <= CLUSTER_MAX_DISTANCE_KM:
                disjoint.union(left, right)
    components: dict[int, list[SyntheticEvent]] = {}
    for index, event in enumerate(eligible):
        components.setdefault(disjoint.find(index), []).append(event)
    clusters: list[TargetCluster] = []
    for members in components.values():
        ordered_members = tuple(
            sorted(
                members, key=lambda event: (event.origin_time_utc, event.event_id.encode("utf-8"))
            )
        )
        member_ids = tuple(
            sorted((event.event_id for event in members), key=lambda value: value.encode("utf-8"))
        )
        identity = _canonical_sha256(
            {
                "domain": "seismoflux.p1.synthetic-target-cluster.v1",
                "horizon_days": horizon_days,
                "issue_id": issue_id,
                "member_event_ids": list(member_ids),
            }
        )
        clusters.append(
            TargetCluster(
                issue_id=issue_id,
                horizon_days=horizon_days,
                cluster_id=f"p1sc-{identity[:24]}",
                member_event_ids=member_ids,
                representative=ordered_members[0],
            )
        )
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                cluster.representative.origin_time_utc,
                cluster.representative.event_id.encode("utf-8"),
                cluster.issue_id.encode("utf-8"),
                cluster.cluster_id.encode("utf-8"),
            ),
        )
    )


def cluster_guarded_exposures(
    issues: tuple[IssueCandidate, ...],
    events_by_issue_id: Mapping[str, tuple[SyntheticEvent, ...]],
    truth_fetched_at_by_issue_id: Mapping[str, datetime],
    *,
    horizon_days: Literal[30, 90],
    grid: tuple[GridCell, ...],
) -> tuple[TargetCluster, ...]:
    """Select guarded on-time exposures, then cluster each window separately."""

    selected = select_guarded_issues(issues, horizon_days=horizon_days)
    selected_ids = {issue.issue_id for issue in selected}
    if set(events_by_issue_id) != selected_ids or set(truth_fetched_at_by_issue_id) != selected_ids:
        raise ValueError("truth inputs must bind exactly the guard-selected issue set")
    clusters: list[TargetCluster] = []
    seen_event_ids: set[str] = set()
    for issue in selected:
        exposure_clusters = cluster_target_events(
            events_by_issue_id[issue.issue_id],
            issue_id=issue.issue_id,
            issue_time_utc=issue.scheduled_issue_time_utc,
            horizon_days=horizon_days,
            truth_fetched_at_utc=truth_fetched_at_by_issue_id[issue.issue_id],
            grid=grid,
        )
        member_ids = {
            event_id for cluster in exposure_clusters for event_id in cluster.member_event_ids
        }
        if seen_event_ids & member_ids:
            raise ValueError("a target event may not be counted across selected exposures")
        seen_event_ids.update(member_ids)
        clusters.extend(exposure_clusters)
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                cluster.representative.origin_time_utc,
                cluster.representative.event_id.encode("utf-8"),
                cluster.issue_id.encode("utf-8"),
                cluster.cluster_id.encode("utf-8"),
            ),
        )
    )


def _cell_for_point(grid: tuple[GridCell, ...], *, x_km: float, y_km: float) -> GridCell:
    matches = [cell for cell in grid if _cell_contains_point(cell, x_km=x_km, y_km=y_km)]
    if len(matches) != 1:
        raise ValueError("target representative must fall inside exactly one frozen grid cell")
    return matches[0]


def score_clusters(
    forecast: DualModelForecast,
    clusters: tuple[TargetCluster, ...],
    *,
    horizon_days: Literal[30, 90],
) -> ScoreSummary:
    """Pair-score cluster representatives against the two same-area alarm maps."""

    if any(cluster.horizon_days != horizon_days for cluster in clusters):
        raise ValueError("all clusters must belong to the requested horizon")
    if any(cluster.issue_id != forecast.issue_id for cluster in clusters):
        raise ValueError("target clusters must bind the exact forecast issue_id")
    B0_alarm = set(forecast.B0_alarm.selected_cell_ids)
    challenger_alarm = set(forecast.B0_R30_alarm.selected_cell_ids)
    scores: list[ClusterScore] = []
    for cluster in clusters:
        representative = cluster.representative
        cell = _cell_for_point(forecast.grid, x_km=representative.x_km, y_km=representative.y_km)
        scores.append(
            ClusterScore(
                issue_id=cluster.issue_id,
                cluster_id=cluster.cluster_id,
                representative_origin_time_utc=representative.origin_time_utc,
                representative_event_id=representative.event_id,
                B0_hit=cell.cell_id in B0_alarm,
                B0_R30_hit=cell.cell_id in challenger_alarm,
            )
        )
    ordered = tuple(
        sorted(
            scores,
            key=lambda score: (
                score.representative_origin_time_utc,
                score.representative_event_id.encode("utf-8"),
                score.issue_id.encode("utf-8"),
                score.cluster_id.encode("utf-8"),
            ),
        )
    )
    return ScoreSummary(horizon_days=horizon_days, scores=ordered)


def paired_bootstrap_interval(scores: tuple[ClusterScore, ...]) -> tuple[float, float] | None:
    """Return the frozen three-look Bonferroni-adjusted paired-cluster interval."""

    if not scores:
        return None
    differences = np.array([score.paired_difference for score in scores], dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sampled_indices = generator.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_REPLICATES, len(differences)),
    )
    bootstrap_gain = 100.0 * np.mean(differences[sampled_indices], axis=1)
    lower, upper = np.quantile(
        bootstrap_gain,
        [BOOTSTRAP_LOWER_QUANTILE, BOOTSTRAP_UPPER_QUANTILE],
    )
    return float(lower), float(upper)


def _final_decision(gain: float, interval_lower: float) -> ReviewDecision:
    if gain <= 0.0:
        return "stop_B0_R30_retain_B0"
    if gain >= 5.0 and interval_lower > 0.0:
        return "confirm_strong_prospective_improvement"
    return "report_uncertain_at_final_review"


def _one_review(
    ordered_scores: tuple[ClusterScore, ...],
    *,
    trigger: ReviewTrigger,
    prefix_count: int,
    look_sequence: int,
    elapsed_months: float,
) -> SequentialReview:
    prefix = ordered_scores[:prefix_count]
    registry_sha = ordered_cluster_registry_sha256(ordered_scores)
    prefix_sha = selected_cluster_prefix_sha256(prefix)
    if not prefix:
        return SequentialReview(
            review_trigger=trigger,
            look_sequence=look_sequence,
            prior_completed_look_count=look_sequence - 1,
            cumulative_cluster_count=0,
            ordered_cluster_registry_sha256=registry_sha,
            selected_cluster_prefix_sha256=prefix_sha,
            elapsed_months=elapsed_months,
            B0_hit_clusters=0,
            B0_R30_hit_clusters=0,
            recall_gain_percentage_points=None,
            sequentially_adjusted_interval_lower=None,
            sequentially_adjusted_interval_upper=None,
            decision="report_evidence_insufficient_at_final_review",
        )
    B0_hits = sum(score.B0_hit for score in prefix)
    challenger_hits = sum(score.B0_R30_hit for score in prefix)
    gain = 100.0 * (challenger_hits - B0_hits) / len(prefix)
    interval = paired_bootstrap_interval(prefix)
    if interval is None:
        raise AssertionError("non-empty paired scores must produce an interval")
    decision: ReviewDecision
    if trigger in {"cluster_10", "cluster_20"}:
        decision = "continue_accumulation"
    else:
        decision = _final_decision(gain, interval[0])
    return SequentialReview(
        review_trigger=trigger,
        look_sequence=look_sequence,
        prior_completed_look_count=look_sequence - 1,
        cumulative_cluster_count=len(prefix),
        ordered_cluster_registry_sha256=registry_sha,
        selected_cluster_prefix_sha256=prefix_sha,
        elapsed_months=elapsed_months,
        B0_hit_clusters=B0_hits,
        B0_R30_hit_clusters=challenger_hits,
        recall_gain_percentage_points=gain,
        sequentially_adjusted_interval_lower=interval[0],
        sequentially_adjusted_interval_upper=interval[1],
        decision=decision,
    )


def build_pending_sequential_reviews(
    primary_scores: ScoreSummary,
    *,
    elapsed_months: float,
    completed_reviews: tuple[SequentialReview, ...] = (),
) -> tuple[SequentialReview, ...]:
    """Emit exact pending 10/20/30 looks, or the frozen 36-month terminal look."""

    if primary_scores.horizon_days != PRIMARY_HORIZON_DAYS:
        raise ValueError("sequential reviews may read only the 30-day primary endpoint")
    if not math.isfinite(elapsed_months) or not 0.0 <= elapsed_months <= 36.0:
        raise ValueError("elapsed_months must be in [0, 36]")
    valid_prefixes: tuple[tuple[ReviewTrigger, ...], ...] = (
        (),
        ("cluster_10",),
        ("cluster_10", "cluster_20"),
        ("cluster_10", "cluster_20", "cluster_30"),
    )
    completed_triggers = tuple(review.review_trigger for review in completed_reviews)
    if completed_triggers not in valid_prefixes:
        raise ValueError("completed triggers must be an exact 10/20/30 prefix")
    ordered_scores = primary_scores.scores
    threshold_by_trigger: dict[ReviewTrigger, int] = {
        "cluster_10": 10,
        "cluster_20": 20,
        "cluster_30": 30,
        "time_36_months": len(ordered_scores),
    }
    for position, prior_review in enumerate(completed_reviews, start=1):
        threshold = threshold_by_trigger[prior_review.review_trigger]
        if len(ordered_scores) < threshold:
            raise ValueError("current registry is shorter than a completed cluster look")
        expected_prior = _one_review(
            ordered_scores,
            trigger=prior_review.review_trigger,
            prefix_count=threshold,
            look_sequence=position,
            elapsed_months=prior_review.elapsed_months,
        )
        immutable_fields = (
            "review_trigger",
            "look_sequence",
            "prior_completed_look_count",
            "cumulative_cluster_count",
            "selected_cluster_prefix_sha256",
            "B0_hit_clusters",
            "B0_R30_hit_clusters",
            "recall_gain_percentage_points",
            "sequentially_adjusted_interval_lower",
            "sequentially_adjusted_interval_upper",
            "decision",
        )
        if any(
            getattr(prior_review, field) != getattr(expected_prior, field)
            for field in immutable_fields
        ):
            raise ValueError("a completed look's frozen cluster prefix or result changed")
    if "cluster_30" in completed_triggers:
        return ()
    count = len(ordered_scores)
    reviews: list[SequentialReview] = []
    completed = list(completed_triggers)
    threshold_looks: tuple[tuple[int, ReviewTrigger], ...] = (
        (10, "cluster_10"),
        (20, "cluster_20"),
        (30, "cluster_30"),
    )
    if elapsed_months < 36.0:
        for threshold, trigger in threshold_looks:
            if count >= threshold and trigger not in completed:
                review = _one_review(
                    ordered_scores,
                    trigger=trigger,
                    prefix_count=threshold,
                    look_sequence=len(completed) + 1,
                    elapsed_months=elapsed_months,
                )
                reviews.append(review)
                completed.append(trigger)
        return tuple(reviews)

    terminal_catch_up_elapsed = math.nextafter(36.0, 0.0)
    if count >= 30:
        for threshold, trigger in threshold_looks:
            if trigger not in completed:
                review = _one_review(
                    ordered_scores,
                    trigger=trigger,
                    prefix_count=threshold,
                    look_sequence=len(completed) + 1,
                    elapsed_months=(36.0 if trigger == "cluster_30" else terminal_catch_up_elapsed),
                )
                reviews.append(review)
                completed.append(trigger)
        return tuple(reviews)
    for threshold, trigger in threshold_looks[:2]:
        if count >= threshold and trigger not in completed:
            review = _one_review(
                ordered_scores,
                trigger=trigger,
                prefix_count=threshold,
                look_sequence=len(completed) + 1,
                elapsed_months=terminal_catch_up_elapsed,
            )
            reviews.append(review)
            completed.append(trigger)
    terminal = _one_review(
        ordered_scores,
        trigger="time_36_months",
        prefix_count=count,
        look_sequence=len(completed) + 1,
        elapsed_months=36.0,
    )
    reviews.append(terminal)
    return tuple(reviews)


__all__ = [
    "BANDWIDTH_KM",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "GRID_CELL_AREA_KM2",
    "GRID_CELL_SIZE_KM",
    "MAXIMUM_ALARM_AREA_KM2",
    "MIXING_ALPHA",
    "AlarmPrefix",
    "ClusterScore",
    "DualModelForecast",
    "GridCell",
    "IssueCandidate",
    "RelativeIntensitySurface",
    "ReviewDecision",
    "ReviewTrigger",
    "ScoreSummary",
    "SequentialReview",
    "SyntheticEvent",
    "TargetCluster",
    "build_dual_model_forecast",
    "build_pending_sequential_reviews",
    "cluster_guarded_exposures",
    "cluster_target_events",
    "deduplicate_source_boundary",
    "gaussian_kde_relative_intensity",
    "make_equal_area_grid",
    "ordered_cluster_registry_sha256",
    "paired_bootstrap_interval",
    "score_clusters",
    "select_guarded_issues",
    "selected_cluster_prefix_sha256",
]
