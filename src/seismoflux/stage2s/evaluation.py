"""Target-blind Stage 2S evaluation primitives.

The functions in this module accept only already-frozen in-memory predictions and
memberships.  They do not open paths, fit models, or make gate decisions outside the
pre-registered rules.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal, SupportsFloat, SupportsIndex, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from pyproj import Geod

Contrast = Literal["S1_minus_S0", "S1_minus_SP"]
Metric = Literal["IG", "recall"]
Model = Literal["S0", "S1", "SP"]
GateStatus = Literal["invalid", "evidence_insufficient", "failed", "passed_development_signal"]
CellKey: TypeAlias = tuple[int, int]
MetricKey: TypeAlias = tuple[Contrast, Metric]
FoldMetricKey: TypeAlias = tuple[Contrast, Metric, int]
HorizonMetricKey: TypeAlias = tuple[Contrast, Metric, int]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CONTRASTS: tuple[Contrast, ...] = ("S1_minus_S0", "S1_minus_SP")
METRICS: tuple[Metric, ...] = ("IG", "recall")
MODEL_IDS: tuple[Model, ...] = ("S0", "S1", "SP")
FOLDS = (1, 2, 3)
HORIZONS = (7, 30, 90)
SIGN_TOLERANCE = 1.0e-12
ADDITIVE_TOLERANCE = 1.0e-10
BOOTSTRAP_REPLICATIONS = 2000
BOOTSTRAP_NAMESPACE = (
    "seismoflux|stage2s-causal-seismicity-development-v1|G1-T|"
    "paired-event-block-bootstrap|PCG64|root-seed=147|v1"
)
BOOTSTRAP_ENTROPY = int.from_bytes(
    hashlib.sha256(BOOTSTRAP_NAMESPACE.encode("utf-8")).digest()[:16],
    "big",
)
_GEOD = Geod(ellps="WGS84")


class Stage2SEvaluationInvalid(ValueError):
    """Raised when an identity or numerical invariant makes evaluation invalid."""


class Stage2SEvidenceInsufficient(ValueError):
    """Raised when a frozen evaluation is valid but lacks required evidence."""


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool | str | bytes | bytearray):
        raise TypeError(f"{label} must be numeric")
    result = float(cast(SupportsFloat | SupportsIndex, value))
    if not math.isfinite(result):
        raise Stage2SEvaluationInvalid(f"{label} must be finite")
    return result


def _readonly_float_vector(value: object, *, label: str, length: int | None = None) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1:
        raise Stage2SEvaluationInvalid(f"{label} must be one-dimensional")
    if length is not None and result.shape != (length,):
        raise Stage2SEvaluationInvalid(f"{label} has the wrong length")
    if not np.isfinite(result).all():
        raise Stage2SEvaluationInvalid(f"{label} must be finite")
    result.setflags(write=False)
    return result


def _readonly_issue_mass_matrix(value: object, *, label: str) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise Stage2SEvaluationInvalid(
            f"{label} must be a non-empty issue-by-operational-cell matrix"
        )
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise Stage2SEvaluationInvalid(f"{label} must be finite and non-negative")
    for issue_index, issue_mass in enumerate(result):
        total = math.fsum(float(value) for value in issue_mass)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=SIGN_TOLERANCE):
            raise Stage2SEvaluationInvalid(f"{label} issue {issue_index} masses must sum to one")
    result.setflags(write=False)
    return result


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


@dataclass(frozen=True, slots=True)
class CellScore:
    """One frozen fold-by-horizon score with per-event paired contributions."""

    fold_index: int
    horizon_days: int
    issue_count: int
    event_ids: tuple[str, ...]
    supported_ig: NDArray[np.bool_]
    hit_by_model: Mapping[Model, BoolArray]
    ig_event_log_ratios: Mapping[Contrast, FloatArray]
    recall_hit_differences: Mapping[Contrast, FloatArray]
    compensator_differences: Mapping[Contrast, float]
    information_gain: Mapping[Contrast, float]
    recall_gain: Mapping[Contrast, float]

    def __post_init__(self) -> None:
        if self.fold_index not in FOLDS or self.horizon_days not in HORIZONS:
            raise Stage2SEvaluationInvalid("cell score has an unknown fold or horizon")
        if (
            not isinstance(self.issue_count, int)
            or isinstance(self.issue_count, bool)
            or self.issue_count <= 0
        ):
            raise Stage2SEvaluationInvalid("cell score issue_count must be positive")
        event_ids = tuple(self.event_ids)
        if not event_ids or len(set(event_ids)) != len(event_ids):
            raise Stage2SEvaluationInvalid("cell score event IDs must be non-empty and unique")
        supported = np.array(self.supported_ig, dtype=np.bool_, copy=True)
        if supported.shape != (len(event_ids),):
            raise Stage2SEvaluationInvalid("supported_ig must align with event IDs")
        supported.setflags(write=False)
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "supported_ig", supported)
        if tuple(self.hit_by_model) != MODEL_IDS:
            raise Stage2SEvaluationInvalid("cell model hits must be ordered S0, S1, SP")
        model_hits: dict[Model, BoolArray] = {}
        for model in MODEL_IDS:
            values = np.array(self.hit_by_model[model], dtype=np.bool_, copy=True)
            if values.shape != (len(event_ids),):
                raise Stage2SEvaluationInvalid(f"{model} cell hits must align with event IDs")
            values.setflags(write=False)
            model_hits[model] = values
        object.__setattr__(self, "hit_by_model", MappingProxyType(model_hits))
        event_terms: dict[Contrast, FloatArray] = {}
        recall_terms: dict[Contrast, FloatArray] = {}
        compensators: dict[Contrast, float] = {}
        information_gain: dict[Contrast, float] = {}
        recall_gain: dict[Contrast, float] = {}
        for contrast in CONTRASTS:
            event_terms[contrast] = _readonly_float_vector(
                self.ig_event_log_ratios[contrast],
                label=f"{contrast} IG event terms",
                length=int(np.count_nonzero(supported)),
            )
            recall_terms[contrast] = _readonly_float_vector(
                self.recall_hit_differences[contrast],
                label=f"{contrast} recall event terms",
                length=len(event_ids),
            )
            compensators[contrast] = _finite_float(
                self.compensator_differences[contrast],
                label="compensator difference",
            )
            information_gain[contrast] = _finite_float(
                self.information_gain[contrast],
                label="information gain",
            )
            recall_gain[contrast] = _finite_float(
                self.recall_gain[contrast],
                label="recall gain",
            )
        object.__setattr__(self, "ig_event_log_ratios", MappingProxyType(event_terms))
        object.__setattr__(self, "recall_hit_differences", MappingProxyType(recall_terms))
        object.__setattr__(
            self,
            "compensator_differences",
            MappingProxyType(compensators),
        )
        object.__setattr__(self, "information_gain", MappingProxyType(information_gain))
        object.__setattr__(self, "recall_gain", MappingProxyType(recall_gain))


def score_fold_horizon(
    *,
    fold_index: int,
    horizon_days: int,
    event_ids: Sequence[str],
    supported_ig: object,
    log_density_by_model: Mapping[str, object],
    hit_by_model: Mapping[str, object],
    operational_mass_by_model: Mapping[str, object],
    shared_rate_per_day: float,
) -> CellScore:
    """Score one frozen cell using continuous event densities and independent compensators."""

    identifiers = tuple(event_ids)
    if tuple(log_density_by_model) != ("S0", "S1", "SP"):
        raise Stage2SEvaluationInvalid("log densities must be ordered S0, S1, SP")
    if tuple(hit_by_model) != ("S0", "S1", "SP"):
        raise Stage2SEvaluationInvalid("hits must be ordered S0, S1, SP")
    if tuple(operational_mass_by_model) != ("S0", "S1", "SP"):
        raise Stage2SEvaluationInvalid("masses must be ordered S0, S1, SP")
    supported = np.asarray(supported_ig, dtype=np.bool_)
    if supported.shape != (len(identifiers),):
        raise Stage2SEvaluationInvalid("supported_ig must align with event IDs")
    supported_count = int(np.count_nonzero(supported))
    if supported_count == 0:
        raise Stage2SEvidenceInsufficient("supported IG cell has zero events")
    if not identifiers:
        raise Stage2SEvidenceInsufficient("recall cell has zero events")

    log_density = {
        model: _readonly_float_vector(
            log_density_by_model[model],
            label=f"{model} log density",
            length=len(identifiers),
        )
        for model in ("S0", "S1", "SP")
    }
    hits: dict[Model, NDArray[np.bool_]] = {}
    for model in MODEL_IDS:
        hit_values = np.asarray(hit_by_model[model], dtype=np.bool_)
        if hit_values.shape != (len(identifiers),):
            raise Stage2SEvaluationInvalid(f"{model} hits must align with event IDs")
        hits[model] = hit_values

    mass_arrays = {
        model: _readonly_issue_mass_matrix(
            operational_mass_by_model[model],
            label=f"{model} issue masses",
        )
        for model in ("S0", "S1", "SP")
    }
    issue_mass_shape = mass_arrays["S0"].shape
    if any(values.shape != issue_mass_shape for values in mass_arrays.values()):
        raise Stage2SEvaluationInvalid("all models must share one issue count and operational grid")
    issue_count = int(issue_mass_shape[0])
    unsupported = np.logical_not(supported)
    for model, hit_values in hits.items():
        if np.any(hit_values[unsupported]):
            raise Stage2SEvaluationInvalid(
                f"{model} must force every unsupported strict-recall target to a miss"
            )

    rate = _finite_float(shared_rate_per_day, label="shared rate")
    if rate <= 0.0:
        raise Stage2SEvidenceInsufficient("shared M5_6 rate must be positive")
    duration = float(horizon_days)
    comparisons: Mapping[Contrast, tuple[Model, Model]] = {
        "S1_minus_S0": ("S1", "S0"),
        "S1_minus_SP": ("S1", "SP"),
    }
    event_terms: dict[Contrast, FloatArray] = {}
    recall_terms: dict[Contrast, FloatArray] = {}
    compensators: dict[Contrast, float] = {}
    information_gain: dict[Contrast, float] = {}
    recall_gain: dict[Contrast, float] = {}
    for contrast, (candidate, comparator) in comparisons.items():
        log_ratio = np.asarray(
            log_density[candidate][supported] - log_density[comparator][supported],
            dtype=np.float64,
        )
        log_ratio.setflags(write=False)
        hit_difference = np.asarray(
            hits[candidate].astype(np.float64) - hits[comparator].astype(np.float64),
            dtype=np.float64,
        )
        hit_difference.setflags(write=False)
        candidate_compensator = math.fsum(
            rate * duration * math.fsum(float(value) for value in issue_mass)
            for issue_mass in mass_arrays[candidate]
        )
        comparator_compensator = math.fsum(
            rate * duration * math.fsum(float(value) for value in issue_mass)
            for issue_mass in mass_arrays[comparator]
        )
        compensator = candidate_compensator - comparator_compensator
        if abs(compensator) > ADDITIVE_TOLERANCE:
            raise Stage2SEvaluationInvalid("paired global compensator differs above tolerance")
        event_terms[contrast] = log_ratio
        recall_terms[contrast] = hit_difference
        compensators[contrast] = compensator
        information_gain[contrast] = (
            math.fsum(float(value) for value in log_ratio) - compensator
        ) / supported_count
        recall_gain[contrast] = math.fsum(float(value) for value in hit_difference) / len(
            identifiers
        )
    return CellScore(
        fold_index=fold_index,
        horizon_days=horizon_days,
        issue_count=issue_count,
        event_ids=identifiers,
        supported_ig=supported,
        hit_by_model=hits,
        ig_event_log_ratios=event_terms,
        recall_hit_differences=recall_terms,
        compensator_differences=compensators,
        information_gain=information_gain,
        recall_gain=recall_gain,
    )


@dataclass(frozen=True, slots=True)
class EventBlock:
    """One physical event carrying all frozen horizon memberships and paired terms."""

    event_id: str
    origin_time_utc: datetime
    fold_index: int
    horizons: tuple[int, ...]
    supported_ig: bool
    ig_by_contrast_horizon: Mapping[tuple[Contrast, int], float]
    recall_by_contrast_horizon: Mapping[tuple[Contrast, int], float]

    def __post_init__(self) -> None:
        if not self.event_id:
            raise Stage2SEvaluationInvalid("event_id must not be empty")
        if self.origin_time_utc.tzinfo is None:
            raise Stage2SEvaluationInvalid("origin_time_utc must be timezone-aware")
        if self.fold_index not in FOLDS:
            raise Stage2SEvaluationInvalid("event block fold is invalid")
        horizons = tuple(sorted(set(self.horizons)))
        if not horizons or any(value not in HORIZONS for value in horizons):
            raise Stage2SEvaluationInvalid("event block horizons are invalid")
        object.__setattr__(self, "horizons", horizons)
        for contrast in CONTRASTS:
            for horizon in horizons:
                _finite_float(
                    self.recall_by_contrast_horizon[(contrast, horizon)],
                    label="recall contribution",
                )
                key = (contrast, horizon)
                if self.supported_ig:
                    _finite_float(self.ig_by_contrast_horizon[key], label="IG contribution")
                elif key in self.ig_by_contrast_horizon:
                    raise Stage2SEvaluationInvalid("unsupported event cannot carry an IG term")

    @property
    def membership_signature(self) -> int:
        bits = "".join("1" if horizon in self.horizons else "0" for horizon in HORIZONS)
        return int(bits, 2)


@dataclass(frozen=True, slots=True)
class FamilyInterval:
    point: float
    lower: float
    upper: float
    replicates: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BootstrapFamilies:
    entropy_uint128: int
    intervals: Mapping[MetricKey, FamilyInterval]
    cell_denominators: Mapping[tuple[int, int, Metric], int]


def bootstrap_draw_indices(
    *,
    stratum_sizes: Sequence[int],
    replications: int = BOOTSTRAP_REPLICATIONS,
) -> tuple[tuple[NDArray[np.int64], ...], ...]:
    """Expose the exact single-stream draw order for deterministic testing and reuse."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in stratum_sizes
    ):
        raise ValueError("stratum sizes must be positive integers")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_ENTROPY))
    all_draws: list[tuple[NDArray[np.int64], ...]] = []
    for _ in range(replications):
        replication: list[NDArray[np.int64]] = []
        for size in stratum_sizes:
            draw = generator.integers(
                low=0,
                high=size,
                size=size,
                dtype=np.int64,
                endpoint=False,
            )
            draw.setflags(write=False)
            replication.append(draw)
        all_draws.append(tuple(replication))
    return tuple(all_draws)


def _macro_values(
    *,
    events: Sequence[EventBlock],
    multiplicities: Sequence[int],
    compensators: Mapping[tuple[Contrast, int, int], float],
    denominators: Mapping[tuple[int, int, Metric], int],
) -> dict[MetricKey, float]:
    values: dict[MetricKey, float] = {}
    for contrast in CONTRASTS:
        cell_ig: list[float] = []
        cell_recall: list[float] = []
        for fold_index in FOLDS:
            for horizon in HORIZONS:
                ig_sum = math.fsum(
                    multiplicity * event.ig_by_contrast_horizon[(contrast, horizon)]
                    for event, multiplicity in zip(events, multiplicities, strict=True)
                    if event.fold_index == fold_index
                    and horizon in event.horizons
                    and event.supported_ig
                )
                recall_sum = math.fsum(
                    multiplicity * event.recall_by_contrast_horizon[(contrast, horizon)]
                    for event, multiplicity in zip(events, multiplicities, strict=True)
                    if event.fold_index == fold_index and horizon in event.horizons
                )
                ig_denominator = denominators[(fold_index, horizon, "IG")]
                recall_denominator = denominators[(fold_index, horizon, "recall")]
                cell_ig.append(
                    (ig_sum - compensators[(contrast, fold_index, horizon)]) / ig_denominator
                )
                cell_recall.append(recall_sum / recall_denominator)
        values[(contrast, "IG")] = math.fsum(cell_ig) / 9.0
        values[(contrast, "recall")] = math.fsum(cell_recall) / 9.0
    return values


def bootstrap_families(
    events: Sequence[EventBlock],
    *,
    compensators: Mapping[tuple[Contrast, int, int], float],
    replications: int = BOOTSTRAP_REPLICATIONS,
) -> BootstrapFamilies:
    """Run the frozen paired physical-event block Bootstrap for both metric families."""

    if replications != BOOTSTRAP_REPLICATIONS:
        raise Stage2SEvaluationInvalid("Stage2S requires exactly 2000 Bootstrap replications")
    ordered_events = tuple(events)
    if not ordered_events or len({event.event_id for event in ordered_events}) != len(
        ordered_events
    ):
        raise Stage2SEvaluationInvalid("Bootstrap events must be non-empty and unique")
    expected_compensators = {
        (contrast, fold_index, horizon)
        for contrast in CONTRASTS
        for fold_index in FOLDS
        for horizon in HORIZONS
    }
    if set(compensators) != expected_compensators:
        raise Stage2SEvaluationInvalid("Bootstrap compensator cells are incomplete")
    frozen_compensators = {
        key: _finite_float(value, label="Bootstrap compensator")
        for key, value in compensators.items()
    }

    denominator_counts: dict[tuple[int, int, Metric], int] = {}
    for fold_index in FOLDS:
        for horizon in HORIZONS:
            recall_count = sum(
                event.fold_index == fold_index and horizon in event.horizons
                for event in ordered_events
            )
            ig_count = sum(
                event.fold_index == fold_index and horizon in event.horizons and event.supported_ig
                for event in ordered_events
            )
            if recall_count == 0 or ig_count == 0:
                raise Stage2SEvidenceInsufficient("a fold-horizon cell has zero events")
            denominator_counts[(fold_index, horizon, "IG")] = ig_count
            denominator_counts[(fold_index, horizon, "recall")] = recall_count

    strata: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, event in enumerate(ordered_events):
        strata[(event.fold_index, event.membership_signature, int(event.supported_ig))].append(
            index
        )
    ordered_strata = tuple(sorted(strata))
    for indices in strata.values():
        indices.sort(
            key=lambda index: (
                ordered_events[index].origin_time_utc,
                _utf8_key(ordered_events[index].event_id),
            )
        )
    draws = bootstrap_draw_indices(
        stratum_sizes=[len(strata[key]) for key in ordered_strata],
        replications=replications,
    )
    point = _macro_values(
        events=ordered_events,
        multiplicities=[1] * len(ordered_events),
        compensators=frozen_compensators,
        denominators=denominator_counts,
    )
    replicate_values = {
        key: np.empty(replications, dtype=np.float64)
        for key in ((contrast, metric) for contrast in CONTRASTS for metric in METRICS)
    }
    for replicate_index, replication_draws in enumerate(draws):
        multiplicities = [0] * len(ordered_events)
        for stratum_key, draw in zip(ordered_strata, replication_draws, strict=True):
            indices = strata[stratum_key]
            for local_index in draw:
                multiplicities[indices[int(local_index)]] += 1
        observed = _macro_values(
            events=ordered_events,
            multiplicities=multiplicities,
            compensators=frozen_compensators,
            denominators=denominator_counts,
        )
        for key, value in observed.items():
            if not math.isfinite(value):
                raise Stage2SEvaluationInvalid("Bootstrap replication is non-finite")
            replicate_values[key][replicate_index] = value

    intervals: dict[MetricKey, FamilyInterval] = {}
    for key, values in replicate_values.items():
        lower, upper = np.quantile(values, [0.0125, 0.9875], method="linear")
        intervals[key] = FamilyInterval(
            point=point[key],
            lower=float(lower),
            upper=float(upper),
            replicates=tuple(float(value) for value in values),
        )
    return BootstrapFamilies(
        entropy_uint128=BOOTSTRAP_ENTROPY,
        intervals=intervals,
        cell_denominators=denominator_counts,
    )


@dataclass(frozen=True, slots=True)
class RegionContribution:
    zone_id: str
    ig_event_count: int
    recall_event_count: int
    contributions: Mapping[MetricKey, float]

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise Stage2SEvaluationInvalid("zone_id must not be empty")
        if self.ig_event_count < 0 or self.recall_event_count < 0:
            raise Stage2SEvaluationInvalid("regional event counts must be non-negative")
        for key in ((contrast, metric) for contrast in CONTRASTS for metric in METRICS):
            _finite_float(self.contributions[key], label="regional contribution")


@dataclass(frozen=True, slots=True)
class RegionalMetricResult:
    event_bearing_zone_count: int
    positive_event_bearing_zone_count: int
    strongest_positive_zone_id: str | None
    strongest_positive_contribution: float | None
    leave_strongest_out_residual: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class RegionRobustness:
    results: Mapping[MetricKey, RegionalMetricResult]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and all(result.passed for result in self.results.values())


def compute_region_robustness(
    regions: Sequence[RegionContribution],
    *,
    primary_metrics: Mapping[MetricKey, float],
    required_zone_count: int = 39,
) -> RegionRobustness:
    """Check additive closure, event-bearing breadth, and metric-specific LORO."""

    ordered = tuple(regions)
    if len(ordered) != required_zone_count or len({item.zone_id for item in ordered}) != len(
        ordered
    ):
        raise Stage2SEvaluationInvalid("regional results must contain every unique frozen zone")
    results: dict[MetricKey, RegionalMetricResult] = {}
    failures: list[str] = []
    for contrast in CONTRASTS:
        for metric in METRICS:
            key = (contrast, metric)
            primary = _finite_float(primary_metrics[key], label="primary metric")
            contribution_sum = math.fsum(item.contributions[key] for item in ordered)
            if not math.isclose(
                contribution_sum,
                primary,
                rel_tol=0.0,
                abs_tol=ADDITIVE_TOLERANCE,
            ):
                raise Stage2SEvaluationInvalid(f"{contrast} {metric} regional closure failed")
            event_bearing = tuple(
                item
                for item in ordered
                if (item.ig_event_count > 0 if metric == "IG" else item.recall_event_count > 0)
            )
            if len(event_bearing) < 2:
                raise Stage2SEvidenceInsufficient(
                    f"{contrast} {metric} has fewer than two event-bearing zones"
                )
            positive = tuple(
                item for item in event_bearing if item.contributions[key] > SIGN_TOLERANCE
            )
            strongest: RegionContribution | None = None
            residual: float | None = None
            passed = len(positive) >= 2
            if positive:
                strongest = min(
                    positive,
                    key=lambda item: (-item.contributions[key], _utf8_key(item.zone_id)),
                )
                residual = primary - strongest.contributions[key]
                passed = passed and residual > SIGN_TOLERANCE
            if not passed:
                failures.append(f"{contrast}:{metric}:regional_breadth_or_LORO_failed")
            results[key] = RegionalMetricResult(
                event_bearing_zone_count=len(event_bearing),
                positive_event_bearing_zone_count=len(positive),
                strongest_positive_zone_id=None if strongest is None else strongest.zone_id,
                strongest_positive_contribution=(
                    None if strongest is None else strongest.contributions[key]
                ),
                leave_strongest_out_residual=residual,
                passed=passed,
            )
    return RegionRobustness(results=results, failures=tuple(failures))


@dataclass(frozen=True, slots=True)
class SequenceEvent:
    event_id: str
    origin_time_utc: datetime
    longitude: float
    latitude: float
    contributions: Mapping[MetricKey, float]
    model_hit_contributions: Mapping[Model, float]

    def __post_init__(self) -> None:
        if not self.event_id or self.origin_time_utc.tzinfo is None:
            raise Stage2SEvaluationInvalid("sequence event identity/time is invalid")
        longitude = _finite_float(self.longitude, label="longitude")
        latitude = _finite_float(self.latitude, label="latitude")
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise Stage2SEvaluationInvalid("sequence event coordinate is invalid")
        for key in ((contrast, metric) for contrast in CONTRASTS for metric in METRICS):
            _finite_float(self.contributions[key], label="sequence contribution")
        if set(self.model_hit_contributions) != set(MODEL_IDS):
            raise Stage2SEvaluationInvalid("sequence event model-hit family is incomplete")
        frozen_hits: dict[Model, float] = {}
        for model in MODEL_IDS:
            hit = _finite_float(
                self.model_hit_contributions[model],
                label="sequence model-hit contribution",
            )
            if hit < 0.0:
                raise Stage2SEvaluationInvalid(
                    "sequence model-hit contribution must be non-negative"
                )
            frozen_hits[model] = hit
        object.__setattr__(
            self,
            "model_hit_contributions",
            MappingProxyType(frozen_hits),
        )


@dataclass(frozen=True, slots=True)
class SequenceComponent:
    component_id: str
    event_ids: tuple[str, ...]
    contributions: Mapping[MetricKey, float]
    model_hit_contributions: Mapping[Model, float]
    model_hit_fractions: Mapping[Model, float | None]
    ig_fractions: Mapping[Contrast, float | None]
    event_fraction: float
    origin_time_span_days: float
    max_pairwise_geodesic_distance_km: float

    def __post_init__(self) -> None:
        if not self.component_id or not self.event_ids:
            raise Stage2SEvaluationInvalid("sequence component identity is empty")
        if self.component_id != min(self.event_ids, key=_utf8_key):
            raise Stage2SEvaluationInvalid("sequence component ID is not the UTF-8 minimum")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise Stage2SEvaluationInvalid("sequence component event IDs are not unique")
        frozen_contributions: dict[MetricKey, float] = {}
        for key in ((contrast, metric) for contrast in CONTRASTS for metric in METRICS):
            frozen_contributions[key] = _finite_float(
                self.contributions[key],
                label="sequence component contribution",
            )
        span = _finite_float(
            self.origin_time_span_days,
            label="sequence component time span",
        )
        distance = _finite_float(
            self.max_pairwise_geodesic_distance_km,
            label="sequence component distance",
        )
        if span < 0.0 or distance < 0.0:
            raise Stage2SEvaluationInvalid(
                "sequence component span and distance must be non-negative"
            )
        object.__setattr__(self, "origin_time_span_days", span)
        object.__setattr__(self, "max_pairwise_geodesic_distance_km", distance)
        object.__setattr__(
            self,
            "contributions",
            MappingProxyType(frozen_contributions),
        )
        if set(self.model_hit_contributions) != set(MODEL_IDS) or set(
            self.model_hit_fractions
        ) != set(MODEL_IDS):
            raise Stage2SEvaluationInvalid("sequence component model-hit family is incomplete")
        hit_raw: dict[Model, float] = {}
        hit_fractions: dict[Model, float | None] = {}
        for model in MODEL_IDS:
            raw = _finite_float(
                self.model_hit_contributions[model],
                label="sequence component model-hit raw",
            )
            if raw < 0.0:
                raise Stage2SEvaluationInvalid(
                    "sequence component model-hit raw must be non-negative"
                )
            fraction_value = self.model_hit_fractions[model]
            fraction = (
                None
                if fraction_value is None
                else _finite_float(
                    fraction_value,
                    label="sequence component model-hit fraction",
                )
            )
            if fraction is not None and not -ADDITIVE_TOLERANCE <= fraction <= (
                1.0 + ADDITIVE_TOLERANCE
            ):
                raise Stage2SEvaluationInvalid(
                    "sequence component model-hit fraction is outside [0, 1]"
                )
            hit_raw[model] = raw
            hit_fractions[model] = fraction
        if set(self.ig_fractions) != set(CONTRASTS):
            raise Stage2SEvaluationInvalid("sequence component IG fractions are incomplete")
        ig_fractions = {
            contrast: (
                None
                if self.ig_fractions[contrast] is None
                else _finite_float(
                    self.ig_fractions[contrast],
                    label="sequence component IG fraction",
                )
            )
            for contrast in CONTRASTS
        }
        event_fraction = _finite_float(
            self.event_fraction,
            label="sequence component event fraction",
        )
        if not 0.0 < event_fraction <= 1.0:
            raise Stage2SEvaluationInvalid("sequence component event fraction must be in (0, 1]")
        object.__setattr__(
            self,
            "model_hit_contributions",
            MappingProxyType(hit_raw),
        )
        object.__setattr__(
            self,
            "model_hit_fractions",
            MappingProxyType(hit_fractions),
        )
        object.__setattr__(self, "ig_fractions", MappingProxyType(ig_fractions))
        object.__setattr__(self, "event_fraction", event_fraction)


@dataclass(frozen=True, slots=True)
class SequenceDiagnostic:
    components: tuple[SequenceComponent, ...]
    global_residual: Mapping[MetricKey, float]
    primary_model_recall: Mapping[Model, float]
    largest_count_component_id: str
    largest_gain_component_id: Mapping[MetricKey, str]
    leave_largest_count_out: Mapping[MetricKey, float]
    leave_largest_gain_out: Mapping[MetricKey, float]
    claim_limited: bool
    interpretation_limit: str

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components or tuple(component.component_id for component in components) != tuple(
            sorted(
                (component.component_id for component in components),
                key=_utf8_key,
            )
        ):
            raise Stage2SEvaluationInvalid(
                "sequence components must be non-empty and UTF-8 ordered"
            )
        metric_keys = {(contrast, metric) for contrast in CONTRASTS for metric in METRICS}
        if (
            set(self.global_residual) != metric_keys
            or set(self.largest_gain_component_id) != metric_keys
            or set(self.leave_largest_count_out) != metric_keys
            or set(self.leave_largest_gain_out) != metric_keys
        ):
            raise Stage2SEvaluationInvalid("sequence diagnostic metric families are incomplete")
        frozen_residual = {
            key: _finite_float(value, label="sequence global residual")
            for key, value in self.global_residual.items()
        }
        frozen_count_leave = {
            key: _finite_float(value, label="sequence largest-count leave-out")
            for key, value in self.leave_largest_count_out.items()
        }
        frozen_gain_leave = {
            key: _finite_float(value, label="sequence largest-gain leave-out")
            for key, value in self.leave_largest_gain_out.items()
        }
        if set(self.primary_model_recall) != set(MODEL_IDS):
            raise Stage2SEvaluationInvalid("sequence primary model-recall family is incomplete")
        model_recall = {
            model: _finite_float(value, label="sequence primary model recall")
            for model, value in self.primary_model_recall.items()
        }
        if any(value < 0.0 for value in model_recall.values()):
            raise Stage2SEvaluationInvalid("sequence primary model recall must be non-negative")
        component_ids = {component.component_id for component in components}
        if self.largest_count_component_id not in component_ids or any(
            component_id not in component_ids
            for component_id in self.largest_gain_component_id.values()
        ):
            raise Stage2SEvaluationInvalid("sequence diagnostic references unknown components")
        expected_limited = any(value <= SIGN_TOLERANCE for value in frozen_gain_leave.values())
        if self.claim_limited is not expected_limited:
            raise Stage2SEvaluationInvalid(
                "sequence claim limit differs from largest-gain leave-outs"
            )
        expected_limit = (
            "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
            if expected_limited
            else "no_sequence_interpretation_limit"
        )
        if self.interpretation_limit != expected_limit:
            raise Stage2SEvaluationInvalid("sequence interpretation limit is inconsistent")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "global_residual",
            MappingProxyType(frozen_residual),
        )
        object.__setattr__(
            self,
            "primary_model_recall",
            MappingProxyType(model_recall),
        )
        object.__setattr__(
            self,
            "largest_gain_component_id",
            MappingProxyType(dict(self.largest_gain_component_id)),
        )
        object.__setattr__(
            self,
            "leave_largest_count_out",
            MappingProxyType(frozen_count_leave),
        )
        object.__setattr__(
            self,
            "leave_largest_gain_out",
            MappingProxyType(frozen_gain_leave),
        )


@dataclass(frozen=True, slots=True)
class SequenceClosureEvidence:
    expected_event_ids: tuple[str, ...]
    global_residual: Mapping[MetricKey, float]
    primary_model_recall: Mapping[Model, float]

    def __post_init__(self) -> None:
        event_ids = tuple(self.expected_event_ids)
        if (
            not event_ids
            or len(set(event_ids)) != len(event_ids)
            or event_ids != tuple(sorted(event_ids, key=_utf8_key))
        ):
            raise Stage2SEvaluationInvalid("sequence closure event IDs must be unique UTF-8 order")
        metric_keys = {(contrast, metric) for contrast in CONTRASTS for metric in METRICS}
        if set(self.global_residual) != metric_keys:
            raise Stage2SEvaluationInvalid("sequence closure residual family is incomplete")
        residual = {
            key: _finite_float(value, label="sequence closure residual")
            for key, value in self.global_residual.items()
        }
        if any(residual[(contrast, "recall")] != 0.0 for contrast in CONTRASTS):
            raise Stage2SEvaluationInvalid("sequence recall global residual must be exactly zero")
        if set(self.primary_model_recall) != set(MODEL_IDS):
            raise Stage2SEvaluationInvalid("sequence closure model-recall family is incomplete")
        model_recall = {
            model: _finite_float(value, label="sequence model recall")
            for model, value in self.primary_model_recall.items()
        }
        if any(value < 0.0 for value in model_recall.values()):
            raise Stage2SEvaluationInvalid("sequence model recall must be non-negative")
        object.__setattr__(self, "expected_event_ids", event_ids)
        object.__setattr__(
            self,
            "global_residual",
            MappingProxyType(residual),
        )
        object.__setattr__(
            self,
            "primary_model_recall",
            MappingProxyType(model_recall),
        )


def build_sequence_closure_evidence(
    cell_scores: Mapping[tuple[int, int], CellScore],
) -> SequenceClosureEvidence:
    """Independently derive sequence identities, compensator residuals, and hits."""

    expected_cells = {(fold_index, horizon) for fold_index in FOLDS for horizon in HORIZONS}
    if set(cell_scores) != expected_cells:
        raise Stage2SEvaluationInvalid("sequence closure requires all nine fold-horizon cells")
    for key, score in cell_scores.items():
        if key != (score.fold_index, score.horizon_days):
            raise Stage2SEvaluationInvalid("sequence closure cell key and score identity differ")
    event_ids = tuple(
        sorted(
            {event_id for score in cell_scores.values() for event_id in score.event_ids},
            key=_utf8_key,
        )
    )
    global_residual: dict[MetricKey, float] = {}
    for contrast in CONTRASTS:
        ig_terms: list[float] = []
        for score in cell_scores.values():
            supported_count = int(np.count_nonzero(score.supported_ig))
            if supported_count == 0:
                raise Stage2SEvidenceInsufficient("sequence closure has a zero-supported IG cell")
            ig_terms.append(
                -score.compensator_differences[contrast]
                / supported_count
                / float(len(expected_cells))
            )
        global_residual[(contrast, "IG")] = math.fsum(ig_terms)
        global_residual[(contrast, "recall")] = 0.0
    primary_model_recall = {
        model: math.fsum(
            float(np.count_nonzero(score.hit_by_model[model]))
            / len(score.event_ids)
            / float(len(expected_cells))
            for score in cell_scores.values()
        )
        for model in MODEL_IDS
    }
    return SequenceClosureEvidence(
        expected_event_ids=event_ids,
        global_residual=global_residual,
        primary_model_recall=primary_model_recall,
    )


def _sequence_components(events: Sequence[SequenceEvent]) -> tuple[tuple[int, ...], ...]:
    parents = list(range(len(events)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            seconds = abs(
                (events[right].origin_time_utc - events[left].origin_time_utc).total_seconds()
            )
            if seconds > 2_592_000:
                continue
            _, _, distance_m = _GEOD.inv(
                events[left].longitude,
                events[left].latitude,
                events[right].longitude,
                events[right].latitude,
            )
            if distance_m <= 75_000.0:
                union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(events)):
        grouped[find(index)].append(index)
    return tuple(
        tuple(indices)
        for indices in sorted(
            grouped.values(),
            key=lambda indices: min(_utf8_key(events[index].event_id) for index in indices),
        )
    )


def compute_sequence_diagnostic(
    events: Sequence[SequenceEvent],
    *,
    primary_metrics: Mapping[MetricKey, float],
    closure: SequenceClosureEvidence,
) -> SequenceDiagnostic:
    """Build frozen 30-day/75-km connected components and fixed-denominator leave-outs."""

    ordered_events = tuple(
        sorted(events, key=lambda item: (item.origin_time_utc, _utf8_key(item.event_id)))
    )
    if not ordered_events or len({item.event_id for item in ordered_events}) != len(ordered_events):
        raise Stage2SEvaluationInvalid("sequence events must be non-empty and unique")
    observed_event_ids = tuple(sorted((item.event_id for item in ordered_events), key=_utf8_key))
    if observed_event_ids != closure.expected_event_ids:
        raise Stage2SEvaluationInvalid(
            "sequence event contributions do not match the independently derived event union"
        )
    metric_keys = {(contrast, metric) for contrast in CONTRASTS for metric in METRICS}
    if set(primary_metrics) != metric_keys:
        raise Stage2SEvaluationInvalid("sequence primary metric family is incomplete")
    primary = {
        key: _finite_float(value, label="sequence primary metric")
        for key, value in primary_metrics.items()
    }
    component_cores: list[
        tuple[
            str,
            tuple[str, ...],
            dict[MetricKey, float],
            dict[Model, float],
            float,
            float,
        ]
    ] = []
    for indices in _sequence_components(ordered_events):
        event_ids = tuple(
            sorted((ordered_events[index].event_id for index in indices), key=_utf8_key)
        )
        component_id = min(event_ids, key=_utf8_key)
        contributions = {
            key: math.fsum(ordered_events[index].contributions[key] for index in indices)
            for key in ((contrast, metric) for contrast in CONTRASTS for metric in METRICS)
        }
        model_hits = {
            model: math.fsum(
                ordered_events[index].model_hit_contributions[model] for index in indices
            )
            for model in MODEL_IDS
        }
        times = [ordered_events[index].origin_time_utc for index in indices]
        origin_time_span_days = (max(times) - min(times)).total_seconds() / 86_400.0
        distances_km: list[float] = []
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                _, _, distance_m = _GEOD.inv(
                    ordered_events[left_index].longitude,
                    ordered_events[left_index].latitude,
                    ordered_events[right_index].longitude,
                    ordered_events[right_index].latitude,
                )
                distances_km.append(float(distance_m) / 1_000.0)
        component_cores.append(
            (
                component_id,
                event_ids,
                contributions,
                model_hits,
                origin_time_span_days,
                max(distances_km) if distances_km else 0.0,
            )
        )
    components = [
        SequenceComponent(
            component_id=component_id,
            event_ids=event_ids,
            contributions=contributions,
            model_hit_contributions=model_hits,
            model_hit_fractions={
                model: (
                    model_hits[model] / closure.primary_model_recall[model]
                    if closure.primary_model_recall[model] > SIGN_TOLERANCE
                    else None
                )
                for model in MODEL_IDS
            },
            ig_fractions={
                contrast: (
                    contributions[(contrast, "IG")] / primary[(contrast, "IG")]
                    if primary[(contrast, "IG")] > SIGN_TOLERANCE
                    else None
                )
                for contrast in CONTRASTS
            },
            event_fraction=len(event_ids) / len(closure.expected_event_ids),
            origin_time_span_days=origin_time_span_days,
            max_pairwise_geodesic_distance_km=distance,
        )
        for (
            component_id,
            event_ids,
            contributions,
            model_hits,
            origin_time_span_days,
            distance,
        ) in component_cores
    ]
    components.sort(key=lambda item: _utf8_key(item.component_id))
    for model in MODEL_IDS:
        observed_hit = math.fsum(
            component.model_hit_contributions[model] for component in components
        )
        if not math.isclose(
            observed_hit,
            closure.primary_model_recall[model],
            rel_tol=0.0,
            abs_tol=ADDITIVE_TOLERANCE,
        ):
            raise Stage2SEvaluationInvalid(f"{model} sequence model-hit additive closure failed")
    for key in ((contrast, metric) for contrast in CONTRASTS for metric in METRICS):
        observed = (
            math.fsum(item.contributions[key] for item in components) + closure.global_residual[key]
        )
        if not math.isclose(
            observed,
            primary[key],
            rel_tol=0.0,
            abs_tol=ADDITIVE_TOLERANCE,
        ):
            raise Stage2SEvaluationInvalid(f"{key} sequence additive closure failed")
    largest_count = min(
        components,
        key=lambda item: (-len(item.event_ids), _utf8_key(item.component_id)),
    )
    largest_gain: dict[MetricKey, str] = {}
    largest_count_residuals: dict[MetricKey, float] = {}
    largest_gain_residuals: dict[MetricKey, float] = {}
    claim_limited = False
    for key in ((contrast, metric) for contrast in CONTRASTS for metric in METRICS):
        strongest = min(
            components,
            key=lambda item: (-item.contributions[key], _utf8_key(item.component_id)),
        )
        largest_count_residual = primary[key] - largest_count.contributions[key]
        largest_gain_residual = primary[key] - strongest.contributions[key]
        largest_gain[key] = strongest.component_id
        largest_count_residuals[key] = largest_count_residual
        largest_gain_residuals[key] = largest_gain_residual
        if largest_gain_residual <= SIGN_TOLERANCE:
            claim_limited = True
    return SequenceDiagnostic(
        components=tuple(components),
        global_residual=closure.global_residual,
        primary_model_recall=closure.primary_model_recall,
        largest_count_component_id=largest_count.component_id,
        largest_gain_component_id=largest_gain,
        leave_largest_count_out=largest_count_residuals,
        leave_largest_gain_out=largest_gain_residuals,
        claim_limited=claim_limited,
        interpretation_limit=(
            "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
            if claim_limited
            else "no_sequence_interpretation_limit"
        ),
    )


@dataclass(frozen=True, slots=True)
class GateDecision:
    status: GateStatus
    reasons: tuple[str, ...]


def decide_stage2s(
    *,
    invalid_reasons: Sequence[str] = (),
    evidence_insufficient_reasons: Sequence[str] = (),
    failed_reasons: Sequence[str] = (),
) -> GateDecision:
    """Apply the frozen decision priority without accepting an external decision callback."""

    invalid = tuple(invalid_reasons)
    insufficient = tuple(evidence_insufficient_reasons)
    failed = tuple(failed_reasons)
    if invalid:
        return GateDecision(status="invalid", reasons=invalid)
    if insufficient:
        return GateDecision(status="evidence_insufficient", reasons=insufficient)
    if failed:
        return GateDecision(status="failed", reasons=failed)
    return GateDecision(status="passed_development_signal", reasons=())


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    """One frozen predictor-feed delay sensitivity result."""

    delay_days: int
    values: Mapping[MetricKey, float]

    def __post_init__(self) -> None:
        if self.delay_days not in {1, 7}:
            raise Stage2SEvaluationInvalid("latency delay must be exactly 1 or 7 days")
        expected = {(contrast, metric) for contrast in CONTRASTS for metric in METRICS}
        if set(self.values) != expected:
            raise Stage2SEvaluationInvalid("latency metrics are incomplete")
        frozen = {
            key: _finite_float(value, label="latency metric") for key, value in self.values.items()
        }
        object.__setattr__(self, "values", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True)
class GateAssessment:
    """Complete internally-derived G1-T decision evidence."""

    decision: GateDecision
    supported_event_union_count: int
    recall_event_union_count: int
    fold_macros: Mapping[FoldMetricKey, float]
    horizon_macros: Mapping[HorizonMetricKey, float]
    overall_macros: Mapping[MetricKey, float]
    claim_limited: bool
    interpretation_limit: str

    def __post_init__(self) -> None:
        expected = (
            "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
            if self.claim_limited
            else "no_sequence_interpretation_limit"
        )
        if self.interpretation_limit != expected:
            raise Stage2SEvaluationInvalid(
                "gate interpretation limit differs from sequence claim limit"
            )


def _cell_metric(score: CellScore, contrast: Contrast, metric: Metric) -> float:
    return score.information_gain[contrast] if metric == "IG" else score.recall_gain[contrast]


def descriptive_sp_minus_s0_point_estimates(
    overall_macros: Mapping[MetricKey, float],
) -> Mapping[Metric, float]:
    """Derive SP-S0 point estimates only; this contrast never enters CI or gate."""

    expected = {(contrast, metric) for contrast in CONTRASTS for metric in METRICS}
    if set(overall_macros) != expected:
        raise Stage2SEvaluationInvalid(
            "descriptive SP-S0 derivation requires all primary overall macros"
        )
    return MappingProxyType(
        {
            metric: _finite_float(
                overall_macros[("S1_minus_S0", metric)] - overall_macros[("S1_minus_SP", metric)],
                label=f"descriptive SP-S0 {metric}",
            )
            for metric in METRICS
        }
    )


def evaluate_stage2s_gate(
    cell_scores: Mapping[tuple[int, int], CellScore],
    *,
    bootstrap: BootstrapFamilies,
    regional: RegionRobustness,
    latency: Sequence[LatencyMetrics],
    sequence: SequenceDiagnostic,
) -> GateAssessment:
    """Derive the frozen G1-T decision without an external gate or metric callback."""

    expected_cells = {(fold_index, horizon) for fold_index in FOLDS for horizon in HORIZONS}
    if set(cell_scores) != expected_cells:
        raise Stage2SEvaluationInvalid("G1-T requires all nine fold-horizon cells")
    for cell_key, score in cell_scores.items():
        if cell_key != (score.fold_index, score.horizon_days):
            raise Stage2SEvaluationInvalid("cell score key and identity differ")

    event_folds: dict[str, int] = {}
    event_support: dict[str, bool] = {}
    for score in cell_scores.values():
        for event_id, supported in zip(
            score.event_ids,
            score.supported_ig,
            strict=True,
        ):
            prior_fold = event_folds.setdefault(event_id, score.fold_index)
            if prior_fold != score.fold_index:
                raise Stage2SEvaluationInvalid(
                    "one assessment event cannot belong to multiple disjoint folds"
                )
            support_value = bool(supported)
            prior_support = event_support.setdefault(event_id, support_value)
            if prior_support is not support_value:
                raise Stage2SEvaluationInvalid(
                    "one physical event has inconsistent support membership"
                )
    recall_union_count = len(event_folds)
    supported_union_count = sum(event_support.values())

    fold_macros: dict[FoldMetricKey, float] = {}
    horizon_macros: dict[HorizonMetricKey, float] = {}
    overall_macros: dict[MetricKey, float] = {}
    for contrast in CONTRASTS:
        for metric in METRICS:
            for fold_index in FOLDS:
                fold_macros[(contrast, metric, fold_index)] = (
                    math.fsum(
                        _cell_metric(
                            cell_scores[(fold_index, horizon)],
                            contrast,
                            metric,
                        )
                        for horizon in HORIZONS
                    )
                    / 3.0
                )
            for horizon in HORIZONS:
                horizon_macros[(contrast, metric, horizon)] = (
                    math.fsum(
                        _cell_metric(
                            cell_scores[(fold_index, horizon)],
                            contrast,
                            metric,
                        )
                        for fold_index in FOLDS
                    )
                    / 3.0
                )
            overall_macros[(contrast, metric)] = (
                math.fsum(fold_macros[(contrast, metric, fold_index)] for fold_index in FOLDS) / 3.0
            )

    expected_interval_keys = {(contrast, metric) for contrast in CONTRASTS for metric in METRICS}
    if (
        bootstrap.entropy_uint128 != BOOTSTRAP_ENTROPY
        or set(bootstrap.intervals) != expected_interval_keys
    ):
        raise Stage2SEvaluationInvalid("Bootstrap family identity is invalid")
    for metric_key, interval in bootstrap.intervals.items():
        if len(interval.replicates) != BOOTSTRAP_REPLICATIONS:
            raise Stage2SEvaluationInvalid("Bootstrap family must contain exactly 2000 rows")
        for label, value in (
            ("point", interval.point),
            ("lower", interval.lower),
            ("upper", interval.upper),
        ):
            _finite_float(value, label=f"Bootstrap {label}")
        if interval.lower > interval.upper:
            raise Stage2SEvaluationInvalid("Bootstrap interval bounds are reversed")
        if not math.isclose(
            interval.point,
            overall_macros[metric_key],
            rel_tol=0.0,
            abs_tol=ADDITIVE_TOLERANCE,
        ):
            raise Stage2SEvaluationInvalid("Bootstrap point and nine-cell macro do not close")
    if set(regional.results) != expected_interval_keys:
        raise Stage2SEvaluationInvalid("regional robustness family is incomplete")
    latency_by_delay = {item.delay_days: item for item in latency}
    if len(latency_by_delay) != len(tuple(latency)) or set(latency_by_delay) != {1, 7}:
        raise Stage2SEvaluationInvalid("latency evidence must contain delays 1 and 7 once each")

    insufficient: list[str] = []
    failed: list[str] = []
    if supported_union_count < 20:
        insufficient.append("supported_IG_unique_physical_event_union_below_20")
    if recall_union_count < 20:
        insufficient.append("full_area_recall_unique_physical_event_union_below_20")
    for delay_days in (1, 7):
        for contrast in CONTRASTS:
            for metric in METRICS:
                if latency_by_delay[delay_days].values[(contrast, metric)] <= SIGN_TOLERANCE:
                    insufficient.append(f"latency_{delay_days}d:{contrast}:{metric}:lte_1e_12")

    for contrast in CONTRASTS:
        if any(horizon_macros[(contrast, "IG", horizon)] <= 0.0 for horizon in HORIZONS):
            failed.append(f"{contrast}:not_all_horizon_IG_points_positive")
        ig_folds = [fold_macros[(contrast, "IG", fold_index)] for fold_index in FOLDS]
        if sum(value > 0.0 for value in ig_folds) < 2:
            failed.append(f"{contrast}:fewer_than_two_positive_fold_IG_macros")
        if sorted(ig_folds)[1] <= 0.0:
            failed.append(f"{contrast}:fold_IG_median_not_positive")
        if bootstrap.intervals[(contrast, "IG")].lower <= 0.0:
            failed.append(f"{contrast}:IG_familywise_lower_not_positive")

        recall_folds = [fold_macros[(contrast, "recall", fold_index)] for fold_index in FOLDS]
        if sum(value > 0.0 for value in recall_folds) < 2:
            failed.append(f"{contrast}:fewer_than_two_positive_fold_recall_macros")
        if sorted(recall_folds)[1] <= 0.0:
            failed.append(f"{contrast}:fold_recall_median_not_positive")
        recall_interval = bootstrap.intervals[(contrast, "recall")]
        if recall_interval.lower <= 0.0:
            failed.append(f"{contrast}:recall_familywise_lower_not_positive")
        if contrast == "S1_minus_S0":
            if overall_macros[(contrast, "recall")] < 0.05:
                failed.append(f"{contrast}:recall_gain_below_5pp")
        elif overall_macros[(contrast, "recall")] <= 0.0:
            failed.append(f"{contrast}:recall_gain_not_positive")
    if not regional.passed:
        failed.extend(regional.failures)

    decision = decide_stage2s(
        evidence_insufficient_reasons=insufficient,
        failed_reasons=failed,
    )
    return GateAssessment(
        decision=decision,
        supported_event_union_count=supported_union_count,
        recall_event_union_count=recall_union_count,
        fold_macros=MappingProxyType(fold_macros),
        horizon_macros=MappingProxyType(horizon_macros),
        overall_macros=MappingProxyType(overall_macros),
        claim_limited=sequence.claim_limited,
        interpretation_limit=sequence.interpretation_limit,
    )


__all__ = [
    "ADDITIVE_TOLERANCE",
    "BOOTSTRAP_ENTROPY",
    "BOOTSTRAP_NAMESPACE",
    "BOOTSTRAP_REPLICATIONS",
    "CONTRASTS",
    "HORIZONS",
    "METRICS",
    "MODEL_IDS",
    "SIGN_TOLERANCE",
    "BootstrapFamilies",
    "CellScore",
    "EventBlock",
    "FamilyInterval",
    "GateAssessment",
    "GateDecision",
    "GateStatus",
    "LatencyMetrics",
    "MetricKey",
    "Model",
    "RegionContribution",
    "RegionRobustness",
    "SequenceClosureEvidence",
    "SequenceDiagnostic",
    "SequenceEvent",
    "Stage2SEvaluationInvalid",
    "Stage2SEvidenceInsufficient",
    "bootstrap_draw_indices",
    "bootstrap_families",
    "build_sequence_closure_evidence",
    "compute_region_robustness",
    "compute_sequence_diagnostic",
    "decide_stage2s",
    "descriptive_sp_minus_s0_point_estimates",
    "evaluate_stage2s_gate",
    "score_fold_horizon",
]
