"""Pure scientific scoring primitives for the S1 catalog baselines.

This module is deliberately a thin, file-I/O-free scoring layer.  It accepts
already frozen forecasts and already constructed unique physical targets; it
does not select models, construct folds, or inspect later observations.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from seismoflux.d1_replay.spatial import (
    FROZEN_D1_AREA_BUDGETS_KM2,
    D1AlarmPrefix,
    select_alarm_prefixes,
)
from seismoflux.multitask_s1.time_magnitude import (
    NB2DispersionQualification,
    TruncatedGRMagnitudeModel,
    nb2_at_least_one_probability,
    nb2_logpmf,
    poisson_at_least_one_probability,
    poisson_logpmf,
)
from seismoflux.stage2s.contracts import SpatialGrid

FloatArray = NDArray[np.float64]
RecallBasis = Literal["all", "anchor", "episode_balanced", "subsequent"]
CountDistribution = Literal["poisson", "nb2"]
ScoreStatus = Literal["evaluable", "not_evaluable"]

LOCATION_RECALL_BASES: Final[tuple[RecallBasis, ...]] = (
    "all",
    "anchor",
    "episode_balanced",
    "subsequent",
)
MAGNITUDE_SUPPORT_COMPARABILITY_NOTE: Final = (
    "M0_is_conditional_on_M>=4_and_M3_is_conditional_on_M>=5;_"
    "their_raw_log_scores_must_not_be_compared_directly"
)
MAGNITUDE_M5_COMMON_SUPPORT_NOTE: Final = (
    "direct_M0_vs_M3_comparison_requires_the_exact_same_unique_M>=5_event_ids"
)
JOINT_M5_FACTORIZATION: Final = (
    "log_P(N_M5plus)+sum_log_f(location|M5plus)+sum_log_P(magnitude_bin|M5plus)"
)


def _nonnegative_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonnegative_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _event_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError("event_ids must contain only non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("duplicate physical event_id in one scoring population")
    return result


def _cell_indices(values: Sequence[object], *, cell_count: int) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("event_cell_indices must contain only integers")
        index = int(value)
        if not 0 <= index < cell_count:
            raise ValueError("event cell index is outside the frozen spatial grid")
        result.append(index)
    return tuple(result)


def _normalized_mass_and_prefixes(
    cell_relative_mass: object,
    grid: SpatialGrid,
) -> tuple[FloatArray, tuple[D1AlarmPrefix, ...]]:
    prefixes = select_alarm_prefixes(cell_relative_mass, grid)
    mass = np.array(cell_relative_mass, dtype=np.float64, copy=True, order="C")
    # ``select_alarm_prefixes`` has already checked shape, finiteness,
    # non-negativity, grid length, and unit normalization.
    mass.setflags(write=False)
    return mass, prefixes


def _cell_log_density(mass: FloatArray, grid: SpatialGrid, cell_index: int) -> float:
    cell_mass = float(mass[cell_index])
    if cell_mass == 0.0:
        return -math.inf
    # Subtracting logs avoids overflow for a very small, but valid, clipped cell.
    return math.log(cell_mass) - math.log(float(grid.clipped_area_km2[cell_index]))


def _sum_log_scores(values: Sequence[float]) -> float:
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("log scores may be finite or negative infinity, never NaN/+infinity")
    return math.fsum(values)


@dataclass(frozen=True, slots=True)
class EventLocationLogDensity:
    """One unique physical event scored in its exact clipped grid cell."""

    event_id: str
    cell_index: int
    log_density_per_km2: float


@dataclass(frozen=True, slots=True)
class AlarmRecallScore:
    """Weighted recall for one complete-cell alarm prefix and target basis."""

    basis: RecallBasis
    area_budget_km2: float
    actual_area_km2: float
    hit_weight: float
    total_weight: float
    recall: float | None


@dataclass(frozen=True, slots=True)
class LocationEvaluation:
    event_log_densities: tuple[EventLocationLogDensity, ...]
    mean_log_density_per_event: float | None
    alarm_recall: tuple[AlarmRecallScore, ...]


def score_location_events(
    cell_relative_mass: object,
    grid: SpatialGrid,
    *,
    event_ids: Sequence[str],
    event_cell_indices: Sequence[int],
    episode_ids: Sequence[str],
    episode_member_counts: Sequence[int],
    is_episode_anchor: Sequence[bool],
) -> LocationEvaluation:
    """Score unique target events and all four frozen alarm-recall populations.

    ``episode_balanced`` assigns each represented event ``1/full_member_count``
    from the frozen all-catalog episode ledger.  It therefore does not silently
    renormalize a partly represented episode to weight one.  ``anchor`` and
    ``subsequent`` may legitimately have zero total weight; their recall is then
    ``None`` (NA), never a fabricated zero.
    """

    if not isinstance(grid, SpatialGrid) or grid.cell_size_km != 25.0:
        raise TypeError("location scoring requires the frozen 25 km SpatialGrid")
    mass, prefixes = _normalized_mass_and_prefixes(cell_relative_mass, grid)
    identifiers = _event_ids(event_ids)
    indices = _cell_indices(event_cell_indices, cell_count=grid.cell_count)
    episodes = tuple(episode_ids)
    member_counts = tuple(
        _nonnegative_count("episode_member_count", value) for value in episode_member_counts
    )
    anchors = tuple(is_episode_anchor)
    lengths = {
        len(identifiers),
        len(indices),
        len(episodes),
        len(member_counts),
        len(anchors),
    }
    if len(lengths) != 1:
        raise ValueError("location target vectors must have one common length")
    if any(not isinstance(value, str) or not value.strip() for value in episodes):
        raise ValueError("episode_ids must contain only non-empty strings")
    if any(value == 0 for value in member_counts):
        raise ValueError("episode_member_counts must be positive")
    if any(not isinstance(value, bool) for value in anchors):
        raise TypeError("is_episode_anchor must contain only booleans")

    event_scores = tuple(
        EventLocationLogDensity(
            event_id=event_id,
            cell_index=cell_index,
            log_density_per_km2=_cell_log_density(mass, grid, cell_index),
        )
        for event_id, cell_index in zip(identifiers, indices, strict=True)
    )
    represented_episode_sizes = Counter(episodes)
    frozen_size_by_episode: dict[str, int] = {}
    for episode, member_count in zip(episodes, member_counts, strict=True):
        previous = frozen_size_by_episode.setdefault(episode, member_count)
        if previous != member_count:
            raise ValueError("one episode has inconsistent frozen member_count values")
    if any(
        frozen_size_by_episode[episode] < represented_count
        for episode, represented_count in represented_episode_sizes.items()
    ):
        raise ValueError("frozen episode member_count is smaller than represented target members")
    weights: dict[RecallBasis, tuple[float, ...]] = {
        "all": tuple(1.0 for _ in identifiers),
        "anchor": tuple(1.0 if anchor else 0.0 for anchor in anchors),
        "episode_balanced": tuple(1.0 / member_count for member_count in member_counts),
        "subsequent": tuple(0.0 if anchor else 1.0 for anchor in anchors),
    }
    recall_scores: list[AlarmRecallScore] = []
    for prefix in prefixes:
        selected = set(int(value) for value in prefix.selected_indices)
        for basis in LOCATION_RECALL_BASES:
            basis_weights = weights[basis]
            total_weight = math.fsum(basis_weights)
            hit_weight = math.fsum(
                weight
                for weight, cell_index in zip(basis_weights, indices, strict=True)
                if cell_index in selected
            )
            recall_scores.append(
                AlarmRecallScore(
                    basis=basis,
                    area_budget_km2=prefix.budget_km2,
                    actual_area_km2=prefix.actual_area_km2,
                    hit_weight=hit_weight,
                    total_weight=total_weight,
                    recall=None if total_weight == 0.0 else hit_weight / total_weight,
                )
            )
    return LocationEvaluation(
        event_log_densities=event_scores,
        mean_log_density_per_event=(
            None
            if not event_scores
            else _sum_log_scores(tuple(item.log_density_per_km2 for item in event_scores))
            / len(event_scores)
        ),
        alarm_recall=tuple(recall_scores),
    )


@dataclass(frozen=True, slots=True)
class TimeExposureScore:
    """One count-window score, including windows with zero observed events.

    ``count_bias`` is forecast minus observed count.  For an unevaluable NB2
    dispersion, the distributional scores are NA while the supplied mean bias
    remains identifiable and the observed zero/nonzero window is retained.
    """

    distribution: CountDistribution
    status: ScoreStatus
    reason: str
    observed_count: int
    expected_count: float
    log_score: float | None
    at_least_one_probability: float | None
    at_least_one_brier: float | None
    count_bias: float


def _evaluable_time_score(
    *,
    distribution: CountDistribution,
    reason: str,
    observed_count: int,
    expected_count: float,
    log_score: float,
    at_least_one_probability: float,
) -> TimeExposureScore:
    target = 1.0 if observed_count > 0 else 0.0
    return TimeExposureScore(
        distribution=distribution,
        status="evaluable",
        reason=reason,
        observed_count=observed_count,
        expected_count=expected_count,
        log_score=log_score,
        at_least_one_probability=at_least_one_probability,
        at_least_one_brier=(at_least_one_probability - target) ** 2,
        count_bias=expected_count - observed_count,
    )


def score_poisson_exposure(*, observed_count: int, expected_count: float) -> TimeExposureScore:
    """Score one Poisson exposure without dropping a zero-event window."""

    observed = _nonnegative_count("observed_count", observed_count)
    expected = _nonnegative_finite("expected_count", expected_count)
    return _evaluable_time_score(
        distribution="poisson",
        reason="poisson_distribution_evaluable",
        observed_count=observed,
        expected_count=expected,
        log_score=poisson_logpmf(observed, expected),
        at_least_one_probability=poisson_at_least_one_probability(expected),
    )


def score_nb2_exposure(
    *,
    observed_count: int,
    expected_count: float,
    qualification: NB2DispersionQualification,
) -> TimeExposureScore:
    """Score one NB2 exposure, preserving its frozen qualification branch."""

    observed = _nonnegative_count("observed_count", observed_count)
    expected = _nonnegative_finite("expected_count", expected_count)
    if not isinstance(qualification, NB2DispersionQualification):
        raise TypeError("qualification must be an NB2DispersionQualification")
    if qualification.status == "not_evaluable":
        return TimeExposureScore(
            distribution="nb2",
            status="not_evaluable",
            reason=qualification.reason,
            observed_count=observed,
            expected_count=expected,
            log_score=None,
            at_least_one_probability=None,
            at_least_one_brier=None,
            count_bias=expected - observed,
        )
    if qualification.status == "poisson_limit":
        log_score = poisson_logpmf(observed, expected)
        probability = poisson_at_least_one_probability(expected)
    else:
        assert qualification.dispersion_k is not None
        log_score = nb2_logpmf(observed, expected, qualification.dispersion_k)
        probability = nb2_at_least_one_probability(expected, qualification.dispersion_k)
    return _evaluable_time_score(
        distribution="nb2",
        reason=qualification.reason,
        observed_count=observed,
        expected_count=expected,
        log_score=log_score,
        at_least_one_probability=probability,
    )


@dataclass(frozen=True, slots=True)
class EventMagnitudeLogScore:
    event_id: str
    magnitude: float
    log_probability: float
    is_m6_plus: bool


@dataclass(frozen=True, slots=True)
class MagnitudeEvaluation:
    """A magnitude score tied to one explicit conditional support."""

    model_id: str
    conditional_support: str
    event_scores: tuple[EventMagnitudeLogScore, ...]
    log_probability_sum: float | None
    mean_log_probability: float | None
    m6_plus_probability: float | None
    mean_m6_plus_brier: float | None
    direct_cross_support_comparison_allowed: bool
    comparability_note: str


def _score_magnitude_events(
    model: TruncatedGRMagnitudeModel,
    *,
    required_model_id: Literal["M0_GR_GLOBAL", "M3_GR_LONG_M5"],
    event_ids: Sequence[str],
    magnitudes: Sequence[float],
) -> MagnitudeEvaluation:
    if not isinstance(model, TruncatedGRMagnitudeModel) or model.model_id != required_model_id:
        raise ValueError(f"magnitude scoring requires {required_model_id}")
    identifiers = _event_ids(event_ids)
    values = tuple(float(value) for value in magnitudes)
    if len(identifiers) != len(values):
        raise ValueError("magnitude event IDs and values must have one common length")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("target magnitudes must be finite")
    event_scores = tuple(
        EventMagnitudeLogScore(
            event_id=event_id,
            magnitude=magnitude,
            log_probability=model.log_probability(magnitude),
            is_m6_plus=magnitude >= 6.0,
        )
        for event_id, magnitude in zip(identifiers, values, strict=True)
    )
    log_values = tuple(item.log_probability for item in event_scores)
    total_log_probability = None if not log_values else _sum_log_scores(log_values)
    probability = model.m6_plus_probability_mass
    mean_brier = (
        None
        if not event_scores
        else math.fsum(
            (probability - (1.0 if item.is_m6_plus else 0.0)) ** 2 for item in event_scores
        )
        / len(event_scores)
    )
    if required_model_id == "M0_GR_GLOBAL":
        support = "M>=4 unique physical events"
        comparability_note = MAGNITUDE_SUPPORT_COMPARABILITY_NOTE
    else:
        support = "M>=5 unique physical events, conditional tail"
        comparability_note = MAGNITUDE_M5_COMMON_SUPPORT_NOTE
    return MagnitudeEvaluation(
        model_id=model.model_id,
        conditional_support=support,
        event_scores=event_scores,
        log_probability_sum=total_log_probability,
        mean_log_probability=(
            None if total_log_probability is None else total_log_probability / len(event_scores)
        ),
        m6_plus_probability=probability,
        mean_m6_plus_brier=mean_brier,
        direct_cross_support_comparison_allowed=False,
        comparability_note=comparability_note,
    )


def score_m0_unique_events(
    model: TruncatedGRMagnitudeModel,
    *,
    event_ids: Sequence[str],
    magnitudes: Sequence[float],
) -> MagnitudeEvaluation:
    """Score M0 on unique M>=4 events and its conditional M>=6 Brier target."""

    return _score_magnitude_events(
        model,
        required_model_id="M0_GR_GLOBAL",
        event_ids=event_ids,
        magnitudes=magnitudes,
    )


def score_m3_conditional_m5_events(
    model: TruncatedGRMagnitudeModel,
    *,
    event_ids: Sequence[str],
    magnitudes: Sequence[float],
) -> MagnitudeEvaluation:
    """Score M3 only on its own M>=5 conditional tail support."""

    return _score_magnitude_events(
        model,
        required_model_id="M3_GR_LONG_M5",
        event_ids=event_ids,
        magnitudes=magnitudes,
    )


def score_m0_conditional_m5_events(
    model: TruncatedGRMagnitudeModel,
    *,
    event_ids: Sequence[str],
    magnitudes: Sequence[float],
) -> MagnitudeEvaluation:
    """Re-normalize M0 on M>=5 for a same-event-set M3 sensitivity comparison."""

    if not isinstance(model, TruncatedGRMagnitudeModel) or model.model_id != "M0_GR_GLOBAL":
        raise ValueError("conditional M>=5 comparison requires M0_GR_GLOBAL")
    identifiers = _event_ids(event_ids)
    values = tuple(float(value) for value in magnitudes)
    if len(identifiers) != len(values):
        raise ValueError("magnitude event IDs and values must have one common length")
    if any(not math.isfinite(value) or value < 5.0 for value in values):
        raise ValueError("conditional M0 target magnitudes must be finite and at least M5")
    m5_plus_probability = model.m5_6_probability_mass + model.m6_plus_probability_mass
    if not 0.0 < m5_plus_probability <= 1.0:
        raise ValueError("M0 must assign positive finite probability to its M>=5 tail")
    log_normalizer = math.log(m5_plus_probability)
    event_scores = tuple(
        EventMagnitudeLogScore(
            event_id=event_id,
            magnitude=magnitude,
            log_probability=model.log_probability(magnitude) - log_normalizer,
            is_m6_plus=magnitude >= 6.0,
        )
        for event_id, magnitude in zip(identifiers, values, strict=True)
    )
    log_values = tuple(item.log_probability for item in event_scores)
    total_log_probability = None if not log_values else _sum_log_scores(log_values)
    m6_probability = model.m6_plus_probability_mass / m5_plus_probability
    mean_brier = (
        None
        if not event_scores
        else math.fsum(
            (m6_probability - (1.0 if item.is_m6_plus else 0.0)) ** 2 for item in event_scores
        )
        / len(event_scores)
    )
    return MagnitudeEvaluation(
        model_id=model.model_id,
        conditional_support="M>=5 unique physical events, M0 re-normalized tail",
        event_scores=event_scores,
        log_probability_sum=total_log_probability,
        mean_log_probability=(
            None if total_log_probability is None else total_log_probability / len(event_scores)
        ),
        m6_plus_probability=m6_probability,
        mean_m6_plus_brier=mean_brier,
        direct_cross_support_comparison_allowed=False,
        comparability_note=MAGNITUDE_M5_COMMON_SUPPORT_NOTE,
    )


@dataclass(frozen=True, slots=True)
class M5TailSensitivityComparison:
    """Paired M3-minus-M0 effects on one exact unique M>=5 event population."""

    event_ids: tuple[str, ...]
    magnitudes: tuple[float, ...]
    m0_conditional_m5: MagnitudeEvaluation
    m3_conditional_m5: MagnitudeEvaluation
    mean_log_score_difference_m3_minus_m0: float | None
    mean_m6_brier_difference_m3_minus_m0: float | None


def compare_m0_m3_on_common_m5_support(
    m0_model: TruncatedGRMagnitudeModel,
    m3_model: TruncatedGRMagnitudeModel,
    *,
    event_ids: Sequence[str],
    magnitudes: Sequence[float],
) -> M5TailSensitivityComparison:
    """Pair M0 and M3 before calculating any same-support sensitivity effect."""

    identifiers = _event_ids(event_ids)
    values = tuple(float(value) for value in magnitudes)
    if len(identifiers) != len(values):
        raise ValueError("paired M>=5 event IDs and values must have one common length")
    m0_score = score_m0_conditional_m5_events(
        m0_model,
        event_ids=identifiers,
        magnitudes=values,
    )
    m3_score = score_m3_conditional_m5_events(
        m3_model,
        event_ids=identifiers,
        magnitudes=values,
    )
    m0_identity = tuple((item.event_id, item.magnitude) for item in m0_score.event_scores)
    m3_identity = tuple((item.event_id, item.magnitude) for item in m3_score.event_scores)
    expected_identity = tuple(zip(identifiers, values, strict=True))
    if m0_identity != expected_identity or m3_identity != expected_identity:
        raise ValueError("M0 and M3 common-support event identities or magnitudes differ")
    if m0_score.mean_log_probability is None:
        mean_log_difference = None
    else:
        assert m3_score.mean_log_probability is not None
        mean_log_difference = m3_score.mean_log_probability - m0_score.mean_log_probability
    if m0_score.mean_m6_plus_brier is None:
        mean_brier_difference = None
    else:
        assert m3_score.mean_m6_plus_brier is not None
        mean_brier_difference = m3_score.mean_m6_plus_brier - m0_score.mean_m6_plus_brier
    return M5TailSensitivityComparison(
        event_ids=identifiers,
        magnitudes=values,
        m0_conditional_m5=m0_score,
        m3_conditional_m5=m3_score,
        mean_log_score_difference_m3_minus_m0=mean_log_difference,
        mean_m6_brier_difference_m3_minus_m0=mean_brier_difference,
    )


@dataclass(frozen=True, slots=True)
class JointM5Score:
    """Minimal non-duplicating factorization for the unique M>=5 population."""

    count_distribution: CountDistribution
    event_count: int
    count_log_score: float | None
    conditional_location_log_density_sum: float
    conditional_magnitude_log_probability_sum: float
    joint_log_score: float | None
    factorization: str


def score_minimal_joint_m5(
    cell_relative_mass: object,
    grid: SpatialGrid,
    magnitude_model: TruncatedGRMagnitudeModel,
    *,
    event_ids: Sequence[str],
    event_cell_indices: Sequence[int],
    event_magnitudes: Sequence[float],
    expected_count: float,
    count_distribution: CountDistribution = "poisson",
    nb2_qualification: NB2DispersionQualification | None = None,
) -> JointM5Score:
    """Score one M>=5 count once, then conditional location and magnitude.

    Empty target windows remain valid: the two empty conditional sums are the
    mathematical neutral value zero, while the count PMF still scores ``N=0``.
    An unevaluable NB2 count component makes the joint score NA, not zero.
    """

    if not isinstance(grid, SpatialGrid) or grid.cell_size_km != 25.0:
        raise TypeError("joint scoring requires the frozen 25 km SpatialGrid")
    if (
        not isinstance(magnitude_model, TruncatedGRMagnitudeModel)
        or magnitude_model.model_id != "M0_GR_GLOBAL"
        or magnitude_model.mc != 4.0
    ):
        raise ValueError("joint M>=5 scoring requires the frozen M0_GR_GLOBAL model")
    mass, _ = _normalized_mass_and_prefixes(cell_relative_mass, grid)
    identifiers = _event_ids(event_ids)
    indices = _cell_indices(event_cell_indices, cell_count=grid.cell_count)
    magnitudes = tuple(float(value) for value in event_magnitudes)
    if len({len(identifiers), len(indices), len(magnitudes)}) != 1:
        raise ValueError("joint M>=5 event vectors must have one common length")
    if any(not math.isfinite(value) or value < 5.0 for value in magnitudes):
        raise ValueError("joint target magnitudes must be finite and at least M5")

    location_logs = tuple(_cell_log_density(mass, grid, cell_index) for cell_index in indices)
    m5_plus_probability = (
        magnitude_model.m5_6_probability_mass + magnitude_model.m6_plus_probability_mass
    )
    if not 0.0 < m5_plus_probability <= 1.0:
        raise ValueError("M0 must assign positive finite probability to its M>=5 tail")
    magnitude_logs = tuple(
        magnitude_model.log_probability(magnitude) - math.log(m5_plus_probability)
        for magnitude in magnitudes
    )
    location_sum = _sum_log_scores(location_logs)
    magnitude_sum = _sum_log_scores(magnitude_logs)
    observed_count = len(identifiers)
    if count_distribution == "poisson":
        if nb2_qualification is not None:
            raise ValueError("Poisson joint scoring must not receive an NB2 qualification")
        exposure = score_poisson_exposure(
            observed_count=observed_count,
            expected_count=expected_count,
        )
    elif count_distribution == "nb2":
        if nb2_qualification is None:
            raise ValueError("NB2 joint scoring requires an explicit frozen qualification")
        exposure = score_nb2_exposure(
            observed_count=observed_count,
            expected_count=expected_count,
            qualification=nb2_qualification,
        )
    else:
        raise ValueError("unknown joint count distribution")
    joint = (
        None
        if exposure.log_score is None
        else _sum_log_scores((exposure.log_score, location_sum, magnitude_sum))
    )
    return JointM5Score(
        count_distribution=count_distribution,
        event_count=observed_count,
        count_log_score=exposure.log_score,
        conditional_location_log_density_sum=location_sum,
        conditional_magnitude_log_probability_sum=magnitude_sum,
        joint_log_score=joint,
        factorization=JOINT_M5_FACTORIZATION,
    )


__all__ = [
    "FROZEN_D1_AREA_BUDGETS_KM2",
    "JOINT_M5_FACTORIZATION",
    "LOCATION_RECALL_BASES",
    "MAGNITUDE_M5_COMMON_SUPPORT_NOTE",
    "MAGNITUDE_SUPPORT_COMPARABILITY_NOTE",
    "AlarmRecallScore",
    "CountDistribution",
    "EventLocationLogDensity",
    "EventMagnitudeLogScore",
    "JointM5Score",
    "LocationEvaluation",
    "M5TailSensitivityComparison",
    "MagnitudeEvaluation",
    "RecallBasis",
    "TimeExposureScore",
    "compare_m0_m3_on_common_m5_support",
    "score_location_events",
    "score_m0_conditional_m5_events",
    "score_m0_unique_events",
    "score_m3_conditional_m5_events",
    "score_minimal_joint_m5",
    "score_nb2_exposure",
    "score_poisson_exposure",
]
