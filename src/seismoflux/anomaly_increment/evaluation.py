"""Pure statistical reductions for the frozen stage-4 anomaly-increment protocol.

This module contains no file readers and no model runner.  It operates only on
explicit arrays or immutable evidence records so that every test can remain
synthetic and target blind during the scoring-code freeze.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, TypeAlias

import numpy as np

from seismoflux.anomaly_increment.preregistration import (
    PRIMARY_MACRO_HORIZONS_DAYS,
    STAGE4_HORIZONS_DAYS,
    Stage4SeedContext,
)

GateStatus: TypeAlias = Literal["passed", "failed", "evidence_insufficient"]
MagnitudeBin: TypeAlias = Literal["M5_6", "M6_plus"]
SameRecallStatus: TypeAlias = Literal[
    "evaluable",
    "zero_reference_hits",
    "unreachable_at_maximum_budget",
]
AdoptionChoice: TypeAlias = Literal["dynamic", "snapshot", "background_only"]
AdoptionStatus: TypeAlias = Literal["adopted", "credible_negative", "evidence_insufficient"]

PRIMARY_ALARM_AREA_KM2 = 600_000
MAXIMUM_ALARM_AREA_KM2 = 960_000
ALARM_AREA_STEP_KM2 = 625


def _finite_float(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _unique_ids(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    items = tuple(values)
    if any(not item or item != item.strip() for item in items):
        raise ValueError(f"{label} must contain non-empty trimmed identifiers")
    if len(set(items)) != len(items):
        raise ValueError(f"{label} must contain unique physical-event identifiers")
    return items


def _horizon_mapping(
    values: Mapping[int, int] | Mapping[int, float],
    *,
    label: str,
) -> tuple[tuple[int, int | float], ...]:
    if tuple(sorted(values)) != PRIMARY_MACRO_HORIZONS_DAYS:
        raise ValueError(f"{label} must contain exactly the 7, 30, and 90 day horizons")
    return tuple((horizon, values[horizon]) for horizon in PRIMARY_MACRO_HORIZONS_DAYS)


def information_gain_per_physical_event(
    *,
    event_ids: Sequence[str],
    candidate_event_log_intensities: Sequence[float],
    comparator_event_log_intensities: Sequence[float],
    candidate_integrated_intensity: float,
    comparator_integrated_intensity: float,
) -> float | None:
    """Point-process log-likelihood difference per unique supported event."""

    ids = _unique_ids(event_ids, label="event_ids")
    candidate_logs = tuple(
        _finite_float(value, label="candidate_event_log_intensities")
        for value in candidate_event_log_intensities
    )
    comparator_logs = tuple(
        _finite_float(value, label="comparator_event_log_intensities")
        for value in comparator_event_log_intensities
    )
    if len(candidate_logs) != len(ids) or len(comparator_logs) != len(ids):
        raise ValueError("one candidate and comparator event term is required per event")
    candidate_integral = _finite_float(
        candidate_integrated_intensity, label="candidate_integrated_intensity"
    )
    comparator_integral = _finite_float(
        comparator_integrated_intensity, label="comparator_integrated_intensity"
    )
    if candidate_integral < 0.0 or comparator_integral < 0.0:
        raise ValueError("integrated intensities must be non-negative")
    if not ids:
        return None
    log_likelihood_difference = (
        math.fsum(candidate_logs)
        - candidate_integral
        - math.fsum(comparator_logs)
        + comparator_integral
    )
    return log_likelihood_difference / len(ids)


@dataclass(frozen=True, slots=True)
class RecallEstimate:
    strict_hit_count: int
    all_study_area_event_count: int
    value: float | None


def strict_recall(
    all_study_area_event_ids: Sequence[str],
    strictly_covered_event_ids: Sequence[str],
) -> RecallEstimate:
    """Recall with unsupported study-area events retained as misses."""

    all_ids = _unique_ids(all_study_area_event_ids, label="all_study_area_event_ids")
    covered = _unique_ids(strictly_covered_event_ids, label="strictly_covered_event_ids")
    unknown = set(covered).difference(all_ids)
    if unknown:
        raise ValueError("strictly covered IDs must be study-area physical events")
    hit_count = len(covered)
    value = None if not all_ids else hit_count / len(all_ids)
    return RecallEstimate(hit_count, len(all_ids), value)


def same_area_recall_gain_percentage_points(
    candidate: RecallEstimate,
    comparator: RecallEstimate,
) -> float | None:
    if candidate.all_study_area_event_count != comparator.all_study_area_event_count:
        raise ValueError("same-area recall estimates must use the same denominator")
    if candidate.value is None or comparator.value is None:
        return None
    return 100.0 * (candidate.value - comparator.value)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        lower = _finite_float(self.lower, label="confidence interval lower")
        upper = _finite_float(self.upper, label="confidence interval upper")
        if lower > upper:
            raise ValueError("confidence interval lower must not exceed upper")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


def percentile_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 1 or not np.isfinite(samples).all():
        raise ValueError("percentile interval requires a non-empty finite vector")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, (tail, 1.0 - tail), method="linear")
    return ConfidenceInterval(float(lower), float(upper), confidence_level)


@dataclass(frozen=True, slots=True)
class EventHorizonMembership:
    event_id: str
    magnitude_bin: MagnitudeBin
    membership: tuple[bool, bool, bool, bool, bool]

    def __post_init__(self) -> None:
        if not self.event_id or self.event_id != self.event_id.strip():
            raise ValueError("event_id must be a non-empty trimmed identifier")
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("magnitude_bin is outside the frozen bins")
        if len(self.membership) != len(STAGE4_HORIZONS_DAYS) or not all(
            isinstance(value, bool) for value in self.membership
        ):
            raise ValueError("membership must be five booleans ordered 7,30,90,180,365")
        if not any(self.membership):
            raise ValueError("events with an empty five-window signature are outside the bootstrap")

    @property
    def signature(self) -> str:
        return "".join("1" if value else "0" for value in self.membership)


@dataclass(frozen=True, slots=True)
class BootstrapSample:
    sampled_indices: tuple[int, ...]
    multiplicities: tuple[int, ...]
    stratum_sample_sizes: tuple[tuple[str, int], ...]
    marginal_counts_by_horizon: tuple[tuple[int, int], ...]


def stratified_five_horizon_bootstrap_indices(
    events: Sequence[EventHorizonMembership],
    *,
    context: Stage4SeedContext,
) -> BootstrapSample:
    """Resample within magnitude x five-bit membership strata without refitting."""

    if context.purpose != "bootstrap" or context.partition_role != "joint":
        raise ValueError("five-window bootstrap requires a joint bootstrap context")
    records = tuple(events)
    if not records:
        raise ValueError("events must not be empty")
    _unique_ids(tuple(item.event_id for item in records), label="event IDs")
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        strata[(item.magnitude_bin, item.signature)].append(index)

    generator = context.generator()
    sampled_indices: list[int] = []
    sample_sizes: list[tuple[str, int]] = []
    for (magnitude_bin, signature), members in sorted(strata.items()):
        size = len(members)
        if size == 1:
            sampled = members
        else:
            draws = generator.integers(0, size, size=size, dtype=np.int64)
            sampled = [members[int(draw)] for draw in draws]
        sampled_indices.extend(sampled)
        sample_sizes.append((f"{magnitude_bin}:{signature}", size))

    multiplicities = [0] * len(records)
    for index in sampled_indices:
        multiplicities[index] += 1
    original_marginals = tuple(
        (horizon, sum(event.membership[position] for event in records))
        for position, horizon in enumerate(STAGE4_HORIZONS_DAYS)
    )
    sampled_marginals = tuple(
        (
            horizon,
            sum(
                multiplicities[index]
                for index, event in enumerate(records)
                if event.membership[position]
            ),
        )
        for position, horizon in enumerate(STAGE4_HORIZONS_DAYS)
    )
    if sampled_marginals != original_marginals:
        raise AssertionError("membership-stratified bootstrap changed a horizon marginal")
    return BootstrapSample(
        sampled_indices=tuple(sampled_indices),
        multiplicities=tuple(multiplicities),
        stratum_sample_sizes=tuple(sample_sizes),
        marginal_counts_by_horizon=sampled_marginals,
    )


def frozen_same_recall_budget_grid() -> tuple[int, ...]:
    return tuple(range(0, MAXIMUM_ALARM_AREA_KM2 + ALARM_AREA_STEP_KM2, ALARM_AREA_STEP_KM2))


@dataclass(frozen=True, slots=True)
class AlarmAreaPoint:
    budget_km2: int
    mean_exact_selected_area_km2: float
    strict_hit_count: int

    def __post_init__(self) -> None:
        if self.budget_km2 not in set(frozen_same_recall_budget_grid()):
            raise ValueError("budget_km2 is outside the frozen 625 km2 grid")
        area = _finite_float(
            self.mean_exact_selected_area_km2, label="mean_exact_selected_area_km2"
        )
        if area < 0.0 or area > self.budget_km2:
            raise ValueError("exact selected area must be within its no-interpolation budget")
        if isinstance(self.strict_hit_count, bool) or self.strict_hit_count < 0:
            raise ValueError("strict_hit_count must be non-negative")


@dataclass(frozen=True, slots=True)
class SameRecallAreaResult:
    status: SameRecallStatus
    reference_hit_count: int
    candidate_budget_km2: int | None
    candidate_mean_exact_area_km2: float | None
    background_mean_exact_area_km2: float
    relative_reduction: float
    bootstrap_numeric_value: float
    pass_eligible: bool


def same_recall_union_area_relative_reduction(
    *,
    background_primary: AlarmAreaPoint,
    candidate_budget_profile: Sequence[AlarmAreaPoint],
) -> SameRecallAreaResult:
    """Evaluate the frozen no-interpolation same-recall area branch."""

    if background_primary.budget_km2 != PRIMARY_ALARM_AREA_KM2:
        raise ValueError("background reference must use the frozen 600000 km2 budget")
    profile = tuple(candidate_budget_profile)
    if tuple(item.budget_km2 for item in profile) != frozen_same_recall_budget_grid():
        raise ValueError("candidate profile must contain the complete frozen budget grid")
    if any(
        right.strict_hit_count < left.strict_hit_count
        or right.mean_exact_selected_area_km2 < left.mean_exact_selected_area_km2
        for left, right in pairwise(profile)
    ):
        raise ValueError("candidate hit counts and exact selected areas must be monotone")
    reference_hits = background_primary.strict_hit_count
    background_area = float(background_primary.mean_exact_selected_area_km2)
    if reference_hits == 0:
        return SameRecallAreaResult(
            status="zero_reference_hits",
            reference_hit_count=0,
            candidate_budget_km2=None,
            candidate_mean_exact_area_km2=None,
            background_mean_exact_area_km2=background_area,
            relative_reduction=0.0,
            bootstrap_numeric_value=0.0,
            pass_eligible=False,
        )
    if background_area <= 0.0:
        raise ValueError("positive reference hits require positive background selected area")
    selected = next((item for item in profile if item.strict_hit_count >= reference_hits), None)
    if selected is None:
        return SameRecallAreaResult(
            status="unreachable_at_maximum_budget",
            reference_hit_count=reference_hits,
            candidate_budget_km2=None,
            candidate_mean_exact_area_km2=None,
            background_mean_exact_area_km2=background_area,
            relative_reduction=0.0,
            bootstrap_numeric_value=0.0,
            pass_eligible=False,
        )
    reduction = 1.0 - selected.mean_exact_selected_area_km2 / background_area
    return SameRecallAreaResult(
        status="evaluable",
        reference_hit_count=reference_hits,
        candidate_budget_km2=selected.budget_km2,
        candidate_mean_exact_area_km2=selected.mean_exact_selected_area_km2,
        background_mean_exact_area_km2=background_area,
        relative_reduction=reduction,
        bootstrap_numeric_value=reduction,
        pass_eligible=True,
    )


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool | None
    observed: str
    requirement: str


@dataclass(frozen=True, slots=True)
class GateOutcome:
    gate_id: Literal["G2", "G3"]
    status: GateStatus
    checks: tuple[GateCheck, ...]
    reasons: tuple[str, ...]
    fold_macro_values: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class G2Evidence:
    unique_union_event_count: int
    event_count_by_horizon: Mapping[int, int]
    information_gain_by_horizon: Mapping[int, float]
    macro_information_gain_interval: ConfidenceInterval | None
    time_permutation_p_value: float | None
    dynamic_minus_coverage_interval: ConfidenceInterval | None
    same_area_recall_gain_percentage_points: float | None
    same_area_recall_gain_interval: ConfidenceInterval | None
    same_recall_area_relative_reduction: float | None
    same_recall_area_interval: ConfidenceInterval | None
    same_recall_branch_evaluable: bool
    permutation_scientific_failure_fraction: float = 0.0
    space_permutation_scientific_failure_fraction: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.unique_union_event_count, bool) or self.unique_union_event_count < 0:
            raise ValueError("unique_union_event_count must be non-negative")
        counts = _horizon_mapping(self.event_count_by_horizon, label="event_count_by_horizon")
        if any(isinstance(value, bool) or int(value) < 0 for _, value in counts):
            raise ValueError("horizon event counts must be non-negative integers")
        gains = _horizon_mapping(
            self.information_gain_by_horizon, label="information_gain_by_horizon"
        )
        for _, value in gains:
            _finite_float(float(value), label="information gain")
        if not 0.0 <= self.permutation_scientific_failure_fraction <= 1.0:
            raise ValueError("time-permutation failure fraction must be in [0, 1]")
        if not 0.0 <= self.space_permutation_scientific_failure_fraction <= 1.0:
            raise ValueError("space-permutation failure fraction must be in [0, 1]")
        intervals = (
            self.macro_information_gain_interval,
            self.dynamic_minus_coverage_interval,
            self.same_area_recall_gain_interval,
            self.same_recall_area_interval,
        )
        if any(
            interval is not None and interval.confidence_level != 0.95 for interval in intervals
        ):
            raise ValueError("G2 intervals must use the frozen 95% confidence level")


def _optional_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def evaluate_g2(evidence: G2Evidence) -> GateOutcome:
    """Apply every frozen G2 statistical, confounding, and practical condition."""

    counts = {horizon: int(value) for horizon, value in evidence.event_count_by_horizon.items()}
    gains = {
        horizon: float(value) for horizon, value in evidence.information_gain_by_horizon.items()
    }
    time_p_value = evidence.time_permutation_p_value
    same_area_value = evidence.same_area_recall_gain_percentage_points
    same_recall_value = evidence.same_recall_area_relative_reduction
    insufficient: list[str] = []
    if evidence.unique_union_event_count < 20:
        insufficient.append("fewer_than_20_unique_union_events")
    if any(counts[horizon] == 0 for horizon in PRIMARY_MACRO_HORIZONS_DAYS):
        insufficient.append("zero_event_primary_horizon")
    if evidence.macro_information_gain_interval is None:
        insufficient.append("missing_macro_information_gain_interval")
    if not _optional_finite(time_p_value):
        insufficient.append("missing_time_permutation_p_value")
    if evidence.dynamic_minus_coverage_interval is None:
        insufficient.append("missing_dynamic_minus_coverage_interval")
    if evidence.permutation_scientific_failure_fraction > 0.01:
        insufficient.append("time_permutation_scientific_failure_fraction_above_0_01")
    if evidence.space_permutation_scientific_failure_fraction > 0.01:
        insufficient.append("space_permutation_scientific_failure_fraction_above_0_01")
    if same_area_value is None or evidence.same_area_recall_gain_interval is None:
        same_area_available = False
    else:
        same_area_available = math.isfinite(same_area_value)
    same_recall_available = (
        evidence.same_recall_branch_evaluable
        and _optional_finite(same_recall_value)
        and evidence.same_recall_area_interval is not None
    )
    if not same_area_available and not same_recall_available:
        insufficient.append("no_evaluable_practical_improvement_branch")

    checks = (
        GateCheck(
            "unique_union_event_count",
            None if evidence.unique_union_event_count < 20 else True,
            str(evidence.unique_union_event_count),
            ">=20",
        ),
        GateCheck(
            "every_primary_horizon_has_events",
            None if any(counts[h] == 0 for h in PRIMARY_MACRO_HORIZONS_DAYS) else True,
            ",".join(f"{h}:{counts[h]}" for h in PRIMARY_MACRO_HORIZONS_DAYS),
            "all >0",
        ),
        GateCheck(
            "all_horizon_information_gain_positive",
            all(gains[horizon] > 0.0 for horizon in PRIMARY_MACRO_HORIZONS_DAYS),
            ",".join(f"{h}:{gains[h]:.12g}" for h in PRIMARY_MACRO_HORIZONS_DAYS),
            "all >0",
        ),
        GateCheck(
            "macro_information_gain_lower_bound",
            None
            if evidence.macro_information_gain_interval is None
            else evidence.macro_information_gain_interval.lower > 0.0,
            "missing"
            if evidence.macro_information_gain_interval is None
            else f"{evidence.macro_information_gain_interval.lower:.12g}",
            ">0",
        ),
        GateCheck(
            "time_permutation_p_value",
            None
            if not _optional_finite(time_p_value)
            else time_p_value is not None and time_p_value <= 0.05,
            "missing" if time_p_value is None else f"{time_p_value:.12g}",
            "<=0.05",
        ),
        GateCheck(
            "dynamic_minus_coverage_lower_bound",
            None
            if evidence.dynamic_minus_coverage_interval is None
            else evidence.dynamic_minus_coverage_interval.lower > 0.0,
            "missing"
            if evidence.dynamic_minus_coverage_interval is None
            else f"{evidence.dynamic_minus_coverage_interval.lower:.12g}",
            ">0",
        ),
        GateCheck(
            "same_area_practical_branch",
            None
            if not same_area_available
            else bool(
                same_area_value is not None
                and same_area_value >= 5.0
                and evidence.same_area_recall_gain_interval is not None
                and evidence.same_area_recall_gain_interval.lower > 0.0
            ),
            "not_evaluable" if not same_area_available else f"{same_area_value:.12g} pp",
            ">=5 pp and lower95>0",
        ),
        GateCheck(
            "same_recall_practical_branch",
            None
            if not same_recall_available
            else bool(
                same_recall_value is not None
                and same_recall_value >= 0.10
                and evidence.same_recall_area_interval is not None
                and evidence.same_recall_area_interval.lower > 0.0
            ),
            "not_evaluable" if not same_recall_available else f"{same_recall_value:.12g}",
            ">=0.10 and lower95>0",
        ),
    )
    if insufficient:
        return GateOutcome("G2", "evidence_insufficient", checks, tuple(insufficient))
    core_checks = checks[2:6]
    practical_passed = checks[6].passed is True or checks[7].passed is True
    passed = all(check.passed is True for check in core_checks) and practical_passed
    reasons = () if passed else tuple(check.name for check in checks if check.passed is False)
    return GateOutcome("G2", "passed" if passed else "failed", checks, reasons)


@dataclass(frozen=True, slots=True)
class G3FoldEvidence:
    fold_id: str
    event_count_by_horizon: Mapping[int, int]
    dynamic_minus_snapshot_information_gain_by_horizon: Mapping[int, float]

    def __post_init__(self) -> None:
        if not self.fold_id or self.fold_id != self.fold_id.strip():
            raise ValueError("fold_id must be a non-empty trimmed identifier")
        counts = _horizon_mapping(self.event_count_by_horizon, label="event_count_by_horizon")
        if any(isinstance(value, bool) or int(value) < 0 for _, value in counts):
            raise ValueError("horizon event counts must be non-negative integers")
        gains = _horizon_mapping(
            self.dynamic_minus_snapshot_information_gain_by_horizon,
            label="dynamic_minus_snapshot_information_gain_by_horizon",
        )
        for _, value in gains:
            _finite_float(float(value), label="dynamic-minus-snapshot information gain")


def evaluate_g3(folds: Sequence[G3FoldEvidence]) -> GateOutcome:
    """Apply the three-fold, three-horizon dynamic-vs-snapshot gate."""

    items = tuple(folds)
    if len(items) != 3 or len({item.fold_id for item in items}) != 3:
        raise ValueError("G3 requires exactly three distinct development folds")
    if any(
        item.event_count_by_horizon[horizon] == 0
        for item in items
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS
    ):
        insufficient_checks = (
            GateCheck(
                "all_fold_horizons_have_events",
                None,
                "at_least_one_zero_event_fold_horizon",
                "all >0",
            ),
        )
        return GateOutcome(
            "G3",
            "evidence_insufficient",
            insufficient_checks,
            ("zero_event_fold_horizon_no_partial_macro",),
        )
    fold_macros = tuple(
        statistics.fmean(
            item.dynamic_minus_snapshot_information_gain_by_horizon[horizon]
            for horizon in PRIMARY_MACRO_HORIZONS_DAYS
        )
        for item in items
    )
    positive_count = sum(value > 0.0 for value in fold_macros)
    median = statistics.median(fold_macros)
    checks: tuple[GateCheck, ...] = (
        GateCheck("positive_fold_count", positive_count >= 2, str(positive_count), ">=2 of 3"),
        GateCheck("median_fold_macro", median > 0.0, f"{median:.12g}", ">0"),
    )
    passed = all(check.passed is True for check in checks)
    return GateOutcome(
        "G3",
        "passed" if passed else "failed",
        checks,
        () if passed else tuple(check.name for check in checks if check.passed is False),
        fold_macros,
    )


@dataclass(frozen=True, slots=True)
class AdoptionDecision:
    choice: AdoptionChoice
    status: AdoptionStatus
    reason: str


def apply_preregistered_adoption_matrix(
    *,
    dynamic_g2: GateOutcome,
    dynamic_g3: GateOutcome,
    snapshot_equivalent_g2: GateOutcome | None,
) -> AdoptionDecision:
    """Conservatively apply the frozen no-post-score-fallback adoption matrix."""

    if dynamic_g2.gate_id != "G2" or dynamic_g3.gate_id != "G3":
        raise ValueError("dynamic gate identities do not match G2/G3")
    if snapshot_equivalent_g2 is not None and snapshot_equivalent_g2.gate_id != "G2":
        raise ValueError("snapshot equivalent gate must be G2")
    if dynamic_g2.status == "evidence_insufficient":
        return AdoptionDecision(
            "background_only",
            "evidence_insufficient",
            "dynamic_G2_evidence_insufficient_retain_background_only",
        )
    if dynamic_g2.status == "failed":
        return AdoptionDecision(
            "background_only",
            "credible_negative",
            "dynamic_G2_not_pass_retain_background_and_stop_complex_models",
        )
    if dynamic_g3.status == "passed":
        return AdoptionDecision("dynamic", "adopted", "dynamic_G2_pass_and_G3_pass")
    if dynamic_g3.status == "evidence_insufficient":
        return AdoptionDecision(
            "background_only",
            "evidence_insufficient",
            "dynamic_G3_evidence_insufficient_no_partial_macro",
        )
    if snapshot_equivalent_g2 is None or snapshot_equivalent_g2.status == "evidence_insufficient":
        return AdoptionDecision(
            "background_only",
            "evidence_insufficient",
            "snapshot_equivalent_G2_evidence_unavailable",
        )
    if snapshot_equivalent_g2.status == "passed":
        return AdoptionDecision(
            "snapshot", "adopted", "dynamic_G3_fail_and_snapshot_equivalent_G2_pass"
        )
    return AdoptionDecision(
        "background_only",
        "credible_negative",
        "dynamic_G3_fail_and_snapshot_equivalent_G2_not_pass",
    )


__all__ = [
    "ALARM_AREA_STEP_KM2",
    "MAXIMUM_ALARM_AREA_KM2",
    "PRIMARY_ALARM_AREA_KM2",
    "AdoptionChoice",
    "AdoptionDecision",
    "AdoptionStatus",
    "AlarmAreaPoint",
    "BootstrapSample",
    "ConfidenceInterval",
    "EventHorizonMembership",
    "G2Evidence",
    "G3FoldEvidence",
    "GateCheck",
    "GateOutcome",
    "GateStatus",
    "MagnitudeBin",
    "RecallEstimate",
    "SameRecallAreaResult",
    "SameRecallStatus",
    "apply_preregistered_adoption_matrix",
    "evaluate_g2",
    "evaluate_g3",
    "frozen_same_recall_budget_grid",
    "information_gain_per_physical_event",
    "percentile_interval",
    "same_area_recall_gain_percentage_points",
    "same_recall_union_area_relative_reduction",
    "stratified_five_horizon_bootstrap_indices",
    "strict_recall",
]
