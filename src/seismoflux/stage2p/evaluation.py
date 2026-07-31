"""In-memory scientific evaluation for the Stage 2P P0/P1/PP comparison.

This module deliberately has no readers, writers, network calls, or dependency
on the stopped Stage 2S evaluator.  Inputs are explicit synthetic or already
frozen target observations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

ModelId: TypeAlias = Literal["P0", "P1", "PP"]
ComparisonId: TypeAlias = Literal["P1_minus_P0", "P1_minus_PP"]
GateStatus: TypeAlias = Literal["passed", "failed", "evidence_insufficient"]
EvaluationOutcomeStatus: TypeAlias = Literal["evaluated", "evidence_insufficient"]

MODEL_IDS: tuple[ModelId, ...] = ("P0", "P1", "PP")
HORIZONS_DAYS = (7, 30, 90)
COMPARISONS: tuple[tuple[ComparisonId, ModelId, ModelId], ...] = (
    ("P1_minus_P0", "P1", "P0"),
    ("P1_minus_PP", "P1", "PP"),
)
BOOTSTRAP_SEED = 147
BOOTSTRAP_REPLICATES = 2_000
BONFERRONI_LOWER_QUANTILE = 0.00625
BONFERRONI_UPPER_QUANTILE = 0.99375


class EvidenceInsufficientError(ValueError):
    """Raised when a frozen sample or bootstrap replicate has no denominator."""


class BootstrapReplicateDenominatorError(EvidenceInsufficientError):
    """A fixed bootstrap replicate cannot evaluate one frozen horizon."""

    def __init__(
        self,
        *,
        replicate_index: int,
        horizon_days: int,
        all_event_denominator: int,
        supported_event_denominator: int,
    ) -> None:
        self.replicate_index = replicate_index
        self.horizon_days = horizon_days
        self.all_event_denominator = all_event_denominator
        self.supported_event_denominator = supported_event_denominator
        super().__init__(
            "bootstrap replicate has a zero horizon denominator; "
            f"replicate_index={replicate_index}, horizon_days={horizon_days}, "
            f"all_event_denominator={all_event_denominator}, "
            f"supported_event_denominator={supported_event_denominator}; "
            "the frozen replicate is retained and no redraw is allowed"
        )


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """One target contribution to one non-overlapping forecast horizon.

    Events outside the shared support remain in strict-recall denominators but
    must have no density or alarm hit.  Supported densities are relative spatial
    densities, not absolute earthquake probabilities.
    """

    event_id: str
    horizon_days: int
    cluster_id: str
    region_id: str
    in_support: bool
    model_densities: Mapping[ModelId, float | None]
    alarm_hits: Mapping[ModelId, bool]

    def __post_init__(self) -> None:
        _identifier(self.event_id, label="event_id")
        _identifier(self.cluster_id, label="cluster_id")
        _identifier(self.region_id, label="region_id")
        if self.horizon_days not in HORIZONS_DAYS:
            raise ValueError("horizon_days must be one of 7, 30, and 90")
        if set(self.model_densities) != set(MODEL_IDS):
            raise ValueError("model_densities must contain exactly P0, P1, and PP")
        if set(self.alarm_hits) != set(MODEL_IDS):
            raise ValueError("alarm_hits must contain exactly P0, P1, and PP")

        densities: dict[ModelId, float | None] = {}
        hits: dict[ModelId, bool] = {}
        for model_id in MODEL_IDS:
            density = self.model_densities[model_id]
            hit = self.alarm_hits[model_id]
            if not isinstance(hit, bool):
                raise ValueError("alarm hits must be boolean")
            hits[model_id] = hit
            if self.in_support:
                if density is None:
                    raise ValueError("supported events require all three densities")
                value = float(density)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("supported densities must be finite and positive")
                densities[model_id] = value
            else:
                if density is not None or hit:
                    raise ValueError(
                        "events outside support must have null densities and be misses"
                    )
                densities[model_id] = None

        object.__setattr__(self, "model_densities", densities)
        object.__setattr__(self, "alarm_hits", hits)


def _density(observation: TargetObservation, model_id: ModelId) -> float:
    value = observation.model_densities[model_id]
    if value is None:
        raise AssertionError("supported observation unexpectedly has no density")
    return value


@dataclass(frozen=True, slots=True)
class ModelRecall:
    event_count: int
    event_hit_count: int
    strict_event_recall: float
    cluster_count: int
    cluster_hit_count: int
    independent_cluster_recall: float
    region_count: int
    region_hit_count: int
    independent_region_recall: float


@dataclass(frozen=True, slots=True)
class ComparisonPoint:
    recall_gain_percentage_points: float
    information_gain_nats_per_event: float
    supported_event_count: int


@dataclass(frozen=True, slots=True)
class HorizonEvaluation:
    horizon_days: int
    model_recall: Mapping[ModelId, ModelRecall]
    comparisons: Mapping[ComparisonId, ComparisonPoint]


@dataclass(frozen=True, slots=True)
class MacroModelRecall:
    strict_event_recall: float
    independent_cluster_recall: float
    independent_region_recall: float


@dataclass(frozen=True, slots=True)
class SimultaneousInterval:
    lower: float
    upper: float
    lower_quantile: float = BONFERRONI_LOWER_QUANTILE
    upper_quantile: float = BONFERRONI_UPPER_QUANTILE


@dataclass(frozen=True, slots=True)
class RemovalDiagnostic:
    endpoint: str
    group_kind: Literal["region", "cluster"]
    removed_id: str | None
    removed_contribution: float
    residual_with_original_denominator: float

    @property
    def remains_positive(self) -> bool:
        return self.residual_with_original_denominator > 0.0


@dataclass(frozen=True, slots=True)
class ComparisonEvaluation:
    comparison_id: ComparisonId
    per_horizon: Mapping[int, ComparisonPoint]
    macro_recall_gain_percentage_points: float
    macro_information_gain_nats_per_event: float
    recall_interval: SimultaneousInterval
    information_gain_interval: SimultaneousInterval
    removal_diagnostics: tuple[RemovalDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ScienceEvaluation:
    horizons: Mapping[int, HorizonEvaluation]
    macro_model_recall: Mapping[ModelId, MacroModelRecall]
    comparisons: Mapping[ComparisonId, ComparisonEvaluation]
    unique_event_count: int
    independent_cluster_count: int
    independent_region_count: int
    bootstrap_seed: int
    bootstrap_replicates: int
    bootstrap_endpoint_order: tuple[str, ...]
    bootstrap_samples: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class ScienceEvaluationOutcome:
    status: EvaluationOutcomeStatus
    evaluation: ScienceEvaluation | None
    reason_code: str | None
    detail: str | None
    failed_bootstrap_replicate_index: int | None
    failed_horizon_days: int | None
    bootstrap_redraw_performed: bool = False


@dataclass(frozen=True, slots=True)
class GateAssessment:
    status: GateStatus
    reasons: tuple[str, ...]


def _validate_observations(
    observations: Sequence[TargetObservation],
) -> tuple[TargetObservation, ...]:
    values = tuple(observations)
    if not values:
        raise EvidenceInsufficientError("at least one target observation is required")
    keys = tuple((item.horizon_days, item.event_id) for item in values)
    if len(set(keys)) != len(keys):
        raise ValueError("event_id must be unique within each horizon")
    present = {item.horizon_days for item in values}
    if present != set(HORIZONS_DAYS):
        raise EvidenceInsufficientError("all 7, 30, and 90 day horizons are required")
    return values


def _model_recall(observations: Sequence[TargetObservation], model_id: ModelId) -> ModelRecall:
    event_count = len(observations)
    if event_count == 0:
        raise EvidenceInsufficientError("strict recall needs a positive denominator")
    event_hit_count = sum(item.alarm_hits[model_id] for item in observations)
    clusters = sorted({item.cluster_id for item in observations})
    regions = sorted({item.region_id for item in observations})
    cluster_hit_count = sum(
        any(item.cluster_id == cluster_id and item.alarm_hits[model_id] for item in observations)
        for cluster_id in clusters
    )
    region_hit_count = sum(
        any(item.region_id == region_id and item.alarm_hits[model_id] for item in observations)
        for region_id in regions
    )
    return ModelRecall(
        event_count=event_count,
        event_hit_count=event_hit_count,
        strict_event_recall=event_hit_count / event_count,
        cluster_count=len(clusters),
        cluster_hit_count=cluster_hit_count,
        independent_cluster_recall=cluster_hit_count / len(clusters),
        region_count=len(regions),
        region_hit_count=region_hit_count,
        independent_region_recall=region_hit_count / len(regions),
    )


def _comparison_point(
    observations: Sequence[TargetObservation],
    candidate: ModelId,
    comparator: ModelId,
) -> ComparisonPoint:
    if not observations:
        raise EvidenceInsufficientError("comparison needs a positive recall denominator")
    supported = tuple(item for item in observations if item.in_support)
    if not supported:
        raise EvidenceInsufficientError(
            "information gain needs a positive supported-event denominator"
        )
    recall_gain = (
        100.0
        * math.fsum(
            float(item.alarm_hits[candidate]) - float(item.alarm_hits[comparator])
            for item in observations
        )
        / len(observations)
    )
    log_ratios = (
        math.log(_density(item, candidate)) - math.log(_density(item, comparator))
        for item in supported
    )
    return ComparisonPoint(
        recall_gain_percentage_points=recall_gain,
        information_gain_nats_per_event=math.fsum(log_ratios) / len(supported),
        supported_event_count=len(supported),
    )


def _macro_point_from_weights(
    observations: Sequence[TargetObservation],
    weights: Mapping[str, int],
    candidate: ModelId,
    comparator: ModelId,
    *,
    replicate_index: int,
) -> tuple[float, float]:
    recall_values: list[float] = []
    information_values: list[float] = []
    for horizon in HORIZONS_DAYS:
        rows = tuple(item for item in observations if item.horizon_days == horizon)
        all_denominator = sum(weights.get(item.cluster_id, 0) for item in rows)
        supported_denominator = sum(
            weights.get(item.cluster_id, 0) for item in rows if item.in_support
        )
        if all_denominator <= 0 or supported_denominator <= 0:
            raise BootstrapReplicateDenominatorError(
                replicate_index=replicate_index,
                horizon_days=horizon,
                all_event_denominator=all_denominator,
                supported_event_denominator=supported_denominator,
            )
        recall_numerator = math.fsum(
            weights.get(item.cluster_id, 0)
            * (float(item.alarm_hits[candidate]) - float(item.alarm_hits[comparator]))
            for item in rows
        )
        information_numerator = math.fsum(
            weights.get(item.cluster_id, 0)
            * (math.log(_density(item, candidate)) - math.log(_density(item, comparator)))
            for item in rows
            if item.in_support
        )
        recall_values.append(100.0 * recall_numerator / all_denominator)
        information_values.append(information_numerator / supported_denominator)
    return math.fsum(recall_values) / 3.0, math.fsum(information_values) / 3.0


def _contribution_by_group(
    observations: Sequence[TargetObservation],
    candidate: ModelId,
    comparator: ModelId,
    *,
    endpoint: Literal["recall", "information_gain"],
    group_kind: Literal["region", "cluster"],
) -> dict[str, float]:
    field = "region_id" if group_kind == "region" else "cluster_id"
    group_ids = sorted({getattr(item, field) for item in observations})
    result = {group_id: 0.0 for group_id in group_ids}
    for horizon in HORIZONS_DAYS:
        rows = tuple(item for item in observations if item.horizon_days == horizon)
        if endpoint == "recall":
            denominator = len(rows)
            for item in rows:
                result[getattr(item, field)] += (
                    100.0
                    * (float(item.alarm_hits[candidate]) - float(item.alarm_hits[comparator]))
                    / denominator
                    / 3.0
                )
        else:
            supported = tuple(item for item in rows if item.in_support)
            if not supported:
                raise EvidenceInsufficientError("information gain needs support in every horizon")
            denominator = len(supported)
            for item in supported:
                result[getattr(item, field)] += (
                    (math.log(_density(item, candidate)) - math.log(_density(item, comparator)))
                    / denominator
                    / 3.0
                )
    return result


def _removal_diagnostic(
    *,
    endpoint: str,
    group_kind: Literal["region", "cluster"],
    global_value: float,
    contributions: Mapping[str, float],
) -> RemovalDiagnostic:
    positives = tuple(
        (group_id, contribution)
        for group_id, contribution in contributions.items()
        if contribution > 0.0
    )
    if not positives:
        return RemovalDiagnostic(endpoint, group_kind, None, 0.0, global_value)
    removed_id, removed = sorted(positives, key=lambda item: (-item[1], item[0]))[0]
    return RemovalDiagnostic(
        endpoint=endpoint,
        group_kind=group_kind,
        removed_id=removed_id,
        removed_contribution=removed,
        residual_with_original_denominator=global_value - removed,
    )


def evaluate_science_targets(
    observations: Sequence[TargetObservation],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> ScienceEvaluation:
    """Evaluate the frozen P0/P1/PP endpoints from explicit target rows."""

    values = _validate_observations(observations)
    if (
        not isinstance(bootstrap_replicates, int)
        or isinstance(bootstrap_replicates, bool)
        or bootstrap_replicates <= 0
    ):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError("bootstrap_seed must be an integer")

    horizons: dict[int, HorizonEvaluation] = {}
    for horizon in HORIZONS_DAYS:
        rows = tuple(item for item in values if item.horizon_days == horizon)
        horizons[horizon] = HorizonEvaluation(
            horizon_days=horizon,
            model_recall={model_id: _model_recall(rows, model_id) for model_id in MODEL_IDS},
            comparisons={
                comparison_id: _comparison_point(rows, candidate, comparator)
                for comparison_id, candidate, comparator in COMPARISONS
            },
        )

    macro_model_recall = {
        model_id: MacroModelRecall(
            strict_event_recall=math.fsum(
                horizons[horizon].model_recall[model_id].strict_event_recall
                for horizon in HORIZONS_DAYS
            )
            / 3.0,
            independent_cluster_recall=math.fsum(
                horizons[horizon].model_recall[model_id].independent_cluster_recall
                for horizon in HORIZONS_DAYS
            )
            / 3.0,
            independent_region_recall=math.fsum(
                horizons[horizon].model_recall[model_id].independent_region_recall
                for horizon in HORIZONS_DAYS
            )
            / 3.0,
        )
        for model_id in MODEL_IDS
    }

    clusters = tuple(sorted({item.cluster_id for item in values}))
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    sample_array = np.empty((bootstrap_replicates, 4), dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        draws = rng.integers(0, len(clusters), size=len(clusters))
        counts = np.bincount(draws, minlength=len(clusters))
        weights = {cluster_id: int(counts[index]) for index, cluster_id in enumerate(clusters)}
        column = 0
        for _, candidate, comparator in COMPARISONS:
            recall, information = _macro_point_from_weights(
                values,
                weights,
                candidate,
                comparator,
                replicate_index=replicate,
            )
            sample_array[replicate, column] = information
            sample_array[replicate, column + 1] = recall
            column += 2

    quantiles = np.quantile(
        sample_array,
        [BONFERRONI_LOWER_QUANTILE, BONFERRONI_UPPER_QUANTILE],
        axis=0,
        method="linear",
    )
    comparisons: dict[ComparisonId, ComparisonEvaluation] = {}
    for comparison_index, (
        comparison_id,
        candidate,
        comparator,
    ) in enumerate(COMPARISONS):
        per_horizon = {
            horizon: horizons[horizon].comparisons[comparison_id] for horizon in HORIZONS_DAYS
        }
        macro_recall = (
            math.fsum(item.recall_gain_percentage_points for item in per_horizon.values()) / 3.0
        )
        macro_information = (
            math.fsum(item.information_gain_nats_per_event for item in per_horizon.values()) / 3.0
        )
        information_column = comparison_index * 2
        recall_column = information_column + 1
        diagnostics: list[RemovalDiagnostic] = []
        endpoint_values: tuple[tuple[Literal["recall", "information_gain"], float], ...] = (
            ("information_gain", macro_information),
            ("recall", macro_recall),
        )
        for endpoint, global_value in endpoint_values:
            for group_kind in ("region", "cluster"):
                contributions = _contribution_by_group(
                    values,
                    candidate,
                    comparator,
                    endpoint=endpoint,
                    group_kind=group_kind,
                )
                diagnostics.append(
                    _removal_diagnostic(
                        endpoint=f"{comparison_id}_{endpoint}",
                        group_kind=group_kind,
                        global_value=global_value,
                        contributions=contributions,
                    )
                )
        comparisons[comparison_id] = ComparisonEvaluation(
            comparison_id=comparison_id,
            per_horizon=per_horizon,
            macro_recall_gain_percentage_points=macro_recall,
            macro_information_gain_nats_per_event=macro_information,
            recall_interval=SimultaneousInterval(
                lower=float(quantiles[0, recall_column]),
                upper=float(quantiles[1, recall_column]),
            ),
            information_gain_interval=SimultaneousInterval(
                lower=float(quantiles[0, information_column]),
                upper=float(quantiles[1, information_column]),
            ),
            removal_diagnostics=tuple(diagnostics),
        )

    endpoint_order = (
        "P1_minus_P0_information_gain",
        "P1_minus_P0_strict_recall",
        "P1_minus_PP_information_gain",
        "P1_minus_PP_strict_recall",
    )
    return ScienceEvaluation(
        horizons=horizons,
        macro_model_recall=macro_model_recall,
        comparisons=comparisons,
        unique_event_count=len({item.event_id for item in values}),
        independent_cluster_count=len(clusters),
        independent_region_count=len({item.region_id for item in values}),
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_endpoint_order=endpoint_order,
        bootstrap_samples=tuple(
            tuple(float(value) for value in row)  # type: ignore[misc]
            for row in sample_array
        ),
    )


def evaluate_science_targets_outcome(
    observations: Sequence[TargetObservation],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> ScienceEvaluationOutcome:
    """Return structured insufficiency without changing or redrawing bootstrap rows."""

    try:
        evaluation = evaluate_science_targets(
            observations,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
    except BootstrapReplicateDenominatorError as error:
        return ScienceEvaluationOutcome(
            status="evidence_insufficient",
            evaluation=None,
            reason_code="bootstrap_replicate_zero_horizon_denominator",
            detail=str(error),
            failed_bootstrap_replicate_index=error.replicate_index,
            failed_horizon_days=error.horizon_days,
        )
    except EvidenceInsufficientError as error:
        return ScienceEvaluationOutcome(
            status="evidence_insufficient",
            evaluation=None,
            reason_code="input_or_fixed_denominator_evidence_insufficient",
            detail=str(error),
            failed_bootstrap_replicate_index=None,
            failed_horizon_days=None,
        )
    return ScienceEvaluationOutcome(
        status="evaluated",
        evaluation=evaluation,
        reason_code=None,
        detail=None,
        failed_bootstrap_replicate_index=None,
        failed_horizon_days=None,
    )


def assess_confirmatory_gate(
    evaluation: ScienceEvaluation,
    *,
    minimum_unique_events: int = 20,
    minimum_clusters: int = 10,
) -> GateAssessment:
    """Apply the preregistered scientific gate without scenario-specific labels."""

    if evaluation.unique_event_count < minimum_unique_events:
        return GateAssessment("evidence_insufficient", ("fewer_than_20_unique_events",))
    if evaluation.independent_cluster_count < minimum_clusters:
        return GateAssessment("evidence_insufficient", ("fewer_than_10_clusters",))

    p0 = evaluation.comparisons["P1_minus_P0"]
    pp = evaluation.comparisons["P1_minus_PP"]
    checks = (
        (
            p0.macro_recall_gain_percentage_points >= 5.0,
            "P1_minus_P0_recall_gain_below_5pp",
        ),
        (p0.recall_interval.lower > 0.0, "P1_minus_P0_recall_interval_not_positive"),
        (pp.recall_interval.lower > 0.0, "P1_minus_PP_recall_interval_not_positive"),
        (
            p0.information_gain_interval.lower > 0.0,
            "P1_minus_P0_information_interval_not_positive",
        ),
        (
            pp.information_gain_interval.lower > 0.0,
            "P1_minus_PP_information_interval_not_positive",
        ),
        (
            all(
                item.remains_positive
                for comparison in evaluation.comparisons.values()
                for item in comparison.removal_diagnostics
            ),
            "largest_region_or_cluster_removal_not_positive",
        ),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    return GateAssessment("passed" if not reasons else "failed", reasons)


def evaluation_to_dict(evaluation: ScienceEvaluation) -> dict[str, Any]:
    """Return a JSON-ready structure for synthetic figures and reports."""

    return asdict(evaluation)


__all__ = [
    "BONFERRONI_LOWER_QUANTILE",
    "BONFERRONI_UPPER_QUANTILE",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "COMPARISONS",
    "HORIZONS_DAYS",
    "MODEL_IDS",
    "BootstrapReplicateDenominatorError",
    "ComparisonId",
    "EvaluationOutcomeStatus",
    "EvidenceInsufficientError",
    "GateAssessment",
    "ModelId",
    "ScienceEvaluation",
    "ScienceEvaluationOutcome",
    "TargetObservation",
    "assess_confirmatory_gate",
    "evaluate_science_targets",
    "evaluate_science_targets_outcome",
    "evaluation_to_dict",
]
