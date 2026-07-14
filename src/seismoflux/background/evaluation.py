"""Frozen stage-2 information-gain, G1, and one-SE evaluation rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.randomness import SeedContext

ModelId = Literal["uniform_poisson", "spatial_poisson", "etas"]
MODEL_SIMPLICITY_ORDER: tuple[ModelId, ...] = (
    "uniform_poisson",
    "spatial_poisson",
    "etas",
)


def information_gain_per_event(
    candidate_log_likelihood: float,
    uniform_log_likelihood: float,
    physical_event_count: int,
) -> float:
    """Return candidate-minus-uniform information gain in nats per event."""

    candidate = float(candidate_log_likelihood)
    uniform = float(uniform_log_likelihood)
    if not math.isfinite(candidate) or not math.isfinite(uniform):
        raise ValueError("log likelihoods must be finite")
    if (
        not isinstance(physical_event_count, int)
        or isinstance(physical_event_count, bool)
        or physical_event_count <= 0
    ):
        raise ValueError("physical_event_count must be a positive integer")
    result = (candidate - uniform) / physical_event_count
    if not math.isfinite(result):
        raise ValueError("information gain must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FoldInformationGain:
    fold_id: str
    physical_event_count: int
    candidate_log_likelihood: float
    uniform_log_likelihood: float
    information_gain_per_event: float

    @classmethod
    def build(
        cls,
        *,
        fold_id: str,
        physical_event_count: int,
        candidate_log_likelihood: float,
        uniform_log_likelihood: float,
    ) -> FoldInformationGain:
        if not fold_id:
            raise ValueError("fold_id must not be empty")
        return cls(
            fold_id=fold_id,
            physical_event_count=physical_event_count,
            candidate_log_likelihood=float(candidate_log_likelihood),
            uniform_log_likelihood=float(uniform_log_likelihood),
            information_gain_per_event=information_gain_per_event(
                candidate_log_likelihood,
                uniform_log_likelihood,
                physical_event_count,
            ),
        )


@dataclass(frozen=True, slots=True)
class BackgroundModelEvidence:
    """Validation and historical-fold evidence for one frozen model snapshot."""

    model_id: ModelId
    validation_physical_event_count: int
    validation_log_likelihood: float
    validation_uniform_log_likelihood: float
    validation_information_gain_per_event: float
    development_folds: tuple[FoldInformationGain, ...]
    fit_succeeded: bool
    numerically_stable: bool
    grid_converged: bool

    @classmethod
    def build(
        cls,
        *,
        model_id: ModelId,
        validation_physical_event_count: int,
        validation_log_likelihood: float,
        validation_uniform_log_likelihood: float,
        development_folds: tuple[FoldInformationGain, ...],
        fit_succeeded: bool,
        numerically_stable: bool,
        grid_converged: bool,
    ) -> BackgroundModelEvidence:
        if model_id not in MODEL_SIMPLICITY_ORDER:
            raise ValueError("unknown background model_id")
        if tuple(item.fold_id for item in development_folds) != (
            "fold_1",
            "fold_2",
            "fold_3",
            "fold_4",
        ):
            raise ValueError("development evidence must contain the four frozen folds in order")
        return cls(
            model_id=model_id,
            validation_physical_event_count=validation_physical_event_count,
            validation_log_likelihood=float(validation_log_likelihood),
            validation_uniform_log_likelihood=float(validation_uniform_log_likelihood),
            validation_information_gain_per_event=information_gain_per_event(
                validation_log_likelihood,
                validation_uniform_log_likelihood,
                validation_physical_event_count,
            ),
            development_folds=development_folds,
            fit_succeeded=bool(fit_succeeded),
            numerically_stable=bool(numerically_stable),
            grid_converged=bool(grid_converged),
        )

    @property
    def eligible_for_selection(self) -> bool:
        return self.fit_succeeded and self.numerically_stable and self.grid_converged

    @property
    def positive_development_fold_count(self) -> int:
        return sum(item.information_gain_per_event > 0.0 for item in self.development_folds)

    @property
    def passes_g1_as_same_model(self) -> bool:
        return (
            self.model_id != "uniform_poisson"
            and self.eligible_for_selection
            and self.validation_information_gain_per_event > 0.0
            and self.positive_development_fold_count >= 3
        )


@dataclass(frozen=True, slots=True)
class G1Assessment:
    passed: bool
    passing_models: tuple[ModelId, ...]
    model_pass: tuple[tuple[ModelId, bool], ...]


def assess_g1(evidence: tuple[BackgroundModelEvidence, ...]) -> G1Assessment:
    by_id = {item.model_id: item for item in evidence}
    if set(by_id) != set(MODEL_SIMPLICITY_ORDER) or len(by_id) != len(evidence):
        raise ValueError("G1 evidence must contain each background model exactly once")
    nonuniform_ids: tuple[ModelId, ...] = ("spatial_poisson", "etas")
    model_pass = tuple(
        (model_id, by_id[model_id].passes_g1_as_same_model) for model_id in nonuniform_ids
    )
    passing = tuple(model_id for model_id, passed in model_pass if passed)
    return G1Assessment(passed=bool(passing), passing_models=passing, model_pass=model_pass)


@dataclass(frozen=True, slots=True)
class OneStandardErrorCandidate:
    model_id: ModelId
    validation_information_gain_per_event: float
    paired_standard_error: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class BackgroundModelSelection:
    validation_best_model_id: ModelId
    selected_model_id: ModelId
    candidates: tuple[OneStandardErrorCandidate, ...]
    excluded_failed_models: tuple[ModelId, ...]


def select_background_model(
    evidence: tuple[BackgroundModelEvidence, ...],
) -> BackgroundModelSelection:
    """Select the simplest successful model within the frozen paired one-SE threshold."""

    by_id = {item.model_id: item for item in evidence}
    if set(by_id) != set(MODEL_SIMPLICITY_ORDER) or len(by_id) != len(evidence):
        raise ValueError("selection evidence must contain each background model exactly once")
    eligible = tuple(item for item in evidence if item.eligible_for_selection)
    if not eligible:
        raise ValueError("no successful and numerically stable background model is selectable")
    best_value = max(item.validation_information_gain_per_event for item in eligible)
    best = next(
        by_id[model_id]
        for model_id in reversed(MODEL_SIMPLICITY_ORDER)
        if by_id[model_id].eligible_for_selection
        and math.isclose(
            by_id[model_id].validation_information_gain_per_event,
            best_value,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    )
    best_folds = np.asarray(
        [item.information_gain_per_event for item in best.development_folds],
        dtype=np.float64,
    )
    audits: list[OneStandardErrorCandidate] = []
    for model_id in MODEL_SIMPLICITY_ORDER:
        candidate = by_id[model_id]
        if not candidate.eligible_for_selection:
            continue
        candidate_folds = np.asarray(
            [item.information_gain_per_event for item in candidate.development_folds],
            dtype=np.float64,
        )
        differences = candidate_folds - best_folds
        paired_se = float(np.std(differences, ddof=1) / math.sqrt(4.0))
        audits.append(
            OneStandardErrorCandidate(
                model_id=model_id,
                validation_information_gain_per_event=(
                    candidate.validation_information_gain_per_event
                ),
                paired_standard_error=paired_se,
                eligible=(
                    candidate.validation_information_gain_per_event >= best_value - paired_se
                ),
            )
        )
    selected = next(
        model_id
        for model_id in MODEL_SIMPLICITY_ORDER
        if any(item.model_id == model_id and item.eligible for item in audits)
    )
    return BackgroundModelSelection(
        validation_best_model_id=best.model_id,
        selected_model_id=selected,
        candidates=tuple(audits),
        excluded_failed_models=tuple(
            model_id
            for model_id in MODEL_SIMPLICITY_ORDER
            if not by_id[model_id].eligible_for_selection
        ),
    )


@dataclass(frozen=True, slots=True)
class InformationGainContributions:
    """Per-physical-event log terms plus one fixed compensator difference."""

    physical_event_ids: tuple[str, ...]
    event_log_intensity_differences: NDArray[np.float64]
    compensator_difference: float

    def __post_init__(self) -> None:
        values = np.asarray(self.event_log_intensity_differences, dtype=np.float64)
        if values.ndim != 1 or values.shape != (len(self.physical_event_ids),):
            raise ValueError("event contributions must align one-to-one with physical event IDs")
        if not self.physical_event_ids or len(set(self.physical_event_ids)) != len(
            self.physical_event_ids
        ):
            raise ValueError("physical event IDs must be non-empty and unique")
        if not np.isfinite(values).all() or not math.isfinite(self.compensator_difference):
            raise ValueError("information-gain contributions must be finite")
        values.setflags(write=False)

    @property
    def information_gain_per_event(self) -> float:
        numerator = (
            float(np.sum(self.event_log_intensity_differences, dtype=np.float64))
            - self.compensator_difference
        )
        return numerator / len(self.physical_event_ids)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    replications: int
    confidence_level: float
    point_estimate: float
    lower: float
    upper: float
    replicate_values: tuple[float, ...]


def bootstrap_information_gain(
    contributions: InformationGainContributions,
    *,
    model_seed_id: Literal[
        "spatial_poisson_vs_uniform_poisson",
        "etas_vs_uniform_poisson",
    ],
    replications: int = 2000,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """Resample physical events while retaining the preregistered compensator."""

    if replications != 2000:
        raise ValueError("stage-2 bootstrap must use exactly 2000 replications")
    if confidence_level != 0.95:
        raise ValueError("stage-2 bootstrap confidence level must remain 0.95")
    values = contributions.event_log_intensity_differences
    count = len(values)
    replicates = np.empty(replications, dtype=np.float64)
    for replicate_index in range(replications):
        generator = SeedContext(
            root_seed=147,
            protocol_version="0.2.0",
            namespace="bootstrap",
            model_id=model_seed_id,
            issue_id="g1_primary_validation_2024-07-01_2025-07-01",
            replicate_index=replicate_index,
        ).generator()
        sampled = generator.integers(0, count, size=count)
        numerator = float(np.sum(values[sampled], dtype=np.float64)) - (
            contributions.compensator_difference
        )
        replicates[replicate_index] = numerator / count
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha], method="linear")
    return BootstrapInterval(
        replications=replications,
        confidence_level=confidence_level,
        point_estimate=contributions.information_gain_per_event,
        lower=float(lower),
        upper=float(upper),
        replicate_values=tuple(float(value) for value in replicates),
    )


__all__ = [
    "MODEL_SIMPLICITY_ORDER",
    "BackgroundModelEvidence",
    "BackgroundModelSelection",
    "BootstrapInterval",
    "FoldInformationGain",
    "G1Assessment",
    "InformationGainContributions",
    "ModelId",
    "OneStandardErrorCandidate",
    "assess_g1",
    "bootstrap_information_gain",
    "information_gain_per_event",
    "select_background_model",
]
