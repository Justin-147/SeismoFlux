"""Audited point-process score evidence for the frozen stage-2 G1 gate."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.artifacts import canonical_json_bytes

ModelId = Literal["uniform_poisson", "spatial_poisson", "etas"]
MODEL_SIMPLICITY_ORDER: tuple[ModelId, ...] = (
    "uniform_poisson",
    "spatial_poisson",
    "etas",
)
EXPECTED_SNAPSHOTS = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _readonly(values: object) -> NDArray[np.float64]:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _nonempty(name: str, value: str) -> str:
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class PointProcessScoreEvidence:
    """Complete event-term and compensator evidence for one fixed snapshot."""

    protocol_sha256: str
    model_id: ModelId
    model_variant_id: str
    parameter_snapshot_id: str
    snapshot_id: str
    fit_end_utc: str
    assessment_start_utc: str
    assessment_end_utc: str
    selected_mc: float
    target_event_ids: tuple[str, ...]
    event_log_intensities: NDArray[np.float64]
    compensator: float
    numerical_gate_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.protocol_sha256) is None:
            raise ValueError("protocol_sha256 must be a lowercase SHA-256 string")
        if self.model_id not in MODEL_SIMPLICITY_ORDER:
            raise ValueError("unknown background model_id")
        _nonempty("model_variant_id", self.model_variant_id)
        _nonempty("parameter_snapshot_id", self.parameter_snapshot_id)
        if self.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("score snapshot is not one of the five frozen snapshots")
        _nonempty("fit_end_utc", self.fit_end_utc)
        _nonempty("assessment_start_utc", self.assessment_start_utc)
        _nonempty("assessment_end_utc", self.assessment_end_utc)
        if not math.isfinite(self.selected_mc) or self.selected_mc <= 0.0:
            raise ValueError("selected_mc must be finite and positive")
        identifiers = tuple(self.target_event_ids)
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("target event IDs must be non-empty and unique")
        log_intensities = _readonly(self.event_log_intensities)
        if log_intensities.ndim != 1 or log_intensities.shape != (len(identifiers),):
            raise ValueError("event log intensities must align one-to-one with target event IDs")
        if not np.isfinite(log_intensities).all():
            raise ValueError("event log intensities must be finite")
        if not math.isfinite(self.compensator) or self.compensator < 0.0:
            raise ValueError("point-process compensator must be finite and non-negative")
        gate_ids = tuple(self.numerical_gate_evidence_ids)
        if (
            not gate_ids
            or any(not value for value in gate_ids)
            or len(set(gate_ids)) != len(gate_ids)
        ):
            raise ValueError("numerical gate evidence IDs must be non-empty and unique")
        object.__setattr__(self, "target_event_ids", identifiers)
        object.__setattr__(self, "event_log_intensities", log_intensities)
        object.__setattr__(self, "numerical_gate_evidence_ids", gate_ids)

    @property
    def event_log_intensity_sum(self) -> float:
        return float(np.sum(self.event_log_intensities, dtype=np.float64))

    @property
    def log_likelihood(self) -> float:
        return self.event_log_intensity_sum - self.compensator

    @property
    def score_id(self) -> str:
        payload = {
            "protocol_sha256": self.protocol_sha256,
            "model_id": self.model_id,
            "model_variant_id": self.model_variant_id,
            "parameter_snapshot_id": self.parameter_snapshot_id,
            "snapshot_id": self.snapshot_id,
            "fit_end_utc": self.fit_end_utc,
            "assessment_start_utc": self.assessment_start_utc,
            "assessment_end_utc": self.assessment_end_utc,
            "selected_mc": self.selected_mc,
            "target_event_ids": self.target_event_ids,
            "event_log_intensities": tuple(float(value) for value in self.event_log_intensities),
            "compensator": self.compensator,
            "numerical_gate_evidence_ids": self.numerical_gate_evidence_ids,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class PairedInformationGainEvidence:
    """Candidate-minus-uniform evidence on an identical physical event set."""

    candidate: PointProcessScoreEvidence
    uniform: PointProcessScoreEvidence
    event_log_intensity_differences: NDArray[np.float64]
    compensator_difference: float
    information_gain_per_event: float | None

    @classmethod
    def build(
        cls,
        *,
        candidate: PointProcessScoreEvidence,
        uniform: PointProcessScoreEvidence,
    ) -> PairedInformationGainEvidence:
        if uniform.model_id != "uniform_poisson":
            raise ValueError("paired baseline score must be uniform_poisson")
        exact_fields = (
            "protocol_sha256",
            "snapshot_id",
            "fit_end_utc",
            "assessment_start_utc",
            "assessment_end_utc",
            "selected_mc",
            "target_event_ids",
        )
        for field_name in exact_fields:
            if getattr(candidate, field_name) != getattr(uniform, field_name):
                raise ValueError(f"candidate and uniform score differ on paired field {field_name}")
        differences = _readonly(candidate.event_log_intensities - uniform.event_log_intensities)
        compensator_difference = candidate.compensator - uniform.compensator
        information_gain = (
            (float(np.sum(differences, dtype=np.float64)) - compensator_difference)
            / len(candidate.target_event_ids)
            if candidate.target_event_ids
            else None
        )
        return cls(
            candidate=candidate,
            uniform=uniform,
            event_log_intensity_differences=differences,
            compensator_difference=compensator_difference,
            information_gain_per_event=information_gain,
        )

    def __post_init__(self) -> None:
        values = _readonly(self.event_log_intensity_differences)
        if values.shape != (len(self.candidate.target_event_ids),):
            raise ValueError("paired event contributions have the wrong shape")
        if not np.isfinite(values).all() or not math.isfinite(self.compensator_difference):
            raise ValueError("paired information-gain evidence must be finite")
        expected = (
            (float(np.sum(values, dtype=np.float64)) - self.compensator_difference) / len(values)
            if len(values)
            else None
        )
        if expected != self.information_gain_per_event:
            raise ValueError("paired information gain differs from its event evidence")
        object.__setattr__(self, "event_log_intensity_differences", values)


@dataclass(frozen=True, slots=True)
class AuditedBackgroundModelEvidence:
    """All five attempts for one exact model variant, including explicit failures."""

    model_id: ModelId
    model_variant_id: str
    protocol_sha256: str
    development_folds: tuple[PairedInformationGainEvidence, ...]
    validation: PairedInformationGainEvidence | None
    failed_snapshot_reasons: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_SIMPLICITY_ORDER:
            raise ValueError("unknown background model_id")
        _nonempty("model_variant_id", self.model_variant_id)
        if _SHA256_PATTERN.fullmatch(self.protocol_sha256) is None:
            raise ValueError("protocol_sha256 must be a lowercase SHA-256 string")
        scored = (*self.development_folds, *((self.validation,) if self.validation else ()))
        for item in scored:
            if item.candidate.model_id != self.model_id:
                raise ValueError("model evidence contains a score from another model family")
            if item.candidate.model_variant_id != self.model_variant_id:
                raise ValueError("model evidence mixes different model variants")
            if item.candidate.protocol_sha256 != self.protocol_sha256:
                raise ValueError("model evidence mixes different protocol fingerprints")
        fold_ids = tuple(item.candidate.snapshot_id for item in self.development_folds)
        expected_fold_order = tuple(
            snapshot for snapshot in EXPECTED_SNAPSHOTS[:4] if snapshot in set(fold_ids)
        )
        if fold_ids != expected_fold_order or len(set(fold_ids)) != len(fold_ids):
            raise ValueError("development scores must be an ordered subset of frozen folds")
        if (
            self.validation is not None
            and self.validation.candidate.snapshot_id != "final_validation"
        ):
            raise ValueError("validation evidence must use the final_validation snapshot")
        failures = tuple(self.failed_snapshot_reasons)
        failed_ids = tuple(item[0] for item in failures)
        if len(set(failed_ids)) != len(failed_ids):
            raise ValueError("failed snapshot IDs must be unique")
        if any(
            snapshot not in EXPECTED_SNAPSHOTS or not reasons or any(not value for value in reasons)
            for snapshot, reasons in failures
        ):
            raise ValueError("failed snapshots require frozen IDs and non-empty reasons")
        scored_ids = {*fold_ids}
        if self.validation is not None:
            scored_ids.add("final_validation")
        if scored_ids.intersection(failed_ids):
            raise ValueError("a snapshot cannot be both scored and failed")
        if scored_ids.union(failed_ids) != set(EXPECTED_SNAPSHOTS):
            raise ValueError("model evidence must account for all five frozen snapshots")
        object.__setattr__(self, "failed_snapshot_reasons", failures)

    @property
    def eligible_for_selection(self) -> bool:
        return (
            not self.failed_snapshot_reasons
            and len(self.development_folds) == 4
            and self.validation is not None
            and all(item.information_gain_per_event is not None for item in self.development_folds)
            and self.validation.information_gain_per_event is not None
        )

    @property
    def positive_development_fold_count(self) -> int:
        return sum(
            item.information_gain_per_event is not None and item.information_gain_per_event > 0.0
            for item in self.development_folds
        )

    @property
    def passes_g1_as_same_model(self) -> bool:
        return (
            self.model_id != "uniform_poisson"
            and self.eligible_for_selection
            and cast(PairedInformationGainEvidence, self.validation).information_gain_per_event
            is not None
            and cast(
                float,
                cast(PairedInformationGainEvidence, self.validation).information_gain_per_event,
            )
            > 0.0
            and self.positive_development_fold_count >= 3
        )


@dataclass(frozen=True, slots=True)
class AuditedG1Assessment:
    passed: bool
    passing_model_variants: tuple[str, ...]
    model_pass: tuple[tuple[ModelId, str, bool], ...]


def assess_audited_g1(
    evidence: tuple[AuditedBackgroundModelEvidence, ...],
) -> AuditedG1Assessment:
    by_id = {item.model_id: item for item in evidence}
    if set(by_id) != set(MODEL_SIMPLICITY_ORDER) or len(by_id) != len(evidence):
        raise ValueError("audited G1 evidence must contain each background model exactly once")
    if len({item.protocol_sha256 for item in evidence}) != 1:
        raise ValueError("audited G1 evidence must use one frozen protocol fingerprint")
    nonuniform: tuple[ModelId, ModelId] = ("spatial_poisson", "etas")
    model_pass = tuple(
        (
            model_id,
            by_id[model_id].model_variant_id,
            by_id[model_id].passes_g1_as_same_model,
        )
        for model_id in nonuniform
    )
    passing = tuple(variant for _, variant, passed in model_pass if passed)
    return AuditedG1Assessment(
        passed=bool(passing),
        passing_model_variants=passing,
        model_pass=model_pass,
    )


@dataclass(frozen=True, slots=True)
class AuditedModelSelection:
    validation_best_model_variant_id: str
    selected_model_variant_id: str
    paired_standard_errors: tuple[tuple[str, float], ...]
    excluded_model_variants: tuple[str, ...]


def select_audited_background_model(
    evidence: tuple[AuditedBackgroundModelEvidence, ...],
) -> AuditedModelSelection:
    """Apply the frozen validation-best and paired one-SE rule to audited evidence."""

    by_id = {item.model_id: item for item in evidence}
    if set(by_id) != set(MODEL_SIMPLICITY_ORDER) or len(by_id) != len(evidence):
        raise ValueError("selection evidence must contain each background model exactly once")
    eligible = tuple(item for item in evidence if item.eligible_for_selection)
    if not eligible:
        raise ValueError("no complete audited background model is selectable")
    validation_values = {
        item.model_id: cast(
            float, cast(PairedInformationGainEvidence, item.validation).information_gain_per_event
        )
        for item in eligible
    }
    best_value = max(validation_values.values())
    best = next(
        by_id[model_id]
        for model_id in reversed(MODEL_SIMPLICITY_ORDER)
        if model_id in validation_values
        and math.isclose(validation_values[model_id], best_value, rel_tol=0.0, abs_tol=1.0e-15)
    )
    best_folds = np.asarray(
        [cast(float, item.information_gain_per_event) for item in best.development_folds],
        dtype=np.float64,
    )
    standard_errors: list[tuple[str, float]] = []
    one_se_eligible: set[str] = set()
    for model_id in MODEL_SIMPLICITY_ORDER:
        item = by_id[model_id]
        if not item.eligible_for_selection:
            continue
        fold_values = np.asarray(
            [cast(float, fold.information_gain_per_event) for fold in item.development_folds],
            dtype=np.float64,
        )
        paired_se = float(np.std(fold_values - best_folds, ddof=1) / math.sqrt(4.0))
        standard_errors.append((item.model_variant_id, paired_se))
        if validation_values[model_id] >= best_value - paired_se:
            one_se_eligible.add(item.model_variant_id)
    selected = next(
        by_id[model_id]
        for model_id in MODEL_SIMPLICITY_ORDER
        if by_id[model_id].model_variant_id in one_se_eligible
    )
    return AuditedModelSelection(
        validation_best_model_variant_id=best.model_variant_id,
        selected_model_variant_id=selected.model_variant_id,
        paired_standard_errors=tuple(standard_errors),
        excluded_model_variants=tuple(
            item.model_variant_id for item in evidence if not item.eligible_for_selection
        ),
    )


__all__ = [
    "MODEL_SIMPLICITY_ORDER",
    "AuditedBackgroundModelEvidence",
    "AuditedG1Assessment",
    "AuditedModelSelection",
    "ModelId",
    "PairedInformationGainEvidence",
    "PointProcessScoreEvidence",
    "assess_audited_g1",
    "select_audited_background_model",
]
