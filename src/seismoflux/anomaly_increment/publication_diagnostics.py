"""Aggregate, coordinate-free publication diagnostics for stage 4.

Only already-computed numerical diagnostics enter this module.  The contracts
contain no paths, target coordinates, cell mappings, or data-loading behavior.
Missing panel inputs are rejected before any static or interactive publication
can be rendered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, TypeAlias

from seismoflux.anomaly_increment.contracts import canonical_mapping_sha256
from seismoflux.anomaly_increment.preregistration import (
    PRIMARY_MACRO_HORIZONS_DAYS,
    STAGE4_HORIZONS_DAYS,
)

MagnitudeBin: TypeAlias = Literal["M5_6", "M6_plus"]
ModelVariant: TypeAlias = Literal[
    "background_no_increment",
    "coverage_only",
    "snapshot",
    "dynamic",
]
IncrementVariant: TypeAlias = Literal["coverage_only", "snapshot", "dynamic"]
PermutationVariant: TypeAlias = Literal["snapshot", "dynamic"]
PermutationKind: TypeAlias = Literal["time", "space"]
EvidenceStatus: TypeAlias = Literal[
    "evaluated",
    "evidence_insufficient_zero_events",
    "evidence_insufficient_no_random_split",
    "exploratory_low_sample",
    "exploratory_low_sample_no_random_split",
]
FoldEvidenceStatus: TypeAlias = Literal[
    "evaluated",
    "evidence_insufficient_zero_events",
]
PermutationEvidenceStatus: TypeAlias = Literal[
    "evaluated",
    "evidence_insufficient_zero_events",
    "evidence_insufficient_no_placebo_injection",
    "evidence_insufficient_scientific_failure_fraction",
]
InformationGainEvidenceStatus: TypeAlias = EvidenceStatus
RegionalInformationGainEvidenceStatus: TypeAlias = Literal[
    "evaluated",
    "evidence_insufficient_zero_supported_events",
]
RegionalRecallEvidenceStatus: TypeAlias = Literal[
    "evaluated",
    "evidence_insufficient_zero_all_events",
]
SameRecallEvidenceStatus: TypeAlias = Literal[
    "evaluated",
    "evidence_insufficient_zero_comparator_recall",
    "evidence_insufficient_target_recall_not_reached",
]
AlarmRecallEvidenceStatus: TypeAlias = Literal[
    "evaluated",
    "evidence_insufficient_zero_all_events",
]

NULL_REPLICATION_COUNT = 1_000
PUBLICATION_ALARM_BUDGETS_KM2: tuple[float, ...] = (
    300_000.0,
    450_000.0,
    600_000.0,
    750_000.0,
    960_000.0,
)
_INCREMENT_VARIANTS: tuple[IncrementVariant, ...] = (
    "coverage_only",
    "snapshot",
    "dynamic",
)
_MODEL_VARIANTS: tuple[ModelVariant, ...] = (
    "background_no_increment",
    *_INCREMENT_VARIANTS,
)


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _integer(value: int, *, label: str, positive: bool) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _finite(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _finite_tuple(
    values: tuple[float, ...],
    *,
    label: str,
    minimum_size: int,
) -> tuple[float, ...]:
    output = tuple(_finite(value, label=label) for value in values)
    if len(output) < minimum_size:
        raise ValueError(f"{label} must contain at least {minimum_size} values")
    return output


def _permutation_null_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    output: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("null_statistics must be numeric")
        resolved = float(value)
        if math.isnan(resolved) or resolved == -math.inf:
            raise ValueError("null_statistics may contain only finite values or +inf failures")
        output.append(resolved)
    if len(output) != NULL_REPLICATION_COUNT:
        raise ValueError("each permutation null distribution must contain 1000 values")
    return tuple(output)


def _json_safe_permutation_value(value: float) -> float | str:
    return "positive_infinity_scientific_failure" if value == math.inf else value


def _strictly_increasing(values: tuple[float, ...], *, label: str) -> None:
    if any(right <= left for left, right in pairwise(values)):
        raise ValueError(f"{label} must be strictly increasing")


def _non_decreasing(values: tuple[float, ...], *, label: str) -> None:
    if any(right < left for left, right in pairwise(values)):
        raise ValueError(f"{label} must be non-decreasing")


@dataclass(frozen=True, slots=True)
class DataMethodFlowDiagnostics:
    training_issue_count: int
    training_cell_count: int
    fitted_feature_count: int
    independent_event_count: int
    study_area_km2: float
    model_variant_count: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_issue_count",
            _integer(self.training_issue_count, label="training_issue_count", positive=True),
        )
        object.__setattr__(
            self,
            "training_cell_count",
            _integer(self.training_cell_count, label="training_cell_count", positive=True),
        )
        object.__setattr__(
            self,
            "fitted_feature_count",
            _integer(self.fitted_feature_count, label="fitted_feature_count", positive=True),
        )
        object.__setattr__(
            self,
            "independent_event_count",
            _integer(
                self.independent_event_count,
                label="independent_event_count",
                positive=False,
            ),
        )
        if self.model_variant_count != len(_MODEL_VARIANTS):
            raise ValueError("model_variant_count must retain all four frozen variants")
        study_area = _finite(self.study_area_km2, label="study_area_km2")
        if study_area <= 0.0:
            raise ValueError("study_area_km2 must be positive")
        object.__setattr__(self, "study_area_km2", study_area)

    def as_mapping(self) -> dict[str, object]:
        return {
            "fitted_feature_count": self.fitted_feature_count,
            "independent_event_count": self.independent_event_count,
            "model_variant_count": self.model_variant_count,
            "study_area_km2": self.study_area_km2,
            "training_cell_count": self.training_cell_count,
            "training_issue_count": self.training_issue_count,
        }


@dataclass(frozen=True, slots=True)
class CoefficientEffectCurve:
    variant: IncrementVariant
    coefficient_name: str
    coefficient_estimate: float
    input_values: tuple[float, ...]
    effect_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.variant not in _INCREMENT_VARIANTS:
            raise ValueError("coefficient effects require a fitted increment variant")
        name = _identifier(self.coefficient_name, label="coefficient_name")
        estimate = _finite(self.coefficient_estimate, label="coefficient_estimate")
        inputs = _finite_tuple(
            self.input_values,
            label="coefficient effect input_values",
            minimum_size=3,
        )
        effects = _finite_tuple(
            self.effect_values,
            label="coefficient effect effect_values",
            minimum_size=3,
        )
        if len(inputs) != len(effects):
            raise ValueError("coefficient effect inputs and effects must align")
        _strictly_increasing(inputs, label="coefficient effect input_values")
        object.__setattr__(self, "coefficient_name", name)
        object.__setattr__(self, "coefficient_estimate", estimate)
        object.__setattr__(self, "input_values", inputs)
        object.__setattr__(self, "effect_values", effects)

    def as_mapping(self) -> dict[str, object]:
        return {
            "coefficient_estimate": self.coefficient_estimate,
            "coefficient_name": self.coefficient_name,
            "effect_values": list(self.effect_values),
            "input_values": list(self.input_values),
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class DistanceLeadDecayDiagnostics:
    distance_km: tuple[float, ...]
    spatial_relative_weight: tuple[float, ...]
    lead_days: tuple[float, ...]
    temporal_relative_weight: tuple[float, ...]

    def __post_init__(self) -> None:
        distance = _finite_tuple(self.distance_km, label="distance_km", minimum_size=3)
        spatial = _finite_tuple(
            self.spatial_relative_weight,
            label="spatial_relative_weight",
            minimum_size=3,
        )
        leads = _finite_tuple(self.lead_days, label="lead_days", minimum_size=3)
        temporal = _finite_tuple(
            self.temporal_relative_weight,
            label="temporal_relative_weight",
            minimum_size=3,
        )
        if len(distance) != len(spatial) or len(leads) != len(temporal):
            raise ValueError("distance/lead nodes must align with their relative weights")
        _strictly_increasing(distance, label="distance_km")
        _strictly_increasing(leads, label="lead_days")
        if distance[0] < 0.0 or leads[0] < 0.0:
            raise ValueError("distance and lead nodes must be non-negative")
        if any(value < 0.0 or value > 1.0 for value in (*spatial, *temporal)):
            raise ValueError("relative decay weights must lie in [0, 1]")
        object.__setattr__(self, "distance_km", distance)
        object.__setattr__(self, "spatial_relative_weight", spatial)
        object.__setattr__(self, "lead_days", leads)
        object.__setattr__(self, "temporal_relative_weight", temporal)

    def as_mapping(self) -> dict[str, object]:
        return {
            "distance_km": list(self.distance_km),
            "lead_days": list(self.lead_days),
            "spatial_relative_weight": list(self.spatial_relative_weight),
            "temporal_relative_weight": list(self.temporal_relative_weight),
        }


@dataclass(frozen=True, slots=True)
class FoldMacroValue:
    fold_index: Literal[1, 2, 3]
    primary_horizon_event_counts: tuple[int, int, int]
    evidence_status: FoldEvidenceStatus
    dynamic_macro_information_gain: float | None
    snapshot_macro_information_gain: float | None

    def __post_init__(self) -> None:
        if self.fold_index not in {1, 2, 3}:
            raise ValueError("fold_index must be one of the three development folds")
        counts = tuple(
            _integer(value, label="fold primary-horizon event count", positive=False)
            for value in self.primary_horizon_event_counts
        )
        if len(counts) != len(PRIMARY_MACRO_HORIZONS_DAYS):
            raise ValueError("fold diagnostics require one count per primary horizon")
        dynamic: float | None
        snapshot: float | None
        if self.evidence_status == "evaluated":
            if (
                any(value == 0 for value in counts)
                or self.dynamic_macro_information_gain is None
                or self.snapshot_macro_information_gain is None
            ):
                raise ValueError("evaluated fold macro requires events in every primary horizon")
            dynamic = _finite(
                self.dynamic_macro_information_gain,
                label="dynamic_macro_information_gain",
            )
            snapshot = _finite(
                self.snapshot_macro_information_gain,
                label="snapshot_macro_information_gain",
            )
        elif self.evidence_status == "evidence_insufficient_zero_events":
            if (
                all(value > 0 for value in counts)
                or self.dynamic_macro_information_gain is not None
                or self.snapshot_macro_information_gain is not None
            ):
                raise ValueError("zero-event fold evidence must not contain a partial macro")
            dynamic = snapshot = None
        else:
            raise ValueError("fold evidence status is outside the frozen protocol")
        object.__setattr__(self, "primary_horizon_event_counts", counts)
        object.__setattr__(self, "dynamic_macro_information_gain", dynamic)
        object.__setattr__(self, "snapshot_macro_information_gain", snapshot)

    def as_mapping(self) -> dict[str, object]:
        return {
            "dynamic_macro_information_gain": self.dynamic_macro_information_gain,
            "evidence_status": self.evidence_status,
            "fold_index": self.fold_index,
            "primary_horizon_event_counts": list(self.primary_horizon_event_counts),
            "snapshot_macro_information_gain": self.snapshot_macro_information_gain,
        }


@dataclass(frozen=True, slots=True)
class PermutationDistribution:
    variant: PermutationVariant
    kind: PermutationKind
    observed_statistic: float | None
    null_statistics: tuple[float, ...]
    evidence_status: PermutationEvidenceStatus = "evaluated"

    def __post_init__(self) -> None:
        if self.variant not in {"snapshot", "dynamic"}:
            raise ValueError("permutation variant must be snapshot or dynamic")
        if self.kind not in {"time", "space"}:
            raise ValueError("permutation kind must be time or space")
        if self.evidence_status in {
            "evaluated",
            "evidence_insufficient_scientific_failure_fraction",
        }:
            if self.observed_statistic is None:
                raise ValueError("populated permutation diagnostics require an observed statistic")
            observed = _finite(self.observed_statistic, label="observed_statistic")
            null_statistics = _permutation_null_tuple(self.null_statistics)
            failures = sum(value == math.inf for value in null_statistics)
            if self.evidence_status == "evaluated" and failures > (NULL_REPLICATION_COUNT // 100):
                raise ValueError(
                    "evaluated permutation evidence exceeds the 1% failure ceiling; "
                    "the 1% ceiling for scientific failures was breached"
                )
            if (
                self.evidence_status == "evidence_insufficient_scientific_failure_fraction"
                and failures <= NULL_REPLICATION_COUNT // 100
            ):
                raise ValueError(
                    "scientific-failure evidence status requires more than 1% failures"
                )
        elif self.evidence_status in {
            "evidence_insufficient_zero_events",
            "evidence_insufficient_no_placebo_injection",
        }:
            if self.observed_statistic is not None or self.null_statistics:
                raise ValueError(
                    "insufficient permutation evidence must not contain invented statistics"
                )
            observed = None
            null_statistics = ()
        else:
            raise ValueError("permutation evidence status is outside the frozen protocol")
        object.__setattr__(self, "observed_statistic", observed)
        object.__setattr__(self, "null_statistics", null_statistics)

    @property
    def scientific_failure_count(self) -> int:
        return sum(value == math.inf for value in self.null_statistics)

    @property
    def p_value(self) -> float | None:
        if self.observed_statistic is None or not self.null_statistics:
            return None
        return (1.0 + sum(value >= self.observed_statistic for value in self.null_statistics)) / (
            len(self.null_statistics) + 1.0
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "evidence_status": self.evidence_status,
            "kind": self.kind,
            "null_statistics": [
                _json_safe_permutation_value(value) for value in self.null_statistics
            ],
            "observed_statistic": self.observed_statistic,
            "p_value": self.p_value,
            "scientific_failure_count": self.scientific_failure_count,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class InformationGainInterval:
    magnitude_bin: MagnitudeBin
    horizon_days: int
    independent_event_count: int
    evidence_status: EvidenceStatus
    point_estimate: float | None
    lower_95: float | None
    upper_95: float | None

    def __post_init__(self) -> None:
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("information-gain magnitude bin is outside the frozen bins")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("information-gain horizon is outside the frozen windows")
        count = _integer(
            self.independent_event_count,
            label="information-gain independent_event_count",
            positive=False,
        )
        if self.evidence_status not in {
            "evaluated",
            "evidence_insufficient_zero_events",
            "evidence_insufficient_no_random_split",
            "exploratory_low_sample",
            "exploratory_low_sample_no_random_split",
        }:
            raise ValueError("information-gain evidence status is outside the frozen protocol")
        values = (self.point_estimate, self.lower_95, self.upper_95)
        point: float | None
        lower: float | None
        upper: float | None
        if self.evidence_status in {
            "evaluated",
            "evidence_insufficient_no_random_split",
            "exploratory_low_sample",
            "exploratory_low_sample_no_random_split",
        }:
            if (
                count <= 0
                or self.point_estimate is None
                or self.lower_95 is None
                or self.upper_95 is None
            ):
                raise ValueError("populated information gain requires events and a full interval")
            if self.evidence_status == "evaluated" and (
                self.horizon_days not in PRIMARY_MACRO_HORIZONS_DAYS
            ):
                raise ValueError("long-window information gain is descriptive, not confirmatory")
            if self.evidence_status == "evidence_insufficient_no_random_split" and (
                self.horizon_days not in {180, 365}
            ):
                raise ValueError("no-random-split evidence status is restricted to long windows")
            if self.evidence_status == "exploratory_low_sample" and (
                self.magnitude_bin != "M6_plus"
                or count >= 10
                or self.horizon_days not in PRIMARY_MACRO_HORIZONS_DAYS
            ):
                raise ValueError(
                    "low-sample exploratory evidence requires M6+, n<10, and a primary window"
                )
            if self.evidence_status == "exploratory_low_sample_no_random_split" and (
                self.magnitude_bin != "M6_plus"
                or count >= 10
                or self.horizon_days not in {180, 365}
            ):
                raise ValueError(
                    "combined exploratory evidence requires M6+, n<10, and a long window"
                )
            if (
                self.magnitude_bin == "M6_plus"
                and count < 10
                and self.evidence_status
                not in {
                    "exploratory_low_sample",
                    "exploratory_low_sample_no_random_split",
                }
            ):
                raise ValueError("M6+ information gain with n<10 must remain exploratory")
            point = _finite(self.point_estimate, label="information-gain point_estimate")
            lower = _finite(self.lower_95, label="information-gain lower_95")
            upper = _finite(self.upper_95, label="information-gain upper_95")
            if lower > upper:
                raise ValueError("information-gain interval lower bound exceeds upper bound")
        elif self.evidence_status == "evidence_insufficient_zero_events":
            if any(value is not None for value in values):
                raise ValueError("insufficient evidence must not serialize invented IG values")
            if count != 0:
                raise ValueError("zero-event evidence status requires an exact zero count")
            point = lower = upper = None
        else:
            raise ValueError("information-gain evidence status is outside the frozen protocol")
        object.__setattr__(self, "independent_event_count", count)
        object.__setattr__(self, "point_estimate", point)
        object.__setattr__(self, "lower_95", lower)
        object.__setattr__(self, "upper_95", upper)

    def as_mapping(self) -> dict[str, object]:
        return {
            "horizon_days": self.horizon_days,
            "independent_event_count": self.independent_event_count,
            "evidence_status": self.evidence_status,
            "lower_95": self.lower_95,
            "magnitude_bin": self.magnitude_bin,
            "point_estimate": self.point_estimate,
            "upper_95": self.upper_95,
        }


@dataclass(frozen=True, slots=True)
class RegionHorizonMetric:
    region_id: str
    horizon_days: int
    information_gain_nats_per_event: float | None
    supported_event_count: int
    all_study_area_event_count: int
    strict_recall: float | None
    information_gain_evidence_status: RegionalInformationGainEvidenceStatus
    strict_recall_evidence_status: RegionalRecallEvidenceStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _identifier(self.region_id, label="region_id"))
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("regional metric horizon is outside the frozen windows")
        supported_count = _integer(
            self.supported_event_count,
            label="regional supported_event_count",
            positive=False,
        )
        all_count = _integer(
            self.all_study_area_event_count,
            label="regional all_study_area_event_count",
            positive=False,
        )
        if supported_count > all_count:
            raise ValueError("supported regional events must be a subset of all study-area events")
        if self.information_gain_evidence_status == "evaluated":
            if supported_count <= 0 or self.information_gain_nats_per_event is None:
                raise ValueError("evaluated regional information gain requires supported events")
            information_gain = _finite(
                self.information_gain_nats_per_event,
                label="regional information_gain_nats_per_event",
            )
        elif self.information_gain_evidence_status == "evidence_insufficient_zero_supported_events":
            if supported_count != 0 or self.information_gain_nats_per_event is not None:
                raise ValueError("zero-supported-event regional information gain must be None")
            information_gain = None
        else:
            raise ValueError("regional information-gain evidence status is outside the protocol")

        if self.strict_recall_evidence_status == "evaluated":
            if all_count <= 0 or self.strict_recall is None:
                raise ValueError("evaluated regional recall requires study-area events")
            recall = _finite(self.strict_recall, label="regional strict_recall")
            if not 0.0 <= recall <= 1.0:
                raise ValueError("regional strict_recall must lie in [0, 1]")
        elif self.strict_recall_evidence_status == "evidence_insufficient_zero_all_events":
            if all_count != 0 or self.strict_recall is not None:
                raise ValueError("zero-all-event regional strict recall must be None")
            recall = None
        else:
            raise ValueError("regional recall evidence status is outside the frozen protocol")
        object.__setattr__(self, "supported_event_count", supported_count)
        object.__setattr__(self, "all_study_area_event_count", all_count)
        object.__setattr__(self, "information_gain_nats_per_event", information_gain)
        object.__setattr__(self, "strict_recall", recall)

    def as_mapping(self) -> dict[str, object]:
        return {
            "all_study_area_event_count": self.all_study_area_event_count,
            "horizon_days": self.horizon_days,
            "information_gain_nats_per_event": self.information_gain_nats_per_event,
            "information_gain_evidence_status": self.information_gain_evidence_status,
            "region_id": self.region_id,
            "strict_recall": self.strict_recall,
            "strict_recall_evidence_status": self.strict_recall_evidence_status,
            "supported_event_count": self.supported_event_count,
        }


@dataclass(frozen=True, slots=True)
class AlarmBudgetRecallPoint:
    budget_km2: float
    selected_alarm_area_km2: float
    strict_recall: float | None
    all_study_area_event_count: int
    evidence_status: AlarmRecallEvidenceStatus

    def __post_init__(self) -> None:
        budget = _finite(self.budget_km2, label="budget_km2")
        area = _finite(self.selected_alarm_area_km2, label="selected_alarm_area_km2")
        count = _integer(
            self.all_study_area_event_count,
            label="alarm-budget all_study_area_event_count",
            positive=False,
        )
        if budget <= 0.0 or area <= 0.0 or area > budget:
            raise ValueError("selected alarm area must lie in (0, budget_km2]")
        if self.evidence_status == "evaluated":
            if count <= 0 or self.strict_recall is None:
                raise ValueError("evaluated alarm-budget recall requires study-area events")
            recall = _finite(self.strict_recall, label="strict_recall")
            if not 0.0 <= recall <= 1.0:
                raise ValueError("strict_recall must lie in [0, 1]")
        elif self.evidence_status == "evidence_insufficient_zero_all_events":
            if count != 0 or self.strict_recall is not None:
                raise ValueError("zero-all-event alarm-budget recall must be None")
            recall = None
        else:
            raise ValueError("alarm-budget recall evidence status is outside the protocol")
        object.__setattr__(self, "budget_km2", budget)
        object.__setattr__(self, "selected_alarm_area_km2", area)
        object.__setattr__(self, "all_study_area_event_count", count)
        object.__setattr__(self, "strict_recall", recall)

    def as_mapping(self) -> dict[str, object]:
        return {
            "all_study_area_event_count": self.all_study_area_event_count,
            "budget_km2": self.budget_km2,
            "evidence_status": self.evidence_status,
            "selected_alarm_area_km2": self.selected_alarm_area_km2,
            "strict_recall": self.strict_recall,
        }


@dataclass(frozen=True, slots=True)
class AlarmBudgetRecallCurve:
    variant: ModelVariant
    magnitude_bin: MagnitudeBin
    horizon_days: int
    points: tuple[AlarmBudgetRecallPoint, ...]

    def __post_init__(self) -> None:
        if self.variant not in _MODEL_VARIANTS:
            raise ValueError("alarm-budget curve variant is outside the frozen variants")
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("alarm-budget curve magnitude bin is outside the frozen bins")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("alarm-budget curve horizon is outside the frozen windows")
        points = tuple(self.points)
        if len(points) < 3:
            raise ValueError("alarm-budget curves require at least three real budget points")
        budgets = tuple(item.budget_km2 for item in points)
        selected_areas = tuple(item.selected_alarm_area_km2 for item in points)
        recalls = tuple(item.strict_recall for item in points)
        _strictly_increasing(budgets, label="alarm budgets")
        _non_decreasing(selected_areas, label="selected alarm areas")
        event_counts = {item.all_study_area_event_count for item in points}
        evidence_statuses = {item.evidence_status for item in points}
        if len(event_counts) != 1 or len(evidence_statuses) != 1:
            raise ValueError("alarm-budget curve points must share one event denominator")
        numeric_recalls = tuple(value for value in recalls if value is not None)
        if numeric_recalls and len(numeric_recalls) != len(recalls):
            raise ValueError("alarm-budget curve must not mix evaluated and missing recall")
        if any(right < left for left, right in pairwise(numeric_recalls)):
            raise ValueError("strict recall must be non-decreasing with alarm area")
        object.__setattr__(self, "points", points)

    def as_mapping(self) -> dict[str, object]:
        return {
            "horizon_days": self.horizon_days,
            "magnitude_bin": self.magnitude_bin,
            "points": [item.as_mapping() for item in self.points],
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class SameRecallAreaReduction:
    magnitude_bin: MagnitudeBin
    horizon_days: int
    target_recall: float | None
    comparator_variant: ModelVariant
    candidate_variant: ModelVariant
    comparator_area_km2: float | None
    candidate_area_km2: float | None
    area_reduction_lower_95: float | None
    area_reduction_upper_95: float | None
    evidence_status: SameRecallEvidenceStatus = "evaluated"

    def __post_init__(self) -> None:
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("same-recall magnitude bin is outside the frozen bins")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("same-recall horizon is outside the frozen windows")
        if self.comparator_variant not in _MODEL_VARIANTS:
            raise ValueError("same-recall comparator variant is outside the frozen variants")
        if self.candidate_variant not in _MODEL_VARIANTS:
            raise ValueError("same-recall candidate variant is outside the frozen variants")
        if self.candidate_variant == self.comparator_variant:
            raise ValueError("same-recall candidate and comparator must differ")
        recall: float | None
        comparator: float | None
        candidate: float | None
        lower: float | None
        upper: float | None
        if self.evidence_status == "evaluated":
            if (
                self.target_recall is None
                or self.comparator_area_km2 is None
                or self.candidate_area_km2 is None
                or self.area_reduction_lower_95 is None
                or self.area_reduction_upper_95 is None
            ):
                raise ValueError("evaluated same-recall diagnostics require a complete interval")
            recall = _finite(self.target_recall, label="same-recall target_recall")
            comparator = _finite(
                self.comparator_area_km2,
                label="same-recall comparator_area_km2",
            )
            candidate = _finite(
                self.candidate_area_km2,
                label="same-recall candidate_area_km2",
            )
            lower = _finite(
                self.area_reduction_lower_95,
                label="same-recall area_reduction_lower_95",
            )
            upper = _finite(
                self.area_reduction_upper_95,
                label="same-recall area_reduction_upper_95",
            )
            if not 0.0 < recall <= 1.0:
                raise ValueError("same-recall target must lie in (0, 1]")
            if comparator <= 0.0 or candidate <= 0.0:
                raise ValueError("same-recall areas must be positive")
            if lower > upper:
                raise ValueError("same-recall area-reduction lower bound exceeds upper bound")
        elif self.evidence_status in {
            "evidence_insufficient_zero_comparator_recall",
            "evidence_insufficient_target_recall_not_reached",
        }:
            if self.area_reduction_lower_95 is not None or self.area_reduction_upper_95 is not None:
                raise ValueError(
                    "non-evaluable same-recall diagnostics must not invent an interval"
                )
            if self.target_recall is None:
                recall = None
            else:
                recall = _finite(self.target_recall, label="same-recall target_recall")
                if not 0.0 <= recall <= 1.0:
                    raise ValueError("non-evaluable same-recall target must lie in [0, 1]")
            comparator = (
                None
                if self.comparator_area_km2 is None
                else _finite(
                    self.comparator_area_km2,
                    label="same-recall comparator_area_km2",
                )
            )
            candidate = (
                None
                if self.candidate_area_km2 is None
                else _finite(
                    self.candidate_area_km2,
                    label="same-recall candidate_area_km2",
                )
            )
            if comparator is not None and comparator <= 0.0:
                raise ValueError("same-recall comparator area must be positive when present")
            if candidate is not None and candidate <= 0.0:
                raise ValueError("same-recall candidate area must be positive when present")
            lower = upper = None
        else:
            raise ValueError("same-recall evidence status is outside the frozen protocol")
        object.__setattr__(self, "target_recall", recall)
        object.__setattr__(self, "comparator_area_km2", comparator)
        object.__setattr__(self, "candidate_area_km2", candidate)
        object.__setattr__(self, "area_reduction_lower_95", lower)
        object.__setattr__(self, "area_reduction_upper_95", upper)

    @property
    def area_reduction_fraction(self) -> float | None:
        if self.comparator_area_km2 is None or self.candidate_area_km2 is None:
            return None
        return 1.0 - self.candidate_area_km2 / self.comparator_area_km2

    def as_mapping(self) -> dict[str, object]:
        return {
            "area_reduction_fraction": self.area_reduction_fraction,
            "area_reduction_lower_95": self.area_reduction_lower_95,
            "area_reduction_upper_95": self.area_reduction_upper_95,
            "candidate_area_km2": self.candidate_area_km2,
            "candidate_variant": self.candidate_variant,
            "comparator_area_km2": self.comparator_area_km2,
            "comparator_variant": self.comparator_variant,
            "evidence_status": self.evidence_status,
            "horizon_days": self.horizon_days,
            "magnitude_bin": self.magnitude_bin,
            "target_recall": self.target_recall,
        }


@dataclass(frozen=True, slots=True)
class PublicationDiagnostics:
    """Complete numerical inputs required by the frozen nine-panel publication."""

    data_flow: DataMethodFlowDiagnostics
    coefficient_effects: tuple[CoefficientEffectCurve, ...]
    distance_lead_decay: DistanceLeadDecayDiagnostics
    fold_macro_values: tuple[FoldMacroValue, ...]
    permutation_distributions: tuple[PermutationDistribution, ...]
    information_gain_intervals: tuple[InformationGainInterval, ...]
    region_ids: tuple[str, ...]
    region_horizon_metrics: tuple[RegionHorizonMetric, ...]
    alarm_budget_curves: tuple[AlarmBudgetRecallCurve, ...]
    same_recall_area_reductions: tuple[SameRecallAreaReduction, ...]

    def __post_init__(self) -> None:
        effects = tuple(self.coefficient_effects)
        if not effects:
            raise ValueError("publication diagnostics require coefficient effect curves")
        if {item.variant for item in effects} != set(_INCREMENT_VARIANTS):
            raise ValueError("coefficient effects must cover coverage, snapshot, and dynamic")
        effect_keys = tuple((item.variant, item.coefficient_name) for item in effects)
        if len(effect_keys) != len(set(effect_keys)):
            raise ValueError("coefficient effect identities must be unique")

        folds = tuple(self.fold_macro_values)
        if tuple(item.fold_index for item in folds) != (1, 2, 3):
            raise ValueError("publication diagnostics require development folds 1, 2, and 3")

        permutations = tuple(self.permutation_distributions)
        permutation_keys = tuple((item.variant, item.kind) for item in permutations)
        required_permutations = (
            ("dynamic", "time"),
            ("dynamic", "space"),
            ("snapshot", "time"),
        )
        optional_snapshot_space = (*required_permutations, ("snapshot", "space"))
        if permutation_keys not in {required_permutations, optional_snapshot_space}:
            raise ValueError(
                "permutation diagnostics require dynamic-time, dynamic-space, and snapshot-time"
            )

        intervals = tuple(self.information_gain_intervals)
        interval_keys = tuple((item.magnitude_bin, item.horizon_days) for item in intervals)
        if not intervals or len(interval_keys) != len(set(interval_keys)):
            raise ValueError("information-gain interval identities must be non-empty and unique")
        expected_interval_keys = tuple(
            (magnitude_bin, horizon)
            for magnitude_bin in ("M5_6", "M6_plus")
            for horizon in STAGE4_HORIZONS_DAYS
        )
        if interval_keys != expected_interval_keys:
            raise ValueError("information-gain diagnostics must cover both bins and all windows")

        region_ids = tuple(
            _identifier(value, label="publication region_id") for value in self.region_ids
        )
        if not region_ids or len(region_ids) != len(set(region_ids)):
            raise ValueError("publication region IDs must be non-empty and unique")
        regional = tuple(self.region_horizon_metrics)
        regional_keys = tuple((item.region_id, item.horizon_days) for item in regional)
        expected_regional = tuple(
            (region_id, horizon) for region_id in region_ids for horizon in STAGE4_HORIZONS_DAYS
        )
        if regional_keys != expected_regional:
            raise ValueError("regional diagnostics must form a complete region-by-horizon matrix")

        curves = tuple(self.alarm_budget_curves)
        curve_keys = tuple((item.variant, item.magnitude_bin, item.horizon_days) for item in curves)
        if len(curves) < 2 or len(curve_keys) != len(set(curve_keys)):
            raise ValueError("publication requires at least two unique alarm-budget curves")
        if not {"background_no_increment", "dynamic"}.issubset({item.variant for item in curves}):
            raise ValueError("alarm-budget curves must include background and dynamic")
        required_curve_keys = {
            (variant, "M5_6", horizon)
            for variant in ("background_no_increment", "dynamic")
            for horizon in PRIMARY_MACRO_HORIZONS_DAYS
        }
        if not required_curve_keys.issubset(set(curve_keys)):
            raise ValueError(
                "alarm-budget curves must cover M5_6 background and dynamic primary horizons"
            )
        budget_nodes = {tuple(point.budget_km2 for point in item.points) for item in curves}
        if len(budget_nodes) != 1:
            raise ValueError("alarm-budget curves must share the same real area budgets")
        if next(iter(budget_nodes)) != PUBLICATION_ALARM_BUDGETS_KM2:
            raise ValueError("alarm-budget curves must retain the five preregistered budgets")

        reductions = tuple(self.same_recall_area_reductions)
        if not reductions:
            raise ValueError("publication requires same-recall area-reduction diagnostics")
        required_reductions = {
            ("M5_6", horizon, "background_no_increment", "dynamic")
            for horizon in PRIMARY_MACRO_HORIZONS_DAYS
        }
        actual_reductions = {
            (
                item.magnitude_bin,
                item.horizon_days,
                item.comparator_variant,
                item.candidate_variant,
            )
            for item in reductions
        }
        if not required_reductions.issubset(actual_reductions):
            raise ValueError("same-recall diagnostics must cover M5_6 primary horizons")
        curve_key_set = set(curve_keys)
        for item in reductions:
            candidate_key = (item.candidate_variant, item.magnitude_bin, item.horizon_days)
            comparator_key = (item.comparator_variant, item.magnitude_bin, item.horizon_days)
            if candidate_key not in curve_key_set or comparator_key not in curve_key_set:
                raise ValueError("same-recall diagnostics must bind published budget curves")

        object.__setattr__(self, "coefficient_effects", effects)
        object.__setattr__(self, "fold_macro_values", folds)
        object.__setattr__(self, "permutation_distributions", permutations)
        object.__setattr__(self, "information_gain_intervals", intervals)
        object.__setattr__(self, "region_ids", region_ids)
        object.__setattr__(self, "region_horizon_metrics", regional)
        object.__setattr__(self, "alarm_budget_curves", curves)
        object.__setattr__(self, "same_recall_area_reductions", reductions)

    def _content_mapping(self) -> dict[str, object]:
        return {
            "alarm_budget_curves": [item.as_mapping() for item in self.alarm_budget_curves],
            "coefficient_effects": [item.as_mapping() for item in self.coefficient_effects],
            "data_flow": self.data_flow.as_mapping(),
            "distance_lead_decay": self.distance_lead_decay.as_mapping(),
            "fold_macro_values": [item.as_mapping() for item in self.fold_macro_values],
            "information_gain_intervals": [
                item.as_mapping() for item in self.information_gain_intervals
            ],
            "permutation_distributions": [
                item.as_mapping() for item in self.permutation_distributions
            ],
            "region_horizon_metrics": [item.as_mapping() for item in self.region_horizon_metrics],
            "region_ids": list(self.region_ids),
            "same_recall_area_reductions": [
                item.as_mapping() for item in self.same_recall_area_reductions
            ],
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._content_mapping())

    @property
    def sha256(self) -> str:
        return self.content_sha256

    def as_mapping(self) -> dict[str, object]:
        return {**self._content_mapping(), "content_sha256": self.content_sha256}


__all__ = [
    "NULL_REPLICATION_COUNT",
    "PUBLICATION_ALARM_BUDGETS_KM2",
    "AlarmBudgetRecallCurve",
    "AlarmBudgetRecallPoint",
    "AlarmRecallEvidenceStatus",
    "CoefficientEffectCurve",
    "DataMethodFlowDiagnostics",
    "DistanceLeadDecayDiagnostics",
    "FoldEvidenceStatus",
    "FoldMacroValue",
    "IncrementVariant",
    "InformationGainEvidenceStatus",
    "InformationGainInterval",
    "MagnitudeBin",
    "ModelVariant",
    "PermutationDistribution",
    "PermutationEvidenceStatus",
    "PermutationKind",
    "PermutationVariant",
    "PublicationDiagnostics",
    "RegionHorizonMetric",
    "RegionalInformationGainEvidenceStatus",
    "RegionalRecallEvidenceStatus",
    "SameRecallAreaReduction",
    "SameRecallEvidenceStatus",
]
