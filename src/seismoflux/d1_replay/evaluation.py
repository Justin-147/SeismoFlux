"""Scientific evaluation for the D1 retrospective development replay.

The independent unit is a preregistered rule-based M5--6 cluster, never a
grid cell.  This module deliberately accepts already-fitted model outcomes so
bootstrap resampling cannot accidentally refit models or treat millions of
cells as independent evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

D1_MODEL_ORDER = (
    "B0",
    "B0_R30",
    "B0_C",
    "B0_C_A_snapshot",
    "B0_C_A_dynamic",
    "B0_R30_C_A_dynamic",
)
D1_HORIZONS_DAYS = (30, 90)
D1_FOLD_IDS = ("fold_1", "fold_2", "fold_3")
D1_AREA_BUDGETS_KM2 = (300_000.0, 450_000.0, 600_000.0, 750_000.0, 960_000.0)
D1_PRIMARY_HORIZON_DAYS = 30
D1_PRIMARY_AREA_KM2 = 600_000.0
D1_ASSESSMENT_ISSUES_PER_FOLD = {30: 8, 90: 3}
D1_BOOTSTRAP_REPLICATIONS = 2_000
D1_BOOTSTRAP_ROOT_SEED = 147

EvidenceLevel = Literal["strong", "promising", "weak", "none_or_harmful"]


def _finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class D1ClusterModelOutcome:
    """One model's result for one active horizon-specific global cluster."""

    cluster_id: str
    fold_id: str
    issue_id: str
    horizon_days: int
    model_id: str
    log_density: float | None
    outside_support: bool
    hit_by_area: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.cluster_id or not self.issue_id:
            raise ValueError("D1 cluster outcome identities must be non-empty")
        if self.fold_id not in D1_FOLD_IDS:
            raise ValueError("D1 cluster outcome fold is not preregistered")
        if self.horizon_days not in D1_HORIZONS_DAYS:
            raise ValueError("D1 cluster outcome horizon is not preregistered")
        if self.model_id not in D1_MODEL_ORDER:
            raise ValueError("D1 cluster outcome model is not preregistered")
        if not isinstance(self.outside_support, bool):
            raise TypeError("outside_support must be boolean")
        if self.outside_support:
            if self.log_density is not None or any(self.hit_by_area):
                raise ValueError("outside-support cluster must have no density and be missed")
        elif self.log_density is None:
            raise ValueError("inside-support cluster must have one finite log density")
        else:
            _finite(self.log_density, label="cluster log density")
        if len(self.hit_by_area) != len(D1_AREA_BUDGETS_KM2):
            raise ValueError("D1 hit vector does not cover all area budgets")
        for hit in self.hit_by_area:
            if not isinstance(hit, bool):
                raise TypeError("D1 alarm hits must be booleans")
        if any(
            left and not right
            for left, right in zip(self.hit_by_area, self.hit_by_area[1:], strict=False)
        ):
            raise ValueError("D1 alarm hits must remain hit as the area prefix expands")

    def hit_at(self, area_km2: float) -> bool:
        try:
            index = D1_AREA_BUDGETS_KM2.index(float(area_km2))
        except ValueError as exc:
            raise ValueError("area is not one of the preregistered D1 budgets") from exc
        return self.hit_by_area[index]


@dataclass(frozen=True, slots=True)
class D1Metric:
    model_id: str
    horizon_days: int
    fold_id: str | None
    area_budget_km2: float
    cluster_count: int
    supported_cluster_count: int
    outside_support_count: int
    hit_count: int
    recall: float | None
    mean_log_density: float | None

    def as_mapping(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "horizon_days": self.horizon_days,
            "fold_id": self.fold_id,
            "area_budget_km2": self.area_budget_km2,
            "cluster_count": self.cluster_count,
            "supported_cluster_count": self.supported_cluster_count,
            "outside_support_count": self.outside_support_count,
            "hit_count": self.hit_count,
            "recall": self.recall,
            "mean_log_density": self.mean_log_density,
        }


@dataclass(frozen=True, slots=True)
class D1IssueAlarmOutcome:
    """Alarm exposure for one issue, including issues with no target cluster."""

    fold_id: str
    issue_id: str
    horizon_days: int
    model_id: str
    actual_area_km2: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.fold_id not in D1_FOLD_IDS or not self.issue_id:
            raise ValueError("D1 issue alarm identity is invalid")
        if self.horizon_days not in D1_HORIZONS_DAYS or self.model_id not in D1_MODEL_ORDER:
            raise ValueError("D1 issue alarm horizon/model is not preregistered")
        if len(self.actual_area_km2) != len(D1_AREA_BUDGETS_KM2):
            raise ValueError("D1 issue alarm does not cover all area budgets")
        previous = -math.inf
        for actual, budget in zip(self.actual_area_km2, D1_AREA_BUDGETS_KM2, strict=True):
            value = _finite(actual, label="actual alarm area")
            if value < previous or value < 0.0 or value > budget:
                raise ValueError("D1 actual alarm areas must be monotone and within budget")
            previous = value


@dataclass(frozen=True, slots=True)
class D1AlarmExposureMetric:
    model_id: str
    horizon_days: int
    fold_id: str | None
    area_budget_km2: float
    issue_count: int
    mean_actual_area_km2: float
    study_area_km2: float
    mean_alarm_fraction: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "horizon_days": self.horizon_days,
            "fold_id": self.fold_id,
            "area_budget_km2": self.area_budget_km2,
            "issue_count": self.issue_count,
            "mean_actual_area_km2": self.mean_actual_area_km2,
            "study_area_km2": self.study_area_km2,
            "mean_alarm_fraction": self.mean_alarm_fraction,
        }


@dataclass(frozen=True, slots=True)
class D1BootstrapEffect:
    model_id: str
    horizon_days: int
    area_budget_km2: float
    observed_hit_gain: int
    observed_recall_gain: float
    lower_95: float
    upper_95: float
    probability_gain_positive: float
    replication_count: int

    def as_mapping(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "horizon_days": self.horizon_days,
            "area_budget_km2": self.area_budget_km2,
            "observed_hit_gain": self.observed_hit_gain,
            "observed_recall_gain": self.observed_recall_gain,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "probability_gain_positive": self.probability_gain_positive,
            "replication_count": self.replication_count,
        }


@dataclass(frozen=True, slots=True)
class D1RawEffectDecision:
    model_id: str
    raw_effect_level: EvidenceLevel
    pooled_hit_gain: int
    pooled_recall_gain: float
    probability_gain_positive: float
    nonworse_fold_count: int
    baseline_efficiency_area_km2: float
    candidate_efficiency_area_km2: float | None
    area_steps_saved: int | None
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "raw_effect_level": self.raw_effect_level,
            "pooled_hit_gain": self.pooled_hit_gain,
            "pooled_recall_gain": self.pooled_recall_gain,
            "probability_gain_positive": self.probability_gain_positive,
            "nonworse_fold_count": self.nonworse_fold_count,
            "baseline_efficiency_area_km2": self.baseline_efficiency_area_km2,
            "candidate_efficiency_area_km2": self.candidate_efficiency_area_km2,
            "area_steps_saved": self.area_steps_saved,
            "reason": self.reason,
        }


def validate_complete_outcomes(
    outcomes: Iterable[D1ClusterModelOutcome],
    *,
    expected_support_by_horizon: Mapping[int, Iterable[tuple[str, str, str]]] | None = None,
) -> tuple[D1ClusterModelOutcome, ...]:
    """Require frozen active-cluster support for all six models per horizon.

    ``expected_support_by_horizon`` is mandatory at the real-run boundary and
    comes directly from ``D1TargetLayer``.  It stays optional only so isolated
    synthetic tests can exercise the metric arithmetic without a full catalog.
    """

    items = tuple(outcomes)
    if not items:
        raise ValueError("D1 evaluation requires cluster outcomes")
    expected_normalized: dict[int, set[tuple[str, str, str]]] | None = None
    if expected_support_by_horizon is not None:
        if set(expected_support_by_horizon) != set(D1_HORIZONS_DAYS):
            raise ValueError("expected D1 support must contain exactly 30d and 90d")
        expected_normalized = {}
        for horizon in D1_HORIZONS_DAYS:
            raw_expected = tuple(expected_support_by_horizon[horizon])
            expected = set(raw_expected)
            if not expected or len(expected) != len(raw_expected):
                raise ValueError("expected D1 support must be non-empty and unique")
            expected_normalized[horizon] = expected
    by_horizon_model: dict[tuple[int, str], set[tuple[str, str, str]]] = defaultdict(set)
    seen: set[tuple[int, str, str]] = set()
    for item in items:
        key = (item.horizon_days, item.cluster_id, item.model_id)
        if key in seen:
            raise ValueError("duplicate D1 cluster/model/horizon outcome")
        seen.add(key)
        by_horizon_model[(item.horizon_days, item.model_id)].add(
            (item.cluster_id, item.fold_id, item.issue_id)
        )
    for horizon in D1_HORIZONS_DAYS:
        reference = by_horizon_model.get((horizon, "B0"))
        if not reference:
            raise ValueError(f"D1 horizon {horizon} has no B0 cluster outcomes")
        if expected_normalized is not None and reference != expected_normalized[horizon]:
            raise ValueError(f"D1 horizon {horizon} outcomes differ from the frozen target support")
        for model in D1_MODEL_ORDER:
            if by_horizon_model.get((horizon, model)) != reference:
                raise ValueError(
                    f"D1 horizon {horizon} model {model} does not share B0 cluster support"
                )
    return items


def summarize_metrics(
    outcomes: Iterable[D1ClusterModelOutcome],
    *,
    expected_support_by_horizon: Mapping[int, Iterable[tuple[str, str, str]]] | None = None,
) -> tuple[D1Metric, ...]:
    items = validate_complete_outcomes(
        outcomes,
        expected_support_by_horizon=expected_support_by_horizon,
    )
    result: list[D1Metric] = []
    for horizon in D1_HORIZONS_DAYS:
        for model in D1_MODEL_ORDER:
            model_items = tuple(
                item for item in items if item.horizon_days == horizon and item.model_id == model
            )
            for fold in (*D1_FOLD_IDS, None):
                selected = (
                    model_items
                    if fold is None
                    else tuple(item for item in model_items if item.fold_id == fold)
                )
                for area_index, area in enumerate(D1_AREA_BUDGETS_KM2):
                    count = len(selected)
                    supported = sum(not item.outside_support for item in selected)
                    outside = count - supported
                    hits = sum(item.hit_by_area[area_index] for item in selected)
                    result.append(
                        D1Metric(
                            model_id=model,
                            horizon_days=horizon,
                            fold_id=fold,
                            area_budget_km2=area,
                            cluster_count=count,
                            supported_cluster_count=supported,
                            outside_support_count=outside,
                            hit_count=hits,
                            recall=None if count == 0 else hits / count,
                            mean_log_density=(
                                None
                                if count == 0 or outside > 0
                                else math.fsum(
                                    item.log_density
                                    for item in selected
                                    if item.log_density is not None
                                )
                                / count
                            ),
                        )
                    )
    return tuple(result)


def validate_complete_alarm_outcomes(
    outcomes: Iterable[D1IssueAlarmOutcome],
    *,
    expected_issues_by_horizon: Mapping[int, Iterable[tuple[str, str]]],
) -> tuple[D1IssueAlarmOutcome, ...]:
    """Bind alarm exposure to every frozen assessment issue, including empty ones."""

    items = tuple(outcomes)
    if set(expected_issues_by_horizon) != set(D1_HORIZONS_DAYS):
        raise ValueError("expected alarm issues must contain exactly 30d and 90d")
    expected: dict[int, set[tuple[str, str]]] = {}
    for horizon in D1_HORIZONS_DAYS:
        raw = tuple(expected_issues_by_horizon[horizon])
        normalized = set(raw)
        if not normalized or len(normalized) != len(raw):
            raise ValueError("expected alarm issue support must be non-empty and unique")
        if any(fold not in D1_FOLD_IDS or not issue for fold, issue in normalized):
            raise ValueError("expected alarm issue identity is invalid")
        expected_per_fold = D1_ASSESSMENT_ISSUES_PER_FOLD[horizon]
        if any(
            sum(fold == candidate_fold for fold, _ in normalized) != expected_per_fold
            for candidate_fold in D1_FOLD_IDS
        ):
            raise ValueError("expected alarm issues differ from the frozen fold water levels")
        expected[horizon] = normalized
    by_horizon_model: dict[tuple[int, str], set[tuple[str, str]]] = defaultdict(set)
    seen: set[tuple[int, str, str]] = set()
    area_by_issue_model: dict[tuple[str, str, str], tuple[float, ...]] = {}
    for item in items:
        key = (item.horizon_days, item.issue_id, item.model_id)
        if key in seen:
            raise ValueError("duplicate D1 issue/model/horizon alarm outcome")
        seen.add(key)
        by_horizon_model[(item.horizon_days, item.model_id)].add((item.fold_id, item.issue_id))
        issue_model_key = (item.fold_id, item.issue_id, item.model_id)
        previous_area = area_by_issue_model.setdefault(issue_model_key, item.actual_area_km2)
        if previous_area != item.actual_area_km2:
            raise ValueError("one issue/model changed alarm area between horizons")
    for horizon in D1_HORIZONS_DAYS:
        for model in D1_MODEL_ORDER:
            if by_horizon_model.get((horizon, model)) != expected[horizon]:
                raise ValueError(f"D1 {horizon}d {model} alarm issues differ from frozen exposures")
    return items


def summarize_alarm_exposure(
    outcomes: Iterable[D1IssueAlarmOutcome],
    *,
    expected_issues_by_horizon: Mapping[int, Iterable[tuple[str, str]]],
    study_area_km2: float,
) -> tuple[D1AlarmExposureMetric, ...]:
    items = validate_complete_alarm_outcomes(
        outcomes,
        expected_issues_by_horizon=expected_issues_by_horizon,
    )
    study_area = _finite(study_area_km2, label="study area")
    if study_area <= 0.0:
        raise ValueError("study area must be positive")
    result: list[D1AlarmExposureMetric] = []
    for horizon in D1_HORIZONS_DAYS:
        for model in D1_MODEL_ORDER:
            model_items = tuple(
                item for item in items if item.horizon_days == horizon and item.model_id == model
            )
            for fold in (*D1_FOLD_IDS, None):
                selected = (
                    model_items
                    if fold is None
                    else tuple(item for item in model_items if item.fold_id == fold)
                )
                if not selected:
                    continue
                for area_index, area in enumerate(D1_AREA_BUDGETS_KM2):
                    mean_area = math.fsum(
                        item.actual_area_km2[area_index] for item in selected
                    ) / len(selected)
                    result.append(
                        D1AlarmExposureMetric(
                            model_id=model,
                            horizon_days=horizon,
                            fold_id=fold,
                            area_budget_km2=area,
                            issue_count=len(selected),
                            mean_actual_area_km2=mean_area,
                            study_area_km2=study_area,
                            mean_alarm_fraction=mean_area / study_area,
                        )
                    )
    return tuple(result)


def paired_cluster_bootstrap(
    outcomes: Iterable[D1ClusterModelOutcome],
    *,
    replications: int = D1_BOOTSTRAP_REPLICATIONS,
    expected_support_by_horizon: Mapping[int, Iterable[tuple[str, str, str]]] | None = None,
) -> tuple[D1BootstrapEffect, ...]:
    """Resample active global clusters with the exact registered PCG64 streams."""

    items = validate_complete_outcomes(
        outcomes,
        expected_support_by_horizon=expected_support_by_horizon,
    )
    if isinstance(replications, bool) or replications <= 0:
        raise ValueError("D1 bootstrap replication count must be positive")
    effects: list[D1BootstrapEffect] = []
    for horizon in D1_HORIZONS_DAYS:
        horizon_items = tuple(item for item in items if item.horizon_days == horizon)
        cluster_ids = tuple(sorted({item.cluster_id for item in horizon_items}))
        index = {cluster_id: position for position, cluster_id in enumerate(cluster_ids)}
        hit_matrix: dict[str, np.ndarray] = {}
        for model in D1_MODEL_ORDER:
            matrix = np.zeros((len(cluster_ids), len(D1_AREA_BUDGETS_KM2)), dtype=np.float64)
            for item in horizon_items:
                if item.model_id == model:
                    matrix[index[item.cluster_id], :] = item.hit_by_area
            hit_matrix[model] = matrix
        replicate_effects = {
            (model, area_index): np.empty(replications, dtype=np.float64)
            for model in D1_MODEL_ORDER[1:]
            for area_index in range(len(D1_AREA_BUDGETS_KM2))
        }
        for replication in range(replications):
            generator = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([D1_BOOTSTRAP_ROOT_SEED, 1, horizon, replication])
                )
            )
            draw = generator.integers(0, len(cluster_ids), size=len(cluster_ids))
            baseline = hit_matrix["B0"][draw, :].mean(axis=0)
            for model in D1_MODEL_ORDER[1:]:
                candidate = hit_matrix[model][draw, :].mean(axis=0)
                for area_index in range(len(D1_AREA_BUDGETS_KM2)):
                    replicate_effects[(model, area_index)][replication] = (
                        candidate[area_index] - baseline[area_index]
                    )
        for model in D1_MODEL_ORDER[1:]:
            for area_index, area in enumerate(D1_AREA_BUDGETS_KM2):
                values = replicate_effects[(model, area_index)]
                observed_candidate = int(hit_matrix[model][:, area_index].sum())
                observed_baseline = int(hit_matrix["B0"][:, area_index].sum())
                lower, upper = np.percentile(values, [2.5, 97.5], method="linear")
                effects.append(
                    D1BootstrapEffect(
                        model_id=model,
                        horizon_days=horizon,
                        area_budget_km2=area,
                        observed_hit_gain=observed_candidate - observed_baseline,
                        observed_recall_gain=(observed_candidate - observed_baseline)
                        / len(cluster_ids),
                        lower_95=float(lower),
                        upper_95=float(upper),
                        probability_gain_positive=float(np.mean(values > 0.0)),
                        replication_count=replications,
                    )
                )
    return tuple(effects)


def minimum_area_reaching_recall(
    metrics: Sequence[D1Metric],
    *,
    model_id: str,
    horizon_days: int,
    target_recall: float,
) -> float | None:
    target = _finite(target_recall, label="target recall")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target recall must lie in [0, 1]")
    candidates = sorted(
        (
            item
            for item in metrics
            if item.model_id == model_id
            and item.horizon_days == horizon_days
            and item.fold_id is None
        ),
        key=lambda item: item.area_budget_km2,
    )
    for item in candidates:
        if item.recall is not None and item.recall >= target:
            return item.area_budget_km2
    return None


def classify_primary_raw_effect(
    metrics: Sequence[D1Metric],
    bootstrap: Sequence[D1BootstrapEffect],
    *,
    model_id: str,
) -> D1RawEffectDecision:
    """Preclassify raw predictive effect before placebo attribution diagnostics.

    This is intentionally not the final scientific conclusion.  Time/space
    placebos, regional direction, and leave-one-cluster-out diagnostics must be
    added before any gain can be attributed to the anomaly mechanism.
    """

    if model_id == "B0" or model_id not in D1_MODEL_ORDER:
        raise ValueError("scientific decision requires one registered candidate model")

    def metric(model: str, fold: str | None, area: float) -> D1Metric:
        matches = [
            item
            for item in metrics
            if item.model_id == model
            and item.horizon_days == D1_PRIMARY_HORIZON_DAYS
            and item.fold_id == fold
            and item.area_budget_km2 == area
        ]
        if len(matches) != 1:
            raise ValueError("primary D1 metric set is incomplete or duplicated")
        return matches[0]

    baseline = metric("B0", None, D1_PRIMARY_AREA_KM2)
    candidate = metric(model_id, None, D1_PRIMARY_AREA_KM2)
    if baseline.recall is None or candidate.recall is None:
        raise ValueError("primary D1 evidence cannot be classified without clusters")
    effect_matches = [
        item
        for item in bootstrap
        if item.model_id == model_id
        and item.horizon_days == D1_PRIMARY_HORIZON_DAYS
        and item.area_budget_km2 == D1_PRIMARY_AREA_KM2
    ]
    if len(effect_matches) != 1:
        raise ValueError("primary D1 bootstrap effect is incomplete or duplicated")
    effect = effect_matches[0]
    fold_gains = tuple(
        metric(model_id, fold, D1_PRIMARY_AREA_KM2).hit_count
        - metric("B0", fold, D1_PRIMARY_AREA_KM2).hit_count
        for fold in D1_FOLD_IDS
    )
    nonworse = sum(gain >= 0 for gain in fold_gains)
    baseline_area = minimum_area_reaching_recall(
        metrics,
        model_id="B0",
        horizon_days=D1_PRIMARY_HORIZON_DAYS,
        target_recall=baseline.recall,
    )
    if baseline_area is None:
        raise ValueError("B0 cannot fail to reach its own primary recall")
    candidate_area = minimum_area_reaching_recall(
        metrics,
        model_id=model_id,
        horizon_days=D1_PRIMARY_HORIZON_DAYS,
        target_recall=baseline.recall,
    )
    area_steps_saved = (
        None
        if candidate_area is None
        else D1_AREA_BUDGETS_KM2.index(baseline_area) - D1_AREA_BUDGETS_KM2.index(candidate_area)
    )
    gain = candidate.hit_count - baseline.hit_count
    probability = effect.probability_gain_positive
    if gain >= 2 and probability >= 0.90 and min(fold_gains) >= -1:
        level: EvidenceLevel = "strong"
        reason = "合并后至少多命中2个震群, 正增益概率不低于0.90, 且单折没有多漏超过1群。"
    elif (gain >= 1 and probability >= 0.80 and nonworse >= 2) or (
        gain == 0 and area_steps_saved is not None and area_steps_saved >= 1
    ):
        level = "promising"
        reason = "达到预登记的多命中稳定性条件, 或在相同命中下至少节省一个面积档。"
    elif (gain > 0 or (area_steps_saved is not None and area_steps_saved > 0)) and min(
        fold_gains
    ) >= -1:
        level = "weak"
        reason = "方向为正且无明显单折伤害, 但独立震群证据尚未达到有希望门。"
    else:
        level = "none_or_harmful"
        reason = "没有增加独立震群命中或节省报警面积, 或折间伤害明显。"
    return D1RawEffectDecision(
        model_id=model_id,
        raw_effect_level=level,
        pooled_hit_gain=gain,
        pooled_recall_gain=candidate.recall - baseline.recall,
        probability_gain_positive=probability,
        nonworse_fold_count=nonworse,
        baseline_efficiency_area_km2=baseline_area,
        candidate_efficiency_area_km2=candidate_area,
        area_steps_saved=area_steps_saved,
        reason=reason,
    )


__all__ = [
    "D1_AREA_BUDGETS_KM2",
    "D1_ASSESSMENT_ISSUES_PER_FOLD",
    "D1_BOOTSTRAP_REPLICATIONS",
    "D1_BOOTSTRAP_ROOT_SEED",
    "D1_FOLD_IDS",
    "D1_HORIZONS_DAYS",
    "D1_MODEL_ORDER",
    "D1_PRIMARY_AREA_KM2",
    "D1_PRIMARY_HORIZON_DAYS",
    "D1AlarmExposureMetric",
    "D1BootstrapEffect",
    "D1ClusterModelOutcome",
    "D1IssueAlarmOutcome",
    "D1Metric",
    "D1RawEffectDecision",
    "classify_primary_raw_effect",
    "minimum_area_reaching_recall",
    "paired_cluster_bootstrap",
    "summarize_alarm_exposure",
    "summarize_metrics",
    "validate_complete_alarm_outcomes",
    "validate_complete_outcomes",
]
