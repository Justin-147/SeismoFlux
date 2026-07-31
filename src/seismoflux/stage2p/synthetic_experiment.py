"""Known-answer synthetic experiments for the Stage 2P science MVP.

The future targets in this module are deliberately placed in forecast cells
with known P0/P1/PP relationships.  That makes this a software and scientific
logic check, not evidence that real earthquakes are predictable.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np

from seismoflux.data.common import canonical_json_bytes
from seismoflux.stage2p.catalog import SyntheticEvent
from seismoflux.stage2p.evaluation import (
    GateAssessment,
    ScienceEvaluation,
    TargetObservation,
    assess_confirmatory_gate,
    evaluate_science_targets,
)
from seismoflux.stage2p.forecast import (
    ScienceForecastBundle,
    ScienceModelId,
    alarm_area_spread_km2,
    build_science_forecast,
)
from seismoflux.stage2s.contracts import SpatialGrid, SpatialQuadratureFamily
from seismoflux.stage2s.spatial import event_cell_index_25km

ScenarioId: TypeAlias = Literal[
    "recent_activity_predictive",
    "no_recent_signal",
    "recent_activity_misleading",
]
SyntheticKnownAnswerStatus: TypeAlias = Literal["passed", "failed"]

ISSUE_TIME = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
QUERY_CUTOFF = ISSUE_TIME - timedelta(minutes=15)
TRAINING_START = datetime(1970, 1, 1, tzinfo=UTC)
GRID_WIDTH_KM = 2_000.0
GRID_HEIGHT_KM = 2_000.0
TARGET_CLUSTER_COUNT = 12
SYNTHETIC_REGION_COUNT = 4
SYNTHETIC_REGION_BY_CLUSTER_INDEX = tuple(
    f"synthetic-region-{cluster_index % SYNTHETIC_REGION_COUNT:02d}"
    for cluster_index in range(TARGET_CLUSTER_COUNT)
)
TARGET_DAYS_AFTER_ISSUE = (2, 20, 48)
MODEL_IDS: tuple[ScienceModelId, ...] = ("P0", "P1", "PP")


@dataclass(frozen=True, slots=True)
class SyntheticTarget:
    event_id: str
    origin_time: datetime
    x_km: float
    y_km: float
    magnitude: float
    cluster_id: str
    region_id: str


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    scenario_id: ScenarioId
    label_zh: str
    known_answer_zh: str
    forecast_events: tuple[SyntheticEvent, ...]


@dataclass(frozen=True, slots=True)
class SyntheticScenarioResult:
    """One synthetic result.

    ``counterfactual_confirmatory_gate`` only checks whether the known-answer
    fixture would satisfy the frozen arithmetic.  It is never real prediction
    evidence and is deliberately excluded from public synthetic artifacts.
    """

    scenario: SyntheticScenario
    forecast: ScienceForecastBundle
    targets: tuple[SyntheticTarget, ...]
    observations: tuple[TargetObservation, ...]
    evaluation: ScienceEvaluation
    counterfactual_confirmatory_gate: GateAssessment
    expected_behavior_passed: bool

    @property
    def synthetic_known_answer_status(self) -> SyntheticKnownAnswerStatus:
        """Return the only pass/fail status suitable for synthetic artifacts."""

        return "passed" if self.expected_behavior_passed else "failed"


def _rectangular_grid(
    cell_size_km: float,
    *,
    width_km: float = GRID_WIDTH_KM,
    height_km: float = GRID_HEIGHT_KM,
) -> SpatialGrid:
    column_count = int(width_km / cell_size_km)
    row_count = int(height_km / cell_size_km)
    if column_count * cell_size_km != width_km or row_count * cell_size_km != height_km:
        raise ValueError("synthetic dimensions must be exact multiples of every grid size")
    rows = np.repeat(np.arange(row_count, dtype=np.int64), column_count)
    columns = np.tile(np.arange(column_count, dtype=np.int64), row_count)
    xy = np.column_stack(
        (
            (columns.astype(np.float64) + 0.5) * cell_size_km,
            (rows.astype(np.float64) + 0.5) * cell_size_km,
        )
    )
    area = np.full(row_count * column_count, cell_size_km**2, dtype=np.float64)
    size = str(cell_size_km).replace(".", "_")
    cell_ids = tuple(
        f"stage2p_synthetic_{size}_r{int(row):03d}_c{int(column):03d}"
        for row, column in zip(rows, columns, strict=True)
    )
    grid_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "role": "stage2p_science_mvp_synthetic_grid",
                "cell_size_km": cell_size_km,
                "width_km": width_km,
                "height_km": height_km,
                "cell_ids": cell_ids,
            }
        )
    ).hexdigest()
    return SpatialGrid(
        grid_id=grid_id,
        cell_size_km=cell_size_km,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_km=xy,
        clipped_area_km2=area,
    )


def build_synthetic_grid_family() -> SpatialQuadratureFamily:
    """Return the target-free 2000 km square 50/25/12.5 km grid family."""

    return SpatialQuadratureFamily(
        grids=(
            _rectangular_grid(50.0),
            _rectangular_grid(25.0),
            _rectangular_grid(12.5),
        )
    )


def _event(
    event_id: str,
    *,
    origin: datetime,
    x_km: float,
    y_km: float,
    magnitude: float = 4.3,
) -> SyntheticEvent:
    return SyntheticEvent(
        id=event_id,
        origin_time=origin,
        first_seen=origin + timedelta(minutes=5),
        x_km=x_km,
        y_km=y_km,
        magnitude=magnitude,
    )


def _long_term_events() -> tuple[SyntheticEvent, ...]:
    centers = ((400.0, 400.0), (1_000.0, 1_000.0), (1_500.0, 1_500.0))
    offsets = tuple(
        (x_offset, y_offset)
        for y_offset in (-45.0, -15.0, 15.0, 45.0)
        for x_offset in (-60.0, -30.0, 0.0, 30.0, 60.0)
    )
    base = datetime(2020, 1, 1, tzinfo=UTC)
    events: list[SyntheticEvent] = []
    for center_index, (center_x, center_y) in enumerate(centers):
        for offset_index, (offset_x, offset_y) in enumerate(offsets):
            events.append(
                _event(
                    f"long-{center_index:02d}-{offset_index:02d}",
                    origin=base + timedelta(days=center_index * 40, hours=offset_index),
                    x_km=center_x + offset_x,
                    y_km=center_y + offset_y,
                    magnitude=4.2 + 0.1 * (offset_index % 3),
                )
            )
    return tuple(events)


def _window_cluster(
    prefix: str,
    *,
    center: tuple[float, float],
    origin: datetime,
) -> tuple[SyntheticEvent, ...]:
    offsets = tuple(
        (x_offset, y_offset)
        for y_offset in (-125.0, 0.0, 125.0)
        for x_offset in (-187.5, -62.5, 62.5, 187.5)
    )
    return tuple(
        _event(
            f"{prefix}-{index:02d}",
            origin=origin + timedelta(minutes=index),
            x_km=center[0] + offset[0],
            y_km=center[1] + offset[1],
            magnitude=4.4,
        )
        for index, offset in enumerate(offsets)
    )


def build_synthetic_scenarios() -> tuple[SyntheticScenario, ...]:
    """Build the three preregistered known-answer input scenarios."""

    long_term = _long_term_events()
    recent_predictive = _window_cluster(
        "recent-predictive",
        center=(1_480.0, 500.0),
        origin=QUERY_CUTOFF - timedelta(days=10),
    )
    preceding_control = _window_cluster(
        "preceding-control",
        center=(500.0, 1_500.0),
        origin=QUERY_CUTOFF - timedelta(days=45),
    )
    recent_misleading = _window_cluster(
        "recent-misleading",
        center=(1_800.0, 200.0),
        origin=QUERY_CUTOFF - timedelta(days=10),
    )
    return (
        SyntheticScenario(
            scenario_id="recent_activity_predictive",
            label_zh="近期活动有效",
            known_answer_zh="未来目标位于 P1 独有报警格, P1 应同时超过 P0 和 PP。",
            forecast_events=long_term + preceding_control + recent_predictive,
        ),
        SyntheticScenario(
            scenario_id="no_recent_signal",
            label_zh="近期活动无增量",
            known_answer_zh="近期窗和对照窗均为空, P1、PP 必须精确退化为 P0。",
            forecast_events=long_term,
        ),
        SyntheticScenario(
            scenario_id="recent_activity_misleading",
            label_zh="近期活动误导",
            known_answer_zh="未来目标位于 P0/PP 报警而 P1 未报警的格, P1 应变差。",
            forecast_events=long_term + recent_misleading,
        ),
    )


def _greedy_spaced_indices(
    candidate_indices: tuple[int, ...],
    *,
    grid: SpatialGrid,
    score_by_index: dict[int, float],
    count: int = TARGET_CLUSTER_COUNT,
    minimum_spacing_km: float = 100.0,
) -> tuple[int, ...]:
    ordered = sorted(
        candidate_indices,
        key=lambda index: (
            -score_by_index[index],
            int(grid.rows[index]),
            int(grid.columns[index]),
            grid.cell_ids[index].encode("utf-8"),
        ),
    )
    selected: list[int] = []
    for index in ordered:
        x_value, y_value = (float(value) for value in grid.query_xy_km[index])
        if all(
            math.hypot(
                x_value - float(grid.query_xy_km[prior, 0]),
                y_value - float(grid.query_xy_km[prior, 1]),
            )
            >= minimum_spacing_km
            for prior in selected
        ):
            selected.append(index)
            if len(selected) == count:
                return tuple(selected)
    raise ValueError("synthetic scenario cannot provide 12 separated known-answer cells")


def _target_cell_indices(
    scenario_id: ScenarioId,
    forecast: ScienceForecastBundle,
) -> tuple[int, ...]:
    grid = forecast.p0.spatial_density.grid_family.at(25.0)
    selected = {
        model.model_id: frozenset(int(index) for index in model.alarm.selected_indices)
        for model in forecast.models
    }
    intensity = {
        model.model_id: np.asarray(
            model.spatial_density.mass_25km / grid.clipped_area_km2,
            dtype=np.float64,
        )
        for model in forecast.models
    }
    if scenario_id == "recent_activity_predictive":
        candidates = tuple(sorted(selected["P1"] - selected["P0"] - selected["PP"]))
        score = {
            index: math.log(float(intensity["P1"][index]))
            - max(
                math.log(float(intensity["P0"][index])),
                math.log(float(intensity["PP"][index])),
            )
            for index in candidates
            if intensity["P0"][index] > 0.0
            and intensity["P1"][index] > 0.0
            and intensity["PP"][index] > 0.0
        }
    elif scenario_id == "recent_activity_misleading":
        candidates = tuple(sorted((selected["P0"] & selected["PP"]) - selected["P1"]))
        score = {
            index: min(
                math.log(float(intensity["P0"][index])),
                math.log(float(intensity["PP"][index])),
            )
            - math.log(float(intensity["P1"][index]))
            for index in candidates
            if intensity["P0"][index] > 0.0
            and intensity["P1"][index] > 0.0
            and intensity["PP"][index] > 0.0
        }
    else:
        candidates = tuple(sorted(selected["P0"] & selected["P1"] & selected["PP"]))
        score = {index: float(intensity["P0"][index]) for index in candidates}
    eligible = tuple(index for index in candidates if index in score)
    return _greedy_spaced_indices(eligible, grid=grid, score_by_index=score)


def _targets(
    scenario: SyntheticScenario,
    forecast: ScienceForecastBundle,
) -> tuple[SyntheticTarget, ...]:
    grid = forecast.p0.spatial_density.grid_family.at(25.0)
    target_indices = _target_cell_indices(scenario.scenario_id, forecast)
    values: list[SyntheticTarget] = []
    for cluster_index, cell_index in enumerate(target_indices):
        x_km, y_km = (float(value) for value in grid.query_xy_km[cell_index])
        for sequence, day in enumerate(TARGET_DAYS_AFTER_ISSUE):
            values.append(
                SyntheticTarget(
                    event_id=(f"{scenario.scenario_id}-target-{cluster_index:02d}-{sequence:02d}"),
                    origin_time=ISSUE_TIME + timedelta(days=day),
                    x_km=x_km,
                    y_km=y_km,
                    magnitude=5.2 + 0.1 * (sequence % 2),
                    cluster_id=f"synthetic-cluster-{cluster_index:02d}",
                    region_id=SYNTHETIC_REGION_BY_CLUSTER_INDEX[cluster_index],
                )
            )
    return tuple(values)


def _alarm_hit(
    forecast: ScienceForecastBundle,
    model_id: ScienceModelId,
    *,
    x_km: float,
    y_km: float,
) -> bool:
    model = forecast.at(model_id)
    grid = model.spatial_density.grid_family.at(25.0)
    row_column = event_cell_index_25km(x_km * 1_000.0, y_km * 1_000.0)
    lookup = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(zip(grid.rows, grid.columns, strict=True))
    }
    index = lookup.get(row_column)
    return index is not None and index in {int(value) for value in model.alarm.selected_indices}


def _observations(
    forecast: ScienceForecastBundle,
    targets: tuple[SyntheticTarget, ...],
) -> tuple[TargetObservation, ...]:
    rows: list[TargetObservation] = []
    for horizon in (7, 30, 90):
        for target in targets:
            if not (ISSUE_TIME < target.origin_time <= ISSUE_TIME + timedelta(days=horizon)):
                continue
            densities = {
                model_id: float(
                    forecast.at(model_id).spatial_density.density(
                        np.asarray([target.x_km], dtype=np.float64),
                        np.asarray([target.y_km], dtype=np.float64),
                    )[0]
                )
                for model_id in MODEL_IDS
            }
            hits = {
                model_id: _alarm_hit(
                    forecast,
                    model_id,
                    x_km=target.x_km,
                    y_km=target.y_km,
                )
                for model_id in MODEL_IDS
            }
            rows.append(
                TargetObservation(
                    event_id=target.event_id,
                    horizon_days=horizon,
                    cluster_id=target.cluster_id,
                    region_id=target.region_id,
                    in_support=True,
                    model_densities=densities,
                    alarm_hits=hits,
                )
            )
    return tuple(rows)


def _expected_behavior(
    scenario_id: ScenarioId,
    forecast: ScienceForecastBundle,
    evaluation: ScienceEvaluation,
    counterfactual_gate: GateAssessment,
) -> bool:
    p0 = evaluation.comparisons["P1_minus_P0"]
    pp = evaluation.comparisons["P1_minus_PP"]
    if scenario_id == "recent_activity_predictive":
        return (
            counterfactual_gate.status == "passed"
            and p0.macro_recall_gain_percentage_points > 0.0
            and pp.macro_recall_gain_percentage_points > 0.0
            and p0.macro_information_gain_nats_per_event > 0.0
            and pp.macro_information_gain_nats_per_event > 0.0
        )
    if scenario_id == "no_recent_signal":
        return (
            forecast.p1.spatial_density is forecast.p0.spatial_density
            and forecast.pp.spatial_density is forecast.p0.spatial_density
            and p0.macro_recall_gain_percentage_points == 0.0
            and pp.macro_recall_gain_percentage_points == 0.0
            and p0.macro_information_gain_nats_per_event == 0.0
            and pp.macro_information_gain_nats_per_event == 0.0
            and counterfactual_gate.status == "failed"
        )
    return (
        p0.macro_recall_gain_percentage_points < 0.0
        and pp.macro_recall_gain_percentage_points < 0.0
        and p0.macro_information_gain_nats_per_event < 0.0
        and pp.macro_information_gain_nats_per_event < 0.0
        and counterfactual_gate.status == "failed"
    )


def run_synthetic_scenario(
    scenario: SyntheticScenario,
    *,
    grid_family: SpatialQuadratureFamily,
    bootstrap_replicates: int = 2_000,
) -> SyntheticScenarioResult:
    """Run one known-answer scenario without reading any external data."""

    forecast = build_science_forecast(
        scenario.forecast_events,
        issue_time=ISSUE_TIME,
        query_cutoff=QUERY_CUTOFF,
        training_start=TRAINING_START,
        grid_family=grid_family,
    )
    targets = _targets(scenario, forecast)
    observations = _observations(forecast, targets)
    evaluation = evaluate_science_targets(
        observations,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=147,
    )
    counterfactual_gate = assess_confirmatory_gate(evaluation)
    expected = _expected_behavior(
        scenario.scenario_id,
        forecast,
        evaluation,
        counterfactual_gate,
    )
    return SyntheticScenarioResult(
        scenario=scenario,
        forecast=forecast,
        targets=targets,
        observations=observations,
        evaluation=evaluation,
        counterfactual_confirmatory_gate=counterfactual_gate,
        expected_behavior_passed=expected,
    )


def run_all_synthetic_scenarios(
    *,
    bootstrap_replicates: int = 2_000,
) -> tuple[SyntheticScenarioResult, ...]:
    """Run all three scenarios on one shared target-free grid family."""

    family = build_synthetic_grid_family()
    results = tuple(
        run_synthetic_scenario(
            scenario,
            grid_family=family,
            bootstrap_replicates=bootstrap_replicates,
        )
        for scenario in build_synthetic_scenarios()
    )
    if not all(result.expected_behavior_passed for result in results):
        failed = tuple(
            result.scenario.scenario_id for result in results if not result.expected_behavior_passed
        )
        raise ValueError(f"synthetic known-answer behavior failed: {failed}")
    return results


def scenario_summary(result: SyntheticScenarioResult) -> dict[str, object]:
    """Return a compact JSON-ready public summary without target coordinates."""

    evaluation = result.evaluation
    comparisons: dict[str, object] = {}
    for comparison_id, comparison in evaluation.comparisons.items():
        comparisons[comparison_id] = {
            "macro_recall_gain_percentage_points": (comparison.macro_recall_gain_percentage_points),
            "macro_information_gain_nats_per_event": (
                comparison.macro_information_gain_nats_per_event
            ),
            "recall_interval": {
                "lower": comparison.recall_interval.lower,
                "upper": comparison.recall_interval.upper,
            },
            "information_gain_interval": {
                "lower": comparison.information_gain_interval.lower,
                "upper": comparison.information_gain_interval.upper,
            },
            "removal_diagnostics": [
                {
                    "endpoint": item.endpoint,
                    "group_kind": item.group_kind,
                    "removed_id": item.removed_id,
                    "residual_with_original_denominator": (item.residual_with_original_denominator),
                    "remains_positive": item.remains_positive,
                }
                for item in comparison.removal_diagnostics
            ],
        }
    bootstrap_sha256 = hashlib.sha256(
        canonical_json_bytes(
            [[float(value) for value in row] for row in evaluation.bootstrap_samples]
        )
    ).hexdigest()
    return {
        "scenario_id": result.scenario.scenario_id,
        "label_zh": result.scenario.label_zh,
        "known_answer_zh": result.scenario.known_answer_zh,
        "synthetic_only": True,
        "real_prediction_evidence": False,
        "forecast_event_count": len(result.scenario.forecast_events),
        "future_target_count": len(result.targets),
        "unique_target_count": evaluation.unique_event_count,
        "independent_cluster_count": evaluation.independent_cluster_count,
        "independent_region_count": evaluation.independent_region_count,
        "alarm_area_km2": {
            model.model_id: model.alarm.actual_area_km2 for model in result.forecast.models
        },
        "maximum_alarm_area_spread_km2": alarm_area_spread_km2(result.forecast),
        "macro_model_recall": {
            model_id: {
                "strict_event_recall": recall.strict_event_recall,
                "independent_cluster_recall": recall.independent_cluster_recall,
                "independent_region_recall": recall.independent_region_recall,
            }
            for model_id, recall in evaluation.macro_model_recall.items()
        },
        "comparisons": comparisons,
        "synthetic_known_answer_status": result.synthetic_known_answer_status,
        "bootstrap": {
            "seed": evaluation.bootstrap_seed,
            "replicates": evaluation.bootstrap_replicates,
            "endpoint_order": list(evaluation.bootstrap_endpoint_order),
            "samples_sha256": bootstrap_sha256,
        },
    }


def experiment_summary(
    results: tuple[SyntheticScenarioResult, ...],
) -> MappingProxyType[str, object]:
    """Return the compact top-level result used by figures and reports."""

    return MappingProxyType(
        {
            "experiment_id": "stage2p-science-mvp-synthetic-v1",
            "generated_from_real_data": False,
            "real_prediction_evidence": False,
            "issue_time_utc": ISSUE_TIME.isoformat().replace("+00:00", "Z"),
            "query_cutoff_utc": QUERY_CUTOFF.isoformat().replace("+00:00", "Z"),
            "models": list(MODEL_IDS),
            "bandwidth_km": 75.0,
            "mixture_weight": 0.5,
            "alarm_budget_km2": 600_000.0,
            "scenarios": [scenario_summary(result) for result in results],
        }
    )


__all__ = [
    "GRID_HEIGHT_KM",
    "GRID_WIDTH_KM",
    "ISSUE_TIME",
    "MODEL_IDS",
    "QUERY_CUTOFF",
    "SYNTHETIC_REGION_BY_CLUSTER_INDEX",
    "SYNTHETIC_REGION_COUNT",
    "ScenarioId",
    "SyntheticKnownAnswerStatus",
    "SyntheticScenario",
    "SyntheticScenarioResult",
    "SyntheticTarget",
    "build_synthetic_grid_family",
    "build_synthetic_scenarios",
    "experiment_summary",
    "run_all_synthetic_scenarios",
    "run_synthetic_scenario",
    "scenario_summary",
]
