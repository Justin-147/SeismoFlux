"""Deterministic synthetic scenarios for the target-blind P1-0B rehearsal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeAlias

from seismoflux.p1_b0_r30.core import (
    CLUSTER_MAX_DISTANCE_KM,
    PRIMARY_HORIZON_DAYS,
    DualModelForecast,
    GridCell,
    ScoreSummary,
    SequentialReview,
    SyntheticEvent,
    TargetCluster,
    build_dual_model_forecast,
    build_pending_sequential_reviews,
    cluster_target_events,
    make_equal_area_grid,
    score_clusters,
)

ExpectedDirection: TypeAlias = Literal["positive", "zero", "negative"]

SYNTHETIC_QUERY_CUTOFF_UTC = datetime(2026, 9, 9, 15, 45, tzinfo=UTC)
SYNTHETIC_ISSUE_TIME_UTC = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
SYNTHETIC_ISSUE_ID = "p1-20260909T160000Z"


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class SyntheticScenarioResult:
    """One explicit model-behaviour scenario with no claim about real earthquakes."""

    scenario_id: str
    label: str
    expected_direction: ExpectedDirection
    interpretation: str
    forecast: DualModelForecast
    target_events: tuple[SyntheticEvent, ...]
    target_clusters: tuple[TargetCluster, ...]
    score: ScoreSummary
    reviews: tuple[SequentialReview, ...]

    @property
    def observed_direction(self) -> ExpectedDirection:
        gain = self.score.recall_gain_percentage_points
        if gain is None or math.isclose(gain, 0.0, abs_tol=1e-12):
            return "zero"
        return "positive" if gain > 0.0 else "negative"

    def as_mapping(self) -> dict[str, object]:
        score_by_cluster = {score.cluster_id: score for score in self.score.scores}
        B0_hits = [score.cluster_id for score in self.score.scores if score.B0_hit]
        challenger_hits = [score.cluster_id for score in self.score.scores if score.B0_R30_hit]
        all_cluster_ids = [score.cluster_id for score in self.score.scores]

        def model_mapping(
            model_id: Literal["B0", "B0_R30"],
        ) -> dict[str, object]:
            if model_id == "B0":
                surface = self.forecast.B0
                alarm = self.forecast.B0_alarm
                hits = B0_hits
                recall = self.score.B0_recall
            else:
                surface = self.forecast.B0_R30
                alarm = self.forecast.B0_R30_alarm
                hits = challenger_hits
                recall = self.score.B0_R30_recall
            hit_set = set(hits)
            return {
                "model_id": model_id,
                "relative_intensity": surface.relative_intensity.tolist(),
                "alarm_cell_ids": list(alarm.selected_cell_ids),
                "alarm_area_km2": alarm.actual_area_km2,
                "next_complete_cell_area_km2": alarm.next_complete_cell_area_km2,
                "hit_cluster_ids": hits,
                "missed_cluster_ids": [
                    cluster_id for cluster_id in all_cluster_ids if cluster_id not in hit_set
                ],
                "recall": recall,
                "value_semantics": "relative_intensity_not_absolute_probability",
            }

        targets: list[dict[str, object]] = []
        for cluster in self.target_clusters:
            score = score_by_cluster[cluster.cluster_id]
            event = cluster.representative
            targets.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "event_id": event.event_id,
                    "origin_time_utc": _utc_text(event.origin_time_utc),
                    "x_km": event.x_km,
                    "y_km": event.y_km,
                    "B0_hit": score.B0_hit,
                    "B0_R30_hit": score.B0_R30_hit,
                }
            )
        return {
            "scenario_id": self.scenario_id,
            "label": self.label,
            "expected_direction": self.expected_direction,
            "observed_direction": self.observed_direction,
            "interpretation": self.interpretation,
            "issue_id": self.forecast.issue_id,
            "scheduled_issue_time_utc": _utc_text(self.forecast.scheduled_issue_time_utc),
            "query_cutoff_utc": _utc_text(self.forecast.query_cutoff_utc),
            "synthetic_only": True,
            "value_semantics": "relative_intensity_not_absolute_probability",
            "grid": {
                "rows": max(cell.row for cell in self.forecast.grid) + 1,
                "columns": max(cell.column for cell in self.forecast.grid) + 1,
                "cell_size_km": 25.0,
                "cells": [cell.as_mapping() for cell in self.forecast.grid],
            },
            "components": {
                "B0": {
                    "active_event_count": self.forecast.B0.active_event_count,
                },
                "R30": {
                    "relative_intensity": self.forecast.R30.relative_intensity.tolist(),
                    "active_event_count": self.forecast.R30.active_event_count,
                    "empty_recent_fallback_to_B0": self.forecast.recent_fallback_to_B0,
                },
            },
            "models": {
                "B0": model_mapping("B0"),
                "B0_R30": model_mapping("B0_R30"),
            },
            "targets": targets,
            "comparison": {
                "cluster_count": self.score.cluster_count,
                "B0_hit_clusters": self.score.B0_hit_clusters,
                "B0_R30_hit_clusters": self.score.B0_R30_hit_clusters,
                "recall_gain_percentage_points": self.score.recall_gain_percentage_points,
                "actual_area_difference_km2": self.forecast.actual_area_difference_km2,
                "area_fairness_status": "passed",
            },
            "sequential_reviews": [review.as_mapping() for review in self.reviews],
        }


def make_synthetic_model_events(*, include_recent: bool = True) -> tuple[SyntheticEvent, ...]:
    """Generate a broad historical southwest pattern and optional northeast recent pulse."""

    events: list[SyntheticEvent] = []
    index = 0
    for row in range(2, 22, 2):
        for column in range(2, 22, 2):
            origin = datetime(2010, 1, 1, tzinfo=UTC) + timedelta(days=31 * index)
            events.append(
                SyntheticEvent(
                    event_id=f"synthetic-history-{index:03d}",
                    origin_time_utc=origin,
                    available_at_utc=origin + timedelta(minutes=10),
                    x_km=(column + 0.5) * 25.0,
                    y_km=(row + 0.5) * 25.0,
                    magnitude=4.3,
                    source_id="synthetic_history",
                )
            )
            index += 1
    if include_recent:
        recent_cells = ((27, 27), (27, 31), (31, 27), (31, 31))
        for recent_index, (row, column) in enumerate(recent_cells):
            origin = SYNTHETIC_QUERY_CUTOFF_UTC - timedelta(days=3 + 4 * recent_index)
            events.append(
                SyntheticEvent(
                    event_id=f"synthetic-recent-{recent_index:02d}",
                    origin_time_utc=origin,
                    available_at_utc=origin + timedelta(minutes=5),
                    x_km=(column + 0.5) * 25.0,
                    y_km=(row + 0.5) * 25.0,
                    magnitude=4.4,
                    source_id="synthetic_ComCat",
                )
            )
    return tuple(events)


def _select_spaced_cells(
    candidates: tuple[GridCell, ...],
    *,
    count: int,
    occupied: tuple[GridCell, ...] = (),
) -> tuple[GridCell, ...]:
    selected = list(occupied)
    added: list[GridCell] = []
    for cell in sorted(
        candidates, key=lambda item: (item.row, item.column, item.cell_id.encode("utf-8"))
    ):
        if all(
            math.hypot(cell.x_km - other.x_km, cell.y_km - other.y_km)
            > CLUSTER_MAX_DISTANCE_KM + 1e-9
            for other in selected
        ):
            selected.append(cell)
            added.append(cell)
            if len(added) == count:
                return tuple(added)
    raise ValueError(f"synthetic scenario needs {count} spatially independent candidate cells")


def _target_cells(
    forecast: DualModelForecast,
    *,
    expected_direction: ExpectedDirection,
) -> tuple[GridCell, ...]:
    B0 = set(forecast.B0_alarm.selected_cell_ids)
    challenger = set(forecast.B0_R30_alarm.selected_cell_ids)
    cells_by_id = {cell.cell_id: cell for cell in forecast.grid}

    def group(cell_ids: set[str]) -> tuple[GridCell, ...]:
        return tuple(cells_by_id[cell_id] for cell_id in cell_ids)

    common = group(B0 & challenger)
    neither = group(set(cells_by_id) - (B0 | challenger))
    if expected_direction == "positive":
        first = _select_spaced_cells(group(challenger - B0), count=4)
        second = _select_spaced_cells(common, count=6, occupied=first)
    elif expected_direction == "negative":
        first = _select_spaced_cells(group(B0 - challenger), count=4)
        second = _select_spaced_cells(common, count=6, occupied=first)
    else:
        first = _select_spaced_cells(neither, count=5)
        second = _select_spaced_cells(common, count=5, occupied=first)
    return first + second


def _make_target_events(
    cells: tuple[GridCell, ...], *, scenario_id: str
) -> tuple[SyntheticEvent, ...]:
    return tuple(
        SyntheticEvent(
            event_id=f"{scenario_id}-target-{index:02d}",
            origin_time_utc=SYNTHETIC_ISSUE_TIME_UTC + timedelta(days=1 + 2 * index),
            available_at_utc=SYNTHETIC_ISSUE_TIME_UTC + timedelta(days=1 + 2 * index, minutes=5),
            x_km=cell.x_km,
            y_km=cell.y_km,
            magnitude=5.3,
            source_id="synthetic_ComCat",
        )
        for index, cell in enumerate(cells)
    )


def build_synthetic_scenario(
    expected_direction: ExpectedDirection,
    *,
    empty_recent: bool = False,
) -> SyntheticScenarioResult:
    """Run one fully in-memory scenario through forecasting, clustering, and scoring."""

    if expected_direction not in {"positive", "zero", "negative"}:
        raise ValueError("expected_direction must be positive, zero, or negative")
    if empty_recent and expected_direction != "zero":
        raise ValueError("empty R30 is an exact-zero fallback scenario")
    scenario_id = "empty_recent_fallback" if empty_recent else expected_direction
    labels = {
        "positive": "正向: 近期成分多命中",
        "zero": "零效应: 两模型命中相同",
        "negative": "负向: 近期成分少命中",
        "empty_recent_fallback": "空近期窗口: 挑战者精确退回长期背景",
    }
    interpretations = {
        "positive": "在相同报警面积内, 预设合成震群让 B0_R30 比 B0 多命中。",
        "zero": "在相同报警面积内, 两张图对预设合成震群的命中数相同。",
        "negative": "在相同报警面积内, 预设合成震群让 B0_R30 比 B0 少命中。",
        "empty_recent_fallback": "最近30天无合成事件时, 两张相对强度图和报警区完全相同。",
    }
    grid = make_equal_area_grid()
    forecast = build_dual_model_forecast(
        make_synthetic_model_events(include_recent=not empty_recent),
        grid,
        issue_id=SYNTHETIC_ISSUE_ID,
        scheduled_issue_time_utc=SYNTHETIC_ISSUE_TIME_UTC,
    )
    target_cells = _target_cells(forecast, expected_direction=expected_direction)
    target_events = _make_target_events(target_cells, scenario_id=scenario_id)
    clusters = cluster_target_events(
        target_events,
        issue_id=SYNTHETIC_ISSUE_ID,
        issue_time_utc=SYNTHETIC_ISSUE_TIME_UTC,
        horizon_days=PRIMARY_HORIZON_DAYS,
        truth_fetched_at_utc=SYNTHETIC_ISSUE_TIME_UTC + timedelta(days=61),
        grid=grid,
    )
    score = score_clusters(forecast, clusters, horizon_days=PRIMARY_HORIZON_DAYS)
    result = SyntheticScenarioResult(
        scenario_id=scenario_id,
        label=labels[scenario_id],
        expected_direction=expected_direction,
        interpretation=interpretations[scenario_id],
        forecast=forecast,
        target_events=target_events,
        target_clusters=clusters,
        score=score,
        reviews=build_pending_sequential_reviews(score, elapsed_months=6.0),
    )
    if result.observed_direction != expected_direction:
        raise AssertionError(
            "synthetic target construction failed to express its declared direction"
        )
    return result


def build_all_synthetic_scenarios() -> tuple[SyntheticScenarioResult, ...]:
    """Return the frozen positive, zero, and negative visual rehearsal scenarios."""

    return (
        build_synthetic_scenario("positive"),
        build_synthetic_scenario("zero"),
        build_synthetic_scenario("negative"),
    )


__all__ = [
    "SYNTHETIC_ISSUE_ID",
    "SYNTHETIC_ISSUE_TIME_UTC",
    "SYNTHETIC_QUERY_CUTOFF_UTC",
    "ExpectedDirection",
    "SyntheticScenarioResult",
    "build_all_synthetic_scenarios",
    "build_synthetic_scenario",
    "make_synthetic_model_events",
]
