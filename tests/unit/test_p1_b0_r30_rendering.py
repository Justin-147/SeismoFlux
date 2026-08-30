# ruff: noqa: RUF001
"""Visual acceptance checks for the P1-0B synthetic-only model comparison."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence

import pytest

from seismoflux.p1_b0_r30.rendering import (
    P1SyntheticRenderingError,
    build_offline_synthetic_explorer_html,
    build_offline_synthetic_forecast_html,
    render_synthetic_forecast_svg,
    render_synthetic_scenarios_svg,
)


def _model(
    *,
    alarm_cells: Sequence[str],
    hits: Sequence[str],
    misses: Sequence[str],
) -> dict[str, object]:
    total = len(hits) + len(misses)
    cell_ids = ["c00", "c01", "c02", "c10", "c11", "c12"]
    ranking = list(alarm_cells) + [cell_id for cell_id in cell_ids if cell_id not in alarm_cells]
    weights = (0.30, 0.25, 0.15, 0.12, 0.10, 0.08)
    weight_by_cell = dict(zip(ranking, weights, strict=True))
    return {
        "relative_intensity": [weight_by_cell[cell_id] for cell_id in cell_ids],
        "alarm_cell_ids": list(alarm_cells),
        "alarm_area_km2": len(alarm_cells) * 625.0,
        "next_complete_cell_area_km2": 625.0,
        "hit_cluster_ids": list(hits),
        "missed_cluster_ids": list(misses),
        "recall": None if total == 0 else len(hits) / total,
    }


def _scenario(
    *,
    scenario_id: str,
    direction: str,
    label: str,
    target_prefix: str,
    b0_hits: int,
    challenger_hits: int,
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for row in range(2):
        for column in range(3):
            cells.append(
                {
                    "cell_id": f"c{row}{column}",
                    "row": row,
                    "column": column,
                    "x_km": column * 25.0,
                    "y_km": row * 25.0,
                    "area_km2": 625.0,
                }
            )
    target_cells = ((0, 0), (0, 2), (1, 1))
    target_ids = [f"{target_prefix}{index}" for index in range(1, 4)]
    targets: list[dict[str, object]] = []
    for target_id, (row, column) in zip(target_ids, target_cells, strict=True):
        targets.append(
            {
                "cluster_id": target_id,
                "event_id": f"event-{target_id}",
                "origin_time_utc": "2040-01-02T00:00:00Z",
                "x_km": column * 25.0,
                "y_km": row * 25.0,
                "B0_hit": target_id in target_ids[:b0_hits],
                "B0_R30_hit": target_id in target_ids[:challenger_hits],
            }
        )
    b0_alarm = ["c00", "c10"] if b0_hits == 1 else ["c00", "c02"]
    challenger_alarm = ["c00", "c10"] if challenger_hits == 1 else ["c00", "c02"]
    if b0_hits == challenger_hits == 2:
        b0_alarm = ["c00", "c02"]
        challenger_alarm = ["c00", "c02"]
    B0_model = _model(
        alarm_cells=b0_alarm,
        hits=target_ids[:b0_hits],
        misses=target_ids[b0_hits:],
    )
    return {
        "issue_id": "synthetic-p1-20400102T000000Z",
        "scheduled_issue_time_utc": "2040-01-02T00:00:00Z",
        "scenario_id": scenario_id,
        "label": label,
        "expected_direction": direction,
        "interpretation": f"{label}用于验证已知方向。",
        "query_cutoff_utc": "2040-01-01T23:45:00Z",
        "grid": {
            "rows": 2,
            "columns": 3,
            "cell_size_km": 25.0,
            "cells": cells,
        },
        "models": {
            "B0": B0_model,
            "B0_R30": _model(
                alarm_cells=challenger_alarm,
                hits=target_ids[:challenger_hits],
                misses=target_ids[challenger_hits:],
            ),
        },
        "components": {
            "B0": {"active_event_count": 100},
            "R30": {
                "relative_intensity": [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
                "active_event_count": 4,
            },
        },
        "targets": targets,
        "comparison": {
            "cluster_count": 3,
            "B0_hit_clusters": b0_hits,
            "B0_R30_hit_clusters": challenger_hits,
            "recall_gain_percentage_points": (challenger_hits - b0_hits) / 3 * 100,
            "actual_area_difference_km2": 0.0,
            "area_fairness_status": "passed",
        },
    }


def _scenarios() -> list[dict[str, object]]:
    return [
        _scenario(
            scenario_id="known-positive",
            direction="positive",
            label="已知正贡献",
            target_prefix="P",
            b0_hits=1,
            challenger_hits=2,
        ),
        _scenario(
            scenario_id="known-zero",
            direction="zero",
            label="已知零增益",
            target_prefix="Z",
            b0_hits=2,
            challenger_hits=2,
        ),
        _scenario(
            scenario_id="known-negative",
            direction="negative",
            label="已知负贡献",
            target_prefix="N",
            b0_hits=2,
            challenger_hits=1,
        ),
    ]


def test_static_svg_is_deterministic_complete_and_plain_about_scope() -> None:
    scenarios = _scenarios()
    first = render_synthetic_scenarios_svg(scenarios)
    second = render_synthetic_scenarios_svg(list(reversed(scenarios)))
    assert first == second
    assert first.startswith(b'<svg xmlns="http://www.w3.org/2000/svg"')
    text = first.decode("utf-8")
    for expected in (
        "纯合成演示，不是真实预测",
        "SYNTHETIC / NOT A REAL FORECAST",
        "纯合成成熟后回放",
        "正向情景",
        "零增益情景",
        "负向情景",
        "B0_R30 = 0.75 × B0 + 0.25 × R30",
        'data-model="B0"',
        'data-model="B0_R30"',
        'data-layer="relative-intensity"',
        'data-layer="synthetic-targets"',
        'data-outcome="hit"',
        'data-outcome="miss"',
        "召回",
        "实际报警面积",
        "面积公平",
    ):
        assert expected in text
    assert "概率" not in text
    assert "absolute probability" not in text.casefold()


def test_html_is_deterministic_offline_self_contained_and_interactive() -> None:
    scenarios = _scenarios()
    first = build_offline_synthetic_explorer_html(scenarios)
    second = build_offline_synthetic_explorer_html(list(reversed(scenarios)))
    assert first == second
    assert first.startswith("<!doctype html>")
    assert first.rstrip().endswith("</html>")
    for forbidden in (
        "http://",
        "https://",
        "<script src=",
        "<link rel=",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "概率",
    ):
        assert forbidden not in first
    for expected in (
        'id="scenario-select"',
        'id="model-select"',
        'id="target-toggle"',
        'data-layer-relative-intensity="true"',
        'data-layer-alarm="true"',
        'data-layer-targets="toggle"',
        'type="application/json"',
        "known-positive",
        "known-zero",
        "known-negative",
        "B0_R30 = 0.75 × B0 + 0.25 × R30",
        "实际报警面积",
        "面积公平",
        "合成数据",
        "不读取真实地震目录",
        "纯合成演示，不是真实预测",
        "合成目标成熟后的结果回放",
    ):
        assert expected in first


def test_issue_time_forecast_artifacts_are_target_blind_and_deterministic() -> None:
    scenario = _scenarios()[0]
    baseline_svg = render_synthetic_forecast_svg(scenario)
    baseline_html = build_offline_synthetic_forecast_html(scenario)
    assert baseline_svg == render_synthetic_forecast_svg(scenario)
    assert baseline_html == build_offline_synthetic_forecast_html(scenario)

    changed_future = copy.deepcopy(scenario)
    changed_future["scenario_id"] = "secret-future-direction"
    changed_future["label"] = "SECRET_FUTURE_LABEL"
    changed_future["expected_direction"] = "negative"
    changed_future["interpretation"] = "SECRET_FUTURE_RESULT"
    changed_future["targets"] = [
        {
            "cluster_id": "FUTURE_CLUSTER_SHOULD_NEVER_APPEAR",
            "event_id": "FUTURE_EVENT_SHOULD_NEVER_APPEAR",
            "origin_time_utc": "2099-12-31T00:00:00Z",
            "x_km": 9_876_543.25,
            "y_km": -9_876_543.25,
            "B0_hit": False,
            "B0_R30_hit": True,
        }
    ]
    changed_future["comparison"] = {
        "cluster_count": 999,
        "B0_hit_clusters": 999,
        "B0_R30_hit_clusters": 0,
        "recall_gain_percentage_points": -100.0,
        "actual_area_difference_km2": 123.0,
        "area_fairness_status": "future_text_must_be_ignored",
    }
    models = changed_future["models"]
    assert isinstance(models, dict)
    for raw_model in models.values():
        assert isinstance(raw_model, dict)
        raw_model["hit_cluster_ids"] = ["FUTURE_CLUSTER_SHOULD_NEVER_APPEAR"]
        raw_model["missed_cluster_ids"] = []
        raw_model["recall"] = 1.0

    changed_svg = render_synthetic_forecast_svg(changed_future)
    changed_html = build_offline_synthetic_forecast_html(changed_future)
    assert changed_svg == baseline_svg
    assert changed_html == baseline_html

    svg_text = baseline_svg.decode("utf-8")
    for artifact in (svg_text, baseline_html):
        assert "纯合成预测图" in artifact
        assert "不含未来目标" in artifact
        assert "B0_R30 = 0.75 × B0 + 0.25 × R30" in artifact
        assert "synthetic-p1-20400102T000000Z" in artifact
        assert "2040-01-02T00:00:00Z" in artifact
        assert "2040-01-01T23:45:00Z" in artifact
        assert "B0 n=100" in artifact or "B0 / R30 数据水位" in artifact
        assert "R30 n=4" in artifact or "100 / 4" in artifact
        assert "1,250 km²" in artifact
        for forbidden in (
            "FUTURE_CLUSTER_SHOULD_NEVER_APPEAR",
            "FUTURE_EVENT_SHOULD_NEVER_APPEAR",
            "SECRET_FUTURE_LABEL",
            "SECRET_FUTURE_RESULT",
            "9876543.25",
            "targets",
            "命中",
            "漏报",
            "召回",
            "预设方向",
        ):
            assert forbidden not in artifact
    for expected in (
        'id="forecast-model-select"',
        'id="intensity-toggle"',
        'id="alarm-toggle"',
        'data-layer-relative-intensity="toggle"',
        'data-layer-alarm="toggle"',
        'type="application/json"',
        'id="cell-inspector"',
        '"ranked_cell_ids"',
        '"rank_by_cell"',
        'addEventListener("mousemove",inspectCell)',
        'addEventListener("click",inspectCell)',
        "relative_intensity=",
        "rank=",
        "alarm=",
    ):
        assert expected in baseline_html
    for forbidden in ("http://", "https://", "fetch(", "<script src="):
        assert forbidden not in baseline_html
    payload_text = baseline_html.split(
        '<script type="application/json" id="forecast-data">', maxsplit=1
    )[1].split("</script>", maxsplit=1)[0]
    payload = json.loads(payload_text)
    assert isinstance(payload, dict)
    payload_models = payload["models"]
    assert isinstance(payload_models, dict)
    for payload_model in payload_models.values():
        assert isinstance(payload_model, dict)
        ranked = payload_model["ranked_cell_ids"]
        rank_by_cell = payload_model["rank_by_cell"]
        assert isinstance(ranked, list)
        assert isinstance(rank_by_cell, dict)
        assert len(ranked) == 6
        assert sorted(rank_by_cell.values()) == list(range(1, 7))
        assert all(rank_by_cell[cell_id] == index for index, cell_id in enumerate(ranked, 1))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty_issue_id", "issue_id must be a non-empty string"),
        ("wrong_query_offset", "exactly 15 minutes before"),
        ("zero_B0_level", "active_event_count must be an integer >= 1"),
        ("negative_R30_level", "active_event_count must be an integer >= 0"),
    ],
)
def test_issue_time_forecast_rejects_invalid_identity_time_or_data_level(
    mutation: str, message: str
) -> None:
    scenario = _scenarios()[0]
    if mutation == "empty_issue_id":
        scenario["issue_id"] = ""
    elif mutation == "wrong_query_offset":
        scenario["query_cutoff_utc"] = "2040-01-01T23:44:00Z"
    elif mutation == "zero_B0_level":
        components = scenario["components"]
        assert isinstance(components, dict)
        B0 = components["B0"]
        assert isinstance(B0, dict)
        B0["active_event_count"] = 0
    else:
        components = scenario["components"]
        assert isinstance(components, dict)
        R30 = components["R30"]
        assert isinstance(R30, dict)
        R30["active_event_count"] = -1
    with pytest.raises(P1SyntheticRenderingError, match=message):
        render_synthetic_forecast_svg(scenario)


def _forecast_only_mapping(
    *,
    rows: int,
    columns: int,
    alarm_cell_count: int,
    cell_size_km: float = 25.0,
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    cell_ids: list[str] = []
    for row in range(rows):
        for column in range(columns):
            cell_id = f"r{row:03d}c{column:03d}"
            cell_ids.append(cell_id)
            cells.append(
                {
                    "cell_id": cell_id,
                    "row": row,
                    "column": column,
                    "x_km": (column + 0.5) * 25.0,
                    "y_km": (row + 0.5) * 25.0,
                    "area_km2": 625.0,
                }
            )
    raw_mass = [float(len(cells) - index) for index in range(len(cells))]
    total = sum(raw_mass)
    intensity = [value / total for value in raw_mass]
    alarm_ids = cell_ids[:alarm_cell_count]
    model: dict[str, object] = {
        "relative_intensity": intensity,
        "alarm_cell_ids": alarm_ids,
        "alarm_area_km2": alarm_cell_count * 625.0,
        "next_complete_cell_area_km2": 625.0,
    }
    return {
        "issue_id": "synthetic-oversized-check",
        "scheduled_issue_time_utc": "2040-01-02T00:00:00Z",
        "query_cutoff_utc": "2040-01-01T23:45:00Z",
        "grid": {
            "rows": rows,
            "columns": columns,
            "cell_size_km": cell_size_km,
            "cells": cells,
        },
        "models": {"B0": copy.deepcopy(model), "B0_R30": copy.deepcopy(model)},
        "components": {
            "B0": {"active_event_count": 1_000},
            "R30": {"active_event_count": 20},
        },
    }


def test_issue_time_forecast_rejects_50_km_grid() -> None:
    mapping = _forecast_only_mapping(rows=2, columns=3, alarm_cell_count=2, cell_size_km=50.0)
    with pytest.raises(P1SyntheticRenderingError, match="must equal the frozen 25 km"):
        render_synthetic_forecast_svg(mapping)


def test_issue_time_forecast_rejects_cell_area_above_625_km2() -> None:
    mapping = _forecast_only_mapping(rows=2, columns=3, alarm_cell_count=2)
    grid = mapping["grid"]
    assert isinstance(grid, dict)
    cells = grid["cells"]
    assert isinstance(cells, list)
    first_cell = cells[0]
    assert isinstance(first_cell, dict)
    first_cell["area_km2"] = 625.1
    with pytest.raises(P1SyntheticRenderingError, match=r"frozen interval \(0, 625\]"):
        render_synthetic_forecast_svg(mapping)


def test_issue_time_forecast_rejects_800000_km2_B0_alarm() -> None:
    mapping = _forecast_only_mapping(rows=40, columns=40, alarm_cell_count=1_280)
    with pytest.raises(P1SyntheticRenderingError, match="exceeds the frozen 600000 km2 cap"):
        render_synthetic_forecast_svg(mapping)


def test_issue_time_forecast_rejects_non_maximal_B0_prefix_below_area_cap() -> None:
    mapping = _forecast_only_mapping(rows=40, columns=40, alarm_cell_count=1)
    with pytest.raises(P1SyntheticRenderingError, match="largest complete-cell prefix"):
        render_synthetic_forecast_svg(mapping)


def test_issue_time_forecast_rejects_gap_large_enough_for_next_complete_cell() -> None:
    areas = (625.0, 500.0, 100.0, 625.0)
    cells = [
        {
            "cell_id": f"c{column}",
            "row": 0,
            "column": column,
            "x_km": (column + 0.5) * 25.0,
            "y_km": 12.5,
            "area_km2": area,
        }
        for column, area in enumerate(areas)
    ]

    def intensity_for_density(density: Sequence[float]) -> list[float]:
        mass = [value * area for value, area in zip(density, areas, strict=True)]
        total = sum(mass)
        return [value / total for value in mass]

    mapping: dict[str, object] = {
        "issue_id": "synthetic-next-cell-check",
        "scheduled_issue_time_utc": "2040-01-02T00:00:00Z",
        "query_cutoff_utc": "2040-01-01T23:45:00Z",
        "grid": {"rows": 1, "columns": 4, "cell_size_km": 25.0, "cells": cells},
        "models": {
            "B0": {
                "relative_intensity": intensity_for_density((4.0, 2.0, 1.0, 3.0)),
                "alarm_cell_ids": ["c0", "c3"],
                "alarm_area_km2": 1_250.0,
                "next_complete_cell_area_km2": 500.0,
            },
            "B0_R30": {
                "relative_intensity": intensity_for_density((4.0, 3.0, 2.0, 1.0)),
                "alarm_cell_ids": ["c0", "c1"],
                "alarm_area_km2": 1_125.0,
                "next_complete_cell_area_km2": 100.0,
            },
        },
        "components": {
            "B0": {"active_event_count": 10},
            "R30": {"active_event_count": 2},
        },
    }
    with pytest.raises(
        P1SyntheticRenderingError,
        match="largest complete-cell prefix within B0 area",
    ):
        render_synthetic_forecast_svg(mapping)


def test_embedded_json_cannot_close_its_script_element() -> None:
    scenarios = _scenarios()
    scenarios[0]["interpretation"] = "</script><script>throw new Error('unsafe')</script>"
    document = build_offline_synthetic_explorer_html(scenarios)
    assert "</script><script>throw" not in document
    assert "\\u003c/script\\u003e" in document


def test_renderer_rejects_area_that_does_not_match_alarm_cells() -> None:
    scenarios = _scenarios()
    models = scenarios[0]["models"]
    assert isinstance(models, dict)
    challenger = models["B0_R30"]
    assert isinstance(challenger, dict)
    challenger["alarm_area_km2"] = 1.0
    with pytest.raises(P1SyntheticRenderingError, match="disagrees with the selected cells"):
        render_synthetic_scenarios_svg(scenarios)


def test_renderer_rejects_noncanonical_cells_before_intensity_can_be_misaligned() -> None:
    scenarios = _scenarios()
    grid = scenarios[0]["grid"]
    assert isinstance(grid, dict)
    cells = grid["cells"]
    assert isinstance(cells, list)
    cells[0], cells[1] = cells[1], cells[0]
    with pytest.raises(P1SyntheticRenderingError, match="canonical row/column/cell_id order"):
        render_synthetic_scenarios_svg(scenarios)


def test_renderer_rejects_target_outside_the_grid_instead_of_snapping_it() -> None:
    scenarios = _scenarios()
    targets = scenarios[0]["targets"]
    assert isinstance(targets, list)
    target = targets[0]
    assert isinstance(target, dict)
    target["x_km"] = 99_999.0
    with pytest.raises(P1SyntheticRenderingError, match="inside exactly one synthetic grid cell"):
        render_synthetic_scenarios_svg(scenarios)


def test_renderer_rejects_alarm_cells_that_skip_the_stable_ranking() -> None:
    scenarios = _scenarios()
    models = scenarios[0]["models"]
    assert isinstance(models, dict)
    B0 = models["B0"]
    assert isinstance(B0, dict)
    B0["alarm_cell_ids"] = ["c00", "c01"]
    with pytest.raises(P1SyntheticRenderingError, match="complete, unbroken stable-ranking prefix"):
        render_synthetic_scenarios_svg(scenarios)


def test_renderer_rejects_hit_lists_that_disagree_with_alarm_membership() -> None:
    scenarios = _scenarios()
    models = scenarios[0]["models"]
    assert isinstance(models, dict)
    B0 = models["B0"]
    assert isinstance(B0, dict)
    B0["hit_cluster_ids"] = ["P2"]
    B0["missed_cluster_ids"] = ["P1", "P3"]
    with pytest.raises(P1SyntheticRenderingError, match="hit/missed lists disagree"):
        render_synthetic_scenarios_svg(scenarios)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("cluster_count", 99, "counts disagree"),
        ("recall_gain_percentage_points", 99.0, "recall gain disagrees"),
        ("actual_area_difference_km2", 10.0, "area difference disagrees"),
        ("area_fairness_status", "claimed_without_check", "recomputable passed"),
    ],
)
def test_renderer_recomputes_comparison_instead_of_trusting_text(
    field: str, bad_value: object, message: str
) -> None:
    scenarios = _scenarios()
    comparison = scenarios[0]["comparison"]
    assert isinstance(comparison, dict)
    comparison[field] = bad_value
    with pytest.raises(P1SyntheticRenderingError, match=message):
        render_synthetic_scenarios_svg(scenarios)


def test_renderer_rejects_wrong_declared_direction() -> None:
    scenarios = _scenarios()
    scenarios[0]["expected_direction"] = "zero"
    with pytest.raises(P1SyntheticRenderingError, match="expected_direction disagrees"):
        render_synthetic_scenarios_svg(scenarios)


def test_renderer_rejects_challenger_with_more_alarm_area() -> None:
    scenarios = _scenarios()
    models = scenarios[0]["models"]
    assert isinstance(models, dict)
    challenger = models["B0_R30"]
    assert isinstance(challenger, dict)
    challenger["alarm_cell_ids"] = ["c00", "c02", "c01"]
    challenger["alarm_area_km2"] = 1_875.0
    with pytest.raises(P1SyntheticRenderingError, match="may not alarm more area"):
        render_synthetic_scenarios_svg(scenarios)


def test_renderer_rejects_full_cell_area_gap_of_625_km2() -> None:
    scenarios = _scenarios()
    models = scenarios[0]["models"]
    assert isinstance(models, dict)
    challenger = models["B0_R30"]
    assert isinstance(challenger, dict)
    challenger["alarm_cell_ids"] = ["c00"]
    challenger["alarm_area_km2"] = 625.0
    challenger["hit_cluster_ids"] = ["P1"]
    challenger["missed_cluster_ids"] = ["P2", "P3"]
    challenger["recall"] = 1 / 3
    targets = scenarios[0]["targets"]
    assert isinstance(targets, list)
    second_target = targets[1]
    assert isinstance(second_target, dict)
    second_target["B0_R30_hit"] = False
    with pytest.raises(
        P1SyntheticRenderingError,
        match="largest complete-cell prefix within B0 area",
    ):
        render_synthetic_scenarios_svg(scenarios)
