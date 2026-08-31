from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET

import pytest

from seismoflux.p1_b0_r30.production_rendering import (
    ProductionRenderingError,
    build_offline_production_forecast_html,
    parse_production_forecast_view,
    render_production_forecast_svg,
)


def _forecast_mapping() -> dict[str, object]:
    cells = [
        {
            "cell_id": f"r{row:02d}c{column:02d}",
            "row": row,
            "column": column,
            "area_km2": 625.0,
            "support_status": "indeterminate" if (row, column) == (1, 2) else "supported",
        }
        for row in range(2)
        for column in range(3)
    ]
    b0_order = [cell["cell_id"] for cell in cells]
    challenger_order = ["r01c02", "r01c01", "r01c00", "r00c00", "r00c01", "r00c02"]
    return {
        "issue_id": "p1-20260909T160000Z",
        "scheduled_issue_time_utc": "2026-09-09T16:00:00Z",
        "query_cutoff_utc": "2026-09-09T15:45:00Z",
        "source_snapshot_sha256": "1" * 64,
        "source_request_sha256": "2" * 64,
        "support_manifest_sha256": "3" * 64,
        "code_commit": "4" * 40,
        "B0_source_count": 5_991,
        "R30_source_count": 1,
        "grid": {"cell_size_km": 25.0, "cells": cells},
        "models": {
            "B0": {
                "normalized_cell_mass": [0.35, 0.25, 0.15, 0.10, 0.08, 0.07],
                "alarm_cell_ids": b0_order,
                "actual_alarm_area_km2": 3_750.0,
                "next_complete_cell_area_km2": None,
            },
            "B0_R30": {
                "normalized_cell_mass": [0.10, 0.10, 0.10, 0.15, 0.25, 0.30],
                "alarm_cell_ids": challenger_order,
                "actual_alarm_area_km2": 3_750.0,
                "next_complete_cell_area_km2": None,
            },
        },
    }


def test_parse_real_forecast_uses_one_support_and_shared_colour_scale() -> None:
    view = parse_production_forecast_view(_forecast_mapping())

    assert view.issue_id == "p1-20260909T160000Z"
    assert view.query_cutoff_utc == "2026-09-09T15:45:00Z"
    assert view.source_snapshot_sha256 == "1" * 64
    assert view.B0_source_count == 5_991
    assert view.R30_source_count == 1
    assert [model.model_id for model in view.models] == ["B0", "B0_R30"]
    assert view.models[0].ranked_cell_ids[0] == "r00c00"
    assert view.models[1].ranked_cell_ids[0] == "r01c02"
    assert view.shared_colour_maximum == pytest.approx(0.35 / 625.0)


def test_static_svg_is_deterministic_future_blind_and_side_by_side() -> None:
    mapping = _forecast_mapping()
    first = render_production_forecast_svg(mapping)
    second = render_production_forecast_svg(copy.deepcopy(mapping))

    assert first == second
    root = ET.fromstring(first)
    assert root.attrib["data-artifact"] == "p1-real-prospective-forecast"
    assert root.attrib["data-source-snapshot-sha256"] == "1" * 64
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    groups = root.findall(".//svg:g[@data-model]", namespace)
    assert [group.attrib["data-model"] for group in groups] == ["B0", "B0_R30"]
    text = first.decode("utf-8")
    assert "T = 2026-09-09T16:00:00Z" in text
    assert "Q = 2026-09-09T15:45:00Z" in text
    assert "相对强度" in text and "不是概率" in text
    assert "长期目录入选数 5,991" in text
    assert "最近30天M4+入选数 1" in text
    assert "不是效果结论" in text
    assert "不含未来地震" in text
    assert 'data-layer="truth"' not in text
    assert "P1-0C" not in text
    assert "历史适配演练" not in text


def test_static_and_interactive_maps_are_north_up() -> None:
    mapping = _forecast_mapping()
    svg = render_production_forecast_svg(mapping).decode("utf-8")

    south = re.search(r'<rect x="[^"]+" y="([^"]+)"[^>]+data-cell-id="r00c00"', svg)
    north = re.search(r'<rect x="[^"]+" y="([^"]+)"[^>]+data-cell-id="r01c00"', svg)
    assert south is not None and north is not None
    assert float(north.group(1)) < float(south.group(1))

    interactive = build_offline_production_forecast_html(mapping)
    assert "forecast.maximum_row-cell.row" in interactive
    assert "forecast.maximum_row-Math.floor" in interactive


def test_offline_html_has_two_maps_no_external_dependency_and_no_truth_layer() -> None:
    mapping = _forecast_mapping()
    first = build_offline_production_forecast_html(mapping)
    second = build_offline_production_forecast_html(copy.deepcopy(mapping))

    assert first == second
    assert 'id="grid-B0"' in first
    assert 'id="grid-B0_R30"' in first
    assert "同一支持、同一报警面积规则" in first
    assert "不是预测效果结论" in first
    assert "相对强度与顺位不是绝对发震概率" in first
    assert "长期目录入选数 <strong>5,991</strong>" in first
    assert "最近30天M4+入选数 <strong>1</strong>" in first
    assert "未来评价窗成熟后必须另存回放" in first
    assert "http://" not in first and "https://" not in first
    assert "<link" not in first and " src=" not in first
    assert "fetch(" not in first and "XMLHttpRequest" not in first
    assert "data-layer-truth" not in first
    assert "P1-0C" not in first and "历史适配演练" not in first

    payload = first.split('<script type="application/json" id="forecast-data">', 1)[1].split(
        "</script>", 1
    )[0]
    decoded = json.loads(payload)
    assert decoded["source_snapshot_sha256"] == "1" * 64
    assert set(decoded["models"]) == {"B0", "B0_R30"}
    assert not any(
        token in key.casefold()
        for key in decoded
        for token in ("truth", "target", "hit", "recall", "outcome")
    )


def test_future_outcome_field_fails_closed_before_rendering() -> None:
    mapping = _forecast_mapping()
    mapping["truth_clusters"] = []

    with pytest.raises(ProductionRenderingError, match="forbidden future-outcome field"):
        render_production_forecast_svg(mapping)


def test_query_cutoff_must_be_exactly_fifteen_minutes_before_t() -> None:
    mapping = _forecast_mapping()
    mapping["query_cutoff_utc"] = "2026-09-09T15:44:59Z"

    with pytest.raises(ProductionRenderingError, match="exactly 15 minutes"):
        parse_production_forecast_view(mapping)


def test_issue_id_must_be_derived_from_scheduled_t() -> None:
    mapping = _forecast_mapping()
    mapping["issue_id"] = "p1-20260916T160000Z"

    with pytest.raises(ProductionRenderingError, match="does not match scheduled issue"):
        parse_production_forecast_view(mapping)


def test_alarm_cells_must_remain_the_unmodified_complete_cell_prefix() -> None:
    mapping = _forecast_mapping()
    models = mapping["models"]
    assert isinstance(models, dict)
    b0 = models["B0"]
    assert isinstance(b0, dict)
    b0["alarm_cell_ids"] = list(reversed(b0["alarm_cell_ids"]))

    with pytest.raises(ProductionRenderingError, match="unmodified largest complete-cell"):
        parse_production_forecast_view(mapping)


def test_frozen_area_cap_selects_exactly_600000_square_kilometres() -> None:
    mapping = _forecast_mapping()
    cells = [
        {
            "cell_id": f"r{row:02d}c{column:02d}",
            "row": row,
            "column": column,
            "area_km2": 625.0,
            "support_status": "supported",
        }
        for row in range(31)
        for column in range(31)
    ]
    grid = mapping["grid"]
    assert isinstance(grid, dict)
    grid["cells"] = cells
    mass = [1.0 / len(cells)] * len(cells)
    selected = [cell["cell_id"] for cell in cells[:960]]
    models = mapping["models"]
    assert isinstance(models, dict)
    for model in models.values():
        assert isinstance(model, dict)
        model["normalized_cell_mass"] = mass
        model["alarm_cell_ids"] = selected
        model["actual_alarm_area_km2"] = 600_000.0
        model["next_complete_cell_area_km2"] = 625.0

    view = parse_production_forecast_view(mapping)

    assert view.models[0].actual_alarm_area_km2 == 600_000.0
    assert view.models[1].actual_alarm_area_km2 == 600_000.0
    assert len(view.models[0].alarm_cell_ids) == 960


def test_model_cannot_assign_mass_outside_frozen_support() -> None:
    mapping = _forecast_mapping()
    grid = mapping["grid"]
    assert isinstance(grid, dict)
    cells = grid["cells"]
    assert isinstance(cells, list)
    cells[0]["support_status"] = "unsupported"

    with pytest.raises(ProductionRenderingError, match="outside the frozen support"):
        parse_production_forecast_view(mapping)


def test_snapshot_hash_and_code_commit_are_strictly_canonical() -> None:
    mapping = _forecast_mapping()
    mapping["source_snapshot_sha256"] = "A" * 64
    with pytest.raises(ProductionRenderingError, match="lowercase SHA-256"):
        parse_production_forecast_view(mapping)

    mapping = _forecast_mapping()
    mapping["code_commit"] = "4" * 39
    with pytest.raises(ProductionRenderingError, match="40-character commit"):
        parse_production_forecast_view(mapping)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("B0_source_count", 0, "B0_source_count must be positive"),
        ("R30_source_count", -1, "R30_source_count must be in"),
        ("R30_source_count", 5_992, "R30_source_count must be in"),
        ("R30_source_count", 1.0, "must be an integer"),
    ],
)
def test_source_counts_fail_closed_when_tampered(field: str, value: object, message: str) -> None:
    mapping = _forecast_mapping()
    mapping[field] = value

    with pytest.raises(ProductionRenderingError, match=message):
        parse_production_forecast_view(mapping)
