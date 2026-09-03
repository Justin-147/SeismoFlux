"""Synthetic fixed-task figure tests; never open stored scientific results."""

import copy
from pathlib import Path
from typing import Any

import pytest

from seismoflux.multitask_s3.strata_figure import (
    BANDS,
    BUDGETS,
    HORIZONS,
    SOURCE_FILTER,
    VIEWS,
    _selected_sample_counts,
    _selected_values,
    render_strata_figure,
)


def _blocks() -> list[dict[str, Any]]:
    blocks = []
    for horizon in HORIZONS:
        rows = [
            {
                **SOURCE_FILTER,
                "horizon_days": horizon,
                "magnitude_band": band,
                "event_view": view,
                "area_budget_km2": budget,
                "national": {
                    "delta_recall_pp": float(index - 2),
                    "unique_event_count": 3,
                    "unique_episode_count": 2,
                },
            }
            for view in VIEWS
            for band in BANDS
            for index, budget in enumerate(BUDGETS)
        ]
        blocks.append(
            {
                "fold_scope": "POOLED_A_DEVELOPMENT",
                "horizon_days": horizon,
                "status": "no_complete_evaluation_windows_NA" if horizon == 365 else "summarized",
                "rows": [] if horizon == 365 else rows,
            }
        )
    return blocks


def test_fixed_complete_matrix_distinguishes_zero_from_explicit_na() -> None:
    values = _selected_values(_blocks())
    assert tuple(values) == VIEWS
    assert all(
        len(matrix) == 10 and all(len(row) == 5 for row in matrix) for matrix in values.values()
    )
    for matrix in values.values():
        assert matrix[0] == [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert matrix[-2:] == [[None] * 5, [None] * 5]


@pytest.mark.parametrize(
    "field,value",
    [
        ("axis", "all_reports_descriptive"),
        ("mode", "secondary_70km"),
        ("reference", "R30_REFERENCE"),
        ("candidate", "CAT_COV"),
        ("fold_scope", "A_DEV_2023_2024"),
    ],
)
def test_other_axes_modes_references_and_folds_are_excluded(field: str, value: str) -> None:
    blocks = _blocks()
    extra = copy.deepcopy(blocks[0]["rows"][0])
    extra[field] = value
    extra["national"]["delta_recall_pp"] = 99.0
    blocks[0]["rows"].append(extra)
    assert _selected_values(blocks)["anchor"][0][0] == -2.0


def test_missing_or_duplicate_selected_tasks_raise() -> None:
    missing = _blocks()
    missing[0]["rows"].pop()
    with pytest.raises(ValueError, match="missing selected task"):
        _selected_values(missing)
    duplicate = _blocks()
    duplicate[0]["rows"].append(duplicate[0]["rows"][0])
    with pytest.raises(ValueError, match="duplicate selected task"):
        _selected_values(duplicate)


def test_missing_or_duplicate_horizon_blocks_raise() -> None:
    with pytest.raises(ValueError, match="missing pooled horizon"):
        _selected_values(_blocks()[:-1])
    blocks = _blocks()
    with pytest.raises(ValueError, match="duplicate pooled horizon"):
        _selected_values([*blocks, blocks[0]])


def test_missing_horizon_must_not_be_silently_represented_as_zero() -> None:
    blocks = _blocks()
    blocks[-1]["status"] = "summarized"
    with pytest.raises(ValueError, match="missing selected task"):
        _selected_values(blocks)


def test_fully_enumerated_365_tasks_can_be_explicit_na_but_never_numeric() -> None:
    blocks = _blocks()
    rows = copy.deepcopy(blocks[0]["rows"])
    for row in rows:
        row["horizon_days"] = 365
        row["national"]["delta_recall_pp"] = None
    blocks[-1].update(status="summarized", rows=rows)
    assert _selected_values(blocks)["anchor"][-1] == [None] * 5
    rows[0]["national"]["delta_recall_pp"] = 0.0
    with pytest.raises(ValueError, match="must remain NA"):
        _selected_values(blocks)


def test_render_low_dpi_preserves_svg_text_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "synthetic_figure"
    result = render_strata_figure(_blocks(), output, dpi=40)
    assert result["source_filter"] == SOURCE_FILTER
    assert result["selected_task_count"] == 200
    assert result["finite_task_count"] == 160
    assert result["NA_task_count"] == 40
    assert result["color_limits_pp"] == [-2.0, 2.0]
    assert result["panel_sample_counts"]["anchor"][0] == {
        "unique_event_count": 3,
        "unique_episode_count": 2,
    }
    assert result["panel_sample_counts"]["anchor"][-1] is None
    png = Path(result["paths"]["png"])
    svg = Path(result["paths"]["svg"])
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    text = svg.read_text(encoding="utf-8")
    assert "<text" in text
    assert "首震" in text and "NA" in text
    original = (png.read_bytes(), svg.read_bytes())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_strata_figure(_blocks(), output, dpi=40)
    assert (png.read_bytes(), svg.read_bytes()) == original


def test_sample_counts_must_be_identical_across_budgets() -> None:
    blocks = _blocks()
    blocks[0]["rows"][1]["national"]["unique_event_count"] = 4
    with pytest.raises(ValueError, match="across alarm budgets"):
        _selected_sample_counts(blocks)
