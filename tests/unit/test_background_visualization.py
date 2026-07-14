from __future__ import annotations

import inspect
import struct

import numpy as np
import pytest
from shapely.geometry import Polygon

from seismoflux.background.grid import EqualAreaGrid, build_clipped_grid
from seismoflux.background.visualization import (
    COLORBAR_LABEL,
    COLORMAP_NAME,
    FIGURE_DPI,
    FIGURE_HEIGHT_PX,
    FIGURE_WIDTH_PX,
    FONT_FAMILY,
    PANEL_TITLES,
    render_conditional_intensity_figure,
    render_conditional_intensity_png,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk_types(payload: bytes) -> tuple[bytes, ...]:
    chunk_types: list[bytes] = []
    offset = len(_PNG_SIGNATURE)
    while offset < len(payload):
        length = int.from_bytes(payload[offset : offset + 4], byteorder="big")
        chunk_types.append(payload[offset + 4 : offset + 8])
        offset += 12 + length
    assert offset == len(payload)
    return tuple(chunk_types)


def _small_clipped_grid() -> EqualAreaGrid:
    study_area = Polygon(
        [
            (-5_000.0, -4_000.0),
            (42_000.0, -4_000.0),
            (42_000.0, 38_000.0),
            (12_000.0, 38_000.0),
            (-5_000.0, 18_000.0),
            (-5_000.0, -4_000.0),
        ],
        holes=[
            [
                (5_000.0, 5_000.0),
                (9_000.0, 5_000.0),
                (9_000.0, 9_000.0),
                (5_000.0, 9_000.0),
                (5_000.0, 5_000.0),
            ]
        ],
    )
    return build_clipped_grid(study_area, cell_size_km=25.0)


def _valid_arrays(cell_count: int) -> tuple[np.ndarray, np.ndarray]:
    background = np.linspace(0.05, 0.35, cell_count, dtype=np.float64)
    trigger = np.linspace(0.0, 0.12, cell_count, dtype=np.float64)
    return background, trigger


def test_small_exactly_clipped_grid_renders_a_fixed_deterministic_png() -> None:
    grid = _small_clipped_grid()
    background, trigger = _valid_arrays(len(grid.cells))

    first = render_conditional_intensity_figure(
        grid,
        background,
        trigger,
        issue_date="2025-06-19",
        model_variant="etas/mc3.0/grid25",
        data_cutoff="2025-06-18T16:00:00Z",
    )
    second = render_conditional_intensity_figure(
        grid,
        background,
        trigger,
        issue_date="2025-06-19",
        model_variant="etas/mc3.0/grid25",
        data_cutoff="2025-06-18T16:00:00Z",
    )

    assert first.png_bytes.startswith(_PNG_SIGNATURE)
    assert first.png_bytes.endswith(b"IEND\xaeB`\x82")
    assert first.png_bytes == second.png_bytes
    assert (
        render_conditional_intensity_png(
            grid,
            background,
            trigger,
            issue_date="2025-06-19",
            model_variant="etas/mc3.0/grid25",
            data_cutoff="2025-06-18T16:00:00Z",
        )
        == first.png_bytes
    )
    width, height = struct.unpack(">II", first.png_bytes[16:24])
    assert (width, height) == (first.width_px, first.height_px)
    assert (width, height, first.dpi) == (FIGURE_WIDTH_PX, FIGURE_HEIGHT_PX, FIGURE_DPI)
    assert first.font_family == FONT_FAMILY
    assert first.colormap_name == COLORMAP_NAME
    assert b"tIME" not in _png_chunk_types(first.png_bytes)
    assert b"Creation Time" not in first.png_bytes


def test_labels_and_metadata_use_only_conditional_intensity_terminology() -> None:
    grid = _small_clipped_grid()
    background, trigger = _valid_arrays(len(grid.cells))

    rendered = render_conditional_intensity_figure(
        grid,
        background,
        trigger,
        issue_date="2025-06-19",
        model_variant="spatial-poisson/grid25",
        data_cutoff="2025-06-18T16:00:00Z",
    )

    assert (
        rendered.panel_titles
        == PANEL_TITLES
        == (
            "背景条件强度",
            "触发条件强度",
            "总条件强度",
        )
    )
    assert rendered.colorbar_label == COLORBAR_LABEL == "条件强度 (期望事件强度)"
    assert dict(rendered.png_metadata)["Geometry"] == (
        "EqualAreaGrid clipped cells and study-area boundary"
    )
    visible_and_metadata_text = "\n".join(
        (
            *rendered.panel_titles,
            rendered.colorbar_label,
            rendered.footer_label,
            *(value for _, value in rendered.png_metadata),
        )
    )
    assert "条件强度" in visible_and_metadata_text
    assert "期望事件强度" in visible_and_metadata_text
    assert "绝对概率" not in visible_and_metadata_text
    assert "发震概率" not in visible_and_metadata_text
    assert "Creation Time" not in dict(rendered.png_metadata)


def test_public_render_api_has_no_target_or_external_spatial_layer_inputs() -> None:
    expected = {
        "grid",
        "background_intensity",
        "trigger_intensity",
        "issue_date",
        "model_variant",
        "data_cutoff",
    }
    for function in (render_conditional_intensity_figure, render_conditional_intensity_png):
        parameters = set(inspect.signature(function).parameters)
        assert parameters == expected
        joined = " ".join(parameters).lower()
        for forbidden in ("target", "epicenter", "earthquake", "anomaly", "fault", "basemap"):
            assert forbidden not in joined


@pytest.mark.parametrize(
    ("which", "replacement", "message"),
    [
        ("background", np.array([1.0, 2.0]), "shape"),
        ("background", np.array([[1.0, 2.0]]), "shape"),
        ("background", np.array([np.nan]), "finite non-negative"),
        ("trigger", np.array([-1.0]), "finite non-negative"),
        ("trigger", np.array([np.inf]), "finite non-negative"),
    ],
)
def test_intensity_arrays_reject_wrong_shape_or_invalid_values(
    which: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    grid = build_clipped_grid(
        Polygon([(0.0, 0.0), (25_000.0, 0.0), (25_000.0, 25_000.0), (0.0, 0.0)]),
        cell_size_km=25.0,
    )
    arrays: dict[str, np.ndarray] = {
        "background": np.array([0.2]),
        "trigger": np.array([0.1]),
    }
    arrays[which] = replacement

    with pytest.raises(ValueError, match=message):
        render_conditional_intensity_figure(
            grid,
            arrays["background"],
            arrays["trigger"],
            issue_date="2025-06-19",
            model_variant="etas/grid25",
            data_cutoff="2025-06-18T16:00:00Z",
        )


def test_total_is_cellwise_component_sum_and_sum_must_remain_finite() -> None:
    grid = build_clipped_grid(
        Polygon([(0.0, 0.0), (25_000.0, 0.0), (25_000.0, 25_000.0), (0.0, 0.0)]),
        cell_size_km=25.0,
    )
    rendered = render_conditional_intensity_figure(
        grid,
        np.array([0.2]),
        np.array([0.1]),
        issue_date="2025-06-19",
        model_variant="etas/grid25",
        data_cutoff="2025-06-18T16:00:00Z",
    )
    assert dict(rendered.png_metadata)["TotalDefinition"] == (
        "background_intensity + trigger_intensity (cellwise)"
    )

    with pytest.raises(ValueError, match="must remain finite"):
        render_conditional_intensity_figure(
            grid,
            np.array([1.0e308]),
            np.array([1.0e308]),
            issue_date="2025-06-19",
            model_variant="etas/grid25",
            data_cutoff="2025-06-18T16:00:00Z",
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("issue_date", "2025-6-19", "canonical YYYY-MM-DD"),
        ("data_cutoff", "2025-06-18T16:00:00+00:00", "canonical"),
        ("data_cutoff", "2025-06-18T16:00:00", "canonical"),
        ("model_variant", "etas model", "safe ASCII"),
        ("model_variant", "绝对概率", "safe ASCII"),
    ],
)
def test_identity_labels_must_be_canonical_and_cannot_inject_unsafe_wording(
    keyword: str,
    value: str,
    message: str,
) -> None:
    grid = _small_clipped_grid()
    background, trigger = _valid_arrays(len(grid.cells))
    identity = {
        "issue_date": "2025-06-19",
        "model_variant": "etas/grid25",
        "data_cutoff": "2025-06-18T16:00:00Z",
    }
    identity[keyword] = value

    with pytest.raises(ValueError, match=message):
        render_conditional_intensity_figure(
            grid,
            background,
            trigger,
            issue_date=identity["issue_date"],
            model_variant=identity["model_variant"],
            data_cutoff=identity["data_cutoff"],
        )
