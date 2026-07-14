"""Deterministic, target-independent stage-2 conditional-intensity maps.

The renderer accepts only a frozen equal-area grid and cell-aligned intensity
arrays.  It deliberately has no catalogue, epicentre, anomaly, fault, basemap,
or candidate-region input, so later outcomes cannot affect the rendered spatial
support.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from seismoflux.background.grid import EqualAreaGrid

PANEL_TITLES: Final[tuple[str, str, str]] = (
    "背景条件强度",
    "触发条件强度",
    "总条件强度",
)
COLORBAR_LABEL: Final = "条件强度 (期望事件强度)"
FIGURE_DPI: Final = 150
FIGURE_WIDTH_IN: Final = 12.0
FIGURE_HEIGHT_IN: Final = 4.4
FIGURE_WIDTH_PX: Final = int(FIGURE_WIDTH_IN * FIGURE_DPI)
FIGURE_HEIGHT_PX: Final = int(FIGURE_HEIGHT_IN * FIGURE_DPI)
FONT_FAMILY: Final = "Microsoft YaHei"
COLORMAP_NAME: Final = "seismoflux_conditional_intensity"
_MODEL_VARIANT_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+:-]{0,127}")
_UTC_TIMESTAMP_PATTERN: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_COLOR_STOPS: Final[tuple[str, ...]] = (
    "#313695",
    "#4575b4",
    "#74add1",
    "#ffffbf",
    "#f46d43",
    "#d73027",
    "#a50026",
)


@dataclass(frozen=True, slots=True)
class ConditionalIntensityRender:
    """PNG bytes plus the exact labels and fixed encoding properties used."""

    png_bytes: bytes
    panel_titles: tuple[str, str, str]
    colorbar_label: str
    footer_label: str
    png_metadata: tuple[tuple[str, str], ...]
    width_px: int
    height_px: int
    dpi: int
    font_family: str
    colormap_name: str


def _canonical_issue_date(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("issue_date must be a canonical YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("issue_date must be a canonical YYYY-MM-DD string") from error
    if parsed.isoformat() != value:
        raise ValueError("issue_date must be a canonical YYYY-MM-DD string")
    return value


def _canonical_data_cutoff(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("data_cutoff must be a canonical UTC timestamp")
    if _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError("data_cutoff must use canonical YYYY-MM-DDTHH:MM:SSZ form")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("data_cutoff must use canonical YYYY-MM-DDTHH:MM:SSZ form") from error
    canonical = parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    if canonical != value:
        raise ValueError("data_cutoff must use canonical YYYY-MM-DDTHH:MM:SSZ form")
    return value


def _canonical_model_variant(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("model_variant must be a string")
    if _MODEL_VARIANT_PATTERN.fullmatch(value) is None:
        raise ValueError("model_variant must use 1-128 safe ASCII identifier characters")
    return value


def _validated_intensity(
    label: str,
    values: ArrayLike,
    *,
    cell_count: int,
) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a numeric one-dimensional array") from error
    if result.shape != (cell_count,):
        raise ValueError(f"{label} must have shape ({cell_count},)")
    if not bool(np.isfinite(result).all()) or bool((result < 0.0).any()):
        raise ValueError(f"{label} must contain only finite non-negative values")
    return np.array(result, dtype=np.float64, copy=True)


def _polygon_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for member in geometry.geoms:
            if isinstance(member, Polygon | MultiPolygon | GeometryCollection):
                parts.extend(_polygon_parts(member))
        if parts:
            return tuple(parts)
    raise ValueError("render geometry must be Polygon or MultiPolygon")


def _ring_path_arrays(coordinates: object) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    vertices = np.asarray(coordinates, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 4:
        raise ValueError("render geometry contains an invalid closed ring")
    codes = np.full(vertices.shape[0], MatplotlibPath.LINETO, dtype=np.uint8)
    codes[0] = MatplotlibPath.MOVETO
    codes[-1] = MatplotlibPath.CLOSEPOLY
    return vertices, codes


def _geometry_path(geometry: BaseGeometry) -> MatplotlibPath:
    vertex_parts: list[NDArray[np.float64]] = []
    code_parts: list[NDArray[np.uint8]] = []
    for raw_polygon in _polygon_parts(geometry):
        polygon = orient(raw_polygon, sign=1.0)
        rings = (polygon.exterior, *polygon.interiors)
        for ring in rings:
            vertices, codes = _ring_path_arrays(ring.coords)
            vertex_parts.append(vertices)
            code_parts.append(codes)
    if not vertex_parts:
        raise ValueError("render geometry must not be empty")
    return MatplotlibPath(
        np.concatenate(vertex_parts, axis=0),
        np.concatenate(code_parts, axis=0),
    )


def _add_intensity_panel(
    axis: Axes,
    *,
    grid: EqualAreaGrid,
    values: NDArray[np.float64],
    title: str,
    colormap: LinearSegmentedColormap,
    normalization: Normalize,
    font: FontProperties,
) -> PatchCollection:
    patches = [PathPatch(_geometry_path(cell.clipped_geometry)) for cell in grid.cells]
    collection = PatchCollection(
        patches,
        cmap=colormap,
        norm=normalization,
        edgecolor="#ffffff",
        linewidth=0.15,
        antialiased=False,
    )
    collection.set_array(values)
    axis.add_collection(collection)
    axis.add_patch(
        PathPatch(
            _geometry_path(grid.study_area_equal_area),
            facecolor="none",
            edgecolor="#202020",
            linewidth=0.7,
            antialiased=True,
            zorder=3,
        )
    )

    minimum_x, minimum_y, maximum_x, maximum_y = (
        float(value) for value in grid.study_area_equal_area.bounds
    )
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    extent = max(width, height)
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("study-area render extent must be finite and positive")
    horizontal_padding = max(width * 0.02, extent * 0.002)
    vertical_padding = max(height * 0.02, extent * 0.002)
    axis.set_xlim(minimum_x - horizontal_padding, maximum_x + horizontal_padding)
    axis.set_ylim(minimum_y - vertical_padding, maximum_y + vertical_padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")
    axis.set_title(title, fontproperties=font, fontsize=11, pad=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return collection


def render_conditional_intensity_figure(
    grid: EqualAreaGrid,
    background_intensity: ArrayLike,
    trigger_intensity: ArrayLike,
    *,
    issue_date: str,
    model_variant: str,
    data_cutoff: str,
) -> ConditionalIntensityRender:
    """Render three immutable conditional-intensity panels entirely in memory."""

    if not isinstance(grid, EqualAreaGrid):
        raise TypeError("grid must be an EqualAreaGrid")
    issue_date_value = _canonical_issue_date(issue_date)
    model_variant_value = _canonical_model_variant(model_variant)
    data_cutoff_value = _canonical_data_cutoff(data_cutoff)
    background = _validated_intensity(
        "background_intensity",
        background_intensity,
        cell_count=len(grid.cells),
    )
    trigger = _validated_intensity(
        "trigger_intensity",
        trigger_intensity,
        cell_count=len(grid.cells),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        total = background + trigger
    if not bool(np.isfinite(total).all()):
        raise ValueError("background_intensity + trigger_intensity must remain finite")

    footer_label = (
        f"起报日: {issue_date_value}  |  模型变体: {model_variant_value}  |  "
        f"数据截止: {data_cutoff_value}"
    )
    png_metadata = (
        ("Software", "SeismoFlux deterministic renderer"),
        ("Title", "SeismoFlux conditional intensity"),
        ("Description", "三面板条件强度 (期望事件强度)"),
        ("DPI", str(FIGURE_DPI)),
        ("FontFamily", FONT_FAMILY),
        ("ColorScale", f"{COLORMAP_NAME} fixed blue-to-red 256"),
        ("IssueDate", issue_date_value),
        ("ModelVariant", model_variant_value),
        ("DataCutoff", data_cutoff_value),
        ("PanelTitles", " | ".join(PANEL_TITLES)),
        ("ColorbarLabel", COLORBAR_LABEL),
        ("TotalDefinition", "background_intensity + trigger_intensity (cellwise)"),
        ("Geometry", "EqualAreaGrid clipped cells and study-area boundary"),
    )

    maximum_total = float(np.max(total))
    display_scale = maximum_total if maximum_total > 0.0 else 1.0
    scaled_components = tuple(values / display_scale for values in (background, trigger, total))
    normalization = Normalize(vmin=0.0, vmax=1.0, clip=True)
    colormap = LinearSegmentedColormap.from_list(
        COLORMAP_NAME,
        _COLOR_STOPS,
        N=256,
    )
    font = FontProperties(family=FONT_FAMILY)
    figure = Figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=FIGURE_DPI,
        facecolor="white",
    )
    canvas = FigureCanvasAgg(figure)
    axes = tuple(figure.add_subplot(1, 3, index + 1) for index in range(3))
    collections = tuple(
        _add_intensity_panel(
            axis,
            grid=grid,
            values=values,
            title=title,
            colormap=colormap,
            normalization=normalization,
            font=font,
        )
        for axis, values, title in zip(
            axes,
            scaled_components,
            PANEL_TITLES,
            strict=True,
        )
    )
    figure.subplots_adjust(left=0.02, right=0.89, bottom=0.16, top=0.92, wspace=0.08)
    colorbar_axis = figure.add_axes((0.915, 0.21, 0.017, 0.60))
    colorbar = figure.colorbar(collections[-1], cax=colorbar_axis)
    tick_positions = (
        tuple(float(value) for value in np.linspace(0.0, 1.0, 6, dtype=np.float64))
        if maximum_total > 0.0
        else (0.0,)
    )
    colorbar.set_ticks(tick_positions)
    colorbar.set_ticklabels(tuple(f"{position * maximum_total:.3g}" for position in tick_positions))
    colorbar.set_label(COLORBAR_LABEL, fontproperties=font, fontsize=9, labelpad=9)
    colorbar.ax.tick_params(labelsize=8, width=0.6, length=3)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontproperties(font)
    figure.text(
        0.02,
        0.055,
        footer_label,
        fontproperties=font,
        fontsize=8.5,
        color="#303030",
        ha="left",
        va="center",
    )

    output = io.BytesIO()
    canvas.print_png(  # type: ignore[no-untyped-call]
        output,
        metadata=dict(png_metadata),
        pil_kwargs={"compress_level": 9, "optimize": False},
    )
    png_bytes = output.getvalue()
    figure.clear()
    return ConditionalIntensityRender(
        png_bytes=png_bytes,
        panel_titles=PANEL_TITLES,
        colorbar_label=COLORBAR_LABEL,
        footer_label=footer_label,
        png_metadata=png_metadata,
        width_px=FIGURE_WIDTH_PX,
        height_px=FIGURE_HEIGHT_PX,
        dpi=FIGURE_DPI,
        font_family=FONT_FAMILY,
        colormap_name=COLORMAP_NAME,
    )


def render_conditional_intensity_png(
    grid: EqualAreaGrid,
    background_intensity: ArrayLike,
    trigger_intensity: ArrayLike,
    *,
    issue_date: str,
    model_variant: str,
    data_cutoff: str,
) -> bytes:
    """Return only the deterministic PNG payload for publication code."""

    return render_conditional_intensity_figure(
        grid,
        background_intensity,
        trigger_intensity,
        issue_date=issue_date,
        model_variant=model_variant,
        data_cutoff=data_cutoff,
    ).png_bytes
