# ruff: noqa: E501, RUF001
"""Target-blind visualisation for a real P1 prospective forecast issue.

The public functions in this module are deliberately pure: they validate one
issue-time-only mapping and return deterministic SVG bytes or a self-contained
HTML string.  They never read a truth catalogue, fetch a URL, or write an
artifact.  The caller that owns the append-only prospective issue directory is
responsible for persisting the returned bytes exactly once.

The input contract contains one frozen support/grid shared by ``B0`` and
``B0_R30``.  Colour and ranking use normalized cell mass divided by the actual
clipped cell area.  The resulting values are conditional *relative intensity*,
not absolute earthquake probability.  Future outcomes are rejected before any
rendering work is performed.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeAlias, cast

_MODEL_IDS = ("B0", "B0_R30")
_CELL_SIZE_KM = 25.0
_FULL_CELL_AREA_KM2 = _CELL_SIZE_KM**2
_INITIAL_AREA_CAP_KM2 = 600_000.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ISSUE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_FUTURE_KEY_TOKENS = (
    "truth",
    "target",
    "cluster",
    "hit",
    "miss",
    "recall",
    "gain",
    "score",
    "outcome",
    "effect",
    "mature",
)

SupportStatus: TypeAlias = Literal["supported", "indeterminate", "unsupported"]


class ProductionRenderingError(ValueError):
    """Raised when a real issue visual input violates its scientific boundary."""


@dataclass(frozen=True, slots=True)
class ProductionCell:
    """One cell in the single support/grid shared by both forecast models."""

    cell_id: str
    row: int
    column: int
    area_km2: float
    support_status: SupportStatus


@dataclass(frozen=True, slots=True)
class ProductionModelView:
    """One issue-time model surface and its complete-cell alarm prefix."""

    model_id: Literal["B0", "B0_R30"]
    normalized_cell_mass: tuple[float, ...]
    relative_intensity_per_km2: tuple[float, ...]
    alarm_cell_ids: tuple[str, ...]
    actual_alarm_area_km2: float
    next_complete_cell_area_km2: float | None
    ranked_cell_ids: tuple[str, ...]
    rank_by_cell: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ProductionForecastView:
    """Validated, future-blind view of one real prospective forecast issue."""

    issue_id: str
    scheduled_issue_time_utc: str
    scheduled_issue_time: datetime
    query_cutoff_utc: str
    query_cutoff: datetime
    source_snapshot_sha256: str
    source_request_sha256: str
    support_manifest_sha256: str
    code_commit: str
    B0_source_count: int
    R30_source_count: int
    cells: tuple[ProductionCell, ...]
    models: tuple[ProductionModelView, ProductionModelView]
    minimum_row: int
    maximum_row: int
    minimum_column: int
    maximum_column: int
    shared_colour_maximum: float


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProductionRenderingError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ProductionRenderingError(f"{path} must be a sequence")
    return cast(Sequence[object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ProductionRenderingError(f"{path} fields differ: {'; '.join(details)}")


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProductionRenderingError(f"{path} must be a non-empty stripped string")
    return value


def _sha256(value: object, path: str) -> str:
    text = _text(value, path)
    if _SHA256_RE.fullmatch(text) is None:
        raise ProductionRenderingError(f"{path} must be a lowercase SHA-256")
    return text


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionRenderingError(f"{path} must be an integer")
    return value


def _number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProductionRenderingError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ProductionRenderingError(f"{path} must be finite and >= {minimum}")
    return result


def _utc_timestamp(value: object, path: str) -> tuple[str, datetime]:
    text = _text(value, path)
    if not text.endswith("Z"):
        raise ProductionRenderingError(f"{path} must use an explicit UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ProductionRenderingError(f"{path} is not an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProductionRenderingError(f"{path} must be UTC")
    return text, parsed.astimezone(UTC)


def _reject_future_fields(value: object, path: str = "forecast") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ProductionRenderingError(f"{path} contains a non-string key")
            key = raw_key.casefold().replace("-", "_")
            token = next((item for item in _FUTURE_KEY_TOKENS if item in key), None)
            if token is not None:
                raise ProductionRenderingError(
                    f"{path}.{raw_key} is a forbidden future-outcome field ({token})"
                )
            _reject_future_fields(child, f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _reject_future_fields(child, f"{path}[{index}]")


def _normalise_cells(value: object) -> tuple[ProductionCell, ...]:
    grid = _mapping(value, "grid")
    _exact_keys(grid, {"cell_size_km", "cells"}, "grid")
    cell_size = _number(grid.get("cell_size_km"), "grid.cell_size_km", minimum=1.0)
    if not math.isclose(cell_size, _CELL_SIZE_KM, rel_tol=0.0, abs_tol=1e-12):
        raise ProductionRenderingError("grid.cell_size_km must remain frozen at 25")
    cells: list[ProductionCell] = []
    for index, item in enumerate(_sequence(grid.get("cells"), "grid.cells")):
        path = f"grid.cells[{index}]"
        raw = _mapping(item, path)
        _exact_keys(raw, {"cell_id", "row", "column", "area_km2", "support_status"}, path)
        status = _text(raw.get("support_status"), f"{path}.support_status")
        if status not in {"supported", "indeterminate", "unsupported"}:
            raise ProductionRenderingError(f"{path}.support_status is invalid")
        area = _number(raw.get("area_km2"), f"{path}.area_km2", minimum=1e-12)
        if area > _FULL_CELL_AREA_KM2 + 1e-9:
            raise ProductionRenderingError(f"{path}.area_km2 exceeds one complete 25 km cell")
        cells.append(
            ProductionCell(
                cell_id=_text(raw.get("cell_id"), f"{path}.cell_id"),
                row=_integer(raw.get("row"), f"{path}.row"),
                column=_integer(raw.get("column"), f"{path}.column"),
                area_km2=area,
                support_status=cast(SupportStatus, status),
            )
        )
    if not cells:
        raise ProductionRenderingError("grid.cells must not be empty")
    canonical = tuple(
        sorted(cells, key=lambda item: (item.row, item.column, item.cell_id.encode("utf-8")))
    )
    if tuple(cells) != canonical:
        raise ProductionRenderingError("grid.cells must use canonical row/column/cell_id order")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ProductionRenderingError("grid cell IDs must be unique")
    if len({(cell.row, cell.column) for cell in cells}) != len(cells):
        raise ProductionRenderingError("grid row/column positions must be unique")
    return tuple(cells)


def _largest_prefix(
    ranked_cells: tuple[ProductionCell, ...], *, maximum_area_km2: float
) -> tuple[ProductionCell, ...]:
    selected: list[ProductionCell] = []
    selected_areas: list[float] = []
    tolerance = max(1e-8, maximum_area_km2 * 1e-12)
    for cell in ranked_cells:
        candidate = math.fsum((*selected_areas, cell.area_km2))
        if candidate > maximum_area_km2 + tolerance:
            break
        selected.append(cell)
        selected_areas.append(cell.area_km2)
    return tuple(selected)


def _normalise_model(
    value: object,
    *,
    model_id: Literal["B0", "B0_R30"],
    cells: tuple[ProductionCell, ...],
    maximum_area_km2: float,
) -> ProductionModelView:
    path = f"models.{model_id}"
    raw = _mapping(value, path)
    _exact_keys(
        raw,
        {
            "normalized_cell_mass",
            "alarm_cell_ids",
            "actual_alarm_area_km2",
            "next_complete_cell_area_km2",
        },
        path,
    )
    mass = tuple(
        _number(item, f"{path}.normalized_cell_mass[{index}]")
        for index, item in enumerate(
            _sequence(raw.get("normalized_cell_mass"), f"{path}.normalized_cell_mass")
        )
    )
    if len(mass) != len(cells):
        raise ProductionRenderingError(f"{path}.normalized_cell_mass must align with grid.cells")
    if not math.isclose(math.fsum(mass), 1.0, rel_tol=1e-8, abs_tol=1e-10):
        raise ProductionRenderingError(f"{path}.normalized_cell_mass must sum to one")
    if any(
        value > 1e-15 and cell.support_status == "unsupported"
        for cell, value in zip(cells, mass, strict=True)
    ):
        raise ProductionRenderingError(f"{path} assigns mass outside the frozen support")
    intensity = tuple(value / cell.area_km2 for cell, value in zip(cells, mass, strict=True))
    intensity_by_id = {cell.cell_id: value for cell, value in zip(cells, intensity, strict=True)}
    ranked_cells = tuple(
        sorted(
            cells,
            key=lambda cell: (
                -intensity_by_id[cell.cell_id],
                cell.row,
                cell.column,
                cell.cell_id.encode("utf-8"),
            ),
        )
    )
    alarm_ids = tuple(
        _text(item, f"{path}.alarm_cell_ids[{index}]")
        for index, item in enumerate(_sequence(raw.get("alarm_cell_ids"), f"{path}.alarm_cell_ids"))
    )
    if len(set(alarm_ids)) != len(alarm_ids):
        raise ProductionRenderingError(f"{path}.alarm_cell_ids contains duplicates")
    expected_prefix = _largest_prefix(ranked_cells, maximum_area_km2=maximum_area_km2)
    expected_ids = tuple(cell.cell_id for cell in expected_prefix)
    if alarm_ids != expected_ids:
        raise ProductionRenderingError(
            f"{path}.alarm_cell_ids must be the unmodified largest complete-cell ranking prefix"
        )
    actual_area = _number(raw.get("actual_alarm_area_km2"), f"{path}.actual_alarm_area_km2")
    recomputed_area = math.fsum(cell.area_km2 for cell in expected_prefix)
    tolerance = max(1e-8, recomputed_area * 1e-10)
    if not math.isclose(actual_area, recomputed_area, rel_tol=0.0, abs_tol=tolerance):
        raise ProductionRenderingError(
            f"{path}.actual_alarm_area_km2 disagrees with selected cells"
        )
    expected_next = (
        ranked_cells[len(expected_prefix)].area_km2
        if len(expected_prefix) < len(ranked_cells)
        else None
    )
    raw_next = raw.get("next_complete_cell_area_km2")
    next_area = (
        None
        if raw_next is None
        else _number(raw_next, f"{path}.next_complete_cell_area_km2", minimum=1e-12)
    )
    if expected_next is None:
        if next_area is not None:
            raise ProductionRenderingError(f"{path}.next_complete_cell_area_km2 must be null")
    elif next_area is None or not math.isclose(next_area, expected_next, rel_tol=0.0, abs_tol=1e-8):
        raise ProductionRenderingError(
            f"{path}.next_complete_cell_area_km2 disagrees with the stable ranking"
        )
    ranking = tuple(cell.cell_id for cell in ranked_cells)
    return ProductionModelView(
        model_id=model_id,
        normalized_cell_mass=mass,
        relative_intensity_per_km2=intensity,
        alarm_cell_ids=alarm_ids,
        actual_alarm_area_km2=actual_area,
        next_complete_cell_area_km2=next_area,
        ranked_cell_ids=ranking,
        rank_by_cell={cell_id: rank for rank, cell_id in enumerate(ranking, start=1)},
    )


def parse_production_forecast_view(
    forecast_mapping: Mapping[str, object],
) -> ProductionForecastView:
    """Validate and normalize one real, issue-time-only forecast view."""

    raw = _mapping(forecast_mapping, "forecast")
    _reject_future_fields(raw)
    _exact_keys(
        raw,
        {
            "issue_id",
            "scheduled_issue_time_utc",
            "query_cutoff_utc",
            "source_snapshot_sha256",
            "source_request_sha256",
            "support_manifest_sha256",
            "code_commit",
            "B0_source_count",
            "R30_source_count",
            "grid",
            "models",
        },
        "forecast",
    )
    issue_id = _text(raw.get("issue_id"), "forecast.issue_id")
    if _ISSUE_ID_RE.fullmatch(issue_id) is None:
        raise ProductionRenderingError(
            "forecast.issue_id contains unsafe or non-canonical characters"
        )
    scheduled_text, scheduled = _utc_timestamp(
        raw.get("scheduled_issue_time_utc"), "forecast.scheduled_issue_time_utc"
    )
    cutoff_text, cutoff = _utc_timestamp(raw.get("query_cutoff_utc"), "forecast.query_cutoff_utc")
    if scheduled - cutoff != timedelta(minutes=15):
        raise ProductionRenderingError("query cutoff Q must be exactly 15 minutes before T")
    expected_issue_id = f"p1-{scheduled.strftime('%Y%m%dT%H%M%SZ')}"
    if issue_id != expected_issue_id:
        raise ProductionRenderingError("forecast.issue_id does not match scheduled issue T")
    cells = _normalise_cells(raw.get("grid"))
    models_raw = _mapping(raw.get("models"), "models")
    _exact_keys(models_raw, set(_MODEL_IDS), "models")
    b0 = _normalise_model(
        models_raw.get("B0"),
        model_id="B0",
        cells=cells,
        maximum_area_km2=_INITIAL_AREA_CAP_KM2,
    )
    challenger = _normalise_model(
        models_raw.get("B0_R30"),
        model_id="B0_R30",
        cells=cells,
        maximum_area_km2=b0.actual_alarm_area_km2,
    )
    area_difference = b0.actual_alarm_area_km2 - challenger.actual_alarm_area_km2
    if area_difference < -1e-8 or area_difference >= _FULL_CELL_AREA_KM2 - 1e-8:
        raise ProductionRenderingError("paired actual alarm areas violate the frozen fairness rule")
    if (
        challenger.next_complete_cell_area_km2 is not None
        and area_difference >= challenger.next_complete_cell_area_km2 - 1e-8
        and not math.isclose(area_difference, 0.0, abs_tol=1e-8)
    ):
        raise ProductionRenderingError("B0_R30 leaves enough area for its next complete cell")
    code_commit = _text(raw.get("code_commit"), "forecast.code_commit")
    if _COMMIT_RE.fullmatch(code_commit) is None:
        raise ProductionRenderingError(
            "forecast.code_commit must be a lowercase 40-character commit"
        )
    B0_source_count = _integer(raw.get("B0_source_count"), "forecast.B0_source_count")
    R30_source_count = _integer(raw.get("R30_source_count"), "forecast.R30_source_count")
    if B0_source_count < 1:
        raise ProductionRenderingError("forecast.B0_source_count must be positive")
    if not 0 <= R30_source_count <= B0_source_count:
        raise ProductionRenderingError("forecast.R30_source_count must be in [0, B0_source_count]")
    shared_maximum = max((*b0.relative_intensity_per_km2, *challenger.relative_intensity_per_km2))
    if not math.isfinite(shared_maximum) or shared_maximum <= 0.0:
        raise ProductionRenderingError("shared relative-intensity colour scale must be positive")
    return ProductionForecastView(
        issue_id=issue_id,
        scheduled_issue_time_utc=scheduled_text,
        scheduled_issue_time=scheduled,
        query_cutoff_utc=cutoff_text,
        query_cutoff=cutoff,
        source_snapshot_sha256=_sha256(
            raw.get("source_snapshot_sha256"), "forecast.source_snapshot_sha256"
        ),
        source_request_sha256=_sha256(
            raw.get("source_request_sha256"), "forecast.source_request_sha256"
        ),
        support_manifest_sha256=_sha256(
            raw.get("support_manifest_sha256"), "forecast.support_manifest_sha256"
        ),
        code_commit=code_commit,
        B0_source_count=B0_source_count,
        R30_source_count=R30_source_count,
        cells=cells,
        models=(b0, challenger),
        minimum_row=min(cell.row for cell in cells),
        maximum_row=max(cell.row for cell in cells),
        minimum_column=min(cell.column for cell in cells),
        maximum_column=max(cell.column for cell in cells),
        shared_colour_maximum=shared_maximum,
    )


def _format_number(value: float, *, decimals: int = 3) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, abs_tol=1e-9):
        return f"{rounded:,}"
    return f"{value:,.{decimals}f}"


def _colour(value: float, maximum: float) -> str:
    ratio = 0.0 if maximum <= 0.0 else min(1.0, max(0.0, value / maximum))
    low = (238, 246, 255)
    high = (30, 92, 157)
    rgb = tuple(round(start + ratio * (end - start)) for start, end in zip(low, high, strict=True))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _panel_geometry(
    forecast: ProductionForecastView, *, width: float, height: float
) -> tuple[float, float, float]:
    columns = forecast.maximum_column - forecast.minimum_column + 1
    rows = forecast.maximum_row - forecast.minimum_row + 1
    available_width = width - 170.0
    available_height = height - 66.0
    scale = min(available_width / columns, available_height / rows)
    return 18.0, 48.0 + (available_height - rows * scale) / 2.0, scale


def _forecast_panel_svg(
    forecast: ProductionForecastView,
    model: ProductionModelView,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    grid_x, grid_y, scale = _panel_geometry(forecast, width=width, height=height)
    alarms = set(model.alarm_cell_ids)
    parts = [
        f'<g data-model="{model.model_id}" transform="translate({x:.1f},{y:.1f})">',
        f'<rect width="{width:.1f}" height="{height:.1f}" rx="12" fill="#fff" stroke="#ccd6e1"/>',
        f'<text x="18" y="30" class="model">{model.model_id}</text>',
        '<g data-layer="relative-intensity" data-value-semantics="relative-intensity-not-probability">',
    ]
    for index, cell in enumerate(forecast.cells):
        cell_x = grid_x + (cell.column - forecast.minimum_column) * scale
        # Grid rows increase northward, while SVG y increases downward.
        cell_y = grid_y + (forecast.maximum_row - cell.row) * scale
        alarm = cell.cell_id in alarms
        dash = (
            ' stroke-dasharray="3,2"'
            if cell.support_status == "indeterminate" and not alarm
            else ""
        )
        fill = _colour(model.relative_intensity_per_km2[index], forecast.shared_colour_maximum)
        parts.append(
            f'<rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{scale:.2f}" height="{scale:.2f}" '
            f'fill="{fill}" stroke="{"#e39400" if alarm else "#c8d3df"}" '
            f'stroke-width="{2.8 if alarm else 0.8:.1f}"{dash} data-cell-id="{html.escape(cell.cell_id)}" '
            f'data-alarm="{str(alarm).lower()}" data-support-status="{cell.support_status}">'
            f"<title>格 {html.escape(cell.cell_id)}；单位面积相对强度 {model.relative_intensity_per_km2[index]:.10g}；"
            f"顺位 {model.rank_by_cell[cell.cell_id]}；实际面积 {cell.area_km2:.6g} km²；"
            f"{'报警格' if alarm else '非报警格'}；support={cell.support_status}</title></rect>"
        )
    parts.extend(
        [
            "</g>",
            f'<text x="{width - 135:.1f}" y="88" class="metric-label">实际报警面积</text>',
            f'<text x="{width - 135:.1f}" y="114" class="metric">{_format_number(model.actual_alarm_area_km2)} km²</text>',
            f'<text x="{width - 135:.1f}" y="154" class="metric-label">完整报警格</text>',
            f'<text x="{width - 135:.1f}" y="180" class="metric">{len(model.alarm_cell_ids):,} 个</text>',
            f'<text x="{width - 135:.1f}" y="220" class="metric-label">数值含义</text>',
            f'<text x="{width - 135:.1f}" y="246" class="note">相对强度</text>',
            f'<text x="{width - 135:.1f}" y="266" class="note">不是概率</text>',
            "</g>",
        ]
    )
    return parts


def render_production_forecast_svg(forecast_mapping: Mapping[str, object]) -> bytes:
    """Render a deterministic static map for one real prospective issue."""

    forecast = parse_production_forecast_view(forecast_mapping)
    area_difference = (
        forecast.models[0].actual_alarm_area_km2 - forecast.models[1].actual_alarm_area_km2
    )
    width = 1400
    height = 860
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" data-artifact="p1-real-prospective-forecast" data-source-snapshot-sha256="{forecast.source_snapshot_sha256}" data-shared-colour-max="{forecast.shared_colour_maximum:.12g}">',
        '<title id="title">P1 真实前瞻起报：B0 与 B0_R30</title>',
        '<desc id="desc">只展示查询截止时已冻结的相对强度和报警格，不含未来地震，不能据此判断预测效果。</desc>',
        "<style>",
        "text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#17243a}",
        ".title{font-size:28px;font-weight:700}.warning{font-size:16px;font-weight:700;fill:#a33226}",
        ".subtitle{font-size:13px;fill:#506176}.model{font-size:19px;font-weight:700}",
        ".metric-label{font-size:12px;fill:#65758a}.metric{font-size:15px;font-weight:650}.note{font-size:12px;fill:#506176}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        '<text x="40" y="42" class="title">P1 真实前瞻起报：同一支持、同一报警面积规则</text>',
        '<text x="40" y="70" class="warning">这是事前风险排序，不是效果结论；图中没有未来地震或命中信息</text>',
        f'<text x="40" y="98" class="subtitle">Issue = {html.escape(forecast.issue_id)}；T = {html.escape(forecast.scheduled_issue_time_utc)}；Q = {html.escape(forecast.query_cutoff_utc)}</text>',
        '<text x="40" y="122" class="subtitle">75 km Gaussian KDE；B0_R30 = 0.75 × B0 + 0.25 × R30；双图共享同一单位面积相对强度色标。</text>',
        f'<text x="40" y="146" class="subtitle">长期目录入选数 {forecast.B0_source_count:,}；最近30天M4+入选数 {forecast.R30_source_count:,}。</text>',
        f'<text x="40" y="170" class="subtitle">source snapshot SHA-256 = {forecast.source_snapshot_sha256}</text>',
        f'<text x="40" y="194" class="subtitle">source request SHA-256 = {forecast.source_request_sha256}</text>',
    ]
    parts.extend(
        _forecast_panel_svg(
            forecast,
            forecast.models[0],
            x=40.0,
            y=218.0,
            width=640.0,
            height=466.0,
        )
    )
    parts.extend(
        _forecast_panel_svg(
            forecast,
            forecast.models[1],
            x=720.0,
            y=218.0,
            width=640.0,
            height=466.0,
        )
    )
    parts.extend(
        [
            f'<text x="40" y="722" class="note">面积公平：B0_R30 实际面积不超过 B0；差值 {_format_number(area_difference)} km²；B0 初始上限 {_format_number(_INITIAL_AREA_CAP_KM2, decimals=0)} km²。</text>',
            f'<text x="40" y="746" class="note">support manifest SHA-256 = {forecast.support_manifest_sha256}</text>',
            f'<text x="40" y="770" class="note">code commit = {forecast.code_commit}</text>',
            '<text x="40" y="802" class="note">颜色只表示各模型内部的条件相对强弱；相对强度、评分和顺位均不是绝对发震概率。</text>',
            '<text x="40" y="826" class="note">预测效果必须等未来评价窗成熟后另行检验，任何未来结果都不得回写本起报图。</text>',
            "</svg>",
        ]
    )
    return ("\n".join(parts) + "\n").encode("utf-8")


def _forecast_json(forecast: ProductionForecastView) -> dict[str, object]:
    return {
        "issue_id": forecast.issue_id,
        "scheduled_issue_time_utc": forecast.scheduled_issue_time_utc,
        "query_cutoff_utc": forecast.query_cutoff_utc,
        "source_snapshot_sha256": forecast.source_snapshot_sha256,
        "source_request_sha256": forecast.source_request_sha256,
        "support_manifest_sha256": forecast.support_manifest_sha256,
        "code_commit": forecast.code_commit,
        "B0_source_count": forecast.B0_source_count,
        "R30_source_count": forecast.R30_source_count,
        "minimum_row": forecast.minimum_row,
        "maximum_row": forecast.maximum_row,
        "minimum_column": forecast.minimum_column,
        "maximum_column": forecast.maximum_column,
        "shared_colour_maximum": forecast.shared_colour_maximum,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "row": cell.row,
                "column": cell.column,
                "area_km2": cell.area_km2,
                "support_status": cell.support_status,
            }
            for cell in forecast.cells
        ],
        "models": {
            model.model_id: {
                "normalized_cell_mass": list(model.normalized_cell_mass),
                "relative_intensity_per_km2": list(model.relative_intensity_per_km2),
                "alarm_cell_ids": list(model.alarm_cell_ids),
                "actual_alarm_area_km2": model.actual_alarm_area_km2,
                "next_complete_cell_area_km2": model.next_complete_cell_area_km2,
                "rank_by_cell": dict(model.rank_by_cell),
            }
            for model in forecast.models
        },
    }


def _safe_json(value: object) -> str:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_offline_production_forecast_html(
    forecast_mapping: Mapping[str, object],
) -> str:
    """Build a deterministic, fully offline two-map issue-time explorer."""

    forecast = parse_production_forecast_view(forecast_mapping)
    payload = _safe_json(_forecast_json(forecast))
    area_difference = (
        forecast.models[0].actual_alarm_area_km2 - forecast.models[1].actual_alarm_area_km2
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1 真实前瞻起报：B0 与 B0_R30</title>
<style>
:root{{--ink:#17243a;--muted:#5c6d82;--paper:#fff;--bg:#eef3f8;--line:#cbd7e4;--alarm:#e39400}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:24px}}header,.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}
h1{{font-size:27px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 10px}}p{{line-height:1.55;margin:7px 0}}.warning{{color:#a33226;font-weight:700}}.muted{{color:var(--muted)}}
.controls{{display:flex;gap:20px;align-items:center;flex-wrap:wrap}}.toggle{{display:flex;gap:7px;align-items:center;font-weight:650}}
.maps{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}canvas{{display:block;width:100%;height:auto;background:#f9fbfd;border:1px solid var(--line);border-radius:8px}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0}}.metric{{background:#f4f7fb;border-radius:8px;padding:11px}}.metric strong{{display:block;font-size:19px;margin-top:4px}}
.inspector{{margin-top:10px;padding:10px;min-height:44px;background:#eef4fa;border-radius:7px;font-family:ui-monospace,Consolas,monospace}}code{{overflow-wrap:anywhere}}@media(max-width:850px){{.maps,.metrics{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main data-artifact="p1-real-prospective-forecast-offline" data-source-snapshot-sha256="{forecast.source_snapshot_sha256}">
<header><h1>P1 真实前瞻起报：同一支持、同一报警面积规则</h1><p class="warning">这是事前风险排序，不是预测效果结论；本页不含未来地震、命中或召回信息。</p><p>Issue = <code>{html.escape(forecast.issue_id)}</code> · T = <code>{html.escape(forecast.scheduled_issue_time_utc)}</code> · Q = <code>{html.escape(forecast.query_cutoff_utc)}</code></p><p>长期目录入选数 <strong>{forecast.B0_source_count:,}</strong>；最近30天M4+入选数 <strong>{forecast.R30_source_count:,}</strong>。</p><p>75 km Gaussian KDE；<code>B0_R30 = 0.75 × B0 + 0.25 × R30</code>。双图共享同一固定 support、格网与单位面积相对强度色标；相对强度与顺位不是绝对发震概率。</p></header>
<section class="card controls" aria-label="起报图控件"><label class="toggle" for="intensity-toggle"><input id="intensity-toggle" type="checkbox" checked>显示单位面积相对强度</label><label class="toggle" for="alarm-toggle"><input id="alarm-toggle" type="checkbox" checked>显示报警格</label><label class="toggle" for="support-toggle"><input id="support-toggle" type="checkbox" checked>标示 support 状态</label></section>
<section class="maps">
<article class="card"><h2>B0：长期背景</h2><canvas id="grid-B0" width="720" height="590" data-model="B0" data-layer-relative-intensity="toggle" data-layer-alarm="toggle" data-layer-support="toggle"></canvas><div class="metrics"><div class="metric">实际报警面积<strong>{_format_number(forecast.models[0].actual_alarm_area_km2)} km²</strong></div><div class="metric">完整报警格<strong>{len(forecast.models[0].alarm_cell_ids):,}</strong></div><div class="metric">数值含义<strong>相对强度</strong></div></div><div class="inspector" id="inspector-B0" aria-live="polite">移动或点击格网查看 B0 起报值。</div></article>
<article class="card"><h2>B0_R30：加入近 30 天活动</h2><canvas id="grid-B0_R30" width="720" height="590" data-model="B0_R30" data-layer-relative-intensity="toggle" data-layer-alarm="toggle" data-layer-support="toggle"></canvas><div class="metrics"><div class="metric">实际报警面积<strong>{_format_number(forecast.models[1].actual_alarm_area_km2)} km²</strong></div><div class="metric">完整报警格<strong>{len(forecast.models[1].alarm_cell_ids):,}</strong></div><div class="metric">数值含义<strong>相对强度</strong></div></div><div class="inspector" id="inspector-B0_R30" aria-live="polite">移动或点击格网查看 B0_R30 起报值。</div></article>
</section>
<section class="card"><h2>科学边界与冻结身份</h2><p>面积公平：B0_R30 不超过 B0；实际面积差 {_format_number(area_difference)} km²；B0 初始上限 {_format_number(_INITIAL_AREA_CAP_KM2, decimals=0)} km²。报警格都是各自原始顺位下不跳格的最大完整前缀。</p><p>source snapshot SHA-256：<code>{forecast.source_snapshot_sha256}</code><br>source request SHA-256：<code>{forecast.source_request_sha256}</code><br>support manifest SHA-256：<code>{forecast.support_manifest_sha256}</code><br>code commit：<code>{forecast.code_commit}</code></p><p class="muted">本页只保存 Q 时已经可用的输入及其预测图层。未来评价窗成熟后必须另存回放，不能回写本页；因此当前图不能说明 B0 或 B0_R30 哪个预测效果更好。</p></section>
<script type="application/json" id="forecast-data">{payload}</script>
<script>
"use strict";
const forecast=JSON.parse(document.getElementById("forecast-data").textContent),intensityToggle=document.getElementById("intensity-toggle"),alarmToggle=document.getElementById("alarm-toggle"),supportToggle=document.getElementById("support-toggle"),views={{}};
function fmt(value){{return new Intl.NumberFormat("zh-CN",{{maximumFractionDigits:3}}).format(value)}}
function colour(value){{const ratio=forecast.shared_colour_maximum>0?Math.max(0,Math.min(1,value/forecast.shared_colour_maximum)):0,low=[238,246,255],high=[30,92,157];return "rgb("+low.map((item,index)=>Math.round(item+ratio*(high[index]-item))).join(",")+")"}}
function render(modelId){{const canvas=document.getElementById("grid-"+modelId),context=canvas.getContext("2d"),model=forecast.models[modelId],alarms=new Set(model.alarm_cell_ids),columns=forecast.maximum_column-forecast.minimum_column+1,rows=forecast.maximum_row-forecast.minimum_row+1,padding=34,scale=Math.min((canvas.width-padding*2)/columns,(canvas.height-padding*2)/rows),originX=(canvas.width-columns*scale)/2,originY=(canvas.height-rows*scale)/2;views[modelId]={{model,alarms,scale,originX,originY}};context.clearRect(0,0,canvas.width,canvas.height);context.fillStyle="#f9fbfd";context.fillRect(0,0,canvas.width,canvas.height);forecast.cells.forEach((cell,index)=>{{const x=originX+(cell.column-forecast.minimum_column)*scale,y=originY+(forecast.maximum_row-cell.row)*scale,isAlarm=alarms.has(cell.cell_id);context.fillStyle=intensityToggle.checked?colour(model.relative_intensity_per_km2[index]):"#fff";context.fillRect(x,y,scale,scale);context.setLineDash(supportToggle.checked&&cell.support_status==="indeterminate"?[3,2]:[]);context.strokeStyle=alarmToggle.checked&&isAlarm?"#e39400":"#c8d3df";context.lineWidth=alarmToggle.checked&&isAlarm?2.5:1;context.strokeRect(x,y,scale,scale);context.setLineDash([])}})}}
function inspect(modelId,event){{const view=views[modelId],canvas=document.getElementById("grid-"+modelId);if(!view)return;const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height,column=Math.floor((x-view.originX)/view.scale)+forecast.minimum_column,row=forecast.maximum_row-Math.floor((y-view.originY)/view.scale),cell=forecast.cells.find(item=>item.row===row&&item.column===column);if(!cell)return;const index=forecast.cells.indexOf(cell),value=view.model.relative_intensity_per_km2[index],rank=view.model.rank_by_cell[cell.cell_id],alarm=view.alarms.has(cell.cell_id);document.getElementById("inspector-"+modelId).textContent="cell_id="+cell.cell_id+" · relative_intensity_per_km2="+value.toExponential(6)+" · rank="+rank+" · area="+fmt(cell.area_km2)+" km² · alarm="+(alarm?"是":"否")+" · support="+cell.support_status}}
for(const modelId of ["B0","B0_R30"]){{const canvas=document.getElementById("grid-"+modelId);canvas.addEventListener("mousemove",event=>inspect(modelId,event));canvas.addEventListener("click",event=>inspect(modelId,event));render(modelId)}}
for(const toggle of [intensityToggle,alarmToggle,supportToggle]){{toggle.addEventListener("change",()=>{{render("B0");render("B0_R30")}})}}
</script>
</main>
</body>
</html>
"""


__all__ = [
    "ProductionCell",
    "ProductionForecastView",
    "ProductionModelView",
    "ProductionRenderingError",
    "build_offline_production_forecast_html",
    "parse_production_forecast_view",
    "render_production_forecast_svg",
]
