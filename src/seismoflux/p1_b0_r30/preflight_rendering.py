# ruff: noqa: E501, RUF001
"""Target-blind P1-0C preflight visualisation.

The forecast renderers in this module accept a deliberately small mapping
whose schema contains issue-time inputs only.  They fail closed when any
future-outcome field is present.  Mature, synthetic known-answer observations
are accepted only by the separate replay renderers and are bound to the exact
SHA-256 of the forecast SVG they annotate.

The functions are pure: they return bytes or text and never write or replace a
forecast artifact on disk.  Add-only persistence is therefore left to the
caller that owns the immutable artifact directory.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeAlias, cast

from seismoflux.d1_replay.spatial import Frozen25kmCellLocator
from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.core import (
    DualModelForecast,
    build_pending_sequential_reviews,
    elapsed_tropical_months,
    locate_point_cell,
)
from seismoflux.p1_b0_r30.preimage import recompute_mature_truth_snapshot

_MODEL_IDS = ("B0", "B0_R30")
_AREA_CAP_KM2 = 600_000.0
_CELL_SIZE_KM = 25.0
_FULL_CELL_AREA_KM2 = _CELL_SIZE_KM**2
_BANDWIDTH_KM = 75.0
_RECENT_WEIGHT = 0.25
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FUTURE_KEY_TOKENS = (
    "truth",
    "target",
    "cluster",
    "hit",
    "miss",
    "recall",
    "gain",
    "review",
    "outcome",
    "score",
    "effect",
)

SupportStatus: TypeAlias = Literal["supported", "indeterminate", "unsupported"]


class P1PreflightRenderingError(ValueError):
    """Raised when a preflight visual input violates its scientific contract."""


@dataclass(frozen=True, slots=True)
class _CatalogWaterline:
    path: str
    sha256: str
    row_count: int
    eligible_B0_event_count: int
    R30_event_count: int
    origin_time_max_utc: str
    available_at_max_utc: str


@dataclass(frozen=True, slots=True)
class _SupportWaterline:
    support_id: str
    manifest_sha256: str
    fit_end_utc: str
    common_mc: float
    fixed_cell_count: int
    supported_cell_count: int
    indeterminate_cell_count: int
    unsupported_cell_count: int
    retained_area_km2: float


@dataclass(frozen=True, slots=True)
class _Cell:
    cell_id: str
    row: int
    column: int
    area_km2: float
    support_status: SupportStatus


@dataclass(frozen=True, slots=True)
class _Model:
    model_id: str
    normalized_cell_mass: tuple[float, ...]
    relative_intensity_per_km2: tuple[float, ...]
    alarm_cell_ids: tuple[str, ...]
    actual_alarm_area_km2: float
    next_complete_cell_area_km2: float | None
    ranked_cell_ids: tuple[str, ...]
    rank_by_cell: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _Forecast:
    rehearsal_id: str
    scheduled_issue_time_utc: str
    scheduled_issue_time: datetime
    query_cutoff_utc: str
    query_cutoff: datetime
    catalog: _CatalogWaterline
    support: _SupportWaterline
    cells: tuple[_Cell, ...]
    models: tuple[_Model, _Model]
    minimum_row: int
    maximum_row: int
    minimum_column: int
    maximum_column: int
    shared_colour_maximum: float


@dataclass(frozen=True, slots=True)
class _Cluster:
    cluster_id: str
    representative_event_id: str
    cell_id: str
    origin_time_utc: str


@dataclass(frozen=True, slots=True)
class _Replay:
    replay_id: str
    forecast_sha256: str
    forecast: _Forecast
    synthetic_raw_response_sha256: str
    synthetic_raw_response_byte_count: int
    cluster_assignment_sha256: str
    ordered_cluster_registry_sha256: str
    horizon_days: int
    mature_after_utc: str
    replay_created_at_utc: str
    clusters: tuple[_Cluster, ...]
    review: Mapping[str, object]


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P1PreflightRenderingError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise P1PreflightRenderingError(f"{path} must be a sequence")
    return cast(Sequence[object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise P1PreflightRenderingError(
            f"{path} keys differ from the frozen schema; missing={missing}, extra={extra}"
        )


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P1PreflightRenderingError(f"{path} must be a non-empty string")
    return value


def _sha256(value: object, path: str) -> str:
    text = _text(value, path)
    if _SHA256_RE.fullmatch(text) is None:
        raise P1PreflightRenderingError(f"{path} must be a lowercase SHA-256")
    return text


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise P1PreflightRenderingError(f"{path} must be an integer >= {minimum}")
    return value


def _signed_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise P1PreflightRenderingError(f"{path} must be an integer")
    return value


def _number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise P1PreflightRenderingError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise P1PreflightRenderingError(f"{path} must be finite and >= {minimum:g}")
    return result


def _utc_timestamp(value: object, path: str) -> tuple[str, datetime]:
    text = _text(value, path)
    if not text.endswith("Z"):
        raise P1PreflightRenderingError(f"{path} must use a canonical UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise P1PreflightRenderingError(f"{path} is not a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise P1PreflightRenderingError(f"{path} must be UTC")
    canonical = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if canonical != text:
        raise P1PreflightRenderingError(f"{path} must use canonical ISO-8601 form")
    return text, parsed.astimezone(UTC)


def _reject_future_fields(value: object, path: str = "forecast") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise P1PreflightRenderingError(f"{path} contains a non-string key")
            key = raw_key.casefold().replace("-", "_")
            token = next((item for item in _FUTURE_KEY_TOKENS if item in key), None)
            if token is not None:
                raise P1PreflightRenderingError(
                    f"{path}.{raw_key} is a forbidden future-outcome field ({token})"
                )
            _reject_future_fields(child, f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _reject_future_fields(child, f"{path}[{index}]")


def _normalise_catalog(value: object, *, query_cutoff: datetime) -> _CatalogWaterline:
    raw = _mapping(value, "catalog")
    _exact_keys(
        raw,
        {
            "path",
            "sha256",
            "row_count",
            "eligible_B0_event_count",
            "R30_event_count",
            "origin_time_max_utc",
            "available_at_max_utc",
        },
        "catalog",
    )
    row_count = _integer(raw.get("row_count"), "catalog.row_count", minimum=1)
    B0_count = _integer(
        raw.get("eligible_B0_event_count"),
        "catalog.eligible_B0_event_count",
        minimum=1,
    )
    R30_count = _integer(raw.get("R30_event_count"), "catalog.R30_event_count")
    if not 0 <= R30_count <= B0_count <= row_count:
        raise P1PreflightRenderingError(
            "catalog counts must satisfy 0 <= R30 <= eligible B0 <= all rows"
        )
    origin_text, origin_max = _utc_timestamp(
        raw.get("origin_time_max_utc"), "catalog.origin_time_max_utc"
    )
    available_text, available_max = _utc_timestamp(
        raw.get("available_at_max_utc"), "catalog.available_at_max_utc"
    )
    if origin_max > query_cutoff or available_max > query_cutoff:
        raise P1PreflightRenderingError(
            "catalog origin/availability waterlines must not exceed query cutoff Q"
        )
    return _CatalogWaterline(
        path=_text(raw.get("path"), "catalog.path"),
        sha256=_sha256(raw.get("sha256"), "catalog.sha256"),
        row_count=row_count,
        eligible_B0_event_count=B0_count,
        R30_event_count=R30_count,
        origin_time_max_utc=origin_text,
        available_at_max_utc=available_text,
    )


def _normalise_support(value: object, *, query_cutoff: datetime) -> _SupportWaterline:
    raw = _mapping(value, "support")
    _exact_keys(
        raw,
        {
            "support_id",
            "manifest_sha256",
            "fit_end_utc",
            "common_mc",
            "fixed_cell_count",
            "supported_cell_count",
            "indeterminate_cell_count",
            "unsupported_cell_count",
            "retained_area_km2",
        },
        "support",
    )
    fit_text, fit_end = _utc_timestamp(raw.get("fit_end_utc"), "support.fit_end_utc")
    if fit_end > query_cutoff:
        raise P1PreflightRenderingError("support fit waterline must not exceed query cutoff Q")
    common_mc = _number(raw.get("common_mc"), "support.common_mc")
    if not math.isclose(common_mc, 4.0, rel_tol=0.0, abs_tol=1e-12):
        raise P1PreflightRenderingError("support.common_mc must remain frozen at 4.0")
    fixed = _integer(raw.get("fixed_cell_count"), "support.fixed_cell_count", minimum=1)
    supported = _integer(raw.get("supported_cell_count"), "support.supported_cell_count")
    indeterminate = _integer(
        raw.get("indeterminate_cell_count"), "support.indeterminate_cell_count"
    )
    unsupported = _integer(raw.get("unsupported_cell_count"), "support.unsupported_cell_count")
    if supported + indeterminate + unsupported != fixed:
        raise P1PreflightRenderingError("support status counts must sum to fixed_cell_count")
    return _SupportWaterline(
        support_id=_text(raw.get("support_id"), "support.support_id"),
        manifest_sha256=_sha256(raw.get("manifest_sha256"), "support.manifest_sha256"),
        fit_end_utc=fit_text,
        common_mc=common_mc,
        fixed_cell_count=fixed,
        supported_cell_count=supported,
        indeterminate_cell_count=indeterminate,
        unsupported_cell_count=unsupported,
        retained_area_km2=_number(
            raw.get("retained_area_km2"), "support.retained_area_km2", minimum=1.0
        ),
    )


def _normalise_cells(value: object) -> tuple[_Cell, ...]:
    raw_grid = _mapping(value, "grid")
    _exact_keys(raw_grid, {"cell_size_km", "cells"}, "grid")
    cell_size = _number(raw_grid.get("cell_size_km"), "grid.cell_size_km", minimum=1.0)
    if not math.isclose(cell_size, _CELL_SIZE_KM, rel_tol=0.0, abs_tol=1e-12):
        raise P1PreflightRenderingError("grid.cell_size_km must remain frozen at 25")
    cells: list[_Cell] = []
    for index, item in enumerate(_sequence(raw_grid.get("cells"), "grid.cells")):
        path = f"grid.cells[{index}]"
        raw = _mapping(item, path)
        _exact_keys(raw, {"cell_id", "row", "column", "area_km2", "support_status"}, path)
        status = _text(raw.get("support_status"), f"{path}.support_status")
        if status not in {"supported", "indeterminate", "unsupported"}:
            raise P1PreflightRenderingError(f"{path}.support_status is invalid")
        area = _number(raw.get("area_km2"), f"{path}.area_km2", minimum=1e-12)
        if area > _FULL_CELL_AREA_KM2 + 1e-9:
            raise P1PreflightRenderingError(f"{path}.area_km2 exceeds a complete 25 km cell")
        row = _signed_integer(raw.get("row"), f"{path}.row")
        column = _signed_integer(raw.get("column"), f"{path}.column")
        cells.append(
            _Cell(
                cell_id=_text(raw.get("cell_id"), f"{path}.cell_id"),
                row=row,
                column=column,
                area_km2=area,
                support_status=cast(SupportStatus, status),
            )
        )
    if not cells:
        raise P1PreflightRenderingError("grid.cells must not be empty")
    canonical = tuple(
        sorted(cells, key=lambda item: (item.row, item.column, item.cell_id.encode()))
    )
    if tuple(cells) != canonical:
        raise P1PreflightRenderingError("grid.cells must be in canonical row/column/cell_id order")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise P1PreflightRenderingError("grid cell IDs must be unique")
    if len({(cell.row, cell.column) for cell in cells}) != len(cells):
        raise P1PreflightRenderingError("grid row/column positions must be unique")
    return tuple(cells)


def _largest_prefix(
    ranked_cells: tuple[_Cell, ...], *, maximum_area_km2: float
) -> tuple[_Cell, ...]:
    selected: list[_Cell] = []
    area = 0.0
    tolerance = max(1e-8, maximum_area_km2 * 1e-12)
    for cell in ranked_cells:
        candidate = area + cell.area_km2
        if candidate > maximum_area_km2 + tolerance:
            break
        selected.append(cell)
        area = candidate
    return tuple(selected)


def _normalise_model(
    value: object,
    *,
    model_id: str,
    cells: tuple[_Cell, ...],
    maximum_area_km2: float,
) -> _Model:
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
    normalized_cell_mass = tuple(
        _number(item, f"{path}.normalized_cell_mass[{index}]")
        for index, item in enumerate(
            _sequence(raw.get("normalized_cell_mass"), f"{path}.normalized_cell_mass")
        )
    )
    if len(normalized_cell_mass) != len(cells):
        raise P1PreflightRenderingError(f"{path}.normalized_cell_mass must align with grid.cells")
    total_mass = math.fsum(normalized_cell_mass)
    if not math.isclose(total_mass, 1.0, rel_tol=1e-8, abs_tol=1e-10):
        raise P1PreflightRenderingError(f"{path}.normalized_cell_mass must sum to one")
    if any(
        mass > 1e-15 and cell.support_status == "unsupported"
        for cell, mass in zip(cells, normalized_cell_mass, strict=True)
    ):
        raise P1PreflightRenderingError(f"{path} assigns mass to an unsupported support cell")
    relative_intensity_per_km2 = tuple(
        mass / cell.area_km2 for cell, mass in zip(cells, normalized_cell_mass, strict=True)
    )
    intensity_by_id = {
        cell.cell_id: intensity
        for cell, intensity in zip(cells, relative_intensity_per_km2, strict=True)
    }
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
        raise P1PreflightRenderingError(f"{path}.alarm_cell_ids contains duplicates")
    expected_prefix = _largest_prefix(ranked_cells, maximum_area_km2=maximum_area_km2)
    expected_ids = tuple(cell.cell_id for cell in expected_prefix)
    if alarm_ids != expected_ids:
        raise P1PreflightRenderingError(
            f"{path}.alarm_cell_ids must be the largest stable complete-cell prefix"
        )
    actual_area = _number(raw.get("actual_alarm_area_km2"), f"{path}.actual_alarm_area_km2")
    recomputed_area = math.fsum(cell.area_km2 for cell in expected_prefix)
    tolerance = max(1e-8, recomputed_area * 1e-10)
    if not math.isclose(actual_area, recomputed_area, rel_tol=0.0, abs_tol=tolerance):
        raise P1PreflightRenderingError(f"{path}.actual_alarm_area_km2 disagrees with cells")
    raw_next = raw.get("next_complete_cell_area_km2")
    expected_next = (
        ranked_cells[len(expected_prefix)].area_km2
        if len(expected_prefix) < len(ranked_cells)
        else None
    )
    if raw_next is None:
        next_area = None
    else:
        next_area = _number(raw_next, f"{path}.next_complete_cell_area_km2", minimum=1e-12)
    if expected_next is None:
        if next_area is not None:
            raise P1PreflightRenderingError(
                f"{path}.next_complete_cell_area_km2 must be null after the final cell"
            )
    elif next_area is None or not math.isclose(next_area, expected_next, rel_tol=0.0, abs_tol=1e-8):
        raise P1PreflightRenderingError(
            f"{path}.next_complete_cell_area_km2 disagrees with the stable ranking"
        )
    ranking = tuple(cell.cell_id for cell in ranked_cells)
    return _Model(
        model_id=model_id,
        normalized_cell_mass=normalized_cell_mass,
        relative_intensity_per_km2=relative_intensity_per_km2,
        alarm_cell_ids=alarm_ids,
        actual_alarm_area_km2=actual_area,
        next_complete_cell_area_km2=next_area,
        ranked_cell_ids=ranking,
        rank_by_cell={cell_id: rank for rank, cell_id in enumerate(ranking, start=1)},
    )


def _normalise_forecast(value: Mapping[str, object]) -> _Forecast:
    _reject_future_fields(value)
    _exact_keys(
        value,
        {
            "rehearsal_id",
            "scheduled_issue_time_utc",
            "query_cutoff_utc",
            "catalog",
            "support",
            "grid",
            "models",
        },
        "forecast",
    )
    rehearsal_id = _text(value.get("rehearsal_id"), "forecast.rehearsal_id")
    if not rehearsal_id.startswith("p1-0c-"):
        raise P1PreflightRenderingError("forecast.rehearsal_id must start with p1-0c-")
    scheduled_text, scheduled = _utc_timestamp(
        value.get("scheduled_issue_time_utc"), "forecast.scheduled_issue_time_utc"
    )
    cutoff_text, cutoff = _utc_timestamp(value.get("query_cutoff_utc"), "forecast.query_cutoff_utc")
    if scheduled - cutoff != timedelta(minutes=15):
        raise P1PreflightRenderingError("query cutoff Q must be exactly 15 minutes before T")
    catalog = _normalise_catalog(value.get("catalog"), query_cutoff=cutoff)
    support = _normalise_support(value.get("support"), query_cutoff=cutoff)
    cells = _normalise_cells(value.get("grid"))
    raw_models = _mapping(value.get("models"), "models")
    if set(raw_models) != set(_MODEL_IDS):
        raise P1PreflightRenderingError("models must contain exactly B0 and B0_R30")
    B0 = _normalise_model(
        raw_models.get("B0"),
        model_id="B0",
        cells=cells,
        maximum_area_km2=_AREA_CAP_KM2,
    )
    challenger = _normalise_model(
        raw_models.get("B0_R30"),
        model_id="B0_R30",
        cells=cells,
        maximum_area_km2=B0.actual_alarm_area_km2,
    )
    area_difference = B0.actual_alarm_area_km2 - challenger.actual_alarm_area_km2
    tolerance = max(1e-8, B0.actual_alarm_area_km2 * 1e-10)
    if area_difference < -tolerance:
        raise P1PreflightRenderingError("B0_R30 may not alarm more area than B0")
    if area_difference > tolerance:
        if not (area_difference < _FULL_CELL_AREA_KM2):
            raise P1PreflightRenderingError("paired alarm-area difference must be below 625 km2")
        if (
            challenger.next_complete_cell_area_km2 is None
            or not area_difference < challenger.next_complete_cell_area_km2 - tolerance
        ):
            raise P1PreflightRenderingError(
                "paired alarm-area difference must be below the challenger next complete cell"
            )
    shared_maximum = max((*B0.relative_intensity_per_km2, *challenger.relative_intensity_per_km2))
    if shared_maximum <= 0.0:
        raise P1PreflightRenderingError("both model surfaces cannot be zero")
    return _Forecast(
        rehearsal_id=rehearsal_id,
        scheduled_issue_time_utc=scheduled_text,
        scheduled_issue_time=scheduled,
        query_cutoff_utc=cutoff_text,
        query_cutoff=cutoff,
        catalog=catalog,
        support=support,
        cells=cells,
        models=(B0, challenger),
        minimum_row=min(cell.row for cell in cells),
        maximum_row=max(cell.row for cell in cells),
        minimum_column=min(cell.column for cell in cells),
        maximum_column=max(cell.column for cell in cells),
        shared_colour_maximum=shared_maximum,
    )


def _format_number(value: float, *, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def _next_area_text(value: float | None) -> str:
    return "无下一格" if value is None else f"{_format_number(value)} km²"


def _colour(value: float, maximum: float) -> str:
    ratio = 0.0 if maximum <= 0.0 else max(0.0, min(1.0, value / maximum))
    low = (238, 246, 255)
    high = (30, 92, 157)
    channels = tuple(
        round(start + ratio * (end - start)) for start, end in zip(low, high, strict=True)
    )
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


def _panel_geometry(
    forecast: _Forecast, *, x: float, y: float, width: float, height: float
) -> tuple[float, float, float]:
    columns = forecast.maximum_column - forecast.minimum_column + 1
    rows = forecast.maximum_row - forecast.minimum_row + 1
    scale = min((width - 170.0) / columns, (height - 64.0) / rows)
    grid_width = columns * scale
    grid_height = rows * scale
    return (
        x + 18.0,
        y + 48.0 + max(0.0, (height - 64.0 - grid_height) / 2.0),
        min(scale, grid_width / columns),
    )


def _forecast_panel_svg(
    forecast: _Forecast,
    model: _Model,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    clusters: tuple[_Cluster, ...] = (),
) -> list[str]:
    grid_x, grid_y, scale = _panel_geometry(forecast, x=x, y=y, width=width, height=height)
    selected = set(model.alarm_cell_ids)
    parts = [
        f'<g data-model="{model.model_id}" data-colour-max="{forecast.shared_colour_maximum:.12g}" '
        f'transform="translate({x:.1f},{y:.1f})">',
        f'<rect x="0" y="0" width="{width:.1f}" height="{height:.1f}" rx="10" '
        'fill="#ffffff" stroke="#cbd7e4"/>',
        f'<text x="18" y="30" class="model">{model.model_id}</text>',
        '<g data-layer="relative-intensity">',
    ]
    cell_by_id = {cell.cell_id: cell for cell in forecast.cells}
    index_by_id = {cell.cell_id: index for index, cell in enumerate(forecast.cells)}
    for cell in forecast.cells:
        index = index_by_id[cell.cell_id]
        cell_x = grid_x - x + (cell.column - forecast.minimum_column) * scale
        cell_y = grid_y - y + (cell.row - forecast.minimum_row) * scale
        alarm = cell.cell_id in selected
        stroke = "#e39400" if alarm else "#c8d3df"
        dash = ' stroke-dasharray="2 1"' if cell.support_status == "indeterminate" else ""
        parts.append(
            f'<rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{scale:.2f}" '
            f'height="{scale:.2f}" fill="{_colour(model.relative_intensity_per_km2[index], forecast.shared_colour_maximum)}" '
            f'stroke="{stroke}" stroke-width="{2.2 if alarm else 0.7:.1f}"{dash} '
            f'data-cell-id="{html.escape(cell.cell_id)}" data-alarm="{str(alarm).lower()}" '
            f'data-relative-intensity-per-km2="{model.relative_intensity_per_km2[index]:.12g}" '
            f'data-support-status="{cell.support_status}"><title>格 {html.escape(cell.cell_id)}；'
            f"单位面积相对强度 {model.relative_intensity_per_km2[index]:.8g} /km²；排名 "
            f"{model.rank_by_cell[cell.cell_id]}；面积 {cell.area_km2:.3f} km²；"
            f"{'报警格' if alarm else '非报警格'}</title></rect>"
        )
    parts.append("</g>")
    if clusters:
        parts.append('<g data-layer="synthetic-mature-clusters">')
        occurrence_by_cell: dict[str, int] = {}
        for cluster in clusters:
            cell = cell_by_id[cluster.cell_id]
            occurrence = occurrence_by_cell.get(cell.cell_id, 0)
            occurrence_by_cell[cell.cell_id] = occurrence + 1
            offset = ((occurrence % 3) - 1) * min(4.0, scale * 0.15)
            centre_x = grid_x - x + (cell.column - forecast.minimum_column + 0.5) * scale + offset
            centre_y = grid_y - y + (cell.row - forecast.minimum_row + 0.5) * scale - offset
            radius = max(3.0, min(7.0, scale * 0.22))
            if cell.cell_id in selected:
                parts.append(
                    f'<circle cx="{centre_x:.2f}" cy="{centre_y:.2f}" r="{radius:.2f}" '
                    f'fill="#138a53" stroke="#ffffff" stroke-width="1.5" '
                    f'data-cluster-id="{html.escape(cluster.cluster_id)}" data-outcome="hit">'
                    f"<title>{html.escape(cluster.cluster_id)}：命中</title></circle>"
                )
            else:
                parts.extend(
                    [
                        f'<g data-cluster-id="{html.escape(cluster.cluster_id)}" '
                        'data-outcome="miss">',
                        f"<title>{html.escape(cluster.cluster_id)}：漏报</title>",
                        f'<path d="M {centre_x - radius:.2f} {centre_y - radius:.2f} '
                        f"L {centre_x + radius:.2f} {centre_y + radius:.2f} "
                        f"M {centre_x + radius:.2f} {centre_y - radius:.2f} "
                        f'L {centre_x - radius:.2f} {centre_y + radius:.2f}" '
                        'stroke="#c33b32" stroke-width="2.5" stroke-linecap="round"/>',
                        "</g>",
                    ]
                )
        parts.append("</g>")
    metric_x = width - 140.0
    parts.extend(
        [
            f'<text x="{metric_x:.1f}" y="78" class="metric-label">实际报警面积</text>',
            f'<text x="{metric_x:.1f}" y="103" class="metric">'
            f"{_format_number(model.actual_alarm_area_km2)} km²</text>",
            f'<text x="{metric_x:.1f}" y="140" class="metric-label">下一完整格</text>',
            f'<text x="{metric_x:.1f}" y="165" class="metric">'
            f"{_next_area_text(model.next_complete_cell_area_km2)}</text>",
            f'<text x="{metric_x:.1f}" y="202" class="metric-label">完整报警格</text>',
            f'<text x="{metric_x:.1f}" y="227" class="metric">'
            f"{len(model.alarm_cell_ids):,} 个</text>",
            "</g>",
        ]
    )
    return parts


def render_preflight_forecast_svg(forecast_mapping: Mapping[str, object]) -> bytes:
    """Render a deterministic, target-blind P1-0C historical rehearsal SVG."""

    forecast = _normalise_forecast(_mapping(forecast_mapping, "forecast"))
    area_difference = (
        forecast.models[0].actual_alarm_area_km2 - forecast.models[1].actual_alarm_area_km2
    )
    width = 1280
    height = 790
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
        f'data-artifact="p1-0c-preflight-forecast" data-shared-colour-max="'
        f'{forecast.shared_colour_maximum:.12g}">',
        '<title id="title">P1-0C 历史适配双模型演练</title>',
        '<desc id="desc">只使用查询截止时已可用输入的双模型单位面积相对强度与完整报警格。</desc>',
        "<style>",
        "text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#17243a}",
        ".title{font-size:27px;font-weight:700}.warning{font-size:17px;font-weight:700;fill:#a33226}",
        ".subtitle{font-size:13px;fill:#506176}.model{font-size:19px;font-weight:700}",
        ".metric-label{font-size:12px;fill:#65758a}.metric{font-size:15px;font-weight:650}",
        ".note{font-size:12px;fill:#506176}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        '<text x="42" y="44" class="title">P1-0C 起报时双模型科学图</text>',
        '<text x="42" y="74" class="warning">历史适配演练，不是真实前瞻 issue</text>',
        f'<text x="42" y="102" class="subtitle">演练 ID = '
        f"{html.escape(forecast.rehearsal_id)}；T = "
        f"{html.escape(forecast.scheduled_issue_time_utc)}；Q = "
        f"{html.escape(forecast.query_cutoff_utc)}</text>",
        f'<text x="42" y="126" class="subtitle">目录水位：origin/available 最大值 '
        f"{html.escape(forecast.catalog.origin_time_max_utc)} / "
        f"{html.escape(forecast.catalog.available_at_max_utc)}；B0 / R30 入选 "
        f"{forecast.catalog.eligible_B0_event_count:,} / {forecast.catalog.R30_event_count:,}</text>",
        f'<text x="42" y="150" class="subtitle">support 水位：fit_end '
        f"{html.escape(forecast.support.fit_end_utc)}；support_id "
        f"{html.escape(forecast.support.support_id)}；Mc={forecast.support.common_mc:.1f}；"
        f"{forecast.support.fixed_cell_count} 格（{forecast.support.supported_cell_count} supported + "
        f"{forecast.support.indeterminate_cell_count} indeterminate + "
        f"{forecast.support.unsupported_cell_count} unsupported）</text>",
        '<text x="42" y="174" class="subtitle">75 km Gaussian KDE；'
        "B0_R30 = 0.75 × B0 + 0.25 × R30；颜色 = normalized_cell_mass / 实际格面积；"
        "双图使用同一色标。</text>",
    ]
    parts.extend(
        _forecast_panel_svg(
            forecast,
            forecast.models[0],
            x=42.0,
            y=196.0,
            width=580.0,
            height=430.0,
        )
    )
    parts.extend(
        _forecast_panel_svg(
            forecast,
            forecast.models[1],
            x=658.0,
            y=196.0,
            width=580.0,
            height=430.0,
        )
    )
    parts.extend(
        [
            f'<text x="42" y="660" class="note">面积公平：B0_R30 不超过 B0；'
            f"实际面积差 {_format_number(area_difference, decimals=3)} km²；"
            f"上限 {_format_number(_AREA_CAP_KM2, decimals=0)} km²。</text>",
            f'<text x="42" y="684" class="note">目录：'
            f"{html.escape(forecast.catalog.path)}；SHA-256 "
            f"{forecast.catalog.sha256}</text>",
            f'<text x="42" y="708" class="note">support manifest SHA-256 '
            f"{forecast.support.manifest_sha256}；保留面积 "
            f"{_format_number(forecast.support.retained_area_km2)} km²。</text>",
            '<text x="42" y="738" class="note">本图只含查询截止时可用的输入、单位面积相对强度、顺位和报警格；'
            "相对强度不是绝对发震概率。</text>",
            '<text x="42" y="762" class="note">报警面积由 25 km 裁剪格实际面积逐格求和；'
            "边界格不得切割或跳过。</text>",
            "</svg>",
        ]
    )
    return ("\n".join(parts) + "\n").encode("utf-8")


def _forecast_json(forecast: _Forecast) -> dict[str, object]:
    return {
        "rehearsal_id": forecast.rehearsal_id,
        "scheduled_issue_time_utc": forecast.scheduled_issue_time_utc,
        "query_cutoff_utc": forecast.query_cutoff_utc,
        "catalog": {
            "path": forecast.catalog.path,
            "sha256": forecast.catalog.sha256,
            "row_count": forecast.catalog.row_count,
            "eligible_B0_event_count": forecast.catalog.eligible_B0_event_count,
            "R30_event_count": forecast.catalog.R30_event_count,
            "origin_time_max_utc": forecast.catalog.origin_time_max_utc,
            "available_at_max_utc": forecast.catalog.available_at_max_utc,
        },
        "support": {
            "support_id": forecast.support.support_id,
            "manifest_sha256": forecast.support.manifest_sha256,
            "fit_end_utc": forecast.support.fit_end_utc,
            "common_mc": forecast.support.common_mc,
            "fixed_cell_count": forecast.support.fixed_cell_count,
            "supported_cell_count": forecast.support.supported_cell_count,
            "indeterminate_cell_count": forecast.support.indeterminate_cell_count,
            "unsupported_cell_count": forecast.support.unsupported_cell_count,
            "retained_area_km2": forecast.support.retained_area_km2,
        },
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
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_preflight_forecast_html(forecast_mapping: Mapping[str, object]) -> str:
    """Build a deterministic, fully offline P1-0C forecast explorer."""

    forecast = _normalise_forecast(_mapping(forecast_mapping, "forecast"))
    payload = _safe_json(_forecast_json(forecast))
    area_difference = (
        forecast.models[0].actual_alarm_area_km2 - forecast.models[1].actual_alarm_area_km2
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1-0C 历史适配双模型演练</title>
<style>
:root{{--ink:#17243a;--muted:#5c6d82;--paper:#fff;--bg:#eef3f8;--line:#cbd7e4;--alarm:#e39400}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:24px}}header,.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}
h1{{font-size:27px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 10px}}p{{line-height:1.55;margin:7px 0}}.warning{{color:#a33226;font-weight:700}}.muted{{color:var(--muted)}}
.controls{{display:flex;gap:18px;align-items:end;flex-wrap:wrap}}label{{display:grid;gap:6px;font-weight:650}}.toggle{{display:flex;gap:7px;align-items:center;padding:8px 0}}select{{font:inherit;padding:8px;border:1px solid #aebdcc;border-radius:7px;background:#fff}}
.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(285px,1fr);gap:16px}}canvas{{display:block;width:100%;height:auto;background:#f9fbfd;border:1px solid var(--line);border-radius:8px}}
.metric{{background:#f4f7fb;border-radius:8px;padding:12px;margin-bottom:9px}}.metric strong{{display:block;font-size:20px;margin-top:4px}}.inspector{{margin-top:10px;padding:10px;background:#eef4fa;border-radius:7px;font-family:ui-monospace,Consolas,monospace}}code{{overflow-wrap:anywhere}}@media(max-width:760px){{.layout{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main data-artifact="p1-0c-preflight-forecast-offline">
<header><h1>P1-0C 起报时双模型科学图</h1><p class="warning">历史适配演练，不是真实前瞻 issue</p><p class="muted">演练 ID = {html.escape(forecast.rehearsal_id)} · T = {html.escape(forecast.scheduled_issue_time_utc)} · Q = {html.escape(forecast.query_cutoff_utc)}</p><p>75 km Gaussian KDE；<code>B0_R30 = 0.75 × B0 + 0.25 × R30</code>。颜色与排序均使用 <code>normalized_cell_mass / 实际格面积</code> 的单位面积相对强度；双模型共享同一色标，相对强度不是绝对发震概率。</p></header>
<section class="card controls" aria-label="起报图控件"><label for="model-select">模型<select id="model-select"><option value="B0">B0</option><option value="B0_R30">B0_R30</option></select></label><label class="toggle" for="intensity-toggle"><input id="intensity-toggle" type="checkbox" checked>显示单位面积相对强度</label><label class="toggle" for="alarm-toggle"><input id="alarm-toggle" type="checkbox" checked>显示报警格</label><label class="toggle" for="support-toggle"><input id="support-toggle" type="checkbox" checked>标示 support 状态</label></section>
<section class="layout"><div class="card"><h2 id="view-title">格网图</h2><canvas id="grid" width="760" height="590" data-layer-relative-intensity="toggle" data-layer-alarm="toggle" data-layer-support="toggle"></canvas><div class="inspector" id="cell-inspector" aria-live="polite">移动或点击格网，查看 cell_id、单位面积相对强度、排名、裁剪面积和 support 状态。</div></div>
<aside class="card"><h2>起报时数据水位</h2><div class="metric">当前实际报警面积<strong id="area-value">—</strong></div><div class="metric">下一完整格面积<strong id="next-area-value">—</strong></div><div class="metric">B0 / R30 入选数<strong>{forecast.catalog.eligible_B0_event_count:,} / {forecast.catalog.R30_event_count:,}</strong></div><p>面积公平：B0_R30 不超过 B0；实际面积差 {_format_number(area_difference, decimals=3)} km²。</p><p>目录 origin / available 最大值：<br>{html.escape(forecast.catalog.origin_time_max_utc)}<br>{html.escape(forecast.catalog.available_at_max_utc)}</p><p>support fit_end：{html.escape(forecast.support.fit_end_utc)}<br>support_id：<code>{html.escape(forecast.support.support_id)}</code><br>Mc={forecast.support.common_mc:.1f}；{forecast.support.fixed_cell_count} 格 = {forecast.support.supported_cell_count} supported + {forecast.support.indeterminate_cell_count} indeterminate + {forecast.support.unsupported_cell_count} unsupported</p></aside></section>
<section class="card"><h2>冻结身份</h2><p>目录：<code>{html.escape(forecast.catalog.path)}</code><br>目录 SHA-256：<code>{forecast.catalog.sha256}</code><br>support manifest SHA-256：<code>{forecast.support.manifest_sha256}</code><br>support 保留面积：{_format_number(forecast.support.retained_area_km2)} km²</p><p class="muted">本页只含查询截止时可用的输入、单位面积相对强度、顺位和报警格。报警面积由 25 km 裁剪格实际面积逐格求和。</p></section>
<script type="application/json" id="forecast-data">{payload}</script>
<script>
"use strict";
const forecast=JSON.parse(document.getElementById("forecast-data").textContent);
const modelSelect=document.getElementById("model-select"),intensityToggle=document.getElementById("intensity-toggle"),alarmToggle=document.getElementById("alarm-toggle"),supportToggle=document.getElementById("support-toggle");
const canvas=document.getElementById("grid"),context=canvas.getContext("2d"),inspector=document.getElementById("cell-inspector");let view=null;
function fmt(value){{return new Intl.NumberFormat("zh-CN",{{maximumFractionDigits:3}}).format(value)}}
function colour(value){{const ratio=forecast.shared_colour_maximum>0?Math.max(0,Math.min(1,value/forecast.shared_colour_maximum)):0,low=[238,246,255],high=[30,92,157];return "rgb("+low.map((item,index)=>Math.round(item+ratio*(high[index]-item))).join(",")+")"}}
function render(){{const model=forecast.models[modelSelect.value],alarms=new Set(model.alarm_cell_ids),columns=forecast.maximum_column-forecast.minimum_column+1,rows=forecast.maximum_row-forecast.minimum_row+1,padding=34,scale=Math.min((canvas.width-padding*2)/columns,(canvas.height-padding*2)/rows),originX=(canvas.width-columns*scale)/2,originY=(canvas.height-rows*scale)/2;view={{model,alarms,scale,originX,originY}};context.clearRect(0,0,canvas.width,canvas.height);context.fillStyle="#f9fbfd";context.fillRect(0,0,canvas.width,canvas.height);forecast.cells.forEach((cell,index)=>{{const x=originX+(cell.column-forecast.minimum_column)*scale,y=originY+(cell.row-forecast.minimum_row)*scale,isAlarm=alarms.has(cell.cell_id);context.fillStyle=intensityToggle.checked?colour(model.relative_intensity_per_km2[index]):"#fff";context.fillRect(x,y,scale,scale);context.setLineDash(supportToggle.checked&&cell.support_status==="indeterminate"?[3,2]:[]);context.strokeStyle=alarmToggle.checked&&isAlarm?"#e39400":"#c8d3df";context.lineWidth=alarmToggle.checked&&isAlarm?2.5:1;context.strokeRect(x,y,scale,scale);context.setLineDash([])}});document.getElementById("view-title").textContent=modelSelect.value+" · 共享色标单位面积相对强度与报警格";document.getElementById("area-value").textContent=fmt(model.actual_alarm_area_km2)+" km²";document.getElementById("next-area-value").textContent=model.next_complete_cell_area_km2===null?"无下一格":fmt(model.next_complete_cell_area_km2)+" km²"}}
function inspect(event){{if(!view)return;const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height,column=Math.floor((x-view.originX)/view.scale)+forecast.minimum_column,row=Math.floor((y-view.originY)/view.scale)+forecast.minimum_row,cell=forecast.cells.find(item=>item.row===row&&item.column===column);if(!cell)return;const index=forecast.cells.indexOf(cell),value=view.model.relative_intensity_per_km2[index],rank=view.model.rank_by_cell[cell.cell_id],alarm=view.alarms.has(cell.cell_id);inspector.textContent="cell_id="+cell.cell_id+" · relative_intensity_per_km2="+value.toExponential(6)+" · rank="+rank+" · area="+fmt(cell.area_km2)+" km² · alarm="+(alarm?"是":"否")+" · support="+cell.support_status}}
modelSelect.addEventListener("change",render);intensityToggle.addEventListener("change",render);alarmToggle.addEventListener("change",render);supportToggle.addEventListener("change",render);canvas.addEventListener("mousemove",inspect);canvas.addEventListener("click",inspect);render();
</script>
</main>
</body>
</html>
"""


def _validate_scientific_forecast_context(
    scientific: object,
    locator: object,
    rendered: _Forecast,
) -> tuple[DualModelForecast, Frozen25kmCellLocator]:
    if not isinstance(scientific, DualModelForecast):
        raise P1PreflightRenderingError("replay.scientific_forecast must be a DualModelForecast")
    if not isinstance(locator, Frozen25kmCellLocator):
        raise P1PreflightRenderingError(
            "replay.scientific_locator must be the frozen 25 km locator"
        )
    if scientific.scheduled_issue_time_utc != rendered.scheduled_issue_time:
        raise P1PreflightRenderingError("scientific forecast issue time differs from the map")
    if scientific.query_cutoff_utc != rendered.query_cutoff:
        raise P1PreflightRenderingError("scientific forecast cutoff differs from the map")
    if len(scientific.grid) != len(rendered.cells) or locator.grid.cell_count != len(
        rendered.cells
    ):
        raise P1PreflightRenderingError("scientific, locator and rendered grids differ in length")
    locator_ids = tuple(locator.grid.cell_ids)
    scientific_ids = tuple(cell.cell_id for cell in scientific.grid)
    rendered_ids = tuple(cell.cell_id for cell in rendered.cells)
    if locator_ids != scientific_ids or locator_ids != rendered_ids:
        raise P1PreflightRenderingError("scientific, locator and rendered grid IDs differ")
    row_shifts: set[int] = set()
    column_shifts: set[int] = set()
    for index, (scientific_cell, rendered_cell) in enumerate(
        zip(scientific.grid, rendered.cells, strict=True)
    ):
        locator_row = int(locator.grid.rows[index])
        locator_column = int(locator.grid.columns[index])
        locator_xy = locator.grid.query_xy_km[index]
        locator_area = float(locator.grid.clipped_area_km2[index])
        row_shifts.add(scientific_cell.row - locator_row)
        column_shifts.add(scientific_cell.column - locator_column)
        if (rendered_cell.row, rendered_cell.column) != (locator_row, locator_column):
            raise P1PreflightRenderingError("rendered grid indices differ from the frozen locator")
        if not (
            math.isclose(
                scientific_cell.x_km,
                float(locator_xy[0]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                scientific_cell.y_km,
                float(locator_xy[1]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                scientific_cell.area_km2,
                locator_area,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                rendered_cell.area_km2,
                locator_area,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise P1PreflightRenderingError(
                "scientific coordinates or areas differ from the frozen 25 km grid"
            )
    if len(row_shifts) != 1 or len(column_shifts) != 1:
        raise P1PreflightRenderingError(
            "synthetic grid may shift only all row/column indices by constants"
        )
    for rendered_model, surface, alarm in (
        (rendered.models[0], scientific.B0, scientific.B0_alarm),
        (rendered.models[1], scientific.B0_R30, scientific.B0_R30_alarm),
    ):
        if tuple(float(item) for item in surface.relative_intensity) != (
            rendered_model.normalized_cell_mass
        ):
            raise P1PreflightRenderingError("scientific model mass differs from the rendered map")
        if alarm.selected_cell_ids != rendered_model.alarm_cell_ids:
            raise P1PreflightRenderingError("scientific alarm mask differs from the rendered map")
        if not math.isclose(
            alarm.actual_area_km2,
            rendered_model.actual_alarm_area_km2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise P1PreflightRenderingError("scientific alarm area differs from the rendered map")
    return scientific, locator


def _normalise_replay(
    value: Mapping[str, object],
    *,
    raw_truth_bytes: bytes,
    scientific_forecast: DualModelForecast,
    scientific_locator: Frozen25kmCellLocator,
) -> _Replay:
    _exact_keys(
        value,
        {
            "replay_id",
            "forecast_sha256",
            "forecast",
            "synthetic_raw_response_sha256",
            "synthetic_raw_response_byte_count",
            "cluster_assignment_sha256",
            "ordered_cluster_registry_sha256",
            "horizon_days",
            "mature_after_utc",
            "replay_created_at_utc",
            "clusters",
            "review",
        },
        "replay",
    )
    replay_id = _text(value.get("replay_id"), "replay.replay_id")
    if not replay_id.startswith("p1-0c-synthetic-"):
        raise P1PreflightRenderingError("replay.replay_id must start with p1-0c-synthetic-")
    raw_forecast = _mapping(value.get("forecast"), "replay.forecast")
    forecast = _normalise_forecast(raw_forecast)
    declared_forecast_sha = _sha256(value.get("forecast_sha256"), "replay.forecast_sha256")
    actual_forecast_sha = hashlib.sha256(render_preflight_forecast_svg(raw_forecast)).hexdigest()
    if declared_forecast_sha != actual_forecast_sha:
        raise P1PreflightRenderingError(
            "replay.forecast_sha256 must equal the exact preflight forecast SVG SHA-256"
        )
    scientific, locator = _validate_scientific_forecast_context(
        scientific_forecast,
        scientific_locator,
        forecast,
    )
    if type(raw_truth_bytes) is not bytes:
        raise P1PreflightRenderingError("replay synthetic raw response must be exact bytes")
    raw_bytes = raw_truth_bytes
    declared_raw_sha = _sha256(
        value.get("synthetic_raw_response_sha256"),
        "replay.synthetic_raw_response_sha256",
    )
    if hashlib.sha256(raw_bytes).hexdigest() != declared_raw_sha:
        raise P1PreflightRenderingError("synthetic raw response SHA-256 differs from exact bytes")
    declared_raw_count = _integer(
        value.get("synthetic_raw_response_byte_count"),
        "replay.synthetic_raw_response_byte_count",
        minimum=1,
    )
    if declared_raw_count != len(raw_bytes):
        raise P1PreflightRenderingError(
            "synthetic raw response byte count differs from exact bytes"
        )
    horizon = _integer(value.get("horizon_days"), "replay.horizon_days", minimum=1)
    if horizon not in {30, 90}:
        raise P1PreflightRenderingError("replay.horizon_days must be 30 or 90")
    mature_text, mature_after = _utc_timestamp(
        value.get("mature_after_utc"), "replay.mature_after_utc"
    )
    expected_maturity = forecast.scheduled_issue_time + timedelta(days=horizon + 30)
    if mature_after != expected_maturity:
        raise P1PreflightRenderingError("replay maturity must equal T + horizon + 30 days")
    created_text, created = _utc_timestamp(
        value.get("replay_created_at_utc"), "replay.replay_created_at_utc"
    )
    if created < mature_after:
        raise P1PreflightRenderingError("mature replay cannot be created before maturity")
    try:
        recomputed = recompute_mature_truth_snapshot(
            raw_bytes,
            scientific,
            horizon_days=cast(Literal[30, 90], horizon),
            truth_fetched_at_utc=created,
        )
    except (TypeError, ValueError) as error:
        raise P1PreflightRenderingError(
            f"synthetic raw truth recomputation failed: {error}"
        ) from error
    declared_cluster_sha = _sha256(
        value.get("cluster_assignment_sha256"),
        "replay.cluster_assignment_sha256",
    )
    if declared_cluster_sha != recomputed.cluster_assignment_sha256:
        raise P1PreflightRenderingError("cluster assignment differs from raw truth recomputation")
    declared_registry_sha = _sha256(
        value.get("ordered_cluster_registry_sha256"),
        "replay.ordered_cluster_registry_sha256",
    )
    if declared_registry_sha != recomputed.ordered_cluster_registry_sha256:
        raise P1PreflightRenderingError("score registry differs from raw truth recomputation")
    cells = {cell.cell_id: cell for cell in forecast.cells}
    clusters: list[_Cluster] = []
    for index, item in enumerate(_sequence(value.get("clusters"), "replay.clusters")):
        path = f"replay.clusters[{index}]"
        raw = _mapping(item, path)
        _exact_keys(
            raw,
            {"cluster_id", "representative_event_id", "cell_id", "origin_time_utc"},
            path,
        )
        cell_id = _text(raw.get("cell_id"), f"{path}.cell_id")
        if cell_id not in cells:
            raise P1PreflightRenderingError(f"{path}.cell_id is outside the frozen forecast grid")
        origin_text, origin = _utc_timestamp(raw.get("origin_time_utc"), f"{path}.origin_time_utc")
        if (
            not forecast.scheduled_issue_time
            < origin
            <= (forecast.scheduled_issue_time + timedelta(days=horizon))
        ):
            raise P1PreflightRenderingError(f"{path}.origin_time_utc is outside (T,T+h]")
        clusters.append(
            _Cluster(
                cluster_id=_text(raw.get("cluster_id"), f"{path}.cluster_id"),
                representative_event_id=_text(
                    raw.get("representative_event_id"), f"{path}.representative_event_id"
                ),
                cell_id=cell_id,
                origin_time_utc=origin_text,
            )
        )
    expected_clusters: list[_Cluster] = []
    for cluster in recomputed.clusters:
        representative = cluster.representative
        if representative.longitude is None or representative.latitude is None:
            raise AssertionError("SyntheticEvent always materializes WGS84 coordinates")
        projected_index = locator.locate_projected(
            representative.x_km * 1_000.0,
            representative.y_km * 1_000.0,
        )
        geographic_index = locator.locate_lonlat(
            representative.longitude,
            representative.latitude,
        )
        if projected_index is None or projected_index != geographic_index:
            raise P1PreflightRenderingError(
                "representative projected and WGS84 coordinates do not locate one frozen cell"
            )
        locator_cell_id = locator.grid.cell_ids[projected_index]
        try:
            scientific_cell_id = locate_point_cell(
                scientific.grid,
                x_km=representative.x_km,
                y_km=representative.y_km,
            ).cell_id
        except ValueError as error:
            raise P1PreflightRenderingError(
                f"representative cannot be located in the scientific grid: {error}"
            ) from error
        if scientific_cell_id != locator_cell_id:
            raise P1PreflightRenderingError(
                "scientific-grid and frozen clipped-cell locators disagree"
            )
        expected_clusters.append(
            _Cluster(
                cluster_id=cluster.cluster_id,
                representative_event_id=representative.event_id,
                cell_id=locator_cell_id,
                origin_time_utc=representative.origin_time_utc.isoformat().replace("+00:00", "Z"),
            )
        )
    if tuple(clusters) != tuple(expected_clusters):
        raise P1PreflightRenderingError(
            "display cluster rows differ from raw cluster and frozen-locator recomputation"
        )
    review = _mapping(value.get("review"), "replay.review")
    expected_review_keys = {
        "horizon_days",
        "review_trigger",
        "look_sequence",
        "prior_completed_look_count",
        "cumulative_cluster_count",
        "ordered_cluster_registry_sha256",
        "selected_cluster_prefix_sha256",
        "elapsed_months",
        "B0_hit_clusters",
        "B0_R30_hit_clusters",
        "recall_gain_percentage_points",
        "sequentially_adjusted_interval_lower",
        "sequentially_adjusted_interval_upper",
        "decision",
    }
    _exact_keys(review, expected_review_keys, "replay.review")
    _number(review.get("elapsed_months"), "replay.review.elapsed_months")
    elapsed_months = elapsed_tropical_months(scientific.scheduled_issue_time_utc, created)
    try:
        reviews = build_pending_sequential_reviews(
            recomputed.scores,
            elapsed_months=elapsed_months,
        )
    except ValueError as error:
        raise P1PreflightRenderingError(f"cluster review recomputation failed: {error}") from error
    if len(reviews) != 1 or reviews[0].review_trigger != "cluster_10":
        raise P1PreflightRenderingError(
            "ten-cluster mature replay must produce exactly the frozen cluster_10 review"
        )
    expected_review = reviews[0].as_mapping()
    if canonical_json_bytes(dict(review)) != canonical_json_bytes(expected_review):
        raise P1PreflightRenderingError("cluster_10 review differs from paired-score recomputation")
    return _Replay(
        replay_id=replay_id,
        forecast_sha256=declared_forecast_sha,
        forecast=forecast,
        synthetic_raw_response_sha256=declared_raw_sha,
        synthetic_raw_response_byte_count=declared_raw_count,
        cluster_assignment_sha256=declared_cluster_sha,
        ordered_cluster_registry_sha256=declared_registry_sha,
        horizon_days=horizon,
        mature_after_utc=mature_text,
        replay_created_at_utc=created_text,
        clusters=tuple(clusters),
        review=expected_review,
    )


def _score(replay: _Replay, model: _Model) -> tuple[int, int, float | None]:
    alarm_ids = set(model.alarm_cell_ids)
    hit_count = sum(cluster.cell_id in alarm_ids for cluster in replay.clusters)
    total = len(replay.clusters)
    return hit_count, total - hit_count, None if total == 0 else hit_count / total


def render_preflight_mature_replay_svg(
    replay_mapping: Mapping[str, object],
    *,
    raw_truth_bytes: bytes,
    scientific_forecast: DualModelForecast,
    scientific_locator: Frozen25kmCellLocator,
) -> bytes:
    """Render a separate synthetic mature replay bound to the forecast SVG hash."""

    replay = _normalise_replay(
        _mapping(replay_mapping, "replay"),
        raw_truth_bytes=raw_truth_bytes,
        scientific_forecast=scientific_forecast,
        scientific_locator=scientific_locator,
    )
    B0_hit, B0_miss, B0_recall = _score(replay, replay.forecast.models[0])
    challenger_hit, challenger_miss, challenger_recall = _score(replay, replay.forecast.models[1])
    gain = (
        None
        if B0_recall is None or challenger_recall is None
        else (challenger_recall - B0_recall) * 100.0
    )
    width = 1280
    height = 830
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
        f'data-artifact="p1-0c-synthetic-mature-replay" data-forecast-sha256="'
        f'{replay.forecast_sha256}" data-shared-colour-max="'
        f'{replay.forecast.shared_colour_maximum:.12g}" data-cluster-assignment-sha256="'
        f'{replay.cluster_assignment_sha256}" data-score-registry-sha256="'
        f'{replay.ordered_cluster_registry_sha256}">',
        '<title id="title">P1-0C 合成原始字节成熟回放</title>',
        '<desc id="desc">另存的合成已知答案回放，绑定原起报 SVG 哈希，不是真实效果证据。</desc>',
        "<style>",
        "text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#17243a}",
        ".title{font-size:27px;font-weight:700}.warning{font-size:17px;font-weight:700;fill:#a33226}",
        ".subtitle{font-size:13px;fill:#506176}.model{font-size:19px;font-weight:700}",
        ".metric-label{font-size:12px;fill:#65758a}.metric{font-size:15px;font-weight:650}",
        ".note{font-size:12px;fill:#506176}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        '<text x="42" y="44" class="title">P1-0C 合成原始字节成熟回放</text>',
        '<text x="42" y="74" class="warning">另存回放，不是真实预测效果证据</text>',
        f'<text x="42" y="102" class="subtitle">forecast SHA-256 = {replay.forecast_sha256}</text>',
        f'<text x="42" y="126" class="subtitle">合成原始响应 SHA-256 = '
        f"{replay.synthetic_raw_response_sha256}；字节数 "
        f"{replay.synthetic_raw_response_byte_count:,}</text>",
        f'<text x="42" y="150" class="subtitle">{replay.horizon_days} 天窗口；成熟时刻 '
        f"{html.escape(replay.mature_after_utc)}；回放形成时刻 "
        f"{html.escape(replay.replay_created_at_utc)}</text>",
        '<text x="42" y="174" class="subtitle">绿色圆点为命中，红叉为漏报；'
        "底图仍使用原起报时共享色标与原报警掩膜。</text>",
    ]
    parts.extend(
        _forecast_panel_svg(
            replay.forecast,
            replay.forecast.models[0],
            x=42.0,
            y=196.0,
            width=580.0,
            height=430.0,
            clusters=replay.clusters,
        )
    )
    parts.extend(
        _forecast_panel_svg(
            replay.forecast,
            replay.forecast.models[1],
            x=658.0,
            y=196.0,
            width=580.0,
            height=430.0,
            clusters=replay.clusters,
        )
    )
    B0_recall_text = "无合成震群" if B0_recall is None else f"{B0_recall * 100.0:.1f}%"
    challenger_recall_text = (
        "无合成震群" if challenger_recall is None else f"{challenger_recall * 100.0:.1f}%"
    )
    gain_text = "不可计算" if gain is None else f"{gain:+.1f} 个百分点"
    parts.extend(
        [
            f'<text x="42" y="660" class="note">B0：命中 {B0_hit}、漏报 '
            f"{B0_miss}、召回 {B0_recall_text}；B0_R30：命中 {challenger_hit}、漏报 "
            f"{challenger_miss}、召回 {challenger_recall_text}。</text>",
            f'<text x="42" y="686" class="note">合成已知答案召回变化：{gain_text}；'
            "该数值只验证原始字节到展示的逻辑。</text>",
            f'<text x="42" y="716" class="note">冻结 cluster_10 复审：B0 '
            f"{replay.review['B0_hit_clusters']}/10，B0_R30 "
            f"{replay.review['B0_R30_hit_clusters']}/10；决定 "
            f"{html.escape(str(replay.review['decision']))}。</text>",
            '<text x="42" y="742" class="note">成熟层只存在于本回放；'
            "原起报 SVG/HTML 的内容和 SHA-256 不得改变。</text>",
            '<text x="42" y="768" class="note">相对强度不是绝对发震概率；'
            "两个模型继续使用同一报警面积规则。</text>",
            f'<text x="42" y="796" class="note">replay_id = '
            f"{html.escape(replay.replay_id)}；合成震群数 {len(replay.clusters)}。</text>",
            "</svg>",
        ]
    )
    return ("\n".join(parts) + "\n").encode("utf-8")


def _replay_json(replay: _Replay) -> dict[str, object]:
    model_scores: dict[str, object] = {}
    for model in replay.forecast.models:
        hit, miss, recall = _score(replay, model)
        model_scores[model.model_id] = {
            "hit_count": hit,
            "miss_count": miss,
            "recall": recall,
        }
    return {
        "replay_id": replay.replay_id,
        "forecast_sha256": replay.forecast_sha256,
        "synthetic_raw_response_sha256": replay.synthetic_raw_response_sha256,
        "synthetic_raw_response_byte_count": replay.synthetic_raw_response_byte_count,
        "cluster_assignment_sha256": replay.cluster_assignment_sha256,
        "ordered_cluster_registry_sha256": replay.ordered_cluster_registry_sha256,
        "horizon_days": replay.horizon_days,
        "mature_after_utc": replay.mature_after_utc,
        "replay_created_at_utc": replay.replay_created_at_utc,
        "forecast": _forecast_json(replay.forecast),
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "representative_event_id": cluster.representative_event_id,
                "cell_id": cluster.cell_id,
                "origin_time_utc": cluster.origin_time_utc,
            }
            for cluster in replay.clusters
        ],
        "model_scores": model_scores,
        "review": dict(replay.review),
    }


def build_preflight_mature_replay_html(
    replay_mapping: Mapping[str, object],
    *,
    raw_truth_bytes: bytes,
    scientific_forecast: DualModelForecast,
    scientific_locator: Frozen25kmCellLocator,
) -> str:
    """Build a fully offline, separate synthetic mature replay explorer."""

    replay = _normalise_replay(
        _mapping(replay_mapping, "replay"),
        raw_truth_bytes=raw_truth_bytes,
        scientific_forecast=scientific_forecast,
        scientific_locator=scientific_locator,
    )
    payload = _safe_json(_replay_json(replay))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1-0C 合成原始字节成熟回放</title>
<style>
:root{{--ink:#17243a;--muted:#5c6d82;--paper:#fff;--bg:#eef3f8;--line:#cbd7e4}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}}main{{max-width:1080px;margin:0 auto;padding:24px}}header,.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}h1{{font-size:27px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 10px}}p{{line-height:1.55}}.warning{{color:#a33226;font-weight:700}}.muted{{color:var(--muted)}}.controls{{display:flex;gap:18px;align-items:end;flex-wrap:wrap}}label{{display:grid;gap:6px;font-weight:650}}.toggle{{display:flex;gap:7px;align-items:center;padding:8px 0}}select{{font:inherit;padding:8px;border:1px solid #aebdcc;border-radius:7px;background:#fff}}.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(270px,1fr);gap:16px}}canvas{{display:block;width:100%;height:auto;background:#f9fbfd;border:1px solid var(--line);border-radius:8px}}.metric{{background:#f4f7fb;border-radius:8px;padding:12px;margin-bottom:9px}}.metric strong{{display:block;font-size:20px;margin-top:4px}}code{{overflow-wrap:anywhere}}@media(max-width:760px){{.layout{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main data-artifact="p1-0c-synthetic-mature-replay-offline" data-forecast-sha256="{replay.forecast_sha256}">
<header><h1>P1-0C 合成原始字节成熟回放</h1><p class="warning">另存回放，不是真实预测效果证据</p><p>原起报 SVG SHA-256：<code>{replay.forecast_sha256}</code><br>合成原始响应 SHA-256：<code>{replay.synthetic_raw_response_sha256}</code>（{replay.synthetic_raw_response_byte_count:,} 字节）<br>分群 SHA-256：<code>{replay.cluster_assignment_sha256}</code><br>配对评分 SHA-256：<code>{replay.ordered_cluster_registry_sha256}</code></p><p>底图颜色沿用原起报的 <code>normalized_cell_mass / 实际格面积</code> 单位面积相对强度及双模型共享色标。</p><p class="muted">{replay.horizon_days} 天窗口 · 成熟时刻 {html.escape(replay.mature_after_utc)} · 回放形成时刻 {html.escape(replay.replay_created_at_utc)}</p></header>
<section class="card controls" aria-label="成熟回放控件"><label for="model-select">模型<select id="model-select"><option value="B0">B0</option><option value="B0_R30">B0_R30</option></select></label><label class="toggle" for="intensity-toggle"><input id="intensity-toggle" type="checkbox" checked>显示单位面积相对强度</label><label class="toggle" for="alarm-toggle"><input id="alarm-toggle" type="checkbox" checked>显示报警格</label><label class="toggle" for="cluster-toggle"><input id="cluster-toggle" type="checkbox" checked>显示合成成熟震群</label></section>
<section class="layout"><div class="card"><h2 id="view-title">成熟后回放</h2><canvas id="grid" width="760" height="590" data-layer-relative-intensity="toggle" data-layer-alarm="toggle" data-layer-synthetic-clusters="toggle"></canvas><p class="muted">绿色圆点为命中，红叉为漏报；底图与报警掩膜来自绑定的原起报 SVG。</p></div><aside class="card"><h2>已知答案结果</h2><div class="metric">召回<strong id="recall-value">—</strong></div><div class="metric">命中 / 漏报<strong id="hit-miss-value">—</strong></div><div class="metric">两模型召回变化<strong id="gain-value">—</strong></div><div class="metric">cluster_10 冻结复审<strong>{html.escape(str(replay.review["decision"]))}</strong></div><p>B0 / B0_R30 命中：{replay.review["B0_hit_clusters"]} / {replay.review["B0_R30_hit_clusters"]}</p><p>实际报警面积：<span id="area-value">—</span></p><p class="muted">这些数值只验证合成原始字节到分群、命中和展示的逻辑，不构成真实预测效果。</p></aside></section>
<section class="card"><h2>不可覆盖关系</h2><p>本文件是另增的成熟回放，引用原起报 SHA-256；不得覆盖或把成熟层写回原起报 SVG/HTML。相对强度不是绝对发震概率。</p></section>
<script type="application/json" id="replay-data">{payload}</script>
<script>
"use strict";
const replay=JSON.parse(document.getElementById("replay-data").textContent),forecast=replay.forecast;
const modelSelect=document.getElementById("model-select"),intensityToggle=document.getElementById("intensity-toggle"),alarmToggle=document.getElementById("alarm-toggle"),clusterToggle=document.getElementById("cluster-toggle"),canvas=document.getElementById("grid"),context=canvas.getContext("2d");
function fmt(value){{return new Intl.NumberFormat("zh-CN",{{maximumFractionDigits:3}}).format(value)}}
function colour(value){{const ratio=forecast.shared_colour_maximum>0?Math.max(0,Math.min(1,value/forecast.shared_colour_maximum)):0,low=[238,246,255],high=[30,92,157];return "rgb("+low.map((item,index)=>Math.round(item+ratio*(high[index]-item))).join(",")+")"}}
function cross(x,y,r){{context.strokeStyle="#c33b32";context.lineWidth=3;context.lineCap="round";context.beginPath();context.moveTo(x-r,y-r);context.lineTo(x+r,y+r);context.moveTo(x+r,y-r);context.lineTo(x-r,y+r);context.stroke()}}
function render(){{const model=forecast.models[modelSelect.value],score=replay.model_scores[modelSelect.value],alarms=new Set(model.alarm_cell_ids),columns=forecast.maximum_column-forecast.minimum_column+1,rows=forecast.maximum_row-forecast.minimum_row+1,padding=34,scale=Math.min((canvas.width-padding*2)/columns,(canvas.height-padding*2)/rows),originX=(canvas.width-columns*scale)/2,originY=(canvas.height-rows*scale)/2;context.clearRect(0,0,canvas.width,canvas.height);context.fillStyle="#f9fbfd";context.fillRect(0,0,canvas.width,canvas.height);forecast.cells.forEach((cell,index)=>{{const x=originX+(cell.column-forecast.minimum_column)*scale,y=originY+(cell.row-forecast.minimum_row)*scale,isAlarm=alarms.has(cell.cell_id);context.fillStyle=intensityToggle.checked?colour(model.relative_intensity_per_km2[index]):"#fff";context.fillRect(x,y,scale,scale);context.strokeStyle=alarmToggle.checked&&isAlarm?"#e39400":"#c8d3df";context.lineWidth=alarmToggle.checked&&isAlarm?2.5:1;context.strokeRect(x,y,scale,scale)}});if(clusterToggle.checked){{const occurrences={{}};replay.clusters.forEach(cluster=>{{const cell=forecast.cells.find(item=>item.cell_id===cluster.cell_id),count=occurrences[cell.cell_id]||0;occurrences[cell.cell_id]=count+1;const offset=((count%3)-1)*Math.min(4,scale*.15),x=originX+(cell.column-forecast.minimum_column+.5)*scale+offset,y=originY+(cell.row-forecast.minimum_row+.5)*scale-offset,r=Math.max(4,Math.min(7,scale*.22));if(alarms.has(cell.cell_id)){{context.fillStyle="#138a53";context.strokeStyle="#fff";context.lineWidth=1.5;context.beginPath();context.arc(x,y,r,0,Math.PI*2);context.fill();context.stroke()}}else{{cross(x,y,r)}}}})}}document.getElementById("view-title").textContent=modelSelect.value+" · 合成成熟震群回放（单位面积相对强度共享色标）";document.getElementById("recall-value").textContent=score.recall===null?"无合成震群":(score.recall*100).toFixed(1)+"%";document.getElementById("hit-miss-value").textContent=score.hit_count+" / "+score.miss_count;const b0=replay.model_scores.B0.recall,challenger=replay.model_scores.B0_R30.recall,gain=b0===null||challenger===null?null:(challenger-b0)*100;document.getElementById("gain-value").textContent=gain===null?"不可计算":(gain>=0?"+":"")+gain.toFixed(1)+" 个百分点";document.getElementById("area-value").textContent=fmt(model.actual_alarm_area_km2)+" km²"}}
modelSelect.addEventListener("change",render);intensityToggle.addEventListener("change",render);alarmToggle.addEventListener("change",render);clusterToggle.addEventListener("change",render);render();
</script>
</main>
</body>
</html>
"""


__all__ = [
    "P1PreflightRenderingError",
    "build_preflight_forecast_html",
    "build_preflight_mature_replay_html",
    "render_preflight_forecast_svg",
    "render_preflight_mature_replay_svg",
]
