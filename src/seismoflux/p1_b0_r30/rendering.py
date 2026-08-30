# ruff: noqa: E501, RUF001
"""Deterministic, synthetic-only views for the P1 B0 versus B0_R30 check.

The functions in this module deliberately accept plain mappings and sequences.
They do not read files, contact a network service, or know anything about the
real prospective catalogue.  The visual contract is therefore usable by the
synthetic runner without coupling the scientific calculation to a UI library.

Input mappings follow ``SyntheticScenarioResult.as_mapping()`` and contain:

* ``scenario_id``, ``label``, ``expected_direction`` and ``interpretation``;
* a regular ``grid`` with cells ordered exactly as each intensity array;
* ``models.B0`` and ``models.B0_R30``;
* synthetic ``targets`` and a ``comparison`` summary.

Both outputs say prominently that they are synthetic demonstrations and not
real forecasts.  Values are described as relative intensity, never as an
absolute chance of an earthquake.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_MODEL_IDS = ("B0", "B0_R30")
_DIRECTION_ORDER = {"positive": 0, "zero": 1, "negative": 2}
_DIRECTION_LABELS = {
    "positive": "正向情景",
    "zero": "零增益情景",
    "negative": "负向情景",
}
_SYNTHETIC_WARNING = "纯合成演示，不是真实预测"
_ENGLISH_WARNING = "SYNTHETIC / NOT A REAL FORECAST"


class P1SyntheticRenderingError(ValueError):
    """Raised when a synthetic visualization payload is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class _Cell:
    cell_id: str
    row: int
    column: int
    x_km: float
    y_km: float
    area_km2: float


@dataclass(frozen=True, slots=True)
class _Target:
    cluster_id: str
    event_id: str
    x_km: float
    y_km: float
    row: int
    column: int
    B0_hit: bool
    B0_R30_hit: bool


@dataclass(frozen=True, slots=True)
class _Model:
    model_id: str
    relative_intensity: tuple[float, ...]
    alarm_cell_ids: frozenset[str]
    alarm_area_km2: float
    next_complete_cell_area_km2: float
    declared_alarm_area_km2: float
    hit_cluster_ids: frozenset[str]
    missed_cluster_ids: frozenset[str]
    recall: float | None


@dataclass(frozen=True, slots=True)
class _ForecastLayer:
    model_id: str
    relative_intensity: tuple[float, ...]
    ranked_cell_ids: tuple[str, ...]
    rank_by_cell: Mapping[str, int]
    alarm_cell_ids: frozenset[str]
    alarm_area_km2: float
    next_complete_cell_area_km2: float


@dataclass(frozen=True, slots=True)
class _ForecastView:
    issue_id: str
    scheduled_issue_time_utc: str
    query_cutoff_utc: str
    B0_active_event_count: int
    R30_active_event_count: int
    rows: int
    columns: int
    cell_size_km: float
    cells: tuple[_Cell, ...]
    models: tuple[_ForecastLayer, _ForecastLayer]


@dataclass(frozen=True, slots=True)
class _Scenario:
    scenario_id: str
    label: str
    expected_direction: str
    interpretation: str
    query_cutoff_utc: str
    rows: int
    columns: int
    cell_size_km: float
    cells: tuple[_Cell, ...]
    targets: tuple[_Target, ...]
    models: tuple[_Model, _Model]
    comparison: Mapping[str, object]


def _as_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P1SyntheticRenderingError(f"{path} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise P1SyntheticRenderingError(f"{path} keys must be strings")
    return value


def _as_sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise P1SyntheticRenderingError(f"{path} must be a sequence")
    return value


def _as_text(value: object, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise P1SyntheticRenderingError(f"{path} must be a non-empty string")
    return value


def _as_utc_timestamp(value: object, path: str) -> tuple[str, datetime]:
    text = _as_text(value, path)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise P1SyntheticRenderingError(f"{path} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise P1SyntheticRenderingError(f"{path} must include an UTC offset")
    utc = parsed.astimezone(UTC)
    canonical = utc.isoformat().replace("+00:00", "Z")
    return canonical, utc


def _as_int(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise P1SyntheticRenderingError(f"{path} must be an integer >= {minimum}")
    return value


def _as_number(value: object, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise P1SyntheticRenderingError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise P1SyntheticRenderingError(f"{path} is outside the allowed range")
    return result


def _as_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise P1SyntheticRenderingError(f"{path} must be boolean")
    return value


def _as_optional_fraction(value: object, path: str) -> float | None:
    if value is None:
        return None
    result = _as_number(value, path, minimum=0.0)
    if result > 1.0:
        raise P1SyntheticRenderingError(f"{path} must be <= 1")
    return result


def _as_string_tuple(value: object, path: str) -> tuple[str, ...]:
    items = _as_sequence(value, path)
    result = tuple(_as_text(item, f"{path}[{index}]") for index, item in enumerate(items))
    if len(result) != len(set(result)):
        raise P1SyntheticRenderingError(f"{path} contains duplicate identifiers")
    return result


def _containing_cell(
    target_x: float,
    target_y: float,
    cells: Sequence[_Cell],
    *,
    path: str,
) -> _Cell:
    matches: list[_Cell] = []
    for cell in cells:
        half_side = math.sqrt(cell.area_km2) / 2.0
        if (
            cell.x_km - half_side <= target_x < cell.x_km + half_side
            and cell.y_km - half_side <= target_y < cell.y_km + half_side
        ):
            matches.append(cell)
    if len(matches) != 1:
        raise P1SyntheticRenderingError(f"{path} must fall inside exactly one synthetic grid cell")
    return matches[0]


def _normalise_grid(
    value: object,
    *,
    path: str,
) -> tuple[int, int, float, tuple[_Cell, ...]]:
    grid = _as_mapping(value, path)
    rows = _as_int(grid.get("rows"), f"{path}.rows", minimum=1)
    columns = _as_int(grid.get("columns"), f"{path}.columns", minimum=1)
    cell_size_km = _as_number(grid.get("cell_size_km"), f"{path}.cell_size_km", minimum=1e-12)
    if not math.isclose(cell_size_km, 25.0, rel_tol=0.0, abs_tol=1e-12):
        raise P1SyntheticRenderingError(f"{path}.cell_size_km must equal the frozen 25 km")
    raw_cells = _as_sequence(grid.get("cells"), f"{path}.cells")
    cells_list: list[_Cell] = []
    for index, item in enumerate(raw_cells):
        cell_path = f"{path}.cells[{index}]"
        raw_cell = _as_mapping(item, cell_path)
        cell = _Cell(
            cell_id=_as_text(raw_cell.get("cell_id"), f"{cell_path}.cell_id"),
            row=_as_int(raw_cell.get("row"), f"{cell_path}.row"),
            column=_as_int(raw_cell.get("column"), f"{cell_path}.column"),
            x_km=_as_number(raw_cell.get("x_km"), f"{cell_path}.x_km"),
            y_km=_as_number(raw_cell.get("y_km"), f"{cell_path}.y_km"),
            area_km2=_as_number(raw_cell.get("area_km2"), f"{cell_path}.area_km2", minimum=1e-12),
        )
        if cell.row >= rows or cell.column >= columns:
            raise P1SyntheticRenderingError(f"{cell_path} lies outside grid dimensions")
        if cell.area_km2 > 625.0:
            raise P1SyntheticRenderingError(
                f"{cell_path}.area_km2 must be in the frozen interval (0, 625]"
            )
        cells_list.append(cell)
    canonical_cells = tuple(
        sorted(
            cells_list,
            key=lambda cell: (cell.row, cell.column, cell.cell_id.encode("utf-8")),
        )
    )
    if tuple(cells_list) != canonical_cells:
        raise P1SyntheticRenderingError(
            f"{path}.cells must be in canonical row/column/cell_id order "
            "so intensity arrays cannot become misaligned"
        )
    cells = tuple(cells_list)
    if len(cells) != rows * columns:
        raise P1SyntheticRenderingError(f"{path} must contain every regular-grid cell")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise P1SyntheticRenderingError(f"{path} contains duplicate cell_id values")
    if len({(cell.row, cell.column) for cell in cells}) != len(cells):
        raise P1SyntheticRenderingError(f"{path} contains duplicate row/column cells")
    return rows, columns, cell_size_km, cells


def _normalise_forecast_layer(
    value: object,
    *,
    model_id: str,
    cells: tuple[_Cell, ...],
) -> _ForecastLayer:
    """Validate only information that exists at the synthetic issue instant."""

    path = f"models.{model_id}"
    raw = _as_mapping(value, path)
    if "model_id" in raw and raw.get("model_id") != model_id:
        raise P1SyntheticRenderingError(f"{path}.model_id disagrees with its mapping key")
    intensity_raw = _as_sequence(raw.get("relative_intensity"), f"{path}.relative_intensity")
    relative_intensity = tuple(
        _as_number(item, f"{path}.relative_intensity[{index}]", minimum=0.0)
        for index, item in enumerate(intensity_raw)
    )
    if len(relative_intensity) != len(cells):
        raise P1SyntheticRenderingError(
            f"{path}.relative_intensity must align one-to-one with grid.cells"
        )
    if not any(value > 0.0 for value in relative_intensity):
        raise P1SyntheticRenderingError(f"{path}.relative_intensity must contain signal")
    if not math.isclose(math.fsum(relative_intensity), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise P1SyntheticRenderingError(f"{path}.relative_intensity must sum to one")

    alarm_cell_id_sequence = _as_string_tuple(raw.get("alarm_cell_ids"), f"{path}.alarm_cell_ids")
    alarm_cell_ids = frozenset(alarm_cell_id_sequence)
    valid_cell_ids = {cell.cell_id for cell in cells}
    if not alarm_cell_ids <= valid_cell_ids:
        raise P1SyntheticRenderingError(f"{path}.alarm_cell_ids contains an unknown cell")
    ranking = tuple(
        cells[index].cell_id
        for index in sorted(
            range(len(cells)),
            key=lambda index: (
                -relative_intensity[index] / cells[index].area_km2,
                cells[index].row,
                cells[index].column,
                cells[index].cell_id.encode("utf-8"),
            ),
        )
    )
    if alarm_cell_id_sequence != ranking[: len(alarm_cell_id_sequence)]:
        raise P1SyntheticRenderingError(
            f"{path}.alarm_cell_ids must be the complete, unbroken stable-ranking prefix"
        )
    declared_area = _as_number(raw.get("alarm_area_km2"), f"{path}.alarm_area_km2", minimum=0.0)
    actual_area = math.fsum(cell.area_km2 for cell in cells if cell.cell_id in alarm_cell_ids)
    area_tolerance = max(1e-7, abs(actual_area) * 1e-9)
    if not math.isclose(declared_area, actual_area, rel_tol=0.0, abs_tol=area_tolerance):
        raise P1SyntheticRenderingError(f"{path}.alarm_area_km2 disagrees with the selected cells")
    next_index = len(alarm_cell_id_sequence)
    if next_index >= len(ranking):
        raise P1SyntheticRenderingError(f"{path} must leave a next complete cell for fairness")
    cells_by_id = {cell.cell_id: cell for cell in cells}
    expected_next_area = cells_by_id[ranking[next_index]].area_km2
    declared_next_area = _as_number(
        raw.get("next_complete_cell_area_km2"),
        f"{path}.next_complete_cell_area_km2",
        minimum=1e-12,
    )
    if not math.isclose(declared_next_area, expected_next_area, rel_tol=0.0, abs_tol=1e-9):
        raise P1SyntheticRenderingError(
            f"{path}.next_complete_cell_area_km2 disagrees with the stable ranking"
        )
    return _ForecastLayer(
        model_id=model_id,
        relative_intensity=relative_intensity,
        ranked_cell_ids=ranking,
        rank_by_cell={cell_id: index + 1 for index, cell_id in enumerate(ranking)},
        alarm_cell_ids=alarm_cell_ids,
        alarm_area_km2=actual_area,
        next_complete_cell_area_km2=expected_next_area,
    )


def _normalise_model(
    value: object,
    *,
    model_id: str,
    cells: tuple[_Cell, ...],
    targets: tuple[_Target, ...],
) -> _Model:
    path = f"models.{model_id}"
    raw = _as_mapping(value, path)
    forecast = _normalise_forecast_layer(value, model_id=model_id, cells=cells)
    alarm_cell_ids = forecast.alarm_cell_ids

    hit_ids = frozenset(_as_string_tuple(raw.get("hit_cluster_ids"), f"{path}.hit_cluster_ids"))
    missed_ids = frozenset(
        _as_string_tuple(raw.get("missed_cluster_ids"), f"{path}.missed_cluster_ids")
    )
    if hit_ids & missed_ids:
        raise P1SyntheticRenderingError(f"{path} marks a target as both hit and missed")
    target_ids = frozenset(target.cluster_id for target in targets)
    if hit_ids | missed_ids != target_ids:
        raise P1SyntheticRenderingError(f"{path} must classify every synthetic target")
    # Do not infer a target's cell from its identifier.  The frozen row/column
    # mapping is the authority even when identifiers are opaque.
    cell_ids_by_position = {(cell.row, cell.column): cell.cell_id for cell in cells}
    expected_hit_ids = frozenset(
        target.cluster_id
        for target in targets
        if cell_ids_by_position[(target.row, target.column)] in alarm_cell_ids
    )
    if hit_ids != expected_hit_ids or missed_ids != target_ids - expected_hit_ids:
        raise P1SyntheticRenderingError(
            f"{path} hit/missed lists disagree with target cells and the alarm prefix"
        )
    for target in targets:
        declared_hit = target.B0_hit if model_id == "B0" else target.B0_R30_hit
        if declared_hit != (target.cluster_id in expected_hit_ids):
            raise P1SyntheticRenderingError(
                f"{path} disagrees with the per-target hit flag for {target.cluster_id}"
            )

    recall = _as_optional_fraction(raw.get("recall"), f"{path}.recall")
    expected_recall = None if not target_ids else len(expected_hit_ids) / len(target_ids)
    if expected_recall is None:
        if recall is not None:
            raise P1SyntheticRenderingError(f"{path}.recall must be null when there are no targets")
    elif recall is None or not math.isclose(recall, expected_recall, abs_tol=1e-12):
        raise P1SyntheticRenderingError(f"{path}.recall disagrees with hit/missed targets")

    return _Model(
        model_id=model_id,
        relative_intensity=forecast.relative_intensity,
        alarm_cell_ids=alarm_cell_ids,
        alarm_area_km2=forecast.alarm_area_km2,
        next_complete_cell_area_km2=forecast.next_complete_cell_area_km2,
        declared_alarm_area_km2=forecast.alarm_area_km2,
        hit_cluster_ids=hit_ids,
        missed_cluster_ids=missed_ids,
        recall=recall,
    )


def _validated_area_difference(
    B0: _ForecastLayer | _Model,
    challenger: _ForecastLayer | _Model,
    *,
    path: str,
    total_grid_area_km2: float,
) -> tuple[float, float]:
    difference = B0.alarm_area_km2 - challenger.alarm_area_km2
    tolerance = max(1e-7, B0.alarm_area_km2 * 1e-9)
    if B0.alarm_area_km2 > 600_000.0 + tolerance:
        raise P1SyntheticRenderingError(
            f"{path}.B0 actual alarm area exceeds the frozen 600000 km2 cap"
        )
    if (
        total_grid_area_km2 > 600_000.0 + tolerance
        and B0.alarm_area_km2 + B0.next_complete_cell_area_km2 <= 600_000.0 + tolerance
    ):
        raise P1SyntheticRenderingError(
            f"{path}.B0 alarm is not the largest complete-cell prefix within 600000 km2"
        )
    if (
        challenger.alarm_area_km2 + challenger.next_complete_cell_area_km2
        <= B0.alarm_area_km2 + tolerance
    ):
        raise P1SyntheticRenderingError(
            f"{path}.B0_R30 alarm is not the largest complete-cell prefix within B0 area"
        )
    if difference < -tolerance:
        raise P1SyntheticRenderingError(f"{path}.B0_R30 may not alarm more area than B0")
    if not math.isclose(difference, 0.0, abs_tol=tolerance) and not (0.0 < difference < 625.0):
        raise P1SyntheticRenderingError(
            f"{path} alarm-area difference must be zero or strictly below 625 km2"
        )
    if not math.isclose(difference, 0.0, abs_tol=tolerance) and not (
        difference < challenger.next_complete_cell_area_km2 - tolerance
    ):
        raise P1SyntheticRenderingError(
            f"{path} alarm-area difference must be smaller than the challenger next complete cell"
        )
    return difference, tolerance


def _normalise_forecast_view(value: Mapping[str, object]) -> _ForecastView:
    """Read the issue-time subset and intentionally ignore every future field."""

    issue_id = _as_text(value.get("issue_id"), "issue_id")
    scheduled_text, scheduled = _as_utc_timestamp(
        value.get("scheduled_issue_time_utc"), "scheduled_issue_time_utc"
    )
    cutoff, cutoff_time = _as_utc_timestamp(value.get("query_cutoff_utc"), "query_cutoff_utc")
    if scheduled - cutoff_time != timedelta(minutes=15):
        raise P1SyntheticRenderingError(
            "query_cutoff_utc must be exactly 15 minutes before scheduled_issue_time_utc"
        )
    rows, columns, cell_size_km, cells = _normalise_grid(value.get("grid"), path="grid")
    raw_models = _as_mapping(value.get("models"), "models")
    if set(raw_models) != set(_MODEL_IDS):
        raise P1SyntheticRenderingError("models must contain exactly B0 and B0_R30")
    components = _as_mapping(value.get("components"), "components")
    B0_component = _as_mapping(components.get("B0"), "components.B0")
    B0_active_event_count = _as_int(
        B0_component.get("active_event_count"),
        "components.B0.active_event_count",
        minimum=1,
    )
    R30 = _as_mapping(components.get("R30"), "components.R30")
    R30_active_event_count = _as_int(
        R30.get("active_event_count"), "components.R30.active_event_count"
    )
    models = tuple(
        _normalise_forecast_layer(raw_models.get(model_id), model_id=model_id, cells=cells)
        for model_id in _MODEL_IDS
    )
    _validated_area_difference(
        models[0],
        models[1],
        path="models",
        total_grid_area_km2=math.fsum(cell.area_km2 for cell in cells),
    )
    return _ForecastView(
        issue_id=issue_id,
        scheduled_issue_time_utc=scheduled_text,
        query_cutoff_utc=cutoff,
        B0_active_event_count=B0_active_event_count,
        R30_active_event_count=R30_active_event_count,
        rows=rows,
        columns=columns,
        cell_size_km=cell_size_km,
        cells=cells,
        models=(models[0], models[1]),
    )


def _normalise_scenario(value: Mapping[str, object]) -> _Scenario:
    scenario_id = _as_text(value.get("scenario_id"), "scenario_id")
    label = _as_text(value.get("label"), f"{scenario_id}.label")
    direction = _as_text(value.get("expected_direction"), f"{scenario_id}.expected_direction")
    if direction not in _DIRECTION_ORDER:
        raise P1SyntheticRenderingError(
            f"{scenario_id}.expected_direction must be positive, zero, or negative"
        )
    interpretation = _as_text(value.get("interpretation"), f"{scenario_id}.interpretation")
    cutoff = _as_text(value.get("query_cutoff_utc"), f"{scenario_id}.query_cutoff_utc")

    rows, columns, cell_size_km, cells = _normalise_grid(
        value.get("grid"), path=f"{scenario_id}.grid"
    )

    raw_targets = _as_sequence(value.get("targets"), f"{scenario_id}.targets")
    targets_list: list[_Target] = []
    for index, item in enumerate(raw_targets):
        path = f"{scenario_id}.targets[{index}]"
        raw_target = _as_mapping(item, path)
        x_km = _as_number(raw_target.get("x_km"), f"{path}.x_km")
        y_km = _as_number(raw_target.get("y_km"), f"{path}.y_km")
        containing = _containing_cell(x_km, y_km, cells, path=path)
        targets_list.append(
            _Target(
                cluster_id=_as_text(raw_target.get("cluster_id"), f"{path}.cluster_id"),
                event_id=_as_text(raw_target.get("event_id"), f"{path}.event_id"),
                x_km=x_km,
                y_km=y_km,
                row=containing.row,
                column=containing.column,
                B0_hit=_as_bool(raw_target.get("B0_hit"), f"{path}.B0_hit"),
                B0_R30_hit=_as_bool(raw_target.get("B0_R30_hit"), f"{path}.B0_R30_hit"),
            )
        )
    targets = tuple(sorted(targets_list, key=lambda target: (target.cluster_id, target.event_id)))
    if len({target.cluster_id for target in targets}) != len(targets):
        raise P1SyntheticRenderingError(
            f"{scenario_id}.targets contains duplicate cluster_id values"
        )
    raw_models = _as_mapping(value.get("models"), f"{scenario_id}.models")
    extra_models = set(raw_models) - set(_MODEL_IDS)
    if extra_models:
        raise P1SyntheticRenderingError(f"{scenario_id}.models contains an unfrozen model")
    models = tuple(
        _normalise_model(raw_models.get(model_id), model_id=model_id, cells=cells, targets=targets)
        for model_id in _MODEL_IDS
    )
    B0, challenger = models
    area_difference, area_tolerance = _validated_area_difference(
        B0,
        challenger,
        path=f"{scenario_id}.models",
        total_grid_area_km2=math.fsum(cell.area_km2 for cell in cells),
    )

    cluster_count = len(targets)
    B0_hit_count = len(B0.hit_cluster_ids)
    challenger_hit_count = len(challenger.hit_cluster_ids)
    derived_gain = (
        None
        if cluster_count == 0
        else 100.0 * (challenger_hit_count - B0_hit_count) / cluster_count
    )
    derived_direction = (
        "zero"
        if derived_gain is None or math.isclose(derived_gain, 0.0, abs_tol=1e-12)
        else ("positive" if derived_gain > 0.0 else "negative")
    )
    if direction != derived_direction:
        raise P1SyntheticRenderingError(
            f"{scenario_id}.expected_direction disagrees with recomputed paired recall"
        )
    if "observed_direction" in value and value.get("observed_direction") != derived_direction:
        raise P1SyntheticRenderingError(
            f"{scenario_id}.observed_direction disagrees with recomputed paired recall"
        )

    raw_comparison = _as_mapping(value.get("comparison"), f"{scenario_id}.comparison")
    declared_cluster_count = _as_int(
        raw_comparison.get("cluster_count"), f"{scenario_id}.comparison.cluster_count"
    )
    declared_B0_hits = _as_int(
        raw_comparison.get("B0_hit_clusters"),
        f"{scenario_id}.comparison.B0_hit_clusters",
    )
    declared_challenger_hits = _as_int(
        raw_comparison.get("B0_R30_hit_clusters"),
        f"{scenario_id}.comparison.B0_R30_hit_clusters",
    )
    if (
        declared_cluster_count != cluster_count
        or declared_B0_hits != B0_hit_count
        or declared_challenger_hits != challenger_hit_count
    ):
        raise P1SyntheticRenderingError(
            f"{scenario_id}.comparison counts disagree with recomputed target outcomes"
        )
    declared_gain_raw = raw_comparison.get("recall_gain_percentage_points")
    if derived_gain is None:
        if declared_gain_raw is not None:
            raise P1SyntheticRenderingError(
                f"{scenario_id}.comparison recall gain must be null without targets"
            )
    else:
        declared_gain = _as_number(
            declared_gain_raw,
            f"{scenario_id}.comparison.recall_gain_percentage_points",
        )
        if not math.isclose(declared_gain, derived_gain, abs_tol=1e-12):
            raise P1SyntheticRenderingError(
                f"{scenario_id}.comparison recall gain disagrees with recomputed recall"
            )
    declared_area_difference = _as_number(
        raw_comparison.get("actual_area_difference_km2"),
        f"{scenario_id}.comparison.actual_area_difference_km2",
    )
    if not math.isclose(
        declared_area_difference, area_difference, rel_tol=0.0, abs_tol=area_tolerance
    ):
        raise P1SyntheticRenderingError(
            f"{scenario_id}.comparison area difference disagrees with selected cells"
        )
    declared_fairness = _as_text(
        raw_comparison.get("area_fairness_status"),
        f"{scenario_id}.comparison.area_fairness_status",
    )
    if declared_fairness != "passed":
        raise P1SyntheticRenderingError(
            f"{scenario_id}.comparison must report the recomputable passed fairness state"
        )
    comparison: dict[str, object] = {
        "cluster_count": cluster_count,
        "B0_hit_clusters": B0_hit_count,
        "B0_R30_hit_clusters": challenger_hit_count,
        "recall_gain_percentage_points": derived_gain,
        "actual_area_difference_km2": area_difference,
        "area_fairness_status": "passed",
    }
    return _Scenario(
        scenario_id=scenario_id,
        label=label,
        expected_direction=direction,
        interpretation=interpretation,
        query_cutoff_utc=cutoff,
        rows=rows,
        columns=columns,
        cell_size_km=cell_size_km,
        cells=cells,
        targets=targets,
        models=(models[0], models[1]),
        comparison=comparison,
    )


def _normalise_scenarios(
    scenarios: Sequence[Mapping[str, object]],
) -> tuple[_Scenario, _Scenario, _Scenario]:
    values = tuple(_normalise_scenario(_as_mapping(item, "scenario")) for item in scenarios)
    if len(values) != 3:
        raise P1SyntheticRenderingError("exactly three synthetic scenarios are required")
    if len({scenario.scenario_id for scenario in values}) != 3:
        raise P1SyntheticRenderingError("scenario_id values must be unique")
    if {scenario.expected_direction for scenario in values} != set(_DIRECTION_ORDER):
        raise P1SyntheticRenderingError(
            "one positive, one zero, and one negative scenario are required"
        )
    ordered = tuple(
        sorted(
            values,
            key=lambda scenario: (
                _DIRECTION_ORDER[scenario.expected_direction],
                scenario.scenario_id,
            ),
        )
    )
    return ordered[0], ordered[1], ordered[2]


def _format_recall(value: float | None) -> str:
    return "无合成目标" if value is None else f"{value * 100:.1f}%"


def _format_number(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, abs_tol=1e-9):
        return f"{rounded:,}"
    return f"{value:,.1f}"


def _cell_colour(value: float, maximum: float) -> str:
    ratio = 0.0 if maximum <= 0.0 else min(1.0, max(0.0, value / maximum))
    low = (238, 246, 255)
    high = (30, 92, 157)
    red = round(low[0] + ratio * (high[0] - low[0]))
    green = round(low[1] + ratio * (high[1] - low[1]))
    blue = round(low[2] + ratio * (high[2] - low[2]))
    return f"rgb({red},{green},{blue})"


def _forecast_panel_svg(
    forecast: _ForecastView,
    model: _ForecastLayer,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    parts = [
        f'<g data-model="{model.model_id}" transform="translate({x:.1f},{y:.1f})">',
        f'<rect x="0" y="0" width="{width:.1f}" height="{height:.1f}" rx="12" '
        'fill="#ffffff" stroke="#ccd6e1"/>',
        f'<text x="18" y="32" class="model">{model.model_id}</text>',
    ]
    max_grid_width = width - 210.0
    max_grid_height = height - 72.0
    scale = min(max_grid_width / forecast.columns, max_grid_height / forecast.rows)
    grid_height = forecast.rows * scale
    grid_x = 18.0
    grid_y = 52.0 + (max_grid_height - grid_height) / 2.0
    maximum = max(model.relative_intensity)
    parts.append('<g data-layer="relative-intensity">')
    for index, cell in enumerate(forecast.cells):
        cell_x = grid_x + cell.column * scale
        cell_y = grid_y + cell.row * scale
        alarm = cell.cell_id in model.alarm_cell_ids
        parts.append(
            f'<rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{scale:.2f}" '
            f'height="{scale:.2f}" fill="{_cell_colour(model.relative_intensity[index], maximum)}" '
            f'stroke="{"#e39400" if alarm else "#d3dde8"}" '
            f'stroke-width="{3.0 if alarm else 0.8:.1f}" '
            f'data-cell-id="{html.escape(cell.cell_id)}" data-alarm="{str(alarm).lower()}">'
            f"<title>格 {html.escape(cell.cell_id)}; 相对强度 "
            f"{model.relative_intensity[index]:.8f}; 排名 {model.rank_by_cell[cell.cell_id]}; "
            f"{'报警格' if alarm else '非报警格'}</title></rect>"
        )
    parts.append("</g>")
    metric_x = width - 170.0
    parts.extend(
        [
            f'<text x="{metric_x:.1f}" y="92" class="metric-label">实际报警面积</text>',
            f'<text x="{metric_x:.1f}" y="120" class="metric-small">'
            f"{_format_number(model.alarm_area_km2)} km²</text>",
            f'<text x="{metric_x:.1f}" y="162" class="metric-label">完整报警格</text>',
            f'<text x="{metric_x:.1f}" y="190" class="metric-small">'
            f"{len(model.alarm_cell_ids):,} 个</text>",
            f'<text x="{metric_x:.1f}" y="232" class="metric-label">数值含义</text>',
            f'<text x="{metric_x:.1f}" y="258" class="note">条件相对强度</text>',
            "</g>",
        ]
    )
    return parts


def render_synthetic_forecast_svg(scenario_mapping: Mapping[str, object]) -> bytes:
    """Render an issue-time synthetic forecast without reading any future field.

    Only issue identity/timing, issue-time data levels, the frozen grid,
    relative-intensity arrays, and alarm prefixes contribute to the bytes.
    Scenario labels, intended direction, future events, and post-maturity
    summaries are deliberately ignored.
    """

    forecast = _normalise_forecast_view(_as_mapping(scenario_mapping, "scenario_mapping"))
    width = 1280
    height = 680
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="forecast-title forecast-desc" '
        'data-artifact="p1-b0-r30-synthetic-forecast">',
        '<title id="forecast-title">P1 B0 与 B0_R30 纯合成预测图</title>',
        '<desc id="forecast-desc">纯合成预测图, 不是真实预测, 不含未来目标。仅展示起报时已冻结的相对强度和报警格。</desc>',
        "<style>",
        "text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#17243a}",
        ".title{font-size:29px;font-weight:700}.warning{font-size:17px;font-weight:700;fill:#a33226}",
        ".subtitle{font-size:14px;fill:#506176}.model{font-size:20px;font-weight:700}",
        ".metric-label{font-size:13px;fill:#65758a}.metric-small{font-size:18px;font-weight:650}",
        ".note{font-size:13px;fill:#506176}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        '<text x="42" y="48" class="title">P1 起报时双模型图</text>',
        '<text x="42" y="80" class="warning">纯合成预测图 · 不是真实预测 · 不含未来目标</text>',
        '<text x="42" y="108" class="subtitle">B0_R30 = 0.75 × B0 + 0.25 × R30；'
        "颜色为条件相对强度，橙框为完整报警格。</text>",
        f'<text x="42" y="134" class="subtitle">Issue = {html.escape(forecast.issue_id)}；'
        f"计划起报 T = {html.escape(forecast.scheduled_issue_time_utc)}</text>",
        f'<text x="42" y="158" class="subtitle">查询截点 Q = '
        f"{html.escape(forecast.query_cutoff_utc)}；数据水位 B0 n={forecast.B0_active_event_count:,}，"
        f"R30 n={forecast.R30_active_event_count:,}。</text>",
    ]
    parts.extend(
        _forecast_panel_svg(
            forecast,
            forecast.models[0],
            x=42.0,
            y=180.0,
            width=580.0,
            height=400.0,
        )
    )
    parts.extend(
        _forecast_panel_svg(
            forecast,
            forecast.models[1],
            x=658.0,
            y=180.0,
            width=580.0,
            height=400.0,
        )
    )
    parts.extend(
        [
            '<text x="42" y="620" class="note">本图只含合成起报时已经可用的信息；'
            "未来观测只能进入另存的成熟后回放，不能改写本图。</text>",
            '<text x="42" y="646" class="note">报警面积以选中格网的实际面积求和；'
            "相对强度与顺位不代表绝对发震机会。</text>",
            "</svg>",
        ]
    )
    return ("\n".join(parts) + "\n").encode("utf-8")


def _forecast_as_json(forecast: _ForecastView) -> dict[str, object]:
    return {
        "issue_id": forecast.issue_id,
        "scheduled_issue_time_utc": forecast.scheduled_issue_time_utc,
        "query_cutoff_utc": forecast.query_cutoff_utc,
        "B0_active_event_count": forecast.B0_active_event_count,
        "R30_active_event_count": forecast.R30_active_event_count,
        "rows": forecast.rows,
        "columns": forecast.columns,
        "cells": [
            {
                "area_km2": cell.area_km2,
                "cell_id": cell.cell_id,
                "column": cell.column,
                "row": cell.row,
            }
            for cell in forecast.cells
        ],
        "models": {
            model.model_id: {
                "alarm_area_km2": model.alarm_area_km2,
                "alarm_cell_ids": sorted(model.alarm_cell_ids),
                "ranked_cell_ids": list(model.ranked_cell_ids),
                "rank_by_cell": dict(model.rank_by_cell),
                "relative_intensity": list(model.relative_intensity),
            }
            for model in forecast.models
        },
    }


def build_offline_synthetic_forecast_html(
    scenario_mapping: Mapping[str, object],
) -> str:
    """Build a target-blind, fully offline issue-time forecast explorer."""

    forecast = _normalise_forecast_view(_as_mapping(scenario_mapping, "scenario_mapping"))
    payload = _safe_embedded_json(_forecast_as_json(forecast))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1 B0 与 B0_R30 纯合成预测图</title>
<style>
:root{{--ink:#17243a;--muted:#5c6d82;--paper:#fff;--bg:#eef3f8;--line:#cbd7e4;--alarm:#e39400}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:24px}}header,.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px}}
h1{{font-size:28px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 12px}}p{{line-height:1.6}}.warning{{color:#a33226;font-weight:750}}.muted{{color:var(--muted)}}
.controls{{display:grid;grid-template-columns:2fr 1fr 1fr;gap:16px;align-items:end}}label{{display:grid;gap:6px;font-weight:650}}select{{font:inherit;padding:9px;border:1px solid #aebdcc;border-radius:8px;background:#fff}}.toggle{{display:flex;gap:8px;align-items:center;padding:9px 0}}
.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(250px,1fr);gap:16px}}canvas{{display:block;width:100%;height:auto;min-height:500px;background:#f9fbfd;border:1px solid var(--line);border-radius:10px}}
.metric{{background:#f4f7fb;border-radius:9px;padding:14px;margin-bottom:10px}}.metric strong{{display:block;font-size:22px;margin-top:5px}}.legend{{font-size:13px;color:var(--muted)}}.inspector{{margin-top:12px;padding:11px;background:#eef4fa;border-radius:8px;font-family:ui-monospace,Consolas,monospace}}@media(max-width:760px){{.controls,.layout{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main data-artifact="p1-b0-r30-synthetic-forecast-offline">
<header><h1>P1 起报时双模型图</h1><p class="warning">纯合成预测图 · 不是真实预测 · 不含未来目标</p><p class="muted">Issue = {html.escape(forecast.issue_id)} · 计划起报 T = {html.escape(forecast.scheduled_issue_time_utc)} · 查询截点 Q = {html.escape(forecast.query_cutoff_utc)}</p><p class="muted">该文件只回放查询截点已经冻结的相对强度与报警格。未来观测不会被嵌入，也不能改写本文件。</p></header>
<section class="card controls" aria-label="预测图控件">
<label for="forecast-model-select">模型<select id="forecast-model-select"><option value="B0">B0</option><option value="B0_R30">B0_R30</option></select></label>
<label class="toggle" for="intensity-toggle"><input id="intensity-toggle" type="checkbox" checked>显示相对强度层</label>
<label class="toggle" for="alarm-toggle"><input id="alarm-toggle" type="checkbox" checked>显示报警格层</label>
</section>
<section class="layout"><div class="card"><h2 id="forecast-view-title">格网图</h2><canvas id="forecast-grid" width="720" height="560" data-layer-relative-intensity="toggle" data-layer-alarm="toggle"></canvas><p class="legend">浅蓝到深蓝表示模型内部条件相对强度由低到高；橙框表示完整报警格。悬停或点击格网可查看冻结排名。</p><div class="inspector" id="cell-inspector" aria-live="polite">悬停或点击一个格网：cell_id、相对强度、排名和报警状态将在这里显示。</div></div>
<aside class="card"><h2>起报时信息</h2><div class="metric">当前模型实际报警面积<strong id="forecast-area">—</strong></div><div class="metric">完整报警格<strong id="forecast-cell-count">—</strong></div><div class="metric">B0 / R30 数据水位<strong>{forecast.B0_active_event_count:,} / {forecast.R30_active_event_count:,}</strong></div><p>B0 实际报警面积：{_format_number(forecast.models[0].alarm_area_km2)} km²<br>B0_R30 实际报警面积：{_format_number(forecast.models[1].alarm_area_km2)} km²</p><p><code>B0_R30 = 0.75 × B0 + 0.25 × R30</code></p><p class="muted" id="forecast-cutoff"></p><p class="muted">相对强度与顺位不代表绝对发震机会。</p></aside></section>
<script type="application/json" id="forecast-data">{payload}</script>
<script>
"use strict";
const forecast=JSON.parse(document.getElementById("forecast-data").textContent);
const modelSelect=document.getElementById("forecast-model-select"),intensityToggle=document.getElementById("intensity-toggle"),alarmToggle=document.getElementById("alarm-toggle");
const canvas=document.getElementById("forecast-grid"),context=canvas.getContext("2d");
const inspector=document.getElementById("cell-inspector");let lastView=null;
function formatNumber(value){{return new Intl.NumberFormat("zh-CN",{{maximumFractionDigits:1}}).format(value)}}
function colour(value,maximum){{const ratio=maximum>0?Math.max(0,Math.min(1,value/maximum)):0,low=[238,246,255],high=[30,92,157];return "rgb("+low.map((item,index)=>Math.round(item+ratio*(high[index]-item))).join(",")+")"}}
function renderForecast(){{const model=forecast.models[modelSelect.value],alarms=new Set(model.alarm_cell_ids),padding=42,scale=Math.min((canvas.width-padding*2)/forecast.columns,(canvas.height-padding*2)/forecast.rows),gridWidth=forecast.columns*scale,gridHeight=forecast.rows*scale,originX=(canvas.width-gridWidth)/2,originY=(canvas.height-gridHeight)/2,maximum=Math.max(...model.relative_intensity);lastView={{model,alarms,scale,originX,originY}};context.clearRect(0,0,canvas.width,canvas.height);context.fillStyle="#f9fbfd";context.fillRect(0,0,canvas.width,canvas.height);forecast.cells.forEach((cell,index)=>{{const x=originX+cell.column*scale,y=originY+cell.row*scale,isAlarm=alarms.has(cell.cell_id);context.fillStyle=intensityToggle.checked?colour(model.relative_intensity[index],maximum):"#ffffff";context.fillRect(x,y,scale,scale);context.strokeStyle=alarmToggle.checked&&isAlarm?"#e39400":"#d3dde8";context.lineWidth=alarmToggle.checked&&isAlarm?3:1;context.strokeRect(x,y,scale,scale)}});document.getElementById("forecast-view-title").textContent=modelSelect.value+" 相对强度与报警格";document.getElementById("forecast-area").textContent=formatNumber(model.alarm_area_km2)+" km²";document.getElementById("forecast-cell-count").textContent=model.alarm_cell_ids.length+" 个";document.getElementById("forecast-cutoff").textContent="查询截点 Q = "+forecast.query_cutoff_utc}}
function inspectCell(event){{if(!lastView)return;const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height,column=Math.floor((x-lastView.originX)/lastView.scale),row=Math.floor((y-lastView.originY)/lastView.scale);if(row<0||row>=forecast.rows||column<0||column>=forecast.columns)return;const index=row*forecast.columns+column,cell=forecast.cells[index],value=lastView.model.relative_intensity[index],rank=lastView.model.rank_by_cell[cell.cell_id],alarm=lastView.alarms.has(cell.cell_id);inspector.textContent="cell_id="+cell.cell_id+" · relative_intensity="+value.toExponential(6)+" · rank="+rank+" · alarm="+(alarm?"是":"否")}}
modelSelect.addEventListener("change",renderForecast);intensityToggle.addEventListener("change",renderForecast);alarmToggle.addEventListener("change",renderForecast);canvas.addEventListener("mousemove",inspectCell);canvas.addEventListener("click",inspectCell);renderForecast();
</script>
</main>
</body>
</html>
"""


def _target_marker_svg(
    *,
    target: _Target,
    model: _Model,
    grid_x: float,
    grid_y: float,
    cell_scale: float,
    offset_index: int,
) -> list[str]:
    offset = ((offset_index % 3) - 1) * min(4.0, cell_scale * 0.12)
    x = grid_x + (target.column + 0.5) * cell_scale + offset
    y = grid_y + (target.row + 0.5) * cell_scale - offset
    safe_id = html.escape(target.cluster_id)
    if target.cluster_id in model.hit_cluster_ids:
        return [
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{max(3.5, cell_scale * 0.12):.2f}" '
            f'fill="#138a53" stroke="#ffffff" stroke-width="1.5" data-target="{safe_id}" '
            'data-outcome="hit"><title>合成目标：命中</title></circle>'
        ]
    radius = max(4.0, cell_scale * 0.13)
    return [
        f'<g data-target="{safe_id}" data-outcome="miss"><title>合成目标：漏报</title>',
        f'<path d="M {x - radius:.2f} {y - radius:.2f} L {x + radius:.2f} {y + radius:.2f} '
        f'M {x + radius:.2f} {y - radius:.2f} L {x - radius:.2f} {y + radius:.2f}" '
        'stroke="#c33b32" stroke-width="3" stroke-linecap="round"/>',
        "</g>",
    ]


def _model_panel_svg(
    scenario: _Scenario,
    model: _Model,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    panel = [
        f'<g data-scenario="{html.escape(scenario.scenario_id)}" '
        f'data-model="{model.model_id}" transform="translate({x:.1f},{y:.1f})">',
        f'<rect x="0" y="0" width="{width:.1f}" height="{height:.1f}" rx="12" '
        'fill="#ffffff" stroke="#ccd6e1"/>',
        f'<text x="18" y="30" class="model">{model.model_id}</text>',
    ]
    max_grid_width = width - 210.0
    max_grid_height = height - 62.0
    scale = min(max_grid_width / scenario.columns, max_grid_height / scenario.rows)
    grid_height = scenario.rows * scale
    grid_x = 18.0
    grid_y = 46.0 + (max_grid_height - grid_height) / 2.0
    maximum = max(model.relative_intensity)
    panel.append('<g data-layer="relative-intensity">')
    for index, cell in enumerate(scenario.cells):
        cell_x = grid_x + cell.column * scale
        cell_y = grid_y + cell.row * scale
        alarm = cell.cell_id in model.alarm_cell_ids
        border = "#e39400" if alarm else "#d3dde8"
        stroke_width = 3.0 if alarm else 1.0
        panel.append(
            f'<rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{scale:.2f}" '
            f'height="{scale:.2f}" fill="{_cell_colour(model.relative_intensity[index], maximum)}" '
            f'stroke="{border}" stroke-width="{stroke_width:.1f}" '
            f'data-cell-id="{html.escape(cell.cell_id)}" data-alarm="{str(alarm).lower()}">'
            f"<title>相对强度 {model.relative_intensity[index]:.6f}; "
            f"{'报警格' if alarm else '非报警格'}</title></rect>"
        )
    panel.append("</g>")
    panel.append('<g data-layer="synthetic-targets">')
    for index, target in enumerate(scenario.targets):
        panel.extend(
            _target_marker_svg(
                target=target,
                model=model,
                grid_x=grid_x,
                grid_y=grid_y,
                cell_scale=scale,
                offset_index=index,
            )
        )
    panel.append("</g>")

    metric_x = width - 176.0
    hits = len(model.hit_cluster_ids)
    misses = len(model.missed_cluster_ids)
    panel.extend(
        [
            f'<text x="{metric_x:.1f}" y="70" class="metric-label">召回</text>',
            f'<text x="{metric_x:.1f}" y="96" class="metric-value">'
            f"{_format_recall(model.recall)}</text>",
            f'<text x="{metric_x:.1f}" y="126" class="metric-label">命中 / 漏报</text>',
            f'<text x="{metric_x:.1f}" y="150" class="metric-small">{hits} / {misses}</text>',
            f'<text x="{metric_x:.1f}" y="180" class="metric-label">实际报警面积</text>',
            f'<text x="{metric_x:.1f}" y="204" class="metric-small">'
            f"{_format_number(model.alarm_area_km2)} km²</text>",
            "</g>",
        ]
    )
    return panel


def render_synthetic_scenarios_svg(
    scenarios: Sequence[Mapping[str, object]],
) -> bytes:
    """Render the positive/zero/negative synthetic comparison as one SVG.

    The result is UTF-8 bytes so a caller can write it atomically without an
    encoding decision.  Scenario ordering is canonical and independent of the
    caller's input order.
    """

    ordered = _normalise_scenarios(scenarios)
    width = 1480
    header_height = 188
    scenario_height = 318
    footer_height = 86
    height = header_height + len(ordered) * scenario_height + footer_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
        'data-artifact="p1-b0-r30-three-scenario-synthetic">',
        '<title id="title">P1 B0 与 B0_R30 纯合成三情景成熟后回放</title>',
        f'<desc id="desc">{_SYNTHETIC_WARNING}。展示相对强度、报警格、合成目标、'
        "命中、漏报、召回和实际报警面积。</desc>",
        "<style>",
        "text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#17243a}",
        ".title{font-size:30px;font-weight:700}.warning{font-size:16px;font-weight:700;fill:#a33226}",
        ".subtitle{font-size:15px;fill:#506176}.scenario{font-size:21px;font-weight:700}",
        ".model{font-size:19px;font-weight:700}.metric-label{font-size:13px;fill:#65758a}",
        ".metric-value{font-size:24px;font-weight:700}.metric-small{font-size:17px;font-weight:650}",
        ".note{font-size:13px;fill:#506176}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        '<text x="42" y="50" class="title">P1 纯合成成熟后回放：B0 对比 B0_R30</text>',
        f'<text x="42" y="82" class="warning">{_SYNTHETIC_WARNING} · {_ENGLISH_WARNING}</text>',
        '<text x="42" y="112" class="subtitle">B0_R30 = 0.75 × B0 + 0.25 × R30；'
        "R30 为 75 km 高斯核的近30天活动相对强度。</text>",
        '<text x="42" y="138" class="subtitle">面积公平：两模型按同一格网和同一报警面积规则选格；'
        "图中报告实际选中面积。</text>",
        '<g aria-label="图例" transform="translate(1018,104)">',
        '<rect x="0" y="0" width="22" height="22" fill="#6b9ac4" stroke="#e39400" stroke-width="3"/>',
        '<text x="30" y="17" class="note">报警格</text>',
        '<circle cx="126" cy="11" r="7" fill="#138a53"/><text x="140" y="17" class="note">命中</text>',
        '<path d="M220 4 L234 18 M234 4 L220 18" stroke="#c33b32" stroke-width="3"/>',
        '<text x="242" y="17" class="note">漏报</text></g>',
    ]

    for scenario_index, scenario in enumerate(ordered):
        top = header_height + scenario_index * scenario_height
        direction_label = _DIRECTION_LABELS[scenario.expected_direction]
        gain = _as_number(
            scenario.comparison.get("recall_gain_percentage_points"),
            f"{scenario.scenario_id}.comparison.recall_gain_percentage_points",
        )
        fairness = _as_text(
            scenario.comparison.get("area_fairness_status"),
            f"{scenario.scenario_id}.comparison.area_fairness_status",
        )
        fairness_label = "通过" if fairness == "passed" else fairness
        parts.extend(
            [
                f'<g data-scenario-row="{html.escape(scenario.scenario_id)}">',
                f'<text x="42" y="{top + 28}" class="scenario">'
                f"{direction_label} · {html.escape(scenario.label)}</text>",
                f'<text x="396" y="{top + 28}" class="note">召回变化 {gain:+.1f} 个百分点 · '
                f"面积公平 {html.escape(fairness_label)}</text>",
                f'<text x="868" y="{top + 28}" class="note">'
                f"{html.escape(scenario.interpretation)}</text>",
            ]
        )
        parts.extend(
            _model_panel_svg(
                scenario,
                scenario.models[0],
                x=42.0,
                y=float(top + 42),
                width=680.0,
                height=252.0,
            )
        )
        parts.extend(
            _model_panel_svg(
                scenario,
                scenario.models[1],
                x=758.0,
                y=float(top + 42),
                width=680.0,
                height=252.0,
            )
        )
        parts.append("</g>")

    footer_y = header_height + len(ordered) * scenario_height + 20
    parts.extend(
        [
            f'<line x1="42" x2="1438" y1="{footer_y}" y2="{footer_y}" stroke="#c7d2df"/>',
            f'<text x="42" y="{footer_y + 28}" class="note">数据说明：仅使用程序生成的合成格网与合成目标；'
            "未读取真实地震目录、异常表或锁定测试。</text>",
            f'<text x="42" y="{footer_y + 52}" class="note">用途：验证计算链能识别“有帮助、无变化、变差”三种已知答案；'
            "这些结果不能证明实际预测有效。</text>",
            "</svg>",
        ]
    )
    return ("\n".join(parts) + "\n").encode("utf-8")


def _scenario_as_json(scenario: _Scenario) -> dict[str, object]:
    cells = [
        {
            "area_km2": cell.area_km2,
            "cell_id": cell.cell_id,
            "column": cell.column,
            "row": cell.row,
        }
        for cell in scenario.cells
    ]
    targets = [
        {
            "cluster_id": target.cluster_id,
            "column": target.column,
            "row": target.row,
        }
        for target in scenario.targets
    ]
    models: dict[str, object] = {}
    for model in scenario.models:
        models[model.model_id] = {
            "alarm_area_km2": model.alarm_area_km2,
            "alarm_cell_ids": sorted(model.alarm_cell_ids),
            "hit_cluster_ids": sorted(model.hit_cluster_ids),
            "missed_cluster_ids": sorted(model.missed_cluster_ids),
            "recall": model.recall,
            "relative_intensity": list(model.relative_intensity),
        }
    return {
        "columns": scenario.columns,
        "comparison": dict(scenario.comparison),
        "expected_direction": scenario.expected_direction,
        "interpretation": scenario.interpretation,
        "label": scenario.label,
        "models": models,
        "query_cutoff_utc": scenario.query_cutoff_utc,
        "rows": scenario.rows,
        "scenario_id": scenario.scenario_id,
        "cells": cells,
        "targets": targets,
    }


def _safe_embedded_json(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        raw.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_offline_synthetic_explorer_html(
    scenarios: Sequence[Mapping[str, object]],
) -> str:
    """Build one self-contained HTML explorer with no external dependency."""

    ordered = _normalise_scenarios(scenarios)
    payload = _safe_embedded_json([_scenario_as_json(scenario) for scenario in ordered])
    options = "".join(
        f'<option value="{html.escape(scenario.scenario_id)}">'
        f"{html.escape(_DIRECTION_LABELS[scenario.expected_direction])} · "
        f"{html.escape(scenario.label)}</option>"
        for scenario in ordered
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1 B0 与 B0_R30 纯合成离线回放</title>
<style>
:root{{--ink:#17243a;--muted:#5c6d82;--paper:#fff;--bg:#eef3f8;--line:#cbd7e4;--alarm:#e39400;--hit:#138a53;--miss:#c33b32}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px}}header,.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px}}
h1{{font-size:28px;margin:0 0 8px}}h2{{font-size:19px;margin:0 0 12px}}p{{line-height:1.6}}.warning{{color:#a33226;font-weight:750;margin:4px 0}}.muted{{color:var(--muted)}}
.controls{{display:grid;grid-template-columns:2fr 1fr 1fr;gap:14px;align-items:end}}label{{display:grid;gap:6px;font-weight:650}}select,input{{font:inherit}}select{{padding:9px;border:1px solid #aebdcc;border-radius:8px;background:white}}
.toggle{{display:flex;gap:8px;align-items:center;padding:9px 0}}.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(270px,1fr);gap:16px}}
canvas{{display:block;width:100%;height:auto;min-height:420px;background:#f9fbfd;border:1px solid var(--line);border-radius:10px}}
.metric-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.metric{{background:#f4f7fb;border-radius:9px;padding:12px}}.metric strong{{display:block;font-size:22px;margin-top:4px}}
.legend{{display:flex;flex-wrap:wrap;gap:18px;font-size:13px;margin-top:12px}}.swatch{{display:inline-block;width:18px;height:18px;vertical-align:middle;margin-right:6px}}
.alarm{{background:#6b9ac4;border:3px solid var(--alarm)}}.hit{{background:var(--hit);border-radius:50%}}.miss{{position:relative}}.miss::before,.miss::after{{content:"";position:absolute;width:20px;height:3px;left:-1px;top:8px;background:var(--miss)}}.miss::before{{transform:rotate(45deg)}}.miss::after{{transform:rotate(-45deg)}}
code{{font-size:14px}}@media(max-width:820px){{.controls,.layout{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main data-artifact="p1-b0-r30-synthetic-offline-explorer">
<header>
<h1>P1 纯合成成熟后回放：B0 对比 B0_R30</h1>
<p class="warning">{_SYNTHETIC_WARNING} · {_ENGLISH_WARNING}</p>
<p class="muted">这是合成目标成熟后的结果回放，用来检查方法能否分清“有帮助、无变化、变差”。它不是起报时原预测图，也不能证明实际预测有效。</p>
</header>
<section class="card controls" aria-label="回放控件">
<label for="scenario-select">合成情景<select id="scenario-select">{options}</select></label>
<label for="model-select">模型<select id="model-select"><option value="B0">B0</option><option value="B0_R30">B0_R30</option></select></label>
<label class="toggle" for="target-toggle"><input id="target-toggle" type="checkbox" checked>显示合成目标层</label>
</section>
<section class="layout">
<div class="card">
<h2 id="view-title">格网回放</h2>
<canvas id="grid-view" width="760" height="520" data-layer-relative-intensity="true" data-layer-alarm="true" data-layer-targets="toggle"></canvas>
<div class="legend" aria-label="图例"><span><i class="swatch alarm"></i>报警格</span><span><i class="swatch hit"></i>命中目标</span><span><i class="swatch miss"></i>漏报目标</span><span>浅蓝 → 深蓝：相对强度由低到高</span></div>
</div>
<aside class="card">
<h2>当前结果</h2>
<div class="metric-grid"><div class="metric">召回<strong id="recall-value">—</strong></div><div class="metric">命中 / 漏报<strong id="hit-miss-value">—</strong></div><div class="metric">实际报警面积<strong id="area-value">—</strong></div><div class="metric">两模型召回变化<strong id="gain-value">—</strong></div></div>
<p id="fairness-value"></p><p id="interpretation-value"></p><p class="muted" id="cutoff-value"></p>
</aside>
</section>
<section class="card" id="method-card">
<h2>方法和数据说明</h2>
<p><strong>组合公式：</strong><code>B0_R30 = 0.75 × B0 + 0.25 × R30</code>；R30 是 75 km 高斯核汇总的近30天活动相对强度。</p>
<p><strong>面积公平：</strong>两个模型使用相同格网和相同报警面积规则；界面同时报告真正选中的格网面积，不用格数冒充面积。</p>
<p><strong>合成数据：</strong>仅使用程序生成的格网和目标，不读取真实地震目录、异常表、人工预测或锁定测试。</p>
<p><strong>读图：</strong>橙框是被模型挑出的报警格；绿色圆点是落入报警格的合成目标，红叉是漏报。颜色只表示同一模型内的相对强弱。</p>
</section>
<script type="application/json" id="scenario-data">{payload}</script>
<script>
"use strict";
const scenarios=JSON.parse(document.getElementById("scenario-data").textContent);
const scenarioSelect=document.getElementById("scenario-select");
const modelSelect=document.getElementById("model-select");
const targetToggle=document.getElementById("target-toggle");
const canvas=document.getElementById("grid-view");
const context=canvas.getContext("2d");
function formatNumber(value){{return new Intl.NumberFormat("zh-CN",{{maximumFractionDigits:1}}).format(value)}}
function cellColour(value,maximum){{const ratio=maximum>0?Math.max(0,Math.min(1,value/maximum)):0;const low=[238,246,255],high=[30,92,157];return "rgb("+low.map((item,index)=>Math.round(item+ratio*(high[index]-item))).join(",")+")"}}
function selectedScenario(){{return scenarios.find(item=>item.scenario_id===scenarioSelect.value)}}
function drawCross(x,y,radius){{context.strokeStyle="#c33b32";context.lineWidth=4;context.lineCap="round";context.beginPath();context.moveTo(x-radius,y-radius);context.lineTo(x+radius,y+radius);context.moveTo(x+radius,y-radius);context.lineTo(x-radius,y+radius);context.stroke()}}
function render(){{
  const scenario=selectedScenario(),model=scenario.models[modelSelect.value];
  const padding=46,availableWidth=canvas.width-padding*2,availableHeight=canvas.height-padding*2;
  const scale=Math.min(availableWidth/scenario.columns,availableHeight/scenario.rows);
  const gridWidth=scenario.columns*scale,gridHeight=scenario.rows*scale;
  const originX=(canvas.width-gridWidth)/2,originY=(canvas.height-gridHeight)/2;
  const maximum=Math.max(...model.relative_intensity),alarms=new Set(model.alarm_cell_ids),hits=new Set(model.hit_cluster_ids);
  context.clearRect(0,0,canvas.width,canvas.height);context.fillStyle="#f9fbfd";context.fillRect(0,0,canvas.width,canvas.height);
  scenario.cells.forEach((cell,index)=>{{const x=originX+cell.column*scale,y=originY+cell.row*scale;context.fillStyle=cellColour(model.relative_intensity[index],maximum);context.fillRect(x,y,scale,scale);context.strokeStyle=alarms.has(cell.cell_id)?"#e39400":"#d3dde8";context.lineWidth=alarms.has(cell.cell_id)?4:1;context.strokeRect(x,y,scale,scale)}});
  if(targetToggle.checked){{scenario.targets.forEach((target,index)=>{{const jitter=((index%3)-1)*Math.min(5,scale*.1),x=originX+(target.column+.5)*scale+jitter,y=originY+(target.row+.5)*scale-jitter,r=Math.max(6,scale*.12);if(hits.has(target.cluster_id)){{context.fillStyle="#138a53";context.strokeStyle="#fff";context.lineWidth=2;context.beginPath();context.arc(x,y,r,0,Math.PI*2);context.fill();context.stroke()}}else{{drawCross(x,y,r)}}}})}}
  const hitCount=model.hit_cluster_ids.length,missCount=model.missed_cluster_ids.length;
  document.getElementById("view-title").textContent=scenario.label+" · "+modelSelect.value;
  document.getElementById("recall-value").textContent=model.recall===null?"无合成目标":(model.recall*100).toFixed(1)+"%";
  document.getElementById("hit-miss-value").textContent=hitCount+" / "+missCount;
  document.getElementById("area-value").textContent=formatNumber(model.alarm_area_km2)+" km²";
  const gain=scenario.comparison.recall_gain_percentage_points;document.getElementById("gain-value").textContent=(gain>=0?"+":"")+gain.toFixed(1)+" 个百分点";
  const fairnessLabel=scenario.comparison.area_fairness_status==="passed"?"通过":scenario.comparison.area_fairness_status;
  document.getElementById("fairness-value").textContent="面积公平检查："+fairnessLabel+"；两模型实际面积差 "+formatNumber(scenario.comparison.actual_area_difference_km2)+" km²。";
  document.getElementById("interpretation-value").textContent="解释："+scenario.interpretation;
  document.getElementById("cutoff-value").textContent="合成查询截点："+scenario.query_cutoff_utc;
}}
scenarioSelect.addEventListener("change",render);modelSelect.addEventListener("change",render);targetToggle.addEventListener("change",render);render();
</script>
</main>
</body>
</html>
"""


def build_offline_explorer_html(scenarios: Sequence[Mapping[str, object]]) -> str:
    """Compatibility name for the P1 synthetic-only offline explorer."""

    return build_offline_synthetic_explorer_html(scenarios)


def render_synthetic_comparison_svg(
    scenarios: Sequence[Mapping[str, object]],
) -> bytes:
    """Compatibility name for the P1 three-scenario static figure."""

    return render_synthetic_scenarios_svg(scenarios)


__all__ = [
    "P1SyntheticRenderingError",
    "build_offline_explorer_html",
    "build_offline_synthetic_explorer_html",
    "build_offline_synthetic_forecast_html",
    "render_synthetic_comparison_svg",
    "render_synthetic_forecast_svg",
    "render_synthetic_scenarios_svg",
]
