# ruff: noqa: E501, RUF001
"""Deterministic static and offline result views for the Stage 2S screen.

The public render boundary accepts an immutable whole-run record plus derived,
coordinate-free map rasters.  It never opens the earthquake catalogue, the
processed-data tree, or a network resource.  The maps encode relative intensity
rank only; they are not absolute earthquake probabilities.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from numpy.typing import ArrayLike, NDArray

from seismoflux.stage2s.records import Stage2SRecordError, Stage2SWholeRunRecord

ModelId = Literal["S0", "S1", "SP"]
ContrastId = Literal["S1_minus_S0", "S1_minus_SP"]
MetricId = Literal["IG", "recall"]

MODEL_IDS: tuple[ModelId, ...] = ("S0", "S1", "SP")
CONTRAST_IDS: tuple[ContrastId, ...] = ("S1_minus_S0", "S1_minus_SP")
METRIC_IDS: tuple[MetricId, ...] = ("IG", "recall")
PROTOCOL_ARTIFACT_NAMES = (
    "S0_S1_SP_same_area_recall_and_information_gain.svg",
    "fold_horizon_region_and_failure_cases.svg",
    "historical_frozen_assessment_relative_intensity_map.svg",
    "historical_frozen_assessment_backtest_explorer.html",
    "historical_frozen_assessment_map_explorer.html",
)
COMPANION_PNG_NAME = "S0_S1_SP_same_area_recall_and_information_gain.png"
ALL_ARTIFACT_NAMES = (*PROTOCOL_ARTIFACT_NAMES, COMPANION_PNG_NAME)
DISPLAY_ALARM_BUDGETS_KM2 = (300_000, 450_000, 600_000, 750_000, 960_000)
FORMAL_ALARM_BUDGET_KM2 = 600_000
_DISPLAY_BUDGET_KEYS = tuple(str(value) for value in DISPLAY_ALARM_BUDGETS_KM2)

_MODEL_LABELS: Mapping[ModelId, str] = MappingProxyType(
    {
        "S0": "S0 长期地震背景",
        "S1": "S1 长期背景 + 最近 30 天地震",
        "SP": "SP 长期背景 + 紧邻过去 30 天对照",
    }
)
_CONTRAST_LABELS: Mapping[ContrastId, str] = MappingProxyType(
    {
        "S1_minus_S0": "S1 − S0（最近地震信息相对长期背景）",
        "S1_minus_SP": "S1 − SP（最近窗口相对过去窗口）",
    }
)
_GATE_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "invalid": "结果无效",
        "evidence_insufficient": "证据不足",
        "failed": "未通过开发门控",
        "passed_development_signal": "通过开发信号门控",
    }
)
_PALETTE = (
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#ffffbf",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
)
_SVG_FONT = "'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif"
_PNG_FONT = FontProperties(family="Microsoft YaHei")
_IG_AXIS_BOUNDS = (0.09, 0.50, 0.82, 0.20)
_RECALL_AXIS_BOUNDS = (0.09, 0.20, 0.82, 0.20)
_RECALL_DISPLAY_HALF_RANGE_FLOOR_PP = 0.1


class Stage2SRenderingError(ValueError):
    """Raised when a result view would be incomplete or scientifically unsafe."""


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool | str | bytes | bytearray):
        raise Stage2SRenderingError(f"{label} must be numeric")
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage2SRenderingError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise Stage2SRenderingError(f"{label} must be finite")
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage2SRenderingError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise Stage2SRenderingError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise Stage2SRenderingError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage2SRenderingError(f"{label} must be non-empty text")
    return value


def _identity_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage2SRenderingError(f"{label} must be an integer identity")
    return value


def _sha256_text(value: object, *, label: str) -> str:
    digest = _text(value, label=label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise Stage2SRenderingError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _sha256_sequence(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(
        _sha256_text(item, label=f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label=label))
    )


def _safe_json(value: object) -> str:
    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): plain(nested) for key, nested in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            return [plain(nested) for nested in item]
        return item

    return json.dumps(
        plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")


@dataclass(frozen=True, slots=True)
class Stage2SMapFrame:
    """One coordinate-free, derived historical relative-intensity raster."""

    issue_time_utc: str
    data_cutoff_utc: str
    fold_index: int
    horizon_days: int
    model_id: ModelId
    relative_intensity_rank: Sequence[Sequence[float | None]]
    study_area_km2: Sequence[Sequence[float | None]]
    alarm_area_fraction_by_budget_km2: Mapping[
        str,
        Sequence[Sequence[float | None]],
    ]
    actual_alarm_area_km2_by_budget: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.fold_index not in (1, 2, 3):
            raise Stage2SRenderingError("map frame fold_index must be 1, 2, or 3")
        if self.horizon_days not in (7, 30, 90):
            raise Stage2SRenderingError("map frame horizon_days must be 7, 30, or 90")
        if self.model_id not in MODEL_IDS:
            raise Stage2SRenderingError("map frame model_id must be S0, S1, or SP")
        _text(self.issue_time_utc, label="map frame issue_time_utc")
        _text(self.data_cutoff_utc, label="map frame data_cutoff_utc")
        rows = tuple(tuple(row) for row in self.relative_intensity_rank)
        areas = tuple(tuple(row) for row in self.study_area_km2)
        if not rows or not rows[0]:
            raise Stage2SRenderingError("map frame raster must not be empty")
        width = len(rows[0])
        if len(rows) > 96 or width > 96:
            raise Stage2SRenderingError("map frame raster exceeds the 96 by 96 publication cap")
        if any(len(row) != width for row in rows):
            raise Stage2SRenderingError("map frame raster rows must share one width")
        if len(areas) != len(rows) or any(len(row) != width for row in areas):
            raise Stage2SRenderingError("study-area raster must align with the map frame raster")
        frozen_rows: list[tuple[float | None, ...]] = []
        frozen_areas: list[tuple[float | None, ...]] = []
        populated = 0
        for row_index, (row, area_row) in enumerate(zip(rows, areas, strict=True)):
            frozen_row: list[float | None] = []
            frozen_area: list[float | None] = []
            for column_index, (raw_value, raw_area) in enumerate(zip(row, area_row, strict=True)):
                if raw_value is None:
                    if raw_area is not None:
                        raise Stage2SRenderingError(
                            "outside-study raster pixels cannot contain study area"
                        )
                    frozen_row.append(None)
                    frozen_area.append(None)
                    continue
                value = _finite_number(
                    raw_value,
                    label=f"relative rank [{row_index},{column_index}]",
                )
                if not 0.0 <= value <= 1.0:
                    raise Stage2SRenderingError("relative intensity rank must be in [0, 1]")
                area = _finite_number(
                    raw_area,
                    label=f"study area [{row_index},{column_index}]",
                )
                if area <= 0.0:
                    raise Stage2SRenderingError(
                        "populated raster pixels must have positive study area"
                    )
                frozen_row.append(value)
                frozen_area.append(area)
                populated += 1
            frozen_rows.append(tuple(frozen_row))
            frozen_areas.append(tuple(frozen_area))
        if populated == 0:
            raise Stage2SRenderingError("map frame must contain at least one study-area pixel")

        raw_fraction_by_budget = _mapping(
            self.alarm_area_fraction_by_budget_km2,
            label="alarm_area_fraction_by_budget_km2",
        )
        raw_actual_by_budget = _mapping(
            self.actual_alarm_area_km2_by_budget,
            label="actual_alarm_area_km2_by_budget",
        )
        if tuple(raw_fraction_by_budget) != _DISPLAY_BUDGET_KEYS:
            raise Stage2SRenderingError("alarm-area fraction budgets or order changed")
        if tuple(raw_actual_by_budget) != _DISPLAY_BUDGET_KEYS:
            raise Stage2SRenderingError("actual alarm-area budgets or order changed")
        frozen_fraction_by_budget: dict[str, tuple[tuple[float | None, ...], ...]] = {}
        frozen_actual_by_budget: dict[str, float] = {}
        for budget_key, budget in zip(
            _DISPLAY_BUDGET_KEYS,
            DISPLAY_ALARM_BUDGETS_KM2,
            strict=True,
        ):
            raw_fraction_rows: tuple[tuple[object, ...], ...] = tuple(
                tuple(
                    _sequence(
                        row,
                        label=f"alarm fraction {budget_key} row {row_index}",
                    )
                )
                for row_index, row in enumerate(
                    _sequence(
                        raw_fraction_by_budget[budget_key],
                        label=f"alarm fraction {budget_key}",
                    )
                )
            )
            if len(raw_fraction_rows) != len(rows) or any(
                len(row) != width for row in raw_fraction_rows
            ):
                raise Stage2SRenderingError(
                    f"alarm fraction {budget_key} must align with the map raster"
                )
            frozen_fraction_rows: list[tuple[float | None, ...]] = []
            closure_terms: list[float] = []
            for row_index, (rank_row, area_row, fraction_row) in enumerate(
                zip(frozen_rows, frozen_areas, raw_fraction_rows, strict=True)
            ):
                frozen_fraction_row: list[float | None] = []
                for column_index, (rank, pixel_area, raw_fraction) in enumerate(
                    zip(rank_row, area_row, fraction_row, strict=True)
                ):
                    if rank is None:
                        if raw_fraction is not None:
                            raise Stage2SRenderingError(
                                "outside-study raster pixels cannot contain alarm area"
                            )
                        frozen_fraction_row.append(None)
                        continue
                    fraction = _finite_number(
                        raw_fraction,
                        label=(f"alarm fraction {budget_key} [{row_index},{column_index}]"),
                    )
                    if not 0.0 <= fraction <= 1.0:
                        raise Stage2SRenderingError("alarm-area fraction must be in [0, 1]")
                    assert pixel_area is not None
                    closure_terms.append(pixel_area * fraction)
                    frozen_fraction_row.append(fraction)
                frozen_fraction_rows.append(tuple(frozen_fraction_row))
            actual = _finite_number(
                raw_actual_by_budget[budget_key],
                label=f"actual alarm area {budget_key}",
            )
            if actual < 0.0 or actual > float(budget) + 1.0e-8:
                raise Stage2SRenderingError(
                    "actual alarm area must be non-negative and within its display budget"
                )
            closed_area = math.fsum(closure_terms)
            if not math.isclose(
                closed_area,
                actual,
                rel_tol=0.0,
                abs_tol=max(1.0e-8, actual * 1.0e-12),
            ):
                raise Stage2SRenderingError(
                    f"alarm-area fraction {budget_key} does not close to exact area"
                )
            frozen_fraction_by_budget[budget_key] = tuple(frozen_fraction_rows)
            frozen_actual_by_budget[budget_key] = actual

        object.__setattr__(self, "relative_intensity_rank", tuple(frozen_rows))
        object.__setattr__(self, "study_area_km2", tuple(frozen_areas))
        object.__setattr__(
            self,
            "alarm_area_fraction_by_budget_km2",
            MappingProxyType(frozen_fraction_by_budget),
        )
        object.__setattr__(
            self,
            "actual_alarm_area_km2_by_budget",
            MappingProxyType(frozen_actual_by_budget),
        )

    def alarm_area_fraction(
        self,
        budget_km2: int = FORMAL_ALARM_BUDGET_KM2,
    ) -> tuple[tuple[float | None, ...], ...]:
        try:
            return cast(
                tuple[tuple[float | None, ...], ...],
                self.alarm_area_fraction_by_budget_km2[str(budget_km2)],
            )
        except KeyError as exc:
            raise Stage2SRenderingError("unknown display alarm-area budget") from exc

    @property
    def actual_alarm_area_km2(self) -> float:
        return self.actual_alarm_area_km2_by_budget[str(FORMAL_ALARM_BUDGET_KM2)]

    @property
    def frame_id(self) -> str:
        return f"fold{self.fold_index}-h{self.horizon_days}-{self.issue_time_utc}-{self.model_id}"


def build_rank_map_frame(
    *,
    issue_time_utc: str,
    data_cutoff_utc: str,
    fold_index: int,
    horizon_days: int,
    model_id: ModelId,
    projected_x_m: ArrayLike,
    projected_y_m: ArrayLike,
    relative_mass: ArrayLike,
    clipped_area_km2: ArrayLike,
    alarm: ArrayLike,
    actual_alarm_area_km2: float,
    raster_width: int = 48,
    raster_height: int = 36,
) -> Stage2SMapFrame:
    """Bin frozen grid points into a coordinate-free rank raster.

    The resulting DTO contains no projected or geographic coordinate table.  Each
    populated pixel stores a tie-aware rank of ``sum(mass) / sum(clipped area)``.
    Alarm overlays retain the exact selected clipped-area fraction per display
    budget; only 600,000 km² is the formal scientific layer.
    """

    if not 4 <= raster_width <= 96 or not 4 <= raster_height <= 96:
        raise Stage2SRenderingError("raster dimensions must each be between 4 and 96")
    x = np.asarray(projected_x_m, dtype=np.float64)
    y = np.asarray(projected_y_m, dtype=np.float64)
    mass = np.asarray(relative_mass, dtype=np.float64)
    area = np.asarray(clipped_area_km2, dtype=np.float64)
    alarm_values = np.asarray(alarm, dtype=np.bool_)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or mass.shape != x.shape
        or area.shape != x.shape
        or alarm_values.shape != x.shape
    ):
        raise Stage2SRenderingError("map-frame point arrays must be aligned one-dimensional arrays")
    if x.size == 0:
        raise Stage2SRenderingError("map-frame point arrays must not be empty")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise Stage2SRenderingError("map-frame projected coordinates must be finite")
    if not np.isfinite(mass).all() or np.any(mass < 0.0):
        raise Stage2SRenderingError("map-frame relative mass must be finite and non-negative")
    if not np.isfinite(area).all() or np.any(area <= 0.0):
        raise Stage2SRenderingError("map-frame clipped area must be finite and positive")
    coordinate_keys = tuple(
        (float(x_value).hex(), float(y_value).hex()) for x_value, y_value in zip(x, y, strict=True)
    )
    if len(set(coordinate_keys)) != len(coordinate_keys):
        raise Stage2SRenderingError("map-frame projected grid points must be unique")
    x_span = float(np.max(x) - np.min(x))
    y_span = float(np.max(y) - np.min(y))
    if x_span <= 0.0 or y_span <= 0.0:
        raise Stage2SRenderingError("map-frame projected extent must be two-dimensional")
    column = np.minimum(
        ((x - float(np.min(x))) / x_span * raster_width).astype(np.int64),
        raster_width - 1,
    )
    row = np.minimum(
        ((float(np.max(y)) - y) / y_span * raster_height).astype(np.int64),
        raster_height - 1,
    )
    grouped_indices: dict[tuple[int, int], list[int]] = {}
    for index in range(x.size):
        grouped_indices.setdefault((int(row[index]), int(column[index])), []).append(index)
    accumulated_density = np.zeros((raster_height, raster_width), dtype=np.float64)
    accumulated_area = np.zeros((raster_height, raster_width), dtype=np.float64)
    populated = np.zeros((raster_height, raster_width), dtype=np.bool_)
    for (row_index, column_index), indices in grouped_indices.items():
        ordered_indices = sorted(
            indices,
            key=lambda index: (float(x[index]), float(y[index])),
        )
        pixel_mass = math.fsum(float(mass[index]) for index in ordered_indices)
        pixel_area = math.fsum(float(area[index]) for index in ordered_indices)
        accumulated_density[row_index, column_index] = pixel_mass / pixel_area
        accumulated_area[row_index, column_index] = pixel_area
        populated[row_index, column_index] = True
    values = accumulated_density[populated]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape, dtype=np.float64)
    if values.size == 1:
        ranks[0] = 1.0
    else:
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            tied_rank = ((start + end - 1) / 2.0) / (values.size - 1)
            ranks[order[start:end]] = tied_rank
            start = end
    rank_raster = np.full(accumulated_density.shape, np.nan, dtype=np.float64)
    rank_raster[populated] = ranks
    rank_rows = tuple(
        tuple(
            None if not populated[i, j] else float(rank_raster[i, j]) for j in range(raster_width)
        )
        for i in range(raster_height)
    )
    study_area_rows = tuple(
        tuple(
            None if not populated[i, j] else float(accumulated_area[i, j])
            for j in range(raster_width)
        )
        for i in range(raster_height)
    )
    formal_actual = math.fsum(
        float(area[index]) for index in range(x.size) if bool(alarm_values[index])
    )
    supplied_formal_actual = _finite_number(
        actual_alarm_area_km2,
        label="actual_alarm_area_km2",
    )
    if not math.isclose(
        formal_actual,
        supplied_formal_actual,
        rel_tol=0.0,
        abs_tol=max(1.0e-8, supplied_formal_actual * 1.0e-12),
    ):
        raise Stage2SRenderingError("formal alarm mask area does not match actual_alarm_area_km2")
    cell_intensity = np.asarray(mass / area, dtype=np.float64)
    ranking = tuple(
        sorted(
            range(x.size),
            key=lambda index: (
                -float(cell_intensity[index]),
                float(y[index]),
                float(x[index]),
            ),
        )
    )
    selected_by_budget: dict[str, NDArray[np.bool_]] = {}
    actual_by_budget: dict[str, float] = {}
    for budget in DISPLAY_ALARM_BUDGETS_KM2:
        budget_key = str(budget)
        if budget == FORMAL_ALARM_BUDGET_KM2:
            selected = np.asarray(alarm_values, dtype=np.bool_)
            actual = formal_actual
        else:
            selected = np.zeros(x.size, dtype=np.bool_)
            running_area = 0.0
            for index in ranking:
                candidate_area = math.fsum((running_area, float(area[index])))
                if candidate_area > float(budget):
                    break
                selected[index] = True
                running_area = candidate_area
            actual = running_area
        selected_by_budget[budget_key] = selected
        actual_by_budget[budget_key] = actual
    fraction_by_budget: dict[str, tuple[tuple[float | None, ...], ...]] = {}
    for budget_key in _DISPLAY_BUDGET_KEYS:
        selected = selected_by_budget[budget_key]
        fraction_raster = np.full(
            (raster_height, raster_width),
            np.nan,
            dtype=np.float64,
        )
        for (row_index, column_index), indices in grouped_indices.items():
            alarm_area = math.fsum(float(area[index]) for index in indices if bool(selected[index]))
            fraction_raster[row_index, column_index] = (
                alarm_area / accumulated_area[row_index, column_index]
            )
        fraction_by_budget[budget_key] = tuple(
            tuple(
                None if not populated[i, j] else float(fraction_raster[i, j])
                for j in range(raster_width)
            )
            for i in range(raster_height)
        )
    return Stage2SMapFrame(
        issue_time_utc=issue_time_utc,
        data_cutoff_utc=data_cutoff_utc,
        fold_index=fold_index,
        horizon_days=horizon_days,
        model_id=model_id,
        relative_intensity_rank=rank_rows,
        study_area_km2=study_area_rows,
        alarm_area_fraction_by_budget_km2=fraction_by_budget,
        actual_alarm_area_km2_by_budget=actual_by_budget,
    )


@dataclass(frozen=True, slots=True)
class Stage2SRenderPayload:
    """Complete immutable render input, separate from model fitting and scoring."""

    record: Stage2SWholeRunRecord
    s0_training_cutoff_utc: str
    recent_origin_window: str
    preceding_origin_window: str
    available_at_cutoff: str
    map_frames: Sequence[Stage2SMapFrame]

    def __post_init__(self) -> None:
        if not isinstance(self.record, Stage2SWholeRunRecord):
            raise TypeError("record must be a Stage2SWholeRunRecord")
        _text(self.s0_training_cutoff_utc, label="s0_training_cutoff_utc")
        _text(self.recent_origin_window, label="recent_origin_window")
        _text(self.preceding_origin_window, label="preceding_origin_window")
        _text(self.available_at_cutoff, label="available_at_cutoff")
        frames = tuple(self.map_frames)
        if not frames:
            raise Stage2SRenderingError("at least one S0/S1/SP map-frame group is required")
        if len(frames) > 180:
            raise Stage2SRenderingError("map frame count exceeds the bounded publication cap")
        if not all(isinstance(frame, Stage2SMapFrame) for frame in frames):
            raise TypeError("map_frames must contain only Stage2SMapFrame values")
        grouped: dict[tuple[int, int, str], set[str]] = {}
        seen_ids: set[str] = set()
        for frame in frames:
            if frame.frame_id in seen_ids:
                raise Stage2SRenderingError("map frame IDs must be unique")
            seen_ids.add(frame.frame_id)
            key = (frame.fold_index, frame.horizon_days, frame.issue_time_utc)
            grouped.setdefault(key, set()).add(frame.model_id)
        if any(models != set(MODEL_IDS) for models in grouped.values()):
            raise Stage2SRenderingError("each map-frame issue group must contain S0, S1, and SP")
        ordered = tuple(
            sorted(
                frames,
                key=lambda frame: (
                    frame.fold_index,
                    frame.horizon_days,
                    frame.issue_time_utc.encode("utf-8"),
                    MODEL_IDS.index(frame.model_id),
                ),
            )
        )
        object.__setattr__(self, "map_frames", ordered)


def _record_map_frame_bindings(
    payload: Stage2SRenderPayload,
) -> tuple[Mapping[str, object], ...]:
    """Close every published map group against final whole-run identities and seals."""

    record = payload.record
    seals = _mapping(record.seal_chain, label="seal_chain")
    required_seal_fields = {
        "fold_fit_receipt_sha256",
        "issue_prediction_seal_sha256",
        "fold_prediction_seal_sha256",
        "master_prediction_seal_sha256",
    }
    if set(seals) != required_seal_fields:
        missing = tuple(sorted(required_seal_fields - set(seals)))
        extra = tuple(sorted(set(seals) - required_seal_fields))
        raise Stage2SRenderingError(
            f"seal_chain fields changed; missing={missing!r}, extra={extra!r}"
        )
    fit_chain = _sha256_sequence(
        seals["fold_fit_receipt_sha256"],
        label="seal_chain.fold_fit_receipt_sha256",
    )
    issue_chain = _sha256_sequence(
        seals["issue_prediction_seal_sha256"],
        label="seal_chain.issue_prediction_seal_sha256",
    )
    fold_chain = _sha256_sequence(
        seals["fold_prediction_seal_sha256"],
        label="seal_chain.fold_prediction_seal_sha256",
    )
    master_sha256 = _sha256_text(
        seals["master_prediction_seal_sha256"],
        label="seal_chain.master_prediction_seal_sha256",
    )

    fold_bindings: dict[int, dict[str, str]] = {}
    ordered_fit_sha256: list[str] = []
    ordered_fold_sha256: list[str] = []
    for index, raw_summary in enumerate(record.fold_fit_summaries):
        summary = _mapping(raw_summary, label=f"fold_fit_summaries[{index}]")
        fold_index = _identity_integer(
            summary.get("fold_index"),
            label=f"fold_fit_summaries[{index}].fold_index",
        )
        if fold_index in fold_bindings:
            raise Stage2SRenderingError("fold-fit identities must be unique")
        fit_sha256 = _sha256_text(
            summary.get("fit_receipt_sha256"),
            label=f"fold_fit_summaries[{index}].fit_receipt_sha256",
        )
        fold_sha256 = _sha256_text(
            summary.get("fold_prediction_seal_sha256"),
            label=f"fold_fit_summaries[{index}].fold_prediction_seal_sha256",
        )
        fold_bindings[fold_index] = {
            "fit_receipt_sha256": fit_sha256,
            "fold_prediction_seal_sha256": fold_sha256,
        }
        ordered_fit_sha256.append(fit_sha256)
        ordered_fold_sha256.append(fold_sha256)
    if tuple(fold_bindings) != (1, 2, 3):
        raise Stage2SRenderingError("fold-fit summaries must be ordered folds 1, 2, and 3")
    if tuple(ordered_fit_sha256) != fit_chain:
        raise Stage2SRenderingError("fold fit receipts do not match the seal chain")
    if tuple(ordered_fold_sha256) != fold_chain:
        raise Stage2SRenderingError("fold prediction seals do not match the seal chain")

    cell_identities = {
        (
            _identity_integer(cell.get("fold_index"), label="cell_scores.fold_index"),
            _identity_integer(cell.get("horizon_days"), label="cell_scores.horizon_days"),
        )
        for cell in record.cell_scores
    }
    issue_bindings: dict[tuple[int, str], dict[str, object]] = {}
    ordered_issue_sha256: list[str] = []
    expected_groups: set[tuple[int, int, str]] = set()
    for index, raw_summary in enumerate(record.issue_prediction_summaries):
        summary = _mapping(raw_summary, label=f"issue_prediction_summaries[{index}]")
        fold_index = _identity_integer(
            summary.get("fold_index"),
            label=f"issue_prediction_summaries[{index}].fold_index",
        )
        if fold_index not in fold_bindings:
            raise Stage2SRenderingError("issue summary references an unknown fold")
        issue_date = _text(
            summary.get("issue_date"),
            label=f"issue_prediction_summaries[{index}].issue_date",
        )
        issue_time_utc = _text(
            summary.get("issue_time_utc"),
            label=f"issue_prediction_summaries[{index}].issue_time_utc",
        )
        raw_horizons = _sequence(
            summary.get("horizons_days"),
            label=f"issue_prediction_summaries[{index}].horizons_days",
        )
        horizons = tuple(
            _identity_integer(
                value,
                label=f"issue_prediction_summaries[{index}].horizons_days[{position}]",
            )
            for position, value in enumerate(raw_horizons)
        )
        if not horizons or any(value not in (7, 30, 90) for value in horizons):
            raise Stage2SRenderingError("issue summary horizons must be a non-empty 7/30/90 subset")
        if horizons != tuple(value for value in (7, 30, 90) if value in set(horizons)):
            raise Stage2SRenderingError("issue summary horizons must be unique and ordered")
        issue_sha256 = _sha256_text(
            summary.get("issue_prediction_seal_sha256"),
            label=f"issue_prediction_summaries[{index}].issue_prediction_seal_sha256",
        )
        raw_areas = _mapping(
            summary.get("actual_alarm_area_km2"),
            label=f"issue_prediction_summaries[{index}].actual_alarm_area_km2",
        )
        actual_area_by_model = {
            model_id: _finite_number(
                raw_areas.get(f"delay0:{model_id}"),
                label=(
                    f"issue_prediction_summaries[{index}].actual_alarm_area_km2.delay0:{model_id}"
                ),
            )
            for model_id in MODEL_IDS
        }
        issue_key = (fold_index, issue_time_utc)
        if issue_key in issue_bindings:
            raise Stage2SRenderingError("issue prediction identities must be unique")
        issue_bindings[issue_key] = {
            "issue_date": issue_date,
            "horizons_days": horizons,
            "issue_prediction_seal_sha256": issue_sha256,
            "actual_alarm_area_km2": actual_area_by_model,
        }
        ordered_issue_sha256.append(issue_sha256)
        for horizon_days in horizons:
            if (fold_index, horizon_days) not in cell_identities:
                raise Stage2SRenderingError(
                    "issue fold/horizon identity is absent from whole-run cell scores"
                )
            expected_groups.add((fold_index, horizon_days, issue_time_utc))
    if tuple(ordered_issue_sha256) != issue_chain:
        raise Stage2SRenderingError("issue prediction seals do not match the seal chain")

    frames_by_group: dict[tuple[int, int, str], dict[ModelId, Stage2SMapFrame]] = {}
    for frame in payload.map_frames:
        if frame.data_cutoff_utc != frame.issue_time_utc:
            raise Stage2SRenderingError("primary map data cutoff must equal its sealed issue time")
        key = (frame.fold_index, frame.horizon_days, frame.issue_time_utc)
        frames_by_group.setdefault(key, {})[frame.model_id] = frame
    observed_groups = set(frames_by_group)
    if observed_groups != expected_groups:
        missing_groups = tuple(sorted(expected_groups - observed_groups))
        extra_groups = tuple(sorted(observed_groups - expected_groups))
        raise Stage2SRenderingError(
            "map-frame issue groups differ from the record; "
            f"missing={missing_groups!r}, extra={extra_groups!r}"
        )

    bindings: list[Mapping[str, object]] = []
    for fold_index, horizon_days, issue_time_utc in sorted(expected_groups):
        issue = issue_bindings[(fold_index, issue_time_utc)]
        area_by_model = cast(Mapping[str, float], issue["actual_alarm_area_km2"])
        frames_by_model = frames_by_group[(fold_index, horizon_days, issue_time_utc)]
        for model_id in MODEL_IDS:
            observed_area = frames_by_model[model_id].actual_alarm_area_km2
            expected_area = area_by_model[model_id]
            if not math.isclose(
                observed_area,
                expected_area,
                rel_tol=0.0,
                abs_tol=max(1.0e-8, expected_area * 1.0e-12),
            ):
                raise Stage2SRenderingError(
                    "map-frame alarm area differs from its sealed issue summary"
                )
        fold = fold_bindings[fold_index]
        bindings.append(
            MappingProxyType(
                {
                    "fold_index": fold_index,
                    "horizon_days": horizon_days,
                    "issue_date": issue["issue_date"],
                    "issue_time_utc": issue_time_utc,
                    "data_cutoff_utc": issue_time_utc,
                    "models": MODEL_IDS,
                    "actual_alarm_area_km2": area_by_model,
                    "fit_receipt_sha256": fold["fit_receipt_sha256"],
                    "issue_prediction_seal_sha256": issue["issue_prediction_seal_sha256"],
                    "fold_prediction_seal_sha256": fold["fold_prediction_seal_sha256"],
                    "master_prediction_seal_sha256": master_sha256,
                }
            )
        )
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class Stage2SRenderedArtifact:
    name: str
    payload: bytes
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.name not in ALL_ARTIFACT_NAMES:
            raise Stage2SRenderingError(f"unknown Stage 2S render artifact {self.name!r}")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise Stage2SRenderingError("render artifact payload must be non-empty bytes")
        object.__setattr__(self, "sha256", hashlib.sha256(self.payload).hexdigest())


@dataclass(frozen=True, slots=True)
class Stage2SRenderedBundle:
    """The five preregistered views plus a deterministic PNG companion."""

    artifacts: Sequence[Stage2SRenderedArtifact]

    def __post_init__(self) -> None:
        values = tuple(self.artifacts)
        if tuple(item.name for item in values) != ALL_ARTIFACT_NAMES:
            raise Stage2SRenderingError("render bundle artifact order or membership changed")
        object.__setattr__(self, "artifacts", values)

    @property
    def artifact_sha256_by_name(self) -> Mapping[str, str]:
        return MappingProxyType({item.name: item.sha256 for item in self.artifacts})

    def artifact(self, name: str) -> Stage2SRenderedArtifact:
        for item in self.artifacts:
            if item.name == name:
                return item
        raise KeyError(name)

    def write_to(self, output_dir: Path, *, check: bool = False) -> None:
        """Create every artifact once, or verify that existing bytes are current."""

        if not output_dir.is_absolute():
            raise Stage2SRenderingError("render output directory must be absolute")
        if check:
            for artifact in self.artifacts:
                destination = output_dir / artifact.name
                try:
                    existing = destination.read_bytes()
                except FileNotFoundError as exc:
                    raise Stage2SRenderingError(f"missing render artifact: {destination}") from exc
                if existing != artifact.payload:
                    raise Stage2SRenderingError(f"stale render artifact: {destination}")
            return
        existing_destinations = tuple(
            output_dir / artifact.name
            for artifact in self.artifacts
            if os.path.lexists(output_dir / artifact.name)
        )
        if existing_destinations:
            raise Stage2SRenderingError(
                f"render artifact already exists: {existing_destinations[0]}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for artifact in self.artifacts:
            destination = output_dir / artifact.name
            try:
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                    0o600,
                )
            except FileExistsError as exc:
                raise Stage2SRenderingError(
                    f"render artifact already exists: {destination}"
                ) from exc
            try:
                offset = 0
                while offset < len(artifact.payload):
                    written = os.write(descriptor, artifact.payload[offset:])
                    if written <= 0:
                        raise Stage2SRenderingError("short write while creating render artifact")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def verify_stage2s_bundle_against_record(
    payload: Stage2SRenderPayload,
    bundle: Stage2SRenderedBundle,
) -> None:
    """Require a final whole-run record to bind all six deterministic render bytes."""

    if not isinstance(payload, Stage2SRenderPayload):
        raise TypeError("payload must be a Stage2SRenderPayload")
    if not isinstance(bundle, Stage2SRenderedBundle):
        raise TypeError("bundle must be a Stage2SRenderedBundle")
    expected_raw = _mapping(
        payload.record.artifact_sha256_by_name,
        label="whole-run artifact_sha256_by_name",
    )
    expected_names = set(ALL_ARTIFACT_NAMES)
    supplied_names = set(expected_raw)
    if supplied_names != expected_names:
        missing = tuple(name for name in ALL_ARTIFACT_NAMES if name not in supplied_names)
        extra = tuple(sorted(supplied_names - expected_names))
        raise Stage2SRenderingError(
            f"whole-run artifact hashes changed; missing={missing!r}, extra={extra!r}"
        )
    actual = bundle.artifact_sha256_by_name
    for name in ALL_ARTIFACT_NAMES:
        expected_sha256 = _sha256_text(
            expected_raw[name],
            label=f"whole-run artifact_sha256_by_name.{name}",
        )
        if expected_sha256 != actual[name]:
            raise Stage2SRenderingError(
                f"render artifact SHA-256 differs from whole-run record: {name}"
            )


@dataclass(frozen=True, slots=True)
class _Interval:
    point: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class _CellView:
    fold_index: int
    horizon_days: int
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _Summary:
    mode_code: str
    status_banner: str
    scope_note: str
    experiment_id: str
    gate_status: str
    gate_label: str
    gate_reasons: tuple[str, ...]
    metrics: Mapping[str, _Interval]
    cells: tuple[_CellView, ...]
    regional: Mapping[str, object]
    sequence: Mapping[str, object]
    latency: tuple[Mapping[str, object], ...]


def _metric_key(contrast: ContrastId, metric: MetricId) -> str:
    return f"{contrast}:{metric}"


def _cell_metric(
    cell: Mapping[str, object],
    *,
    contrast: ContrastId,
    metric: MetricId,
) -> float:
    field = "information_gain" if metric == "IG" else "recall_gain"
    family = cell.get(field)
    if isinstance(family, Mapping) and contrast in family:
        return _finite_number(family[contrast], label=f"cell {field}.{contrast}")
    fallback = cell.get(metric)
    return _finite_number(fallback, label=f"cell fallback {metric}")


def _extract_summary(record: Stage2SWholeRunRecord) -> _Summary:
    if record.mode == "synthetic_acceptance":
        mode_code = "SYNTHETIC"
        status_banner = "SYNTHETIC · 合成数据 · 仅代码与流程验收"
        scope_note = "DEVELOPMENT 未运行；REAL DATA 未读取；不能据此声称预测效果提升。"
    else:
        mode_code = "DEVELOPMENT_REAL"
        status_banner = "REAL DATA · DEVELOPMENT · 历史回溯"
        scope_note = "复用开发期历史评估；不是独立验证、锁定测试、当前预测或前瞻预测。"
    gate = _mapping(record.gate_evidence, label="gate_evidence")
    status = _text(gate.get("status"), label="gate_evidence.status")
    if status not in _GATE_LABELS:
        raise Stage2SRenderingError("unknown Stage 2S gate status")
    raw_reasons = gate.get("reasons", ())
    reasons = tuple(
        _text(value, label="gate reason")
        for value in _sequence(raw_reasons, label="gate_evidence.reasons")
    )
    identity = _mapping(record.identity, label="identity")
    experiment_id = str(identity.get("experiment_id", "stage2s"))
    overall = gate.get("overall_macros", {})
    overall_mapping = _mapping(overall, label="gate_evidence.overall_macros")
    cells: list[_CellView] = []
    for raw_cell in record.cell_scores:
        cell = _mapping(raw_cell, label="cell_scores item")
        fold_index = int(_finite_number(cell.get("fold_index"), label="cell fold_index"))
        horizon_days = int(_finite_number(cell.get("horizon_days"), label="cell horizon_days"))
        values = {
            _metric_key(contrast, metric): _cell_metric(
                cell,
                contrast=contrast,
                metric=metric,
            )
            for contrast in CONTRAST_IDS
            for metric in METRIC_IDS
        }
        cells.append(
            _CellView(
                fold_index=fold_index,
                horizon_days=horizon_days,
                values=MappingProxyType(values),
            )
        )
    interval_root = _mapping(
        record.bootstrap_summary.get("intervals", {}),
        label="bootstrap_summary.intervals",
    )
    metrics: dict[str, _Interval] = {}
    for contrast in CONTRAST_IDS:
        for metric in METRIC_IDS:
            key = _metric_key(contrast, metric)
            cell_values = [cell.values[key] for cell in cells]
            fallback_point = math.fsum(cell_values) / len(cell_values)
            point = _finite_number(
                overall_mapping.get(key, fallback_point),
                label=f"overall macro {key}",
            )
            raw_interval = interval_root.get(key)
            if isinstance(raw_interval, Mapping):
                interval = _mapping(raw_interval, label=f"bootstrap interval {key}")
                point = _finite_number(interval.get("point", point), label=f"{key} point")
                lower = _finite_number(interval.get("lower", point), label=f"{key} lower")
                upper = _finite_number(interval.get("upper", point), label=f"{key} upper")
            else:
                lower = point
                upper = point
            if lower > point or point > upper:
                raise Stage2SRenderingError(f"bootstrap interval {key} is not ordered")
            metrics[key] = _Interval(point=point, lower=lower, upper=upper)
    regional = _mapping(record.regional_evidence, label="regional_evidence")
    sequence = _mapping(record.sequence_evidence, label="sequence_evidence")
    claim_limited = sequence.get("claim_limited")
    if claim_limited is True:
        status_banner += " · 仅支持震群相关续发"
        scope_note += " 去除主导震群后优势不稳，禁止解释为广泛区域提升。"
    latency = tuple(
        _mapping(value, label="latency_evidence item") for value in record.latency_evidence
    )
    return _Summary(
        mode_code=mode_code,
        status_banner=status_banner,
        scope_note=scope_note,
        experiment_id=experiment_id,
        gate_status=status,
        gate_label=_GATE_LABELS[status],
        gate_reasons=reasons,
        metrics=MappingProxyType(metrics),
        cells=tuple(cells),
        regional=regional,
        sequence=sequence,
        latency=latency,
    )


def _display_value(metric: MetricId, value: float) -> str:
    if metric == "recall":
        return f"{value * 100:+.1f} 个百分点"
    return f"{value:+.3f}"


def _display_short(metric: MetricId, value: float) -> str:
    if metric == "recall":
        return f"{value * 100:+.1f}pp"
    return f"{value:+.3f}"


def _palette_color(rank: float) -> str:
    index = min(int(rank * len(_PALETTE)), len(_PALETTE) - 1)
    return _PALETTE[index]


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str = "body",
    anchor: str | None = None,
) -> str:
    anchor_part = f' text-anchor="{escape(anchor, quote=True)}"' if anchor else ""
    return (
        f'<text x="{x:g}" y="{y:g}" class="{escape(css_class, quote=True)}"'
        f"{anchor_part}>{escape(value)}</text>"
    )


def _svg_box(
    x: float,
    y: float,
    width: float,
    height: float,
    css_class: str,
    *,
    radius: float = 16,
) -> str:
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{radius:g}" class="{escape(css_class, quote=True)}"/>'
    )


def _svg_header(
    *,
    title: str,
    description: str,
    summary: _Summary,
    width: int,
    height: int,
) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(description)}</desc>',
        "<style>",
        f"text{{font-family:{_SVG_FONT};fill:#102a43}}",
        ".title{font-size:34px;font-weight:700}.subtitle{font-size:17px;fill:#486581}",
        ".section{font-size:21px;font-weight:700}.heading{font-size:18px;font-weight:700}",
        ".body{font-size:15px}.small{font-size:13px;fill:#486581}",
        ".value{font-size:24px;font-weight:700}.status{font-size:15px;font-weight:700;fill:#fff}",
        ".panel{fill:#fff;stroke:#bcccdc;stroke-width:1.5}",
        ".blue{fill:#eaf2ff;stroke:#2f80ed;stroke-width:2}",
        ".green{fill:#e8f7f1;stroke:#2d9d78;stroke-width:2}",
        ".orange{fill:#fff3e7;stroke:#f2994a;stroke-width:2}",
        ".purple{fill:#f1edff;stroke:#7b61ff;stroke-width:2}",
        ".warning{fill:#fff0f0;stroke:#c0392b;stroke-width:2}",
        ".track{stroke:#d9e2ec;stroke-width:14;stroke-linecap:round}",
        ".zero{stroke:#627d98;stroke-width:1}.ci{stroke:#102a43;stroke-width:3}",
        ".pointA{fill:#2f80ed}.pointB{fill:#f2994a}.divider{stroke:#d9e2ec;stroke-width:1}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        _svg_text(60, 58, title, css_class="title"),
        _svg_text(60, 91, summary.scope_note, css_class="subtitle"),
        _svg_box(width - 535, 30, 475, 46, "warning", radius=23),
        _svg_text(
            width - 297.5,
            59,
            summary.status_banner,
            css_class="heading",
            anchor="middle",
        ),
    ]


def _svg_metric_bar(
    *,
    x: float,
    y: float,
    width: float,
    interval: _Interval,
    scale: float,
    css_class: str,
    metric: MetricId,
) -> list[str]:
    half = width / 2.0

    def project(value: float) -> float:
        return x + half + max(-1.0, min(1.0, value / scale)) * (half - 8)

    return [
        f'<line x1="{x:g}" y1="{y:g}" x2="{x + width:g}" y2="{y:g}" class="track"/>',
        f'<line x1="{x + half:g}" y1="{y - 13:g}" x2="{x + half:g}" y2="{y + 13:g}" class="zero"/>',
        f'<line x1="{project(interval.lower):g}" y1="{y:g}" '
        f'x2="{project(interval.upper):g}" y2="{y:g}" class="ci"/>',
        f'<circle cx="{project(interval.point):g}" cy="{y:g}" r="8" class="{css_class}"/>',
        _svg_text(
            x + width + 20,
            y + 6,
            _display_value(metric, interval.point),
            css_class="heading",
        ),
    ]


def render_overall_results_svg(payload: Stage2SRenderPayload) -> bytes:
    """Render the principal same-area IG and strict-recall comparison."""

    summary = _extract_summary(payload.record)
    lines = _svg_header(
        title="Stage 2S 总体效果：同一报警面积下的公平比较",
        description=(
            "比较 S1 与长期背景 S0、过去窗口对照 SP 的信息增益和严格召回增量，"
            "并显示开发门控与证据边界。"
        ),
        summary=summary,
        width=1600,
        height=1120,
    )
    lines.extend(
        [
            _svg_text(60, 145, "用了什么方法", css_class="section"),
            _svg_box(60, 170, 440, 155, "blue"),
            _svg_text(85, 207, "S0｜长期背景", css_class="heading"),
            _svg_text(85, 241, "只用长期 M4+ 地震分布，回答“哪里长期更活跃”。"),
            _svg_text(85, 273, f"训练截止：{payload.s0_training_cutoff_utc}", css_class="small"),
            _svg_text(85, 299, "作为所有比较共同的合法背景。", css_class="small"),
            _svg_box(580, 170, 440, 155, "green"),
            _svg_text(605, 207, "S1｜最近地震候选", css_class="heading"),
            _svg_text(605, 241, "在 S0 上加入起报前最近 30 天 M4+ 地震分布。"),
            _svg_text(605, 273, f"窗口：{payload.recent_origin_window}", css_class="small"),
            _svg_text(605, 299, "检验近期地震是否带来额外空间信息。", css_class="small"),
            _svg_box(1100, 170, 440, 155, "orange"),
            _svg_text(1125, 207, "SP｜过去窗口对照", css_class="heading"),
            _svg_text(1125, 241, "把最近窗口换成紧邻的更早 30 天，结构保持相同。"),
            _svg_text(1125, 273, f"窗口：{payload.preceding_origin_window}", css_class="small"),
            _svg_text(1125, 299, "排除“任意地震多发区都显得有效”的假象。", css_class="small"),
            _svg_text(60, 380, "总体效果（3 折 × 7/30/90 天宏平均）", css_class="section"),
        ]
    )
    ig_scale = max(
        1.0e-12,
        *(
            abs(value)
            for contrast in CONTRAST_IDS
            for interval in (summary.metrics[_metric_key(contrast, "IG")],)
            for value in (interval.lower, interval.upper, interval.point)
        ),
    )
    recall_scale = max(
        1.0e-12,
        *(
            abs(value)
            for contrast in CONTRAST_IDS
            for interval in (summary.metrics[_metric_key(contrast, "recall")],)
            for value in (interval.lower, interval.upper, interval.point)
        ),
    )
    lines.extend(
        [
            _svg_box(60, 405, 1480, 205, "panel"),
            _svg_text(90, 443, "信息增益 IG（正值表示 S1 的空间评分更好）", css_class="heading"),
            _svg_text(90, 489, "S1 − S0"),
            _svg_text(90, 553, "S1 − SP"),
        ]
    )
    lines.extend(
        _svg_metric_bar(
            x=255,
            y=484,
            width=950,
            interval=summary.metrics["S1_minus_S0:IG"],
            scale=ig_scale,
            css_class="pointA",
            metric="IG",
        )
    )
    lines.extend(
        _svg_metric_bar(
            x=255,
            y=548,
            width=950,
            interval=summary.metrics["S1_minus_SP:IG"],
            scale=ig_scale,
            css_class="pointB",
            metric="IG",
        )
    )
    lines.extend(
        [
            _svg_box(60, 635, 1480, 205, "panel"),
            _svg_text(
                90,
                673,
                "严格召回增量（固定报警面积 ≤ 600,000 km²；正值表示命中比例提高）",
                css_class="heading",
            ),
            _svg_text(90, 719, "S1 − S0"),
            _svg_text(90, 783, "S1 − SP"),
        ]
    )
    lines.extend(
        _svg_metric_bar(
            x=255,
            y=714,
            width=950,
            interval=summary.metrics["S1_minus_S0:recall"],
            scale=recall_scale,
            css_class="pointA",
            metric="recall",
        )
    )
    lines.extend(
        _svg_metric_bar(
            x=255,
            y=778,
            width=950,
            interval=summary.metrics["S1_minus_SP:recall"],
            scale=recall_scale,
            css_class="pointB",
            metric="recall",
        )
    )
    gate_class = "green" if summary.gate_status == "passed_development_signal" else "warning"
    lines.extend(
        [
            _svg_text(60, 895, "门控结论与解释边界", css_class="section"),
            _svg_box(60, 920, 1480, 132, gate_class),
            _svg_text(90, 960, summary.gate_label, css_class="value"),
            _svg_text(
                90,
                993,
                (
                    "这表示合成流程按预期工作，不是预测提升证据。"
                    if summary.mode_code == "SYNTHETIC"
                    else "这只是在复用开发期历史回溯中达到预注册标准，仍需独立前瞻检验。"
                ),
            ),
            _svg_text(
                90,
                1022,
                "图中只表达相对空间强度、信息增益和固定面积召回；不表示绝对发震概率。",
                css_class="small",
            ),
            '<line x1="60" y1="1080" x2="1540" y2="1080" class="divider"/>',
            _svg_text(
                60,
                1103,
                f"experiment_id: {summary.experiment_id}｜available_at：{payload.available_at_cutoff}",
                css_class="small",
            ),
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _regional_result_rows(summary: _Summary) -> tuple[tuple[str, Mapping[str, object]], ...]:
    raw_results = summary.regional.get("results", {})
    if not isinstance(raw_results, Mapping):
        return ()
    values: list[tuple[str, Mapping[str, object]]] = []
    for key in sorted(raw_results, key=lambda item: str(item).encode("utf-8")):
        raw_value = raw_results[key]
        if isinstance(key, str) and isinstance(raw_value, Mapping):
            values.append((key, cast(Mapping[str, object], raw_value)))
    return tuple(values)


def _regional_zone_rows(summary: _Summary) -> tuple[Mapping[str, object], ...]:
    raw_regions = _sequence(
        summary.regional.get("regions", ()),
        label="regional_evidence.regions",
    )
    regions: list[Mapping[str, object]] = []
    seen_zone_ids: set[str] = set()
    required_contributions = {
        f"{contrast}:{metric}" for contrast in CONTRAST_IDS for metric in METRIC_IDS
    }
    for index, raw_region in enumerate(raw_regions):
        region = _mapping(raw_region, label=f"regional_evidence.regions[{index}]")
        zone_id = _text(region.get("zone_id"), label="regional zone_id")
        if zone_id in seen_zone_ids:
            raise Stage2SRenderingError("regional zone IDs must be unique")
        seen_zone_ids.add(zone_id)
        raw_contributions = _mapping(
            region.get("contributions"),
            label=f"regional contributions {zone_id}",
        )
        if set(raw_contributions) != required_contributions:
            raise Stage2SRenderingError("each regional row must contain both contrasts and metrics")
        contributions = {
            key: _finite_number(value, label=f"regional contribution {zone_id}:{key}")
            for key, value in raw_contributions.items()
        }
        regions.append(
            MappingProxyType(
                {
                    "zone_id": zone_id,
                    "ig_event_count": int(
                        _finite_number(
                            region.get("ig_event_count"),
                            label=f"regional IG event count {zone_id}",
                        )
                    ),
                    "recall_event_count": int(
                        _finite_number(
                            region.get("recall_event_count"),
                            label=f"regional recall event count {zone_id}",
                        )
                    ),
                    "contributions": MappingProxyType(contributions),
                }
            )
        )
    if len(regions) != 39:
        raise Stage2SRenderingError("regional explorer requires all 39 frozen zones")
    return tuple(
        sorted(
            regions,
            key=lambda item: cast(str, item["zone_id"]).encode("utf-8"),
        )
    )


def _optional_fraction(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, label=label)


def _sequence_diagnostic_view(summary: _Summary) -> dict[str, object]:
    sequence = summary.sequence
    component_count = _identity_integer(
        sequence.get("component_count"),
        label="sequence_evidence.component_count",
    )
    event_unit_count = _identity_integer(
        sequence.get("event_resampling_unit_count"),
        label="sequence_evidence.event_resampling_unit_count",
    )
    largest_count_id = _text(
        sequence.get("largest_count_component_id"),
        label="sequence_evidence.largest_count_component_id",
    )
    components = tuple(
        _mapping(item, label=f"sequence_evidence.components[{index}]")
        for index, item in enumerate(
            _sequence(
                sequence.get("components"),
                label="sequence_evidence.components",
            )
        )
    )
    largest_count = next(
        (
            component
            for component in components
            if component.get("component_id") == largest_count_id
        ),
        None,
    )
    if largest_count is None:
        raise Stage2SRenderingError("largest-count sequence component detail is missing")
    model_hits = _mapping(
        largest_count.get("model_hits"),
        label="largest-count sequence model_hits",
    )
    model_hit_fractions = {
        model_id: _optional_fraction(
            _mapping(
                model_hits.get(model_id),
                label=f"largest-count sequence model_hits.{model_id}",
            ).get("fraction"),
            label=f"largest-count sequence model_hits.{model_id}.fraction",
        )
        for model_id in MODEL_IDS
    }
    information_gain = _mapping(
        largest_count.get("information_gain"),
        label="largest-count sequence information_gain",
    )
    information_gain_fractions = {
        contrast: _optional_fraction(
            _mapping(
                information_gain.get(contrast),
                label=f"largest-count sequence information_gain.{contrast}",
            ).get("fraction"),
            label=f"largest-count sequence information_gain.{contrast}.fraction",
        )
        for contrast in CONTRAST_IDS
    }
    leave_largest_count = _mapping(
        sequence.get("leave_largest_count_out"),
        label="sequence_evidence.leave_largest_count_out",
    )
    leave_largest_gain = _mapping(
        sequence.get("leave_largest_gain_out"),
        label="sequence_evidence.leave_largest_gain_out",
    )
    largest_gain_ids = _mapping(
        sequence.get("largest_gain_component_id"),
        label="sequence_evidence.largest_gain_component_id",
    )
    metric_keys = tuple(
        f"{contrast}:{metric}" for contrast in CONTRAST_IDS for metric in METRIC_IDS
    )
    return {
        "component_count": component_count,
        "event_resampling_unit_count": event_unit_count,
        "claim_limited": sequence.get("claim_limited"),
        "interpretation_limit": sequence.get("interpretation_limit"),
        "largest_count_component": {
            "component_id": largest_count_id,
            "event_count": _identity_integer(
                largest_count.get("event_count"),
                label="largest-count sequence event_count",
            ),
            "event_fraction": _finite_number(
                largest_count.get("event_fraction"),
                label="largest-count sequence event_fraction",
            ),
            "origin_time_span_days": _finite_number(
                largest_count.get("origin_time_span_days"),
                label="largest-count sequence origin_time_span_days",
            ),
            "max_pairwise_geodesic_distance_km": _finite_number(
                largest_count.get("max_pairwise_geodesic_distance_km"),
                label="largest-count sequence max_pairwise_geodesic_distance_km",
            ),
            "model_hit_fractions": model_hit_fractions,
            "information_gain_fractions": information_gain_fractions,
        },
        "leave_largest_count_out": {
            key: _finite_number(
                leave_largest_count.get(key),
                label=f"sequence_evidence.leave_largest_count_out.{key}",
            )
            for key in metric_keys
        },
        "largest_gain_component_id": {
            key: _text(
                largest_gain_ids.get(key),
                label=f"sequence_evidence.largest_gain_component_id.{key}",
            )
            for key in metric_keys
        },
        "leave_largest_gain_out": {
            key: _finite_number(
                leave_largest_gain.get(key),
                label=f"sequence_evidence.leave_largest_gain_out.{key}",
            )
            for key in metric_keys
        },
    }


def _display_fraction(value: object) -> str:
    if value is None:
        return "不可估"
    return f"{_finite_number(value, label='display fraction') * 100:.1f}%"


def _short_identifier(value: object, *, limit: int = 13) -> str:
    text = _text(value, label="sequence component identifier")
    return text if len(text) <= limit else f"{text[:limit]}…"


def _sequence_leave_out_text(
    values: Mapping[str, object],
    *,
    contrast: ContrastId,
) -> str:
    return (
        f"IG {_display_short('IG', _finite_number(values[f'{contrast}:IG'], label='IG leave-out'))}"
        " · 召回 "
        f"{_display_short('recall', _finite_number(values[f'{contrast}:recall'], label='recall leave-out'))}"
    )


def render_diagnostics_svg(payload: Stage2SRenderPayload) -> bytes:
    """Render fold/horizon, regional, LORO, sequence, latency, and failure evidence."""

    summary = _extract_summary(payload.record)
    sequence_view = _sequence_diagnostic_view(summary)
    lines = _svg_header(
        title="Stage 2S 稳健性与失败案例检查",
        description=("逐折逐窗口展示信息增益和召回，并汇总区域、留一地区、震群、延迟与失败原因。"),
        summary=summary,
        width=1600,
        height=1500,
    )
    lines.extend(
        [
            _svg_text(60, 145, "逐折 × 预测窗口：S1 相对两个基线", css_class="section"),
            _svg_text(
                60,
                174,
                "每格依次为 S1−S0 的 IG / 召回增量，以及 S1−SP 的 IG / 召回增量。",
                css_class="small",
            ),
        ]
    )
    cell_width = 460
    cell_height = 128
    for index, cell in enumerate(summary.cells):
        row = index // 3
        column = index % 3
        x = 60 + column * 505
        y = 200 + row * 150
        lines.extend(
            [
                _svg_box(x, y, cell_width, cell_height, "panel"),
                _svg_text(
                    x + 22,
                    y + 34,
                    f"Fold {cell.fold_index}｜{cell.horizon_days} 天",
                    css_class="heading",
                ),
                _svg_text(
                    x + 22,
                    y + 67,
                    "S1−S0  "
                    f"IG {_display_short('IG', cell.values['S1_minus_S0:IG'])}  ·  "
                    "召回 "
                    f"{_display_short('recall', cell.values['S1_minus_S0:recall'])}",
                ),
                _svg_text(
                    x + 22,
                    y + 98,
                    "S1−SP  "
                    f"IG {_display_short('IG', cell.values['S1_minus_SP:IG'])}  ·  "
                    "召回 "
                    f"{_display_short('recall', cell.values['S1_minus_SP:recall'])}",
                ),
            ]
        )
    lines.extend(
        [
            _svg_text(60, 690, "区域分散性与 LORO（剔除最强正贡献区后）", css_class="section"),
            _svg_box(60, 715, 920, 375, "panel"),
        ]
    )
    region_rows = _regional_result_rows(summary)
    if region_rows:
        for index, (key, result) in enumerate(region_rows[:6]):
            y = 755 + index * 38
            positive = result.get("positive_event_bearing_zone_count", "—")
            total = result.get("event_bearing_zone_count", "—")
            residual = result.get("leave_strongest_out_residual", "—")
            passed = result.get("passed", "—")
            residual_text = (
                f"{_finite_number(residual, label='regional residual'):+.3f}"
                if isinstance(residual, int | float) and not isinstance(residual, bool)
                else str(residual)
            )
            lines.extend(
                [
                    _svg_text(88, y, key.replace(":", " / "), css_class="heading"),
                    _svg_text(345, y, f"正贡献区 {positive}/{total}"),
                    _svg_text(565, y, f"LORO 残差 {residual_text}"),
                    _svg_text(810, y, f"通过：{passed}"),
                ]
            )
    else:
        lines.append(_svg_text(88, 760, "区域明细不可用；门控必须按证据不足处理。"))
    largest_count = _mapping(
        sequence_view["largest_count_component"],
        label="sequence diagnostic largest_count_component",
    )
    model_hit_fractions = _mapping(
        largest_count["model_hit_fractions"],
        label="sequence diagnostic model_hit_fractions",
    )
    information_gain_fractions = _mapping(
        largest_count["information_gain_fractions"],
        label="sequence diagnostic information_gain_fractions",
    )
    leave_largest_count = _mapping(
        sequence_view["leave_largest_count_out"],
        label="sequence diagnostic leave_largest_count_out",
    )
    largest_gain_ids = _mapping(
        sequence_view["largest_gain_component_id"],
        label="sequence diagnostic largest_gain_component_id",
    )
    leave_largest_gain = _mapping(
        sequence_view["leave_largest_gain_out"],
        label="sequence diagnostic leave_largest_gain_out",
    )
    lines.extend(
        [
            _svg_box(1010, 715, 530, 375, "purple"),
            _svg_text(1038, 755, "震群/序列敏感性", css_class="heading"),
            _svg_text(
                1038,
                785,
                (
                    f"事件块重采样单位 {sequence_view['event_resampling_unit_count']}｜"
                    f"震群 {sequence_view['component_count']}｜"
                    f"限制结论 {sequence_view['claim_limited']}"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                813,
                (
                    f"最大震群 {_short_identifier(largest_count['component_id'])}｜"
                    f"事件 {largest_count['event_count']}（"
                    f"{_display_fraction(largest_count['event_fraction'])}）"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                839,
                (
                    f"跨度 {_finite_number(largest_count['origin_time_span_days'], label='span'):.1f} 天｜"
                    "最大两两距离 "
                    f"{_finite_number(largest_count['max_pairwise_geodesic_distance_km'], label='distance'):.1f} km"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                865,
                (
                    "命中占比 "
                    f"S0 {_display_fraction(model_hit_fractions['S0'])}｜"
                    f"S1 {_display_fraction(model_hit_fractions['S1'])}｜"
                    f"SP {_display_fraction(model_hit_fractions['SP'])}"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                891,
                (
                    "IG 占比 "
                    f"S1−S0 {_display_fraction(information_gain_fractions['S1_minus_S0'])}｜"
                    f"S1−SP {_display_fraction(information_gain_fractions['S1_minus_SP'])}"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                919,
                "去最大事件数震群 S1−S0｜"
                + _sequence_leave_out_text(
                    leave_largest_count,
                    contrast="S1_minus_S0",
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                945,
                "去最大事件数震群 S1−SP｜"
                + _sequence_leave_out_text(
                    leave_largest_count,
                    contrast="S1_minus_SP",
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                973,
                (
                    "去最大增益 S1−S0 IG｜"
                    f"{_short_identifier(largest_gain_ids['S1_minus_S0:IG'])} "
                    f"{_display_short('IG', _finite_number(leave_largest_gain['S1_minus_S0:IG'], label='largest gain IG leave-out'))}"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                999,
                (
                    "去最大增益 S1−S0 召回｜"
                    f"{_short_identifier(largest_gain_ids['S1_minus_S0:recall'])} "
                    f"{_display_short('recall', _finite_number(leave_largest_gain['S1_minus_S0:recall'], label='largest gain recall leave-out'))}"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                1025,
                (
                    "去最大增益 S1−SP IG｜"
                    f"{_short_identifier(largest_gain_ids['S1_minus_SP:IG'])} "
                    f"{_display_short('IG', _finite_number(leave_largest_gain['S1_minus_SP:IG'], label='largest gain SP IG leave-out'))}"
                ),
                css_class="small",
            ),
            _svg_text(
                1038,
                1051,
                (
                    "去最大增益 S1−SP 召回｜"
                    f"{_short_identifier(largest_gain_ids['S1_minus_SP:recall'])} "
                    f"{_display_short('recall', _finite_number(leave_largest_gain['S1_minus_SP:recall'], label='largest gain SP recall leave-out'))}"
                ),
                css_class="small",
            ),
            _svg_text(60, 1110, "报告延迟敏感性", css_class="section"),
        ]
    )
    for index, latency in enumerate(summary.latency):
        delay = latency.get("delay_days", "—")
        metrics = latency.get("metrics", {})
        metric_map = cast(Mapping[str, object], metrics) if isinstance(metrics, Mapping) else {}
        x = 60 + index * 750
        lines.extend(
            [
                _svg_box(x, 1135, 710, 130, "blue" if index == 0 else "orange"),
                _svg_text(x + 25, 1173, f"额外延迟 {delay} 天", css_class="heading"),
                _svg_text(
                    x + 25,
                    1207,
                    "S1−S0  "
                    "IG "
                    f"{_display_short('IG', _finite_number(metric_map.get('S1_minus_S0:IG'), label='latency IG'))}  ·  "
                    "召回 "
                    f"{_display_short('recall', _finite_number(metric_map.get('S1_minus_S0:recall'), label='latency recall'))}",
                ),
                _svg_text(
                    x + 25,
                    1237,
                    "S1−SP  "
                    "IG "
                    f"{_display_short('IG', _finite_number(metric_map.get('S1_minus_SP:IG'), label='latency SP IG'))}  ·  "
                    "召回 "
                    f"{_display_short('recall', _finite_number(metric_map.get('S1_minus_SP:recall'), label='latency SP recall'))}",
                ),
            ]
        )
    lines.extend(
        [
            _svg_text(60, 1290, "失败/停止原因", css_class="section"),
            _svg_box(60, 1315, 1480, 102, "warning"),
        ]
    )
    reason_text = (
        "；".join(summary.gate_reasons[:3])
        if summary.gate_reasons
        else "无登记失败原因；仍须遵守证据范围与独立验证边界。"
    )
    lines.extend(
        [
            _svg_text(88, 1355, f"门控：{summary.gate_label}", css_class="heading"),
            _svg_text(88, 1388, reason_text[:145], css_class="small"),
            _svg_text(
                60,
                1455,
                "相对强度不是绝对发震概率；历史开发回溯也不是当前或前瞻预测。",
                css_class="small",
            ),
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _first_map_group(payload: Stage2SRenderPayload) -> tuple[Stage2SMapFrame, ...]:
    first = payload.map_frames[0]
    key = (first.fold_index, first.horizon_days, first.issue_time_utc)
    frames = tuple(
        frame
        for frame in payload.map_frames
        if (frame.fold_index, frame.horizon_days, frame.issue_time_utc) == key
    )
    by_model = {frame.model_id: frame for frame in frames}
    return tuple(by_model[model_id] for model_id in MODEL_IDS)


def _svg_map_panel(
    frame: Stage2SMapFrame,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    rows = frame.relative_intensity_rank
    alarm_fractions = frame.alarm_area_fraction()
    row_count = len(rows)
    column_count = len(rows[0])
    cell_size = min(width / column_count, height / row_count)
    actual_width = cell_size * column_count
    actual_height = cell_size * row_count
    offset_x = x + (width - actual_width) / 2.0
    offset_y = y + (height - actual_height) / 2.0
    lines: list[str] = []
    for row_index, (row, fraction_row) in enumerate(zip(rows, alarm_fractions, strict=True)):
        for column_index, (value, alarm_fraction) in enumerate(zip(row, fraction_row, strict=True)):
            if value is None:
                continue
            cell_x = offset_x + column_index * cell_size
            cell_y = offset_y + row_index * cell_size
            stroke = "#f7f9fc"
            stroke_width = max(0.15, cell_size * 0.025)
            lines.append(
                f'<rect x="{cell_x:.3f}" y="{cell_y:.3f}" width="{cell_size + 0.04:.3f}" '
                f'height="{cell_size + 0.04:.3f}" fill="{_palette_color(value)}" '
                f'stroke="{stroke}" stroke-width="{stroke_width:.3f}"/>'
            )
            if alarm_fraction is not None and alarm_fraction > 0.0:
                lines.append(
                    f'<rect x="{cell_x + stroke_width:.3f}" y="{cell_y + stroke_width:.3f}" '
                    f'width="{max(0.0, cell_size - 2 * stroke_width):.3f}" '
                    f'height="{max(0.0, cell_size - 2 * stroke_width):.3f}" '
                    f'fill="#111827" fill-opacity="{alarm_fraction:.6f}" '
                    'stroke="none"/>'
                )
    return lines


def render_relative_intensity_map_svg(payload: Stage2SRenderPayload) -> bytes:
    """Render one representative historical S0/S1/SP map triplet."""

    summary = _extract_summary(payload.record)
    frames = _first_map_group(payload)
    first = frames[0]
    lines = _svg_header(
        title="Stage 2S 历史相对强度地图",
        description=("同一历史起报时点下 S0、S1、SP 的派生相对强度秩栅格及固定面积报警区。"),
        summary=summary,
        width=1600,
        height=980,
    )
    lines.extend(
        [
            _svg_text(
                60,
                140,
                (
                    f"Fold {first.fold_index}｜{first.horizon_days} 天｜"
                    f"历史起报 {first.issue_time_utc}｜数据截止 {first.data_cutoff_utc}"
                ),
                css_class="section",
            ),
            _svg_text(
                60,
                171,
                "颜色表示模型内相对强度顺位；深色覆盖透明度等于概化像素内正式报警面积占比。",
                css_class="small",
            ),
        ]
    )
    panel_width = 460
    positions = (60, 570, 1080)
    for frame, x in zip(frames, positions, strict=True):
        lines.extend(
            [
                _svg_box(x, 205, panel_width, 615, "panel"),
                _svg_text(
                    x + panel_width / 2,
                    243,
                    _MODEL_LABELS[frame.model_id],
                    css_class="heading",
                    anchor="middle",
                ),
            ]
        )
        lines.extend(
            _svg_map_panel(
                frame,
                x=x + 20,
                y=265,
                width=panel_width - 40,
                height=465,
            )
        )
        lines.extend(
            [
                _svg_text(
                    x + panel_width / 2,
                    768,
                    f"正式 600,000 km² 门实际入选：{frame.actual_alarm_area_km2:,.0f} km²",
                    anchor="middle",
                ),
                _svg_text(
                    x + panel_width / 2,
                    795,
                    "派生秩栅格；未嵌入原始查询网格坐标表",
                    css_class="small",
                    anchor="middle",
                ),
            ]
        )
    legend_x = 435
    for index, color in enumerate(_PALETTE):
        lines.append(
            f'<rect x="{legend_x + index * 80}" y="858" width="80" height="20" fill="{color}"/>'
        )
    lines.extend(
        [
            _svg_text(415, 903, "较低", css_class="small", anchor="end"),
            _svg_text(795, 903, "模型内相对强度顺位", css_class="small", anchor="middle"),
            _svg_text(1175, 903, "较高", css_class="small"),
            _svg_text(
                60,
                947,
                (
                    f"{summary.status_banner}｜相对强度不是绝对发震概率；"
                    "本图不预测未来地震总数或具体日期。"
                ),
                css_class="small",
            ),
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _configure_metric_axis(
    axis: Axes,
    *,
    summary: _Summary,
    metric: MetricId,
) -> None:
    intervals = [summary.metrics[_metric_key(contrast, metric)] for contrast in CONTRAST_IDS]
    scale = max(
        1.0e-12,
        *(abs(value) for interval in intervals for value in (interval.lower, interval.upper)),
    )
    factor = 100.0 if metric == "recall" else 1.0
    y_positions = np.array([1.0, 0.0], dtype=np.float64)
    points = np.array([interval.point * factor for interval in intervals], dtype=np.float64)
    lower_errors = np.array(
        [(interval.point - interval.lower) * factor for interval in intervals],
        dtype=np.float64,
    )
    upper_errors = np.array(
        [(interval.upper - interval.point) * factor for interval in intervals],
        dtype=np.float64,
    )
    axis.axvline(0.0, color="#627d98", linewidth=0.8)
    axis.errorbar(
        points,
        y_positions,
        xerr=np.vstack((lower_errors, upper_errors)),
        fmt="o",
        color="#102a43",
        markerfacecolor="#2f80ed",
        markeredgecolor="#102a43",
        markersize=7,
        capsize=4,
        linewidth=1.6,
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(("S1 − S0", "S1 − SP"), fontproperties=_PNG_FONT, fontsize=10)
    adjusted_scale = scale * factor
    if metric == "recall":
        adjusted_scale = max(adjusted_scale, _RECALL_DISPLAY_HALF_RANGE_FLOOR_PP)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set_xlim(-adjusted_scale * 1.18, adjusted_scale * 1.18)
    axis.grid(axis="x", color="#d9e2ec", linewidth=0.6)
    axis.tick_params(axis="x", labelsize=9, width=0.6)
    axis.set_title(
        ("信息增益 IG（对数评分差）" if metric == "IG" else "严格召回增量（百分点；报警面积固定）"),
        fontproperties=_PNG_FONT,
        fontsize=11,
        pad=9,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    for y_position, interval in zip(y_positions, intervals, strict=True):
        axis.text(
            interval.point * factor,
            y_position + 0.22,
            _display_short(metric, interval.point),
            fontproperties=_PNG_FONT,
            fontsize=9,
            ha="center",
            va="bottom",
            color="#102a43",
        )


def render_overall_results_png(payload: Stage2SRenderPayload) -> bytes:
    """Render a deterministic PNG companion to the principal SVG."""

    summary = _extract_summary(payload.record)
    figure = Figure(figsize=(16.0, 10.0), dpi=100, facecolor="#f7f9fc")
    canvas = FigureCanvasAgg(figure)
    figure.text(
        0.04,
        0.945,
        "Stage 2S 总体效果：同一报警面积下的公平比较",
        fontproperties=_PNG_FONT,
        fontsize=22,
        color="#102a43",
        ha="left",
        va="top",
    )
    figure.text(
        0.04,
        0.902,
        summary.status_banner,
        fontproperties=_PNG_FONT,
        fontsize=12,
        color="#c0392b",
        ha="left",
        va="top",
    )
    method_lines = (
        "S0：长期 M4+ 地震背景",
        "S1：S0 + 起报前最近 30 天 M4+ 地震",
        "SP：S0 + 紧邻的过去 30 天 M4+ 地震对照",
    )
    for index, value in enumerate(method_lines):
        figure.text(
            0.06 + index * 0.32,
            0.835,
            value,
            fontproperties=_PNG_FONT,
            fontsize=11,
            color="#102a43",
            bbox={
                "boxstyle": "round,pad=0.7",
                "facecolor": ("#eaf2ff", "#e8f7f1", "#fff3e7")[index],
                "edgecolor": ("#2f80ed", "#2d9d78", "#f2994a")[index],
                "linewidth": 1.0,
            },
            ha="left",
            va="center",
        )
    ig_axis = figure.add_axes(_IG_AXIS_BOUNDS, facecolor="#ffffff")
    recall_axis = figure.add_axes(_RECALL_AXIS_BOUNDS, facecolor="#ffffff")
    _configure_metric_axis(ig_axis, summary=summary, metric="IG")
    _configure_metric_axis(recall_axis, summary=summary, metric="recall")
    figure.text(
        0.04,
        0.115,
        f"门控：{summary.gate_label}",
        fontproperties=_PNG_FONT,
        fontsize=15,
        color="#102a43",
        ha="left",
        va="center",
    )
    figure.text(
        0.04,
        0.074,
        summary.scope_note,
        fontproperties=_PNG_FONT,
        fontsize=10,
        color="#486581",
        ha="left",
        va="center",
    )
    figure.text(
        0.04,
        0.041,
        "只表达相对空间强度与固定面积效果；不是绝对发震概率。",
        fontproperties=_PNG_FONT,
        fontsize=10,
        color="#486581",
        ha="left",
        va="center",
    )
    output = io.BytesIO()
    canvas.print_png(  # type: ignore[no-untyped-call]
        output,
        metadata={
            "Software": "SeismoFlux deterministic Stage2S renderer",
            "Title": "Stage2S same-area information gain and recall",
            "EvidenceMode": summary.mode_code,
            "GateStatus": summary.gate_status,
            "ProbabilityClaim": "relative intensity only; not absolute probability",
        },
        pil_kwargs={"compress_level": 9, "optimize": False},
    )
    result = output.getvalue()
    figure.clear()
    return result


def _fold_provenance(record: Stage2SWholeRunRecord) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for raw in record.fold_fit_summaries:
        item = _mapping(raw, label="fold fit summary")
        values.append(
            {
                "fold_index": item.get("fold_index"),
                "alpha_R_by_delay": item.get("alpha_R_by_delay", item.get("alpha_R", "—")),
                "alpha_P_by_delay": item.get("alpha_P_by_delay", item.get("alpha_P", "—")),
                "shared_rate_per_day": item.get("shared_rate_per_day", "—"),
            }
        )
    return values


def _seal_provenance(record: Stage2SWholeRunRecord) -> dict[str, object]:
    seals = _mapping(record.seal_chain, label="seal_chain")
    return {
        "fold_fit_receipt_sha256": seals.get("fold_fit_receipt_sha256", "—"),
        "issue_prediction_seal_sha256": seals.get("issue_prediction_seal_sha256", "—"),
        "fold_prediction_seal_sha256": seals.get("fold_prediction_seal_sha256", "—"),
        "master_prediction_seal_sha256": seals.get("master_prediction_seal_sha256", "—"),
    }


def _actual_alarm_areas(record: Stage2SWholeRunRecord) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for raw in record.issue_prediction_summaries:
        item = _mapping(raw, label="issue prediction summary")
        values.append(
            {
                "fold_index": item.get("fold_index"),
                "issue_date": item.get("issue_date", item.get("issue_time_utc", "—")),
                "actual_alarm_area_km2": item.get("actual_alarm_area_km2", "—"),
                "issue_prediction_seal_sha256": item.get(
                    "issue_prediction_seal_sha256",
                    "—",
                ),
            }
        )
    return values


def _provenance_mapping(payload: Stage2SRenderPayload) -> dict[str, object]:
    return {
        "S0_training_cutoff": payload.s0_training_cutoff_utc,
        "R_and_RP_origin_windows": {
            "R": payload.recent_origin_window,
            "RP": payload.preceding_origin_window,
        },
        "available_at_cutoff": payload.available_at_cutoff,
        "fold_alpha_R_alpha_P_and_shared_rate": _fold_provenance(payload.record),
        "fold_and_horizon": [
            {"fold_index": cell["fold_index"], "horizon_days": cell["horizon_days"]}
            for cell in payload.record.cell_scores
        ],
        "issue_fold_and_master_seal_sha256": _seal_provenance(payload.record),
        "actual_alarm_area_km2": _actual_alarm_areas(payload.record),
        "map_frame_bindings": _record_map_frame_bindings(payload),
    }


def _backtest_data(payload: Stage2SRenderPayload) -> dict[str, object]:
    summary = _extract_summary(payload.record)
    regional_results = {key: dict(value) for key, value in _regional_result_rows(summary)}
    expected_regional_keys = {
        f"{contrast}:{metric}" for contrast in CONTRAST_IDS for metric in METRIC_IDS
    }
    if set(regional_results) != expected_regional_keys:
        raise Stage2SRenderingError("regional explorer requires both contrasts and metrics")
    regions = [dict(value) for value in _regional_zone_rows(summary)]
    cell_failures = [
        {
            "fold_index": cell.fold_index,
            "horizon_days": cell.horizon_days,
            "contrast": contrast,
            "metric": metric,
            "value": cell.values[f"{contrast}:{metric}"],
        }
        for cell in summary.cells
        for contrast in CONTRAST_IDS
        for metric in METRIC_IDS
        if cell.values[f"{contrast}:{metric}"] <= 0.0
    ]
    latency_rows: list[dict[str, object]] = []
    required_latency_keys = {
        f"{contrast}:{metric}" for contrast in CONTRAST_IDS for metric in METRIC_IDS
    }
    for expected_delay, raw_latency in zip((1, 7), summary.latency, strict=True):
        delay = int(
            _finite_number(
                raw_latency.get("delay_days"),
                label="latency delay_days",
            )
        )
        metrics = _mapping(raw_latency.get("metrics"), label=f"latency {delay} metrics")
        if delay != expected_delay or set(metrics) != required_latency_keys:
            raise Stage2SRenderingError(
                "latency explorer requires ordered 1/7-day values for both contrasts"
            )
        latency_rows.append(
            {
                "delay_days": delay,
                "metrics": {
                    key: _finite_number(value, label=f"latency {delay}:{key}")
                    for key, value in metrics.items()
                },
            }
        )
    return {
        "mode": summary.mode_code,
        "status_banner": summary.status_banner,
        "scope_note": summary.scope_note,
        "gate_status": summary.gate_status,
        "gate_label": summary.gate_label,
        "gate_reasons": list(summary.gate_reasons),
        "metrics": {
            key: {
                "point": value.point,
                "lower": value.lower,
                "upper": value.upper,
            }
            for key, value in summary.metrics.items()
        },
        "cells": [
            {
                "fold_index": cell.fold_index,
                "horizon_days": cell.horizon_days,
                "values": dict(cell.values),
            }
            for cell in summary.cells
        ],
        "regional": {
            "passed": summary.regional.get("passed", "—"),
            "failures": summary.regional.get("failures", ()),
            "results": regional_results,
            "regions": regions,
        },
        "sequence": _sequence_diagnostic_view(summary),
        "latency": latency_rows,
        "cell_failures": cell_failures,
        "provenance": _provenance_mapping(payload),
    }


def build_backtest_explorer_html(payload: Stage2SRenderPayload) -> str:
    """Build a completely offline, coordinate-free backtest explorer."""

    data = _safe_json(_backtest_data(payload))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2S 历史开发回溯效果浏览器</title>
<style>
:root{{--ink:#102a43;--muted:#486581;--line:#bcccdc;--paper:#f7f9fc;--panel:#fff;
--blue:#2f80ed;--orange:#f2994a;--green:#2d9d78;--warn:#c0392b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px}}h1{{margin:0 0 8px;font-size:28px}}
h2{{margin:30px 0 14px;font-size:20px}}p{{line-height:1.65}}.banner{{border:2px solid var(--warn);
background:#fff0f0;padding:14px 18px;border-radius:12px;font-weight:700}}.scope{{color:var(--muted)}}
.methods,.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.metrics{{grid-template-columns:1fr 1fr}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
 .method b{{display:block;margin-bottom:8px}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
 button,select{{font:inherit;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:8px;
 padding:9px 13px;cursor:pointer}}button[aria-pressed="true"],button.selected{{background:var(--ink);color:#fff}}
.metric-value{{font-size:25px;font-weight:700;margin:8px 0}}.interval{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.cell{{text-align:left;min-height:90px}}
.cell strong,.cell span{{display:block}}.cell span{{margin-top:7px}}dl{{display:grid;
grid-template-columns:minmax(240px,1fr) 2fr;gap:8px 20px}}dt{{font-weight:700}}dd{{margin:0;
word-break:break-word}}pre{{white-space:pre-wrap;word-break:break-word;background:#eef3f8;padding:12px;
 border-radius:8px}}.warning{{border-left:5px solid var(--warn);padding-left:14px}}
 .failure-list{{margin:0;padding-left:22px;line-height:1.65}}.failure-list:empty::before{{
 content:"没有登记失败项";color:var(--muted)}}.region-line{{line-height:1.75}}
@media(max-width:760px){{main{{padding:16px}}.methods,.metrics,.grid{{grid-template-columns:1fr}}
dl{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<h1>Stage 2S 历史开发回溯效果</h1>
<div class="banner" id="status"></div>
<p class="scope" id="scope"></p>
<h2>方法</h2>
<section class="methods" aria-label="三个公平比较模型">
<div class="card method"><b>S0 长期背景</b>回答“哪里长期更活跃”。</div>
<div class="card method"><b>S1 最近地震候选</b>S0 加起报前最近 30 天 M4+ 地震。</div>
<div class="card method"><b>SP 过去窗口对照</b>S0 加紧邻的更早 30 天 M4+ 地震。</div>
</section>
<h2>总体效果</h2>
<div class="controls" aria-label="选择比较基线">
<button type="button" data-contrast="S1_minus_S0" aria-pressed="true">S1 − S0</button>
<button type="button" data-contrast="S1_minus_SP" aria-pressed="false">S1 − SP</button>
</div>
<section class="metrics" aria-live="polite">
<div class="card"><b>信息增益 IG</b><div class="metric-value" id="ig-value"></div>
<div class="interval" id="ig-ci"></div></div>
<div class="card"><b>严格召回增量（固定报警面积）</b><div class="metric-value" id="recall-value"></div>
<div class="interval" id="recall-ci"></div></div>
</section>
<h2>逐折 × 预测窗口</h2>
 <div class="grid" id="cell-grid"></div>
 <section class="card" id="cell-detail" aria-live="polite"></section>
 <h2>39 个冻结区域贡献与 LORO</h2>
 <div class="controls">
 <label>选择冻结区域
 <select id="region-select" aria-label="选择冻结 construction zone"></select></label>
 </div>
 <section class="card region-line" id="region-detail" aria-live="polite"></section>
 <h2>震群/序列敏感性</h2>
 <section class="card" aria-live="polite">
 <p id="sequence-summary"></p>
 <dl>
 <dt>最大事件数震群</dt><dd id="sequence-largest"></dd>
 <dt>该震群命中占比</dt><dd id="sequence-hit-fractions"></dd>
 <dt>该震群 IG 占比</dt><dd id="sequence-ig-fractions"></dd>
 <dt>去最大事件数震群</dt><dd><pre id="sequence-largest-count-out"></pre></dd>
 <dt>去各指标最大增益震群</dt><dd><pre id="sequence-largest-gain-out"></pre></dd>
 </dl>
 </section>
 <h2>目录可用延迟敏感性</h2>
 <div class="grid" id="latency-grid" aria-live="polite"></div>
 <h2>失败与脆弱案例</h2>
 <section class="card warning"><ul class="failure-list" id="failure-list"></ul></section>
 <h2>门控与稳健性</h2>
<section class="card warning">
<p><b id="gate-label"></b></p><p id="gate-detail"></p>
<p id="robustness"></p>
</section>
<h2>完整溯源</h2>
<section class="card">
<dl>
<dt>S0_training_cutoff</dt><dd id="prov-s0"></dd>
<dt>R_and_RP_origin_windows</dt><dd id="prov-windows"></dd>
<dt>available_at_cutoff</dt><dd id="prov-availability"></dd>
<dt>fold_alpha_R_alpha_P_and_shared_rate</dt><dd><pre id="prov-fold"></pre></dd>
<dt>fold_and_horizon</dt><dd id="prov-cells"></dd>
<dt>issue_fold_and_master_seal_sha256</dt><dd><pre id="prov-seals"></pre></dd>
<dt>actual_alarm_area_km2</dt><dd><pre id="prov-area"></pre></dd>
<dt>map_frame_bindings</dt><dd><pre id="prov-map-bindings"></pre></dd>
</dl>
</section>
<p class="scope"><b>解释边界：</b>这里只表达相对空间强度、信息增益和固定面积召回；
不是绝对发震概率，也不是当前或前瞻预测。</p>
</main>
<script>
"use strict";
 const DATA={data};
 let selectedContrast="S1_minus_S0";
 let selectedCell=DATA.cells[0];
 let selectedRegion=DATA.regional.regions[0];
 const fmtIG=v=>(v>=0?"+":"")+v.toFixed(3);
 const fmtRecall=v=>(v>=0?"+":"")+(v*100).toFixed(1)+" 个百分点";
function metric(key){{return DATA.metrics[selectedContrast+":"+key]}}
function renderOverall(){{
 const ig=metric("IG"),recall=metric("recall");
 document.getElementById("ig-value").textContent=fmtIG(ig.point);
 document.getElementById("ig-ci").textContent="同时区间 ["+fmtIG(ig.lower)+", "+fmtIG(ig.upper)+"]";
 document.getElementById("recall-value").textContent=fmtRecall(recall.point);
 document.getElementById("recall-ci").textContent="同时区间 ["+fmtRecall(recall.lower)+", "+fmtRecall(recall.upper)+"]";
}}
function renderCells(){{
 const grid=document.getElementById("cell-grid");grid.replaceChildren();
 DATA.cells.forEach(cell=>{{
  const button=document.createElement("button");button.type="button";button.className="cell";
  button.classList.toggle("selected",cell===selectedCell);
  button.setAttribute("aria-pressed",cell===selectedCell?"true":"false");
  const title=document.createElement("strong");title.textContent="Fold "+cell.fold_index+" · "+cell.horizon_days+" 天";
  const value=document.createElement("span");
  value.textContent="IG "+fmtIG(cell.values[selectedContrast+":IG"])+" · 召回 "+fmtRecall(cell.values[selectedContrast+":recall"]);
  button.append(title,value);button.addEventListener("click",()=>{{selectedCell=cell;renderCells();renderDetail()}});
  grid.append(button);
 }});
}}
function renderDetail(){{
 const values=selectedCell.values;
 document.getElementById("cell-detail").textContent=
 "当前格：Fold "+selectedCell.fold_index+"，"+selectedCell.horizon_days+" 天；"+
 "S1−S0 为 IG "+fmtIG(values["S1_minus_S0:IG"])+"、召回 "+fmtRecall(values["S1_minus_S0:recall"])+"；"+
 "S1−SP 为 IG "+fmtIG(values["S1_minus_SP:IG"])+"、召回 "+fmtRecall(values["S1_minus_SP:recall"])+"。";
 }}
 const regionSelect=document.getElementById("region-select");
 DATA.regional.regions.forEach(region=>{{
  const option=document.createElement("option");option.value=region.zone_id;
  option.textContent=region.zone_id;regionSelect.append(option);
 }});
 function renderRegion(){{
  const contribution=selectedRegion.contributions;
  const igResult=DATA.regional.results[selectedContrast+":IG"];
  const recallResult=DATA.regional.results[selectedContrast+":recall"];
  document.getElementById("region-detail").textContent=
   "区域 "+selectedRegion.zone_id+"｜IG 目标数 "+selectedRegion.ig_event_count+
   "｜召回目标数 "+selectedRegion.recall_event_count+"｜当前对比 IG 贡献 "+
   fmtIG(contribution[selectedContrast+":IG"])+"｜召回贡献 "+
   fmtRecall(contribution[selectedContrast+":recall"])+"。当前对比全局 LORO：IG 去最大贡献区 "+
   (igResult.strongest_positive_zone_id??"—")+" 后残差 "+
   fmtIG(igResult.leave_strongest_out_residual)+"；召回去最大贡献区 "+
   (recallResult.strongest_positive_zone_id??"—")+" 后残差 "+
   fmtRecall(recallResult.leave_strongest_out_residual)+"。";
 }}
 regionSelect.addEventListener("change",()=>{{
  selectedRegion=DATA.regional.regions.find(region=>region.zone_id===regionSelect.value);
 if(!selectedRegion)throw new Error("冻结区域缺失");renderRegion();
 }});
 function renderSequence(){{
  const sequence=DATA.sequence;
  const largest=sequence.largest_count_component;
  const pct=value=>value===null?"不可估":(value*100).toFixed(1)+"%";
  document.getElementById("sequence-summary").textContent=
   "Bootstrap 事件块重采样单位 "+sequence.event_resampling_unit_count+
   "；连通震群 "+sequence.component_count+"；是否限制结论："+sequence.claim_limited+"。";
  document.getElementById("sequence-largest").textContent=
   largest.component_id+"｜事件 "+largest.event_count+"（"+pct(largest.event_fraction)+
   "）｜跨度 "+largest.origin_time_span_days.toFixed(1)+" 天｜最大两两距离 "+
   largest.max_pairwise_geodesic_distance_km.toFixed(1)+" km";
  document.getElementById("sequence-hit-fractions").textContent=
   "S0 "+pct(largest.model_hit_fractions.S0)+"｜S1 "+pct(largest.model_hit_fractions.S1)+
   "｜SP "+pct(largest.model_hit_fractions.SP);
  document.getElementById("sequence-ig-fractions").textContent=
   "S1−S0 "+pct(largest.information_gain_fractions.S1_minus_S0)+
   "｜S1−SP "+pct(largest.information_gain_fractions.S1_minus_SP);
  const leaveText=values=>
   "S1−S0：IG "+fmtIG(values["S1_minus_S0:IG"])+"，召回 "+
   fmtRecall(values["S1_minus_S0:recall"])+"；S1−SP：IG "+
   fmtIG(values["S1_minus_SP:IG"])+"，召回 "+fmtRecall(values["S1_minus_SP:recall"]);
  document.getElementById("sequence-largest-count-out").textContent=
   leaveText(sequence.leave_largest_count_out);
  const gainIds=sequence.largest_gain_component_id,gainLeave=sequence.leave_largest_gain_out;
  document.getElementById("sequence-largest-gain-out").textContent=
   "S1−S0 IG "+gainIds["S1_minus_S0:IG"]+" → "+fmtIG(gainLeave["S1_minus_S0:IG"])+
   "；召回 "+gainIds["S1_minus_S0:recall"]+" → "+
   fmtRecall(gainLeave["S1_minus_S0:recall"])+"；S1−SP IG "+
   gainIds["S1_minus_SP:IG"]+" → "+fmtIG(gainLeave["S1_minus_SP:IG"])+
   "；召回 "+gainIds["S1_minus_SP:recall"]+" → "+
   fmtRecall(gainLeave["S1_minus_SP:recall"]);
 }}
 function renderLatency(){{
  const grid=document.getElementById("latency-grid");grid.replaceChildren();
  DATA.latency.forEach(item=>{{
   const metrics=item.metrics,card=document.createElement("section");card.className="card";
   const heading=document.createElement("b");heading.textContent="额外延迟 "+item.delay_days+" 天";
   const first=document.createElement("p");first.textContent="S1−S0：IG "+
    fmtIG(metrics["S1_minus_S0:IG"])+"，召回 "+fmtRecall(metrics["S1_minus_S0:recall"]);
   const second=document.createElement("p");second.textContent="S1−SP：IG "+
    fmtIG(metrics["S1_minus_SP:IG"])+"，召回 "+fmtRecall(metrics["S1_minus_SP:recall"]);
   card.append(heading,first,second);grid.append(card);
  }});
 }}
 function renderFailures(){{
  const failures=[...DATA.gate_reasons,...DATA.regional.failures];
  DATA.cell_failures.forEach(item=>failures.push(
   "Fold "+item.fold_index+" / "+item.horizon_days+" 天 / "+item.contrast+" / "+
   item.metric+" 未呈正向："+item.value));
  if(DATA.sequence.claim_limited===true)failures.push(
   "去除主导震群后优势不稳：结论仅支持震群相关续发，不支持广泛区域提升");
  const list=document.getElementById("failure-list");list.replaceChildren();
  [...new Set(failures)].forEach(text=>{{
   const item=document.createElement("li");item.textContent=text;list.append(item);
  }});
 }}
 document.querySelectorAll("[data-contrast]").forEach(button=>button.addEventListener("click",()=>{{
 selectedContrast=button.dataset.contrast;
 document.querySelectorAll("[data-contrast]").forEach(peer=>peer.setAttribute(
  "aria-pressed",peer===button?"true":"false"));
  renderOverall();renderCells();renderRegion();
}}));
document.getElementById("status").textContent=DATA.status_banner;
document.getElementById("scope").textContent=DATA.scope_note;
document.getElementById("gate-label").textContent=DATA.gate_label;
document.getElementById("gate-detail").textContent=DATA.gate_reasons.length?
 DATA.gate_reasons.join("；"):"无登记失败原因，但解释边界仍然有效。";
document.getElementById("robustness").textContent=
 "区域检查通过："+DATA.regional.passed+"；震群数："+DATA.sequence.component_count+
 "；震群限制结论："+DATA.sequence.claim_limited+"。";
const P=DATA.provenance;
document.getElementById("prov-s0").textContent=P.S0_training_cutoff;
document.getElementById("prov-windows").textContent=JSON.stringify(P.R_and_RP_origin_windows);
document.getElementById("prov-availability").textContent=P.available_at_cutoff;
document.getElementById("prov-fold").textContent=JSON.stringify(P.fold_alpha_R_alpha_P_and_shared_rate,null,2);
document.getElementById("prov-cells").textContent=JSON.stringify(P.fold_and_horizon);
document.getElementById("prov-seals").textContent=JSON.stringify(P.issue_fold_and_master_seal_sha256,null,2);
document.getElementById("prov-area").textContent=JSON.stringify(P.actual_alarm_area_km2,null,2);
document.getElementById("prov-map-bindings").textContent=JSON.stringify(P.map_frame_bindings,null,2);
 renderOverall();renderCells();renderDetail();renderRegion();renderSequence();renderLatency();renderFailures();
</script>
</body>
</html>
"""


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


def _map_frame_png(
    frame: Stage2SMapFrame,
    *,
    budget_km2: int = FORMAL_ALARM_BUDGET_KM2,
    pixel_scale: int = 5,
) -> bytes:
    """Encode one derived map raster as deterministic RGB PNG bytes."""

    rows = frame.relative_intensity_rank
    alarm_fractions = frame.alarm_area_fraction(budget_km2)
    height = len(rows) * pixel_scale
    width = len(rows[0]) * pixel_scale
    background = (247, 249, 252)
    alarm_color = np.asarray((17, 24, 39), dtype=np.float64)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, :] = background
    for row_index, (row, fraction_row) in enumerate(zip(rows, alarm_fractions, strict=True)):
        for column_index, (value, alarm_fraction) in enumerate(zip(row, fraction_row, strict=True)):
            if value is None:
                continue
            y0 = row_index * pixel_scale
            x0 = column_index * pixel_scale
            rank_color = np.asarray(_hex_rgb(_palette_color(value)), dtype=np.float64)
            fraction = 0.0 if alarm_fraction is None else alarm_fraction
            mixed = np.rint((1.0 - fraction) * rank_color + fraction * alarm_color).astype(np.uint8)
            image[y0 : y0 + pixel_scale, x0 : x0 + pixel_scale, :] = mixed
    raw_rows = b"".join(
        b"\x00" + image[row_index].tobytes(order="C") for row_index in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(raw_rows, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _map_explorer_data(payload: Stage2SRenderPayload) -> dict[str, object]:
    summary = _extract_summary(payload.record)
    return {
        "mode": summary.mode_code,
        "status_banner": summary.status_banner,
        "scope_note": summary.scope_note,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "issue_time_utc": frame.issue_time_utc,
                "data_cutoff_utc": frame.data_cutoff_utc,
                "fold_index": frame.fold_index,
                "horizon_days": frame.horizon_days,
                "model_id": frame.model_id,
                "model_label": _MODEL_LABELS[frame.model_id],
                "actual_alarm_area_km2_by_budget": dict(frame.actual_alarm_area_km2_by_budget),
                "png_data_url_by_budget": {
                    str(budget): (
                        "data:image/png;base64,"
                        + base64.b64encode(_map_frame_png(frame, budget_km2=budget)).decode("ascii")
                    )
                    for budget in DISPLAY_ALARM_BUDGETS_KM2
                },
            }
            for frame in payload.map_frames
        ],
        "display_alarm_budgets_km2": list(DISPLAY_ALARM_BUDGETS_KM2),
        "formal_alarm_budget_km2": FORMAL_ALARM_BUDGET_KM2,
        "provenance": _provenance_mapping(payload),
    }


def build_map_explorer_html(payload: Stage2SRenderPayload) -> str:
    """Build a single-file offline explorer backed only by derived PNG frames."""

    data = _safe_json(_map_explorer_data(payload))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2S 历史相对强度地图浏览器</title>
<style>
:root{{--ink:#102a43;--muted:#486581;--line:#bcccdc;--paper:#f7f9fc;--panel:#fff;--warn:#c0392b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px}}h1{{margin:0 0 8px;font-size:28px}}
h2{{margin:28px 0 12px;font-size:20px}}.banner{{border:2px solid var(--warn);background:#fff0f0;
padding:14px 18px;border-radius:12px;font-weight:700}}.scope{{color:var(--muted);line-height:1.65}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}button,select{{font:inherit;color:var(--ink);
background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px 13px}}button{{cursor:pointer}}
button[aria-pressed="true"]{{background:var(--ink);color:#fff}}.map{{background:var(--panel);
border:1px solid var(--line);border-radius:12px;padding:16px;text-align:center}}
.map img{{display:block;width:100%;max-height:570px;object-fit:contain;image-rendering:pixelated}}
.detail{{margin-top:12px;line-height:1.6}}.legend{{height:18px;margin:15px auto 6px;max-width:660px;
background:linear-gradient(90deg,#313695,#4575b4,#74add1,#abd9e9,#ffffbf,#fdae61,#f46d43,#d73027,#a50026)}}
.legend-labels{{display:flex;justify-content:space-between;max-width:660px;margin:auto;color:var(--muted)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
dl{{display:grid;grid-template-columns:minmax(240px,1fr) 2fr;gap:8px 20px}}dt{{font-weight:700}}
dd{{margin:0;word-break:break-word}}pre{{white-space:pre-wrap;word-break:break-word;background:#eef3f8;
padding:12px;border-radius:8px}}@media(max-width:700px){{main{{padding:16px}}dl{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<h1>Stage 2S 历史相对强度地图</h1>
<div class="banner" id="status"></div><p class="scope" id="scope"></p>
<div class="controls">
<label>历史起报与折/窗口
<select id="issue-select" aria-label="选择历史起报与折窗口"></select></label>
<label>展示面积预算
<select id="area-select" aria-label="选择展示面积预算"></select></label>
</div>
<p class="banner" id="budget-note"></p>
<div class="controls" aria-label="选择模型">
<button type="button" data-model="S0" aria-pressed="true">S0 长期背景</button>
<button type="button" data-model="S1" aria-pressed="false">S1 最近地震</button>
<button type="button" data-model="SP" aria-pressed="false">SP 过去窗口对照</button>
</div>
<section class="map">
<img id="map-image" alt="派生历史相对强度秩栅格；深色程度表示概化像素内报警面积占比">
<div class="legend" aria-hidden="true"></div><div class="legend-labels"><span>相对较低</span>
<span>模型内相对强度顺位</span><span>相对较高</span></div>
<p class="scope">报警叠层越深，表示该概化像素内入选报警面积占比越高。</p>
<div class="detail" id="map-detail" aria-live="polite"></div>
</section>
<h2>完整溯源</h2>
<section class="card">
<dl>
<dt>S0_training_cutoff</dt><dd id="prov-s0"></dd>
<dt>R_and_RP_origin_windows</dt><dd id="prov-windows"></dd>
<dt>available_at_cutoff</dt><dd id="prov-availability"></dd>
<dt>fold_alpha_R_alpha_P_and_shared_rate</dt><dd><pre id="prov-fold"></pre></dd>
<dt>fold_and_horizon</dt><dd id="prov-cells"></dd>
<dt>issue_fold_and_master_seal_sha256</dt><dd><pre id="prov-seals"></pre></dd>
<dt>actual_alarm_area_km2</dt><dd><pre id="prov-area"></pre></dd>
<dt>map_frame_bindings</dt><dd><pre id="prov-map-bindings"></pre></dd>
</dl>
</section>
<p class="scope"><b>解释边界：</b>地图颜色只表示同一模型内部的相对强度顺位，
不是绝对发震概率；这是历史开发回溯，不是当前或前瞻预测。交互文件只嵌入派生 PNG，
不嵌入原始查询网格坐标表或裁剪几何。</p>
</main>
<script>
"use strict";
const DATA={data};
const groupKey=frame=>frame.fold_index+"|"+frame.horizon_days+"|"+frame.issue_time_utc;
const groups=[...new Set(DATA.frames.map(groupKey))];
let selectedGroup=groups[0],selectedModel="S0",selectedBudget=String(DATA.formal_alarm_budget_km2);
const issueSelect=document.getElementById("issue-select");
groups.forEach(key=>{{
 const frame=DATA.frames.find(value=>groupKey(value)===key);
 const option=document.createElement("option");option.value=key;
 option.textContent="Fold "+frame.fold_index+" · "+frame.horizon_days+" 天 · "+frame.issue_time_utc;
 issueSelect.append(option);
}});
const areaSelect=document.getElementById("area-select");
DATA.display_alarm_budgets_km2.forEach(budget=>{{
 const option=document.createElement("option");option.value=String(budget);
 option.textContent=budget.toLocaleString("zh-CN")+" km²"+
  (budget===DATA.formal_alarm_budget_km2?"（正式门）":"（仅展示性派生）");
 if(budget===DATA.formal_alarm_budget_km2)option.selected=true;areaSelect.append(option);
}});
function render(){{
 const frame=DATA.frames.find(value=>groupKey(value)===selectedGroup&&value.model_id===selectedModel);
 if(!frame)throw new Error("完整的 S0/S1/SP 派生地图组缺失");
 const image=document.getElementById("map-image");image.src=frame.png_data_url_by_budget[selectedBudget];
 image.alt=frame.model_label+"的派生历史相对强度秩栅格；深色程度表示概化像素内报警面积占比";
 const formal=Number(selectedBudget)===DATA.formal_alarm_budget_km2;
 document.getElementById("budget-note").textContent=formal?
  "正式科学门：报警面积固定 600,000 km²；图中深色比例来自精确裁剪面积。":
  "仅展示性派生；正式门固定 600,000 km²。当前面积层不重算、不替代任何正式成绩。";
 document.getElementById("map-detail").textContent=
 frame.model_label+"｜Fold "+frame.fold_index+"｜"+frame.horizon_days+" 天｜历史起报 "+
 frame.issue_time_utc+"｜数据截止 "+frame.data_cutoff_utc+"｜展示预算 "+
 Number(selectedBudget).toLocaleString("zh-CN")+" km²｜实际入选 "+
 frame.actual_alarm_area_km2_by_budget[selectedBudget].toLocaleString("zh-CN")+" km²";
}}
issueSelect.addEventListener("change",()=>{{selectedGroup=issueSelect.value;render()}});
areaSelect.addEventListener("change",()=>{{selectedBudget=areaSelect.value;render()}});
document.querySelectorAll("[data-model]").forEach(button=>button.addEventListener("click",()=>{{
 selectedModel=button.dataset.model;
 document.querySelectorAll("[data-model]").forEach(peer=>peer.setAttribute(
  "aria-pressed",peer===button?"true":"false"));
 render();
}}));
document.getElementById("status").textContent=DATA.status_banner;
document.getElementById("scope").textContent=DATA.scope_note;
const P=DATA.provenance;
document.getElementById("prov-s0").textContent=P.S0_training_cutoff;
document.getElementById("prov-windows").textContent=JSON.stringify(P.R_and_RP_origin_windows);
document.getElementById("prov-availability").textContent=P.available_at_cutoff;
document.getElementById("prov-fold").textContent=JSON.stringify(P.fold_alpha_R_alpha_P_and_shared_rate,null,2);
document.getElementById("prov-cells").textContent=JSON.stringify(P.fold_and_horizon);
document.getElementById("prov-seals").textContent=JSON.stringify(P.issue_fold_and_master_seal_sha256,null,2);
document.getElementById("prov-area").textContent=JSON.stringify(P.actual_alarm_area_km2,null,2);
document.getElementById("prov-map-bindings").textContent=JSON.stringify(P.map_frame_bindings,null,2);
render();
</script>
</body>
</html>
"""


def render_stage2s_bundle(payload: Stage2SRenderPayload) -> Stage2SRenderedBundle:
    """Render all preregistered Stage 2S result files without opening any input path."""

    artifacts = (
        Stage2SRenderedArtifact(
            PROTOCOL_ARTIFACT_NAMES[0],
            render_overall_results_svg(payload),
        ),
        Stage2SRenderedArtifact(
            PROTOCOL_ARTIFACT_NAMES[1],
            render_diagnostics_svg(payload),
        ),
        Stage2SRenderedArtifact(
            PROTOCOL_ARTIFACT_NAMES[2],
            render_relative_intensity_map_svg(payload),
        ),
        Stage2SRenderedArtifact(
            PROTOCOL_ARTIFACT_NAMES[3],
            build_backtest_explorer_html(payload).encode("utf-8"),
        ),
        Stage2SRenderedArtifact(
            PROTOCOL_ARTIFACT_NAMES[4],
            build_map_explorer_html(payload).encode("utf-8"),
        ),
        Stage2SRenderedArtifact(
            COMPANION_PNG_NAME,
            render_overall_results_png(payload),
        ),
    )
    return Stage2SRenderedBundle(artifacts)


def _record_from_mapping(value: object) -> Stage2SWholeRunRecord:
    source = dict(_mapping(value, label="record"))
    supplied_hash = source.pop("run_record_sha256", None)
    schema_version = source.pop("schema_version", None)
    record_type = source.pop("record_type", None)
    if schema_version != 1 or record_type != "stage2s_whole_run_record":
        raise Stage2SRenderingError(
            "record must be the canonical Stage 2S whole-run schema version 1"
        )
    if supplied_hash is None:
        raise Stage2SRenderingError("whole-run record is missing run_record_sha256")
    supplied_sha256 = _sha256_text(
        supplied_hash,
        label="whole-run record run_record_sha256",
    )
    required = (
        "mode",
        "identity",
        "input_receipts",
        "fold_fit_summaries",
        "issue_prediction_summaries",
        "seal_chain",
        "cell_scores",
        "bootstrap_summary",
        "bootstrap_rows",
        "regional_evidence",
        "sequence_evidence",
        "descriptive_point_estimates",
        "latency_evidence",
        "gate_evidence",
        "artifact_sha256_by_name",
    )
    if set(source) != set(required):
        missing = tuple(key for key in required if key not in source)
        extra = tuple(key for key in source if key not in required)
        raise Stage2SRenderingError(
            f"whole-run record fields changed; missing={missing!r}, extra={extra!r}"
        )
    try:
        record = Stage2SWholeRunRecord(
            mode=cast(Literal["synthetic_acceptance", "formal_development"], source["mode"]),
            identity=cast(Mapping[str, object], source["identity"]),
            input_receipts=cast(Mapping[str, object], source["input_receipts"]),
            fold_fit_summaries=cast(
                Sequence[Mapping[str, object]],
                source["fold_fit_summaries"],
            ),
            issue_prediction_summaries=cast(
                Sequence[Mapping[str, object]],
                source["issue_prediction_summaries"],
            ),
            seal_chain=cast(Mapping[str, object], source["seal_chain"]),
            cell_scores=cast(Sequence[Mapping[str, object]], source["cell_scores"]),
            bootstrap_summary=cast(Mapping[str, object], source["bootstrap_summary"]),
            bootstrap_rows=cast(Sequence[Sequence[float]], source["bootstrap_rows"]),
            regional_evidence=cast(Mapping[str, object], source["regional_evidence"]),
            sequence_evidence=cast(Mapping[str, object], source["sequence_evidence"]),
            descriptive_point_estimates=cast(
                Mapping[str, object],
                source["descriptive_point_estimates"],
            ),
            latency_evidence=cast(Sequence[Mapping[str, object]], source["latency_evidence"]),
            gate_evidence=cast(Mapping[str, object], source["gate_evidence"]),
            artifact_sha256_by_name=cast(
                Mapping[str, object],
                source["artifact_sha256_by_name"],
            ),
        )
    except (Stage2SRecordError, TypeError) as exc:
        raise Stage2SRenderingError(f"invalid whole-run record: {exc}") from exc
    if supplied_sha256 != record.run_record_sha256:
        raise Stage2SRenderingError("whole-run record SHA-256 does not match its content")
    return record


def _decode_canonical_json(value: object) -> object:
    if isinstance(value, Mapping):
        if set(value) == {"$seismoflux_type", "hex"}:
            if value.get("$seismoflux_type") != "float" or not isinstance(
                value.get("hex"),
                str,
            ):
                raise Stage2SRenderingError("invalid canonical float value")
            try:
                result = float.fromhex(cast(str, value["hex"]))
            except ValueError as exc:
                raise Stage2SRenderingError("invalid canonical float hex value") from exc
            if not math.isfinite(result):
                raise Stage2SRenderingError("canonical float must be finite")
            return result
        return {str(key): _decode_canonical_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_canonical_json(item) for item in value]
    return value


def parse_stage2s_render_payload(payload: bytes) -> Stage2SRenderPayload:
    """Parse one canonical JSON render payload for the standalone CLI."""

    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2SRenderingError("render payload must be valid UTF-8 JSON") from exc
    root = _mapping(_decode_canonical_json(raw), label="render payload")
    required_root_fields = {
        "record",
        "s0_training_cutoff_utc",
        "recent_origin_window",
        "preceding_origin_window",
        "available_at_cutoff",
        "map_frames",
    }
    if set(root) != required_root_fields:
        missing = tuple(sorted(required_root_fields - set(root)))
        extra = tuple(sorted(set(root) - required_root_fields))
        raise Stage2SRenderingError(
            f"render payload fields changed; missing={missing!r}, extra={extra!r}"
        )
    frames = []
    required_frame_fields = {
        "issue_time_utc",
        "data_cutoff_utc",
        "fold_index",
        "horizon_days",
        "model_id",
        "relative_intensity_rank",
        "study_area_km2",
        "alarm_area_fraction_by_budget_km2",
        "actual_alarm_area_km2_by_budget",
    }
    for index, raw_frame in enumerate(
        _sequence(root.get("map_frames"), label="render payload map_frames")
    ):
        item = _mapping(raw_frame, label=f"map_frames[{index}]")
        if set(item) != required_frame_fields:
            missing = tuple(sorted(required_frame_fields - set(item)))
            extra = tuple(sorted(set(item) - required_frame_fields))
            raise Stage2SRenderingError(
                f"map_frames[{index}] fields changed; missing={missing!r}, extra={extra!r}"
            )
        frames.append(
            Stage2SMapFrame(
                issue_time_utc=_text(
                    item.get("issue_time_utc"),
                    label=f"map_frames[{index}].issue_time_utc",
                ),
                data_cutoff_utc=_text(
                    item.get("data_cutoff_utc"),
                    label=f"map_frames[{index}].data_cutoff_utc",
                ),
                fold_index=int(
                    _finite_number(
                        item.get("fold_index"),
                        label=f"map_frames[{index}].fold_index",
                    )
                ),
                horizon_days=int(
                    _finite_number(
                        item.get("horizon_days"),
                        label=f"map_frames[{index}].horizon_days",
                    )
                ),
                model_id=cast(ModelId, item.get("model_id")),
                relative_intensity_rank=cast(
                    Sequence[Sequence[float | None]],
                    item.get("relative_intensity_rank"),
                ),
                study_area_km2=cast(
                    Sequence[Sequence[float | None]],
                    item.get("study_area_km2"),
                ),
                alarm_area_fraction_by_budget_km2=cast(
                    Mapping[str, Sequence[Sequence[float | None]]],
                    item.get("alarm_area_fraction_by_budget_km2"),
                ),
                actual_alarm_area_km2_by_budget=cast(
                    Mapping[str, float],
                    item.get("actual_alarm_area_km2_by_budget"),
                ),
            )
        )
    result = Stage2SRenderPayload(
        record=_record_from_mapping(root.get("record")),
        s0_training_cutoff_utc=_text(
            root.get("s0_training_cutoff_utc"),
            label="s0_training_cutoff_utc",
        ),
        recent_origin_window=_text(
            root.get("recent_origin_window"),
            label="recent_origin_window",
        ),
        preceding_origin_window=_text(
            root.get("preceding_origin_window"),
            label="preceding_origin_window",
        ),
        available_at_cutoff=_text(
            root.get("available_at_cutoff"),
            label="available_at_cutoff",
        ),
        map_frames=frames,
    )
    _record_map_frame_bindings(result)
    return result


__all__ = [
    "ALL_ARTIFACT_NAMES",
    "COMPANION_PNG_NAME",
    "DISPLAY_ALARM_BUDGETS_KM2",
    "FORMAL_ALARM_BUDGET_KM2",
    "PROTOCOL_ARTIFACT_NAMES",
    "Stage2SMapFrame",
    "Stage2SRenderPayload",
    "Stage2SRenderedArtifact",
    "Stage2SRenderedBundle",
    "Stage2SRenderingError",
    "build_backtest_explorer_html",
    "build_map_explorer_html",
    "build_rank_map_frame",
    "parse_stage2s_render_payload",
    "render_diagnostics_svg",
    "render_overall_results_png",
    "render_overall_results_svg",
    "render_relative_intensity_map_svg",
    "render_stage2s_bundle",
    "verify_stage2s_bundle_against_record",
]
