# ruff: noqa: E501, RUF001
"""Deterministic static and offline-interactive views for Stage 2P synthetic results."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Literal

import numpy as np

from seismoflux.data.common import canonical_json_bytes
from seismoflux.stage2p.evaluation import HORIZONS_DAYS
from seismoflux.stage2p.synthetic_experiment import (
    GRID_HEIGHT_KM,
    GRID_WIDTH_KM,
    ISSUE_TIME,
    MODEL_IDS,
    SyntheticScenarioResult,
    experiment_summary,
)

_DISPLAY_CELL_KM = 50.0
_PANEL_SIZE = 400.0
_PANEL_GAP = 36.0
_PANEL_LEFT = 78.0
_PANEL_TOP = 112.0
_PALETTE = (
    (247, 251, 255),
    (198, 219, 239),
    (107, 174, 214),
    (33, 113, 181),
    (8, 48, 107),
)
_MODEL_LABELS = {"P0": "P0 长期背景", "P1": "P1 加近期活动", "PP": "PP 过去窗对照"}


@dataclass(frozen=True, slots=True)
class RenderedArtifact:
    name: str
    content: bytes
    media_type: str

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name in {".", ".."}
            or Path(self.name).is_absolute()
            or Path(self.name).name != self.name
        ):
            raise ValueError("artifact name must be one safe basename")
        if not isinstance(self.content, bytes):
            raise TypeError("artifact content must be bytes")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("artifact media_type must be a non-empty string")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def _grid_shape(result: SyntheticScenarioResult) -> tuple[int, int]:
    grid = result.forecast.p0.spatial_density.grid_family.at(25.0)
    rows = int(np.max(grid.rows)) + 1
    columns = int(np.max(grid.columns)) + 1
    if rows * columns != grid.cell_count:
        raise ValueError("synthetic visualization requires one complete rectangular grid")
    if rows % 2 or columns % 2:
        raise ValueError("25 km display grid must aggregate exactly to aligned 50 km cells")
    return rows, columns


def _display_layers(
    result: SyntheticScenarioResult,
    model_id: Literal["P0", "P1", "PP"],
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = _grid_shape(result)
    model = result.forecast.at(model_id)
    mass = np.asarray(model.spatial_density.mass_25km, dtype=np.float64).reshape(rows, columns)
    display_mass = mass.reshape(rows // 2, 2, columns // 2, 2).sum(axis=(1, 3))
    intensity = display_mass / (_DISPLAY_CELL_KM**2)
    selected = np.zeros((rows, columns), dtype=np.float64)
    selected.flat[np.asarray(model.alarm.selected_indices, dtype=np.int64)] = 1.0
    alarm_fraction = selected.reshape(rows // 2, 2, columns // 2, 2).mean(axis=(1, 3))
    return intensity, alarm_fraction


def _scenario_color_limits(result: SyntheticScenarioResult) -> tuple[float, float]:
    positive_logs: list[float] = []
    for model_id in MODEL_IDS:
        intensity, _ = _display_layers(result, model_id)
        positive_logs.extend(math.log10(float(value)) for value in intensity.ravel() if value > 0.0)
    if not positive_logs:
        raise ValueError("synthetic display intensity must contain positive values")
    values = np.asarray(positive_logs, dtype=np.float64)
    lower = float(np.quantile(values, 0.02, method="linear"))
    upper = float(np.quantile(values, 0.995, method="linear"))
    if not upper > lower:
        upper = lower + 1.0
    return lower, upper


def _scaled_intensity(
    intensity: np.ndarray,
    limits: tuple[float, float],
) -> np.ndarray:
    lower, upper = limits
    positive = intensity[intensity > 0.0]
    floor = float(np.min(positive)) if positive.size else 10.0**lower
    logs = np.log10(np.maximum(intensity, floor))
    result = np.asarray(
        np.clip((logs - lower) / (upper - lower), 0.0, 1.0),
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def _rgb(value: float) -> str:
    scaled = min(max(float(value), 0.0), 1.0)
    position = scaled * (len(_PALETTE) - 1)
    lower_index = min(math.floor(position), len(_PALETTE) - 2)
    fraction = position - lower_index
    lower = _PALETTE[lower_index]
    upper = _PALETTE[lower_index + 1]
    channels = tuple(
        round(low + fraction * (high - low)) for low, high in zip(lower, upper, strict=True)
    )
    return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}"


def _target_hit_lookup(
    result: SyntheticScenarioResult,
    *,
    horizon: int,
) -> dict[str, Mapping[str, bool]]:
    return {
        row.event_id: {str(model_id): bool(hit) for model_id, hit in row.alarm_hits.items()}
        for row in result.observations
        if row.horizon_days == horizon
    }


def _target_day(result: SyntheticScenarioResult, event_id: str) -> int:
    target = next(item for item in result.targets if item.event_id == event_id)
    return int((target.origin_time - ISSUE_TIME).total_seconds() // 86_400)


def _marker_svg(
    *,
    x: float,
    y: float,
    hit: bool,
    day: int,
) -> str:
    offsets = {2: (-3.0, -2.0), 20: (3.0, -2.0), 48: (0.0, 3.0)}
    x_offset, y_offset = offsets.get(day, (0.0, 0.0))
    marker_x = x + x_offset
    marker_y = y + y_offset
    if hit:
        if day == 20:
            points = (
                f"{marker_x:.2f},{marker_y - 4:.2f} "
                f"{marker_x - 4:.2f},{marker_y + 4:.2f} "
                f"{marker_x + 4:.2f},{marker_y + 4:.2f}"
            )
            return (
                f'<polygon points="{points}" fill="#19a463" stroke="#ffffff" stroke-width="1.2"/>'
            )
        if day == 48:
            points = (
                f"{marker_x:.2f},{marker_y - 4:.2f} "
                f"{marker_x - 4:.2f},{marker_y:.2f} "
                f"{marker_x:.2f},{marker_y + 4:.2f} "
                f"{marker_x + 4:.2f},{marker_y:.2f}"
            )
            return (
                f'<polygon points="{points}" fill="#19a463" stroke="#ffffff" stroke-width="1.2"/>'
            )
        return (
            f'<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="3.8" '
            'fill="#19a463" stroke="#ffffff" stroke-width="1.2"/>'
        )
    return (
        f'<path d="M {marker_x - 4:.2f} {marker_y - 4:.2f} '
        f"L {marker_x + 4:.2f} {marker_y + 4:.2f} "
        f"M {marker_x + 4:.2f} {marker_y - 4:.2f} "
        f'L {marker_x - 4:.2f} {marker_y + 4:.2f}" '
        'stroke="#d43f3a" stroke-width="2.2" stroke-linecap="round"/>'
    )


def _panel_svg(
    result: SyntheticScenarioResult,
    *,
    model_id: Literal["P0", "P1", "PP"],
    panel_index: int,
    limits: tuple[float, float],
) -> str:
    intensity, alarm_fraction = _display_layers(result, model_id)
    scaled = _scaled_intensity(intensity, limits)
    row_count, column_count = intensity.shape
    cell_width = _PANEL_SIZE / column_count
    cell_height = _PANEL_SIZE / row_count
    left = _PANEL_LEFT + panel_index * (_PANEL_SIZE + _PANEL_GAP)
    top = _PANEL_TOP
    pieces = [
        f'<g aria-label="{escape(_MODEL_LABELS[model_id])}">',
        (
            f'<text x="{left:.1f}" y="{top - 32:.1f}" class="panel-title">'
            f"{escape(_MODEL_LABELS[model_id])}</text>"
        ),
        (
            f'<text x="{left:.1f}" y="{top - 12:.1f}" class="panel-meta">'
            f"报警面积 {result.forecast.at(model_id).alarm.actual_area_km2:,.0f} km² · "
            f"90天命中 {result.evaluation.horizons[90].model_recall[model_id].event_hit_count}/"
            f"{result.evaluation.horizons[90].model_recall[model_id].event_count}</text>"
        ),
    ]
    for row in range(row_count):
        display_row = row_count - 1 - row
        for column in range(column_count):
            x_value = left + column * cell_width
            y_value = top + display_row * cell_height
            pieces.append(
                f'<rect x="{x_value:.2f}" y="{y_value:.2f}" '
                f'width="{cell_width + 0.05:.2f}" height="{cell_height + 0.05:.2f}" '
                f'fill="{_rgb(float(scaled[row, column]))}"/>'
            )
            fraction = float(alarm_fraction[row, column])
            if fraction == 0.0:
                pieces.append(
                    f'<rect x="{x_value:.2f}" y="{y_value:.2f}" '
                    f'width="{cell_width + 0.05:.2f}" height="{cell_height + 0.05:.2f}" '
                    'fill="#ffffff" fill-opacity="0.68"/>'
                )
            elif fraction < 1.0:
                pieces.append(
                    f'<rect x="{x_value:.2f}" y="{y_value:.2f}" '
                    f'width="{cell_width + 0.05:.2f}" height="{cell_height + 0.05:.2f}" '
                    f'fill="#ffffff" fill-opacity="{0.68 * (1.0 - fraction):.3f}"/>'
                )
    pieces.append(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{_PANEL_SIZE:.2f}" '
        f'height="{_PANEL_SIZE:.2f}" fill="none" stroke="#28323c" stroke-width="1.2"/>'
    )
    hit_lookup = _target_hit_lookup(result, horizon=90)
    for target in result.targets:
        x_value = left + target.x_km / GRID_WIDTH_KM * _PANEL_SIZE
        y_value = top + (1.0 - target.y_km / GRID_HEIGHT_KM) * _PANEL_SIZE
        pieces.append(
            _marker_svg(
                x=x_value,
                y=y_value,
                hit=bool(hit_lookup[target.event_id][model_id]),
                day=_target_day(result, target.event_id),
            )
        )
    pieces.append("</g>")
    return "".join(pieces)


def _bar(
    *,
    x: float,
    y: float,
    width: float,
    value: float,
    minimum: float,
    maximum: float,
    color: str,
    label: str,
) -> str:
    span = maximum - minimum
    zero_x = x + width * (0.0 - minimum) / span
    value_x = x + width * (value - minimum) / span
    left = min(zero_x, value_x)
    bar_width = max(abs(value_x - zero_x), 1.0)
    return (
        f'<text x="{x - 8:.1f}" y="{y + 13:.1f}" text-anchor="end" class="metric-label">'
        f"{escape(label)}</text>"
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="18" '
        'fill="#edf1f4"/>'
        f'<line x1="{zero_x:.1f}" y1="{y - 2:.1f}" x2="{zero_x:.1f}" '
        f'y2="{y + 20:.1f}" stroke="#4d5966" stroke-width="1"/>'
        f'<rect x="{left:.1f}" y="{y + 2:.1f}" width="{bar_width:.1f}" height="14" '
        f'fill="{color}"/>'
        f'<text x="{x + width + 8:.1f}" y="{y + 13:.1f}" class="metric-value">'
        f"{value:+.3f}</text>"
    )


def render_scenario_svg(result: SyntheticScenarioResult) -> bytes:
    """Render one three-model known-answer map with comparison metrics."""

    width = 1_450
    height = 720
    limits = _scenario_color_limits(result)
    p0 = result.evaluation.comparisons["P1_minus_P0"]
    pp = result.evaluation.comparisons["P1_minus_PP"]
    pieces = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title-{result.scenario.scenario_id} '
            f'desc-{result.scenario.scenario_id}">'
        ),
        (
            "<style>"
            "text{font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;fill:#16212b}"
            ".title{font-size:24px;font-weight:600}.subtitle{font-size:14px;fill:#4d5966}"
            ".warning{font-size:14px;font-weight:600;fill:#a12622}"
            ".panel-title{font-size:17px;font-weight:600}.panel-meta{font-size:12px;fill:#4d5966}"
            ".metric-label{font-size:12px}.metric-value{font-size:12px;font-family:monospace}"
            ".legend{font-size:12px;fill:#35414d}"
            "</style>"
        ),
        (
            f'<title id="title-{result.scenario.scenario_id}">'
            f"Stage 2P 合成演练：{escape(result.scenario.label_zh)}</title>"
        ),
        (
            f'<desc id="desc-{result.scenario.scenario_id}">'
            "P0、P1、PP 相对强度、同面积报警区、合成目标命中漏报和模型对比。"
            "</desc>"
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="72" y="38" class="title">{escape(result.scenario.label_zh)}</text>'
            '<text x="72" y="64" class="warning">纯合成演练，不是真实预测证据</text>'
            f'<text x="72" y="88" class="subtitle">{escape(result.scenario.known_answer_zh)}</text>'
        ),
    ]
    for panel_index, model_id in enumerate(MODEL_IDS):
        pieces.append(
            _panel_svg(
                result,
                model_id=model_id,
                panel_index=panel_index,
                limits=limits,
            )
        )
    pieces.extend(
        [
            '<text x="78" y="548" class="legend">鲜明区域＝600,000 km²报警前缀；淡化区域＝未报警</text>',
            '<text x="78" y="570" class="legend">●/▲/◆ 绿色＝2/20/48天目标命中；红色 ×＝漏报</text>',
            '<text x="580" y="548" class="legend">颜色＝相对空间强度（对数色阶），不是绝对发震概率</text>',
            (
                f'<text x="580" y="570" class="legend">样本：'
                f"{result.evaluation.unique_event_count} 个唯一目标，"
                f"{result.evaluation.independent_cluster_count} 个震群块；"
                "合成已知答案检查="
                f"{'通过' if result.synthetic_known_answer_status == 'passed' else '失败'}</text>"
            ),
            '<text x="78" y="614" class="panel-title">模型对比（宏平均）</text>',
            _bar(
                x=320,
                y=596,
                width=300,
                value=p0.macro_recall_gain_percentage_points,
                minimum=-100.0,
                maximum=100.0,
                color="#2f80c1",
                label="P1-P0 召回百分点",
            ),
            _bar(
                x=320,
                y=626,
                width=300,
                value=pp.macro_recall_gain_percentage_points,
                minimum=-100.0,
                maximum=100.0,
                color="#8a5fb5",
                label="P1-PP 召回百分点",
            ),
            _bar(
                x=910,
                y=596,
                width=300,
                value=p0.macro_information_gain_nats_per_event,
                minimum=-2.5,
                maximum=2.5,
                color="#2f80c1",
                label="P1-P0 信息增益",
            ),
            _bar(
                x=910,
                y=626,
                width=300,
                value=pp.macro_information_gain_nats_per_event,
                minimum=-2.5,
                maximum=2.5,
                color="#8a5fb5",
                label="P1-PP 信息增益",
            ),
            (
                '<text x="78" y="690" class="subtitle">'
                "报警面积相同；未来目标只用于演练评价，没有参与预测图生成。"
                "</text>"
            ),
            "</svg>",
        ]
    )
    return "".join(pieces).encode("utf-8")


def render_comparison_svg(results: Sequence[SyntheticScenarioResult]) -> bytes:
    """Render one compact across-scenario comparison chart."""

    values = tuple(results)
    if len(values) != 3:
        raise ValueError("comparison figure requires exactly three scenarios")
    width, height = 1_200, 600
    pieces = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="summary-title">'
        ),
        (
            "<style>"
            "text{font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;fill:#16212b}"
            ".title{font-size:24px;font-weight:600}.subtitle{font-size:14px;fill:#4d5966}"
            ".label{font-size:14px;font-weight:600}.small{font-size:12px;fill:#4d5966}"
            "</style>"
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<title id="summary-title">Stage 2P 三种合成情景对比</title>',
        '<text x="58" y="42" class="title">三种已知答案能否被正确区分</text>',
        '<text x="58" y="68" class="subtitle">纯合成演练，不是真实预测证据</text>',
    ]
    model_colors = {"P0": "#7b8794", "P1": "#2f80c1", "PP": "#8a5fb5"}
    for row_index, result in enumerate(values):
        y_base = 128 + row_index * 150
        pieces.append(
            f'<text x="58" y="{y_base:.1f}" class="label">{escape(result.scenario.label_zh)}</text>'
        )
        pieces.append(
            f'<text x="58" y="{y_base + 23:.1f}" class="small">'
            "合成已知答案检查="
            f"{'通过' if result.synthetic_known_answer_status == 'passed' else '失败'}</text>"
        )
        for model_index, model_id in enumerate(MODEL_IDS):
            recall = result.evaluation.macro_model_recall[model_id].independent_cluster_recall
            x_value = 320 + model_index * 205
            pieces.append(
                f'<text x="{x_value:.1f}" y="{y_base - 18:.1f}" class="small">'
                f"{model_id} 震群召回 {100.0 * recall:.0f}%</text>"
            )
            pieces.append(
                f'<rect x="{x_value:.1f}" y="{y_base:.1f}" width="170" height="18" fill="#edf1f4"/>'
            )
            pieces.append(
                f'<rect x="{x_value:.1f}" y="{y_base:.1f}" '
                f'width="{170.0 * recall:.1f}" height="18" fill="{model_colors[model_id]}"/>'
            )
        p0 = result.evaluation.comparisons["P1_minus_P0"]
        pp = result.evaluation.comparisons["P1_minus_PP"]
        pieces.append(
            f'<text x="320" y="{y_base + 54:.1f}" class="small">'
            f"P1-P0: 召回 {p0.macro_recall_gain_percentage_points:+.1f} pp，"
            f"IG {p0.macro_information_gain_nats_per_event:+.3f} nats/event</text>"
        )
        pieces.append(
            f'<text x="730" y="{y_base + 54:.1f}" class="small">'
            f"P1-PP: 召回 {pp.macro_recall_gain_percentage_points:+.1f} pp，"
            f"IG {pp.macro_information_gain_nats_per_event:+.3f} nats/event</text>"
        )
    pieces.extend(
        [
            '<text x="58" y="575" class="subtitle">'
            "结论：实现会在近期信号有效时奖励 P1，无信号时给出零增益，误导时给出负结果。"
            "</text>",
            "</svg>",
        ]
    )
    return "".join(pieces).encode("utf-8")


def _interactive_payload(
    results: Sequence[SyntheticScenarioResult],
) -> dict[str, object]:
    scenarios: list[dict[str, object]] = []
    for result in results:
        limits = _scenario_color_limits(result)
        models: dict[str, object] = {}
        for model_id in MODEL_IDS:
            intensity, alarm = _display_layers(result, model_id)
            scaled = _scaled_intensity(intensity, limits)
            models[model_id] = {
                "intensity": [round(float(value), 6) for value in scaled.ravel()],
                "alarm_fraction": [round(float(value), 2) for value in alarm.ravel()],
                "actual_area_km2": result.forecast.at(model_id).alarm.actual_area_km2,
                "recall_by_horizon": {
                    str(horizon): {
                        "event_recall": (
                            result.evaluation.horizons[horizon]
                            .model_recall[model_id]
                            .strict_event_recall
                        ),
                        "cluster_recall": (
                            result.evaluation.horizons[horizon]
                            .model_recall[model_id]
                            .independent_cluster_recall
                        ),
                        "region_recall": (
                            result.evaluation.horizons[horizon]
                            .model_recall[model_id]
                            .independent_region_recall
                        ),
                    }
                    for horizon in HORIZONS_DAYS
                },
            }
        target_rows: list[dict[str, object]] = []
        hit_lookup = _target_hit_lookup(result, horizon=90)
        for target in result.targets:
            target_rows.append(
                {
                    "event_id": target.event_id,
                    "x": target.x_km / _DISPLAY_CELL_KM,
                    "y": target.y_km / _DISPLAY_CELL_KM,
                    "day": _target_day(result, target.event_id),
                    "hits": dict(hit_lookup[target.event_id]),
                }
            )
        scenarios.append(
            {
                "scenario_id": result.scenario.scenario_id,
                "label_zh": result.scenario.label_zh,
                "known_answer_zh": result.scenario.known_answer_zh,
                "grid_rows": int(GRID_HEIGHT_KM / _DISPLAY_CELL_KM),
                "grid_columns": int(GRID_WIDTH_KM / _DISPLAY_CELL_KM),
                "models": models,
                "targets": target_rows,
                "comparisons": {
                    comparison_id: {
                        "recall_gain_pp": comparison.macro_recall_gain_percentage_points,
                        "information_gain": (comparison.macro_information_gain_nats_per_event),
                        "recall_ci": [
                            comparison.recall_interval.lower,
                            comparison.recall_interval.upper,
                        ],
                        "information_ci": [
                            comparison.information_gain_interval.lower,
                            comparison.information_gain_interval.upper,
                        ],
                    }
                    for comparison_id, comparison in result.evaluation.comparisons.items()
                },
                "synthetic_known_answer_status": (result.synthetic_known_answer_status),
            }
        )
    return {
        "experiment": dict(experiment_summary(tuple(results))),
        "scenarios": scenarios,
    }


def build_offline_explorer_html(results: Sequence[SyntheticScenarioResult]) -> str:
    """Build a self-contained no-fetch HTML explorer."""

    payload = json.dumps(
        _interactive_payload(results),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeismoFlux Stage 2P 合成科学演练</title>
<style>
:root{{color-scheme:light;font-family:"Noto Sans SC","Microsoft YaHei",sans-serif}}
body{{margin:0;background:#f5f7f9;color:#16212b}}
main{{max-width:1100px;margin:auto;padding:20px}}
h1{{font-size:24px;margin:0 0 4px}} .warning{{color:#a12622;font-weight:700}}
.controls{{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0}}
label{{display:grid;gap:4px;font-size:13px}} select{{font:inherit;padding:7px 10px}}
.layout{{display:grid;grid-template-columns:minmax(300px,680px) minmax(240px,1fr);gap:18px}}
canvas{{width:100%;height:auto;background:#fff;border:1px solid #c8d0d8}}
.metrics{{background:#fff;border:1px solid #d6dde3;padding:14px}}
.metrics dl{{display:grid;grid-template-columns:1fr auto;gap:8px 12px;margin:8px 0}}
.metrics dt{{color:#4d5966}} .metrics dd{{margin:0;font-variant-numeric:tabular-nums}}
.legend{{font-size:13px;color:#4d5966;margin-top:10px}}
@media(max-width:760px){{.layout{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<h1>Stage 2P：P0 / P1 / PP 直观对比</h1>
<div class="warning">纯合成演练，不是真实预测证据</div>
<div class="controls">
<label>情景<select id="scenario"></select></label>
<label>模型<select id="model"><option>P0</option><option>P1</option><option>PP</option></select></label>
<label>未来窗口<select id="horizon"><option value="7">7 天</option><option value="30">30 天</option><option value="90" selected>90 天</option></select></label>
</div>
<p id="known-answer"></p>
<div class="layout">
<div><canvas id="map" width="640" height="640" role="img" aria-label="合成相对强度、报警区和未来目标地图"></canvas>
<div class="legend">鲜明色块＝报警区，淡色＝未报警；绿色目标＝命中，红色 ×＝漏报；颜色只表示相对强度。</div></div>
<section class="metrics" aria-live="polite"><h2 id="metric-title"></h2><dl id="metric-list"></dl></section>
</div>
<p class="legend">所有数据都写在本页内；离线打开不会请求网络。目标按 2、20、48 天逐步出现。相对强度不是绝对发震概率。</p>
</main>
<script>
"use strict";
const payload={payload};
const scenarioSelect=document.getElementById("scenario");
const modelSelect=document.getElementById("model");
const horizonSelect=document.getElementById("horizon");
const canvas=document.getElementById("map");
const context=canvas.getContext("2d");
const known=document.getElementById("known-answer");
const metricTitle=document.getElementById("metric-title");
const metricList=document.getElementById("metric-list");
for(const item of payload.scenarios){{
  const option=document.createElement("option"); option.value=item.scenario_id;
  option.textContent=item.label_zh; scenarioSelect.appendChild(option);
}}
const palette=[[247,251,255],[198,219,239],[107,174,214],[33,113,181],[8,48,107]];
function color(value){{
  const position=Math.max(0,Math.min(1,value))*(palette.length-1);
  const low=Math.min(Math.floor(position),palette.length-2), fraction=position-low;
  const rgb=palette[low].map((v,i)=>Math.round(v+fraction*(palette[low+1][i]-v)));
  return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
}}
function scenario(){{return payload.scenarios.find(item=>item.scenario_id===scenarioSelect.value);}}
function addMetric(label,value){{
  const dt=document.createElement("dt"); dt.textContent=label;
  const dd=document.createElement("dd"); dd.textContent=value;
  metricList.append(dt,dd);
}}
function draw(){{
  const item=scenario(), model=modelSelect.value, horizon=Number(horizonSelect.value);
  const layer=item.models[model], rows=item.grid_rows, columns=item.grid_columns;
  const cell=canvas.width/columns; context.clearRect(0,0,canvas.width,canvas.height);
  for(let row=0;row<rows;row++) for(let column=0;column<columns;column++){{
    const index=row*columns+column, y=(rows-1-row)*cell;
    context.fillStyle=color(layer.intensity[index]); context.fillRect(column*cell,y,cell+0.3,cell+0.3);
    const alarm=layer.alarm_fraction[index];
    if(alarm===0){{context.fillStyle="rgba(255,255,255,.68)";context.fillRect(column*cell,y,cell+0.3,cell+0.3);}}
    else if(alarm<1){{context.fillStyle=`rgba(255,255,255,${{.68*(1-alarm)}})`;context.fillRect(column*cell,y,cell+0.3,cell+0.3);}}
  }}
  for(const target of item.targets.filter(target=>target.day<=horizon)){{
    const x=target.x*cell, y=canvas.height-target.y*cell, hit=target.hits[model];
    context.lineWidth=3; context.strokeStyle=hit?"#087f4f":"#c4312d"; context.fillStyle="#19a463";
    context.beginPath();
    if(hit){{context.arc(x,y,5,0,Math.PI*2);context.fill();context.stroke();}}
    else{{context.moveTo(x-6,y-6);context.lineTo(x+6,y+6);context.moveTo(x+6,y-6);context.lineTo(x-6,y+6);context.stroke();}}
  }}
  known.textContent=item.known_answer_zh;
  metricTitle.textContent=`${{item.label_zh}} · ${{model}} · ${{horizon}}天`;
  metricList.replaceChildren();
  const recall=layer.recall_by_horizon[String(horizon)];
  addMetric("实际报警面积",`${{layer.actual_area_km2.toLocaleString()}} km²`);
  addMetric("事件严格召回",`${{(100*recall.event_recall).toFixed(1)}}%`);
  addMetric("独立震群召回",`${{(100*recall.cluster_recall).toFixed(1)}}%`);
  addMetric("独立区域召回",`${{(100*recall.region_recall).toFixed(1)}}%`);
  for(const [name,value] of Object.entries(item.comparisons)){{
    addMetric(`${{name}} 召回差`,`${{value.recall_gain_pp>=0?"+":""}}${{value.recall_gain_pp.toFixed(1)}} pp`);
    addMetric(`${{name}} 信息增益`,`${{value.information_gain>=0?"+":""}}${{value.information_gain.toFixed(3)}} nats/event`);
  }}
  addMetric(
    "合成已知答案检查",
    item.synthetic_known_answer_status==="passed"?"通过":"失败"
  );
}}
for(const element of [scenarioSelect,modelSelect,horizonSelect]) element.addEventListener("change",draw);
draw();
</script>
</body>
</html>
"""


def render_artifacts(
    results: Sequence[SyntheticScenarioResult],
) -> tuple[RenderedArtifact, ...]:
    """Return deterministic public-safe synthetic result artifacts."""

    values = tuple(results)
    if len(values) != 3:
        raise ValueError("the synthetic MVP requires exactly three scenarios")
    artifacts: list[RenderedArtifact] = []
    for result in values:
        artifacts.append(
            RenderedArtifact(
                name=f"{result.scenario.scenario_id}.svg",
                content=render_scenario_svg(result),
                media_type="image/svg+xml",
            )
        )
    artifacts.extend(
        (
            RenderedArtifact(
                name="scenario_comparison.svg",
                content=render_comparison_svg(values),
                media_type="image/svg+xml",
            ),
            RenderedArtifact(
                name="stage2p_science_mvp_explorer.html",
                content=build_offline_explorer_html(values).encode("utf-8"),
                media_type="text/html",
            ),
            RenderedArtifact(
                name="metrics.json",
                content=canonical_json_bytes(dict(experiment_summary(values))),
                media_type="application/json",
            ),
        )
    )
    return tuple(artifacts)


def write_artifacts(
    artifacts: Sequence[RenderedArtifact],
    output_directory: Path,
    *,
    check: bool = False,
) -> Mapping[str, str]:
    """Create or verify one deterministic artifact bundle."""

    values = tuple(artifacts)
    names = tuple(artifact.name for artifact in values)
    if len(set(names)) != len(names):
        raise ValueError("artifact names must be unique")
    output = output_directory.resolve()
    if check:
        if not output.is_dir():
            raise FileNotFoundError(f"artifact directory does not exist: {output}")
        actual_names = {entry.name for entry in output.iterdir()}
        if actual_names != set(names):
            raise ValueError("artifact directory contents differ from expected bundle")
    else:
        output.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for artifact in values:
        destination = output / artifact.name
        if check:
            if destination.read_bytes() != artifact.content:
                raise ValueError(f"rendered artifact differs: {artifact.name}")
        else:
            destination.write_bytes(artifact.content)
        hashes[artifact.name] = artifact.sha256
    return hashes


__all__ = [
    "RenderedArtifact",
    "build_offline_explorer_html",
    "render_artifacts",
    "render_comparison_svg",
    "render_scenario_svg",
    "write_artifacts",
]
