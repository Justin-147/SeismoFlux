# ruff: noqa: RUF001
"""Geometry-free, deterministic overview for the stage-2R-1 public result."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Literal


@dataclass(frozen=True, slots=True)
class LocalSupportVisualSnapshot:
    """One snapshot's support coverage and primary paired information gain."""

    snapshot_id: str
    supported_area_fraction: float
    spatial_information_gain: float | None
    etas_information_gain: float | None

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("visual snapshot_id must not be empty")
        if not math.isfinite(self.supported_area_fraction) or not (
            0.0 <= self.supported_area_fraction <= 1.0
        ):
            raise ValueError("supported_area_fraction must be finite and in [0, 1]")
        for value in (self.spatial_information_gain, self.etas_information_gain):
            if value is not None and not math.isfinite(value):
                raise ValueError("visual information gain must be finite when present")


@dataclass(frozen=True, slots=True)
class LocalSupportVisualHorizon:
    """Zero-publication-delay secondary diagnostic for one horizon."""

    horizon_days: int
    spatial_information_gain: float | None
    etas_information_gain: float | None

    def __post_init__(self) -> None:
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("visual horizon must be one of the frozen horizons")
        for value in (self.spatial_information_gain, self.etas_information_gain):
            if value is not None and not math.isfinite(value):
                raise ValueError("visual horizon information gain must be finite when present")


@dataclass(frozen=True, slots=True)
class LocalSupportVisualBootstrap:
    """Validation point estimate and preregistered 95% bootstrap interval."""

    model_id: str
    point_estimate: float | None
    lower: float | None
    upper: float | None

    def __post_init__(self) -> None:
        if self.model_id not in {"spatial_poisson", "etas"}:
            raise ValueError("visual bootstrap model_id is unknown")
        values = (self.point_estimate, self.lower, self.upper)
        if any(value is None for value in values):
            if any(value is not None for value in values):
                raise ValueError("visual bootstrap interval must be complete or absent")
            return
        point, lower, upper = (float(value) for value in values if value is not None)
        if not all(math.isfinite(value) for value in (point, lower, upper)):
            raise ValueError("visual bootstrap interval must be finite")
        if lower > upper:
            raise ValueError("visual bootstrap interval must be ordered")


@dataclass(frozen=True, slots=True)
class LocalSupportVisualSensitivity:
    """ETAS information gain with and without eligible unsupported parents."""

    snapshot_id: str
    primary_information_gain: float | None
    excluded_information_gain: float | None

    def __post_init__(self) -> None:
        if self.snapshot_id not in {"fold_1", "fold_3"}:
            raise ValueError("visual ETAS sensitivity is frozen to fold_1/fold_3")
        for value in (self.primary_information_gain, self.excluded_information_gain):
            if value is not None and not math.isfinite(value):
                raise ValueError("visual ETAS sensitivity must be finite when present")

    @property
    def difference(self) -> float | None:
        if self.primary_information_gain is None or self.excluded_information_gain is None:
            return None
        return self.excluded_information_gain - self.primary_information_gain


_SNAPSHOT_LABELS = {
    "fold_1": "折1",
    "fold_2": "折2",
    "fold_3": "折3",
    "fold_4": "折4",
    "final_validation": "最终验证",
}


def _text(x: float, y: float, value: str, *, size: int = 18, fill: str = "#243447") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'fill="{fill}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">'
        f"{escape(value)}</text>"
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    radius: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.0, width):.1f}" '
        f'height="{max(0.0, height):.1f}" rx="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def _ig_extent(
    snapshots: tuple[LocalSupportVisualSnapshot, ...],
    horizons: tuple[LocalSupportVisualHorizon, ...],
    bootstrap: tuple[LocalSupportVisualBootstrap, ...],
) -> float:
    values: list[float] = [0.0]
    for snapshot in snapshots:
        values.extend(
            value
            for value in (
                snapshot.spatial_information_gain,
                snapshot.etas_information_gain,
            )
            if value is not None
        )
    for horizon_item in horizons:
        values.extend(
            value
            for value in (
                horizon_item.spatial_information_gain,
                horizon_item.etas_information_gain,
            )
            if value is not None
        )
    for bootstrap_item in bootstrap:
        values.extend(
            value
            for value in (
                bootstrap_item.point_estimate,
                bootstrap_item.lower,
                bootstrap_item.upper,
            )
            if value is not None
        )
    return max(0.05, max(abs(value) for value in values) * 1.15)


def render_local_support_results_svg(
    *,
    snapshots: tuple[LocalSupportVisualSnapshot, ...],
    horizons: tuple[LocalSupportVisualHorizon, ...],
    bootstrap: tuple[LocalSupportVisualBootstrap, ...],
    sensitivity: tuple[LocalSupportVisualSensitivity, ...],
    g1_ls_status: Literal["passed", "failed", "not_evaluable"],
    selected_model_variant_id: str,
) -> bytes:
    """Render a public overview without accepting geometry, coordinates, or event locations."""

    if tuple(item.snapshot_id for item in snapshots) != tuple(_SNAPSHOT_LABELS):
        raise ValueError("visual snapshots must contain the frozen five-snapshot order")
    if tuple(item.horizon_days for item in horizons) != (7, 30, 90, 180, 365):
        raise ValueError("visual horizons must contain the frozen horizon order")
    if tuple(item.model_id for item in bootstrap) != ("spatial_poisson", "etas"):
        raise ValueError("visual bootstrap must follow the frozen model order")
    if tuple(item.snapshot_id for item in sensitivity) != ("fold_1", "fold_3"):
        raise ValueError("visual ETAS sensitivity must contain fold_1/fold_3")
    if g1_ls_status not in {"passed", "failed", "not_evaluable"}:
        raise ValueError("visual G1-LS status is unknown")
    if not selected_model_variant_id:
        raise ValueError("selected model variant must not be empty")

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1060" '
        'viewBox="0 0 1440 1060" role="img" aria-labelledby="title desc">',
        '<title id="title">SeismoFlux G1-LS 背景模型结果总览</title>',
        (
            '<desc id="desc">数据支持率、模型流程、配对信息增益、置信区间、'
            "时窗诊断与父历史敏感性。</desc>"
        ),
        _rect(0, 0, 1440, 1060, fill="#f5f7fb"),
        _text(55, 55, "SeismoFlux · G1-LS 背景模型结果总览", size=30, fill="#102a43"),
        _text(
            55,
            86,
            "所有比较使用同一局部支持域、同一目标事件和同一补偿积分域；数值是条件强度信息增益，不是绝对发震概率。",
            size=15,
            fill="#52667a",
        ),
    ]

    status_fill = {
        "passed": "#13795b",
        "failed": "#b42318",
        "not_evaluable": "#7b5e00",
    }[g1_ls_status]
    status_text = {
        "passed": "G1-LS 通过",
        "failed": "G1-LS 未通过",
        "not_evaluable": "G1-LS 未评估",
    }[g1_ls_status]
    parts.extend(
        (
            _rect(1130, 30, 250, 58, fill=status_fill, radius=10),
            _text(1170, 68, status_text, size=22, fill="#ffffff"),
        )
    )

    panel_fill = "#ffffff"
    panel_stroke = "#d9e2ec"
    parts.extend(
        (
            _rect(40, 115, 1360, 150, fill=panel_fill, stroke=panel_stroke, radius=12),
            _text(65, 148, "用的数据与方法", size=21, fill="#102a43"),
        )
    )
    stages = (
        ("因果目录", "每个拟合截止日前已可用的地震"),
        ("固定局部支持", "500 km格；raw Mc>4只屏蔽本格"),
        ("三类背景", "均匀 Poisson → KDE → ETAS"),
        ("同域配对评分", "连续时间似然 + 补偿积分"),
        ("门控与选择", "4 折稳定性 + 最终验证 + one-SE"),
    )
    stage_x = (65.0, 330.0, 595.0, 860.0, 1125.0)
    for index, ((title, subtitle), x) in enumerate(zip(stages, stage_x, strict=True)):
        parts.append(_rect(x, 165, 210, 72, fill="#eaf2ff", stroke="#9fbff0", radius=8))
        parts.append(_text(x + 14, 191, title, size=17, fill="#174ea6"))
        parts.append(_text(x + 14, 219, subtitle, size=12, fill="#52667a"))
        if index < len(stages) - 1:
            parts.append(_line(x + 215, 201, x + 252, 201, stroke="#7b93ad", width=2.0))
            parts.append(
                f'<path d="M {x + 246:.1f} 195 L {x + 254:.1f} 201 '
                f'L {x + 246:.1f} 207" fill="none" stroke="#7b93ad" '
                'stroke-width="2"/>'
            )

    parts.extend(
        (
            _rect(40, 285, 650, 285, fill=panel_fill, stroke=panel_stroke, radius=12),
            _text(65, 322, "A. 每个快照保留的研究区面积", size=21, fill="#102a43"),
            _text(
                65,
                346,
                "预登记门槛为 95%；raw Mc>4 只改变本格支持状态，不提高其他格的排除状态。",
                size=13,
                fill="#52667a",
            ),
        )
    )
    for index, snapshot in enumerate(snapshots):
        y = 378.0 + index * 36.0
        label = _SNAPSHOT_LABELS[snapshot.snapshot_id]
        parts.append(_text(65, y + 17, label, size=14))
        parts.append(_rect(160, y, 450, 20, fill="#edf1f5", radius=4))
        parts.append(
            _rect(
                160,
                y,
                450 * snapshot.supported_area_fraction,
                20,
                fill="#277da1" if snapshot.supported_area_fraction >= 0.95 else "#b42318",
                radius=4,
            )
        )
        parts.append(
            _text(
                618,
                y + 16,
                f"{snapshot.supported_area_fraction * 100:.2f}%",
                size=13,
                fill="#243447",
            )
        )

    parts.extend(
        (
            _rect(710, 285, 690, 285, fill=panel_fill, stroke=panel_stroke, radius=12),
            _text(735, 322, "B. 主评估：相对均匀基线的信息增益", size=21, fill="#102a43"),
            _text(
                735,
                346,
                "正值更好；同一行的模型在完全相同的目标与曝光上比较。",
                size=13,
                fill="#52667a",
            ),
        )
    )
    extent = _ig_extent(snapshots, horizons, bootstrap)
    plot_left, plot_right, zero_x = 850.0, 1360.0, 1105.0
    scale = (plot_right - plot_left) / (2.0 * extent)
    parts.append(_line(zero_x, 365, zero_x, 545, stroke="#708090", width=1.2))
    parts.append(_text(plot_left, 365, f"−{extent:.3g}", size=11, fill="#6b7c8f"))
    parts.append(_text(zero_x - 4, 365, "0", size=11, fill="#6b7c8f"))
    parts.append(_text(plot_right - 40, 365, f"+{extent:.3g}", size=11, fill="#6b7c8f"))
    for index, snapshot in enumerate(snapshots):
        y = 390.0 + index * 34.0
        parts.append(_text(735, y + 12, _SNAPSHOT_LABELS[snapshot.snapshot_id], size=13))
        for offset, value, color in (
            (0.0, snapshot.spatial_information_gain, "#f8961e"),
            (13.0, snapshot.etas_information_gain, "#577590"),
        ):
            if value is None:
                parts.append(_text(1320, y + offset + 9, "未评分", size=10, fill="#8b98a5"))
                continue
            width = abs(value) * scale
            x = zero_x if value >= 0.0 else zero_x - width
            parts.append(_rect(x, y + offset, width, 9, fill=color, radius=2))
    parts.append(_rect(1180, 535, 14, 9, fill="#f8961e", radius=2))
    parts.append(_text(1200, 544, "空间 Poisson/KDE", size=11))
    parts.append(_rect(1300, 535, 14, 9, fill="#577590", radius=2))
    parts.append(_text(1320, 544, "ETAS", size=11))

    parts.extend(
        (
            _rect(40, 590, 650, 245, fill=panel_fill, stroke=panel_stroke, radius=12),
            _text(65, 627, "C. 最终验证的 95% bootstrap 区间", size=21, fill="#102a43"),
            _text(
                65,
                651,
                "区间跨 0 表示证据不确定；门控仍按冻结规则判定。",
                size=13,
                fill="#52667a",
            ),
        )
    )
    b_left, b_right, b_zero = 190.0, 645.0, 417.5
    b_scale = (b_right - b_left) / (2.0 * extent)
    parts.append(_line(b_zero, 675, b_zero, 797, stroke="#708090", width=1.2))
    for index, bootstrap_item in enumerate(bootstrap):
        y = 704.0 + index * 64.0
        label = "空间 Poisson/KDE" if bootstrap_item.model_id == "spatial_poisson" else "ETAS"
        parts.append(_text(65, y + 6, label, size=14))
        if (
            bootstrap_item.point_estimate is None
            or bootstrap_item.lower is None
            or bootstrap_item.upper is None
        ):
            parts.append(
                _text(
                    190,
                    y + 6,
                    "未计算（模型未产生合格最终分数）",
                    size=13,
                    fill="#8b98a5",
                )
            )
            continue
        x1 = b_zero + bootstrap_item.lower * b_scale
        x2 = b_zero + bootstrap_item.upper * b_scale
        xp = b_zero + bootstrap_item.point_estimate * b_scale
        parts.append(_line(x1, y, x2, y, stroke="#334e68", width=4.0))
        parts.append(f'<circle cx="{xp:.1f}" cy="{y:.1f}" r="7" fill="#d64545"/>')
        interval_label = (
            f"{bootstrap_item.point_estimate:+.4f} "
            f"[{bootstrap_item.lower:+.4f}, {bootstrap_item.upper:+.4f}]"
        )
        parts.append(_text(505, y + 6, interval_label, size=12))

    parts.extend(
        (
            _rect(710, 590, 690, 245, fill=panel_fill, stroke=panel_stroke, radius=12),
            _text(735, 627, "D. 次级时窗诊断（报告延迟 0 天）", size=21, fill="#102a43"),
            _text(
                735,
                651,
                "仅作积分诊断，不参与 G1-LS；正值代表优于均匀基线。",
                size=13,
                fill="#52667a",
            ),
        )
    )
    h_left, h_width = 900.0, 425.0
    for index, horizon_item in enumerate(horizons):
        x = h_left + index * (h_width / 5.0)
        parts.append(_text(x + 16, 686, f"{horizon_item.horizon_days}天", size=12))
        for row, value, color in (
            (0, horizon_item.spatial_information_gain, "#f8961e"),
            (1, horizon_item.etas_information_gain, "#577590"),
        ):
            y = 700.0 + row * 58.0
            if value is None:
                fill = "#edf1f5"
                label = "—"
            else:
                strength = min(1.0, abs(value) / extent)
                fill = color if value > 0.0 else "#c8553d"
                opacity = 0.25 + 0.75 * strength
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="75" height="42" rx="5" '
                    f'fill="{fill}" fill-opacity="{opacity:.3f}"/>'
                )
                label = f"{value:+.3f}"
            if value is None:
                parts.append(_rect(x, y, 75, 42, fill=fill, radius=5))
            parts.append(_text(x + 13, y + 27, label, size=12, fill="#172b4d"))
    parts.append(_text(735, 727, "空间", size=13))
    parts.append(_text(735, 785, "ETAS", size=13))

    parts.extend(
        (
            _rect(40, 855, 1360, 150, fill=panel_fill, stroke=panel_stroke, radius=12),
            _text(65, 890, "E. ETAS unsupported 父历史敏感性", size=21, fill="#102a43"),
            _text(
                65,
                915,
                "同一目标与支持域：比较纳入合格 unsupported 条件父事件和完全排除后的信息增益。",
                size=13,
                fill="#52667a",
            ),
            _text(595, 945, "纳入父历史", size=13, fill="#52667a"),
            _text(815, 945, "完全排除", size=13, fill="#52667a"),
            _text(1035, 945, "差值（排除−纳入）", size=13, fill="#52667a"),
        )
    )
    for index, sensitivity_item in enumerate(sensitivity):
        y = 967.0 + index * 24.0
        parts.append(_text(420, y, _SNAPSHOT_LABELS[sensitivity_item.snapshot_id], size=13))
        primary_value = sensitivity_item.primary_information_gain
        excluded_value = sensitivity_item.excluded_information_gain
        difference = sensitivity_item.difference
        primary_label = "未评分" if primary_value is None else f"{primary_value:+.4f}"
        excluded_label = "未评分" if excluded_value is None else f"{excluded_value:+.4f}"
        difference_label = "—" if difference is None else f"{difference:+.4f}"
        parts.append(_text(610, y, primary_label, size=13))
        parts.append(_text(830, y, excluded_label, size=13))
        parts.append(_text(1080, y, difference_label, size=13))

    parts.append(
        _text(
            55,
            1042,
            (
                f"最终 one-SE 选择：{selected_model_variant_id}　｜　锁定测试：未运行　｜　"
                "unsupported 区域在后续全域指标中仍计为未覆盖"
            ),
            size=15,
            fill="#334e68",
        )
    )
    parts.append("</svg>")
    return "".join(parts).encode("utf-8")


__all__ = [
    "LocalSupportVisualBootstrap",
    "LocalSupportVisualHorizon",
    "LocalSupportVisualSensitivity",
    "LocalSupportVisualSnapshot",
    "render_local_support_results_svg",
]
