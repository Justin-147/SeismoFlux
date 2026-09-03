"""One fixed scientific comparison figure, without selecting tasks or best cases."""

# ruff: noqa: RUF001
# Chinese figure typography intentionally uses fullwidth punctuation.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

HORIZONS = (7, 30, 90, 180, 365)
BANDS = ("Ms5_6", "Ms6_plus")
BUDGETS = (300000, 450000, 600000, 750000, 960000)
VIEWS = ("anchor", "episode_balanced", "all", "subsequent")
SOURCE_FILTER = {
    "fold_scope": "POOLED_A_DEVELOPMENT",
    "axis": "primary_nonoverlap",
    "candidate": "CAT_DYN",
    "reference": "CATALOG",
    "mode": "strict",
}
TITLES = {
    "anchor": "首震｜每条震序的第一个地震",
    "episode_balanced": "震序平衡｜每个成员按 1/n 计权",
    "all": "全部地震｜保留每个事件",
    "subsequent": "后续地震｜首震之后的成员",
}


def _selected_values(blocks: Sequence[Mapping[str, Any]]) -> dict[str, list[list[float | None]]]:
    selected: dict[int, Mapping[str, Any]] = {}
    for block in blocks:
        if block.get("fold_scope") != SOURCE_FILTER["fold_scope"]:
            continue
        horizon = block["horizon_days"]
        if horizon not in HORIZONS:
            raise ValueError("unregistered horizon in the pooled development blocks")
        if horizon in selected:
            raise ValueError("duplicate pooled horizon block")
        selected[horizon] = block
    if set(selected) != set(HORIZONS):
        raise ValueError("missing pooled horizon block")

    values: dict[tuple[str, int, str, int], float | None] = {}
    for horizon, block in selected.items():
        if block.get("status") == "no_complete_evaluation_windows_NA":
            if horizon != 365 or block["rows"]:
                raise ValueError(
                    "only the empty 365-day frozen block is registered as no-window NA"
                )
            for view in VIEWS:
                for band in BANDS:
                    for budget in BUDGETS:
                        values[(view, horizon, band, budget)] = None
            continue
        for row in block["rows"]:
            if any(row.get(field) != expected for field, expected in SOURCE_FILTER.items()):
                continue
            if row["horizon_days"] != horizon:
                raise ValueError("row horizon disagrees with its pooled block")
            view, band, budget = row["event_view"], row["magnitude_band"], row["area_budget_km2"]
            if view not in VIEWS or band not in BANDS or budget not in BUDGETS:
                raise ValueError(
                    "unregistered view, band or area in selected scientific comparison"
                )
            key = (view, horizon, band, budget)
            if key in values:
                raise ValueError("duplicate selected task")
            raw_value = row["national"]["delta_recall_pp"]
            if raw_value is not None and (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, Real)
                or not math.isfinite(float(raw_value))
                or not -100 <= raw_value <= 100
            ):
                raise ValueError(
                    "recall change must be a finite percentage-point value or explicit NA"
                )
            value = None if raw_value is None else float(raw_value)
            if horizon == 365 and value is not None:
                raise ValueError(
                    "the frozen 365-day task has no complete windows and must remain NA"
                )
            values[key] = value
    expected_keys = {
        (view, horizon, band, budget)
        for view in VIEWS
        for horizon in HORIZONS
        for band in BANDS
        for budget in BUDGETS
    }
    if set(values) != expected_keys:
        raise ValueError("missing selected task; unavailable tasks must supply explicit NA")
    return {
        view: [
            [values[(view, horizon, band, budget)] for budget in BUDGETS]
            for horizon in HORIZONS
            for band in BANDS
        ]
        for view in VIEWS
    }


def _font_name() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Microsoft YaHei UI", "Noto Sans CJK SC"):
        if name in available:
            return name
    raise RuntimeError("no supported common Chinese font found for the scientific figure")


def _selected_sample_counts(
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, int] | None]]:
    counts: dict[tuple[str, int, str], dict[str, int]] = {}
    for block in blocks:
        if block.get("fold_scope") != SOURCE_FILTER["fold_scope"]:
            continue
        for row in block["rows"]:
            if any(row.get(field) != value for field, value in SOURCE_FILTER.items()):
                continue
            horizon = row["horizon_days"]
            if horizon == 365:
                continue
            sample = {
                field: row["national"][field]
                for field in ("unique_event_count", "unique_episode_count")
            }
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in sample.values()):
                raise ValueError("sample counts must be nonnegative integers")
            if sample["unique_episode_count"] > sample["unique_event_count"]:
                raise ValueError("unique episodes cannot exceed unique events")
            key = (row["event_view"], horizon, row["magnitude_band"])
            if key in counts and counts[key] != sample:
                raise ValueError("sample counts disagree across alarm budgets")
            counts[key] = sample
    return {
        view: [
            None if horizon == 365 else counts[(view, horizon, band)]
            for horizon in HORIZONS
            for band in BANDS
        ]
        for view in VIEWS
    }


def render_strata_figure(
    blocks: Sequence[Mapping[str, Any]], output_dir: Path, *, dpi: int = 180
) -> dict[str, Any]:
    """Render one 2x2 nationwide comparison from already-summarized predictions.

    No model, budget, reference or favourable subgroup is selected. Extra folds,
    overlapping-report axes, buffered modes and other contrasts are excluded.
    Every fixed task is required; explicit NA is never replaced by zero. DPI
    changes rendering resolution only. Existing PNG/SVG targets are not replaced.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    panels = _selected_values(blocks)
    samples = _selected_sample_counts(blocks)
    font = _font_name()
    output_dir = Path(output_dir)
    paths = {
        extension: output_dir / f"05_strata_recall_changes.{extension}"
        for extension in ("png", "svg")
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("a strata figure target already exists; refusing to overwrite")
    finite = [
        value for rows in panels.values() for row in rows for value in row if value is not None
    ]
    maximum = max((abs(value) for value in finite), default=0.0)
    limit = maximum if maximum > 0 else 1.0
    cmap = LinearSegmentedColormap.from_list(
        "seismoflux_orange_white_blue", ["#bd612e", "#fafbfc", "#286794"]
    )
    cmap = cmap.with_extremes(bad="#dfe3e8")
    norm = Normalize(vmin=-limit, vmax=limit)
    row_labels = [
        f"{horizon}天 · {'Ms 5–6' if band == 'Ms5_6' else 'Ms ≥6'}"
        for horizon in HORIZONS
        for band in BANDS
    ]
    with mpl.rc_context({"font.family": font, "svg.fonttype": "none", "axes.unicode_minus": False}):
        figure = Figure(figsize=(14.5, 13.3), facecolor="#ffffff")
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2)
        figure.subplots_adjust(
            left=0.12, right=0.965, top=0.865, bottom=0.255, wspace=0.30, hspace=0.30
        )
        figure.text(
            0.065,
            0.96,
            "异常信息的改善，体现在哪类地震任务上？",
            fontsize=23,
            fontweight="bold",
            color="#17364b",
        )
        figure.text(
            0.065,
            0.924,
            "已有历史开发预测｜同一全国报警面积预算｜动态异常模型 − 地震目录背景",
            fontsize=12.5,
            color="#526776",
        )
        image = None
        for axis, view in zip(axes.flat, VIEWS, strict=True):
            array = np.asarray(
                [[np.nan if value is None else value for value in row] for row in panels[view]],
                dtype=float,
            )
            image = axis.imshow(np.ma.masked_invalid(array), cmap=cmap, norm=norm, aspect="auto")
            axis.set_title(TITLES[view], loc="left", fontsize=13, fontweight="bold", pad=13)
            sample_labels = [
                row_label
                + "\n"
                + (
                    "无完整窗"
                    if sample is None
                    else f"{sample['unique_event_count']}事件 / "
                    f"{sample['unique_episode_count']}震序"
                )
                for row_label, sample in zip(row_labels, samples[view], strict=True)
            ]
            axis.set_yticks(range(len(row_labels)), sample_labels, fontsize=8.5)
            axis.set_xticks(range(len(BUDGETS)), ["30", "45", "60", "75", "96"], fontsize=10)
            axis.set_xlabel("全国报警面积预算（万 km²）", fontsize=10, labelpad=7)
            axis.set_xticks(np.arange(-0.5, len(BUDGETS), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.8)
            axis.tick_params(which="both", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            for y, row in enumerate(panels[view]):
                for x, value in enumerate(row):
                    label = "NA" if value is None else "0" if value == 0 else f"{value:+.2g}"
                    color = (
                        "white" if value is not None and abs(value) > 0.62 * limit else "#21384a"
                    )
                    axis.text(x, y, label, ha="center", va="center", fontsize=10, color=color)
            for y in (1.5, 3.5, 5.5, 7.5):
                axis.axhline(y, color="white", linewidth=2.2)
        if image is None:
            raise RuntimeError("no scientific panels rendered")
        color_axis = figure.add_axes((0.29, 0.185, 0.46, 0.019))
        colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.set_label(
            "召回率变化（百分点）｜橙：减少命中　蓝：增加命中", fontsize=11, labelpad=7
        )
        colorbar.outline.set_visible(False)
        figure.text(
            0.065,
            0.125,
            "读图：0 表示没有变化；灰色 NA 表示无可评价事件或完整预报窗，不是零效果。"
            "365 天目前无完整评价窗。",
            fontsize=10,
            color="#465967",
        )
        figure.text(
            0.065,
            0.091,
            "震序平衡：n 是全历史固定震序的成员数；每个成员按 1/n 计权，"
            "不在当前年份、地区或子组内重新归一化。",
            fontsize=10,
            color="#465967",
        )
        figure.text(
            0.065,
            0.057,
            "这是同一历史开发任务的点估计，不是成功概率，也不代表胜过所有基线；"
            "区间用于说明不确定性，不作为采用门槛。",
            fontsize=10,
            color="#465967",
        )
        figure.text(
            0.065,
            0.031,
            "震级档：5 ≤ Ms < 6；Ms ≥ 6。行旁为不同事件数 / 不同震序数，不等同独立样本数。",
            fontsize=9.5,
            color="#687985",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension, path in paths.items():
            with path.open("xb") as stream:
                figure.savefig(stream, format=extension, dpi=dpi, facecolor=figure.get_facecolor())
        figure.clear()
    return {
        "status": "rendered",
        "paths": {extension: str(path.resolve()) for extension, path in paths.items()},
        "source_filter": dict(SOURCE_FILTER),
        "event_views": list(VIEWS),
        "row_order": [
            {"horizon_days": horizon, "magnitude_band": band}
            for horizon in HORIZONS
            for band in BANDS
        ],
        "area_budgets_km2": list(BUDGETS),
        "panel_values": panels,
        "panel_sample_counts": samples,
        "selected_task_count": len(VIEWS) * len(HORIZONS) * len(BANDS) * len(BUDGETS),
        "finite_task_count": len(finite),
        "NA_task_count": 200 - len(finite),
        "color_limits_pp": [-limit, limit],
        "effect_unit": "percentage_points",
        "font_family": font,
        "svg_text_preserved": True,
        "dpi": dpi,
        "interpretation": "fixed_development_comparison_not_success_probability_"
        "or_all_baselines_win",
    }
