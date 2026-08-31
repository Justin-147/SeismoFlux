"""Build the public-facing SeismoFlux scientific figures and offline explorer.

This module is deliberately separated from the frozen prospective runtime.  It
reads immutable retrospective artifacts and produces publication-only views;
it never fits a model, changes a score, or reads the locked test set.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pyproj import CRS, Geod, Transformer


matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei",
            "Arial",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


EQUAL_AREA_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+datum=WGS84 +units=m +no_defs +type=crs"
)
AREA_BUDGETS = (300_000, 450_000, 600_000, 750_000, 960_000)
PRIMARY_AREA = 600_000
MODEL_COLORS = {"B0": "#64748B", "B0_R30": "#D85A30"}
INK = "#172235"
MUTED = "#5D6B7E"
GRID = "#DCE4EC"
PALE = "#F3F6F8"
HIT = "#168F73"
GAIN = "#D85A30"
MISS = "#D5DDE5"
BLUE = "#1F6F8B"
GOLD = "#E3A62F"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    cluster_id: str
    title: str
    short_title: str
    issue_id: str
    issue_time_utc: str
    target_time_utc: str
    target_lon: float
    target_lat: float
    target_magnitude: float
    representative_cell_index: int
    b0_rank: int
    r30_rank: int
    nearby_summary: str
    interpretation: str
    role: str


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="subei_20231201",
        cluster_id="10f4293cb49fa075cacd293ce19e263df01b6888a5ec14d24f7d6371604c9936",
        title="甘肃酒泉肃北 M5.0",
        short_title="肃北 M5.0",
        issue_id="d96368e8-8c1c-523f-b8c9-9fd6beb35226",
        issue_time_utc="2023-11-22T16:00:00Z",
        target_time_utc="2023-12-01T14:55:56Z",
        target_lon=97.27,
        target_lat=39.30,
        target_magnitude=5.0,
        representative_cell_index=9739,
        b0_rank=3912,
        r30_rank=194,
        nearby_summary="起报前29天，目标附近14.5 km出现M5.5",
        interpretation="近期活动把一个长期背景中不突出的区域推入等面积报警区。",
        role="正文震例：地域上区别于库车热点。",
    ),
    CaseSpec(
        case_id="zaduo_20240307",
        cluster_id="dccab079d56e831efd655c8c5533e561d23738244464b19747759b70255d3976",
        title="青海玉树杂多 M5.5",
        short_title="杂多 M5.5",
        issue_id="29609dfd-b7f4-5025-8e96-d05b88192bb6",
        issue_time_utc="2024-03-06T16:00:00Z",
        target_time_utc="2024-03-07T10:06:30Z",
        target_lon=93.01,
        target_lat=33.58,
        target_magnitude=5.5,
        representative_cell_index=6071,
        b0_rank=1669,
        r30_rank=153,
        nearby_summary="起报前约46小时，6.7 km处出现M5.3",
        interpretation="模型捕捉的是活跃震群的延续，不应解释为全新区域发现。",
        role="补充震例：用于说明模型擅长延续性活动。",
    ),
    CaseSpec(
        case_id="kuche_20240822",
        cluster_id="fe1c02a83a0ec5f0d22d3de7f35e742173f438d638882c4103208d020aa7abb4",
        title="新疆库车 M5.0（2024年8月）",
        short_title="库车 M5.0",
        issue_id="cc26ca23-737e-5e78-8c5e-29bf8b40e588",
        issue_time_utc="2024-08-07T16:00:00Z",
        target_time_utc="2024-08-21T23:38:00Z",
        target_lon=83.97,
        target_lat=40.92,
        target_magnitude=5.0,
        representative_cell_index=11934,
        b0_rank=2146,
        r30_rank=3,
        nearby_summary="起报前30天，75 km内有3个M4+事件",
        interpretation="近期活动使目标格升至全国第3；与10月震例落在同一25 km格。",
        role="补充震例：不能视作单独的空间重复证据。",
    ),
    CaseSpec(
        case_id="kuche_20241026",
        cluster_id="b33ea2715499fbd3679393bf2cc3b64a541f307a9a48f644026c5b6cf8df67ab",
        title="新疆库车 M5.5（2024年10月）",
        short_title="库车 M5.5",
        issue_id="928b1f44-c48b-57a6-b52c-09fee1e8e76e",
        issue_time_utc="2024-10-16T16:00:00Z",
        target_time_utc="2024-10-26T08:35:57Z",
        target_lon=83.93,
        target_lat=40.98,
        target_magnitude=5.5,
        representative_cell_index=11934,
        b0_rank=1725,
        r30_rank=3,
        nearby_summary="起报前30天，75 km内有5个M4+；最近M4.9约提前4小时",
        interpretation="近期活动把目标格从长期背景第1725位推到第3位。",
        role="正文主震例：空间和时间信号最清楚。",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "figures": root / "figures",
        "interactive": root / "interactive",
        "figure_data": root / "figure_data",
        "build": root / "build",
        "qa": root / "qa",
        "manuscript": root / "manuscript",
        "presentation": root / "presentation",
        "poster": root / "poster",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _save_figure(fig: plt.Figure, base: Path, *, dpi: int = 220) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _metric_index(summary: Mapping[str, Any]) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    result: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in summary["all_metrics"]:
        if row["fold_id"] is None:
            result[(int(row["horizon_days"]), int(row["area_budget_km2"]), row["model_id"])] = row
    return result


def _bootstrap_index(summary: Mapping[str, Any]) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    return {
        (int(row["horizon_days"]), int(row["area_budget_km2"]), row["model_id"]): row
        for row in summary["paired_cluster_bootstrap"]
    }


def _write_metric_csv(
    path: Path,
    metrics: Mapping[tuple[int, int, str], Mapping[str, Any]],
    bootstrap: Mapping[tuple[int, int, str], Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "horizon_days",
                "area_budget_km2",
                "model_id",
                "cluster_count",
                "hit_count",
                "recall",
                "recall_gain_vs_b0",
                "bootstrap_lower_95",
                "bootstrap_upper_95",
                "bootstrap_positive_gain_replicate_proportion",
            ],
        )
        writer.writeheader()
        for horizon in (30, 90):
            for area in AREA_BUDGETS:
                baseline = metrics[(horizon, area, "B0")]
                for model in ("B0", "B0_R30"):
                    row = metrics[(horizon, area, model)]
                    boot = bootstrap.get((horizon, area, model))
                    writer.writerow(
                        {
                            "horizon_days": horizon,
                            "area_budget_km2": area,
                            "model_id": model,
                            "cluster_count": row["cluster_count"],
                            "hit_count": row["hit_count"],
                            "recall": f"{row['recall']:.8f}",
                            "recall_gain_vs_b0": f"{row['recall'] - baseline['recall']:.8f}",
                            "bootstrap_lower_95": "" if boot is None else f"{boot['lower_95']:.8f}",
                            "bootstrap_upper_95": "" if boot is None else f"{boot['upper_95']:.8f}",
                            "bootstrap_positive_gain_replicate_proportion": ""
                            if boot is None
                            else f"{boot['probability_gain_positive']:.8f}",
                        }
                    )


def _figure_primary_result(
    metrics: Mapping[tuple[int, int, str], Mapping[str, Any]],
    bootstrap: Mapping[tuple[int, int, str], Mapping[str, Any]],
    output_base: Path,
) -> None:
    fig = plt.figure(figsize=(14.2, 7.8), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 3, width_ratios=[0.9, 1.35, 1.05], height_ratios=[1, 1])
    ax_bar = fig.add_subplot(grid_spec[:, 0])
    ax_curve = fig.add_subplot(grid_spec[0, 1:])
    ax_fold = fig.add_subplot(grid_spec[1, 1])
    ax_scope = fig.add_subplot(grid_spec[1, 2])

    primary = [metrics[(30, PRIMARY_AREA, model)] for model in ("B0", "B0_R30")]
    values = [row["recall"] * 100 for row in primary]
    bars = ax_bar.bar(
        [0, 1],
        values,
        width=0.62,
        color=[MODEL_COLORS["B0"], MODEL_COLORS["B0_R30"]],
        zorder=3,
    )
    ax_bar.set_ylim(0, 52)
    ax_bar.set_xticks([0, 1], ["长期背景\nB0", "加入近30天活动\nB0_R30"], fontsize=11)
    ax_bar.set_ylabel("规则化震群召回（%）", color=INK)
    ax_bar.set_title("相同报警面积上限，多命中4个震群", loc="left", fontsize=15, fontweight="bold", color=INK)
    ax_bar.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for bar, row in zip(bars, primary, strict=True):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.4,
            f"{row['hit_count']}/{row['cluster_count']}\n{row['recall']:.1%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=INK,
        )
    boot = bootstrap[(30, PRIMARY_AREA, "B0_R30")]
    ax_bar.text(
        0.5,
        47.8,
        "+19.05个百分点",
        ha="center",
        va="center",
        fontsize=12,
        color=GAIN,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFF1EB", "ec": "none"},
    )
    ax_bar.text(
        0.02,
        -0.18,
        f"30天窗口；报警面积≈60万 km²（研究区约6.4%）\n"
        f"配对Bootstrap 95%区间：{boot['lower_95']:+.2%} 至 {boot['upper_95']:+.2%}",
        transform=ax_bar.transAxes,
        fontsize=9.5,
        color=MUTED,
        va="top",
    )

    x = np.array(AREA_BUDGETS) / 10_000
    for model in ("B0", "B0_R30"):
        y = [metrics[(30, area, model)]["recall"] * 100 for area in AREA_BUDGETS]
        ax_curve.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=MODEL_COLORS[model],
            label=("长期背景 B0" if model == "B0" else "B0 + 近30天活动"),
        )
    ax_curve.axvline(PRIMARY_AREA / 10_000, color="#9AA7B5", linestyle="--", linewidth=1)
    ax_curve.annotate(
        "30万 km²时已命中7/21\n超过B0在60万 km²的5/21",
        xy=(30, metrics[(30, 300_000, "B0_R30")]["recall"] * 100),
        xytext=(45, 46),
        arrowprops={"arrowstyle": "->", "color": GAIN, "lw": 1.2},
        fontsize=10,
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID},
    )
    ax_curve.set_xlabel("报警面积（万 km²）")
    ax_curve.set_ylabel("规则化震群召回（%）")
    ax_curve.set_ylim(0, 62)
    ax_curve.set_title("面积—召回曲线（30天）", loc="left", fontsize=15, fontweight="bold", color=INK)
    ax_curve.grid(color=GRID, linewidth=0.8)
    ax_curve.legend(frameon=False, ncol=2, loc="lower right")

    fold_names = ["2023–2024", "2024–2025", "2025–2026"]
    baseline_hits = np.array([3, 2, 0])
    candidate_hits = np.array([5, 4, 0])
    indices = np.arange(3)
    ax_fold.bar(indices - 0.18, baseline_hits, 0.36, color=MODEL_COLORS["B0"], label="B0")
    ax_fold.bar(indices + 0.18, candidate_hits, 0.36, color=MODEL_COLORS["B0_R30"], label="B0_R30")
    for x_index, before, after in zip(indices, baseline_hits, candidate_hits, strict=True):
        ax_fold.text(x_index, max(before, after) + 0.25, f"{before}→{after}", ha="center", fontsize=10, color=INK)
    ax_fold.set_xticks(indices, fold_names, fontsize=9.5)
    ax_fold.set_ylim(0, 6.3)
    ax_fold.set_ylabel("命中震群数")
    ax_fold.set_title("三段时间外推", loc="left", fontsize=14, fontweight="bold", color=INK)
    ax_fold.grid(axis="y", color=GRID, linewidth=0.8)
    ax_fold.legend(frameon=False, ncol=2, fontsize=9)

    ax_scope.axis("off")
    ax_scope.set_title("证据边界", loc="left", fontsize=14, fontweight="bold", color=INK)
    scope_lines = [
        ("已", "所有起报均只用当时可见目录"),
        ("已", "两模型使用相同60万 km²上限与同一21群"),
        ("已", "21群中无B0命中而R30丢失的反向案例"),
        ("限", "四个新增命中集中在西北，第三折无改善"),
        ("限", "目前仍是历史时间外推，不是真实前瞻证明"),
    ]
    for index, (mark, line) in enumerate(scope_lines):
        y = 0.88 - index * 0.17
        color = HIT if mark == "已" else GOLD
        ax_scope.text(0.02, y, mark, color=color, fontsize=11, fontweight="bold", va="center")
        ax_scope.text(0.11, y, line, color=INK, fontsize=10.5, va="center", wrap=True)
    ax_scope.add_patch(
        FancyBboxPatch(
            (0, 0.02),
            1,
            0.95,
            transform=ax_scope.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor="#F8FAFC",
            edgecolor=GRID,
            linewidth=1,
            zorder=-1,
        )
    )

    fig.suptitle(
        "SeismoFlux：近期地震活动在固定报警面积下提高历史区域召回",
        x=0.01,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    _save_figure(fig, output_base)


def _cluster_rows(observed: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in observed["outcomes"]
        if int(row["horizon_days"]) == 30 and row["model_id"] in {"B0", "B0_R30"}
    ]
    grouped: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for row in selected:
        key = (
            row["cluster_id"],
            row["fold_id"],
            row["issue_id"],
            row["issue_time_utc"],
            int(row["representative_cell_index"]),
        )
        record = grouped.setdefault(
            key,
            {
                "cluster_id": row["cluster_id"],
                "fold_id": row["fold_id"],
                "issue_id": row["issue_id"],
                "issue_time_utc": row["issue_time_utc"],
                "representative_cell_index": int(row["representative_cell_index"]),
            },
        )
        record[row["model_id"]] = bool(row["hit_by_area"][2])
        record[f"{row['model_id']}_log_density"] = float(row["log_density"])
    rows = list(grouped.values())
    rows.sort(key=lambda item: (item["fold_id"], item["issue_time_utc"], item["cluster_id"]))
    case_by_cluster = {case.cluster_id: case for case in CASES}
    for index, row in enumerate(rows, start=1):
        case = case_by_cluster.get(row["cluster_id"])
        row["row_id"] = index
        row["case_id"] = "" if case is None else case.case_id
        row["label"] = (
            f"{index:02d}  {row['issue_time_utc'][:10]}  {row['fold_id'].replace('fold_', '折')}"
            if case is None
            else f"{index:02d}  {case.short_title}"
        )
        row["outcome"] = (
            "gain"
            if (not row["B0"] and row["B0_R30"])
            else "loss"
            if (row["B0"] and not row["B0_R30"])
            else "both_hit"
            if row["B0"] and row["B0_R30"]
            else "both_miss"
        )
    return rows


def _write_cluster_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "row_id",
        "fold_id",
        "issue_time_utc",
        "cluster_id",
        "representative_cell_index",
        "B0",
        "B0_R30",
        "outcome",
        "case_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _figure_cluster_outcomes(rows: Sequence[Mapping[str, Any]], output_base: Path) -> None:
    fig, (ax_matrix, ax_summary) = plt.subplots(
        1,
        2,
        figsize=(12.8, 9.2),
        gridspec_kw={"width_ratios": [1.45, 1]},
        constrained_layout=True,
    )
    ax_matrix.set_xlim(-0.1, 2.2)
    ax_matrix.set_ylim(len(rows) - 0.5, -0.5)
    for index, row in enumerate(rows):
        ax_matrix.text(-0.08, index, row["label"], ha="right", va="center", fontsize=9.2, color=INK)
        for model_index, model in enumerate(("B0", "B0_R30")):
            hit = bool(row[model])
            color = HIT if hit else MISS
            edge = GAIN if row["outcome"] == "gain" and model == "B0_R30" else "white"
            ax_matrix.scatter(
                model_index + 0.45,
                index,
                s=240,
                marker="s",
                c=color,
                edgecolors=edge,
                linewidths=2.5 if edge == GAIN else 1,
                zorder=3,
            )
            ax_matrix.text(
                model_index + 0.45,
                index,
                "命中" if hit else "漏报",
                ha="center",
                va="center",
                fontsize=7.3,
                color="white" if hit else MUTED,
                fontweight="bold" if hit else "normal",
            )
        if row["outcome"] == "gain":
            ax_matrix.annotate(
                "新增",
                xy=(1.05, index),
                xytext=(1.67, index),
                ha="center",
                va="center",
                color=GAIN,
                fontsize=8.5,
                fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "color": GAIN, "lw": 1.2},
            )
    ax_matrix.set_xticks([0.45, 1.45], ["长期背景 B0", "B0 + 近30天活动"], fontsize=11)
    ax_matrix.xaxis.tick_top()
    ax_matrix.tick_params(length=0)
    ax_matrix.set_yticks([])
    ax_matrix.set_title("21个规则化震群逐一对照", loc="left", fontsize=15, fontweight="bold", color=INK, pad=26)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    ax_summary.axis("off")
    counts = {
        "both_hit": sum(row["outcome"] == "both_hit" for row in rows),
        "gain": sum(row["outcome"] == "gain" for row in rows),
        "loss": sum(row["outcome"] == "loss" for row in rows),
        "both_miss": sum(row["outcome"] == "both_miss" for row in rows),
    }
    blocks = [
        ("两者都命中", counts["both_hit"], HIT, "长期背景已经覆盖"),
        ("R30新增命中", counts["gain"], GAIN, "本研究的净增益"),
        ("R30反向丢失", counts["loss"], "#B42318", "主终点中没有发生"),
        ("两者都漏报", counts["both_miss"], "#94A3B8", "仍需改进的主要部分"),
    ]
    for index, (label, count, color, note) in enumerate(blocks):
        y = 0.88 - index * 0.19
        ax_summary.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.09),
                0.94,
                0.145,
                transform=ax_summary.transAxes,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor="#F8FAFC",
                edgecolor=GRID,
            )
        )
        ax_summary.text(0.08, y, str(count), transform=ax_summary.transAxes, fontsize=28, fontweight="bold", color=color, va="center")
        ax_summary.text(0.26, y + 0.025, label, transform=ax_summary.transAxes, fontsize=12, fontweight="bold", color=INK, va="center")
        ax_summary.text(0.26, y - 0.035, note, transform=ax_summary.transAxes, fontsize=9.5, color=MUTED, va="center")
    ax_summary.text(
        0.02,
        0.07,
        "读图方式：每一行是一个按75 km / 30天规则合并后的规则化震群。\n"
        "这张图展示全部样本，正文震例只负责解释‘为什么’，不能替代总体统计。",
        transform=ax_summary.transAxes,
        fontsize=10,
        color=MUTED,
        va="bottom",
    )
    ax_summary.set_title("净变化：+4群，0群反向丢失", loc="left", fontsize=15, fontweight="bold", color=INK)
    fig.suptitle("不是只挑好例：全部21群的命中变化", x=0.01, ha="left", fontsize=20, fontweight="bold", color=INK)
    _save_figure(fig, output_base)


def _load_case_score_rows(scores_path: Path, case: CaseSpec) -> dict[str, dict[str, np.ndarray]]:
    columns = [
        "model_id",
        "cell_index",
        "query_x_m",
        "query_y_m",
        "relative_strength_per_km2",
        "rank",
        "alarm_600000",
    ]
    table = pq.read_table(
        scores_path,
        columns=columns,
        filters=[
            ("issue_id", "=", case.issue_id),
            ("horizon_days", "=", 30),
            ("model_id", "in", ["B0", "B0_R30"]),
        ],
    )
    frame = table.to_pydict()
    transformer = Transformer.from_crs(CRS.from_user_input(EQUAL_AREA_CRS), CRS.from_epsg(4326), always_xy=True)
    result: dict[str, dict[str, np.ndarray]] = {}
    model_array = np.array(frame["model_id"], dtype=object)
    for model in ("B0", "B0_R30"):
        mask = model_array == model
        x_m = np.asarray(frame["query_x_m"], dtype=float)[mask]
        y_m = np.asarray(frame["query_y_m"], dtype=float)[mask]
        lon, lat = transformer.transform(x_m, y_m)
        result[model] = {
            "cell_index": np.asarray(frame["cell_index"], dtype=int)[mask],
            "lon": np.asarray(lon, dtype=float),
            "lat": np.asarray(lat, dtype=float),
            "strength": np.asarray(frame["relative_strength_per_km2"], dtype=float)[mask],
            "rank": np.asarray(frame["rank"], dtype=int)[mask],
            "alarm": np.asarray(frame["alarm_600000"], dtype=bool)[mask],
        }
    return result


def _load_catalog(catalog_path: Path) -> dict[str, np.ndarray]:
    columns = [
        "origin_time_utc",
        "available_at",
        "longitude",
        "latitude",
        "magnitude",
        "inside_study_area",
    ]
    frame = pq.read_table(catalog_path, columns=columns).to_pydict()
    return {
        "origin": np.asarray(frame["origin_time_utc"], dtype="datetime64[us]"),
        "available": np.asarray(frame["available_at"], dtype="datetime64[us]"),
        "lon": np.asarray(frame["longitude"], dtype=float),
        "lat": np.asarray(frame["latitude"], dtype=float),
        "magnitude": np.asarray(frame["magnitude"], dtype=float),
        "inside": np.asarray(frame["inside_study_area"], dtype=bool),
    }


def _recent_events(catalog: Mapping[str, np.ndarray], case: CaseSpec, *, radius_km: float = 250.0) -> list[dict[str, float]]:
    issue = _parse_utc(case.issue_time_utc)
    start = issue - timedelta(days=30)
    issue_np = np.datetime64(issue.replace(tzinfo=None), "us")
    start_np = np.datetime64(start.replace(tzinfo=None), "us")
    mask = (
        (catalog["origin"] > start_np)
        & (catalog["origin"] <= issue_np)
        & (catalog["available"] <= issue_np)
        & (catalog["magnitude"] >= 4.0)
        & catalog["inside"]
    )
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    geod = Geod(ellps="WGS84")
    target_lon = np.full(len(indices), case.target_lon)
    target_lat = np.full(len(indices), case.target_lat)
    _, _, distances_m = geod.inv(
        target_lon,
        target_lat,
        catalog["lon"][indices],
        catalog["lat"][indices],
    )
    selected: list[dict[str, float]] = []
    for index, distance_m in zip(indices, distances_m, strict=True):
        if distance_m <= radius_km * 1_000:
            origin_us = int(catalog["origin"][index].astype("datetime64[us]").astype(np.int64))
            origin = datetime.fromtimestamp(origin_us / 1_000_000, tz=timezone.utc)
            selected.append(
                {
                    "lon": float(catalog["lon"][index]),
                    "lat": float(catalog["lat"][index]),
                    "magnitude": float(catalog["magnitude"][index]),
                    "distance_km": float(distance_m / 1_000),
                    "days_before_issue": float((issue - origin).total_seconds() / 86_400),
                }
            )
    selected.sort(key=lambda row: row["days_before_issue"], reverse=True)
    return selected


def _case_local_payload(
    score_rows: Mapping[str, Mapping[str, np.ndarray]],
    case: CaseSpec,
    recent: Sequence[Mapping[str, float]],
    *,
    lon_pad: float = 3.1,
    lat_pad: float = 2.35,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model, rows in score_rows.items():
        local = (
            (rows["lon"] >= case.target_lon - lon_pad)
            & (rows["lon"] <= case.target_lon + lon_pad)
            & (rows["lat"] >= case.target_lat - lat_pad)
            & (rows["lat"] <= case.target_lat + lat_pad)
        )
        strengths = rows["strength"][local]
        positive = strengths[strengths > 0]
        floor = float(np.min(positive)) if len(positive) else 1e-30
        log_strength = np.log10(np.maximum(strengths, floor))
        low, high = np.quantile(log_strength, [0.05, 0.98]) if len(log_strength) else (0.0, 1.0)
        denominator = max(float(high - low), 1e-12)
        norm = np.clip((log_strength - low) / denominator, 0, 1)
        models[model] = [
            [
                round(float(lon), 4),
                round(float(lat), 4),
                round(float(value), 4),
                bool(alarm),
                int(rank),
            ]
            for lon, lat, value, alarm, rank in zip(
                rows["lon"][local],
                rows["lat"][local],
                norm,
                rows["alarm"][local],
                rows["rank"][local],
                strict=True,
            )
        ]
    issue = _parse_utc(case.issue_time_utc)
    target = _parse_utc(case.target_time_utc)
    return {
        "id": case.case_id,
        "title": case.title,
        "shortTitle": case.short_title,
        "issueLocal": (issue + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
        "targetLocal": (target + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
        "daysAfterIssue": round((target - issue).total_seconds() / 86_400, 2),
        "target": [case.target_lon, case.target_lat, case.target_magnitude],
        "ranks": {"B0": case.b0_rank, "B0_R30": case.r30_rank},
        "nearbySummary": case.nearby_summary,
        "interpretation": case.interpretation,
        "role": case.role,
        "bounds": [
            case.target_lon - lon_pad,
            case.target_lon + lon_pad,
            case.target_lat - lat_pad,
            case.target_lat + lat_pad,
        ],
        "recent": [
            [
                round(float(row["lon"]), 3),
                round(float(row["lat"]), 3),
                round(float(row["magnitude"]), 1),
                round(float(row["distance_km"]), 1),
                round(float(row["days_before_issue"]), 2),
            ]
            for row in recent
        ],
        "models": models,
    }


def _draw_local_map(
    ax: plt.Axes,
    rows: Mapping[str, np.ndarray],
    case: CaseSpec,
    recent: Sequence[Mapping[str, float]],
    model: str,
    *,
    lon_pad: float = 3.1,
    lat_pad: float = 2.35,
) -> None:
    local = (
        (rows["lon"] >= case.target_lon - lon_pad)
        & (rows["lon"] <= case.target_lon + lon_pad)
        & (rows["lat"] >= case.target_lat - lat_pad)
        & (rows["lat"] <= case.target_lat + lat_pad)
    )
    lon = rows["lon"][local]
    lat = rows["lat"][local]
    strength = rows["strength"][local]
    alarm = rows["alarm"][local]
    positive = strength[strength > 0]
    floor = np.min(positive) if len(positive) else 1e-30
    log_strength = np.log10(np.maximum(strength, floor))
    low, high = np.quantile(log_strength, [0.05, 0.98]) if len(log_strength) else (0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("seismoflux", ["#F7FAFC", "#9CC9D4", "#1F6F8B", "#17324D"])
    ax.scatter(lon, lat, c=log_strength, cmap=cmap, norm=Normalize(low, high), marker="s", s=17, linewidths=0, alpha=0.95)
    ax.scatter(lon[alarm], lat[alarm], facecolors="none", edgecolors=GOLD, marker="s", s=25, linewidths=0.75, alpha=0.95)
    if recent:
        ax.scatter(
            [row["lon"] for row in recent],
            [row["lat"] for row in recent],
            s=[18 + 28 * (row["magnitude"] - 4) for row in recent],
            facecolors="#F6C453",
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
            zorder=5,
        )
    ax.scatter([case.target_lon], [case.target_lat], marker="*", s=180, color="#C83E3A", edgecolors="white", linewidths=1, zorder=8)
    ax.set_xlim(case.target_lon - lon_pad, case.target_lon + lon_pad)
    ax.set_ylim(case.target_lat - lat_pad, case.target_lat + lat_pad)
    ax.set_aspect(1 / max(math.cos(math.radians(case.target_lat)), 0.35))
    ax.grid(color="#FFFFFF", linewidth=0.7, alpha=0.9)
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    rank = case.b0_rank if model == "B0" else case.r30_rank
    status = "漏报" if model == "B0" else "命中"
    ax.set_title(
        f"{'长期背景 B0' if model == 'B0' else 'B0 + 近30天活动'}\n目标格第{rank}名 · {status}",
        fontsize=11.3,
        fontweight="bold",
        color=INK if model == "B0" else GAIN,
    )


def _draw_case_timeline(
    ax: plt.Axes,
    case: CaseSpec,
    recent: Sequence[Mapping[str, float]],
) -> None:
    nearby = [row for row in recent if row["distance_km"] <= 75]
    ax.axvspan(-30, 0, color="#F2F7F9")
    ax.axvline(0, color=INK, linewidth=1.4)
    ax.hlines(0, -30, 30, color="#9BA8B5", linewidth=1)
    for row in nearby:
        x = -row["days_before_issue"]
        height = 0.12 + (row["magnitude"] - 4) * 0.2
        ax.vlines(x, 0, height, color=BLUE, linewidth=1.4)
        ax.scatter([x], [height], s=28 + 34 * (row["magnitude"] - 4), color="#F6C453", edgecolors=BLUE, linewidths=0.7, zorder=4)
    issue = _parse_utc(case.issue_time_utc)
    target = _parse_utc(case.target_time_utc)
    target_day = (target - issue).total_seconds() / 86_400
    ax.vlines(target_day, 0, 0.56, color="#C83E3A", linewidth=1.8)
    ax.scatter([target_day], [0.56], marker="*", s=145, color="#C83E3A", zorder=5)
    ax.text(target_day, 0.68, f"目标 M{case.target_magnitude:.1f}\n起报后{target_day:.1f}天", ha="center", fontsize=9.5, color="#A12B28", fontweight="bold")
    ax.text(0, -0.13, "起报 T", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_xlim(-30.5, 30.5)
    ax.set_ylim(-0.22, 0.95)
    ax.set_yticks([])
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_xlabel("相对起报日（天）", fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("目标附近75 km的时间线", loc="left", fontsize=11.3, fontweight="bold", color=INK)
    ax.text(
        0.01,
        0.98,
        case.nearby_summary,
        transform=ax.transAxes,
        va="top",
        fontsize=9.3,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": GRID},
    )


def _figure_case_studies(
    case_data: Mapping[str, Mapping[str, Any]],
    output_base: Path,
) -> None:
    selected = (CASES[3], CASES[0])
    fig = plt.figure(figsize=(15.5, 10.8))
    spec = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.05, 1.05],
        left=0.05,
        right=0.985,
        top=0.82,
        bottom=0.15,
        hspace=0.78,
        wspace=0.30,
    )
    for row_index, case in enumerate(selected):
        data = case_data[case.case_id]
        for col_index, model in enumerate(("B0", "B0_R30")):
            ax = fig.add_subplot(spec[row_index, col_index])
            _draw_local_map(ax, data["scores"][model], case, data["recent"], model)
        timeline = fig.add_subplot(spec[row_index, 2])
        _draw_case_timeline(timeline, case, data["recent"])
        timeline.text(
            0.01,
            -0.30,
            case.interpretation,
            transform=timeline.transAxes,
            fontsize=9.5,
            color=INK,
            va="top",
            wrap=True,
        )
        figure_y = 0.855 if row_index == 0 else 0.455
        fig.text(0.01, figure_y, case.title, fontsize=15, fontweight="bold", color=INK)
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none", markeredgecolor=GOLD, markersize=9, label="60万 km²报警格"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F6C453", markeredgecolor=BLUE, markersize=8, label="起报前30天M4+"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#C83E3A", markeredgecolor="#C83E3A", markersize=12, label="事后叠加目标地震"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "两个说明性震例：相同面积下，近期活动如何改变空间排序",
        x=0.01,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.99,
        0.014,
        "震例为结果形成后选出的解释性案例；总体证据以全部21个规则化震群为准。",
        ha="right",
        fontsize=9,
        color=MUTED,
    )
    _save_figure(fig, output_base, dpi=230)


def _figure_case_rank_shifts(output_base: Path) -> None:
    fig, (ax_rank, ax_note) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.6),
        gridspec_kw={"width_ratios": [1.55, 1]},
        constrained_layout=True,
    )
    y = np.arange(len(CASES))
    for index, case in enumerate(CASES):
        ax_rank.plot([case.r30_rank, case.b0_rank], [index, index], color="#CBD5E1", linewidth=3, zorder=1)
        ax_rank.scatter(case.b0_rank, index, s=75, color=MODEL_COLORS["B0"], zorder=3)
        ax_rank.scatter(case.r30_rank, index, s=95, color=MODEL_COLORS["B0_R30"], zorder=4)
        ax_rank.text(case.b0_rank * 1.08, index + 0.12, str(case.b0_rank), color=MODEL_COLORS["B0"], fontsize=9)
        ax_rank.text(max(case.r30_rank / 1.25, 1.4), index - 0.16, str(case.r30_rank), color=MODEL_COLORS["B0_R30"], fontsize=9, ha="right")
    ax_rank.set_xscale("log")
    ax_rank.set_xlim(1, 6000)
    ax_rank.set_yticks(y, [case.short_title for case in CASES])
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("目标格在15,697个网格中的名次（对数坐标；越靠左越高）")
    ax_rank.grid(axis="x", color=GRID, linewidth=0.8)
    ax_rank.set_title("四个新增命中的排名变化", loc="left", fontsize=15, fontweight="bold", color=INK)
    ax_rank.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["B0"], markersize=8, label="B0"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["B0_R30"], markersize=8, label="B0_R30"),
        ],
        frameon=False,
        loc="lower right",
    )
    for spine in ax_rank.spines.values():
        spine.set_visible(False)

    ax_note.axis("off")
    ax_note.set_title("如何理解这些案例", loc="left", fontsize=15, fontweight="bold", color=INK)
    notes = [
        ("最强变化", "库车两个震例都升至全国第3名，但位于同一25 km格。"),
        ("机制线索", "改善与起报前30天的近场M4+活动一致，主要反映地震聚集。"),
        ("谨慎解释", "杂多案例属于活跃序列延续；四例全部来自西北地区。"),
        ("统计主体", "案例只解释模型行为，+4群的证据仍来自全部21群的成对比较。"),
    ]
    for index, (title, body) in enumerate(notes):
        y0 = 0.86 - index * 0.21
        ax_note.text(0.02, y0, title, transform=ax_note.transAxes, fontsize=11.5, fontweight="bold", color=GAIN if index == 0 else INK, va="top")
        ax_note.text(0.02, y0 - 0.075, body, transform=ax_note.transAxes, fontsize=10, color=MUTED, va="top", wrap=True)
    ax_note.add_patch(
        FancyBboxPatch(
            (0, 0.02),
            0.98,
            0.94,
            transform=ax_note.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor="#F8FAFC",
            edgecolor=GRID,
            zorder=-1,
        )
    )
    fig.suptitle("新增命中来自目标格的大幅升位，而不是扩大报警面积", x=0.01, ha="left", fontsize=20, fontweight="bold", color=INK)
    _save_figure(fig, output_base)


def _figure_method_overview(output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 6.1))
    ax.axis("off")
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 6.1)
    boxes = [
        (0.35, 3.20, 2.35, 1.55, "历史目录", "1970年至起报日T\n仅使用当时已可见M4+", "#EAF2F6", BLUE),
        (3.10, 3.20, 2.35, 1.55, "长期背景 B0", "75 km平滑\n描述长期空间易发性", "#EEF1F5", MODEL_COLORS["B0"]),
        (3.10, 0.95, 2.35, 1.55, "近期活动 R30", "(T−30天, T]\n描述短期地震聚集", "#FFF0E9", MODEL_COLORS["B0_R30"]),
        (5.85, 2.08, 2.55, 1.65, "冻结混合", "75% B0 + 25% R30\n各折仅在更早历史中确定", "#FFF8E7", GOLD),
        (8.80, 2.08, 2.25, 1.65, "网格排序", "15,697个约25 km格\n输出相对强度与顺位", "#EAF4F1", HIT),
        (11.45, 2.08, 2.35, 1.65, "固定面积评价", "选取约60万 km²\n事后评价30天M5–6震群", "#F2EBF7", "#7C5AA6"),
    ]
    for x, y, width, height, title, body, face, edge in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6,
            )
        )
        ax.text(x + 0.18, y + height - 0.32, title, fontsize=13, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.18, y + height - 0.72, body, fontsize=10.2, color=MUTED, va="top", linespacing=1.45)
    arrows = [
        ((2.70, 3.98), (3.10, 3.98)),
        ((2.70, 3.55), (3.10, 1.72)),
        ((5.45, 3.70), (5.85, 3.15)),
        ((5.45, 1.72), (5.85, 2.63)),
        ((8.40, 2.90), (8.80, 2.90)),
        ((11.05, 2.90), (11.45, 2.90)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#8695A5"})
    ax.text(
        0.35,
        5.52,
        "科学问题：在不提高报警面积上限的前提下，近期地震活动能否覆盖更多规则化震群？",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.35,
        0.30,
        "因果边界：T之后的地震只用于报警冻结后的评分；真实震中不参与候选生成、网格加密或模型训练。",
        fontsize=10.5,
        color=MUTED,
    )
    _save_figure(fig, output_base)


def _interactive_html(payload: Mapping[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeismoFlux 科学结果浏览器</title>
<style>
:root{{--ink:#172235;--muted:#5d6b7e;--line:#dce4ec;--paper:#f4f7f9;--white:#fff;--base:#64748b;--gain:#d85a30;--hit:#168f73;--gold:#e3a62f;--blue:#1f6f8b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);font-family:Arial,"Microsoft YaHei",sans-serif;color:var(--ink)}}
.shell{{max-width:1440px;margin:0 auto;padding:28px}}.hero{{background:linear-gradient(120deg,#17324d 0%,#1f6f8b 68%,#2c8fa3 100%);color:white;border-radius:24px;padding:38px 42px;box-shadow:0 16px 42px rgba(23,50,77,.18)}}
.eyebrow{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;opacity:.78;font-weight:700}}h1{{font-size:clamp(28px,4vw,52px);line-height:1.08;max-width:1000px;margin:15px 0 12px}}.hero p{{max-width:950px;line-height:1.65;font-size:17px;margin:0;opacity:.9}}
.warning{{margin-top:20px;display:inline-flex;gap:9px;align-items:center;background:rgba(255,255,255,.12);padding:9px 13px;border-radius:999px;font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0}}.card{{background:white;border:1px solid var(--line);border-radius:18px;padding:19px 20px;box-shadow:0 6px 20px rgba(23,34,53,.05)}}.card .k{{color:var(--muted);font-size:12px;font-weight:700;letter-spacing:.06em}}.card .v{{font-size:30px;font-weight:800;margin-top:7px}}.card .s{{font-size:12px;color:var(--muted);margin-top:4px;line-height:1.5}}.accent{{color:var(--gain)}}
.tabs{{display:flex;gap:8px;margin:26px 0 16px;flex-wrap:wrap}}button,select{{font:inherit}}.tab{{border:1px solid var(--line);background:white;padding:10px 16px;border-radius:999px;cursor:pointer;color:var(--muted);font-weight:700}}.tab.active{{background:var(--ink);color:white;border-color:var(--ink)}}.panel{{display:none}}.panel.active{{display:block}}.section{{background:white;border:1px solid var(--line);border-radius:20px;padding:24px;margin-bottom:18px;box-shadow:0 6px 20px rgba(23,34,53,.04)}}h2{{margin:0 0 7px;font-size:24px}}.lead{{margin:0 0 20px;color:var(--muted);line-height:1.65}}.chart{{width:100%;height:auto;display:block}}.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin:10px 0}}.dot{{width:10px;height:10px;border-radius:99px;display:inline-block;margin-right:6px}}
.matrix{{display:grid;grid-template-columns:minmax(150px,1fr) 95px 110px 92px;gap:6px;align-items:center;max-width:900px}}.matrix .head{{font-size:12px;color:var(--muted);font-weight:700;padding:8px}}.matrix .label{{font-size:13px;padding:7px 9px}}.state{{text-align:center;padding:7px;border-radius:8px;font-size:12px;font-weight:700}}.hit{{background:#dff3ec;color:#0f6c57}}.miss{{background:#edf1f5;color:#718096}}.new{{color:var(--gain);font-weight:800}}
.case-controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:18px}}select{{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:white;min-width:260px}}.case-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.map-card{{border:1px solid var(--line);border-radius:16px;padding:13px;background:#fbfcfd}}.map-card h3{{margin:0 0 8px;font-size:16px}}.map{{background:white;border-radius:10px;width:100%;aspect-ratio:1.35;display:block}}.case-meta{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}.pill{{background:#f6f8fa;border-radius:12px;padding:12px}}.pill b{{display:block;font-size:17px}}.pill span{{font-size:11px;color:var(--muted)}}.explain{{border-left:4px solid var(--gain);padding:12px 14px;background:#fff6f1;border-radius:0 12px 12px 0;line-height:1.65}}.limits li{{margin:9px 0;line-height:1.6;color:#344154}}footer{{color:var(--muted);font-size:12px;text-align:center;padding:18px}}
@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}.case-grid{{grid-template-columns:1fr}}.case-meta{{grid-template-columns:1fr 1fr}}.matrix{{grid-template-columns:minmax(120px,1fr) 75px 86px 60px}}.shell{{padding:14px}}.hero{{padding:26px 24px}}}}
</style>
</head>
<body><main class="shell">
<section class="hero"><div class="eyebrow">SeismoFlux · Retrospective scientific evidence</div><h1>相同报警面积上限下，近期地震活动多覆盖了4个规则化震群</h1><p>长期地震背景 B0 与近30天活动 R30 按冻结权重混合。在中国大陆历史时间外推中，30天、600,000 km²面积上限下的命中由5/21提高到9/21。</p><div class="warning">● 当前证据是历史时间外推；截至2026年8月31日真实前瞻为0期</div></section>
<section class="cards"><div class="card"><div class="k">长期背景</div><div class="v">5/21</div><div class="s">23.81% 召回</div></div><div class="card"><div class="k">加入近30天活动</div><div class="v accent">9/21</div><div class="s">42.86% 召回</div></div><div class="card"><div class="k">净增益</div><div class="v accent">+4群</div><div class="s">+19.05个百分点</div></div><div class="card"><div class="k">报警代价</div><div class="v">≈6.4%</div><div class="s">约60万 km² / 研究区</div></div></section>
<nav class="tabs"><button class="tab active" data-tab="performance">总体效果</button><button class="tab" data-tab="clusters">全部21群</button><button class="tab" data-tab="cases">震例回放</button><button class="tab" data-tab="scope">证据边界</button></nav>
<section id="performance" class="panel active"><div class="section"><h2>面积—召回关系</h2><p class="lead">面积越大，报警代价越高。两条曲线在相同面积上限下比较，避免用扩大报警范围换取命中。</p><svg id="curve" class="chart" viewBox="0 0 1000 440" role="img" aria-label="面积召回曲线"></svg><div class="legend"><span><i class="dot" style="background:var(--base)"></i>长期背景 B0</span><span><i class="dot" style="background:var(--gain)"></i>B0 + 近30天活动</span></div></div></section>
<section id="clusters" class="panel"><div class="section"><h2>全部21个规则化震群</h2><p class="lead">每行一个按75 km / 30天规则合并的震群；“新增”表示 B0 漏报而 B0_R30 命中。没有发生反向丢失。</p><div id="matrix" class="matrix"></div></div></section>
<section id="cases" class="panel"><div class="section"><h2>四个新增命中震例</h2><p class="lead">地图中的红星为评分后叠加的目标地震，不参与预测；黄色边框是相同60万 km²面积下的报警格。</p><div class="case-controls"><label for="case-select">选择震例：</label><select id="case-select"></select></div><div id="case-meta" class="case-meta"></div><div class="case-grid"><div class="map-card"><h3>长期背景 B0</h3><svg id="map-b0" class="map" viewBox="0 0 620 455"></svg></div><div class="map-card"><h3>B0 + 近30天活动</h3><svg id="map-r30" class="map" viewBox="0 0 620 455"></svg></div></div><div id="case-explain" class="explain"></div></div></section>
<section id="scope" class="panel"><div class="section"><h2>结论能说到哪里</h2><ul class="limits"><li>结果来自三个按时间向未来外推的历史折，不读取起报日之后的信息生成预测。</li><li>四个新增命中全部位于西北地区，其中两个落在同一25 km网格；第三折为0/7→0/7。</li><li>配对Bootstrap按21个震群重采样，没有进行地域区块Bootstrap；0.9905不是模型正确概率。</li><li>异常表在本次冻结方案中没有带来额外主终点命中；已有置乱检验针对异常增量，不针对B0_R30增量。</li><li>真实前瞻协议已授权并冻结，但截至2026年8月31日尚未产生首期合法预测；未来需等待30/90天真值成熟后才能判断实际增益。</li></ul></div></section>
<footer>相对强度和网格顺位不是绝对发震概率。页面完全离线，不加载外部脚本、字体或地图。</footer>
</main>
<script>
const DATA={data_json};
const NS='http://www.w3.org/2000/svg';
const el=(tag,attrs={{}})=>{{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,String(v)));return n}};
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')}}));
function drawCurve(){{const s=document.getElementById('curve');s.innerHTML='';const L=90,R=950,T=40,B=365;[0,20,40,60].forEach(v=>{{const y=B-v/60*(B-T);s.append(el('line',{{x1:L,x2:R,y1:y,y2:y,stroke:'#dce4ec'}}));const t=el('text',{{x:L-14,y:y+5,'text-anchor':'end',fill:'#5d6b7e','font-size':14}});t.textContent=v+'%';s.append(t)}});DATA.metrics.areas.forEach((a,i)=>{{const x=L+i/(DATA.metrics.areas.length-1)*(R-L);const t=el('text',{{x,y:B+30,'text-anchor':'middle',fill:'#5d6b7e','font-size':14}});t.textContent=(a/10000)+'万';s.append(t)}});[['B0','#64748b'],['B0_R30','#d85a30']].forEach(([m,c])=>{{const pts=DATA.metrics[m].map((v,i)=>[L+i/(DATA.metrics.areas.length-1)*(R-L),B-v*100/60*(B-T)]);s.append(el('polyline',{{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:c,'stroke-width':5,'stroke-linecap':'round','stroke-linejoin':'round'}}));pts.forEach((p,i)=>{{s.append(el('circle',{{cx:p[0],cy:p[1],r:7,fill:'white',stroke:c,'stroke-width':4}}));const t=el('text',{{x:p[0],y:p[1]-15,'text-anchor':'middle',fill:c,'font-size':14,'font-weight':700}});t.textContent=DATA.metrics.hits[m][i]+'/21';s.append(t)}})}});const title=el('text',{{x:(L+R)/2,y:425,'text-anchor':'middle',fill:'#5d6b7e','font-size':15}});title.textContent='报警面积';s.append(title)}}
function drawMatrix(){{const m=document.getElementById('matrix');m.innerHTML='<div class="head">震群 / 起报</div><div class="head">B0</div><div class="head">B0_R30</div><div class="head">变化</div>';DATA.clusters.forEach(r=>{{m.insertAdjacentHTML('beforeend',`<div class="label">${{r.label}}</div><div class="state ${{r.B0?'hit':'miss'}}">${{r.B0?'命中':'漏报'}}</div><div class="state ${{r.B0_R30?'hit':'miss'}}">${{r.B0_R30?'命中':'漏报'}}</div><div class="${{r.outcome==='gain'?'new':''}}">${{r.outcome==='gain'?'新增':'—'}}</div>` )}})}}
function color(v){{const a=[247,250,252],b=[31,111,139];return `rgb(${{a[0]+(b[0]-a[0])*v}},${{a[1]+(b[1]-a[1])*v}},${{a[2]+(b[2]-a[2])*v}})`}}
function drawMap(id,c,m){{const s=document.getElementById(id);s.innerHTML='';const [xmin,xmax,ymin,ymax]=c.bounds;const sx=x=>45+(x-xmin)/(xmax-xmin)*530,sy=y=>405-(y-ymin)/(ymax-ymin)*350;for(let i=0;i<=5;i++){{const x=45+i*106;s.append(el('line',{{x1:x,x2:x,y1:55,y2:405,stroke:'#e7edf2'}}));const y=55+i*70;s.append(el('line',{{x1:45,x2:575,y1:y,y2:y,stroke:'#e7edf2'}}))}}c.models[m].forEach(p=>{{s.append(el('rect',{{x:sx(p[0])-3.3,y:sy(p[1])-3.3,width:6.6,height:6.6,fill:color(p[2]),stroke:p[3]?'#e3a62f':'none','stroke-width':p[3]?1.4:0}}))}});c.recent.forEach(p=>s.append(el('circle',{{cx:sx(p[0]),cy:sy(p[1]),r:3.5+(p[2]-4)*3,fill:'#f6c453',stroke:'#1f6f8b','stroke-width':1}})));const [tx,ty]=[sx(c.target[0]),sy(c.target[1])];const star=el('text',{{x:tx,y:ty+8,'text-anchor':'middle',fill:'#c83e3a','font-size':28,'font-weight':900}});star.textContent='★';s.append(star);const label=el('text',{{x:55,y:35,fill:m==='B0_R30'?'#d85a30':'#172235','font-size':17,'font-weight':700}});label.textContent=`目标格排名：${{c.ranks[m]}} · ${{m==='B0_R30'?'命中':'漏报'}}`;s.append(label)}}
function renderCase(i){{const c=DATA.cases[i];document.getElementById('case-meta').innerHTML=`<div class="pill"><span>起报（北京时间）</span><b>${{c.issueLocal}}</b></div><div class="pill"><span>目标地震</span><b>${{c.targetLocal}} · M${{c.target[2].toFixed(1)}}</b></div><div class="pill"><span>排名变化</span><b class="accent">${{c.ranks.B0}} → ${{c.ranks.B0_R30}}</b></div><div class="pill"><span>起报后发生</span><b>${{c.daysAfterIssue}} 天</b></div>`;document.getElementById('case-explain').innerHTML=`<b>${{c.title}}</b><br>${{c.nearbySummary}}。${{c.interpretation}}<br><span style="color:var(--muted);font-size:12px">${{c.role}}</span>`;drawMap('map-b0',c,'B0');drawMap('map-r30',c,'B0_R30')}}
const sel=document.getElementById('case-select');DATA.cases.forEach((c,i)=>sel.add(new Option(c.title,i)));sel.addEventListener('change',()=>renderCase(Number(sel.value)));drawCurve();drawMatrix();renderCase(0);
</script></body></html>"""


def _write_case_csv(path: Path, cases: Sequence[CaseSpec]) -> None:
    fields = [
        "case_id",
        "cluster_id",
        "title",
        "issue_time_utc",
        "target_time_utc",
        "target_longitude",
        "target_latitude",
        "target_magnitude",
        "b0_rank",
        "b0_r30_rank",
        "nearby_summary",
        "interpretation",
        "role",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "cluster_id": case.cluster_id,
                    "title": case.title,
                    "issue_time_utc": case.issue_time_utc,
                    "target_time_utc": case.target_time_utc,
                    "target_longitude": case.target_lon,
                    "target_latitude": case.target_lat,
                    "target_magnitude": case.target_magnitude,
                    "b0_rank": case.b0_rank,
                    "b0_r30_rank": case.r30_rank,
                    "nearby_summary": case.nearby_summary,
                    "interpretation": case.interpretation,
                    "role": case.role,
                }
            )


def build(project_root: Path, output_root: Path) -> None:
    paths = _ensure_dirs(output_root)
    observed_path = project_root / "data/interim/d1/observed_078e950/observed_result.json"
    scores_path = project_root / "data/interim/d1/observed_078e950/d1_cell_scores.parquet"
    summary_path = project_root / "outputs/visualizations/d1_078e950_final_attribution/d1_science_summary.json"
    catalog_path = project_root / "data/processed/stage1/debc98054172a4a1/earthquake_event.parquet"
    observed = _read_json(observed_path)
    summary = _read_json(summary_path)
    metrics = _metric_index(summary)
    bootstrap = _bootstrap_index(summary)
    cluster_rows = _cluster_rows(observed)
    catalog = _load_catalog(catalog_path)

    _write_metric_csv(paths["figure_data"] / "primary_metrics.csv", metrics, bootstrap)
    _write_cluster_csv(paths["figure_data"] / "cluster_outcomes_30d_600k.csv", cluster_rows)
    _write_case_csv(paths["figure_data"] / "illustrative_case_summary.csv", CASES)

    _figure_primary_result(metrics, bootstrap, paths["figures"] / "figure_01_primary_result")
    _figure_cluster_outcomes(cluster_rows, paths["figures"] / "figure_02_all_cluster_outcomes")

    case_data: dict[str, dict[str, Any]] = {}
    case_payloads: list[dict[str, Any]] = []
    for case in CASES:
        scores = _load_case_score_rows(scores_path, case)
        recent = _recent_events(catalog, case)
        case_data[case.case_id] = {"scores": scores, "recent": recent}
        case_payloads.append(_case_local_payload(scores, case, recent))
    _figure_case_studies(case_data, paths["figures"] / "figure_03_case_studies")
    _figure_case_rank_shifts(paths["figures"] / "figure_04_case_rank_shifts")
    _figure_method_overview(paths["figures"] / "figure_05_method_overview")

    interactive_payload = {
        "metrics": {
            "areas": list(AREA_BUDGETS),
            "B0": [metrics[(30, area, "B0")]["recall"] for area in AREA_BUDGETS],
            "B0_R30": [metrics[(30, area, "B0_R30")]["recall"] for area in AREA_BUDGETS],
            "hits": {
                "B0": [metrics[(30, area, "B0")]["hit_count"] for area in AREA_BUDGETS],
                "B0_R30": [metrics[(30, area, "B0_R30")]["hit_count"] for area in AREA_BUDGETS],
            },
        },
        "clusters": [
            {
                "label": row["label"],
                "B0": row["B0"],
                "B0_R30": row["B0_R30"],
                "outcome": row["outcome"],
            }
            for row in cluster_rows
        ],
        "cases": case_payloads,
    }
    (paths["build"] / "science_payload.json").write_text(
        json.dumps(interactive_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (paths["interactive"] / "seismoflux_science_explorer.html").write_text(
        _interactive_html(interactive_payload), encoding="utf-8"
    )

    captions = """# 图件说明

1. **Figure 01 — 主结果。** 30天、相同600,000 km²报警面积上限下，B0_R30 相对 B0 从5/21提高到9/21；右上为五档面积—召回曲线，左下为三段时间外推，右下为证据边界。
2. **Figure 02 — 全部21群。** 逐震群展示两模型命中状态。B0_R30新增4群，没有反向丢失；该图用于防止仅凭精选震例判断模型。
3. **Figure 03 — 两个说明性震例。** 展示库车M5.5与肃北M5.0的长期背景、近期活动混合结果和时间线。目标地震均为报警冻结后事后叠加。
4. **Figure 04 — 四个新增命中的排名变化。** 横轴为目标格名次（对数坐标），越靠左表示优先级越高。两个库车事件落在同一25 km格。
5. **Figure 05 — 方法概览。** 从当时可见历史目录到固定面积评价的严格时间流程，不包含锁定测试或未来前瞻结果。

所有模型值来自冻结历史回放；相对强度与顺位不是绝对发震概率。震例为结果形成后选出的解释性案例，总体统计以全部21个规则化震群为准。
"""
    (paths["root"] / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/publication/seismoflux_b0_r30_v1"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    build(project_root, output_root.resolve())


if __name__ == "__main__":
    main()
