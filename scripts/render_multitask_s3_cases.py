"""Show every changed primary anchor, with all registered budgets retained."""
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HORIZONS = (7, 30, 90, 180)
BUDGETS = (300000, 450000, 600000, 750000, 960000)
COLORS = ("#ffffff", "#e4e8ec", "#8d9eab", "#158574", "#ce5f45")
CODES = {"both_miss": 1, "both_hit": 2, "gained": 3, "lost": 4}


def build_case_matrix(ledger):
    rows = [
        row
        for row in ledger["rows"]
        if row["primary_nonoverlap"]
        and row["event_view"] == "anchor"
        and row["mode"] == "strict"
        and row["comparison"] == "CAT_DYN_minus_CATALOG"
    ]
    changed = {row["event_id"] for row in rows if row["classification"] in ("gained", "lost")}
    events = sorted(changed, key=lambda event: (ledger["events"][event]["origin_time_utc"], event))
    tasks = [(horizon, budget) for horizon in HORIZONS for budget in BUDGETS]
    matrix = np.zeros((len(events), len(tasks)), dtype=int)
    seen = set()
    for row in rows:
        if row["event_id"] not in changed:
            continue
        key = (row["event_id"], row["horizon_days"], row["area_budget_km2"])
        if key in seen:
            raise ValueError("one earthquake has multiple primary exposures for this task")
        seen.add(key)
        position = tasks.index((row["horizon_days"], row["area_budget_km2"]))
        matrix[events.index(row["event_id"]), position] = CODES[row["classification"]]
    return events, tasks, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    if ledger.get("local_only") is not True:
        raise ValueError("this figure contains earthquake information and must remain local")
    events, tasks, matrix = build_case_matrix(ledger)
    if not events:
        raise ValueError("no changed primary anchor cases to display")
    plt.rcParams.update(
        {"font.family": "Microsoft YaHei", "svg.fonttype": "none", "axes.unicode_minus": False}
    )
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#f7f9fb")
    fig.subplots_adjust(left=0.26, right=0.975, top=0.77, bottom=0.19)
    ax.imshow(matrix, cmap=ListedColormap(COLORS), vmin=0, vmax=4, aspect="auto")
    labels = []
    for event_id in events:
        event = ledger["events"][event_id]
        day = datetime.fromisoformat(event["origin_time_utc"]).astimezone(ZoneInfo("Asia/Shanghai"))
        labels.append(
            f"{day:%Y-%m-%d}   Ms {event['magnitude']:.1f}\n"
            f"{event['longitude']:.2f}°E, {event['latitude']:.2f}°N"
        )
    ax.set_yticks(range(len(events)), labels, fontsize=10)
    ax.set_xticks(range(len(tasks)), [f"{budget / 10000:g}" for _, budget in tasks], fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(events), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    ax.tick_params(axis="y", pad=12)
    ax.set_xlabel("每档全国报警面积上限（万平方公里）", fontsize=11, labelpad=14)
    for block, horizon in enumerate(HORIZONS):
        ax.text(
            block * 5 + 2,
            -0.85,
            f"未来 {horizon} 天",
            ha="center",
            fontsize=12,
            weight="bold",
            color="#203747",
        )
        if block:
            ax.axvline(block * 5 - 0.5, color="#506777", linewidth=1.8)
    for row, column in np.argwhere(matrix >= 3):
        ax.text(
            column,
            row,
            "+" if matrix[row, column] == 3 else "−",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
            color="white",
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.045, 0.94, "异常信息改变了哪些震例的结果？", fontsize=23, weight="bold", color="#173747"
    )
    fig.text(
        0.045,
        0.885,
        "目录 + 异常变化  vs  目录基线  |  严格网格命中 · 主起报轴 · 历史开发期",
        fontsize=12,
        color="#506777",
    )
    legend_labels = ("此主起报轴无该事件", "共同漏报", "共同命中", "新增命中", "丢失命中")
    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#c7cfd4", label=label)
            for color, label in zip(COLORS, legend_labels, strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.58, 0.075),
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.045,
        0.035,
        "展示在任一预设任务下发生得失变化的全部震例；同一地震跨列重复，不是新增独立样本。正负变化均保留，非独立验证。",
        fontsize=10,
        color="#506777",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        target = args.output_dir / f"04_all_changed_anchor_cases.{suffix}"
        if target.exists():
            raise FileExistsError("preserve the existing case figure")
        fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
