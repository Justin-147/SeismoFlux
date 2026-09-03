"""Plot every registered short/mid-horizon S3 initial effect, without model selection."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seismoflux.multitask_s3.preparation import sha256, write_json

HORIZONS = (7, 30, 90, 180)
BANDS = (("Ms5_6", "5.0 ≤ Ms < 6.0"), ("Ms6_plus", "Ms ≥ 6.0"))
COLORS = ("#8b929d", "#586680", "#4c8caf", "#d3a342", "#aa4b65")
SPATIAL = ("CATALOG", "R30_REFERENCE", "CAT_COV", "CAT_SNAP", "CAT_DYN")
LABELS = ("地震目录", "近期活动参考", "加报告覆盖", "加异常现状", "加现状与趋势")


def render(source: Path, output: Path) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload["status"] != "initial_development_effects_complete_S3_not_complete":
        raise ValueError("initial S3 effect summary is not complete")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "font.size": 11,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#bac2ca",
            "text.color": "#263245",
            "axes.labelcolor": "#263245",
            "xtick.color": "#596575",
            "ytick.color": "#596575",
        }
    )
    footer = (
        "历史开发对比，不是独立检验；仅使用主间隔起报。365天无完整评价窗。"
        "\n置乱归因与完整分层仍待完成；小幅改善保留，不能据此声称普遍有效。"
    )
    points = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    x = np.arange(len(HORIZONS))
    for ax, (band, title) in zip(axes, BANDS, strict=True):
        for offset, (variant, label, color) in enumerate(zip(SPATIAL, LABELS, COLORS, strict=True)):
            heights, fractions = [], []
            for horizon in HORIZONS:
                row = payload["pooled_folds"][f"h{horizon:03d}__{band}"]["primary_nonoverlap"]
                alarm = next(
                    item
                    for item in row["spatial"][variant]["alarms"]
                    if item["area_budget_km2"] == 600000
                )
                score = alarm["strict"]["anchor"]
                heights.append(score["recall"] * 100 if score["recall"] is not None else np.nan)
                fractions.append(f"{score['hits']}/{score['total']}")
                points.append({"horizon_days": horizon, "band": band, "model": variant, **score})
            bars = ax.bar(x + (offset - 2) * 0.15, heights, width=0.135, color=color, label=label)
            for bar, value, fraction in zip(bars, heights, fractions, strict=True):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 1.6,
                        fraction,
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        rotation=90,
                    )
        for index, horizon in enumerate(HORIZONS):
            count = payload["pooled_folds"][f"h{horizon:03d}__{band}"]["primary_nonoverlap"][
                "sample_counts"
            ]["unique_anchors"]
            if count == 0:
                ax.text(index, 5, "无目标震例", ha="center", fontsize=10, color="#838c99")
        ax.set_title(title, loc="left", pad=15, fontweight="bold")
        ax.set_xticks(x, [f"{h}天" for h in HORIZONS])
        ax.set_ylim(0, 120)
        ax.set_yticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_xlabel("预报时长")
    axes[0].set_ylabel("独立首震命中率（柱顶为命中数 / 目标数）")
    fig.suptitle(
        "同样的报警预算，异常信息是否让地震少漏报？", x=0.06, ha="left", fontsize=19, y=0.98
    )
    fig.text(
        0.06, 0.90, "两段较晚历史开发时期 · 报警预算 60万 km² · 严格落入报警格内", color="#657183"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.13), ncol=5, frameon=False
    )
    fig.text(0.06, 0.025, footer, fontsize=9, color="#687383")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.29, wspace=0.13)
    first = output / "01_fixed_area_anchor_recall"
    for suffix in ("png", "svg"):
        fig.savefig(first.with_suffix(f".{suffix}"), dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    count_points = []
    for ax, (band, title) in zip(axes, BANDS, strict=True):
        for index, (design, label, color) in enumerate(
            zip(("COV", "SNAP", "DYN"), LABELS[2:], COLORS[2:], strict=True)
        ):
            values = []
            for horizon in HORIZONS:
                contrast = payload["pooled_folds"][f"h{horizon:03d}__{band}"]["primary_nonoverlap"][
                    "count_contrasts"
                ][f"T0_CAL_{design}_minus_T0_CAL"]
                value = contrast["delta_poisson_log_score_mean"]
                values.append(np.nan if value is None else value)
                count_points.append(
                    {
                        "horizon_days": horizon,
                        "band": band,
                        "design": design,
                        "delta_mean_log_score": value,
                    }
                )
            ax.bar(x + (index - 1) * 0.22, values, width=0.2, color=color, label=label)
        ax.axhline(0, color="#344154", linewidth=1)
        ax.set_title(title, loc="left", pad=14, fontweight="bold")
        ax.set_xticks(x, [f"{h}天" for h in HORIZONS])
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_xlabel("预报时长")
    axes[0].set_ylabel("相对校准目录模型的平均对数评分变化")
    fig.suptitle("异常信息是否改善未来地震次数的判断？", x=0.06, ha="left", fontsize=19, y=0.97)
    fig.text(
        0.06,
        0.89,
        "零线为校准目录参照：高于零更好，低于零更差；包含没有地震的时窗",
        color="#657183",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.13), ncol=3, frameon=False
    )
    fig.text(0.06, 0.025, footer, fontsize=9, color="#687383")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.81, bottom=0.28, wspace=0.25)
    second = output / "02_count_information_change"
    for suffix in ("png", "svg"):
        fig.savefig(second.with_suffix(f".{suffix}"), dpi=180, facecolor="white")
    plt.close(fig)
    result: dict[str, object] = {
        "source_sha256": sha256(source),
        "local_only": True,
        "alarm_budget_km2": 600000,
        "spatial_points": points,
        "count_points": count_points,
        "figures": [first.name, second.name],
        "scope": "initial_historical_development_not_full_S3",
    }
    write_json(output / "render_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(arguments.source, arguments.output)
