# ruff: noqa: E501, RUF001
"""S2-A aggregate figures and local-only offline replay of completed scores.

No training or scoring is performed. The existing replay serializer is reused
read-only; S2 labels, scientific scope, case choice and reference are explicit.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath


def _shared_renderer() -> Any:
    path = Path(__file__).with_name("render_multitask_s1_c2b.py")
    spec = importlib.util.spec_from_file_location("seismoflux_s2a_replay_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError("Saved-score replay helpers are unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared = _shared_renderer()
AREA_BUDGETS = _shared.AREA_BUDGETS
REQUIRED_ARTIFACTS = _shared.REQUIRED_ARTIFACTS
_sha256 = _shared._sha256
_json_for_script = _shared._json_for_script
REFERENCE = "C2B_D0_MULTISCALE"
MODEL_LABELS = {
    REFERENCE: "目录｜多尺度（主参考）",
    "C2B_D0_AGE_WEIGHTED": "目录｜年龄加权",
    "C0_L3_B0_R30_CAUSAL": "目录｜长期＋近期",
    "C0_L0_UNIFORM": "均匀空间参考",
    "S2A_SIMPLE_FAULT_ONLY": "简化断层｜纯几何",
    "S2A_SIMPLE_CATALOG_MIX": "简化断层｜精细＋目录",
    "S2A_SIMPLE_COARSE_MIX": "简化断层｜粗化＋目录",
    "S2A_TRACE_FAULT_ONLY": "详细迹线｜纯几何",
    "S2A_TRACE_CATALOG_MIX": "详细迹线｜精细＋目录",
    "S2A_TRACE_COARSE_MIX": "详细迹线｜粗化＋目录",
}
MIX_MODELS = tuple(model for model in MODEL_LABELS if model.endswith("_MIX"))
PAIR_AXES = (
    ("S2A_SIMPLE_CATALOG_MIX", REFERENCE, "简化精细 − 目录"),
    ("S2A_TRACE_CATALOG_MIX", REFERENCE, "详细精细 − 目录"),
    ("S2A_SIMPLE_CATALOG_MIX", "S2A_SIMPLE_COARSE_MIX", "简化精细 − 粗化"),
    ("S2A_TRACE_CATALOG_MIX", "S2A_TRACE_COARSE_MIX", "详细精细 − 粗化"),
)
STATIC_STEMS = ("01_main_anchor_recall", "02_fault_geometry_area_curves", "03_geometry_net_hits")
CASE_STEM = "04_selected_gain_and_failure_local_only"
HTML_NAME = "seismoflux_s2a_fault_geometry_replay.html"
FILENAMES = (
    *(f"{stem}.{suffix}" for stem in (*STATIC_STEMS, CASE_STEM) for suffix in ("png", "svg")),
    HTML_NAME,
)
SNAPSHOT_NOTE = (
    "断层资料为 2026 年收集快照：仅检验对历史地点分布的补充信息，"
    "不代表当时可发布的预测，也不是真正前瞻检验。"
)


def _footnote(summary: dict[str, Any]) -> str:
    prefix = "合成测试示例，不代表科学结果。\n" if summary.get("synthetic_fixture") else ""
    return (
        prefix
        + SNAPSHOT_NOTE
        + "\n"
        + (
            "各时限／震级分别评价；不合并扩大独立样本。相对空间分配不是绝对发震概率。\n"
            "同一报警预算，实际完整格面积略有差异；粗化保留既定大区总质量，不是空间置乱或迁移检验。"
        )
    )


def _load(output_root: Path) -> tuple[Any, ...]:
    loaded = _shared._load(output_root)
    summary = loaded[0]
    if set(summary["model_ids"]) != set(MODEL_LABELS):
        raise ValueError("S2-A requires the six frozen geometry models and four references")
    if summary.get("scientific_role") != (
        "current_static_geometry_retrospective_development_not_historical_prospective"
    ):
        raise ValueError("S2-A modern static snapshot role is missing or changed")
    return loaded


def _replay_data(*args: Any) -> dict[str, Any]:
    data = _shared._replay_data(*args)
    data["model_labels"] = {model: MODEL_LABELS[model] for model in data["summary"]["model_ids"]}
    data["default_reference"] = REFERENCE
    return data


def _static_figures(summary: dict[str, Any], root: Path) -> None:
    _shared._style()
    curves = {_shared._curve_key(row): row for row in summary["curves"]}
    models = list(MODEL_LABELS)
    main = [curves[(model, 30, "M5_6", 0.0, 600000.0)] for model in models]
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    values = [
        np.nan if row["anchor_recall"] is None else row["anchor_recall"] * 100 for row in main
    ]
    colors = [
        "#24465e"
        if model == REFERENCE
        else "#0b8c82"
        if "SIMPLE" in model
        else "#81579b"
        if "TRACE" in model
        else "#91a3b1"
        for model in models
    ]
    ax.barh(np.arange(len(models)), np.nan_to_num(values), height=0.65, color=colors, zorder=3)
    upper = max(5.0, max((v for v in values if np.isfinite(v)), default=0.0))
    for index, (row, value) in enumerate(zip(main, values, strict=True)):
        label = (
            "无可评价锚点"
            if not np.isfinite(value)
            else f"{value:.1f}%  ·  {row['anchor_hits']:g}/{row['anchor_total']:g}"
        )
        ax.text((value if np.isfinite(value) else 0) + upper * 0.015, index, label, va="center")
    ax.set_yticks(np.arange(len(models)), [MODEL_LABELS[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, upper * 1.35)
    ax.set_xlabel("独立首震锚点严格区域召回（%）")
    ax.grid(axis="x", color="#e6edf1", zorder=0)
    ax.set_title("加入断层位置，能否在同样报警预算下多找到地震区域？", loc="left", pad=30)
    fig.text(
        0.31, 0.90, "30 天 · Ms 5–6 · 60 万 km² · 严格 0 km；主参考为目录多尺度模型", fontsize=10
    )
    fig.text(0.035, 0.035, _footnote(summary), fontsize=9, color="#536775", linespacing=1.5)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.88, bottom=0.23)
    _shared._save_figure(fig, root, STATIC_STEMS[0])

    horizons = summary["horizons_days"]
    bands = summary["magnitude_bins"]
    fig, axes = plt.subplots(len(bands), len(horizons), figsize=(20, 9.5), squeeze=False)
    lines = (
        (REFERENCE, "#24465e", "-"),
        ("S2A_SIMPLE_CATALOG_MIX", "#0b8c82", "-"),
        ("S2A_SIMPLE_COARSE_MIX", "#0b8c82", "--"),
        ("S2A_TRACE_CATALOG_MIX", "#81579b", "-"),
        ("S2A_TRACE_COARSE_MIX", "#81579b", "--"),
    )
    for row_index, band in enumerate(bands):
        for col_index, horizon in enumerate(horizons):
            ax = axes[row_index, col_index]
            for model, color, style in lines:
                selected = [curves[(model, horizon, band, 0.0, budget)] for budget in AREA_BUDGETS]
                recall = [
                    np.nan if r["anchor_recall"] is None else r["anchor_recall"] * 100
                    for r in selected
                ]
                ax.plot(
                    np.asarray(AREA_BUDGETS) / 10000,
                    recall,
                    color=color,
                    linestyle=style,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.8,
                    label=MODEL_LABELS[model],
                )
            total = curves[(REFERENCE, horizon, band, 0.0, 600000.0)]["anchor_total"]
            ax.set_title(
                f"{_shared.BAND_LABELS[band]}｜{horizon} 天\n独立锚点 N={total:g}", fontsize=12
            )
            ax.set_xticks([30, 60, 96])
            ax.set_ylim(0, 105)
            ax.grid(color="#e6edf1", linewidth=0.6)
            if row_index == len(bands) - 1:
                ax.set_xlabel("报警面积预算（万 km²）")
            if col_index == 0:
                ax.set_ylabel("严格区域召回（%）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "断层增量是否只在特定时限、震级或报警面积下出现？", x=0.055, ha="left", y=0.995, fontsize=18
    )
    fig.text(0.055, 0.025, _footnote(summary), fontsize=9, color="#536775", linespacing=1.4)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.84, bottom=0.20, hspace=0.44, wspace=0.18)
    _shared._save_figure(fig, root, STATIC_STEMS[1])

    lookup = {
        (r["candidate_model_id"], r["reference_model_id"], r["horizon_days"], r["magnitude_bin"]): r
        for r in summary["pairings"]
        if r["hit_tolerance_km"] == 0 and r["area_budget_km2"] == 600000
    }
    matrices = [
        np.asarray(
            [[lookup[(a, b, h, band)]["net_hits"] for h in horizons] for a, b, _ in PAIR_AXES],
            dtype=float,
        )
        for band in bands
    ]
    limit = max(1.0, *(float(np.max(np.abs(matrix))) for matrix in matrices))
    fig, axes = plt.subplots(1, len(bands), figsize=(16, 6.9), squeeze=False)
    for band, matrix, ax in zip(bands, matrices, axes[0], strict=True):
        mesh = ax.imshow(
            matrix,
            cmap="BrBG",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            aspect="auto",
        )
        for i, (a, b, _) in enumerate(PAIR_AXES):
            for j, horizon in enumerate(horizons):
                pairing = lookup[(a, b, horizon, band)]
                ax.text(
                    j,
                    i,
                    f"{int(matrix[i, j]):+d}\n+{pairing['gained']:g}/−{pairing['lost']:g}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if abs(matrix[i, j]) > limit * 0.65 else "#243746",
                )
        ax.set_xticks(
            np.arange(len(horizons)),
            [
                f"{h} 天\nN={curves[(REFERENCE, h, band, 0.0, 600000.0)]['anchor_total']:g}"
                for h in horizons
            ],
        )
        ax.set_yticks(np.arange(len(PAIR_AXES)), [label for _, _, label in PAIR_AXES])
        ax.set_title(_shared.BAND_LABELS[band], pad=13)
    fig.colorbar(mesh, ax=list(axes[0]), fraction=0.024, pad=0.025, label="净增加的独立锚点命中数")
    fig.suptitle("断层的补充信息：精细位置，还是大区分布？", x=0.05, ha="left", y=0.99, fontsize=18)
    fig.text(
        0.05,
        0.90,
        "60 万 km² · 严格 0 km；大字为净命中，下行为新增／丢失。不同列样本数不同，不相加。",
        fontsize=10,
    )
    fig.text(0.05, 0.035, _footnote(summary), fontsize=9, color="#536775", linespacing=1.5)
    # Explicit positions leave room for long Chinese row labels and the color bar.
    axes[0, 0].set_position([0.17, 0.29, 0.29, 0.54])
    axes[0, 1].set_position([0.64, 0.29, 0.29, 0.54])
    fig.axes[-1].set_position([0.95, 0.29, 0.012, 0.54])
    _shared._save_figure(fig, root, STATIC_STEMS[2])


def _select_cases(data: dict[str, Any]) -> dict[str, Any]:
    """Choose one mixed candidate, then show its gains and failures honestly."""
    main = {
        row["model_id"]: row
        for row in data["summary"]["curves"]
        if row["horizon_days"] == 30
        and row["magnitude_bin"] == "M5_6"
        and row["area_budget_km2"] == 600000
        and row["hit_tolerance_km"] == 0
    }
    candidate = max(
        MIX_MODELS,
        key=lambda model: -1.0
        if main[model]["anchor_recall"] is None
        else main[model]["anchor_recall"],
    )
    bit = 1 << AREA_BUDGETS.index(600000.0)
    rows = []
    for index, issue in enumerate(data["issues"]):
        if issue["horizon_days"] != 30 or issue["magnitude_bin"] != "M5_6":
            continue
        anchors = [e for e in issue["events"] if e["anchor"]]
        if not anchors:
            continue
        hits = [
            (bool(e["hits"][candidate]["0"] & bit), bool(e["hits"][REFERENCE]["0"] & bit))
            for e in anchors
        ]
        gain = sum(a and not b for a, b in hits)
        loss = sum(b and not a for a, b in hits)
        rows.append(
            {
                "issue_index": index,
                "issue_time_us": issue["issue_us"],
                "fold_id": issue["fold"],
                "gained": gain,
                "lost": loss,
                "net_hits": gain - loss,
                "common_missed": sum(not a and not b for a, b in hits),
                "anchor_total": len(anchors),
            }
        )
    cases = []
    for kind, sign, field in (("gain", 1, "gained"), ("failure", -1, "lost")):
        eligible = [row for row in rows if sign * row["net_hits"] > 0]
        if eligible:
            selected = min(
                eligible,
                key=lambda r: (-sign * r["net_hits"], -r[field], r["issue_time_us"], r["fold_id"]),
            )
            label = "净新增命中较多一期" if sign == 1 else "净损失命中较多一期"
            rule = "signed_net_then_directional_count_then_date_fold"
        elif rows:
            used = {row["issue_index"] for row in cases}
            pools = [
                (r, "有新增但非净改善" if sign == 1 else "有丢失但非净损失", field)
                for r in rows
                if r[field] > 0
            ]
            pools += [(r, "共同漏报", "common_missed") for r in rows if r["common_missed"] > 0]
            pools += [(r, "非空期", None) for r in rows]
            distinct = [item for item in pools if item[0]["issue_index"] not in used]
            options = distinct or pools
            priority = {field: 0, "common_missed": 1, None: 2}
            selected, fallback_label, count = min(
                options,
                key=lambda item: (
                    priority[item[2]],
                    -item[0][item[2]] if item[2] else 0,
                    item[0]["issue_time_us"],
                    item[0]["fold_id"],
                ),
            )
            label = (
                ("无净新增命中期" if sign == 1 else "无净损失命中期") + "；回退" + fallback_label
            )
            if selected["issue_index"] in used:
                label += "（无其他非空期）"
            rule = "no_signed_net_prefer_distinct_then_direction_common_miss_nonempty"
        else:
            continue
        cases.append(
            {**selected, "kind": kind, "label": label, "selection": rule, "highlight": field}
        )
    return {
        "status": "available" if cases else "no_main_anchor_events",
        "candidate_model_id": candidate,
        "reference_model_id": REFERENCE,
        "candidate_selection": "highest_development_main_anchor_recall_among_four_frozen_mixtures_then_fixed_MODEL_LABELS_order",
        "scope": "development_illustration_not_independent_optimality_or_adoption_evidence",
        "area_budget_km2": 600000,
        "horizon_days": 30,
        "magnitude_bin": "M5_6",
        "hit_tolerance_km": 0,
        "cases": cases,
    }


def _case_figures(data: dict[str, Any], root: Path) -> dict[str, Any]:
    selection = _select_cases(data)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)
    patches, all_points = [], []
    for cell in data["grid"]:
        vertices, codes = [], []
        for polygon in cell["polygons"]:
            for ring in polygon:
                vertices.extend(ring)
                codes.extend(
                    [PlotPath.MOVETO, *([PlotPath.LINETO] * (len(ring) - 2)), PlotPath.CLOSEPOLY]
                )
        all_points.extend(vertices)
        patches.append(PathPatch(PlotPath(np.asarray(vertices), codes)))
    points_array = np.asarray(all_points)
    lower, upper = points_array.min(axis=0), points_array.max(axis=0)
    if upper[1] - lower[1] < (upper[0] - lower[0]) * 0.45:
        middle = (upper[1] + lower[1]) / 2
        half = (upper[0] - lower[0]) * 0.225
        lower[1], upper[1] = middle - half, middle + half
    bit = 1 << AREA_BUDGETS.index(600000.0)
    for i in range(2):
        if i >= len(selection["cases"]):
            for ax in axes[i]:
                ax.text(0.5, 0.5, "无可展示的主任务非空震例", transform=ax.transAxes, ha="center")
                ax.set_axis_off()
            continue
        case = selection["cases"][i]
        issue = data["issues"][case["issue_index"]]
        events = [event for event in issue["events"] if event["anchor"]]
        for j, model in enumerate((selection["candidate_model_id"], REFERENCE)):
            ax = axes[i, j]
            plan = data["plans"][issue["forecasts"][model]]
            area = next(row for row in plan["areas"] if row[0] == 600000)
            ax.add_collection(PatchCollection(patches, facecolor="#e8eef2", edgecolor="none"))
            ax.add_collection(
                PatchCollection(
                    [patches[k] for k in plan["order"][: area[2]]],
                    facecolor="#0b8c82" if j == 0 else "#426c91",
                    edgecolor="none",
                )
            )
            hits = 0
            for event in events:
                hit = bool(event["hits"][model]["0"] & bit)
                hits += hit
                ax.scatter(
                    event["lon"],
                    event["lat"],
                    marker="o" if hit else "x",
                    color="#193a4c" if hit else "#bd3544",
                    s=44,
                    zorder=4,
                )
                left = bool(event["hits"][selection["candidate_model_id"]]["0"] & bit)
                right = bool(event["hits"][REFERENCE]["0"] & bit)
                if (left and not right) if case["kind"] == "gain" else (right and not left):
                    ax.scatter(
                        event["lon"],
                        event["lat"],
                        marker="o",
                        s=135,
                        facecolors="none",
                        edgecolors="#913f98" if i == 0 else "#dd7d24",
                        linewidths=1.8,
                        zorder=5,
                    )
            ax.set_xlim(lower[0] - 0.1, upper[0] + 0.1)
            ax.set_ylim(lower[1] - 0.1, upper[1] + 0.1)
            ax.set_aspect(1 / np.cos(np.deg2rad(points_array[:, 1].mean())))
            ax.set_xlabel("经度（°E）")
            ax.set_ylabel("纬度（°N）")
            date = (
                pd.Timestamp(issue["issue_us"], unit="us", tz="UTC")
                .tz_convert("Asia/Shanghai")
                .strftime("%Y-%m-%d")
            )
            ax.set_title(
                f"{case['label']}｜{date}（北京时间）\n{MODEL_LABELS[model]}：{hits}/{len(events)}；实际 {area[1] / 10000:.2f} 万 km²\n新增 {case['gained']}，丢失 {case['lost']}，净 {case['net_hits']:+d}",
                loc="left",
                fontsize=10,
                pad=10,
            )
    fig.suptitle("断层增量的正反震例（仅本机开发展示）", x=0.055, ha="left", fontsize=18)
    fig.text(
        0.055,
        0.925,
        "同一期左为选定混合模型，右为目录多尺度主参考；30 天 · Ms 5–6 · 60 万 km² · 严格 0 km。",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.025,
        "候选仅按四个混合模型的开发主任务召回最大选作展示；平分按固定模型顺序。\n优先展示净新增／净损失最多一期；没有正／负例时明确标注回退，不制造成功或失败。\n● 命中，× 漏报；紫环为新增，橙环为丢失。命中沿用保存评分，不由绘图重新计算。\n"
        + _footnote(data["summary"]),
        fontsize=9,
        color="#536775",
        linespacing=1.4,
    )
    fig.subplots_adjust(left=0.055, right=0.98, top=0.855, bottom=0.25, hspace=0.55, wspace=0.20)
    _shared._save_figure(fig, root, CASE_STEM)
    return selection


def _html_template() -> str:
    page = _shared.HTML_TEMPLATE
    page = page.replace(
        "SeismoFlux｜C2B 开发期离线回放（仅本机）", "SeismoFlux｜S2-A 断层增量离线回放（仅本机）"
    )
    page = page.replace("C2B：目录数据与位置模型的历史比较", "S2-A：断层位置能否补充地震目录？")
    start = page.index('<p class="note">完整工作线：')
    end = page.index("</p>", start) + 4
    page = (
        page[:start]
        + '<p class="note">'
        + SNAPSHOT_NOTE
        + " 目录沿用用户提供的 Ms 数值；主参考为目录多尺度模型。分别加入简化断层或详细迹线，并以保留大区权重的粗图对照。模型输出是相对空间分配，不是绝对发震概率。此页只回放保存产物，不训练、不评分。</p>"
        + page[end:]
    )
    page = page.replace(
        'candidate.value=S.model_ids.includes("C2B_D0_K75")?"C2B_D0_K75":S.model_ids[0];',
        'candidate.value=D.default_candidate||"S2A_SIMPLE_CATALOG_MIX";',
    )
    page = page.replace(
        'reference.value=S.model_ids.includes("C0_L3_B0_R30_CAUSAL")?"C0_L3_B0_R30_CAUSAL":S.model_ids[0];',
        "reference.value=D.default_reference;",
    )
    page = page.replace(
        "最终科学价值需结合主锚点、各折、各时限及负结果复审。",
        "需结合精细／粗化对照、各折、各时限及负结果复审。",
    )
    page = page.replace("不能据此宣称模型已通过采用门控", "不能替代独立样本的检验")
    page = page.replace(
        "PNG/SVG 为不含逐事件内容的聚合图。",
        "只有前三张聚合图可公开；震例 PNG/SVG 和 HTML 均仅本机。",
    )
    return page


HTML_TEMPLATE = _html_template()


def render(output_root: Path, render_root: Path | None = None) -> Path:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = output_root.resolve()
    destination = render_root.resolve() if render_root is not None else output / "rendered"
    loaded = _load(output)
    data = _replay_data(*loaded)
    destination.mkdir(parents=True, exist_ok=True)
    _static_figures(loaded[0], destination)
    selection = _case_figures(data, destination)
    data["default_candidate"] = selection["candidate_model_id"]
    data["case_selection"] = selection
    encoded = base64.b64encode((destination / f"{STATIC_STEMS[0]}.png").read_bytes()).decode(
        "ascii"
    )
    html_path = destination / HTML_NAME
    html_path.write_text(
        HTML_TEMPLATE.replace("__MAIN_IMAGE__", encoded).replace(
            "__DATA__", _json_for_script(data)
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source_score_manifest_sha256": _sha256(output / "score_phase/score_manifest.json"),
        "scientific_role": loaded[0]["scientific_role"],
        "static_snapshot_caveat": SNAPSHOT_NOTE,
        "reference_model_id": REFERENCE,
        "timestamp_unit": "us",
        "empty_exposures_retained": True,
        "network_resources": [],
        "synthetic_fixture": bool(loaded[0].get("synthetic_fixture")),
        "case_selection": selection,
        "artifacts": [
            {
                "path": name,
                "sha256": _sha256(destination / name),
                "audience": "local_only_contains_event_data"
                if name == HTML_NAME or name.startswith(CASE_STEM)
                else "public_aggregate_no_event_coordinates",
            }
            for name in FILENAMES
        ],
    }
    (destination / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="S2-A 聚合图与本机离线历史回放；不运行评分")
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/multitask_s2/s2a_fault_geometry_v1")
    )
    parser.add_argument("--render-root", type=Path)
    args = parser.parse_args()
    print(render(args.output_root, args.render_root))


if __name__ == "__main__":
    main()
