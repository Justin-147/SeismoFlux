# ruff: noqa: E501, RUF001
"""S2-C aggregate figures and local-only offline replay of completed scores.

No training or scoring is performed. The existing C2B serializer and offline
controls are reused read-only; same-support rate controls remain explicit.
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
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath


def _shared_renderer() -> Any:
    path = Path(__file__).with_name("render_multitask_s1_c2b.py")
    spec = importlib.util.spec_from_file_location("seismoflux_s2c_replay_helpers", path)
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
    "C0_L0_UNIFORM": "全国均匀参考",
    "S2C_UNIT_ONLY": "变形区支撑｜纯层",
    "S2C_UNIT_CATALOG_MIX": "变形区支撑＋目录",
    "S2C_STRAIN_ONLY": "应变强度｜纯层",
    "S2C_STRAIN_CATALOG_MIX": "应变强度＋目录",
}
MIX_MODELS = tuple(model for model in MODEL_LABELS if model.endswith("_CATALOG_MIX"))
PAIR_AXES = (
    ("S2C_STRAIN_CATALOG_MIX", "S2C_UNIT_CATALOG_MIX", "强度＋目录 − 支撑＋目录"),
    ("S2C_STRAIN_ONLY", "S2C_UNIT_ONLY", "纯强度 − 纯支撑"),
    ("S2C_UNIT_CATALOG_MIX", REFERENCE, "支撑＋目录 − 目录"),
    ("S2C_STRAIN_CATALOG_MIX", REFERENCE, "强度＋目录 − 目录"),
    ("S2C_UNIT_ONLY", "C0_L0_UNIFORM", "纯支撑 − 全国均匀"),
    ("S2C_STRAIN_ONLY", "C0_L0_UNIFORM", "纯强度 − 全国均匀"),
)
STATIC_STEMS = (
    "01_main_anchor_recall",
    "02_strain_area_curves",
    "03_strength_support_net_hits",
)
CASE_STEM = "04_selected_gain_and_failure_local_only"
HTML_NAME = "seismoflux_s2c_strain_replay.html"
FILENAMES = (
    *(f"{stem}.{suffix}" for stem in (*STATIC_STEMS, CASE_STEM) for suffix in ("png", "svg")),
    HTML_NAME,
)
SNAPSHOT_NOTE = (
    "GSRM v2.1 标注发布于 2014 年：全四折为历史回看；2015—2019 仍是开发比较，不是独立测试。"
)
ATTRIBUTION = "GSRM / GEM Foundation 2014 · Kreemer, Blewitt & Klein (2014) · CC-BY-NC-SA 3.0"
ROLE = "GSRM_2014_static_retrospective_development_with_post_release_era_slice_not_independent_test"


def _footnote(summary: dict[str, Any]) -> str:
    prefix = "合成测试示例，不代表科学结果。\n" if summary.get("synthetic_fixture") else ""
    return (
        prefix + SNAPSHOT_NOTE + "\n"
        "同一报警面积预算；分母按各评价时限、震级及时间分列，不跨列累加。零应变不等于无地震风险。\n"
        "相对空间分配不是绝对发震概率；当前文件未证明与 2014 历史字节完全相同。\n" + ATTRIBUTION
    )


def _load(output_root: Path) -> tuple[Any, ...]:
    loaded = _shared._load(output_root)
    summary = loaded[0]
    if set(summary["model_ids"]) != set(MODEL_LABELS):
        raise ValueError("S2-C requires four strain models and four references")
    if summary.get("scientific_role") != ROLE:
        raise ValueError("S2-C snapshot role changed")
    if summary["horizons_days"] != [7, 30, 90, 180, 365] or set(summary["magnitude_bins"]) != {
        "M5_6",
        "M6_plus",
    }:
        raise ValueError("S2-C frozen axes changed")
    expected = {
        (m, h, b, t, a)
        for m in MODEL_LABELS
        for h in summary["horizons_days"]
        for b in summary["magnitude_bins"]
        for t in (0.0, 70.0)
        for a in AREA_BUDGETS
    }
    for part in (summary, summary["post_release_development"]):
        keys = [_shared._curve_key(row) for row in part["curves"]]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("Incomplete evaluation slice")
        pairs = {(r["candidate_model_id"], r["reference_model_id"]) for r in part["pairings"]}
        if pairs != {(a, b) for a, b, _ in PAIR_AXES}:
            raise ValueError("Six registered comparisons required")
    if not summary.get("synthetic_fixture") and (
        summary["main_anchor_count"] != 147
        or summary["post_release_development"]["main_anchor_count"] != 32
    ):
        raise ValueError("S2-C main anchor denominator mismatch")
    return loaded


def _replay_data(*args: Any) -> dict[str, Any]:
    data = _shared._replay_data(*args)
    data["model_labels"] = {model: MODEL_LABELS[model] for model in data["summary"]["model_ids"]}
    data["default_reference"] = REFERENCE
    return data


def _static_figures(summary: dict[str, Any], root: Path) -> None:
    _shared._style()
    parts = (summary, summary["post_release_development"])
    titles = ("2000—2019｜全四折历史回看", "2015—2019｜产品发布后开发")
    colors = [
        "#24465e",
        "#91a3b1",
        "#7892a3",
        "#bec8cc",
        "#a79270",
        "#bb914e",
        "#42a497",
        "#087f77",
    ]
    models = list(MODEL_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, part, title in zip(axes, parts, titles, strict=True):
        lookup = {_shared._curve_key(r): r for r in part["curves"]}
        rows = [lookup[(m, 30, "M5_6", 0.0, 600000.0)] for m in models]
        values = [np.nan if r["anchor_recall"] is None else 100 * r["anchor_recall"] for r in rows]
        ax.barh(range(len(models)), values, color=colors, height=0.65)
        for i, (r, v) in enumerate(zip(rows, values, strict=True)):
            label = (
                "无可评价锚点"
                if not np.isfinite(v)
                else f"{v:.1f}% · {r['anchor_hits']:g}/{r['anchor_total']:g}"
            )
            ax.text((v if np.isfinite(v) else 0) + 1, i, label, va="center", fontsize=9)
        ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 125)
        ax.set_xlabel("独立首震锚点严格区域召回（%）")
        ax.set_title(title, loc="left", pad=17)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("应变信息是否减少漏报？｜30 天 · Ms 5–6 · 60 万 km² · 严格 0 km", fontsize=17)
    fig.text(0.035, 0.025, _footnote(summary), fontsize=9, color="#536775", linespacing=1.5)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.87, bottom=0.24, wspace=0.65)
    _shared._save_figure(fig, root, STATIC_STEMS[0])

    fig, axes = plt.subplots(2, 5, figsize=(20, 11))
    for i, (part, title) in enumerate(zip(parts, titles, strict=True)):
        lookup = {_shared._curve_key(r): r for r in part["curves"]}
        for j, h in enumerate(summary["horizons_days"]):
            ax = axes[i, j]
            for m, color in zip(models, colors, strict=True):
                rows = [lookup[(m, h, "M5_6", 0.0, a)] for a in AREA_BUDGETS]
                ax.plot(
                    np.array(AREA_BUDGETS) / 10000,
                    [
                        np.nan if r["anchor_recall"] is None else 100 * r["anchor_recall"]
                        for r in rows
                    ],
                    color=color,
                    marker="o",
                    markersize=3,
                    linestyle="--" if m.endswith("_ONLY") else "-",
                    label=MODEL_LABELS[m],
                )
            total = lookup[(REFERENCE, h, "M5_6", 0.0, 600000.0)]["anchor_total"]
            ax.set_title(f"{title}\n{h} 天 · Ms 5–6 · N={total:g}", fontsize=10)
            ax.set_ylim(0, 105)
            ax.set_xticks([30, 60, 96])
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel("严格区域召回（%）")
            if i == 1:
                ax.set_xlabel("报警预算（万 km²）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("不同时间窗口与报警面积：保留所有模型的正负表现", fontsize=17)
    fig.text(0.04, 0.025, _footnote(summary), fontsize=9, color="#536775", linespacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.20, hspace=0.4, wspace=0.22)
    _shared._save_figure(fig, root, STATIC_STEMS[1])

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for ax, part, title in zip(axes, parts, titles, strict=True):
        rows = [
            next(
                r
                for r in part["pairings"]
                if r["candidate_model_id"] == a
                and r["reference_model_id"] == b
                and r["horizon_days"] == 30
                and r["magnitude_bin"] == "M5_6"
                and r["hit_tolerance_km"] == 0
                and r["area_budget_km2"] == 600000
            )
            for a, b, _ in PAIR_AXES
        ]
        y = np.arange(6)
        ax.barh(y, [r["gained"] for r in rows], color="#087f77", label="新增命中")
        ax.barh(y, [-r["lost"] for r in rows], color="#ba535b", label="丢失命中")
        for i, r in enumerate(rows):
            ax.text(
                r["gained"] + 0.25,
                i,
                f"净 {r['net_hits']:+g}（+{r['gained']:g}/−{r['lost']:g}）",
                va="center",
                fontsize=9,
            )
        limit = max(1, max(max(r["gained"], r["lost"]) for r in rows))
        ax.set_xlim(-limit * 1.2, limit * 2.9)
        ax.axvline(0, color="#647887", lw=0.8)
        ax.set_yticks(y, [label for _, _, label in PAIR_AXES], fontsize=9)
        ax.invert_yaxis()
        total = next(
            r["anchor_total"]
            for r in part["curves"]
            if r["model_id"] == REFERENCE
            and r["horizon_days"] == 30
            and r["magnitude_bin"] == "M5_6"
            and r["area_budget_km2"] == 600000
            and r["hit_tolerance_km"] == 0
        )
        ax.set_title(f"{title} · N={total:g}", loc="left")
        ax.set_xlabel("独立首震锚点命中数")
        ax.legend(frameon=False)
    fig.suptitle("六组预定对照：收益来自变形区的位置，还是应变强弱？", fontsize=17)
    fig.text(
        0.035,
        0.16,
        "30 天 · Ms 5–6 · 60 万 km² · 严格 0 km；六行是相关对照，不是六个独立实验。",
        fontsize=10,
    )
    fig.text(0.035, 0.025, _footnote(summary), fontsize=9, color="#536775", linespacing=1.4)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.87, bottom=0.25, wspace=0.7)
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
        "candidate_selection": "highest_development_main_anchor_recall_among_two_frozen_mixtures_then_fixed_MODEL_LABELS_order",
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
    fig.suptitle("应变层的正反震例（仅本机开发展示）", x=0.055, ha="left", fontsize=18)
    fig.text(
        0.055,
        0.925,
        "同一期左为选定混合模型，右为目录多尺度主参考；30 天 · Ms 5–6 · 60 万 km² · 严格 0 km。",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.025,
        "候选仅按两个冻结混合模型的开发主任务召回最大选作展示；平分按固定模型顺序。\n优先展示净新增／净损失最多一期；没有正／负例时明确标注回退，不制造成功或失败。\n● 命中，× 漏报；紫环为新增，橙环为丢失。命中沿用保存评分，不由绘图重新计算。\n"
        + _footnote(data["summary"]),
        fontsize=9,
        color="#536775",
        linespacing=1.4,
    )
    fig.subplots_adjust(left=0.055, right=0.98, top=0.855, bottom=0.25, hspace=0.55, wspace=0.20)
    _shared._save_figure(fig, root, CASE_STEM)
    return selection


def _html_template() -> str:
    page = _shared.HTML_TEMPLATE.replace(
        "SeismoFlux｜C2B 开发期离线回放（仅本机）", "SeismoFlux｜S2-C 应变离线回放（仅本机）"
    )
    page = page.replace("C2B：目录数据与位置模型的历史比较", "S2-C：应变信息能否补充地震目录？")
    start = page.index('<p class="note">完整工作线：')
    end = page.index("</p>", start) + 4
    page = (
        page[:start]
        + '<p class="note">'
        + SNAPSHOT_NOTE
        + " 变形区支撑与应变强弱分别对照；零质量不是缺样本，不删地震。"
        + ATTRIBUTION
        + "</p>"
        + page[end:]
    )
    page = page.replace(
        '<div class="controls"><label>时限',
        '<div class="controls"><label>评价分列 <select id="slice"><option value="all">全四折历史回看</option><option value="post">2015—2019 开发</option></select></label><label>时限',
        1,
    )
    page = page.replace(
        "S=D.summary,$=id=>document.getElementById(id);",
        'S=D.summary,$=id=>document.getElementById(id);\nconst slice=$("slice"); slice.value="all"; const selectedSummary=()=>slice.value==="post"?S.post_release_development:S;',
    )
    page = page.replace(
        "const row=S.curves.find", "const row=selectedSummary().curves.find"
    ).replace("p=S.pairings.find", "p=selectedSummary().pairings.find")
    page = page.replace(
        "i.horizon_days===+horizon.value&&i.magnitude_bin===band.value",
        'i.horizon_days===+horizon.value&&i.magnitude_bin===band.value&&(slice.value==="all"||S.post_release_development.fold_ids.includes(i.fold))',
    )
    page = page.replace("[horizon,band].forEach", "[horizon,band,slice].forEach")
    page = page.replace(
        "function logLabel(row){",
        'function logLabel(row){if(row.log_density_status==="negative_infinity_from_zero_mass")return "−∞（真实零质量，不是无事件）";',
    )
    page = page.replace(
        'candidate.value=S.model_ids.includes("C2B_D0_K75")?"C2B_D0_K75":S.model_ids[0];',
        'candidate.value=D.default_candidate||"S2C_STRAIN_CATALOG_MIX";',
    )
    page = page.replace(
        'reference.value=S.model_ids.includes("C0_L3_B0_R30_CAUSAL")?"C0_L3_B0_R30_CAUSAL":S.model_ids[0];',
        "reference.value=D.default_reference;",
    )
    page = page.replace(
        "PNG/SVG 为不含逐事件内容的聚合图。",
        "仅前三张聚合图可公开；案例图及 HTML 仅本机。" + ATTRIBUTION,
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
        "source_attribution": ATTRIBUTION,
        "strain_derived_artifact_license": "CC-BY-NC-SA-3.0",
        "evaluation_slices": [
            "all_four_development_folds_retrospective",
            "post_release_2015_2019_development",
        ],
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
    parser = argparse.ArgumentParser(description="S2-C 聚合图与本机离线历史回放；不运行评分")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir",
        "--output-root",
        dest="run_dir",
        type=Path,
        default=Path("outputs/multitask_s2/s2c_strain_v1"),
    )
    parser.add_argument("--output-dir", "--render-root", dest="output_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir if args.run_dir.is_absolute() else args.project_root / args.run_dir
    destination = args.output_dir
    if destination is not None and not destination.is_absolute():
        destination = args.project_root / destination
    print(render(run, destination))


if __name__ == "__main__":
    main()
