# ruff: noqa: E501, RUF001
"""Aggregate scientific figures and a local-only, self-contained C2B replay.

Read only hash-verified score artifacts. This module never trains, scores,
opens catalogs or holdouts, or changes saved forecasts. Maps use saved clipped
grid geometry; all hit symbols come from saved event_results.hit, not the map.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
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
from pyproj import Transformer
from shapely import from_wkt
from shapely.ops import transform

from seismoflux.background.grid import EQUAL_AREA_CRS

AREA_BUDGETS = (300000.0, 450000.0, 600000.0, 750000.0, 960000.0)
AXES = ["fold_id", "horizon_days", "magnitude_bin", "issue_time_us"]
MODEL_LABELS = {
    "C0_L0_UNIFORM": "C0 L0 均匀参考",
    "C0_L1_REGIONAL_CONSTANT": "C0 L1 分区活动率",
    "C0_L2_KDE_CAUSAL": "C0 L2 因果 KDE",
    "C0_L2_KDE75_LEGACY": "C0 L2 固定 75 km",
    "C0_L3_B0_R30_CAUSAL": "C0 L3 长期＋近期",
    "C2B_D0_K75": "D0｜KDE 75 km",
    "C2B_D1_K75": "D1｜KDE 75 km",
    "C2B_D2_K75": "D2｜KDE 75 km",
    "C2B_D0_R30": "D0｜长期＋近期 30 天",
    "C2B_D1_R30": "D1｜长期＋近期 30 天",
    "C2B_D0_MULTISCALE": "D0｜多空间尺度",
    "C2B_D0_AGE_WEIGHTED": "D0｜相对年龄加权",
    "C2B_D0_RIDGE_CORE": "D0｜岭组合核心",
    "C2B_D0_RIDGE_M5": "D0｜岭组合＋M5",
}
BAND_LABELS = {"M5_6": "Ms 5–6", "M6_plus": "Ms 6+"}
PALETTE = ("#264b73", "#008777", "#cb7a19", "#9b4d96", "#427eaf", "#ad3c43")
STATIC_STEMS = ("01_main_anchor_recall", "02_grouped_area_curves", "03_paired_contributions")
CASE_STEM = "04_selected_gain_and_failure_local_only"
HTML_NAME = "seismoflux_s1c2b_replay.html"
FILENAMES = (
    *(f"{stem}.{suffix}" for stem in (*STATIC_STEMS, CASE_STEM) for suffix in ("png", "svg")),
    HTML_NAME,
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "exposure_results.parquet",
    "event_results.parquet",
    "paired_anchor_results.parquet",
    "alarm_prefixes.parquet",
    "grid_geometry.parquet",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(
    output_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score = (output_root / "score_phase").resolve()
    manifest = json.loads((score / "score_manifest.json").read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    names = [record["path"] for record in records]
    if len(names) != len(set(names)) or not set(REQUIRED_ARTIFACTS).issubset(names):
        raise ValueError("C2B score manifest omits or duplicates required artifacts")
    # A finalized artifact manifest is the C2B completion gate; there is no C2A complete flag.
    for record in records:
        path = (score / record["path"]).resolve()
        if (
            not path.is_relative_to(score)
            or not path.is_file()
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Score artifact path or hash changed: {record['path']}")
    summary = json.loads((score / "summary.json").read_text(encoding="utf-8"))
    if summary.get("holdout_read") is not False or summary.get("locked_test_run") is not False:
        raise ValueError("Only sealed development results may be rendered")
    if summary.get("new_independent_test_evidence") is not False:
        raise ValueError("C2B replay is not new independent test evidence")
    models = summary["model_ids"]
    if len(set(models)) != len(models) or not models:
        raise ValueError("Invalid model axis")
    keys = [
        (
            row["model_id"],
            row["horizon_days"],
            row["magnitude_bin"],
            row["hit_tolerance_km"],
            row["area_budget_km2"],
        )
        for row in summary["curves"]
    ]
    tolerances = {key[3] for key in keys}
    expected = {
        (model, horizon, band, tolerance, area)
        for model in models
        for horizon in summary["horizons_days"]
        for band in summary["magnitude_bins"]
        for tolerance in tolerances
        for area in AREA_BUDGETS
    }
    if (
        0 not in tolerances
        or not tolerances.issubset({0, 70})
        or len(keys) != len(set(keys))
        or set(keys) != expected
    ):
        raise ValueError("Incomplete or duplicate model/horizon/band/tolerance/budget curve axis")
    return (
        summary,
        pd.read_parquet(score / "event_results.parquet"),
        pd.read_parquet(score / "alarm_prefixes.parquet"),
        pd.read_parquet(score / "grid_geometry.parquet"),
        pd.read_parquet(score / "exposure_results.parquet"),
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#acb8c2",
            "text.color": "#253642",
            "axes.labelcolor": "#253642",
            "xtick.color": "#253642",
            "ytick.color": "#253642",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: Any, root: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(root / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def _footnote(summary: dict[str, Any]) -> str:
    prefix = "合成测试示例，不代表科学结果。\n" if summary.get("synthetic_fixture") else ""
    return prefix + (
        "开发期位置模型比较；不是独立确认或真实未来预测。各时限、震级分别评价，不合并扩大样本量。\n"
        "同一预算下的实际完整格面积可能略有差异；网格前缀尚不等于最终有限连通危险区产品。\n"
        "D0：1970 起规范目录 Ms≥4；D1：1980 起 M3 来源 Ms≥4；D2：1950 起 M5 来源 Ms≥5。"
    )


def _panel_groups(models: list[str]) -> list[tuple[str, list[str]]]:
    groups = [
        ("既有参考模型", [model for model in models if model.startswith("C0_")]),
        (
            "数据面板对照（同一模型）",
            [
                model
                for model in models
                if model.endswith(("_K75", "_R30")) and model.startswith("C2B_")
            ],
        ),
        ("同 D0 数据的模型对照", [model for model in models if model.startswith("C2B_D0_")]),
    ]
    covered = {model for _, items in groups for model in items}
    if extra := [model for model in models if model not in covered]:
        groups.append(("其他已保存模型", extra))
    return [(name, items) for name, items in groups if items]


def _curve_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["model_id"],
        int(row["horizon_days"]),
        row["magnitude_bin"],
        float(row["hit_tolerance_km"]),
        float(row["area_budget_km2"]),
    )


def _static_figures(summary: dict[str, Any], root: Path) -> None:
    _style()
    models = summary["model_ids"]
    curves = {_curve_key(row): row for row in summary["curves"]}
    main = [curves[(model, 30, "M5_6", 0.0, 600000.0)] for model in models]
    fig, ax = plt.subplots(figsize=(12.8, max(7.8, len(models) * 0.43 + 2.0)))
    values = [
        np.nan if row["anchor_recall"] is None else row["anchor_recall"] * 100 for row in main
    ]
    positions = np.arange(len(models))
    ax.barh(
        positions,
        np.nan_to_num(values),
        height=0.66,
        color=["#7b8d9d" if model.startswith("C0_") else "#008777" for model in models],
        zorder=3,
    )
    upper = max(5, max((value for value in values if np.isfinite(value)), default=0))
    for index, (row, value) in enumerate(zip(main, values, strict=True)):
        label = (
            "不可评价（无锚点）"
            if not np.isfinite(value)
            else f"{value:.1f}%  ·  {row['anchor_hits']:g}/{row['anchor_total']:g}"
        )
        ax.text(
            (0 if not np.isfinite(value) else value) + upper * 0.015,
            index,
            label,
            va="center",
            fontsize=10,
        )
    ax.set_yticks(positions, [MODEL_LABELS.get(model, model) for model in models])
    ax.invert_yaxis()
    ax.set_xlim(0, upper * 1.38)
    ax.set_xlabel("固定首震锚点严格区域召回（%）")
    ax.grid(axis="x", color="#e5ebef", zorder=0)
    fig.suptitle(
        "主锚点：同一报警预算下，14 个模型的命中与召回"
        if len(models) == 14
        else "主锚点：全部已保存模型的命中与召回",
        x=0.05,
        ha="left",
        fontsize=17,
    )
    fig.text(
        0.05,
        0.925,
        "30 天 · Ms 5–6 · 严格 0 km · 60 万 km² 预算 · 按预定模型轴展示，不按结果排序",
        color="#526573",
    )
    fig.text(0.05, 0.035, _footnote(summary), fontsize=9, color="#526573", linespacing=1.6)
    fig.subplots_adjust(left=0.255, right=0.96, top=0.88, bottom=0.17)
    _save_figure(fig, root, STATIC_STEMS[0])

    groups = _panel_groups(models)
    fig, axes = plt.subplots(1, len(groups), figsize=(6.0 * len(groups), 7.5), squeeze=False)
    for ax, (title, panel_models) in zip(axes[0], groups, strict=True):
        for index, model in enumerate(panel_models):
            rows = [curves[(model, 30, "M5_6", 0.0, area)] for area in AREA_BUDGETS]
            y = [
                np.nan if row["anchor_recall"] is None else row["anchor_recall"] * 100
                for row in rows
            ]
            ax.plot(
                np.asarray(AREA_BUDGETS) / 10000,
                y,
                marker=("o", "s", "^", "D", "v", "P")[index % 6],
                linestyle="--" if index >= 3 else "-",
                linewidth=1.9,
                color=PALETTE[index % 6],
                label=MODEL_LABELS.get(model, model),
            )
        ax.set_title(title, loc="left", pad=76)
        ax.set_xticks(np.asarray(AREA_BUDGETS) / 10000)
        ax.set_xlabel("报警面积预算（万 km²）")
        ax.set_ylim(bottom=0)
        ax.grid(color="#e7ecf0")
        ax.legend(
            frameon=False,
            fontsize=8.5,
            loc="lower left",
            bbox_to_anchor=(0, 1.015),
            ncol=2,
            columnspacing=0.9,
            handlelength=2,
        )
    axes[0, 0].set_ylabel("固定首震锚点召回（%）")
    maximum = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(0, maximum * 1.12)
    fig.suptitle("五档面积曲线：参考、数据与模型分别比较", x=0.045, ha="left", fontsize=18)
    fig.text(
        0.045,
        0.90,
        "30 天 · Ms 5–6 · 严格 0 km；全部五档预算均保留，三面板采用同一纵轴。",
        color="#526573",
    )
    fig.text(0.045, 0.035, _footnote(summary), fontsize=9, color="#526573", linespacing=1.6)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.25, wspace=0.22)
    _save_figure(fig, root, STATIC_STEMS[1])

    pairs = [
        row
        for row in summary["pairings"]
        if row["hit_tolerance_km"] == 0
        and row["area_budget_km2"] == 600000
        and row.get("purpose") != "existing_reference_main_anchor_only"
    ]
    identities = list(
        dict.fromkeys((row["candidate_model_id"], row["reference_model_id"]) for row in pairs)
    )
    horizons, bands = summary["horizons_days"], summary["magnitude_bins"]
    lookup = {
        (
            row["candidate_model_id"],
            row["reference_model_id"],
            row["horizon_days"],
            row["magnitude_bin"],
        ): row
        for row in pairs
    }
    limit = max(
        1.0,
        max(
            (abs(row["delta_recall_pp"]) for row in pairs if row["delta_recall_pp"] is not None),
            default=1.0,
        ),
    )
    fig, axes = plt.subplots(
        1, len(bands), figsize=(19.0, max(6.5, len(identities) * 0.7 + 2.4)), squeeze=False
    )
    cmap = plt.colormaps["RdBu"].with_extremes(bad="#e7ecf0")
    if not identities:
        identities = [("无预登记配对", "")]
    for ax, band in zip(axes[0], bands, strict=True):
        values = np.full((len(identities), len(horizons)), np.nan)
        for i, (candidate, reference) in enumerate(identities):
            for j, horizon in enumerate(horizons):
                row = lookup.get((candidate, reference, horizon, band))
                if row is not None and row["delta_recall_pp"] is not None:
                    values[i, j] = row["delta_recall_pp"]
                label = (
                    "NA"
                    if not np.isfinite(values[i, j])
                    else f"{values[i, j]:+.1f} pp\n{row['net_hits']:+d}/{row['anchor_total']}"
                )
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                    if np.isfinite(values[i, j]) and abs(values[i, j]) > limit * 0.65
                    else "#253642",
                )
        im = ax.imshow(
            values, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto"
        )
        ax.set_xticks(np.arange(len(horizons)), [f"{h} 天" for h in horizons])
        ax.set_yticks(
            np.arange(len(identities)),
            [f"{MODEL_LABELS.get(c, c)}\n− {MODEL_LABELS.get(r, r)}" for c, r in identities],
            fontsize=9,
        )
        ax.set_title(BAND_LABELS.get(band, band), loc="left", pad=15)
        ax.tick_params(length=0, pad=9)
    fig.suptitle("预登记配对贡献：改善、零变化和损失全部保留", x=0.035, ha="left", fontsize=18)
    fig.text(
        0.035,
        0.92,
        "60 万 km² · 严格 0 km · 格内：召回差（百分点）／净命中数与锚点总数；NA 表示无可评价配对。",
        color="#526573",
    )
    fig.text(
        0.035,
        0.045,
        "区间在离线页按已保存结果展示：2,000 次配对非重叠时间曝光重采样（含空期），不是独立震序重采样。\n"
        + _footnote(summary),
        fontsize=9,
        color="#526573",
        linespacing=1.6,
    )
    fig.subplots_adjust(left=0.17, right=0.94, top=0.84, bottom=0.22, wspace=0.75)
    cax = fig.add_axes((0.955, 0.27, 0.012, 0.5))
    fig.colorbar(im, cax=cax, label="候选 − 参考（百分点）")
    _save_figure(fig, root, STATIC_STEMS[2])


def _geometry_polygons(geometry: Any) -> list[Any]:
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [polygon for part in geometry.geoms for polygon in _geometry_polygons(part)]
    return []


def _replay_data(
    summary: dict[str, Any],
    events: pd.DataFrame,
    alarms: pd.DataFrame,
    grid: pd.DataFrame,
    exposures: pd.DataFrame,
) -> dict[str, Any]:
    """Compact nested area prefixes, keeping all independent axis identities."""
    cells = grid.sort_values("cell_index")
    if not np.array_equal(cells.cell_index.to_numpy(), np.arange(len(cells))) or cells.empty:
        raise ValueError("Grid indices must be contiguous, unchanged and nonempty")
    models = summary["model_ids"]
    tolerances = sorted({float(row["hit_tolerance_km"]) for row in summary["curves"]})
    exposure_keys = [*AXES, "model_id", "area_budget_km2", "hit_tolerance_km"]
    if exposures.duplicated(exposure_keys).any():
        raise ValueError("Duplicate exposure identity")
    issues: dict[tuple[Any, ...], dict[str, Any]] = {}
    expected_rows = len(models) * len(AREA_BUDGETS) * len(tolerances)
    for key, group in exposures.groupby(AXES, sort=True):
        if (
            len(group) != expected_rows
            or set(group.model_id) != set(models)
            or group.event_count.nunique() != 1
            or group.anchor_total.nunique() != 1
        ):
            raise ValueError("Exposure model axis or fixed target population differs")
        issue_us = int(key[3])
        # Never reinterpret microseconds as pandas' default nanosecond integers.
        label = pd.Timestamp(issue_us, unit="us", tz="UTC").isoformat()
        issues[key] = {
            "fold": str(key[0]),
            "horizon_days": int(key[1]),
            "magnitude_bin": str(key[2]),
            "issue_us": issue_us,
            "label": label,
            "event_count": int(group.iloc[0].event_count),
            "anchor_total": int(group.iloc[0].anchor_total),
            "forecasts": {},
            "events": [],
        }
    if (
        len(issues) != summary["target_exposure_band_count"]
        or len(exposures[[*AXES[:2], AXES[3]]].drop_duplicates())
        != summary["primary_issue_horizon_count"]
    ):
        raise ValueError("Replay lost exposure axes, including empty periods")
    plans, plan_lookup = [], {}
    for key, group in alarms.groupby([*AXES, "model_id"], sort=True):
        axis, model = key[:-1], key[-1]
        if axis not in issues or model not in models:
            raise ValueError("Alarm axis leaves exposure population")
        group = group.sort_values("area_budget_km2")
        if tuple(group.area_budget_km2) != AREA_BUDGETS:
            raise ValueError("Replay requires all five area budgets")
        order = [int(value) for value in group.iloc[-1].selected_cell_indices]
        if len(set(order)) != len(order) or any(not 0 <= value < len(cells) for value in order):
            raise ValueError("Invalid alarm cell index")
        areas = []
        for row in group.itertuples(index=False):
            prefix = [int(value) for value in row.selected_cell_indices]
            if prefix != order[: len(prefix)]:
                raise ValueError("Saved alarm prefixes are not nested")
            if not 0 <= float(row.actual_area_km2) <= float(row.area_budget_km2):
                raise ValueError("Actual area leaves saved budget")
            areas.append([float(row.area_budget_km2), float(row.actual_area_km2), len(prefix)])
        plan = {"order": order, "areas": areas}
        token = json.dumps(plan, separators=(",", ":"))
        if token not in plan_lookup:
            plan_lookup[token] = len(plans)
            plans.append(plan)
        issues[axis]["forecasts"][model] = plan_lookup[token]
    if any(set(issue["forecasts"]) != set(models) for issue in issues.values()):
        raise ValueError("Replay is missing an alarm model or an empty period")
    event_keys = [*AXES, "event_id"]
    if events.duplicated([*event_keys, "model_id", "area_budget_km2", "hit_tolerance_km"]).any():
        raise ValueError("Duplicate saved event hit identity")
    event_lookup = {}
    for row in events.drop_duplicates(event_keys).itertuples(index=False):
        axis = (row.fold_id, row.horizon_days, row.magnitude_bin, row.issue_time_us)
        if axis not in issues:
            raise ValueError("Event axis leaves exposure population")
        event = {
            "id": str(row.event_id),
            "episode": str(row.episode_id),
            "anchor": bool(row.is_episode_anchor),
            "cell": int(row.cell_index),
            "lon": float(row.longitude),
            "lat": float(row.latitude),
            "hits": {},
        }
        issues[axis]["events"].append(event)
        event_lookup[(*axis, row.event_id)] = event
    masks: dict[tuple[Any, ...], tuple[int, int]] = {}
    for row in events.itertuples(index=False):
        key = (
            row.fold_id,
            row.horizon_days,
            row.magnitude_bin,
            row.issue_time_us,
            row.event_id,
            row.model_id,
            float(row.hit_tolerance_km),
        )
        bit = 1 << AREA_BUDGETS.index(float(row.area_budget_km2))
        seen, hits = masks.get(key, (0, 0))
        masks[key] = (seen | bit, hits | (bit if row.hit else 0))
    for key, (seen, hits) in masks.items():
        if seen != (1 << len(AREA_BUDGETS)) - 1:
            raise ValueError("Event hit axis omits area budgets")
        event_lookup[key[:-2]]["hits"].setdefault(key[-2], {})[str(int(key[-1]))] = hits
    for issue in issues.values():
        if (
            len(issue["events"]) != issue["event_count"]
            or sum(event["anchor"] for event in issue["events"]) != issue["anchor_total"]
        ):
            raise ValueError("Saved targets disagree with exposure counts")
        for event in issue["events"]:
            if set(event["hits"]) != set(models) or any(
                set(value) != {str(int(t)) for t in tolerances} for value in event["hits"].values()
            ):
                raise ValueError("Event hit axis omits models or tolerances")
    transformer = Transformer.from_crs(EQUAL_AREA_CRS, "EPSG:4326", always_xy=True)
    geometry = []
    for row in cells.itertuples(index=False):
        projected = from_wkt(row.clipped_geometry_wkt_equal_area_m)
        polygons = _geometry_polygons(transform(transformer.transform, projected))
        if not polygons:
            raise ValueError("Grid cell has no saved polygon")
        geometry.append(
            {
                "lon": float(row.longitude),
                "lat": float(row.latitude),
                "polygons": [
                    [
                        [list(map(float, coordinate[:2])) for coordinate in ring.coords]
                        for ring in [polygon.exterior, *polygon.interiors]
                    ]
                    for polygon in polygons
                ],
            }
        )
    return {
        "summary": summary,
        "issues": sorted(
            issues.values(),
            key=lambda row: (
                row["horizon_days"],
                row["magnitude_bin"],
                row["issue_us"],
                row["fold"],
            ),
        ),
        "plans": plans,
        "grid": geometry,
        "model_labels": {model: MODEL_LABELS.get(model, model) for model in models},
        "band_labels": BAND_LABELS,
        "area_budgets": list(AREA_BUDGETS),
        "tolerances": tolerances,
    }


def _json_for_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _select_cases(data: dict[str, Any]) -> dict[str, Any]:
    """Deterministic development-only illustration, never a model adoption rule."""
    summary = data["summary"]
    reference = "C0_L3_B0_R30_CAUSAL"
    candidates = [model for model in summary["model_ids"] if not model.startswith("C0_")]
    if not candidates or reference not in summary["model_ids"]:
        return {
            "status": "not_available",
            "reason": "C2B candidates or saved C0 L3 reference absent",
            "cases": [],
        }
    main = {
        row["model_id"]: row
        for row in summary["curves"]
        if row["horizon_days"] == 30
        and row["magnitude_bin"] == "M5_6"
        and row["area_budget_km2"] == 600000
        and row["hit_tolerance_km"] == 0
    }
    candidate = max(
        candidates,
        key=lambda model: -1.0
        if main[model]["anchor_recall"] is None
        else main[model]["anchor_recall"],
    )
    bit = 1 << AREA_BUDGETS.index(600000.0)
    rows = []
    for index, issue in enumerate(data["issues"]):
        if issue["horizon_days"] != 30 or issue["magnitude_bin"] != "M5_6":
            continue
        events = [event for event in issue["events"] if event["anchor"]]
        if not events:
            continue
        hits = [
            (bool(event["hits"][candidate]["0"] & bit), bool(event["hits"][reference]["0"] & bit))
            for event in events
        ]
        rows.append(
            {
                "issue_index": index,
                "fold_id": issue["fold"],
                "issue_time_us": issue["issue_us"],
                "gained": sum(a and not b for a, b in hits),
                "lost": sum(not a and b for a, b in hits),
                "net_hits": sum(int(a) - int(b) for a, b in hits),
                "common_missed": sum(not a and not b for a, b in hits),
                "candidate_missed": sum(not a for a, _ in hits),
                "anchor_total": len(events),
            }
        )
    selected = []
    for kind, field in (("gain", "gained"), ("failure", "lost")):
        direction = 1 if kind == "gain" else -1
        eligible = [row for row in rows if direction * row["net_hits"] > 0]
        if eligible:
            row = min(
                eligible,
                key=lambda row: (
                    -direction * row["net_hits"],
                    -row[field],
                    row["issue_time_us"],
                    row["fold_id"],
                ),
            )
            label = "净新增命中较多一期" if kind == "gain" else "净损失命中较多一期"
            selection = (
                "positive_net_then_gained_descending_then_issue_time_us_fold_id"
                if kind == "gain"
                else "negative_net_then_lost_descending_then_issue_time_us_fold_id"
            )
        else:
            if not rows:
                continue
            used = {case["issue_index"] for case in selected}
            tiers = (
                (
                    [row for row in rows if row[field] > 0],
                    field,
                    "有新增但非净改善" if kind == "gain" else "有丢失但非净损失",
                ),
                ([row for row in rows if row["common_missed"] > 0], "common_missed", "共同漏报"),
                (rows, "earliest_nonempty", "非空期"),
            )
            # Seek a distinct issue across fallback tiers first. Reuse is explicit
            # and unavoidable only when there is no other nonempty main issue.
            for allow_repeat in (False, True):
                for pool, fallback_field, _fallback_label in tiers:
                    options = [
                        row for row in pool if allow_repeat or row["issue_index"] not in used
                    ]
                    if options:
                        row = min(
                            options,
                            key=lambda row: (
                                0
                                if fallback_field == "earliest_nonempty"
                                else -row[fallback_field],
                                row["issue_time_us"],
                                row["fold_id"],
                            ),
                        )
                        break
                else:
                    continue
                break
            label = (
                ("无净新增命中期" if kind == "gain" else "无净损失命中期")
                + "；回退"
                + _fallback_label
            )
            if row["issue_index"] in used:
                label += "（无其他非空期）"
            selection = (
                "no_signed_net_fallback_"
                + fallback_field
                + "_prefer_distinct_then_count_descending_issue_time_us_fold_id"
            )
        selected.append(
            {**row, "kind": kind, "label": label, "selection": selection, "highlight": field}
        )
    return {
        "status": "available" if selected else "no_main_anchor_events",
        "candidate_model_id": candidate,
        "reference_model_id": reference,
        "candidate_selection": "highest_saved_development_main_anchor_recall_among_C2B_then_summary_model_order",
        "display_rule_version": 2,
        "fallback_repeat_policy": "seek_distinct_issue_across_directional_common_miss_nonempty_tiers_before_reuse",
        "scope": "development_illustration_not_independent_optimality_or_adoption_evidence",
        "horizon_days": 30,
        "magnitude_bin": "M5_6",
        "hit_tolerance_km": 0,
        "area_budget_km2": 600000,
        "cases": selected,
    }


def _case_figures(data: dict[str, Any], root: Path) -> dict[str, Any]:
    selection = _select_cases(data)
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 12.0), squeeze=False)
    patches = []
    for cell in data["grid"]:
        vertices, codes = [], []
        for polygon in cell["polygons"]:
            for ring in polygon:
                vertices.extend(ring)
                codes.extend(
                    [PlotPath.MOVETO, *([PlotPath.LINETO] * (len(ring) - 2)), PlotPath.CLOSEPOLY]
                )
        patches.append(PathPatch(PlotPath(np.asarray(vertices), codes)))
    points = np.asarray(
        [
            point
            for cell in data["grid"]
            for polygon in cell["polygons"]
            for ring in polygon
            for point in ring
        ]
    )
    lower, upper = points.min(axis=0), points.max(axis=0)
    # Display-only padding prevents a nearly horizontal synthetic/local domain
    # collapsing the map. No alarm boundary, selected cell or paid area changes.
    minimum_latitude_span = (upper[0] - lower[0]) * 0.45
    if upper[1] - lower[1] < minimum_latitude_span:
        middle = (lower[1] + upper[1]) / 2
        lower[1], upper[1] = middle - minimum_latitude_span / 2, middle + minimum_latitude_span / 2
    pad = np.maximum((upper - lower) * 0.025, 0.05)
    bit = 1 << AREA_BUDGETS.index(600000.0)
    for row_index in range(2):
        if row_index >= len(selection["cases"]):
            for ax in axes[row_index]:
                ax.text(
                    0.5,
                    0.5,
                    "无可展示的主锚点震例",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            continue
        case = selection["cases"][row_index]
        issue = data["issues"][case["issue_index"]]
        events = [event for event in issue["events"] if event["anchor"]]
        for col_index, model in enumerate(
            (selection["candidate_model_id"], selection["reference_model_id"])
        ):
            ax = axes[row_index, col_index]
            plan = data["plans"][issue["forecasts"][model]]
            area = next(area for area in plan["areas"] if area[0] == 600000)
            selected = plan["order"][: area[2]]
            ax.add_collection(
                PatchCollection(patches, facecolor="#e7eef2", edgecolor="none", zorder=1)
            )
            ax.add_collection(
                PatchCollection(
                    [patches[index] for index in selected],
                    facecolor="#008777" if col_index == 0 else "#427eaf",
                    edgecolor="none",
                    zorder=2,
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
                    s=48,
                    linewidths=1.5,
                    color="#172f3f" if hit else "#bd323f",
                    zorder=4,
                )
                candidate_hit = bool(event["hits"][selection["candidate_model_id"]]["0"] & bit)
                reference_hit = bool(event["hits"][selection["reference_model_id"]]["0"] & bit)
                highlighted = (
                    candidate_hit and not reference_hit
                    if case["highlight"] == "gained"
                    else reference_hit and not candidate_hit
                )
                if highlighted:
                    ax.scatter(
                        event["lon"],
                        event["lat"],
                        marker="o",
                        s=135,
                        linewidths=1.6,
                        facecolors="none",
                        edgecolors="#933d93" if case["highlight"] == "gained" else "#e08122",
                        zorder=5,
                    )
            ax.set_xlim(lower[0] - pad[0], upper[0] + pad[0])
            ax.set_ylim(lower[1] - pad[1], upper[1] + pad[1])
            ax.set_aspect(1 / np.cos(np.deg2rad(points[:, 1].mean())))
            ax.set_xlabel("经度（°E）")
            ax.set_ylabel("纬度（°N）")
            ax.grid(color="#edf1f4", linewidth=0.6, zorder=0)
            date = (
                pd.Timestamp(issue["issue_us"], unit="us", tz="UTC")
                .tz_convert("Asia/Shanghai")
                .strftime("%Y-%m-%d")
            )
            ax.set_title(
                f"{case['label']}｜{date}\n净 {case['net_hits']:+d}（新增 {case['gained']}／丢失 {case['lost']}）\n{MODEL_LABELS.get(model, model)}：保存命中 {hits}/{len(events)}；实际 {area[1] / 10000:.2f} 万 km²",
                loc="left",
                fontsize=11,
                pad=12,
            )
    fig.suptitle(
        "开发展示震例：同一候选的新增命中与失败对照（仅本机）", x=0.055, ha="left", fontsize=18
    )
    fig.text(
        0.055,
        0.925,
        "30 天 · Ms 5–6 · 严格 0 km · 60 万 km²；每行是同一期，左为候选，右为已保存 C0 L3。",
        fontsize=10,
        color="#526573",
    )
    fig.text(
        0.055,
        0.035,
        "候选仅按开发主锚点召回最大、固定模型轴打破平分选作展示；不代表独立最优或采用证据。\n成功期优先净增最多、再新增最多；失败期优先净损最多、再丢失最多，同分按日期、折排序。\n无净正／负期则如实回退有新增／丢失期、共同漏报或非空期，优先不重复。● 命中，× 漏报；成功行紫环标新增，失败行橙环标丢失。\n全域范围仅用于展示；报警多边形、面积和命中完全沿用保存产物。含逐事件位置，PNG/SVG 不纳入公开聚合图。"
        + ("\n合成测试示例，不代表科学结果。" if data["summary"].get("synthetic_fixture") else ""),
        fontsize=9,
        color="#526573",
        linespacing=1.6,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.20, hspace=0.53, wspace=0.20)
    _save_figure(fig, root, CASE_STEM)
    return selection


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>SeismoFlux｜C2B 开发期离线回放（仅本机）</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f9;color:#243746;font:15px/1.6 "Microsoft YaHei",Arial,sans-serif}main{max-width:1420px;margin:auto;padding:26px 22px}h1{font-size:27px;line-height:1.35}h2{font-size:20px;margin:25px 0 10px}h3{font-size:16px;margin:4px 0}p{margin:8px 0}.scope{background:#eaf0f4;border-left:4px solid #264b73;padding:12px 16px}.note{font-size:13px;color:#526573}.warning{color:#9e3340}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:15px 0;align-items:center}.controls label{display:flex;align-items:center;gap:7px}select,button{font:inherit;padding:6px 8px;border:1px solid #aab8c2;border-radius:3px;background:white;max-width:100%}button{cursor:pointer}button:disabled{opacity:.4}input[type=range]{flex:1;min-width:150px}.figure{width:100%;height:auto;background:white;display:block}.maps{display:grid;grid-template-columns:1fr 1fr;gap:20px}.mapstats{min-height:44px;font-size:13px}canvas{width:100%;height:430px;background:white;display:block}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;background:white;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce5eb;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#eaf0f4}.positive{color:#008777}.negative{color:#ad3c43}.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}.figure-links{display:flex;flex-wrap:wrap;gap:12px}details{margin:14px 0}footer{border-top:1px solid #dce5eb;margin-top:26px;padding-top:16px;font-size:13px;color:#526573}@media(max-width:850px){.maps{grid-template-columns:1fr}main{padding:18px 12px}h1{font-size:23px}canvas{height:360px}.controls label{max-width:100%}}
</style></head><body><main><h1>C2B：目录数据与位置模型的历史比较</h1>
<div class="scope">仅本机使用：本页含逐事件震中及命中数据，不得作为公开页面发布。<br>这是已经评分的开发期历史回放，不是真实未来预测，不代表绝对发震概率，也不是新增独立测试证据。<p id="synthetic" class="warning"></p></div>
<p class="note">完整工作线：C0 参考 → C2A 输入敏感性 → C2B 目录与简单模型比较 → 科学价值复审。当前页只读冻结产物，不训练、不评分、不修改历史预测。D0：1970 起规范目录 Ms≥4；D1：1980 起 M3 目录来源 Ms≥4；D2：1950 起 M5 目录来源 Ms≥5。数据面板变化是来源、时间和震级等的总体差异，不能视为纯年份或纯震级效应。</p>
<details><summary>主锚点静态图（聚合图可单独公开）</summary><img class="figure" src="data:image/png;base64,__MAIN_IMAGE__" alt="30天Ms5–6严格0km与60万平方公里主锚点的全部模型命中和召回"></details>
<div class="controls"><label>时限 <select id="horizon"></select></label><label>震级 <select id="band"></select></label><label>预算 <select id="area"></select></label><label>命中口径 <select id="tolerance"></select></label></div>
<div class="controls"><label>候选模型 <select id="candidate"></select></label><label>参考模型 <select id="reference"></select></label></div>
<h2>同一评价轴的聚合结果</h2><div class="tablewrap"><table><thead><tr><th>模型</th><th>锚点命中／总数</th><th>召回</th><th>实际面积均值（万 km²）</th><th>曝光／空期</th><th>平均空间对数密度</th></tr></thead><tbody id="metric-table"></tbody></table></div>
<p id="pair-summary"></p><p id="pair-ci" class="note"></p><p class="note">配对区间沿用评分产物：2,000 次配对非重叠时间曝光重采样，含空期，不是独立震序重采样。未保存的模型组合不补造区间；NA、零变化、负变化全部保留。</p>
<h2>按起报日期回放（含空期）</h2><div class="controls"><button id="previous" type="button">上一期</button><input id="issue" type="range" min="0" value="0" aria-label="起报日期滑块"><button id="next" type="button">下一期</button><label>日期 <select id="issue-select"></select></label><label>目标显示 <select id="view"><option value="anchor">固定首震锚点</option><option value="all">全部目标事件</option></select></label></div>
<p id="issue-label" aria-live="polite"></p><p id="issue-change"></p><div class="maps"><section><h3 id="title-candidate"></h3><p class="mapstats" id="stats-candidate"></p><canvas id="map-candidate" role="img" aria-label="候选模型保存的报警格及真实目标震中"></canvas></section><section><h3 id="title-reference"></h3><p class="mapstats" id="stats-reference"></p><canvas id="map-reference" role="img" aria-label="参考模型保存的报警格及真实目标震中"></canvas></section></div>
<div class="legend"><span>蓝色：共同报警格</span><span class="positive">绿色：候选新增报警格</span><span class="negative">红色：参考独有报警格</span><span>● 命中</span><span>× 漏报</span><span>紫色外环：候选新增命中</span></div>
<p class="note">地图绘制评分产物的真实裁剪格多边形及目录震中；仅为经纬度展示，不推断行政边界。严格与 70 km 命中均读取保存的 event_results.hit；70 km 是距裁剪报警格多边形的次要容差，不扩大计费面积。切换时限或震级会切换独立日期轴，空期不删除。</p>
<div class="tablewrap"><table><thead><tr><th>本期目标事件</th><th>目标类型</th><th>经度</th><th>纬度</th><th>候选</th><th>参考</th><th>变化</th></tr></thead><tbody id="event-table"></tbody></table></div>
<footer>所有数据、几何和脚本均已嵌入，可直接断网打开。本页只回放保存的预测和评价，不能据此宣称模型已通过采用门控；最终科学价值需结合主锚点、各折、各时限及负结果复审。HTML/事件数据仅本机；PNG/SVG 为不含逐事件内容的聚合图。</footer></main>
<script id="replay-data" type="application/json">__DATA__</script><script>
"use strict";
const D=JSON.parse(document.getElementById("replay-data").textContent),S=D.summary,$=id=>document.getElementById(id);
const horizon=$("horizon"),band=$("band"),area=$("area"),tolerance=$("tolerance"),candidate=$("candidate"),reference=$("reference"),slider=$("issue"),dateSelect=$("issue-select"),view=$("view");
const number=(x,n=1)=>x===null||x===undefined?"NA":Number(x).toLocaleString("zh-CN",{minimumFractionDigits:n,maximumFractionDigits:n}),signed=(x,n=1)=>x===null||x===undefined?"NA":(x>0?"+":"")+number(x,n),date=iso=>new Date(iso).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
S.horizons_days.forEach(h=>horizon.add(new Option(`${h} 天`,h)));horizon.value=S.horizons_days.includes(30)?"30":String(S.horizons_days[0]);S.magnitude_bins.forEach(b=>band.add(new Option(D.band_labels[b]||b,b)));band.value=S.magnitude_bins.includes("M5_6")?"M5_6":S.magnitude_bins[0];D.area_budgets.forEach(a=>area.add(new Option(`${a/10000} 万 km²`,a)));area.value="600000";D.tolerances.forEach(t=>tolerance.add(new Option(t===0?"严格 0 km（主指标）":"70 km（次要容差）",t)));tolerance.value="0";S.model_ids.forEach(m=>{candidate.add(new Option(D.model_labels[m],m));reference.add(new Option(D.model_labels[m],m));});candidate.value=S.model_ids.includes("C2B_D0_K75")?"C2B_D0_K75":S.model_ids[0];reference.value=S.model_ids.includes("C0_L3_B0_R30_CAUSAL")?"C0_L3_B0_R30_CAUSAL":S.model_ids[0];if(S.synthetic_fixture)$("synthetic").textContent="合成测试示例：仅验证渲染与交互，不代表科学结果。";
let visibleIssues=[];const axis=row=>row.horizon_days===+horizon.value&&row.magnitude_bin===band.value&&row.area_budget_km2===+area.value&&row.hit_tolerance_km===+tolerance.value;
function addRow(body,values){const tr=body.insertRow();values.forEach(value=>tr.insertCell().textContent=value);return tr;}
function logLabel(row){if(row.log_density_status==="negative_infinity_from_saved_C0_zero_mass")return "−∞（已保存 C0 零质量）";if(row.log_density_status==="no_events")return "NA（无事件）";return row.event_mean_log_density===null?`NA（${row.log_density_status||"状态未提供"}）`:number(row.event_mean_log_density,4);}
function drawMetrics(){const body=$("metric-table");body.textContent="";[candidate.value,reference.value].forEach(m=>{const row=S.curves.find(c=>c.model_id===m&&axis(c));addRow(body,[D.model_labels[m],`${number(row.anchor_hits,0)} / ${number(row.anchor_total,0)}`,row.anchor_recall===null?"NA":number(row.anchor_recall*100)+"%",`${number(row.actual_area_mean_km2/10000,2)}（${number(row.actual_area_min_km2/10000,2)}–${number(row.actual_area_max_km2/10000,2)}）`,`${row.exposure_count} / ${row.empty_exposure_count}`,logLabel(row)]);});let p=S.pairings.find(p=>p.candidate_model_id===candidate.value&&p.reference_model_id===reference.value&&axis(p)),reverse=false;if(!p){p=S.pairings.find(p=>p.candidate_model_id===reference.value&&p.reference_model_id===candidate.value&&axis(p));reverse=!!p;}if(!p){$("pair-summary").textContent="此模型组合／评价轴未保存预登记配对结果；不补造置信区间。下方仍可查看已保存命中的逐期差异。";$("pair-ci").textContent="";return;}const sign=reverse?-1:1,delta=p.delta_recall_pp===null?null:sign*p.delta_recall_pp;$("pair-summary").textContent=`候选 − 参考：新增命中 ${reverse?p.lost:p.gained}，丢失命中 ${reverse?p.gained:p.lost}；净变化 ${signed(sign*p.net_hits,0)}；召回差 ${signed(delta)} 个百分点。`;const ci=p.bootstrap_ci95_pp===null?null:reverse?[-p.bootstrap_ci95_pp[1],-p.bootstrap_ci95_pp[0]]:p.bootstrap_ci95_pp;$("pair-ci").textContent=`95% 配对区间：${ci===null?"NA（无可评价区间）":`[${signed(ci[0])}, ${signed(ci[1])}] 个百分点`}。各折净命中：${p.per_fold.map(f=>`${f.fold_id} ${signed(sign*f.net_hits,0)}/${f.anchor_total}`).join("；")}。`;}
function rebuildDates(){const previous=visibleIssues[+slider.value];visibleIssues=D.issues.filter(i=>i.horizon_days===+horizon.value&&i.magnitude_bin===band.value);dateSelect.textContent="";visibleIssues.forEach((issue,i)=>dateSelect.add(new Option(`${date(issue.label)} · ${issue.fold}${issue.event_count===0?" · 空期":""}`,i)));slider.max=Math.max(0,visibleIssues.length-1);const index=previous?visibleIssues.findIndex(i=>i.issue_us===previous.issue_us&&i.fold===previous.fold):-1;slider.value=String(Math.max(0,index));}
function hit(event,model){return Boolean(event.hits[model][tolerance.value]&(1<<D.area_budgets.indexOf(+area.value)));}
function plan(issue,model){const p=D.plans[issue.forecasts[model]],a=p.areas.find(a=>a[0]===+area.value);return {cells:new Set(p.order.slice(0,a[2])),actual:a[1]};}
let lonMin=Infinity,lonMax=-Infinity,latMin=Infinity,latMax=-Infinity;D.grid.forEach(cell=>cell.polygons.forEach(poly=>poly.forEach(ring=>ring.forEach(p=>{lonMin=Math.min(lonMin,p[0]);lonMax=Math.max(lonMax,p[0]);latMin=Math.min(latMin,p[1]);latMax=Math.max(latMax,p[1]);}))));const lonPad=Math.max(.1,(lonMax-lonMin)*.035),latPad=Math.max(.1,(latMax-latMin)*.05);lonMin-=lonPad;lonMax+=lonPad;latMin-=latPad;latMax+=latPad;
function drawMap(which,model,selected,other,events){const canvas=$("map-"+which),ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext("2d");c.scale(ratio,ratio);c.fillStyle="white";c.fillRect(0,0,w,h);const l=44,r=14,top=14,b=30,x=lon=>l+(lon-lonMin)/(lonMax-lonMin)*(w-l-r),y=lat=>h-b-(lat-latMin)/(latMax-latMin)*(h-top-b);c.font='11px "Microsoft YaHei",Arial,sans-serif';c.strokeStyle="#e1e8ed";c.fillStyle="#526573";c.lineWidth=.6;for(let i=0;i<=4;i++){const lon=lonMin+(lonMax-lonMin)*i/4,lat=latMin+(latMax-latMin)*i/4;c.beginPath();c.moveTo(x(lon),top);c.lineTo(x(lon),h-b);c.moveTo(l,y(lat));c.lineTo(w-r,y(lat));c.stroke();c.textAlign=i===0?"left":i===4?"right":"center";c.fillText(number(lon,0)+"°E",x(lon),h-8);c.textAlign="right";c.fillText(number(lat,0)+"°N",l-5,y(lat)+4);}function cellPath(cell){c.beginPath();cell.polygons.forEach(poly=>poly.forEach(ring=>{ring.forEach((p,i)=>i?c.lineTo(x(p[0]),y(p[1])):c.moveTo(x(p[0]),y(p[1])));c.closePath();}));}c.fillStyle="#eaf0f3";D.grid.forEach(cell=>{cellPath(cell);c.fill("evenodd");});selected.forEach(index=>{c.fillStyle=other.has(index)?"#427eaf":which==="candidate"?"#008777":"#ad3c43";cellPath(D.grid[index]);c.fill("evenodd");});events.forEach(e=>{const xx=x(e.lon),yy=y(e.lat),savedHit=hit(e,model);c.lineWidth=1.8;if(savedHit){c.fillStyle="#122b39";c.strokeStyle="white";c.beginPath();c.arc(xx,yy,4,0,Math.PI*2);c.fill();c.stroke();}else{c.strokeStyle="#a9303b";c.beginPath();c.moveTo(xx-4,yy-4);c.lineTo(xx+4,yy+4);c.moveTo(xx-4,yy+4);c.lineTo(xx+4,yy-4);c.stroke();}if(hit(e,candidate.value)&&!hit(e,reference.value)){c.strokeStyle="#923b92";c.beginPath();c.arc(xx,yy,7,0,Math.PI*2);c.stroke();}});}
function drawIssue(){const i=+slider.value,issue=visibleIssues[i];dateSelect.value=String(i);$("previous").disabled=i<=0;$("next").disabled=i>=visibleIssues.length-1;if(!issue)return;const events=issue.events.filter(e=>view.value==="all"||e.anchor),a=plan(issue,candidate.value),b=plan(issue,reference.value),gain=events.filter(e=>hit(e,candidate.value)&&!hit(e,reference.value)).length,loss=events.filter(e=>!hit(e,candidate.value)&&hit(e,reference.value)).length;$("issue-label").textContent=`第 ${i+1}/${visibleIssues.length} 期 · ${date(issue.label)}（北京时间） · ${issue.fold} · ${issue.horizon_days} 天 · ${D.band_labels[issue.magnitude_bin]||issue.magnitude_bin} · 全部目标 ${issue.event_count}，首震锚点 ${issue.anchor_total}`;$("issue-change").textContent=`当前显示目标：新增命中 ${gain}，丢失命中 ${loss}，净变化 ${signed(gain-loss,0)}；候选新增报警 ${[...a.cells].filter(c=>!b.cells.has(c)).length} 格，参考独有 ${[...b.cells].filter(c=>!a.cells.has(c)).length} 格。`;[["candidate",candidate.value,a,b],["reference",reference.value,b,a]].forEach(([which,model,p,other])=>{$("title-"+which).textContent=D.model_labels[model];$("stats-"+which).textContent=`保存命中 ${events.filter(e=>hit(e,model)).length}/${events.length} · 实际面积 ${number(p.actual/10000,2)} 万 km² · ${p.cells.size} 格`;drawMap(which,model,p.cells,other.cells,events);});const body=$("event-table");body.textContent="";if(!events.length){const td=body.insertRow().insertCell();td.colSpan=7;td.textContent=issue.event_count===0?"本期为空期；仍保留在时间曝光、面积记录和回放中。":"本期无首震锚点；可切换查看全部目标事件。";}else events.forEach(e=>{const ah=hit(e,candidate.value),bh=hit(e,reference.value),change=ah&&!bh?"新增命中":!ah&&bh?"丢失命中":"不变",row=addRow(body,[e.id,e.anchor?"首震锚点":"后续事件",number(e.lon,3),number(e.lat,3),ah?"命中":"漏报",bh?"命中":"漏报",change]);row.lastChild.className=ah&&!bh?"positive":!ah&&bh?"negative":"";});}
function update(){drawMetrics();drawIssue();}[candidate,reference,area,tolerance].forEach(control=>control.addEventListener("change",update));[horizon,band].forEach(control=>control.addEventListener("change",()=>{rebuildDates();update();}));view.addEventListener("change",drawIssue);slider.addEventListener("input",drawIssue);dateSelect.addEventListener("change",()=>{slider.value=dateSelect.value;drawIssue();});$("previous").addEventListener("click",()=>{slider.value=+slider.value-1;drawIssue();});$("next").addEventListener("click",()=>{slider.value=+slider.value+1;drawIssue();});window.addEventListener("resize",drawIssue);rebuildDates();update();
</script></body></html>"""


def render(output_root: Path, render_root: Path | None = None) -> Path:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = output_root.resolve()
    destination = render_root.resolve() if render_root is not None else output / "rendered"
    summary, events, alarms, grid, exposures = _load(output)
    data = _replay_data(summary, events, alarms, grid, exposures)
    destination.mkdir(parents=True, exist_ok=True)
    _static_figures(summary, destination)
    case_selection = _case_figures(data, destination)
    encoded = base64.b64encode((destination / f"{STATIC_STEMS[0]}.png").read_bytes()).decode(
        "ascii"
    )
    page = HTML_TEMPLATE.replace("__MAIN_IMAGE__", encoded).replace(
        "__DATA__", _json_for_script(data)
    )
    html_path = destination / HTML_NAME
    html_path.write_text(page, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_score_manifest_sha256": _sha256(output / "score_phase/score_manifest.json"),
        "scientific_role": "development_location_model_comparison_not_independent_confirmation",
        "event_coordinate_semantics": "original_catalog_epicentre_from_saved_score_artifact",
        "alarm_geometry_semantics": "saved_clipped_equal_area_cell_polygon_transformed_to_WGS84_for_display_only",
        "hit_semantics": "saved_event_results_hit_for_each_model_horizon_band_issue_budget_tolerance",
        "timestamp_unit": "us",
        "empty_exposures_retained": True,
        "network_resources": [],
        "synthetic_fixture": bool(summary.get("synthetic_fixture", False)),
        "case_selection": case_selection,
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
    parser = argparse.ArgumentParser(description="C2B 聚合科学图与本机独立离线历史回放；不执行评分")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/multitask_s1/s1c2b_catalog_models_v1"),
        help="C2B run root（其中包含 score_phase）",
    )
    parser.add_argument("--render-root", type=Path)
    args = parser.parse_args()
    print(render(args.output_root, args.render_root))


if __name__ == "__main__":
    main()
