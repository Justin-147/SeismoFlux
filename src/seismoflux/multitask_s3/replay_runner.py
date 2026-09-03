"""Build a local offline S3 replay from immutable predictions and saved outcomes.

Only alarm geometry is reconstructed with the original ranking rule. No target
catalog, feature store, model fit, new hit score, or null result is opened.
"""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml
from pyproj import Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s1.c2b_score import log_alarm_prefixes
from seismoflux.multitask_s1.runner_inputs import load_verified_spatial_inputs
from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, build_fold_calendar
from seismoflux.multitask_s3.case_ledger import CONTRASTS, VARIANTS
from seismoflux.multitask_s3.preparation import sha256, write_json
from seismoflux.multitask_s3.runner import (
    BANDS,
    COUNT_VARIANTS,
    read_prediction_block,
    verify_complete_predictions,
)

LABELS = {
    "CATALOG": "长期地震目录",
    "R30_REFERENCE": "长期＋近期地震目录",
    "CAT_COV": "目录＋报告覆盖",
    "CAT_SNAP": "目录＋异常现状",
    "CAT_DYN": "目录＋异常现状与变化",
    "T0": "目录原始次数率",
    "T0_CAL": "目录次数率校准",
    "T0_CAL_COV": "校准＋报告覆盖",
    "T0_CAL_SNAP": "校准＋异常现状",
    "T0_CAL_DYN": "校准＋异常现状与变化",
}


def geometry_polygons(geometry: Any) -> list[Any]:
    """Same polygon traversal as the existing S1/S2 renderer; keep holes."""
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [polygon for part in geometry.geoms for polygon in geometry_polygons(part)]
    return []


def serialize_geometry(geometries: Sequence[Any]) -> dict[str, Any]:
    cells = []
    bounds = []
    for geometry in geometries:
        polygons = geometry_polygons(geometry)
        if not polygons:
            raise ValueError("each frozen cell must have polygon geometry")
        # Decimetre rounding is display-only; costs and hits use unrounded inputs.
        cells.append(
            [
                [
                    [[round(float(x), 1), round(float(y), 1)] for x, y in ring.coords]
                    for ring in (polygon.exterior, *polygon.interiors)
                ]
                for polygon in polygons
            ]
        )
        bounds.append(geometry.bounds)
    if not bounds:
        raise ValueError("the display grid cannot be empty")
    return {
        "crs": str(EQUAL_AREA_CRS),
        "coordinate_units": "metres",
        "display_rounding_metres": 0.1,
        "bounds": [
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        ],
        "cells": cells,
    }


def frame_from_saved_diagnostics(
    records: Sequence[Mapping[str, Any]],
    *,
    model_alarms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Copy both bands and all saved hit flags; preserve empty earthquake windows."""
    if len(records) != 2 or {r["magnitude_band"] for r in records} != set(BANDS):
        raise ValueError("each replay date needs both formal magnitude bands, including empty ones")
    first = records[0]
    identity = (first["fold_id"], first["horizon_days"], first["issue_time_utc"])
    if any(
        (r["fold_id"], r["horizon_days"], r["issue_time_utc"]) != identity
        or r["primary_nonoverlap"] != first["primary_nonoverlap"]
        for r in records
    ):
        raise ValueError("bands must refer to the identical issue, horizon and reporting axis")
    if set(model_alarms) != set(VARIANTS):
        raise ValueError("all five saved spatial variants are required")
    issue = datetime.fromisoformat(str(first["issue_time_utc"]))
    if issue.tzinfo is None:
        raise ValueError("replay issue must be timezone-aware")
    models = {
        variant: {"alarms": [dict(alarm) for alarm in model_alarms[variant]]}
        for variant in VARIANTS
    }
    bands = {}
    for record in records:
        ids = list(record["target_event_ids"])
        anchor = list(record["anchor_mask"])
        if len(anchor) != len(ids) or any(not isinstance(v, bool) for v in anchor):
            raise ValueError("saved event and anchor arrays must be aligned")
        if set(record["spatial"]) != set(VARIANTS) or set(record["count"]) != set(COUNT_VARIANTS):
            raise ValueError("saved record lost a registered spatial or count model")
        outcomes: dict[str, list[dict[str, Any]]] = {}
        for variant in VARIANTS:
            saved = record["spatial"][variant]["alarms"]
            shown = {a["area_budget_km2"]: a for a in model_alarms[variant]}
            if len(saved) != len(shown) or {a["area_budget_km2"] for a in saved} != set(shown):
                raise ValueError("display alarm budgets differ from saved evaluation")
            outcomes[variant] = []
            for alarm in saved:
                display = shown[alarm["area_budget_km2"]]
                if not math.isclose(
                    display["actual_area_km2"], alarm["actual_area_km2"], rel_tol=0, abs_tol=1e-7
                ):
                    raise ValueError("display alarm area differs from the original paid cost")
                hits = alarm["_local"]
                if alarm["secondary_70km"]["status"] != "scored":
                    raise ValueError("do not invent an unscored 70-km outcome")
                for key in ("strict_hits", "secondary_70km_hits"):
                    if len(hits[key]) != len(ids) or any(
                        not isinstance(v, bool) for v in hits[key]
                    ):
                        raise ValueError("saved hit arrays must be complete and aligned")
                outcomes[variant].append(
                    {
                        "area_budget_km2": alarm["area_budget_km2"],
                        "strict_hits": list(hits["strict_hits"]),
                        "secondary_70km_hits": list(hits["secondary_70km_hits"]),
                    }
                )
        bands[record["magnitude_band"]] = {
            "event_ids": ids,
            "anchor_mask": anchor,
            "outcomes": outcomes,
            "counts": {variant: dict(record["count"][variant]) for variant in COUNT_VARIANTS},
        }
    return {
        "id": "|".join(map(str, identity)),
        "fold_id": identity[0],
        "horizon_days": identity[1],
        "issue_time_utc": identity[2],
        "target_end_utc": (issue + timedelta(days=int(identity[1]))).isoformat(),
        "primary_nonoverlap": first["primary_nonoverlap"],
        "models": models,
        "bands": bands,
    }


def render_trial(
    *,
    project_root: Path,
    data_root: Path,
    prediction_dir: Path,
    score_dir: Path,
    case_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project_root.resolve()
    prediction, score, cases, output = (
        p.resolve() for p in (prediction_dir, score_dir, case_dir, output_dir)
    )
    allowed = project / "outputs/multitask_s3"
    if any(not p.is_relative_to(allowed) for p in (prediction, score, cases, output)):
        raise ValueError("replay inputs and output must remain in the local S3 workspace")
    final = output / "seismoflux_s3a_replay.html"
    if final.exists() or (output / "replay_manifest.json").exists():
        raise FileExistsError("preserve an existing replay; do not overwrite a finished artifact")
    manifest_path = prediction / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_hash = sha256(manifest_path)
    issues = tuple(datetime.fromisoformat(value) for value in manifest["issue_times_utc"])
    truth_cutoff = datetime.fromisoformat(manifest["truth_cutoff_utc"])
    calendars = {
        f"{fold}__h{h:03d}": build_fold_calendar(
            issues, fold_id=fold, horizon_days=h, truth_cutoff=truth_cutoff
        )
        for fold in FOLDS
        for h in HORIZONS
    }
    verify_complete_predictions(prediction, manifest, calendars)
    score_progress = json.loads((score / "score_progress.json").read_text(encoding="utf-8"))
    ledger_path = cases / "case_ledger_local.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    diagnostic_path = score / "event_diagnostics_local.json"
    diagnostic_hash = sha256(diagnostic_path)
    if (
        score_progress["status"] != "complete"
        or score_progress["prediction_manifest_sha256"] != prediction_hash
        or ledger.get("local_only") is not True
        or ledger["provenance"]["prediction_manifest_sha256"] != prediction_hash
        or ledger["provenance"]["event_diagnostics_sha256"] != diagnostic_hash
    ):
        raise ValueError("replay needs the same completed, local-only scored evidence")
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostics.get("local_only") is not True:
        raise ValueError("event outcomes must remain local-only")
    prepared = manifest["identity"]["prepared_inputs"]
    protocol_path = project / "configs/multitask_s3_anomaly.yaml"
    if sha256(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("protocol differs from the saved prediction")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    budgets = list(protocol["support"]["alarm_area_budgets_km2"])
    domain, _, area_hash = load_verified_spatial_inputs(data_root)
    grid = domain.operational_grid
    if (
        grid.grid_id != prepared["grid_id"]
        or grid.cell_count != prepared["grid_cells"]
        or area_hash != prepared["study_area_sha256"]
    ):
        raise ValueError("display geometry differs from the frozen prediction grid")
    geometry = serialize_geometry(domain.locator.clipped_geometries)
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for record in diagnostics["records"]:
        record_key = (record["fold_id"], record["horizon_days"], record["issue_time_utc"])
        grouped.setdefault(record_key, []).append(record)
    frames = []
    unevaluable = []
    consumed = set()
    for key, calendar in calendars.items():
        saved = read_prediction_block(
            prediction / manifest["completed"][key]["file"],
            identity=manifest["identity"],
            calendar=calendar,
        )
        if not calendar.evaluation_issues:
            unevaluable.append(
                {
                    "fold_id": calendar.fold_id,
                    "horizon_days": calendar.horizon_days,
                    "reason": "本年度开发折没有完整评价窗口；NA，不是零命中",
                }
            )
        for index, issue in enumerate(calendar.evaluation_issues):
            frame_key = (calendar.fold_id, calendar.horizon_days, issue.isoformat())
            alarms = {
                variant: log_alarm_prefixes(saved["spatial_log_mass"][index, column], grid, budgets)
                for column, variant in enumerate(VARIANTS)
            }
            frames.append(frame_from_saved_diagnostics(grouped[frame_key], model_alarms=alarms))
            consumed.add(frame_key)
        print(f"Replay geometry {key}: {len(calendar.evaluation_issues)} dates", flush=True)
    if consumed != set(grouped):
        raise ValueError("some saved diagnostic windows were not included in the replay")
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    events = {}
    for event_id, metadata in ledger["events"].items():
        x, y = transformer.transform(metadata["longitude"], metadata["latitude"])
        events[event_id] = {**metadata, "x_m": float(x), "y_m": float(y)}
    shown_ids = {
        event_id
        for frame in frames
        for band in frame["bands"].values()
        for event_id in band["event_ids"]
    }
    if shown_ids != set(events):
        raise ValueError("replay events differ from the complete existing case ledger")
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_manifest_sha256": prediction_hash,
        "event_diagnostics_sha256": diagnostic_hash,
        "case_ledger_sha256": sha256(ledger_path),
        "grid_id": grid.grid_id,
        "study_area_sha256": area_hash,
        "model_refitted": False,
        "hits_recomputed": False,
        "new_targets_read": False,
        "null_scores_read": False,
        "alarm_geometry": "original_log_alarm_prefixes_on_immutable_saved_predictions",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
    }
    payload = {
        "version": "S3A_REPLAY_V1",
        "local_only": True,
        "notes": [
            "较晚历史开发；非独立测试、非真实前瞻。",
            "回放使用已保存预测与结局；重叠窗口不是新增独立地震样本。",
            "报警网格不是最终最多10区域产品；70km仅辅助命中判定，不扩张图上报警面积。",
            "次数头两震级带共享乘子；没有单独学习震级分布。",
        ],
        "budgets": budgets,
        "contrasts": [
            {
                "id": f"{candidate}_minus_{reference}",
                "candidate": candidate,
                "reference": reference,
                "label": f"{LABELS[candidate]} 对比 {LABELS[reference]}",
            }
            for candidate, reference in CONTRASTS
        ],
        "variants": LABELS,
        "geometry": geometry,
        "events": events,
        "frames": frames,
        "unevaluable": unevaluable,
        "provenance": provenance,
    }
    from seismoflux.multitask_s3.replay_html import render_replay_html

    page = render_replay_html(payload)
    # No output exists until all old evidence has been validated and assembled.
    output.mkdir(parents=True, exist_ok=True)
    temporary = final.with_suffix(".tmp.html")
    temporary.write_text(page, encoding="utf-8")
    os.replace(temporary, final)
    result = {
        "status": "local_replay_created",
        "local_only": True,
        "frames": len(frames),
        "band_windows": len(diagnostics["records"]),
        "unique_events": len(events),
        "unevaluable": unevaluable,
        "html_file": final.name,
        "html_bytes": final.stat().st_size,
        "html_sha256": sha256(final),
        "provenance": provenance,
        "gui_validation": "not_claimed_browser_file_policy_previously_denied",
    }
    write_json(output / "replay_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "data-root",
        "prediction-dir",
        "score-dir",
        "case-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("replay must run with one numerical-library thread")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    result = render_trial(**vars(args))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
