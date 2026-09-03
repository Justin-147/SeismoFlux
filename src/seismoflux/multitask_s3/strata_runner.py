"""Explain saved S3 effects by full-history episodes and fixed geographic strata.

No predictions, targets, hits, model fits, or null scores are recomputed. All
output remains local. Regional summaries use the unchanged nationwide alarm
budget; they are descriptive temporal-development strata, not spatial tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from seismoflux.multitask_s0 import build_episodes, verify_authoritative_catalog_identity
from seismoflux.multitask_s1.runner_inputs import load_verified_spatial_inputs
from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, build_fold_calendar
from seismoflux.multitask_s3.case_ledger import CONTRASTS, MODES
from seismoflux.multitask_s3.input_waterlevel import load_development_catalog
from seismoflux.multitask_s3.preparation import sha256, write_json
from seismoflux.multitask_s3.strata_summary import summarize_paired_group
from seismoflux.multitask_s3.strata_uncertainty import paired_uncertainty

BANDS = ("Ms5_6", "Ms6_plus")
VIEWS = ("anchor", "episode_balanced", "subsequent", "all")
AXES = ("primary_nonoverlap", "all_reports_descriptive")
MAPPING_PATH = "interim/stage4/anomaly_increment_r2/construction_zone_cell_mapping.parquet"
MAPPING_SHA256 = "171a500de9f9dd475f2c37a5426debc7c6f2d34ddd418056729c39b27118108e"


def episode_membership(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Group each formal band on the complete bounded history, never a subset."""
    result: dict[str, dict[str, Any]] = {}
    for band in BANDS:
        selected = frame.inside_study_area & (frame.magnitude >= (5 if band == "Ms5_6" else 6))
        if band == "Ms5_6":
            selected &= frame.magnitude < 6
        for episode in build_episodes(frame.loc[selected], max_time_days=30, max_distance_km=75.0):
            for event_id in cast(list[str], episode["member_event_ids"]):
                identifier = str(event_id)
                if identifier in result:
                    raise ValueError("event belongs to more than one formal-band episode")
                result[identifier] = {
                    "magnitude_band": band,
                    "episode_id": str(episode["episode_id"]),
                    "global_member_count": int(cast(int, episode["member_count"])),
                    "is_anchor": identifier == episode["anchor_event_id"],
                }
    return result


def align_region_aliases(mapping: pd.DataFrame, cell_ids: Sequence[str]) -> tuple[str, ...]:
    """Reuse the S0 UTF-8 ordered aliases, not event-derived boundaries."""
    if mapping.isna().any().any() or mapping.cell_id.astype(str).duplicated().any():
        raise ValueError("static region mapping has missing or duplicate identities")
    by_cell = dict(
        zip(mapping.cell_id.astype(str), mapping.construction_zone_id.astype(str), strict=True)
    )
    if set(by_cell) != set(cell_ids) or len(by_cell) != len(cell_ids):
        raise ValueError("static region mapping differs from the frozen grid")
    zones = sorted(set(by_cell.values()), key=lambda value: value.encode("utf-8"))
    if len(zones) != 39:
        raise ValueError("the frozen region mapping must retain its 39 nonempty zones")
    aliases = {zone: f"atomic_block_{index:02d}" for index, zone in enumerate(zones, 1)}
    return tuple(aliases[by_cell[cell]] for cell in cell_ids)


def attach_membership(
    rows: Sequence[Mapping[str, Any]],
    membership: Mapping[str, Mapping[str, Any]],
    region_by_cell: Sequence[str],
) -> list[dict[str, Any]]:
    """Attach only identities/weights; preserve every stored hit and cost."""
    result = []
    for row in rows:
        info = membership[row["event_id"]]
        if (
            info["magnitude_band"] != row["magnitude_band"]
            or bool(info["is_anchor"]) != (row["event_view"] == "anchor")
            or row["event_view"] not in ("anchor", "subsequent")
        ):
            raise ValueError("saved event band or anchor differs from full-history episodes")
        cell = row["target_cell_index"]
        if (
            isinstance(cell, bool)
            or not isinstance(cell, int)
            or not 0 <= cell < len(region_by_cell)
        ):
            raise ValueError("saved target cell is outside the unchanged grid")
        result.append({**row, **info, "region_id": region_by_cell[cell]})
    return result


def view_rows(rows: Sequence[Mapping[str, Any]], view: str) -> list[dict[str, Any]]:
    if view not in VIEWS:
        raise ValueError("unregistered event view")
    return [
        {**row, "weight": 1.0 / row["global_member_count"] if view == "episode_balanced" else 1.0}
        for row in rows
        if view in ("all", "episode_balanced")
        or (view == "anchor" and row["is_anchor"])
        or (view == "subsequent" and not row["is_anchor"])
    ]


def summarize_task(
    rows: Sequence[Mapping[str, Any]],
    *,
    issue_keys: Sequence[str],
    all_regions: Sequence[str],
    identity: Mapping[str, Any],
    primary: bool,
    member_counts: Mapping[str, int],
) -> dict[str, Any]:
    """National paired summaries plus all observed regions and explicit empty strata."""
    result = dict(identity)
    summary = summarize_paired_group(rows)
    result["national"] = {key: value for key, value in summary.items() if key != "_local"}
    result["issue_count"] = len(issue_keys)
    result["issues_without_selected_events"] = len(issue_keys) - len(
        summary["_local"]["issue_clusters"]
    )
    if not set(summary["_local"]["issue_clusters"]) <= set(issue_keys):
        raise ValueError("a stored event exposure is absent from the complete issue calendar")
    result["uncertainty"] = (
        paired_uncertainty(
            summary,
            issue_keys=issue_keys,
            label="S3A_STRATA|" + json.dumps(dict(identity), sort_keys=True),
            global_member_counts=member_counts,
        )
        if primary
        else {"status": "descriptive_only_overlapping_reports_not_independent_resamples"}
    )
    regional: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        regional[row["region_id"]].append(row)
    result["regions"] = [
        {
            "region_id": region,
            **{
                key: value
                for key, value in summarize_paired_group(group).items()
                if key != "_local"
            },
        }
        for region, group in sorted(regional.items())
    ]
    result["empty_region_ids_NA"] = sorted(set(all_regions) - set(regional))
    result["regional_interpretation"] = (
        "descriptive_only_same_nationwide_alarm_cost_not_spatial_holdout"
    )
    return result


def summarize_block(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    fold_scope: str,
    horizon: int,
    budgets: Sequence[float],
    all_regions: Sequence[str],
    member_counts: Mapping[str, int],
) -> dict[str, Any]:
    folds = set(FOLDS) if fold_scope == "POOLED_A_DEVELOPMENT" else {fold_scope}
    relevant = [r for r in records if r["fold_id"] in folds and r["horizon_days"] == horizon]
    if not relevant:
        return {
            "status": "no_complete_evaluation_windows_NA",
            "rows": [],
            "magnitude_bands": list(BANDS),
        }
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["fold_id"] in folds and row["horizon_days"] == horizon:
            key = (
                row["magnitude_band"],
                row["candidate"],
                row["reference"],
                row["area_budget_km2"],
                row["mode"],
            )
            grouped[key].append(row)
    output = []
    for axis in AXES:
        primary = axis == "primary_nonoverlap"
        for band in BANDS:
            selected = [
                r
                for r in relevant
                if r["magnitude_band"] == band and (not primary or r["primary_nonoverlap"])
            ]
            issue_keys = sorted(f"{r['fold_id']}|{r['issue_time_utc']}" for r in selected)
            if len(set(issue_keys)) != len(issue_keys):
                raise ValueError("duplicate saved diagnostic issue in the same task")
            for candidate, reference in CONTRASTS:
                for budget in budgets:
                    costs = {}
                    for variant in (candidate, reference):
                        paid = [
                            float(a["actual_area_km2"])
                            for r in selected
                            for a in r["spatial"][variant]["alarms"]
                            if a["area_budget_km2"] == budget
                        ]
                        if len(paid) != len(selected):
                            raise ValueError("saved alarm costs do not cover all evaluation issues")
                        costs[variant] = [min(paid), max(paid)] if paid else None
                    for mode, _ in MODES:
                        group = [
                            r
                            for r in grouped[(band, candidate, reference, budget, mode)]
                            if not primary or r["primary_nonoverlap"]
                        ]
                        for view in VIEWS:
                            identity = {
                                "fold_scope": fold_scope,
                                "horizon_days": horizon,
                                "magnitude_band": band,
                                "axis": axis,
                                "event_view": view,
                                "candidate": candidate,
                                "reference": reference,
                                "mode": mode,
                                "area_budget_km2": budget,
                            }
                            summary = summarize_task(
                                view_rows(group, view),
                                issue_keys=issue_keys,
                                all_regions=all_regions,
                                identity=identity,
                                primary=primary,
                                member_counts=member_counts,
                            )
                            summary["nationwide_actual_area_min_max_km2"] = costs
                            output.append(summary)
    return {"status": "summarized", "rows": output}


def summarize_trial(
    *,
    project_root: Path,
    data_root: Path,
    prediction_dir: Path,
    score_dir: Path,
    case_dir: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    project = project_root.resolve()
    prediction, scores, cases, output = (
        p.resolve() for p in (prediction_dir, score_dir, case_dir, output_dir)
    )
    allowed = project / "outputs/multitask_s3"
    if any(not p.is_relative_to(allowed) for p in (prediction, scores, cases, output)):
        raise ValueError("strata inputs and outputs must remain in the local S3 workspace")
    final_path = output / "strata_manifest.json"
    if output.exists() and (not resume or final_path.exists()):
        raise FileExistsError("preserve existing strata output; resume only an incomplete same run")

    def read(path: Path) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    prediction_path = prediction / "prediction_manifest.json"
    manifest = read(prediction_path)
    prediction_hash = sha256(prediction_path)
    progress = read(scores / "score_progress.json")
    ledger_path, diagnostic_path = (
        cases / "case_ledger_local.json",
        scores / "event_diagnostics_local.json",
    )
    ledger, diagnostics = read(ledger_path), read(diagnostic_path)
    prepared = manifest["identity"]["prepared_inputs"]
    provenance = ledger["provenance"]
    if (
        manifest["status"] != "predictions_complete"
        or progress["status"] != "complete"
        or progress["prediction_manifest_sha256"] != prediction_hash
        or ledger["status"] != "ledger_complete_no_case_selection"
        or ledger.get("local_only") is not True
        or diagnostics.get("local_only") is not True
        or provenance["prediction_manifest_sha256"] != prediction_hash
        or provenance["event_diagnostics_sha256"] != sha256(diagnostic_path)
        or provenance["science_scores_sha256"] != sha256(scores / "science_scores.json")
        or provenance["catalog_identity"] != prepared["catalog_identity"]
        or datetime.fromisoformat(provenance["truth_cutoff_utc"])
        != datetime.fromisoformat(manifest["truth_cutoff_utc"])
        or any(
            provenance[name] is not False
            for name in ("model_refitted", "hits_recomputed", "new_evaluation_role_accessed")
        )
    ):
        raise ValueError("strata require the same completed prediction, score and case evidence")
    protocol_path = project / "configs/multitask_s3_anomaly.yaml"
    if sha256(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("frozen S3 protocol changed")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    cutoff = datetime.fromisoformat(manifest["truth_cutoff_utc"])
    issues = tuple(datetime.fromisoformat(value) for value in manifest["issue_times_utc"])
    expected = {}
    for fold in FOLDS:
        for horizon in HORIZONS:
            calendar = build_fold_calendar(
                issues, fold_id=fold, horizon_days=horizon, truth_cutoff=cutoff
            )
            for issue in calendar.evaluation_issues:
                for band in BANDS:
                    expected[(fold, horizon, band, issue.isoformat())] = (
                        issue in calendar.primary_evaluation_issues
                    )
    records = diagnostics["records"]
    observed = {
        (r["fold_id"], r["horizon_days"], r["magnitude_band"], r["issue_time_utc"]): r[
            "primary_nonoverlap"
        ]
        for r in records
    }
    if len(observed) != len(records) or observed != expected:
        raise ValueError(
            "saved diagnostics must retain the complete frozen calendar, including empty windows"
        )
    catalog_path = data_root / protocol["access"]["catalog"]
    if verify_authoritative_catalog_identity(catalog_path) != prepared["catalog_identity"]:
        raise ValueError("episode metadata catalog differs from original trial")
    frame = load_development_catalog(catalog_path, truth_cutoff=cutoff)
    membership = episode_membership(frame)
    metadata = frame.set_index("event_id")
    for event_id, event in ledger["events"].items():
        original = metadata.loc[event_id]
        if any(
            float(event[k]) != float(original[k]) for k in ("longitude", "latitude", "magnitude")
        ) or any(
            pd.Timestamp(event[k]) != original[k] for k in ("origin_time_utc", "available_at")
        ):
            raise ValueError("stored event metadata differs from the original bounded catalog")
    for record in records:
        for event_id, anchor in zip(record["target_event_ids"], record["anchor_mask"], strict=True):
            if (
                membership[event_id]["magnitude_band"] != record["magnitude_band"]
                or membership[event_id]["is_anchor"] != anchor
            ):
                raise ValueError("saved anchors differ from the same full-history episode rule")
    domain, _, area_hash = load_verified_spatial_inputs(data_root)
    grid = domain.operational_grid
    if (
        grid.grid_id != prepared["grid_id"]
        or grid.cell_count != prepared["grid_cells"]
        or area_hash != prepared["study_area_sha256"]
    ):
        raise ValueError("strata grid differs from the frozen prediction grid")
    mapping_path = data_root / MAPPING_PATH
    if sha256(mapping_path) != MAPPING_SHA256:
        raise ValueError("frozen construction-zone mapping changed")
    mapping = pq.read_table(
        mapping_path, columns=["cell_id", "construction_zone_id"], use_threads=False
    ).to_pandas()
    region_by_cell = align_region_aliases(mapping, grid.cell_ids)
    rows = attach_membership(ledger["rows"], membership, region_by_cell)
    member_counts = {
        info["episode_id"]: info["global_member_count"] for info in membership.values()
    }
    identity = {
        "prediction_manifest_sha256": prediction_hash,
        "case_ledger_sha256": sha256(ledger_path),
        "event_diagnostics_sha256": sha256(diagnostic_path),
        "mapping_sha256": MAPPING_SHA256,
        "protocol_sha256": prepared["protocol_sha256"],
        "catalog_identity": prepared["catalog_identity"],
        "truth_cutoff_utc": manifest["truth_cutoff_utc"],
        "grid_id": grid.grid_id,
        "implementation_sha256": {
            name: sha256(project / f"src/seismoflux/multitask_s3/{name}.py")
            for name in ("strata_runner", "strata_summary", "strata_uncertainty")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "strata_progress.json"
    state = (
        read(state_path)
        if resume
        else {
            "identity": identity,
            "status": "summarizing",
            "completed": {},
            "active_pid": os.getpid(),
        }
    )
    if state["identity"] != identity:
        raise ValueError("resume must retain identical saved evidence and summarization code")
    state["active_pid"] = os.getpid()
    write_json(state_path, state)
    for fold_scope in (*FOLDS, "POOLED_A_DEVELOPMENT"):
        for horizon in HORIZONS:
            key = f"{fold_scope}__h{horizon:03d}"
            if key in state["completed"]:
                saved = state["completed"][key]
                if sha256(output / saved["file"]) != saved["sha256"]:
                    raise ValueError("saved strata checkpoint changed")
                continue
            block = summarize_block(
                rows,
                records,
                fold_scope=fold_scope,
                horizon=horizon,
                budgets=protocol["support"]["alarm_area_budgets_km2"],
                all_regions=sorted(set(region_by_cell)),
                member_counts=member_counts,
            )
            block.update({"local_only": True, "fold_scope": fold_scope, "horizon_days": horizon})
            path = output / f"{key}.json"
            if path.exists():
                if read(path) != block:
                    raise ValueError(
                        "unregistered strata checkpoint differs; preserve it for review"
                    )
            else:
                write_json(path, block)
            state["completed"][key] = {
                "file": path.name,
                "sha256": sha256(path),
                "status": block["status"],
                "summary_rows": len(block["rows"]),
            }
            state["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
            write_json(state_path, state)
            print(f"Strata {key}: {len(block['rows'])} task/view summaries saved", flush=True)
    state["status"] = "strata_complete"
    state["active_pid"] = None
    write_json(state_path, state)
    result = {
        "status": "strata_complete",
        "local_only": True,
        "identity": identity,
        "completed": state["completed"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "model_refitted": False,
        "hits_recomputed": False,
        "new_evaluation_role_accessed": False,
        "null_scores_read": False,
        "regions": sorted(set(region_by_cell)),
        "interpretation": (
            "same_saved_predictions; 1/full_bounded_episode_size; "
            "regional_temporal_development_not_spatial_test; intervals_not_adoption_gates"
        ),
    }
    write_json(final_path, result)
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("strata summaries require one numerical-library thread")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    print(json.dumps(summarize_trial(**vars(args)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
