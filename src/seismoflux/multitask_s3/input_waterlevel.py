"""Bounded S3-A input integration and sample counts, with no candidate scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import yaml

from seismoflux.multitask_s0 import (
    CATALOG_COLUMNS,
    build_episodes,
    verify_authoritative_catalog_identity,
)
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_HISTORY_START_UTC,
    catalog_event_table_from_frame,
    load_verified_spatial_inputs,
)
from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, REPORT_END, build_fold_calendar
from seismoflux.multitask_s3.catalog_background import build_catalog_background_components
from seismoflux.multitask_s3.features import load_issue_features, read_report_issue_metadata


def load_development_catalog(path: Path, *, truth_cutoff: datetime) -> pd.DataFrame:
    """Read only authorized historical conditioning/development rows and fields."""
    selection = (
        (ds.field("origin_time_utc") >= pa.scalar(CATALOG_HISTORY_START_UTC))
        & (ds.field("origin_time_utc") < pa.scalar(REPORT_END))
        & (ds.field("available_at") <= pa.scalar(truth_cutoff))
        & (ds.field("inside_study_area") == True)  # noqa: E712
        & (ds.field("magnitude") >= 4.0)
    )
    frame = (
        ds.dataset(path, format="parquet")
        .to_table(columns=list(CATALOG_COLUMNS), filter=selection, use_threads=False)
        .to_pandas()
    )
    for name in ("origin_time_utc", "available_at"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if (
        not frame.empty
        and not (
            (frame.origin_time_utc >= CATALOG_HISTORY_START_UTC)
            & (frame.origin_time_utc < REPORT_END)
            & (frame.available_at <= truth_cutoff)
        ).all()
    ):
        raise ValueError("bounded catalog reader returned unauthorized rows")
    return frame.sort_values(["origin_time_utc", "event_id"], kind="mergesort", ignore_index=True)


def summarize_windows(
    frame: pd.DataFrame,
    issues: tuple[datetime, ...],
    horizon: int,
    available_by: datetime,
    anchors: dict[str, set[str]],
) -> dict[str, object]:
    """Counts only; distinguish overlapping occurrences from unique earthquakes."""
    allowed = frame[frame.available_at <= available_by]
    unique: dict[str, set[str]] = {"Ms4_plus": set(), "Ms5_6": set(), "Ms6_plus": set()}
    occurrences = {name: 0 for name in unique}
    empty = {name: 0 for name in unique}
    for issue in issues:
        window = allowed[
            (allowed.origin_time_utc > issue)
            & (allowed.origin_time_utc <= issue + timedelta(days=horizon))
        ]
        panels = {
            "Ms4_plus": window,
            "Ms5_6": window[(window.magnitude >= 5) & (window.magnitude < 6)],
            "Ms6_plus": window[window.magnitude >= 6],
        }
        for band, panel in panels.items():
            identifiers = set(panel.event_id.astype(str))
            unique[band].update(identifiers)
            occurrences[band] += len(identifiers)
            empty[band] += int(not identifiers)
    return {
        "issue_count": len(issues),
        "first_issue_utc": issues[0].isoformat() if issues else None,
        "last_issue_utc": issues[-1].isoformat() if issues else None,
        "event_occurrences": occurrences,
        "unique_events": {name: len(ids) for name, ids in unique.items()},
        "empty_issues": empty,
        "fixed_first_anchor_events": {
            name: len(unique[name] & ids) for name, ids in anchors.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if args.output_dir.exists():
        raise FileExistsError("preserve prior input checks; select a new named output directory")
    protocol_path = args.project_root / "configs/multitask_s3_anomaly.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    source = args.data_root / protocol["access"]["feature_store"]
    catalog_path = args.data_root / protocol["access"]["catalog"]
    s0_ledger = json.loads(
        (
            args.project_root
            / "outputs/multitask_s0/s0_score_blind_20260901/catalog_sample_ledger.json"
        ).read_text(encoding="utf-8")
    )
    truth_cutoff = datetime.fromisoformat(
        s0_ledger["truth_cutoff_derivation"]["maximum_origin_time_utc"].replace("Z", "+00:00")
    )
    print("Verifying fixed source bytes and authorized report metadata.", flush=True)
    with source.open("rb") as handle:
        feature_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if feature_hash != protocol["access"]["feature_store_sha256"]:
        raise ValueError("S3 feature source identity mismatch")
    catalog_identity = verify_authoritative_catalog_identity(catalog_path)
    metadata = read_report_issue_metadata(source, report_end_exclusive=REPORT_END)
    issues = tuple(item.issue_time_utc for item in metadata)
    if not issues:
        raise ValueError("no authorized real report dates")
    print(f"Authorized report dates: {len(issues)}. Reading bounded catalog columns.", flush=True)
    frame = load_development_catalog(catalog_path, truth_cutoff=truth_cutoff)
    anchors: dict[str, set[str]] = {}
    for name, lower, upper in [("Ms5_6", 5.0, 6.0), ("Ms6_plus", 6.0, float("inf"))]:
        panel = frame[(frame.magnitude >= lower) & (frame.magnitude < upper)]
        anchors[name] = {str(e["anchor_event_id"]) for e in build_episodes(panel)}
    result: dict[str, object] = {
        "status": "input_waterlevel_only_no_new_model_skill",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "feature_store_sha256": feature_hash,
        "catalog_identity": catalog_identity,
        "truth_cutoff_utc_from_frozen_S0_metadata": truth_cutoff.isoformat(),
        "allowed_report_count": len(issues),
        "allowed_report_first": issues[0].isoformat(),
        "allowed_report_last": issues[-1].isoformat(),
        "bounded_inside_Ms4_catalog_rows": len(frame),
        "folds": {},
        "scores_read_or_computed": False,
    }
    folds = {}
    for fold_id in FOLDS:
        folds[fold_id] = {}
        for horizon in HORIZONS:
            c = build_fold_calendar(
                issues, fold_id=fold_id, horizon_days=horizon, truth_cutoff=truth_cutoff
            )
            folds[fold_id][str(horizon)] = {
                "training": summarize_windows(
                    frame, c.training_issues, horizon, c.label_fit_cutoff, anchors
                ),
                "evaluation_primary": summarize_windows(
                    frame, c.primary_evaluation_issues, horizon, truth_cutoff, anchors
                ),
                "evaluation_all_reports_descriptive": summarize_windows(
                    frame, c.evaluation_issues, horizon, truth_cutoff, anchors
                ),
                "inner": {
                    i.block_id: {
                        "training": summarize_windows(
                            frame, i.training_issues, horizon, i.label_fit_cutoff, anchors
                        ),
                        "validation": summarize_windows(
                            frame, i.validation_issues, horizon, c.label_fit_cutoff, anchors
                        ),
                    }
                    for i in c.inner
                },
            }
    result["folds"] = folds
    print(
        "Checking the first real report against independent grid and causal background.",
        flush=True,
    )
    domain, grid, area_hash = load_verified_spatial_inputs(args.data_root)
    first = issues[0]
    features = load_issue_features(
        source,
        issue_times_utc=[first],
        expected_cell_ids=domain.operational_grid.cell_ids,
        expected_grid_id=domain.operational_grid.grid_id,
        report_end_exclusive=FOLDS["A_DEV_2023_2024"][1],
    )[first]
    background = build_catalog_background_components(
        catalog_event_table_from_frame(frame), grid, first
    ).for_horizon(30)
    result["first_report_integration"] = {
        "issue_time_utc": first.isoformat(),
        "grid_id": domain.operational_grid.grid_id,
        "grid_cells": grid.cell_count,
        "study_area_sha256": area_hash,
        "feature_shape": list(features.values.shape),
        "feature_missing_counts": np.isnan(features.values).sum(axis=0).tolist(),
        "spatial_mass_sum": float(np.exp(background.primary_log_mass).sum()),
        "history_counts": dict(background.waterlevel.history_counts),
        "history_cutoff_utc": background.waterlevel.data_cutoff_utc.isoformat(),
        "T0_30day_expected_counts": dict(background.expected_counts),
        "interpretation": "input_integration_only_not_an_anomaly_prediction_or_evaluation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "input_waterlevel.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": result["status"],
                "reports": len(issues),
                "path": str(args.output_dir / "input_waterlevel.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
