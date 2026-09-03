"""Thin local-only S3 scoring of already completed, immutable predictions.

This produces initial historical-development effects, not complete S3 acceptance.
It never fits a model, changes predictions, opens locked roles, or applies an
adoption threshold. All output, including event diagnostics, remains local.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from numpy.typing import NDArray
from pyproj import Transformer
from shapely.strtree import STRtree

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s0 import verify_authoritative_catalog_identity
from seismoflux.multitask_s1.c2b_score import projected_near_cells
from seismoflux.multitask_s1.runner_inputs import load_verified_spatial_inputs
from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, S3FoldCalendar, build_fold_calendar
from seismoflux.multitask_s3.input_waterlevel import load_development_catalog
from seismoflux.multitask_s3.preparation import sha256, write_json
from seismoflux.multitask_s3.runner import (
    BANDS,
    COUNT_VARIANTS,
    SPATIAL_VARIANTS,
    read_prediction_block,
    verify_complete_predictions,
)
from seismoflux.multitask_s3.scoring import (
    pair_spatial,
    score_count,
    score_spatial,
    summarize_spatial,
)
from seismoflux.multitask_s3.targets import build_window_targets, prepare_anchor_ids
from seismoflux.stage2s.contracts import SpatialGrid

SPATIAL_CONTRASTS = (
    *((variant, "CATALOG") for variant in SPATIAL_VARIANTS if variant != "CATALOG"),
    ("CAT_DYN", "CAT_COV"),
    ("CAT_DYN", "CAT_SNAP"),
)
COUNT_CONTRASTS = (
    *((variant, "T0_CAL") for variant in COUNT_VARIANTS if variant != "T0_CAL"),
    ("T0_CAL_DYN", "T0_CAL_COV"),
    ("T0_CAL_DYN", "T0_CAL_SNAP"),
)
PENDING = (
    "episode_balanced_view",
    "regional_strata",
    "200_time_and_200_space_placebos_per_fold",
    "static_figures_and_local_interactive_case_replay",
    "scientific_value_review_and_full_S3_acceptance",
)


def summarize_count(scores_by_issue: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Equal-issue count scores, retaining complete zero-earthquake windows."""
    rows = list(scores_by_issue.values())
    if not rows:
        return {
            "status": "no_issues",
            "issue_count": 0,
            "observed_count_total": None,
            "expected_count_total": None,
            "poisson_log_score_sum": None,
            "poisson_log_score_mean": None,
            "brier_at_least_one_mean": None,
            "count_bias_expected_minus_observed_mean": None,
        }
    finite_scores = all(row["poisson_log_score"] is not None for row in rows)
    log_total = (
        math.fsum(float(row["poisson_log_score"]) for row in rows) if finite_scores else None
    )
    return {
        "status": "summarized" if finite_scores else "contains_nonfinite_poisson_score",
        "issue_count": len(rows),
        "empty_issue_count": sum(int(row["observed_count"] == 0) for row in rows),
        "observed_count_total": sum(int(row["observed_count"]) for row in rows),
        "expected_count_total": math.fsum(float(row["expected_count"]) for row in rows),
        "poisson_log_score_sum": log_total,
        "poisson_log_score_mean": None if log_total is None else log_total / len(rows),
        "brier_at_least_one_mean": math.fsum(float(row["brier_at_least_one"]) for row in rows)
        / len(rows),
        "count_bias_expected_minus_observed_mean": math.fsum(
            float(row["count_bias_expected_minus_observed"]) for row in rows
        )
        / len(rows),
    }


def pair_count(
    candidate: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(candidate) != set(reference) or any(
        candidate[issue]["observed_count"] != reference[issue]["observed_count"]
        for issue in candidate
    ):
        raise ValueError("count contrasts require the same issues and observed target counts")
    c_summary, r_summary = summarize_count(candidate), summarize_count(reference)
    result: dict[str, Any] = {
        "status": "paired" if candidate else "no_issues",
        "issue_count": len(candidate),
        "candidate": c_summary,
        "reference": r_summary,
        "delta_direction": "candidate_minus_reference; log_score_higher_better_brier_lower_better",
    }
    for field in (
        "poisson_log_score_sum",
        "poisson_log_score_mean",
        "brier_at_least_one_mean",
        "count_bias_expected_minus_observed_mean",
    ):
        c_value, r_value = c_summary[field], r_summary[field]
        result[f"delta_{field}"] = (
            float(c_value) - float(r_value) if c_value is not None and r_value is not None else None
        )
    return result


def summarize_axis(records: Sequence[dict[str, Any]], *, primary: bool) -> dict[str, Any]:
    selected = [row for row in records if row["primary_nonoverlap"] or not primary]
    spatial = {
        variant: {row["issue_key"]: row["spatial"][variant] for row in selected}
        for variant in SPATIAL_VARIANTS
    }
    counts = {
        variant: {row["issue_key"]: row["count"][variant] for row in selected}
        for variant in COUNT_VARIANTS
    }
    if len({row["issue_key"] for row in selected}) != len(selected):
        raise ValueError("duplicate exposure in one horizon/band summary")
    event_ids = {identifier for row in selected for identifier in row["target_event_ids"]}
    anchors = {
        identifier
        for row in selected
        for identifier, anchor in zip(row["target_event_ids"], row["anchor_mask"], strict=True)
        if anchor
    }
    return {
        "status": "summarized" if selected else "no_complete_issues_NA",
        "axis": "primary_nonoverlap" if primary else "all_reports_descriptive",
        "independence_note": (
            "nonoverlapping_h_plus_30d_issue_axis_not_proof_of_independent_earthquakes"
            if primary
            else "overlapping_windows_occurrences_not_independent_samples"
        ),
        "sample_counts": {
            "issue_count": len(selected),
            "event_occurrences": sum(len(row["target_event_ids"]) for row in selected),
            "unique_events": len(event_ids),
            "anchor_occurrences": sum(sum(row["anchor_mask"]) for row in selected),
            "unique_anchors": len(anchors),
        },
        "spatial": {variant: summarize_spatial(rows) for variant, rows in spatial.items()},
        "count": {variant: summarize_count(rows) for variant, rows in counts.items()},
        "spatial_contrasts": {
            f"{candidate}_minus_{reference}": pair_spatial(
                spatial[candidate], spatial[reference], bootstrap_nonoverlapping_issues=primary
            )
            for candidate, reference in SPATIAL_CONTRASTS
        },
        "count_contrasts": {
            f"{candidate}_minus_{reference}": pair_count(counts[candidate], counts[reference])
            for candidate, reference in COUNT_CONTRASTS
        },
    }


def score_prediction_block(
    prediction: Mapping[str, Any],
    calendar: S3FoldCalendar,
    *,
    frame: pd.DataFrame,
    cell_indices: NDArray[np.int64],
    anchor_ids: Mapping[str, set[str]],
    near_cells_by_event: Mapping[str, set[int]],
    grid: SpatialGrid,
    budgets_km2: Sequence[float],
    truth_cutoff: datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pure block wiring after the caller verifies all prediction files first."""
    metadata = prediction["metadata"]
    if (
        metadata["spatial_variants"] != list(SPATIAL_VARIANTS)
        or metadata["count_variants"] != list(COUNT_VARIANTS)
        or metadata["magnitude_bands"] != list(BANDS)
    ):
        raise ValueError("saved variant or magnitude order differs from frozen S3")
    n = len(calendar.evaluation_issues)
    if prediction["spatial_log_mass"].shape != (n, 5, grid.cell_count) or prediction[
        "count_log_mean"
    ].shape != (n, 5, 2):
        raise ValueError("prediction arrays do not match this complete block")
    local: list[dict[str, Any]] = []
    primary_issues = set(calendar.primary_evaluation_issues)
    for index, issue in enumerate(calendar.evaluation_issues):
        labels = build_window_targets(
            frame,
            issue_time=issue,
            horizon_days=calendar.horizon_days,
            available_by=truth_cutoff,
            cell_indices=cell_indices,
            cell_count=grid.cell_count,
            anchor_ids_by_band=anchor_ids,
        )
        for band_index, band in enumerate(BANDS):
            target = labels.bands[band]
            nearby = [near_cells_by_event[event_id] for event_id in target.event_ids]
            local.append(
                {
                    "fold_id": calendar.fold_id,
                    "horizon_days": calendar.horizon_days,
                    "magnitude_band": band,
                    "issue_time_utc": issue.isoformat(),
                    "issue_key": f"{calendar.fold_id}|{issue.isoformat()}",
                    "primary_nonoverlap": issue in primary_issues,
                    "target_event_ids": list(target.event_ids),
                    "target_cell_indices": target.cell_indices.tolist(),
                    "anchor_mask": target.anchor_mask.tolist(),
                    "spatial": {
                        variant: score_spatial(
                            prediction["spatial_log_mass"][index, variant_index],
                            targets=target,
                            grid=grid,
                            budgets_km2=budgets_km2,
                            near_cells=nearby,
                        )
                        for variant_index, variant in enumerate(SPATIAL_VARIANTS)
                    },
                    "count": {
                        variant: score_count(
                            float(prediction["count_log_mean"][index, variant_index, band_index]),
                            target.event_count,
                        )
                        for variant_index, variant in enumerate(COUNT_VARIANTS)
                    },
                }
            )
    summary = {
        "fold_id": calendar.fold_id,
        "horizon_days": calendar.horizon_days,
        "status": "scored" if n else "unavailable_no_complete_outer_window",
        "training_and_inner_metadata": metadata["models"],
        "bands": {
            band: {
                axis: summarize_axis(
                    [row for row in local if row["magnitude_band"] == band], primary=primary
                )
                for axis, primary in (
                    ("primary_nonoverlap", True),
                    ("all_reports_descriptive", False),
                )
            }
            for band in BANDS
        },
    }
    return summary, local


def score_trial(
    *, project_root: Path, data_root: Path, prediction_dir: Path, output_dir: Path
) -> dict[str, Any]:
    project, prediction, output = (
        project_root.resolve(),
        prediction_dir.resolve(),
        output_dir.resolve(),
    )
    allowed = project / "outputs/multitask_s3"
    if (
        not prediction.is_relative_to(allowed)
        or not output.is_relative_to(allowed)
        or output == prediction
    ):
        raise ValueError("prediction and scoring must use distinct local S3 output directories")
    manifest_path = prediction / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = tuple(datetime.fromisoformat(value) for value in manifest["issue_times_utc"])
    truth_cutoff = datetime.fromisoformat(manifest["truth_cutoff_utc"])
    calendars = {
        f"{fold}__h{horizon:03d}": build_fold_calendar(
            issues, fold_id=fold, horizon_days=horizon, truth_cutoff=truth_cutoff
        )
        for fold in FOLDS
        for horizon in HORIZONS
    }
    # This is deliberately before any catalog, target, or coordinate access.
    verify_complete_predictions(prediction, manifest, calendars)
    if output.exists():
        raise FileExistsError("preserve existing local scores; choose an explicit new output path")
    identity = manifest["identity"]
    prepared = identity["prepared_inputs"]
    protocol_path = project / "configs/multitask_s3_anomaly.yaml"
    if sha256(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("frozen protocol changed since prediction")
    for name, expected in identity["implementation_sha256"].items():
        if sha256(project / f"src/seismoflux/{name}.py") != expected:
            raise ValueError("prediction implementation changed; do not silently reinterpret it")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    catalog_path = data_root / protocol["access"]["catalog"]
    if verify_authoritative_catalog_identity(catalog_path) != prepared["catalog_identity"]:
        raise ValueError("evaluation catalog differs from prediction inputs")
    domain, _, area_hash = load_verified_spatial_inputs(data_root)
    grid = domain.operational_grid
    if (
        grid.grid_id != prepared["grid_id"]
        or grid.cell_count != prepared["grid_cells"]
        or area_hash != prepared["study_area_sha256"]
    ):
        raise ValueError("evaluation grid differs from prediction inputs")
    frame = load_development_catalog(catalog_path, truth_cutoff=truth_cutoff)
    positions = [
        domain.locator.locate_lonlat(float(lon), float(lat))
        for lon, lat in zip(frame["longitude"], frame["latitude"], strict=True)
    ]
    cell_indices = np.array([-1 if value is None else value for value in positions], dtype=np.int64)
    anchors = prepare_anchor_ids(frame)
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    x, y = transformer.transform(frame["longitude"].to_numpy(), frame["latitude"].to_numpy())
    neighborhoods = projected_near_cells(STRtree(domain.locator.clipped_geometries), x, y)
    nearby = dict(zip(frame["event_id"].astype(str), neighborhoods, strict=True))
    output.mkdir(parents=True)
    progress: dict[str, Any] = {
        "status": "scoring",
        "local_only": True,
        "active_pid": os.getpid(),
        "prediction_manifest_sha256": sha256(manifest_path),
        "completed_blocks": [],
        "total_blocks": len(calendars),
    }
    write_json(output / "score_progress.json", progress)
    blocks: dict[str, Any] = {}
    local_records: list[dict[str, Any]] = []
    for key, calendar in calendars.items():
        saved = read_prediction_block(
            prediction / manifest["completed"][key]["file"], identity=identity, calendar=calendar
        )
        blocks[key], rows = score_prediction_block(
            saved,
            calendar,
            frame=frame,
            cell_indices=cell_indices,
            anchor_ids=anchors,
            near_cells_by_event=nearby,
            grid=grid,
            budgets_km2=protocol["support"]["alarm_area_budgets_km2"],
            truth_cutoff=truth_cutoff,
        )
        local_records.extend(rows)
        progress["completed_blocks"].append(key)
        progress["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
        write_json(output / "score_progress.json", progress)
        print(f"Scored {key}; predictions unchanged.", flush=True)
    summary = {
        "status": "initial_development_effects_complete_S3_not_complete",
        "local_only": True,
        "scope": "registered_A_historical_development_not_independent_test",
        "prediction_identity": identity,
        "prediction_manifest_sha256": sha256(manifest_path),
        "truth_cutoff_utc": truth_cutoff.isoformat(),
        "scoring_sha256": sha256(Path(__file__)),
        "blocks": blocks,
        "pooled_folds": {
            f"h{horizon:03d}__{band}": {
                axis: summarize_axis(
                    [
                        row
                        for row in local_records
                        if row["horizon_days"] == horizon and row["magnitude_band"] == band
                    ],
                    primary=primary,
                )
                for axis, primary in (
                    ("primary_nonoverlap", True),
                    ("all_reports_descriptive", False),
                )
            }
            for horizon in HORIZONS
            for band in BANDS
        },
        "pending": list(PENDING),
        "adoption_threshold": None,
    }
    write_json(
        output / "event_diagnostics_local.json", {"local_only": True, "records": local_records}
    )
    write_json(output / "science_scores.json", summary)
    progress.update(
        status="complete", active_pid=None, last_checkpoint_utc=datetime.now(UTC).isoformat()
    )
    write_json(output / "score_progress.json", progress)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "data-root", "prediction-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("launch numerical libraries with one thread each")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    score_trial(
        project_root=args.project_root,
        data_root=args.data_root,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
    )
    print("Initial S3 historical effects saved locally; additional S3 diagnostics remain pending.")


if __name__ == "__main__":
    main()
