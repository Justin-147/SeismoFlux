"""Fit the frozen S3-A trial from prepared reports, saving predictions before scores.

No outer target windows or effect scores are evaluated here. Each fold/horizon
is a resumable local-only prediction block, including explicit unavailable rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from scipy.special import logsumexp  # type: ignore[import-untyped]

from seismoflux.multitask_s0 import verify_authoritative_catalog_identity
from seismoflux.multitask_s1.c2b_models import mix_log_masses
from seismoflux.multitask_s1.runner_inputs import load_verified_spatial_inputs
from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, S3FoldCalendar, build_fold_calendar
from seismoflux.multitask_s3.input_waterlevel import load_development_catalog
from seismoflux.multitask_s3.preparation import read_issue_cache, sha256, write_json
from seismoflux.multitask_s3.targets import build_window_targets, prepare_anchor_ids
from seismoflux.multitask_s3.training import (
    Design,
    S3InnerBlock,
    S3TrainingSample,
    select_and_fit,
)

SPATIAL_VARIANTS = ("CATALOG", "R30_REFERENCE", "CAT_COV", "CAT_SNAP", "CAT_DYN")
COUNT_VARIANTS = ("T0", "T0_CAL", "T0_CAL_COV", "T0_CAL_SNAP", "T0_CAL_DYN")
BANDS = ("Ms5_6", "Ms6_plus")
DESIGNS: tuple[Design, ...] = ("COV", "SNAP", "DYN")


def calendar_metadata(calendar: S3FoldCalendar) -> dict[str, Any]:
    # Round-trip converts all nested dataclass times, not labels or target records.
    return dict(json.loads(json.dumps(asdict(calendar), default=lambda value: value.isoformat())))


def horizon_background(
    cache: Mapping[str, Any], horizon_days: int
) -> tuple[np.ndarray, np.ndarray]:
    if horizon_days not in HORIZONS:
        raise ValueError("unregistered S3 horizon")
    weights = (1.0 / 3,) * 3 if horizon_days == 365 else (0.5, 0.5, 0.0)
    mass = mix_log_masses([cache[f"kernel_{scale}"] for scale in (25, 75, 150)], weights)
    rates = cache["metadata"]["expected_counts_per_day"]
    means = np.array([float(rates[band]) * horizon_days for band in BANDS])
    total = float(rates["Ms5_plus"]) * horizon_days
    if (
        not np.isfinite(means).all()
        or np.any(means <= 0)
        or not math.isfinite(total)
        or not math.isclose(float(means.sum()), total, rel_tol=1e-12, abs_tol=1e-14)
    ):
        raise ValueError("prepared count offsets must be positive disjoint formal-band rates")
    return mass, means


def predict_block(
    calendar: S3FoldCalendar,
    *,
    caches: Mapping[datetime, Mapping[str, Any]],
    frame: pd.DataFrame,
    cell_indices: np.ndarray,
    anchor_ids: Mapping[str, set[str]],
    areas_km2: np.ndarray,
) -> dict[str, Any]:
    """Fit only training/inner labels; predict outer features without outer labels."""
    size = len(areas_km2)
    issues = calendar.evaluation_issues
    metadata: dict[str, Any] = {
        "calendar": calendar_metadata(calendar),
        "spatial_variants": list(SPATIAL_VARIANTS),
        "count_variants": list(COUNT_VARIANTS),
        "magnitude_bands": list(BANDS),
        "outer_effect_scores_computed": False,
        "local_only": True,
        "models": {},
        "status": "predictions_complete" if issues else "unavailable_no_complete_outer_window",
        "shared_band_multiplier": True,
        "learns_magnitude_distribution": False,
    }
    spatial = np.empty((len(issues), len(SPATIAL_VARIANTS), size), dtype=np.float64)
    counts = np.empty((len(issues), len(COUNT_VARIANTS), len(BANDS)), dtype=np.float64)
    if not issues:
        # No prediction can be evaluated at this horizon. Preserve NA, not zero skill.
        return {"metadata": metadata, "spatial_log_mass": spatial, "count_log_mean": counts}

    needed = set(calendar.training_issues) | set(issues)
    for inner in calendar.inner:
        needed.update(inner.training_issues)
        needed.update(inner.validation_issues)
    backgrounds = {
        issue: horizon_background(caches[issue], calendar.horizon_days) for issue in needed
    }
    sample_cache: dict[tuple[datetime, datetime], S3TrainingSample] = {}

    def sample(issue: datetime, cutoff: datetime) -> S3TrainingSample:
        key = (issue, cutoff)
        if key not in sample_cache:
            labels = build_window_targets(
                frame,
                issue_time=issue,
                horizon_days=calendar.horizon_days,
                available_by=cutoff,
                cell_indices=cell_indices,
                cell_count=size,
                anchor_ids_by_band=anchor_ids,
            )
            background, means = backgrounds[issue]
            sample_cache[key] = S3TrainingSample(
                issue,
                caches[issue]["features"],
                background,
                float(means.sum()),
                labels.spatial_counts_ms4.astype(np.float64),
                labels.count_ms5plus,
            )
        return sample_cache[key]

    training = tuple(sample(issue, calendar.label_fit_cutoff) for issue in calendar.training_issues)
    inner_blocks = tuple(
        S3InnerBlock(
            inner.block_id,
            tuple(sample(issue, inner.label_fit_cutoff) for issue in inner.training_issues),
            tuple(sample(issue, calendar.label_fit_cutoff) for issue in inner.validation_issues),
        )
        for inner in calendar.inner
    )
    fits = {
        design: select_and_fit(
            training, inner_blocks=inner_blocks, design=design, areas_km2=areas_km2
        )
        for design in DESIGNS
    }
    metadata["models"] = {design: fit.to_dict() for design, fit in fits.items()}
    for row, issue in enumerate(issues):
        features = caches[issue]["features"]
        background, means = backgrounds[issue]
        total = float(means.sum())
        band_log_ratio = np.log(means) - math.log(total)
        spatial[row, 0] = background
        spatial[row, 1] = caches[issue]["r30_log_mass"]
        counts[row, 0] = np.log(means)
        counts[row, 1] = fits["COV"].predict_calibrated_log_mean(total) + band_log_ratio
        for column, design in enumerate(DESIGNS, start=2):
            spatial[row, column] = fits[design].predict_log_mass(features, background)
            counts[row, column] = fits[design].predict_log_mean(features, total) + band_log_ratio
    if not np.isfinite(spatial).all() or not np.isfinite(counts).all():
        raise FloatingPointError("predictions must preserve finite log masses and log means")
    if not np.allclose(logsumexp(spatial, axis=2), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("predicted spatial masses are not normalized")
    return {"metadata": metadata, "spatial_log_mass": spatial, "count_log_mean": counts}


def write_prediction_block(path: Path, result: dict[str, Any], identity: dict[str, Any]) -> None:
    metadata = {**result["metadata"], "identity": identity}
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.array(json.dumps(metadata, allow_nan=False)),
        spatial_log_mass=result["spatial_log_mass"],
        count_log_mean=result["count_log_mean"],
    )
    os.replace(temporary, path)


def read_prediction_block(
    path: Path, *, identity: dict[str, Any], calendar: S3FoldCalendar
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        spatial = np.array(payload["spatial_log_mass"])
        counts = np.array(payload["count_log_mean"])
    if metadata["identity"] != identity or metadata["calendar"] != calendar_metadata(calendar):
        raise ValueError("prediction block belongs to another frozen trial or calendar")
    n = len(calendar.evaluation_issues)
    cells = int(identity["prepared_inputs"]["grid_cells"])
    if spatial.shape != (n, 5, cells) or counts.shape != (n, 5, 2):
        raise ValueError("prediction block does not cover every registered variant and issue")
    if not np.isfinite(spatial).all() or not np.isfinite(counts).all():
        raise ValueError("invalid prediction log values")
    if n and not np.allclose(logsumexp(spatial, axis=2), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("prediction log masses are not normalized")
    return {"metadata": metadata, "spatial_log_mass": spatial, "count_log_mean": counts}


def verify_complete_predictions(
    output: Path, manifest: dict[str, Any], calendars: Mapping[str, S3FoldCalendar]
) -> None:
    """The later scoring entry point must call this before constructing outer labels."""
    if manifest["status"] != "predictions_complete" or set(manifest["completed"]) != set(calendars):
        raise ValueError("all registered prediction blocks must be saved before any outer scoring")
    for key, calendar in calendars.items():
        record = manifest["completed"][key]
        path = output / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("saved prediction changed; preserve it and investigate")
        read_prediction_block(path, identity=manifest["identity"], calendar=calendar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "data-root", "prepared-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(
        os.environ.get(key) != "1"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("launch numerical libraries with one thread each")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    prepared = args.prepared_dir.resolve()
    allowed = project / "outputs/multitask_s3"
    if (
        not output.is_relative_to(allowed)
        or not prepared.is_relative_to(allowed)
        or output == prepared
    ):
        raise ValueError("prediction and preparation must use distinct local S3 output directories")
    source_manifest = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    if source_manifest["status"] != "complete" or source_manifest.get("failures"):
        raise ValueError("actual-date preparation must finish before training")
    prepared_identity = source_manifest["identity"]
    protocol_path = project / "configs/multitask_s3_anomaly.yaml"
    if sha256(protocol_path) != prepared_identity["protocol_sha256"]:
        raise ValueError("protocol changed since preparation")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    identity = {
        "prepared_inputs": prepared_identity,
        "prepared_report_sha256": {
            issue: record["sha256"]
            for issue, record in sorted(source_manifest["completed"].items())
        },
        "implementation_sha256": {
            name: sha256(project / f"src/seismoflux/{name}.py")
            for name in (
                "multitask_s3/runner",
                "multitask_s3/calendar",
                "multitask_s3/targets",
                "multitask_s3/training",
                "multitask_s3/models",
                "multitask_s3/features",
                "multitask_s1/c2b_models",
            )
        },
    }
    issues = tuple(datetime.fromisoformat(value) for value in source_manifest["issue_times_utc"])
    if set(source_manifest["completed"]) != {issue.isoformat() for issue in issues}:
        raise ValueError("prepared report inventory is incomplete")
    truth_cutoff = datetime.fromisoformat(source_manifest["truth_cutoff_utc"])
    calendars = {
        f"{fold}__h{horizon:03d}": build_fold_calendar(
            issues, fold_id=fold, horizon_days=horizon, truth_cutoff=truth_cutoff
        )
        for fold in FOLDS
        for horizon in HORIZONS
    }
    if output.exists() and not args.resume:
        raise FileExistsError("resume this same named trial instead of duplicating it")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "prediction.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    manifest_path = output / "prediction_manifest.json"
    manifest: dict[str, Any] = {
        "status": "initializing",
        "identity": identity,
        "active_pid": os.getpid(),
        "workers": min(args.workers, len(FOLDS)),
        "completed": {},
        "failures": {},
        "total_blocks": len(calendars),
        "outer_effect_scores_computed": False,
        "local_only": True,
        "truth_cutoff_utc": truth_cutoff.isoformat(),
        "issue_times_utc": [issue.isoformat() for issue in issues],
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
    }
    try:
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous["identity"] != identity:
                raise ValueError(
                    "resume requires the same frozen inputs and fitting implementation"
                )
            manifest["completed"] = previous["completed"]
            if previous["status"] == "predictions_complete":
                verify_complete_predictions(output, previous, calendars)
                print(
                    "All predictions already saved; do not refit or inspect scores here.",
                    flush=True,
                )
                return
        for key, record in manifest["completed"].items():
            if key not in calendars or sha256(output / record["file"]) != record["sha256"]:
                raise ValueError("prior completed block changed")
            read_prediction_block(
                output / record["file"], identity=identity, calendar=calendars[key]
            )
        catalog_path = args.data_root / protocol["access"]["catalog"]
        if (
            verify_authoritative_catalog_identity(catalog_path)
            != prepared_identity["catalog_identity"]
        ):
            raise ValueError("canonical catalog differs from prepared background")
        domain, grid, area_hash = load_verified_spatial_inputs(args.data_root)
        if (
            domain.operational_grid.grid_id != prepared_identity["grid_id"]
            or grid.cell_count != prepared_identity["grid_cells"]
            or area_hash != prepared_identity["study_area_sha256"]
        ):
            raise ValueError("independent grid differs from prepared features")
        frame = load_development_catalog(catalog_path, truth_cutoff=truth_cutoff)
        positions = [
            domain.locator.locate_lonlat(float(lon), float(lat))
            for lon, lat in zip(frame["longitude"], frame["latitude"], strict=True)
        ]
        cell_indices = np.array(
            [-1 if value is None else value for value in positions], dtype=np.int64
        )
        anchor_ids = prepare_anchor_ids(frame)
        caches = {}
        for issue in issues:
            record = source_manifest["completed"][issue.isoformat()]
            path = prepared / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError("prepared report changed; do not recompute or silently substitute")
            caches[issue] = read_issue_cache(path, issue_time=issue, identity=prepared_identity)
            for array in caches[issue].values():
                if isinstance(array, np.ndarray):
                    array.setflags(write=False)
        manifest["status"] = "predicting"
        write_json(manifest_path, manifest)
        # Only two fold chains run concurrently. Horizons in one fold stay sequential.
        checkpoint_lock = Lock()

        def run_fold(fold: str) -> None:
            for horizon in HORIZONS:
                key = f"{fold}__h{horizon:03d}"
                if key in manifest["completed"]:
                    continue
                path = output / f"{key}.npz"
                if path.exists():
                    result = read_prediction_block(path, identity=identity, calendar=calendars[key])
                else:
                    result = predict_block(
                        calendars[key],
                        caches=caches,
                        frame=frame,
                        cell_indices=cell_indices,
                        anchor_ids=anchor_ids,
                        areas_km2=grid.area_km2,
                    )
                    write_prediction_block(path, result, identity)
                record = {
                    "file": path.name,
                    "sha256": sha256(path),
                    "status": result["metadata"]["status"],
                    "evaluation_issues": len(calendars[key].evaluation_issues),
                    "saved_at_utc": datetime.now(UTC).isoformat(),
                }
                with checkpoint_lock:
                    manifest["completed"][key] = record
                    manifest["completed_blocks"] = len(manifest["completed"])
                    manifest["last_checkpoint_utc"] = record["saved_at_utc"]
                    write_json(manifest_path, manifest)
                print(f"Saved {key}; outer scoring still pending.", flush=True)

        with ThreadPoolExecutor(max_workers=manifest["workers"]) as executor:
            futures = {executor.submit(run_fold, fold): fold for fold in FOLDS}
            for future in as_completed(futures):
                fold = futures[future]
                try:
                    future.result()
                except Exception as error:
                    with checkpoint_lock:
                        manifest["failures"][fold] = f"{type(error).__name__}: {error}"
                        manifest["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
                        write_json(manifest_path, manifest)
        manifest["status"] = (
            "predictions_complete" if len(manifest["completed"]) == len(calendars) else "incomplete"
        )
        manifest["active_pid"] = None
        manifest["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, manifest)
        if manifest["status"] != "predictions_complete":
            raise RuntimeError(
                "prediction blocks incomplete; inspect checkpoint without changing the trial"
            )
        verify_complete_predictions(output, manifest, calendars)
        print(
            "Every outer prediction is saved. Outer scoring is a separate next phase.", flush=True
        )
    except Exception as error:
        manifest["status"] = "incomplete"
        manifest["active_pid"] = None
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
        # Preserve a prior manifest when a resume identity check fails.
        if (
            not manifest_path.exists()
            or json.loads(manifest_path.read_text(encoding="utf-8"))["identity"] == identity
        ):
            write_json(manifest_path, manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
