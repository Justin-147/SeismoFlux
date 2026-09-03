"""Prepare actual A-date features and causal catalog references once per report.

This local-only cache is not the completed input-waterlevel check and does not
fit anomaly parameters or read effect scores. It feeds the frozen S3 training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml

from seismoflux.multitask_s0 import verify_authoritative_catalog_identity
from seismoflux.multitask_s1.runner_inputs import (
    catalog_event_table_from_frame,
    load_verified_spatial_inputs,
)
from seismoflux.multitask_s3.calendar import REPORT_END
from seismoflux.multitask_s3.catalog_background import build_catalog_background_components
from seismoflux.multitask_s3.features import load_issue_features, read_report_issue_metadata
from seismoflux.multitask_s3.input_waterlevel import load_development_catalog


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_issue_cache(
    path: Path,
    *,
    issue_time: datetime,
    identity: dict[str, Any],
    features: np.ndarray,
    kernel_log_masses: dict[float, np.ndarray],
    r30_log_mass: np.ndarray,
    expected_counts_per_day: dict[str, float],
) -> dict[str, Any]:
    metadata = {
        "issue_time_utc": issue_time.astimezone(UTC).isoformat(),
        "identity": identity,
        "expected_counts_per_day": expected_counts_per_day,
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        features=features,
        kernel_25=kernel_log_masses[25.0],
        kernel_75=kernel_log_masses[75.0],
        kernel_150=kernel_log_masses[150.0],
        r30_log_mass=r30_log_mass,
    )
    os.replace(temporary, path)
    return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def read_issue_cache(
    path: Path, *, issue_time: datetime, identity: dict[str, Any]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        if metadata["identity"] != identity:
            raise ValueError("prepared cache belongs to different frozen inputs")
        if metadata["issue_time_utc"] != issue_time.astimezone(UTC).isoformat():
            raise ValueError("prepared cache belongs to another actual report")
        result = {
            key: np.array(payload[key], copy=True)
            for key in payload.files
            if key != "metadata_json"
        }
    size = int(identity["grid_cells"])
    if result["features"].shape != (size, 20) or np.isinf(result["features"]).any():
        raise ValueError("prepared features are not the frozen 20-column grid")
    for name in ("kernel_25", "kernel_75", "kernel_150", "r30_log_mass"):
        values = result[name]
        if values.shape != (size,) or np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("prepared catalog reference has invalid cells")
        if not np.isclose(np.exp(values).sum(), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("prepared catalog reference is not normalized")
    result["metadata"] = metadata
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("launch with OMP/OPENBLAS/MKL_NUM_THREADS=1 to reserve CPU capacity")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = args.output_dir.resolve()
    allowed = (args.project_root / "outputs/multitask_s3").resolve()
    if not output.is_relative_to(allowed):
        raise ValueError("local S3 caches must remain inside the active project outputs")
    if output.exists() and not args.resume:
        raise FileExistsError("use --resume for this same named preparation")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "preparation.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    manifest_path = output / "preparation.json"
    manifest: dict[str, Any] = {
        "phase": "actual_A_date_background_and_features",
        "status": "initializing",
        "active_pid": os.getpid(),
        "workers": args.workers,
        "anomaly_parameters_trained": False,
        "outer_effect_scores_computed": False,
        "local_only": True,
        "completed": {},
    }
    try:
        protocol_path = args.project_root / "configs/multitask_s3_anomaly.yaml"
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        accepted = json.loads(
            (
                args.project_root / "outputs/multitask_s3/s3a_input_v1/input_waterlevel.json"
            ).read_text(encoding="utf-8")
        )
        if sha256(protocol_path) != accepted["protocol_sha256"]:
            raise ValueError("S3 protocol changed since the accepted inputs")
        source = args.data_root / protocol["access"]["feature_store"]
        catalog_path = args.data_root / protocol["access"]["catalog"]
        print("Checking frozen inputs for actual-date reference preparation.", flush=True)
        if sha256(source) != accepted["feature_store_sha256"]:
            raise ValueError("S3 feature source changed")
        catalog_identity = verify_authoritative_catalog_identity(catalog_path)
        if catalog_identity != accepted["catalog_identity"]:
            raise ValueError("S3 catalog identity changed")
        truth_cutoff = datetime.fromisoformat(accepted["truth_cutoff_utc_from_frozen_S0_metadata"])
        reports = read_report_issue_metadata(source, report_end_exclusive=REPORT_END)
        issues = tuple(record.issue_time_utc for record in reports)
        if len(issues) != accepted["allowed_report_count"]:
            raise ValueError("authorized report dates differ from accepted inputs")
        frame = load_development_catalog(catalog_path, truth_cutoff=truth_cutoff)
        catalog = catalog_event_table_from_frame(frame)
        domain, grid, area_hash = load_verified_spatial_inputs(args.data_root)
        identity = {
            "protocol_sha256": accepted["protocol_sha256"],
            "feature_store_sha256": accepted["feature_store_sha256"],
            "catalog_identity": catalog_identity,
            "grid_id": domain.operational_grid.grid_id,
            "grid_cells": grid.cell_count,
            "study_area_sha256": area_hash,
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("identity") != identity:
                raise ValueError("resume identity mismatch; preserve the prior preparation")
            manifest["completed"] = previous["completed"]
        manifest.update(
            identity=identity,
            total_reports=len(issues),
            issue_times_utc=[issue.isoformat() for issue in issues],
            code_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True
            ).strip(),
            truth_cutoff_utc=truth_cutoff.isoformat(),
            status="preparing",
            failures={},
        )
        pending = []
        for issue in issues:
            key = issue.isoformat()
            if key in manifest["completed"]:
                entry = manifest["completed"][key]
                if sha256(output / entry["file"]) != entry["sha256"]:
                    raise ValueError("completed reference cache changed; do not overwrite it")
            else:
                pending.append(issue)
        write_json(manifest_path, manifest)

        def prepare(issue: datetime) -> dict[str, Any]:
            path = output / (issue.strftime("%Y%m%dT%H%M%SZ") + ".npz")
            if path.exists():
                read_issue_cache(path, issue_time=issue, identity=identity)
                return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            features = load_issue_features(
                source,
                issue_times_utc=[issue],
                expected_cell_ids=domain.operational_grid.cell_ids,
                expected_grid_id=domain.operational_grid.grid_id,
                report_end_exclusive=REPORT_END,
            )[issue]
            background = build_catalog_background_components(catalog, grid, issue)
            counts_30 = background.for_horizon(30).expected_counts
            return write_issue_cache(
                path,
                issue_time=issue,
                identity=identity,
                features=features.values,
                kernel_log_masses=dict(background.kernel_log_masses),
                r30_log_mass=background.r30_reference_log_mass,
                expected_counts_per_day={key: value / 30.0 for key, value in counts_30.items()},
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(prepare, issue): issue for issue in pending}
            for future in as_completed(futures):
                key = futures[future].isoformat()
                try:
                    manifest["completed"][key] = future.result()
                except Exception as error:
                    manifest["failures"][key] = f"{type(error).__name__}: {error}"
                manifest["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
                manifest["completed_reports"] = len(manifest["completed"])
                write_json(manifest_path, manifest)
                print(
                    f"Prepared {len(manifest['completed'])}/{len(issues)} reports; "
                    f"failures={len(manifest['failures'])}; last={key}",
                    flush=True,
                )
        manifest["completed_reports"] = len(manifest["completed"])
        manifest["status"] = (
            "complete" if len(manifest["completed"]) == len(issues) else "incomplete"
        )
        manifest["active_pid"] = None
        manifest["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, manifest)
        if manifest["status"] != "complete":
            raise RuntimeError(
                "some actual-date references failed; inspect checkpoint, do not change the trial"
            )
        print(
            "Actual-date inputs ready; anomaly fitting and outer scoring are still pending.",
            flush=True,
        )
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
