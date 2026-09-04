"""Read the unchanged frozen S3 null inputs for process-based resumption.

This adapter deliberately leaves the original scientific modules and prediction
identity untouched. It only accepts an existing trial whose time and space
source-reconstruction checks passed, and never writes or scores anything.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from seismoflux.multitask_s0 import verify_authoritative_catalog_identity
from seismoflux.multitask_s1.runner_inputs import load_verified_spatial_inputs
from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, REPORT_END, build_fold_calendar
from seismoflux.multitask_s3.input_waterlevel import load_development_catalog
from seismoflux.multitask_s3.null_inputs import load_radius_bases
from seismoflux.multitask_s3.null_runner import (
    KINDS,
    REPLICATES,
    ROOT_SEED,
    _data_path,
    progress_counts,
)
from seismoflux.multitask_s3.null_state_inputs import (
    load_all_zone_ids,
    load_construction_strata,
    load_issue_snapshots,
)
from seismoflux.multitask_s3.preparation import read_issue_cache, sha256
from seismoflux.multitask_s3.targets import prepare_anchor_ids


def _local_file(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("frozen file escaped its registered directory")
    return path


def load_frozen_context(
    *,
    project_root: Path,
    data_root: Path,
    prepared_dir: Path,
    reference_prediction_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate and load the original two-fold, 200+200 trial without writes.

    The caller owns the trial lock and execution checkpoint. The returned arrays
    in each issue cache are readonly exactly as in the original null runner;
    callers may persist this context to a local readonly shared-memory cache.
    """
    project, output, prepared, reference = (
        path.resolve()
        for path in (project_root, output_dir, prepared_dir, reference_prediction_dir)
    )
    data_root = data_root.resolve()
    allowed = project / "outputs/multitask_s3"
    if (
        any(not path.is_relative_to(allowed) for path in (output, prepared, reference))
        or len({output, prepared, reference}) != 3
    ):
        raise ValueError("use distinct local S3 directories")
    check = manifest.get("source_reconstruction_check", {})
    if check.get("time_identity") != "passed" or check.get("space_identity") != "passed":
        raise ValueError("parallel resume requires passed time and space reconstruction checks")
    progress_counts(manifest["completed"], manifest["failures"])
    source = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    reference_path = reference / "prediction_manifest.json"
    original = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        source["status"] != "complete"
        or source.get("failures")
        or original["status"] != "predictions_complete"
        or original["identity"]["prepared_inputs"] != source["identity"]
    ):
        raise ValueError("original preparation and predictions must be completed and unchanged")
    protocol_path = project / "configs/multitask_s3_anomaly.yaml"
    if sha256(protocol_path) != source["identity"]["protocol_sha256"]:
        raise ValueError("S3 protocol changed since the completed real-feature trial")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["placebos"]["root_seed"] != ROOT_SEED
        or protocol["placebos"]["time_replicates_per_fold"] != REPLICATES
        or protocol["placebos"]["space_replicates_per_fold"] != REPLICATES
    ):
        raise ValueError("registered root seed and 200+200 count must not change")
    for name, expected in original["identity"]["implementation_sha256"].items():
        if sha256(project / f"src/seismoflux/{name}.py") != expected:
            raise ValueError("frozen real-feature model implementation changed")
    old_sources = yaml.safe_load(
        (project / "configs/d1_retrospective_development.yaml").read_text(encoding="utf-8")
    )["data"]
    spatial_sources = old_sources["spatial_strata"]
    state_path = _data_path(data_root, old_sources["anomaly_features"]["state_history_path"])
    source_paths = {
        "feature_store": data_root / protocol["access"]["feature_store"],
        "state_history": state_path,
        **{
            name: _data_path(
                data_root, spatial_sources["local_coordinate_artifacts_not_committed"][name]
            )
            for name in ("cell_mapping", "entity_mapping")
        },
    }
    source_hashes = {
        "feature_store": protocol["access"]["feature_store_sha256"],
        "state_history": protocol["access"]["state_history_sha256"],
        **{
            name: spatial_sources["local_artifact_sha256"][name]
            for name in ("cell_mapping", "entity_mapping")
        },
    }
    names = (
        "multitask_s3/null_runner",
        "multitask_s3/null_features",
        "multitask_s3/null_inputs",
        "multitask_s3/null_space",
        "multitask_s3/null_state_inputs",
        "features/anomaly/trajectory",
        "features/anomaly/snapshot",
        "features/anomaly/state",
        "features/anomaly/spatial",
        "d1_replay/placebos",
    )
    identity = {
        **original["identity"],
        "reference_prediction_manifest_sha256": sha256(reference_path),
        "null_source_sha256": source_hashes,
        "null_implementation_sha256": {
            name: sha256(project / f"src/seismoflux/{name}.py") for name in names
        },
    }
    identity["seed_namespace"] = {
        "root": ROOT_SEED,
        "kinds": list(KINDS),
        "folds": list(FOLDS),
        "replicate_indices": [0, REPLICATES - 1],
        "words": ["root", "kind_index", "fold_index", "replicate", "horizon_if_time_else_0"],
    }
    if manifest["identity"] != identity:
        raise ValueError("resume requires the same frozen null implementation, sources and trial")
    issues = tuple(datetime.fromisoformat(value) for value in source["issue_times_utc"])
    truth = datetime.fromisoformat(source["truth_cutoff_utc"])
    if (
        manifest["issue_times_utc"] != [time.isoformat() for time in issues]
        or manifest["truth_cutoff_utc"] != truth.isoformat()
    ):
        raise ValueError("checkpoint report axis or truth cutoff changed")
    calendars = {
        (fold, h): build_fold_calendar(issues, fold_id=fold, horizon_days=h, truth_cutoff=truth)
        for fold in FOLDS
        for h in HORIZONS
    }
    for calendar in calendars.values():
        if calendar.report_issues != issues[: len(calendar.report_issues)]:
            raise ValueError("fold history must be the complete registered prefix")
    for name, path in source_paths.items():
        if sha256(path) != source_hashes[name]:
            raise ValueError(f"immutable null source changed: {name}")
    catalog_path = data_root / protocol["access"]["catalog"]
    if (
        verify_authoritative_catalog_identity(catalog_path)
        != source["identity"]["catalog_identity"]
    ):
        raise ValueError("catalog changed since the real-feature trial")
    domain, grid, area_hash = load_verified_spatial_inputs(data_root)
    if (
        domain.operational_grid.grid_id != source["identity"]["grid_id"]
        or grid.cell_count != source["identity"]["grid_cells"]
        or area_hash != source["identity"]["study_area_sha256"]
    ):
        raise ValueError("null grid differs from the completed predictions")
    caches = {}
    for issue in issues:
        entry = source["completed"][issue.isoformat()]
        path = _local_file(prepared, entry["file"])
        if (
            sha256(path) != entry["sha256"]
            or original["identity"]["prepared_report_sha256"][issue.isoformat()] != entry["sha256"]
        ):
            raise ValueError("prepared report changed; do not recalculate it")
        caches[issue] = read_issue_cache(path, issue_time=issue, identity=source["identity"])
        for value in caches[issue].values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    for key, entry in manifest["completed"].items():
        if entry["file"] != f"{key}.npz":
            raise ValueError("completed null block filename differs from registered task")
        if sha256(_local_file(output, entry["file"])) != entry["sha256"]:
            raise ValueError(f"completed null block changed: {key}")
    radius = load_radius_bases(
        source_paths["feature_store"],
        issue_times_utc=issues,
        expected_cell_ids=domain.operational_grid.cell_ids,
        expected_grid_id=domain.operational_grid.grid_id,
        report_end_exclusive=REPORT_END,
    )
    bases = np.stack([radius[issue] for issue in issues])
    features = np.stack([caches[issue]["features"] for issue in issues])
    del radius
    snapshots = load_issue_snapshots(
        state_path, issue_times_utc=issues, report_end_exclusive=REPORT_END
    )
    strata = load_construction_strata(
        source_paths["entity_mapping"],
        snapshots_by_issue=snapshots,
        report_end_exclusive=REPORT_END,
    )
    zones = load_all_zone_ids(source_paths["cell_mapping"])
    query_xy_m = domain.operational_grid.query_xy_km * 1000.0
    frame = load_development_catalog(catalog_path, truth_cutoff=truth)
    positions = [
        domain.locator.locate_lonlat(float(lon), float(lat))
        for lon, lat in zip(frame["longitude"], frame["latitude"], strict=True)
    ]
    cells = np.array([-1 if value is None else value for value in positions], dtype=np.int64)
    anchors = prepare_anchor_ids(frame)
    return {
        "identity": identity,
        "issues": issues,
        "truth": truth,
        "calendars": calendars,
        "caches": caches,
        "bases": bases,
        "features": features,
        "snapshots": snapshots,
        "strata": strata,
        "zones": zones,
        "query_xy_m": query_xy_m,
        "frame": frame,
        "cells": cells,
        "anchors": anchors,
        "areas_km2": grid.area_km2,
    }
