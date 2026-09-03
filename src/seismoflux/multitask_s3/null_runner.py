"""Resumable frozen 200+200 S3 null predictions, without outer effect scoring.

The scientific trial and fitted model functions are unchanged. This thin runner
only substitutes the registered counterfactual features and saves every block.
All null predictions must be terminal before a separate scoring phase starts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pyarrow as pa
import yaml

from seismoflux.multitask_s0 import verify_authoritative_catalog_identity
from seismoflux.multitask_s1.runner_inputs import load_verified_spatial_inputs
from seismoflux.multitask_s3.calendar import (
    FOLDS,
    HORIZONS,
    REPORT_END,
    S3FoldCalendar,
    build_fold_calendar,
)
from seismoflux.multitask_s3.input_waterlevel import load_development_catalog
from seismoflux.multitask_s3.null_features import (
    DYNAMIC_INDICES,
    SNAPSHOT_AND_MASK_INDICES,
    permute_time_features,
    rebuild_dynamic_values,
)
from seismoflux.multitask_s3.null_inputs import load_radius_bases
from seismoflux.multitask_s3.null_space import permute_space_features, permute_space_issue
from seismoflux.multitask_s3.null_state_inputs import (
    load_all_zone_ids,
    load_construction_strata,
    load_issue_snapshots,
)
from seismoflux.multitask_s3.preparation import read_issue_cache, sha256, write_json
from seismoflux.multitask_s3.runner import (
    predict_block,
    read_prediction_block,
    write_prediction_block,
)
from seismoflux.multitask_s3.targets import prepare_anchor_ids

KINDS = ("time", "space")
REPLICATES = 200
ROOT_SEED = 147


@dataclass(frozen=True, slots=True)
class NullTask:
    kind: str
    fold_id: str
    replicate: int
    horizon_days: int

    def __post_init__(self) -> None:
        if (
            self.kind not in KINDS
            or self.fold_id not in FOLDS
            or self.horizon_days not in HORIZONS
            or isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or not 0 <= self.replicate < REPLICATES
        ):
            raise ValueError("unregistered null task")

    @property
    def key(self) -> str:
        return f"{self.kind}__{self.fold_id}__r{self.replicate:03d}__h{self.horizon_days:03d}"

    @property
    def seed_words(self) -> list[int]:
        # Space fields are the same for all horizons of this fold/replicate.
        return [
            ROOT_SEED,
            KINDS.index(self.kind),
            tuple(FOLDS).index(self.fold_id),
            self.replicate,
            self.horizon_days if self.kind == "time" else 0,
        ]


def registered_tasks() -> tuple[NullTask, ...]:
    return tuple(
        NullTask(kind, fold, replicate, horizon)
        for kind in KINDS
        for fold in FOLDS
        for replicate in range(REPLICATES)
        for horizon in HORIZONS
    )


def progress_counts(completed: Mapping[str, Any], failures: Mapping[str, Any]) -> dict[str, Any]:
    finished = set(completed) | set(failures)
    if set(completed) & set(failures):
        raise ValueError("a null block cannot be both completed and failed")
    tasks = registered_tasks()
    if not finished <= {task.key for task in tasks}:
        raise ValueError("checkpoint contains an unregistered null block")
    groups = {}
    for kind in KINDS:
        groups[kind] = {}
        for fold in FOLDS:
            done = failed = 0
            for replicate in range(REPLICATES):
                keys = {NullTask(kind, fold, replicate, horizon).key for horizon in HORIZONS}
                if keys <= finished:
                    if keys & set(failures):
                        failed += 1
                    else:
                        done += 1
            groups[kind][fold] = {
                "completed_replicates": done,
                "failed_replicates": failed,
                "registered_replicates": REPLICATES,
            }
    return {
        "by_kind_fold": groups,
        "completed_blocks": len(completed),
        "failed_blocks": len(failures),
        "total_blocks": len(tasks),
        "terminal_percent": 100.0 * len(finished) / len(tasks),
    }


def save_or_resume_block(
    *,
    path: Path,
    identity: dict[str, Any],
    task: NullTask,
    calendar: S3FoldCalendar,
    construct: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Reuse an orphan saved block after interruption; never overwrite predictions."""
    if path.exists():
        result = read_prediction_block(path, identity=identity, calendar=calendar)
        if result["metadata"].get("null_task") != asdict(task):
            raise ValueError("saved null block belongs to another replicate")
    else:
        result = construct()
        result["metadata"]["null_task"] = asdict(task)
        result["metadata"]["seed_words"] = task.seed_words
        result["metadata"]["role"] = "offline_attribution_not_causal_forecast"
        write_prediction_block(path, result, identity)
    return {
        "file": path.name,
        "sha256": sha256(path),
        "status": result["metadata"]["status"],
        "evaluation_issues": len(calendar.evaluation_issues),
        "model_statuses": {
            design: {"spatial": model["spatial_status"], "count": model["count"]["status"]}
            for design, model in result["metadata"]["models"].items()
        },
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }


def overlay_features(caches, times, features):
    if features.shape != (len(times), caches[times[0]]["features"].shape[0], 20):
        raise ValueError("null features do not align to complete report history")
    return {time: {**caches[time], "features": features[index]} for index, time in enumerate(times)}


def _data_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or relative.parts[0] != "data":
        raise ValueError("borrowed immutable source path must be rooted at data/")
    path = (root / Path(*relative.parts[1:])).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("immutable source escaped data root")
    return path


def run_null_predictions(
    *,
    project_root: Path,
    data_root: Path,
    prepared_dir: Path,
    reference_prediction_dir: Path,
    output_dir: Path,
    workers: int = 2,
    kinds: tuple[str, ...] = KINDS,
    resume: bool = False,
):
    project, output, prepared, reference = (
        path.resolve()
        for path in (project_root, output_dir, prepared_dir, reference_prediction_dir)
    )
    allowed = project / "outputs/multitask_s3"
    if (
        any(not path.is_relative_to(allowed) for path in (output, prepared, reference))
        or len({output, prepared, reference}) != 3
        or workers not in (1, 2, 3)
        or not kinds
        or not set(kinds) <= set(KINDS)
    ):
        raise ValueError("use distinct local S3 directories and at most three fold workers")
    if output.exists() and not resume:
        raise FileExistsError("resume this named null trial instead of creating duplicate work")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "null_prediction.lock"
    with lock_path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    manifest_path = output / "null_prediction_manifest.json"
    manifest = None
    try:
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
        issues = tuple(datetime.fromisoformat(value) for value in source["issue_times_utc"])
        truth = datetime.fromisoformat(source["truth_cutoff_utc"])
        calendars = {
            (fold, h): build_fold_calendar(issues, fold_id=fold, horizon_days=h, truth_cutoff=truth)
            for fold in FOLDS
            for h in HORIZONS
        }
        previous = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        if previous and previous["identity"] != identity:
            raise ValueError(
                "resume requires the same frozen null implementation, sources and trial"
            )
        manifest = previous or {
            "identity": identity,
            "completed": {},
            "failures": {},
            "local_only": True,
            "issue_times_utc": [time.isoformat() for time in issues],
            "truth_cutoff_utc": truth.isoformat(),
            "outer_effect_scores_computed": False,
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True
            ).strip(),
        }
        selected = {task.key for task in registered_tasks() if task.kind in kinds}
        if selected <= (set(manifest["completed"]) | set(manifest["failures"])):
            print("Selected null predictions already terminal; do not repeat them.", flush=True)
            return manifest
        manifest.update(
            status="loading_frozen_inputs",
            active_pid=os.getpid(),
            workers=min(workers, len(FOLDS)),
            requested_kinds=list(kinds),
            current_tasks={},
            last_checkpoint_utc=datetime.now(UTC).isoformat(),
            **progress_counts(manifest["completed"], manifest["failures"]),
        )
        write_json(manifest_path, manifest)
        print("Loading only frozen authorized inputs for null predictions.", flush=True)
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
            path = prepared / entry["file"]
            if (
                sha256(path) != entry["sha256"]
                or original["identity"]["prepared_report_sha256"][issue.isoformat()]
                != entry["sha256"]
            ):
                raise ValueError("prepared report changed; do not recalculate it")
            caches[issue] = read_issue_cache(path, issue_time=issue, identity=source["identity"])
            for value in caches[issue].values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
        for key, entry in manifest["completed"].items():
            if sha256(output / entry["file"]) != entry["sha256"]:
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
        if "source_reconstruction_check" not in manifest:
            print(
                "Checking unchanged-time dynamic reconstruction, not fitting a null replicate.",
                flush=True,
            )
            dynamic = rebuild_dynamic_values(issues, bases)
            if not np.allclose(
                dynamic, features[:, :, DYNAMIC_INDICES], rtol=1e-11, atol=1e-12, equal_nan=True
            ):
                raise ValueError("identity time reconstruction differs from the existing features")
            manifest["source_reconstruction_check"] = {
                "time_identity": "passed",
                "space_identity": "pending",
            }
            del dynamic
            write_json(manifest_path, manifest)
        snapshots = strata = zones = None
        query_xy_m = domain.operational_grid.query_xy_km * 1000.0
        if "space" in kinds:
            snapshots = load_issue_snapshots(
                state_path, issue_times_utc=issues, report_end_exclusive=REPORT_END
            )
            strata = load_construction_strata(
                source_paths["entity_mapping"],
                snapshots_by_issue=snapshots,
                report_end_exclusive=REPORT_END,
            )
            zones = load_all_zone_ids(source_paths["cell_mapping"])
            if manifest["source_reconstruction_check"]["space_identity"] != "passed":

                class IdentityRng:
                    def permutation(self, values):
                        return (
                            np.arange(values) if np.isscalar(values) else np.asarray(values).copy()
                        )

                for index in sorted({0, len(issues) // 2, len(issues) - 1}):
                    issue = issues[index]
                    result = permute_space_issue(
                        snapshot=snapshots[issue],
                        strata_by_state_id=strata,
                        all_zone_ids=zones,
                        query_xy_m=query_xy_m,
                        features=features[index],
                        rng=IdentityRng(),
                    )
                    if not (
                        np.allclose(
                            result.features[:, SNAPSHOT_AND_MASK_INDICES],
                            features[index][:, SNAPSHOT_AND_MASK_INDICES],
                            rtol=1e-11,
                            atol=1e-12,
                            equal_nan=True,
                        )
                        and np.allclose(
                            result.radius_bases,
                            bases[index],
                            rtol=1e-11,
                            atol=1e-12,
                            equal_nan=True,
                        )
                    ):
                        raise ValueError(
                            "identity spatial reconstruction differs from the existing features"
                        )
                manifest["source_reconstruction_check"]["space_identity"] = "passed"
                write_json(manifest_path, manifest)
        frame = load_development_catalog(catalog_path, truth_cutoff=truth)
        positions = [
            domain.locator.locate_lonlat(float(lon), float(lat))
            for lon, lat in zip(frame["longitude"], frame["latitude"], strict=True)
        ]
        cells = np.array([-1 if value is None else value for value in positions], dtype=np.int64)
        anchors = prepare_anchor_ids(frame)
        checkpoint_lock = Lock()
        manifest["status"] = "predicting_nulls"
        write_json(manifest_path, manifest)

        def checkpoint():
            manifest.update(progress_counts(manifest["completed"], manifest["failures"]))
            manifest["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
            write_json(manifest_path, manifest)

        def run_fold(fold):
            times = calendars[(fold, HORIZONS[0])].report_issues
            n = len(times)
            if times != issues[:n]:
                raise ValueError("fold history must be the complete registered prefix")
            for kind in kinds:
                for replicate in range(REPLICATES):
                    tasks = [NullTask(kind, fold, replicate, h) for h in HORIZONS]
                    pending = [
                        task
                        for task in tasks
                        if task.key not in manifest["completed"]
                        and task.key not in manifest["failures"]
                    ]
                    if not pending:
                        continue
                    with checkpoint_lock:
                        manifest["current_tasks"][fold] = {
                            "kind": kind,
                            "replicate": replicate,
                            "phase": "features",
                        }
                        checkpoint()
                    print(
                        f"Starting {kind} {fold} replicate {replicate + 1}/{REPLICATES}.",
                        flush=True,
                    )
                    space_result = None
                    space_error = None
                    if kind == "space" and any(
                        calendars[(fold, task.horizon_days)].evaluation_issues for task in pending
                    ):
                        try:
                            space_result = permute_space_features(
                                issue_times_utc=times,
                                snapshots_by_issue={time: snapshots[time] for time in times},
                                strata_by_state_id=strata,
                                all_zone_ids=zones,
                                query_xy_m=query_xy_m,
                                features=features[:n],
                                rng=np.random.default_rng(
                                    np.random.SeedSequence(tasks[0].seed_words)
                                ),
                            )
                        except (FloatingPointError, OverflowError) as error:
                            space_error = error
                    for task in pending:
                        calendar = calendars[(fold, task.horizon_days)]
                        with checkpoint_lock:
                            manifest["current_tasks"][fold] = {
                                "kind": kind,
                                "replicate": replicate,
                                "horizon_days": task.horizon_days,
                                "phase": "rebuild_fit_save",
                            }
                            checkpoint()

                        def construct(
                            calendar=calendar,
                            task=task,
                            kind=kind,
                            space_result=space_result,
                            space_error=space_error,
                        ):
                            diagnostics = {"status": "unavailable_no_complete_outer_window"}
                            null_caches = caches
                            if calendar.evaluation_issues:
                                if space_error is not None:
                                    raise space_error
                                null = (
                                    space_result
                                    if kind == "space"
                                    else permute_time_features(
                                        issue_times_utc=times,
                                        features=features[:n],
                                        radius_bases=bases[:n],
                                        fold_id=fold,
                                        horizon_days=task.horizon_days,
                                        truth_cutoff=truth,
                                        rng=np.random.default_rng(
                                            np.random.SeedSequence(task.seed_words)
                                        ),
                                    )
                                )
                                diagnostics = null.diagnostics
                                null_caches = overlay_features(caches, times, null.features)
                            result = predict_block(
                                calendar,
                                caches=null_caches,
                                frame=frame,
                                cell_indices=cells,
                                anchor_ids=anchors,
                                areas_km2=grid.area_km2,
                            )
                            result["metadata"]["null_diagnostics"] = diagnostics
                            return result

                        try:
                            entry = save_or_resume_block(
                                path=output / f"{task.key}.npz",
                                identity=identity,
                                task=task,
                                calendar=calendar,
                                construct=construct,
                            )
                        except (FloatingPointError, OverflowError) as error:
                            with checkpoint_lock:
                                manifest["failures"][task.key] = {
                                    "error": f"{type(error).__name__}: {error}",
                                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                                }
                                checkpoint()
                            print(f"Recorded failed block {task.key}: {error}", flush=True)
                        else:
                            # A file/checkpoint/identity/resource failure is an interruption,
                            # not a scientific replicate failure or grounds for replacement.
                            with checkpoint_lock:
                                manifest["completed"][task.key] = entry
                                checkpoint()
                            print(f"Saved {task.key}; no outer effect scoring.", flush=True)
                    del space_result
            with checkpoint_lock:
                manifest["current_tasks"].pop(fold, None)
                checkpoint()

        with ThreadPoolExecutor(max_workers=min(workers, len(FOLDS))) as executor:
            futures = [executor.submit(run_fold, fold) for fold in FOLDS]
            for future in as_completed(futures):
                future.result()
        finished = set(manifest["completed"]) | set(manifest["failures"])
        complete = finished == {task.key for task in registered_tasks()}
        manifest.update(
            status=(
                "all_null_predictions_terminal"
                if complete
                else "selected_kind_predictions_terminal_others_pending"
            ),
            active_pid=None,
            all_null_predictions_terminal=complete,
        )
        checkpoint()
        print("Selected null predictions terminal. Scoring remains a separate phase.", flush=True)
        return manifest
    except Exception as error:
        if manifest is not None:
            manifest.update(
                status="interrupted_or_input_error",
                active_pid=None,
                error=f"{type(error).__name__}: {error}",
                last_checkpoint_utc=datetime.now(UTC).isoformat(),
            )
            write_json(manifest_path, manifest)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "data-root",
        "prepared-dir",
        "reference-prediction-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--kind", choices=("both", "time", "space"), default="both")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("launch numerical libraries with one thread each")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    run_null_predictions(
        project_root=args.project_root,
        data_root=args.data_root,
        prepared_dir=args.prepared_dir,
        reference_prediction_dir=args.reference_prediction_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        kinds=KINDS if args.kind == "both" else (args.kind,),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
