"""User-authorized eight-process scheduling of the unchanged frozen S3 null trial.

The original null runner, prediction identity, inputs and all saved NPZs remain
unchanged. Execution provenance is recorded separately from scientific identity.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import multiprocessing
import os
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from seismoflux.multitask_s3.null_parallel_context import write_parallel_context
from seismoflux.multitask_s3.null_parallel_inputs import load_frozen_context
from seismoflux.multitask_s3.null_parallel_worker import initialize_worker, run_replicate
from seismoflux.multitask_s3.null_runner import NullTask, progress_counts, registered_tasks
from seismoflux.multitask_s3.preparation import sha256, write_json

EXECUTION_AUTHORIZATION = "2026-09-04_user_authorized_eight_parallel_processes"
EXECUTION_MODULES = (
    "null_parallel",
    "null_parallel_context",
    "null_parallel_inputs",
    "null_parallel_worker",
)


def pending_groups(manifest: dict[str, Any]) -> tuple[tuple[NullTask, ...], ...]:
    """Partition only unfinished registered blocks, preserving each space field."""
    progress_counts(manifest["completed"], manifest["failures"])
    finished = set(manifest["completed"]) | set(manifest["failures"])
    groups: dict[tuple[str, str, int], list[NullTask]] = {}
    for task in registered_tasks():
        if task.key not in finished:
            groups.setdefault((task.kind, task.fold_id, task.replicate), []).append(task)
    return tuple(tuple(group) for group in groups.values())


def execution_identity(project: Path) -> dict[str, Any]:
    return {
        "authorization": EXECUTION_AUTHORIZATION,
        "unit": "kind_fold_replicate_with_shared_space_field_across_horizons",
        "numerical_library_threads": 1,
        "implementation_sha256": {
            name: sha256(project / f"src/seismoflux/multitask_s3/{name}.py")
            for name in EXECUTION_MODULES
        },
    }


def check_execution_identity(manifest: dict[str, Any], current: dict[str, Any]) -> None:
    previous = manifest.get("parallel_execution_identity")
    if previous is not None and previous != current:
        raise ValueError("parallel execution changed; do not silently resume changed code")


def record_result(manifest: dict[str, Any], tasks: tuple[NullTask, ...], result) -> None:
    expected = {task.key for task in tasks}
    complete, failed = result["completed"], result["failures"]
    if set(complete) & set(failed) or (set(complete) | set(failed)) != expected:
        raise ValueError("worker did not return exactly its assigned blocks")
    if expected & (set(manifest["completed"]) | set(manifest["failures"])):
        raise ValueError("worker attempted to replace a registered result")
    manifest["completed"].update(complete)
    manifest["failures"].update(failed)


def run_parallel(
    *,
    project_root: Path,
    data_root: Path,
    prepared_dir: Path,
    reference_prediction_dir: Path,
    output_dir: Path,
    workers: int = 8,
) -> dict[str, Any]:
    project = project_root.resolve()
    output, prepared, reference = (
        output_dir.resolve(),
        prepared_dir.resolve(),
        reference_prediction_dir.resolve(),
    )
    allowed = project / "outputs/multitask_s3"
    if (
        isinstance(workers, bool)
        or workers not in range(1, 9)
        or any(not path.is_relative_to(allowed) for path in (output, prepared, reference))
        or len({output, prepared, reference}) != 3
    ):
        raise ValueError("use distinct S3 directories and one to eight authorized workers")
    manifest_path = output / "null_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = pending_groups(manifest)
    if not groups:
        print("All frozen blocks already terminal; no new run.", flush=True)
        return manifest
    current_identity = execution_identity(project)
    check_execution_identity(manifest, current_identity)
    if manifest.get("outer_effect_scores_computed") is not False:
        raise ValueError("this prediction-only continuation must precede null scoring")
    lock_path = output / "null_prediction.lock"
    with lock_path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    original_manifest_sha = sha256(manifest_path)
    original_science_identity = copy.deepcopy(manifest["identity"])

    def checkpoint() -> None:
        if manifest["identity"] != original_science_identity:
            raise ValueError("scientific identity must not change during parallel scheduling")
        manifest.update(progress_counts(manifest["completed"], manifest["failures"]))
        manifest["last_checkpoint_utc"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, manifest)

    try:
        manifest.update(
            status="loading_parallel_frozen_inputs",
            active_pid=os.getpid(),
            workers=workers,
            worker_pids=[],
            current_tasks={},
            requested_kinds=["time", "space"],
            execution_mode="replicate_process_pool",
        )
        checkpoint()
        print("Loading unchanged frozen inputs once for shared read-only workers.", flush=True)
        context = load_frozen_context(
            project_root=project,
            data_root=data_root,
            prepared_dir=prepared,
            reference_prediction_dir=reference,
            output_dir=output,
            manifest=manifest,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        context_path = output / f"parallel_context_{stamp}_{os.getpid()}" / "context.pkl"
        write_parallel_context(context_path, context)
        del context
        gc.collect()
        manifest["parallel_execution_identity"] = current_identity
        manifest.setdefault("execution_history", []).append(
            {
                "started_at_utc": datetime.now(UTC).isoformat(),
                "prior_manifest_sha256": original_manifest_sha,
                "preserved_completed_blocks": len(manifest["completed"]),
                "preserved_failed_blocks": len(manifest["failures"]),
                "execution_identity": current_identity,
                "workers": workers,
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=project, text=True
                ).strip(),
                "context_file": str(context_path.relative_to(output)),
                "context_pickle_sha256": sha256(context_path),
            }
        )
        manifest["status"] = "predicting_nulls"
        checkpoint()
        queue = iter(groups)
        fatal_errors: list[str] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(str(context_path),),
        ) as executor:
            active = {}

            def submit_next() -> bool:
                tasks = next(queue, None)
                if tasks is None:
                    return False
                future = executor.submit(run_replicate, tasks, str(output))
                active[future] = tasks
                first = tasks[0]
                group_key = first.key.rsplit("__h", 1)[0]
                manifest["current_tasks"][group_key] = {
                    "kind": first.kind,
                    "fold_id": first.fold_id,
                    "replicate": first.replicate,
                    "pending_horizons": [task.horizon_days for task in tasks],
                    "phase": "running_in_process_pool",
                }
                # CPython's process registry is used only for monitoring, never scheduling.
                manifest["worker_pids"] = sorted(getattr(executor, "_processes", {}))
                return True

            for _ in range(workers):
                if not submit_next():
                    break
            checkpoint()
            print(f"Started {len(active)} independent replicate workers.", flush=True)
            while active:
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    tasks = active.pop(future)
                    group_key = tasks[0].key.rsplit("__h", 1)[0]
                    try:
                        result = future.result()
                        record_result(manifest, tasks, result)
                        print(
                            f"Saved group {group_key}; {len(manifest['completed'])}/4000 blocks; "
                            f"worker {result['pid']}; no outer scoring.",
                            flush=True,
                        )
                    except Exception as error:
                        fatal_errors.append(f"{group_key}: {type(error).__name__}: {error}")
                        manifest["status"] = "draining_workers_after_error"
                        print(f"Execution error: {fatal_errors[-1]}", flush=True)
                    manifest["current_tasks"].pop(group_key, None)
                    checkpoint()
                if not fatal_errors:
                    while len(active) < workers and submit_next():
                        pass
                if done:
                    checkpoint()
            # All processes leave before clearing the coordinator lock, including on error.
        if fatal_errors:
            raise RuntimeError("; ".join(fatal_errors))
        if pending_groups(manifest):
            raise RuntimeError("queue ended before all frozen blocks were terminal")
        manifest.update(
            status="all_null_predictions_terminal",
            active_pid=None,
            worker_pids=[],
            current_tasks={},
            all_null_predictions_terminal=True,
        )
        checkpoint()
        print("All frozen null predictions terminal; scoring remains separate.", flush=True)
        return manifest
    except Exception as error:
        manifest.update(
            status="interrupted_or_input_error",
            active_pid=None,
            worker_pids=[],
            error=f"{type(error).__name__}: {error}",
        )
        checkpoint()
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "data-root",
        "prepared-dir",
        "reference-prediction-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError("launch all numerical libraries with one thread each")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    run_parallel(**vars(args))


if __name__ == "__main__":
    main()
