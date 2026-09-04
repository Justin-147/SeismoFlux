"""Process-local execution of unchanged, frozen S3 null-replicate blocks.

Only the parent owns the manifest. A worker owns one fold/kind/replicate group
and writes its uniquely named prediction files through the original immutable
save-or-resume implementation. No outer-effect scoring is performed here.
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from seismoflux.multitask_s3.calendar import HORIZONS
from seismoflux.multitask_s3.null_features import permute_time_features
from seismoflux.multitask_s3.null_parallel_context import read_parallel_context
from seismoflux.multitask_s3.null_runner import (
    NullTask,
    overlay_features,
    save_or_resume_block,
)
from seismoflux.multitask_s3.null_space import permute_space_features
from seismoflux.multitask_s3.preparation import write_json
from seismoflux.multitask_s3.runner import predict_block

_CONTEXT: dict[str, Any] | None = None
_CONTEXT_KEYS = frozenset(
    {
        "identity",
        "issues",
        "truth",
        "calendars",
        "caches",
        "bases",
        "features",
        "snapshots",
        "strata",
        "zones",
        "query_xy_m",
        "frame",
        "cells",
        "anchors",
        "areas_km2",
    }
)
_FAILURE_RECEIPT_SCHEMA = "s3_null_numeric_failure_v1"


def _read_failure_receipt(
    path: Path, *, identity: dict[str, Any], task: NullTask
) -> dict[str, str]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {"schema", "identity", "null_task", "seed_words", "error", "recorded_at_utc"}
        or receipt["schema"] != _FAILURE_RECEIPT_SCHEMA
        or receipt["identity"] != identity
        or receipt["null_task"] != asdict(task)
        or receipt["seed_words"] != task.seed_words
        or not isinstance(receipt["error"], str)
        or not receipt["error"].startswith(("FloatingPointError: ", "OverflowError: "))
        or not isinstance(receipt["recorded_at_utc"], str)
    ):
        raise ValueError("failure receipt differs from the frozen trial, task, seed or schema")
    recorded = datetime.fromisoformat(receipt["recorded_at_utc"])
    if recorded.tzinfo is None or recorded.utcoffset() != UTC.utcoffset(recorded):
        raise ValueError("failure receipt must preserve its UTC recording time")
    return {"error": receipt["error"], "recorded_at_utc": receipt["recorded_at_utc"]}


def _save_failure_receipt(
    path: Path, *, identity: dict[str, Any], task: NullTask, error: Exception
) -> dict[str, str]:
    if path.exists():
        return _read_failure_receipt(path, identity=identity, task=task)
    entry = {
        "error": f"{type(error).__name__}: {error}",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    # Each task belongs to exactly one worker; atomic replacement publishes only
    # a complete receipt. A later task interruption cannot erase this failure.
    write_json(
        path,
        {
            "schema": _FAILURE_RECEIPT_SCHEMA,
            "identity": identity,
            "null_task": asdict(task),
            "seed_words": task.seed_words,
            **entry,
        },
    )
    return entry


def _set_below_normal() -> None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel32.SetPriorityClass.restype = ctypes.c_int
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000):
            raise ctypes.WinError(ctypes.get_last_error())


def initialize_worker(context_path: str) -> None:
    """Load verified local input context once, sharing array pages read-only."""
    global _CONTEXT
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    _set_below_normal()
    context = read_parallel_context(Path(context_path))
    if not isinstance(context, dict) or not context.keys() >= _CONTEXT_KEYS:
        raise ValueError("parallel worker requires the complete verified frozen context")
    _CONTEXT = context


def run_replicate(tasks: tuple[NullTask, ...], output_dir: str) -> dict[str, Any]:
    if _CONTEXT is None:
        raise RuntimeError("initialize the frozen context before executing null tasks")
    return run_replicate_with_context(_CONTEXT, tasks, output_dir)


def run_replicate_with_context(
    context: dict[str, Any], tasks: tuple[NullTask, ...], output_dir: str
) -> dict[str, Any]:
    """Run an ordered pending-horizon subset with the original seed namespace."""
    if (
        not isinstance(tasks, tuple)
        or not tasks
        or not all(isinstance(task, NullTask) for task in tasks)
    ):
        raise ValueError("a worker needs a nonempty tuple of registered null tasks")
    first = tasks[0]
    if len({(task.kind, task.fold_id, task.replicate) for task in tasks}) != 1 or tuple(
        task.horizon_days for task in tasks
    ) != tuple(h for h in HORIZONS if any(task.horizon_days == h for task in tasks)):
        raise ValueError("one worker group requires unique horizons in frozen order")
    if not context.keys() >= _CONTEXT_KEYS:
        raise ValueError("parallel worker requires the complete verified frozen context")
    output = Path(output_dir)
    if not output.is_dir():
        raise ValueError("the parent must create the null output directory before dispatch")

    kind, fold = first.kind, first.fold_id
    calendars = context["calendars"]
    times = calendars[(fold, HORIZONS[0])].report_issues
    n = len(times)
    if times != context["issues"][:n]:
        raise ValueError("fold history must be the complete registered prefix")
    caches, features, bases = context["caches"], context["features"], context["bases"]
    completed: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for task in tasks:
        receipt_path = output / f"{task.key}.failure.json"
        if receipt_path.exists():
            if (output / f"{task.key}.npz").exists():
                raise ValueError("a null task cannot have both predictions and a failure receipt")
            failures[task.key] = _read_failure_receipt(
                receipt_path, identity=context["identity"], task=task
            )
    pending = tuple(task for task in tasks if task.key not in failures)
    space_result = None
    space_error = None
    if kind == "space" and any(
        calendars[(fold, task.horizon_days)].evaluation_issues
        and not (output / f"{task.key}.npz").exists()
        for task in pending
    ):
        try:
            space_result = permute_space_features(
                issue_times_utc=times,
                snapshots_by_issue={time: context["snapshots"][time] for time in times},
                strata_by_state_id=context["strata"],
                all_zone_ids=context["zones"],
                query_xy_m=context["query_xy_m"],
                features=features[:n],
                rng=np.random.default_rng(np.random.SeedSequence(first.seed_words)),
            )
        except (FloatingPointError, OverflowError) as error:
            space_error = error

    for task in pending:
        calendar = calendars[(fold, task.horizon_days)]

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
                        truth_cutoff=context["truth"],
                        rng=np.random.default_rng(np.random.SeedSequence(task.seed_words)),
                    )
                )
                diagnostics = null.diagnostics
                null_caches = overlay_features(caches, times, null.features)
            result = predict_block(
                calendar,
                caches=null_caches,
                frame=context["frame"],
                cell_indices=context["cells"],
                anchor_ids=context["anchors"],
                areas_km2=context["areas_km2"],
            )
            result["metadata"]["null_diagnostics"] = diagnostics
            return result

        block_path = output / f"{task.key}.npz"
        existed_before_execution = block_path.exists()
        try:
            entry = save_or_resume_block(
                path=block_path,
                identity=context["identity"],
                task=task,
                calendar=calendar,
                construct=construct,
            )
        except (FloatingPointError, OverflowError) as error:
            failures[task.key] = _save_failure_receipt(
                output / f"{task.key}.failure.json",
                identity=context["identity"],
                task=task,
                error=error,
            )
        else:
            entry["execution_provenance"] = (
                "reused_existing_file"
                if existed_before_execution
                else "parallel_replicate_worker_v1"
            )
            completed[task.key] = entry
    return {"completed": completed, "failures": failures, "pid": os.getpid()}
