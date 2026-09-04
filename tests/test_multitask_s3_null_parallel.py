"""Synthetic scheduling, provenance and shared-input regression checks."""

import copy
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from seismoflux.multitask_s3.calendar import HORIZONS
from seismoflux.multitask_s3.null_parallel import (
    check_execution_identity,
    pending_groups,
    record_result,
)
from seismoflux.multitask_s3.null_parallel_context import (
    read_parallel_context,
    write_parallel_context,
)
from seismoflux.multitask_s3.null_runner import NullTask, registered_tasks

FOLD = "A_DEV_2023_2024"


def test_partition_preserves_all_4000_keys_and_partial_and_failed_blocks():
    manifest = {"completed": {}, "failures": {}}
    groups = pending_groups(manifest)
    assert len(groups) == 800
    assert {task.key for group in groups for task in group} == {
        task.key for task in registered_tasks()
    }
    for group in groups:
        assert len({(t.kind, t.fold_id, t.replicate) for t in group}) == 1
        assert tuple(t.horizon_days for t in group) == HORIZONS
    manifest["completed"][groups[0][0].key] = {"sha256": "old"}
    manifest["failures"][groups[0][1].key] = {"error": "original numerical failure"}
    pending = pending_groups(manifest)
    assert pending[0] == groups[0][2:]
    assert sum(map(len, pending)) == 3998


def test_execution_change_rejected_without_rewriting_science_identity():
    manifest = {"identity": {"original": "unchanged"}}
    current = {"authorization": "synthetic", "implementation_sha256": {"runner": "a"}}
    before = copy.deepcopy(manifest)
    check_execution_identity(manifest, current)
    assert manifest == before
    manifest["parallel_execution_identity"] = copy.deepcopy(current)
    check_execution_identity(manifest, current)
    current["implementation_sha256"]["runner"] = "b"
    with pytest.raises(ValueError, match="execution changed"):
        check_execution_identity(manifest, current)
    assert manifest["identity"] == before["identity"]


def test_out_of_order_group_results_preserve_old_entries_and_reject_duplicates():
    tasks = tuple(NullTask("space", FOLD, 3, h) for h in HORIZONS)
    manifest = {"completed": {"historical": {"sha256": "unchanged"}}, "failures": {}}
    complete = {t.key: {"sha256": t.key} for t in reversed(tasks[1:])}
    failed = {tasks[0].key: {"error": "synthetic numerical failure"}}
    result = {"completed": complete, "failures": failed, "pid": 123}
    record_result(manifest, tasks, result)
    assert manifest["completed"]["historical"] == {"sha256": "unchanged"}
    assert len(manifest["completed"]) == 5 and len(manifest["failures"]) == 1
    with pytest.raises(ValueError, match="replace"):
        record_result(manifest, tasks, result)
    result["failures"] = {}
    with pytest.raises(ValueError, match="exactly"):
        record_result({"completed": {}, "failures": {}}, tasks, result)


def test_shared_numeric_arrays_preserve_values_dtypes_aliases_and_readonly(tmp_path):
    array = np.arange(40000, dtype=np.float64).reshape(200, 200)
    original = {"large": array, "alias": array, "small": np.array([np.nan, 2.0]), "text": "x"}
    path = tmp_path / "context" / "context.pkl"
    write_parallel_context(path, original)
    restored = read_parallel_context(path)
    assert isinstance(restored["large"], np.memmap)
    assert restored["large"] is restored["alias"]
    assert not restored["large"].flags.writeable
    np.testing.assert_array_equal(restored["large"], original["large"])
    np.testing.assert_array_equal(restored["small"], original["small"])
    assert restored["text"] == "x"
    with pytest.raises(ValueError):
        restored["large"][0, 0] = 3
    with pytest.raises(FileExistsError):
        write_parallel_context(path, original)


def _spawn_probe(path):
    context = read_parallel_context(path)
    time.sleep(0.2)
    return {
        "pid": os.getpid(),
        "sum": float(context["large"].sum()),
        "readonly": not context["large"].flags.writeable,
        "threads": [
            os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        ],
    }


def test_windows_spawn_can_share_context_across_eight_processes(tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    path = tmp_path / "spawn_context" / "context.pkl"
    data = np.arange(20000, dtype=np.float64)
    write_parallel_context(path, {"large": data})
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_spawn_probe, [path] * 24))
    assert len({result["pid"] for result in results}) == 8
    assert all(result["sum"] == float(data.sum()) for result in results)
    assert all(result["readonly"] and result["threads"] == ["1"] * 3 for result in results)
