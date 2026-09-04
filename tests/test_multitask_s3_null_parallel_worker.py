"""Synthetic equivalence checks; no catalog, held-out targets, or scores are read."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from seismoflux.multitask_s3 import null_parallel_worker as worker
from seismoflux.multitask_s3.calendar import HORIZONS, S3FoldCalendar
from seismoflux.multitask_s3.null_runner import NullTask
from seismoflux.multitask_s3.runner import calendar_metadata, read_prediction_block

FOLD = "A_DEV_2023_2024"
ISSUE = datetime(2023, 7, 20, 16, tzinfo=UTC)


def context():
    calendars = {
        (FOLD, h): S3FoldCalendar(FOLD, h, (ISSUE,), ISSUE, (), (ISSUE,) if h < 365 else (), (), ())
        for h in HORIZONS
    }
    features = np.ones((1, 2, 20))
    return {
        "identity": {"prepared_inputs": {"grid_cells": 2}},
        "issues": (ISSUE,),
        "truth": ISSUE,
        "calendars": calendars,
        "caches": {ISSUE: {"features": features[0], "kernel_25": np.array([0.4, 0.6])}},
        "bases": np.zeros((1, 2, 5)),
        "features": features,
        "snapshots": {ISSUE: "synthetic_snapshot"},
        "strata": {"synthetic": "stratum"},
        "zones": (0, 1),
        "query_xy_m": np.zeros((2, 2)),
        "frame": "synthetic_frame",
        "cells": np.array([0, 1]),
        "anchors": np.array([0, 1]),
        "areas_km2": np.ones(2),
    }


def install_predict(monkeypatch):
    calls = []

    def predict(calendar, **kwargs):
        calls.append((calendar, kwargs))
        available = bool(calendar.evaluation_issues)
        return {
            "metadata": {
                "calendar": calendar_metadata(calendar),
                "models": {"COV": {"spatial_status": "fitted", "count": {"status": "fitted"}}},
                "status": (
                    "predictions_complete" if available else "unavailable_no_complete_outer_window"
                ),
            },
            "spatial_log_mass": np.full((len(calendar.evaluation_issues), 5, 2), -np.log(2)),
            "count_log_mean": np.zeros((len(calendar.evaluation_issues), 5, 2)),
        }

    monkeypatch.setattr(worker, "predict_block", predict)
    return calls


def read_saved(tmp_path, ctx, task):
    return read_prediction_block(
        tmp_path / f"{task.key}.npz",
        identity=ctx["identity"],
        calendar=ctx["calendars"][(task.fold_id, task.horizon_days)],
    )


def test_initializer_uses_readonly_mmap_single_arrow_threads_and_low_priority(monkeypatch):
    ctx = context()
    observed = []
    monkeypatch.setattr(worker.pa, "set_cpu_count", lambda value: observed.append(("cpu", value)))
    monkeypatch.setattr(
        worker.pa, "set_io_thread_count", lambda value: observed.append(("io", value))
    )
    monkeypatch.setattr(worker, "_set_below_normal", lambda: observed.append(("priority", "low")))

    def load(path):
        observed.append((path, "readonly_context"))
        return ctx

    monkeypatch.setattr(worker, "read_parallel_context", load)
    monkeypatch.setattr(worker, "_CONTEXT", None)
    worker.initialize_worker("synthetic_context.pkl")
    assert worker._CONTEXT is ctx
    assert observed == [
        ("cpu", 1),
        ("io", 1),
        ("priority", "low"),
        (Path("synthetic_context.pkl"), "readonly_context"),
    ]


def test_run_requires_initialization(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_CONTEXT", None)
    with pytest.raises(RuntimeError, match="initialize"):
        worker.run_replicate((NullTask("time", FOLD, 0, 7),), str(tmp_path))


@pytest.mark.parametrize(
    "tasks",
    [
        (),
        (NullTask("time", FOLD, 0, 7), NullTask("time", FOLD, 0, 7)),
        (NullTask("time", FOLD, 0, 30), NullTask("time", FOLD, 0, 7)),
        (NullTask("time", FOLD, 0, 7), NullTask("space", FOLD, 0, 30)),
        (NullTask("time", FOLD, 0, 7), NullTask("time", FOLD, 1, 30)),
        (NullTask("time", FOLD, 0, 7), NullTask("time", "A_DEV_2024_2025", 0, 30)),
    ],
)
def test_nonunique_or_mixed_worker_group_rejected(tmp_path, tasks):
    with pytest.raises(ValueError):
        worker.run_replicate_with_context(context(), tasks, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_time_horizons_keep_original_seed_and_na_avoids_permutation(tmp_path, monkeypatch):
    ctx = context()
    calls = install_predict(monkeypatch)
    observed = []

    def permute(**kwargs):
        observed.append((kwargs["horizon_days"], kwargs["rng"].integers(0, 2**32, size=8)))
        assert kwargs["truth_cutoff"] == ctx["truth"]
        assert kwargs["fold_id"] == FOLD
        assert kwargs["issue_times_utc"] == ctx["issues"]
        return SimpleNamespace(
            features=np.full_like(ctx["features"], kwargs["horizon_days"]),
            diagnostics={"synthetic_horizon": kwargs["horizon_days"]},
        )

    monkeypatch.setattr(worker, "permute_time_features", permute)
    tasks = tuple(NullTask("time", FOLD, 12, h) for h in (7, 30, 365))
    result = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    assert result["pid"] == os.getpid() and not result["failures"]
    assert set(result["completed"]) == {task.key for task in tasks}
    assert all(
        entry["execution_provenance"] == "parallel_replicate_worker_v1"
        for entry in result["completed"].values()
    )
    assert [h for h, _ in observed] == [7, 30]
    for (h, values), task in zip(observed, tasks[:2], strict=True):
        expected = np.random.default_rng(np.random.SeedSequence(task.seed_words)).integers(
            0, 2**32, size=8
        )
        np.testing.assert_array_equal(values, expected)
        saved = read_saved(tmp_path, ctx, task)
        assert saved["metadata"]["seed_words"] == task.seed_words
        assert saved["metadata"]["null_diagnostics"] == {"synthetic_horizon": h}
    assert calls[-1][1]["caches"] is ctx["caches"]
    assert read_saved(tmp_path, ctx, tasks[-1])["metadata"]["null_diagnostics"] == {
        "status": "unavailable_no_complete_outer_window"
    }
    np.testing.assert_array_equal(ctx["features"], 1)
    assert all(call[1]["frame"] == "synthetic_frame" for call in calls)
    assert not (tmp_path / "null_prediction_manifest.json").exists()


def test_space_pending_subset_shares_one_original_seed_field_across_horizons(tmp_path, monkeypatch):
    ctx = context()
    calls = install_predict(monkeypatch)
    observed = []
    field = np.full_like(ctx["features"], 9)

    def permute(**kwargs):
        observed.append(kwargs["rng"].integers(0, 2**32, size=8))
        assert kwargs["snapshots_by_issue"] == ctx["snapshots"]
        assert kwargs["strata_by_state_id"] is ctx["strata"]
        return SimpleNamespace(features=field, diagnostics={"synthetic_space": 9})

    monkeypatch.setattr(worker, "permute_space_features", permute)
    tasks = tuple(NullTask("space", FOLD, 15, h) for h in (30, 90, 365))
    result = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    assert len(observed) == 1 and not result["failures"]
    expected = np.random.default_rng(
        np.random.SeedSequence(NullTask("space", FOLD, 15, 7).seed_words)
    ).integers(0, 2**32, size=8)
    np.testing.assert_array_equal(observed[0], expected)
    for _, kwargs in calls[:2]:
        assert np.shares_memory(kwargs["caches"][ISSUE]["features"], field)
        assert kwargs["caches"][ISSUE]["kernel_25"] is ctx["caches"][ISSUE]["kernel_25"]
    assert calls[2][1]["caches"] is ctx["caches"]


def test_partial_resume_reuses_orphan_file_without_refitting_or_overwriting(tmp_path, monkeypatch):
    ctx = context()
    calls = install_predict(monkeypatch)
    permuted = []

    def permute(**kwargs):
        permuted.append(kwargs["horizon_days"])
        return SimpleNamespace(features=ctx["features"].copy(), diagnostics={"synthetic": True})

    monkeypatch.setattr(worker, "permute_time_features", permute)
    tasks = tuple(NullTask("time", FOLD, 1, h) for h in (7, 30))
    worker.run_replicate_with_context(ctx, tasks[:1], str(tmp_path))
    first_path = tmp_path / f"{tasks[0].key}.npz"
    before = first_path.read_bytes()
    result = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    assert permuted == [7, 30] and len(calls) == 2
    assert first_path.read_bytes() == before
    assert len(result["completed"]) == 2
    assert result["completed"][tasks[0].key]["execution_provenance"] == "reused_existing_file"
    assert (
        result["completed"][tasks[1].key]["execution_provenance"] == "parallel_replicate_worker_v1"
    )


@pytest.mark.parametrize("error_type", [FloatingPointError, OverflowError])
def test_only_numeric_errors_are_terminal_and_na_still_saved(tmp_path, monkeypatch, error_type):
    ctx = context()
    install_predict(monkeypatch)

    def fail(**kwargs):
        raise error_type("synthetic_numeric_failure")

    monkeypatch.setattr(worker, "permute_space_features", fail)
    tasks = tuple(NullTask("space", FOLD, 1, h) for h in (7, 30, 365))
    result = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    assert set(result["failures"]) == {task.key for task in tasks[:2]}
    assert set(result["completed"]) == {tasks[-1].key}
    assert all(error_type.__name__ in value["error"] for value in result["failures"].values())


def test_time_numeric_failure_does_not_replace_seed_or_abort_other_horizon(tmp_path, monkeypatch):
    ctx = context()
    install_predict(monkeypatch)
    observed = []

    def permute(**kwargs):
        observed.append(kwargs["horizon_days"])
        if kwargs["horizon_days"] == 7:
            raise FloatingPointError("synthetic")
        return SimpleNamespace(features=ctx["features"].copy(), diagnostics={})

    monkeypatch.setattr(worker, "permute_time_features", permute)
    tasks = tuple(NullTask("time", FOLD, 1, h) for h in (7, 30))
    result = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    assert observed == [7, 30]
    assert set(result["failures"]) == {tasks[0].key}
    assert set(result["completed"]) == {tasks[1].key}


def test_numeric_failure_survives_later_io_interruption_without_recalculation(
    tmp_path, monkeypatch
):
    ctx = context()
    install_predict(monkeypatch)
    observed = []

    def interrupted_permute(**kwargs):
        horizon = kwargs["horizon_days"]
        observed.append(horizon)
        if horizon == 7:
            raise FloatingPointError("synthetic_numeric_failure")
        raise OSError("synthetic_later_interruption")

    tasks = tuple(NullTask("time", FOLD, 25, h) for h in (7, 30))
    monkeypatch.setattr(worker, "permute_time_features", interrupted_permute)
    with pytest.raises(OSError, match="synthetic_later_interruption"):
        worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    receipt_path = tmp_path / f"{tasks[0].key}.failure.json"
    before = receipt_path.read_bytes()
    assert not (tmp_path / f"{tasks[1].key}.failure.json").exists()
    receipt = json.loads(before)
    assert receipt["identity"] == ctx["identity"]
    assert receipt["seed_words"] == tasks[0].seed_words
    assert receipt["null_task"]["horizon_days"] == 7

    def resumed_permute(**kwargs):
        observed.append(kwargs["horizon_days"])
        assert kwargs["horizon_days"] == 30
        return SimpleNamespace(features=ctx["features"].copy(), diagnostics={})

    monkeypatch.setattr(worker, "permute_time_features", resumed_permute)
    result = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    assert observed == [7, 30, 30]
    assert set(result["failures"]) == {tasks[0].key}
    assert set(result["completed"]) == {tasks[1].key}
    assert result["failures"][tasks[0].key]["recorded_at_utc"] == receipt["recorded_at_utc"]
    assert receipt_path.read_bytes() == before


@pytest.mark.parametrize("field", ["identity", "null_task", "seed_words", "error"])
def test_failure_receipt_wrong_identity_task_seed_or_error_is_rejected(
    tmp_path, monkeypatch, field
):
    ctx = context()
    task = NullTask("time", FOLD, 28, 7)

    def numeric_fail(**kwargs):
        raise OverflowError("synthetic")

    monkeypatch.setattr(worker, "permute_time_features", numeric_fail)
    worker.run_replicate_with_context(ctx, (task,), str(tmp_path))
    path = tmp_path / f"{task.key}.failure.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = {
        "identity": {"prepared_inputs": {"grid_cells": 3}},
        "null_task": {"kind": "time", "fold_id": FOLD, "replicate": 29, "horizon_days": 7},
        "seed_words": [147, 0, 0, 29, 7],
        "error": "MemoryError: not a scientific failure",
    }[field]
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="failure receipt differs"):
        worker.run_replicate_with_context(ctx, (task,), str(tmp_path))


def test_all_space_failures_resume_without_rebuilding_and_na_remains_successful(
    tmp_path, monkeypatch
):
    ctx = context()
    install_predict(monkeypatch)
    calls = []

    def numeric_fail(**kwargs):
        calls.append(True)
        raise OverflowError("synthetic")

    monkeypatch.setattr(worker, "permute_space_features", numeric_fail)
    tasks = tuple(NullTask("space", FOLD, 35, h) for h in (7, 30, 365))
    first = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    second = worker.run_replicate_with_context(ctx, tasks, str(tmp_path))
    assert calls == [True]
    assert second["failures"] == first["failures"]
    assert set(second["completed"]) == {tasks[-1].key}
    assert not (tmp_path / f"{tasks[-1].key}.failure.json").exists()


@pytest.mark.parametrize("error_type", [ValueError, OSError, MemoryError])
def test_input_file_and_resource_errors_propagate_as_interruptions(
    tmp_path, monkeypatch, error_type
):
    ctx = context()

    def fail(**kwargs):
        raise error_type("synthetic_interruption")

    monkeypatch.setattr(worker, "permute_space_features", fail)
    with pytest.raises(error_type, match="synthetic_interruption"):
        worker.run_replicate_with_context(ctx, (NullTask("space", FOLD, 1, 7),), str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["time", "space"])
def test_real_fitter_outputs_match_original_sequential_composition(tmp_path, monkeypatch, kind):
    """Use the unchanged fitter on synthetic inputs, not an outer-score fixture."""
    ctx = context()
    train0 = datetime(2022, 7, 20, 16, tzinfo=UTC)
    train1 = train0 + timedelta(days=7)
    times = (train0, train1, ISSUE)
    features = np.arange(3 * 2 * 20, dtype=float).reshape(3, 2, 20) / 100.0
    caches = {
        time: {
            "features": features[index],
            "kernel_25": np.log([0.6, 0.4]),
            "kernel_75": np.log([0.4, 0.6]),
            "kernel_150": np.log([0.3, 0.7]),
            "r30_log_mass": np.log([0.7, 0.3]),
            "metadata": {
                "expected_counts_per_day": {
                    "Ms5_6": 0.01,
                    "Ms6_plus": 0.005,
                    "Ms5_plus": 0.015,
                }
            },
        }
        for index, time in enumerate(times)
    }
    calendar = S3FoldCalendar(
        FOLD, 7, times, datetime(2023, 6, 1, tzinfo=UTC), (train0, train1), (ISSUE,), (ISSUE,), ()
    )
    frame = pd.DataFrame(
        [
            {
                "event_id": f"synthetic-{index}",
                "origin_time_utc": time + timedelta(days=1),
                "available_at": time + timedelta(days=1),
                "magnitude": 5.2,
                "inside_study_area": True,
            }
            for index, time in enumerate((train0, train1))
        ]
    )
    ctx.update(
        issues=times,
        caches=caches,
        features=features,
        bases=np.zeros((3, 2, 5)),
        frame=frame,
        anchors={"Ms5_6": set(), "Ms6_plus": set()},
        snapshots={time: "synthetic_snapshot" for time in times},
    )
    ctx["calendars"][(FOLD, 7)] = calendar
    task = NullTask(kind, FOLD, 23, 7)

    def permute(**kwargs):
        return SimpleNamespace(
            features=kwargs["rng"].normal(size=features.shape),
            diagnostics={"status": "synthetic_replicate"},
        )

    monkeypatch.setattr(worker, "permute_time_features", permute)
    monkeypatch.setattr(worker, "permute_space_features", permute)
    # This is the original runner's construct composition, evaluated separately.
    null = permute(rng=np.random.default_rng(np.random.SeedSequence(task.seed_words)))
    expected = worker.predict_block(
        calendar,
        caches=worker.overlay_features(caches, times, null.features),
        frame=frame,
        cell_indices=ctx["cells"],
        anchor_ids=ctx["anchors"],
        areas_km2=ctx["areas_km2"],
    )
    worker.run_replicate_with_context(ctx, (task,), str(tmp_path))
    actual = read_saved(tmp_path, ctx, task)
    np.testing.assert_array_equal(actual["spatial_log_mass"], expected["spatial_log_mass"])
    np.testing.assert_array_equal(actual["count_log_mean"], expected["count_log_mean"])
    assert actual["metadata"]["models"] == expected["metadata"]["models"]
    assert actual["metadata"]["null_diagnostics"] == null.diagnostics
