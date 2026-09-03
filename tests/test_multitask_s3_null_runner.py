"""Synthetic bookkeeping and immutable block tests for the frozen null runner."""

from datetime import UTC, datetime

import numpy as np
import pytest

from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, S3FoldCalendar
from seismoflux.multitask_s3.null_runner import (
    NullTask,
    _data_path,
    overlay_features,
    progress_counts,
    registered_tasks,
    save_or_resume_block,
)
from seismoflux.multitask_s3.runner import calendar_metadata

FOLD = "A_DEV_2023_2024"
ISSUE = datetime(2023, 7, 20, 16, tzinfo=UTC)


def test_registered_count_and_seed_namespace_are_fixed_and_space_reuses_fields():
    tasks = registered_tasks()
    assert len(tasks) == len({task.key for task in tasks}) == 2 * 2 * 200 * 5
    assert {task.replicate for task in tasks} == set(range(200))
    for kind in ("time", "space"):
        for fold in FOLDS:
            assert sum(task.kind == kind and task.fold_id == fold for task in tasks) == 1000
    first = NullTask("time", FOLD, 0, 7)
    assert first.seed_words == [147, 0, 0, 0, 7]
    assert first.seed_words != NullTask("time", FOLD, 0, 30).seed_words
    assert NullTask("space", FOLD, 0, 7).seed_words == NullTask("space", FOLD, 0, 365).seed_words
    assert NullTask("space", FOLD, 0, 7).seed_words != NullTask("space", FOLD, 1, 7).seed_words
    assert (
        NullTask("space", FOLD, 0, 7).seed_words
        != NullTask("space", "A_DEV_2024_2025", 0, 7).seed_words
    )
    np.testing.assert_array_equal(
        np.random.default_rng(np.random.SeedSequence(first.seed_words)).permutation(20),
        np.random.default_rng(np.random.SeedSequence(first.seed_words)).permutation(20),
    )


def test_progress_requires_all_horizons_and_preserves_failed_replicates():
    keys = [NullTask("time", FOLD, 0, h).key for h in HORIZONS]
    completed = {key: {} for key in keys[:-1]}
    result = progress_counts(completed, {})
    assert result["by_kind_fold"]["time"][FOLD]["completed_replicates"] == 0
    completed[keys[-1]] = {"status": "unavailable_no_complete_outer_window"}
    result = progress_counts(completed, {})
    assert result["by_kind_fold"]["time"][FOLD]["completed_replicates"] == 1
    completed.pop(keys[0])
    result = progress_counts(completed, {keys[0]: {"error": "synthetic_failure"}})
    assert result["by_kind_fold"]["time"][FOLD]["failed_replicates"] == 1
    assert result["completed_blocks"] == 4 and result["failed_blocks"] == 1
    assert result["terminal_percent"] == 100 * 5 / 4000
    with pytest.raises(ValueError, match="both completed"):
        progress_counts({keys[0]: {}}, {keys[0]: {}})
    with pytest.raises(ValueError, match="unregistered"):
        progress_counts({"not_a_task": {}}, {})


@pytest.mark.parametrize(
    "kind,fold,replicate,h",
    [
        ("other", FOLD, 0, 30),
        ("time", "old_fold", 0, 30),
        ("time", FOLD, 200, 30),
        ("time", FOLD, -1, 30),
        ("time", FOLD, True, 30),
        ("time", FOLD, 0, 14),
    ],
)
def test_unregistered_tasks_are_rejected(kind, fold, replicate, h):
    with pytest.raises(ValueError):
        NullTask(kind, fold, replicate, h)


def test_saved_orphan_block_is_reused_without_refit_and_wrong_replicate_rejected(tmp_path):
    calendar = S3FoldCalendar(FOLD, 30, (ISSUE,), ISSUE, (), (ISSUE,), (ISSUE,), ())
    identity = {"prepared_inputs": {"grid_cells": 2}}
    task = NullTask("time", FOLD, 0, 30)
    calls = []

    def construct():
        calls.append(True)
        return {
            "metadata": {
                "calendar": calendar_metadata(calendar),
                "models": {"COV": {"spatial_status": "fitted", "count": {"status": "fitted"}}},
                "status": "predictions_complete",
            },
            "spatial_log_mass": np.full((1, 5, 2), -np.log(2)),
            "count_log_mean": np.zeros((1, 5, 2)),
        }

    path = tmp_path / "block.npz"
    one = save_or_resume_block(
        path=path, identity=identity, task=task, calendar=calendar, construct=construct
    )
    before = path.read_bytes()
    two = save_or_resume_block(
        path=path, identity=identity, task=task, calendar=calendar, construct=construct
    )
    assert len(calls) == 1 and one["sha256"] == two["sha256"] and path.read_bytes() == before
    with pytest.raises(ValueError, match="another replicate"):
        save_or_resume_block(
            path=path,
            identity=identity,
            task=NullTask("time", FOLD, 1, 30),
            calendar=calendar,
            construct=construct,
        )
    assert len(calls) == 1 and path.read_bytes() == before


def test_feature_overlay_does_not_change_background_offset_or_original_values():
    original = np.ones((2, 20))
    background = np.log([0.4, 0.6])
    rates = {"rates": "synthetic"}
    cache = {ISSUE: {"features": original, "kernel_25": background, "metadata": rates}}
    changed = np.zeros((1, 2, 20))
    overlaid = overlay_features(cache, (ISSUE,), changed)
    assert overlaid[ISSUE]["kernel_25"] is background
    assert overlaid[ISSUE]["metadata"] is rates
    np.testing.assert_array_equal(cache[ISSUE]["features"], original)
    assert np.shares_memory(overlaid[ISSUE]["features"], changed)


def test_borrowed_paths_are_data_relative_and_cannot_escape(tmp_path):
    assert _data_path(tmp_path, "data/inside/file.parquet") == tmp_path / "inside/file.parquet"
    for value in ("outside/file.parquet", "data/../elsewhere/file.parquet"):
        with pytest.raises(ValueError):
            _data_path(tmp_path, value)
