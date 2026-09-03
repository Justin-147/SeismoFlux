"""Synthetic S3 runner wiring only; no real catalog, outer score, or locked test."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp  # type: ignore[import-untyped]

from seismoflux.multitask_s3 import runner
from seismoflux.multitask_s3.calendar import (
    S3FoldCalendar,
    S3InnerCalendar,
    build_fold_calendar,
)
from seismoflux.multitask_s3.preparation import sha256
from seismoflux.multitask_s3.targets import build_window_targets

TRAIN_0 = datetime(2022, 7, 20, 16, tzinfo=UTC)
TRAIN_1 = TRAIN_0 + timedelta(days=7)
VALIDATION = datetime(2023, 1, 19, 16, tzinfo=UTC)
OUTER = datetime(2023, 7, 20, 16, tzinfo=UTC)
INNER_CUTOFF = datetime(2022, 12, 1, 16, tzinfo=UTC)
OUTER_CUTOFF = datetime(2023, 6, 1, 16, tzinfo=UTC)
TIMES = (TRAIN_0, TRAIN_1, VALIDATION, OUTER)


def _calendar() -> S3FoldCalendar:
    return S3FoldCalendar(
        fold_id="A_DEV_2023_2024",
        horizon_days=7,
        report_issues=TIMES,
        label_fit_cutoff=OUTER_CUTOFF,
        training_issues=(TRAIN_0, TRAIN_1),
        evaluation_issues=(OUTER,),
        primary_evaluation_issues=(OUTER,),
        inner=(S3InnerCalendar("A_I1", INNER_CUTOFF, (TRAIN_0,), (VALIDATION,)),),
    )


def _cache(index: int) -> dict[str, Any]:
    features = np.zeros((3, 20))
    features[:, 0] = np.array([0.0, 1.0, 2.0]) + index * 0.1
    features[:, 12] = np.array([0.1, 0.2, 0.3]) + index
    features[:, 13] = np.array([0.3, 0.1, 0.2]) + index * 0.5
    features[:, 8] = np.array([-1.0, 0.0, 1.0]) * index
    return {
        "features": features,
        "kernel_25": np.log(np.array([0.5, 0.3, 0.2])),
        "kernel_75": np.log(np.array([0.2, 0.5, 0.3])),
        "kernel_150": np.log(np.array([0.1, 0.2, 0.7])),
        "r30_log_mass": np.log(np.array([0.3, 0.6, 0.1])),
        "metadata": {
            "expected_counts_per_day": {"Ms5_6": 0.01, "Ms6_plus": 0.005, "Ms5_plus": 0.015}
        },
    }


def _frame() -> pd.DataFrame:
    specifications = [
        ("early", TRAIN_0 + timedelta(days=1), None, 5.2),
        (
            "delayed_early",
            TRAIN_0 + timedelta(days=2),
            datetime(2022, 12, 15, tzinfo=UTC),
            6.1,
        ),
        ("ms4_training", TRAIN_1 + timedelta(days=1), None, 4.3),
        ("validation", VALIDATION + timedelta(days=1), None, 5.4),
        ("validation_late", VALIDATION + timedelta(days=2), OUTER_CUTOFF + timedelta(days=1), 6.2),
        ("outer_unmapped", OUTER + timedelta(days=1), None, 7.0),
    ]
    return pd.DataFrame(
        [
            {
                "event_id": identifier,
                "origin_time_utc": origin,
                "available_at": origin if availability is None else availability,
                "magnitude": magnitude,
                "inside_study_area": True,
            }
            for identifier, origin, availability, magnitude in specifications
        ]
    )


def _predict(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "caches": {issue: _cache(index) for index, issue in enumerate(TIMES)},
        "frame": _frame(),
        # Any accidental selection of unavailable validation or outer labels fails.
        "cell_indices": np.array([0, 1, 2, 0, -1, -1], dtype=np.int64),
        "anchor_ids": {"Ms5_6": set(), "Ms6_plus": set()},
        "areas_km2": np.array([1.0, 2.0, 3.0]),
    }
    arguments.update(overrides)
    return runner.predict_block(_calendar(), **arguments)


def _identity() -> dict[str, Any]:
    return {"prepared_inputs": {"grid_cells": 3}, "synthetic_trial": "runner-v1"}


def test_predict_block_trains_registered_variants_and_normalizes_spatial_masses() -> None:
    result = _predict()
    spatial, counts, metadata = (
        result["spatial_log_mass"],
        result["count_log_mean"],
        result["metadata"],
    )
    assert spatial.shape == (1, 5, 3)
    assert counts.shape == (1, 5, 2)
    assert np.isfinite(spatial).all() and np.isfinite(counts).all()
    np.testing.assert_allclose(logsumexp(spatial, axis=2), np.zeros((1, 5)), atol=1e-12)
    expected_background = 0.5 * np.exp(_cache(3)["kernel_25"]) + 0.5 * np.exp(
        _cache(3)["kernel_75"]
    )
    np.testing.assert_allclose(np.exp(spatial[0, 0]), expected_background)
    np.testing.assert_array_equal(spatial[0, 1], _cache(3)["r30_log_mass"])
    np.testing.assert_allclose(np.exp(counts[0, 0]), [0.07, 0.035])
    assert metadata["spatial_variants"] == list(runner.SPATIAL_VARIANTS)
    assert metadata["count_variants"] == list(runner.COUNT_VARIANTS)
    assert metadata["status"] == "predictions_complete"
    assert metadata["outer_effect_scores_computed"] is False
    assert metadata["local_only"] is True
    for model in metadata["models"].values():
        assert model["count"]["training_issue_count"] == 2
        assert model["count"]["event_count"] == 2
        assert model["spatial"]["event_count"] == 3
        assert model["training_performance"]["issue_count"] == 2


def test_count_variants_share_one_multiplier_between_disjoint_magnitude_bands() -> None:
    result = _predict()
    log_counts = result["count_log_mean"][0]
    corrections = log_counts - log_counts[0]
    np.testing.assert_allclose(corrections[:, 0], corrections[:, 1], atol=1e-12)
    np.testing.assert_allclose(log_counts[:, 0] - log_counts[:, 1], np.log(2.0), atol=1e-12)
    assert result["metadata"]["shared_band_multiplier"] is True
    assert result["metadata"]["learns_magnitude_distribution"] is False
    expected_calibrated_total = 2.0 / 2.0
    assert np.exp(log_counts[1]).sum() == pytest.approx(expected_calibrated_total)


def test_targets_are_only_training_or_inner_dates_with_their_correct_availability_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = build_window_targets
    calls: dict[tuple[datetime, datetime], tuple[int, int]] = {}

    def spy(frame: pd.DataFrame, **kwargs: Any) -> Any:
        issue, cutoff = kwargs["issue_time"], kwargs["available_by"]
        assert issue != OUTER
        assert issue in (TRAIN_0, TRAIN_1, VALIDATION)
        labels = real_builder(frame, **kwargs)
        assert (issue, cutoff) not in calls
        calls[(issue, cutoff)] = (labels.count_ms4plus, labels.count_ms5plus)
        return labels

    monkeypatch.setattr(runner, "build_window_targets", spy)
    result = _predict()
    assert calls == {
        (TRAIN_0, OUTER_CUTOFF): (2, 2),
        (TRAIN_1, OUTER_CUTOFF): (1, 0),
        (TRAIN_0, INNER_CUTOFF): (1, 1),
        (VALIDATION, OUTER_CUTOFF): (1, 1),
    }
    for model in result["metadata"]["models"].values():
        inner = model["selection"]["inner_scores"][0]
        assert inner["training"]["count_event_count"] == 1
        assert inner["validation"]["count_event_count"] == 1
    assert result["metadata"]["outer_effect_scores_computed"] is False


def test_365_day_without_complete_outer_window_returns_empty_na_without_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = build_fold_calendar(
        TIMES,
        fold_id="A_DEV_2023_2024",
        horizon_days=365,
        truth_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert not calendar.evaluation_issues

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an unavailable horizon must not construct labels or fit")

    monkeypatch.setattr(runner, "build_window_targets", forbidden)
    monkeypatch.setattr(runner, "select_and_fit", forbidden)
    result = runner.predict_block(
        calendar,
        caches={},
        frame=pd.DataFrame(),
        cell_indices=np.empty(0, dtype=np.int64),
        anchor_ids={},
        areas_km2=np.ones(3),
    )
    assert result["spatial_log_mass"].shape == (0, 5, 3)
    assert result["count_log_mean"].shape == (0, 5, 2)
    assert result["metadata"]["status"] == "unavailable_no_complete_outer_window"
    assert result["metadata"]["models"] == {}
    assert result["metadata"]["outer_effect_scores_computed"] is False
    assert "scores" not in result


def test_horizon_background_uses_frozen_kernel_weights_and_disjoint_count_offsets() -> None:
    cache = _cache(0)
    spatial, means = runner.horizon_background(cache, 365)
    expected = sum(np.exp(cache[f"kernel_{scale}"]) for scale in (25, 75, 150)) / 3.0
    np.testing.assert_allclose(np.exp(spatial), expected)
    np.testing.assert_allclose(means, [3.65, 1.825])
    with pytest.raises(ValueError, match="unregistered"):
        runner.horizon_background(cache, 14)
    cache["metadata"]["expected_counts_per_day"]["Ms5_plus"] = 0.1
    with pytest.raises(ValueError, match="disjoint"):
        runner.horizon_background(cache, 7)


def test_prediction_cache_round_trip_and_identity_calendar_checks(tmp_path: Path) -> None:
    result = _predict()
    path, identity, calendar = tmp_path / "block.npz", _identity(), _calendar()
    runner.write_prediction_block(path, result, identity)
    loaded = runner.read_prediction_block(path, identity=identity, calendar=calendar)
    np.testing.assert_array_equal(loaded["spatial_log_mass"], result["spatial_log_mass"])
    np.testing.assert_array_equal(loaded["count_log_mean"], result["count_log_mean"])
    assert loaded["metadata"]["identity"] == identity
    assert loaded["metadata"]["models"] == result["metadata"]["models"]
    assert not path.with_suffix(".tmp.npz").exists()
    with pytest.raises(ValueError, match="another frozen trial"):
        runner.read_prediction_block(
            path, identity={**identity, "changed": True}, calendar=calendar
        )
    with pytest.raises(ValueError, match="calendar"):
        runner.read_prediction_block(
            path, identity=identity, calendar=replace(calendar, label_fit_cutoff=INNER_CUTOFF)
        )


@pytest.mark.parametrize("corruption", ["variant_shape", "nonfinite", "unnormalized"])
def test_prediction_cache_rejects_incomplete_or_invalid_arrays(
    tmp_path: Path, corruption: str
) -> None:
    result = _predict()
    if corruption == "variant_shape":
        result["spatial_log_mass"] = result["spatial_log_mass"][:, :4]
    elif corruption == "nonfinite":
        result["count_log_mean"][0, 0, 0] = np.nan
    else:
        result["spatial_log_mass"][0, 0, 0] += 1.0
    path = tmp_path / "invalid.npz"
    runner.write_prediction_block(path, result, _identity())
    with pytest.raises(ValueError):
        runner.read_prediction_block(path, identity=_identity(), calendar=_calendar())


def test_no_scoring_guard_opens_files_until_every_registered_prediction_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendars = {"A1_h007": _calendar(), "A2_h007": replace(_calendar(), fold_id="A_DEV_2024_2025")}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("must reject the incomplete manifest before reading prediction files")

    monkeypatch.setattr(runner, "sha256", forbidden)
    cases: list[tuple[str, dict[str, Any]]] = [
        ("predicting", {}),
        ("predictions_complete", {"A1_h007": {}}),
        ("predictions_complete", {"A1_h007": {}, "A2_h007": {}, "extra": {}}),
    ]
    for status, completed in cases:
        with pytest.raises(ValueError, match="before any outer scoring"):
            runner.verify_complete_predictions(
                tmp_path, {"status": status, "completed": completed}, calendars
            )


def test_complete_prediction_guard_checks_hash_and_saved_identity(tmp_path: Path) -> None:
    key, calendar, identity = "A1_h007", _calendar(), _identity()
    path = tmp_path / "block.npz"
    runner.write_prediction_block(path, _predict(), identity)
    manifest: dict[str, Any] = {
        "status": "predictions_complete",
        "identity": identity,
        "completed": {key: {"file": path.name, "sha256": sha256(path)}},
    }
    runner.verify_complete_predictions(tmp_path, manifest, {key: calendar})
    with pytest.raises(ValueError, match="another frozen trial"):
        runner.verify_complete_predictions(
            tmp_path, {**manifest, "identity": {**identity, "changed": True}}, {key: calendar}
        )
    manifest["completed"][key]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="saved prediction changed"):
        runner.verify_complete_predictions(tmp_path, manifest, {key: calendar})


def test_unavailable_block_also_round_trips_as_empty_predictions(tmp_path: Path) -> None:
    calendar = replace(
        _calendar(), horizon_days=365, evaluation_issues=(), primary_evaluation_issues=()
    )
    result = runner.predict_block(
        calendar,
        caches={},
        frame=pd.DataFrame(),
        cell_indices=np.empty(0, dtype=np.int64),
        anchor_ids={},
        areas_km2=np.ones(3),
    )
    path = tmp_path / "unavailable.npz"
    runner.write_prediction_block(path, result, _identity())
    loaded = runner.read_prediction_block(path, identity=_identity(), calendar=calendar)
    assert loaded["spatial_log_mass"].size == 0
    assert loaded["count_log_mean"].size == 0
    assert loaded["metadata"]["status"] == "unavailable_no_complete_outer_window"
