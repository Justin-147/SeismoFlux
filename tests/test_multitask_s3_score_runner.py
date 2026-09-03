"""Synthetic-only checks for the thin S3 local scoring entry point."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import box

from seismoflux.multitask_s3 import score_runner
from seismoflux.multitask_s3.calendar import S3FoldCalendar
from seismoflux.multitask_s3.preparation import sha256, write_json
from seismoflux.multitask_s3.runner import BANDS, COUNT_VARIANTS, SPATIAL_VARIANTS
from seismoflux.multitask_s3.scoring import score_count
from seismoflux.stage2s.contracts import SpatialGrid

ISSUE = datetime(2023, 7, 20, 16, tzinfo=UTC)
TIMES = (ISSUE, ISSUE + timedelta(days=7), ISSUE + timedelta(days=63))
TRUTH = datetime(2024, 7, 1, tzinfo=UTC)


def _grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="synthetic_scoring_grid",
        cell_size_km=25,
        cell_ids=("c0", "c1", "c2"),
        rows=np.zeros(3, dtype=np.int64),
        columns=np.arange(3, dtype=np.int64),
        query_xy_km=np.column_stack((np.arange(3) * 25, np.zeros(3))),
        clipped_area_km2=np.array([100.0, 100.0, 100.0]),
    )


def _calendar() -> S3FoldCalendar:
    return S3FoldCalendar(
        "A_DEV_2023_2024",
        30,
        TIMES,
        ISSUE - timedelta(days=50),
        (),
        TIMES,
        (TIMES[0], TIMES[2]),
        (),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": identifier,
                "origin_time_utc": ISSUE + timedelta(days=days),
                "available_at": ISSUE + timedelta(days=days),
                "magnitude": magnitude,
                "inside_study_area": True,
                "longitude": lon,
                "latitude": 30.0,
            }
            for identifier, days, magnitude, lon in (
                ("e_anchor", 1, 5.2, 101.0),
                ("e_subsequent", 8, 5.3, 102.0),
                ("e_ms6", 9, 6.2, 100.0),
            )
        ]
    )


def _prediction(calendar: S3FoldCalendar | None = None) -> dict[str, Any]:
    calendar = _calendar() if calendar is None else calendar
    n = len(calendar.evaluation_issues)
    spatial = np.log(
        np.array(
            [
                [0.6, 0.3, 0.1],
                [0.1, 0.3, 0.6],
                [0.6, 0.3, 0.1],
                [0.1, 0.3, 0.6],
                [0.1, 0.6, 0.3],
            ]
        )
    )
    counts = np.log(np.array([[0.5, 0.25], [1.0, 0.5], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]))
    return {
        "metadata": {
            "spatial_variants": list(SPATIAL_VARIANTS),
            "count_variants": list(COUNT_VARIANTS),
            "magnitude_bands": list(BANDS),
            "models": {"COV": {"training_synthetic": True}},
        },
        "spatial_log_mass": np.repeat(spatial[None, :, :], n, axis=0),
        "count_log_mean": np.repeat(counts[None, :, :], n, axis=0),
    }


def _score(**overrides: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arguments: dict[str, Any] = {
        "frame": _frame(),
        "cell_indices": np.array([1, 2, 0], dtype=np.int64),
        "anchor_ids": {"Ms5_6": {"e_anchor"}, "Ms6_plus": {"e_ms6"}},
        "near_cells_by_event": {"e_anchor": {0, 1}, "e_subsequent": {1, 2}, "e_ms6": {0, 1}},
        "grid": _grid(),
        "budgets_km2": [100.0, 200.0],
        "truth_cutoff": TRUTH,
    }
    arguments.update(overrides)
    return score_runner.score_prediction_block(_prediction(), _calendar(), **arguments)


def test_primary_and_replay_axes_do_not_confuse_occurrences_with_unique_events() -> None:
    summary, local = _score()
    primary = summary["bands"]["Ms5_6"]["primary_nonoverlap"]
    replay = summary["bands"]["Ms5_6"]["all_reports_descriptive"]
    assert primary["sample_counts"] == {
        "issue_count": 2,
        "event_occurrences": 2,
        "unique_events": 2,
        "anchor_occurrences": 1,
        "unique_anchors": 1,
    }
    assert replay["sample_counts"]["issue_count"] == 3
    assert replay["sample_counts"]["event_occurrences"] == 3
    assert replay["sample_counts"]["unique_events"] == 2
    assert summary["training_and_inner_metadata"] == _prediction()["metadata"]["models"]
    assert len(local) == 6
    assert "target_event_ids" in local[0]
    assert '"e_anchor"' not in json.dumps(summary)
    assert len(primary["spatial"]) == len(primary["count"]) == 5


def test_paired_results_show_gains_losses_and_70km_without_changing_paid_area() -> None:
    summary, _ = _score()
    primary = summary["bands"]["Ms5_6"]["primary_nonoverlap"]
    contrast = primary["spatial_contrasts"]["CAT_DYN_minus_CAT_COV"]["alarms"][0]
    anchors = contrast["strict"]["views"]["anchor"]
    assert (anchors["gained"], anchors["lost"], anchors["net_hits"]) == (1, 0, 1)
    assert contrast["candidate_actual_area_mean_km2"] == 100.0
    assert contrast["reference_actual_area_mean_km2"] == 100.0
    assert contrast["secondary_70km"]["views"]["anchor"]["net_hits"] == 0
    ms6 = summary["bands"]["Ms6_plus"]["primary_nonoverlap"]
    loss = ms6["spatial_contrasts"]["CAT_DYN_minus_CATALOG"]["alarms"][0]
    assert loss["strict"]["views"]["anchor"]["lost"] == 1
    assert "CAT_DYN_minus_CAT_SNAP" in primary["spatial_contrasts"]
    assert "T0_CAL_DYN_minus_T0_CAL_SNAP" in primary["count_contrasts"]


def test_zero_windows_remain_count_observations_and_logmeans_are_not_exponentiated_for_scores() -> (
    None
):
    summary, _ = _score()
    primary = summary["bands"]["Ms5_6"]["primary_nonoverlap"]
    count = primary["count"]["T0_CAL_DYN"]
    assert count["issue_count"] == 2 and count["empty_issue_count"] == 1
    expected = (
        score_count(np.log(3.0), 2)["poisson_log_score"]
        + score_count(np.log(3.0), 0)["poisson_log_score"]
    )
    assert count["poisson_log_score_sum"] == pytest.approx(expected)
    prediction = _prediction()
    prediction["count_log_mean"][:, 4, :] = -1000.0
    result, _ = score_runner.score_prediction_block(
        prediction,
        _calendar(),
        frame=_frame(),
        cell_indices=np.array([1, 2, 0]),
        anchor_ids={"Ms5_6": {"e_anchor"}, "Ms6_plus": {"e_ms6"}},
        near_cells_by_event={"e_anchor": {1}, "e_subsequent": {2}, "e_ms6": {0}},
        grid=_grid(),
        budgets_km2=[100.0],
        truth_cutoff=TRUTH,
    )
    tiny = result["bands"]["Ms5_6"]["primary_nonoverlap"]["count"]["T0_CAL_DYN"]
    assert tiny["poisson_log_score_sum"] == pytest.approx(-2000.0 - np.log(2.0))
    assert tiny["expected_count_total"] == 0.0


def test_unavailable_horizon_is_na_without_constructing_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = replace(
        _calendar(), horizon_days=365, evaluation_issues=(), primary_evaluation_issues=()
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("no complete issue means no target construction")

    monkeypatch.setattr(score_runner, "build_window_targets", forbidden)
    summary, local = score_runner.score_prediction_block(
        _prediction(calendar),
        calendar,
        frame=pd.DataFrame(),
        cell_indices=np.empty(0, dtype=np.int64),
        anchor_ids={},
        near_cells_by_event={},
        grid=_grid(),
        budgets_km2=[100.0],
        truth_cutoff=TRUTH,
    )
    assert summary["status"] == "unavailable_no_complete_outer_window"
    assert local == []
    axis = summary["bands"]["Ms5_6"]["primary_nonoverlap"]
    assert axis["status"] == "no_complete_issues_NA"
    assert axis["count"]["T0"]["poisson_log_score_mean"] is None
    assert axis["count"]["T0"]["observed_count_total"] is None
    assert axis["spatial"]["CAT_DYN"]["alarms"] == []


def test_count_pairs_validate_exact_exposure_and_target_alignment() -> None:
    candidate, reference = {"i1": score_count(0.0, 0)}, {"i1": score_count(np.log(2.0), 0)}
    paired = score_runner.pair_count(candidate, reference)
    assert paired["delta_poisson_log_score_sum"] == pytest.approx(1.0)
    assert paired["delta_brier_at_least_one_mean"] < 0
    with pytest.raises(ValueError, match="same issues"):
        score_runner.pair_count(candidate, {"i2": reference["i1"]})
    with pytest.raises(ValueError, match="target counts"):
        score_runner.pair_count(candidate, {"i1": score_count(0.0, 1)})


def test_scoring_entry_rejects_incomplete_predictions_before_any_catalog_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = tmp_path / "outputs/multitask_s3/prediction"
    prediction.mkdir(parents=True)
    write_json(
        prediction / "prediction_manifest.json",
        {
            "status": "predicting",
            "completed": {},
            "issue_times_utc": [time.isoformat() for time in TIMES],
            "truth_cutoff_utc": TRUTH.isoformat(),
        },
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("catalog must stay unread until all ten predictions pass verification")

    monkeypatch.setattr(score_runner, "load_development_catalog", forbidden)
    monkeypatch.setattr(score_runner, "verify_authoritative_catalog_identity", forbidden)
    output = tmp_path / "outputs/multitask_s3/scores"
    with pytest.raises(ValueError, match="before any outer scoring"):
        score_runner.score_trial(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            prediction_dir=prediction,
            output_dir=output,
        )
    assert not output.exists()


def test_thin_trial_writes_separate_local_diagnostics_and_explicit_pending_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = tmp_path / "outputs/multitask_s3/prediction"
    prediction.mkdir(parents=True)
    configs = tmp_path / "configs"
    configs.mkdir()
    protocol = configs / "multitask_s3_anomaly.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "access": {"catalog": "synthetic.parquet"},
                "support": {"alarm_area_budgets_km2": [100.0, 200.0]},
            }
        ),
        encoding="utf-8",
    )
    grid = _grid()
    prepared = {
        "protocol_sha256": sha256(protocol),
        "catalog_identity": "synthetic_identity",
        "grid_id": grid.grid_id,
        "grid_cells": 3,
        "study_area_sha256": "synthetic_area",
    }
    identity = {"prepared_inputs": prepared, "implementation_sha256": {}}
    completed = {
        f"{fold}__h{horizon:03d}": {"file": f"{fold}_{horizon}.npz"}
        for fold in ("A_DEV_2023_2024", "A_DEV_2024_2025")
        for horizon in (7, 30, 90, 180, 365)
    }
    write_json(
        prediction / "prediction_manifest.json",
        {
            "status": "predictions_complete",
            "identity": identity,
            "completed": completed,
            "issue_times_utc": [time.isoformat() for time in TIMES],
            "truth_cutoff_utc": TRUTH.isoformat(),
        },
    )
    calls = []

    def verify(path: Path, manifest: dict[str, Any], calendars: Any) -> None:
        assert len(calendars) == 10
        calls.append("verified_all_ten")

    def load(*args: object, **kwargs: object) -> pd.DataFrame:
        assert calls == ["verified_all_ten"]
        return _frame()

    def read(path: Path, *, identity: dict[str, Any], calendar: S3FoldCalendar) -> dict[str, Any]:
        return _prediction(calendar)

    domain = SimpleNamespace(
        operational_grid=grid,
        locator=SimpleNamespace(
            locate_lonlat=lambda lon, lat: {100.0: 0, 101.0: 1, 102.0: 2}[lon],
            clipped_geometries=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
        ),
    )
    monkeypatch.setattr(score_runner, "verify_complete_predictions", verify)
    monkeypatch.setattr(
        score_runner, "verify_authoritative_catalog_identity", lambda path: "synthetic_identity"
    )
    monkeypatch.setattr(
        score_runner, "load_verified_spatial_inputs", lambda path: (domain, None, "synthetic_area")
    )
    monkeypatch.setattr(score_runner, "load_development_catalog", load)
    monkeypatch.setattr(
        score_runner,
        "prepare_anchor_ids",
        lambda frame: {"Ms5_6": {"e_anchor"}, "Ms6_plus": {"e_ms6"}},
    )
    monkeypatch.setattr(score_runner, "projected_near_cells", lambda tree, x, y: [{1}, {2}, {0}])
    monkeypatch.setattr(score_runner, "read_prediction_block", read)
    output = tmp_path / "outputs/multitask_s3/scores"
    result = score_runner.score_trial(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        prediction_dir=prediction,
        output_dir=output,
    )
    assert result["local_only"] is True
    assert result["status"] == "initial_development_effects_complete_S3_not_complete"
    assert len(result["blocks"]) == 10
    assert len(result["pooled_folds"]) == 10
    assert result["adoption_threshold"] is None
    assert "200_time_and_200_space_placebos_per_fold" in result["pending"]
    scores_text = (output / "science_scores.json").read_text(encoding="utf-8")
    diagnostics = json.loads((output / "event_diagnostics_local.json").read_text(encoding="utf-8"))
    progress = json.loads((output / "score_progress.json").read_text(encoding="utf-8"))
    assert '"e_anchor"' not in scores_text
    assert diagnostics["local_only"] is True and diagnostics["records"]
    assert progress["status"] == "complete" and len(progress["completed_blocks"]) == 10
    with pytest.raises(FileExistsError):
        score_runner.score_trial(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            prediction_dir=prediction,
            output_dir=output,
        )
