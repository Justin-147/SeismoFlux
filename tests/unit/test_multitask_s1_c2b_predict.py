from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seismoflux.multitask_s1 import c2b_predict as runner
from seismoflux.multitask_s1.c2b_inputs import c2b_catalog_from_frames
from seismoflux.multitask_s1.runner_inputs import InnerExposure

ROOT = Path(__file__).resolve().parents[2]


def components():
    base = np.log(np.asarray([0.25, 0.75]))
    return {name: base.copy() for name in runner.COMPONENT_IDS}


def sample(block="I1", issue="1990-01-01T00:00:00+00:00", *, horizon=30, available=None):
    time = datetime.fromisoformat(issue)
    end = time + timedelta(days=horizon)
    visible = runner._epoch_us(end if available is None else available)
    return runner.InnerSample(
        "C_DEV_2000_2004",
        block,
        horizon,
        time,
        end,
        components(),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([visible, visible], dtype=np.int64),
    )


def test_timestamp_microseconds_and_finite_log_tail():
    assert runner._epoch_us(datetime(1970, 1, 1, tzinfo=UTC)) == 0
    assert runner._epoch_us(datetime(2000, 1, 1, 0, 0, 0, 1, tzinfo=UTC)) == 946684800000001
    with pytest.raises(ValueError, match="timezone"):
        runner._epoch_us(datetime(2000, 1, 1))
    runner._validate_log_mass(np.asarray([0.0, -2000.0]), (2,))
    with pytest.raises(ValueError, match="normalized"):
        runner._validate_log_mass(np.asarray([-2.0, -2.0]), (2,))


def test_exact_empty_recent_and_alpha_zero_retain_base():
    base = components()["K75"]
    other = np.log([0.5, 0.5])
    assert runner._mix_recent(base, other, 0.0) is base
    assert runner._mix_recent(base, base.copy(), 0.75) is base


def test_ridge_training_tail_purge_and_label_availability():
    boundary = datetime(1995, 1, 1, tzinfo=UTC)
    cutoff = boundary - timedelta(days=30)
    exact = sample(issue=(cutoff - timedelta(days=30)).isoformat())
    late_end = sample(issue=(cutoff - timedelta(days=30) + timedelta(microseconds=1)).isoformat())
    wrong_block = sample(block="I2", issue="1990-01-01T00:00:00+00:00")
    assert runner.legal_ridge_training([exact, late_end, wrong_block], ["I1"], boundary) == [exact]
    delayed = sample(
        issue="1990-01-01T00:00:00+00:00", available=cutoff + timedelta(microseconds=1)
    )
    assert delayed.counts(cutoff).tolist() == [0.0, 0.0]
    assert delayed.counts(cutoff + timedelta(microseconds=1)).tolist() == [1.0, 1.0]


def test_empty_selection_blocks_are_na_not_zero_and_ties_keep_base():
    protocol = runner.load_protocol(ROOT)
    cutoff = datetime(2000, 1, 1, tzinfo=UTC)
    empty = runner.select_kernel_parameters([], cutoff, np.ones(2), protocol)
    assert empty["multiscale"]["selected"] == "K75"
    assert empty["age"]["selected"] == (0.0, 90)
    assert all(value is None for value in empty["age"]["candidates"][0]["block_scores"])
    nonempty = runner.select_kernel_parameters(
        [sample("I1"), sample("I2")], cutoff, np.ones(2), protocol
    )
    assert nonempty["multiscale"]["selected"] == "K75"
    assert nonempty["age"]["selected"] == (0.0, 90)
    assert nonempty["age"]["status"] == "selected_from_earlier_blocks"


def test_ridge_cv_passes_only_earlier_training_rows_and_distinct_cutoffs(monkeypatch):
    fold = {
        "id": "C_DEV_2000_2004",
        "outer_start": "2000-01-01T00:00:00+00:00",
        "inner_blocks": [
            {"id": "I1", "start": "1985-01-01T00:00:00+00:00"},
            {"id": "I2", "start": "1990-01-01T00:00:00+00:00"},
            {"id": "I3", "start": "1995-01-01T00:00:00+00:00"},
        ],
    }
    samples = [
        sample("I1", "1988-01-01T00:00:00+00:00"),
        sample("I2", "1993-01-01T00:00:00+00:00"),
        sample("I3", "1998-01-01T00:00:00+00:00"),
    ]
    seen = []

    def fake_fit(issues, areas, *, ridge_lambda):
        seen.append(([issue.issue_id.split(":")[1] for issue in issues], ridge_lambda))
        return SimpleNamespace(
            predict_log_mass=lambda base, features: base,
            to_dict=lambda: {
                "status": "synthetic",
                "event_count": sum(issue.future_counts.sum() for issue in issues),
            },
        )

    monkeypatch.setattr(runner, "fit_spatial_ridge", fake_fit)
    _, evidence = runner.select_and_fit_ridge(samples, fold, np.ones(2), with_m5=False)
    assert seen == [
        (["I1"], 10.0),
        (["I1", "I2"], 10.0),
        (["I1"], 1.0),
        (["I1", "I2"], 1.0),
        (["I1"], 0.1),
        (["I1", "I2"], 0.1),
        (["I1", "I2", "I3"], 10.0),
    ]
    assert evidence["selected_lambda"] == 10.0
    assert (
        evidence["validation"][0]["branches"][0]["training_label_cutoff_utc"]
        == "1989-12-02T00:00:00+00:00"
    )


def test_real_calendar_only_has_396_pairs_no_target_load():
    counts = [
        sum(
            len(runner._expected_horizon_axis(ROOT, fold, horizon))
            for fold in runner.DEVELOPMENT_FOLD_IDS
        )
        for horizon in runner.HORIZONS
    ]
    assert counts == [176, 116, 56, 32, 16]


def test_partial_final_manifest_never_unlocks_scoring(tmp_path, monkeypatch):
    identity = {"protocol_id": "synthetic"}
    monkeypatch.setattr(runner, "load_protocol", lambda root: {})
    monkeypatch.setattr(runner, "_identity", lambda protocol: identity)
    (tmp_path / "prediction_manifest.json").write_text(
        json.dumps(
            {**identity, "folds": [{"fold_id": fold} for fold in runner.DEVELOPMENT_FOLD_IDS[:3]]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_verify_fold",
        lambda *args: pytest.fail("incomplete run must fail before fold reading"),
    )
    with pytest.raises(ValueError, match="all four"):
        runner.verify_prediction_manifest(tmp_path, tmp_path)


def test_all_nine_models_share_axis_and_no_ridge_is_exact_base():
    protocol = runner.load_protocol(ROOT)
    kernel_selection = {"multiscale": {"selected": "K75"}, "age": {"selected": (0.0, 90)}}
    values = components()
    output = runner._prediction_models(values, kernel_selection, [None, None], protocol)
    assert output.shape == (9, 2)
    for row in output:
        np.testing.assert_array_equal(row, values["K75"])


def test_json_checkpoints_atomic_and_never_overwrite_completed(tmp_path):
    path = tmp_path / "record.json"
    runner._write_json(path, {"stage": "complete"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"stage": "complete"}
    assert not path.with_suffix(".partial.json").exists()
    with pytest.raises(FileExistsError):
        runner._write_json(path, {"stage": "replacement"})


def test_synthetic_horizon_artifacts_roundtrip_and_feed_location_scoring(tmp_path, monkeypatch):
    from seismoflux.multitask_s1.c2b_score import score_exposure
    from seismoflux.stage2s.contracts import SpatialGrid

    identity = {"protocol_id": "synthetic_interface_only"}
    fold_id = runner.DEVELOPMENT_FOLD_IDS[0]
    records = []
    monkeypatch.setattr(runner, "load_protocol", lambda project: {"inputs": {"grid_cells": 2}})
    monkeypatch.setattr(runner, "_expected_horizon_axis", lambda project, fold, horizon: [0])
    for horizon in runner.HORIZONS:
        payload = tmp_path / f"h{horizon}.npz"
        logs = np.broadcast_to(np.log([0.75, 0.25]), (1, 9, 2)).copy()
        runner._atomic_npz(
            payload,
            {
                "fold_id": np.asarray(fold_id),
                "model_ids": np.asarray(runner.MODEL_IDS),
                "issue_times_us": np.asarray([0], dtype=np.int64),
                "horizons_days": np.asarray([horizon], dtype=np.int64),
                "log_cell_mass": logs,
            },
        )
        metadata = tmp_path / f"h{horizon}.json"
        runner._write_json(
            metadata,
            {
                "identity": identity,
                "fold_id": fold_id,
                "horizon_days": horizon,
                "predictions": runner._record(tmp_path, payload),
            },
        )
        records.append(runner._record(tmp_path, metadata, horizon_days=horizon))
    fold_path = tmp_path / "fold.json"
    runner._write_json(fold_path, {"identity": identity, "fold_id": fold_id, "horizons": records})
    record = runner._record(tmp_path, fold_path, fold_id=fold_id)
    runner._verify_fold(tmp_path, tmp_path, record, identity)
    arrays = runner.load_fold_arrays(tmp_path, record)
    assert arrays["log_cell_mass"].shape == (5, 9, 2)
    assert arrays["horizons_days"].tolist() == list(runner.HORIZONS)
    grid = SpatialGrid(
        grid_id="synthetic",
        cell_size_km=25.0,
        cell_ids=("c0", "c1"),
        rows=np.zeros(2, dtype=np.int64),
        columns=np.asarray([0, 1], dtype=np.int64),
        query_xy_km=np.asarray([[0.0, 0.0], [25.0, 0.0]]),
        clipped_area_km2=np.asarray([625.0, 625.0]),
    )
    target = {
        "event_ids": ["e"],
        "event_cell_indices": [0],
        "episode_ids": ["episode"],
        "global_episode_member_counts": [1],
        "is_episode_anchor": [True],
        "event_longitudes": [100.0],
        "event_latitudes": [30.0],
    }
    exposure, events, _ = score_exposure(
        log_mass=arrays["log_cell_mass"][1, 0],
        grid=grid,
        target=target,
        fold_id=fold_id,
        horizon_days=30,
        issue_time_us=0,
        magnitude_bin="M5_6",
        model_id=runner.MODEL_IDS[0],
        budgets=[625.0],
        near_cells=[{0}],
    )
    assert exposure[0]["anchor_hits"] == 1
    assert events[0]["log_density_per_km2"] == pytest.approx(np.log(0.75) - np.log(625.0))


def test_inner_targets_are_global_m4_and_never_reused_as_feature_input():
    import pandas as pd

    rows = []
    for name, time, magnitude, inside in (
        ("train", "1989-01-01", 4.0, True),
        ("at_issue", "1990-01-01", 5.0, True),
        ("target", "1990-01-02", 4.0, True),
        ("too_small", "1990-01-03", 3.9, True),
        ("outside", "1990-01-04", 5.0, False),
        ("beyond", "1990-02-02", 5.0, True),
    ):
        instant = pd.Timestamp(time, tz="UTC")
        rows.append(
            dict(
                event_id=name,
                origin_time_utc=instant,
                available_at=instant,
                magnitude=magnitude,
                longitude=100.0,
                latitude=30.0,
                inside_study_area=inside,
                catalog_sources=["s"],
            )
        )
    source = pd.DataFrame(
        [
            dict(
                source_record_id="s",
                source_id="earthquake_catalog_m3_plus",
                origin_time_utc=pd.Timestamp("1980-01-01", tz="UTC"),
                available_at=pd.Timestamp("1980-01-01", tz="UTC"),
            )
        ]
    )
    catalog = c2b_catalog_from_frames(pd.DataFrame(rows), source)
    issue = datetime(1990, 1, 1, tzinfo=UTC)
    calls = []

    def get_components(only_issue):
        calls.append(only_issue)
        return components()

    inputs = SimpleNamespace(
        catalog=catalog.table,
        inner_exposures=(InnerExposure("C_DEV_2000_2004", "I1", 30, (issue,)),),
        spatial_domain=SimpleNamespace(locator=SimpleNamespace(locate_lonlat=lambda lon, lat: 0)),
    )
    fold = {
        "id": "C_DEV_2000_2004",
        "outer_start": "2000-01-01T00:00:00Z",
        "inner_blocks": [{"id": "I1", "end": "1995-01-01T00:00:00Z"}],
    }
    result = runner._inner_samples(inputs, fold, 30, SimpleNamespace(get=get_components))
    assert calls == [issue]
    assert result[0].counts(datetime(2000, 1, 1, tzinfo=UTC)).tolist() == [1.0, 0.0]
