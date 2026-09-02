"""Synthetic checks only: no real target or saved prediction is opened."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from seismoflux.multitask_s1 import c2b_score
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s2 import strain_score as scorer
from seismoflux.multitask_s2.strain import blend_log_masses
from seismoflux.stage2s.contracts import SpatialGrid

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = yaml.safe_load((ROOT / "configs/multitask_s2_c_strain.yaml").read_text("utf-8"))
MODELS = tuple(PROTOCOL["models"])
REFERENCES = tuple(PROTOCOL["evaluation"]["references"])
BUDGETS = PROTOCOL["evaluation"]["area_budgets_km2"]


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")


def _grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="synthetic-s2c-not-real",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.zeros(4, dtype=np.int64),
        columns=np.arange(4, dtype=np.int64),
        query_xy_km=np.array([[0, 0], [25, 0], [50, 0], [75, 0]], dtype=float),
        clipped_area_km2=np.array([200000.0, 250000.0, 300000.0, 300000.0]),
    )


def _target(empty=False, event_id="synthetic-event") -> dict:
    value = {
        "event_ids": [event_id],
        "episode_ids": [event_id + "-episode"],
        "global_episode_member_counts": [3],
        "is_episode_anchor": [True],
        "event_cell_indices": [2],
        "event_longitudes": [100.0],
        "event_latitudes": [30.0],
    }
    return {key: [] for key in value} if empty else value


def _arguments(empty=False):
    return dict(
        grid=_grid(),
        target=_target(empty),
        fold_id=DEVELOPMENT_FOLD_IDS[0],
        horizon_days=30,
        issue_time_us=0,
        magnitude_bin="M5_6",
        model_id=MODELS[0],
        budgets=BUDGETS,
        near_cells=[] if empty else [{2}],
    )


def test_finite_and_alpha_zero_scores_equal_old_catalog_algorithm_exactly():
    catalog = np.log([0.4, 0.3, 0.2, 0.1])
    physical = np.array([math.log(0.6), math.log(0.4), -math.inf, -math.inf])
    alpha_zero = blend_log_masses(catalog, physical, 0.0)
    np.testing.assert_array_equal(alpha_zero, catalog)
    for empty in (False, True):
        arguments = _arguments(empty)
        assert scorer.score_exposure(log_mass=alpha_zero, **arguments) == c2b_score.score_exposure(
            log_mass=catalog, **arguments
        )
    assert scorer.log_alarm_prefixes(catalog, _grid(), BUDGETS) == c2b_score.log_alarm_prefixes(
        catalog, _grid(), BUDGETS
    )


def test_true_zero_retains_event_denominators_and_zero_ties():
    logs = np.array([math.log(0.6), math.log(0.4), -math.inf, -math.inf])
    exposures, events, alarms = scorer.score_exposure(log_mass=logs, **_arguments())
    assert len(exposures) == len(events) == 10
    assert len(alarms) == 5
    assert all(row["event_count"] == row["anchor_total"] == 1 for row in exposures)
    assert all(row["event_mean_log_density"] == -math.inf for row in exposures)
    assert all(row["log_density_per_km2"] == -math.inf for row in events)
    assert alarms[0]["selected_cell_indices"] == [0]
    assert alarms[3]["selected_cell_indices"] == [0, 1, 2]
    assert all(row["actual_area_km2"] <= row["area_budget_km2"] for row in alarms)
    assert not events[0]["hit"]
    assert events[6]["hit"]
    empty, empty_events, _ = scorer.score_exposure(log_mass=logs, **_arguments(True))
    assert not empty_events
    assert all(row["event_count"] == 0 and row["event_mean_log_density"] is None for row in empty)


def test_tiny_finite_density_never_underflows_into_zero_ranking():
    logs = np.array([0.0, -math.inf, -1000.0, -math.inf])
    prefixes = scorer.log_alarm_prefixes(logs, _grid(), BUDGETS)
    assert prefixes[2]["selected"] == [0, 2]


def test_json_distinguishes_negative_infinity_from_empty_and_rejects_nan():
    curves = [{"event_mean_log_density": value} for value in (-math.inf, None, -10.0)]
    scorer._json_log_scores(curves)
    assert [item["log_density_status"] for item in curves] == [
        "negative_infinity_from_zero_mass",
        "no_events",
        "finite",
    ]
    json.dumps(curves, allow_nan=False)
    for invalid in (math.nan, math.inf):
        with pytest.raises(ValueError, match="NaN"):
            scorer._json_log_scores([{"event_mean_log_density": invalid}])


def test_all_fold_axes_allow_negative_infinity_but_not_invalid_or_all_zero():
    protocol = {
        "calendar": {
            "outer_folds": DEVELOPMENT_FOLD_IDS,
            "horizons_days": [30],
            "outer_total_issue_horizon_pairs": 4,
            "outer_issue_counts_per_horizon": [4],
        },
        "models": {model: {} for model in MODELS},
        "inputs": {"grid_cells": 4},
    }
    manifest = {"folds": [{"fold_id": fold} for fold in DEVELOPMENT_FOLD_IDS]}
    values = np.array([math.log(0.6), math.log(0.4), -math.inf, -math.inf])
    arrays = {
        fold: {
            "horizons_days": np.array([30]),
            "issue_times_us": np.array([0]),
            "model_ids": np.array(MODELS),
            "log_cell_mass": np.tile(values, (1, 4, 1)),
        }
        for fold in DEVELOPMENT_FOLD_IDS
    }
    assert len(scorer._prediction_axes(manifest, arrays, protocol, MODELS, (30,))) == 4
    for invalid in (math.nan, math.inf, -math.inf):
        damaged = copy.deepcopy(arrays)
        damaged[DEVELOPMENT_FOLD_IDS[0]]["log_cell_mass"][0, 0] = invalid
        with pytest.raises(ValueError, match="log mass"):
            scorer._prediction_axes(manifest, damaged, protocol, MODELS, (30,))


@pytest.mark.parametrize("failure", ["verifier", "three_folds"])
def test_all_prediction_folds_are_required_before_any_target_access(tmp_path, monkeypatch, failure):
    module = ModuleType("seismoflux.multitask_s2.strain_predict")
    module.MODEL_IDS, module.HORIZONS, module.PROTOCOL_PATH = (
        MODELS,
        (7, 30, 90, 180, 365),
        Path("unused"),
    )
    module.load_protocol = lambda project: {"outputs": {"root": "outputs/multitask_s2/synthetic"}}
    module._output_root = (
        lambda project, protocol, output: output or project / protocol["outputs"]["root"]
    )

    def verify(*args):
        if failure == "verifier":
            raise ValueError("four folds incomplete")
        return {"folds": [{"fold_id": fold} for fold in DEVELOPMENT_FOLD_IDS[:3]]}

    module.verify_prediction_manifest = verify
    module.load_fold_arrays = lambda *args: pytest.fail("incomplete fold set")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("target read"))
    with pytest.raises(ValueError, match="four"):
        scorer.run_score_phase(project_root=tmp_path, data_root=tmp_path)


def test_registered_later_slice_has_same_comparisons_and_only_its_own_samples():
    collections = {name: [] for name in scorer.TABLE_NAMES}
    logs = np.log([0.4, 0.3, 0.2, 0.1])
    for fold in DEVELOPMENT_FOLD_IDS:
        for horizon in PROTOCOL["calendar"]["horizons_days"]:
            for band in scorer.BANDS:
                for issue in (0, 86400000000):
                    target = _target(issue != 0, f"synthetic-{fold}-{horizon}-{band}")
                    for model in (*REFERENCES, *MODELS):
                        rows = scorer.score_exposure(
                            log_mass=logs,
                            grid=_grid(),
                            target=target,
                            fold_id=fold,
                            horizon_days=horizon,
                            issue_time_us=issue,
                            magnitude_bin=band,
                            model_id=model,
                            budgets=BUDGETS,
                            near_cells=[{2}] if issue == 0 else [],
                        )
                        for name, frame_rows in zip(scorer.TABLE_NAMES, rows, strict=True):
                            collections[name].extend(frame_rows)
    frames = {name: pd.DataFrame(rows) for name, rows in collections.items()}
    exposures, events = frames[scorer.TABLE_NAMES[0]], frames[scorer.TABLE_NAMES[1]]
    full, _ = scorer._slice_summary(
        exposures,
        events,
        PROTOCOL,
        folds=DEVELOPMENT_FOLD_IDS,
        expected_main_anchors=4,
        role="synthetic-only",
    )
    later, paired = scorer._slice_summary(
        exposures,
        events,
        PROTOCOL,
        folds=(DEVELOPMENT_FOLD_IDS[-1],),
        expected_main_anchors=1,
        role="synthetic-later-only",
    )
    assert len(full["curves"]) == len(later["curves"]) == 800
    assert len(full["pairings"]) == len(later["pairings"]) == 600
    assert full["primary_issue_horizon_count"] == 40
    assert later["primary_issue_horizon_count"] == 10
    assert set(paired.fold_id) == {DEVELOPMENT_FOLD_IDS[-1]}
    assert later["new_independent_test_evidence"] is False
    assert all(row["net_hits"] == 0 for row in later["pairings"])
    assert all(len(row["per_fold"]) == 1 for row in later["pairings"])
    json.dumps(later, allow_nan=False)
    damaged = exposures.copy()
    damaged.loc[0, "event_log_density_sum"] = math.nan
    with pytest.raises(ValueError, match="silently omitted"):
        scorer._summarize(damaged, events, PROTOCOL)
    with pytest.raises(ValueError, match="six"):
        scorer._summarize(exposures, events, {"planned_pairs": PROTOCOL["planned_pairs"][:-1]})
