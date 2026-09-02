from __future__ import annotations

import json
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from shapely.geometry import box
from shapely.strtree import STRtree

from seismoflux.multitask_s1.c2b_score import (
    _write_json,
    exposure_bootstrap,
    log_alarm_prefixes,
    projected_near_cells,
    run_score_phase,
    score_exposure,
    summarize,
    validate_targets,
)
from seismoflux.multitask_s1.development_predict import LOCATION_MODEL_IDS
from seismoflux.stage2s.contracts import SpatialGrid


def _grid():
    return SpatialGrid(
        grid_id="synthetic-only",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.zeros(4, dtype=np.int64),
        columns=np.arange(4),
        query_xy_km=np.array([[0, 0], [25, 0], [50, 0], [75, 0]], dtype=float),
        clipped_area_km2=np.array([200000.0, 250000.0, 300000.0, 300000.0]),
    )


def _target(empty=False):
    if empty:
        return {key: [] for key in _target()}
    return {
        "event_ids": ["e"],
        "event_cell_indices": [2],
        "episode_ids": ["episode"],
        "global_episode_member_counts": [4],
        "is_episode_anchor": [True],
        "event_longitudes": [100.0],
        "event_latitudes": [30.0],
    }


def test_target_adapter_preserves_empty_periods_and_separate_horizons():
    axes = {
        ("C_DEV_2000_2004", horizon, issue) for horizon in (7, 30) for issue in (0, 86400000000)
    }
    rows = []
    for fold, horizon, issue in sorted(axes):
        for band in ("M5_6", "M6_plus"):
            target = _target(empty=issue != 0 or band != "M5_6")
            for model in LOCATION_MODEL_IDS:
                time = pd.Timestamp(issue, unit="us", tz="UTC")
                payload = {
                    **target,
                    "fold_id": fold,
                    "horizon_days": horizon,
                    "issue_time_utc": time.isoformat(),
                    "magnitude_bin": band,
                    "model_id": model,
                    "metric": "spatial_log_density",
                    "event_count": len(target["event_ids"]),
                    "catalog_delay_hours": 24,
                    "hit_tolerance_km": 0.0,
                    "episode_definition": "full_catalog_fixed_anchor_30d_75km_by_magnitude_bin",
                }
                rows.append(
                    {
                        "score_family": "location",
                        "fold_id": fold,
                        "horizon_days": horizon,
                        "issue_time_utc": time,
                        "model_id": model,
                        "payload_json": json.dumps(payload),
                    }
                )
    frame = pd.DataFrame(rows)
    result = validate_targets(frame, axes, cell_count=4, expected_main_anchors=1)
    assert len(result) == 8
    assert sum(not target["event_ids"] for target in result.values()) == 6
    with pytest.raises(ValueError, match="omit empty"):
        validate_targets(frame.iloc[1:], axes, cell_count=4, expected_main_anchors=1)
    duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_targets(duplicate, axes, cell_count=4, expected_main_anchors=1)


def test_log_mass_tail_is_finite_and_prefix_is_no_skip():
    grid = _grid()
    logs = np.array([0.0, -1.0, -1000.0, -1001.0])
    logs -= logsumexp(logs)
    prefix = log_alarm_prefixes(logs, grid, [400000.0])[0]
    assert prefix["selected"] == [0]
    assert prefix["actual_area_km2"] == 200000.0
    scored, events, _ = score_exposure(
        log_mass=logs,
        grid=grid,
        target=_target(),
        fold_id="f",
        horizon_days=30,
        issue_time_us=0,
        magnitude_bin="M5_6",
        model_id="m",
        budgets=[400000.0],
        near_cells=[{0, 2}],
    )
    assert scored[0]["anchor_hits"] == 0
    assert scored[1]["anchor_hits"] == 1
    assert scored[1]["actual_area_km2"] == scored[0]["actual_area_km2"] == 200000.0
    assert events[0]["log_density_per_km2"] == pytest.approx(logs[2] - np.log(300000.0))
    assert np.isfinite(events[0]["log_density_per_km2"])
    assert scored[1]["episode_balanced_hits"] == 0.25


def test_empty_exposure_retained_with_no_event_mean():
    scored, events, alarms = score_exposure(
        log_mass=np.log(np.ones(4) / 4),
        grid=_grid(),
        target=_target(True),
        fold_id="f",
        horizon_days=7,
        issue_time_us=0,
        magnitude_bin="M6_plus",
        model_id="m",
        budgets=[600000.0],
        near_cells=[],
    )
    assert len(scored) == 2 and len(alarms) == 1 and not events
    assert all(row["anchor_total"] == 0 and row["event_mean_log_density"] is None for row in scored)


def test_paired_bootstrap_keeps_zero_periods_and_ratios_not_mean_recalls():
    delta = np.array([1.0, 0.0, 0.0])
    totals = np.array([2.0, 1.0, 0.0])
    assert exposure_bootstrap(delta, totals) == exposure_bootstrap(delta, totals)
    assert exposure_bootstrap(delta, totals) == [0.0, 50.0]
    assert exposure_bootstrap(np.zeros(2), np.zeros(2)) is None


def test_70km_uses_polygon_distance_and_includes_exact_boundary():
    tree = STRtree([box(0, 0, 25000, 25000)])
    neighborhoods = projected_near_cells(tree, [95000, 95000.001, 12500], [12500] * 3)
    assert neighborhoods == [{0}, set(), {0}]


def test_incomplete_prediction_gate_precedes_any_target_read(tmp_path, monkeypatch):
    module = ModuleType("seismoflux.multitask_s1.c2b_predict")
    module.MODEL_IDS = ("synthetic",)
    module.PROTOCOL_PATH = "unused.yaml"
    module.load_protocol = lambda project: {"outputs": {"root": "synthetic_output"}}
    module.load_fold_arrays = lambda *args: pytest.fail("must not load partial arrays")

    def incomplete(*args):
        raise ValueError("four folds not complete")

    module.verify_prediction_manifest = incomplete
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *args, **kwargs: pytest.fail("must not open targets")
    )
    with pytest.raises(ValueError, match="four folds not complete"):
        run_score_phase(project_root=tmp_path, data_root=tmp_path)


def test_summary_json_handles_pandas_group_key_scalars(tmp_path):
    path = tmp_path / "summary.json"
    _write_json(path, {"horizon_days": np.int64(30), "area": np.float64(600000)})
    assert json.loads(path.read_text()) == {"horizon_days": 30, "area": 600000}


def test_summary_reports_paired_gain_without_pooling_and_retains_empty_period():
    exposure_rows, event_rows = [], []
    for horizon in (7, 30):
        for issue in (0, 86400000000):
            for model, masses in (
                ("candidate", [0.1, 0.1, 0.7, 0.1]),
                ("reference", [0.7, 0.1, 0.1, 0.1]),
            ):
                rows, events, _ = score_exposure(
                    log_mass=np.log(masses),
                    grid=_grid(),
                    target=_target(issue != 0),
                    fold_id="f",
                    horizon_days=horizon,
                    issue_time_us=issue,
                    magnitude_bin="M5_6",
                    model_id=model,
                    budgets=[300000.0],
                    near_cells=[] if issue else [{2}],
                )
                exposure_rows.extend(rows)
                event_rows.extend(events)
    curves, pairs, details = summarize(
        pd.DataFrame(exposure_rows),
        pd.DataFrame(event_rows),
        [["candidate", "reference", "synthetic"]],
        (),
    )
    assert len(curves) == 8 and len(pairs) == 4
    assert all(row["exposure_count"] == 2 and row["empty_exposure_count"] == 1 for row in curves)
    assert all(row["anchor_total"] == 1 and row["net_hits"] == 1 for row in pairs)
    assert len(details) == 4
