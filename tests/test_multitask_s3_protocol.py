"""Score-blind checks of the finite S3-A scientific protocol, not model skill."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return yaml.safe_load((ROOT / "configs/multitask_s3_anomaly.yaml").read_text(encoding="utf-8"))


def test_only_registered_a_development_roles_are_opened():
    p = _protocol()
    s0 = yaml.safe_load((ROOT / "configs/multitask_s0.yaml").read_text(encoding="utf-8"))
    roles = {f["id"]: f for f in s0["anomaly_time_folds"]["outer_folds"]}
    assert p["access"]["allowed_roles"] == ["A_DEV_2023_2024", "A_DEV_2024_2025"]
    for fold in p["folds"]:
        old = roles[fold["id"]]
        assert fold["target_start"][:10] == old["target_block_start"]
        assert fold["target_end_exclusive"][:10] == old["target_block_end_exclusive"]
        assert old["role"] == "development"
    assert "A_AUDIT_2025_2026" in p["access"]["forbidden_score_roles"]
    assert p["access"]["report_end_exclusive"] == "2025-07-01T00:00:00+08:00"
    assert not p["access"]["read_all_205_reports_then_filter"]


def test_designs_are_exactly_nested_5_16_20_without_hidden_missing_columns():
    f = _protocol()["features"]
    assert len(f["raw"]) == 16
    assert len(f["missing_controls"]) == 4
    all_ids = [c["id"] for c in f["raw"] + f["missing_controls"]]
    assert len(set(all_ids)) == 20
    designs = f["designs"]
    for name, size in [("COV", 5), ("SNAP", 16), ("DYN", 20)]:
        assert len(designs[name]) == len(set(designs[name])) == size
    assert set(designs["COV"]) < set(designs["SNAP"]) < set(designs["DYN"])
    assert set(designs["DYN"]) == set(all_ids)
    assert set(designs["DYN"]) - set(designs["SNAP"]) == {
        "listed_slope", "listed_acceleration", "first_seen_slope", "dynamic_missing"
    }


def test_inner_blocks_end_before_outer_label_embargo():
    p = _protocol()
    for fold in p["folds"]:
        cutoff = datetime.fromisoformat(fold["target_start"]) - timedelta(days=30)
        for key in fold["inner_blocks"]:
            start, end = map(datetime.fromisoformat, p["inner_blocks"][key])
            assert start < end <= cutoff
    selection = p["models"]["selection"]
    assert selection["required_evaluable_inner_blocks"] == 2
    assert selection["fewer_inner_blocks_lambda"] == 10
    assert selection["ridge_candidates"] == [0.1, 1.0, 10.0]


def test_inherited_tasks_and_no_national_veto():
    p = _protocol()
    assert p["targets"]["horizons_days"] == [7, 30, 90, 180, 365]
    assert p["targets"]["formal_magnitude_bands"] == ["Ms5_6", "Ms6_plus"]
    assert p["support"]["nationwide_quality_area_veto"] is None
    assert not p["support"]["local_Mc_mask"]
    assert p["support"]["alarm_area_budgets_km2"] == [300000, 450000, 600000, 750000, 960000]
    assert p["time"]["catalog_latency_hours"] == 24
    for weights in p["background"]["weights_by_horizon"].values():
        assert abs(sum(weights) - 1) < 1e-14
    assert not p["background"]["new_catalog_parameter_search"]


def test_count_units_and_offline_null_boundary_are_explicit():
    p = _protocol()
    count = p["models"]["count"]
    assert "complete h-day window" in count["formula"]
    assert not count["learns_magnitude_distribution"]
    assert not count["intercept_penalized"]
    assert "T0_CAL" in count["variants"]
    null = p["placebos"]
    assert null["time_replicates_per_fold"] == null["space_replicates_per_fold"] == 200
    assert null["role"].startswith("offline_attribution")
    assert "training_and_validation_issue_cutoffs_for_h" in null["time"]["boundaries"]
    assert not null["space"]["source_time_changes"]
    assert "coverage_missing_control" in null["fixed"]
    assert p["evaluation"]["save_all_outer_predictions_before_reading_any_outer_effect_scores"]
