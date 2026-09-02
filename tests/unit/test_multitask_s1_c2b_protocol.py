"""Configuration-only science checks: never load outer target rows or raw catalogs."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load(
        (ROOT / "configs/multitask_s1_c2b_catalog_models.yaml").read_text("utf-8")
    )


@pytest.fixture(scope="module")
def parent(protocol):
    return yaml.safe_load((ROOT / protocol["parent_contract"]).read_text("utf-8"))


def test_three_panels_match_aggregate_ledger_and_source_semantics(protocol):
    inputs = protocol["inputs"]
    path = ROOT / inputs["panel_ledger"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == inputs["panel_ledger_sha256"]
    ledger = json.loads(payload)
    assert len(ledger["rows"]) == 24
    assert (
        inputs["membership"] == "any_source_record_visible_at_issue_cutoff_not_anchor_source_only"
    )
    assert inputs["effective_type_local_two_sources"] == "Ms"
    assert inputs["numeric_magnitude_conversion"] == "none"
    assert inputs["external_catalog_Ms_assumption"] is False
    expected = {
        "D0_CANONICAL_M4_1970": (1970, 4.0, None),
        "D1_M3SOURCE_M4_1980": (1980, 4.0, "earthquake_catalog_m3_plus"),
        "D2_M5SOURCE_M5_1950": (1950, 5.0, "earthquake_catalog_m5_plus"),
    }
    assert set(protocol["panels"]) == set(expected)
    assert {(row["panel_id"], row["cutoff_year_local"]) for row in ledger["rows"]} == {
        (panel_id, year)
        for panel_id in expected
        for year in (1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020)
    }
    for panel_id, (year, magnitude, source) in expected.items():
        panel = protocol["panels"][panel_id]
        assert panel == {
            "start_local": f"{year}-01-01T00:00:00+08:00",
            "magnitude_minimum": magnitude,
            "required_source": source,
        }
        for row in ledger["rows"]:
            if row["panel_id"] != panel_id:
                continue
            assert row["required_any_source_id"] == source
            assert row["magnitude_minimum_inclusive"] == magnitude
            assert row["effective_magnitude_type"] == "Ms"
            assert datetime.fromisoformat(
                row["training_start_inclusive_utc"]
            ) == datetime.fromisoformat(panel["start_local"])
    for field, ledger_name in (
        ("canonical_catalog", "canonical_event"),
        ("source_records", "source_record"),
    ):
        assert inputs[f"{field}_sha256"] == ledger["inputs"][ledger_name]["file_sha256"]
    assert inputs["canonical_catalog_rows"] == 40898
    assert inputs["source_record_rows"] == 43785


def test_exactly_nine_models_and_valid_paired_comparisons(protocol):
    fixed = protocol["fixed_models"]
    selected = protocol["selected_models"]
    ridge = protocol["ridge"]["models"]
    model_ids = set(fixed) | set(selected) | set(ridge)
    assert (len(fixed), len(selected), len(ridge), len(model_ids)) == (5, 2, 2, 9)
    assert all(
        model["panel"] in protocol["panels"] for model in [*fixed.values(), *selected.values()]
    )
    assert len(protocol["planned_pairs"]) == 9
    for pair in protocol["planned_pairs"]:
        assert len(pair) == 3
        candidate, reference, explanation = pair
        assert candidate in model_ids and reference in model_ids
        assert candidate != reference and explanation
    assert any(
        pair[:2] == ["C2B_D0_RIDGE_M5", "C2B_D0_RIDGE_CORE"] for pair in protocol["planned_pairs"]
    )


def test_finite_multiscale_and_age_candidates_are_normalized(protocol):
    selected = protocol["selected_models"]
    scale = selected["C2B_D0_MULTISCALE"]
    assert scale["bandwidth_order_km"] == protocol["location_math"]["kernel_bandwidths_km"]
    assert scale["bandwidth_order_km"] == [25.0, 75.0, 150.0]
    assert len(scale["candidates"]) == 4
    assert set(scale["tie_order"]) == set(scale["candidates"])
    for weights in scale["candidates"].values():
        assert len(weights) == 3 and all(
            math.isfinite(weight) and weight >= 0 for weight in weights
        )
        assert sum(weights) == pytest.approx(1.0, abs=1e-15)
    age = selected["C2B_D0_AGE_WEIGHTED"]
    assert age["half_life_days_candidates"] == [7.0, 30.0, 90.0]
    assert age["alpha_candidates"] == [0.0, 0.25, 0.5, 0.75]
    assert age["unique_candidates"] == 1 + 3 * 3 == 10
    assert age["alpha_zero"] == "one_exact_KDE75_candidate_no_half_life_search"
    for alpha in age["alpha_candidates"]:
        assert 0 <= alpha <= 1 and alpha + (1 - alpha) == pytest.approx(1.0)
    for name in ("C2B_D0_R30", "C2B_D1_R30"):
        assert protocol["fixed_models"][name]["alpha"] == 0.25
    assert protocol["location_math"]["boundary_normalization"] == (
        "each_component_over_whole_national_grid_before_mixing"
    )


def test_ridge_features_regularization_and_legal_forward_cv(protocol, parent):
    ridge = protocol["ridge"]
    core = ridge["models"]["C2B_D0_RIDGE_CORE"]
    full = ridge["models"]["C2B_D0_RIDGE_M5"]
    assert len(core) == 3 and len(full) == 4
    assert full[:-1] == core and full[-1] == "log_D2_K75_over_K75"
    assert ridge["lambda_candidates"] == [0.1, 1.0, 10.0]
    assert ridge["cross_validation"] == [
        {"train_blocks": ["I1"], "validate_block": "I2"},
        {"train_blocks": ["I1", "I2"], "validate_block": "I3"},
    ]
    assert ridge["training_target_end"] == "at_or_before_validation_block_start_minus_30d"
    assert ridge["label_available_cutoff"] == "at_or_before_validation_block_start_minus_30d"
    assert "separately_for_each_CV_branch" in ridge["scaler"]["fit_scope"]
    assert ridge["fewer_than_two_nonempty_validation_blocks"] == "fixed_lambda_10_then_legal_refit"
    assert "outer_start_minus_30d" in ridge["final_refit"]
    for fold in parent["outer_folds"]:
        blocks = {block["id"]: block for block in fold["inner_blocks"]}
        assert set(blocks) == {"I1", "I2", "I3"}
        for branch in ridge["cross_validation"]:
            validation_start = datetime.fromisoformat(blocks[branch["validate_block"]]["start"])
            for training_id in branch["train_blocks"]:
                assert datetime.fromisoformat(blocks[training_id]["end"]) <= validation_start
        assert datetime.fromisoformat(blocks["I3"]["end"]) <= (
            datetime.fromisoformat(fold["outer_start"]) - timedelta(days=30)
        )


def test_five_horizons_396_calendar_pairs_from_dates_without_targets(protocol, parent):
    calendar = protocol["calendar"]
    assert (
        calendar["horizons_days"]
        == parent["causal_boundaries"]["horizons_days"]
        == [7, 30, 90, 180, 365]
    )
    assert calendar["outer_folds"] == parent["execution_scope"]["enabled_outer_folds"]
    assert (
        calendar["main_catalog_delay_hours"]
        == parent["causal_boundaries"]["main_catalog_delay_hours"]
        == 24
    )
    assert (
        calendar["outer_embargo_days"]
        == parent["causal_boundaries"]["parameter_label_embargo_days"]
        == 30
    )
    counts = []
    for horizon in calendar["horizons_days"]:
        count = 0
        for fold in parent["outer_folds"]:
            start = datetime.fromisoformat(fold["outer_start"])
            end = datetime.fromisoformat(fold["outer_end"])
            issue = start + timedelta(days=(3 - start.weekday()) % 7)
            next_selected = issue
            while issue + timedelta(days=horizon) <= end:
                if issue >= next_selected:
                    count += 1
                    next_selected = issue + timedelta(days=horizon + 30)
                issue += timedelta(days=7)
        counts.append(count)
    assert counts == calendar["outer_issue_counts_per_horizon"] == [176, 116, 56, 32, 16]
    assert sum(counts) == calendar["outer_total_issue_horizon_pairs"] == 396


def test_no_new_gate_masks_or_forbidden_data_and_fixed_fair_evaluation(protocol):
    parent_run = yaml.safe_load((ROOT / protocol["parent_run_contract"]).read_text("utf-8"))
    assert protocol["inputs"]["national_support_percentage_gate"] is None
    assert protocol["inputs"]["local_Mc_masks_applied"] is False
    for key in (
        "holdout_2020_2022_opened",
        "audit_2023_plus_opened",
        "locked_test_run",
        "future_catalog_downloads",
        "frozen_P1_modified",
        "science_first_Stage4_drafts_touched",
        "C2A_masks_crossed_with_models",
        "ETAS_or_negative_binomial_reopened",
    ):
        assert protocol["run_boundary"][key] is False
    assert protocol["run_boundary"]["save_all_four_fold_predictions_before_outer_scoring"] is True
    assert protocol["selection"]["retrospective_targets_may_inform_features_or_boundaries"] is False
    assert protocol["evaluation"]["area_budgets_km2"] == parent_run["metrics"]["area_budgets_km2"]
    assert (
        protocol["inputs"]["grid_id"]
        == parent_run["input_identities"]["operational_grid"]["grid_id"]
    )
    assert protocol["evaluation"]["pooling_horizons_or_budgets_to_inflate_sample_size"] is False
    assert protocol["evaluation"]["confidence_interval_is_hard_adoption_gate"] is False
    assert protocol["evaluation"]["time_magnitude_or_joint_refit"] is False
    assert protocol["evaluation"]["expected_main_anchor_count"] == 147
    assert protocol["resources"]["numerical_threads_per_worker"] == 1
    assert protocol["resources"]["maximum_fold_workers"] <= 3
    assert protocol["resources"]["minimum_reserved_physical_cores"] >= 2
