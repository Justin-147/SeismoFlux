"""Finite S2-A science contract; these tests do not open event or model score rows."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load((ROOT / "configs/multitask_s2_a_fault_geometry.yaml").read_text("utf-8"))


def test_frozen_catalog_identity_and_same_calendar(protocol):
    inputs = protocol["inputs"]
    path = ROOT / inputs["catalog_protocol"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == inputs["catalog_protocol_sha256"]
    parent = yaml.safe_load(path.read_text("utf-8"))
    for key in (
        "outer_folds",
        "horizons_days",
        "outer_issue_counts_per_horizon",
        "outer_total_issue_horizon_pairs",
    ):
        assert protocol["calendar"][key] == parent["calendar"][key]
    for key in ("grid_id", "grid_cells", "grid_cell_size_km", "grid_area_km2"):
        assert inputs[key] == parent["inputs"][key]
    assert protocol["calendar"]["catalog_delay_hours"] == 24
    assert protocol["calendar"]["outer_embargo_days"] == 30
    assert inputs["local_two_catalogs_effective_magnitude_type"] == "Ms"
    assert inputs["numeric_magnitude_conversion"] == "none"
    assert inputs["external_catalog_Ms_assumption"] is False


def test_two_geometry_sources_and_no_future_attributes(protocol):
    inputs = protocol["inputs"]
    sources = inputs["geometry_sources"]
    assert set(sources) == {"SIMPLE", "TRACE"}
    assert sources["SIMPLE"]["allowed_columns"] == ["fault_segment_id", "simplified_geometry_wkb"]
    assert sources["TRACE"]["allowed_columns"] == [
        "trace_id",
        "geometry_wkb",
        "usable_for_geometry",
    ]
    assert sources["SIMPLE"]["expected_usable_lines"] == 519
    assert sources["TRACE"]["expected_usable_lines"] == 7215
    assert inputs["original_historical_model_eligible"] is False
    assert inputs["source_eligibility_flags_modified"] is False
    assert inputs["crosswalk_attribute_transfer"] is False
    assert "elapsed_ratio_at_snapshot" in inputs["excluded_features"]
    assert inputs["cell_to_block_mapping"]["blocks"] == 39
    assert inputs["cell_to_block_mapping"]["sha256"] == (
        "171a500de9f9dd475f2c37a5426debc7c6f2d34ddd418056729c39b27118108e"
    )


def test_six_models_thirteen_mixture_candidates_and_explicit_ties(protocol):
    models, math = protocol["models"], protocol["geometry_math"]
    assert len(models) == 6
    for source in ("SIMPLE", "TRACE"):
        assert models[f"S2A_{source}_FAULT_ONLY"] == {
            "source": source,
            "representation": "fine",
            "family": "fault_only",
        }
        for suffix, representation in (("CATALOG_MIX", "fine"), ("COARSE_MIX", "coarse")):
            assert models[f"S2A_{source}_{suffix}"] == {
                "source": source,
                "representation": representation,
                "family": "catalog_mixture",
            }
    assert math["scales_km"] == [25.0, 75.0, 150.0]
    assert math["power"] == 2.0
    assert math["scale_tie_order_km"] == [75.0, 150.0, 25.0]
    assert math["alpha_candidates"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert math["unique_mixture_candidates_per_source_and_representation"] == 1 + 4 * 3
    assert math["alpha_zero"] == "one_exact_catalog_candidate_scale_null"
    assert math["coarse_control_is_formal_permutation_or_transfer_test"] is False
    assert "identical" in math["pure_fault_scale_alarm_ranking"]


def test_inner_catalog_never_uses_later_validation_labels(protocol):
    selection = protocol["selection"]
    assert selection["inner_catalog_branches"] == [
        {"train_blocks": ["I1"], "validate_block": "I2"},
        {"train_blocks": ["I1", "I2"], "validate_block": "I3"},
    ]
    assert selection["inner_catalog_minimum_nonempty_blocks"] == 2
    assert "explicit_exact_K75" in selection["I2_catalog_consequence"]
    assert selection["inner_catalog_training_end_and_label_visibility"] == (
        "at_or_before_validation_start_minus_30d"
    )
    assert selection["geometry_validation_blocks"] == ["I2", "I3"]
    assert selection["geometry_label_end_and_visibility"] == ("at_or_before_outer_start_minus_30d")
    assert selection["minimum_nonempty_blocks_for_geometry_selection"] == 1
    assert selection["zero_nonempty_blocks_mixture"] == "exact_catalog_alpha_zero"
    assert selection["zero_nonempty_blocks_fault_only"] == "fixed_scale_75km"
    assert selection["geometry_or_blocks_may_use_target_coordinates"] is False


def test_identical_cost_targets_and_complete_finite_comparisons(protocol):
    evaluation = protocol["evaluation"]
    assert evaluation["magnitude_bands"] == ["Ms5_6", "Ms6_plus"]
    assert evaluation["area_budgets_km2"] == [300000.0, 450000.0, 600000.0, 750000.0, 960000.0]
    assert evaluation["expected_main_anchor_count"] == 147
    assert evaluation["strict_hit_tolerance_km"] == 0
    assert evaluation["secondary_hit_tolerance_km"] == 70
    assert evaluation["bootstrap_replicates"] == 2000
    assert evaluation["confidence_interval_is_hard_adoption_gate"] is False
    all_ids = set(protocol["models"]) | set(evaluation["references"])
    pairs = protocol["planned_pairs"]
    assert len(pairs) == 11
    assert len({(pair[0], pair[1]) for pair in pairs}) == 11
    for candidate, reference, reason in pairs:
        assert candidate in all_ids and reference in all_ids
        assert candidate != reference and reason
    for source in ("SIMPLE", "TRACE"):
        assert any(
            pair[:2] == [f"S2A_{source}_CATALOG_MIX", f"S2A_{source}_COARSE_MIX"] for pair in pairs
        )


def test_no_support_veto_no_forbidden_run_and_controlled_resources(protocol):
    assert protocol["inputs"]["national_support_percentage_gate"] is None
    assert protocol["inputs"]["local_Mc_masks_applied"] is False
    boundary = protocol["run_boundary"]
    assert boundary["save_all_four_fold_predictions_before_outer_scoring"] is True
    for key in (
        "holdout_2020_2022_opened",
        "audit_2023_plus_opened",
        "locked_test_run",
        "future_catalog_downloads",
        "frozen_P1_modified",
        "science_first_Stage4_drafts_touched",
        "old_S1_predictions_or_scores_modified",
        "post_score_additional_scale_or_weight_search",
        "slip_attributes_hazard_or_strain_in_this_run",
    ):
        assert boundary[key] is False
    resources = protocol["resources"]
    assert resources["default_fold_workers"] == 2
    assert resources["maximum_fold_workers"] == 3
    assert resources["minimum_reserved_physical_cores"] >= 2
    assert resources["numerical_threads_per_worker"] == 1
