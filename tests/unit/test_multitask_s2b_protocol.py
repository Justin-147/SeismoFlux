"""S2-B registration checks: only configuration and method documents are read."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
LAYERS = ("COMMON_UNIT", "COMMON_GEO", "COMMON_GD", "NATIVE_UNIT", "NATIVE_GD")
GEO = "geologic_total_rate_mm_per_year"
GD = "geodetic_total_rate_mm_per_year"


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load((ROOT / "configs/multitask_s2_b_slip_rate.yaml").read_text("utf-8"))


@pytest.fixture(scope="module")
def parent(protocol):
    path = ROOT / protocol["parent_protocol"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["parent_protocol_sha256"]
    return yaml.safe_load(path.read_text("utf-8"))


def test_parent_identity_method_document_and_no_new_evidence_claim(protocol, parent):
    assert protocol["parent_protocol"] == "configs/multitask_s2_a_fault_geometry.yaml"
    assert protocol["parent_protocol_sha256"] == (
        "d6e19dca67063030e8eafdfd766f13f2310a5e52790cc4b2b7fe8707cf58b5c9"
    )
    assert protocol["parent_contract"] == parent["parent_contract"]
    assert (ROOT / protocol["methods_document"]).is_file()
    assert protocol["S1_development_results_seen"] is True
    assert protocol["S2A_development_results_seen"] is True
    assert protocol["S2B_predictions_or_scores_seen_at_registration"] is False
    assert "not_historical_prospective" in protocol["scientific_role"]
    assert protocol["science_review"]["present_value"] == (
        "necessary_preparation_no_S2B_predictive_evidence"
    )


def test_catalog_grid_calendar_and_inner_selection_are_unchanged(protocol, parent):
    for field in ("calendar", "selection", "resources"):
        assert protocol[field] == parent[field]
    inputs = protocol["inputs"]
    for key in (
        "catalog_protocol",
        "catalog_protocol_sha256",
        "catalog_run",
        "catalog_prediction_manifest_sha256",
        "catalog_score_manifest_sha256",
        "catalog_main_model",
        "catalog_panel",
        "local_two_catalogs_effective_magnitude_type",
        "numeric_magnitude_conversion",
        "external_catalog_Ms_assumption",
        "old_catalog_cache_and_predictions",
        "missing_old_component_cache",
        "grid_id",
        "grid_cells",
        "grid_cell_size_km",
        "grid_area_km2",
        "national_support_percentage_gate",
        "local_Mc_masks_applied",
    ):
        assert inputs[key] == parent["inputs"][key]
    assert (
        hashlib.sha256((ROOT / inputs["catalog_protocol"]).read_bytes()).hexdigest()
        == (inputs["catalog_protocol_sha256"])
    )
    assert inputs["local_two_catalogs_effective_magnitude_type"] == "Ms"
    assert inputs["numeric_magnitude_conversion"] == "none"
    assert inputs["national_support_percentage_gate"] is None
    assert inputs["local_Mc_masks_applied"] is False
    assert sum(protocol["calendar"]["outer_issue_counts_per_horizon"]) == 396


def test_same_original_segment_table_only_two_total_rates_and_direct_geometry(protocol, parent):
    inputs, source = protocol["inputs"], protocol["inputs"]["fault_segments"]
    original = parent["inputs"]["geometry_sources"]["SIMPLE"]
    for key in (
        "path",
        "sha256",
        "id_column",
        "geometry_column",
        "expected_rows",
        "expected_usable_lines",
    ):
        assert source[key] == original[key]
    assert source["expected_unique_ids"] == 519
    assert source["allowed_columns"] == ["fault_segment_id", "simplified_geometry_wkb", GEO, GD]
    assert source["rate_units"] == "mm_per_year"
    assert source["expected_geologic_nonnull_and_positive"] == 385
    assert source["expected_geodetic_nonnull_and_positive"] == 515
    assert source["expected_both_nonnull"] == 385
    assert source["expected_geodetic_only"] == 130
    assert source["expected_both_missing"] == 4
    assert source["expected_both_nonnull"] + source["expected_geodetic_only"] == 515
    assert source["expected_both_nonnull"] + source["expected_geodetic_only"] + 4 == 519
    assert source["geologic_rate_range_mm_per_year"] == [0.01, 21.2]
    assert source["geodetic_rate_range_mm_per_year"] == [0.04, 18.22]
    assert inputs["original_historical_model_eligible"] is False
    assert inputs["source_eligibility_flags_modified"] is False
    assert inputs["crosswalk_attribute_transfer"] is False
    assert inputs["detailed_trace_attributes_used"] is False
    assert inputs["automatic_rate_imputation"] is False
    assert inputs["missing_rate_policy"] == (
        "unknown_not_zero_exclude_only_from_required_line_panel_not_earthquake_targets"
    )
    assert "no_component_reconstruction_no_winsorization" in inputs["rate_values"]
    assert "elapsed_ratio_at_snapshot" in inputs["excluded_features"]
    assert "long_term_hazard_score" in inputs["excluded_features"]


def test_common_native_support_and_five_weight_layers(protocol):
    assert protocol["panels"] == {
        "COMMON": {"required_nonnull_columns": [GEO, GD], "expected_segments": 385},
        "NATIVE_GD": {"required_nonnull_columns": [GD], "expected_segments": 515},
    }
    assert tuple(protocol["layers"]) == LAYERS
    assert protocol["layers"] == {
        "COMMON_UNIT": {"panel": "COMMON", "weight_column": None, "constant_weight": 1.0},
        "COMMON_GEO": {"panel": "COMMON", "weight_column": GEO, "constant_weight": None},
        "COMMON_GD": {"panel": "COMMON", "weight_column": GD, "constant_weight": None},
        "NATIVE_UNIT": {"panel": "NATIVE_GD", "weight_column": None, "constant_weight": 1.0},
        "NATIVE_GD": {"panel": "NATIVE_GD", "weight_column": GD, "constant_weight": None},
    }
    assert protocol["geometry_math"]["target_or_grid_filtering_by_layer_support"] is False
    assert protocol["geometry_math"]["layer_support_selection"] == (
        "rate_nonnull_only_not_earthquake_coordinates_or_results"
    )


def test_gaussian_length_weighted_integral_normalizes_only_after_line_sum(protocol):
    math = protocol["geometry_math"]
    assert math["kernel"] == "K_s(r)=exp(-r^2/(2*s^2))/(2*pi*s^2)"
    assert math["line_integral"] == (
        "q_s(x_i)=sum_j_in_panel(v_j*integral_over_L_j(K_s(norm(x_i-u))*d_length))"
    )
    assert math["normalized_cell_mass"] == "F_s(i)=A_i*q_s(x_i)/sum_k(A_k*q_s(x_k))"
    assert math["per_line_unit_mass_normalization"] is False
    assert math["line_contribution"] == (
        "proportional_to_projected_length_times_original_rate_or_unit_weight"
    )
    assert math["distance_units"] == "km"
    assert math["numerical_representation"] == "finite_float64_normalized_log_cell_mass"
    for key in (
        "Gaussian_tail_truncation_radius_km",
        "artificial_probability_floor",
        "additional_background_weight",
    ):
        assert math[key] is None


def test_fixed_quadrature_and_one_target_blind_diagnostic_are_not_selection_axes(protocol):
    numeric = protocol["numerical_integration"]
    assert numeric["method"] == "midpoint_quadrature_on_each_original_projected_straight_edge"
    assert numeric["subinterval_count"] == "ceil(edge_length_km/maximum_step_km)"
    assert numeric["quadrature_point_weight"] == "represented_edge_length_km_times_layer_weight"
    assert numeric["zero_length_edges"] == "zero_contribution"
    assert numeric["same_panel_layers_share_quadrature_points"] is True
    assert numeric["production_maximum_step_km"] == 3.125
    assert numeric["diagnostic_maximum_step_km"] == 6.25
    assert numeric["diagnostic_runs"] == 1
    assert numeric["diagnostic_uses_earthquake_targets"] is False
    assert numeric["diagnostic_metrics"] == [
        "normalized_layer_total_variation",
        "alarm_cell_differences_at_fixed_budgets",
    ]
    assert numeric["production_always_uses_fine_step"] is True
    assert numeric["step_selected_by_outer_results"] is False
    assert numeric["repeated_adaptive_refinement"] is False
    assert numeric["national_percentage_agreement_gate"] is None


def test_ten_models_finite_thirteen_candidates_and_inherited_ties(protocol, parent):
    models, math = protocol["models"], protocol["geometry_math"]
    assert tuple(models) == tuple(
        f"S2B_{layer}_{suffix}" for layer in LAYERS for suffix in ("ONLY", "CATALOG_MIX")
    )
    for layer in LAYERS:
        assert models[f"S2B_{layer}_ONLY"] == {"layer": layer, "family": "fault_only"}
        assert models[f"S2B_{layer}_CATALOG_MIX"] == {
            "layer": layer,
            "family": "catalog_mixture",
        }
    for key in (
        "scales_km",
        "scale_tie_order_km",
        "alpha_candidates",
        "alpha_zero",
        "mixture_tie_order",
    ):
        assert math[key] == parent["geometry_math"][key]
    candidates = [(0.0, None)] + [
        (alpha, scale)
        for alpha in math["alpha_candidates"]
        if alpha > 0.0
        for scale in math["scale_tie_order_km"]
    ]
    assert len(set(candidates)) == math["unique_mixture_candidates_per_layer"] == 13
    assert candidates[:4] == [(0.0, None), (0.25, 75.0), (0.25, 150.0), (0.25, 25.0)]
    assert protocol["selection"]["tie_tolerance"] == 1.0e-12


def test_identical_evaluation_and_all_fourteen_ordered_pairs(protocol, parent):
    evaluation = dict(protocol["evaluation"])
    inherited = dict(parent["evaluation"])
    assert evaluation.pop("reuse_same_C0_target_vectors_after_all_S2B_predictions") is True
    assert inherited.pop("reuse_same_C0_target_vectors_after_all_S2A_predictions") is True
    assert evaluation == inherited
    expected = [
        ("COMMON_GEO_CATALOG_MIX", "COMMON_UNIT_CATALOG_MIX"),
        ("COMMON_GD_CATALOG_MIX", "COMMON_UNIT_CATALOG_MIX"),
        ("NATIVE_GD_CATALOG_MIX", "NATIVE_UNIT_CATALOG_MIX"),
        ("COMMON_UNIT_CATALOG_MIX", "C2B_D0_MULTISCALE"),
        ("COMMON_GEO_CATALOG_MIX", "C2B_D0_MULTISCALE"),
        ("COMMON_GD_CATALOG_MIX", "C2B_D0_MULTISCALE"),
        ("NATIVE_UNIT_CATALOG_MIX", "C2B_D0_MULTISCALE"),
        ("NATIVE_GD_CATALOG_MIX", "C2B_D0_MULTISCALE"),
        ("COMMON_GD_CATALOG_MIX", "COMMON_GEO_CATALOG_MIX"),
        ("NATIVE_GD_CATALOG_MIX", "COMMON_GD_CATALOG_MIX"),
        ("NATIVE_UNIT_CATALOG_MIX", "COMMON_UNIT_CATALOG_MIX"),
        ("COMMON_GEO_ONLY", "COMMON_UNIT_ONLY"),
        ("COMMON_GD_ONLY", "COMMON_UNIT_ONLY"),
        ("NATIVE_GD_ONLY", "NATIVE_UNIT_ONLY"),
    ]
    expected = [(f"S2B_{a}", b if b.startswith("C2B_") else f"S2B_{b}") for a, b in expected]
    pairs = protocol["planned_pairs"]
    assert [tuple(pair[:2]) for pair in pairs] == expected
    assert len(set(expected)) == 14
    all_ids = set(protocol["models"]) | set(evaluation["references"])
    for candidate, reference, reason in pairs:
        assert candidate in all_ids and reference in all_ids and candidate != reference
        assert reason
    conditions = len(protocol["calendar"]["horizons_days"]) * len(evaluation["magnitude_bands"])
    conditions *= len(evaluation["area_budgets_km2"]) * 2  # strict 0 km and auxiliary 70 km
    assert conditions == 100
    assert len(all_ids) * conditions == len(pairs) * conditions == 1400
    text = (ROOT / protocol["methods_document"]).read_text("utf-8")
    table_pairs = []
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) >= 5 and fields[1].strip().isdigit():
            table_pairs.append((fields[2].strip(), fields[3].strip()))
    assert table_pairs == expected


def test_no_forbidden_runs_old_artifact_changes_or_new_claims(protocol):
    boundary = protocol["run_boundary"]
    assert boundary["before_real_run"] == "protocol_and_implementation_acceptance_test_commit_push"
    assert boundary["save_all_four_fold_predictions_before_outer_scoring"] is True
    assert boundary["current_snapshot_role_must_be_displayed"] is True
    for key in (
        "holdout_2020_2022_opened",
        "audit_2023_plus_opened",
        "locked_test_run",
        "future_catalog_downloads",
        "frozen_P1_modified",
        "science_first_Stage4_drafts_touched",
        "old_S1_predictions_or_scores_modified",
        "old_S2A_predictions_or_scores_modified",
        "S2A_predictor_rerun",
        "post_score_additional_scale_or_weight_search",
        "post_score_rate_transform_or_panel_change",
        "hazard_strain_or_dynamic_anomalies_in_this_run",
        "rate_components_or_moment_rates_reconstructed",
        "support_or_target_selection_by_earthquake_results",
    ):
        assert boundary[key] is False
    assert protocol["outputs"]["root"] == "outputs/multitask_s2/s2b_slip_rate_v1"
    assert protocol["outputs"]["preserve_completed_checkpoints"] is True
    assert protocol["resources"]["default_fold_workers"] == 2
    assert protocol["resources"]["maximum_fold_workers"] == 3
    assert protocol["resources"]["minimum_reserved_physical_cores"] >= 2
    assert protocol["resources"]["numerical_threads_per_worker"] == 1
    assert protocol["resources"]["priority"] == "BelowNormal"
