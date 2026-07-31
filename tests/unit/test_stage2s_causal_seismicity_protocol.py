from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from shapely.geometry import box

from seismoflux.anomaly_increment.preregistration import verify_content_sha256
from seismoflux.features.anomaly.grid import build_stage3_query_grid

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "causal_seismicity_screen.yaml"
FOLD_PATH = ROOT / "data" / "manifests" / "causal_seismicity_screen_fold_manifest.json"
INPUT_PATH = (
    ROOT / "data" / "manifests" / "causal_seismicity_screen_target_blind_input_contract.json"
)
RESEARCH_CONFIG_PATH = ROOT / "configs" / "research_protocol.yaml"
SOURCE_FOLD_PATH = ROOT / "data" / "manifests" / "anomaly_increment_r2_fold_manifest.json"
ACCEPTANCE_PATH = ROOT / "docs" / "phase2s0_causal_seismicity_protocol_acceptance.md"
STAGE2P_REVIEW_PATH = ROOT / "docs" / "phase2p_target_blind_scientific_route_review.md"
STAGE2P_HANDOFF_PATH = ROOT / "docs" / "restart_handoff_2026-07-31_stage2p_route_review.md"
SCIENCE_REVIEW_PATH = ROOT / "docs" / "scientific_value_review_and_model_composition.md"
STAGE2P_SVG_PATH = ROOT / "docs" / "stage2p_route_selection.svg"
SHANGHAI = ZoneInfo("Asia/Shanghai")


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[cast(str, key)] = loader.construct_object(  # type: ignore[no-untyped-call]
            value_node,
            deep=deep,
        )
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(payload, dict)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_issue_utc(value: str) -> datetime:
    local = datetime.combine(date.fromisoformat(value), time.min, tzinfo=SHANGHAI)
    return local.astimezone(UTC)


def test_protocol_has_one_target_blind_stage2s_authority() -> None:
    config = _load_yaml(CONFIG_PATH)

    assert config["schema_version"] == 1
    assert config["protocol_version"] == "0.2.3"
    assert config["experiment_id"] == "stage2s-causal-seismicity-development-v1"
    assert config["gate_id"] == "G1-T"
    assert config["status"] == (
        "target_blind_protocol_content_accepted_remote_tag_required_for_execution"
    )
    assert config["frozen_on"] == "2026-07-30"
    assert config["governance"]["stage"] == {
        "protocol": "Stage2S-0",
        "development_attempt": "Stage2S-1",
        "branch_role": "background_dynamics_not_stage4_anomaly_revival",
    }
    assert (
        config["governance"]["maximum_new_foundational_protocol_corrections_after_this_freeze"] == 0
    )
    assert config["governance"]["on_new_foundational_P0"] == "stop_stage2s_and_retain_75km_KDE"
    assert config["governance"]["execution_authority_requires_remote_protocol_tag"] is True


def test_stage4_route_is_stopped_and_cannot_be_reused() -> None:
    config = _load_yaml(CONFIG_PATH)
    governance = config["governance"]
    models = config["allowed_models"]

    assert governance["stage4a_route_status"] == "stopped_before_code_freeze"
    assert governance["stage4a_attempt_consumed"] is False
    assert governance["stage4a_code_or_result_reuse"] == "forbidden"
    assert models["anomaly_feature_or_table_use"] == "forbidden"
    assert models["full_etas_hawkes"] == "forbidden"
    assert models["tree_neural_weak_or_self_supervised_model"] == "forbidden"
    contract = _load_json(INPUT_PATH)
    assert (
        "any_src/seismoflux/anomaly_increment/kde_dev_*.py_regardless_of_tracking_state"
        in contract["forbidden_sources"]
    )
    assert (
        "AST_transitive_import_closure_has_no_stage4_kde_dev_module_or_test_regardless_tracking_state"
        in config["execution_control"][
            "code_stage_non_target_preflight_must_pass_before_attempt_registration"
        ]
    )


def test_only_S0_S1_and_SP_are_allowed() -> None:
    config = _load_yaml(CONFIG_PATH)
    models = config["allowed_models"]
    mixtures = config["mixtures"]

    assert models["exact_order"] == ["S0", "S1", "SP"]
    assert models["confirmatory_candidate"] == "S1"
    assert models["comparators"] == ["S0", "SP"]
    assert models["additional_candidate_model"] == "forbidden"
    assert mixtures["S1"]["formula"] == "(1-alpha_R_fold)*S0 + alpha_R_fold*R"
    assert mixtures["SP"]["formula"] == "(1-alpha_P_fold)*S0 + alpha_P_fold*RP"


def test_long_term_background_is_the_frozen_fold4_75km_model() -> None:
    config = _load_yaml(CONFIG_PATH)
    background = config["long_term_background"]

    assert background["source_snapshot"] == "fold_4"
    assert background["fit_end_utc"] == "2019-12-31T16:00:00Z"
    assert background["selected_mc"] == 4.0
    assert background["bandwidth_km"] == 75.0
    assert background["support_reconstruction_uses_all_magnitudes_before_mc"] is True
    assert background["bandwidth_mc_support_or_domain_reselection"] == "forbidden"
    assert background["final_validation_background_use"] == "forbidden"
    assert background["public_registry_as_inference_payload"] == "forbidden"
    assert background["support_id"] == "local-support-788851371baf0e3b"
    assert background["support_common_mc"] == 4.0
    assert background["support_retained_area_fraction"] == 1.0
    assert background["support_retained_area_m2"] == 9415305754432.771
    assert background["support_historical_event_count"] == 24173
    assert background["support_base_cell_size_km"] == 500.0
    assert background["support_parent_cell_size_km"] == 1000.0


def test_recent_and_past_control_windows_are_unique_and_strictly_causal() -> None:
    recent = _load_yaml(CONFIG_PATH)["recent_seismicity"]

    assert recent["source_magnitude_minimum"] == 4.0
    assert recent["source_magnitude_weight"] == 1.0
    assert recent["spatial_bandwidth_km"] == 75.0
    assert recent["temporal_decay"] == "none"
    assert recent["most_recent_window"]["origin_interval"] == "(T-30d,T]"
    assert recent["most_recent_window"]["available_at_lte_issue_time"] is True
    assert recent["preceding_window_control"]["origin_interval"] == "(T-60d,T-30d]"
    assert recent["preceding_window_control"]["available_at_lte"] == "T_minus_30d"
    assert (
        recent["preceding_window_control"]["same_length_kernel_magnitude_and_domain_as_R"] is True
    )
    assert recent["density_changes_inside_forecast_horizon"] == "forbidden"
    assert recent["second_window_length_decay_bandwidth_or_magnitude_weight"] == "forbidden"


def test_continuous_density_scoring_and_grid_mass_operators_are_unique() -> None:
    density = _load_yaml(CONFIG_PATH)["continuous_spatial_density"]

    assert density["operational_normalization_grid_km"] == 12.5
    assert density["event_density_operator"].replace(" ", "") == (
        "continuous_normalized_gaussian_KDE_at_projected_event_coordinate_not_cell"
        "_interpolation_or_cell_average"
    )
    assert density["same_operator_for_alpha_fit_and_assessment_IG"] is True
    assert density["density_floor_or_clipping"] == "forbidden"
    assert density["operational_25km_mass"] == (
        "aggregate_operational_12_5km_normalized_masses_to_aligned_parent"
    )
    assert density["alarm_area_denominator"] == "exact_study_area_clipped_25km_parent_area"
    assert density["operational_25km_mass_sum_absolute_tolerance"] == 1.0e-12
    assert density["primary_convergence_pair_km"] == [25.0, 12.5]
    assert density["diagnostic_convergence_pair_km"] == [50.0, 25.0]
    assert density["convergence_failure_status"] == "invalid"


def test_weights_are_fit_only_on_h007_training_rows() -> None:
    mixtures = _load_yaml(CONFIG_PATH)["mixtures"]
    solver = mixtures["exact_solver"]

    assert mixtures["weight_bounds_inclusive"] == [0.0, 1.0]
    assert mixtures["one_weight_per_model_per_fold"] is True
    assert mixtures["weight_shared_across_7_30_90_day_assessments"] is True
    assert mixtures["weight_fit_rows"] == "expanding_nonoverlapping_h007_fit_exposures_only"
    assert mixtures["assessment_target_or_metric_use_for_weight"] == "forbidden"
    assert mixtures["fit_target_population"] == "unique_supported_M5_6_events_only"
    assert mixtures["fit_target_availability_rule"] == (
        "available_at_lte_that_fold_fit_target_end_inclusive_utc"
    )
    assert solver["first_derivative_formula"].startswith("sum_i_qX_i_minus_q0_i")
    assert solver["second_derivative_formula"].startswith("negative_sum_i_square")
    assert solver["flat_rule"] == "max_abs_qX_div_q0_minus_1_lte_1e_12"
    assert solver["derivative_sign_tolerance"] == 1.0e-12
    assert solver["concavity_positive_tolerance"] == 1.0e-12
    assert solver["if_derivative_at_zero_lte_positive_tolerance"] == 0.0
    assert solver["if_derivative_at_one_gte_negative_tolerance"] == 1.0
    assert solver["otherwise_bisection_iterations"] == 64
    assert solver["bisection_update_if_midpoint_derivative_gt_zero"] == "replace_left_endpoint"
    assert solver["bisection_return"] == "arithmetic_mean_of_final_left_and_right_endpoints"
    assert solver["completely_flat_tie"] == 0.0
    assert solver["density_floor_or_ratio_clipping"] == "forbidden"


def test_all_models_share_one_M5_6_rate_and_exact_compensator() -> None:
    shared = _load_yaml(CONFIG_PATH)["shared_rate_and_compensator"]

    assert shared["estimate"] == (
        "supported_M5_6_fit_event_count_divided_by_total_h007_fit_exposure_days"
    )
    assert shared["shared_exactly_by"] == ["S0", "S1", "SP"]
    assert shared["source_M4_rate_use"] == "forbidden"
    assert shared["candidate_specific_rate"] == "forbidden"
    assert shared["zero_or_nonfinite_rate_action"] == "evidence_insufficient"
    assert shared["cell_mass_sum_each_model_expected"] == 1.0
    assert shared["cell_mass_sum_each_model_absolute_tolerance"] == 1.0e-12
    assert shared["compensator_mass_source"] == "operational_12_5km_masses_aggregated_to_25km"
    assert shared["paired_global_compensator_difference_expected"] == 0.0
    assert shared["paired_global_compensator_difference_absolute_tolerance"] == 1.0e-10
    assert shared["write_zero_without_independent_numerical_recomputation"] == "forbidden"


def test_fold_manifest_is_target_blind_and_expanding() -> None:
    manifest = _load_json(FOLD_PATH)
    folds = cast(list[dict[str, Any]], manifest["folds"])

    assert manifest["status"] == "target_blind_calendar_only_no_execution_or_scoring_authority"
    assert manifest["target_bands_mutually_disjoint"] is True
    assert (
        manifest["security"]["contains_target_ids_coordinates_scores_hits_or_model_results"]
        is False
    )
    assert [fold["fold_index"] for fold in folds] == [1, 2, 3]
    fit_lists = [cast(list[str], fold["fit_issue_dates_local_h007"]) for fold in folds]
    assert fit_lists[1][: len(fit_lists[0])] == fit_lists[0]
    assert fit_lists[2][: len(fit_lists[1])] == fit_lists[1]
    assert [len(values) for values in fit_lists] == [12, 25, 38]


def test_fold_windows_are_nonoverlapping_and_fit_precedes_assessment() -> None:
    folds = cast(list[dict[str, Any]], _load_json(FOLD_PATH)["folds"])
    prior_assessment_end: datetime | None = None

    for fold in folds:
        fit_dates = cast(list[str], fold["fit_issue_dates_local_h007"])
        fit_issue_times = [_local_issue_utc(value) for value in fit_dates]
        assert all(right - left == timedelta(days=7) for left, right in pairwise(fit_issue_times))
        fit_end = datetime.fromisoformat(
            cast(str, fold["fit_target_end_inclusive_utc"]).replace("Z", "+00:00")
        )
        assert fit_end == _local_issue_utc(fit_dates[-1]) + timedelta(days=7)

        band = cast(dict[str, str], fold["assessment_band"])
        band_start = datetime.fromisoformat(band["start_exclusive_utc"].replace("Z", "+00:00"))
        band_end = datetime.fromisoformat(band["end_inclusive_utc"].replace("Z", "+00:00"))
        assert fit_end < band_start < band_end
        if prior_assessment_end is not None:
            assert prior_assessment_end < band_start
        prior_assessment_end = band_end

        by_horizon = cast(dict[str, list[str]], fold["assessment_issue_dates_local_by_horizon"])
        for horizon_text, issue_dates in by_horizon.items():
            horizon = int(horizon_text)
            windows = [
                (_local_issue_utc(issue), _local_issue_utc(issue) + timedelta(days=horizon))
                for issue in issue_dates
            ]
            assert all(start >= band_start and end <= band_end for start, end in windows)
            assert all(left[1] <= right[0] for left, right in pairwise(windows))


def test_fold_manifest_is_an_exact_safe_projection_of_the_bound_source_calendar() -> None:
    manifest = _load_json(FOLD_PATH)
    source = _load_json(SOURCE_FOLD_PATH)
    contract = _load_json(INPUT_PATH)
    config_source = _load_yaml(CONFIG_PATH)["source_contracts"]["rolling_fold_manifest"][
        "source_design"
    ]
    source_folds = cast(list[dict[str, Any]], source["joint_macro_rolling_folds"])
    projected_folds = cast(list[dict[str, Any]], manifest["folds"])

    assert _sha256(SOURCE_FOLD_PATH) == manifest["source_design"]["file_sha256"]
    assert verify_content_sha256(source) is True
    assert manifest["source_design"]["content_sha256"] == source["content_sha256"]
    assert config_source["path"] == manifest["source_design"]["path"]
    assert config_source["sha256"] == _sha256(SOURCE_FOLD_PATH)
    assert config_source["content_sha256"] == source["content_sha256"]
    assert contract["calendar"]["source_design_path"] == manifest["source_design"]["path"]
    assert contract["calendar"]["source_design_file_sha256"] == _sha256(SOURCE_FOLD_PATH)
    assert contract["calendar"]["source_design_content_sha256"] == source["content_sha256"]
    assert len(cast(str, source["content_sha256"])) == 64
    assert manifest["issue_semantics"]["target_window"] == "(T,T+h]"
    assert source["target_window_rule"] == "(issue_time,issue_time+h]"
    assert (
        manifest["issue_semantics"]["training_target_end_strictly_before_assessment_target_start"]
        is source["training_target_end_must_be_strictly_before_assessment_target_start"]
    )
    for projected, original in zip(projected_folds, source_folds, strict=True):
        assert projected["fold_index"] == original["fold_index"]
        assert projected["fit_target_end_inclusive_utc"] == original["fit_target_end_inclusive_utc"]
        assert projected["assessment_band"] == original["assessment_band"]
        assert projected["fit_issue_dates_local_h007"] == [
            value.removeprefix("development-h007-")
            for value in cast(list[str], original["fit_exposure_ids_7d"])
        ]
        expected_assessment = {
            str(horizon): [
                value.removeprefix(f"development-h{horizon:03d}-")
                for value in cast(
                    list[str],
                    original["assessment_exposure_ids_by_horizon"][str(horizon)],
                )
            ]
            for horizon in (7, 30, 90)
        }
        assert projected["assessment_issue_dates_local_by_horizon"] == expected_assessment


def test_source_contract_hashes_are_exact_without_opening_processed_data() -> None:
    config = _load_yaml(CONFIG_PATH)
    sources = config["source_contracts"]
    safe_paths = {
        "data_catalog": ROOT / sources["data_catalog"]["path"],
        "background_local_support_manifest": ROOT
        / sources["background_local_support_manifest"]["path"],
        "background_registry": ROOT / sources["background_registry"]["path"],
        "rolling_fold_manifest": ROOT / sources["rolling_fold_manifest"]["path"],
        "target_blind_input_contract": ROOT / sources["target_blind_input_contract"]["path"],
        "spatial_strata_manifest": ROOT / sources["spatial_strata_manifest"]["path"],
    }
    expected = {
        name: sources[name]["sha256"]
        for name in (
            "data_catalog",
            "background_local_support_manifest",
            "background_registry",
            "rolling_fold_manifest",
            "target_blind_input_contract",
            "spatial_strata_manifest",
        )
    }

    assert {name: _sha256(path) for name, path in safe_paths.items()} == expected
    catalog = sources["earthquake_catalog"]
    assert catalog["path"].startswith("data/processed/")
    assert catalog["byte_open_or_stat_before_remote_code_tag"] == "forbidden"
    assert catalog["byte_open_before_registered_attempt_and_target_read_CAS"] == "forbidden"
    study_area = sources["study_area"]
    assert study_area["path"].startswith("data/processed/")
    assert study_area["byte_open_or_stat_before_remote_code_tag"] == "forbidden"
    assert study_area["projected_geometry_sha256"] == (
        "c5074034de71fd24ff2d9e0277c49389436e41b720a82fdb45f620b67b2c7231"
    )
    assert study_area["projected_total_area_m2"] == 9415305754432.771
    for path_key, hash_key in (
        ("loader_source_path", "loader_source_sha256_at_protocol"),
        ("builder_source_path", "builder_source_sha256_at_protocol"),
        ("grid_primitives_source_path", "grid_primitives_source_sha256_at_protocol"),
    ):
        assert _sha256(ROOT / study_area[path_key]) == study_area[hash_key]
    assert study_area["projected_geometry_identity_algorithm"] == {
        "source_path": "src/seismoflux/background/local_support.py",
        "source_sha256_at_protocol": (
            "a66f6b113c9dd8da4b013a7869f0396dc3d643ac4fce63ebccefd374ce7e8aaa"
        ),
        "normalize": "shapely.normalize_on_projected_geometry",
        "serialize": (
            "shapely.to_wkb_hex_false_output_dimension_2_byte_order_1_include_srid_false"
        ),
        "digest": "SHA256_of_serialized_WKB_bytes",
    }
    assert (
        _sha256(ROOT / study_area["projected_geometry_identity_algorithm"]["source_path"])
        == study_area["projected_geometry_identity_algorithm"]["source_sha256_at_protocol"]
    )
    assert study_area["operational_grid_family_builder_symbol"] == (
        "seismoflux.background.grid.build_equal_area_grid_family"
    )
    assert study_area["operational_quadrature_representative_point"] == (
        "shapely.point_on_surface_of_exact_support_clipped_geometry"
    )


def test_target_blind_input_contract_preserves_cell_to_zone_mapping() -> None:
    contract = _load_json(INPUT_PATH)
    spatial = contract["spatial_inputs"]
    mapping = spatial["cell_mapping"]

    assert contract["security"] == {
        "real_catalog_bytes_read": False,
        "development_target_read": False,
        "independent_validation_target_read": False,
        "locked_test_read": False,
        "study_area_or_query_grid_bytes_read": False,
        "execution_authority": False,
    }
    assert mapping["row_count"] == 15697
    assert mapping["required_nonempty_zone_count"] == 39
    assert spatial["adapter_exact_return_fields"] == [
        "query_grid",
        "construction_zone_id_by_cell_id",
    ]
    assert spatial["query_grid_exact_runtime_fields"] == [
        "grid_id",
        "equal_area_crs",
        "cell_size_km",
        "cell_ids",
        "rows",
        "columns",
        "query_xy_m",
        "clipped_area_km2",
    ]
    assert spatial["mapping_keys_equal_query_grid_cell_ids"] is True
    assert spatial["mapping_grid_id_rows_columns_and_query_xy_match_runtime_query_grid"] is True
    assert (
        spatial["query_grid_coordinates_and_areas_are_local_target_independent_runtime_fields"]
        is True
    )
    assert (
        spatial["query_grid_coordinates_for_candidate_refinement_or_parameter_selection"] is False
    )
    assert spatial["raw_query_grid_coordinate_or_clipped_geometry_table_publication"] is False
    assert (
        spatial["derived_historical_result_map_rendering_after_result_without_raw_coordinate_table"]
        is True
    )
    assert spatial["construction_zone_mapping_contains_coordinates_or_geometry"] is False
    assert spatial["cell_to_zone_mapping_may_not_be_dropped"] is True


def test_frozen_query_grid_builder_has_complete_synthetic_runtime_fields() -> None:
    first = build_stage3_query_grid(box(109.9, 34.9, 110.1, 35.1))
    second = build_stage3_query_grid(box(109.9, 34.9, 110.1, 35.1))

    assert first.grid_id == second.grid_id
    assert first.equal_area_crs == second.equal_area_crs
    assert first.cell_size_km == 25.0
    assert first.cell_count > 0
    assert len(first.cell_ids) == len(set(first.cell_ids))
    np.testing.assert_array_equal(first.rows, second.rows)
    np.testing.assert_array_equal(first.columns, second.columns)
    np.testing.assert_array_equal(first.query_xy_m, second.query_xy_m)
    np.testing.assert_array_equal(first.clipped_area_km2, second.clipped_area_km2)
    assert np.all(np.isfinite(first.query_xy_m))
    assert np.all(first.clipped_area_km2 > 0.0)
    assert np.all(first.clipped_area_km2 <= 625.0)


def test_alarm_area_is_a_fixed_complete_600000km2_prefix() -> None:
    alarm = _load_yaml(CONFIG_PATH)["alarm_area"]

    assert alarm["query_grid_cell_size_km"] == 25.0
    assert alarm["target_independent_grid_only"] is True
    assert alarm["primary_budget_km2"] == 600000.0
    assert alarm["mass_source"] == (
        "operational_12_5km_normalized_masses_aggregated_to_aligned_25km_parent"
    )
    assert alarm["area_source"] == "exact_study_area_clipped_runtime_query_grid_cell_area"
    assert alarm["rank_value"] == (
        "aggregated_integrated_cell_mass_divided_by_exact_study_area_clipped_cell_area"
    )
    assert alarm["tie_break"] == [
        "cell_row_ascending",
        "cell_column_ascending",
        "cell_id_ascending",
    ]
    assert alarm["selection"] == "complete_prefix_with_cumulative_exact_area_lte_budget"
    assert alarm["partial_cell_skip_oversized_cell_or_budget_expansion"] == "forbidden"


def test_supported_information_gain_and_full_area_recall_have_distinct_frozen_denominators() -> (
    None
):
    config = _load_yaml(CONFIG_PATH)
    targets = config["calendar_and_targets"]
    evaluation = config["evaluation"]

    assert targets["information_gain_target"]["population"] == (
        "base_target_and_inside_exact_fold4_supported_domain"
    )
    assert targets["information_gain_target"]["unsupported_target_event_log_action"] == "excluded"
    assert targets["strict_recall_target"]["population"] == "all_base_targets_in_full_study_area"
    assert targets["strict_recall_target"]["unsupported_target_action"] == (
        "included_in_denominator_and_forced_miss_for_all_models"
    )
    assert targets["minimum_supported_IG_unique_physical_events_union"] == 20
    assert targets["minimum_full_area_recall_unique_physical_events_union"] == 20
    assert evaluation["information_gain"][
        "event_density_grid_lookup_interpolation_or_cell_average"
    ] == ("forbidden")
    assert evaluation["strict_recall"]["unsupported_target_hit"] == "forced_zero_all_models"


def test_bootstrap_has_two_preregistered_family_members() -> None:
    bootstrap = _load_yaml(CONFIG_PATH)["bootstrap"]

    assert bootstrap["replications"] == 2000
    assert bootstrap["unit"] == (
        "deduplicated_physical_event_block_with_all_fold_horizon_memberships"
    )
    assert bootstrap["strata"].endswith("x_supported_IG_bit")
    assert bootstrap["replicate_indices"] == "0_through_1999"
    assert bootstrap["entropy_uint128"].startswith("unsigned_big_endian_integer")
    assert bootstrap["empty_stratum_RNG_call"] == "none"
    assert bootstrap["point_estimate_included_as_replication"] is False
    assert bootstrap["quantile"]["method"] == "numpy_linear_frozen_explicitly"
    assert bootstrap["horizon_membership_signature"] == {
        "text_order": "b7_b30_b90",
        "binary_integer": "int_of_concatenated_b7b30b90_base_2",
        "example_001": "90_day_only",
    }
    assert bootstrap["same_multiplicity_for_all_models_contrasts_IG_and_recall"] is True
    assert bootstrap["predictions_alarms_rates_and_weights_refit_inside_bootstrap"] is False
    assert bootstrap["sequence_component_count_and_event_resampling_unit_count_reported"] is True
    assert "not_sequence_independence" in bootstrap["interval_scope"]
    assert bootstrap["failed_replication_replacement_or_seed_change"] == "forbidden"
    for family in ("information_gain", "fixed_area_recall"):
        interval = bootstrap["interval_families"][family]
        assert interval["members"] == ["S1_minus_S0", "S1_minus_SP"]
        assert interval["member_interval_coverage"] == 0.975
        assert interval["lower_percentile"] == 1.25
        assert interval["upper_percentile"] == 98.75


def test_decision_gate_requires_information_gain_recall_fold_and_region_evidence() -> None:
    gate = _load_yaml(CONFIG_PATH)["decision_gate"]
    required = set(gate["passed_development_signal_requires_all"])

    assert {
        "all_three_horizon_S1_minus_S0_IG_point_estimates_positive",
        "all_three_horizon_S1_minus_SP_IG_point_estimates_positive",
        "S1_minus_S0_overall_macro_IG_familywise_lower_gt_zero",
        "S1_minus_SP_overall_macro_IG_familywise_lower_gt_zero",
        "S1_minus_S0_fixed_600000km2_recall_gain_gte_5pp",
        "S1_minus_S0_fixed_600000km2_recall_familywise_lower_gt_zero",
        "S1_minus_SP_fixed_600000km2_recall_familywise_lower_gt_zero",
        "at_least_two_of_three_S1_minus_S0_fold_macro_recall_gains_positive",
        "at_least_two_of_three_S1_minus_SP_fold_macro_recall_gains_positive",
        "S1_minus_S0_fold_macro_recall_gain_median_positive",
        "S1_minus_SP_fold_macro_recall_gain_median_positive",
        "both_contrasts_regional_additive_and_metric_specific_LORO_passed",
        "both_delays_both_contrasts_IG_and_recall_gt_1e_12",
    }.issubset(required)
    assert gate["result_based_30_60_90_365_day_exponential_or_random_lag_retry"] == "forbidden"


def test_regional_mapping_and_LORO_are_frozen_before_target_read() -> None:
    regional = _load_yaml(CONFIG_PATH)["regional_robustness"]

    assert regional["region_id_source"] == "construction_zone_id_by_cell_id"
    assert regional["contrasts_required"] == ["S1_minus_S0", "S1_minus_SP"]
    assert regional["minimum_IG_event_bearing_zones"] == 2
    assert regional["minimum_recall_event_bearing_zones"] == 2
    assert regional["minimum_positive_additive_IG_event_bearing_zones_each_contrast"] == 2
    assert regional["minimum_positive_additive_recall_event_bearing_zones_each_contrast"] == 2
    assert regional["positive_definition"] == (
        "event_bearing_AND_additive_contribution_gt_sign_tolerance"
    )
    assert regional["report_all_39_zones_including_zero_event_zones"] is True
    assert regional["additive_closure_absolute_tolerance"] == 1.0e-10
    assert regional["strongest_positive_zone_each_contrast_each_metric"]["tie_break"] == (
        "unsigned_UTF8_construction_zone_id_bytes_ascending"
    )
    loro = regional["leave_one_region_out"]
    assert loro["type"] == "fixed_denominator_additive_residual"
    assert loro["model_weight_rate_density_alarm_mask_ranking_or_denominator_refit"] == "forbidden"
    assert loro["residual_macro_IG_each_contrast_gt_sign_tolerance"] is True
    assert loro["residual_macro_recall_gain_each_contrast_gt_sign_tolerance"] is True


def test_sequence_diagnostic_keeps_clusters_in_the_primary_result() -> None:
    sequence = _load_yaml(CONFIG_PATH)["sequence_dominance_diagnostic"]
    algorithm = sequence["sequence_algorithm"]

    assert sequence["role"] == "required_claim_scope_diagnostic_not_primary_pass_gate"
    assert sequence["retain_all_sequences_in_primary_analysis"] is True
    assert algorithm["graph_edge_if"] == {
        "absolute_origin_time_difference_seconds_lte": 2592000,
        "WGS84_geodesic_distance_m_lte": 75000.0,
        "conjunction": "AND",
    }
    assert algorithm["components"] == "undirected_connected_components"
    assert algorithm["component_id"] == "unsigned_UTF8_bytes_smallest_event_id"
    assert algorithm["event_order"] == [
        "origin_time_utc_ascending",
        "unsigned_UTF8_event_id_bytes_ascending",
    ]
    assert sequence["additive_statistics"]["contrasts"] == ["S1_minus_S0", "S1_minus_SP"]
    assert sequence["additive_statistics"]["global_compensator_assignment"] == (
        "separate_unassigned_residual_not_attributed_to_any_sequence"
    )
    assert sequence["largest_gain_component_each_contrast_each_metric"]["IG"] == (
        "maximum_additive_IG_event_component"
    )
    assert sequence["largest_gain_component_each_contrast_each_metric"]["recall"] == (
        "maximum_additive_recall_gain_component"
    )
    assert sequence["leave_out"]["refit_reweight_rerank_alarm_or_denominator_change"] == "forbidden"
    assert "sequence_component_count" in sequence["report"]
    assert "event_resampling_unit_count" in sequence["report"]
    assert "component_max_pairwise_geodesic_distance_km" in sequence["report"]
    assert sequence["aftershock_or_cluster_deletion_from_primary_result"] == "forbidden"


def test_latency_sensitivities_do_not_change_target_windows_or_select_a_delay() -> None:
    latency = _load_yaml(CONFIG_PATH)["catalog_latency_sensitivity"]

    assert latency["simulated_additional_delay_days"] == [1, 7]
    assert latency["run_each_delay_separately_and_require_both"] is True
    assert latency["recent_window_as_of_rule"] == "available_at_plus_delay_lte_T"
    assert latency["preceding_window_as_of_rule"] == "available_at_plus_delay_lte_T_minus_30d"
    assert latency["origin_interval_boundaries_unchanged"] is True
    assert latency["eligibility"] == (
        "intersection_of_fixed_origin_interval_and_delayed_availability_rule"
    )
    assert latency["same_window_kernel_domain_solver_and_assessment_as_primary"] is True
    assert latency["report_contrasts_each_delay"] == [
        "S1_delay_minus_S0",
        "S1_delay_minus_SP_delay",
    ]
    assert latency["threshold"] == 1.0e-12
    assert latency["selecting_best_delay_or_combining_delays"] == "forbidden"
    assert latency["any_required_metric_lte_threshold_action"] == (
        "evidence_insufficient_fragile_signal_stop_route"
    )


def test_attempt_and_target_read_are_still_zero_and_single_use() -> None:
    config = _load_yaml(CONFIG_PATH)
    governance = config["governance"]
    execution = config["execution_control"]

    assert governance["development_target_read_count_at_preregistration"] == 0
    assert governance["independent_validation_target_read_count_at_preregistration"] == 0
    assert governance["locked_test_read_count_at_preregistration"] == 0
    assert governance["development_scientific_attempts_allowed"] == 1
    assert execution["attempt_ledger"]["only_transition"] == "absent_to_registered_once_by_O_EXCL"
    assert execution["target_read_ledger"]["only_transition"] == (
        "absent_to_claimed_before_first_catalog_byte_by_O_EXCL"
    )
    assert (
        execution["target_read_ledger"][
            "claim_consumes_attempt_even_if_catalog_validation_or_run_fails"
        ]
        is True
    )
    single_open = execution["catalog_single_open"]
    assert single_open["physical_open_calls_allowed"] == 1
    assert single_open["path_stat_GetFileHash_or_parquet_path_open"] == "forbidden"
    assert single_open["parquet_parse"] == "pyarrow_BufferReader_over_same_in_memory_bytes"
    chain = single_open["prediction_seal_chain"]
    assert chain["fold_fit_receipt_path_pattern"] == (
        "data/interim/stage2s/causal_seismicity_screen/fold_{fold}/fit_receipt.json"
    )
    assert chain["issue_path_pattern"] == (
        "data/interim/stage2s/causal_seismicity_screen/fold_{fold}/"
        "issue_{YYYY-MM-DD}/prediction_seal.json"
    )
    assert chain["fold_order"] == [1, 2, 3]
    assert chain["all_issue_seals_in_fold_bind_same_fold_fit_receipt_sha256"] is True
    assert "previous_fold_prediction_seal_sha256_if_any" in chain["fold_fit_receipt_binds"]
    assert "no_assessment_membership_score_hit_or_metric_exposed" in chain["fold_fit_receipt_binds"]
    assert "current_fold_fit_receipt_sha256" in chain["each_fold_binds"]
    assert chain["master_required_before_assessment_target_view_and_scoring"] is True
    assert chain["master_binds"].count("non_target_preflight_receipt_sha256") == 1
    assert execution["rolling_role_separation"][
        "later_fold_fit_view_before_prior_fold_prediction_seal"
    ] == ("forbidden")
    assert execution["rolling_role_separation"][
        "prior_fold_score_hit_or_candidate_metric_to_later_fit_or_predictor"
    ] == ("forbidden")
    assert (
        "synthetic_cross_fold_and_cross_issue_role_overlap_precedence_and_seal_order"
        in execution["code_stage_non_target_preflight_must_pass_before_attempt_registration"]
    )
    assert execution["workers"] == 1
    assert execution["BLAS_threads_each_library"] == 1
    assert execution["reserve_physical_cores_minimum"] == 2


def test_rolling_fold_overlap_requires_the_frozen_seal_chain() -> None:
    manifest = _load_json(FOLD_PATH)
    config = _load_yaml(CONFIG_PATH)
    folds = cast(list[dict[str, Any]], manifest["folds"])
    chain = config["execution_control"]["catalog_single_open"]["prediction_seal_chain"]
    run_order = config["execution_control"]["development_run_order"]

    for earlier, later in pairwise(folds):
        assert datetime.fromisoformat(
            cast(str, later["fit_target_end_inclusive_utc"]).replace("Z", "+00:00")
        ) > datetime.fromisoformat(
            cast(str, earlier["assessment_band"]["start_exclusive_utc"]).replace("Z", "+00:00")
        )
    assert chain["fold_order"] == [1, 2, 3]
    assert "process_folds_strictly_1_then_2_then_3_with_prior_fold_seal_required" in run_order
    assert "expose_all_assessment_target_memberships_only_after_master_prediction_seal" in run_order


def test_existing_validation_is_not_relabelled_as_independent_stage2s_evidence() -> None:
    config = _load_yaml(CONFIG_PATH)
    governance = config["governance"]

    assert governance["independent_validation_authorized"] is False
    assert (
        "reused_stage2r_background_selection_period"
        in governance["existing_2022_2023_development_role"]
    )
    assert governance["existing_2024_2025_validation_role"] == (
        "previously_scored_by_stage2r_not_an_independent_stage2s_target"
    )
    assert config["long_term_background"]["final_validation_background_use"] == "forbidden"


def test_static_interactive_and_forecast_outputs_are_preregistered_without_false_claims() -> None:
    outputs = _load_yaml(CONFIG_PATH)["deliverables_after_real_development_attempt"]
    rules = outputs["visualization_rules"]

    assert outputs["static_before_target_result_required"] == [
        "stage2s_data_method_causal_timeline.svg"
    ]
    assert "S0_S1_SP_same_area_recall_and_information_gain.svg" in outputs["static"]
    assert "historical_frozen_assessment_relative_intensity_map.svg" in outputs["static"]
    assert "historical_frozen_assessment_backtest_explorer.html" in outputs["interactive_offline"]
    assert "historical_frozen_assessment_map_explorer.html" in outputs["interactive_offline"]
    assert outputs["interactive_provenance_panel_required_fields"] == [
        "S0_training_cutoff",
        "R_and_RP_origin_windows",
        "available_at_cutoff",
        "fold_alpha_R_alpha_P_and_shared_rate",
        "fold_and_horizon",
        "issue_fold_and_master_seal_sha256",
        "actual_alarm_area_km2",
    ]
    assert rules["baseline_candidate_and_control_visible_together"] is True
    assert rules["synthetic_or_engineering_chart_labeled_as_prediction_result"] == "forbidden"
    assert rules["absolute_earthquake_probability_claim"] == "forbidden"
    assert rules["stage2s_outputs_must_be_labeled_historical_reused_development_assessment"] is True
    assert rules["current_or_prospective_forecast_visual_in_stage2s"] == "forbidden"
    assert (
        "separate_post_result_prospective_protocol_required"
        in outputs["current_forecast_generation_authority"]
    )


def test_blueprint_and_research_protocol_agree_with_stage2s_and_stage2p_contracts() -> None:
    config = _load_yaml(CONFIG_PATH)
    research = _load_yaml(RESEARCH_CONFIG_PATH)
    blueprint = (ROOT / "SEISMOFLUX_IMPLEMENTATION_HANDOFF.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "causal_seismicity_screen_protocol.md").read_text(encoding="utf-8")
    research_doc = (ROOT / "docs" / "research_protocol.md").read_text(encoding="utf-8")
    acceptance = ACCEPTANCE_PATH.read_text(encoding="utf-8")
    route_review = STAGE2P_REVIEW_PATH.read_text(encoding="utf-8")
    stage2p_handoff = STAGE2P_HANDOFF_PATH.read_text(encoding="utf-8")
    science_review = SCIENCE_REVIEW_PATH.read_text(encoding="utf-8")
    stage2p_svg = STAGE2P_SVG_PATH.read_text(encoding="utf-8")

    assert "### 阶段2S" in blueprint and "因果近期地震活动增量" in blueprint
    assert "`S0`" in blueprint and "`S1`" in blueprint and "`SP`" in blueprint
    assert "本协议冻结后允许的新增 foundational 修订数为 0" in blueprint
    assert "开发目标读取" in protocol and "独立验证目标读取" in protocol
    assert "不是随机置乱" in protocol
    assert "不能" in protocol and "称为 Stage 2S 独立验证" in protocol
    assert "## 0.3 阶段 2S 因果近期地震活动预登记" in research_doc
    assert "### 阶段2P" in blueprint and "真正前瞻近期地震活动证据采集" in blueprint
    assert "## 0.5 阶段 2P 真正前瞻路线复审" in research_doc
    assert "本地结论" in acceptance and "协议内容通过" in acceptance
    assert "对预测效果的直接提升" in acceptance and "尚无" in acceptance
    assert "只允许逐路径暂存以下 10 个文件" in acceptance
    assert research["protocol_version"] == "1.4.0"
    stage2s = research["stage_2s_preregistration"]
    assert stage2s["machine_protocol"]["sha256"] == _sha256(CONFIG_PATH)
    assert stage2s["fold_manifest"]["sha256"] == _sha256(FOLD_PATH)
    assert stage2s["target_blind_input_contract"]["sha256"] == _sha256(INPUT_PATH)
    assert stage2s["models"] == config["allowed_models"]["exact_order"]
    stage2p = research["stage_2p_route_review"]
    assert stage2p["stage_id"] == "Stage2P-1A"
    assert stage2p["status"] == "accepted"
    assert stage2p["execution_authorized"] is False
    assert stage2p["real_issue_authorized"] is False
    assert stage2p["stage2p1_protocol_frozen"] is True
    assert (
        stage2p["stage2p1_next_authorized_action"]
        == "final_validation_commit_push_tag_readback_then_stage2p1b_synthetic_only"
    )
    assert stage2p["stage2p1_protocol_path"] == "configs/prospective_recent_seismicity.yaml"
    assert (
        stage2p["stage2p1_record_schema_path"] == "data/contracts/stage2p_prospective_records.json"
    )
    assert stage2p["research_only_not_stage10_or_G8"] is True
    assert stage2p["new_target_read_count"] == 0
    assert stage2p["locked_test_read_count"] == 0
    assert stage2p["locked_test_run_count"] == 0
    assert stage2p["locked_test_bypass_authorized"] is False
    assert stage2p["models"] == {
        "P0": "causal_long_term_75km_KDE_rebuilt_each_issue_from_same_T_snapshot",
        "P1": "0.5_P0_T_plus_0.5_recent_30d_M4plus_75km_KDE",
        "PP": "0.5_P0_T_plus_0.5_prior_origin_30d_M4plus_75km_KDE_known_at_T",
    }
    assert stage2p["P0_freezes_method_not_2023_density"] is True
    assert stage2p["recent_components"]["empty_R30_action"] == "P1_equals_P0_exactly"
    assert stage2p["recent_components"]["empty_RP30_action"] == "PP_equals_P0_exactly"
    assert stage2p["minimum_complete_mature_issues"] == 52
    assert stage2p["minimum_unique_deduplicated_M5_6_events"] == 20
    assert stage2p["primary_target"]["magnitude_bin"] == "M5_6"
    assert stage2p["primary_target"]["M6plus_role"].endswith("never_satisfies_primary_sample_gate")
    assert stage2p["maximum_on_time_issues"] == 104
    analysis = stage2p["exposure_and_analysis_contract"]
    assert analysis["formal_exposures"].endswith("nonoverlap_within_each_horizon")
    assert analysis["zero_event_exposures_retained"] is True
    assert analysis["intermediate_confirmatory_effect_display_or_testing_forbidden"] is True
    pass_gate = stage2p["pass_gate"]
    assert pass_gate["P1_minus_P0_macro_information_gain_familywise_lower_gt_zero"] is True
    assert pass_gate["P1_minus_PP_macro_information_gain_familywise_lower_gt_zero"] is True
    assert pass_gate["P1_minus_P0_recall_gain_familywise_lower_gt_zero"] is True
    assert pass_gate["P1_minus_PP_recall_gain_familywise_lower_gt_zero"] is True
    assert pass_gate["P1_minus_P0_recall_gain_pp_minimum"] == 5
    source_contract = stage2p["stage2p1_mandatory_source_snapshot_contract"]
    assert source_contract["seal_completed_before_issue_T"] is True
    assert (
        source_contract["append_only_previous_issue_hash_and_remote_time_anchor_required"] is True
    )
    assert (
        "source_id_institution_endpoint_or_file_identity_version_and_license"
        in (source_contract["must_bind"])
    )
    assert "seal_completed_at_utc_and_issue_T_utc" in source_contract["must_bind"]
    truth_contract = stage2p["stage2p1_mandatory_target_truth_contract"]
    assert truth_contract["required_before_first_issue"] is True
    assert "magnitude_location_and_identity_revision_policy" in truth_contract["must_bind"]
    assert (
        "immutable_target_cohort_hash_and_append_only_evaluation_revision_policy"
        in truth_contract["must_bind"]
    )
    model_contract = stage2p["stage2p1_mandatory_model_identity_contract"]["must_bind"]
    assert "study_polygon_projection_support_grid_and_region_map_hashes" in model_contract
    assert "code_protocol_input_model_and_output_density_hashes" in model_contract
    assert (
        stage2p["publication_acl"]["raw_normalized_deduplicated_rows_and_exact_coordinates"]
        == "local_restricted_unless_license_allows"
    )
    assert (
        stage2p["publication_acl"]["public_artifacts"]
        == "hashes_counts_aggregate_grids_and_license_permitted_overlays_only"
    )
    assert stage2p["anomaly_followup"]["freeze_before_any_P_effect_metric_is_unsealed"] is True
    assert stage2p["construction_followup"]["horizons_days"] == [90, 180, 365]
    assert stage2p["pass_interpretation"].endswith("not_G7_G8_stage10_or_business_promotion")
    assert stage2p["science_value_category"] == "necessary_enabler"
    assert "唯一去重 `M5_6" in route_review
    assert "确认性累计对比在唯一正式判定前保持密封" in route_review
    assert "tracked 清单元数据显示" in stage2p_handoff
    assert "不替代" in science_review and "G7/G8" in science_review
    assert "0.5\u00d7P0 + 0.5\u00d7R30" in stage2p_svg
    assert "20 个唯一 M5\u20136" in stage2p_svg
    assert "20 个独立 M5+" not in route_review + stage2p_handoff + stage2p_svg
    ET.parse(STAGE2P_SVG_PATH)


def test_science_value_review_is_complete_and_not_a_prediction_claim() -> None:
    review = _load_yaml(CONFIG_PATH)["science_value_review"]

    assert review["science_value_category"] == "necessary_enabler"
    assert review["direct_prediction_improvement"] == "none_before_real_development_attempt"
    assert review["decision"] == "adjust_to_stage2s_causal_seismicity_screen"
    assert review["next_scientific_test"] == (
        "run_one_three_fold_S1_minus_S0_and_SP_development_attempt"
    )
    assert "any_new_foundational_P0" in review["stop_condition"]
