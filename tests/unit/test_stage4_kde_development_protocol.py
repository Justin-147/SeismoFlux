from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "anomaly_increment_kde_dev.yaml"
PROTOCOL_DOCUMENT_PATH = ROOT / "docs" / "anomaly_increment_kde_dev_protocol.md"

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "protocol_version",
    "stage",
    "experiment_id",
    "gate_id",
    "status",
    "frozen_on",
    "language",
    "authority",
    "freeze",
    "science_question",
    "inputs",
    "access_boundary",
    "background",
    "variants",
    "features",
    "model",
    "calendar",
    "alarm_region",
    "randomness",
    "evaluation",
    "decision_gate",
    "execution_control",
    "science_value_review",
    "resources",
    "outputs",
}

PUBLIC_JSON_ALLOWLIST = {
    "data/manifests/anomaly_feature_dictionary.json",
    "data/manifests/anomaly_feature_registry.json",
    "data/manifests/anomaly_increment_kde_dev_inherited_contracts.json",
    "data/manifests/anomaly_increment_r2_feature_set.json",
    "data/manifests/anomaly_increment_r2_fold_manifest.json",
    "data/manifests/anomaly_increment_r2_spatial_strata.json",
    "data/manifests/background_local_support_model_registry.json",
    "data/manifests/etas_numerical_qualification_result.json",
}


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
        result[key] = loader.construct_object(  # type: ignore[no-untyped-call]
            value_node,
            deep=deep,
        )
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml(path: Path = CONFIG_PATH) -> dict[str, Any]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(payload, dict)
    return payload


def _load_json(relative_path: str) -> dict[str, Any]:
    normalized = Path(relative_path).as_posix()
    assert normalized in PUBLIC_JSON_ALLOWLIST
    assert "data/processed" not in normalized
    payload = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(relative_path: str) -> str:
    normalized = Path(relative_path).as_posix()
    assert not Path(relative_path).is_absolute()
    assert ".." not in Path(relative_path).parts
    assert normalized != "data/processed"
    assert not normalized.startswith("data/processed/")
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _content_sha256(payload: dict[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    serialized = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_mapping_keys(child))
    return keys


def test_protocol_is_unique_target_blind_and_has_exact_stage_identity() -> None:
    config = _load_yaml()

    assert set(config) == EXPECTED_TOP_LEVEL
    assert config["schema_version"] == 1
    assert config["protocol_version"] == "0.4.2"
    assert config["stage"] == "4A"
    assert config["experiment_id"] == "stage4-kde-development-v1"
    assert config["gate_id"] == "S4-KDE-DEV"
    assert config["status"] == "target_blind_protocol_accepted_locally_pending_remote_tag"

    freeze = config["freeze"]
    assert freeze["development_target_reads_before_freeze"] == 0
    assert freeze["locked_test_reads_before_freeze"] == 0
    assert freeze["scores_seen_before_freeze"] is False
    assert freeze["target_counts_seen_before_freeze"] is False
    assert freeze["development_scientific_attempts_allowed"] == 1
    assert freeze["attempt_identity"] == "stage4-kde-development-v1-attempt-1"
    assert {
        freeze["protocol_tag"],
        freeze["expected_code_tag"],
        freeze["expected_result_tag"],
    } == {
        "v0.3.2-kde-anomaly-increment-protocol",
        "v0.3.2-kde-anomaly-increment-code",
        "v0.3.2-kde-anomaly-increment-result",
    }
    assert freeze["protocol_tests_and_documents_allowed_before_protocol_tag"] is True
    assert (
        freeze[
            "protocol_commit_and_tag_required_before_scoring_implementation_and_real_input_adapter"
        ]
        is True
    )
    assert freeze["maximum_score_blind_code_correction_cycles_before_target_read"] == 1
    assert freeze["maximum_same_identity_infrastructure_resumes"] == 1
    assert freeze["post_target_read_code_config_or_identity_repairs_allowed"] == 0
    assert freeze["stage4b_independent_validation_authorized"] is False
    assert freeze["locked_test_authorized"] is False


def test_authority_and_q2_negative_route_pruning_are_exact() -> None:
    config = _load_yaml()
    authority = config["authority"]

    for identity in (
        authority["blueprint"],
        authority["research_protocol"],
        authority["research_protocol_document"],
        authority["science_value_policy"],
    ):
        assert _sha256(identity["path"]) == identity["sha256"]

    q2_identity = config["inputs"]["q2_etas_result"]
    assert _sha256(q2_identity["path"]) == q2_identity["sha256"]
    q2 = _load_json(q2_identity["path"])
    assert q2["qualification_status"] == "not_evaluable"
    assert q2["anomaly_feature_read"] is False
    assert q2["assessment_event_read"] is False
    assert q2["locked_test_read_or_run"] is False

    assert config["science_question"]["etas_status"] == "not_evaluable_not_scored"
    assert config["science_question"]["etas_absence_is_not_zero_score"] is True
    assert config["freeze"]["old_r2_execution_authority_reused"] is False


def test_public_input_identities_and_inherited_contracts_are_byte_exact() -> None:
    config = _load_yaml()
    inputs = config["inputs"]

    public_identities = (
        "background_registry",
        "stage3_registry",
        "feature_dictionary",
        "inherited_contract_allowlist",
        "inherited_fold_contract",
        "inherited_feature_contract",
        "inherited_spatial_strata_contract",
        "environment_lock",
    )
    for name in public_identities:
        identity = inputs[name]
        assert _sha256(identity["path"]) == identity["sha256"]

    for name in (
        "inherited_fold_contract",
        "inherited_feature_contract",
        "inherited_spatial_strata_contract",
    ):
        identity = inputs[name]
        manifest = _load_json(identity["path"])
        assert manifest["content_sha256"] == identity["content_sha256"]
        assert _content_sha256(manifest) == identity["content_sha256"]

    stage3 = _load_json(inputs["stage3_registry"]["path"])
    assert stage3["status"] == "accepted_feature_only_no_target_scoring"
    assert stage3["stage4_allowed"] is True
    assert stage3["aggregate"]["target_or_earthquake_label_read_count"] == 0
    assert stage3["aggregate"]["snapshot_count"] == 205
    assert stage3["aggregate"]["feature_row_count"] == 3_217_885

    inherited = _load_json(inputs["inherited_contract_allowlist"]["path"])
    assert inherited["old_r2_execution_authority_reused"] is False
    assert inherited["old_r2_randomness_reused"] is False
    assert inherited["stage4a_loader_must_reject_any_pointer_outside_allowlist"] is True
    assert set(inherited["source_contracts"]) == {"fold", "feature", "spatial_strata"}
    for source in inherited["source_contracts"].values():
        assert set(source["allowed_json_pointers"])
        assert all(pointer.startswith("/") for pointer in source["allowed_json_pointers"])
    assert "/horizons" not in inherited["source_contracts"]["fold"]["allowed_json_pointers"]
    assert (
        "/preprocessing_contract"
        not in inherited["source_contracts"]["feature"]["allowed_json_pointers"]
    )
    forbidden = set(inherited["forbidden_inherited_semantics"])
    assert {"authorization", "attempt", "execution_seal", "old_random_stream"} <= forbidden


def test_only_exact_low_level_scientific_symbols_may_be_reused() -> None:
    primitives = _load_yaml()["inputs"]["reusable_scientific_primitives"]
    allowlist = primitives["exact_symbol_allowlist"]

    assert set(allowlist) == {"evaluation", "model", "preprocessing", "integration"}
    assert primitives["any_unlisted_module_or_symbol"] == "forbidden"
    assert set(allowlist["evaluation"]["symbols"]) == {
        "information_gain_per_physical_event",
        "strict_recall",
        "same_area_recall_gain_percentage_points",
        "percentile_interval",
    }
    for identity in allowlist.values():
        assert _sha256(identity["path"]) == identity["sha256"]
        assert set(identity["symbols"])

    explicitly_forbidden = set(
        primitives["explicitly_forbidden_old_r2_orchestrators_and_semantics"]
    )
    assert {
        "src/seismoflux/anomaly_increment/scoring_pipeline.py",
        "src/seismoflux/anomaly_increment/formal_run.py",
        "src/seismoflux/anomaly_increment/placebo.py",
        "evaluation.evaluate_g2",
        "evaluation.same_recall_union_area_relative_reduction",
        "evaluation.stratified_five_horizon_bootstrap_indices",
        "preregistration.Stage4SeedContext",
    } <= explicitly_forbidden


def test_background_variants_and_feature_contrasts_are_frozen() -> None:
    config = _load_yaml()
    registry = _load_json(config["inputs"]["background_registry"]["path"])
    selected = registry["science"]["g1_ls_and_selection"]

    assert registry["gate_name"] == "G1-LS"
    assert selected["g1_ls"]["passed"] is True
    assert (
        selected["selection"]["selected_model_variant_id"] == "spatial_poisson/gaussian_kde_bw75km"
    )
    assert config["background"]["variant_id"] == selected["selection"]["selected_model_variant_id"]
    assert config["background"]["bandwidth_km"] == 75.0
    snapshot = config["inputs"]["development_background_snapshot"]
    assert snapshot == {
        "source": "inputs.background_model_payload",
        "snapshot_id": "fold_4",
        "parameter_snapshot_id": (
            "83a0c60d4b62ba6a6e849ac2d5f430001d054b7aec3af40f76193180a18bf4c5"
        ),
        "fit_end_utc": "2019-12-31T16:00:00Z",
        "support_id": "local-support-788851371baf0e3b",
        "compensator_domain_id": (
            "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55"
        ),
        "common_mc": 4.0,
        "supported_area_fraction": 1.0,
        "causal_for_all_stage4a_assessment_dates": True,
    }
    assert config["inputs"]["final_validation_background_snapshot"]["use_in_stage4a"] == (
        "forbidden_future_backfill"
    )
    assert config["background"]["spatial_snapshot_for_every_stage4a_fold"] == (
        "inputs.development_background_snapshot"
    )

    variants = config["variants"]
    assert list(variants) == ["B0", "C0", "B1", "B2", "contribution_contrasts"]
    assert variants["B0"]["logical_feature_set"] == []
    assert variants["C0"]["logical_feature_count"] == 9
    assert variants["B1"]["logical_feature_count"] == 17
    assert variants["B2"]["logical_feature_count"] == 22
    assert variants["contribution_contrasts"] == {
        "full_dynamic_system": "B2_minus_B0",
        "reporting_coverage": "C0_minus_B0",
        "all_anomaly_beyond_coverage": "B2_minus_C0",
        "snapshot_anomaly_beyond_coverage": "B1_minus_C0",
        "trajectory_increment": "B2_minus_B1",
    }

    feature_manifest = _load_json(config["inputs"]["inherited_feature_contract"]["path"])
    assert list(feature_manifest["feature_sets"]) == ["coverage_only", "dynamic", "snapshot"]
    assert len(feature_manifest["feature_sets"]["coverage_only"]["logical_features"]) == 9
    assert len(feature_manifest["feature_sets"]["snapshot"]["logical_features"]) == 17
    assert len(feature_manifest["feature_sets"]["dynamic"]["logical_features"]) == 22
    groups = config["features"]["interpretation_groups"]
    grouped_features = [item for group in groups.values() for item in group]
    assert len(grouped_features) == 22
    assert len(set(grouped_features)) == 22
    assert set(grouped_features) == set(
        feature_manifest["feature_sets"]["dynamic"]["logical_features"]
    )
    interpretation = config["features"]["feature_interpretation_outputs"]
    assert interpretation["per_fold_standardized_coefficient_point_estimates"] == "required"
    assert interpretation["coefficient_across_three_folds_summary"] == (
        "median_min_max_no_confidence_interval_or_p_value"
    )
    assert interpretation["fitted_model_group_zero_out_sensitivity_without_refit"] == {
        "role": "required_diagnostic_not_gate",
        "metrics": [
            "delta_overall_macro_information_gain",
            "delta_fixed_600000km2_strict_recall_percentage_points",
            "target_free_cell_intensity_rank_spearman",
        ],
    }
    assert interpretation["causal_contribution_claim"] == "forbidden"


def test_same_condition_model_calendar_alarm_and_randomness_are_frozen() -> None:
    config = _load_yaml()

    assert config["features"]["confirmatory_radius_km"] == 200
    assert config["features"]["lead_decay_half_life_days"] == 90.0
    assert config["features"]["feature_subset_search"] == "forbidden"
    assert config["model"]["family"] == "shared_ridge_poisson_point_process_increment"
    assert config["model"]["ridge_lambda"] == 1.0
    assert config["model"]["hyperparameter_search"] == "forbidden"
    assert config["model"]["tree_or_neural_models"] == "forbidden"

    assert config["calendar"]["primary_horizons_days"] == [7, 30, 90]
    assert config["calendar"]["rolling_fold_count"] == 3
    assert config["calendar"]["random_split"] == "forbidden"
    fold_manifest = _load_json(config["inputs"]["inherited_fold_contract"]["path"])
    assert len(fold_manifest["joint_macro_rolling_folds"]) == 3
    assert config["calendar"]["descriptive_only_horizons_days"] == [180, 365]

    alarm = config["alarm_region"]
    assert alarm["integration_grid_spacing_km"] == 25
    assert alarm["primary_union_area_km2"] == 600_000
    assert alarm["grid_and_boundaries_target_independent"] is True
    assert alarm["same_area_for_all_variants_and_placebos"] is True
    assert alarm["selection"] == "complete_ranked_prefix_with_cumulative_exact_area_lte_budget"
    assert alarm["partial_cell_selection"] == "forbidden"
    assert alarm["skip_over_budget_cell_and_continue"] == "forbidden"
    assert alarm["descriptive_area_recall_curve"]["pass_eligible"] is False
    molchan = alarm["descriptive_molchan_curve"]
    assert molchan["pass_eligible"] is False
    assert molchan["budgets"] == "same_as_descriptive_area_recall_curve"
    assert molchan["selection"] == "same_complete_prefix_rule"
    assert molchan["unsupported_targets"] == "counted_as_misses"
    assert molchan["aggregation_order"] == (
        "arithmetic_mean_within_fold_across_7_30_90_then_across_three_folds"
    )
    assert math.isclose(molchan["fold_horizon_weight"], 1 / 9)
    assert molchan["endpoints"] == {
        "zero_budget_alarm_fraction": 0.0,
        "zero_budget_miss_rate": 1.0,
        "stop_budget_km2": 960000,
        "beyond_stop": "not_computed",
    }
    assert molchan["interpolation"] == "forbidden"

    randomness = config["randomness"]
    assert randomness["root_seed"] == 147
    assert randomness["canonical_context_fields"][0] == "root_seed_decimal"
    assert randomness["root_seed_is_included_as_canonical_context_field"] is True
    assert randomness["bootstrap"]["replications"] == 2000
    assert randomness["time_permutation"]["replications"] == 1000
    assert randomness["space_permutation"]["replications"] == 1000
    assert randomness["old_r2_randomness_manifest_reused"] is False
    assert randomness["worker_count_invariant"] is True
    assert randomness["failed_permutation_replication_handling"][
        "denominator_remains_configured_replications_plus_one"
    ]
    assert randomness["bootstrap"]["failed_replication_action"] == (
        "evidence_insufficient_no_drop_or_replacement"
    )
    bootstrap = randomness["bootstrap"]
    assert bootstrap["population"] == (
        "all_unique_M5_6_physical_event_ids_in_three_fold_three_horizon_union"
    )
    assert bootstrap["strata"] == ("fold_id_x_three_bit_7_30_90_horizon_membership_signature")
    assert bootstrap["sampling"] == (
        "with_replacement_within_each_stratum_same_stratum_sample_size"
    )
    assert bootstrap["fold_horizon_marginal_event_counts_each_replication"] == ("preserved_exactly")
    assert bootstrap["same_multiplicity_vector_for_all_variants_comparators_and_metrics"]
    assert bootstrap["singleton_stratum_action"] == "deterministic_self_draw"
    assert bootstrap["original_event_to_fold_and_horizon_membership_is_frozen_before_resampling"]
    intervals = bootstrap["intervals"]
    assert intervals["marginal_percentile_95"] == "required_diagnostic_not_gate"
    simultaneous = intervals["confirmatory_familywise_simultaneous"]
    assert simultaneous["method"] == "bonferroni_two_sided_percentile"
    assert simultaneous["comparisons_per_family"] == 4
    assert math.isclose(simultaneous["component_two_sided_confidence_level"], 0.9875)
    assert math.isclose(simultaneous["lower_percentile"], 0.00625)
    assert math.isclose(simultaneous["upper_percentile"], 0.99375)
    assert simultaneous["gate_uses_simultaneous_lower_bounds_not_marginal_bounds"]


def test_scientific_gate_requires_effect_placebos_practical_gain_and_stability() -> None:
    config = _load_yaml()
    evaluation = config["evaluation"]

    assert evaluation["candidate_variants"] == ["B2", "B1"]
    assert evaluation["background_comparators"] == ["B0", "C0"]
    assert evaluation["primary_magnitude_bin"] == "M5_6"
    assert evaluation["primary_magnitude_range"] == "[5.0,6.0)"
    assert evaluation["unique_physical_event_union_minimum"] == 20
    assert evaluation["horizon_point_estimate_must_be_positive"] == [7, 30, 90]
    assert evaluation["fold_positive_count_minimum"] == 2
    assert evaluation["fold_macro_median_must_be_positive"] is True
    assert (
        evaluation["macro_information_gain_familywise_simultaneous_95pct_lower_must_be_positive"]
        is True
    )
    assert evaluation["time_permutation_p_max"] == 0.05
    assert evaluation["space_permutation_p_max"] == 0.05
    max_t = evaluation["multiple_candidate_control"]
    assert max_t["method"] == "paired_single_step_maxT"
    assert max_t["independent_null_families"] == [
        "time_permutation",
        "space_permutation",
    ]
    assert max_t["candidate_statistics"] == {
        "B1": "overall_macro_ig_B1_minus_C0",
        "B2": "overall_macro_ig_B2_minus_C0",
    }
    assert max_t["each_replication_family_statistic"] == ("max_of_B1_and_B2_candidate_statistics")
    assert max_t["required_for_adopted_candidate"] == [
        "adjusted_time_permutation_p_lte_0_05",
        "adjusted_space_permutation_p_lte_0_05",
    ]
    assert max_t["B2_minus_B1_role"] == ("component_adoption_gate_only_not_new_stage4a_positive")
    assert max_t["unadjusted_candidate_p_values"] == "diagnostic_only_not_gate"
    b2_invalid = max_t["B2_observed_invalid_handling"]
    assert b2_invalid == {
        "shrink_family_to_B1_only": "forbidden",
        "B1_adoption_after_B2_invalid": "forbidden",
        "overall_status": "evidence_insufficient",
        "replacement_or_recompute": "forbidden",
    }
    common_gate = evaluation["candidate_common_gate"]
    assert set(common_gate["required_against_each_of_B0_and_C0"]) == {
        "all_three_horizon_ig_point_estimates_positive",
        "at_least_two_of_three_fold_macros_positive",
        "three_fold_macro_median_positive",
        "overall_macro_ig_familywise_simultaneous_95pct_lower_positive",
        ("fixed_600000km2_recall_gain_gte_5pp_and_familywise_simultaneous_95pct_lower_positive"),
        "regional_robustness",
    }
    assert common_gate["required_once_per_candidate_for_anomaly_beyond_C0_maxT_family"] == [
        "maxT_adjusted_time_permutation_p_lte_0_05",
        "maxT_adjusted_space_permutation_p_lte_0_05",
    ]

    practical = evaluation["practical_improvement"]
    assert practical["pass_rule"] == "fixed_600000_km2_recall_only"
    assert practical["same_area_strict_recall_gain_percentage_points_minimum"] == 5.0
    assert practical["familywise_simultaneous_95pct_lower_must_be_positive"] is True
    assert practical["required_against"] == ["B0", "C0"]
    assert practical["same_recall_union_area_relative_reduction"]["role"] == (
        "descriptive_only_not_pass_eligible"
    )
    assert evaluation["dynamic_additional_gate"]["contrast"] == "B2_minus_B1"
    assert evaluation["dynamic_additional_gate"]["failure_action"] == (
        "reject_dynamic_only_then_evaluate_snapshot"
    )

    regional = evaluation["regional_robustness"]
    assert regional["minimum_positive_event_bearing_regions"] == 2
    assert regional["leave_strongest_contribution_region_out_macro_ig_must_be_positive"]
    assert regional["leave_strongest_contribution_region_out_95pct_interval"] == (
        "required_diagnostic_not_gate"
    )
    assert evaluation["full_study_area_target_denominator"] is True
    assert config["decision_gate"]["candidate_gate_applies_independently_to_B2_and_B1"]


def test_macro_and_alarm_reference_examples_have_unique_executable_semantics() -> None:
    config = _load_yaml()
    aggregation = config["evaluation"]["aggregation"]
    reference = config["evaluation"]["aggregation_reference_example"]
    rows = reference["fold_rows_horizons_7_30_90"]
    fold_macros = [statistics.fmean(row) for row in rows]
    horizon_macros = [statistics.fmean(row[index] for row in rows) for index in range(3)]

    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(fold_macros, reference["expected_fold_macros"], strict=True)
    )
    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(
            horizon_macros,
            reference["expected_horizon_macros"],
            strict=True,
        )
    )
    assert math.isclose(
        statistics.fmean(fold_macros),
        reference["expected_overall_macro"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert aggregation["fold_macro"] == "arithmetic_mean_of_7_30_90_cell_information_gains"
    assert aggregation["overall_macro"] == "arithmetic_mean_of_three_fold_macros"
    assert aggregation["null_replication_statistic"] == "exact_same_overall_macro_after_refit"

    alarm = config["alarm_region"]
    example = alarm["reference_example"]
    ranked = sorted(
        example["cells"],
        key=lambda item: (-item["score"], item["row"], item["column"], item["cell_id"]),
    )
    assert [item["cell_id"] for item in ranked] == example["expected_ranked_ids"]
    selected: list[dict[str, Any]] = []
    area = 0.0
    for cell in ranked:
        next_area = area + cell["exact_area_km2"]
        if next_area > example["budget_km2"]:
            break
        selected.append(cell)
        area = next_area
    assert [item["cell_id"] for item in selected] == example["expected_complete_prefix_ids"]
    assert area == example["expected_exact_selected_area_km2"]


def test_target_access_resource_and_visualization_boundaries_fail_closed() -> None:
    config = _load_yaml()
    access = config["access_boundary"]

    assert access["before_code_tag"]["real_processed_input_bytes"] == "forbidden"
    assert access["before_code_tag"]["development_target_bytes"] == "forbidden"
    assert access["after_code_tag"]["development_target_read"] == (
        "only_inside_single_registered_attempt"
    )
    assert access["after_code_tag"]["independent_validation_targets"] == "forbidden"
    assert access["target_use"]["grid_construction"] == "forbidden"
    assert access["target_use"]["spatial_refinement"] == "forbidden"
    assert access["target_use"]["alarm_threshold_selection"] == "forbidden"
    roles = access["earthquake_catalog_roles"]
    assert roles["frozen_spatial_kde"]["catalog_rebuild_in_stage4a"] == "forbidden"
    assert roles["fold_rate_head_training"]["assessment_rows_allowed"] is False
    assert roles["assessment_targets"]["use_outside_scoring_and_hit_miss"] == "forbidden"

    resources = config["resources"]
    assert resources["minimum_reserved_physical_cores"] >= 2
    assert resources["maximum_logical_workers"] == 6
    assert resources["effective_worker_formula"] == "max(1,min(6,physical_cores-2))"
    assert resources["minimum_detected_physical_cores_to_run"] == 3
    assert resources["blas_threads_per_process"] == 1
    assert resources["process_priority"] == "below_normal"
    assert resources["gpu"]["new_backend_engineering_for_this_gate"] == "forbidden"

    outputs = config["outputs"]
    assert outputs["public_interactive_forecast"].endswith("/index.html")
    assert outputs["public_interactive_aggregate_results"].endswith("/index.html")
    assert outputs["local_retrospective"].startswith("data/interim/")
    assert outputs["retrospective_must_be_labeled_historical_not_prospective"] is True
    assert "raw_coordinates" in outputs["public_outputs_forbid"]
    schemas = outputs["schemas"]
    assert "target_payload" in schemas["public_target_free_forecast"]["forbidden"]
    assert "hit_or_miss" in schemas["public_target_free_forecast"]["forbidden"]
    assert "target_overlay" in schemas["local_historical_retrospective"]["required"]
    assert schemas["local_historical_retrospective"]["publication"] == "forbidden_gitignored"
    assert (
        "per_fold_horizon_event_and_exposure_counts"
        in schemas["public_aggregate_results"]["required"]
    )
    aggregate_required = set(schemas["public_aggregate_results"]["required"])
    assert {
        "final_status_selected_candidate_failed_gates_and_science_value_decision",
        "paired_maxT_adjusted_time_and_space_p_values_with_unadjusted_diagnostic_labels",
        "complete_area_recall_curve",
        "molchan_miss_rate_alarm_area_curve",
    } <= aggregate_required
    assert (
        "paired_maxT_adjusted_time_and_space_p_values_with_unadjusted_diagnostic_labels"
        in outputs["result_manifest_required_fields"]
    )
    assert (
        "source_library_total_vs_attempt_used_snapshot_and_feature_row_counts_by_fold_"
        "with_unused_reason" in outputs["result_manifest_required_fields"]
    )
    assert (
        "familywise_simultaneous_95pct_IG_and_fixed_area_recall_bounds_with_"
        "marginal_diagnostic_labels" in outputs["result_manifest_required_fields"]
    )
    assert {
        "plain_language_data_and_model_flow",
        (
            "source_library_total_vs_attempt_used_snapshot_and_feature_row_counts_by_"
            "fold_with_unused_reason"
        ),
        (
            "familywise_simultaneous_95pct_IG_and_fixed_area_recall_bounds_with_"
            "marginal_diagnostic_labels"
        ),
    } <= aggregate_required
    status_policy = outputs["target_free_forecast_status_policy"]
    assert status_policy["applies_to"] == [
        "public_target_free_forecast",
        "local_full_fidelity_target_free_forecast",
    ]
    assert status_policy["passed"] == ("selected_B1_or_B2_primary_with_B0_optional_comparator")
    assert status_policy["failed_or_evidence_insufficient"] == (
        "B0_primary_B1_B2_default_off_and_labeled_not_adopted_diagnostic"
    )
    assert status_policy["invalid"] == "do_not_publish_forecast_page"
    assert outputs["target_free_stage4a_pages_evidence_role"] == (
        "historical_or_development_replay_not_prospective_evidence"
    )
    assert schemas["public_target_free_forecast"]["status_policy"] == (
        "outputs.target_free_forecast_status_policy"
    )
    assert schemas["local_full_fidelity_target_free_forecast"]["status_policy"] == (
        "outputs.target_free_forecast_status_policy"
    )
    for replay_schema in (
        schemas["public_target_free_forecast"],
        schemas["local_full_fidelity_target_free_forecast"],
    ):
        assert (
            "historical_or_development_replay_not_prospective_evidence_warning"
            in replay_schema["required"]
        )
    prospective = outputs["prospective_issue_contract"]
    assert prospective["activation"] == (
        "only_after_stage4a_result_tag_and_frozen_selected_or_B0_fallback_model"
    )
    assert prospective["issue_time_must_precede_target_window"] is True
    assert prospective["backfill"] == "forbidden"
    assert prospective["overwrite_or_delete_after_issue"] == "forbidden"
    assert {
        "data_cutoff_and_7_30_90_day_windows",
        "input_manifest_sha256",
        "model_identity_and_payload_sha256",
        "protocol_config_sha256",
        "code_commit_tag_and_seal_sha256",
        "immutable_archive_path",
    } <= set(prospective["required_identity_fields"])
    assert prospective["target_payload_hit_miss_or_future_event_count"] == "forbidden"
    flow = set(outputs["plain_language_data_and_model_flow_required_content"])
    assert {
        "frozen_75km_spatial_KDE_uses_history_only_through_2019_12_31",
        "each_fold_rate_head_uses_only_earthquakes_in_its_prior_training_exposures",
        "assessment_M5_6_earthquakes_are_used_only_for_scoring_and_hit_miss",
        "205_causal_anomaly_snapshots_and_3217885_feature_rows",
        "source_library_totals_are_not_the_attempt_sample_size",
        (
            "per_fold_actual_used_snapshot_and_feature_row_counts_with_future_or_"
            "validation_holdout_reasons"
        ),
        "B0_to_C0_to_B1_to_B2_component_flow",
        "human_prediction_location_magnitude_time_and_free_text_forbidden",
    } <= flow
    assert outputs["static_science_summary_schema"] == (
        "same_required_scientific_fields_as_public_aggregate_results"
    )


def test_minimal_execution_seal_ledgers_and_resume_identity_are_mandatory() -> None:
    config = _load_yaml()
    control = config["execution_control"]

    assert control["code_seal"]["created_and_committed_at_code_tag"] is True
    assert {
        "protocol_config_sha256",
        "expected_input_manifest_sha256",
        "frozen_input_seal_sha256",
        "source_commit_S",
        "expected_code_tag_name",
        "exact_allowed_seal_commit_changed_paths",
        "runner_and_scientific_module_sha256",
        "environment_lock_sha256",
        "randomness_manifest_sha256",
        "initial_zero_attempt_ledger_sha256",
    } <= set(control["code_seal"]["binds"])
    assert set(control["code_seal"]["must_not_bind"]) == {
        "seal_commit_C_hash",
        "code_seal_own_sha256",
    }
    graph = control["code_freeze_commit_graph"]
    assert graph["source_commit"]["identity_symbol"] == "S"
    assert graph["seal_commit"]["identity_symbol"] == "C"
    assert graph["seal_commit"]["exact_parent"] == "S"
    assert graph["seal_commit"]["allowed_changed_paths_from_S"] == [
        "data/manifests/anomaly_increment_kde_dev_input.json",
        "data/manifests/anomaly_increment_kde_dev_randomness.json",
        "data/manifests/anomaly_increment_kde_dev_code_seal.json",
        "data/manifests/anomaly_increment_kde_dev_attempt_ledger.json",
    ]
    assert graph["seal_commit"]["any_other_changed_path"] == "forbidden"
    assert graph["seal_commit"]["self_commit_hash_embedded_in_tracked_tree"] == "forbidden"
    assert graph["code_tag"] == {
        "name": "v0.3.2-kde-anomaly-increment-code",
        "points_to": "C",
    }
    assert graph["preflight_remote_verification"] == [
        "tag_peels_to_C",
        "C_has_exactly_one_parent_S",
        "C_minus_S_changed_paths_equal_allowed_changed_paths_from_S",
        "source_tree_module_and_schema_hashes_match_code_seal",
        "seal_artifact_cross_hashes_are_acyclic_and_match",
    ]
    assert "source_commit" in control["input_identity_seal"]["derivation"]
    assert "code_commit" not in control["input_identity_seal"]["derivation"]
    assert control["expected_input_manifest"]["created_and_committed_at_code_tag"] is True
    assert (
        control["expected_input_manifest"][
            "generated_only_from_protocol_declared_identities_without_opening_real_inputs"
        ]
        is True
    )
    assert control["input_identity_seal"][
        "written_to_randomness_manifest_as_frozen_input_seal_sha256"
    ]
    assert control["initial_attempt_ledger"][
        "created_and_committed_at_code_tag_with_operation_count_zero"
    ]
    assert control["initial_attempt_ledger"]["maximum_registered_scientific_attempts"] == 1
    assert control["initial_attempt_ledger"]["initial_zero_payload_binds"] == [
        "protocol_config_sha256",
        "expected_input_manifest_sha256",
        "frozen_input_seal_sha256",
        "source_commit_S",
        "expected_code_tag_name",
    ]
    assert control["initial_attempt_ledger"]["initial_zero_payload_must_not_bind"] == [
        "code_seal_sha256",
        "seal_commit_C_hash",
    ]
    assert control["initial_attempt_ledger"][
        "runtime_registration_appends_verified_code_seal_sha256"
    ]
    assert control["initial_attempt_ledger"]["registration"] == (
        "atomic_compare_and_swap_existing_zero_ledger_to_registered"
    )
    assert control["runtime_target_read_ledger"][
        "atomically_created_at_preflight_start_with_operation_count_zero"
    ]
    assert control["runtime_target_read_ledger"]["maximum_logical_target_open_sessions"] == 1
    assert control["preflight_receipt"]["target_bytes_read_during_preflight"] is False
    assert control["preflight_receipt"]["binds_zero_target_read_ledger_sha256_and_operation_count"]
    assert control["attempt_start_sequence"] == [
        "verify_remote_code_tag_and_code_seal",
        "atomically_create_or_verify_same_identity_target_read_ledger_at_zero",
        "write_preflight_receipt_bound_to_zero_target_read_ledger",
        "verify_preflight_receipt_and_zero_target_read_count",
        "atomically_register_single_attempt",
        "verify_runtime_target_read_ledger_remains_zero_and_bound_to_preflight_receipt",
        "open_target_once_through_single_adapter",
    ]
    assert control["attempt_start_sequence"][-1] == ("open_target_once_through_single_adapter")
    assert control["resume"]["same_identity_only"] is True
    assert control["resume"]["completed_checkpoint_deletion_replacement_or_recompute"] == (
        "forbidden"
    )
    assert control["direct_target_reader_outside_single_adapter"] == "forbidden"


def test_science_value_gate_is_mandatory_and_no_scores_are_embedded() -> None:
    config = _load_yaml()
    review = config["science_value_review"]
    decision = config["decision_gate"]

    assert review["required_immediately_after_result"] is True
    assert review["required_fields"] == [
        "science_value_category",
        "evidence",
        "decision",
        "next_scientific_test",
        "stop_condition",
    ]
    assert review["missing_review_blocks_stage_completion"] is True
    assert decision["passed"]["science_value_category"] == "direct_improvement"
    assert decision["passed"]["G2_or_G3_claim"] == "forbidden"
    assert decision["failed"]["science_value_category"] == "no_material_progress"
    assert decision["evidence_insufficient"]["science_value_category"] == ("no_material_progress")
    assert decision["invalid"]["science_value_category"] == ("not_assigned_until_root_cause_review")
    assert decision["candidate_status_priority"] == [
        "invalid",
        "evidence_insufficient",
        "failed",
        "passed",
    ]
    resolution = decision["overall_status_resolution"]
    assert resolution["nonselected_candidate_failed_status_cannot_override_adopted_pass"]
    assert resolution["B2_only_invalid_action"] == (
        "overall_evidence_insufficient_no_candidate_adoption"
    )
    assert resolution["shared_invalid_precedes_candidate_adoption"] is True
    assert resolution["explicit_B2_only_invalid_cases"] == {
        "any_B1_status": "overall_evidence_insufficient",
    }
    assert resolution["resolution_order"] == [
        "any_shared_invalid_condition_means_overall_invalid",
        "any_adopted_candidate_passes_means_overall_passed",
        (
            "no_adopted_candidate_and_any_evaluable_candidate_evidence_insufficient_or_"
            "B2_only_invalid_means_overall_evidence_insufficient"
        ),
        "both_candidates_valid_evidence_sufficient_and_no_adopted_candidate_means_overall_failed",
    ]
    assert set(decision["candidate_truth_table"]["invalid"]["reasons"]) == {
        "candidate_fit_or_score_invalid",
        "candidate_nonfinite_primary_output",
    }
    assert decision["candidate_validity"]["B2_invalid_action"] == (
        "overall_evidence_insufficient_no_candidate_adoption"
    )
    assert decision["candidate_validity"]["B1_invalid_action"].startswith("overall_invalid")

    forbidden_result_keys = {
        "event_count",
        "information_gain_nats_per_event",
        "observed_recall",
        "observed_score",
        "result_status",
        "score_id",
        "target_event_count",
    }
    assert _all_mapping_keys(config).isdisjoint(forbidden_result_keys)

    protocol_text = PROTOCOL_DOCUMENT_PATH.read_text(encoding="utf-8")
    blueprint_text = (ROOT / "SEISMOFLUX_IMPLEMENTATION_HANDOFF.md").read_text(encoding="utf-8")
    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for token in (
        "direct_improvement",
        "necessary_enabler",
        "no_material_progress",
        "science_value_category",
        "next_scientific_test",
        "stop_condition",
    ):
        assert token in protocol_text
        assert token in blueprint_text
    assert "科学价值复审" in agents_text
    assert "没有实质推动时停止继续堆叠工程" in agents_text
