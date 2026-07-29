from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "background_etas_numerical_qualification.yaml"
START_MANIFEST_PATH = ROOT / "data" / "manifests" / "etas_numerical_repair_start_manifest.json"

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "protocol_version",
    "protocol_revision",
    "stage",
    "status",
    "frozen_on",
    "blueprint",
    "scientific_goal",
    "publication",
    "frozen_identity",
    "target_blindness",
    "snapshots",
    "model",
    "repair",
    "optimizer",
    "qualification",
    "attempt",
    "resources",
    "outputs",
    "decision",
}
SNAPSHOT_ORDER = ["fold_1", "fold_2", "fold_3", "fold_4", "final_validation"]


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


def _load_config() -> dict[str, Any]:
    payload = yaml.load(CONFIG_PATH.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(payload, dict)
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_yaml_is_unique_complete_and_has_no_unknown_top_level_blocks() -> None:
    config = _load_config()

    assert set(config) == EXPECTED_TOP_LEVEL
    assert config["schema_version"] == 1
    assert config["protocol_version"] == "0.2.2"
    assert config["stage"] == "2-ETAS-Q"
    assert config["status"] == "preregistered_target_blind"


def test_five_snapshots_and_frozen_starts_form_exactly_25_unique_combinations() -> None:
    config = _load_config()
    manifest = json.loads(START_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert config["snapshots"]["order"] == SNAPSHOT_ORDER
    assert [row["snapshot_id"] for row in config["snapshots"]["entries"]] == SNAPSHOT_ORDER
    assert [row["snapshot_id"] for row in manifest["snapshots"]] == SNAPSHOT_ORDER
    combinations = [
        (snapshot["snapshot_id"], start["start_index"])
        for snapshot in manifest["snapshots"]
        for start in snapshot["starts"]
    ]
    assert combinations == [
        (snapshot_id, start_index) for snapshot_id in SNAPSHOT_ORDER for start_index in range(5)
    ]
    assert len(combinations) == len(set(combinations)) == 25
    assert (
        manifest["vector_payload_sha256"]
        == config["frozen_identity"]["start_vector_payload_sha256"]
    )
    assert _sha256(START_MANIFEST_PATH) == config["frozen_identity"]["start_manifest_sha256"]


def test_scientific_model_optimizer_and_thresholds_match_frozen_parent() -> None:
    config = _load_config()
    parent = _load_yaml(ROOT / config["frozen_identity"]["parent_protocol_path"])

    expected_model = {
        "variant_id": "etas/d25_q1.5_gamma1_cut300_kde_bw75",
        "background_kde_bandwidth_km": 75.0,
        "maximum_magnitude": 9.5,
        "spatial_kernel": {"d_km2": 25.0, "q": 1.5, "gamma": 1.0, "cutoff_km": 300.0},
        "temporal_kernel": {
            "history_parent_cutoff_days": 3650.0,
            "form": "normalized_infinite_support_omori_utsu",
        },
        "branching_ratio_maximum": 0.95,
        "parameter_order": [
            "background_rate_per_day",
            "productivity_k",
            "alpha",
            "c_days",
            "p",
        ],
        "transformed_parameter_order": [
            "log_background_rate_per_day",
            "log_productivity_k",
            "log_alpha",
            "log_c_days",
            "log_p_minus_one",
        ],
        "parameter_bounds": {
            "background_rate_per_day": [0.01, 10.0],
            "productivity_k": [0.0001, 0.5],
            "alpha": [0.05, 2.0],
            "c_days": [0.001, 30.0],
            "p": [1.01, 2.5],
        },
        "quadrature_grid_km": [50.0, 25.0, 12.5],
    }
    assert config["model"] == expected_model
    assert config["model"]["variant_id"] == parent["etas_model"]["variant_id"]
    assert (
        config["model"]["background_kde_bandwidth_km"]
        == parent["etas_model"]["background_kde_bandwidth_km"]
    )
    assert config["model"]["maximum_magnitude"] == parent["etas_model"]["maximum_magnitude"]
    assert config["model"]["spatial_kernel"] == {
        **parent["etas_model"]["spatial_kernel"],
        "cutoff_km": parent["etas_model"]["spatial_kernel"]["cutoff_km"],
    }
    assert config["model"]["temporal_kernel"] == parent["etas_model"]["temporal_kernel"]
    assert (
        config["model"]["branching_ratio_maximum"]
        == parent["etas_model"]["branching_ratio_maximum"]
    )
    assert config["model"]["parameter_order"] == parent["etas_model"]["parameter_order"]
    assert (
        config["model"]["transformed_parameter_order"]
        == parent["etas_model"]["transformed_parameter_order"]
    )
    assert config["model"]["parameter_bounds"] == parent["etas_model"]["parameter_bounds"]
    assert config["optimizer"]["options"] == {
        "ftol": 1.0e-12,
        "gtol": 1.0e-6,
        "maxiter": 500,
        "maxfun": 100000,
        "maxls": 20,
        "gradient_relative_step": 1.0e-6,
    }
    assert config["optimizer"]["options"] == {
        "ftol": parent["optimizer"]["ftol"],
        "gtol": parent["optimizer"]["gtol"],
        "maxiter": parent["optimizer"]["maximum_iterations"],
        "maxfun": parent["optimizer"]["maxfun"],
        "maxls": parent["optimizer"]["maxls"],
        "gradient_relative_step": 1.0e-6,
    }
    assert config["qualification"] == {
        "exact_snapshot_count": 5,
        "exact_completed_start_row_count": 25,
        "all_five_snapshots_required": True,
        "minimum_converged_starts": 4,
        "gradient_infinity_norm_maximum": 1.0e-4,
        "best_three_relative_objective_range_maximum": 1.0e-4,
        "best_three_transformed_parameter_range_maximum": 0.1,
        "hessian_minimum_eigenvalue": 1.0e-8,
        "hessian_condition_number_maximum": 1.0e10,
        "hessian_method": "central_second_difference_of_objective_on_transformed_scale",
        "hessian_relative_step": 1.0e-4,
        "hessian_step_formula": "relative_step_times_max_1_abs_parameter",
        "hessian_symmetrize": True,
        "hessian_invalid_stencil_action": ("fail_etas_stability_without_one_sided_substitution"),
        "hessian_condition_number_definition": (
            "largest_eigenvalue_divided_by_smallest_eigenvalue_of_"
            "symmetric_positive_definite_hessian"
        ),
        "branching_ratio_maximum_exclusive": 0.95,
        "grid_25_to_12_5_expected_count_relative_difference_maximum": 0.02,
        "grid_25_to_12_5_density_l1_maximum": 0.05,
        "independent_recalculation_required": True,
        "result_is_evaluable_iff_every_snapshot_passes_every_gate": True,
    }
    parent_gate = parent["qualification"]["per_snapshot_conjunctive_requirements"]
    assert (
        config["qualification"]["minimum_converged_starts"]
        == parent_gate["minimum_converged_starts"]
    )
    assert (
        config["qualification"]["gradient_infinity_norm_maximum"]
        == parent_gate["every_counted_converged_gradient_infinity_norm_lte"]
    )
    assert (
        config["qualification"]["best_three_relative_objective_range_maximum"]
        == parent_gate["best_three_relative_objective_range_lte"]
    )
    assert (
        config["qualification"]["best_three_transformed_parameter_range_maximum"]
        == parent_gate["best_three_transformed_parameter_maximum_range_lte"]
    )
    assert (
        config["qualification"]["hessian_minimum_eigenvalue"]
        == parent_gate["hessian_minimum_eigenvalue_gte"]
    )
    assert (
        config["qualification"]["hessian_condition_number_maximum"]
        == parent_gate["hessian_condition_number_lte"]
    )
    assert config["qualification"]["hessian_relative_step"] == 1.0e-4
    parent_stability = _load_yaml(ROOT / config["frozen_identity"]["parent_background_path"])[
        "etas"
    ]["numerical_stability"]
    for key in (
        "hessian_method",
        "hessian_relative_step",
        "hessian_step_formula",
        "hessian_symmetrize",
        "hessian_invalid_stencil_action",
        "hessian_condition_number_definition",
    ):
        assert config["qualification"][key] == parent_stability[key]


def test_snapshot_cutoffs_support_domains_counts_and_parent_roles_match_parent() -> None:
    config = _load_config()
    parent = _load_yaml(ROOT / config["frozen_identity"]["parent_protocol_path"])
    parent_entries = parent["snapshots"]["entries"]
    entries = config["snapshots"]["entries"]

    assert len(entries) == len(parent_entries) == 5
    for observed, expected in zip(entries, parent_entries, strict=True):
        for key in (
            "snapshot_id",
            "fit_end_utc",
            "support_id",
            "compensator_domain_id",
            "retained_area_fraction",
        ):
            assert observed[key] == expected[key]
        assert observed["include_eligible_unsupported_parent_history"] is (
            expected["parent_role"] == "include_prevalidated_eligible_unsupported_history"
        )
    assert [row["fit_event_count"] for row in entries] == parent["fit_input_bundle"][
        "expected_parent_result_counts"
    ]["fit_event_count"]
    assert [row["parent_event_count"] for row in entries] == parent["fit_input_bundle"][
        "expected_parent_result_counts"
    ]["parent_event_count"]
    assert [row["immigrant_kde_training_event_count"] for row in entries] == parent[
        "fit_input_bundle"
    ]["expected_parent_result_counts"]["immigrant_kde_training_event_count"]
    assert config["snapshots"]["unsupported_parent_sensitivity_required_for"] == [
        "fold_1",
        "fold_3",
    ]
    assert (
        config["snapshots"]["unsupported_parent_sensitivity_method"]
        == "reevaluate_primary_selected_parameters_without_refit"
    )
    assert config["snapshots"]["unsupported_parent_sensitivity_optimizer_call_count"] == 0
    assert config["snapshots"][
        "unsupported_parent_sensitivity_is_diagnostic_not_qualification_gate"
    ]
    assert (
        config["snapshots"]["local_mc_spatial_isolation"]
        == "raw_mc_above_common_mc_marks_only_that_fixed_cell_unsupported"
    )
    assert config["snapshots"]["local_mc_spatial_isolation_must_not_propagate_to_other_cells"]
    assert config["snapshots"]["no_eligible_temporal_completeness_layer_is_global_hard_failure"]
    assert config["snapshots"]["support_reconstruction_must_equal_frozen_manifest"]
    assert config["snapshots"]["support_thresholds_may_not_be_recomputed_or_relaxed"]


def test_target_blindness_and_frozen_tracked_file_hashes_are_enforced() -> None:
    config = _load_config()
    blindness = config["target_blindness"]

    assert blindness["anomaly_feature_read"] is False
    assert blindness["stage4_target_read"] is False
    assert blindness["assessment_event_read"] is False
    assert blindness["prior_score_read"] is False
    assert blindness["information_gain_computation"] is False
    assert blindness["forecast_hit_computation"] is False
    assert blindness["locked_test_read_or_run"] is False
    assert blindness["formal_target_consumer_count"] == 0

    identity = config["frozen_identity"]
    for path_key, hash_key in (
        ("blueprint", "blueprint_sha256"),
        ("parent_protocol_path", "parent_protocol_sha256"),
        ("parent_background_path", "parent_background_sha256"),
        ("project_config_path", "project_config_sha256"),
        ("uv_lock_path", "uv_lock_sha256"),
        ("support_manifest_path", "support_manifest_sha256"),
        ("issue_manifest_path", "issue_manifest_sha256"),
        ("start_manifest_path", "start_manifest_sha256"),
        ("production_fixture_path", "production_fixture_sha256"),
        ("independent_fixture_path", "independent_fixture_sha256"),
    ):
        source = ROOT / (config[path_key] if path_key == "blueprint" else identity[path_key])
        assert _sha256(source) == identity[hash_key]


def test_publication_attempt_and_output_rules_freeze_one_non_overwriting_result() -> None:
    config = _load_config()
    interrupted_key = (
        "interrupted_in_memory_fit_without_snapshot_result_is_not_a_completed_scientific_result"
    )

    assert config["publication"]["exact_order"] == ["protocol", "code", "qualification_result"]
    assert config["publication"]["positive_and_negative_results_use_same_result_tag"] is True
    expected_attempt = {
        "attempt_id": "etas_qualification_q1",
        "root": "data/interim/stage2/etas_numerical_qualification/attempts/etas_qualification_q1",
        "snapshot_result_path_template": "snapshots/{snapshot_id}.json",
        "exactly_one_scientific_attempt": True,
        "automatic_retry": False,
        "fit_etas_call_count_per_missing_snapshot": 1,
        "snapshot_result_contains_all_five_start_rows": True,
        "snapshot_result_write": "same_directory_temporary_file_then_atomic_create_if_absent",
        "completed_snapshot_result_is_immutable": True,
        "interrupted_attempt_may_fill_missing_snapshot_only": True,
        "resumed_missing_snapshot_reuses_identical_input_hash_and_five_start_vectors": True,
        "completed_start_row_replacement": False,
        "final_result_create_once": True,
        "positive_or_negative_result_must_be_published": True,
    }
    expected_attempt[interrupted_key] = True
    assert config["attempt"] == expected_attempt
    assert config["resources"]["max_workers"] == 1
    assert config["resources"]["blas_threads"] == 1
    assert config["resources"]["reserve_physical_cores_minimum"] >= 2
    assert config["scientific_goal"]["candidate_after_this_attempt_forbidden"] is True
    assert config["outputs"]["static_figure"].endswith(".svg")
    assert config["outputs"]["interactive_report"].endswith("/index.html")
    assert "terminal_transformed_hex" in config["outputs"]["required_public_fields"]
    assert "input_manifest_sha256" in config["outputs"]["result_manifest_binds"]
    assert config["outputs"][
        "verification_must_rebuild_inputs_and_not_trust_derived_result_metrics"
    ]
