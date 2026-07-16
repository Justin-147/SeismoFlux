from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from seismoflux.anomaly_increment.preregistration import (
    STAGE4_RANDOM_EVALUATION_IDS,
    Stage4SeedContext,
    build_exposure_preregistration,
    build_random_input_seal,
    build_randomness_manifest,
    protocol_design_sha256,
    select_feature_storage_columns,
    validate_stage4_protocol_bundle,
    verify_content_sha256,
)

PROTOCOL_PATH = Path("configs/anomaly_increment_r2.yaml")
FOLD_MANIFEST_PATH = Path("data/manifests/anomaly_increment_r2_fold_manifest.json")
FEATURE_SET_PATH = Path("data/manifests/anomaly_increment_r2_feature_set.json")
RANDOMNESS_PATH = Path("data/manifests/anomaly_increment_r2_randomness.json")
SPATIAL_STRATA_PATH = Path("data/manifests/anomaly_increment_r2_spatial_strata.json")
CLI_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_stage4_preregistration.py"


def _load_preregistration_cli() -> Any:
    module_name = "seismoflux_stage4_preregistration_cli_test"
    spec = importlib.util.spec_from_file_location(module_name, CLI_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load stage-4 preregistration CLI from {CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


preregistration_cli = _load_preregistration_cli()

GENERATED_MANIFEST_PATHS = {
    "fold": FOLD_MANIFEST_PATH,
    "feature_set": FEATURE_SET_PATH,
    "randomness": RANDOMNESS_PATH,
    "spatial_strata": SPATIAL_STRATA_PATH,
}

# These are metadata identities frozen by accepted stages 1--3.  This test never
# opens the earthquake target; it only checks that the score-free protocol points to
# the preregistered identity.
EXPECTED_INPUTS: dict[str, tuple[str, str]] = {
    "research_protocol": (
        "configs/research_protocol.yaml",
        "e488575ef06329fe19ec282224d813f560c575552537abb2b4e227479f443dd5",
    ),
    "research_protocol_document": (
        "docs/research_protocol.md",
        "aa7fd98862030da3dc836b67852fc24fc9770015efcd75cfb0b1075088492dcc",
    ),
    "background_fold_manifest": (
        "data/manifests/background_fold_manifest.json",
        "3a06f9fc1527a5b4e943de874fa81bda19d0afb0aeec832fb2e0af95f20f2e92",
    ),
    "local_support_fold_manifest": (
        "data/manifests/background_local_support_fold_manifest.json",
        "d7ae5266c9143ed0a67a9954da52039b2753f108698fb05477466a6d5b934e38",
    ),
    "local_support_manifest": (
        "data/manifests/background_local_support_manifest.json",
        "632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d",
    ),
    "background_registry": (
        "data/manifests/background_local_support_model_registry.json",
        "774531a806a9591398f1be3474a00fdf89626b515b2cebb41c597f7dc6c1c67b",
    ),
    "background_model_manifest": (
        "models/registry/background_local_support/5c8fa982bb94b9ad/manifest.json",
        "1b8c67ac82a83d6885dee06ea32b8907b333c6118c5ec7968f7b779589447acf",
    ),
    "background_model_payload": (
        "models/registry/background_local_support/5c8fa982bb94b9ad/poisson_kde.json",
        "63db433a0ea490de2674cb84c485f8efd8215374d70f31f22810da29f0d7b142",
    ),
    "stage3_registry": (
        "data/manifests/anomaly_feature_registry.json",
        "65635e18ab94e5e114948a6a7a5d1533a665fee1f6f862abcc5498394f85ce00",
    ),
    "stage3_feature_store": (
        "data/processed/stage3/anomaly_history/"
        "anomaly-feature-bundle-de7547faa9f87541/anomaly_feature_store.parquet",
        "cd383d52cbb85ebba0e495e58f6e3d50d350f952e5432d5cd21386f6224042ef",
    ),
    "stage3_anomaly_state_history": (
        "data/processed/stage3/anomaly_history/"
        "anomaly-feature-bundle-de7547faa9f87541/anomaly_state_history.parquet",
        "d948d59851778ad4c73be398e521883520fdd87880b0604628986cee2cb2813a",
    ),
    "anomaly_observation": (
        "data/processed/stage1/debc98054172a4a1/anomaly_observation.parquet",
        "a4837a6a865ca0d4939fd45d8d8e9c3e4fd90cf0a78e9bb07d57335d5743b80c",
    ),
    "anomaly_report_period": (
        "data/processed/stage1/debc98054172a4a1/anomaly_report_period.parquet",
        "aac44418994b6b99c633dcf536e452ad315439336bc580a13f719d95a069e176",
    ),
    "feature_dictionary": (
        "data/manifests/anomaly_feature_dictionary.json",
        "60b08bed829a3e496bdbea14e3c0afa64809d69eb0a81532be88af7c7efbe050",
    ),
    "earthquake_target": (
        "data/processed/stage1/debc98054172a4a1/earthquake_event.parquet",
        "2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347",
    ),
    "study_area": (
        "data/processed/china_mainland.geojson",
        "5e5dcf012e080882161c95bf592a1ee39a0f0fdad7114bcff58d645aeb30bb02",
    ),
    "environment_lock": (
        "uv.lock",
        "34188ff1d0aa38996233412d36ef65eb5076205c376634368ddd66582097efa5",
    ),
    "construction_linework_l1": (
        "PlotBD/CN-block-L1.gmt",
        "30d7fabbea95040fed596a37dfd07970c6e7699187c27ad064471db25ef5d5cd",
    ),
    "construction_linework_l2": (
        "PlotBD/CN-block-L2.gmt",
        "189b81655411225ad3d7a1860829835ad23b843c239134529bbad9f2d8d98c33",
    ),
}

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PRIMARY_HORIZONS = (7, 30, 90)
ALL_HORIZONS = (7, 30, 90, 180, 365)


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _load_yaml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha256(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _public_stage4_anomaly_issue_dates() -> tuple[date, ...]:
    """Recreate the public weekly calendar needed by the stage-4 fold builder."""

    # The accepted public data-quality report freezes 2024 period 36 as missing;
    # its Thursday issue date is therefore absent rather than imputed as zero.
    missing_issue = date(2024, 9, 5)
    current = date(2022, 7, 21)
    last_stage4_assessment_issue = date(2025, 6, 26)
    values: list[date] = []
    while current <= last_stage4_assessment_issue:
        if current != missing_issue:
            values.append(current)
        current += timedelta(days=7)
    assert len(values) == 153
    return tuple(values)


def _expected_cli_check_result(protocol: dict[str, Any]) -> dict[str, Any]:
    validation = validate_stage4_protocol_bundle(
        protocol,
        fold_manifest=_load_json(FOLD_MANIFEST_PATH),
        feature_manifest=_load_json(FEATURE_SET_PATH),
        randomness_manifest=_load_json(RANDOMNESS_PATH),
        spatial_manifest=_load_json(SPATIAL_STRATA_PATH),
    )
    return {
        "action": "check",
        "local_artifact_count": 4,
        "manifest_count": 4,
        "protocol_design_sha256": protocol_design_sha256(protocol),
        "random_input_seal_sha256": _load_json(RANDOMNESS_PATH)["frozen_input_seal_sha256"],
        "target_read_count": 0,
        "validation_content_sha256": validation["content_sha256"],
    }


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {item for child in value.values() for item in _walk_keys(child)}
    if isinstance(value, list):
        return {item for child in value for item in _walk_keys(child)}
    return set()


def _independent_seed_reference(context: dict[str, Any]) -> dict[str, Any]:
    """Recompute a vector without calling Stage4SeedContext or its helpers."""

    digest = hashlib.sha256(
        b"\x00".join(
            (
                b"seismoflux-stage4",
                b"147",
                _canonical_json_bytes(context),
            )
        )
    ).digest()
    entropy = int.from_bytes(digest[:16], byteorder="big", signed=False)

    def generator() -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(entropy))

    return {
        "context": context,
        "context_sha256": hashlib.sha256(_canonical_json_bytes(context)).hexdigest(),
        "reference_permutation_n5": generator().permutation(5).tolist(),
        "reference_permutation_n7": generator().permutation(7).tolist(),
        "reference_resample_indices_n5": generator()
        .integers(0, 5, size=5, dtype=np.int64)
        .tolist(),
        "reference_uint64": int(generator().integers(0, 2**64, dtype=np.uint64)),
    }


def test_score_free_protocol_binds_exact_inputs_four_manifests_and_two_freezes() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)

    assert protocol["protocol_version"] == "0.4.1"
    assert protocol["frozen_on"] == "2026-07-16"
    assert protocol["stage"] == 4
    assert protocol["status"] == "preregistered_before_any_stage4_target_score"
    assert protocol["hypotheses"] == ["H1", "H2"]
    assert protocol["gates"] == ["G2", "G3"]

    freeze = protocol["freeze"]
    assert freeze["execution_revision"] == "r2"
    assert freeze["corrects_execution_revision"] == "r1"
    assert freeze["pre_score_tag"] == "v0.3.1-anomaly-increment-protocol-r2"
    assert freeze["results_tag"] == "v0.3.1-anomaly-increment-r2"
    assert freeze["protocol_tag_authorizes_only_score_free_implementation"] is True
    assert freeze["scores_seen_before_freeze"] is False
    assert freeze["target_counts_seen_before_freeze"] is False
    assert freeze["locked_test_results_seen"] is False
    assert freeze["validation_is_not_a_tuning_partition"] is True
    assert freeze["formal_validation_scientific_runs_allowed"] == 1

    scoring_freeze = freeze["scoring_code_freeze"]
    assert scoring_freeze["required"] is True
    assert scoring_freeze["expected_tag"] == ("v0.3.1-anomaly-increment-scoring-code-r2")
    assert scoring_freeze["required_seal_path"] == (
        "data/manifests/anomaly_increment_r2_scoring_seal.json"
    )
    assert scoring_freeze["selected_table_logical_identity"] == {
        "method_id": "arrow_ipc_selected_table_logical_identity_r1",
        "sha256_domain_separator_ascii": ("seismoflux.selected-table-logical-identity.r1"),
        "sha256_domain_separator_nul_terminated": True,
        "top_level_schema_metadata": "excluded",
        "field_name_order_type_nullability_and_metadata": "preserved_exactly",
        "null_payload": "canonical_type_zero",
        "validity_bitmap": "preserved_with_length_padding_zeroed",
        "boolean_value_padding": "zeroed_outside_logical_length",
        "chunking_and_slice_offsets": "canonicalized",
        "field_metadata_key_order": "bytewise_ascending",
        "supported_types": [
            "boolean",
            "signed_integer",
            "unsigned_integer",
            "floating_point",
            "timestamp",
            "utf8_string",
        ],
        "valid_payload_bits": "preserved_exactly",
        "unsupported_types": "fail_closed",
    }
    assert {
        "scoring_code_commit_pushed",
        "scoring_code_tag_pushed",
        "execution_seal_verified",
        "synthetic_end_to_end_passed",
        "cpu_float64_numerical_regression_passed",
        "restricted_spatial_artifact_hashes_verified",
        "formal_attempt_count_equals_zero",
        "target_read_count_equals_zero",
        "logical_arrow_identity_r1_verified",
    } <= set(scoring_freeze["required_before_target_read"])
    assert scoring_freeze["gpu_if_not_equivalent_at_code_freeze"] == (
        "lock_formal_run_to_cpu_float64"
    )

    global_gate = set(freeze["required_before_any_target_read_or_score"])
    assert {
        "protocol_commit_pushed",
        "pre_score_tag_pushed",
        "generated_manifest_hashes_verified",
        "topology_gate_passed",
        "all_non_target_tests_passed",
        "scoring_code_commit_pushed",
        "scoring_code_tag_pushed",
        "execution_seal_verified",
        "formal_attempt_count_equals_zero",
        "target_read_count_equals_zero",
    } <= global_gate
    assert freeze["pre_score_tag"] != scoring_freeze["expected_tag"]

    inputs = protocol["inputs"]
    assert set(inputs) == set(EXPECTED_INPUTS)
    for input_id, (expected_path, expected_sha256) in EXPECTED_INPUTS.items():
        recorded = inputs[input_id]
        assert recorded["path"] == expected_path
        assert recorded["sha256"] == expected_sha256
        assert SHA256_RE.fullmatch(recorded["sha256"])

    feature_store = inputs["stage3_feature_store"]
    assert feature_store["bundle_id"] == "anomaly-feature-bundle-de7547faa9f87541"
    assert feature_store["content_sha256"] == (
        "9f2f25f3f0d27f435fc6e4ee191538f70b88de959d9b357cef464b6884f8bfc9"
    )
    assert feature_store["schema_sha256"] == (
        "0d7a9eab211324fcf3d03f6a55ef0f5136a6132f0ce0c56a72cb0a92912e6808"
    )
    target_metadata = inputs["earthquake_target"]
    assert target_metadata["unavailable_before_protocol_freeze"] is True
    assert target_metadata["human_prediction_fields_forbidden"] is True

    generated = protocol["generated_manifests"]
    assert set(generated) == set(GENERATED_MANIFEST_PATHS)
    for manifest_id, path in GENERATED_MANIFEST_PATHS.items():
        assert generated[manifest_id] == {
            "path": path.as_posix(),
            "sha256": _file_sha256(path),
        }


def test_r2_corrective_contract_freezes_placebos_resources_coverage_and_publication() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)

    retirement = protocol["freeze"]["r1_retirement"]
    assert retirement["formal_attempt_ledger"]["operation_count"] == 0
    assert retirement["target_read_ledger"]["operation_count"] == 0
    assert retirement["target_bytes_observed"] is False
    assert retirement["reusable_for_r2_authorization"] is False

    permutations = protocol["evaluation"]["permutations"]
    assert permutations["formal_requests"] == [
        {"kind": "time", "model_variant": "dynamic"},
        {"kind": "space", "model_variant": "dynamic"},
        {"kind": "time", "model_variant": "snapshot"},
        {"kind": "space", "model_variant": "snapshot"},
    ]
    assert permutations["formal_checkpoint_request_identities"] == [
        "time-dynamic",
        "space-dynamic",
        "time-snapshot",
        "space-snapshot",
    ]
    assert permutations["checkpoint_identity_pattern"] == "kind-model_variant"
    assert permutations["exact_request_set_required"] is True
    assert permutations["mappings_paired_across_dynamic_and_snapshot"] is True
    g2 = protocol["evaluation"]["gates"]["G2"]
    assert g2["evaluated_model_variants"] == ["dynamic", "snapshot"]
    assert g2["required_primary_placebos_by_variant"] == {
        "dynamic": ["time", "space"],
        "snapshot": ["time", "space"],
    }
    assert g2["primary_time_permutation_p_lte"] == 0.05
    assert g2["primary_space_permutation_p_lte"] == 0.05
    assert g2["both_primary_p_values_required_for_each_evaluated_variant"] is True
    assert g2["reporting_confound_guard_applies_independently_to_variants"] == [
        "dynamic",
        "snapshot",
    ]
    assert g2["candidate_minus_coverage_only_macro_information_gain_lower_95pct_bound_gt"] == 0
    assert {branch["candidate_variant"] for branch in g2["practical_improvement_any_of"]} == {
        "current_evaluated_candidate_variant"
    }

    compute = protocol["compute"]
    assert compute["max_workers"] == 6
    assert compute["logical_cpu_affinity_limit"] == 6
    assert compute["process_priority"] == "below_normal"
    assert compute["nested_parallelism"] is False
    assert compute["blas_threads_per_worker"] == 1

    coverage = protocol["inputs"]["earthquake_target"]["frozen_catalog_coverage"]
    assert coverage["observed_origin_time_max_utc"] == "2026-07-09T04:25:56Z"
    assert coverage["observed_available_at_max_utc"] == "2026-07-09T04:25:56Z"
    assert coverage["frozen_validation_window_end_max_utc"] == "2025-07-18T16:00:00Z"
    assert (
        coverage["all_frozen_validation_window_endpoints_must_be_lte_both_catalog_maxima"] is True
    )
    assert coverage["missing_or_short_coverage_action"].startswith("fail_closed")

    publication = protocol["publication"]
    assert publication["result_identity_requires"] == [
        "dynamic_G2",
        "snapshot_equivalent_G2",
        "time_dynamic_placebo_result_distribution",
        "space_dynamic_placebo_result_distribution",
        "time_snapshot_placebo_result_distribution",
        "space_snapshot_placebo_result_distribution",
        "dynamic_G3",
        "adoption_decision",
        "adopted_variant_metrics_table",
    ]
    isolation = publication["spatial_output_isolation"]
    spatial_files = (
        isolation["forecast_target_free_files"]
        + isolation["retrospective_target_bearing_local_restricted_files"]
    )
    assert isolation["physical_file_count"] == 4
    assert len(spatial_files) == len(set(spatial_files)) == 4
    assert isolation["target_payload_in_forecast_files_forbidden"] is True
    assert isolation["automatic_cross_file_target_loading_forbidden"] is True
    assert isolation["public_forecast_artifact_validator"] == {
        "reject_artifact_classifications": ["local_restricted", "target_bearing"],
        "forbidden_payload_fields": [
            "event_id",
            "target_coordinates",
            "target_longitude",
            "target_latitude",
            "epicenter_longitude",
            "epicenter_latitude",
            "hit_status",
            "target_marker",
        ],
        "validation_scope": ("parsed_static_dom_and_recursively_deserialized_interactive_payload"),
        "keyword_scan_or_ui_hiding_sufficient": False,
        "failure_action": "fail_closed_forbid_publication",
    }
    assert publication["limitations"] == {
        "earthquake_available_at_assumption": (
            "available_at_equals_origin_time_is_an_optimistic_timeliness_assumption"
        ),
        "bootstrap_interval_scope": (
            "conditional_on_fixed_fitted_model_and_excludes_refit_uncertainty"
        ),
        "etas_comparator_status": "not_evaluable",
        "allowed_increment_claim": "relative_to_frozen_kde_background_only",
        "incremental_value_over_etas_claim_forbidden": True,
    }
    assert publication["display_semantics"] == {
        "coverage_only_option_required": True,
        "aggregate_retrospective_view": {
            "issue_and_model_controls": "hidden_or_disabled",
            "required_summary_label_template_zh": "全部{N}个起报日汇总",
            "issue_count_source": "frozen_issue_calendar",
        },
        "peak_value_100pct": {
            "required_label_zh": "峰值网格百分位",
            "prediction_accuracy_term_forbidden": True,
        },
        "relative_strength": {
            "formula": "peak_integrated_grid_intensity/mean_integrated_grid_intensity",
            "absolute_probability_interpretation_forbidden": True,
        },
        "adoption": {
            "adoption_card_required": True,
            "adopted_variant_required": True,
        },
        "latest_retrospective_landmark": {
            "required_label_zh": "最新冻结日历地标",
            "current_forecast_implication_forbidden": True,
        },
        "forecast_spatial": {
            "rendered_variant": "adopted_variant",
            "unadopted_dynamic_required_label": "research_candidate",
            "unadopted_dynamic_may_not_be_current_forecast": True,
        },
        "placebo_static_panel_layout": {
            "required_panels": [
                "time_dynamic",
                "space_dynamic",
                "time_snapshot",
                "space_snapshot",
            ],
            "all_panels_within_render_bounds_required": True,
            "render_boundary_test_required": True,
        },
    }


def test_all_four_public_manifests_are_content_addressed_and_score_free() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    forbidden_result_fields = set(protocol["forbidden_result_fields_before_freeze"])
    assert {
        "target_ids",
        "target_locations",
        "target_count",
        "model_score",
        "information_gain",
        "hit_result",
        "permutation_result",
        "p_value",
        "selected_anomaly_model_id",
        "score_id",
        "locked_test_result",
    } <= forbidden_result_fields

    for path in GENERATED_MANIFEST_PATHS.values():
        document = _load_json(path)
        assert SHA256_RE.fullmatch(document["content_sha256"])
        assert document["content_sha256"] == _content_sha256(document), path
        assert verify_content_sha256(document) is True
        assert forbidden_result_fields.isdisjoint(_walk_keys(document)), path


def test_score_free_builders_exactly_regenerate_fold_feature_randomness_and_seal() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    background = _load_json(Path(protocol["inputs"]["background_fold_manifest"]["path"]))
    dictionary = _load_json(Path(protocol["inputs"]["feature_dictionary"]["path"]))
    spatial = _load_json(SPATIAL_STRATA_PATH)

    rebuilt_fold = build_exposure_preregistration(
        background,
        anomaly_issue_dates_local=_public_stage4_anomaly_issue_dates(),
    )
    rebuilt_feature = select_feature_storage_columns(dictionary, protocol)
    seal = build_random_input_seal(
        protocol,
        fold_manifest=rebuilt_fold,
        feature_manifest=rebuilt_feature,
        spatial_manifest=spatial,
    )
    rebuilt_randomness = build_randomness_manifest(
        frozen_input_seal_sha256=cast(str, seal["content_sha256"])
    )

    assert rebuilt_fold == _load_json(FOLD_MANIFEST_PATH)
    assert rebuilt_feature == _load_json(FEATURE_SET_PATH)
    assert rebuilt_randomness == _load_json(RANDOMNESS_PATH)
    assert (
        seal["expected_scoring_code_tag"]
        == (protocol["freeze"]["scoring_code_freeze"]["expected_tag"])
    )
    assert seal["normalized_protocol_design_sha256"] == protocol_design_sha256(protocol)
    assert seal["spatial_manifest_content_sha256"] == spatial["content_sha256"]
    assert seal["spatial_local_artifacts"] == spatial["local_artifacts"]


def test_joint_folds_use_one_shared_7d_fit_and_mutually_disjoint_target_bands() -> None:
    manifest = _load_json(FOLD_MANIFEST_PATH)

    assert manifest["shared_model_fit_exposure_horizon_days"] == 7
    assert manifest["single_parameter_vector_per_variant_and_fit_scope"] is True
    assert manifest["joint_macro_target_bands_mutually_disjoint"] is True
    assert manifest["model_fit_authority"] == (
        "joint_macro_rolling_folds_and_formal_validation_fit_only"
    )
    assert manifest["rolling_fold_rule"] == (
        "three_disjoint_90d_target_bands_with_expanding_shared_7d_fit"
    )

    expected_counts = {
        "development": {7: 50, 30: 10, 90: 4, 180: 2, 365: 1},
        "validation": {7: 51, 30: 11, 90: 4, 180: 2, 365: 1},
    }
    exposure_by_id: dict[str, dict[str, Any]] = {}
    for horizon_days in ALL_HORIZONS:
        horizon = manifest["horizons"][str(horizon_days)]
        assert horizon["assessment_only_not_model_fit"] is True
        expected_role = "primary" if horizon_days in PRIMARY_HORIZONS else "evidence_insufficient"
        assert horizon["role"] == expected_role
        for partition in ("development", "validation"):
            exposures = horizon[f"{partition}_exposures"]
            assert len(exposures) == expected_counts[partition][horizon_days]
            assert len({item["id"] for item in exposures}) == len(exposures)
            previous_end: date | None = None
            for exposure in exposures:
                issue = date.fromisoformat(exposure["issue_date_local"])
                target_start = date.fromisoformat(exposure["target_start_exclusive_local"])
                target_end = date.fromisoformat(exposure["target_end_inclusive_local"])
                assert target_start == issue
                assert target_end == issue + timedelta(days=horizon_days)
                if previous_end is not None:
                    assert target_start >= previous_end
                previous_end = target_end
                assert exposure["id"] not in exposure_by_id
                exposure_by_id[exposure["id"]] = exposure

    folds = manifest["joint_macro_rolling_folds"]
    assert len(folds) == 3
    previous_band_end: datetime | None = None
    previous_fit_ids: set[str] = set()
    assessment_ids_seen: set[str] = set()
    seven_day_development = manifest["horizons"]["7"]["development_exposures"]
    for expected_index, fold in enumerate(folds, start=1):
        assert fold["fold_index"] == expected_index
        assert fold["fit_scope_id"] == f"development-fold-{expected_index}"
        assert fold["assessment_horizons_days"] == list(PRIMARY_HORIZONS)
        assert fold["model_fit_scope"] == "one_per_variant_shared_across_all_horizons"
        assert fold["preprocessor_fit_scope"] == "same_7d_fit_exposures_as_model"

        band_start = _parse_utc(fold["assessment_band"]["start_exclusive_utc"])
        band_end = _parse_utc(fold["assessment_band"]["end_inclusive_utc"])
        assert band_start < band_end
        if previous_band_end is not None:
            assert previous_band_end < band_start
        previous_band_end = band_end

        fit_ids = fold["fit_exposure_ids_7d"]
        assert fit_ids
        assert all(identifier.startswith("development-h007-") for identifier in fit_ids)
        expected_fit_ids = [
            exposure["id"]
            for exposure in seven_day_development
            if _parse_utc(exposure["target_end_inclusive_utc"]) < band_start
        ]
        assert fit_ids == expected_fit_ids
        assert previous_fit_ids < set(fit_ids) if previous_fit_ids else True
        previous_fit_ids = set(fit_ids)
        assert _parse_utc(fold["fit_target_end_inclusive_utc"]) < band_start

        fold_assessment_ids: set[str] = set()
        for horizon_days in PRIMARY_HORIZONS:
            horizon_ids = fold["assessment_exposure_ids_by_horizon"][str(horizon_days)]
            assert horizon_ids
            assert all(
                identifier.startswith(f"development-h{horizon_days:03d}-")
                for identifier in horizon_ids
            )
            for identifier in horizon_ids:
                exposure = exposure_by_id[identifier]
                assert band_start <= _parse_utc(exposure["issue_time_utc"])
                assert _parse_utc(exposure["target_end_inclusive_utc"]) <= band_end
            fold_assessment_ids.update(horizon_ids)
        assert assessment_ids_seen.isdisjoint(fold_assessment_ids)
        assessment_ids_seen.update(fold_assessment_ids)

        pools = fold["time_permutation_feature_pools"]
        assert set(pools["fit"]).isdisjoint(pools["assessment"])
        assert pools["pool_crossing_forbidden"] is True
        assert pools["pseudo_history_concatenation"] == ("permuted_fit_then_permuted_assessment")

    formal = manifest["formal_validation_fit"]
    assert formal["fit_exposure_ids_7d"] == [item["id"] for item in seven_day_development]
    assert formal["model_fit_scope"] == ("one_per_variant_shared_across_all_five_horizons")
    assert formal["validation_refit_forbidden"] is True
    assert set(formal["assessment_exposure_ids_by_horizon"]) == {str(item) for item in ALL_HORIZONS}
    earliest_validation = min(
        _parse_utc(exposure["issue_time_utc"])
        for horizon_days in ALL_HORIZONS
        for exposure in manifest["horizons"][str(horizon_days)]["validation_exposures"]
    )
    assert _parse_utc(formal["fit_target_end_inclusive_utc"]) < earliest_validation


def test_feature_manifest_freezes_typed_design_columns_not_quality_predictors() -> None:
    manifest = _load_json(FEATURE_SET_PATH)
    feature_sets = manifest["feature_sets"]
    assert set(feature_sets) == {"coverage_only", "snapshot", "dynamic"}
    assert manifest["selection"] == {
        "confirmatory_kernel": "gaussian",
        "confirmatory_scale_km": 200,
        "trajectory_kernel": "closed_ball",
        "trajectory_scale_km": 200,
        "trajectory_source_series": ["listed_count", "first_seen_count"],
    }
    assert manifest["forbidden_source_match_count"] == 0
    assert manifest["manual_prediction_feature_use_count"] == 0
    assert manifest["source_duration_feature_use_count"] == 0

    expected_counts = {
        "coverage_only": (9, 9, 3, 18),
        "snapshot": (17, 17, 8, 34),
        "dynamic": (22, 27, 38, 54),
    }
    output_sets: dict[str, set[str]] = {}
    for feature_set_id, (
        logical_count,
        source_count,
        audit_count,
        design_count,
    ) in expected_counts.items():
        feature_set = feature_sets[feature_set_id]
        logical = feature_set["logical_features"]
        source = feature_set["source_value_columns"]
        audit = feature_set["audit_only_quality_columns"]
        design = feature_set["design_columns"]
        outputs = feature_set["design_output_columns"]
        assert (len(logical), len(source), len(audit), len(design)) == (
            logical_count,
            source_count,
            audit_count,
            design_count,
        )
        assert len(outputs) == design_count
        assert len(outputs) == len(set(outputs))
        assert source_count * 2 == design_count
        assert set(source).isdisjoint(audit)
        assert set(outputs).isdisjoint(audit)
        assert feature_set["null_reason_validity_and_sample_count_predictors_forbidden"] is True
        assert (
            feature_set["source_value_columns_sha256"]
            == hashlib.sha256(_canonical_json_bytes(source)).hexdigest()
        )
        assert (
            feature_set["audit_only_quality_columns_sha256"]
            == hashlib.sha256(_canonical_json_bytes(audit)).hexdigest()
        )
        assert (
            feature_set["design_columns_sha256"]
            == hashlib.sha256(_canonical_json_bytes(design)).hexdigest()
        )
        assert (
            feature_set["design_output_columns_sha256"]
            == hashlib.sha256(_canonical_json_bytes(outputs)).hexdigest()
        )

        by_source: dict[str, list[dict[str, Any]]] = {}
        for column in design:
            by_source.setdefault(column["source_column"], []).append(column)
            assert column["penalty_factor"] == 1.0
            assert column["categorical_encoding"] is None
            assert column["zero_scale_action"] == (
                "coefficient_fixed_zero_only_when_training_min_equals_max"
            )
        assert set(by_source) == set(source)
        for source_column, pair in by_source.items():
            assert [column["role"] for column in pair] == ["value", "missing_indicator"]
            assert [column["output_column"] for column in pair] == [
                f"value__{source_column}",
                f"missing__{source_column}",
            ]
            assert pair[0]["imputation"] == "training_median_after_transform"
            assert pair[1]["imputation"] == "none"
            assert pair[1]["transform"] == "is_null_boolean"
            assert pair[1]["standardization"] == "none"
        assert outputs == [column["output_column"] for column in design]
        output_sets[feature_set_id] = set(outputs)

    assert output_sets["coverage_only"] < output_sets["snapshot"]
    assert output_sets["snapshot"] < output_sets["dynamic"]
    assert manifest["fault_features"] == {
        "forbidden_in_stage4_model": True,
        "status": "deferred_to_stage_5",
    }


def test_typed_rng_vectors_cover_zero_one_and_last_for_every_legal_context() -> None:
    manifest = _load_json(RANDOMNESS_PATH)

    assert manifest["root_seed"] == 147
    assert manifest["generator"] == "numpy.random.PCG64"
    assert manifest["worker_count_invariant"] is True
    assert manifest["direct_integer_subseeds_forbidden"] is True
    assert manifest["context_canonicalization"] == "canonical_json_utf8_sort_keys_no_nan"
    assert manifest["paired_result_dimensions_not_seed_fields"] == [
        "model_variant",
        "magnitude_bin",
        "horizon",
        "metric",
    ]
    assert manifest["pairing_scope"] == (
        "shared_across_model_variants_magnitude_bins_all_five_horizons_and_metrics"
    )
    assert SHA256_RE.fullmatch(manifest["frozen_input_seal_sha256"])

    families = manifest["families"]
    expected = {
        "bootstrap": (2000, {"joint"}, 12),
        "time_permutation": (1000, {"fit", "assessment"}, 24),
        "space_permutation": (1000, {"fit", "assessment"}, 24),
    }
    forbidden_context_fields = {
        "model_variant",
        "magnitude_bin",
        "horizon",
        "metric",
        "worker_count",
    }
    for purpose, (replications, roles, vector_count) in expected.items():
        family = families[purpose]
        assert family["replications"] == replications
        vectors = family["reference_vectors"]
        assert len(vectors) == vector_count
        observed_contexts: set[tuple[str, str, int]] = set()
        for vector in vectors:
            context = vector["context"]
            assert set(context) == set(family["context_fields"])
            assert forbidden_context_fields.isdisjoint(context)
            assert context["purpose"] == purpose
            assert context["evaluation_id"] in STAGE4_RANDOM_EVALUATION_IDS
            assert context["partition_role"] in roles
            assert context["replication_index"] in {0, 1, replications - 1}
            assert context["frozen_input_seal_sha256"] == (manifest["frozen_input_seal_sha256"])
            assert vector == _independent_seed_reference(context)
            observed_contexts.add(
                (
                    context["evaluation_id"],
                    context["partition_role"],
                    context["replication_index"],
                )
            )
            if purpose == "space_permutation":
                assert context["issue_id"] == "reference-issue-only"
                assert context["construction_stratum_id"] == ("reference-zone-inside-only")
            else:
                assert "issue_id" not in context
                assert "construction_stratum_id" not in context

        expected_contexts = {
            (evaluation_id, role, index)
            for evaluation_id in STAGE4_RANDOM_EVALUATION_IDS
            for role in roles
            for index in (0, 1, replications - 1)
        }
        assert observed_contexts == expected_contexts

    assert families["bootstrap"]["full_model_refit_each_replication"] is False
    assert families["bootstrap"]["per_horizon_marginal_event_count_preserved"] is True
    assert (
        families["time_permutation"][
            "anomaly_increment_coefficients_and_preprocessor_refit_each_replication"
        ]
        is True
    )
    assert (
        families["space_permutation"][
            "anomaly_increment_coefficients_and_preprocessor_refit_each_replication"
        ]
        is True
    )


def test_entity_spatial_strata_and_topology_gates_are_both_green() -> None:
    manifest = _load_json(SPATIAL_STRATA_PATH)
    protocol = _load_yaml(PROTOCOL_PATH)

    topology = manifest["topology_gate"]
    entity_gate = manifest["entity_stratification_gate"]
    input_hash_gate = manifest["input_hash_gate"]
    assert manifest["spatial_placebo_implementation_authorized"] is True
    assert manifest["stage4_target_read_authorized"] is False
    assert topology["passed"] is True
    assert entity_gate["passed"] is True
    assert input_hash_gate["passed"] is True
    assert input_hash_gate["all_hashes_match"] is True
    assert entity_gate["required_for_scoring_authorization"] is True
    assert input_hash_gate["required_for_scoring_authorization"] is True

    assert topology["dangles_count"] == 0
    assert topology["cuts_count"] == 0
    assert topology["invalid_count"] == 0
    assert topology["polygonize_precision_m"] == 1
    assert topology["input_segment_order_identity_stable"] is True
    assert topology["zone_count_invariant_across_precision"] is True
    assert topology["every_fixed_query_cell_exactly_one_zone"] is True
    assert topology["fixed_query_grid_assignment"] == {
        "ambiguous_count": 0,
        "assigned_once_count": 15697,
        "query_cell_count": 15697,
        "unassigned_count": 0,
    }
    assert [item["precision_m"] for item in topology["precision_sensitivity"]] == [
        0.001,
        0.01,
        0.1,
        1,
        10,
    ]
    assert {item["zone_count"] for item in topology["precision_sensitivity"]} == {65}

    aggregate = manifest["aggregate"]
    assert aggregate["zone_count"] == 65
    assert aggregate["assigned_query_cell_count"] == 15697
    assert aggregate["assigned_nonempty_zone_count"] == 39
    assert aggregate["zero_cell_zone_count"] == 26
    assert aggregate["singleton_nonempty_grid_zone_count"] == 2
    assert aggregate["boundary_tie_count"] == 0
    assert aggregate["precision_snap_count"] == 0
    assert aggregate["assigned_spatial_entity_state_count"] == 165841
    assert aggregate["spatially_ineligible_entity_state_count"] == 143
    assert aggregate["total_entity_state_count"] == 165984
    assert (
        aggregate["assigned_spatial_entity_state_count"]
        + aggregate["spatially_ineligible_entity_state_count"]
        == aggregate["total_entity_state_count"]
    )
    assert aggregate["entity_permutation_group_count"] == (
        aggregate["entity_permutation_singleton_group_count"]
        + aggregate["entity_permutation_identical_coordinate_group_count"]
        + aggregate["entity_permutation_nontrivial_group_count"]
    )
    assert aggregate["entity_boundary_tie_count"] == 0
    assert aggregate["entity_nearest_distance_tie_count"] == 0
    assert aggregate["entity_precision_snap_count"] == 0

    assert entity_gate == {
        "all_entity_states_accounted_for": True,
        "all_spatially_eligible_states_assigned_once": True,
        "boundary_tie_rule": "lexicographically_smallest_zone_id",
        "outside_rule": "nearest_zone_id_with_separate_outside_flag_stratum",
        "passed": True,
        "required_for_scoring_authorization": True,
        "spatially_ineligible_rule": "fixed_exclusion_never_permuted",
    }
    space = protocol["evaluation"]["permutations"]["space"]
    assert space["unit"] == (
        "entity_coordinate_pair_bijection_within_issue_and_construction_stratum"
    )
    assert space["spatially_ineligible_entities"] == (
        "remain_excluded_and_never_enter_permutation_pool"
    )
    assert space["zero_query_cell_construction_zones"]["cross_zone_fallback_forbidden"] is True
    assert space["singleton_or_identical_coordinate_stratum_action"] == (
        "identity_and_report_no_effect"
    )
    assert space["linework_or_zone_may_not_enter_model_features"] is True

    assert set(manifest["local_artifacts"]) == {
        "cell_mapping",
        "entity_mapping",
        "connectors",
        "zone_geometry",
    }
    for artifact in manifest["local_artifacts"].values():
        assert set(artifact) == {"byte_count", "media_type", "sha256"}
        assert artifact["byte_count"] > 0
        assert SHA256_RE.fullmatch(artifact["sha256"])
    assert manifest["publication_safety"] == {
        "contains_coordinates": False,
        "contains_geometry": False,
        "contains_per_cell_mapping": False,
        "contains_wkb_or_geojson": False,
        "raw_linework_redistributed": False,
    }
    security = manifest["security"]
    assert security["coordinate_bearing_artifacts_local_only"] is True
    assert security["earthquake_catalog_read"] is False
    assert security["target_or_score_read"] is False
    assert security["geometry_used_as_model_feature"] is False

    forbidden_public_keys = {
        "coordinates",
        "geometry",
        "wkb",
        "geojson",
        "longitude",
        "latitude",
        "cell_id",
        "cell_to_stratum",
        "per_cell_mapping",
        "assignments",
    }
    assert forbidden_public_keys.isdisjoint({key.casefold() for key in _walk_keys(manifest)})


def test_model_gates_placebos_gpu_and_locked_test_remain_frozen() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    model = protocol["model"]
    assert model["background_variant_id"] == "spatial_poisson/gaussian_kde_bw75km"
    assert model["family"] == "ridge_regularized_poisson_point_process"
    assert model["fixed_hyperparameters"] == {
        "spatial_scale_km": 200,
        "anomaly_half_life_days": 90,
        "ridge_lambda": 1.0,
    }
    assert model["hyperparameter_selection_from_validation_forbidden"] is True
    assert model["gradient_boosting_allowed"] is False
    assert model["neural_model_allowed"] is False

    evaluation = protocol["evaluation"]
    assert evaluation["bootstrap"]["replications"] == 2000
    assert evaluation["bootstrap"]["horizons_jointly_resampled_days"] == list(ALL_HORIZONS)
    for null_id in ("time", "space"):
        permutation = evaluation["permutations"][null_id]
        assert permutation["replications"] == 1000
        assert (
            permutation["anomaly_increment_coefficients_and_preprocessor_refit_each_replication"]
            is True
        )
        assert permutation["frozen_background_and_target_rate_head_reused_each_replication"]
        assert permutation["background_and_targets_fixed"] is True

    g2 = evaluation["gates"]["G2"]
    assert g2["evaluation_partition"] == "formal_validation_once"
    assert g2["primary_horizons_days"] == list(PRIMARY_HORIZONS)
    assert g2["minimum_unique_independent_events"] == {
        "threshold": 20,
        "definition": "deduplicated_union_of_physical_event_ids_across_three_primary_horizons",
    }
    assert g2["macro_information_gain_lower_95pct_bound_gt"] == 0
    assert g2["primary_time_permutation_p_lte"] == 0.05
    assert g2["candidate_minus_coverage_only_macro_information_gain_lower_95pct_bound_gt"] == 0
    g3 = evaluation["gates"]["G3"]
    assert g3["evaluation_partition"] == "development_joint_macro_rolling_folds_only"
    assert g3["target_bands_must_be_mutually_disjoint"] is True
    assert g3["one_model_fit_per_variant_per_fold_shared_across_horizons"] is True
    assert g3["outer_fold_count_required"] == 3

    gpu = protocol["compute"]["gpu_equivalence"]
    assert protocol["compute"]["default_backend"] == "cpu_float64"
    assert protocol["compute"]["reserve_physical_cores"] >= 2
    assert gpu["status"] == "optional_acceleration_blocked_until_equivalence_passes"
    assert gpu["cpu_path_required"] is True
    assert gpu["random_permutations_generated_on_cpu_and_transferred_unchanged"] is True
    assert gpu["float64_required"] is True
    assert gpu["failure_action"] == "cpu_only"

    locked = protocol["locked_test"]
    assert locked == {
        "action": "do_not_run",
        "run": False,
        "issue_dates_start": None,
        "issue_dates_end": None,
        "target_count": None,
        "target_ids": [],
        "score_ids": [],
        "artifact_ids": [],
        "result": None,
        "cohort_manifest_path": None,
        "formal_test_execution_forbidden_in_stage4": True,
        "formal_test_belongs_to_stage9": True,
        "outcomes_must_remain_unread": True,
    }


def test_builders_and_typed_rng_fail_closed_under_design_mutations() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    background = _load_json(Path(protocol["inputs"]["background_fold_manifest"]["path"]))
    dictionary = _load_json(Path(protocol["inputs"]["feature_dictionary"]["path"]))
    fold = _load_json(FOLD_MANIFEST_PATH)
    feature = _load_json(FEATURE_SET_PATH)
    spatial = _load_json(SPATIAL_STRATA_PATH)

    bad_background = deepcopy(background)
    bad_background["partitions"]["development"]["actual_issue_date_count"] += 1
    with pytest.raises(ValueError, match="actual issue-date count mismatch"):
        build_exposure_preregistration(
            bad_background,
            anomaly_issue_dates_local=_public_stage4_anomaly_issue_dates(),
        )

    missing_issue_calendar = _public_stage4_anomaly_issue_dates()[1:]
    with pytest.raises(ValueError, match="every scored issue"):
        build_exposure_preregistration(
            background,
            anomaly_issue_dates_local=missing_issue_calendar,
        )

    bad_protocol = deepcopy(protocol)
    selected_name = bad_protocol["features"]["coverage_controls"][0]
    selected_definition = next(
        item for item in dictionary["features"] if item["name"] == selected_name
    )
    bad_dictionary = deepcopy(dictionary)
    bad_selected_definition = next(
        item for item in bad_dictionary["features"] if item["name"] == selected_name
    )
    bad_selected_definition["causal_sources"] = [
        *selected_definition["causal_sources"],
        "earthquake_target",
    ]
    with pytest.raises(ValueError, match="forbidden causal-source tokens"):
        select_feature_storage_columns(bad_dictionary, bad_protocol)

    missing_transform_protocol = deepcopy(protocol)
    del missing_transform_protocol["preprocessing"]["transform_by_logical_feature"][selected_name]
    with pytest.raises(ValueError, match="must exactly cover"):
        select_feature_storage_columns(dictionary, missing_transform_protocol)

    tampered_fold = deepcopy(fold)
    tampered_fold["shared_model_fit_exposure_horizon_days"] = 30
    assert verify_content_sha256(tampered_fold) is False
    with pytest.raises(ValueError, match="fold manifest content hash is invalid"):
        build_random_input_seal(
            protocol,
            fold_manifest=tampered_fold,
            feature_manifest=feature,
            spatial_manifest=spatial,
        )

    original_seal = build_random_input_seal(
        protocol,
        fold_manifest=fold,
        feature_manifest=feature,
        spatial_manifest=spatial,
    )
    changed_tag_protocol = deepcopy(protocol)
    changed_tag_protocol["freeze"]["scoring_code_freeze"]["expected_tag"] += "-mutated"
    changed_seal = build_random_input_seal(
        changed_tag_protocol,
        fold_manifest=fold,
        feature_manifest=feature,
        spatial_manifest=spatial,
    )
    assert changed_seal["content_sha256"] != original_seal["content_sha256"]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_randomness_manifest(frozen_input_seal_sha256="not-a-seal")

    valid_seal = cast(str, original_seal["content_sha256"])
    invalid_contexts = (
        {
            "purpose": "bootstrap",
            "evaluation_id": "development-fold-1",
            "partition_role": "fit",
            "replicate_index": 0,
            "frozen_input_seal_sha256": valid_seal,
        },
        {
            "purpose": "time_permutation",
            "evaluation_id": "development-fold-1",
            "partition_role": "joint",
            "replicate_index": 0,
            "frozen_input_seal_sha256": valid_seal,
        },
        {
            "purpose": "space_permutation",
            "evaluation_id": "development-fold-1",
            "partition_role": "fit",
            "replicate_index": 1000,
            "frozen_input_seal_sha256": valid_seal,
            "issue_id": "issue",
            "construction_stratum_id": "zone",
        },
    )
    for kwargs in invalid_contexts:
        with pytest.raises(ValueError):
            Stage4SeedContext(**cast(Any, kwargs))


def test_cli_check_verifies_public_bundle_with_a_synthetic_restricted_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    assert protocol["spatial_permutation_topology"]["reproducibility"]["check_command"] == (
        ".venv/Scripts/python.exe scripts/build_stage4_preregistration.py check"
    )

    spatial = _load_json(SPATIAL_STRATA_PATH)
    configured = protocol["spatial_permutation_topology"]["local_restricted_artifacts"]
    artifact_to_config = {
        "cell_mapping": "cell_mapping",
        "entity_mapping": "entity_mapping",
        "connectors": "connectors",
        "zone_geometry": "zone_geometry",
    }
    synthetic_paths: dict[str, Path] = {}
    for artifact_id, config_key in artifact_to_config.items():
        path = tmp_path / f"{artifact_id}.synthetic"
        path.write_bytes(b"\x00" * spatial["local_artifacts"][artifact_id]["byte_count"])
        synthetic_paths[config_key] = path

    real_project_path = preregistration_cli._project_path
    real_verify_file = preregistration_cli._verify_file

    def project_path(value: object, *, label: str) -> Path:
        if label in synthetic_paths:
            assert value == configured[label]
            return synthetic_paths[label]
        return cast(Path, real_project_path(value, label=label))

    synthetic_path_set = set(synthetic_paths.values())

    def verify_file(path: Path, expected_sha256: object, *, label: str) -> None:
        if path in synthetic_path_set:
            artifact_id = next(
                key for key, config_key in artifact_to_config.items() if config_key in label
            )
            assert expected_sha256 == spatial["local_artifacts"][artifact_id]["sha256"]
            return
        real_verify_file(path, expected_sha256, label=label)

    monkeypatch.setattr(preregistration_cli, "_project_path", project_path)
    monkeypatch.setattr(preregistration_cli, "_verify_file", verify_file)
    result = preregistration_cli.check(protocol)
    assert result == _expected_cli_check_result(protocol)


def test_cli_check_local_restricted_artifact_integration_or_explicit_skip() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    configured = protocol["spatial_permutation_topology"]["local_restricted_artifacts"]
    artifact_paths = [
        Path(configured[key])
        for key in ("cell_mapping", "entity_mapping", "connectors", "zone_geometry")
    ]
    missing = [path.as_posix() for path in artifact_paths if not path.is_file()]
    if missing:
        pytest.skip(
            "Local restricted stage-4 spatial artifacts are unavailable: " + ", ".join(missing)
        )

    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "check"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)
    assert result == _expected_cli_check_result(protocol)


def test_cli_check_fails_closed_when_a_restricted_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    configured = protocol["spatial_permutation_topology"]["local_restricted_artifacts"]
    local_config_keys = {"cell_mapping", "entity_mapping", "connectors", "zone_geometry"}
    real_project_path = preregistration_cli._project_path

    def project_path(value: object, *, label: str) -> Path:
        if label in local_config_keys:
            assert value == configured[label]
            return tmp_path / f"missing-{label}"
        return cast(Path, real_project_path(value, label=label))

    monkeypatch.setattr(preregistration_cli, "_project_path", project_path)
    with pytest.raises(FileNotFoundError, match="missing-cell_mapping"):
        preregistration_cli.check(protocol)


def test_cli_check_rejects_a_tampered_public_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    copied_paths: dict[str, Path] = {}
    for manifest_id, source in GENERATED_MANIFEST_PATHS.items():
        destination = tmp_path / source.name
        shutil.copyfile(source, destination)
        copied_paths[manifest_id] = destination
    tampered = _load_json(copied_paths["feature_set"])
    tampered["manual_prediction_feature_use_count"] = 1
    copied_paths["feature_set"].write_text(
        json.dumps(tampered, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preregistration_cli,
        "_generated_paths",
        lambda _protocol: copied_paths,
    )

    with pytest.raises(ValueError, match="invalid content_sha256"):
        preregistration_cli.check(protocol)
