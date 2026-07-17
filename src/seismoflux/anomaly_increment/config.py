"""Strict, target-blind loading of the frozen stage-4 protocol bundle.

This module deliberately has no target-catalog loader.  The earthquake target is
represented only by the identity strings that were frozen in the protocol.  In
particular, loading or validating a :class:`Stage4ProtocolBundle` must not call
``exists()``, ``stat()``, ``open()`` or a hashing function on the target path.
The only target byte access lives in :mod:`seismoflux.anomaly_increment.target_access`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, cast

import yaml

from seismoflux.anomaly_increment.immutable_file import (
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
    sha256_existing_immutable_file,
)
from seismoflux.anomaly_increment.preregistration import (
    protocol_design_sha256,
    validate_stage4_protocol_bundle,
)

STAGE4_EXECUTION_REVISION: Final = "r2"
STAGE4_PROTOCOL_PATH: Final = Path("configs/anomaly_increment_r2.yaml")
STAGE4_PROTOCOL_TAG: Final = "v0.3.1-anomaly-increment-protocol-r2"
STAGE4_SCORING_CODE_TAG: Final = "v0.3.1-anomaly-increment-scoring-code-r2"
STAGE4_RESULT_TAG: Final = "v0.3.1-anomaly-increment-r2"
STAGE4_R2_PROTOCOL_DESIGN_SHA256: Final = (
    "fd51d7f19306c48f95d89905416b1e38b9f2ab0078b4d5078ab83857278e193d"
)
STAGE4_SCORING_SEAL_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r2_scoring_seal.json"
)
STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/formal_preflight_receipt.json"
)
STAGE4_QUALIFICATION_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/scoring_qualification.json"
)
STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/logical_identity_worker_replay.json"
)
STAGE4_JUNIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/qualification_stage4.junit.xml"
)
STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/qualification_full_non_target.junit.xml"
)
STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r2_attempt_ledger.json"
)
STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r2_target_read_ledger.json"
)
STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/checkpoints"
)

_STAGE4_SCORING_FREEZE_PATHS: Final[tuple[tuple[str, PurePosixPath], ...]] = (
    ("required_seal_path", STAGE4_SCORING_SEAL_RELATIVE_PATH),
    ("formal_preflight_receipt_path", STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH),
    ("qualification_path", STAGE4_QUALIFICATION_RELATIVE_PATH),
    ("stage4_junit_path", STAGE4_JUNIT_RELATIVE_PATH),
    ("full_non_target_junit_path", STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH),
    ("formal_attempt_ledger_path", STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH),
    ("target_read_ledger_path", STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH),
    ("checkpoint_root", STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH),
)

_STAGE4_LOGICAL_IDENTITY_CONTRACT: Final[dict[str, object]] = {
    "method_id": "arrow_ipc_selected_table_logical_identity_r1",
    "sha256_domain_separator_ascii": "seismoflux.selected-table-logical-identity.r1",
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

_STAGE4_CURRENT_EXECUTION_AUTHORIZATIONS: Final[dict[str, bool]] = {
    "protocol_tag_allowed": True,
    "target_blind_protocol_manifest_restricted_input_and_test_hardening_allowed": True,
    "scientific_scoring_implementation_allowed": False,
    "scientific_scoring_code_commit_allowed": False,
    "scientific_scoring_code_tag_allowed": False,
    "formal_preflight_allowed": False,
    "qualification_allowed": False,
    "scoring_seal_allowed": False,
    "formal_attempt_ledger_creation_allowed": False,
    "target_read_ledger_creation_allowed": False,
    "formal_target_read_allowed": False,
    "formal_scoring_allowed": False,
    "result_tag_allowed": False,
}
_STAGE4_EXECUTION_ACTION_PERMISSION: Final[dict[str, str]] = {
    "formal_preflight": "formal_preflight_allowed",
    "qualification": "qualification_allowed",
    "scoring_seal": "scoring_seal_allowed",
    "formal_attempt_ledger_creation": "formal_attempt_ledger_creation_allowed",
    "target_read_ledger_creation": "target_read_ledger_creation_allowed",
    "formal_target_read": "formal_target_read_allowed",
    "formal_scoring": "formal_scoring_allowed",
}

JsonObject = dict[str, object]


def _mapping(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(JsonObject, value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _read_yaml_mapping(path: Path) -> JsonObject:
    payload = read_existing_immutable_bytes(path, label=str(path))
    return _mapping(yaml.safe_load(payload.decode("utf-8")), label=str(path))


def _read_json_mapping(path: Path) -> JsonObject:
    payload = read_existing_immutable_bytes(path, label=str(path))
    return _mapping(json.loads(payload.decode("utf-8")), label=str(path))


def _file_sha256(path: Path) -> str:
    return sha256_existing_immutable_file(path, label=str(path))


def validate_stage4_r2_execution_contract(protocol: Mapping[str, object]) -> None:
    """Require the exact R2 execution namespace before any scoring authorization."""

    if protocol.get("protocol_version") != "0.4.1":
        raise ValueError("stage-4 scientific protocol_version must remain 0.4.1")
    if protocol.get("frozen_on") != "2026-07-17":
        raise ValueError("stage-4 R2 final target-blind freeze date changed")
    if protocol.get("status") != (
        "target_blind_protocol_frozen_execution_blocked_before_target_read"
    ):
        raise ValueError("stage-4 R2 blocked protocol status changed")
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    if freeze.get("execution_revision") != STAGE4_EXECUTION_REVISION:
        raise ValueError("stage-4 execution revision must be r2")
    if freeze.get("corrects_execution_revision") != "r1":
        raise ValueError("stage-4 R2 must identify the corrected r1 execution")
    if (
        freeze.get("execution_authorization_status")
        != "blocked_before_target_read_etas_comparator_not_evaluable"
        or freeze.get("formal_validation_scientific_runs_allowed") != 0
        or freeze.get(
            "future_new_execution_revision_maximum_formal_validation_scientific_runs_allowed"
        )
        != 1
        or freeze.get(
            "unblock_requires_separate_target_blind_etas_repair_protocol_and_new_execution_revision"
        )
        is not True
    ):
        raise ValueError("stage-4 R2 ETAS fail-closed execution status changed")
    if (
        _mapping(
            freeze.get("current_execution_authorizations"),
            label="freeze.current_execution_authorizations",
        )
        != _STAGE4_CURRENT_EXECUTION_AUTHORIZATIONS
    ):
        raise ValueError("stage-4 R2 current execution authorization matrix changed")
    if freeze.get("execution_revision_document") != "docs/anomaly_increment_protocol_r2.md":
        raise ValueError("stage-4 R2 execution-revision document path changed")
    if freeze.get("readiness_incident_document") != (
        "docs/phase4_scoring_readiness_incident_r0.md"
    ):
        raise ValueError("stage-4 R2 readiness-incident document path changed")
    if freeze.get("pre_score_tag") != STAGE4_PROTOCOL_TAG:
        raise ValueError("stage-4 R2 protocol tag changed")
    if freeze.get("results_tag") != STAGE4_RESULT_TAG:
        raise ValueError("stage-4 R2 results tag changed")

    scoring = _mapping(
        freeze.get("scoring_code_freeze"),
        label="freeze.scoring_code_freeze",
    )
    if scoring.get("current_r2_status") != "reserved_not_authorized_not_created":
        raise ValueError("stage-4 R2 reserved scoring-freeze status changed")
    if scoring.get("expected_tag") != STAGE4_SCORING_CODE_TAG:
        raise ValueError("stage-4 R2 scoring-code tag changed")
    for key, expected in _STAGE4_SCORING_FREEZE_PATHS:
        if scoring.get(key) != expected.as_posix():
            raise ValueError(f"stage-4 R2 scoring freeze path changed: {key}")
    logical_identity = _mapping(
        scoring.get("selected_table_logical_identity"),
        label="freeze.scoring_code_freeze.selected_table_logical_identity",
    )
    if logical_identity != _STAGE4_LOGICAL_IDENTITY_CONTRACT:
        raise ValueError("stage-4 R2 inherited logical identity contract changed")
    required_before_target_read = [
        "scoring_code_commit_pushed",
        "scoring_code_tag_pushed",
        "execution_seal_verified",
        "synthetic_end_to_end_passed",
        "cpu_float64_numerical_regression_passed",
        "restricted_spatial_artifact_hashes_verified",
        "restricted_local_artifact_access_control_verified",
        "identity_time_mapping_reproduces_accepted_snapshot_and_trajectory_columns",
        "identity_space_mapping_reproduces_accepted_200km_scientific_columns",
        "placebo_coverage_values_and_null_bitmap_exactly_unchanged",
        "fixed_small_permutation_matches_stage3_low_level_reference",
        "worker_count_invariance_passed",
        "logical_arrow_identity_r1_verified",
        "frozen_full_non_target_junit_count_matches_actual_scoring_freeze_suite",
        "formal_attempt_count_equals_zero",
        "target_read_count_equals_zero",
    ]
    if scoring.get("required_before_target_read") != required_before_target_read:
        raise ValueError("stage-4 R2 pre-target scoring requirements changed")
    if scoring.get("test_count_binding") != {
        "freeze_time": "after_scoring_implementation_and_all_non_target_tests_are_final",
        "expected_test_count": "UNFROZEN",
        "actual_junit_count_must_equal_frozen_count": True,
        "protocol_acceptance_snapshot_count_may_not_be_reused": True,
    }:
        raise ValueError("stage-4 R2 final JUnit-count binding changed")

    retirement = _mapping(
        freeze.get("r1_retirement"),
        label="freeze.r1_retirement",
    )
    if retirement.get("protocol_design_sha256") != (
        "c15d3bbca5cef4b363a79e183d715124256a12088873d81cd77de489766b32de"
    ):
        raise ValueError("stage-4 R1 retirement protocol identity changed")
    if retirement.get("scoring_seal_file_sha256") != (
        "a6e8dc9ac283813edb62e301114d4985ae332b9c607584c987a4297efe5978f3"
    ):
        raise ValueError("stage-4 R1 retirement seal identity changed")
    for ledger_name, expected_file, expected_content in (
        (
            "formal_attempt_ledger",
            "9ac5e5e080c1d5425f985cb3091b94c0da69d211469589d26ae8bfc314088142",
            "cadc80e5a0f00ffce241f910409750b01e3f410d910dda5d5aad0ff3033d2448",
        ),
        (
            "target_read_ledger",
            "0a49450cc1006ccd0ced26fba30330417f1ec8667c5cedf0bf04242f158210c8",
            "4c1fb843edfa8f59f37f137d8d68962cbdae5991cc8809302d39d240f0b395b6",
        ),
    ):
        ledger = _mapping(
            retirement.get(ledger_name),
            label=f"freeze.r1_retirement.{ledger_name}",
        )
        if (
            ledger.get("file_sha256") != expected_file
            or ledger.get("content_sha256") != expected_content
            or ledger.get("operation_count") != 0
        ):
            raise ValueError(f"stage-4 R1 retirement ledger changed: {ledger_name}")
    if retirement.get("target_bytes_observed") is not False:
        raise ValueError("stage-4 R1 retirement target-read status changed")
    if retirement.get("reusable_for_r2_authorization") is not False:
        raise ValueError("stage-4 R1 artifacts may not authorize R2")

    background = _mapping(protocol.get("background"), label="background")
    if _mapping(background.get("etas"), label="background.etas") != {
        "observed_comparator_status": "not_evaluable",
        "required_status_for_G2_and_any_anomaly_adoption": "evaluable",
        "reason": "failed_numerical_stability_no_qualified_parameters",
        "numerical_retry_in_stage4": "forbidden_without_separate_pre_score_protocol",
        "substitution_claim_forbidden": True,
        "not_evaluable_action": "G2_evidence_insufficient_retain_background_only",
        "overrides_all_candidate_gate_and_adoption_rules": True,
        "stage4_target_read_authorized_while_not_evaluable": False,
        "repair_must_be_target_blind_and_complete_before_new_execution_revision": True,
    }:
        raise ValueError("stage-4 R2 ETAS comparator prerequisite changed")

    inputs = _mapping(protocol.get("inputs"), label="inputs")
    target = _mapping(inputs.get("earthquake_target"), label="inputs.earthquake_target")
    coverage = _mapping(
        target.get("frozen_catalog_coverage"),
        label="inputs.earthquake_target.frozen_catalog_coverage",
    )
    expected_coverage = {
        "basis_document": "docs/data_quality_report.md",
        "basis_document_sha256": (
            "f4bf6633ce433b2c8d85d9d6d36cecd4c6824889f80e4528eeaae4de055ee9de"
        ),
        "observed_origin_time_max_utc": "2026-07-09T04:25:56Z",
        "observed_available_at_max_utc": "2026-07-09T04:25:56Z",
        "frozen_validation_window_end_max_utc": "2025-07-18T16:00:00Z",
        "all_frozen_validation_window_endpoints_must_be_lte_both_catalog_maxima": True,
        "verify_after_authorized_target_open_before_first_score": True,
        "missing_or_short_coverage_action": (
            "fail_closed_register_invalid_attempt_and_do_not_score"
        ),
        "available_at_equals_origin_time_is_optimistic_timeliness_assumption": True,
    }
    if coverage != expected_coverage:
        raise ValueError("stage-4 R2 earthquake-catalog coverage contract changed")

    evaluation = _mapping(protocol.get("evaluation"), label="evaluation")
    if evaluation.get("machine_status_vocabularies") != {
        "scientific_gate_status": {
            "allowed": ["passed", "failed", "evidence_insufficient", "not_reached"],
            "legacy_fail_status_forbidden": True,
        },
        "comparator_evaluability": {
            "allowed": ["evaluable", "not_evaluable"],
        },
        "execution_artifact_state": {
            "allowed": ["not_authorized_not_computed", "not_authorized_not_created"],
            "distinct_from_scientific_gate_status": True,
        },
    }:
        raise ValueError("stage-4 R2 machine-status vocabularies changed")
    if _mapping(
        evaluation.get("information_gain"),
        label="evaluation.information_gain",
    ) != {
        "formula": (
            "point_process_log_likelihood_difference_including_integrated_compensator_"
            "divided_by_unique_physical_events"
        ),
        "unit": "nats_per_physical_event",
        "candidate_variant": "dynamic",
        "comparator_variant": "etas_background_no_increment",
    }:
        raise ValueError("stage-4 R2 primary direct-ETAS information-gain contract changed")
    permutations = _mapping(evaluation.get("permutations"), label="evaluation.permutations")
    expected_requests = [
        {"kind": "time", "model_variant": "dynamic"},
        {"kind": "space", "model_variant": "dynamic"},
        {"kind": "time", "model_variant": "snapshot"},
        {"kind": "space", "model_variant": "snapshot"},
    ]
    if permutations.get("formal_requests") != expected_requests:
        raise ValueError("stage-4 R2 requires exactly four formal placebo requests")
    if permutations.get("formal_checkpoint_request_identities") != [
        "time-dynamic",
        "space-dynamic",
        "time-snapshot",
        "space-snapshot",
    ]:
        raise ValueError("stage-4 R2 formal checkpoint identities changed")
    if (
        permutations.get("primary_statistic_comparator_variant") != "etas_background_no_increment"
        or permutations.get(
            "every_observed_and_null_result_object_binds_etas_artifact_parameter_"
            "and_qualification_sha256"
        )
        is not True
        or permutations.get("checkpoint_identity_pattern") != "kind-model_variant"
        or permutations.get("exact_request_set_required") is not True
        or permutations.get("mappings_paired_across_dynamic_and_snapshot") is not True
    ):
        raise ValueError("stage-4 R2 paired placebo request contract changed")
    multiple_comparisons = _mapping(
        evaluation.get("multiple_comparisons"),
        label="evaluation.multiple_comparisons",
    )
    if multiple_comparisons != {
        "exploratory_method": "holm",
        "alpha": 0.05,
        "exploratory_holm_families": {
            "family_definition": "candidate_variant_x_placebo_kind",
            "exact_families": [
                {
                    "family_id": "dynamic_time",
                    "candidate_variant": "dynamic",
                    "placebo_kind": "time",
                },
                {
                    "family_id": "dynamic_space",
                    "candidate_variant": "dynamic",
                    "placebo_kind": "space",
                },
                {
                    "family_id": "snapshot_time",
                    "candidate_variant": "snapshot",
                    "placebo_kind": "time",
                },
                {
                    "family_id": "snapshot_space",
                    "candidate_variant": "snapshot",
                    "placebo_kind": "space",
                },
            ],
            "magnitude_bins": ["M5_6", "M6_plus"],
            "horizons_days": [7, 30, 90, 180, 365],
            "member_key": ["magnitude_bin", "horizon_days"],
            "complete_cartesian_product_required": True,
            "exact_member_count_per_family": 10,
        },
        "G2_primary_macro_endpoint_in_holm_family": False,
        "confirmatory_gatekeeping": {
            "etas_comparator_prerequisite": {
                "required_for_any_G2_gate_reached_or_anomaly_adoption": True,
                "current_status": "not_evaluable",
                "required_status": "evaluable",
                "current_action": (
                    "all_candidate_qualification_not_reached_and_stage4_target_read_forbidden"
                ),
                "target_scores_may_not_resolve_missing_comparator": True,
            },
            "current_r2_execution_state": {
                "formal_validation_scientific_runs_authorized": 0,
                "dynamic_qualification_gate": {
                    "gate_reached": False,
                    "gate_reached_reason": "etas_comparator_prerequisite_not_evaluable",
                    "qualification_status": "not_reached",
                },
                "snapshot_qualification_gate": {
                    "gate_reached": False,
                    "gate_reached_reason": "etas_comparator_prerequisite_not_evaluable",
                    "qualification_status": "not_reached",
                },
                "all_four_placebo_computations_state": "not_authorized_not_computed",
                "all_four_placebo_result_objects_state": "not_authorized_not_created",
                "candidate_gate_record_state": "not_authorized_not_created",
            },
            "future_post_etas_preregistered_design": {
                "applies_only_if_etas_comparator_status": "evaluable",
                "requires_new_execution_revision": True,
                "direct_candidate_minus_frozen_etas_track_required_for_every_candidate": True,
                "direct_etas_track_must_pass_before_any_anomaly_adoption": True,
                "etas_evaluable_status_alone_never_satisfies_G2": True,
                "comparator_contract": {
                    "mandatory_primary_scientific_comparator": {
                        "variant_id": "etas_background_no_increment",
                        "status_required": "evaluable",
                        "frozen_before_target_read": True,
                        "required_bindings": [
                            "etas_artifact_sha256",
                            "etas_parameter_snapshot_sha256",
                            "etas_numerical_qualification_evidence_sha256",
                        ],
                    },
                    "secondary_required_background_comparator": {
                        "variant_id": "kde_background_no_increment",
                        "frozen_input_binding": "inputs.background_model_payload.sha256",
                        "frozen_artifact_sha256": (
                            "63db433a0ea490de2674cb84c485f8efd8215374d70f31f22810da29f0d7b142"
                        ),
                        "never_substitute_etas": True,
                    },
                    "required_reporting_confound_comparator": {
                        "variant_id": "coverage_only",
                        "never_substitute_etas": True,
                    },
                    (
                        "all_three_comparator_tracks_required_for_development_and_"
                        "regional_qualification"
                    ): True,
                },
                "primary_candidate": "dynamic",
                "snapshot_role": (
                    "conditional_fallback_only_after_dynamic_G2_pass_and_G3_not_pass"
                ),
                "snapshot_independent_rescue_when_dynamic_G2_not_pass": "forbidden",
                "dynamic_and_snapshot_are_not_parallel_alpha_entries": True,
                (
                    "all_four_observed_raw_statistics_and_full_null_distributions_always_computed"
                ): True,
                "all_four_placebo_result_objects_always_bound_to_result_identity": True,
                "raw_computation_does_not_imply_gate_reached": True,
                "time_and_space_placebo_rule": "intersection_union_both_must_pass",
                "candidate_joint_placebo_p_for_reporting": "max_time_and_space_p",
                "practical_any_of_branches_were_preregistered_before_scores": True,
                "practical_branch_each_uses_two_sided_95pct_lower_bound_gt_zero": True,
                "practical_branch_union_one_sided_alpha_upper_bound": 0.05,
                "dynamic_qualification_gate": {
                    "gate_reached": True,
                    "gate_reached_reason": "primary_confirmatory_candidate",
                },
                "snapshot_qualification_gate": {
                    "gate_reached_true_iff": {
                        "dynamic_full_qualification_status": "passed",
                        "G3_status_in": ["failed", "evidence_insufficient"],
                    },
                    "reached_reason_codes": [
                        "dynamic_full_qualification_passed_and_G3_failed",
                        "dynamic_full_qualification_passed_and_G3_evidence_insufficient",
                    ],
                    "otherwise": {
                        "gate_reached": False,
                        "qualification_status": "not_reached",
                        "raw_statistics_role": "diagnostic_only",
                        "confirmatory_conclusion_forbidden": True,
                    },
                    "not_reached_reason_codes": [
                        "dynamic_full_qualification_failed",
                        "dynamic_full_qualification_evidence_insufficient",
                        "dynamic_full_qualification_not_reached",
                        "dynamic_full_qualification_passed_and_G3_passed",
                        "dynamic_full_qualification_passed_and_G3_not_reached",
                    ],
                },
                "gate_record_identity_requires": [
                    "gate_reached",
                    "gate_reached_reason",
                    "gate_record_sha256",
                ],
                "gate_record_sha256_payload_requires": [
                    "candidate_variant",
                    "gate_reached",
                    "gate_reached_reason",
                    "qualification_status",
                    "etas_artifact_sha256",
                    "etas_parameter_snapshot_sha256",
                    "etas_numerical_qualification_evidence_sha256",
                    "formal_G2_result_object_sha256",
                    "development_increment_stability_result_sha256",
                    "regional_stability_result_sha256",
                    "time_placebo_result_object_sha256",
                    "space_placebo_result_object_sha256",
                ],
            },
        },
    }:
        raise ValueError("stage-4 R2 multiple-comparisons contract changed")
    gates = _mapping(evaluation.get("gates"), label="evaluation.gates")
    g2 = _mapping(gates.get("G2"), label="evaluation.gates.G2")
    if (
        g2.get("candidate_variant") != "dynamic"
        or g2.get("snapshot_equivalent_candidate_variant") != "snapshot"
        or g2.get("comparator_variant") != "etas_background_no_increment"
        or g2.get("candidate_minus_etas_macro_information_gain_lower_95pct_bound_gt") != 0
        or g2.get("etas_evaluable_status_alone_never_satisfies_G2") is not True
        or g2.get("direct_candidate_minus_etas_G2_result_required_for_each_evaluated_variant")
        is not True
        or g2.get("evaluated_model_variants") != ["dynamic", "snapshot"]
        or g2.get("required_primary_placebos_by_variant")
        != {"dynamic": ["time", "space"], "snapshot": ["time", "space"]}
        or g2.get("primary_time_permutation_p_lte") != 0.05
        or g2.get("primary_space_permutation_p_lte") != 0.05
        or g2.get("both_primary_p_values_required_for_each_evaluated_variant") is not True
        or g2.get("same_practical_improvement_thresholds_apply_per_evaluated_variant") is not True
        or g2.get("reporting_confound_guard_each_evaluated_candidate_gt_coverage_only") is not True
        or g2.get("reporting_confound_guard_applies_independently_to_variants")
        != ["dynamic", "snapshot"]
        or g2.get("candidate_minus_coverage_only_macro_information_gain_lower_95pct_bound_gt") != 0
        or g2.get("candidate_qualification_requires")
        != [
            "formal_validation_G2_core",
            "direct_candidate_minus_frozen_etas_G2_track",
            "development_increment_stability",
            "regional_stability",
        ]
        or g2.get("candidate_qualification_components_must_all_pass") is not True
        or g2.get("candidate_qualification_nonpass_statuses") != ["failed", "evidence_insufficient"]
        or g2.get("dynamic_is_only_confirmatory_H1_G2_entry") is not True
        or g2.get("snapshot_is_conditional_G3_fallback_only") is not True
    ):
        raise ValueError("stage-4 R2 dynamic/snapshot G2 contract changed")
    if g2.get("practical_improvement_any_of") != [
        {
            "metric": "same_area_strict_recall_gain_percentage_points",
            "evaluation_partition": "formal_validation_once",
            "candidate_variant": "current_evaluated_candidate_variant",
            "comparator_variant": "etas_background_no_increment",
            "magnitude_bin": "M5_6",
            "horizons_days": [7, 30, 90],
            "macro_weighting": "equal_horizon",
            "threshold_gte": 5,
            "lower_95pct_bound_gt": 0,
            "area_budget_km2": 600000,
            "per_issue_selection": (
                "largest_complete_deterministic_cell_prefix_with_exact_area_lte_budget"
            ),
            "denominator": "all_study_area_targets_unsupported_count_as_misses",
        },
        {
            "metric": "same_recall_union_area_relative_reduction",
            "evaluation_partition": "formal_validation_once",
            "candidate_variant": "current_evaluated_candidate_variant",
            "comparator_variant": "etas_background_no_increment",
            "magnitude_bin": "M5_6",
            "horizons_days": [7, 30, 90],
            "macro_weighting": "equal_horizon",
            "reference_hits_per_horizon": ("etas_background_strict_hit_count_at_600000km2"),
            "candidate_search": "frozen_625km2_budget_grid_no_interpolation",
            "candidate_budget_per_horizon": "minimum_budget_reaching_reference_hit_count",
            "area_value": "exact_selected_complete_cell_prefix_area",
            "exposure_area_aggregation": (
                "equal_weight_mean_exact_selected_area_across_nonoverlapping_exposures_within_horizon"
            ),
            "per_horizon_reduction": (
                "one_minus_candidate_mean_exact_area_div_background_mean_exact_area"
            ),
            "macro_reduction": "equal_weight_mean_of_three_per_horizon_reductions",
            (
                "bootstrap_area_values_fixed_event_weights_only_"
                "change_hit_counts_and_selected_budget"
            ): True,
            "zero_reference_hit_action": "branch_not_evaluable_and_not_pass",
            "unreachable_at_960000km2_action": "branch_not_evaluable_and_not_pass",
            "bootstrap_invalid_branch_value_action": (
                "retain_numeric_reduction_0.0_as_failure_not_drop"
            ),
            "threshold_gte": 0.10,
            "lower_95pct_bound_gt": 0,
        },
    ]:
        raise ValueError("stage-4 R2 practical-improvement branches changed")

    development_stability = _mapping(
        gates.get("development_increment_stability"),
        label="evaluation.gates.development_increment_stability",
    )
    if development_stability != {
        "role": "required_candidate_qualification_component",
        "evaluation_partition": ("three_frozen_mutually_disjoint_development_rolling_folds"),
        "evaluated_model_variants": ["dynamic", "snapshot"],
        "model_variant_roles": {
            "dynamic": "confirmatory_H1_G2_candidate",
            "snapshot": "conditional_G3_fallback_only",
        },
        "comparators": [
            "etas_background_no_increment",
            "kde_background_no_increment",
            "coverage_only",
        ],
        "primary_magnitude_bin": "M5_6",
        "horizons_days": [7, 30, 90],
        "macro_weighting": "equal_horizon",
        "per_horizon_statistic": ("candidate_minus_comparator_information_gain_nats_per_event"),
        "fold_statistic": "equal_weight_mean_of_all_three_horizon_statistics",
        "outer_fold_count_required": 3,
        "minimum_positive_folds_per_variant_and_comparator": 2,
        "median_fold_macro_information_gain_gt_per_variant_and_comparator": 0,
        "every_comparator_track_must_pass": True,
        "any_fold_horizon_with_zero_scored_events_action": (
            "evidence_insufficient_no_partial_macro"
        ),
        "target_bands_must_be_mutually_disjoint": True,
        "one_model_fit_per_variant_per_fold_shared_across_horizons": True,
        "same_background_coverage_preprocessor_penalty_and_exposures_required": True,
        "result_identity": {
            "exact_horizon_row_key": [
                "development_fold_id",
                "model_variant",
                "comparator_variant",
                "horizon_days",
            ],
            "required_horizon_row_values": [
                "information_gain_nats_per_event",
                "supported_unique_physical_event_count",
            ],
            "complete_fold_variant_comparator_horizon_cartesian_product_required": True,
            "expected_horizon_row_count": 54,
            "exact_fold_macro_row_key": [
                "development_fold_id",
                "model_variant",
                "comparator_variant",
            ],
            "required_fold_macro_row_values": [
                "equal_weight_macro_information_gain_nats_per_event",
                "positive_horizon_count",
                "status",
            ],
            "expected_fold_macro_row_count": 18,
            "exact_candidate_comparator_summary_key": [
                "model_variant",
                "comparator_variant",
            ],
            "required_candidate_comparator_summary_values": [
                "positive_fold_count",
                "median_fold_macro_information_gain_nats_per_event",
                "status",
            ],
            "expected_candidate_comparator_summary_count": 6,
            "allowed_statuses": ["passed", "failed", "evidence_insufficient"],
            "missing_row_or_value_action": "evidence_insufficient",
        },
        "failure_action": ("candidate_not_G2_qualified_and_publish_no_cross_split_stability"),
        "insufficient_action": ("candidate_not_G2_qualified_evidence_insufficient_no_random_split"),
    }:
        raise ValueError("stage-4 R2 development increment stability contract changed")

    adoption = _mapping(
        evaluation.get("adoption_matrix"),
        label="evaluation.adoption_matrix",
    )
    if adoption != {
        "etas_comparator_not_evaluable": (
            "retain_background_only_and_block_stage4_target_read_until_new_execution_revision"
        ),
        "etas_not_evaluable_rule_overrides_all_rows_below": True,
        "dynamic_G2_pass_and_G3_pass": "adopt_dynamic_for_stage5_comparison",
        "dynamic_G2_pass_and_G3_not_pass_and_snapshot_equivalent_G2_pass": ("adopt_snapshot_only"),
        "dynamic_G2_pass_and_G3_not_pass_and_snapshot_equivalent_G2_not_pass": (
            "retain_background_only"
        ),
        "dynamic_G2_not_pass": ("retain_background_only_and_stop_complex_anomaly_models"),
        "etas_evaluable_but_direct_candidate_minus_etas_not_pass": (
            "retain_best_frozen_non_anomaly_background_and_stop_anomaly_adoption"
        ),
        "direct_etas_track_pass_required_for_any_anomaly_adoption": True,
        (
            "dynamic_G2_pass_means_direct_etas_and_all_candidate_qualification_components_passed"
        ): True,
        (
            "snapshot_equivalent_G2_pass_means_direct_etas_and_all_candidate_"
            "qualification_components_passed"
        ): True,
        "G2_not_pass_statuses": ["failed", "evidence_insufficient"],
        "G3_not_pass_statuses": ["failed", "evidence_insufficient"],
        "snapshot_evaluated_for_adoption_only_after_dynamic_G2_pass_and_G3_not_pass": True,
        "snapshot_independent_rescue_when_dynamic_G2_not_pass": "forbidden",
        "kde_background_no_increment": ("secondary_background_diagnostic_never_an_etas_substitute"),
        "coverage_only": "reporting_confound_diagnostic_never_an_anomaly_adoption",
        "snapshot_equivalent_G2_uses_same_thresholds_placebos_and_practical_metric": True,
        "no_post_score_fallback_choice": True,
    }:
        raise ValueError("stage-4 R2 fixed-sequence adoption contract changed")

    regional_stability = _mapping(
        evaluation.get("regional_stability"),
        label="evaluation.regional_stability",
    )
    if regional_stability != {
        "role": "required_candidate_qualification_component_and_regional_diagnostic",
        "strata": "construction_subzones_from_score_blind_linework_topology",
        "fixed_nonempty_query_grid_zone_count": 39,
        "fixed_region_order": "construction_zone_id_ascending",
        "target_may_not_create_merge_split_or_reorder_regions": True,
        "evaluation_partition": "formal_validation_once",
        "evaluated_model_variants": ["dynamic", "snapshot"],
        "model_variant_roles": {
            "dynamic": "confirmatory_H1_G2_candidate",
            "snapshot": "conditional_G3_fallback_only",
        },
        "comparators": [
            "etas_background_no_increment",
            "kde_background_no_increment",
            "coverage_only",
        ],
        "primary_magnitude_bin": "M5_6",
        "horizons_days": [7, 30, 90],
        "strict_recall_area_budget_km2": 600000,
        "macro_weighting": "equal_horizon",
        "per_region_horizon_contribution_formula": (
            "(region_event_log_intensity_difference-"
            "region_integrated_compensator_difference)/"
            "global_supported_unique_event_count_for_horizon"
        ),
        "zero_global_supported_event_action": "evidence_insufficient",
        "per_region_macro_contribution": ("equal_weight_mean_of_three_horizon_contributions"),
        "sum_region_macro_contributions_must_equal_global_macro_information_gain": True,
        "equality_absolute_tolerance": 1.0e-12,
        "strongest_region_selection": (
            "maximum_macro_contribution_then_construction_zone_id_ascending"
        ),
        "leave_strongest_region_out_residual_formula": (
            "global_macro_information_gain-maximum_region_macro_contribution"
        ),
        "bootstrap": {
            "reuse_joint_physical_event_bootstrap": True,
            "replications": 2000,
            "confidence_level": 0.95,
            "interval_method": "percentile",
            "integrated_compensator_fixed": True,
            "strongest_region_reselected_each_replication": True,
            "same_event_weights_shared_across_variants_comparators_horizons_and_regions": (True),
        },
        "pass_requirements_per_variant_and_comparator": {
            "observed_leave_strongest_region_out_residual_gt": 0,
            "lower_95pct_bound_leave_strongest_region_out_residual_gt": 0,
            "minimum_positive_event_bearing_regions": 2,
            "positive_event_bearing_region_definition": (
                "supported_unique_event_count_gte_1_and_macro_contribution_gt_0"
            ),
        },
        "every_comparator_track_must_pass": True,
        "missing_or_incomplete_region_mapping_action": "evidence_insufficient",
        "fewer_than_two_evaluable_event_regions_action": "evidence_insufficient",
        "required_by_horizon": [
            "information_gain",
            "independent_event_count",
            "strict_recall",
        ],
        "failure_action": ("candidate_not_G2_qualified_and_publish_no_cross_region_stability"),
        "insufficient_action": "candidate_not_G2_qualified_evidence_insufficient",
        "improvement_in_only_one_region_action": (
            "candidate_not_G2_qualified_and_no_stable_increment_claim"
        ),
        "strict_recall_diagnostic": {
            "denominator_field": "region_all_study_area_target_count",
            "positive_denominator_rule": (
                "candidate_or_comparator_strict_recall_equals_its_hit_count_divided_by_denominator"
            ),
            "zero_denominator_rule": {
                "candidate_strict_hit_count_required": 0,
                "comparator_strict_hit_count_required": 0,
                "candidate_strict_recall_required": None,
                "comparator_strict_recall_required": None,
                "strict_recall_evaluability_required": ("not_evaluable_no_region_targets"),
            },
            "positive_denominator_evaluability_required": "evaluable",
            "zero_denominator_is_diagnostic_not_track_evidence_insufficient": True,
        },
        "result_identity": {
            "exact_region_id_order_source": (
                "verified_local_restricted_zone_geometry_construction_zone_id_ascending"
            ),
            "exact_horizon_row_key": [
                "model_variant",
                "comparator_variant",
                "construction_zone_id",
                "horizon_days",
            ],
            "required_horizon_row_values": [
                "information_gain_contribution_nats_per_global_supported_event",
                "supported_unique_physical_event_count",
                "global_supported_unique_physical_event_count",
                "region_event_log_intensity_difference",
                "region_integrated_compensator_difference",
                "region_all_study_area_target_count",
                "global_all_study_area_target_count",
                "candidate_strict_hit_count",
                "comparator_strict_hit_count",
                "candidate_strict_recall",
                "comparator_strict_recall",
                "strict_recall_evaluability",
            ],
            (
                "global_supported_unique_physical_event_count_shared_across_regions_"
                "variants_and_comparators_within_horizon"
            ): True,
            (
                "sum_region_supported_unique_physical_event_count_must_equal_global_"
                "supported_unique_physical_event_count_within_horizon"
            ): True,
            (
                "region_all_study_area_target_count_shared_between_candidate_and_"
                "comparator_within_region_horizon"
            ): True,
            (
                "global_all_study_area_target_count_shared_across_regions_and_tracks_within_horizon"
            ): True,
            (
                "sum_region_all_study_area_target_count_must_equal_global_all_study_"
                "area_target_count"
            ): True,
            "complete_variant_comparator_region_horizon_cartesian_product_required": True,
            "expected_horizon_row_count": 702,
            "exact_region_macro_row_key": [
                "model_variant",
                "comparator_variant",
                "construction_zone_id",
            ],
            "required_region_macro_row_values": [
                "equal_weight_macro_information_gain_contribution",
                "supported_unique_physical_event_count_union",
                "positive_event_bearing_region",
                "is_strongest_region",
            ],
            "supported_unique_physical_event_count_union_definition": (
                "exact_physical_event_id_union_across_7_30_90_day_horizons"
            ),
            "expected_region_macro_row_count": 234,
            "exact_track_summary_key": ["model_variant", "comparator_variant"],
            "required_track_summary_values": [
                "global_macro_information_gain",
                "sum_region_macro_contributions",
                "strongest_construction_zone_id",
                "strongest_region_macro_contribution",
                "leave_strongest_region_out_residual",
                "residual_lower_95pct",
                "residual_upper_95pct",
                "positive_event_bearing_region_count",
                "status",
            ],
            "expected_track_summary_count": 6,
            "exactly_one_is_strongest_region_true_per_track": True,
            "is_strongest_region_true_id_must_equal_track_summary_strongest_construction_zone_id": (
                True
            ),
            "strongest_region_tie_break_must_match": (
                "maximum_macro_contribution_then_construction_zone_id_ascending"
            ),
            "exact_bootstrap_row_key": [
                "model_variant",
                "comparator_variant",
                "replication_index",
            ],
            "required_bootstrap_row_values": [
                "global_macro_information_gain",
                "strongest_construction_zone_id",
                "strongest_region_macro_contribution",
                "leave_strongest_region_out_residual",
            ],
            "complete_track_replication_cartesian_product_required": True,
            "replication_index_domain": {
                "minimum_inclusive": 0,
                "maximum_inclusive": 1999,
                "exact_integer_sequence_per_track_required": True,
                "duplicate_or_extra_rows_forbidden": True,
            },
            "expected_bootstrap_row_count": 12000,
            "allowed_statuses": ["passed", "failed", "evidence_insufficient"],
            "missing_duplicate_or_nonfinite_row_action": "evidence_insufficient",
        },
    }:
        raise ValueError("stage-4 R2 regional stability contract changed")

    topology = _mapping(
        protocol.get("spatial_permutation_topology"),
        label="spatial_permutation_topology",
    )
    restricted_artifacts = _mapping(
        topology.get("local_restricted_artifacts"),
        label="spatial_permutation_topology.local_restricted_artifacts",
    )
    if restricted_artifacts.get("access_control") != {
        "schema_version": 3,
        "policy_id": "stage4_restricted_local_artifact_access_v3",
        "required_before_target_read": True,
        "receipt_bound_to_scoring_qualification_and_execution_seal": True,
        "retained_handle_verification": {
            "exact_path_count": 5,
            "directory_and_four_frozen_artifacts_required": True,
            "no_follow_open_required": True,
            "regular_single_link_files_and_non_reparse_paths_required": True,
            "directory_handle_retained_until_all_file_checks_complete": True,
            "all_four_file_hashes_computed_on_retained_handles": True,
            "entry_and_exit_owner_permission_samples_on_same_retained_handle_required": True,
            "entry_and_exit_canonical_permission_descriptor_bytes_must_match": True,
            "handle_and_path_identity_and_state_reverified_before_release": True,
        },
        "windows": {
            "descriptor_query": "GetSecurityInfo_on_retained_handle",
            "queried_by_handle_required": True,
            "allowed_principal_roles": [
                "current_process_user",
                "local_system",
                "builtin_administrators",
            ],
            "owner_must_be_in_allowed_principal_roles": True,
            "current_process_user_explicit_full_control_required": True,
            "inherit_only_ace_may_not_satisfy_current_object_full_control": True,
            "full_control_access_mask_hex": "0x001f01ff",
            "dacl_present_required": True,
            "dacl_protected_required": True,
            "dacl_defaulted_required": False,
            "inherited_ace_count_required": 0,
            "security_descriptor_revision_required": 1,
            "security_descriptor_control_protected_bit_hex": "0x1000",
            "deny_or_unknown_ace_forbidden": True,
            "ace_for_unauthorized_principal_forbidden": True,
        },
        "posix": {
            "platform_family_required": "linux",
            "queried_by_handle_required": True,
            "owner_must_equal_effective_uid": True,
            "owner_only_file_mode": "0600",
            "owner_only_directory_mode": "0700",
            "filesystem_locality_required": "local",
            "acl_model_required": "classic_mode_bits_without_acl_xattrs",
            "acl_related_xattrs_must_equal": [],
            "allowed_local_classic_filesystems": [
                {
                    "filesystem_type": "ext2_ext3_ext4",
                    "filesystem_magic_hex": "0x0000ef53",
                },
                {"filesystem_type": "tmpfs", "filesystem_magic_hex": "0x01021994"},
                {"filesystem_type": "xfs", "filesystem_magic_hex": "0x58465342"},
                {"filesystem_type": "btrfs", "filesystem_magic_hex": "0x9123683e"},
                {"filesystem_type": "f2fs", "filesystem_magic_hex": "0xf2f52010"},
            ],
        },
        "unsupported_or_unverifiable_platform_action": "fail_closed",
    }:
        raise ValueError("stage-4 R2 restricted-artifact access-control contract changed")

    compute = _mapping(protocol.get("compute"), label="compute")
    if (
        compute.get("max_workers") != 6
        or compute.get("logical_cpu_affinity_limit") != 6
        or compute.get("process_priority") != "below_normal"
        or compute.get("nested_parallelism") is not False
        or compute.get("blas_threads_per_worker") != 1
        or compute.get("blas_environment_must_be_set_before_numpy_or_scipy_import") is not True
        or compute.get("resource_control_receipt_required_before_target_read") is not True
    ):
        raise ValueError("stage-4 R2 resource-control contract changed")

    publication = _mapping(protocol.get("publication"), label="publication")
    isolation = _mapping(
        publication.get("spatial_output_isolation"),
        label="publication.spatial_output_isolation",
    )
    expected_forecast_files = [
        "outputs/visualizations/anomaly_increment_r2_forecast_spatial.svg",
        "outputs/visualizations/anomaly_increment_r2_forecast_spatial.html",
    ]
    expected_retrospective_files = [
        "outputs/visualizations/anomaly_increment_r2_retrospective_target_local.svg",
        "outputs/visualizations/anomaly_increment_r2_retrospective_target_local.html",
    ]
    if (
        isolation.get("physical_file_count") != 4
        or isolation.get("forecast_target_free_files") != expected_forecast_files
        or isolation.get("retrospective_target_bearing_local_restricted_files")
        != expected_retrospective_files
        or len(set(expected_forecast_files + expected_retrospective_files)) != 4
        or isolation.get("target_payload_in_forecast_files_forbidden") is not True
        or isolation.get("automatic_cross_file_target_loading_forbidden") is not True
        or isolation.get(
            "retrospective_target_bearing_files_require_restricted_access_before_bytes_written"
        )
        is not True
        or isolation.get("retrospective_access_control_receipt_required_before_publication")
        is not True
    ):
        raise ValueError("stage-4 R2 spatial-output isolation contract changed")
    retrospective_access = _mapping(
        isolation.get("retrospective_access_control"),
        label="publication.spatial_output_isolation.retrospective_access_control",
    )
    if retrospective_access != {
        "receipt_path": (
            "outputs/visualizations/anomaly_increment_r2_retrospective_acl_receipt.json"
        ),
        "restricted_parent_directory": "outputs/visualizations",
        "parent_directory_restricted_before_target_file_creation": True,
        "zero_byte_destination_created_before_target_bytes": True,
        "zero_byte_destination_acl_verified_before_target_bytes": True,
        "same_verified_handle_held_from_acl_check_through_final_write": True,
        "file_identity_before_and_after_write_must_match": True,
        "ordinary_temporary_file_may_not_receive_target_bytes_first": True,
        "atomic_replace_from_unrestricted_or_unverified_temporary_file_forbidden": True,
        "required_for_each_retrospective_file": True,
        "receipt_bindings": [
            "target_relative_path",
            "verified_zero_byte_file_identity",
            "final_file_identity",
            "final_file_sha256",
            "final_file_acl_descriptor_sha256",
            "restricted_parent_directory_acl_descriptor_sha256",
        ],
        "receipt_and_retrospective_files_forbidden_from_public_bundle": True,
    }:
        raise ValueError("stage-4 R2 retrospective access-control contract changed")
    public_validator = _mapping(
        isolation.get("public_forecast_artifact_validator"),
        label="publication.spatial_output_isolation.public_forecast_artifact_validator",
    )
    if public_validator != {
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
    }:
        raise ValueError("stage-4 R2 public forecast artifact validator changed")
    if _mapping(
        publication.get("result_identity_contract"),
        label="publication.result_identity_contract",
    ) != {
        "current_r2_status": "not_authorized_not_created",
        "applies_only_to_new_execution_revision_after_etas_evaluable": True,
        "future_required_objects": [
            "frozen_etas_artifact_parameter_and_numerical_qualification_receipt",
            "etas_event_term_and_integrated_compensator_score_object",
            "direct_candidate_minus_frozen_etas_G2_tracks",
            "dynamic_minus_etas_formal_G2_result_object",
            "snapshot_minus_etas_formal_G2_result_object",
            "dynamic_G2",
            "snapshot_equivalent_G2",
            "dynamic_development_increment_stability",
            "snapshot_development_increment_stability",
            "development_fold_variant_comparator_horizon_values_counts_macros_and_statuses",
            "dynamic_regional_stability",
            "snapshot_regional_stability",
            "regional_contribution_tables_and_residual_bootstrap_distributions",
            "time_dynamic_placebo_result_distribution",
            "space_dynamic_placebo_result_distribution",
            "time_snapshot_placebo_result_distribution",
            "space_snapshot_placebo_result_distribution",
            "dynamic_G3",
            "candidate_gatekeeping_sequence",
            "candidate_gate_reached_reason_and_gate_record_sha256",
            "candidate_gate_record_direct_etas_and_all_component_result_sha256_bindings",
            "adoption_decision",
            "adoption_decision_direct_etas_status_and_result_sha256",
            "adopted_variant_metrics_table",
        ],
    }:
        raise ValueError("stage-4 R2 result identity contract changed")
    if _mapping(
        publication.get("adopted_variant_mapping"),
        label="publication.adopted_variant_mapping",
    ) != {
        "background_only": {
            "variant_source": (
                "best_frozen_non_anomaly_background_selected_before_stage4_target_read"
            ),
            "allowed_variants": [
                "etas_background_no_increment",
                "kde_background_no_increment",
            ],
            "selection_evidence_sha256_required": True,
            "stage4_target_or_anomaly_result_may_not_affect_selection": True,
        },
        "snapshot": "snapshot",
        "dynamic": "dynamic",
    }:
        raise ValueError("stage-4 R2 adopted-variant mapping changed")
    rendering = _mapping(
        publication.get("spatial_rendering_contract"),
        label="publication.spatial_rendering_contract",
    )
    if rendering.get("center_point_fallback_warning_text") != (
        "中心点示意，非面积几何；报警面积以数值为准"  # noqa: RUF001
    ):
        raise ValueError("stage-4 R2 center-point warning changed")
    axes = _mapping(publication.get("plot_axis_contract"), label="publication.plot_axis_contract")
    if (
        axes.get("numeric_ticks_required") is not True
        or axes.get("molchan_x_domain") != [0.0, 1.0]
        or axes.get("fixed_area_x_domain_km2") != [0, 960000]
    ):
        raise ValueError("stage-4 R2 plot-axis contract changed")
    limitations = _mapping(publication.get("limitations"), label="publication.limitations")
    if limitations != {
        "earthquake_available_at_assumption": (
            "available_at_equals_origin_time_is_an_optimistic_timeliness_assumption"
        ),
        "bootstrap_interval_scope": (
            "conditional_on_fixed_fitted_model_and_excludes_refit_uncertainty"
        ),
        "etas_comparator_status": "not_evaluable",
        "formal_increment_claim_authorized": False,
        "allowed_formal_increment_claim": "none",
        "kde_comparison_role": "target_blind_diagnostic_design_only_not_a_formal_result",
        "incremental_value_over_etas_claim_forbidden": True,
        "etas_not_evaluable_blocks_G2_adoption_and_target_read": True,
    }:
        raise ValueError("stage-4 R2 publication limitations changed")
    display_semantics = _mapping(
        publication.get("display_semantics"),
        label="publication.display_semantics",
    )
    if display_semantics != {
        "etas_background_primary_comparator_option_required": True,
        "coverage_only_option_required": True,
        "kde_background_diagnostic_option_required": True,
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
            "formula": ("peak_integrated_grid_intensity/mean_integrated_grid_intensity"),
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
    }:
        raise ValueError("stage-4 R2 publication display semantics changed")
    if protocol_design_sha256(protocol) != STAGE4_R2_PROTOCOL_DESIGN_SHA256:
        raise ValueError("stage-4 R2 complete protocol design changed")


class Stage4R2ExecutionBlockedError(RuntimeError):
    """Raised when blocked R2 is asked to create or consume scoring state."""


def require_stage4_r2_execution_action(
    protocol: Mapping[str, object],
    *,
    action: str,
) -> None:
    """Fail closed for every R2 action that ETAS qualification currently blocks."""

    validate_stage4_r2_execution_contract(protocol)
    try:
        permission_key = _STAGE4_EXECUTION_ACTION_PERMISSION[action]
    except KeyError as exc:
        raise ValueError(f"unknown stage-4 R2 execution action: {action}") from exc
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    permissions = _mapping(
        freeze.get("current_execution_authorizations"),
        label="freeze.current_execution_authorizations",
    )
    if permissions.get(permission_key) is not True:
        raise Stage4R2ExecutionBlockedError(
            "stage-4 R2 is blocked before target read because the required frozen "
            f"ETAS comparator is not evaluable; action forbidden: {action}"
        )


def stage4_scoring_freeze_relative_path(
    protocol: Mapping[str, object],
    key: str,
) -> PurePosixPath:
    """Return one validated R2 scoring path from the frozen machine contract."""

    validate_stage4_r2_execution_contract(protocol)
    paths = dict(_STAGE4_SCORING_FREEZE_PATHS)
    try:
        expected = paths[key]
    except KeyError as exc:
        raise ValueError(f"unknown stage-4 R2 scoring freeze path: {key}") from exc
    scoring = _mapping(
        _mapping(protocol.get("freeze"), label="freeze").get("scoring_code_freeze"),
        label="freeze.scoring_code_freeze",
    )
    raw = _string(scoring.get(key), label=f"freeze.scoring_code_freeze.{key}")
    relative = PurePosixPath(raw)
    if relative != expected or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"stage-4 R2 scoring freeze path changed: {key}")
    return relative


@dataclass(frozen=True, slots=True)
class ExpectedTargetIdentity:
    """Unobserved target identity copied verbatim from the frozen protocol.

    The path is intentionally a string rather than a :class:`~pathlib.Path` so
    ordinary protocol inspection cannot accidentally probe the filesystem.
    """

    relative_path: str
    expected_file_sha256: str
    expected_content_sha256: str
    expected_schema_sha256: str
    contract_relative_path: str
    expected_contract_sha256: str
    physical_event_id_column: str

    def __post_init__(self) -> None:
        if (
            Path(self.relative_path).is_absolute()
            or Path(self.contract_relative_path).is_absolute()
        ):
            raise ValueError("stage-4 target identity paths must remain repository-relative")
        for label, value in (
            ("expected_file_sha256", self.expected_file_sha256),
            ("expected_content_sha256", self.expected_content_sha256),
            ("expected_schema_sha256", self.expected_schema_sha256),
            ("expected_contract_sha256", self.expected_contract_sha256),
        ):
            _sha256(value, label=label)
        if self.physical_event_id_column != "event_id":
            raise ValueError("stage-4 physical event identity column changed")

    def as_expected_metadata(self) -> JsonObject:
        """Return score-free metadata; this method performs no filesystem access."""

        return {
            "relative_path": self.relative_path,
            "expected_file_sha256": self.expected_file_sha256,
            "expected_content_sha256": self.expected_content_sha256,
            "expected_schema_sha256": self.expected_schema_sha256,
            "contract_relative_path": self.contract_relative_path,
            "expected_contract_sha256": self.expected_contract_sha256,
            "physical_event_id_column": self.physical_event_id_column,
            "observed": False,
        }


@dataclass(frozen=True, slots=True)
class FrozenManifest:
    """One public, content-addressed score-free manifest."""

    manifest_id: str
    relative_path: str
    expected_file_sha256: str
    document: MappingProxyType[str, object]

    @property
    def content_sha256(self) -> str:
        return _sha256(self.document.get("content_sha256"), label=f"{self.manifest_id}.content")


@dataclass(frozen=True, slots=True)
class Stage4ProtocolBundle:
    """Validated protocol plus four public score-free manifests."""

    repository_root: Path
    protocol_path: Path
    protocol: MappingProxyType[str, object]
    fold: FrozenManifest
    feature_set: FrozenManifest
    randomness: FrozenManifest
    spatial_strata: FrozenManifest
    expected_target: ExpectedTargetIdentity
    validation_receipt: MappingProxyType[str, object]

    @property
    def design_sha256(self) -> str:
        return protocol_design_sha256(self.protocol)

    @property
    def random_input_seal_sha256(self) -> str:
        return _sha256(
            self.validation_receipt.get("random_input_seal_sha256"),
            label="random_input_seal_sha256",
        )

    @property
    def manifests(self) -> tuple[FrozenManifest, ...]:
        return (self.fold, self.feature_set, self.randomness, self.spatial_strata)

    def score_free_identity(self) -> JsonObject:
        """Return the complete identity allowed before the first target read."""

        return {
            "protocol_design_sha256": self.design_sha256,
            "protocol_tag": STAGE4_PROTOCOL_TAG,
            "expected_scoring_code_tag": STAGE4_SCORING_CODE_TAG,
            "expected_result_tag": STAGE4_RESULT_TAG,
            "random_input_seal_sha256": self.random_input_seal_sha256,
            "manifest_file_sha256": {
                item.manifest_id: item.expected_file_sha256 for item in self.manifests
            },
            "manifest_content_sha256": {
                item.manifest_id: item.content_sha256 for item in self.manifests
            },
            "expected_target": self.expected_target.as_expected_metadata(),
            "formal_attempt_count": 0,
            "target_read_count": 0,
            "locked_test_run": False,
        }


def _expected_target_identity(protocol: JsonObject) -> ExpectedTargetIdentity:
    inputs = _mapping(protocol.get("inputs"), label="inputs")
    target = _mapping(inputs.get("earthquake_target"), label="inputs.earthquake_target")
    if target.get("unavailable_before_protocol_freeze") is not True:
        raise ValueError("target must remain unavailable before the stage-4 protocol freeze")
    if target.get("human_prediction_fields_forbidden") is not True:
        raise ValueError("human prediction fields must remain forbidden")
    return ExpectedTargetIdentity(
        relative_path=_string(target.get("path"), label="earthquake_target.path"),
        expected_file_sha256=_sha256(target.get("sha256"), label="earthquake_target.sha256"),
        expected_content_sha256=_sha256(
            target.get("content_sha256"), label="earthquake_target.content_sha256"
        ),
        expected_schema_sha256=_sha256(
            target.get("schema_sha256"), label="earthquake_target.schema_sha256"
        ),
        contract_relative_path=_string(
            target.get("contract_path"), label="earthquake_target.contract_path"
        ),
        expected_contract_sha256=_sha256(
            target.get("contract_sha256"), label="earthquake_target.contract_sha256"
        ),
        physical_event_id_column=_string(
            target.get("physical_event_id_column"),
            label="earthquake_target.physical_event_id_column",
        ),
    )


def _load_manifest(
    repository_root: Path,
    *,
    manifest_id: str,
    declaration: object,
) -> FrozenManifest:
    metadata = _mapping(declaration, label=f"generated_manifests.{manifest_id}")
    relative_path = _string(metadata.get("path"), label=f"{manifest_id}.path")
    if Path(relative_path).is_absolute():
        raise ValueError("generated manifest paths must remain repository-relative")
    expected = _sha256(metadata.get("sha256"), label=f"{manifest_id}.sha256")
    path = repository_root / relative_path
    require_existing_real_directory_tree(
        repository_root,
        path.parent,
        label=f"stage-4 {manifest_id} manifest directory",
    )
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(f"{manifest_id} file hash differs from the frozen protocol")
    document = _read_json_mapping(path)
    return FrozenManifest(
        manifest_id=manifest_id,
        relative_path=relative_path,
        expected_file_sha256=expected,
        document=MappingProxyType(document),
    )


def load_stage4_protocol_bundle(
    repository_root: Path | None = None,
    *,
    protocol_relative_path: Path = STAGE4_PROTOCOL_PATH,
) -> Stage4ProtocolBundle:
    """Load and cross-validate the frozen score-free bundle without probing targets."""

    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.resolve()
    )
    if protocol_relative_path.is_absolute():
        raise ValueError("the stage-4 protocol path must be repository-relative")
    if protocol_relative_path != STAGE4_PROTOCOL_PATH:
        raise ValueError("stage-4 execution must use the sole R2 protocol path")
    protocol_path = root / protocol_relative_path
    require_existing_real_directory_tree(
        root,
        protocol_path.parent,
        label="stage-4 protocol directory",
    )
    protocol = _read_yaml_mapping(protocol_path)
    validate_stage4_r2_execution_contract(protocol)

    generated = _mapping(protocol.get("generated_manifests"), label="generated_manifests")
    fold = _load_manifest(root, manifest_id="fold", declaration=generated.get("fold"))
    feature_set = _load_manifest(
        root,
        manifest_id="feature_set",
        declaration=generated.get("feature_set"),
    )
    randomness = _load_manifest(
        root,
        manifest_id="randomness",
        declaration=generated.get("randomness"),
    )
    spatial = _load_manifest(
        root,
        manifest_id="spatial_strata",
        declaration=generated.get("spatial_strata"),
    )
    receipt = validate_stage4_protocol_bundle(
        protocol,
        fold_manifest=fold.document,
        feature_manifest=feature_set.document,
        randomness_manifest=randomness.document,
        spatial_manifest=spatial.document,
    )
    if receipt.get("target_read_count") != 0:
        raise ValueError("score-free bundle unexpectedly reports a target read")
    return Stage4ProtocolBundle(
        repository_root=root,
        protocol_path=protocol_path,
        protocol=MappingProxyType(protocol),
        fold=fold,
        feature_set=feature_set,
        randomness=randomness,
        spatial_strata=spatial,
        expected_target=_expected_target_identity(protocol),
        validation_receipt=MappingProxyType(receipt),
    )


__all__ = [
    "STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH",
    "STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH",
    "STAGE4_EXECUTION_REVISION",
    "STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH",
    "STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH",
    "STAGE4_JUNIT_RELATIVE_PATH",
    "STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH",
    "STAGE4_PROTOCOL_PATH",
    "STAGE4_PROTOCOL_TAG",
    "STAGE4_QUALIFICATION_RELATIVE_PATH",
    "STAGE4_RESULT_TAG",
    "STAGE4_SCORING_CODE_TAG",
    "STAGE4_SCORING_SEAL_RELATIVE_PATH",
    "STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH",
    "ExpectedTargetIdentity",
    "FrozenManifest",
    "Stage4ProtocolBundle",
    "load_stage4_protocol_bundle",
    "stage4_scoring_freeze_relative_path",
    "validate_stage4_r2_execution_contract",
]
