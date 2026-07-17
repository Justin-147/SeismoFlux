from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from stage4_formal_preflight_fixture import make_formal_preflight_receipt

import seismoflux.anomaly_increment.attempt_ledger as attempt_ledger_module
import seismoflux.anomaly_increment.authorization as authorization_module
import seismoflux.anomaly_increment.config as config_module
import seismoflux.anomaly_increment.formal_preflight as formal_preflight_module
import seismoflux.anomaly_increment.immutable_file as immutable_file_module
import seismoflux.anomaly_increment.qualification as qualification_module
import seismoflux.anomaly_increment.target_access as target_access_module
from seismoflux.anomaly_increment.attempt_ledger import (
    STAGE4_TARGET_SCOPE,
    Stage4LedgerError,
    Stage4OperationAlreadyConsumedError,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _complete_stage4_operation_generic as complete_stage4_operation,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _initialize_stage4_ledger_generic as initialize_stage4_ledger,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _read_stage4_ledger_generic as read_stage4_ledger,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _recover_interrupted_stage4_operations_generic as recover_interrupted_stage4_operations,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _registered_stage4_attempt_generic as registered_stage4_attempt,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _reserve_stage4_operation_generic as reserve_stage4_operation,
)
from seismoflux.anomaly_increment.authorization import (
    STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
    STAGE4_FROZEN_PROTOCOL_PATHS,
    STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
    GitStage4RepositoryAdapter,
    PublicRepositoryEvidence,
    Stage4RepositoryEvidence,
    Stage4ScoringNotAuthorizedError,
    Stage4TargetAuthorization,
    authorize_stage4_target_access,
    build_stage4_scoring_seal,
    load_stage4_scoring_seal,
    stage4_execution_binding_id,
    write_stage4_scoring_seal_atomic,
)
from seismoflux.anomaly_increment.compute import BackendEquivalenceEvidence
from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_PREFLIGHT_RECEIPT_PATH,
    FormalPreflightReceipt,
)
from seismoflux.anomaly_increment.formal_preflight import (
    _load_formal_preflight_receipt_generic as load_formal_preflight_receipt,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
)
from seismoflux.anomaly_increment.preregistration import (
    protocol_design_sha256,
    with_content_sha256,
)
from seismoflux.anomaly_increment.qualification import (
    REQUIRED_TESTS_BY_QUALIFICATION,
    PytestRunEvidence,
    ScoreBlindInputEvidence,
    Stage4QualificationError,
    Stage4QualificationEvidence,
    build_stage4_qualification_evidence,
    expected_target_identity_from_protocol,
    parse_pytest_junit_evidence,
    validate_stage4_qualification_against_formal_preflight,
)
from seismoflux.anomaly_increment.qualification import (
    _load_stage4_qualification_evidence_generic as load_stage4_qualification_evidence,
)
from seismoflux.anomaly_increment.qualification import (
    _write_stage4_qualification_evidence_atomic_generic as write_qualification,
)
from seismoflux.anomaly_increment.restricted_access import (
    RESTRICTED_ARTIFACT_IDS,
    RestrictedAccessError,
    RestrictedLocalArtifactAccessControlEvidence,
    RestrictedPathAccessObservation,
    canonical_permission_descriptor,
    restricted_access_contract_sha256,
)
from seismoflux.anomaly_increment.target_access import (
    Stage4LockedTestForbiddenError,
    consume_authorized_stage4_target,
    forbid_stage4_locked_test_access,
    require_stage4_execution_scope,
)
from seismoflux.background.execution import CommandResult

PROTOCOL_TAG = "v0.3.1-anomaly-increment-protocol-r2"
SCORING_TAG = "v0.3.1-anomaly-increment-scoring-code-r2"
RESULT_TAG = "v0.3.1-anomaly-increment-r2"
CODE_COMMIT = "2" * 40
PROTOCOL_COMMIT = "1" * 40
PROTOCOL_TAG_OBJECT = "3" * 40
SCORING_TAG_OBJECT = "4" * 40
EVIDENCE_SHA256 = "a" * 64
BINDING_SHA256 = "b" * 64
SYNTHETIC_FROZEN_TEST_COUNT = 128


@pytest.fixture(autouse=True)
def _synthetic_frozen_test_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qualification_module,
        "FROZEN_FULL_NON_TARGET_TEST_COUNT",
        SYNTHETIC_FROZEN_TEST_COUNT,
    )

    # These tests exercise the pre-registered future execution machinery in
    # isolation. Production entry points remain blocked by the real R2 protocol;
    # dedicated fail-closed tests cover that boundary without this fixture.
    def allow_future_execution(*args: object, **kwargs: object) -> None:
        return None

    def synthetic_scoring_freeze_relative_path(
        protocol: dict[str, Any],
        key: str,
    ) -> PurePosixPath:
        freeze = cast(dict[str, Any], protocol["freeze"])
        scoring = cast(dict[str, Any], freeze["scoring_code_freeze"])
        raw = scoring[key]
        assert isinstance(raw, str)
        return PurePosixPath(raw)

    monkeypatch.setattr(
        attempt_ledger_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        authorization_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        target_access_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        qualification_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        formal_preflight_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        seal_script,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        qualification_script,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        preflight_script,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    # Synthetic target identities necessarily differ from the immutable real R2
    # design hash.  Only these downstream unit tests replace the strict path
    # validator; production modules retain the real validator and guard.
    monkeypatch.setattr(
        authorization_module,
        "stage4_scoring_freeze_relative_path",
        synthetic_scoring_freeze_relative_path,
    )
    monkeypatch.setattr(
        seal_script,
        "stage4_scoring_freeze_relative_path",
        synthetic_scoring_freeze_relative_path,
    )
    monkeypatch.setattr(
        qualification_script,
        "stage4_scoring_freeze_relative_path",
        synthetic_scoring_freeze_relative_path,
    )


def _load_seal_script() -> ModuleType:
    path = Path("scripts/build_stage4_scoring_seal.py").resolve()
    spec = importlib.util.spec_from_file_location("stage4_scoring_seal_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stage-4 scoring seal script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qualification_script() -> ModuleType:
    path = Path("scripts/build_stage4_scoring_qualification.py").resolve()
    spec = importlib.util.spec_from_file_location("stage4_scoring_qualification_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stage-4 scoring qualification script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_preflight_script() -> ModuleType:
    path = Path("scripts/build_stage4_formal_preflight.py").resolve()
    spec = importlib.util.spec_from_file_location("stage4_formal_preflight_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stage-4 formal preflight script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


seal_script = _load_seal_script()
qualification_script = _load_qualification_script()
preflight_script = _load_preflight_script()


def _protocol(
    target_sha256: str,
    target_path: str = "synthetic/target.bin",
    *,
    gpu_backend_frozen: bool = False,
) -> dict[str, Any]:
    return {
        "protocol_version": "0.4.1",
        "frozen_on": "2026-07-17",
        "generated_manifests": {},
        "freeze": {
            "execution_revision": "r2",
            "corrects_execution_revision": "r1",
            "execution_revision_document": "docs/anomaly_increment_protocol_r2.md",
            "readiness_incident_document": ("docs/phase4_scoring_readiness_incident_r0.md"),
            "r1_retirement": {
                "reason": (
                    "scientific_gate_and_publication_contract_corrected_before_any_target_read"
                ),
                "protocol_design_sha256": (
                    "c15d3bbca5cef4b363a79e183d715124256a12088873d81cd77de489766b32de"
                ),
                "scoring_seal_file_sha256": (
                    "a6e8dc9ac283813edb62e301114d4985ae332b9c607584c987a4297efe5978f3"
                ),
                "formal_attempt_ledger": {
                    "file_sha256": (
                        "9ac5e5e080c1d5425f985cb3091b94c0da69d211469589d26ae8bfc314088142"
                    ),
                    "content_sha256": (
                        "cadc80e5a0f00ffce241f910409750b01e3f410d910dda5d5aad0ff3033d2448"
                    ),
                    "operation_count": 0,
                },
                "target_read_ledger": {
                    "file_sha256": (
                        "0a49450cc1006ccd0ced26fba30330417f1ec8667c5cedf0bf04242f158210c8"
                    ),
                    "content_sha256": (
                        "4c1fb843edfa8f59f37f137d8d68962cbdae5991cc8809302d39d240f0b395b6"
                    ),
                    "operation_count": 0,
                },
                "target_bytes_observed": False,
                "reusable_for_r2_authorization": False,
            },
            "pre_score_tag": PROTOCOL_TAG,
            "results_tag": RESULT_TAG,
            "protocol_tag_authorizes_only_score_free_implementation": True,
            "required_before_any_target_read_or_score": [
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
            ],
            "scoring_code_freeze": {
                "expected_tag": SCORING_TAG,
                "required_seal_path": ("data/manifests/anomaly_increment_r2_scoring_seal.json"),
                "formal_preflight_receipt_path": (
                    "data/interim/stage4/anomaly_increment_r2/formal_preflight_receipt.json"
                ),
                "qualification_path": (
                    "data/interim/stage4/anomaly_increment_r2/scoring_qualification.json"
                ),
                "stage4_junit_path": (
                    "data/interim/stage4/anomaly_increment_r2/qualification_stage4.junit.xml"
                ),
                "full_non_target_junit_path": (
                    "data/interim/stage4/anomaly_increment_r2/"
                    "qualification_full_non_target.junit.xml"
                ),
                "formal_attempt_ledger_path": (
                    "data/manifests/anomaly_increment_r2_attempt_ledger.json"
                ),
                "target_read_ledger_path": (
                    "data/manifests/anomaly_increment_r2_target_read_ledger.json"
                ),
                "checkpoint_root": ("data/interim/stage4/anomaly_increment_r2/checkpoints"),
                "selected_table_logical_identity": {
                    "method_id": "arrow_ipc_selected_table_logical_identity_r1",
                    "sha256_domain_separator_ascii": (
                        "seismoflux.selected-table-logical-identity.r1"
                    ),
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
                },
                "gpu_if_not_equivalent_at_code_freeze": ("lock_formal_run_to_cpu_float64"),
                "required_before_target_read": [
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
                ],
                "test_count_binding": {
                    "freeze_time": (
                        "after_scoring_implementation_and_all_non_target_tests_are_final"
                    ),
                    "actual_junit_count_must_equal_frozen_count": True,
                    "protocol_acceptance_snapshot_count_may_not_be_reused": True,
                    "expected_test_count": "UNFROZEN",
                },
            },
        },
        "compute": {
            "max_workers": 6,
            "logical_cpu_affinity_limit": 6,
            "process_priority": "below_normal",
            "nested_parallelism": False,
            "blas_threads_per_worker": 1,
            "blas_environment_must_be_set_before_numpy_or_scipy_import": True,
            "resource_control_receipt_required_before_target_read": True,
            "gpu_available_reference": {
                "python_gpu_backend_installed_at_freeze": gpu_backend_frozen,
            },
            "gpu_equivalence": {
                "objective_relative_tolerance": 1.0e-10,
                "gradient_max_abs_tolerance": 1.0e-8,
                "coefficient_max_abs_tolerance": 1.0e-8,
                "integrated_intensity_relative_tolerance": 1.0e-8,
            },
        },
        "spatial_permutation_topology": {
            "local_restricted_artifacts": {
                "directory": "data/interim/stage4/anomaly_increment_r2",
                "cell_mapping": (
                    "data/interim/stage4/anomaly_increment_r2/"
                    "construction_zone_cell_mapping.parquet"
                ),
                "connectors": (
                    "data/interim/stage4/anomaly_increment_r2/construction_zone_connectors.json"
                ),
                "entity_mapping": (
                    "data/interim/stage4/anomaly_increment_r2/"
                    "construction_zone_entity_mapping.parquet"
                ),
                "zone_geometry": (
                    "data/interim/stage4/anomaly_increment_r2/construction_zones.parquet"
                ),
                "access_control": {
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
                        (
                            "entry_and_exit_owner_permission_samples_on_same_retained_"
                            "handle_required"
                        ): True,
                        ("entry_and_exit_canonical_permission_descriptor_bytes_must_match"): True,
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
                        ("inherit_only_ace_may_not_satisfy_current_object_full_control"): True,
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
                            {
                                "filesystem_type": "tmpfs",
                                "filesystem_magic_hex": "0x01021994",
                            },
                            {
                                "filesystem_type": "xfs",
                                "filesystem_magic_hex": "0x58465342",
                            },
                            {
                                "filesystem_type": "btrfs",
                                "filesystem_magic_hex": "0x9123683e",
                            },
                            {
                                "filesystem_type": "f2fs",
                                "filesystem_magic_hex": "0xf2f52010",
                            },
                        ],
                    },
                    "unsupported_or_unverifiable_platform_action": "fail_closed",
                },
            }
        },
        "inputs": {
            "earthquake_target": {
                "path": target_path,
                "sha256": target_sha256,
                "content_sha256": "5" * 64,
                "schema_sha256": "6" * 64,
                "contract_path": "synthetic/target_contract.json",
                "contract_sha256": "7" * 64,
                "physical_event_id_column": "event_id",
                "unavailable_before_protocol_freeze": True,
                "human_prediction_fields_forbidden": True,
                "frozen_catalog_coverage": {
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
                },
            }
        },
        "evaluation": {
            "machine_status_vocabularies": {
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
            },
            "permutations": {
                "formal_requests": [
                    {"kind": "time", "model_variant": "dynamic"},
                    {"kind": "space", "model_variant": "dynamic"},
                    {"kind": "time", "model_variant": "snapshot"},
                    {"kind": "space", "model_variant": "snapshot"},
                ],
                "formal_checkpoint_request_identities": [
                    "time-dynamic",
                    "space-dynamic",
                    "time-snapshot",
                    "space-snapshot",
                ],
                "checkpoint_identity_pattern": "kind-model_variant",
                "exact_request_set_required": True,
                "mappings_paired_across_dynamic_and_snapshot": True,
            },
            "multiple_comparisons": {
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
                    "primary_candidate": "dynamic",
                    "snapshot_role": (
                        "conditional_fallback_only_after_dynamic_G2_pass_and_G3_not_pass"
                    ),
                    "snapshot_independent_rescue_when_dynamic_G2_not_pass": "forbidden",
                    "dynamic_and_snapshot_are_not_parallel_alpha_entries": True,
                    "all_four_observed_raw_statistics_and_full_null_distributions_"
                    "always_computed": True,
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
                        "time_placebo_result_object_sha256",
                        "space_placebo_result_object_sha256",
                    ],
                },
            },
            "gates": {
                "G2": {
                    "candidate_variant": "dynamic",
                    "snapshot_equivalent_candidate_variant": "snapshot",
                    "evaluated_model_variants": ["dynamic", "snapshot"],
                    "required_primary_placebos_by_variant": {
                        "dynamic": ["time", "space"],
                        "snapshot": ["time", "space"],
                    },
                    "primary_time_permutation_p_lte": 0.05,
                    "primary_space_permutation_p_lte": 0.05,
                    "both_primary_p_values_required_for_each_evaluated_variant": True,
                    "same_practical_improvement_thresholds_apply_per_evaluated_variant": True,
                    "reporting_confound_guard_each_evaluated_candidate_gt_coverage_only": True,
                    "reporting_confound_guard_applies_independently_to_variants": [
                        "dynamic",
                        "snapshot",
                    ],
                    "candidate_minus_coverage_only_macro_information_gain_lower_95pct_bound_gt": 0,
                    "candidate_qualification_requires": [
                        "formal_validation_G2_core",
                        "development_increment_stability",
                        "regional_stability",
                    ],
                    "candidate_qualification_components_must_all_pass": True,
                    "candidate_qualification_nonpass_statuses": [
                        "failed",
                        "evidence_insufficient",
                    ],
                    "dynamic_is_only_confirmatory_H1_G2_entry": True,
                    "snapshot_is_conditional_G3_fallback_only": True,
                    "practical_improvement_any_of": [
                        {
                            "metric": "same_area_strict_recall_gain_percentage_points",
                            "evaluation_partition": "formal_validation_once",
                            "candidate_variant": "current_evaluated_candidate_variant",
                            "comparator_variant": "background_no_increment",
                            "magnitude_bin": "M5_6",
                            "horizons_days": [7, 30, 90],
                            "macro_weighting": "equal_horizon",
                            "threshold_gte": 5,
                            "lower_95pct_bound_gt": 0,
                            "area_budget_km2": 600000,
                            "per_issue_selection": (
                                "largest_complete_deterministic_cell_prefix_with_"
                                "exact_area_lte_budget"
                            ),
                            "denominator": ("all_study_area_targets_unsupported_count_as_misses"),
                        },
                        {
                            "metric": "same_recall_union_area_relative_reduction",
                            "evaluation_partition": "formal_validation_once",
                            "candidate_variant": "current_evaluated_candidate_variant",
                            "comparator_variant": "background_no_increment",
                            "magnitude_bin": "M5_6",
                            "horizons_days": [7, 30, 90],
                            "macro_weighting": "equal_horizon",
                            "reference_hits_per_horizon": (
                                "background_strict_hit_count_at_600000km2"
                            ),
                            "candidate_search": ("frozen_625km2_budget_grid_no_interpolation"),
                            "candidate_budget_per_horizon": (
                                "minimum_budget_reaching_reference_hit_count"
                            ),
                            "area_value": "exact_selected_complete_cell_prefix_area",
                            "exposure_area_aggregation": (
                                "equal_weight_mean_exact_selected_area_across_"
                                "nonoverlapping_exposures_within_horizon"
                            ),
                            "per_horizon_reduction": (
                                "one_minus_candidate_mean_exact_area_div_background_mean_exact_area"
                            ),
                            "macro_reduction": (
                                "equal_weight_mean_of_three_per_horizon_reductions"
                            ),
                            "bootstrap_area_values_fixed_event_weights_only_change_hit_"
                            "counts_and_selected_budget": True,
                            "zero_reference_hit_action": ("branch_not_evaluable_and_not_pass"),
                            "unreachable_at_960000km2_action": (
                                "branch_not_evaluable_and_not_pass"
                            ),
                            "bootstrap_invalid_branch_value_action": (
                                "retain_numeric_reduction_0.0_as_failure_not_drop"
                            ),
                            "threshold_gte": 0.10,
                            "lower_95pct_bound_gt": 0,
                        },
                    ],
                },
                "development_increment_stability": {
                    "role": "required_candidate_qualification_component",
                    "evaluation_partition": (
                        "three_frozen_mutually_disjoint_development_rolling_folds"
                    ),
                    "evaluated_model_variants": ["dynamic", "snapshot"],
                    "model_variant_roles": {
                        "dynamic": "confirmatory_H1_G2_candidate",
                        "snapshot": "conditional_G3_fallback_only",
                    },
                    "comparators": ["background_no_increment", "coverage_only"],
                    "primary_magnitude_bin": "M5_6",
                    "horizons_days": [7, 30, 90],
                    "macro_weighting": "equal_horizon",
                    "per_horizon_statistic": (
                        "candidate_minus_comparator_information_gain_nats_per_event"
                    ),
                    "fold_statistic": ("equal_weight_mean_of_all_three_horizon_statistics"),
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
                        "expected_horizon_row_count": 36,
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
                        "expected_fold_macro_row_count": 12,
                        "exact_candidate_comparator_summary_key": [
                            "model_variant",
                            "comparator_variant",
                        ],
                        "required_candidate_comparator_summary_values": [
                            "positive_fold_count",
                            "median_fold_macro_information_gain_nats_per_event",
                            "status",
                        ],
                        "expected_candidate_comparator_summary_count": 4,
                        "allowed_statuses": [
                            "passed",
                            "failed",
                            "evidence_insufficient",
                        ],
                        "missing_row_or_value_action": "evidence_insufficient",
                    },
                    "failure_action": (
                        "candidate_not_G2_qualified_and_publish_no_cross_split_stability"
                    ),
                    "insufficient_action": (
                        "candidate_not_G2_qualified_evidence_insufficient_no_random_split"
                    ),
                },
            },
            "adoption_matrix": {
                "dynamic_G2_pass_and_G3_pass": "adopt_dynamic_for_stage5_comparison",
                "dynamic_G2_pass_and_G3_not_pass_and_snapshot_equivalent_G2_pass": (
                    "adopt_snapshot_only"
                ),
                "dynamic_G2_pass_and_G3_not_pass_and_snapshot_equivalent_G2_not_pass": (
                    "retain_background_only"
                ),
                "dynamic_G2_not_pass": ("retain_background_only_and_stop_complex_anomaly_models"),
                "dynamic_G2_pass_means_all_candidate_qualification_components_passed": True,
                "snapshot_equivalent_G2_pass_means_all_candidate_qualification_components_"
                "passed": True,
                "G2_not_pass_statuses": ["failed", "evidence_insufficient"],
                "G3_not_pass_statuses": ["failed", "evidence_insufficient"],
                "snapshot_evaluated_for_adoption_only_after_dynamic_G2_pass_and_G3_not_pass": True,
                "snapshot_independent_rescue_when_dynamic_G2_not_pass": "forbidden",
                "coverage_only": "reporting_confound_diagnostic_never_an_anomaly_adoption",
                "snapshot_equivalent_G2_uses_same_thresholds_placebos_and_practical_metric": True,
                "no_post_score_fallback_choice": True,
            },
            "regional_stability": {
                "role": ("required_candidate_qualification_component_and_regional_diagnostic"),
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
                "comparators": ["background_no_increment", "coverage_only"],
                "primary_magnitude_bin": "M5_6",
                "horizons_days": [7, 30, 90],
                "macro_weighting": "equal_horizon",
                "per_region_horizon_contribution_formula": (
                    "(region_event_log_intensity_difference-"
                    "region_integrated_compensator_difference)/"
                    "global_supported_unique_event_count_for_horizon"
                ),
                "zero_global_supported_event_action": "evidence_insufficient",
                "per_region_macro_contribution": (
                    "equal_weight_mean_of_three_horizon_contributions"
                ),
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
                    "same_event_weights_shared_across_variants_comparators_horizons_and_"
                    "regions": True,
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
                "failure_action": (
                    "candidate_not_G2_qualified_and_publish_no_cross_region_stability"
                ),
                "insufficient_action": ("candidate_not_G2_qualified_evidence_insufficient"),
                "improvement_in_only_one_region_action": (
                    "candidate_not_G2_qualified_and_no_stable_increment_claim"
                ),
            },
        },
        "publication": {
            "spatial_output_isolation": {
                "physical_file_count": 4,
                "forecast_target_free_files": [
                    "outputs/visualizations/anomaly_increment_r2_forecast_spatial.svg",
                    "outputs/visualizations/anomaly_increment_r2_forecast_spatial.html",
                ],
                "retrospective_target_bearing_local_restricted_files": [
                    "outputs/visualizations/anomaly_increment_r2_retrospective_target_local.svg",
                    "outputs/visualizations/anomaly_increment_r2_retrospective_target_local.html",
                ],
                "target_payload_in_forecast_files_forbidden": True,
                "automatic_cross_file_target_loading_forbidden": True,
                "retrospective_target_bearing_files_require_restricted_access_before_bytes_"
                "written": True,
                "retrospective_access_control_receipt_required_before_publication": True,
                "retrospective_access_control": {
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
                },
                "public_forecast_artifact_validator": {
                    "reject_artifact_classifications": [
                        "local_restricted",
                        "target_bearing",
                    ],
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
                    "validation_scope": (
                        "parsed_static_dom_and_recursively_deserialized_interactive_payload"
                    ),
                    "keyword_scan_or_ui_hiding_sufficient": False,
                    "failure_action": "fail_closed_forbid_publication",
                },
            },
            "result_identity_requires": [
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
                "adoption_decision",
                "adopted_variant_metrics_table",
            ],
            "spatial_rendering_contract": {
                "center_point_fallback_warning_text": (
                    "中心点示意，非面积几何；报警面积以数值为准"  # noqa: RUF001
                ),
            },
            "plot_axis_contract": {
                "numeric_ticks_required": True,
                "molchan_x_domain": [0.0, 1.0],
                "fixed_area_x_domain_km2": [0, 960000],
            },
            "limitations": {
                "earthquake_available_at_assumption": (
                    "available_at_equals_origin_time_is_an_optimistic_timeliness_assumption"
                ),
                "bootstrap_interval_scope": (
                    "conditional_on_fixed_fitted_model_and_excludes_refit_uncertainty"
                ),
                "etas_comparator_status": "not_evaluable",
                "allowed_increment_claim": "relative_to_frozen_kde_background_only",
                "incremental_value_over_etas_claim_forbidden": True,
            },
            "display_semantics": {
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
            },
        },
    }


def _repository() -> Stage4RepositoryEvidence:
    return Stage4RepositoryEvidence(
        code_commit=CODE_COMMIT,
        branch="codex/stage4-increment-scoring-code",
        upstream="origin/codex/stage4-increment-scoring-code",
        remote="origin",
        remote_repository="github.com/Justin-147/SeismoFlux",
        remote_branch_ref="refs/heads/codex/stage4-increment-scoring-code",
        remote_branch_commit=CODE_COMMIT,
        public_repository=PublicRepositoryEvidence(
            full_name="Justin-147/SeismoFlux",
            visibility="public",
            private=False,
            html_url="https://github.com/Justin-147/SeismoFlux",
        ),
        protocol_tag=PROTOCOL_TAG,
        protocol_tag_object=PROTOCOL_TAG_OBJECT,
        protocol_tag_commit=PROTOCOL_COMMIT,
        scoring_code_tag=SCORING_TAG,
        scoring_code_tag_object=SCORING_TAG_OBJECT,
        scoring_code_tag_commit=CODE_COMMIT,
        frozen_blob_ids=tuple((path, "8" * 40) for path in STAGE4_FROZEN_PROTOCOL_PATHS),
    )


def _restricted_access(
    protocol: dict[str, Any],
) -> RestrictedLocalArtifactAccessControlEvidence:
    local = protocol["spatial_permutation_topology"]["local_restricted_artifacts"]
    file_hashes = {
        artifact_id: (str(index + 1) * 64)[:64]
        for index, artifact_id in enumerate(RESTRICTED_ARTIFACT_IDS)
    }
    observations = [
        RestrictedPathAccessObservation(
            relative_path=local["directory"],
            path_kind="directory",
            verified_sha256=None,
            owner_principal="uid:1000",
            link_count=1,
            reparse_point=False,
            permission_descriptor_json=canonical_permission_descriptor(
                {
                    "acl_model": "classic_mode_bits_without_acl_xattrs",
                    "acl_related_xattrs": [],
                    "filesystem_locality": "local",
                    "filesystem_magic_hex": "0x0000ef53",
                    "filesystem_type": "ext2_ext3_ext4",
                    "mode_octal": "0700",
                    "owner_uid": 1000,
                    "platform_family": "linux",
                    "queried_by_handle": True,
                }
            ),
        )
    ]
    for artifact_id in RESTRICTED_ARTIFACT_IDS:
        observations.append(
            RestrictedPathAccessObservation(
                relative_path=local[artifact_id],
                path_kind="regular_file",
                verified_sha256=file_hashes[artifact_id],
                owner_principal="uid:1000",
                link_count=1,
                reparse_point=False,
                permission_descriptor_json=canonical_permission_descriptor(
                    {
                        "acl_model": "classic_mode_bits_without_acl_xattrs",
                        "acl_related_xattrs": [],
                        "filesystem_locality": "local",
                        "filesystem_magic_hex": "0x0000ef53",
                        "filesystem_type": "ext2_ext3_ext4",
                        "mode_octal": "0600",
                        "owner_uid": 1000,
                        "platform_family": "linux",
                        "queried_by_handle": True,
                    }
                ),
            )
        )
    return RestrictedLocalArtifactAccessControlEvidence(
        protocol_design_sha256=protocol_design_sha256(protocol),
        access_contract_sha256=restricted_access_contract_sha256(protocol),
        platform="posix",
        current_principal="uid:1000",
        directory_relative_path=local["directory"],
        file_artifacts=tuple(
            (artifact_id, local[artifact_id], file_hashes[artifact_id])
            for artifact_id in RESTRICTED_ARTIFACT_IDS
        ),
        observations=tuple(sorted(observations, key=lambda item: item.relative_path)),
    )


def _score_blind_inputs(protocol: dict[str, Any]) -> ScoreBlindInputEvidence:
    access = _restricted_access(protocol)
    return ScoreBlindInputEvidence(
        protocol_design_sha256=protocol_design_sha256(protocol),
        random_input_seal_sha256="9" * 64,
        protocol_validation_sha256="c" * 64,
        observed_project_input_hashes=(("environment_lock", "d" * 64),),
        generated_manifest_hashes=(("fold", "e" * 64),),
        restricted_spatial_artifact_hashes=tuple(
            sorted((artifact_id, digest) for artifact_id, _, digest in access.file_artifacts)
        ),
        restricted_access_control=access,
    )


def _preflight_receipt(
    protocol: dict[str, Any],
    inputs: ScoreBlindInputEvidence,
) -> FormalPreflightReceipt:
    return replace(
        make_formal_preflight_receipt(),
        protocol_design_sha256=protocol_design_sha256(protocol),
        random_input_seal_sha256=inputs.random_input_seal_sha256,
        score_blind_input_evidence_sha256=inputs.content_sha256,
        scoring_code_commit=CODE_COMMIT,
    )


def _pytest_xml(*, full: bool, full_count: int | None = None) -> bytes:
    test_ids = {
        test_id for values in REQUIRED_TESTS_BY_QUALIFICATION.values() for test_id in values
    }
    if full:
        index = 0
        frozen_count = (
            full_count
            if full_count is not None
            else qualification_module.FROZEN_FULL_NON_TARGET_TEST_COUNT
        )
        assert frozen_count is not None
        while len(test_ids) < frozen_count:
            test_ids.add(f"tests.unit.synthetic_full::test_dummy_{index:04d}")
            index += 1
    suite = ET.Element(
        "testsuite",
        {
            "name": "pytest",
            "errors": "0",
            "failures": "0",
            "skipped": "0",
            "tests": str(len(test_ids)),
        },
    )
    for test_id in sorted(test_ids):
        class_name, test_name = test_id.split("::", maxsplit=1)
        ET.SubElement(suite, "testcase", {"classname": class_name, "name": test_name})
    root = ET.Element("testsuites", {"name": "pytest tests"})
    root.append(suite)
    return cast(bytes, ET.tostring(root, encoding="utf-8"))


def _pytest_evidence(*, full: bool) -> PytestRunEvidence:
    return parse_pytest_junit_evidence(_pytest_xml(full=full))


def test_restricted_access_evidence_round_trip_and_tampering_fail_closed() -> None:
    evidence = _restricted_access(_protocol("0" * 64))
    assert (
        RestrictedLocalArtifactAccessControlEvidence.from_mapping(evidence.as_mapping()) == evidence
    )
    tampered = evidence.as_mapping()
    tampered["current_principal"] = "uid:1001"
    with pytest.raises(RestrictedAccessError, match="hash or schema"):
        RestrictedLocalArtifactAccessControlEvidence.from_mapping(tampered)


@pytest.mark.parametrize("delta", (-1, 1))
def test_full_junit_count_must_equal_the_synthetic_frozen_count(delta: int) -> None:
    protocol = _protocol("0" * 64)
    inputs = _score_blind_inputs(protocol)
    mismatched = parse_pytest_junit_evidence(
        _pytest_xml(full=True, full_count=SYNTHETIC_FROZEN_TEST_COUNT + delta)
    )
    with pytest.raises((Stage4QualificationError, ValueError), match="count"):
        build_stage4_qualification_evidence(
            protocol,
            scoring_code_commit=CODE_COMMIT,
            score_blind_input_evidence=inputs,
            formal_preflight_receipt=_preflight_receipt(protocol, inputs),
            logical_identity_replay_audit_sha256="c" * 64,
            stage4_pytest=_pytest_evidence(full=False),
            full_pytest=mismatched,
        )


def _qualification(
    protocol: dict[str, Any],
    inputs: ScoreBlindInputEvidence,
    *,
    gpu_requested: bool = True,
    gpu_passed: bool = False,
) -> Stage4QualificationEvidence:
    gpu_evidence = (
        BackendEquivalenceEvidence(
            candidate_backend="gpu_float64",
            objective_relative_error=0.0,
            gradient_max_abs_error=0.0,
            coefficient_max_abs_error=0.0,
            integrated_intensity_relative_error=0.0,
            random_mapping_byte_identity=True,
            scientific_decision_identity=True,
            repeated_run_identity=True,
            worker_count_identity=True,
        )
        if gpu_passed
        else None
    )
    return build_stage4_qualification_evidence(
        protocol,
        scoring_code_commit=CODE_COMMIT,
        score_blind_input_evidence=inputs,
        formal_preflight_receipt=_preflight_receipt(protocol, inputs),
        logical_identity_replay_audit_sha256="c" * 64,
        stage4_pytest=_pytest_evidence(full=False),
        full_pytest=_pytest_evidence(full=True),
        gpu_requested=gpu_requested,
        gpu_equivalence=gpu_evidence,
    )


def _logical_replay_audit_document(
    protocol: dict[str, Any],
    inputs: ScoreBlindInputEvidence,
) -> dict[str, object]:
    receipts = [
        {
            "accepted_table_sha256": hashlib.sha256(f"logical-replay:{index}".encode()).hexdigest(),
            "issue_id": f"anomaly-issue-{index:03d}",
            "issue_index": index,
            "issue_report_id": f"report-{index:03d}",
            "recomputed_table_sha256": hashlib.sha256(
                f"logical-replay:{index}".encode()
            ).hexdigest(),
        }
        for index in range(153)
    ]
    reproduction_sha256 = "d" * 64
    audit = with_content_sha256(
        {
            "grid_id": "stage4-grid-25km",
            "identity_method": "arrow_ipc_selected_table_logical_identity_r1",
            "issue_count": 153,
            "query_chunk_size": 256,
            "reproduction_identity_sha256": reproduction_sha256,
            "role": "stage4_r1_primary_grid_logical_identity_worker_replay",
            "source_columns": ["signal"],
            "source_input_sha256": inputs.content_sha256,
            "target_bytes_read": False,
            "target_path_observed": False,
            "worker_replays": [
                {
                    "receipts": receipts,
                    "reproduction_identity_sha256": reproduction_sha256,
                    "spatial_workers": workers,
                }
                for workers in (1, 2)
            ],
        }
    )
    return with_content_sha256(
        {
            "audit": audit,
            "locked_test_run": False,
            "protocol_design_sha256": protocol_design_sha256(protocol),
            "random_input_seal_sha256": inputs.random_input_seal_sha256,
            "schema_version": 1,
            "scoring_code_commit": CODE_COMMIT,
            "target_bytes_read": False,
            "target_path_observed": False,
            "worktree_clean_before_and_after": True,
        }
    )


def _seal_fixture(
    tmp_path: Path,
    payload: bytes = b"synthetic-target",
) -> tuple[
    dict[str, Any],
    ScoreBlindInputEvidence,
    Stage4QualificationEvidence,
    Path,
    Path,
    Path,
]:
    protocol = _protocol(hashlib.sha256(payload).hexdigest())
    inputs = _score_blind_inputs(protocol)
    qualification = _qualification(protocol, inputs)
    preflight_receipt = _preflight_receipt(protocol, inputs)
    repository = _repository()
    binding = stage4_execution_binding_id(repository, inputs, qualification)
    attempt_path = tmp_path.joinpath(*Path(STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH).parts)
    target_path = tmp_path.joinpath(*Path(STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH).parts)
    attempt = initialize_stage4_ledger(
        attempt_path,
        kind="formal_attempt",
        execution_binding_id=binding,
    )
    target = initialize_stage4_ledger(
        target_path,
        kind="target_read",
        execution_binding_id=binding,
    )
    seal = build_stage4_scoring_seal(
        protocol,
        repository=repository,
        score_blind_inputs=inputs,
        qualification=qualification,
        formal_preflight_receipt=preflight_receipt,
        attempt_ledger=attempt,
        target_read_ledger=target,
    )
    seal_path = tmp_path / "data" / "manifests" / "anomaly_increment_r2_scoring_seal.json"
    write_stage4_scoring_seal_atomic(seal_path, seal, protocol=protocol)
    preflight_path = tmp_path.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(preflight_receipt.as_mapping(), sort_keys=True),
        encoding="utf-8",
    )
    return protocol, inputs, qualification, attempt_path, target_path, seal_path


class _RepositoryAdapter:
    def __init__(self, evidence: Stage4RepositoryEvidence) -> None:
        self.evidence = evidence
        self.calls = 0

    def observe(
        self,
        project_root: Path,
        *,
        protocol_tag: str,
        scoring_code_tag: str,
        allowed_untracked_paths: Any = (),
    ) -> Stage4RepositoryEvidence:
        del project_root, allowed_untracked_paths
        assert protocol_tag == PROTOCOL_TAG
        assert scoring_code_tag == SCORING_TAG
        self.calls += 1
        return self.evidence


class _PublicProbe:
    def observe(self) -> PublicRepositoryEvidence:
        return _repository().public_repository


def _authorize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: bytes = b"synthetic-target",
) -> tuple[Stage4TargetAuthorization, dict[str, Any], Path, Path]:
    protocol, inputs, _, attempt_path, target_path, seal_path = _seal_fixture(
        tmp_path,
        payload,
    )
    monkeypatch.setattr(authorization_module, "observe_score_blind_inputs", lambda *_: inputs)
    authorization = authorize_stage4_target_access(
        tmp_path,
        protocol,
        scoring_seal_path=seal_path,
        attempt_ledger_path=attempt_path,
        target_read_ledger_path=target_path,
        repository_adapter=_RepositoryAdapter(_repository()),
    )
    return authorization, protocol, attempt_path, target_path


def test_expected_target_identity_is_metadata_only_and_never_requires_target_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "synthetic" / "target.bin"
    protocol = _protocol("0" * 64, missing.relative_to(tmp_path).as_posix())

    identity = expected_target_identity_from_protocol(protocol)

    assert identity["path"] == "synthetic/target.bin"
    assert not missing.exists()


def test_r2_authorization_paths_are_disjoint_from_retired_execution_namespaces() -> None:
    assert STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH == (
        "data/manifests/anomaly_increment_r2_attempt_ledger.json"
    )
    assert STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH == (
        "data/manifests/anomaly_increment_r2_target_read_ledger.json"
    )
    assert FORMAL_PREFLIGHT_RECEIPT_PATH.as_posix() == (
        "data/interim/stage4/anomaly_increment_r2/formal_preflight_receipt.json"
    )
    assert STAGE4_FROZEN_PROTOCOL_PATHS == (
        "configs/anomaly_increment_r2.yaml",
        "data/manifests/anomaly_increment_r2_feature_set.json",
        "data/manifests/anomaly_increment_r2_fold_manifest.json",
        "data/manifests/anomaly_increment_r2_randomness.json",
        "data/manifests/anomaly_increment_r2_spatial_strata.json",
        "docs/anomaly_increment_protocol.md",
        "docs/anomaly_increment_protocol_r1.md",
        "docs/anomaly_increment_protocol_r2.md",
        "docs/phase4_scoring_readiness_incident_r0.md",
        "docs/phase4_protocol_r1_acceptance.md",
        "docs/phase4_protocol_r2_acceptance.md",
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("freeze", "execution_revision", "r1"),
        ("freeze", "pre_score_tag", "v0.3.0-anomaly-increment-protocol-r1"),
        ("freeze", "results_tag", "v0.3.0-anomaly-increment-r1"),
        (
            "scoring",
            "expected_tag",
            "v0.3.0-anomaly-increment-scoring-code-r1",
        ),
        (
            "scoring",
            "formal_attempt_ledger_path",
            "data/manifests/anomaly_increment_attempt_ledger.json",
        ),
        ("logical", "top_level_schema_metadata", "included"),
    ),
)
def test_raw_authorization_and_seal_freeze_tags_fail_closed_on_r1_or_drift(
    section: str,
    key: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = yaml.safe_load(Path("configs/anomaly_increment_r2.yaml").read_text("utf-8"))
    assert isinstance(loaded, dict)
    protocol = deepcopy(loaded)
    freeze = cast(dict[str, Any], protocol["freeze"])
    scoring = cast(dict[str, Any], freeze["scoring_code_freeze"])
    if section == "freeze":
        freeze[key] = value
    elif section == "scoring":
        scoring[key] = value
    else:
        logical = cast(dict[str, Any], scoring["selected_table_logical_identity"])
        logical[key] = value

    # Restore the real guard for this contract-boundary test.  The module-wide
    # synthetic fixture bypass is only for downstream future-execution tests.
    monkeypatch.setattr(
        authorization_module,
        "require_stage4_r2_execution_action",
        config_module.require_stage4_r2_execution_action,
    )
    monkeypatch.setattr(
        seal_script,
        "require_stage4_r2_execution_action",
        config_module.require_stage4_r2_execution_action,
    )

    with pytest.raises(
        Stage4ScoringNotAuthorizedError,
        match="R2 execution is blocked",
    ):
        authorization_module._freeze_tags(protocol, action="formal_target_read")
    with pytest.raises(ValueError, match="stage-4"):
        seal_script._freeze_tags(protocol)


def test_scoring_seal_and_qualification_round_trip_are_content_addressed(
    tmp_path: Path,
) -> None:
    protocol, inputs, qualification, _, _, seal_path = _seal_fixture(tmp_path)
    seal = load_stage4_scoring_seal(seal_path, protocol=protocol)
    qualification_path = tmp_path / "qualification.json"
    write_qualification(qualification_path, qualification)

    assert load_stage4_qualification_evidence(qualification_path) == qualification
    assert seal.score_blind_inputs == inputs
    assert (
        dict(seal.expected_target_identity)["sha256"]
        == protocol["inputs"]["earthquake_target"]["sha256"]
    )
    assert seal.as_mapping()["target_observation"] == {
        "expected_identity_copied_from_protocol": True,
        "path_observed": False,
        "read_count": 0,
        "stat_called": False,
    }
    receipt = _preflight_receipt(protocol, inputs)
    assert qualification.formal_preflight_receipt_sha256 == receipt.content_sha256
    assert qualification.space_placebo_resource_observation_sha256 == (
        receipt.space_placebo_resource_observation.content_sha256
    )
    changed_resource = qualification.as_mapping()
    changed_resource["space_placebo_resource_observation_sha256"] = "0" * 64
    resource_variant = Stage4QualificationEvidence.from_mapping(changed_resource)
    assert resource_variant.content_sha256 == qualification.content_sha256
    with pytest.raises(Stage4QualificationError, match="resource observation"):
        validate_stage4_qualification_against_formal_preflight(
            resource_variant,
            receipt,
        )


def test_frozen_governance_writers_are_create_only_and_idempotent(
    tmp_path: Path,
) -> None:
    protocol, _, qualification, _, _, seal_path = _seal_fixture(tmp_path)
    seal = load_stage4_scoring_seal(seal_path, protocol=protocol)
    original_seal = seal_path.read_bytes()
    assert (
        write_stage4_scoring_seal_atomic(seal_path, seal, protocol=protocol)
        == hashlib.sha256(original_seal).hexdigest()
    )
    changed_seal = replace(seal, initial_attempt_ledger_sha256="0" * 64)
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="different bytes"):
        write_stage4_scoring_seal_atomic(
            seal_path,
            changed_seal,
            protocol=protocol,
        )
    assert seal_path.read_bytes() == original_seal

    qualification_path = tmp_path / "qualification.json"
    write_qualification(qualification_path, qualification)
    original_qualification = qualification_path.read_bytes()
    assert (
        write_qualification(
            qualification_path,
            qualification,
        )
        == hashlib.sha256(original_qualification).hexdigest()
    )
    changed_mapping = qualification.as_mapping()
    changed_mapping["space_placebo_resource_observation_sha256"] = "0" * 64
    changed_qualification = Stage4QualificationEvidence.from_mapping(changed_mapping)
    with pytest.raises(Stage4QualificationError, match="different bytes"):
        write_qualification(
            qualification_path,
            changed_qualification,
        )
    assert qualification_path.read_bytes() == original_qualification

    receipt_path = tmp_path / "formal_preflight_receipt.json"
    preflight_script._write_atomic(
        receipt_path,
        {"receipt": "frozen"},
        protocol=protocol,
    )
    original_receipt = receipt_path.read_bytes()
    preflight_script._write_atomic(
        receipt_path,
        {"receipt": "frozen"},
        protocol=protocol,
    )
    with pytest.raises(ValueError, match="different bytes"):
        preflight_script._write_atomic(
            receipt_path,
            {"receipt": "changed"},
            protocol=protocol,
        )
    assert receipt_path.read_bytes() == original_receipt


def _replace_with_same_byte_hardlink(path: Path, payload: bytes) -> Path:
    path.unlink(missing_ok=True)
    backing = path.with_name(f"{path.name}.hardlink-source")
    backing.write_bytes(payload)
    try:
        path.hardlink_to(backing)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable on this test filesystem: {exc}")
    assert path.stat().st_nlink > 1
    return backing


def test_frozen_governance_writers_reject_same_byte_hardlinks(tmp_path: Path) -> None:
    protocol, _, qualification, _, _, seal_path = _seal_fixture(tmp_path)
    seal = load_stage4_scoring_seal(seal_path, protocol=protocol)
    seal_payload = seal_path.read_bytes()
    _replace_with_same_byte_hardlink(seal_path, seal_payload)
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="single-link"):
        write_stage4_scoring_seal_atomic(seal_path, seal, protocol=protocol)

    qualification_path = tmp_path / "hardlinked-qualification.json"
    write_qualification(qualification_path, qualification)
    qualification_payload = qualification_path.read_bytes()
    _replace_with_same_byte_hardlink(qualification_path, qualification_payload)
    with pytest.raises(Stage4QualificationError, match="single-link"):
        write_qualification(qualification_path, qualification)

    receipt_path = tmp_path / "hardlinked-formal-preflight.json"
    preflight_script._write_atomic(
        receipt_path,
        {"receipt": "frozen"},
        protocol=protocol,
    )
    receipt_payload = receipt_path.read_bytes()
    _replace_with_same_byte_hardlink(receipt_path, receipt_payload)
    with pytest.raises(ValueError, match="single-link"):
        preflight_script._write_atomic(
            receipt_path,
            {"receipt": "frozen"},
            protocol=protocol,
        )


def test_immutable_reader_detects_path_replacement_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "frozen.json"
    replacement = tmp_path / "replacement.json"
    target.write_bytes(b'{"value":"original"}\n')
    replacement.write_bytes(b'{"value":"replacement"}\n')
    real_open = immutable_file_module._open_no_follow

    def replace_then_open(path: Path) -> int:
        replacement.replace(path)
        return real_open(path)

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", replace_then_open)
    with pytest.raises(UnsafeImmutableFileError, match="changed between lstat"):
        read_existing_immutable_bytes(target, label="synthetic frozen artifact")


def test_immutable_reader_rejects_symlink_or_windows_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    link = tmp_path / "linked.json"
    payload = b'{"value":"must-not-be-read-through-link"}\n'
    source.write_bytes(payload)
    try:
        link.symlink_to(source)
    except OSError:
        link.unlink(missing_ok=True)
        real_lstat = os.lstat
        linked_path = os.path.normcase(os.path.abspath(os.fspath(link)))

        def reparse_lstat(
            path: os.PathLike[str] | str,
            *,
            dir_fd: int | None = None,
        ) -> Any:
            observed = os.path.normcase(os.path.abspath(os.fspath(path)))
            if observed == linked_path:
                return SimpleNamespace(
                    st_file_attributes=0x0400,
                    st_mode=stat.S_IFLNK,
                    st_nlink=1,
                )
            if dir_fd is None:
                return real_lstat(path)
            return real_lstat(path, dir_fd=dir_fd)

        monkeypatch.setattr(os, "lstat", reparse_lstat)

    def forbidden_open(path: Path) -> int:
        raise AssertionError(f"linked artifact reached no-follow open: {path}")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)

    with pytest.raises(UnsafeImmutableFileError, match="non-reparse"):
        read_existing_immutable_bytes(link, label="synthetic frozen artifact")
    assert source.read_bytes() == payload


def test_authorization_reloads_canonical_typed_preflight_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, inputs, _, attempt_path, target_path, seal_path = _seal_fixture(tmp_path)
    preflight_path = tmp_path.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    value = json.loads(preflight_path.read_text("utf-8"))
    value["content_sha256"] = "0" * 64
    preflight_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(authorization_module, "observe_score_blind_inputs", lambda *_: inputs)

    with pytest.raises(Stage4ScoringNotAuthorizedError, match="formal preflight"):
        authorize_stage4_target_access(
            tmp_path,
            protocol,
            scoring_seal_path=seal_path,
            attempt_ledger_path=attempt_path,
            target_read_ledger_path=target_path,
            repository_adapter=_RepositoryAdapter(_repository()),
        )


def test_gpu_is_locked_to_cpu_without_a_frozen_backend_even_with_typed_equivalence() -> None:
    protocol = _protocol("0" * 64)
    inputs = _score_blind_inputs(protocol)

    assert _qualification(protocol, inputs, gpu_requested=True).formal_backend == "cpu_float64"
    assert (
        _qualification(protocol, inputs, gpu_requested=True, gpu_passed=True).formal_backend
        == "cpu_float64"
    )
    gpu_protocol = _protocol("0" * 64, gpu_backend_frozen=True)
    gpu_inputs = _score_blind_inputs(gpu_protocol)
    assert (
        _qualification(
            gpu_protocol,
            gpu_inputs,
            gpu_requested=True,
            gpu_passed=True,
        ).formal_backend
        == "gpu_float64"
    )


def test_incomplete_or_tampered_qualification_fails_closed(tmp_path: Path) -> None:
    protocol = _protocol("0" * 64)
    inputs = _score_blind_inputs(protocol)
    suite = ET.Element(
        "testsuite",
        {"name": "pytest", "errors": "0", "failures": "0", "skipped": "0", "tests": "1"},
    )
    ET.SubElement(
        suite,
        "testcase",
        {"classname": "tests.unit.incomplete", "name": "test_only_one"},
    )
    incomplete = parse_pytest_junit_evidence(ET.tostring(suite, encoding="utf-8"))
    with pytest.raises(Stage4QualificationError, match="omits"):
        build_stage4_qualification_evidence(
            protocol,
            scoring_code_commit=CODE_COMMIT,
            score_blind_input_evidence=inputs,
            formal_preflight_receipt=_preflight_receipt(protocol, inputs),
            logical_identity_replay_audit_sha256="c" * 64,
            stage4_pytest=incomplete,
            full_pytest=_pytest_evidence(full=True),
        )

    changed_protocol = json.loads(json.dumps(protocol))
    changed_protocol["freeze"]["scoring_code_freeze"]["required_before_target_read"].remove(
        "worker_count_invariance_passed"
    )
    changed_inputs = _score_blind_inputs(changed_protocol)
    with pytest.raises(Stage4QualificationError, match="protocol gate set"):
        build_stage4_qualification_evidence(
            changed_protocol,
            scoring_code_commit=CODE_COMMIT,
            score_blind_input_evidence=changed_inputs,
            formal_preflight_receipt=_preflight_receipt(
                changed_protocol,
                changed_inputs,
            ),
            logical_identity_replay_audit_sha256="c" * 64,
            stage4_pytest=_pytest_evidence(full=False),
            full_pytest=_pytest_evidence(full=True),
        )

    valid = _qualification(protocol, inputs)
    path = tmp_path / "qualification.json"
    write_qualification(path, valid)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["target_read_count"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Stage4QualificationError, match="hash or schema"):
        load_stage4_qualification_evidence(path)

    value = valid.as_mapping()
    checks = value["checks"]
    assert isinstance(checks, dict)
    synthetic = checks["synthetic_end_to_end_passed"]
    assert isinstance(synthetic, dict)
    synthetic["required_test_ids"] = []
    resource_sha256 = value.pop("space_placebo_resource_observation_sha256")
    value = with_content_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    value["space_placebo_resource_observation_sha256"] = resource_sha256
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Stage4QualificationError, match="check invalid"):
        load_stage4_qualification_evidence(path)


def test_required_qualification_nodes_exist_as_real_test_functions() -> None:
    for test_ids in REQUIRED_TESTS_BY_QUALIFICATION.values():
        for node_id in test_ids:
            class_name, test_name = node_id.split("::", maxsplit=1)
            test_path = Path(*class_name.split(".")).with_suffix(".py")
            assert test_path.is_file(), f"missing qualification test file: {test_path}"
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
            functions = {
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
            function_name = test_name.split("[", maxsplit=1)[0]
            assert function_name in functions, f"missing qualification test node: {node_id}"


def test_empty_ledger_initialization_and_terminal_updates_are_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt.json"
    first = initialize_stage4_ledger(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
    )
    assert (
        initialize_stage4_ledger(
            path,
            kind="formal_attempt",
            execution_binding_id=BINDING_SHA256,
        )
        == first
    )
    start = datetime(2026, 7, 15, tzinfo=UTC)
    mutation = reserve_stage4_operation(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
        operation_id="dev-fold-1-run",
        scope="development-fold-1",
        authorization_id=EVIDENCE_SHA256,
        clock=lambda: start,
    )
    repeated = reserve_stage4_operation(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
        operation_id="dev-fold-1-run",
        scope="development-fold-1",
        authorization_id=EVIDENCE_SHA256,
        clock=lambda: start + timedelta(days=1),
    )
    assert mutation.changed is True
    assert repeated.changed is False
    completed = complete_stage4_operation(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
        operation_id="dev-fold-1-run",
        status="succeeded",
        result_sha256="1" * 64,
        clock=lambda: start + timedelta(seconds=1),
    )
    repeated_completion = complete_stage4_operation(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
        operation_id="dev-fold-1-run",
        status="succeeded",
        result_sha256="1" * 64,
        clock=lambda: start + timedelta(days=1),
    )
    assert completed.changed is True
    assert repeated_completion.changed is False
    assert read_stage4_ledger(path).succeeded_count == 1


def test_interrupted_reservation_cannot_be_recovered_without_frozen_lease_proof(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt.json"
    initialize_stage4_ledger(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
    )
    reserve_stage4_operation(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
        operation_id="formal-run",
        scope="formal-validation",
        authorization_id=EVIDENCE_SHA256,
    )

    with pytest.raises(Stage4LedgerError, match="owner/lease/staleness"):
        recover_interrupted_stage4_operations(
            path,
            kind="formal_attempt",
            execution_binding_id=BINDING_SHA256,
        )

    retained = read_stage4_ledger(path)
    assert retained.operation_count == 1
    assert retained.started_count == 1


def test_registered_attempt_retains_software_failure(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    initialize_stage4_ledger(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
    )
    with (
        pytest.raises(ArithmeticError),
        registered_stage4_attempt(
            path,
            execution_binding_id=BINDING_SHA256,
            operation_id="formal-run",
            scope="formal-validation",
            authorization_id=EVIDENCE_SHA256,
        ),
    ):
        raise ArithmeticError("synthetic failure")

    ledger = read_stage4_ledger(path)
    assert ledger.failed_count == 1
    assert ledger.records[0].failure_code == "arithmetic_error"


def test_ledger_tamper_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    initialize_stage4_ledger(
        path,
        kind="target_read",
        execution_binding_id=BINDING_SHA256,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution_binding_id"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Stage4LedgerError, match="content hash"):
        read_stage4_ledger(path)


def test_ledger_hardlink_alias_is_rejected_before_any_file_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "synthetic-target.bin"
    payload = b"target bytes must never enter the ledger decoder"
    target.write_bytes(payload)
    alias = tmp_path / "attempt.json"
    alias.hardlink_to(target)

    def forbidden_open(path: Path) -> int:
        raise AssertionError(f"unsafe alias reached open: {path}")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    with pytest.raises(Stage4LedgerError, match="safe single-link"):
        read_stage4_ledger(alias)
    assert target.read_bytes() == payload


def test_ledger_lock_hardlink_alias_is_rejected_before_any_file_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "attempt.json"
    lock_key = hashlib.sha256(os.path.abspath(os.fspath(ledger)).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / "seismoflux-stage4-ledger-locks" / lock_key
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.unlink(missing_ok=True)
    target = tmp_path / "synthetic-target.bin"
    payload = b"target bytes must never become a process lock"
    target.write_bytes(payload)
    lock_path.hardlink_to(target)

    def forbidden_open(path: Path, flags: int = os.O_RDONLY) -> int:
        del flags
        raise AssertionError(f"unsafe lock alias reached open: {path}")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    try:
        with pytest.raises(Stage4LedgerError, match="process lock is unsafe"):
            initialize_stage4_ledger(
                ledger,
                kind="formal_attempt",
                execution_binding_id=BINDING_SHA256,
            )
        assert target.read_bytes() == payload
    finally:
        lock_path.unlink(missing_ok=True)


def test_ledger_post_replace_verification_rejects_target_alias_without_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "attempt.json"
    initialize_stage4_ledger(
        path,
        kind="formal_attempt",
        execution_binding_id=BINDING_SHA256,
    )
    target = tmp_path / "synthetic-target.bin"
    payload = b"target bytes must survive ledger replacement ambiguity"
    target.write_bytes(payload)
    real_replace = os.replace

    def replace_then_alias(source: str, destination: Path) -> None:
        real_replace(source, destination)
        Path(destination).unlink()
        Path(destination).hardlink_to(target)

    monkeypatch.setattr(os, "replace", replace_then_alias)
    with pytest.raises(Stage4LedgerError, match="safe single-link"):
        reserve_stage4_operation(
            path,
            kind="formal_attempt",
            execution_binding_id=BINDING_SHA256,
            operation_id="dev-fold-1-run",
            scope="development-fold-1",
            authorization_id=EVIDENCE_SHA256,
        )
    assert target.read_bytes() == payload


@pytest.mark.parametrize(
    ("loader", "requires_protocol"),
    (
        (load_stage4_scoring_seal, True),
        (load_stage4_qualification_evidence, False),
        (load_formal_preflight_receipt, False),
    ),
)
def test_governance_loaders_reject_target_hardlink_before_any_file_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    requires_protocol: bool,
) -> None:
    target = tmp_path / "synthetic-target.bin"
    payload = b"target bytes must never enter a governance loader"
    target.write_bytes(payload)
    alias = tmp_path / "governance-artifact.json"
    alias.hardlink_to(target)

    def forbidden_open(path: Path) -> int:
        raise AssertionError(f"unsafe alias reached open: {path}")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    with pytest.raises(Exception, match="cannot read"):
        if requires_protocol:
            loader(alias, protocol=_protocol("0" * 64))
        else:
            loader(alias)
    assert target.read_bytes() == payload


def test_atomic_target_scope_allows_only_one_concurrent_reservation(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    initialize_stage4_ledger(
        path,
        kind="target_read",
        execution_binding_id=BINDING_SHA256,
    )

    def reserve(index: int) -> bool:
        try:
            return reserve_stage4_operation(
                path,
                kind="target_read",
                execution_binding_id=BINDING_SHA256,
                operation_id=f"target-read-{index}",
                scope=STAGE4_TARGET_SCOPE,
                authorization_id=EVIDENCE_SHA256,
            ).changed
        except Stage4OperationAlreadyConsumedError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(reserve, (1, 2)))

    assert sorted(results) == [False, True]
    assert read_stage4_ledger(path).operation_count == 1


def test_authorization_does_not_require_target_to_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, protocol, _, _ = _authorize(monkeypatch, tmp_path)

    assert authorization.expected_target_mapping()["path"] == "synthetic/target.bin"
    assert not (tmp_path / protocol["inputs"]["earthquake_target"]["path"]).exists()


def test_shadow_empty_ledger_cannot_reauthorize_after_canonical_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic-target-for-shadow-ledger"
    authorization, protocol, attempt_path, _ = _authorize(monkeypatch, tmp_path, payload)
    target_file = tmp_path / "synthetic" / "target.bin"
    target_file.parent.mkdir(parents=True)
    target_file.write_bytes(payload)
    consume_authorized_stage4_target(
        tmp_path,
        authorization,
        protocol=protocol,
        operation_id="canonical-target-read",
        consumer=lambda value: value,
    )
    shadow = tmp_path / "shadow" / "empty_target_ledger.json"
    initialize_stage4_ledger(
        shadow,
        kind="target_read",
        execution_binding_id=authorization.execution_binding_id,
    )
    seal_path = tmp_path / "data" / "manifests" / "anomaly_increment_r2_scoring_seal.json"

    with pytest.raises(Stage4ScoringNotAuthorizedError, match="sole repository-root path"):
        authorize_stage4_target_access(
            tmp_path,
            protocol,
            scoring_seal_path=seal_path,
            attempt_ledger_path=attempt_path,
            target_read_ledger_path=shadow,
            repository_adapter=_RepositoryAdapter(_repository()),
        )


def test_capability_cannot_be_transferred_to_shadow_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic-target-for-capability-transfer"
    authorization, _, _, _ = _authorize(monkeypatch, tmp_path, payload)
    shadow = tmp_path / "shadow" / "empty_target_ledger.json"
    initialize_stage4_ledger(
        shadow,
        kind="target_read",
        execution_binding_id=authorization.execution_binding_id,
    )
    transferred = replace(authorization, target_read_ledger_path=shadow)

    with pytest.raises(Stage4ScoringNotAuthorizedError, match="sole repository-root path"):
        consume_authorized_stage4_target(
            tmp_path,
            transferred,
            protocol={},
            operation_id="shadow-capability-target-read",
            consumer=lambda value: value,
        )


def test_first_authorized_consumption_reads_synthetic_target_once_and_records_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"purely-synthetic-target"
    authorization, _, _, target_ledger = _authorize(monkeypatch, tmp_path, payload)
    path = tmp_path / "synthetic" / "target.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    calls = 0
    original = target_access_module._read_target_bytes_once

    def counted(candidate: Path) -> bytes:
        nonlocal calls
        calls += 1
        return original(candidate)

    monkeypatch.setattr(target_access_module, "_read_target_bytes_once", counted)
    consumed = consume_authorized_stage4_target(
        tmp_path,
        authorization,
        protocol={},
        operation_id="first-target-consumption",
        consumer=lambda value: value.decode("ascii"),
    )

    assert consumed.value == payload.decode("ascii")
    assert calls == 1
    ledger = read_stage4_ledger(target_ledger)
    assert ledger.operation_count == 1 and ledger.succeeded_count == 1
    with pytest.raises(Stage4OperationAlreadyConsumedError):
        consume_authorized_stage4_target(
            tmp_path,
            authorization,
            protocol={},
            operation_id="second-target-consumption",
            consumer=lambda value: value,
        )
    assert calls == 1


def test_target_consumer_failure_is_durable_and_cannot_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic-bad-parser-input"
    authorization, _, _, target_ledger = _authorize(monkeypatch, tmp_path, payload)
    path = tmp_path / "synthetic" / "target.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    def fail(_: bytes) -> object:
        raise ValueError("synthetic parser failure")

    with pytest.raises(ValueError, match="synthetic parser"):
        consume_authorized_stage4_target(
            tmp_path,
            authorization,
            protocol={},
            operation_id="failed-target-consumption",
            consumer=fail,
        )
    ledger = read_stage4_ledger(target_ledger)
    assert ledger.operation_count == 1 and ledger.failed_count == 1
    assert ledger.records[0].failure_code == "value_error"
    with pytest.raises(Stage4OperationAlreadyConsumedError):
        consume_authorized_stage4_target(
            tmp_path,
            authorization,
            protocol={},
            operation_id="retry-is-forbidden",
            consumer=lambda value: value,
        )


def test_target_hash_mismatch_is_registered_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, _, _, target_ledger = _authorize(monkeypatch, tmp_path, b"expected")
    path = tmp_path / "synthetic" / "target.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed")

    with pytest.raises(target_access_module.Stage4TargetAccessError, match="differ"):
        consume_authorized_stage4_target(
            tmp_path,
            authorization,
            protocol={},
            operation_id="identity-mismatch",
            consumer=lambda value: value,
        )

    ledger = read_stage4_ledger(target_ledger)
    assert ledger.failed_count == 1
    assert ledger.records[0].failure_code == "target_access_contract_failure"


def test_locked_test_is_unconditionally_forbidden() -> None:
    for scope in ("locked-test", "stage9-locked_test_only", "test"):
        with pytest.raises(Stage4LockedTestForbiddenError):
            require_stage4_execution_scope(scope)
    with pytest.raises(Stage4LockedTestForbiddenError):
        forbid_stage4_locked_test_access()


def test_target_leakage_boundary_is_visible_in_ast_and_only_target_module_opens_target() -> None:
    root = Path("src/seismoflux/anomaly_increment")
    stage4_files = (
        *sorted(root.rglob("*.py")),
        *sorted(Path("scripts").glob("build_stage4_*.py")),
    )
    dangerous_calls = {
        "exists",
        "open",
        "read_bytes",
        "read_parquet",
        "read_table",
        "sha256_file",
        "stat",
    }
    for path in stage4_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "earthquake_event.parquet" not in source
        if path.name == "target_access.py":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            if call_name not in dangerous_calls:
                continue
            segment = (ast.get_source_segment(source, node) or "").casefold()
            direct_target_reference = (
                "earthquake_target" in segment
                or "target_path" in segment
                or "target_file" in segment
            ) and "contract" not in segment
            assert not direct_target_reference, (
                f"direct target filesystem call escaped sole entrance: {path}:{node.lineno}"
            )
    qualification_source = (root / "qualification.py").read_text(encoding="utf-8")
    assert 'if input_id == "earthquake_target":\n            continue' in qualification_source
    target_source = (root / "target_access.py").read_text(encoding="utf-8")
    tree = ast.parse(target_source)
    open_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
    ]
    assert len(open_calls) == 1


def _git_runner(*, stale_remote: bool = False) -> Any:
    branch_ref = "refs/heads/codex/stage4-increment-scoring-code"

    def run(command: tuple[str, ...], _: Path) -> CommandResult:
        if command == ("git", "rev-parse", "--is-inside-work-tree"):
            return CommandResult(0, "true\n")
        if command == ("git", "rev-parse", "--verify", "HEAD^{commit}"):
            return CommandResult(0, f"{CODE_COMMIT}\n")
        if command == ("git", "status", "--porcelain=v1", "--untracked-files=all"):
            return CommandResult(0, "")
        if command == ("git", "symbolic-ref", "--quiet", "--short", "HEAD"):
            return CommandResult(0, "codex/stage4-increment-scoring-code\n")
        if command == (
            "git",
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return CommandResult(0, "origin/codex/stage4-increment-scoring-code\n")
        if command == ("git", "rev-parse", "--verify", "@{upstream}^{commit}"):
            return CommandResult(0, f"{CODE_COMMIT}\n")
        if command == (
            "git",
            "rev-parse",
            "--verify",
            f"refs/tags/{PROTOCOL_TAG}^{{tag}}",
        ):
            return CommandResult(0, f"{PROTOCOL_TAG_OBJECT}\n")
        if command == (
            "git",
            "rev-parse",
            "--verify",
            f"refs/tags/{PROTOCOL_TAG}^{{commit}}",
        ):
            return CommandResult(0, f"{PROTOCOL_COMMIT}\n")
        if command == (
            "git",
            "merge-base",
            "--is-ancestor",
            PROTOCOL_COMMIT,
            CODE_COMMIT,
        ):
            return CommandResult(0)
        if command == (
            "git",
            "rev-parse",
            "--verify",
            f"refs/tags/{SCORING_TAG}^{{tag}}",
        ):
            return CommandResult(0, f"{SCORING_TAG_OBJECT}\n")
        if command == (
            "git",
            "rev-parse",
            "--verify",
            f"refs/tags/{SCORING_TAG}^{{commit}}",
        ):
            return CommandResult(0, f"{CODE_COMMIT}\n")
        if command[:3] == ("git", "rev-parse", "--verify"):
            revision = command[3]
            for path in STAGE4_FROZEN_PROTOCOL_PATHS:
                if revision in {f"{PROTOCOL_TAG}:{path}", f"HEAD:{path}"}:
                    return CommandResult(0, f"{'8' * 40}\n")
        if command == ("git", "remote", "get-url", "origin"):
            return CommandResult(0, "https://github.com/Justin-147/SeismoFlux.git\n")
        if command[:3] == ("git", "ls-remote", "--refs"):
            remote_commit = "f" * 40 if stale_remote else CODE_COMMIT
            return CommandResult(
                0,
                "\n".join(
                    (
                        f"{remote_commit}\t{branch_ref}",
                        f"{PROTOCOL_TAG_OBJECT}\trefs/tags/{PROTOCOL_TAG}",
                        f"{SCORING_TAG_OBJECT}\trefs/tags/{SCORING_TAG}",
                    )
                )
                + "\n",
            )
        return CommandResult(1, stderr=f"unexpected command: {command!r}")

    return run


def test_git_adapter_requires_protocol_scoring_tags_and_live_remote_identity(
    tmp_path: Path,
) -> None:
    evidence = GitStage4RepositoryAdapter(
        runner=_git_runner(),
        public_probe=_PublicProbe(),
    ).observe(
        tmp_path,
        protocol_tag=PROTOCOL_TAG,
        scoring_code_tag=SCORING_TAG,
    )

    assert evidence == _repository()
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="live public branch"):
        GitStage4RepositoryAdapter(
            runner=_git_runner(stale_remote=True),
            public_probe=_PublicProbe(),
        ).observe(
            tmp_path,
            protocol_tag=PROTOCOL_TAG,
            scoring_code_tag=SCORING_TAG,
        )


def test_cli_generate_and_check_are_target_unread_and_fail_after_ledger_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic-target"
    protocol = _protocol(hashlib.sha256(payload).hexdigest())
    inputs = _score_blind_inputs(protocol)
    qualification = _qualification(protocol, inputs)
    preflight_receipt = _preflight_receipt(protocol, inputs)
    qualification_path = (
        tmp_path
        / "data"
        / "interim"
        / "stage4"
        / "anomaly_increment_r2"
        / "scoring_qualification.json"
    )
    logical_replay_path = (
        tmp_path
        / "data"
        / "interim"
        / "stage4"
        / "anomaly_increment_r2"
        / "logical_identity_worker_replay.json"
    )
    logical_replay_path.parent.mkdir(parents=True, exist_ok=True)
    logical_replay_path.write_text(
        json.dumps(
            _logical_replay_audit_document(protocol, inputs),
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_qualification(qualification_path, qualification)
    preflight_path = tmp_path.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(preflight_receipt.as_mapping(), sort_keys=True),
        encoding="utf-8",
    )
    attempt_path = tmp_path.joinpath(*Path(STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH).parts)
    target_path = tmp_path.joinpath(*Path(STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH).parts)
    monkeypatch.setattr(seal_script, "observe_score_blind_inputs", lambda *_: inputs)
    adapter = _RepositoryAdapter(_repository())

    generated = seal_script.generate(
        tmp_path,
        protocol,
        qualification_path=qualification_path,
        preflight_receipt_path=preflight_path,
        attempt_ledger_path=attempt_path,
        target_read_ledger_path=target_path,
        repository_adapter=adapter,
    )
    checked = seal_script.check(
        tmp_path,
        protocol,
        qualification_path=qualification_path,
        preflight_receipt_path=preflight_path,
        attempt_ledger_path=attempt_path,
        target_read_ledger_path=target_path,
        repository_adapter=adapter,
    )

    assert generated["target_read_count"] == 0
    assert checked["verified"] is True
    assert not (tmp_path / "synthetic" / "target.bin").exists()
    with pytest.raises(ValueError, match="frozen R2 path"):
        seal_script.check(
            tmp_path,
            protocol,
            qualification_path=(
                tmp_path
                / "data"
                / "interim"
                / "stage4"
                / "anomaly_increment"
                / "scoring_qualification.json"
            ),
            preflight_receipt_path=preflight_path,
            attempt_ledger_path=attempt_path,
            target_read_ledger_path=target_path,
            repository_adapter=adapter,
        )
    binding = stage4_execution_binding_id(_repository(), inputs, qualification)
    reserve_stage4_operation(
        target_path,
        kind="target_read",
        execution_binding_id=binding,
        operation_id="already-consumed",
        scope=STAGE4_TARGET_SCOPE,
        authorization_id=EVIDENCE_SHA256,
    )
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="zero attempts"):
        seal_script.check(
            tmp_path,
            protocol,
            qualification_path=qualification_path,
            preflight_receipt_path=preflight_path,
            attempt_ledger_path=attempt_path,
            target_read_ledger_path=target_path,
            repository_adapter=adapter,
        )


def test_qualification_script_derives_real_junit_preflight_and_gpu_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol("0" * 64)
    inputs = _score_blind_inputs(protocol)
    receipt = _preflight_receipt(protocol, inputs)
    preflight_path = tmp_path.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(receipt.as_mapping(), sort_keys=True),
        encoding="utf-8",
    )
    stage4_junit = (
        tmp_path
        / "data"
        / "interim"
        / "stage4"
        / "anomaly_increment_r2"
        / "qualification_stage4.junit.xml"
    )
    full_junit = (
        tmp_path
        / "data"
        / "interim"
        / "stage4"
        / "anomaly_increment_r2"
        / "qualification_full_non_target.junit.xml"
    )
    stage4_junit.parent.mkdir(parents=True, exist_ok=True)
    stage4_junit.write_bytes(_pytest_xml(full=False))
    full_junit.write_bytes(_pytest_xml(full=True))
    qualification_path = (
        tmp_path
        / "data"
        / "interim"
        / "stage4"
        / "anomaly_increment_r2"
        / "scoring_qualification.json"
    )
    logical_replay_path = (
        tmp_path
        / "data"
        / "interim"
        / "stage4"
        / "anomaly_increment_r2"
        / "logical_identity_worker_replay.json"
    )
    logical_replay_path.write_text(
        json.dumps(
            _logical_replay_audit_document(protocol, inputs),
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification_script,
        "observe_score_blind_inputs",
        lambda *_: inputs,
    )

    def git_head(_command: tuple[str, ...], _cwd: Path) -> CommandResult:
        return CommandResult(returncode=0, stdout=f"{CODE_COMMIT}\n")

    generated = qualification_script.generate(
        tmp_path,
        protocol,
        stage4_junit_path=stage4_junit,
        full_junit_path=full_junit,
        preflight_receipt_path=preflight_path,
        qualification_path=qualification_path,
        logical_replay_audit_path=logical_replay_path,
        git_runner=git_head,
    )
    checked = qualification_script.check(
        tmp_path,
        protocol,
        stage4_junit_path=stage4_junit,
        full_junit_path=full_junit,
        preflight_receipt_path=preflight_path,
        qualification_path=qualification_path,
        logical_replay_audit_path=logical_replay_path,
        git_runner=git_head,
    )

    assert generated["gpu_requested"] is True
    assert generated["gpu_status"] == "blocked_no_frozen_backend"
    assert generated["formal_backend"] == "cpu_float64"
    assert generated["formal_preflight_receipt_sha256"] == receipt.content_sha256
    assert checked["verified"] is True
    assert not (tmp_path / "synthetic" / "target.bin").exists()
    with pytest.raises(ValueError, match="frozen R2 path"):
        qualification_script.generate(
            tmp_path,
            protocol,
            stage4_junit_path=(tmp_path / "data" / "interim" / "stage4.junit.xml"),
            full_junit_path=full_junit,
            preflight_receipt_path=preflight_path,
            qualification_path=qualification_path,
            logical_replay_audit_path=logical_replay_path,
            git_runner=git_head,
        )
    with pytest.raises(ValueError, match="frozen R2 path"):
        qualification_script.generate(
            tmp_path,
            protocol,
            stage4_junit_path=stage4_junit,
            full_junit_path=full_junit,
            preflight_receipt_path=preflight_path,
            qualification_path=(tmp_path / "data" / "interim" / "qualification.json"),
            logical_replay_audit_path=logical_replay_path,
            git_runner=git_head,
        )
