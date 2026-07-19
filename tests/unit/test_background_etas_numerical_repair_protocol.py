from __future__ import annotations

import ast
import base64
import hashlib
import json
import subprocess
import types
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import shapely
import shapely._geometry as shapely_geometry
import shapely.creation as shapely_creation
import shapely.geometry.base as shapely_base
import shapely.lib as shapely_lib
import shapely.predicates as shapely_predicates
import yaml
from shapely.geometry import Point
from shapely.geometry import point as shapely_point

from seismoflux.background.etas_fit import ETASParameterBounds, optimizer_start
from seismoflux.background.grid import GridSpec, cell_id
from seismoflux.background.randomness import SeedContext

PROTOCOL_PATH = Path("configs/background_etas_numerical_repair.yaml")
START_MANIFEST_PATH = Path("data/manifests/etas_numerical_repair_start_manifest.json")
PROTOCOL_DOCUMENT_PATH = Path("docs/background_etas_numerical_repair_protocol.md")
PROTOCOL_ACCEPTANCE_PATH = Path("docs/phase2_etas_numerical_repair_protocol_acceptance.md")
RESTART_HANDOFF_PATH = Path("docs/restart_handoff_2026-07-19_stage2_etas_repair_protocol.md")
STAGE4_R2_PROTOCOL_PATH = Path("configs/anomaly_increment_r2.yaml")
R1_PROTOCOL_COMMIT = "da916454c908e0cbe4a7526f56a8f837331a3c7c"

SNAPSHOT_ORDER = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
FIT_ENDS = (
    "2004-12-31T16:00:00Z",
    "2009-12-31T16:00:00Z",
    "2014-12-31T16:00:00Z",
    "2019-12-31T16:00:00Z",
    "2023-06-30T16:00:00Z",
)
SUPPORT_IDS = (
    "local-support-f06e7c7496ea2357",
    "local-support-eaee903b28c55ace",
    "local-support-f86126dbec5bb79b",
    "local-support-788851371baf0e3b",
    "local-support-f6816ab6c6581306",
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate YAML mapping keys instead of silently keeping the last value."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = cast(Any, loader).construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = cast(Any, loader).construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
COMPENSATOR_DOMAIN_IDS = (
    "154062341fe6e2b68625f90b832219617ce5c61b418a3ff31f4df36f50e8fb1f",
    "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55",
    "8e41c306592739c634ca85bf2540dbcfeb2086e64e908a41b4f90db7cd1f94f1",
    "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55",
    "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55",
)
RETAINED_AREA_FRACTIONS = (0.9734474900209907, 1.0, 0.9972058595099415, 1.0, 1.0)
PARENT_ROLES = (
    "include_prevalidated_eligible_unsupported_history",
    "supported_and_external_buffer_history",
    "include_prevalidated_eligible_unsupported_history",
    "supported_and_external_buffer_history",
    "supported_and_external_buffer_history",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger_entry_sha256(entry: dict[str, Any]) -> str:
    return _canonical_payload_sha256(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )


def _ast_sha256(node: ast.AST) -> str:
    payload = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recursive_code_names(code: types.CodeType) -> set[str]:
    """Collect global/attribute names from a function and all nested code objects."""

    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_recursive_code_names(constant))
    return names


def _without_named_class_method(
    source_text: str,
    *,
    class_name: str,
    method_name: str,
) -> tuple[ast.Module, str]:
    """Remove one allowed method while preserving every other source byte and AST node."""

    module = ast.parse(source_text)
    matching_classes = [
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(matching_classes) == 1
    class_node = matching_classes[0]
    matching_methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
    ]
    assert len(matching_methods) == 1
    method_node = matching_methods[0]
    assert method_node.end_lineno is not None

    first_line = min(
        [method_node.lineno, *(decorator.lineno for decorator in method_node.decorator_list)]
    )
    source_lines = source_text.splitlines(keepends=True)
    source_without_method = "".join(
        source_lines[: first_line - 1] + source_lines[method_node.end_lineno :]
    )
    class_node.body = [node for node in class_node.body if node is not method_node]
    return module, source_without_method


def test_protocol_yaml_loader_rejects_duplicate_mapping_keys() -> None:
    duplicate = "root:\n  value: 1\n  value: 2\n"
    try:
        yaml.load(duplicate, Loader=_UniqueKeySafeLoader)
    except yaml.constructor.ConstructorError:
        return
    raise AssertionError("duplicate YAML keys must fail closed")


def test_all_declared_dotted_protocol_refs_resolve() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    resolved: list[tuple[str, str]] = []

    def resolve(reference: str) -> None:
        current: object = protocol
        for component in reference.split("."):
            assert isinstance(current, dict), reference
            assert component in current, (reference, component)
            current = current[component]

    def walk(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for raw_key, item in value.items():
                key = str(raw_key)
                item_path = (*path, key)
                if isinstance(item, str) and (key.endswith("_ref") or key.endswith("_schema_ref")):
                    resolve(item)
                    resolved.append((".".join(item_path), item))
                if key.endswith("_refs") or key.endswith("_schema_refs"):
                    assert isinstance(item, dict), item_path
                    for nested_key, nested_reference in item.items():
                        assert isinstance(nested_reference, str), (*item_path, nested_key)
                        resolve(nested_reference)
                        resolved.append((".".join((*item_path, str(nested_key))), nested_reference))
                walk(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(protocol)
    assert len(resolved) >= 110


def test_repair_protocol_is_independent_target_blind_and_not_yet_executed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)

    assert protocol["protocol_version"] == "0.2.2"
    assert protocol["protocol_revision"] == "r2"
    assert protocol["stage"] == "2-ETAS-R"
    assert protocol["status"] == "preregistered_target_blind_before_any_repair_fit"
    assert protocol["preregistered_on"] == "2026-07-17"
    assert protocol["revised_on"] == "2026-07-19"
    assert protocol["revision_reason"] == (
        "freeze_qualification_attempt_local_staged_public_paths_and_byte_exact_"
        "materialization_without_changing_any_scientific_input_or_fit_rule"
    )
    publication = protocol["publication"]
    assert publication["protocol_tag"] == "v0.2.2-background-etas-repair-protocol-r2"
    assert publication["qualification_code_tag"] == "v0.2.2-background-etas-repair-code"
    assert publication["qualification_result_tag"] == (
        "v0.2.2-background-etas-numerical-qualification"
    )
    assert publication["comparator_adapter_code_tag"] == (
        "v0.2.2-background-etas-comparator-adapter"
    )
    assert publication["comparator_receipt_tag"] == "v0.2.2-background-etas-comparator"
    assert publication["qualification_result_tag_freezes_evaluable_or_not_evaluable"] is True
    assert publication["negative_result_requires_same_qualification_result_tag"] is True
    assert publication["adapter_code_tag_allowed_only_after_positive_qualification_result_tag"]
    assert publication["new_stage4_revision_requires_comparator_receipt_tag"] is True
    assert protocol["repair_code_scope_from_protocol_tag"]["comparison_base"] == (
        "v0.2.2-background-etas-repair-protocol-r2"
    )
    assert publication["exact_order"] == [
        "protocol_commit_push_and_remote_tag_verification",
        "repair_code_and_tests_commit_push_and_remote_tag_verification",
        "qualification_execution_and_positive_or_negative_result_commit_push_and_remote_tag_verification",
        "positive_only_adapter_code_and_tests_commit_push_and_remote_tag_verification",
        "positive_only_adapter_artifact_and_global_receipt_commit_push_and_remote_tag_verification",
        "positive_only_new_stage4_revision",
    ]
    assert protocol["parent"]["etas_status"] == "not_evaluable"
    assert protocol["parent"]["etas_primary_snapshot_converged_starts"] == [0] * 5
    assert protocol["parent"]["parent_scientific_result_may_not_be_reinterpreted"] is True

    target_blind = protocol["target_blindness"]
    assert target_blind["mode"] == "fit_only_before_any_stage4_target_read"
    assert target_blind["stage4_formal_target_consumer_read_count_required"] == 0
    assert target_blind["stage4_assessment_row_materialization_count_required"] == 0
    assert target_blind["stage2_causal_fit_source_access_must_be_separately_ledgered"] is True
    assert target_blind["locked_test_run_required"] is False
    for forbidden in (
        "anomaly_feature_read_allowed",
        "anomaly_result_read_allowed",
        "stage4_formal_target_read_allowed",
        "stage4_score_read_allowed",
        "stage9_locked_test_allowed",
        "stage2_holdout_assessment_interval_construction_allowed",
        "stage2_assessment_target_event_read_allowed",
        "information_gain_computation_allowed",
        "score_id_creation_allowed",
        "model_selection_allowed",
        "prior_stage2_scores_as_tuning_inputs_allowed",
        "parameter_bound_or_threshold_relaxation_allowed",
        "new_or_replacement_optimizer_starts_allowed",
        "failed_snapshot_omission_allowed",
    ):
        assert target_blind[forbidden] is False

    assert protocol["locked_test"] == {
        "run": False,
        "target_count": None,
        "score_ids": [],
        "artifact_ids": [],
        "result": None,
    }


def test_all_repair_input_bindings_match_current_frozen_bytes() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    bindings = protocol["input_bindings"]
    for name, binding in bindings.items():
        path = Path(binding["path"])
        assert path.is_file(), name
        assert _sha256(path) == binding["sha256"], name

    assert bindings["parent_model_registry"]["role"] == (
        "provenance_only_not_read_by_qualification"
    )
    assert bindings["local_support_fold_manifest"]["role"] == (
        "provenance_only_not_read_by_fit_or_qualification"
    )

    local = protocol["local_restricted_input_identities"]
    assert local["ci_policy"] == "verify_frozen_metadata_only_and_never_require_local_files"
    assert local["local_acceptance_policy"] == (
        "every_file_must_exist_and_match_after_code_tag_and_before_execution"
    )
    for name, binding in local.items():
        if name in {"ci_policy", "local_acceptance_policy"}:
            continue
        path_text = binding.get("path", binding.get("path_from_parent_config"))
        assert isinstance(path_text, str) and path_text
        assert isinstance(binding["sha256"], str) and len(binding["sha256"]) == 64

    source = local["stage2_catalog_source"]
    assert source["same_physical_file_as_later_stage4_catalog"] is True
    assert source["allowed_action_after_code_tag"] == "stage2_causal_fit_source_query_only"
    assert source["stage4_formal_target_consumer_action_forbidden"] is True
    assert source["source_to_internal_field_mapping"] == {
        "event_id": "physical_event_id",
        "origin_time_utc": "origin_time",
        "available_at": "available_time",
        "longitude": "longitude",
        "latitude": "latitude",
        "magnitude": "magnitude",
        "inside_study_area": "inside_study_area",
    }
    assert local["frozen_kde_payload"]["role"] == (
        "provenance_only_not_read_by_fit_or_qualification"
    )


def test_five_fit_only_snapshots_are_frozen_without_assessment_intervals() -> None:
    snapshots = _load_yaml(PROTOCOL_PATH)["snapshots"]

    assert tuple(snapshots["order"]) == SNAPSHOT_ORDER
    assert snapshots["fit_start_utc"] == "2000-01-01T00:00:00Z"
    assert snapshots["history_start_utc"] == "1970-01-01T00:00:00Z"
    assert snapshots["common_mc"] == 4.0
    entries = snapshots["entries"]
    assert tuple(item["snapshot_id"] for item in entries) == SNAPSHOT_ORDER
    assert tuple(item["fit_end_utc"] for item in entries) == FIT_ENDS
    assert tuple(item["support_id"] for item in entries) == SUPPORT_IDS
    assert tuple(item["compensator_domain_id"] for item in entries) == COMPENSATOR_DOMAIN_IDS
    assert tuple(item["retained_area_fraction"] for item in entries) == (RETAINED_AREA_FRACTIONS)
    assert tuple(item["parent_role"] for item in entries) == PARENT_ROLES
    assert all("assessment_start_utc" not in item for item in entries)
    assert all("assessment_end_utc" not in item for item in entries)

    bundle = _load_yaml(PROTOCOL_PATH)["fit_input_bundle"]
    assert bundle["source_query_may_not_construct_or_return_assessment_rows"] is True
    assert bundle["column_projection_and_timestamp_predicate_must_be_applied_at_reader_boundary"]
    assert bundle["full_source_table_materialization_then_filter_forbidden"] is True
    assert bundle["reader_spy_must_prove_no_later_or_unavailable_row_was_materialized"]
    assert bundle["local_bundle_classification"] == "local_restricted"
    assert (
        bundle[
            "post_protocol_freeze_code_tag_required_before_any_new_qualification_source_open_stat_hash_query_or_bundle_inspection"
        ]
        is True
    )
    disclosed_probe = bundle["prefreeze_disclosed_source_probe"]
    probe_guard = (
        "no_additional_probe_allowed_after_protocol_freeze_before_remote_code_tag_verification"
    )
    assert disclosed_probe[probe_guard] is True
    assert {key: value for key, value in disclosed_probe.items() if key != probe_guard} == {
        "purpose": "protocol_drafting_verification_of_frozen_file_identity",
        "action": "one_read_only_file_level_sha256_check",
        "decoded_or_returned_row_count": 0,
        "fit_or_assessment_cohort_constructed": False,
        "stage4_formal_target_consumer_count": 0,
        "stage4_assessment_row_materialization_count": 0,
        "qualification_attempt_or_fit_input_bundle_created": False,
    }
    assert bundle["event_order"] == "origin_time_then_physical_event_id"
    selection = bundle["selection"]
    assert selection["fit_events"] == (
        "inside_snapshot_supported_domain_and_magnitude_gte_common_mc_and_origin_gt_fit_start_"
        "and_origin_lte_fit_end_and_available_at_lte_fit_end"
    )
    assert selection["parent_time"] == (
        "origin_gte_max_history_start_and_fit_start_minus_3650_days_and_origin_lte_fit_end_and_"
        "available_at_lte_fit_end"
    )
    assert selection["later_or_unavailable_event_count_required"] == 0
    expected_forbidden = {
        "event_id",
        "physical_event_id",
        "event_coordinates",
        "origin_time",
        "available_time",
        "longitude",
        "latitude",
        "projected_x",
        "projected_y",
        "assessment_event_count",
        "assessment_event_ids",
        "model_score",
        "information_gain",
        "score_id",
    }
    assert expected_forbidden == set(bundle["public_manifest_forbidden_fields"])
    counts = bundle["expected_parent_result_counts"]
    assert tuple(counts["snapshot_order"]) == SNAPSHOT_ORDER
    assert counts["fit_event_count"] == [385, 1287, 1828, 2342, 2734]
    assert counts["parent_event_count"] == [1875, 2874, 3592, 4263, 4802]
    assert counts["immigrant_kde_training_event_count"] == [3189, 4182, 4722, 5237, 5629]
    assert bundle["scientific_fit_input_sha256"]["must_be_bound_by_every_parameter_snapshot"]
    scientific_fields = [
        "snapshot_id",
        "fit_interval",
        "support_id",
        "compensator_domain_id",
        "common_mc_hex",
        "aki_b_hex",
        "beta_hex",
        "model_spec",
        "parameter_bounds",
        "optimizer_options",
        "stability_thresholds",
        "exact_start_vectors",
        "ordered_fit_events",
        "ordered_parent_events_and_roles",
        "ordered_quadrature_containers",
        "immigrant_kde_payload",
    ]
    assert bundle["scientific_fit_input_sha256"]["includes"] == scientific_fields
    assert (
        bundle["scientific_fit_input_record_schemas"]["scientific_fit_input_payload_fields_exact"]
        == scientific_fields
    )
    assert bundle["persistence"]["content_addressed_no_overwrite"] is True

    ledger = bundle["source_access_ledger"]
    assert ledger["path"] == "data/manifests/etas_numerical_repair_source_access_ledger.json"
    assert ledger["classification"] == "local_restricted_gitignored"
    assert ledger["selected_complete_acquisition_attempt_required_completed_action_counts"] == {
        "stage2_fit_source_metadata": 1,
        "stage2_fit_source_rows": 5,
    }
    assert ledger["selected_complete_acquisition_attempt_allowed_actions_exact"] == [
        "stage2_fit_source_metadata",
        "stage2_fit_source_rows",
    ]
    assert ledger["selected_complete_acquisition_attempt_exact_access_pair_count"] == 6
    assert ledger["selected_complete_acquisition_attempt_exact_ledger_entry_count"] == 12
    access_pairs = ledger["selected_complete_acquisition_attempt_ordered_access_pairs"]
    assert [item["pair_index"] for item in access_pairs] == list(range(6))
    assert [item["snapshot_id_or_null"] for item in access_pairs] == [None, *SNAPSHOT_ORDER]
    assert ledger["event_type_values_exact"] == ["intent", "completed", "aborted"]
    assert ledger["any_missing_extra_alias_duplicate_or_unknown_entry_field_forbidden"]
    reader_contract = bundle["reader_boundary_contract"]
    projection = dict(reader_contract["projection_payload"])
    projection_sha256 = projection.pop("canonical_sha256")
    assert _canonical_payload_sha256(projection) == projection_sha256
    predicates = reader_contract["predicate_payload_and_sha256_by_snapshot"]
    assert tuple(predicates) == SNAPSHOT_ORDER
    for index, snapshot_id in enumerate(SNAPSHOT_ORDER, start=1):
        predicate = dict(predicates[snapshot_id])
        predicate_sha256 = predicate.pop("canonical_sha256")
        assert _canonical_payload_sha256(predicate) == predicate_sha256
        assert access_pairs[index]["reader_projection_sha256_or_null"] == projection_sha256
        assert access_pairs[index]["reader_predicate_sha256_or_null"] == predicate_sha256
    pair_payload = {"ordered_access_pairs": access_pairs}
    pair_identity = bundle["public_source_access_receipt"][
        "selected_attempt_ordered_access_pair_identity"
    ]
    assert _canonical_payload_sha256(pair_payload) == pair_identity["reference_sha256"]
    assert ledger[
        "global_ledger_action_counts_may_exceed_selected_attempt_counts_after_interruption"
    ]
    two_phase = ledger["source_access_two_phase_protocol"]
    assert two_phase["intent_entry_fsynced_before_any_source_open_stat_hash_or_reader_call"]
    assert two_phase["existing_entries_may_not_be_deleted_truncated_or_rewritten"] is True
    assert two_phase["sealed_bundle_retry_must_reuse_verified_bundle_without_reopening_source"]
    assert two_phase["exactly_one_intent_and_exactly_one_completed_xor_aborted_terminal_per_pair"]
    assert two_phase["terminal_must_bind_intent_entry_sha256"] is True
    assert two_phase[
        "terminal_must_match_intent_action_snapshot_projection_predicate_code_commit_and_source_sha256"
    ]
    assert ledger["zero_counts_required_on_every_entry"] == {
        "stage4_formal_target_consumer_count_after": 0,
        "stage4_assessment_row_materialization_count_after": 0,
    }
    assert "full_source_table_materialization_then_filter" in ledger["forbidden_actions"]
    receipt = bundle["public_source_access_receipt"]
    assert receipt["stage4_formal_target_consumer_count_required"] == 0
    assert receipt["stage4_assessment_row_materialization_count_required"] == 0
    assert "local_ledger_content_sha256" in receipt["fields_exact"]

    replay = bundle["parent_replay_membership_equivalence"]
    assert replay["source_commit"] == "34fa7b4a491a062ff6e86daecf5568539661b42f"
    assert replay["counts_only_equivalence_forbidden"] is True
    assert replay["repair_and_parent_replay_ordered_identities_must_match_exactly"] is True
    assert replay["parent_replay_scientific_fit_input_sha256_required"] is True
    assert replay[
        "repair_scientific_fit_input_sha256_must_equal_parent_replay_scientific_fit_input_sha256"
    ]
    assert replay["full_value_payloads_required_by_snapshot"] == scientific_fields
    assert replay["full_value_payloads_must_equal_scientific_fit_input_includes_exactly"]
    assert replay[
        "parent_replay_must_construct_the_same_canonical_payload_schema_and_identical_bytes_before_hashing"
    ]
    assert replay["frozen_source_blobs"] == {
        "src/seismoflux/background/pipeline_local_support_etas.py": (
            "11b0b70ff900694780281e8da21123269c6463f1"
        ),
        "src/seismoflux/background/pipeline_poisson.py": (
            "63eab3bf4a62a0052ac05f287b9941fff5a946e5"
        ),
    }


def test_source_access_ledger_hash_chain_reference_is_non_self_referential() -> None:
    ledger = _load_yaml(PROTOCOL_PATH)["fit_input_bundle"]["source_access_ledger"]
    reference = ledger["hash_chain_reference_vector"]
    entry_0: dict[str, Any] = {
        "sequence": 0,
        "occurred_at_utc": "2026-07-17T00:00:00Z",
        "previous_entry_sha256_or_null": None,
        "intent_entry_sha256_or_null": None,
        "acquisition_attempt_id": "acq-0001",
        "access_id": "access-0001",
        "event_type": "intent",
        "code_tag_commit": "0" * 40,
        "source_sha256": "1" * 64,
        "action": "stage2_fit_source_metadata",
        "snapshot_id_or_null": None,
        "reader_projection_sha256_or_null": None,
        "reader_predicate_sha256_or_null": None,
        "materialized_row_count_or_null": None,
        "returned_row_count_or_null": None,
        "stage4_formal_target_consumer_count_after": 0,
        "stage4_assessment_row_materialization_count_after": 0,
    }
    entry_0["entry_sha256"] = _ledger_entry_sha256(entry_0)
    assert entry_0["entry_sha256"] == reference["entry_0_sha256"]

    entry_1: dict[str, Any] = {
        "sequence": 1,
        "occurred_at_utc": "2026-07-17T00:00:01Z",
        "previous_entry_sha256_or_null": entry_0["entry_sha256"],
        "intent_entry_sha256_or_null": entry_0["entry_sha256"],
        "acquisition_attempt_id": "acq-0001",
        "access_id": "access-0001",
        "event_type": "completed",
        "code_tag_commit": "0" * 40,
        "source_sha256": "1" * 64,
        "action": "stage2_fit_source_metadata",
        "snapshot_id_or_null": None,
        "reader_projection_sha256_or_null": None,
        "reader_predicate_sha256_or_null": None,
        "materialized_row_count_or_null": 0,
        "returned_row_count_or_null": 0,
        "stage4_formal_target_consumer_count_after": 0,
        "stage4_assessment_row_materialization_count_after": 0,
    }
    entry_1["entry_sha256"] = _ledger_entry_sha256(entry_1)
    assert entry_1["entry_sha256"] == reference["entry_1_sha256"]
    payload = {"schema_version": 1, "entries": [entry_0, entry_1]}
    assert _canonical_payload_sha256(payload) == reference["final_ledger_content_sha256"]

    mutated_0 = dict(entry_0)
    mutated_0["returned_row_count_or_null"] = 1
    mutated_0["entry_sha256"] = _ledger_entry_sha256(mutated_0)
    assert mutated_0["entry_sha256"] != reference["entry_0_sha256"]
    mutated_1 = dict(entry_1)
    mutated_1["previous_entry_sha256_or_null"] = mutated_0["entry_sha256"]
    mutated_1["intent_entry_sha256_or_null"] = mutated_0["entry_sha256"]
    mutated_1["entry_sha256"] = _ledger_entry_sha256(mutated_1)
    assert mutated_1["entry_sha256"] != reference["entry_1_sha256"]
    mutated_payload = {"schema_version": 1, "entries": [mutated_0, mutated_1]}
    assert _canonical_payload_sha256(mutated_payload) != reference["final_ledger_content_sha256"]


def test_scientific_fit_input_integer_encoding_keeps_only_grid_indices_signed() -> None:
    schemas = _load_yaml(PROTOCOL_PATH)["fit_input_bundle"]["scientific_fit_input_record_schemas"]
    assert schemas["quadrature_cell_integer_field_types"] == {
        "row": "strict_base10_integer",
        "column": "strict_base10_integer",
    }
    assert schemas["integer_encoding"] == {
        "signed_fields_exact": [
            "ordered_quadrature_containers.cells.row",
            "ordered_quadrature_containers.cells.column",
        ],
        "signed_field_encoding": "strict_base10_integer",
        "every_other_integer_field_encoding": "strict_nonnegative_base10_integer",
        "python_bool_is_not_an_integer": True,
    }
    spec = GridSpec(25.0)
    assert cell_id(spec, row=-1, column=2) == "g25000000_r-0000001_c+0000002"
    with pytest.raises(TypeError, match="row must be an integer"):
        cell_id(spec, row=True, column=2)


def test_parent_protocol_twenty_five_optimizer_starts_are_hex_exact() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    manifest = _load_json(START_MANIFEST_PATH)
    without_identity = dict(manifest)
    recorded_identity = without_identity.pop("vector_payload_sha256")
    encoded = json.dumps(
        without_identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == recorded_identity
    assert (
        recorded_identity
        == protocol["input_bindings"]["optimizer_start_manifest"]["vector_payload_sha256"]
    )

    assert manifest["seed_protocol_version"] == "0.2.1"
    assert manifest["root_seed"] == 147
    assert manifest["bit_generator"] == "numpy.random.PCG64"
    assert tuple(item["snapshot_id"] for item in manifest["snapshots"]) == SNAPSHOT_ORDER
    assert sum(len(item["starts"]) for item in manifest["snapshots"]) == 25

    bounds = ETASParameterBounds().transformed()
    for snapshot in manifest["snapshots"]:
        assert snapshot["model_id"] == f"etas/{snapshot['snapshot_id']}"
        assert [item["start_index"] for item in snapshot["starts"]] == list(range(5))
        for item in snapshot["starts"]:
            regenerated = optimizer_start(
                bounds,
                root_seed=147,
                protocol_version="0.2.1",
                model_id=snapshot["model_id"],
                start_index=item["start_index"],
            )
            assert [float(value).hex() for value in regenerated] == item["transformed_hex"]


def test_repair_does_not_widen_bounds_or_change_the_optimizer() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    model = protocol["etas_model"]
    assert model["parameter_bounds"] == {
        "background_rate_per_day": [0.01, 10.0],
        "productivity_k": [0.0001, 0.5],
        "alpha": [0.05, 2.0],
        "c_days": [0.001, 30.0],
        "p": [1.01, 2.5],
    }
    assert model["spatial_kernel"] == {
        "d_km2": 25.0,
        "q": 1.5,
        "gamma": 1.0,
        "cutoff_km": 300.0,
    }
    assert model["branching_ratio_maximum"] == 0.95

    repair = protocol["repair"]
    assert repair["affected_upper_bounds"] == {
        "background_rate_per_day": {
            "exact": "0x1.4000000000000p+3",
            "old_decoded": "0x1.4000000000001p+3",
        },
        "c_days": {
            "exact": "0x1.e000000000000p+4",
            "old_decoded": "0x1.e000000000001p+4",
        },
    }
    assert {
        "widen_physical_bounds",
        "tolerance_based_contains",
        "clip_out_of_domain_transformed_coordinates",
        "move_transformed_bounds_inward",
        "nextafter_expand_physical_domain",
    } == set(repair["forbidden_implementation"])
    assert protocol["randomness"]["seed_protocol_version"] == "0.2.1"
    assert protocol["optimizer"]["retry_with_new_seed_allowed"] is False
    assert protocol["optimizer"]["alternate_optimizer_allowed"] is False
    scope = protocol["repair_code_scope_from_protocol_tag"]
    etas_fit_rule = scope["existing_source_change_rules"]["src/seismoflux/background/etas_fit.py"]
    assert etas_fit_rule["only_symbol_allowed_to_change"] == (
        "ETASParameterBounds.from_transformed"
    )
    assert etas_fit_rule["every_other_ast_node_and_source_byte_range_must_match_protocol_tag"]
    assert scope["any_other_tracked_path_change_forbidden"] is True
    static_receipt = scope["etas_fit_static_call_graph_receipt"]
    assert static_receipt["protocol_tag_etas_fit_git_blob_oid"] == (
        "827ff2b8801c46ed5059231a5df64ce15320c0cf"
    )
    assert static_receipt["symbol_ast_sha256"] == {
        "fit_etas": "9c5da8d64c4f71424184056d2962e013703fd286a4f89ee739956ffbd5bf6caf",
        "etas_objective": "62778f464a77ff3b5ba08db421da2e807121c04b058ecd675e22b18527406b6d",
        "run_five_start_lbfgsb": "05a16648d6b5356db855db004e16d8032092de35884a72b5fd046b28f162a368",
        "three_point_gradient": "6be88b0786084ac637b5cff9aa1f2ad161ace87a594c6b4827df71c2d2db3e3d",
        "optimizer_start": "75ad95aff2c97e86ac1c526a21b18f7c78329121f818e74a97ab4f7cb4d50e87",
        "audit_stability": "1847456b17de82d88003cc0b0672600bac9fe34f8a52f6218a4d64209feec41d",
        "scipy_optimize_minimize_import": (
            "e396c4c61341edb49816428cb3f47557febea6867899a88cdaa91e2b8a786fc4"
        ),
    }
    assert static_receipt["all_other_module_ast_and_remaining_source_bytes_must_equal_protocol_tag"]
    assert static_receipt[
        "run_five_start_must_have_exactly_one_syntactic_minimize_call_inside_range_5"
    ]
    etas_fit_path = Path("src/seismoflux/background/etas_fit.py")
    baseline_bytes = subprocess.run(
        ["git", "cat-file", "blob", static_receipt["protocol_tag_etas_fit_git_blob_oid"]],
        check=True,
        capture_output=True,
    ).stdout
    assert (
        hashlib.sha256(baseline_bytes).hexdigest()
        == static_receipt["protocol_tag_etas_fit_file_sha256"]
    )
    baseline_text = baseline_bytes.decode("utf-8")
    current_text = etas_fit_path.read_bytes().decode("utf-8")
    baseline_without_allowed_ast, baseline_without_allowed_source = _without_named_class_method(
        baseline_text,
        class_name="ETASParameterBounds",
        method_name="from_transformed",
    )
    current_without_allowed_ast, current_without_allowed_source = _without_named_class_method(
        current_text,
        class_name="ETASParameterBounds",
        method_name="from_transformed",
    )
    assert current_without_allowed_source == baseline_without_allowed_source
    assert _ast_sha256(current_without_allowed_ast) == _ast_sha256(baseline_without_allowed_ast)

    baseline_module = ast.parse(baseline_text)
    current_module = ast.parse(current_text)
    baseline_symbol_nodes = {
        node.name: node
        for node in baseline_module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    current_symbol_nodes = {
        node.name: node
        for node in current_module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    baseline_import_node = next(
        node
        for node in baseline_module.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "scipy.optimize"
        and any(alias.name == "minimize" for alias in node.names)
    )
    current_import_node = next(
        node
        for node in current_module.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "scipy.optimize"
        and any(alias.name == "minimize" for alias in node.names)
    )
    for symbol in (
        "fit_etas",
        "etas_objective",
        "run_five_start_lbfgsb",
        "three_point_gradient",
        "optimizer_start",
        "audit_stability",
    ):
        expected_ast_sha256 = static_receipt["symbol_ast_sha256"][symbol]
        assert _ast_sha256(baseline_symbol_nodes[symbol]) == expected_ast_sha256
        assert _ast_sha256(current_symbol_nodes[symbol]) == expected_ast_sha256
    expected_import_ast_sha256 = static_receipt["symbol_ast_sha256"][
        "scipy_optimize_minimize_import"
    ]
    assert _ast_sha256(baseline_import_node) == expected_import_ast_sha256
    assert _ast_sha256(current_import_node) == expected_import_ast_sha256

    runtime_baseline = scope["repair_code_tag_prerequisite_public_artifacts"][
        "optimizer_runtime_baseline"
    ]
    expected_baseline_paths = [
        "src/seismoflux/background/etas_fit.py",
        "src/seismoflux/background/etas_numerical_repair.py",
        "src/seismoflux/background/etas_numerical_repair_io.py",
        "src/seismoflux/background/etas_numerical_repair_evidence.py",
        "src/seismoflux/background/visualization_etas_numerical_repair.py",
        "scripts/run_background_etas_numerical_repair.py",
        "tests/unit/test_etas_fit.py",
        "tests/unit/test_background_etas_numerical_repair.py",
    ]
    assert (
        runtime_baseline["prospective_project_blob_oid_and_file_sha256_map_exact_paths"]
        == expected_baseline_paths
    )
    assert set(runtime_baseline["prospective_project_blob_map_excludes_exact_paths"]) == {
        "data/manifests/etas_numerical_repair_code_diff_receipt.json",
        "data/manifests/etas_numerical_repair_optimizer_runtime_baseline.json",
    }
    assert runtime_baseline[
        "baseline_is_completed_before_code_diff_receipt_and_neither_artifact_hashes_itself"
    ]
    runtime_file_contract = runtime_baseline["python_runtime_file_map_contract"]
    assert runtime_file_contract["record_fields_exact"] == [
        "runtime_role",
        "origin_kind",
        "root_role",
        "canonical_root_relative_path",
        "file_sha256",
        "file_size",
    ]
    assert runtime_file_contract["root_role_values_exact"] == [
        "base_prefix",
        "venv_prefix",
        "windows_system_root",
    ]
    assert (
        runtime_file_contract["required_role_root_contract"]["active_venv_python_executable"]
        == "venv_prefix"
    )
    assert runtime_file_contract["runtime_role_values_exact"] == [
        "base_python_executable",
        "active_venv_python_executable",
        "python_shared_library",
        "runtime_dependency_module_origin",
        "loaded_native_dependency",
    ]
    assert runtime_file_contract["allowed_runtime_role_origin_kind_combinations_exact"] == {
        "base_python_executable": ["python_executable"],
        "active_venv_python_executable": ["python_executable"],
        "python_shared_library": ["python_shared_library"],
        "runtime_dependency_module_origin": ["stdlib_source", "stdlib_extension"],
        "loaded_native_dependency": ["loaded_native_dependency"],
    }
    assert (
        "every_loaded_native_dependency_of_verified_python_numpy_scipy_shapely_runtime_closure_once"
        in runtime_file_contract["required_runtime_coverage_requirements"]
    )
    assert (
        "every_loaded_shapely_geos_and_geos_c_native_image_once"
        in runtime_file_contract["required_runtime_coverage_requirements"]
    )
    assert (
        "every_loaded_python_shared_library_once"
        in runtime_file_contract["required_runtime_coverage_requirements"]
    )
    assert runtime_file_contract["native_dependency_capture_contract"][
        "libcrypto_vcruntime_and_transitive_BLAS_LAPACK_runtime_dependencies_may_not_be_omitted"
    ]
    native_capture = runtime_file_contract["native_dependency_capture_contract"]
    assert native_capture["qualification_preflight_must_run_exact_same_fixed_warmup_before_capture"]
    warmup = native_capture["synthetic_runtime_warmup_receipt_contract"]
    evidence_rules = scope["new_repair_module_execution_rules"][
        "src/seismoflux/background/etas_numerical_repair_evidence.py"
    ]
    assert evidence_rules["optimizer_call_forbidden_except_exact_runtime_warmup_callable"]
    assert evidence_rules["exact_runtime_warmup_optimizer_exception"] == (
        "one_direct_scipy_optimize_minimize_LBFGSB_call_with_frozen_synthetic_receipt_only"
    )
    assert evidence_rules["exact_runtime_warmup_geometry_exception"] == (
        "construct_two_fixed_synthetic_shapely_Points_read_x_y_and_call_"
        "BaseGeometry_equals_once_without_real_geometry"
    )
    assert evidence_rules[
        "runtime_warmup_may_not_import_or_call_etas_fit_etas_objective_run_five_start_or_any_real_scientific_orchestration"
    ]
    assert warmup["exact_callable_qualified_name"] == (
        "seismoflux.background.etas_numerical_repair_evidence._run_fixed_optimizer_runtime_warmup"
    )
    assert warmup["invocation_order_exact"] == [
        "scipy_optimize_minimize_lbfgsb",
        "numpy_linalg_solve",
        "scipy_spatial_ckdtree_query_ball_point",
        "shapely_point_xy_and_base_geometry_equals",
    ]
    assert warmup["receipt_fields_exact"] == [
        "schema_version",
        "callable_identity",
        "invocation_order",
        "canonical_input_payload",
        "canonical_input_payload_sha256",
        "canonical_output_payload",
        "canonical_output_payload_sha256",
        "shapely_runtime_binding_identity",
        "synthetic_runtime_warmup_receipt_sha256",
    ]
    assert warmup["shapely_runtime_binding_identity_fields_exact"] == [
        "fixed_path_branch_decision_receipt_sha256",
        "point_public_alias_dependency_record_sha256",
        "ordered_point_constructor_chain_dependency_record_sha256",
        "ordered_point_x_chain_dependency_record_sha256",
        "ordered_point_y_chain_dependency_record_sha256",
        "ordered_equals_chain_dependency_record_sha256",
        "shapely_runtime_binding_identity_sha256",
    ]
    branch_receipt = warmup["shapely_fixed_path_branch_decision_receipt"]
    assert branch_receipt["exact_values"] == {
        "point_argument_count": 2,
        "point_coordinate_array_dtype": "float64",
        "point_coordinate_array_ndim": 1,
        "point_numeric_dtype_branch": True,
        "deprecation_warn_from_comparison_executed": True,
        "deprecation_category_and_make_msg_branch_executed": False,
        "deprecation_warning_branch_taken": False,
        "multithreading_object_array_count": 0,
        "points_y_is_none": True,
        "points_z_is_none": True,
        "points_indices_is_none": True,
    }
    assert _canonical_payload_sha256(branch_receipt["exact_values"]) == (
        "4d010d7cdb5f1b7d35502d7b8c52db79e94f9c2a2b6b37fdfaac9925f80def37"
    )
    chain_preimages = warmup["shapely_runtime_binding_identity_chain_preimages"]
    assert chain_preimages["chain_preimage_item_fields_exact"] == [
        "dependency_record_id",
        "dependency_record_identity_sha256",
    ]
    assert chain_preimages["point_public_alias_dependency_record_id_exact"] == (
        "shapely.geometry.point.Point@direct_binding"
    )
    assert chain_preimages["ordered_point_constructor_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.point.Point.__new__@direct_descriptor",
        "numpy.array@direct_binding",
        "numpy.ndarray.squeeze@direct_descriptor",
        "numpy.ndarray.ndim@direct_descriptor",
        "numpy.issubdtype@direct_binding",
        "numpy.ndarray.dtype@direct_descriptor",
        "numpy.number@direct_binding",
        "shapely.creation.points@deprecation_wrapper",
        "shapely.creation.points@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely.creation.points@wrapped_python_function",
        "shapely.creation._xyz_to_coords@direct_binding",
        "numpy.intc@direct_binding",
        "shapely.lib.points@native_numpy_ufunc",
        "shapely.geometry.point.Point@direct_binding",
    ]
    assert chain_preimages["ordered_point_x_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.point.Point.x@direct_descriptor",
        "shapely._geometry.get_x@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_x@wrapped_python_function",
        "shapely.lib.get_x@native_numpy_ufunc",
    ]
    assert chain_preimages["ordered_point_y_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.point.Point.y@direct_descriptor",
        "shapely._geometry.get_y@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_y@wrapped_python_function",
        "shapely.lib.get_y@native_numpy_ufunc",
    ]
    assert chain_preimages["ordered_equals_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.base.BaseGeometry.equals@direct_descriptor",
        "shapely.predicates.equals@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely.predicates.equals@wrapped_python_function",
        "shapely.lib.equals@native_numpy_ufunc",
        "shapely.geometry.base._maybe_unpack@direct_binding",
        "numpy.generic.ndim@direct_descriptor",
        "numpy.generic.item@direct_descriptor",
    ]
    assert chain_preimages["ordered_chain_dependency_record_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_complete_ordered_chain_preimage_items"
    )
    assert chain_preimages["identity_field_to_preimage_exact"] == {
        "ordered_point_constructor_chain_dependency_record_sha256": (
            "ordered_point_constructor_chain_dependency_record_ids_exact"
        ),
        "ordered_point_x_chain_dependency_record_sha256": (
            "ordered_point_x_chain_dependency_record_ids_exact"
        ),
        "ordered_point_y_chain_dependency_record_sha256": (
            "ordered_point_y_chain_dependency_record_ids_exact"
        ),
        "ordered_equals_chain_dependency_record_sha256": (
            "ordered_equals_chain_dependency_record_ids_exact"
        ),
    }
    chain_fields = [
        "ordered_point_constructor_chain_dependency_record_ids_exact",
        "ordered_point_x_chain_dependency_record_ids_exact",
        "ordered_point_y_chain_dependency_record_ids_exact",
        "ordered_equals_chain_dependency_record_ids_exact",
    ]
    fake_runtime_map = {
        record_id: hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        for field in chain_fields
        for record_id in chain_preimages[field]
    }
    expected_chain_aggregate_sha256 = {
        "ordered_point_constructor_chain_dependency_record_ids_exact": (
            "f8514cede60afcbf9db4b476b730a0d2bd4c2a815519b4d848d59d4f9ba365c7"
        ),
        "ordered_point_x_chain_dependency_record_ids_exact": (
            "7e705d74b231f9627a742107e3ef21b2a2d27774f6580bc1b4796e29ab229d35"
        ),
        "ordered_point_y_chain_dependency_record_ids_exact": (
            "136d1e9a00b8575cda707eb632d4a53640af47f99887b0e165b9b45273ee6124"
        ),
        "ordered_equals_chain_dependency_record_ids_exact": (
            "9b8ca09c557665ce1f907ca948ff828de0b82a27078b72af5844779ac4740cd0"
        ),
    }
    for field in chain_fields:
        preimage = [
            {
                "dependency_record_id": record_id,
                "dependency_record_identity_sha256": fake_runtime_map[record_id],
            }
            for record_id in chain_preimages[field]
        ]
        assert _canonical_payload_sha256(preimage) == expected_chain_aggregate_sha256[field]
        assert (
            _canonical_payload_sha256(list(reversed(preimage)))
            != (expected_chain_aggregate_sha256[field])
        )
    assert chain_preimages[
        "every_chain_item_dependency_record_id_must_resolve_exactly_once_in_same_runtime_callable_dependency_map_and_item_sha256_must_equal_that_record_dependency_record_identity_sha256"
    ]
    assert warmup["baseline_and_qualification_full_receipt_bytes_must_be_equal"]
    assert warmup[
        "baseline_and_qualification_shapely_input_output_and_descriptor_identity_must_be_equal_byte_for_byte"
    ]
    assert warmup[
        "warmup_must_force_first_use_loading_of_shapely_geometry_predicates_geos_and_geos_c_before_capture"
    ]
    shapely_warmup = warmup["canonical_input_payload_exact"][
        "shapely_point_xy_and_base_geometry_equals"
    ]
    assert shapely_warmup["defining_descriptor_qualified_names_exact"] == [
        "shapely.geometry.point.Point.x",
        "shapely.geometry.point.Point.y",
    ]
    assert warmup[
        "warmup_optimizer_calls_are_preflight_only_and_must_not_enter_five_snapshot_twenty_five_start_diagnostics"
    ]
    assert "synthetic_runtime_warmup_receipt" in runtime_baseline["fields_exact"]
    assert native_capture["single_file_classification_precedence_exact"] == [
        "active_venv_python_executable__python_executable",
        "base_python_executable__python_executable",
        "python_shared_library__python_shared_library",
        "runtime_dependency_module_origin__stdlib_extension",
        "loaded_native_dependency__loaded_native_dependency",
    ]
    assert native_capture["classification_examples"] == {
        "python311_dll": "python_shared_library__python_shared_library_only",
        "python3_dll": "python_shared_library__python_shared_library_only",
        "_hashlib_pyd": "runtime_dependency_module_origin__stdlib_extension_only",
        "_ctypes_pyd": "runtime_dependency_module_origin__stdlib_extension_only",
        "base_prefix_Library_bin_libcrypto_dll": (
            "loaded_native_dependency__loaded_native_dependency"
        ),
    }
    assert native_capture["loaded_image_coverage_record_predicate_exact"] == (
        "runtime_role_is_active_venv_python_executable_or_python_shared_library_or_"
        "loaded_native_dependency_OR_runtime_role_is_runtime_dependency_module_origin_"
        "and_origin_kind_is_stdlib_extension"
    )
    assert runtime_file_contract["canonical_record_key_exact"] == [
        "root_role",
        "canonical_root_relative_path",
    ]
    assert runtime_file_contract["canonical_root_relative_path_normalization_exact"] == (
        "resolve_final_path_then_selected_root_relative_then_forward_slash_then_unicode_NFC_"
        "then_windows_ordinal_lowercase"
    )
    dependency_contract = runtime_baseline["runtime_callable_dependency_map_contract"]
    assert {
        "project_class",
        "project_dunder_method",
        "project_property",
        "scipy_class",
        "shapely_callable",
        "shapely_property",
    } <= set(dependency_contract["dependency_kind_values_exact"])
    assert dependency_contract["closure_membership_values_exact"] == [
        "optimizer_fit_runtime_closure",
        "synthetic_runtime_warmup_closure",
        "three_grid_runtime_closure",
        "adapter_artifact_runtime_closure",
    ]
    assert "verified_shapely_RECORD_file" in dependency_contract["origin_kind_values_exact"]
    assert dependency_contract[
        "project_class_requires_defining_module_attribute_identity_project_blob_and_class_ast_sha256_with_null_code_object_sha256"
    ]
    assert dependency_contract[
        "project_property_requires_exact_class_descriptor_identity_and_fget_ast_and_canonical_code_object_sha256"
    ]
    assert dependency_contract["property_may_not_be_classified_as_project_class_method"]
    assert dependency_contract["record_key_exact"] == [
        "canonical_binding_path",
        "callable_layer",
    ]
    assert {
        "dependency_record_id",
        "binding_alias_paths_exact",
        "wrapped_target_dependency_record_id_or_null",
        "closure_cell_bindings",
        "native_ufunc_identity_or_null",
        "dependency_record_identity_sha256",
    } <= set(dependency_contract["record_fields_exact"])
    assert {
        "deprecation_wrapper",
        "multithreading_wrapper",
        "wrapped_python_function",
        "native_numpy_ufunc",
    } <= set(dependency_contract["callable_layer_values_exact"])
    assert dependency_contract[
        "wrapper___wrapped___must_be_identical_to_target_record_object_and_multithreading_closure_func_cell"
    ]
    assert dependency_contract["closure_cell_role_values_exact"] == [
        "callable_traversal_target",
        "executed_fixed_path_noncallable_configuration",
        "inert_unexecuted_branch_decorator_configuration",
    ]
    noncallable_cells = dependency_contract["deprecation_wrapper_noncallable_closure_cells_exact"]
    assert set(noncallable_cells) == {
        "category",
        "make_msg",
        "warn_from",
    }
    assert noncallable_cells["warn_from"]["cell_role"] == (
        "executed_fixed_path_noncallable_configuration"
    )
    for cell_name in ("category", "make_msg"):
        assert noncallable_cells[cell_name]["cell_role"] == (
            "inert_unexecuted_branch_decorator_configuration"
        )
    for cell in noncallable_cells.values():
        assert set(cell["nontraversed_value_identity"]) == set(
            dependency_contract["nontraversed_value_identity_fields_exact"]
        )
    assert {
        "shapely_distribution_name_and_version",
        "shapely_dist_info_RECORD_sha256",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_complete_installed_distribution_verification_map_sha256",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256",
        "runtime_callable_dependency_map",
        "runtime_callable_dependency_map_sha256",
        "three_grid_runtime_dependency_closure_sha256",
    } <= set(runtime_baseline["fields_exact"])
    test_rule = scope["existing_test_change_rules"]["tests/unit/test_etas_fit.py"]
    assert test_rule["only_new_test_functions_may_be_appended"]
    assert test_rule["deletion_rename_skip_xfail_or_assertion_weakening_forbidden"]
    current_test_module = ast.parse(Path("tests/unit/test_etas_fit.py").read_text(encoding="utf-8"))
    current_test_names = [
        node.name
        for node in current_test_module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
    ]
    frozen_test_names = test_rule["frozen_protocol_tag_test_function_names_exact"]
    assert test_rule["frozen_test_count_exact"] == 19
    assert current_test_names[: len(frozen_test_names)] == frozen_test_names
    assert len(current_test_names) == len(set(current_test_names))
    assert test_rule["duplicate_definition_shadow_collection_disappearance_skip_or_xfail_forbidden"]


def test_qualification_and_stage4_receipt_are_fail_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    diagnostics = protocol["diagnostics"]
    assert diagnostics["primary_record_count"] == 25
    assert diagnostics["exact_row_key"] == ["snapshot_id", "start_index"]
    assert diagnostics["exact_snapshot_and_start_cartesian_product_required"] is True
    assert diagnostics["nonfinite_serialization"] == "null_plus_explicit_failure_code"
    assert diagnostics["missing_failed_start_row_action"] == (
        "invalidate_attempt_without_qualification_result"
    )
    assert "attempt_id" in diagnostics["required_fields"]
    assert "optimizer_invocation_receipt_sha256" in diagnostics["required_fields"]
    assert diagnostics["any_unlisted_failure_code_invalidates_attempt"] is True
    assert diagnostics[
        "required_fields_are_exact_and_any_extra_alias_duplicate_or_unknown_field_is_forbidden"
    ]
    assert "raw_scipy_success" in diagnostics["required_fields"]
    assert "etas_start_scipy_converged" in diagnostics["required_fields"]
    assert diagnostics["start_failure_code_first_match_precedence"] == [
        "terminal_vector_missing_or_nonfinite",
        "terminal_physical_decode_invalid",
        "objective_missing_or_nonfinite",
        "gradient_missing_or_nonfinite",
        "scipy_not_success",
        "gradient_threshold_exceeded",
    ]
    assert diagnostics["every_row_must_bind_one_actual_started_and_completed_optimizer_invocation"]
    assert "fit_etas_call_closing_receipt_sha256" in diagnostics["required_fields"]
    assert "diagnostic_row_sha256" in diagnostics["required_fields"]
    assert diagnostics["diagnostic_row_sha256_formula"].endswith(
        "row_without_diagnostic_row_sha256"
    )
    invocation = protocol["optimizer_invocation_receipt_protocol"]
    assert invocation["exact_fit_call_count"] == 5
    assert invocation["exact_optimizer_invocation_count"] == 25
    assert invocation["completion_kind_for_valid_execution"] == "returned_normally"
    assert invocation[
        "any_exception_or_missing_minimize_call_is_invalid_execution_not_numerical_negative"
    ]
    assert invocation["transparent_wrapper"][
        "wrapper_must_call_captured_original_scipy_minimize_exactly_once_with_same_objects_and_values"
    ]
    assert invocation["transparent_wrapper"][
        "wrapper_must_return_exact_same_OptimizeResult_object_without_copy_or_mutation"
    ]
    assert invocation["transparent_wrapper"][
        "original_module_global_minimize_must_be_restored_in_finally_and_identity_rechecked"
    ]
    invocation_fields = invocation["optimizer_invocation_receipt_fields"]
    assert "raw_OptimizeResult_canonical_payload" in invocation_fields
    assert "raw_OptimizeResult_canonical_sha256" in invocation_fields
    assert not {
        "raw_OptimizeResult_x_hex",
        "raw_OptimizeResult_fun_hex_or_null",
        "raw_OptimizeResult_success",
        "raw_OptimizeResult_status",
        "raw_OptimizeResult_nit",
        "raw_OptimizeResult_nfev",
        "raw_OptimizeResult_njev_or_null",
        "raw_OptimizeResult_message",
    } & set(invocation_fields)
    raw_schema = invocation["canonical_subpayload_schemas"]["raw_OptimizeResult_canonical_sha256"]
    assert raw_schema["receipt_must_embed_complete_payload_and_separate_recomputable_sha256"]
    assert raw_schema["exact_keys_required"] == [
        "fun",
        "hess_inv",
        "jac",
        "message",
        "nfev",
        "nit",
        "njev",
        "status",
        "success",
        "x",
    ]
    assert raw_schema["numeric_scalar_wrapper_fields_exact"] == [
        "value_hex_or_null",
        "numeric_state",
        "original_nonfinite_kind_or_null",
    ]
    assert raw_schema["numeric_state_values_exact"] == ["finite", "nonfinite", "absent_null"]
    assert raw_schema["vector_shapes_if_present_exact"] == {"x": [5], "jac": [5]}
    assert raw_schema["present_vector_runtime_type_dtype_layout_exact"] == {
        "type": "numpy.ndarray",
        "dtype": "float64",
        "C_contiguous": True,
    }
    assert raw_schema[
        "hess_inv_array_element_wrappers_may_use_only_finite_or_nonfinite_never_absent_null"
    ]
    assert invocation["valid_execution_requires"] == [
        "five_fit_etas_calls_started_and_returned_normally",
        "twenty_five_original_scipy_minimize_calls_started_and_returned_normally",
        "exact_five_calls_per_snapshot_in_start_index_order",
        "no_preoptimizer_initial_objective_short_circuit",
        "no_wrapper_observed_exception",
        "original_callable_restored_after_every_snapshot",
        "returned_fit_result_contains_exactly_the_same_five_observed_start_results_in_order",
    ]
    assert invocation["receipt_hash_DAG_order"] == (
        "fit_call_opening_then_optimizer_invocations_then_fit_call_closing_then_"
        "diagnostic_rows_then_three_grid_gate_evidence_then_snapshot_gate_result_then_"
        "fit_attempt_snapshot"
    )
    closing_fields = invocation["fit_etas_call_closing_receipt_fields"]
    assert closing_fields[0] == "schema_version"
    assert {
        "returned_five_start_results_canonical_payload",
        "returned_five_start_results_canonical_sha256",
        "returned_fit_result_canonical_payload",
        "returned_fit_result_canonical_sha256",
    } <= set(closing_fields)
    returned_fit_schema = invocation["canonical_subpayload_schemas"][
        "returned_fit_result_canonical_payload"
    ]
    assert returned_fit_schema["result_branch_values_exact"] == [
        "no_stability_eligible_start",
        "stability_eligible_but_unstable",
        "stable",
    ]
    assert returned_fit_schema["hessian_metric_state_values_exact"] == [
        "finite",
        "nonfinite",
        "absent",
    ]
    closure = invocation["runtime_callable_preconditions"]["runtime_global_dependency_closure"]
    assert {
        "ETASModelSpec.validate_event",
        "ETASModelSpec.magnitude_span",
        "PointAreaQuadrature.integrate",
        "PointAreaQuadrature.inverse_power_masses",
        "SeedContext.fields",
        "SeedContext.digest",
        "SeedContext.entropy",
        "SeedContext.generator",
    } <= set(closure["critical_class_methods_and_properties"])
    expected_edges = {
        "fit_etas": [
            "ETASParameterBounds.transformed",
            "ETASParameterBounds.from_transformed",
            "ETASModelSpec.validate_parameters",
        ],
        "etas_objective.<locals>.objective": ["ETASParameterBounds.from_transformed"],
        "evaluate_prepared_likelihood": ["ETASModelSpec.validate_parameters"],
        "observed_hessian_delta_uncertainty": [
            "ETASModelSpec.validate_parameters",
            "ETASModelSpec.branching_ratio",
        ],
        "prepare_etas_likelihood": [
            "ETASModelSpec.validate_event",
            "PointAreaQuadrature.integrate",
        ],
        "_prepare_compensator_arrays": ["PointAreaQuadrature.inverse_power_masses"],
        "_spatial_parent_mass": ["PointAreaQuadrature.integrate"],
        "ETASParameterBounds.from_transformed": [
            "ETASParameterBounds.transformed",
            "ETASParameterBounds.contains",
        ],
        "ETASModelSpec.validate_parameters": ["ETASModelSpec.branching_ratio"],
        "ETASModelSpec.branching_ratio": ["ETASModelSpec.magnitude_span"],
        "_truncated_gr_expectation_alpha_derivative": ["ETASModelSpec.magnitude_span"],
        "optimizer_start": ["SeedContext.generator"],
        "SeedContext.generator": ["SeedContext.entropy"],
        "SeedContext.entropy": ["SeedContext.digest"],
        "SeedContext.digest": ["SeedContext.fields"],
        "_evaluate_background_density_many": [
            "seismoflux.background.pipeline_etas._KDEBackgroundDensity.density_many",
            "seismoflux.background.pipeline_etas._KDEBackgroundDensity.__call__",
        ],
        "_validated_background_density": [
            "seismoflux.background.pipeline_etas._KDEBackgroundDensity.__call__"
        ],
        "seismoflux.background.pipeline_etas._KDEBackgroundDensity.density_many": [
            "seismoflux.background.poisson.SpatialPoissonModel.density"
        ],
        "seismoflux.background.pipeline_etas._KDEBackgroundDensity.__call__": [
            "seismoflux.background.poisson.SpatialPoissonModel.density_scalar"
        ],
        "seismoflux.background.poisson.SpatialPoissonModel.density": [
            "seismoflux.background.poisson.GaussianMixtureFamily.raw_densities"
        ],
        "seismoflux.background.poisson.SpatialPoissonModel.density_scalar": [
            "seismoflux.background.poisson.SpatialPoissonModel.density"
        ],
        "seismoflux.background.poisson.GaussianMixtureFamily.raw_densities": [
            "seismoflux.background.poisson.GaussianMixtureFamily.training_event_count"
        ],
    }
    assert closure["explicit_scientific_instance_method_edges"] == expected_edges
    assert {target for targets in expected_edges.values() for target in targets} == set(
        closure["critical_class_methods_and_properties"]
    )
    assert closure[
        "protocol_tag_pre_repair_edge_absent_but_repair_code_tag_edge_required_exact"
    ] == {"ETASParameterBounds.from_transformed": ["ETASParameterBounds.transformed"]}
    assert closure["third_party_scientific_instance_descriptors_exact"] == [
        "numpy.random.Generator.uniform",
        "scipy.spatial.cKDTree.query_ball_point",
    ]
    assert closure["explicit_third_party_scientific_instance_descriptor_edges"] == {
        "optimizer_start": ["numpy.random.Generator.uniform"],
        "_query_ball_indices": ["scipy.spatial.cKDTree.query_ball_point"],
    }
    assert invocation["runtime_callable_preconditions"]["critical_global_object_identity_edges"][
        "optimizer_start"
    ] == ["SeedContext"]
    assert "run_etas_numerical_repair_qualification" not in closure["roots"]
    assert "_validate_optimizer_runtime" not in closure["roots"]
    assert closure["nonrecursive_orchestration_identity_only_roots"] == [
        "run_etas_numerical_repair_qualification",
        "_validate_optimizer_runtime",
    ]
    assert closure[
        "every_unresolved_frozen_project_scientific_instance_LOAD_ATTR_LOAD_METHOD_or_call_must_be_in_explicit_edges_exactly_once"
    ]
    assert closure[
        "every_separately_enumerated_third_party_scientific_descriptor_call_must_be_in_explicit_third_party_edges_exactly_once"
    ]

    qualification = protocol["qualification"]
    assert qualification["evaluable_requires_all_five_primary_snapshots_pass"] is True
    assert qualification["partial_success_adoption_allowed"] is False
    assert qualification["threshold_relaxation_after_results_allowed"] is False
    assert qualification["any_valid_numerical_gate_failure_action"] == (
        "publish_target_blind_numerical_negative_and_keep_stage4_blocked"
    )
    classification = qualification["outcome_classification"]
    assert set(classification) == {"evaluable", "not_evaluable", "invalid_execution"}
    assert qualification["invalid_execution_may_not_publish_qualification_manifest_or_result_tag"]
    assert qualification["implementation_exception_may_not_be_reclassified_as_numerical_negative"]
    fit_payload = protocol["optimizer_invocation_receipt_protocol"]["canonical_subpayload_schemas"][
        "returned_fit_result_canonical_payload"
    ]
    stability_fields = fit_payload["stability_fields_exact"]
    assert "best_three_relative_objective_range_nonfinite_kind_or_null" in stability_fields
    assert "best_three_transformed_parameter_range_nonfinite_kind_or_null" in stability_fields
    assert fit_payload["stability_range_metric_state_contract"] == {
        "finite": {"value_hex": "required", "nonfinite_kind": None},
        "nonfinite": {"value_hex": None, "nonfinite_kind": "required"},
        "absent": {"value_hex": None, "nonfinite_kind": None},
    }
    assert fit_payload[
        "either_stability_range_metric_nonfinite_requires_stable_false_and_corresponding_frozen_spread_failure_reason"
    ]
    requirements = qualification["per_snapshot_conjunctive_requirements"]
    assert requirements == {
        "exact_start_count": 5,
        "minimum_converged_starts": 4,
        "every_counted_converged_gradient_infinity_norm_lte": 1.0e-4,
        "best_three_relative_objective_range_lte": 1.0e-4,
        "best_three_transformed_parameter_maximum_range_lte": 0.1,
        "hessian_minimum_eigenvalue_gte": 1.0e-8,
        "hessian_condition_number_lte": 1.0e10,
        "branching_ratio_lt": 0.95,
        "fit_only_25_to_12_5km_expected_count_relative_difference_lte": 0.02,
        "fit_only_25_to_12_5km_density_l1_lte": 0.05,
    }
    selected = qualification["selected_start_rule"]
    assert selected["exact_order"] == "objective_ascending_then_start_index_ascending"
    assert selected["selected_start"] == "first_stability_eligible_row_in_exact_order"
    assert selected["selected_start_must_equal_hessian_evaluation_point"] is True
    assert selected["selected_start_must_equal_fit_result_best_parameters_and_objective"] is True
    assert selected["mismatch_action"] == "invalid_execution_without_qualification_result"
    assert protocol["numerical_regression"]["primary_grid_gate"] == "25_to_12_5km"
    assert protocol["numerical_regression"]["diagnostic_grid_pair"] == (
        "50_to_25km_record_required_not_a_pass_gate"
    )

    receipt = protocol["stage4_receipt"]
    assert receipt["required_only_if_qualification_status"] == "evaluable"
    assert receipt["role_order"] == ["development", "formal_validation", "prospective"]
    assert receipt["exact_global_frozen_comparator_receipt_count"] == 1
    assert receipt[
        "global_receipt_contains_complete_ordered_role_mapping_and_role_parameter_hashes"
    ]
    assert receipt["selected_role_field_in_global_receipt_forbidden"] is True
    assert receipt[
        "required_global_hashes_are_sibling_bindings_in_stage4_evidence_not_fields_nested_inside_each_other"
    ]
    assert receipt[
        "frozen_etas_comparator_receipt_may_not_include_adapter_artifact_closing_seal_sha256"
    ]
    assert receipt["required_global_hashes"] == [
        "etas_artifact_sha256",
        "etas_parameter_set_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "frozen_etas_comparator_receipt_sha256",
        "adapter_artifact_closing_seal_sha256",
    ]
    assert receipt["required_per_role_hashes"] == ["etas_parameter_snapshot_sha256"]
    assert receipt["parameter_set_role_mapping"] == {
        "development": "fold_4",
        "formal_validation": "final_validation",
        "prospective": "final_validation",
    }
    assert receipt["qualification_must_bind_all_five_snapshot_parameter_hashes"] is True
    assert receipt["adapter_has_no_file_io"] is True
    assert receipt["window_events_may_not_backfill_issue_time_forecast_map"] is True
    assert receipt["static_kde_adapter_may_not_be_renamed_or_substituted_as_etas"] is True
    assert receipt["new_stage4_execution_revision_required"] is True
    assert receipt["current_stage4_r2_remains_blocked"] is True
    expected_stage4_binding_chain = [
        "protocol_design_sha256",
        "random_input_seal",
        "ScoreBlindInputEvidence",
        "FormalPreflightReceipt",
        "Stage4QualificationEvidence",
        "Stage4ScoringSeal_and_execution_binding",
        "TargetBlindFormalContext",
        "Stage4InMemoryPlan",
        "PlaceboRequest_and_PlaceboSource",
        "placebo_checkpoint_and_result",
        "final_registry_model_card_and_fingerprint",
    ]
    assert receipt["future_stage4_revision_required_binding_chain"] == (
        expected_stage4_binding_chain
    )
    stage4_object_contract = receipt["future_stage4_every_binding_object_contract"]
    assert stage4_object_contract[
        "object_names_must_equal_future_stage4_revision_required_binding_chain_exactly"
    ]
    assert stage4_object_contract["every_object_must_bind_as_sibling_external_hashes"] == [
        "frozen_etas_comparator_receipt_sha256",
        "adapter_artifact_closing_seal_sha256",
    ]
    assert stage4_object_contract[
        "every_object_must_also_bind_etas_artifact_and_parameter_set_sha256"
    ]

    stage4_r2 = _load_yaml(STAGE4_R2_PROTOCOL_PATH)
    r2_required = stage4_r2["evaluation"]["multiple_comparisons"]["confirmatory_gatekeeping"][
        "future_post_etas_preregistered_design"
    ]["comparator_contract"]["mandatory_primary_scientific_comparator"]["required_bindings"]
    repair_role_required = {
        "etas_artifact_sha256",
        "etas_parameter_snapshot_sha256",
        "etas_numerical_qualification_evidence_sha256",
    }
    assert set(r2_required) == repair_role_required
    assert repair_role_required <= (
        set(receipt["required_global_hashes"]) | set(receipt["required_per_role_hashes"])
    )


def test_three_grid_runtime_dependency_closure_includes_shapely_and_post_return_chain() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    runtime = protocol["optimizer_invocation_receipt_protocol"]["runtime_callable_preconditions"]
    closure = runtime["three_grid_runtime_dependency_closure"]

    assert closure["closure_membership_name_exact"] == "three_grid_runtime_closure"
    assert closure["primary_root_exact"] == (
        "seismoflux.background.pipeline_etas._grid_gate_evidence"
    )
    assert set(closure["post_return_property_roots_exact"]) == {
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.passed",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.failure_reasons",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.numerical_evidence_id",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.passed",
    }

    edges = closure["project_dependency_edges_exact"]
    assert edges["seismoflux.background.pipeline_etas._grid_gate_evidence"] == [
        "seismoflux.background.grid.EqualAreaGridFamily.at",
        "seismoflux.background.adapters.point_area_quadrature_from_grid",
        "seismoflux.background.etas_fit.evaluate_etas_cell_expected_masses",
        "seismoflux.background.pipeline_etas._canonical_sha256",
        "seismoflux.background.pipeline_etas.ETASGridResolutionEvidence",
        "seismoflux.background.grid.diagnose_three_grid_convergence",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence",
    ]
    assert edges[
        "seismoflux.background.etas_fit.PointAreaQuadrature.inverse_power_weighted_point_sums"
    ] == [
        "seismoflux.background.etas_fit._query_ball_indices",
        "seismoflux.background.etas_fit._vectorized_inverse_power_density_squared",
        "seismoflux.background.etas_fit._readonly_float",
    ]
    assert edges[
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.numerical_evidence_id"
    ] == [
        "seismoflux.background.pipeline_etas._canonical_sha256",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.comparisons",
    ]
    assert edges["seismoflux.background.grid.EqualAreaGrid.cell_ids"] == [
        "seismoflux.background.grid.GridCell.id"
    ]
    assert edges["seismoflux.background.grid.GridCell.id"] == ["seismoflux.background.grid.cell_id"]
    assert edges["seismoflux.background.artifacts._canonicalize"] == [
        "seismoflux.background.artifacts._canonicalize"
    ]

    expected_dunders = {
        "seismoflux.background.etas_fit.QuadraturePoint.__post_init__",
        "seismoflux.background.etas_fit.PointAreaQuadrature.__post_init__",
        "seismoflux.background.etas_fit.ETASExpectedCellMasses.__post_init__",
        "seismoflux.background.pipeline_etas.ETASGridResolutionEvidence.__post_init__",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.__post_init__",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.__post_init__",
    }
    assert set(closure["project_dunder_methods_required_exact"]) == expected_dunders
    assert {
        "seismoflux.background.etas_fit.PointAreaQuadrature.inverse_power_weighted_point_sums",
        "seismoflux.background.grid.EqualAreaGridFamily.at",
        "seismoflux.background.pipeline_etas._KDEBackgroundDensity.density_many",
        "seismoflux.background.poisson.GaussianMixtureFamily.raw_densities",
    } <= set(closure["project_class_methods_required_exact"])
    assert {
        "seismoflux.background.grid.EqualAreaGrid.cell_ids",
        "seismoflux.background.grid.GridCell.id",
        "seismoflux.background.grid.GridSpec.cell_size_mm",
        "seismoflux.background.grid.GridConvergenceDiagnostics.passed",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.comparisons",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.passed",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.passed",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.failure_reasons",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.numerical_evidence_id",
    } <= set(closure["project_properties_required_exact"])

    scipy = closure["scipy_dependencies_exact"]
    assert scipy["class_and_constructor"] == "scipy.spatial.cKDTree"
    assert scipy["instance_method"] == "scipy.spatial.cKDTree.query_ball_point"
    assert scipy["exact_edges"] == {
        "seismoflux.background.etas_fit.PointAreaQuadrature.__post_init__": [
            "scipy.spatial.cKDTree"
        ],
        "seismoflux.background.etas_fit._query_ball_indices": [
            "scipy.spatial.cKDTree.query_ball_point"
        ],
    }

    shapely_contract = closure["shapely_dependencies_exact"]
    assert shapely_contract["public_class_binding"] == "shapely.geometry.Point"
    assert shapely_contract["defining_class_qualified_name"] == "shapely.geometry.point.Point"
    assert shapely_contract["properties"] == [
        "shapely.geometry.point.Point.x",
        "shapely.geometry.point.Point.y",
    ]
    assert shapely_contract["instance_methods"] == ["shapely.geometry.base.BaseGeometry.equals"]
    assert shapely_contract["public_alias_bindings_exact"] == {
        "shapely.geometry.Point": "shapely.geometry.point.Point",
        "shapely.points": "shapely.creation.points",
        "shapely.get_x": "shapely._geometry.get_x",
        "shapely.get_y": "shapely._geometry.get_y",
        "shapely.equals": "shapely.predicates.equals",
    }
    warmup_ids = set(shapely_contract["synthetic_runtime_warmup_dependency_record_ids_exact"])
    three_grid_ids = set(shapely_contract["three_grid_runtime_dependency_record_ids_exact"])
    assert {
        "shapely.geometry.point.Point@direct_binding",
        "shapely.geometry.point.Point.__new__@direct_descriptor",
        "numpy.array@direct_binding",
        "numpy.ndarray.squeeze@direct_descriptor",
        "numpy.ndarray.ndim@direct_descriptor",
        "numpy.issubdtype@direct_binding",
        "numpy.number@direct_binding",
        "shapely.creation.points@deprecation_wrapper",
        "shapely.creation.points@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "numpy.ndarray.dtype@direct_descriptor",
        "shapely.creation.points@wrapped_python_function",
        "numpy.intc@direct_binding",
        "shapely.lib.points@native_numpy_ufunc",
    } <= warmup_ids
    assert {
        "numpy.array@direct_binding",
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_x@multithreading_wrapper",
        "shapely._geometry.get_x@wrapped_python_function",
        "shapely.lib.get_x@native_numpy_ufunc",
        "shapely._geometry.get_y@multithreading_wrapper",
        "shapely._geometry.get_y@wrapped_python_function",
        "shapely.lib.get_y@native_numpy_ufunc",
        "shapely.predicates.equals@multithreading_wrapper",
        "shapely.predicates.equals@wrapped_python_function",
        "shapely.lib.equals@native_numpy_ufunc",
        "numpy.generic.ndim@direct_descriptor",
        "numpy.generic.item@direct_descriptor",
    } <= three_grid_ids
    constructor_only = warmup_ids - three_grid_ids
    assert "shapely.geometry.point.Point@direct_binding" in constructor_only
    assert "shapely.lib.points@native_numpy_ufunc" in constructor_only
    assert not any("Point.__new__" in record_id for record_id in three_grid_ids)
    assert shapely_contract[
        "point_constructor_dependency_record_ids_except_shared_numpy_array_and_ndarray_bindings_must_be_disjoint_from_three_grid_runtime_dependency_record_ids"
    ]
    assert shapely_contract[
        "three_grid_runtime_dependency_record_ids_must_equal_exact_intersection_of_synthetic_warmup_and_three_grid_lists"
    ]
    layered_edges = shapely_contract["layered_dependency_record_edges_exact"]
    assert layered_edges["seismoflux.background.adapters.point_area_quadrature_from_grid"] == [
        "shapely.geometry.point.Point.x@direct_descriptor",
        "shapely.geometry.point.Point.y@direct_descriptor",
    ]
    assert (
        "shapely.geometry.point.Point@direct_binding"
        in layered_edges[
            "seismoflux.background.etas_numerical_repair_evidence."
            "_run_fixed_optimizer_runtime_warmup"
        ]
    )
    assert layered_edges["shapely.geometry.point.Point.__new__@direct_descriptor"] == [
        "numpy.array@direct_binding",
        "numpy.ndarray.squeeze@direct_descriptor",
        "numpy.ndarray.ndim@direct_descriptor",
        "numpy.issubdtype@direct_binding",
        "numpy.ndarray.dtype@direct_descriptor",
        "numpy.number@direct_binding",
        "shapely.creation.points@deprecation_wrapper",
        "shapely.geometry.point.Point@direct_binding",
    ]
    assert layered_edges["shapely._geometry.get_x@multithreading_wrapper"] == [
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_x@wrapped_python_function",
    ]
    assert layered_edges["shapely._geometry.get_x@wrapped_python_function"] == [
        "shapely.lib.get_x@native_numpy_ufunc"
    ]
    assert layered_edges["shapely._geometry.get_y@multithreading_wrapper"] == [
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_y@wrapped_python_function",
    ]
    assert layered_edges["shapely.predicates.equals@multithreading_wrapper"] == [
        "numpy.ndarray@direct_binding",
        "shapely.predicates.equals@wrapped_python_function",
    ]
    assert layered_edges["shapely.predicates.equals@wrapped_python_function"] == [
        "shapely.lib.equals@native_numpy_ufunc"
    ]
    assert layered_edges["shapely.geometry.base._maybe_unpack@direct_binding"] == [
        "numpy.generic.ndim@direct_descriptor",
        "numpy.generic.item@direct_descriptor",
    ]

    assert shapely.geometry.Point is shapely_point.Point
    assert shapely.points is shapely_creation.points
    assert shapely.get_x is shapely_geometry.get_x
    assert shapely.get_y is shapely_geometry.get_y
    assert shapely.equals is shapely_predicates.equals

    synthetic_left = Point(1.25, -0.5)
    synthetic_right = Point(1.25, -0.5)
    assert synthetic_left.x.hex() == "0x1.4000000000000p+0"
    assert synthetic_left.y.hex() == "-0x1.0000000000000p-1"
    assert synthetic_left.equals(synthetic_right) is True
    assert Point.x.fget is not None
    assert Point.y.fget is not None
    assert Point.x.fget.__globals__["shapely"].get_x is shapely_geometry.get_x
    assert Point.y.fget.__globals__["shapely"].get_y is shapely_geometry.get_y
    assert Point.equals is shapely_base.BaseGeometry.equals
    assert Point.equals.__globals__["shapely"].equals is shapely_predicates.equals
    assert Point.equals.__globals__["_maybe_unpack"] is shapely_base._maybe_unpack
    raw_equals = shapely_predicates.equals(synthetic_left, synthetic_right)
    assert isinstance(raw_equals, np.generic)
    assert raw_equals.ndim == 0
    assert raw_equals.item() is True

    points_outer = cast(Any, shapely_creation.points)
    points_multithreading = cast(Any, points_outer.__wrapped__)
    points_original = cast(Any, points_multithreading.__wrapped__)
    assert points_outer is not points_multithreading
    assert points_multithreading is not points_original
    points_outer_cells = dict(
        zip(points_outer.__code__.co_freevars, points_outer.__closure__, strict=True)
    )
    points_multithreading_cells = dict(
        zip(
            points_multithreading.__code__.co_freevars,
            points_multithreading.__closure__,
            strict=True,
        )
    )
    assert points_outer_cells["func"].cell_contents is points_multithreading
    assert points_multithreading_cells["func"].cell_contents is points_original
    assert points_outer_cells["category"].cell_contents is DeprecationWarning
    assert points_outer_cells["warn_from"].cell_contents == 3
    make_msg = points_outer_cells["make_msg"].cell_contents
    assert type(make_msg).__module__ == "functools"
    assert type(make_msg).__qualname__ == "_lru_cache_wrapper"
    assert make_msg.__wrapped__.__module__ == "shapely.decorators"
    assert make_msg.__wrapped__.__qualname__.endswith("<locals>.make_msg")
    assert {"array", "ndim", "issubdtype", "dtype", "number", "points", "Point"} <= set(
        Point.__new__.__code__.co_names
    )
    assert Point.__new__.__globals__["Point"] is Point
    assert Point.__new__.__globals__["np"].ndarray is np.ndarray
    assert {"ndarray", "dtype", "flags", "writeable"} <= _recursive_code_names(
        points_multithreading.__code__
    )
    assert points_original.__globals__["lib"].points is shapely_lib.points

    for wrapper, native_name in (
        (cast(Any, shapely_geometry.get_x), "get_x"),
        (cast(Any, shapely_geometry.get_y), "get_y"),
        (cast(Any, shapely_predicates.equals), "equals"),
    ):
        original = cast(Any, wrapper.__wrapped__)
        closure_cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__, strict=True))
        assert closure_cells["func"].cell_contents is original
        assert {"ndarray", "dtype", "flags", "writeable"} <= _recursive_code_names(wrapper.__code__)
        assert wrapper.__globals__["np"].ndarray is np.ndarray
        assert original.__globals__["lib"].__dict__[native_name] is getattr(
            shapely_lib, native_name
        )

    native_expectations = {
        "points": {
            "nin": 2,
            "nout": 1,
            "nargs": 3,
            "ntypes": 1,
            "types": ["di->O"],
            "identity": None,
            "signature": "(d),()->()",
        },
        "get_x": {
            "nin": 1,
            "nout": 1,
            "nargs": 2,
            "ntypes": 1,
            "types": ["O->d"],
            "identity": None,
            "signature": None,
        },
        "get_y": {
            "nin": 1,
            "nout": 1,
            "nargs": 2,
            "ntypes": 1,
            "types": ["O->d"],
            "identity": None,
            "signature": None,
        },
        "equals": {
            "nin": 2,
            "nout": 1,
            "nargs": 3,
            "ntypes": 1,
            "types": ["OO->?"],
            "identity": None,
            "signature": None,
        },
    }
    for native_name, expected in native_expectations.items():
        native = getattr(shapely_lib, native_name)
        assert isinstance(native, np.ufunc)
        assert native.__name__ == native_name
        for field, value in expected.items():
            assert getattr(native, field) == value

    shapely_distribution = distribution("shapely")
    shapely_extension_path = Path(shapely_lib.__file__).resolve()
    matching_record_rows = [
        file
        for file in shapely_distribution.files or []
        if Path(str(shapely_distribution.locate_file(file))).resolve() == shapely_extension_path
    ]
    assert len(matching_record_rows) == 1
    shapely_extension_record = matching_record_rows[0]
    assert shapely_extension_record.hash is not None
    assert shapely_extension_record.hash.mode == "sha256"
    encoded_digest = shapely_extension_record.hash.value
    padding = "=" * (-len(encoded_digest) % 4)
    assert base64.urlsafe_b64decode(encoded_digest + padding).hex() == _sha256(
        shapely_extension_path
    )
    assert shapely_extension_record.size == shapely_extension_path.stat().st_size
    assert isinstance(Point.x, property)
    assert isinstance(Point.y, property)
    assert shapely_contract[
        "every_native_backed_call_requires_complete_shapely_RECORD_and_GEOS_loaded_image_map_match"
    ]
    memberships = shapely_contract["exact_closure_memberships_by_dependency_record_id"]
    assert set(memberships) == warmup_ids | three_grid_ids
    for record_id in constructor_only:
        assert memberships[record_id] == ["synthetic_runtime_warmup_closure"]
    for record_id in warmup_ids & three_grid_ids:
        assert memberships[record_id] == [
            "synthetic_runtime_warmup_closure",
            "three_grid_runtime_closure",
        ]
    assert closure[
        "runtime_callable_dependency_map_closure_membership_for_every_three_grid_reachable_record_must_include_three_grid_runtime_closure"
    ]
    assert closure[
        "synthetic_warmup_only_records_must_include_synthetic_runtime_warmup_closure_and_must_not_claim_three_grid_runtime_closure"
    ]
    assert closure[
        "no_reachable_record_may_be_missing_and_no_unreachable_record_may_claim_three_grid_runtime_closure_membership"
    ]

    runtime_seal = protocol["qualification_execution_seal"]["optimizer_runtime_code_seal"]
    assert runtime_seal["expected_shapely_distribution_version_from_uv_lock"] == "2.1.2"
    assert runtime_seal["complete_installed_distribution_RECORD_validation"]["distributions"] == [
        "numpy",
        "scipy",
        "shapely",
    ]
    required_runtime_fields = {
        "shapely_distribution_name_and_version",
        "shapely_dist_info_RECORD_sha256",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_complete_installed_distribution_verification_map_sha256",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256",
        "runtime_callable_dependency_map",
        "runtime_callable_dependency_map_sha256",
        "three_grid_runtime_dependency_closure_sha256",
    }
    assert required_runtime_fields <= set(runtime_seal["fields_exact"])
    public_runtime = protocol["outputs"]["public_qualification_seal_schema"]
    assert runtime_seal["fields_exact"] == public_runtime["runtime_fields_exact"]
    runtime_baseline = protocol["repair_code_scope_from_protocol_tag"][
        "repair_code_tag_prerequisite_public_artifacts"
    ]["optimizer_runtime_baseline"]
    full_hash_pairs = {
        "numpy_complete_installed_distribution_verification_map": (
            "numpy_complete_installed_distribution_verification_map_sha256"
        ),
        "scipy_complete_installed_distribution_verification_map": (
            "scipy_complete_installed_distribution_verification_map_sha256"
        ),
        "shapely_complete_installed_distribution_verification_map": (
            "shapely_complete_installed_distribution_verification_map_sha256"
        ),
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map": (
            "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256"
        ),
        "numpy_runtime_config_safe_projection": "numpy_runtime_config_canonical_sha256",
        "runtime_callable_dependency_map": "runtime_callable_dependency_map_sha256",
    }
    baseline_fields = set(runtime_baseline["fields_exact"])
    runtime_fields = set(runtime_seal["fields_exact"])
    for full_field, hash_field in full_hash_pairs.items():
        assert {full_field, hash_field} <= baseline_fields
        assert {full_field, hash_field} <= runtime_fields
    crosswalks = runtime_seal["code_tag_baseline_equality"][
        "exact_full_object_and_sibling_hash_crosswalks"
    ]
    assert set(crosswalks) == set(full_hash_pairs)
    for full_field, hash_field in full_hash_pairs.items():
        assert crosswalks[full_field]["baseline_full_field"] == full_field
        assert crosswalks[full_field]["runtime_full_field"] == full_field
        assert crosswalks[full_field]["baseline_sha256_field"] == hash_field
        assert crosswalks[full_field]["runtime_sha256_field"] == hash_field
    assert runtime_seal["code_tag_baseline_equality"][
        "every_baseline_full_object_must_equal_runtime_full_object_byte_for_byte_and_each_sibling_sha256_must_equal_recomputed_hash_of_both"
    ]
    assert public_runtime["opening_runtime_baseline_identity_crosswalk"][
        "all_three_pairs_must_be_equal_and_recompute_from_the_exact_remote_repair_code_tag_baseline_blob"
    ]
    assert public_runtime["runtime_full_object_and_sibling_hash_crosswalk_ref"] == (
        "qualification_execution_seal.optimizer_runtime_code_seal.code_tag_baseline_equality."
        "exact_full_object_and_sibling_hash_crosswalks"
    )
    assert (
        public_runtime["runtime_nested_field_schema_refs"]["shapely_distribution_name_and_version"]
        == "outputs.canonical_nested_schemas.distribution_name_and_version"
    )
    assert public_runtime["runtime_nested_field_schema_refs"][
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map"
    ] == (
        "outputs.canonical_nested_schemas."
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map"
    )


def test_content_identities_and_typed_adapter_contract_are_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    identities = protocol["content_addressing"]["identities"]
    assert set(identities) == {
        "optimizer_runtime_code_seal_sha256",
        "fit_etas_call_opening_receipt_sha256",
        "optimizer_invocation_receipt_sha256",
        "fit_etas_call_closing_receipt_sha256",
        "fit_attempt_snapshot_sha256",
        "three_grid_gate_evidence_sha256",
        "etas_parameter_snapshot_sha256",
        "etas_parameter_set_sha256",
        "etas_numerical_negative_evidence_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "etas_artifact_sha256",
        "frozen_etas_comparator_receipt_sha256",
        "etas_issue_forecast_query_nodes_sha256",
        "etas_issue_simulation_context_sha256",
        "etas_issue_simulation_batch_payload_sha256",
        "etas_issue_simulation_catalog_receipt_sha256",
        "etas_issue_forecast_input_sha256",
        "etas_issue_forecast_projection_receipt_sha256",
        "etas_retrospective_likelihood_input_sha256",
        "etas_causal_parent_receipt_sha256",
        "etas_retrospective_likelihood_terms_sha256",
    }
    adapter_public_schema = protocol["outputs"]["adapter_public_artifact_schemas"]
    assert (
        identities["etas_artifact_sha256"]["includes"]
        == adapter_public_schema["artifact_manifest_fields_exact"][:-1]
    )
    assert "five_etas_parameter_snapshot_sha256" in identities["etas_artifact_sha256"]["includes"]
    assert "five_parameter_snapshot_sha256" not in identities["etas_artifact_sha256"]["includes"]
    assert "model_spec_by_snapshot" in identities["etas_artifact_sha256"]["includes"]
    assert "model_spec" not in identities["etas_artifact_sha256"]["includes"]
    assert (
        identities["frozen_etas_comparator_receipt_sha256"]["includes"]
        == (adapter_public_schema["global_receipt_fields_exact"][:-1])
    )
    invocation_protocol = protocol["optimizer_invocation_receipt_protocol"]
    assert (
        invocation_protocol["fit_etas_call_opening_receipt_identity_fields_exact"]
        == (invocation_protocol["fit_etas_call_opening_receipt_fields"][:-1])
    )
    assert (
        invocation_protocol["optimizer_invocation_receipt_identity_fields_exact"]
        == (invocation_protocol["optimizer_invocation_receipt_fields"][:-1])
    )
    assert (
        invocation_protocol["fit_etas_call_closing_receipt_identity_fields_exact"]
        == (invocation_protocol["fit_etas_call_closing_receipt_fields"][:-1])
    )
    fit_attempt_fields = invocation_protocol["fit_attempt_snapshot_payload_fields_exact"]
    assert identities["fit_attempt_snapshot_sha256"]["includes_ref"] == (
        "optimizer_invocation_receipt_protocol.fit_attempt_snapshot_payload_fields_exact"
    )
    assert {
        "fit_etas_call_opening_receipt_sha256",
        "ordered_five_optimizer_invocation_receipt_sha256",
        "fit_etas_call_closing_receipt_sha256",
        "ordered_five_diagnostic_row_sha256",
        "three_grid_gate_evidence_sha256_or_null",
        "snapshot_gate_result_sha256",
    } <= set(fit_attempt_fields)
    assert "aki_b_beta_two_bin_masses" in identities["etas_parameter_snapshot_sha256"]["includes"]
    assert (
        identities["etas_parameter_set_sha256"]["includes"]
        == protocol["outputs"]["public_parameter_registry_schema"][
            "parameter_set_identity_payload_fields_exact"
        ]
    )
    qualification_identity = identities["etas_numerical_qualification_evidence_sha256"]
    qualification_manifest_schema = protocol["outputs"]["public_qualification_manifest_schema"]
    qualification_projection = qualification_manifest_schema[
        "qualification_evidence_projection_fields_exact"
    ]
    assert qualification_projection == qualification_manifest_schema["top_level_fields_exact"][:-2]
    assert {
        "opening_execution_seal_sha256",
        "qualification_input_seal_sha256",
        "public_source_access_receipt",
        "parent_replay_membership_identity_sha256_by_snapshot",
        "fit_attempt_snapshot_sha256_by_snapshot",
        "diagnostic_rows",
        "snapshot_gate_results",
    } <= set(qualification_projection)
    assert qualification_identity["identity_projection_fields_exact_ref"] == (
        "outputs.public_qualification_manifest_schema.qualification_evidence_projection_fields_exact"
    )
    assert qualification_identity["branch_invariants"] == {
        "evaluable": {
            "etas_parameter_set_sha256": "required",
            "etas_numerical_negative_evidence_sha256": None,
        },
        "not_evaluable": {
            "etas_parameter_set_sha256": None,
            "etas_numerical_negative_evidence_sha256": "required",
        },
    }
    assert (
        identities["etas_numerical_negative_evidence_sha256"][
            "parameter_snapshot_or_set_artifacts_allowed"
        ]
        is False
    )
    assert (
        identities["etas_numerical_negative_evidence_sha256"]["includes"]
        == (qualification_manifest_schema["numerical_negative_evidence_fields_exact"][1:])
    )
    assert (
        identities["frozen_etas_comparator_receipt_sha256"]["selected_role_field_allowed"] is False
    )
    simulation_context = identities["etas_issue_simulation_context_sha256"]
    assert set(simulation_context["explicitly_excludes"]) == {
        "grid_family",
        "horizons_days",
        "query_nodes",
        "magnitude_output_bins",
        "stage4_targets_or_results",
    }
    assert {
        "etas_issue_simulation_context_sha256",
        "etas_issue_simulation_batch_payload_sha256",
        "etas_issue_simulation_catalog_receipt_sha256",
        "grid_family_sha256",
        "horizons_days",
        "etas_issue_forecast_query_nodes_sha256",
    } <= set(identities["etas_issue_forecast_input_sha256"]["includes"])
    projection_receipt = identities["etas_issue_forecast_projection_receipt_sha256"]
    assert projection_receipt["circular_hash_reference_forbidden"] is True
    simulation_receipt_identity = identities["etas_issue_simulation_catalog_receipt_sha256"]

    contract = protocol["adapter_contract"]
    interfaces = contract["typed_interfaces"]
    assert set(interfaces) == {
        "ETASKnownParentEvent",
        "ETASRetrospectiveWindow",
        "ETASRetrospectiveTargetEvent",
        "ETASRetrospectiveEventQueryNode",
        "ETASBaselineQueryNodes",
        "ETASBaselineMeasure",
        "ETASRetrospectiveLikelihoodInput",
        "ETASRetrospectiveEventTerm",
        "ETASRetrospectiveLikelihoodTerms",
        "ETASIssueSimulationContext",
        "ETASIssueSimulationInput",
        "ETASFuturePropagationEvent",
        "ETASIssueSimulationBatch",
        "ETASIssueSimulationOutput",
        "ETASIssueForecastQueryNodes",
        "ETASIssueForecastInput",
        "ETASIssueForecastMeasure",
        "ETASIssueForecastField",
        "ETASIssueSimulationCatalogReceipt",
        "ETASIssueForecastReplicateNodeDiagnostic",
        "ETASIssueForecastProjectionReceipt",
    }
    assert interfaces["ETASIssueSimulationInput"]["target_or_event_result_fields_allowed"] is False
    assert interfaces["ETASIssueForecastInput"]["target_or_event_result_fields_allowed"] is False
    assert interfaces["ETASKnownParentEvent"]["field_aliases_allowed"] is False
    assert interfaces["ETASKnownParentEvent"]["fields"] == [
        "physical_event_id",
        "origin_time_utc",
        "available_time_utc",
        "x_y_hex",
        "magnitude_hex",
        "inside_supported_domain",
        "inside_study_area",
        "inside_parent_domain",
        "parent_role",
    ]
    assert interfaces["ETASKnownParentEvent"]["parent_role_values"] == [
        "supported",
        "true_external_buffer",
        "unsupported_conditional",
    ]
    assert interfaces["ETASKnownParentEvent"]["missing_extra_unknown_or_alias_field_rejected"]
    assert interfaces["ETASRetrospectiveTargetEvent"][
        "missing_extra_unknown_or_alias_field_rejected"
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"][
        "ordered_targets_use_ETASRetrospectiveTargetEvent_schema"
    ]
    assert interfaces["ETASRetrospectiveWindow"]["fields"] == [
        "window_start_utc",
        "window_end_utc",
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"]["nested_schema_refs"] == {
        "window": "adapter_contract.typed_interfaces.ETASRetrospectiveWindow",
        "ordered_targets": "adapter_contract.typed_interfaces.ETASRetrospectiveTargetEvent",
        "ordered_event_query_nodes": (
            "adapter_contract.typed_interfaces.ETASRetrospectiveEventQueryNode"
        ),
        "ordered_known_parent_events": "adapter_contract.typed_interfaces.ETASKnownParentEvent",
        "baseline_query_nodes": "adapter_contract.typed_interfaces.ETASBaselineQueryNodes",
    }
    assert (
        identities["etas_retrospective_likelihood_input_sha256"]["includes"]
        == interfaces["ETASRetrospectiveLikelihoodInput"]["fields"][:-1]
    )
    assert interfaces["ETASRetrospectiveLikelihoodInput"][
        "selected_role_snapshot_and_parameter_sha_must_match_global_receipt_mapping"
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"]["window_interval"] == (
        "(window_start_utc,window_end_utc]"
    )
    assert interfaces["ETASRetrospectiveTargetEvent"]["target_to_cell_mapping"] == (
        "exact_covers_then_row_column_cell_id_tie_break_from_frozen_stage4_R2_25km_grid"
    )
    assert interfaces["ETASRetrospectiveTargetEvent"][
        "exact_coordinates_used_only_for_frozen_cell_mapping_and_never_for_query_refinement_or_point_intensity"
    ]
    assert interfaces["ETASRetrospectiveEventQueryNode"][
        "target_micro_move_within_same_cell_must_leave_query_node_and_intensity_bytes_unchanged"
    ]
    assert interfaces["ETASRetrospectiveEventQueryNode"][
        "target_query_node_crosswalk_fields_exact"
    ] == [
        "target.event_query_node_id_equals_node.event_query_node_id",
        "target.physical_event_id_equals_node.physical_event_id",
        "target.frozen_25km_grid_id_equals_node.frozen_25km_grid_id",
        "target.frozen_25km_cell_id_equals_node.frozen_25km_cell_id",
        "target.frozen_25km_row_equals_node.frozen_25km_row",
        "target.frozen_25km_column_equals_node.frozen_25km_column",
        "target.origin_time_utc_equals_node.event_time_utc",
        "target.magnitude_bin_equals_node.magnitude_bin",
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"]["known_parent_collection_semantics"] == (
        "complete_authorized_M4_plus_causal_catalog_through_window_end_including_scored_"
        "targets_and_unscored_M4_to_lt_M5_events"
    )
    assert (
        interfaces["ETASIssueSimulationInput"][
            "every_parent_available_time_lte_knowledge_cutoff_required"
        ]
        is True
    )
    assert interfaces["ETASIssueSimulationInput"]["every_parent_origin_lte_issue_time_required"]
    assert interfaces["ETASIssueSimulationBatch"]["hidden_global_cache_allowed"] is False
    assert (
        identities["etas_issue_simulation_batch_payload_sha256"]["includes"]
        == interfaces["ETASIssueSimulationBatch"]["fields"][:-1]
    )
    simulation_batch = interfaces["ETASIssueSimulationBatch"]
    assert simulation_batch["replicate_catalog_fields_exact"] == [
        "replicate_index",
        "ordered_future_events",
        "replicate_catalog_sha256",
    ]
    assert simulation_batch[
        "every_ordered_replicate_catalog_sha256_must_recompute_from_and_equal_same_index_catalog_own_sha"
    ]
    assert (
        interfaces["ETASIssueSimulationBatch"]["public_serialization_or_event_row_exposure_allowed"]
        is False
    )
    assert interfaces["ETASIssueSimulationContext"][
        "ordered_known_parent_events_use_ETASKnownParentEvent_schema"
    ]
    assert interfaces["ETASIssueSimulationBatch"][
        "context_object_contains_known_parent_events_once_and_every_replicate_catalog_contains_future_events_only"
    ]
    future_event = interfaces["ETASFuturePropagationEvent"]
    assert future_event["output_eligible_iff_inside_supported_domain_and_magnitude_gte_5"]
    assert set(future_event["domain_role_truth_table"]) == {
        "supported",
        "true_external_buffer",
        "eligible_unsupported",
    }
    assert future_event["replicate_index_must_equal_containing_replicate_catalog_replicate_index"]
    assert future_event[
        "origin_time_must_be_strictly_after_context_issue_time_and_lte_context_issue_time_plus_365_days"
    ]
    assert interfaces["ETASIssueSimulationBatch"][
        "projection_node_intensity_uses_context_known_parents_plus_same_replicate_future_parents_strictly_before_node_time"
    ]
    assert interfaces["ETASIssueForecastInput"]["projection_may_not_resimulate_or_use_hidden_cache"]
    assert interfaces["ETASRetrospectiveLikelihoodTerms"]["alarm_ranking_consumer_allowed"] is False
    assert (
        interfaces["ETASIssueForecastField"]["retrospective_likelihood_consumer_allowed"] is False
    )
    assert (
        interfaces["ETASIssueForecastField"]["permutation_duplicate_or_missing_cell_rejected"]
        is True
    )
    assert (
        "ordered_query_node_measure_payload_sha256_excluding_projection_receipt"
        in (interfaces["ETASIssueForecastField"]["fields"])
    )
    assert (
        "ordered_query_node_measure_sha256" not in (interfaces["ETASIssueForecastField"]["fields"])
    )
    assert interfaces["ETASIssueForecastProjectionReceipt"]["acyclic_construction_order"] == [
        "build_and_hash_ordered_replicate_node_intensity_diagnostic_rows_with_no_projection_receipt_field",
        "hash_ordered_node_measure_rows_with_projection_receipt_field_omitted",
        "build_and_hash_ordered_cell_rows_with_projection_receipt_field_omitted_and_only_the_receipt_free_node_payload_sha",
        "hash_projection_receipt_from_input_catalog_receipt_adapter_replicate_diagnostic_and_both_receipt_free_payload_hashes",
        "fill_projection_receipt_sha_into_node_and_cell_rows_without_rehashing_payload_identities",
    ]
    assert (
        projection_receipt["includes"] == interfaces["ETASIssueForecastProjectionReceipt"]["fields"]
    )
    assert interfaces["ETASIssueForecastMeasure"]["variant_id_exact"] == (
        "etas_background_no_increment"
    )
    assert (
        interfaces["ETASIssueForecastMeasure"]["anomaly_factor_input_or_weighting_allowed"] is False
    )
    assert interfaces["ETASIssueForecastMeasure"][
        "outside_selected_support_positive_zero_fields_exact"
    ] == [
        "conditional_ground_intensity_mean_per_day_km2_hex",
        "conditional_ground_intensity_standard_error_hex",
        "conditional_bin_intensity_mean_per_day_km2_hex",
        "weighted_expected_count_hex",
    ]
    replicate_diagnostic = interfaces["ETASIssueForecastReplicateNodeDiagnostic"]
    assert replicate_diagnostic["outside_selected_support_positive_zero_measure_fields_exact"] == [
        "conditional_ground_intensity_per_day_km2_hex",
        "conditional_bin_intensity_per_day_km2_hex",
        "weighted_expected_count_hex",
    ]
    assert replicate_diagnostic[
        "replicate_index_query_node_id_inside_selected_support_and_magnitude_bin_mass_are_not_zeroed"
    ]
    assert (
        interfaces["ETASRetrospectiveLikelihoodTerms"][
            "exact_event_intensity_count_must_equal_ordered_target_count"
        ]
        is True
    )
    assert interfaces["ETASRetrospectiveLikelihoodTerms"]["terms_must_bind_complete_input_sha256"]
    assert (
        "ordered_baseline_node_measures" in interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"]
    )
    assert (
        "node_level_baseline_measure"
        not in interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"]
    )
    assert interfaces["ETASRetrospectiveLikelihoodTerms"][
        "ordered_baseline_node_measures_use_ETASBaselineMeasure_schema"
    ]
    assert (
        identities["etas_retrospective_likelihood_terms_sha256"]["includes"]
        == interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"][:-1]
    )
    causal_parent_identity = identities["etas_causal_parent_receipt_sha256"]
    assert causal_parent_identity["parent_identity_map_item_fields_exact"] == [
        "query_node_id",
        "ordered_causal_parent_identity_sha256",
    ]
    assert causal_parent_identity[
        "target_parent_identity_map_exact_order_must_equal_input_ordered_event_query_nodes"
    ]
    retrospective_terms = interfaces["ETASRetrospectiveLikelihoodTerms"]
    assert retrospective_terms["ordered_target_identity_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_complete_input_ordered_targets"
    )
    assert retrospective_terms[
        "causal_parent_receipt_sha256_must_equal_recomputed_etas_causal_parent_receipt_sha256"
    ]
    assert set(interfaces["ETASRetrospectiveEventTerm"]["field_types"]) == set(
        interfaces["ETASRetrospectiveEventTerm"]["fields"]
    )
    assert (
        "etas_retrospective_likelihood_input_sha256"
        in interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"]
    )
    catalog_receipt = interfaces["ETASIssueSimulationCatalogReceipt"]
    assert simulation_receipt_identity["includes"] == catalog_receipt["fields"]
    assert "branching_process_domain_and_marks" in catalog_receipt["fields"]
    assert "simulation_controls" in catalog_receipt["fields"]
    assert catalog_receipt["ordered_seed_context_digests_exact_count"] == 128
    assert catalog_receipt["ordered_seed_context_digest_item_fields_exact"] == [
        "replicate_index",
        "seed_context_digest_hex",
    ]
    assert catalog_receipt[
        "each_seed_digest_item_must_equal_same_index_replicate_diagnostic_seed_context_digest"
    ]
    assert catalog_receipt[
        "each_replicate_diagnostic_replicate_catalog_sha256_must_equal_same_index_batch_catalog_sha256"
    ]
    assert catalog_receipt["event_cap_status_fields_exact"] == [
        "maximum_events_per_replicate",
        "any_event_cap_hit",
        "ordered_hit_replicate_indices",
        "forecast_valid",
    ]
    assert catalog_receipt["forecast_valid_iff_any_event_cap_hit_false"]
    assert (
        identities["etas_issue_simulation_context_sha256"]["includes"]
        == interfaces["ETASIssueSimulationContext"]["identity_projection_fields_exact"]
    )
    assert (
        identities["etas_issue_forecast_input_sha256"]["includes"]
        == interfaces["ETASIssueForecastInput"]["identity_projection_fields_exact"]
    )
    forecast = contract["issue_forecast_definition"]
    assert forecast["all_future_descendant_generations_included"] is True
    assert forecast["simulation_replicates"] == 128
    assert forecast["sliced_horizons_days"] == [7, 30, 90, 180, 365]
    assert forecast["replicate_index_first"] == 0
    assert forecast["replicate_index_last_inclusive"] == 127
    assert (
        forecast[
            "one_simulated_catalog_per_role_issue_replicate_reused_across_all_horizons_grids_and_magnitude_bins"
        ]
        is True
    )
    domain = forecast["branching_process_domain_and_marks"]
    assert domain["M4_to_lt_M5_events_are_latent_propagating_not_output_events"] is True
    assert domain["every_nonabsorbed_future_event_with_magnitude_gte_4_is_a_propagating_parent"]
    assert domain["propagation_outer_boundary"] == (
        "absorbing_without_spatial_kernel_renormalization"
    )
    assert domain[
        "event_cap_counts_all_attempted_future_ground_events_including_absorbed_and_M4_to_lt_M5"
    ]
    assert domain["output_measure_domain"] == (
        "every_preregistered_full_grid_query_node_with_exact_positive_zero_outside_selected_snapshot_support"
    )
    assert domain["support_partial_cell_measure_uses_frozen_clipped_area_weight"]
    downstream = contract["downstream_dynamic_anomaly_composition_boundary"]
    assert downstream["owner"] == "future_stage4_candidate_pipeline_not_etas_adapter"
    assert downstream["weighting_must_be_per_query_node_before_cell_aggregation"]
    assert downstream[
        "pure_etas_artifact_global_receipt_simulation_and_projection_receipts_must_remain_unchanged"
    ]
    assert contract["magnitude_bins"]["learned_stage4_bin_rate_head_on_etas_track_allowed"] is False
    assert set(contract["separation_property_tests"]) == {
        "mutate_or_append_future_window_targets_leaves_issue_forecast_bytes_and_sha_unchanged",
        "issue_forecast_rejects_any_parent_available_time_after_knowledge_cutoff",
        "alarm_order_accepts_only_ETASIssueForecastField",
        "retrospective_terms_cannot_be_passed_as_alarm_field",
        "baseline_measure_retains_issue_cell_time_bin_node_identity",
        "issue_forecast_exactly_preserves_query_node_and_cell_order",
        "future_target_mutation_cannot_change_seed_context_or_simulation_receipt",
        "same_simulated_catalog_is_reused_across_horizons_grids_and_magnitude_bins",
        "pure_etas_adapter_rejects_anomaly_factor_input_and_preserves_no_increment_variant",
        "outside_support_nodes_and_cells_are_present_with_exact_positive_zero_values",
        "target_micro_move_within_same_frozen_25km_cell_leaves_retrospective_event_intensity_unchanged",
        "unscored_M4_to_lt_M5_window_event_changes_later_intensity_without_creating_event_term",
    }


def test_issue_simulation_output_has_exact_batch_receipt_crosswalk() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    contract = protocol["adapter_contract"]
    interfaces = contract["typed_interfaces"]
    output = interfaces["ETASIssueSimulationOutput"]
    receipt = interfaces["ETASIssueSimulationCatalogReceipt"]
    crosswalk = output["batch_and_receipt_exact_crosswalk"]

    assert set(crosswalk) == {
        "batch_payload_sha256_equal_across",
        "catalog_receipt_sha256_equal_across",
        "direct_receipt_batch_field_equalities_exact",
        "global_receipt_sha256_equal_across",
        "selected_role_snapshot_parameter_must_match_global_receipt_role_mapping_exactly",
        "adapter_code_blob_sha256_equal_across",
        "environment_lock_sha256_equal_across",
        "branching_process_domain_and_marks_exact_crosswalk",
        "simulation_controls_exact_crosswalk",
        "ordered_seed_context_digests_must_have_exact_indices_zero_through_127_and_each_digest_must_recompute_from_frozen_seed_context_and_same_batch_context",
        "each_seed_digest_must_equal_same_index_replicate_diagnostic_seed_context_digest",
        "each_replicate_diagnostic_seed_entropy_must_equal_first_16_digest_bytes_as_unsigned_big_endian_decimal",
        "ordered_replicate_catalog_sha256_exact_crosswalk",
        "every_replicate_catalog_and_diagnostic_index_must_equal_zero_through_127_once_each_in_ascending_order",
        "event_cap_status_must_recompute_exactly_from_ordered_replicate_catalog_diagnostics_and_frozen_maximum",
        "event_cap_hit_requires_forecast_valid_false_and_forbids_any_forecast_projection",
        "any_missing_extra_alias_duplicate_order_context_issue_role_snapshot_parameter_batch_receipt_model_artifact_control_seed_catalog_or_environment_mismatch_action",
    }
    assert len(crosswalk["batch_payload_sha256_equal_across"]) == 4
    assert len(crosswalk["catalog_receipt_sha256_equal_across"]) == 2
    assert set(crosswalk["direct_receipt_batch_field_equalities_exact"]) == {
        "etas_issue_simulation_context_sha256",
        "issue_id",
        "selected_role",
        "selected_snapshot_id",
        "selected_parameter_snapshot_sha256",
    }
    for values in crosswalk["direct_receipt_batch_field_equalities_exact"].values():
        assert len(values) == 3
        assert any(value.startswith("issue_simulation_batch.") for value in values)
        assert any(value.startswith("etas_issue_simulation_catalog_receipt.") for value in values)

    marks = crosswalk["branching_process_domain_and_marks_exact_crosswalk"]
    assert set(marks) == set(receipt["branching_process_domain_and_marks_fields_exact"])
    assert len(marks["maximum_magnitude_hex"]) == 3
    assert len(marks["beta_hex"]) == 4
    assert len(marks["immigrant_density_artifact_sha256"]) == 2
    assert len(marks["propagation_domain_artifact_sha256"]) == 3
    assert marks["ground_magnitude_lower_hex"] == [
        "etas_issue_simulation_catalog_receipt.branching_process_domain_and_marks.ground_magnitude_lower_hex",
        "canonical_python_float64_hex_of_ETASFuturePropagationEvent_magnitude_range_inclusive_lower",
    ]

    controls = crosswalk["simulation_controls_exact_crosswalk"]
    assert set(controls) == set(receipt["simulation_controls_fields_exact"])
    assert receipt["simulation_controls_exact_values"] == {
        "simulation_replicates": 128,
        "longest_horizon_days": 365,
        "maximum_events_per_replicate": 100000,
        "bit_generator": "numpy.random.PCG64",
        "seed_namespace": "etas_issue_forecast",
    }
    assert controls["seed_namespace"] == [
        "etas_issue_simulation_catalog_receipt.simulation_controls.seed_namespace",
        "adapter_contract.issue_forecast_definition.seed_context_contract.namespace",
    ]
    assert (
        receipt["simulation_controls_exact_values"]["seed_namespace"]
        == contract["issue_forecast_definition"]["seed_context_contract"]["namespace"]
    )
    assert len(crosswalk["ordered_replicate_catalog_sha256_exact_crosswalk"]) == 3
    for required in (
        "selected_role_snapshot_parameter_must_match_global_receipt_role_mapping_exactly",
        "ordered_seed_context_digests_must_have_exact_indices_zero_through_127_and_each_digest_must_recompute_from_frozen_seed_context_and_same_batch_context",
        "each_seed_digest_must_equal_same_index_replicate_diagnostic_seed_context_digest",
        "each_replicate_diagnostic_seed_entropy_must_equal_first_16_digest_bytes_as_unsigned_big_endian_decimal",
        "every_replicate_catalog_and_diagnostic_index_must_equal_zero_through_127_once_each_in_ascending_order",
        "event_cap_status_must_recompute_exactly_from_ordered_replicate_catalog_diagnostics_and_frozen_maximum",
        "event_cap_hit_requires_forecast_valid_false_and_forbids_any_forecast_projection",
    ):
        assert crosswalk[required] is True
    assert (
        crosswalk[
            "any_missing_extra_alias_duplicate_order_context_issue_role_snapshot_parameter_batch_receipt_model_artifact_control_seed_catalog_or_environment_mismatch_action"
        ]
        == "reject_atomic_simulation_output_and_generate_no_projection"
    )


def test_qualification_execution_seals_require_clean_remote_frozen_code() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    seal = protocol["qualification_execution_seal"]
    opening = seal["opening_seal"]
    assert opening[
        "created_before_any_new_qualification_attempt_source_open_stat_hash_query_or_bundle_inspection_after_protocol_freeze"
    ]
    repository = opening["repository_requirements"]
    for required in (
        "worktree_clean",
        "head_equals_repair_code_tag_commit",
        "named_upstream_exists",
        "upstream_commit_equals_head",
        "remote_repair_code_tag_resolves_to_head_and_is_verified",
        "protocol_tag_commit_and_remote_tag_verified",
        "protocol_package_paths_and_git_blob_oids_equal_protocol_tag_commit",
        "repair_code_diff_from_protocol_tag_matches_exact_allowlist_and_receipt",
        "remote_identity_and_tag_objects_verified_with_network",
    ):
        assert repository[required] is True
    assert repository["remote_repository_slug"] == "Justin-147/SeismoFlux"
    assert repository["upstream_branch"] == "origin/codex/stage2-etas-numerical-repair"
    assert opening["protocol_package_paths"] == [
        ".gitignore",
        "configs/background_etas_numerical_repair.yaml",
        "data/manifests/etas_numerical_repair_start_manifest.json",
        "docs/background_etas_numerical_repair_protocol.md",
        "docs/phase2_etas_numerical_repair_protocol_acceptance.md",
        "tests/unit/test_background_etas_numerical_repair_protocol.py",
    ]
    assert (
        "tests/unit/test_stage4_anomaly_increment_runtime.py"
        not in opening["protocol_package_paths"]
    )
    assert (
        "docs/restart_handoff_2026-07-19_stage2_etas_repair_protocol.md"
        not in opening["protocol_package_paths"]
    )
    assert opening["every_protocol_package_path_must_exist_as_regular_file_before_protocol_commit"]
    assert opening[
        "every_protocol_package_path_must_resolve_to_git_blob_in_protocol_commit_and_remote_tag"
    ]
    for package_path in opening["protocol_package_paths"]:
        assert Path(package_path).is_file()
    assert PROTOCOL_ACCEPTANCE_PATH in {
        Path(package_path) for package_path in opening["protocol_package_paths"]
    }
    runtime = seal["optimizer_runtime_code_seal"]
    assert runtime["expected_numpy_distribution_version_from_uv_lock"] == "2.4.6"
    assert runtime["expected_scipy_distribution_version_from_uv_lock"] == "1.17.1"
    assert runtime["ordinary_system_python_or_any_other_distribution_version_must_fail_closed"]
    assert runtime["runtime_module_global_minimize_must_be_same_object_as_scipy_optimize_minimize"]
    assert runtime["absolute_paths_hostnames_usernames_or_environment_secrets_allowed"] is False
    assert (
        runtime["fields_exact"]
        == _load_yaml(PROTOCOL_PATH)["outputs"]["public_qualification_seal_schema"][
            "runtime_fields_exact"
        ]
    )
    assert (
        protocol["content_addressing"]["identities"]["optimizer_runtime_code_seal_sha256"][
            "includes"
        ]
        == runtime["fields_exact"][:-1]
    )
    assert {
        "numpy_runtime_config_safe_projection",
        "runtime_callable_dependency_map_sha256",
        "python_runtime_file_sha256_and_size_map",
        "attempt_id",
        "checked_at_utc",
    } <= set(runtime["fields_exact"])
    live_runtime_rechecks = runtime["live_runtime_module_and_native_image_reenumeration"]
    assert live_runtime_rechecks["checkpoints"] == [
        "before_each_snapshot_fit",
        "after_each_snapshot_fit",
        "before_qualification_closing_seal",
    ]
    assert live_runtime_rechecks[
        "canonical_runtime_file_map_must_equal_code_tag_baseline_and_optimizer_runtime_code_seal_byte_for_byte"
    ]
    local_acceptance = seal["local_restricted_input_acceptance_receipt"]
    assert local_acceptance["every_local_restricted_input_must_exist_and_match_frozen_sha256"]
    assert local_acceptance["canonical_local_restricted_input_acceptance_receipt_sha256_required"]
    qualification = seal["qualification_input_seal"]
    assert qualification[
        "created_after_fit_bundle_and_source_access_ledger_are_sealed_before_any_fit"
    ]
    assert {
        "public_source_access_receipt_sha256",
        "local_source_access_ledger_content_sha256",
        "fit_input_manifest_file_and_content_sha256",
        "parent_replay_membership_identity_sha256_by_snapshot",
        "parent_replay_scientific_fit_input_sha256_by_snapshot",
        "local_restricted_input_acceptance_receipt_sha256",
        "optimizer_runtime_code_seal_sha256",
    } <= set(qualification["includes"])
    assert qualification["unchanged_rechecks"] == [
        "before_each_snapshot_fit",
        "after_each_snapshot_fit",
        "before_qualification_result_finalization",
        "before_public_artifact_materialization",
    ]
    interrupted = seal["interrupted_attempts"]
    assert interrupted["retry_may_not_replace_or_delete_prior_diagnostic_rows"] is True
    assert interrupted["selecting_better_retry_result_forbidden"] is True
    assert interrupted["qualification_uses_first_complete_protocol_valid_attempt_only"] is True
    failure_receipt = interrupted["invalid_execution_local_failure_receipt"]
    assert failure_receipt[
        "every_completed_or_observation_list_is_append_only_and_preserves_actual_logical_order"
    ]
    assert failure_receipt[
        "completed_optimizer_and_diagnostic_lists_may_be_sparse_ordered_subsequences_after_a_caught_start_failure"
    ]
    assert failure_receipt["completed_receipt_list_cardinalities"] == {
        "ordered_completed_fit_call_opening_receipt_sha256": "zero_to_five",
        "ordered_completed_optimizer_invocation_receipt_sha256": "zero_to_twenty_five",
        "ordered_completed_fit_call_closing_receipt_sha256": "zero_to_five",
        "ordered_completed_fit_attempt_snapshot_sha256": "zero_to_five",
        "ordered_completed_diagnostic_row_sha256": "zero_to_twenty_five",
        "ordered_optimizer_call_observation_log": "zero_to_twenty_five",
    }
    assert "ordered_optimizer_call_observation_log" in failure_receipt["exact_fields"]
    assert failure_receipt["optimizer_call_observation_status_values_exact"] == [
        "completed_valid",
        "failed",
    ]
    assert {
        "fit_etas_raised_after_optimizer_return_before_returned_result_crosswalk",
        "returned_start_result_crosswalk_failed",
    } <= set(failure_receipt["incomplete_wrapper_failure_phase_values_exact"])
    assert failure_receipt["safe_failure_evidence_kind_values_exact"] == [
        "none",
        "complete_raw_OptimizeResult",
        "raw_schema_failure_type_state_projection",
    ]
    allowed_failure_evidence_kinds = set(failure_receipt["safe_failure_evidence_kind_values_exact"])
    assert all(
        set(contract["allowed_kinds"]) <= allowed_failure_evidence_kinds
        for contract in failure_receipt["safe_failure_evidence_phase_contract"].values()
    )
    assert failure_receipt[
        "every_failed_observation_sha256_must_resolve_to_exactly_one_canonical_observation_file_with_matching_recomputed_preimage"
    ]
    assert failure_receipt[
        "observation_file_set_must_equal_failed_projection_of_observation_log_with_no_missing_extra_or_overwrite"
    ]
    assert failure_receipt[
        "evidence_layer_may_not_reimplement_fit_postprocessing_or_fabricate_completed_invocation_receipt_after_fit_failure"
    ]
    assert failure_receipt[
        "ordered_completed_optimizer_invocation_receipt_sha256_must_equal_completed_valid_projection_of_observation_log"
    ]
    closing = seal["closing_seal"]
    assert closing["all_25_rows_and_five_snapshot_attempts_must_share_exact_attempt_id"]
    assert closing["preclosing_invalid_execution_has_no_closing_qualification_seal"]
    assert {
        "optimizer_runtime_code_seal_sha256",
        "ordered_five_fit_etas_call_opening_receipt_sha256",
        "ordered_25_optimizer_invocation_receipt_sha256",
        "ordered_five_fit_etas_call_closing_receipt_sha256",
    } <= set(closing["includes"])
    assert closing["self_or_mutual_hash_reference_forbidden"] is True
    assert (
        "staged_public_payload_identity_excluding_closing_seal_and_qualification_evidence"
        in (closing["includes"])
    )


def test_qualification_public_result_staging_paths_are_attempt_local_and_fail_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    seal = protocol["qualification_execution_seal"]
    staging = seal["qualification_public_result_staging"]
    outputs = protocol["outputs"]

    local_root = protocol["fit_input_bundle"]["local_root"]
    attempt_root = f"{local_root}/attempts/{{attempt_id}}"
    staged_root = f"{attempt_root}/staged_public"
    assert staging["fit_input_local_root_ref"] == "fit_input_bundle.local_root"
    assert staging["attempt_root_path_template"] == attempt_root
    assert staging["staged_public_root_path_template"] == staged_root
    attempt_id_contract = staging["attempt_id_path_component_contract"]
    assert attempt_id_contract["fullmatch_regex"] == "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    assert attempt_id_contract["single_ascii_component_only"] is True
    assert (
        attempt_id_contract[
            "slash_backslash_colon_control_character_NUL_drive_UNC_empty_dot_and_dot_dot_forbidden"
        ]
        is True
    )
    assert (
        attempt_id_contract[
            "trailing_dot_or_space_and_case_insensitive_windows_reserved_device_basename_forbidden"
        ]
        is True
    )
    assert (
        attempt_id_contract[
            "exact_case_must_match_the_existing_same_attempt_fit_input_directory_name"
        ]
        is True
    )
    assert staging["staged_to_final_record_fields_exact"] == ["staged_path", "final_path"]

    common = staging["common_staged_to_final_path_mapping_exact"]
    expected_common_final_paths = {
        "fit_input_manifest": outputs["fit_input_manifest"],
        "opening_execution_seal": outputs["opening_execution_seal"],
        "optimizer_runtime_code_seal": outputs["optimizer_runtime_code_seal"],
        "qualification_input_seal": outputs["qualification_input_seal"],
        "report": outputs["report"],
        "static_diagnostic": outputs["static_diagnostic"],
        "interactive_diagnostic": outputs["interactive_diagnostic"],
        "qualification_closing_seal": outputs["qualification_closing_seal"],
        "qualification_manifest": outputs["qualification_manifest"],
    }
    assert len(common) == 9
    assert {key: record["final_path"] for key, record in common.items()} == (
        expected_common_final_paths
    )

    evaluable = staging["evaluable_additional_staged_to_final_path_mapping_exact"]
    expected_evaluable_final_paths = {
        "parameter_snapshots": (
            "models/registry/background_etas_numerical_repair/parameter_snapshots.json"
        ),
        "parameter_set_manifest": (
            "models/registry/background_etas_numerical_repair/parameter_set_manifest.json"
        ),
    }
    assert len(evaluable) == 2
    assert {key: record["final_path"] for key, record in evaluable.items()} == (
        expected_evaluable_final_paths
    )
    assert staging["not_evaluable_additional_staged_to_final_path_mapping_exact"] == {}

    preclosing_common_keys = staging["preclosing_common_logical_artifact_keys_exact"]
    assert preclosing_common_keys == [
        "fit_input_manifest",
        "opening_execution_seal",
        "optimizer_runtime_code_seal",
        "qualification_input_seal",
        "report",
        "static_diagnostic",
        "interactive_diagnostic",
    ]
    assert [common[key]["final_path"] for key in preclosing_common_keys] == outputs[
        "canonical_nested_schemas"
    ]["staged_public_payload_identity"]["common_complete_file_paths_exact"]
    assert staging["preclosing_evaluable_additional_logical_artifact_keys_exact"] == [
        "parameter_snapshots",
        "parameter_set_manifest",
    ]
    assert [
        evaluable[key]["final_path"]
        for key in staging["preclosing_evaluable_additional_logical_artifact_keys_exact"]
    ] == outputs["canonical_nested_schemas"]["staged_public_payload_identity"][
        "evaluable_additional_complete_file_paths_exact"
    ]
    assert staging["preclosing_not_evaluable_additional_logical_artifact_keys_exact"] == []

    closing_key = staging["qualification_closing_seal_logical_artifact_key_exact"]
    manifest_key = staging["qualification_manifest_logical_artifact_key_exact"]
    assert closing_key == "qualification_closing_seal"
    assert manifest_key == "qualification_manifest"
    assert closing_key not in preclosing_common_keys
    assert manifest_key not in preclosing_common_keys
    not_evaluable_order = [*preclosing_common_keys, closing_key, manifest_key]
    evaluable_order = [
        *preclosing_common_keys,
        *staging["preclosing_evaluable_additional_logical_artifact_keys_exact"],
        closing_key,
        manifest_key,
    ]
    assert staging["not_evaluable_public_materialization_logical_artifact_order_exact"] == (
        not_evaluable_order
    )
    assert staging["evaluable_public_materialization_logical_artifact_order_exact"] == (
        evaluable_order
    )
    assert not_evaluable_order[-1] == manifest_key
    assert evaluable_order[-1] == manifest_key

    all_records = [*common.values(), *evaluable.values()]
    assert len({record["final_path"] for record in all_records}) == 11
    assert len({record["staged_path"] for record in all_records}) == 11
    for record in all_records:
        assert set(record) == set(staging["staged_to_final_record_fields_exact"])
        assert record["staged_path"] == f"{staged_root}/{record['final_path']}"
        assert not record["final_path"].startswith(("/", "\\"))
        assert ".." not in record["final_path"].split("/")
        assert "\\" not in record["final_path"]

    ignore_probe_attempt_id = "r2-ignore-probe"
    ignored_staged_probe = f"{staged_root}/probe".format(attempt_id=ignore_probe_attempt_id)
    ignored_staged_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", ignored_staged_probe],
        check=False,
    )
    assert ignored_staged_check.returncode == 0
    for record in all_records:
        staged_path = record["staged_path"].format(attempt_id=ignore_probe_attempt_id)
        staged_path_check = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", staged_path],
            check=False,
        )
        assert staged_path_check.returncode == 0, staged_path
        public_path_check = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", record["final_path"]],
            check=False,
        )
        assert public_path_check.returncode == 1, record["final_path"]
        base_tree_check = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", R1_PROTOCOL_COMMIT, "--", record["final_path"]],
            check=False,
            capture_output=True,
            text=True,
        )
        assert base_tree_check.returncode == 0, base_tree_check.stderr
        assert base_tree_check.stdout == "", record["final_path"]

    result_diff = seal["qualification_result_tag_diff_from_repair_code_tag"]
    assert set(expected_common_final_paths.values()) == set(result_diff["common_exact_added_paths"])
    assert set(expected_evaluable_final_paths.values()) == set(
        result_diff["evaluable_additional_exact_paths"]
    )
    assert result_diff["not_evaluable_additional_exact_paths"] == []
    assert result_diff["invalid_execution_tracked_added_paths"] == []
    assert result_diff[
        "every_evaluable_or_not_evaluable_result_path_must_be_absent_in_repair_code_tag_and_have_exact_git_name_status_A_with_null_old_blob_oid"
    ]
    assert result_diff["overwrite_delete_rename_copy_or_other_modified_name_status_forbidden"]
    assert result_diff["exact_name_status_A_blob_oid_and_binary_patch_receipt_required"]

    staged_identity = outputs["canonical_nested_schemas"]["staged_public_payload_identity"]
    assert staged_identity[
        "ordered_complete_file_path_sha256_map_keys_must_equal_common_plus_qualification_status_branch_additional_paths_exactly"
    ]
    assert staged_identity[
        "every_map_key_must_equal_the_final_path_of_exactly_one_same_branch_qualification_public_result_staging_preclosing_record"
    ]
    assert staged_identity[
        "every_map_value_must_equal_sha256_of_complete_reopened_bytes_at_that_same_record_staged_path"
    ]
    assert staged_identity[
        "every_reopened_staged_path_must_be_non_reparse_regular_file_with_stable_size_hash_bytes_and_declared_schema_or_visual_contract"
    ]
    assert staged_identity[
        "aggregate_content_sha256_must_recompute_after_all_mapped_staged_files_are_reopened_and_before_closing_seal"
    ]

    for required in (
        "staged_public_root_must_equal_fit_input_local_root_plus_attempts_attempt_id_staged_public",
        "attempt_root_must_be_the_existing_non_reparse_fit_input_attempt_directory_strictly_below_fit_input_local_root_attempts",
        "staged_public_root_must_be_gitignored_attempt_exclusive_absent_before_staging_and_created_new",
        "preclosing_mappings_must_equal_staged_public_payload_identity_common_and_branch_additional_paths_exactly",
        "branch_materialization_order_must_equal_exact_preclosing_common_then_branch_additional_then_closing_then_manifest_and_manifest_must_be_last",
        "rollback_order_must_be_the_exact_reverse_of_the_successfully_created_prefix_of_the_same_branch_materialization_order",
        "staged_public_payload_identity_ordered_path_sha256_map_key_set_must_equal_same_branch_preclosing_mapping_final_paths_exactly",
        "staged_public_payload_identity_ordered_path_sha256_map_key_must_equal_each_mapped_record_final_path_and_value_must_equal_sha256_of_complete_reopened_bytes_at_same_record_staged_path",
        "staged_public_payload_identity_every_mapped_staged_file_must_reopen_as_non_reparse_regular_file_with_stable_size_sha256_and_complete_bytes_before_and_after_identity_construction",
        "staged_public_payload_identity_every_reopened_staged_file_must_validate_against_its_declared_strict_public_file_schema_or_visualization_contract",
        "staged_public_payload_identity_aggregate_must_recompute_from_exact_branch_status_ordered_path_sha256_map_and_preclosing_manifest_projection_sha256",
        "qualification_closing_seal_must_use_its_independent_staged_path_only_after_preclosing_identity_and_final_clean_repository_identity_are_frozen",
        "qualification_manifest_must_use_its_independent_staged_path_only_after_closing_seal_and_complete_qualification_evidence_and_manifest_content_sha256_are_frozen",
        "closing_seal_and_qualification_manifest_must_not_enter_staged_public_payload_identity",
        "every_staged_path_must_equal_staged_public_root_plus_its_exact_final_repository_relative_path",
        "staged_root_and_every_staged_or_final_path_must_resolve_strictly_inside_its_declared_root_without_symlink_junction_mount_point_or_other_reparse_escape",
        "every_existing_ancestor_must_be_reopened_and_verified_as_the_same_non_reparse_directory_before_each_create_copy_or_remove",
        "every_staged_and_final_file_must_be_absent_before_its_first_creation_and_no_historical_result_may_be_overwritten",
        "every_staged_file_must_use_exclusive_sibling_temp_write_flush_fsync_atomic_no_clobber_install_then_reopen_hash_and_byte_verification",
        "qualification_manifest_staged_file_must_strict_parse_recompute_all_own_and_cross_file_hashes_reserialize_byte_identically_and_reopen_before_public_materialization",
        "public_materialization_may_begin_only_after_all_branch_required_staged_files_closing_seal_and_qualification_manifest_reopen_checks_and_final_clean_repository_recheck_pass",
        "public_materialization_source_bytes_must_be_read_only_from_the_exact_mapped_staged_file_and_never_regenerated",
        "public_materialization_must_use_exclusive_destination_sibling_temp_byte_copy_flush_fsync_reopen_hash_and_byte_verification_atomic_no_clobber_install_then_final_reopen_verification",
        "every_materialized_final_file_must_equal_its_staged_source_in_size_sha256_and_complete_bytes",
        "every_final_result_path_must_be_absent_in_repair_code_tag_and_prepublication_worktree_then_have_git_status_A_only_after_successful_materialization",
        "tracked_status_for_every_materialized_path_must_be_exact_A_with_no_overwrite_delete_rename_or_other_modified_path_allowed",
        "prepublication_recheck_must_verify_clean_repository_unchanged_head_and_upstream_all_final_paths_absent_and_complete_staged_nine_or_eleven_file_set_byte_exact",
        "evaluable_must_materialize_exactly_all_nine_common_and_two_evaluable_additional_files",
        "not_evaluable_must_materialize_exactly_all_nine_common_and_no_additional_files",
        "not_evaluable_parameter_artifact_root_must_be_absent_before_during_and_after_materialization",
        "any_materialization_failure_must_rollback_in_reverse_creation_order_only_same_invocation_new_final_files_that_reopen_as_non_reparse_regular_files_and_still_match_their_exact_staged_bytes",
        "rollback_must_retain_all_attempt_staging_and_local_failure_evidence_and_must_never_remove_preexisting_mismatched_ambiguous_or_unverified_paths",
        "rollback_may_remove_only_attempt_unique_destination_sibling_temps_after_strict_parent_and_name_reverification_and_may_not_recursively_remove_any_directory",
        "rollback_must_reopen_and_verify_every_removed_final_path_is_absent_else_publication_failure_requires_manual_remediation_without_result_commit_or_tag",
        "post_closing_publication_failure_must_preserve_the_reopened_staged_closing_seal_and_any_installed_manifest_or_attempt_unique_manifest_temp_failure_evidence_and_publish_no_result_or_tag",
        "same_attempt_byte_exact_public_materialization_retry_allowed_only_after_complete_verified_rollback_clean_repository_unchanged_head_and_upstream_all_final_paths_absent_and_all_staged_bytes_and_hashes_unchanged",
        "same_attempt_publication_retry_may_not_rerun_fit_recompute_replace_or_modify_any_staged_payload_closing_seal_or_qualification_manifest_or_select_a_new_result",
        "failed_retry_precondition_permanently_invalidates_attempt_for_publication_and_forbids_result_commit_or_tag",
    ):
        assert staging[required] is True
    assert staging[
        "any_preclosing_path_creation_reopen_hash_byte_or_cross_file_mismatch_action"
    ] == ("invalid_execution_without_closing_seal_public_result_commit_or_qualification_result_tag")
    assert staging[
        "any_post_closing_copy_reopen_hash_byte_cross_file_or_rollback_mismatch_action"
    ] == ("publication_failure_without_public_result_commit_or_qualification_result_tag")

    failure_receipt = staging["publication_failure_receipt"]
    assert failure_receipt["path_template"] == (
        f"{attempt_root}/publication_failures/"
        "{publication_failure_sequence_decimal_zero_padded_6}.json"
    )
    assert failure_receipt["classification"] == "local_restricted_gitignored_append_only"
    assert failure_receipt["schema_version_exact"] == 1
    assert failure_receipt["fields_exact"] == [
        "schema_version",
        "attempt_id",
        "publication_failure_sequence",
        "failed_at_utc",
        "failure_phase",
        "failure_code",
        "exception_type_or_null",
        "qualification_closing_seal_sha256",
        "qualification_manifest_staging_state",
        "qualification_manifest_content_sha256_or_null",
        "qualification_manifest_file_sha256_or_null",
        "ordered_created_final_paths",
        "ordered_rolled_back_final_paths",
        "staged_file_size_and_sha256_or_null_by_final_path",
        "repository_identity_after_rollback",
        "rollback_complete",
        "retry_eligible",
        "previous_publication_failure_receipt_sha256_or_null",
        "publication_failure_receipt_sha256",
    ]
    assert failure_receipt["failure_phase_values_exact"] == [
        "qualification_manifest_construction",
        "qualification_manifest_sibling_temp_write",
        "qualification_manifest_atomic_no_clobber_install",
        "qualification_manifest_reopen",
        "qualification_manifest_schema_byte_cross_file_validation",
        "pre_materialization_recheck",
        "destination_sibling_temp_copy",
        "destination_sibling_temp_reopen",
        "atomic_no_clobber_install",
        "final_file_reopen",
        "rollback",
        "post_rollback_recheck",
    ]
    assert failure_receipt["qualification_manifest_staging_state_values_exact"] == [
        "not_constructed",
        "canonical_bytes_constructed_not_validly_reopened",
        "reopened_bytes_not_validated",
        "reopened_valid",
    ]
    manifest_state_by_phase = failure_receipt[
        "qualification_manifest_staging_state_required_by_failure_phase_exact"
    ]
    assert set(manifest_state_by_phase) == set(failure_receipt["failure_phase_values_exact"])
    assert manifest_state_by_phase["qualification_manifest_construction"] == "not_constructed"
    for phase in (
        "qualification_manifest_sibling_temp_write",
        "qualification_manifest_atomic_no_clobber_install",
        "qualification_manifest_reopen",
    ):
        assert manifest_state_by_phase[phase] == "canonical_bytes_constructed_not_validly_reopened"
    assert (
        manifest_state_by_phase["qualification_manifest_schema_byte_cross_file_validation"]
        == "reopened_bytes_not_validated"
    )
    for phase in failure_receipt["failure_phase_values_exact"][5:]:
        assert manifest_state_by_phase[phase] == "reopened_valid"
    assert failure_receipt[
        "qualification_manifest_sha_required_null_state_by_staging_state_exact"
    ] == {
        "not_constructed": {"content_sha256": None, "file_sha256": None},
        "canonical_bytes_constructed_not_validly_reopened": {
            "content_sha256": None,
            "file_sha256": None,
        },
        "reopened_bytes_not_validated": {
            "content_sha256": None,
            "file_sha256": "required",
        },
        "reopened_valid": {"content_sha256": "required", "file_sha256": "required"},
    }
    assert failure_receipt["first_sequence_exact"] == 0
    assert failure_receipt["maximum_sequence_exact"] == 999999
    assert failure_receipt["sequence_exhaustion_action"] == (
        "publication_failure_requires_manual_remediation_without_retry_result_commit_or_tag"
    )
    assert failure_receipt["attempt_id_schema_ref"] == (
        "qualification_execution_seal.qualification_public_result_staging."
        "attempt_id_path_component_contract"
    )
    assert failure_receipt["failed_at_utc_type"] == "canonical_RFC3339_UTC_instant_with_Z"
    assert failure_receipt["failure_code_fullmatch_regex"] == "[a-z][a-z0-9_]{0,127}"
    assert failure_receipt["staged_file_size_or_null_type"] == (
        "strict_nonnegative_base10_integer_or_null_only_for_manifest_before_reopened_bytes"
    )
    assert failure_receipt["staged_file_sha256_or_null_type"] == (
        "lowercase_hex_length_64_or_null_only_for_manifest_before_reopened_bytes"
    )
    assert failure_receipt["rollback_complete_and_retry_eligible_type"] == "strict_boolean"
    assert failure_receipt[
        "staged_file_size_and_sha256_or_null_by_final_path_value_fields_exact"
    ] == [
        "file_size_or_null",
        "file_sha256_or_null",
    ]
    assert (
        failure_receipt["staged_file_size_and_sha256_or_null_by_final_path_key_order"]
        == "unicode_codepoint_ascending"
    )
    assert failure_receipt["repository_identity_after_rollback_schema_ref"] == (
        "outputs.canonical_nested_schemas.final_clean_repository_identity"
    )
    for required in (
        "path_sequence_component_must_equal_sequence_as_exact_six_digit_zero_padded_decimal",
        "each_later_sequence_must_equal_previous_sequence_plus_one",
        "previous_receipt_sha256_is_null_only_for_sequence_zero_else_must_equal_immediately_previous_reopened_receipt_own_sha256",
        "ordered_created_final_paths_must_equal_successfully_created_prefix_of_same_branch_materialization_order",
        "ordered_rolled_back_final_paths_must_equal_exact_successful_reverse_rollback_projection_of_ordered_created_final_paths",
        "any_extra_missing_alias_duplicate_unknown_or_noncanonical_top_level_or_nested_field_forbidden",
        "staged_file_size_and_sha256_or_null_by_final_path_key_set_must_equal_complete_same_branch_nine_or_eleven_final_paths",
        "every_non_manifest_staged_file_size_and_sha256_value_must_be_nonnull_and_equal_reopened_same_attempt_mapped_staged_regular_file_complete_bytes",
        "manifest_staged_file_size_and_sha256_value_must_be_null_null_before_reopened_bytes_and_nonnull_equal_reopened_complete_bytes_for_reopened_bytes_not_validated_or_reopened_valid",
        "qualification_closing_seal_sha256_must_equal_reopened_same_attempt_staged_closing_seal_own_sha256",
        "nonnull_qualification_manifest_content_sha256_must_equal_reopened_valid_same_attempt_staged_manifest_own_content_sha256",
        "nonnull_qualification_manifest_file_sha256_must_equal_sha256_of_complete_reopened_same_attempt_staged_manifest_file_bytes",
        "manifest_staging_failure_phases_require_empty_created_and_rolled_back_final_paths_retry_eligible_false_and_no_same_attempt_materialization_retry",
        "retry_eligible_true_iff_rollback_complete_repository_clean_head_upstream_unchanged_all_final_paths_absent_and_all_staged_sizes_hashes_and_bytes_unchanged",
        "same_attempt_receipt_files_must_be_complete_gapless_append_only_hash_chain_and_may_never_be_overwritten_truncated_deleted_or_reordered",
    ):
        assert failure_receipt[required] is True
    failure_receipt_probe = failure_receipt["path_template"].format(
        attempt_id=ignore_probe_attempt_id,
        publication_failure_sequence_decimal_zero_padded_6="000000",
    )
    failure_receipt_ignore_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", failure_receipt_probe],
        check=False,
    )
    assert failure_receipt_ignore_check.returncode == 0


def test_issue_forecast_seed_context_exact_bytes_and_reference_vector() -> None:
    forecast = _load_yaml(PROTOCOL_PATH)["adapter_contract"]["issue_forecast_definition"]
    seed = forecast["seed_context_contract"]
    reference = seed["reference_vector"]
    context = SeedContext(
        root_seed=reference["root_seed"],
        protocol_version=reference["protocol_version"],
        namespace=reference["namespace"],
        model_id=reference["model_id"],
        issue_id=reference["issue_id"],
        replicate_index=reference["replicate_index"],
    )
    assert list(context.fields()) == reference["expected_fields"]
    payload = b"\x00".join(field.encode("utf-8") for field in context.fields())
    assert hashlib.sha256(payload).hexdigest() == reference["digest_sha256"]
    assert str(context.entropy()) == reference["entropy_uint128_decimal"]
    assert f"{context.entropy():032x}" == reference["entropy_uint128_hex"]
    assert [float(value).hex() for value in context.generator().random(4)] == reference[
        "pcg64_first_four_uniform_float64_hex"
    ]
    assert seed["fields_in_exact_order"] == [
        "literal_seismoflux",
        "root_seed_base10_without_leading_zero",
        "protocol_version",
        "namespace",
        "model_id",
        "issue_id",
        "replicate_index_zero_padded_eight_decimal_digits",
    ]
    assert seed["replicate_output_order"] == "replicate_index_ascending"
    assert seed["grid_horizon_query_node_or_output_bin_changes_must_not_change_seed_context_digest"]

    local_issue = datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert local_issue.astimezone(UTC) == datetime(2019, 12, 31, 16, tzinfo=UTC)


def test_adapter_artifacts_require_clean_remote_frozen_adapter_code() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    scope = protocol["adapter_code_scope_from_positive_qualification_result_tag"]
    assert scope[
        "qualification_result_artifact_paths_must_be_blob_identical_to_positive_result_tag"
    ]
    assert scope["protocol_package_paths_must_be_blob_identical_to_protocol_tag"]
    assert scope["any_other_tracked_path_change_forbidden"] is True
    code_payload = scope["adapter_code_payload_identity"]
    assert code_payload["schema_version_exact"] == 1
    assert code_payload["field_types"] == {
        "schema_version": "strict_base10_integer",
        "adapter_code_tag_commit": "lowercase_git_commit_oid_length_40",
        "ordered_path_records": "ordered_list",
        "adapter_code_blob_sha256": "lowercase_hex_length_64",
    }
    assert code_payload["ordered_path_records_item_field_types"] == {
        "repository_relative_path": "exact_allowlisted_forward_slash_repository_relative_path",
        "git_blob_oid": "lowercase_git_blob_oid_length_40",
        "file_sha256": "lowercase_hex_length_64",
        "file_size": "nonnegative_strict_base10_integer",
    }
    assert code_payload["ordered_paths_exact_ref"] == (
        "adapter_code_scope_from_positive_qualification_result_tag.allowed_changed_or_added_paths"
    )
    assert code_payload["ordered_path_records_item_fields_exact"] == [
        "repository_relative_path",
        "git_blob_oid",
        "file_sha256",
        "file_size",
    ]
    assert code_payload["adapter_code_blob_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_schema_version_adapter_code_tag_commit_"
        "and_complete_ordered_path_records"
    )
    seal = protocol["adapter_artifact_execution_seal"]
    attempt = seal["attempt_protocol"]
    assert attempt["adapter_artifact_attempt_id_required"]
    assert attempt["independent_gitignored_staging_directory_per_attempt"]
    assert attempt["interrupted_attempt_directory_and_failure_receipt_must_be_retained"]
    assert attempt["overwrite_delete_or_reuse_prior_attempt_directory_forbidden"]
    assert attempt["only_first_complete_protocol_valid_attempt_may_be_published"]
    ledger = attempt["append_only_attempt_ledger"]
    assert ledger["event_type_values"] == ["intent", "ready_to_close", "completed", "aborted"]
    assert set(ledger["event_type_required_null_state_matrix"]) == {
        "intent",
        "ready_to_close",
        "completed",
        "aborted",
    }
    assert ledger["event_type_required_null_state_matrix"]["completed"]["protocol_valid_exact"]
    assert (
        ledger["event_type_required_null_state_matrix"]["aborted"]["protocol_valid_exact"] is False
    )
    assert set(ledger["aborted_failure_phase_sha_matrix"]) == {
        "before_opening",
        "after_opening_before_ready",
        "after_ready_before_closing",
        "after_closing_before_completed",
    }
    assert ledger["completed_requires_ready_to_close_immediately_before_it"]
    assert ledger["aborted_predecessor_and_sha_state_must_match_declared_failure_phase"]
    assert ledger["top_level_fields_exact"] == [
        "schema_version",
        "entries",
        "ledger_content_sha256",
    ]
    assert ledger["first_entry_sequence_exact"] == 0
    assert ledger[
        "every_subsequent_previous_entry_sha256_must_equal_immediately_preceding_entry_recomputed_sha256"
    ]
    empty_ledger_reference = ledger["empty_ledger_reference_vector"]
    assert (
        _canonical_payload_sha256(empty_ledger_reference["canonical_payload"])
        == (empty_ledger_reference["ledger_content_sha256"])
    )
    opening = seal["opening_seal"]
    repository = opening["repository_requirements"]
    for required in (
        "worktree_clean",
        "head_equals_adapter_code_tag_commit",
        "upstream_commit_equals_head",
        "remote_adapter_code_tag_resolves_to_head_and_is_verified_with_network",
        "remote_positive_qualification_result_tag_and_commit_verified",
        "adapter_source_and_test_blob_oids_equal_adapter_code_tag",
        "protocol_package_blob_oids_equal_protocol_tag",
        "adapter_code_diff_from_positive_result_tag_matches_exact_allowlist_and_receipt",
        "qualification_result_artifact_blob_oids_equal_positive_result_tag",
        "repair_and_core_dependency_blob_oids_equal_positive_result_tag",
        "every_other_tracked_path_is_unchanged_or_an_exact_allowed_adapter_path",
    ):
        assert repository[required] is True
    assert repository["remote_repository_slug"] == "Justin-147/SeismoFlux"
    runtime = seal["artifact_runtime_preflight"]
    assert runtime[
        "created_after_attempt_intent_fsync_and_before_opening_seal_or_any_local_or_public_artifact_generation"
    ]
    assert (
        runtime[
            "target_real_catalog_fit_input_event_row_anomaly_score_or_historical_result_access_allowed"
        ]
        is False
    )
    assert runtime["expected_shapely_distribution_version"] == "2.1.2"
    assert runtime["schema_version_exact"] == 1
    assert runtime["scalar_field_types"] == {
        "schema_version": "strict_base10_integer",
        "checked_at_utc": "canonical_RFC3339_UTC_instant_with_Z",
        "adapter_artifact_attempt_id": "nonempty_unicode_NFC_string",
        "adapter_code_tag_commit": "lowercase_git_commit_oid_length_40",
        "adapter_code_blob_sha256": "lowercase_hex_length_64",
        "environment_lock_sha256": "lowercase_hex_length_64",
        "shapely_dist_info_RECORD_sha256": "lowercase_hex_length_64",
        "shapely_complete_installed_distribution_verification_map_sha256": (
            "lowercase_hex_length_64"
        ),
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256": (
            "lowercase_hex_length_64"
        ),
        "adapter_runtime_file_sha256_and_size_map_sha256": "lowercase_hex_length_64",
        "adapter_geometry_callable_dependency_map_sha256": "lowercase_hex_length_64",
        "adapter_artifact_runtime_seal_sha256": "lowercase_hex_length_64",
    }
    assert {
        "adapter_code_blob_sha256",
        "isolated_launcher_identity",
        "synthetic_adapter_geometry_warmup_receipt",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_complete_installed_distribution_verification_map_sha256",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256",
        "adapter_runtime_file_sha256_and_size_map",
        "adapter_runtime_file_sha256_and_size_map_sha256",
        "adapter_geometry_callable_dependency_map",
        "adapter_geometry_callable_dependency_map_sha256",
        "qualification_runtime_shared_identity_crosswalk",
        "adapter_artifact_runtime_seal_sha256",
    } <= set(runtime["fields_exact"])
    assert runtime["synthetic_adapter_geometry_warmup"]["receipt_schema_version_exact"] == 1
    assert (
        "isolated_launcher_identity"
        in runtime["qualification_runtime_shared_identity_fields_exact"]
    )
    nested_runtime_object_fields = {
        "isolated_launcher_identity",
        "synthetic_adapter_geometry_warmup_receipt",
        "windows_runtime_identity",
        "python_implementation_version_abi_platform_and_executable_sha256",
        "shapely_distribution_name_and_version",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "adapter_runtime_file_sha256_and_size_map",
        "adapter_geometry_callable_dependency_map",
        "qualification_runtime_shared_identity_crosswalk",
    }
    assert set(runtime["scalar_field_types"]).isdisjoint(nested_runtime_object_fields)
    assert set(runtime["scalar_field_types"]) | nested_runtime_object_fields == set(
        runtime["fields_exact"]
    )
    adapter_runtime_schema = protocol["outputs"]["canonical_nested_schemas"][
        "adapter_artifact_runtime_seal"
    ]
    assert adapter_runtime_schema["scalar_field_types_ref"] == (
        "adapter_artifact_execution_seal.artifact_runtime_preflight.scalar_field_types"
    )
    assert adapter_runtime_schema["python_runtime_identity_schema_ref"] == (
        "outputs.canonical_nested_schemas.python_runtime_identity"
    )
    shared_identity_crosswalk_schema = protocol["outputs"]["canonical_nested_schemas"][
        "qualification_runtime_shared_identity_crosswalk"
    ]
    assert shared_identity_crosswalk_schema["field_types"] == {
        "positive_qualification_optimizer_runtime_code_seal_sha256": ("lowercase_hex_length_64"),
        "shared_field_names_exact": "ordered_list_of_exact_runtime_field_names",
        "qualification_runtime_shared_identity_crosswalk_sha256": ("lowercase_hex_length_64"),
    }
    assert runtime[
        "same_isolated_launcher_and_single_thread_environment_contract_as_qualification_runtime_required"
    ]
    assert runtime["synthetic_adapter_geometry_warmup"]["operation_order_exact"] == [
        "construct_fixed_box_and_point",
        "fixed_buffer",
        "normalize_geometry",
        "serialize_big_endian_2D_WKB_without_SRID",
        "deserialize_WKB",
        "compare_roundtrip_geometry_equals_normalized_buffer",
        "covers_including_boundary",
    ]
    warmup = runtime["synthetic_adapter_geometry_warmup"]
    assert warmup["receipt_fields_exact"] == [
        "schema_version",
        "operation_order",
        "canonical_input_payload",
        "canonical_input_payload_sha256",
        "canonical_output_payload",
        "canonical_output_payload_sha256",
        "shapely_runtime_binding_identity",
        "synthetic_adapter_geometry_warmup_receipt_sha256",
    ]
    assert warmup["receipt_field_types"] == {
        "schema_version": "strict_base10_integer",
        "operation_order": "ordered_list_of_nonempty_unicode_NFC_strings",
        "canonical_input_payload": "strict_object",
        "canonical_input_payload_sha256": "lowercase_hex_length_64",
        "canonical_output_payload": "strict_object",
        "canonical_output_payload_sha256": "lowercase_hex_length_64",
        "shapely_runtime_binding_identity": "strict_object",
        "synthetic_adapter_geometry_warmup_receipt_sha256": "lowercase_hex_length_64",
    }
    expected_warmup_input = {
        "box_min_max_xy_python_float_hex": [
            "0x0.0p+0",
            "0x0.0p+0",
            "0x1.0000000000000p+1",
            "0x1.0000000000000p+1",
        ],
        "point_xy_python_float_hex": [
            "0x1.0000000000000p+0",
            "0x1.0000000000000p+0",
        ],
        "buffer_distance_python_float_hex": "0x1.0000000000000p+0",
        "buffer_quad_segs": 8,
        "wkb_byte_order": 0,
        "wkb_output_dimension": 2,
        "wkb_include_srid": False,
        "wkb_flavor": "extended",
    }
    assert warmup["canonical_input_payload_exact"] == expected_warmup_input
    assert list(expected_warmup_input) == warmup["canonical_input_payload_fields_exact"]
    assert warmup["canonical_input_payload_field_types"] == {
        "box_min_max_xy_python_float_hex": ("exact_length_4_ordered_python_float64_hex_list"),
        "point_xy_python_float_hex": "exact_length_2_ordered_python_float64_hex_list",
        "buffer_distance_python_float_hex": "finite_python_float64_hex",
        "buffer_quad_segs": "positive_strict_base10_integer",
        "wkb_byte_order": "strict_base10_integer",
        "wkb_output_dimension": "strict_base10_integer",
        "wkb_include_srid": "strict_boolean",
        "wkb_flavor": "nonempty_unicode_NFC_string",
    }
    assert warmup["canonical_output_payload_field_types"] == {
        "normalized_buffer_big_endian_2D_WKB_lowercase_hex": ("nonempty_even_length_lowercase_hex"),
        "roundtrip_geometry_big_endian_2D_WKB_lowercase_hex": (
            "nonempty_even_length_lowercase_hex"
        ),
        "roundtrip_wkb_byte_equal": "strict_boolean",
        "roundtrip_geometry_equals_normalized_buffer": "strict_boolean",
        "roundtrip_geometry_covers_fixed_point": "strict_boolean",
    }
    assert set(warmup["canonical_output_payload_derivation_exact"]) == set(
        warmup["canonical_output_payload_fields_exact"]
    )
    assert _canonical_payload_sha256(expected_warmup_input) == (
        "26bef17f5fc6fe2a3c6dc1690eeaa7de8faa9f43e735190ab2b5809caa6268a0"
    )
    assert warmup["canonical_input_and_output_payload_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_complete_payload"
    )
    assert warmup[
        "both_sibling_payload_sha256_values_must_be_recomputed_from_their_complete_exact_payload_before_receipt_sha256"
    ]
    assert warmup["callable_bindings_exact"] == [
        "shapely.geometry.box",
        "shapely.geometry.Point",
        "shapely.buffer",
        "shapely.normalize",
        "shapely.to_wkb",
        "shapely.from_wkb",
        "shapely.equals",
        "shapely.covers",
    ]
    box_coordinates = [
        float.fromhex(value)
        for value in cast(list[str], expected_warmup_input["box_min_max_xy_python_float_hex"])
    ]
    point_coordinates = [
        float.fromhex(value)
        for value in cast(list[str], expected_warmup_input["point_xy_python_float_hex"])
    ]
    fixed_box = shapely.box(*box_coordinates)
    fixed_point = Point(*point_coordinates)
    fixed_buffer = shapely.buffer(
        fixed_box,
        float.fromhex(cast(str, expected_warmup_input["buffer_distance_python_float_hex"])),
        quad_segs=cast(int, expected_warmup_input["buffer_quad_segs"]),
    )
    normalized_buffer = shapely.normalize(fixed_buffer)
    normalized_wkb = shapely.to_wkb(
        normalized_buffer,
        hex=False,
        output_dimension=expected_warmup_input["wkb_output_dimension"],
        byte_order=expected_warmup_input["wkb_byte_order"],
        include_srid=expected_warmup_input["wkb_include_srid"],
        flavor=expected_warmup_input["wkb_flavor"],
    )
    assert isinstance(normalized_wkb, bytes)
    roundtrip_geometry = shapely.from_wkb(normalized_wkb)
    roundtrip_wkb = shapely.to_wkb(
        roundtrip_geometry,
        hex=False,
        output_dimension=expected_warmup_input["wkb_output_dimension"],
        byte_order=expected_warmup_input["wkb_byte_order"],
        include_srid=expected_warmup_input["wkb_include_srid"],
        flavor=expected_warmup_input["wkb_flavor"],
    )
    assert isinstance(roundtrip_wkb, bytes)
    observed_warmup_output = {
        "normalized_buffer_big_endian_2D_WKB_lowercase_hex": normalized_wkb.hex(),
        "roundtrip_geometry_big_endian_2D_WKB_lowercase_hex": roundtrip_wkb.hex(),
        "roundtrip_wkb_byte_equal": roundtrip_wkb == normalized_wkb,
        "roundtrip_geometry_equals_normalized_buffer": bool(
            shapely.equals(roundtrip_geometry, normalized_buffer)
        ),
        "roundtrip_geometry_covers_fixed_point": bool(
            shapely.covers(roundtrip_geometry, fixed_point)
        ),
    }
    assert list(observed_warmup_output) == warmup["canonical_output_payload_fields_exact"]
    assert {
        key: observed_warmup_output[key] for key in warmup["canonical_output_boolean_values_exact"]
    } == warmup["canonical_output_boolean_values_exact"]
    assert len(_canonical_payload_sha256(observed_warmup_output)) == 64
    warmup_schema = protocol["outputs"]["canonical_nested_schemas"][
        "synthetic_adapter_geometry_warmup_receipt"
    ]
    assert warmup_schema["canonical_input_payload_fields_exact_ref"].endswith(
        ".canonical_input_payload_fields_exact"
    )
    assert warmup_schema["canonical_output_payload_fields_exact_ref"].endswith(
        ".canonical_output_payload_fields_exact"
    )
    assert warmup_schema["canonical_input_and_output_payload_sha256_formula_ref"].endswith(
        ".canonical_input_and_output_payload_sha256_formula"
    )
    assert runtime[
        "every_shared_identity_field_must_equal_same_field_in_positive_qualification_optimizer_runtime_code_seal_byte_for_byte"
    ]
    closing = seal["closing_seal"]
    assert closing["self_or_mutual_hash_reference_forbidden"] is True
    assert {
        "adapter_artifact_attempt_id",
        "adapter_artifact_opening_seal_sha256",
        "etas_artifact_sha256",
        "frozen_etas_comparator_receipt_sha256",
        "staged_adapter_payload_identity_excluding_closing_seal",
    } <= set(closing["includes"])
    outputs = protocol["outputs"]
    public_paths = outputs["adapter_public_artifacts"]
    assert public_paths["static_contract_visual"] == "docs/background_etas_comparator_contract.svg"
    assert public_paths["interactive_contract_visual"] == (
        "outputs/interactive/background_etas_comparator/index.html"
    )
    public_schema = outputs["adapter_public_artifact_schemas"]
    assert {
        "adapter_code_blob_sha256",
        "adapter_artifact_runtime_seal_sha256",
    } <= set(public_schema["artifact_manifest_fields_exact"])
    assert {
        "adapter_code_blob_sha256",
        "adapter_artifact_runtime_seal",
    } <= set(public_schema["opening_seal_fields_exact"])
    assert set(public_schema["artifact_manifest_nested_field_schema_refs"]) == {
        "model_spec_by_snapshot",
        "immigrant_density_artifact_sha256_by_snapshot",
        "propagation_domain_artifact_sha256_by_snapshot",
        "five_etas_parameter_snapshot_sha256",
    }
    model_specs = outputs["canonical_nested_schemas"]["adapter_model_spec_by_snapshot"]
    assert model_specs["keys_exact"] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    ]
    assert model_specs[
        "every_value_must_equal_same_snapshot_positive_qualification_parameter_snapshot_and_scientific_fit_input_model_spec_field_for_field_and_byte_for_byte"
    ]
    assert set(public_schema["opening_seal_nested_field_schema_refs"]) == {
        "repository_identity",
        "adapter_code_commit_and_remote_tag",
        "adapter_artifact_runtime_seal",
        "positive_qualification_result_tag_commit",
        "five_etas_parameter_snapshot_sha256",
        "protocol_package_tree_identity",
    }
    assert set(public_schema["global_receipt_nested_field_schema_refs"]) == {
        "ordered_complete_stage4_role_mapping",
        "etas_parameter_snapshot_sha256_by_role",
        "protocol_code_environment_hashes",
    }
    staged_adapter_schema = outputs["canonical_nested_schemas"]["staged_adapter_payload_identity"]
    assert staged_adapter_schema["logical_artifact_keys_exact"] == [
        "opening_seal",
        "artifact_manifest",
        "global_comparator_receipt",
        "report",
        "static_contract_visual",
        "interactive_contract_visual",
    ]
    assert public_schema["publication_manifest_self_path"] == (
        "data/manifests/background_etas_comparator_publication.json"
    )
    assert public_schema[
        "exact_public_path_file_sha256_map_keys_must_equal_comparator_receipt_tag_exact_added_paths_excluding_publication_manifest_self_path"
    ]
    assert public_schema[
        "publication_manifest_self_path_must_not_appear_in_its_internal_file_sha256_map"
    ]
    assert public_schema[
        "publication_manifest_final_file_sha256_is_frozen_only_by_result_commit_and_remote_annotated_tag_not_embedded_in_self"
    ]
    assert public_schema["contract_visual_any_unlisted_field_or_external_network_request_forbidden"]


def test_adapter_public_result_package_has_strict_cross_file_identity_closure() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    outputs = protocol["outputs"]
    public_schema = outputs["adapter_public_artifact_schemas"]
    invariants = public_schema["adapter_public_result_cross_file_invariants"]

    assert invariants["adapter_artifact_attempt_id_equal_across"] == [
        "opening_seal.adapter_artifact_attempt_id",
        "artifact_manifest.adapter_artifact_attempt_id",
        "closing_seal.adapter_artifact_attempt_id",
        "publication_manifest.adapter_artifact_attempt_id",
        "selected_attempt_ledger.intent.adapter_artifact_attempt_id",
        "selected_attempt_ledger.ready_to_close.adapter_artifact_attempt_id",
        "selected_attempt_ledger.completed.adapter_artifact_attempt_id",
    ]
    assert invariants[
        "global_receipt_same_attempt_must_be_uniquely_bound_by_its_exact_opening_seal_and_artifact_manifest_hashes"
    ]
    assert set(invariants) >= {
        "adapter_artifact_opening_seal_sha256_equal_across",
        "adapter_code_blob_sha256_equal_across",
        "adapter_artifact_runtime_seal_sha256_equal_across",
        "etas_artifact_sha256_equal_across",
        "frozen_etas_comparator_receipt_sha256_equal_across",
        "adapter_artifact_closing_seal_sha256_equal_across",
        "publication_manifest_content_sha256_must_equal_recomputed_canonical_content_identity",
        "selected_attempt_ledger_entry_and_prefix_equalities",
        "parameter_and_qualification_equalities",
        "code_and_environment_equalities",
        "staged_logical_artifact_path_mapping_exact",
        "final_public_logical_artifact_path_mapping_exact",
        "complete_nonpublication_public_materialization_path_mapping_exact",
    }
    assert len(invariants["adapter_artifact_opening_seal_sha256_equal_across"]) == 8
    assert invariants["adapter_code_blob_sha256_equal_across"] == [
        "recomputed_adapter_code_payload_identity.adapter_code_blob_sha256",
        "opening_seal.adapter_code_blob_sha256",
        "opening_seal.adapter_artifact_runtime_seal.adapter_code_blob_sha256",
        "artifact_manifest.adapter_code_blob_sha256",
        "global_receipt.protocol_code_environment_hashes.adapter_code_blob_sha256",
    ]
    assert invariants["adapter_artifact_runtime_seal_sha256_equal_across"] == [
        "recomputed_opening_seal.adapter_artifact_runtime_seal."
        "adapter_artifact_runtime_seal_sha256",
        "opening_seal.adapter_artifact_runtime_seal.adapter_artifact_runtime_seal_sha256",
        "artifact_manifest.adapter_artifact_runtime_seal_sha256",
        "local_payload_manifest.adapter_artifact_runtime_seal_sha256",
        "every_immigrant_density_payload.adapter_artifact_runtime_seal_sha256",
        "every_propagation_domain_payload.adapter_artifact_runtime_seal_sha256",
        "global_receipt.protocol_code_environment_hashes.adapter_artifact_runtime_seal_sha256",
    ]
    assert len(invariants["etas_artifact_sha256_equal_across"]) == 7
    assert len(invariants["frozen_etas_comparator_receipt_sha256_equal_across"]) == 6
    assert len(invariants["adapter_artifact_closing_seal_sha256_equal_across"]) == 4

    parameters = invariants["parameter_and_qualification_equalities"]
    assert set(parameters) == {
        "etas_parameter_set_sha256_equal_across",
        "five_etas_parameter_snapshot_sha256_equal_across",
        "global_role_parameter_hashes_must_equal_frozen_role_projection_of_five_snapshot_map",
        "global_role_projection_exact",
        "etas_numerical_qualification_evidence_sha256_equal_across",
        "artifact_model_spec_by_snapshot_must_equal_same_snapshot_model_spec_in_each_positive_qualification_parameter_snapshot_and_scientific_fit_input",
        "artifact_adapter_code_blob_sha256_must_equal_recomputed_canonical_adapter_code_payload_at_verified_remote_adapter_code_tag_commit",
    }
    assert parameters["global_role_projection_exact"] == {
        "development": "fold_4",
        "formal_validation": "final_validation",
        "prospective": "final_validation",
    }
    code_environment = invariants["code_and_environment_equalities"]
    assert set(code_environment) == {
        "protocol_tag_commit",
        "repair_code_tag_commit",
        "positive_qualification_result_tag_commit",
        "adapter_code_tag_commit",
        "environment_lock_sha256",
        "adapter_code_diff_receipt_sha256",
        "adapter_code_blob_sha256",
        "adapter_artifact_runtime_seal_sha256",
    }
    repository_schema = outputs["canonical_nested_schemas"]["adapter_repository_identity"]
    assert {
        "remote_protocol_tag_commit",
        "remote_repair_code_tag_commit",
        "remote_positive_qualification_result_tag_commit",
        "remote_adapter_code_tag_commit",
    } <= set(repository_schema["fields_exact"])
    assert repository_schema["every_remote_tag_commit_must_be_network_verified_before_opening_seal"]
    assert invariants["acyclic_construction_and_reference_order_exact"] == [
        "attempt_ledger_intent_append_and_fsync",
        "adapter_runtime_preflight_and_staged_opening_seal",
        "ten_local_restricted_artifact_payloads",
        "local_restricted_payload_manifest",
        "artifact_manifest",
        "global_receipt",
        "report",
        "static_contract_visual",
        "interactive_contract_visual",
        "attempt_ledger_ready_to_close",
        "staged_closing_seal",
        "public_materialization_and_reopen_verification_of_seven_nonpublication_files",
        "attempt_ledger_completed",
        "publication_manifest",
        "result_commit_and_remote_annotated_tag",
    ]

    assert invariants["final_public_logical_artifact_path_mapping_exact"] == {
        key: outputs["adapter_public_artifacts"][key]
        for key in outputs["canonical_nested_schemas"]["staged_adapter_payload_identity"][
            "logical_artifact_keys_exact"
        ]
    }
    staged_root = invariants["staged_public_root_path_template"]
    assert staged_root.endswith("/{adapter_artifact_attempt_id}/staged_public")
    assert set(invariants["staged_logical_artifact_path_mapping_exact"]) == set(
        invariants["final_public_logical_artifact_path_mapping_exact"]
    )
    assert all(
        path.startswith(f"{staged_root}/")
        for path in invariants["staged_logical_artifact_path_mapping_exact"].values()
    )
    assert invariants["closing_seal_staged_path_template"].startswith(f"{staged_root}/")
    assert "closing_seal" not in invariants["staged_logical_artifact_path_mapping_exact"]
    assert len(invariants["staged_logical_artifact_path_mapping_exact"]) == 6
    expected_nonpublication_mapping = {
        "opening_seal": "data/manifests/background_etas_adapter_opening_seal.json",
        "artifact_manifest": (
            "models/registry/background_etas_comparator/etas_artifact_manifest.json"
        ),
        "global_comparator_receipt": "data/manifests/background_etas_comparator_receipt.json",
        "closing_seal": "data/manifests/background_etas_adapter_closing_seal.json",
        "report": "docs/background_etas_comparator_report.md",
        "static_contract_visual": "docs/background_etas_comparator_contract.svg",
        "interactive_contract_visual": "outputs/interactive/background_etas_comparator/index.html",
    }
    assert len(expected_nonpublication_mapping) == 7
    assert (
        invariants["complete_nonpublication_public_materialization_path_mapping_exact"]
        == expected_nonpublication_mapping
    )
    publication_self_path = outputs["adapter_public_artifact_schemas"][
        "publication_manifest_self_path"
    ]
    expected_nonpublication_paths = [
        path
        for path in outputs["adapter_public_artifacts"]["comparator_receipt_tag_exact_added_paths"]
        if path != publication_self_path
    ]
    assert len(expected_nonpublication_paths) == 7
    assert set(expected_nonpublication_mapping.values()) == set(expected_nonpublication_paths)
    assert (
        invariants["closing_seal_final_public_path_exact"]
        == outputs["adapter_public_artifacts"]["closing_seal"]
    )
    for required in (
        "every_staged_logical_artifact_hash_must_equal_sha256_of_exact_staged_file_bytes_at_mapped_path",
        "every_staged_logical_artifact_file_must_be_byte_identical_after_public_materialization",
        "staged_logical_hash_must_equal_publication_path_map_value_for_same_materialized_path",
        "staged_payload_identity_recomputed_aggregate_must_equal_closing_seal_staged_adapter_payload_identity",
        "publication_path_map_value_for_every_key_must_equal_sha256_of_exact_final_repository_relative_path_file_bytes",
        "publication_path_map_must_be_recomputed_after_materialization_and_completed_then_before_publication_manifest_atomic_write",
        "deterministic_publication_manifest_atomic_write_may_be_retried_after_completed_only_if_final_ledger_and_all_seven_file_bytes_are_unchanged",
        "publication_path_map_self_path_excluded_and_final_self_file_sha_frozen_only_by_result_commit_and_remote_tag",
        "all_seven_nonpublication_final_paths_must_be_absent_before_materialization_and_no_historical_result_may_be_overwritten",
        "seven_file_materialization_must_stage_destination_sibling_temps_flush_fsync_reopen_hash_then_atomic_replace_and_reopen_verify_each_exact_byte_payload",
        "any_materialization_failure_before_completed_must_remove_only_same_attempt_newly_created_final_files_after_exact_hash_and_attempt_identity_match_retain_all_attempt_staging_and_append_after_closing_before_completed_aborted",
        "completed_may_be_appended_only_after_all_seven_nonpublication_final_files_are_present_byte_identical_and_match_their_staged_or_closing_hashes",
        "opening_may_not_reference_artifact_global_closing_publication_or_final_ledger_hash",
        "artifact_may_not_reference_global_closing_publication_or_final_ledger_hash",
        "global_receipt_may_not_reference_closing_publication_or_final_ledger_hash",
        "closing_may_not_reference_publication_or_final_ledger_hash",
        "staged_logical_map_may_not_contain_closing_publication_attempt_ledger_or_its_own_aggregate_as_a_logical_artifact",
        "all_six_preclosing_staged_files_must_use_temp_write_flush_fsync_atomic_replace_reopen_hash_and_byte_verification_before_ready_to_close",
        "closing_seal_staged_file_must_use_temp_write_flush_fsync_atomic_replace_reopen_hash_and_byte_verification_after_ready_to_close",
    ):
        assert invariants[required] is True
    assert (
        invariants[
            "any_missing_extra_alias_duplicate_hash_byte_identity_attempt_parameter_model_code_environment_or_ledger_mismatch_action"
        ]
        == "invalidate_adapter_attempt_and_publish_nothing"
    )


def test_issue_forecast_rng_contexts_are_distinct_by_role_and_replicate() -> None:
    seed = _load_yaml(PROTOCOL_PATH)["adapter_contract"]["issue_forecast_definition"][
        "seed_context_contract"
    ]
    role_snapshots = {
        "development": "fold_4",
        "formal_validation": "final_validation",
        "prospective": "final_validation",
    }
    parameter_sha_by_snapshot = {
        snapshot: hashlib.sha256(snapshot.encode()).hexdigest()
        for snapshot in set(role_snapshots.values())
    }
    context_sha = "1" * 64
    digests = {
        SeedContext(
            root_seed=seed["root_seed"],
            protocol_version=seed["protocol_version"],
            namespace=seed["namespace"],
            model_id=(f"etas/{role}/{snapshot}/{parameter_sha_by_snapshot[snapshot]}"),
            issue_id=f"stage4/{role}/2024-01-01/{context_sha}",
            replicate_index=replicate,
        ).digest()
        for role, snapshot in role_snapshots.items()
        for replicate in range(128)
    }
    assert len(digests) == 3 * 128


def test_new_outputs_do_not_overlap_historical_stage2_paths() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    new_paths = {protocol["fit_input_bundle"]["local_root"]}
    new_paths.update(
        value for value in protocol["outputs"].values() if isinstance(value, str) and "/" in value
    )
    historical_prefixes = {
        "data/processed/stage2R/local_support",
        "models/registry/background_local_support",
        "outputs/backtests/background_local_support",
        "outputs/experiments/background_local_support",
        "data/manifests/background_local_support_model_registry.json",
        "docs/background_local_support_report.md",
    }
    for new_path in new_paths:
        assert all(
            new_path != historical and not new_path.startswith(f"{historical}/")
            for historical in historical_prefixes
        )
    public_probe = "models/registry/background_etas_numerical_repair/probe.json"
    comparator_probe = "models/registry/background_etas_comparator/probe.json"
    local_probe = "data/processed/stage2R/etas_numerical_repair_adapter_payload/probe.json"
    adapter_ledger_probe = "data/manifests/etas_numerical_repair_adapter_attempt_ledger.json"
    source_ledger_probe = "data/manifests/etas_numerical_repair_source_access_ledger.json"
    public_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", public_probe],
        check=False,
    )
    local_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", local_probe],
        check=False,
    )
    comparator_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", comparator_probe],
        check=False,
    )
    adapter_ledger_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", adapter_ledger_probe],
        check=False,
    )
    source_ledger_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", source_ledger_probe],
        check=False,
    )
    assert public_check.returncode == 1
    assert comparator_check.returncode == 1
    assert local_check.returncode == 0
    assert adapter_ledger_check.returncode == 0
    assert source_ledger_check.returncode == 0
    outputs = protocol["outputs"]
    assert outputs["public_parameter_artifact_root_must_not_be_gitignored"]
    branch = outputs["parameter_artifact_branch_contract"]
    assert branch["evaluable"] == {
        "root_presence_in_qualification_result_tag": "required",
        "exact_files": ["parameter_snapshots.json", "parameter_set_manifest.json"],
    }
    assert branch["not_evaluable"]["root_presence_in_qualification_result_tag"] == "forbidden"
    assert branch["not_evaluable"]["exact_files"] == []
    assert branch["invalid_execution"]["public_qualification_result_allowed"] is False
    registry_schema = outputs["public_parameter_registry_schema"]
    assert registry_schema["any_unlisted_file_field_nested_field_or_alias_forbidden"]
    assert registry_schema["unknown_missing_duplicate_or_noncanonical_field_rejected"]
    assert (
        registry_schema[
            "event_identifier_coordinate_time_target_score_or_training_row_fields_allowed"
        ]
        is False
    )
    manifest_schema = outputs["public_qualification_manifest_schema"]
    cross_file = manifest_schema["cross_file_input_identity_invariants"]
    assert cross_file["every_mapping_key_order_exactly_snapshot_key_order_and_every_value_equal"]
    assert cross_file["any_missing_extra_or_mismatched_cross_file_identity_action"] == (
        "invalid_execution_without_public_result"
    )
    assert set(cross_file) >= {
        "public_source_access_receipt_canonical_sha256_equal_across",
        "fit_input_manifest_file_and_content_sha256_equal_across",
        "five_snapshot_scientific_fit_input_sha256_mapping_equal_across",
        "five_snapshot_parent_replay_scientific_fit_input_sha256_mapping_equal_across",
        "five_snapshot_parent_replay_membership_identity_sha256_mapping_equal_across",
        "start_manifest_file_and_vector_payload_sha256_equal_across",
        "source_and_reader_identity_equal_across",
        "global_source_ledger_identity_equal_across",
        "selected_source_acquisition_attempt_id_equal_across",
        "single_qualification_attempt_id_equal_across",
        "frozen_execution_identity_equal_across",
        "qualification_branch_values_equal_across",
    }
    start_pair = outputs["canonical_nested_schemas"][
        "start_manifest_file_and_vector_payload_sha256_pair"
    ]
    assert start_pair["fields_exact"] == ["file_sha256", "vector_payload_sha256"]
    seal_schema = outputs["public_qualification_seal_schema"]
    assert set(seal_schema["opening_nested_field_schema_refs"]) == {
        "repository_identity",
        "protocol_package_blob_oid_by_path",
        "repair_code_commit_and_remote_tag",
        "optimizer_runtime_baseline_blob_and_content_sha256",
        "all_public_input_binding_sha256",
        "frozen_local_restricted_input_identity_metadata",
    }
    assert (
        "python_runtime_file_sha256_and_size_map" in seal_schema["runtime_nested_field_schema_refs"]
    )
    assert (
        seal_schema["runtime_nested_field_schema_refs"]["synthetic_runtime_warmup_receipt"]
        == "outputs.canonical_nested_schemas.synthetic_runtime_warmup_receipt"
    )
    warmup_schema = outputs["canonical_nested_schemas"]["synthetic_runtime_warmup_receipt"]
    assert warmup_schema["full_receipt_must_equal_code_tag_baseline_byte_for_byte"]
    launcher_schema = outputs["canonical_nested_schemas"]["isolated_launcher_identity"]
    assert launcher_schema["sys_path_role_record_fields_exact"] == [
        "order_index",
        "role",
        "root_role",
        "canonical_root_relative_path",
        "entry_kind",
        "regular_file_sha256_or_null",
        "regular_file_size_or_null",
    ]
    assert launcher_schema["sys_path_root_role_values_exact"] == [
        "base_prefix",
        "venv_prefix",
        "workspace",
    ]
    assert launcher_schema["pth_record_fields_exact"] == [
        "root_role",
        "canonical_root_relative_path",
        "file_sha256",
    ]
    assert launcher_schema["pth_root_role_values_exact"] == ["venv_prefix"]
    startup_environment = launcher_schema["startup_environment_required_exact_values"]
    assert startup_environment["SETUPTOOLS_USE_DISTUTILS"] == "stdlib"
    assert {
        key: startup_environment[key]
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    } == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert {
        "COVERAGE_PROCESS_START",
        "COVERAGE_PROCESS_CONFIG",
        "COV_CORE_SOURCE",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
    } <= set(launcher_schema["startup_environment_required_absent_names_exact"])
    assert launcher_schema["meta_path_exact_ordered_finders"] == [
        "_frozen_importlib.BuiltinImporter",
        "_frozen_importlib.FrozenImporter",
        "_frozen_importlib_external.PathFinder",
    ]
    assert launcher_schema[
        "distutils_hack_coverage_pytest_cov_or_any_other_meta_path_finder_forbidden"
    ]
    assert (
        outputs["canonical_nested_schemas"]["windows_runtime_identity"]["platform_system_exact"]
        == "Windows"
    )
    assert registry_schema["evaluable_cross_file_identity_invariants"][
        "every_snapshot_record_hash_must_recompute_and_match_its_mapping_value"
    ]
    assert registry_schema["not_evaluable_cross_file_absence_invariants"] == {
        "qualification_manifest_parameter_set_and_snapshot_mapping_must_be_null": True,
        "parameter_artifact_root_and_both_registry_files_must_be_absent": True,
    }
    assert set(protocol["outputs"]["public_visual_forbidden_fields"]) == {
        "event_id",
        "physical_event_id",
        "event_coordinates",
        "coordinates",
        "longitude",
        "latitude",
        "projected_x",
        "projected_y",
        "target_rows",
        "assessment_event_id",
        "assessment_event_count",
        "assessment_scores",
        "model_score",
        "information_gain",
        "score_id",
    }
    assert protocol["outputs"]["any_public_visual_field_not_in_allowlist_forbidden"] is True
    report_contract = outputs["public_report_contract"]
    assert report_contract[
        "public_visual_forbidden_fields_apply_to_report_text_tables_links_alt_text_and_embedded_payloads"
    ]
    assert (
        report_contract[
            "absolute_path_hostname_username_environment_secret_or_local_ledger_detail_allowed"
        ]
        is False
    )
    assert set(protocol["outputs"]["public_visual_allowed_fields"]) == {
        "snapshot_id",
        "start_index",
        "numerical_status",
        "objective",
        "gradient_infinity_norm",
        "iterations",
        "function_evaluations",
        "parameter_name",
        "parameter_value",
        "gate_name",
        "gate_status",
        "failure_code",
    }
    assert protocol["outputs"][
        "allowlist_applies_to_embedded_json_tooltips_dom_attributes_downloads_and_accessibility_text"
    ]
    assert protocol["outputs"]["interactive_external_network_requests_allowed"] is False
    assert protocol["outputs"]["interactive_may_embed_only_allowlisted_static_payload"] is True
    domains = protocol["outputs"]["public_visual_value_domains"]
    assert set(domains["parameter_name"]) == {
        "background_rate_per_day",
        "productivity_k",
        "alpha",
        "c_days",
        "p",
        "branching_ratio",
    }


def test_parameter_snapshot_derivation_and_crosswalk_is_strict_and_complete() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    registry = protocol["outputs"]["public_parameter_registry_schema"]
    crosswalk = registry["parameter_snapshot_derivation_and_crosswalk"]

    assert (
        "etas_numerical_qualification_evidence_sha256"
        not in registry["parameter_snapshots_json_top_level_fields"]
    )
    assert (
        "etas_numerical_qualification_evidence_sha256"
        not in registry["parameter_set_manifest_json_fields"]
    )
    registry_invariants = registry["evaluable_cross_file_identity_invariants"]
    assert "qualification_evidence_sha256_equal_across" not in registry_invariants
    assert registry_invariants[
        "parameter_registry_files_may_not_contain_qualification_closing_seal_qualification_evidence_or_manifest_content_sha256"
    ]

    assert crosswalk["applicability"] == (
        "evaluable_branch_only_after_complete_protocol_valid_five_snapshot_qualification"
    )
    assert crosswalk["source_records_exact"] == [
        "complete_scientific_fit_input_payload",
        "fit_etas_call_opening_receipt",
        "fit_etas_call_closing_receipt",
        "selected_diagnostic_row",
        "snapshot_gate_result",
    ]
    source_selection = crosswalk["source_selection"]
    assert source_selection[
        "closing_receipt_opening_receipt_sha256_must_resolve_to_the_same_selected_opening_receipt"
    ]
    assert source_selection["every_source_snapshot_id_must_equal_parameter_snapshot_snapshot_id"]
    assert source_selection["cross_snapshot_or_ambiguous_source_selection_forbidden"]

    field_sources = crosswalk["record_field_derivation_source_exact"]
    assert set(field_sources) == set(registry["parameter_snapshot_record_fields"])
    assert field_sources["scientific_fit_input_sha256"] == (
        "complete_scientific_fit_input_payload_canonical_sha256"
    )
    assert field_sources["model_spec"] == "complete_scientific_fit_input_payload.model_spec"
    assert field_sources["selected_start_index"] == (
        "snapshot_gate_result.selected_start_index_or_null"
    )
    assert field_sources["physical_parameters_hex"].endswith(
        "returned_fit_result_canonical_payload.best_parameters_hex_or_null"
    )
    assert field_sources["hessian_and_uncertainty"] == (
        "exact_projection_of_fit_etas_call_closing_receipt.returned_fit_result_canonical_payload"
    )

    scientific = crosswalk["scientific_input_and_model_spec_crosswalk"]
    assert scientific["scientific_fit_input_sha256_must_equal"] == [
        "canonical_complete_scientific_fit_input_payload_sha256",
        "fit_etas_call_opening_receipt.scientific_fit_input_sha256",
        "qualification_manifest.scientific_fit_input_sha256_by_snapshot_matching_value",
    ]
    assert scientific[
        "model_spec_must_equal_complete_scientific_fit_input_payload_model_spec_field_for_field_and_byte_for_byte"
    ]
    assert scientific["model_spec_fields_exact_ref"] == (
        "fit_input_bundle.scientific_fit_input_record_schemas.model_spec_fields_exact"
    )
    assert scientific[
        "fit_etas_opening_exact_fit_arguments_preimage_ETASModelSpec_payload_must_equal_parameter_snapshot_model_spec"
    ]

    selected = crosswalk["selected_start_and_diagnostic_crosswalk"]
    assert selected["snapshot_gate_result_requirements"] == {
        "numerical_status": "evaluable",
        "selected_start_index_or_null": "required",
        "ordered_failure_codes": "empty",
    }
    assert selected[
        "selected_start_index_must_equal_first_stability_eligible_row_under_frozen_objective_then_start_index_order"
    ]
    assert selected["selected_diagnostic_row_must_have"] == {
        "etas_start_scipy_converged": True,
        "terminal_transformed_hex_or_null": "required",
        "objective_hex_or_null": "required",
        "gradient_infinity_norm_hex_or_null": "required",
        "physical_parameters_hex_or_null": "required",
        "failure_code_or_null": None,
    }
    assert selected["selected_terminal_vector_must_equal_byte_for_byte"] == [
        "selected_diagnostic_row.terminal_transformed_hex_or_null",
        "matching_selected_returned_start_result.final_transformed_hex",
    ]
    assert selected["selected_terminal_transformed_sha256_formula"].endswith(
        "exact_ordered_five_selected_terminal_float64_hex_strings"
    )
    assert selected[
        "hessian_evaluation_point_sha256_must_equal_selected_terminal_transformed_sha256"
    ]

    parameter_values = crosswalk["parameter_value_crosswalk"]
    assert parameter_values["transformed_field_to_selected_terminal_index_exact"] == {
        "log_background_rate_per_day": 0,
        "log_productivity_k": 1,
        "log_alpha": 2,
        "log_c_days": 3,
        "log_p_minus_one": 4,
    }
    assert parameter_values["physical_parameters_hex_must_equal_byte_for_byte"] == [
        "fit_etas_call_closing_receipt.returned_fit_result_canonical_payload."
        "best_parameters_hex_or_null",
        "selected_diagnostic_row.physical_parameters_hex_or_null",
        "frozen_repaired_ETASParameterBounds.from_transformed_of_selected_terminal_vector",
    ]
    assert parameter_values[
        "endpoint_aware_decode_may_not_be_reimplemented_clipped_toleranced_or_recomputed_by_plain_exp"
    ]

    uncertainty = crosswalk["hessian_and_uncertainty_crosswalk"]
    assert uncertainty["returned_fit_requirements"] == {
        "stability_stable": True,
        "hessian_success": True,
        "uncertainty_or_null": "required",
    }
    assert uncertainty["field_mapping_exact"] == {
        "observed_hessian_transformed_hex": (
            "returned_fit_result.stability.hessian.matrix_hex_or_null"
        ),
        "minimum_eigenvalue_hex": (
            "returned_fit_result.stability.hessian.minimum_eigenvalue_hex_or_null"
        ),
        "condition_number_hex": (
            "returned_fit_result.stability.hessian.condition_number_hex_or_null"
        ),
        "transformed_covariance_hex": (
            "returned_fit_result.uncertainty_or_null.transformed_covariance_hex"
        ),
        "physical_covariance_hex": (
            "returned_fit_result.uncertainty_or_null.physical_covariance_hex"
        ),
        "confidence_level_hex": ("returned_fit_result.uncertainty_or_null.confidence_level_hex"),
        "parameter_delta_estimates": (
            "returned_fit_result.uncertainty_or_null.parameter_estimates"
        ),
        "branching_ratio_delta_estimate": (
            "returned_fit_result.uncertainty_or_null.branching_ratio"
        ),
    }
    assert uncertainty[
        "branching_ratio_delta_estimate_estimate_hex_must_equal_snapshot_gate_result_branching_ratio_hex_or_null"
    ]
    assert uncertainty["hessian_and_uncertainty_sha256_must_recompute_from_complete_exact_object"]

    masses = crosswalk["aki_b_beta_and_bin_mass_crosswalk"]
    assert masses["beta_hex_must_equal_byte_for_byte"] == [
        "complete_scientific_fit_input_payload.beta_hex",
        "complete_scientific_fit_input_payload.model_spec.beta_hex",
        "parameter_snapshot.model_spec.beta_hex",
    ]
    assert masses["magnitude_bin_definition_and_formula_ref"] == "adapter_contract.magnitude_bins"
    assert masses[
        "beta_hex_must_equal_python_float64_hex_of_float_aki_b_times_math_log_10_under_frozen_build_etas_model_spec"
    ]
    assert masses[
        "M5_6_mass_hex_must_be_recomputed_from_frozen_beta_mc_mmax_lower_5_upper_6_formula"
    ]
    assert masses[
        "M6_plus_mass_hex_must_be_recomputed_from_frozen_beta_mc_mmax_lower_6_upper_mmax_formula"
    ]
    assert masses["payload_sha256_must_recompute_from_complete_exact_object_without_payload_sha256"]

    closure = crosswalk["gate_and_record_closure"]
    assert closure[
        "snapshot_gate_result_sha256_must_recompute_from_complete_same_snapshot_gate_result"
    ]
    assert closure["snapshot_gate_result_sha256_must_equal_fit_attempt_snapshot_bound_gate_sha256"]
    assert closure[
        "all_parameter_snapshot_record_fields_except_own_sha256_must_be_derived_exactly_once_by_record_field_derivation_source_exact"
    ]
    assert (
        closure["any_missing_extra_mismatch_noncanonical_value_or_failed_recomputation_action"]
        == "invalid_execution_without_parameter_artifact_or_public_qualification_result"
    )


def test_qualification_gate_and_public_result_crosswalks_are_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    invocation = protocol["optimizer_invocation_receipt_protocol"]
    qualification = protocol["qualification"]
    outputs = protocol["outputs"]
    manifest = outputs["public_qualification_manifest_schema"]

    assert (
        "three_grid_gate_evidence_sha256_or_null"
        in invocation["fit_attempt_snapshot_payload_fields_exact"]
    )
    grid = qualification["three_grid_gate_evidence_protocol"]
    assert grid["exact_existing_evaluator"] == (
        "seismoflux.background.pipeline_etas._grid_gate_evidence"
    )
    assert grid["grid_size_order_km_exact"] == [50.0, 25.0, 12.5]
    assert grid[
        "evidence_present_iff_selected_start_index_is_nonnull_and_all_upstream_stability_gates_required_before_grid_evaluation_pass"
    ]
    failure_ordering = qualification["snapshot_failure_code_order_and_deduplication"]
    assert failure_ordering["append_each_failed_gate_once_in_exact_order"]
    skip = failure_ordering["prerequisite_skip_and_failure_code_semantics"]
    truth = failure_ordering["gate_dependency_and_status_truth_table"]
    assert truth["ordered_gate_names_exact"] == [
        "minimum_converged_starts",
        "converged_gradient",
        "best_three_objective_spread",
        "best_three_parameter_spread",
        "hessian_minimum_eigenvalue",
        "hessian_condition_number",
        "branching_ratio",
        "grid_25_to_12_5_expected_count_difference",
        "grid_25_to_12_5_density_l1",
    ]
    assert truth["gate_status_values_exact"] == [
        "passed",
        "failed",
        "not_run_upstream_gate",
    ]
    expected_gate_prerequisites_and_failure_codes = {
        "minimum_converged_starts": ([], "insufficient_converged_starts"),
        "converged_gradient": (
            ["minimum_converged_starts_passed"],
            "converged_gradient_threshold_exceeded",
        ),
        "best_three_objective_spread": (
            ["minimum_converged_starts_passed", "converged_gradient_passed"],
            "best_three_objective_spread_exceeded",
        ),
        "best_three_parameter_spread": (
            ["minimum_converged_starts_passed", "converged_gradient_passed"],
            "best_three_parameter_spread_exceeded",
        ),
        "hessian_minimum_eigenvalue": (
            ["both_best_three_spread_siblings_passed"],
            "hessian_invalid_or_minimum_eigenvalue_failed",
        ),
        "hessian_condition_number": (
            ["hessian_minimum_eigenvalue_passed"],
            "hessian_condition_number_exceeded",
        ),
        "branching_ratio": (
            ["hessian_condition_number_passed", "selected_start_nonnull"],
            "branching_ratio_gate_failed",
        ),
        "grid_25_to_12_5_expected_count_difference": (
            ["branching_ratio_passed"],
            "grid_25_to_12_5_expected_count_difference_exceeded",
        ),
        "grid_25_to_12_5_density_l1": (
            ["branching_ratio_passed"],
            "grid_25_to_12_5_density_l1_exceeded",
        ),
    }
    assert list(expected_gate_prerequisites_and_failure_codes) == truth["ordered_gate_names_exact"]
    for gate_name, (
        prerequisites,
        failure_code,
    ) in expected_gate_prerequisites_and_failure_codes.items():
        assert truth[gate_name]["prerequisites"] == prerequisites
        assert truth[gate_name]["failure_code"] == failure_code
    assert truth["best_three_objective_spread"]["evaluation_group"] == (
        "best_three_spread_siblings"
    )
    assert truth["best_three_parameter_spread"]["evaluation_group"] == (
        "best_three_spread_siblings"
    )
    assert truth["grid_25_to_12_5_expected_count_difference"]["evaluation_group"] == (
        "three_grid_metric_siblings"
    )
    assert truth["grid_25_to_12_5_density_l1"]["evaluation_group"] == ("three_grid_metric_siblings")
    assert truth["sibling_group_rule"] == (
        "once_all_shared_prerequisites_pass_both_sibling_metrics_are_evaluated_and_each_"
        "status_and_failure_code_is_derived_independently"
    )
    assert truth["downstream_rule"] == (
        "any_failed_or_not_run_prerequisite_makes_each_dependent_gate_not_run_upstream_"
        "gate_with_null_gate_metric_and_no_failure_code"
    )
    assert truth["hessian_failure_rule"] == (
        "invalid_absent_nonfinite_or_below_minimum_fails_only_hessian_minimum_"
        "eigenvalue_and_skips_condition_and_all_downstream_gates"
    )
    assert truth["ordered_gate_status_records_must_contain_every_ordered_gate_once_in_exact_order"]
    assert truth[
        "ordered_failure_codes_must_equal_exact_order_projection_of_gate_status_failed_to_declared_failure_code"
    ]
    assert skip == {
        "fewer_than_minimum_stability_eligible_starts": (
            "count_failed_and_every_downstream_gate_not_run_upstream_gate_with_null_"
            "snapshot_gate_metric"
        ),
        "best_three_metrics_in_closing_stability_payload_when_count_or_gradient_gate_failed": (
            "retained_only_in_closing_diagnostic_payload_but_snapshot_gate_metrics_are_null_"
            "and_both_spread_gates_are_not_run_upstream_gate"
        ),
        "hessian_metrics_absent_because_any_count_gradient_or_spread_prerequisite_failed": (
            "both_hessian_snapshot_gate_metrics_null_and_hessian_minimum_and_condition_"
            "gates_not_run_upstream_gate"
        ),
        "hessian_attempted_but_nonfinite_invalid_or_below_minimum": (
            "append_hessian_invalid_or_minimum_eigenvalue_failed_once"
        ),
        "hessian_finite_positive_but_condition_number_above_maximum": (
            "append_hessian_condition_number_exceeded_once"
        ),
        (
            "branching_ratio_absent_because_any_hessian_prerequisite_failed_or_selected_"
            "start_is_null"
        ): ("skipped_null_and_do_not_append_branching_ratio_gate_failed"),
        (
            "three_grid_evidence_absent_because_any_required_upstream_stability_or_"
            "branching_gate_failed"
        ): (
            "both_grid_gates_skipped_all_grid_metrics_null_and_do_not_append_either_grid_"
            "failure_code"
        ),
        "numerical_status_when_any_downstream_gate_is_skipped": (
            "not_evaluable_from_the_actual_recorded_upstream_failure_codes_only"
        ),
        "skipped_gate_status_value_exact": "not_run_upstream_gate",
    }

    gate_fields = manifest["snapshot_gate_result_fields_exact"]
    assert {
        "hessian_minimum_eigenvalue_nonfinite_kind_or_null",
        "hessian_condition_number_nonfinite_kind_or_null",
        "three_grid_gate_evidence_sha256_or_null",
        "ordered_gate_status_records",
    } <= set(gate_fields)
    gate_crosswalk = manifest["snapshot_gate_result_derivation_and_crosswalk"]
    assert gate_crosswalk["selected_start_index_exact_source"] == (
        "qualification.selected_start_rule_applied_to_same_snapshot_five_diagnostic_rows"
    )
    assert gate_crosswalk["three_grid_gate_evidence_sha256_exact_source"] == (
        "same_snapshot_frozen_three_grid_gate_evidence_payload"
    )
    assert gate_crosswalk["prerequisite_skip_and_failure_code_semantics_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "prerequisite_skip_and_failure_code_semantics"
    )
    assert gate_crosswalk["ordered_gate_status_records_exact_source_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "gate_dependency_and_status_truth_table"
    )
    assert manifest["snapshot_ordered_gate_status_records_schema_ref"] == (
        "outputs.canonical_nested_schemas.qualification_ordered_gate_status_records"
    )
    gate_status_schema = outputs["canonical_nested_schemas"][
        "qualification_ordered_gate_status_records"
    ]
    assert gate_status_schema["exact_length"] == 9
    assert gate_status_schema["canonical_container_type_exact"] == "ordered_list_not_mapping"
    assert gate_status_schema["item_fields_exact"] == ["gate_name", "gate_status"]
    assert gate_status_schema["gate_name_order_exact_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "gate_dependency_and_status_truth_table.ordered_gate_names_exact"
    )
    assert gate_status_schema["gate_status_values_exact_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "gate_dependency_and_status_truth_table.gate_status_values_exact"
    )
    assert gate_status_schema["every_gate_name_must_appear_exactly_once_in_declared_order"]
    assert (
        gate_crosswalk["any_source_hash_value_state_order_status_or_cross_snapshot_mismatch_action"]
        == "invalid_execution_without_qualification_result"
    )

    excluded = manifest[
        "staged_preclosing_projection_fields_must_equal_top_level_fields_minus_exact_three_identity_fields"
    ]
    assert excluded == [
        "qualification_closing_seal_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "qualification_manifest_content_sha256",
    ]
    assert manifest["staged_preclosing_projection_fields_exact"] == [
        field for field in manifest["top_level_fields_exact"] if field not in excluded
    ]
    staged = outputs["canonical_nested_schemas"]["staged_public_payload_identity"]
    assert (
        "qualification_manifest_preclosing_projection_sha256_excluding_closing_seal_"
        "qualification_evidence_and_manifest_content" in staged["fields_exact"]
    )
    assert staged[
        "every_preclosing_staged_file_byte_payload_must_exclude_direct_and_transitive_references_to"
    ] == [
        "qualification_closing_seal_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "qualification_manifest_content_sha256",
    ]
    closing = protocol["qualification_execution_seal"]["closing_seal"]
    assert closing["qualification_publication_hash_DAG_order_exact"].endswith(
        "etas_numerical_qualification_evidence_sha256_then_qualification_manifest_content_sha256"
    )

    public_crosswalk = manifest["qualification_public_result_crosswalk"]
    negative_scalar = public_crosswalk["closing_seal_scalar_siblings_must_equal_manifest"][
        "etas_numerical_negative_evidence_sha256_or_null"
    ]
    assert negative_scalar == {
        "evaluable": {"closing_value": None, "manifest_nested_value": None},
        "not_evaluable": (
            "closing_value_equals_qualification_manifest.etas_numerical_negative_"
            "evidence_or_null.etas_numerical_negative_evidence_sha256"
        ),
    }
    assert public_crosswalk[
        "closing_seal_ordered_25_optimizer_sha256_must_equal_manifest_receipt_own_sha256_projection"
    ]
    assert public_crosswalk[
        "closing_seal_ordered_25_diagnostic_sha256_must_equal_manifest_row_own_sha256_projection"
    ]
    assert public_crosswalk["not_evaluable_negative_evidence_crosswalk"][
        "ordered_failure_codes"
    ] == (
        "stable_first_occurrence_deduplication_of_gate_rows_ordered_by_snapshot_then_"
        "frozen_gate_order"
    )
    assert (
        public_crosswalk[
            "any_missing_extra_duplicate_hash_value_order_branch_or_cross_snapshot_mismatch_action"
        ]
        == "invalid_execution_without_public_materialization_or_result_tag"
    )
    input_crosswalk = manifest["cross_file_input_identity_invariants"]
    assert (
        "local_restricted_input_acceptance_receipt.observed_sha256_by_input."
        "stage2_catalog_source" in input_crosswalk["source_and_reader_identity_equal_across"]
    )
    assert (
        "qualification_evidence.public_source_access_receipt.local_ledger_content_sha256"
        in input_crosswalk["global_source_ledger_identity_equal_across"]
    )
    assert input_crosswalk[
        "qualification_input_seal_public_source_access_receipt_sha256_must_recompute_from_the_exact_public_receipt_in_this_crosswalk"
    ]


def test_adapter_local_restricted_payloads_have_strict_byte_and_source_closure() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    outputs = protocol["outputs"]
    schemas = outputs["canonical_nested_schemas"]
    local_protocol = outputs["adapter_local_restricted_artifact_protocol"]
    public_schema = outputs["adapter_public_artifact_schemas"]
    invariants = public_schema["adapter_public_result_cross_file_invariants"]
    local_equalities = invariants["local_restricted_payload_equalities"]

    assert local_protocol["exact_payload_count"] == 10
    assert local_protocol["exact_payload_count_per_kind"] == 5
    staging_root = protocol["adapter_artifact_execution_seal"]["attempt_protocol"][
        "staging_directory_template"
    ]
    assert local_protocol["root_path_template"] == f"{staging_root}/local_restricted"
    assert local_protocol[
        "root_path_template_must_equal_adapter_execution_staging_directory_template_plus_local_restricted"
    ]
    assert local_protocol["payload_manifest_schema_ref"] == (
        "outputs.canonical_nested_schemas.adapter_local_restricted_payload_manifest"
    )
    assert local_protocol["immigrant_density_artifact_schema_ref"] == (
        "outputs.canonical_nested_schemas.adapter_immigrant_density_artifact_payload"
    )
    assert local_protocol["propagation_domain_artifact_schema_ref"] == (
        "outputs.canonical_nested_schemas.adapter_propagation_domain_artifact_payload"
    )
    assert local_protocol[
        "payload_files_and_manifest_must_be_write_temp_flush_fsync_atomic_replace_then_reopened_and_byte_verified_before_public_artifact_construction"
    ]

    density = schemas["adapter_immigrant_density_artifact_payload"]
    assert density["schema_version_exact"] == 1
    assert "adapter_artifact_runtime_seal_sha256" in density["fields_exact"]
    assert density["identity_fields_exact"] == [
        field for field in density["fields_exact"] if field != "immigrant_density_artifact_sha256"
    ]
    assert density["immigrant_kde_payload_fields_exact_ref"] == (
        "fit_input_bundle.scientific_fit_input_record_schemas.immigrant_kde_payload_fields_exact"
    )
    assert density[
        "immigrant_kde_payload_must_equal_same_snapshot_positive_qualification_complete_scientific_fit_input_immigrant_kde_payload_field_for_field_and_byte_for_byte"
    ]

    propagation = schemas["adapter_propagation_domain_artifact_payload"]
    assert propagation["schema_version_exact"] == 1
    assert "adapter_artifact_runtime_seal_sha256" in propagation["fields_exact"]
    assert propagation["identity_fields_exact"] == [
        field
        for field in propagation["fields_exact"]
        if field != "propagation_domain_artifact_sha256"
    ]
    assert propagation["parent_selection_source_commit_exact_ref"] == (
        "adapter_contract.issue_forecast_definition.branching_process_domain_and_marks."
        "parent_0_2_1_propagation_membership.source_commit"
    )
    assert propagation["parent_selection_source_blob_git_oid_by_path_exact_ref"] == (
        "fit_input_bundle.parent_replay_membership_equivalence.frozen_source_blobs"
    )
    assert propagation["frozen_study_area_source_sha256_exact_source"] == (
        "local_restricted_input_acceptance_receipt.observed_sha256_by_input.study_area"
    )
    assert propagation["exact_buffer_distance_km"] == 300.0
    assert propagation["geometry_canonicalization_exact"] == (
        "GEOSNormalize_then_OGC_WKB_big_endian_2D_without_SRID_lowercase_hex"
    )

    local_manifest = schemas["adapter_local_restricted_payload_manifest"]
    assert local_manifest["schema_version_exact"] == 1
    assert "adapter_artifact_runtime_seal_sha256" in local_manifest["fields_exact"]
    assert local_manifest["identity_fields_exact"] == [
        field
        for field in local_manifest["fields_exact"]
        if field != "local_restricted_payload_content_sha256"
    ]
    assert local_manifest["snapshot_order_exact"] == list(SNAPSHOT_ORDER)
    assert local_manifest["exact_local_artifact_path_file_sha256_map_keys"] == (
        "exact_ten_paths_from_both_snapshot_record_maps"
    )
    assert local_manifest[
        "every_record_file_sha256_must_equal_sha256_of_exact_fsynced_file_bytes_at_repository_relative_path_and_same_path_map_value"
    ]
    assert local_manifest[
        "every_artifact_file_must_parse_as_its_declared_strict_schema_and_reserialize_byte_identically"
    ]

    assert local_equalities["local_restricted_payload_content_sha256_equal_across"] == [
        "recomputed_local_payload_manifest_content_sha256",
        "local_payload_manifest.local_restricted_payload_content_sha256",
        "artifact_manifest.local_restricted_payload_content_sha256",
    ]
    assert local_equalities["payload_manifest_runtime_seal_sha256_equal_across"] == [
        "local_payload_manifest.adapter_artifact_runtime_seal_sha256",
        "opening_seal.adapter_artifact_runtime_seal.adapter_artifact_runtime_seal_sha256",
        "artifact_manifest.adapter_artifact_runtime_seal_sha256",
        "global_receipt.protocol_code_environment_hashes.adapter_artifact_runtime_seal_sha256",
    ]
    for key in (
        "immigrant_density_artifact_sha256_by_snapshot_crosswalk",
        "propagation_domain_artifact_sha256_by_snapshot_crosswalk",
    ):
        crosswalk = local_equalities[key]
        assert crosswalk["all_three_maps_must_be_key_and_value_identical_in_exact_snapshot_order"]
    assert local_equalities[
        "every_local_record_file_sha256_must_equal_sha256_of_exact_reopened_payload_file_bytes_and_same_exact_path_map_value"
    ]
    assert local_equalities[
        "every_reopened_payload_file_must_strict_parse_recompute_own_content_sha_and_reserialize_to_identical_bytes"
    ]
    assert invariants["acyclic_construction_and_reference_order_exact"][2:5] == [
        "ten_local_restricted_artifact_payloads",
        "local_restricted_payload_manifest",
        "artifact_manifest",
    ]


def test_protocol_document_states_the_same_stop_boundary() -> None:
    document = PROTOCOL_DOCUMENT_PATH.read_text(encoding="utf-8")
    for required in (
        "v0.2.2-background-etas-repair-protocol-r2",
        "Stage 4 formal target consumer 调用 0",
        "恰好有 25 行",
        "不得复用旧 `run_local_support_etas_pipeline`",
        "FrozenETASComparatorReceipt",
        "ETASIssueForecastInput/Field",
        "ETASIssueForecastQueryNodes/Measure",
        "SeedContext",
        "4≤M<5",
        "实际 Python shared library",
        "完整规范化的原始 `OptimizeResult` payload",
        "`ordered_optimizer_call_observation_log`",
        "四个公开 seal (opening、runtime、input、closing)",
        "只能包含规定章节中的聚合数值诊断和公开协议/工件 SHA",
        "当前阶段 4 R2 继续保持目标读取前硬停",
        "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/{attempt_id}/staged_public",
        "pre-closing identity 只覆盖 7 个 common 文件",
        "同一 attempt 仅重试 byte-exact public materialization",
    ):
        assert required in document


def test_acceptance_and_restart_handoff_share_the_frozen_boundaries() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    acceptance = PROTOCOL_ACCEPTANCE_PATH.read_text(encoding="utf-8")
    handoff = RESTART_HANDOFF_PATH.read_text(encoding="utf-8")
    protocol_document = PROTOCOL_DOCUMENT_PATH.read_text(encoding="utf-8")
    fullwidth_colon = "\uff1a"
    fullwidth_left_parenthesis = "\uff08"

    package_paths = protocol["qualification_execution_seal"]["opening_seal"][
        "protocol_package_paths"
    ]
    assert package_paths == [
        ".gitignore",
        "configs/background_etas_numerical_repair.yaml",
        "data/manifests/etas_numerical_repair_start_manifest.json",
        "docs/background_etas_numerical_repair_protocol.md",
        "docs/phase2_etas_numerical_repair_protocol_acceptance.md",
        "tests/unit/test_background_etas_numerical_repair_protocol.py",
    ]
    for package_path in package_paths:
        assert f"`{package_path}`" in acceptance
        assert f"`{package_path}`" in handoff

    for document in (acceptance, handoff):
        assert "v0.2.2-background-etas-repair-protocol" in document
        assert "v0.2.2-background-etas-repair-protocol-r1" in document
        assert "v0.2.2-background-etas-repair-protocol-r2" in document
        assert "codex/stage2-etas-numerical-repair" in document
        assert "dae6403" in document
        assert f"阶段 9 锁定测试{fullwidth_colon}未运行" in document
        assert "not_run_upstream_gate" in document
        assert "adapter runtime preflight" in document
        assert "artifact、global receipt、报告、静态 SVG、离线 HTML" in document
        assert (
            "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/"
            "{attempt_id}/staged_public"
        ) in document
        assert "staged→final" in document
        assert "post-closing" in document

    assert f"状态{fullwidth_colon}通过{fullwidth_left_parenthesis}本地协议工程验收" in acceptance
    assert "进行中" not in acceptance
    assert "之后补填" not in acceptance
    for final_evidence in (
        "1213 passed in 329.82s",
        "failures=0",
        "errors=0",
        "skipped=0",
        "406edffc53ad4f49a83638611d9aa376026882c89e67d87adcc5712da9dc2916",
        "Success: no issues found in 216 source files",
        "Runtime/Shapely 闭包与四链前像",
        "Adapter runtime、九门真值表与七文件 DAG",
        "六文件 protocol package、引用和三文档一致性",
    ):
        assert final_evidence in acceptance

    assert f"Stage 4 formal target consumer 调用{fullwidth_colon}`0`" in acceptance
    assert f"Stage 4 assessment 行物化{fullwidth_colon}`0`" in acceptance
    assert f"Stage 4 formal target consumer 调用{fullwidth_colon}0" in handoff
    assert f"assessment row 物化{fullwidth_colon}0" in handoff
    assert "## 5. 精确续接步骤" in handoff
    assert "## 6. 后续阶段计划" in handoff
    for future_stage in ("阶段 2R-A", "阶段 2R-B", "阶段 2R-C", "新阶段 4 修订"):
        assert future_stage in handoff

    assert "固定双坐标 `Point` 构造只归 synthetic warmup" in acceptance
    assert "类型化 closure cell 和 native ufunc" in acceptance
    assert "canonical_binding_path + callable_layer" in handoff
    assert (
        "ledger intent append+fsync → adapter runtime preflight+staged opening" in protocol_document
    )
    assert "七个非-publication 文件公开物化并 reopen 验证 → ledger completed" in protocol_document
