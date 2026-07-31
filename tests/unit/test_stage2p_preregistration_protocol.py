from __future__ import annotations

import ast
import copy
import hashlib
import json
import struct
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

from seismoflux.anomaly_increment.preregistration import (
    verify_content_sha256,
    with_content_sha256,
)
from seismoflux.data.common import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "prospective_recent_seismicity.yaml"
RESEARCH_CONFIG_PATH = ROOT / "configs" / "research_protocol.yaml"
SCHEMA_PATH = ROOT / "data" / "contracts" / "stage2p_prospective_records.json"
SHA256_PATTERN = "^[0-9a-f]{64}$"
RFC3161_EXCLUDED_TOP_LEVEL_FIELDS = (
    "timestamp_attempt_evidence",
    "remote_timestamp",
    "content_sha256",
)
MISSED_AUDIT_EXCLUDED_TOP_LEVEL_FIELDS = (
    "missed_audit_timestamp_attempt_evidence",
    "missed_audit_remote_timestamp",
    "content_sha256",
)
TRUTH_RETRY_OFFSETS_HOURS = (0, 6, 24, 72, 168)
ENDPOINT_IDS = (
    "P1_minus_P0_macro_information_gain",
    "P1_minus_PP_macro_information_gain",
    "P1_minus_P0_macro_recall_gain",
    "P1_minus_PP_macro_recall_gain",
)
EXPECTED_SCHEMA_HASHES = {
    "FormalPreferredFieldRow": "6ce072a6511237c69f86c6134e7499e1726a556b5f74cbfa29ff541a2164bae9",
    "FormalScientificTargetRow": "1d7aaef2ab13000af39c532f88b39838eac637cc9207c85c917cf46b8304494e",
    "EffectRowsManifest": "8b5251145e5207f63e4f20f4104768f3c1bfd08f458cdfce913c9757e7816e32",
    "ResultBundleManifest": "3895e6bf772333deac0069470346a60a5a889560956c1ad686105f3720d9ec5e",
}
TABLE_ARTIFACT_IDENTITY_FIELDS = {
    "artifact_id",
    "byte_count",
    "row_count",
    "file_sha256",
    "content_sha256",
    "schema_sha256",
    "sort_order_sha256",
    "table_role",
    "serialization_profile",
    "row_schema_ref",
    "sort_profile",
    "local_restricted",
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
        result[cast(str, key)] = loader.construct_object(  # type: ignore[no-untyped-call]
            value_node,
            deep=deep,
        )
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_config() -> dict[str, Any]:
    value = yaml.load(CONFIG_PATH.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _load_research_config() -> dict[str, Any]:
    value = yaml.load(
        RESEARCH_CONFIG_PATH.read_text(encoding="utf-8"),
        Loader=UniqueKeyLoader,
    )
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _load_schema() -> dict[str, Any]:
    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schema_definition_sha256(name: str) -> str:
    definition = _load_schema()["$defs"][name]
    return hashlib.sha256(canonical_json_bytes(definition)).hexdigest()


def _canonical_file_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _float64_bits_hex(value: float) -> str:
    return struct.pack(">d", value).hex()


def _def_validator(name: str) -> Draft202012Validator:
    schema = _load_schema()
    return Draft202012Validator(
        {
            "$schema": schema["$schema"],
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{name}",
        },
        format_checker=FormatChecker(),
    )


def _assert_valid_def(name: str, instance: object) -> None:
    errors = sorted(
        _def_validator(name).iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    assert not errors, "\n".join(error.message for error in errors)


def _assert_invalid_def(name: str, instance: object) -> None:
    with pytest.raises(ValidationError):
        _def_validator(name).validate(instance)


def _record_preimage_sha256(record: Mapping[str, object]) -> str:
    core = {
        key: value for key, value in record.items() if key not in RFC3161_EXCLUDED_TOP_LEVEL_FIELDS
    }
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _missed_audit_preimage_sha256(record: Mapping[str, object]) -> str:
    core = {
        key: value
        for key, value in record.items()
        if key not in MISSED_AUDIT_EXCLUDED_TOP_LEVEL_FIELDS
    }
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _tsa_attempt(
    *,
    index: int = 0,
    selected: bool = True,
    core_frozen_at_utc: str = "2026-09-09T15:55:00Z",
    preimage_sha256: str | None = None,
) -> dict[str, object]:
    authority = "http://timestamp.digicert.com" if index == 0 else "http://timestamp.sectigo.com"
    frozen = _utc(core_frozen_at_utc)
    request_started = frozen + timedelta(seconds=5 + index * 20)
    if selected:
        generated = request_started + timedelta(seconds=5)
        received = generated + timedelta(seconds=5)
        return {
            "attempt_index": index,
            "authority_url": authority,
            "request_content_type": "application/timestamp-query",
            "response_content_type": "application/timestamp-reply",
            "request_byte_count": 96,
            "request_sha256": _sha256(f"tsa-request-{index}"),
            "attempt_preimage_sha256": preimage_sha256 or _sha256("tsa-preimage"),
            "nonce_sha256": _sha256(f"tsa-nonce-{index}"),
            "certReq": True,
            "request_started_at_utc": request_started.isoformat().replace("+00:00", "Z"),
            "attempt_completed_at_utc": received.isoformat().replace("+00:00", "Z"),
            "response_received_at_utc": received.isoformat().replace("+00:00", "Z"),
            "http_status": 200,
            "response_byte_count": 512,
            "response_sha256": _sha256(f"tsa-response-{index}"),
            "authority_identity_sha256": _sha256(f"tsa-authority-{index}"),
            "trust_chain_sha256": _sha256(f"tsa-chain-{index}"),
            "genTime_utc": generated.isoformat().replace("+00:00", "Z"),
            "offline_trust_path_valid": True,
            "genTime_before_deadline": True,
            "outcome": "selected_valid",
        }
    completed = request_started + timedelta(seconds=10)
    return {
        "attempt_index": index,
        "authority_url": authority,
        "request_content_type": "application/timestamp-query",
        "response_content_type": None,
        "request_byte_count": 96,
        "request_sha256": _sha256(f"tsa-request-{index}"),
        "attempt_preimage_sha256": preimage_sha256 or _sha256("tsa-preimage"),
        "nonce_sha256": _sha256(f"tsa-nonce-{index}"),
        "certReq": True,
        "request_started_at_utc": request_started.isoformat().replace("+00:00", "Z"),
        "attempt_completed_at_utc": completed.isoformat().replace("+00:00", "Z"),
        "response_received_at_utc": None,
        "http_status": None,
        "response_byte_count": 0,
        "response_sha256": None,
        "authority_identity_sha256": None,
        "trust_chain_sha256": None,
        "genTime_utc": None,
        "offline_trust_path_valid": False,
        "genTime_before_deadline": False,
        "outcome": "network_failure",
    }


def _timestamp_proof(
    *,
    subject_type: str,
    preimage_sha256: str,
    selected_attempt: Mapping[str, object],
    deadline_utc: str = "2026-09-09T16:00:00Z",
    proof_field_name: str = "remote_timestamp",
    preimage_profile: str = "stage2p_rfc3161_core_v1",
) -> dict[str, object]:
    return {
        "subject_type": subject_type,
        "proof_field_name": proof_field_name,
        "final_hash_field_name": "content_sha256",
        "preimage_profile": preimage_profile,
        "mechanism": "RFC3161",
        "digest_algorithm": "SHA256",
        "ordered_authorities": [
            "http://timestamp.digicert.com",
            "http://timestamp.sectigo.com",
        ],
        "request_content_type": "application/timestamp-query",
        "response_content_type": "application/timestamp-reply",
        "nonce_required": True,
        "certReq": True,
        "preimage_sha256": preimage_sha256,
        "deadline_utc": deadline_utc,
        "selected_attempt_index": selected_attempt["attempt_index"],
        "selected_response_sha256": selected_attempt["response_sha256"],
        "message_imprint_matches_preimage": True,
        "nonce_matches_request": True,
        "tsa_policy_oid": "2.16.840.1.114412.7.1",
        "timestamping_EKU_verified": True,
        "selection_rule": "first_offline_trust_path_valid_response_with_genTime_before_deadline",
        "verification_code_sha256": _sha256("rfc3161-verifier"),
        "verified": True,
    }


def _seal_timestamped_record(
    payload: Mapping[str, object],
    *,
    subject_type: str,
    selected_attempt_index: int = 0,
    deadline_utc: str = "2026-09-09T16:00:00Z",
) -> dict[str, object]:
    record = copy.deepcopy(dict(payload))
    core_frozen_at_utc = cast(
        str,
        record.get(
            "record_core_frozen_at_utc",
            record.get("frozen_at_utc", "2026-09-09T15:55:00Z"),
        ),
    )
    record["record_core_frozen_at_utc"] = core_frozen_at_utc
    record["timestamp_deadline_utc"] = deadline_utc
    preimage_sha256 = _record_preimage_sha256(record)
    selected = _tsa_attempt(
        index=selected_attempt_index,
        selected=True,
        core_frozen_at_utc=core_frozen_at_utc,
        preimage_sha256=preimage_sha256,
    )
    if selected_attempt_index == 0:
        attempts = [selected]
    else:
        attempts = [
            _tsa_attempt(
                index=0,
                selected=False,
                core_frozen_at_utc=core_frozen_at_utc,
                preimage_sha256=preimage_sha256,
            ),
            selected,
        ]
    record["timestamp_attempt_evidence"] = attempts
    record["remote_timestamp"] = _timestamp_proof(
        subject_type=subject_type,
        preimage_sha256=preimage_sha256,
        selected_attempt=selected,
        deadline_utc=deadline_utc,
    )
    return with_content_sha256(record)


def _seal_missed_audit_record(
    payload: Mapping[str, object],
    *,
    deadline_utc: str,
) -> dict[str, object]:
    record = copy.deepcopy(dict(payload))
    record["missed_audit_timestamp_attempt_evidence"] = []
    record["missed_audit_remote_timestamp"] = None
    preimage_sha256 = _missed_audit_preimage_sha256(record)
    selected = _tsa_attempt(
        index=0,
        selected=True,
        core_frozen_at_utc=cast(str, record["record_core_frozen_at_utc"]),
        preimage_sha256=preimage_sha256,
    )
    record["missed_audit_timestamp_attempt_evidence"] = [selected]
    record["missed_audit_remote_timestamp"] = _timestamp_proof(
        subject_type="IssueInputSnapshotRecord",
        preimage_sha256=preimage_sha256,
        selected_attempt=selected,
        deadline_utc=deadline_utc,
        proof_field_name="missed_audit_remote_timestamp",
        preimage_profile="stage2p_missed_audit_rfc3161_core_v1",
    )
    return with_content_sha256(record)


def _timestamp_semantic_errors(record: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    proof = record.get("remote_timestamp")
    attempts = record.get("timestamp_attempt_evidence")
    if not isinstance(proof, Mapping) or not isinstance(attempts, Sequence):
        return ["missing_top_level_timestamp_evidence"]
    if proof.get("preimage_sha256") != _record_preimage_sha256(record):
        errors.append("preimage_sha256_mismatch")
    if proof.get("deadline_utc") != record.get("timestamp_deadline_utc"):
        errors.append("proof_deadline_must_equal_core_deadline")
    core_frozen = record.get("record_core_frozen_at_utc")
    deadline = record.get("timestamp_deadline_utc")
    previous_completed: datetime | None = None
    for _index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            errors.append("tsa_attempt_not_mapping")
            continue
        request_started = attempt.get("request_started_at_utc")
        completed = attempt.get("attempt_completed_at_utc")
        if attempt.get("attempt_preimage_sha256") != proof.get("preimage_sha256"):
            errors.append("tsa_attempt_preimage_sha256_mismatch")
        if not all(
            isinstance(value, str)
            for value in (core_frozen, request_started, completed, deadline)
        ) or not (
            _utc(cast(str, core_frozen))
            <= _utc(cast(str, request_started))
            <= _utc(cast(str, completed))
            <= _utc(cast(str, deadline))
        ):
            errors.append("tsa_attempt_time_order_invalid")
        if (
            previous_completed is not None
            and isinstance(request_started, str)
            and _utc(request_started) < previous_completed
        ):
            errors.append("tsa_authority_attempts_overlap_or_reverse")
        if isinstance(completed, str):
            previous_completed = _utc(completed)
        response_received = attempt.get("response_received_at_utc")
        if attempt.get("outcome") == "network_failure":
            if response_received is not None:
                errors.append("network_failure_response_received_must_be_null")
        elif response_received != completed:
            errors.append("response_received_must_equal_attempt_completed")
    selected_index = proof.get("selected_attempt_index")
    if not isinstance(selected_index, int) or selected_index >= len(attempts):
        errors.append("selected_attempt_index_out_of_range")
    else:
        selected = attempts[selected_index]
        if not isinstance(selected, Mapping):
            errors.append("selected_attempt_not_mapping")
        else:
            if selected.get("response_sha256") != proof.get("selected_response_sha256"):
                errors.append("selected_response_sha256_mismatch")
            gen_time = selected.get("genTime_utc")
            request_started = selected.get("request_started_at_utc")
            response_received = selected.get("response_received_at_utc")
            completed = selected.get("attempt_completed_at_utc")
            core_frozen = record.get("record_core_frozen_at_utc")
            deadline = proof.get("deadline_utc")
            if (
                not isinstance(gen_time, str)
                or not isinstance(deadline, str)
                or _utc(gen_time) >= _utc(deadline)
            ):
                errors.append("selected_genTime_not_before_deadline")
            if not all(
                isinstance(value, str)
                for value in (
                    core_frozen,
                    request_started,
                    gen_time,
                    response_received,
                    completed,
                    deadline,
                )
            ) or not (
                _utc(cast(str, core_frozen))
                <= _utc(cast(str, request_started))
                <= _utc(cast(str, gen_time))
                <= _utc(cast(str, response_received))
                == _utc(cast(str, completed))
                <= _utc(cast(str, deadline))
            ):
                errors.append("selected_tsa_time_order_invalid")
    if (
        record.get("record_type") == "IssueInputSnapshotRecord"
        and isinstance(record.get("issue_time_utc"), str)
        and proof.get("deadline_utc")
        != (
            _utc(cast(str, record["issue_time_utc"])) - timedelta(minutes=5)
        )
        .isoformat()
        .replace("+00:00", "Z")
    ):
        errors.append("issue_candidate_timestamp_deadline_must_equal_T_minus_5_minutes")
    if (
        record.get("record_type") == "EvaluationFreezeRecord"
        and isinstance(record.get("record_core_frozen_at_utc"), str)
        and isinstance(record.get("timestamp_deadline_utc"), str)
        and _utc(cast(str, record["timestamp_deadline_utc"]))
        != _utc(cast(str, record["record_core_frozen_at_utc"])) + timedelta(minutes=5)
    ):
        errors.append("evaluation_timestamp_deadline_must_equal_core_freeze_plus_5_minutes")
    if not verify_content_sha256(record):
        errors.append("content_sha256_mismatch")

    def find_nested_proof(value: object, *, depth: int) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if depth > 0 and key in {"remote_timestamp", "timestamp_attempt_evidence"}:
                    errors.append("nested_timestamp_proof_forbidden")
                find_nested_proof(item, depth=depth + 1)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for item in value:
                find_nested_proof(item, depth=depth + 1)

    for key, value in record.items():
        if key not in {"remote_timestamp", "timestamp_attempt_evidence"}:
            find_nested_proof(value, depth=1)
    return errors


def _forecast_identity(*, actual_area_km2: float = 599_750.0) -> dict[str, object]:
    return {
        "density_sha256": _sha256("density"),
        "grid_identity_sha256": _sha256("grid"),
        "complete_grid_cell_count": 1000,
        "complete_grid_score_rows_sha256": _sha256("grid-score-rows"),
        "ranked_grid_rows_sha256": _sha256("ranked-grid-rows"),
        "ranking_rule": "mass_per_exact_clipped_area_desc_then_row_column_cell_id_ascending",
        "alarm_area_budget_km2": 600000,
        "alarm_prefix_cell_count": 96,
        "alarm_prefix_rows_sha256": _sha256("alarm-prefix-rows"),
        "selected_alarm_cell_ids_sha256": _sha256("alarm-cell-ids"),
        "alarm_mask_sha256": _sha256("alarm-mask"),
        "actual_alarm_area_km2": actual_area_km2,
        "remaining_budget_km2": 600000.0 - actual_area_km2,
        "alarm_prefix_termination_reason": "next_complete_cell_would_exceed_budget",
        "next_unselected_rank_position_1_based": 97,
        "next_unselected_cell_id": "r0096c0000",
        "next_unselected_complete_cell_area_km2": 300.0,
        "next_unselected_ranked_row_sha256": _sha256("next-ranked-row"),
        "complete_cells_only": True,
        "prefix_maximal_under_budget": True,
        "local_artifact_bundle_sha256": _sha256("local-forecast-bundle"),
    }


def _query_request(*, start: str, end: str, label: str) -> dict[str, object]:
    return {
        "method": "GET",
        "endpoint": "https://earthquake.usgs.gov/fdsnws/event/1/query",
        "starttime_utc": start,
        "endtime_utc": end,
        "bbox": [73.446961, 20.22909, 135.08583, 53.557926],
        "minmagnitude": 3.9,
        "eventtype": "earthquake",
        "format": "geojson",
        "orderby": "time-asc",
        "limit": 20000,
        "includeallorigins": False,
        "includeallmagnitudes": False,
        "includedeleted": False,
        "includesuperseded": "omit_not_applicable_without_eventid",
        "reviewstatus": "omit_use_default_all",
        "offset": 1,
        "jsonerror": True,
        "nodata": 204,
        "catalog": "omit",
        "contributor": "omit",
        "canonical_parameter_order": [
            "starttime",
            "endtime",
            "minlongitude",
            "minlatitude",
            "maxlongitude",
            "maxlatitude",
            "minmagnitude",
            "eventtype",
            "format",
            "orderby",
            "limit",
            "includeallorigins",
            "includeallmagnitudes",
            "includedeleted",
            "offset",
            "jsonerror",
            "nodata",
        ],
        "canonical_url_utf8_sha256": _sha256(f"query-url-{label}"),
    }


def _count_request(*, start: str, end: str, label: str) -> dict[str, object]:
    return {
        "method": "GET",
        "endpoint": "https://earthquake.usgs.gov/fdsnws/event/1/count",
        "starttime_utc": start,
        "endtime_utc": end,
        "bbox": [73.446961, 20.22909, 135.08583, 53.557926],
        "minmagnitude": 3.9,
        "eventtype": "earthquake",
        "format": "geojson",
        "includedeleted": False,
        "includesuperseded": "omit_not_applicable_without_eventid",
        "reviewstatus": "omit_use_default_all",
        "jsonerror": True,
        "nodata": 204,
        "catalog": "omit",
        "contributor": "omit",
        "excluded_query_only_parameters": [
            "orderby",
            "limit",
            "offset",
            "includeallorigins",
            "includeallmagnitudes",
        ],
        "canonical_parameter_order": [
            "starttime",
            "endtime",
            "minlongitude",
            "minlatitude",
            "maxlongitude",
            "maxlatitude",
            "minmagnitude",
            "eventtype",
            "format",
            "includedeleted",
            "jsonerror",
            "nodata",
        ],
        "canonical_url_utf8_sha256": _sha256(f"count-url-{label}"),
    }


def _count_preflight(
    *,
    start: str,
    end: str,
    label: str,
    parsed_count: int | None,
) -> dict[str, object]:
    if parsed_count is None:
        return {
            "request": _count_request(start=start, end=end, label=label),
            "fetch_started_at_utc": None,
            "fetch_completed_at_utc": None,
            "http_status": None,
            "response_content_type": None,
            "response_body_byte_count": None,
            "response_headers_sha256": None,
            "raw_response_sha256": None,
            "geojson_parse_verified": None,
            "parsed_count": None,
            "outcome": "network_failure",
            "failure_code": "network_failure",
        }
    return {
        "request": _count_request(start=start, end=end, label=label),
        "fetch_started_at_utc": "2026-09-09T15:44:01Z",
        "fetch_completed_at_utc": "2026-09-09T15:44:30Z",
        "http_status": 200,
        "response_content_type": "application/json",
        "response_body_byte_count": 48,
        "response_headers_sha256": _sha256(f"count-headers-{label}"),
        "raw_response_sha256": _sha256(f"count-raw-{label}"),
        "geojson_parse_verified": True,
        "parsed_count": parsed_count,
        "outcome": "succeeded",
        "failure_code": None,
    }


def _source_acquisition(
    *,
    role: str,
    http_status: int = 204,
    query_count: int = 0,
    feature_count: int = 0,
    response_body_byte_count: int = 0,
) -> dict[str, object]:
    if role == "prospective_increment":
        query_start = "2026-07-09T04:25:56Z"
        query_end = "2026-09-09T15:45:00Z"
        start_filter = "origin_time_utc_strictly_greater_than_cutover"
        end_filter = "origin_time_utc_lte_query_end"
    elif role == "formal_freeze_full_cohort":
        query_start = "2026-09-09T16:00:00Z"
        query_end = "2028-09-01T16:00:00Z"
        start_filter = "origin_time_utc_strictly_greater_than_formal_cohort_start"
        end_filter = "origin_time_utc_lte_formal_cohort_end"
    else:
        query_start = "2026-09-09T16:00:00Z"
        query_end = "2026-09-16T16:00:00Z"
        start_filter = "origin_time_utc_strictly_greater_than_target_start"
        end_filter = "origin_time_utc_lte_target_end"
    captured_response_headers = {
        "date": "Wed, 09 Sep 2026 15:46:00 GMT",
        "etag": None,
        "last_modified": None,
        "content_type": None if http_status == 204 else "application/json",
        "content_length": str(response_body_byte_count),
    }
    raw_response_sha256 = (
        hashlib.sha256(b"").hexdigest()
        if response_body_byte_count == 0
        else _sha256(f"raw-file-{role}")
    )
    return {
        "source_id": "usgs_anss_comcat_fdsn_event_api_v1",
        "source_role": role,
        "institution": "United_States_Geological_Survey",
        "endpoint_identity_sha256": _sha256("endpoint"),
        "api_version": "1",
        "license_identity_sha256": _sha256("license"),
        "query_start_utc": query_start,
        "query_end_utc": query_end,
        "query_request": _query_request(
            start=query_start,
            end=query_end,
            label=f"acquisition-{role}",
        ),
        "count_preflight": _count_preflight(
            start=query_start,
            end=query_end,
            label=f"acquisition-{role}",
            parsed_count=query_count,
        ),
        "local_starttime_filter": start_filter,
        "local_endtime_filter": end_filter,
        "request_identity_sha256": _sha256(f"request-{role}"),
        "query_count_preflight_request_sha256": _sha256(f"count-request-{role}"),
        "query_count": query_count,
        "feature_count": feature_count,
        "query_limit": 20000,
        "includeallorigins": False,
        "includeallmagnitudes": False,
        "includedeleted": False,
        "includesuperseded": "omit_not_applicable_without_eventid",
        "reviewstatus": "omit_use_default_all",
        "offset": 1,
        "jsonerror": True,
        "nodata": 204,
        "catalog": "omit",
        "contributor": "omit",
        "captured_response_headers": captured_response_headers,
        "response_headers_sha256": hashlib.sha256(
            canonical_json_bytes(captured_response_headers)
        ).hexdigest(),
        "response_content_type": None if http_status == 204 else "application/json",
        "response_body_byte_count": response_body_byte_count,
        "geojson_parse_verified": http_status == 200,
        "fetch_started_at_utc": "2026-09-09T15:45:01Z",
        "fetch_completed_at_utc": "2026-09-09T15:46:00Z",
        "http_status": http_status,
        "raw_response": {
            "artifact_id": _sha256(f"raw-artifact-{role}"),
            "byte_count": response_body_byte_count,
            "file_sha256": raw_response_sha256,
            "local_restricted": True,
        },
        "first_seen_policy_sha256": _sha256("first-seen-policy"),
        "user_agent_identity_sha256": _sha256("user-agent"),
        "TLS_peer_certificate_chain_sha256": _sha256("tls-chain"),
        "redirect_count": 0,
    }


def _truth_attempt(*, index: int, succeeded: bool) -> dict[str, object]:
    offset = TRUTH_RETRY_OFFSETS_HOURS[index]
    maturity_due = _utc("2026-10-16T16:00:00Z")
    scheduled = maturity_due + timedelta(hours=offset)
    if succeeded:
        return {
            "query_role": "temporally_independent_mature_truth",
            "attempt_index": index,
            "retry_offset_hours": offset,
            "scheduled_at_utc": scheduled.isoformat().replace("+00:00", "Z"),
            "fetch_started_at_utc": (scheduled + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "fetch_completed_at_utc": (scheduled + timedelta(seconds=30))
            .isoformat()
            .replace("+00:00", "Z"),
            "issue_time_utc": "2026-09-09T16:00:00Z",
            "horizon_days": 7,
            "target_start_exclusive_utc": "2026-09-09T16:00:00Z",
            "target_end_inclusive_utc": "2026-09-16T16:00:00Z",
            "maturity_due_at_utc": "2026-10-16T16:00:00Z",
            "query_request": _query_request(
                start="2026-09-09T16:00:00Z",
                end="2026-09-16T16:00:00Z",
                label=f"truth-{index}",
            ),
            "count_preflight": _count_preflight(
                start="2026-09-09T16:00:00Z",
                end="2026-09-16T16:00:00Z",
                label=f"truth-{index}",
                parsed_count=0,
            ),
            "request_identity_sha256": _sha256(f"truth-request-{index}"),
            "query_count_preflight_request_sha256": _sha256(f"truth-count-{index}"),
            "query_count": 0,
            "feature_count": 0,
            "query_limit": 20000,
            "includeallorigins": False,
            "includeallmagnitudes": False,
            "includedeleted": False,
            "includesuperseded": "omit_not_applicable_without_eventid",
            "reviewstatus": "omit_use_default_all",
            "offset": 1,
            "jsonerror": True,
            "nodata": 204,
            "catalog": "omit",
            "contributor": "omit",
            "local_starttime_filter": "origin_time_utc_strictly_greater_than_target_start",
            "local_endtime_filter": "origin_time_utc_lte_target_end",
            "http_status": 204,
            "response_content_type": None,
            "response_body_byte_count": 0,
            "geojson_parse_verified": False,
            "response_headers_sha256": _sha256(f"truth-headers-{index}"),
            "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
            "exchange_outcome": "response_received",
            "outcome": "succeeded",
            "selected_as_truth_snapshot": True,
            "failure_code": None,
        }
    return {
        "query_role": "temporally_independent_mature_truth",
        "attempt_index": index,
        "retry_offset_hours": offset,
        "scheduled_at_utc": scheduled.isoformat().replace("+00:00", "Z"),
        "fetch_started_at_utc": None,
        "fetch_completed_at_utc": None,
        "issue_time_utc": "2026-09-09T16:00:00Z",
        "horizon_days": 7,
        "target_start_exclusive_utc": "2026-09-09T16:00:00Z",
        "target_end_inclusive_utc": "2026-09-16T16:00:00Z",
        "maturity_due_at_utc": "2026-10-16T16:00:00Z",
        "query_request": _query_request(
            start="2026-09-09T16:00:00Z",
            end="2026-09-16T16:00:00Z",
            label=f"truth-{index}",
        ),
        "count_preflight": _count_preflight(
            start="2026-09-09T16:00:00Z",
            end="2026-09-16T16:00:00Z",
            label=f"truth-{index}",
            parsed_count=None,
        ),
        "request_identity_sha256": _sha256(f"truth-request-{index}"),
        "query_count_preflight_request_sha256": _sha256(f"truth-count-{index}"),
        "query_count": None,
        "feature_count": None,
        "query_limit": 20000,
        "includeallorigins": False,
        "includeallmagnitudes": False,
        "includedeleted": False,
        "includesuperseded": "omit_not_applicable_without_eventid",
        "reviewstatus": "omit_use_default_all",
        "offset": 1,
        "jsonerror": True,
        "nodata": 204,
        "catalog": "omit",
        "contributor": "omit",
        "local_starttime_filter": "origin_time_utc_strictly_greater_than_target_start",
        "local_endtime_filter": "origin_time_utc_lte_target_end",
        "http_status": None,
        "response_content_type": None,
        "response_body_byte_count": None,
        "geojson_parse_verified": None,
        "response_headers_sha256": None,
        "raw_response_sha256": None,
        "exchange_outcome": "not_attempted_count_preflight_failed_or_limit_reached",
        "outcome": "network_failure",
        "selected_as_truth_snapshot": False,
        "failure_code": "network_failure",
    }


def _truth_attempt_semantic_errors(
    attempts: Sequence[Mapping[str, object]],
    *,
    expect_success: bool,
) -> list[str]:
    errors: list[str] = []
    succeeded_indices: list[int] = []
    maturity_due = _utc("2026-10-16T16:00:00Z")
    for position, attempt in enumerate(attempts):
        if attempt.get("query_role") != "temporally_independent_mature_truth":
            errors.append("wrong_truth_query_role")
        if attempt.get("attempt_index") != position:
            errors.append("attempt_index_not_contiguous")
        expected_offset = TRUTH_RETRY_OFFSETS_HOURS[position]
        if attempt.get("retry_offset_hours") != expected_offset:
            errors.append("retry_offset_wrong_order")
        expected_scheduled = maturity_due + timedelta(hours=expected_offset)
        if _utc(cast(str, attempt["scheduled_at_utc"])) != expected_scheduled:
            errors.append("scheduled_at_not_due_plus_retry_offset")
        if attempt.get("outcome") == "succeeded":
            succeeded_indices.append(position)
    if expect_success:
        if succeeded_indices != [len(attempts) - 1]:
            errors.append("first_success_must_be_last_attempt")
    elif succeeded_indices or len(attempts) != len(TRUTH_RETRY_OFFSETS_HOURS):
        errors.append("unavailable_requires_all_five_failures")
    return errors


def _failed_issue_fetch_attempt(*, query_end_utc: str) -> dict[str, object]:
    return {
        "query_role": "issue_input",
        "attempt_index": 0,
        "scheduled_at_utc": query_end_utc,
        "fetch_started_at_utc": None,
        "fetch_completed_at_utc": None,
        "query_start_utc": "2026-07-09T04:25:56Z",
        "query_end_utc": query_end_utc,
        "query_request": _query_request(
            start="2026-07-09T04:25:56Z",
            end=query_end_utc,
            label="failed-issue",
        ),
        "count_preflight": _count_preflight(
            start="2026-07-09T04:25:56Z",
            end=query_end_utc,
            label="failed-issue",
            parsed_count=None,
        ),
        "request_identity_sha256": _sha256("issue-request"),
        "query_count_preflight_request_sha256": _sha256("issue-count-request"),
        "query_count": None,
        "feature_count": None,
        "query_limit": 20000,
        "includeallorigins": False,
        "includeallmagnitudes": False,
        "includedeleted": False,
        "includesuperseded": "omit_not_applicable_without_eventid",
        "reviewstatus": "omit_use_default_all",
        "offset": 1,
        "jsonerror": True,
        "nodata": 204,
        "catalog": "omit",
        "contributor": "omit",
        "local_starttime_filter": "origin_time_utc_strictly_greater_than_cutover",
        "local_endtime_filter": "origin_time_utc_lte_query_end",
        "http_status": None,
        "response_content_type": None,
        "response_body_byte_count": None,
        "geojson_parse_verified": None,
        "response_headers_sha256": None,
        "raw_response_sha256": None,
        "exchange_outcome": "not_attempted_count_preflight_failed_or_limit_reached",
        "outcome": "network_failure",
        "failure_code": "network_failure",
    }


def _successful_issue_fetch_attempt(*, query_end_utc: str) -> dict[str, object]:
    attempt = _failed_issue_fetch_attempt(query_end_utc=query_end_utc)
    attempt.update(
        {
            "fetch_started_at_utc": "2026-09-09T15:45:01Z",
            "fetch_completed_at_utc": "2026-09-09T15:46:00Z",
            "count_preflight": _count_preflight(
                start="2026-07-09T04:25:56Z",
                end=query_end_utc,
                label="successful-issue",
                parsed_count=0,
            ),
            "query_count": 0,
            "feature_count": 0,
            "http_status": 204,
            "response_content_type": None,
            "response_body_byte_count": 0,
            "geojson_parse_verified": False,
            "response_headers_sha256": _sha256("successful-issue-headers"),
            "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
            "exchange_outcome": "response_received",
            "outcome": "succeeded",
            "failure_code": None,
        }
    )
    return attempt


def _missed_issue_record(
    *,
    scheduled_sequence: int = 1,
    local_time: str = "2026-09-10T00:00:00+08:00",
    previous_sha256: str | None = None,
) -> dict[str, object]:
    issue_local = datetime.fromisoformat(local_time)
    issue_utc = issue_local.astimezone(UTC)
    query_end = issue_utc - timedelta(minutes=15)
    issue_utc_text = issue_utc.isoformat().replace("+00:00", "Z")
    query_end_text = query_end.isoformat().replace("+00:00", "Z")
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "IssueInputSnapshotRecord",
        "experiment_id": "stage2p-prospective-recent-seismicity-v1",
        "protocol_version": "0.2.4",
        "stage_id": "Stage2P-1A",
        "issue_id": f"stage2p-issue-{issue_local:%Y%m%dT000000}+0800",
        "scheduled_issue_sequence": scheduled_sequence,
        "on_time_issue_sequence": None,
        "issue_time_local": local_time,
        "issue_time_utc": issue_utc_text,
        "query_end_utc": query_end_text,
        "record_core_frozen_at_utc": (query_end + timedelta(seconds=30))
        .isoformat()
        .replace("+00:00", "Z"),
        "timestamp_deadline_utc": (issue_utc - timedelta(minutes=5))
        .isoformat()
        .replace("+00:00", "Z"),
        "cohort_definition_sha256": _sha256("cohort"),
        "protocol_config_sha256": _sha256("protocol"),
        "code_commit": "a" * 40,
        "code_tag": "v0.2.4-prospective-seismicity-code",
        "code_manifest_sha256": _sha256("code-manifest"),
        "previous_issue_record_sha256": previous_sha256,
        "status": "missed_issue",
        "attempt_evidence": [_failed_issue_fetch_attempt(query_end_utc=query_end_text)],
        "timestamp_attempt_evidence": [],
        "remote_timestamp": None,
        "missed_audit_timestamp_deadline_utc": issue_utc_text,
        "missed_audit_timestamp_attempt_evidence": [],
        "missed_audit_remote_timestamp": None,
        "source_snapshot": None,
        "prediction_seal": None,
        "failure_code": "network_failure",
        "failed_candidate_on_time_core": None,
        "prediction_generated": False,
        "prediction_installed": False,
        "backfill": False,
    }
    return _seal_missed_audit_record(payload, deadline_utc=issue_utc_text)


def _timestamp_failure_missed_issue_record() -> dict[str, object]:
    record = _missed_issue_record()
    record.pop("content_sha256")
    issue_time_utc = cast(str, record["issue_time_utc"])
    query_end_utc = cast(str, record["query_end_utc"])
    candidate_core_frozen_at_utc = "2026-09-09T15:54:00Z"
    candidate_deadline_utc = "2026-09-09T15:55:00Z"
    candidate_sha256 = _sha256("failed-candidate-on-time-core")
    record.update(
        {
            "record_core_frozen_at_utc": "2026-09-09T15:56:00Z",
            "timestamp_deadline_utc": candidate_deadline_utc,
            "attempt_evidence": [
                _successful_issue_fetch_attempt(query_end_utc=query_end_utc)
            ],
            "timestamp_attempt_evidence": [
                _tsa_attempt(
                    index=0,
                    selected=False,
                    core_frozen_at_utc=candidate_core_frozen_at_utc,
                    preimage_sha256=candidate_sha256,
                ),
                _tsa_attempt(
                    index=1,
                    selected=False,
                    core_frozen_at_utc=candidate_core_frozen_at_utc,
                    preimage_sha256=candidate_sha256,
                ),
            ],
            "remote_timestamp": None,
            "missed_audit_timestamp_deadline_utc": issue_time_utc,
            "missed_audit_timestamp_attempt_evidence": [],
            "missed_audit_remote_timestamp": None,
            "failure_code": "timestamp_failure",
            "failed_candidate_on_time_core": {
                "profile": "stage2p_failed_candidate_on_time_core_v1",
                "candidate_core_sha256": candidate_sha256,
                "candidate_core_artifact_sha256": candidate_sha256,
                "issue_id": record["issue_id"],
                "source_snapshot_sha256": _sha256("failed-candidate-source-snapshot"),
                "prediction_seal_sha256": _sha256("failed-candidate-prediction-seal"),
                "core_frozen_at_utc": candidate_core_frozen_at_utc,
                "timestamp_deadline_utc": candidate_deadline_utc,
                "local_restricted": True,
            },
            "prediction_generated": True,
            "prediction_installed": False,
        }
    )
    return _seal_missed_audit_record(record, deadline_utc=issue_time_utc)


def _issue_semantic_errors(
    record: Mapping[str, object],
    *,
    previous_record: Mapping[str, object] | None,
    previous_on_time_sequence: int,
    first_authorized_utc: str = "2026-09-09T16:00:00Z",
) -> list[str]:
    errors: list[str] = []
    local = datetime.fromisoformat(cast(str, record["issue_time_local"]))
    utc = _utc(cast(str, record["issue_time_utc"]))
    query_end = _utc(cast(str, record["query_end_utc"]))
    if local.weekday() != 3 or local.hour != 0 or local.minute != 0 or local.second != 0:
        errors.append("issue_not_Thursday_midnight_Asia_Shanghai")
    if local.utcoffset() != timedelta(hours=8) or local.astimezone(UTC) != utc:
        errors.append("local_and_utc_issue_time_disagree")
    if query_end != utc - timedelta(minutes=15):
        errors.append("Q_must_equal_T_minus_15_minutes")
    if _utc(cast(str, record["timestamp_deadline_utc"])) != utc - timedelta(minutes=5):
        errors.append("candidate_timestamp_deadline_must_equal_T_minus_5_minutes")
    if utc < _utc(first_authorized_utc):
        errors.append("issue_before_first_issue_not_before")
    expected_issue_id = f"stage2p-issue-{local:%Y%m%dT000000}+0800"
    if record.get("issue_id") != expected_issue_id:
        errors.append("issue_id_not_derived_from_local_T")
    if previous_record is None:
        if utc != _utc(first_authorized_utc):
            errors.append("first_issue_not_first_authorized_rule_time")
    else:
        previous_utc = _utc(cast(str, previous_record["issue_time_utc"]))
        if utc != previous_utc + timedelta(days=7):
            errors.append("issue_T_not_exactly_previous_T_plus_7_days")
    expected_scheduled = (
        1 if previous_record is None else cast(int, previous_record["scheduled_issue_sequence"]) + 1
    )
    if record.get("scheduled_issue_sequence") != expected_scheduled:
        errors.append("scheduled_issue_sequence_not_contiguous")
    expected_previous_hash = None if previous_record is None else previous_record["content_sha256"]
    if record.get("previous_issue_record_sha256") != expected_previous_hash:
        errors.append("previous_issue_hash_chain_broken")
    status = record.get("status")
    expected_on_time = previous_on_time_sequence + 1 if status == "on_time" else None
    if record.get("on_time_issue_sequence") != expected_on_time:
        errors.append("on_time_issue_sequence_not_contiguous")
    if status == "missed_issue":
        audit_deadline = record.get("missed_audit_timestamp_deadline_utc")
        audit_proof = record.get("missed_audit_remote_timestamp")
        audit_attempts = record.get("missed_audit_timestamp_attempt_evidence")
        if audit_deadline != record.get("issue_time_utc"):
            errors.append("missed_audit_deadline_must_equal_T")
        if (
            not isinstance(record.get("record_core_frozen_at_utc"), str)
            or _utc(cast(str, record["record_core_frozen_at_utc"]))
            > utc - timedelta(minutes=4)
        ):
            errors.append("missed_audit_core_must_freeze_by_T_minus_4_minutes")
        if not isinstance(audit_proof, Mapping) or not isinstance(audit_attempts, Sequence):
            errors.append("missing_missed_audit_timestamp")
        else:
            audit_preimage = _missed_audit_preimage_sha256(record)
            if audit_proof.get("preimage_sha256") != audit_preimage:
                errors.append("missed_audit_preimage_sha256_mismatch")
            if audit_proof.get("deadline_utc") != record.get("issue_time_utc"):
                errors.append("missed_audit_proof_deadline_must_equal_T")
            for attempt in audit_attempts:
                if (
                    not isinstance(attempt, Mapping)
                    or attempt.get("attempt_preimage_sha256") != audit_preimage
                ):
                    errors.append("missed_audit_attempt_preimage_mismatch")
            selected_index = audit_proof.get("selected_attempt_index")
            if isinstance(selected_index, int) and selected_index < len(audit_attempts):
                selected = audit_attempts[selected_index]
                if (
                    not isinstance(selected, Mapping)
                    or not isinstance(selected.get("genTime_utc"), str)
                    or _utc(cast(str, selected["genTime_utc"])) >= utc
                ):
                    errors.append("missed_audit_genTime_not_before_T")
        candidate = record.get("failed_candidate_on_time_core")
        candidate_attempts = record.get("timestamp_attempt_evidence")
        if record.get("failure_code") == "timestamp_failure":
            if not isinstance(candidate, Mapping):
                errors.append("timestamp_failure_missing_failed_candidate")
            else:
                candidate_sha256 = candidate.get("candidate_core_sha256")
                if candidate.get("candidate_core_artifact_sha256") != candidate_sha256:
                    errors.append("failed_candidate_core_artifact_sha256_mismatch")
                if candidate.get("issue_id") != record.get("issue_id"):
                    errors.append("failed_candidate_issue_id_mismatch")
                if candidate.get("timestamp_deadline_utc") != record.get(
                    "timestamp_deadline_utc"
                ):
                    errors.append("failed_candidate_deadline_mismatch")
                if not isinstance(candidate_attempts, Sequence) or not candidate_attempts:
                    errors.append("timestamp_failure_missing_candidate_tsa_attempts")
                else:
                    for attempt in candidate_attempts:
                        if (
                            not isinstance(attempt, Mapping)
                            or attempt.get("attempt_preimage_sha256")
                            != candidate_sha256
                        ):
                            errors.append("candidate_tsa_attempt_preimage_mismatch")
        elif candidate is not None:
            errors.append("pre_candidate_failure_must_not_claim_candidate")
    if not verify_content_sha256(record):
        errors.append("content_sha256_mismatch")
    return errors


def _horizon_evaluability(horizon_days: int, *, evaluable: bool) -> dict[str, object]:
    count = 8 if evaluable else 0
    return {
        "horizon_days": horizon_days,
        "complete_exposure_count": 8,
        "truth_available_complete_exposure_count": 8,
        "selected_truth_snapshot_unavailable_count": 0,
        "selected_exposure_manifest_sha256": _sha256(f"exposures-{horizon_days}"),
        "truth_availability_manifest_sha256": _sha256(f"truth-availability-{horizon_days}"),
        "supported_unique_M5_6_event_count": count,
        "full_study_area_unique_M5_6_event_count": count,
        "unique_target_cluster_count": 4 if evaluable else 0,
        "all_required_forecast_densities_finite_positive": True,
        "evaluable": evaluable,
        "unevaluable_reason": None if evaluable else "zero_supported_denominator",
    }


def _target_set_identity(
    *,
    label: str = "target",
    event_count: int = 24,
) -> dict[str, object]:
    target_rows = _table_artifact(
        f"{label}-rows",
        table_name="scientific_target_rows",
        row_count=event_count,
    )
    return {
        "unique_event_count": event_count,
        "ordered_event_ids_sha256": _sha256(f"{label}-ordered-target-event-ids"),
        "target_rows_content_sha256": target_rows["content_sha256"],
        "region_membership_sha256": _sha256(f"{label}-target-regions"),
        "target_rows": target_rows,
    }


def _empty_target_set_identity(label: str) -> dict[str, object]:
    return _target_set_identity(label=label, event_count=0)


def _empty_horizon_evaluability(horizon_days: int) -> dict[str, object]:
    return {
        "horizon_days": horizon_days,
        "complete_exposure_count": 0,
        "truth_available_complete_exposure_count": 0,
        "selected_truth_snapshot_unavailable_count": 0,
        "selected_exposure_manifest_sha256": _sha256(
            f"empty-exposures-{horizon_days}"
        ),
        "truth_availability_manifest_sha256": _sha256(
            f"empty-truth-availability-{horizon_days}"
        ),
        "supported_unique_M5_6_event_count": 0,
        "full_study_area_unique_M5_6_event_count": 0,
        "unique_target_cluster_count": 0,
        "all_required_forecast_densities_finite_positive": True,
        "evaluable": False,
        "unevaluable_reason": "no_complete_exposure",
    }


def _unavailable_horizon_evaluability(
    horizon_days: int,
    *,
    scheduled_issue_cap_terminal: bool,
) -> dict[str, object]:
    return {
        "horizon_days": horizon_days,
        "complete_exposure_count": 8,
        "truth_available_complete_exposure_count": 8,
        "selected_truth_snapshot_unavailable_count": 0,
        "selected_exposure_manifest_sha256": _sha256(
            f"unavailable-exposures-{horizon_days}"
        ),
        "truth_availability_manifest_sha256": _sha256(
            f"unavailable-truth-availability-{horizon_days}"
        ),
        "supported_unique_M5_6_event_count": None,
        "full_study_area_unique_M5_6_event_count": None,
        "unique_target_cluster_count": None,
        "all_required_forecast_densities_finite_positive": None,
        "evaluable": False,
        "unevaluable_reason": (
            "scheduled_issue_cap_terminal_before_formal_freeze"
            if scheduled_issue_cap_terminal
            else "formal_freeze_unavailable"
        ),
    }


def _table_artifact(
    label: str,
    *,
    table_name: str,
    row_count: int,
    table_role: str | None = None,
) -> dict[str, object]:
    config = _load_config()
    artifact_registry = config["artifact_profile_registry"]
    if table_name in artifact_registry["tables"]:
        table_spec = artifact_registry["tables"][table_name]
        serialization_profile = artifact_registry["canonical_jsonl"]["profile"]
    else:
        formal_registry = config["target_cohort"]["formal_freeze_source_manifest"][
            "derived_table_registry"
        ]
        table_spec = formal_registry["tables"][table_name]
        serialization_profile = formal_registry["serialization"]["profile"]

    resolved_table_role = table_role or table_spec.get("table_role")
    if resolved_table_role is None:
        raise ValueError(f"table_role is required for {table_name}")
    allowed_table_roles = table_spec.get("table_roles")
    if allowed_table_roles is not None and resolved_table_role not in allowed_table_roles:
        raise ValueError(f"unsupported table_role for {table_name}")

    file_sha256 = (
        hashlib.sha256(b"").hexdigest()
        if row_count == 0
        else _sha256(f"{label}-file")
    )
    content_sha256 = (
        hashlib.sha256(canonical_json_bytes([])).hexdigest()
        if row_count == 0
        else _sha256(f"{label}-content")
    )
    return {
        "artifact_id": file_sha256,
        "byte_count": 0 if row_count == 0 else row_count * 128,
        "row_count": row_count,
        "file_sha256": file_sha256,
        "content_sha256": content_sha256,
        "schema_sha256": table_spec["expected_row_schema_sha256"],
        "sort_order_sha256": table_spec["expected_sort_order_sha256"],
        "table_role": resolved_table_role,
        "serialization_profile": serialization_profile,
        "row_schema_ref": table_spec["row_schema_ref"],
        "sort_profile": table_spec["sort_profile"],
        "local_restricted": True,
    }


def _event_set_identity(
    component_role: str,
    *,
    event_count: int,
) -> dict[str, object]:
    event_rows = _table_artifact(
        f"{component_role}-event-rows",
        table_name="event_set_rows",
        row_count=event_count,
        table_role=f"{component_role}_event_rows",
    )
    return {
        "component_role": component_role,
        "event_count": event_count,
        "ordered_event_ids_sha256": _sha256(f"{component_role}-ordered-event-ids"),
        "maximum_origin_time_utc": (
            "2026-09-01T00:00:00Z" if event_count else None
        ),
        "maximum_available_at_utc": (
            "2026-09-02T00:00:00Z" if event_count else None
        ),
        "event_rows_content_sha256": event_rows["content_sha256"],
        "event_rows": event_rows,
    }


def _source_snapshot() -> dict[str, object]:
    cutover_rows = _table_artifact(
        "cutover-cross-source-match",
        table_name="cutover_match_rows",
        row_count=2,
    )
    snapshot_sha256 = _sha256("source-snapshot")
    return {
        "snapshot_id": snapshot_sha256,
        "previous_source_snapshot_sha256": None,
        "issue_time_utc": "2026-09-10T00:00:00Z",
        "query_start_utc": "2026-07-09T04:25:56Z",
        "query_end_utc": "2026-09-09T15:45:00Z",
        "acquisition": _source_acquisition(
            role="prospective_increment",
            http_status=200,
            query_count=2,
            feature_count=2,
            response_body_byte_count=512,
        ),
        "source_query_policy_sha256": _sha256("source-query-policy"),
        "parser_code_sha256": _sha256("source-parser-code"),
        "normalization_config_sha256": _sha256("source-normalization-config"),
        "deduplication_code_sha256": _sha256("source-deduplication-code"),
        "deduplication_config_sha256": _sha256("source-deduplication-config"),
        "deduplication_policy_sha256": _sha256("source-deduplication-policy"),
        "revision_policy_sha256": _sha256("source-revision-policy"),
        "historical_baseline_content_sha256": (
            "2005f0ec465978829d0832e7228f22ecd34f1a7e9f268598979de72a5295e404"
        ),
        "cutover_cross_source_match_count": 2,
        "cutover_cross_source_match_sha256": cutover_rows["content_sha256"],
        "cutover_cross_source_match_rows": cutover_rows,
        "normalized_rows": _table_artifact(
            "source-normalized",
            table_name="source_normalized_rows",
            row_count=2,
        ),
        "deduplicated_rows": _table_artifact(
            "source-deduplicated",
            table_name="source_deduplicated_rows",
            row_count=2,
        ),
        "causal_model_view": _table_artifact(
            "causal-model-view",
            table_name="causal_model_view_rows",
            row_count=2,
        ),
        "P0_event_set": _event_set_identity("P0", event_count=2),
        "R30_event_set": _event_set_identity("R30", event_count=2),
        "RP30_event_set": _event_set_identity("RP30", event_count=2),
        "seal_completed_at_utc": "2026-09-09T15:50:00Z",
        "snapshot_sha256": snapshot_sha256,
    }


def _truth_snapshot() -> dict[str, object]:
    window_rows = _table_artifact(
        "truth-window-membership",
        table_name="truth_window_membership_rows",
        row_count=2,
    )
    snapshot_sha256 = _sha256("truth-snapshot")
    return {
        "truth_snapshot_id": snapshot_sha256,
        "target_start_exclusive_utc": "2026-09-09T16:00:00Z",
        "target_end_inclusive_utc": "2026-09-16T16:00:00Z",
        "maturity_due_at_utc": "2026-09-23T16:00:00Z",
        "retry_offset_hours": 0,
        "acquisition": _source_acquisition(
            role="temporally_independent_mature_truth",
            http_status=200,
            query_count=2,
            feature_count=2,
            response_body_byte_count=512,
        ),
        "parser_code_sha256": _sha256("truth-parser-code"),
        "normalization_config_sha256": _sha256("truth-normalization-config"),
        "deduplication_code_sha256": _sha256("truth-deduplication-code"),
        "revision_policy_sha256": _sha256("truth-revision-policy"),
        "formal_origin_time_policy_sha256": _sha256("formal-origin-time-policy"),
        "window_membership_sha256": window_rows["content_sha256"],
        "window_membership_rows": window_rows,
        "normalized_rows": _table_artifact(
            "truth-normalized",
            table_name="truth_normalized_rows",
            row_count=2,
        ),
        "deduplicated_rows": _table_artifact(
            "truth-deduplicated",
            table_name="truth_deduplicated_rows",
            row_count=2,
        ),
        "realized_target_set": _target_set_identity(
            label="truth-target",
            event_count=2,
        ),
        "seal_completed_at_utc": "2026-09-23T16:01:00Z",
        "truth_snapshot_sha256": snapshot_sha256,
    }


def _selected_exposure_manifest(
    issue_prediction_seal_sha256: Sequence[str],
    *,
    empty_rows: bool,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if not empty_rows:
        for ordinal, (horizon_days, issue_index) in enumerate(
            ((7, 0), (30, 1), (90, 2)),
            start=1,
        ):
            issue_time = datetime(2028, 1, 6, tzinfo=UTC) + timedelta(
                days=7 * issue_index
            )
            issue_time_utc = issue_time.isoformat().replace("+00:00", "Z")
            target_end = (issue_time + timedelta(days=horizon_days)).isoformat().replace(
                "+00:00",
                "Z",
            )
            rows.append(
                {
                    "horizon_days": horizon_days,
                    "selection_ordinal_1_based": ordinal,
                    "scheduled_issue_sequence": issue_index + 1,
                    "issue_id": f"stage2p-selected-issue-{issue_index + 1:03d}",
                    "issue_time_utc": issue_time_utc,
                    "prediction_seal_sha256": issue_prediction_seal_sha256[
                        issue_index
                    ],
                    "target_start_exclusive_utc": issue_time_utc,
                    "target_end_inclusive_utc": target_end,
                    "selected_truth_record_sha256": _sha256(
                        f"selected-truth-record-{issue_index}"
                    ),
                    "selected_truth_revision_sequence": 0,
                    "truth_record_status": "mature_truth_sealed",
                    "truth_available": True,
                }
            )
    return {
        "profile": "stage2p_selected_exposure_manifest_v1",
        "selection_rule": (
            "earliest_issue_greedy_nonoverlap_separately_within_each_horizon"
        ),
        "selection_inputs": [
            "on_time_status",
            "issue_time_utc",
            "horizon_days",
        ],
        "horizon_order": [7, 30, 90],
        "zero_event_exposures_retained": True,
        "unavailable_exposure_replacement_forbidden": True,
        "candidate_issue_prediction_seal_sha256": list(
            issue_prediction_seal_sha256
        ),
        "rows": rows,
    }


def _alarm_area_manifest(
    selected_exposure_manifest: Mapping[str, object],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    seen_prediction_seals: set[str] = set()
    rows = cast(Sequence[Mapping[str, object]], selected_exposure_manifest["rows"])
    for row in rows:
        prediction_seal_sha256 = cast(str, row["prediction_seal_sha256"])
        if prediction_seal_sha256 in seen_prediction_seals:
            continue
        seen_prediction_seals.add(prediction_seal_sha256)
        entries.append(
            {
                "scheduled_issue_sequence": row["scheduled_issue_sequence"],
                "issue_id": row["issue_id"],
                "prediction_seal_sha256": prediction_seal_sha256,
                "forecast_bundle_manifest_sha256": _sha256(
                    f"forecast-bundle-{prediction_seal_sha256}"
                ),
                "P0_actual_alarm_area_km2_float64_hex": _float64_bits_hex(
                    599_750.0
                ),
                "P1_actual_alarm_area_km2_float64_hex": _float64_bits_hex(
                    599_900.0
                ),
                "PP_actual_alarm_area_km2_float64_hex": _float64_bits_hex(
                    599_800.0
                ),
                "maximum_pairwise_actual_alarm_area_difference_km2_float64_hex": (
                    _float64_bits_hex(150.0)
                ),
                "within_maximum_pairwise_difference": True,
            }
        )
    return {
        "profile": "stage2p_alarm_area_manifest_v1",
        "selected_exposure_manifest_union_sha256": _canonical_file_sha256(
            selected_exposure_manifest
        ),
        "maximum_allowed_pairwise_difference_km2_float64_hex": _float64_bits_hex(
            1_000.0
        ),
        "entries": entries,
    }


def _alarm_area_comparison(
    alarm_area_manifest_sha256: str,
    *,
    entry_count: int,
) -> dict[str, object]:
    return {
        "profile": "stage2p_alarm_area_comparison_v1",
        "alarm_area_manifest_sha256": alarm_area_manifest_sha256,
        "entry_count": entry_count,
        "maximum_pairwise_actual_alarm_area_difference_km2_float64_hex": (
            _float64_bits_hex(150.0 if entry_count else 0.0)
        ),
        "all_entries_within_maximum_pairwise_difference": True,
    }


def _formal_freeze_source_manifest(
    *,
    formal_window_binding_count: int,
) -> dict[str, object]:
    acquisition = _source_acquisition(
        role="formal_freeze_full_cohort",
        http_status=200,
        query_count=24,
        feature_count=24,
        response_body_byte_count=4096,
    )
    acquisition["count_preflight"].update(  # type: ignore[union-attr]
        {
            "fetch_started_at_utc": "2028-09-09T15:44:00Z",
            "fetch_completed_at_utc": "2028-09-09T15:44:30Z",
        }
    )
    acquisition.update(
        {
            "fetch_started_at_utc": "2028-09-09T15:45:00Z",
            "fetch_completed_at_utc": "2028-09-09T15:46:00Z",
        }
    )
    snapshot_identity = {
        "profile": "stage2p_formal_freeze_source_snapshot_v1",
        "source_id": acquisition["source_id"],
        "canonical_query_request_sha256": acquisition["request_identity_sha256"],
        "response_headers_sha256": acquisition["response_headers_sha256"],
        "raw_response_sha256": acquisition["raw_response"]["file_sha256"],  # type: ignore[index]
        "response_content_type": acquisition["response_content_type"],
        "response_body_byte_count": acquisition["response_body_byte_count"],
        "http_status": acquisition["http_status"],
        "fetch_started_at_utc": acquisition["fetch_started_at_utc"],
        "fetch_completed_at_utc": acquisition["fetch_completed_at_utc"],
    }
    manifest: dict[str, object] = {
        "status": "succeeded_single_full_cohort_response",
        "evaluation_scope_sha256": _sha256("formal-evaluation-scope"),
        "global_target_start_exclusive_utc": acquisition["query_start_utc"],
        "global_target_end_inclusive_utc": acquisition["query_end_utc"],
        "source_acquisition": acquisition,
        "source_snapshot_sha256": hashlib.sha256(
            canonical_json_bytes(snapshot_identity)
        ).hexdigest(),
        "snapshot_observed_at_utc": acquisition["fetch_completed_at_utc"],
        "freeze_completed_at_utc": "2028-09-09T15:50:00Z",
        "query_batch_count": 1,
        "ordered_query_request_sha256": [
            acquisition["request_identity_sha256"],
        ],
        "ordered_response_headers_sha256": [
            acquisition["response_headers_sha256"],
        ],
        "ordered_raw_response_sha256": [
            acquisition["raw_response"]["file_sha256"],  # type: ignore[index]
        ],
        "normalized_rows": _table_artifact(
            "formal-normalized-rows",
            table_name="normalized_rows",
            row_count=24,
        ),
        "deduplicated_rows": _table_artifact(
            "formal-deduplicated-rows",
            table_name="deduplicated_rows",
            row_count=24,
        ),
        "preferred_field_rows": _table_artifact(
            "formal-preferred-field-rows",
            table_name="preferred_field_rows",
            row_count=24,
        ),
        "window_membership_rows": _table_artifact(
            "formal-window-membership",
            table_name="window_membership_rows",
            row_count=formal_window_binding_count,
        ),
        "formal_window_target_bindings": _table_artifact(
            "formal-window-target-bindings",
            table_name="formal_window_target_bindings",
            row_count=formal_window_binding_count,
        ),
        "parser_code_sha256": _sha256("formal-parser-code"),
        "normalization_config_sha256": _sha256("formal-normalization-config"),
        "deduplication_code_sha256": _sha256("formal-deduplication-code"),
        "deduplication_config_sha256": _sha256("formal-deduplication-config"),
        "revision_policy_sha256": _sha256("formal-revision-policy"),
        "code_manifest_sha256": _sha256("code-manifest"),
        "local_restricted": True,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def _formal_freeze_failure_evidence(
    status: str = "not_run_scheduled_issue_cap_terminal",
) -> dict[str, object]:
    failure_details = {
        "not_run_no_complete_scope": (
            "scope_selection",
            "no_complete_scope",
        ),
        "not_run_scheduled_issue_cap_terminal": (
            "scope_selection",
            "scheduled_issue_cap_terminal",
        ),
        "failed_count_preflight": (
            "count_preflight",
            "count_preflight_failed",
        ),
        "failed_count_limit": (
            "count_preflight",
            "count_limit_exceeded",
        ),
        "failed_query_fetch": (
            "query_fetch",
            "query_fetch_failed",
        ),
        "failed_query_parse_or_count_mismatch": (
            "query_parse_or_count_consistency",
            "query_parse_or_count_mismatch",
        ),
        "failed_local_derivation_or_freeze": (
            "durable_freeze",
            "local_derivation_or_freeze_failed",
        ),
    }
    failure_stage, failure_code = failure_details[status]
    has_scope = status not in {
        "not_run_no_complete_scope",
        "not_run_scheduled_issue_cap_terminal",
    }
    has_preflight = status not in {
        "not_run_no_complete_scope",
        "not_run_scheduled_issue_cap_terminal",
        "failed_count_preflight",
    }
    has_query_failure_artifact = status in {
        "failed_query_fetch",
        "failed_query_parse_or_count_mismatch",
        "failed_local_derivation_or_freeze",
    }
    evidence: dict[str, object] = {
        "status": status,
        "evaluation_scope_sha256": _sha256("formal-evaluation-scope"),
        "scheduled_at_utc": "2028-09-09T15:40:00Z",
        "global_target_start_exclusive_utc": (
            "2026-09-09T16:00:00Z" if has_scope else None
        ),
        "global_target_end_inclusive_utc": (
            "2028-09-01T16:00:00Z" if has_scope else None
        ),
        "count_preflight": (
            _count_preflight(
                start="2026-09-09T16:00:00Z",
                end="2028-09-01T16:00:00Z",
                label=f"formal-failure-{status}",
                parsed_count=24,
            )
            if has_preflight
            else None
        ),
        "query_failure_artifact_sha256": (
            _sha256(f"query-failure-{status}")
            if has_query_failure_artifact
            else None
        ),
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "contains_target_or_effect_rows": False,
        "local_restricted": True,
    }
    evidence["failure_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(evidence)
    ).hexdigest()
    return evidence


def _endpoint_result(endpoint_id: str) -> dict[str, object]:
    return {
        "endpoint_id": endpoint_id,
        "point_estimate": 0.08,
        "familywise_lower_bound": 0.01,
        "familywise_upper_bound": 0.15,
        "familywise_lower_bound_gt_zero": True,
        "bootstrap_replications": 2000,
        "lower_quantile": 0.00625,
        "upper_quantile": 0.99375,
        "percentile_method": "numpy_linear",
        "point_contribution_rows_sha256": _sha256(f"point-{endpoint_id}"),
        "bootstrap_distribution_sha256": _sha256(f"bootstrap-{endpoint_id}"),
    }


def _robustness_result(endpoint_id: str) -> dict[str, object]:
    return {
        "endpoint_id": endpoint_id,
        "region_contribution_count": 39,
        "region_contribution_rows_sha256": _sha256(f"region-{endpoint_id}"),
        "region_additive_closure_error": 0.0,
        "region_additive_closure_verified": True,
        "largest_positive_region_identity_sha256": _sha256(
            f"largest-positive-region-{endpoint_id}"
        ),
        "point_estimate_after_largest_positive_region_removal": 0.01,
        "remains_positive_after_largest_positive_region_removal": True,
        "target_cluster_count": 10,
        "cluster_contribution_rows_sha256": _sha256(f"cluster-{endpoint_id}"),
        "cluster_additive_closure_error": 0.0,
        "cluster_additive_closure_verified": True,
        "largest_positive_cluster_identity_sha256": _sha256(
            f"largest-positive-cluster-{endpoint_id}"
        ),
        "point_estimate_after_largest_positive_cluster_removal": 0.01,
        "remains_positive_after_largest_positive_cluster_removal": True,
    }


def _endpoint_artifact_map(
    *,
    table_name: str,
    label: str,
    row_count: int,
) -> dict[str, dict[str, object]]:
    return {
        endpoint_id: _table_artifact(
            f"{label}-{endpoint_id}",
            table_name=table_name,
            row_count=row_count,
        )
        for endpoint_id in ENDPOINT_IDS
    }


def _effect_rows_manifest(input_freeze_sha256: str) -> dict[str, object]:
    return {
        "profile": "stage2p_effect_rows_manifest_v1",
        "formal_run_id": _sha256("formal-run"),
        "input_freeze_sha256": input_freeze_sha256,
        "evaluation_code_commit": "b" * 40,
        "evaluation_code_sha256": _sha256("evaluation-code"),
        "cluster_membership_rows": _table_artifact(
            "effect-cluster-membership",
            table_name="cluster_membership_rows",
            row_count=24,
        ),
        "point_contribution_rows": _endpoint_artifact_map(
            table_name="point_contribution_rows",
            label="point-contribution",
            row_count=24,
        ),
        "region_contribution_rows": _endpoint_artifact_map(
            table_name="region_contribution_rows",
            label="region-contribution",
            row_count=39,
        ),
        "cluster_contribution_rows": _endpoint_artifact_map(
            table_name="cluster_contribution_rows",
            label="cluster-contribution",
            row_count=10,
        ),
        "bootstrap_index_rows": _table_artifact(
            "bootstrap-index",
            table_name="bootstrap_index_rows",
            row_count=20_000,
        ),
    }


def _bootstrap_distribution_rows() -> dict[str, dict[str, object]]:
    return _endpoint_artifact_map(
        table_name="bootstrap_distribution_rows",
        label="bootstrap-distribution",
        row_count=2_000,
    )


def _result_bundle_manifest(
    input_freeze_sha256: str,
    *,
    failure_stage: str | None = None,
) -> dict[str, object]:
    bootstrap_distribution_rows = _bootstrap_distribution_rows()
    slots: dict[str, object] = {
        "effect_rows_manifest_sha256": _sha256("complete-effect-rows-manifest"),
        "alarm_area_comparison_sha256": _sha256("complete-alarm-area-comparison"),
        "bootstrap_distribution_rows": bootstrap_distribution_rows,
        "endpoint_results_sha256": _sha256("complete-endpoint-results"),
        "robustness_results_sha256": _sha256("complete-robustness-results"),
    }
    if failure_stage is not None:
        first_unavailable_slot = {
            "effect_rows_open": 0,
            "alarm_area_comparison": 1,
            "bootstrap": 2,
            "endpoint_evaluation": 3,
            "robustness_evaluation": 4,
            "result_bundle_install": 5,
            "result_seal": 5,
        }[failure_stage]
        slot_names = tuple(slots)
        for slot_name in slot_names[first_unavailable_slot:]:
            slots[slot_name] = None
    return {
        "profile": "stage2p_result_bundle_manifest_v1",
        "formal_run_id": _sha256("formal-run"),
        "input_freeze_sha256": input_freeze_sha256,
        "evaluation_code_commit": "b" * 40,
        "evaluation_code_sha256": _sha256("evaluation-code"),
        "execution_status": "valid" if failure_stage is None else "invalid_execution",
        "failure_stage": failure_stage,
        "failure_code": None if failure_stage is None else f"{failure_stage}_failed",
        "available_audit_artifact_sha256": (
            [] if failure_stage is None else [_sha256(f"audit-{failure_stage}")]
        ),
        **slots,
    }


def _confirmatory_result(
    input_freeze_sha256: str,
    *,
    alarm_area_manifest_sha256: str,
    alarm_area_entry_count: int,
) -> dict[str, object]:
    effect_rows_manifest = _effect_rows_manifest(input_freeze_sha256)
    effect_rows_manifest_sha256 = _canonical_file_sha256(effect_rows_manifest)
    bootstrap_distribution_rows = _bootstrap_distribution_rows()
    bootstrap_replicates_sha256 = _canonical_file_sha256(
        bootstrap_distribution_rows
    )
    alarm_area_comparison = _alarm_area_comparison(
        alarm_area_manifest_sha256,
        entry_count=alarm_area_entry_count,
    )
    alarm_area_comparison_sha256 = _canonical_file_sha256(alarm_area_comparison)
    endpoint_results = {
        endpoint_id: _endpoint_result(endpoint_id) for endpoint_id in ENDPOINT_IDS
    }
    robustness_results = {
        endpoint_id: _robustness_result(endpoint_id) for endpoint_id in ENDPOINT_IDS
    }
    for endpoint_id in ENDPOINT_IDS:
        endpoint_results[endpoint_id]["point_contribution_rows_sha256"] = cast(
            Mapping[str, object],
            effect_rows_manifest["point_contribution_rows"],
        )[endpoint_id]["content_sha256"]  # type: ignore[index]
        endpoint_results[endpoint_id]["bootstrap_distribution_sha256"] = (
            bootstrap_distribution_rows[endpoint_id]["content_sha256"]
        )
        robustness_results[endpoint_id]["region_contribution_rows_sha256"] = cast(
            Mapping[str, object],
            effect_rows_manifest["region_contribution_rows"],
        )[endpoint_id]["content_sha256"]  # type: ignore[index]
        robustness_results[endpoint_id]["cluster_contribution_rows_sha256"] = cast(
            Mapping[str, object],
            effect_rows_manifest["cluster_contribution_rows"],
        )[endpoint_id]["content_sha256"]  # type: ignore[index]

    result_bundle_manifest = _result_bundle_manifest(input_freeze_sha256)
    result_bundle_manifest.update(
        {
            "effect_rows_manifest_sha256": effect_rows_manifest_sha256,
            "alarm_area_comparison_sha256": alarm_area_comparison_sha256,
            "bootstrap_distribution_rows": bootstrap_distribution_rows,
            "endpoint_results_sha256": _canonical_file_sha256(endpoint_results),
            "robustness_results_sha256": _canonical_file_sha256(
                robustness_results
            ),
        }
    )
    return {
        "formal_run_id": _sha256("formal-run"),
        "formal_effect_look_number": 1,
        "input_freeze_sha256": input_freeze_sha256,
        "effect_rows_manifest_sha256": effect_rows_manifest_sha256,
        "bootstrap_replicates_sha256": bootstrap_replicates_sha256,
        "alarm_area_comparison_sha256": alarm_area_comparison_sha256,
        "maximum_pairwise_actual_alarm_area_difference_km2": (
            150.0 if alarm_area_entry_count else 0.0
        ),
        "evaluation_code_commit": "b" * 40,
        "evaluation_code_sha256": _sha256("evaluation-code"),
        "endpoint_results": endpoint_results,
        "robustness_results": robustness_results,
        "execution_status": "valid",
        "all_four_familywise_lower_bounds_gt_zero": True,
        "P1_minus_P0_macro_recall_point_gain_gte_0_05": True,
        "all_four_region_removals_remain_positive": True,
        "all_four_cluster_removals_remain_positive": True,
        "formal_gate_passed": True,
        "decision": "pass_direct_improvement",
        "stop_action": "promote_P1_to_next_independent_validation_without_test_tuning",
        "additional_confirmatory_looks_authorized": False,
        "test_tuning_authorized": False,
        "sealed_at_utc": "2028-09-09T16:01:00Z",
        "result_bundle_sha256": _canonical_file_sha256(result_bundle_manifest),
    }


def _evaluation_input_freeze(
    *,
    trigger: int,
    sample_gate_met: bool,
    scheduled_issue_count: int | None = None,
    formal_freeze_status: str | None = None,
) -> dict[str, object]:
    if trigger not in {52, 104, 130}:
        raise ValueError("unsupported synthetic trigger")
    resolved_formal_freeze_status = formal_freeze_status or (
        "not_run_scheduled_issue_cap_terminal"
        if trigger == 130
        else "succeeded_single_full_cohort_response"
    )
    allowed_formal_freeze_statuses = {
        "succeeded_single_full_cohort_response",
        "not_run_no_complete_scope",
        "not_run_scheduled_issue_cap_terminal",
        "failed_count_preflight",
        "failed_count_limit",
        "failed_query_fetch",
        "failed_query_parse_or_count_mismatch",
        "failed_local_derivation_or_freeze",
    }
    if resolved_formal_freeze_status not in allowed_formal_freeze_statuses:
        raise ValueError("unsupported synthetic formal-freeze status")
    formal_freeze_succeeded = (
        resolved_formal_freeze_status
        == "succeeded_single_full_cohort_response"
    )
    formal_freeze_no_scope = (
        resolved_formal_freeze_status == "not_run_no_complete_scope"
    )
    if sample_gate_met and not formal_freeze_succeeded:
        raise ValueError("formal-freeze unavailability cannot satisfy the sample gate")
    effective_sample_gate_met = sample_gate_met and formal_freeze_succeeded

    if trigger == 130:
        checkpoint = 3
        on_time_count = 80
        trigger_reason = "scheduled_issue_cap_130"
        status = "evidence_insufficient"
        evaluation_sequence = 2
        previous_freeze = _sha256("previous-evaluation-52")
    else:
        checkpoint = 1 if trigger == 52 else 2
        on_time_count = trigger
        trigger_reason = f"on_time_checkpoint_{trigger}"
        status = (
            "confirmatory_effects_authorized"
            if effective_sample_gate_met
            else ("continue_blind_to_104" if trigger == 52 else "evidence_insufficient")
        )
        evaluation_sequence = checkpoint
        previous_freeze = None if trigger == 52 else _sha256("previous-evaluation-52")
    issue_count = on_time_count
    truth_count = issue_count * 3 if trigger in {52, 104} else 0
    evaluable = effective_sample_gate_met
    ordered_issue_prediction_seal_sha256 = [
        _sha256(f"issue-seal-{index}") for index in range(issue_count)
    ]
    selected_exposure_manifest = _selected_exposure_manifest(
        ordered_issue_prediction_seal_sha256,
        empty_rows=formal_freeze_no_scope,
    )
    selected_exposure_manifest_union_sha256 = _canonical_file_sha256(
        selected_exposure_manifest
    )
    alarm_area_manifest = _alarm_area_manifest(selected_exposure_manifest)
    alarm_area_manifest_sha256 = _canonical_file_sha256(alarm_area_manifest)
    formal_freeze_source_manifest = (
        _formal_freeze_source_manifest(formal_window_binding_count=truth_count)
        if formal_freeze_succeeded
        else None
    )
    formal_freeze_failure_evidence = (
        None
        if formal_freeze_succeeded
        else _formal_freeze_failure_evidence(resolved_formal_freeze_status)
    )
    if formal_freeze_succeeded:
        target_count = 24 if evaluable else 19
        union_unique_supported_event_count: int | None = target_count
        union_unique_full_study_area_event_count: int | None = target_count
        unique_target_cluster_count: int | None = 10 if evaluable else 9
        realized_target_union: dict[str, object] | None = _target_set_identity(
            label="formal-realized-target",
            event_count=target_count,
        )
        horizon_evaluability = [
            _horizon_evaluability(horizon, evaluable=evaluable)
            for horizon in (7, 30, 90)
        ]
        final_window_membership_sha256: str | None = cast(
            Mapping[str, object],
            formal_freeze_source_manifest["window_membership_rows"],
        )["content_sha256"]  # type: ignore[assignment]
        cluster_membership_sha256: str | None = cast(
            str,
            _table_artifact(
                "evaluation-cluster-membership",
                table_name="cluster_membership_rows",
                row_count=unique_target_cluster_count,
            )["content_sha256"],
        )
        bootstrap_status = (
            "passed" if effective_sample_gate_met else "not_run_basic_gate_failed"
        )
    elif formal_freeze_no_scope:
        union_unique_supported_event_count = 0
        union_unique_full_study_area_event_count = 0
        unique_target_cluster_count = 0
        realized_target_union = _empty_target_set_identity("no-complete-scope")
        horizon_evaluability = [
            _empty_horizon_evaluability(horizon) for horizon in (7, 30, 90)
        ]
        final_window_membership_sha256 = cast(
            str,
            _table_artifact(
                "empty-formal-window-membership",
                table_name="window_membership_rows",
                row_count=0,
            )["content_sha256"],
        )
        cluster_membership_sha256 = cast(
            str,
            _table_artifact(
                "empty-cluster-membership",
                table_name="cluster_membership_rows",
                row_count=0,
            )["content_sha256"],
        )
        bootstrap_status = "not_run_basic_gate_failed"
    else:
        union_unique_supported_event_count = None
        union_unique_full_study_area_event_count = None
        unique_target_cluster_count = None
        realized_target_union = None
        horizon_evaluability = [
            _unavailable_horizon_evaluability(
                horizon,
                scheduled_issue_cap_terminal=(
                    resolved_formal_freeze_status
                    == "not_run_scheduled_issue_cap_terminal"
                ),
            )
            for horizon in (7, 30, 90)
        ]
        final_window_membership_sha256 = None
        cluster_membership_sha256 = None
        bootstrap_status = "not_run_formal_freeze_unavailable"

    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "EvaluationFreezeRecord",
        "experiment_id": "stage2p-prospective-recent-seismicity-v1",
        "protocol_version": "0.2.4",
        "stage_id": "Stage2P-1A",
        "cohort_definition_sha256": _sha256("cohort-definition"),
        "code_manifest_sha256": _sha256("code-manifest"),
        "formal_freeze_status": resolved_formal_freeze_status,
        "formal_freeze_source_manifest": formal_freeze_source_manifest,
        "formal_freeze_source_manifest_sha256": (
            formal_freeze_source_manifest["manifest_sha256"]
            if formal_freeze_source_manifest is not None
            else None
        ),
        "formal_freeze_failure_evidence": formal_freeze_failure_evidence,
        "formal_freeze_failure_evidence_sha256": (
            formal_freeze_failure_evidence["failure_evidence_sha256"]
            if formal_freeze_failure_evidence is not None
            else None
        ),
        "evaluation_sequence": evaluation_sequence,
        "phase": "input_freeze",
        "checkpoint_number": checkpoint,
        "trigger_reason": trigger_reason,
        "scheduled_issue_count": (
            trigger if scheduled_issue_count is None else scheduled_issue_count
        ),
        "trigger_on_time_issue_count": on_time_count,
        "previous_evaluation_freeze_sha256": previous_freeze,
        "input_freeze_sha256": None,
        "frozen_at_utc": "2028-09-09T16:00:00Z",
        "effect_rows_opened_at_utc": None,
        "all_90d_windows_mature": trigger != 130,
        "ordered_issue_prediction_seal_sha256": (
            ordered_issue_prediction_seal_sha256
        ),
        "ordered_truth_record_sha256": [
            _sha256(f"truth-record-{index}") for index in range(truth_count)
        ],
        "selected_exposure_manifest": selected_exposure_manifest,
        "selected_exposure_manifest_union_sha256": (
            selected_exposure_manifest_union_sha256
        ),
        "truth_availability_manifest_union_sha256": _sha256("truth-availability-union"),
        "final_window_membership_sha256": final_window_membership_sha256,
        "realized_target_union": realized_target_union,
        "union_unique_supported_M5_6_event_count": (
            union_unique_supported_event_count
        ),
        "union_unique_full_study_area_M5_6_event_count": (
            union_unique_full_study_area_event_count
        ),
        "unique_target_cluster_count": unique_target_cluster_count,
        "horizon_evaluability": horizon_evaluability,
        "sample_gate_met": effective_sample_gate_met,
        "cluster_membership_sha256": cluster_membership_sha256,
        "bootstrap_plan_sha256": _sha256("bootstrap-plan"),
        "bootstrap_preflight": {
            "status": bootstrap_status,
            "index_matrix_sha256": _sha256("bootstrap-index-matrix")
            if effective_sample_gate_met
            else None,
            "planned_replication_count": 2000,
            "generated_replication_count": (
                2000 if effective_sample_gate_met else 0
            ),
            "zero_denominator_replication_count": (
                0 if effective_sample_gate_met else None
            ),
            "frozen_before_effect_rows_open": effective_sample_gate_met,
            "redraw_or_discard_performed": False,
        },
        "statistics_policy_sha256": _sha256("statistics-policy"),
        "region_map_sha256": _sha256("region-map"),
        "alarm_area_manifest": alarm_area_manifest,
        "alarm_area_manifest_sha256": alarm_area_manifest_sha256,
        "evaluation_code_commit": "b" * 40,
        "evaluation_code_sha256": _sha256("evaluation-code"),
        "environment_lock_file_sha256": (
            "5d18c0e213a77dae9e133be97deef6756cbf5c9ee3d002b42bca6b2ea36e6eac"
        ),
        "pyproject_file_sha256": (
            "cb12f7c87853909994296cd33c4022c78ebdbf178c4add1f97d5f333246d2fd8"
        ),
        "status": status,
        "confirmatory_effects_authorized": effective_sample_gate_met,
        "confirmatory_effect_rows_opened_before_freeze": False,
        "confirmatory_result": None,
    }
    return _seal_timestamped_record(
        payload,
        subject_type="EvaluationFreezeRecord",
        deadline_utc="2028-09-09T16:05:00Z",
    )


def _evaluation_result_seal(input_freeze: Mapping[str, object]) -> dict[str, object]:
    checkpoint = cast(int, input_freeze["checkpoint_number"])
    trigger = cast(int, input_freeze["trigger_on_time_issue_count"])
    payload = {
        key: copy.deepcopy(value)
        for key, value in input_freeze.items()
        if key not in RFC3161_EXCLUDED_TOP_LEVEL_FIELDS
    }
    input_sha256 = cast(str, input_freeze["content_sha256"])
    payload.update(
        {
            "evaluation_sequence": 2 if checkpoint == 1 else 3,
            "phase": "result_seal",
            "previous_evaluation_freeze_sha256": input_sha256,
            "input_freeze_sha256": input_sha256,
            "frozen_at_utc": "2028-09-09T16:01:00Z",
            "record_core_frozen_at_utc": "2028-09-09T16:01:00Z",
            "effect_rows_opened_at_utc": "2028-09-09T16:00:30Z",
            "status": "confirmatory_result_sealed",
            "sample_gate_met": True,
            "confirmatory_effects_authorized": True,
            "confirmatory_result": _confirmatory_result(
                input_sha256,
                alarm_area_manifest_sha256=cast(
                    str,
                    input_freeze["alarm_area_manifest_sha256"],
                ),
                alarm_area_entry_count=len(
                    cast(
                        Sequence[object],
                        cast(
                            Mapping[str, object],
                            input_freeze["alarm_area_manifest"],
                        )["entries"],
                    )
                ),
            ),
            "trigger_on_time_issue_count": trigger,
        }
    )
    return _seal_timestamped_record(
        payload,
        subject_type="EvaluationFreezeRecord",
        deadline_utc="2028-09-09T16:06:00Z",
    )


def test_governance_is_accepted_frozen_and_only_closure_then_synthetic_is_open() -> None:
    config = _load_config()
    protocol = config["protocol"]

    assert protocol["protocol_version"] == "0.2.4"
    assert protocol["stage_id"] == "Stage2P-1A"
    assert protocol["status"] == "accepted"
    assert protocol["protocol_tag"] == "v0.2.4-prospective-seismicity-protocol"
    assert protocol["code_tag"] == "v0.2.4-prospective-seismicity-code"
    assert protocol["stage2p1_protocol_frozen"] is True
    assert protocol["execution_authorized"] is False
    assert protocol["real_issue_authorized"] is False
    assert protocol["real_catalog_read_authorized"] is False
    assert protocol["real_network_fetch_authorized"] is False
    assert protocol["next_authorized_action"] == (
        "final_validation_commit_push_tag_readback_then_stage2p1b_synthetic_only"
    )
    assert protocol["locked_test_read_count"] == 0
    assert protocol["locked_test_run_count"] == 0

    stage = _load_research_config()["stage_2p_route_review"]
    assert stage["stage_id"] == "Stage2P-1A"
    assert stage["status"] == "accepted"
    assert stage["stage2p1_protocol_frozen"] is True
    assert stage["execution_authorized"] is False
    assert stage["real_issue_authorized"] is False
    assert stage["stage2p1_real_catalog_or_network_read_authorized"] is False
    assert (
        stage["stage2p1_next_authorized_action"]
        == "final_validation_commit_push_tag_readback_then_stage2p1b_synthetic_only"
    )
    assert stage["stage2p1_protocol_path"] == ("configs/prospective_recent_seismicity.yaml")
    assert stage["stage2p1_record_schema_path"] == (
        "data/contracts/stage2p_prospective_records.json"
    )
    assert stage["stage2p1_first_issue_not_before_local"] == ("2026-09-10T00:00:00+08:00")


def test_stage2p_imports_use_only_the_exact_frozen_symbol_allowlist() -> None:
    config = _load_config()
    isolation = config["implementation_isolation"]
    allowed = set(cast(Sequence[str], isolation["allowed_pure_reuse"]))
    forbidden = set(cast(Sequence[str], isolation["forbidden_reuse"]))
    expected_allowed = {
        "seismoflux.background.grid.point_cell_index",
        "seismoflux.background.grid.project_study_area_to_equal_area",
        "seismoflux.background.grid.build_equal_area_grid_family",
        "seismoflux.background.grid.aggregate_fine_masses_to_coarse",
        "seismoflux.background.grid.diagnose_three_grid_convergence",
        "seismoflux.background.grid.require_three_grid_convergence",
        "seismoflux.data.common.canonical_json_bytes",
        "seismoflux.stage2s.spatial.aggregate_operational_mass_to_25km",
        "seismoflux.stage2s.spatial.build_normalized_kde",
        "seismoflux.stage2s.spatial.build_recent_component",
        "seismoflux.stage2s.spatial.event_cell_index_25km",
        "seismoflux.stage2s.spatial.mix_density",
        "seismoflux.stage2s.spatial.select_alarm_prefix",
    }
    assert allowed == expected_allowed
    assert allowed.isdisjoint(forbidden)

    controlled_prefixes = (
        "seismoflux.background",
        "seismoflux.stage2s",
        "seismoflux.anomaly_increment",
        "seismoflux.data",
    )
    stage2p_root = ROOT / "src" / "seismoflux" / "stage2p"
    historical_source_paths = (
        stage2p_root / "__init__.py",
        stage2p_root / "validation.py",
    )
    for source_path in historical_source_paths:
        assert source_path.is_file()
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(controlled_prefixes):
                        pytest.fail(
                            f"{source_path}: module import {alias.name!r} bypasses "
                            "the exact symbol allowlist"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("seismoflux.stage2p"):
                    continue
                if not module.startswith(controlled_prefixes):
                    continue
                for alias in node.names:
                    assert alias.name != "*", (
                        f"{source_path}: wildcard import from {module!r} is forbidden"
                    )
                    symbol = f"{module}.{alias.name}"
                    assert symbol in allowed, (
                        f"{source_path}: controlled reuse {symbol!r} is not in the "
                        "exact frozen allowlist"
                    )
                    assert not any(
                        symbol == denied or symbol.startswith(f"{denied}.")
                        for denied in forbidden
                    ), f"{source_path}: forbidden reuse {symbol!r}"


def test_timestamp_attachments_are_local_restricted_ignored_and_cannot_escape() -> None:
    config = _load_config()
    attachment_paths = config["remote_timestamp"]["attachment_paths"]
    local_restricted = config["storage_and_publication"]["local_restricted"]
    expected_root = PurePosixPath("data/interim/stage2p/timestamp_attachments")
    assert PurePosixPath(local_restricted["timestamp_attachment_root"]) == expected_root

    for template_name in ("request", "response"):
        template = cast(str, attachment_paths[template_name])
        expanded = template.format(
            record_type="IssueInputSnapshotRecord",
            preimage_sha256="a" * 64,
            attempt_index=0,
        )
        relative_path = PurePosixPath(expanded)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert relative_path.is_relative_to(expected_root)
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", expanded],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert ignored.returncode == 0, (
            f"timestamp attachment template {template_name!r} is not Git-ignored"
        )


def test_comcat_cutover_query_first_seen_and_model_windows_are_exact() -> None:
    config = _load_config()
    baseline = config["historical_baseline"]
    comcat = config["comcat"]
    query = comcat["exact_query"]
    calendar = config["calendar"]
    snapshot = config["issue_input_snapshot"]

    assert baseline["cutoff_utc"] == "2026-07-09T04:25:56Z"
    assert (
        baseline["raw_inventory_sha256"]
        == "c6b817d48317d3228f6709cdcb77b934ccb80be99ae43fef25ef8fc0378f94bf"
    )
    assert (
        baseline["normalized_file_sha256"]
        == "2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347"
    )
    assert (
        baseline["normalized_content_sha256"]
        == "2005f0ec465978829d0832e7228f22ecd34f1a7e9f268598979de72a5295e404"
    )
    assert comcat["source_id"] == "usgs_anss_comcat_fdsn_event_api_v1"
    assert comcat["endpoint"] == "https://earthquake.usgs.gov/fdsnws/event/1/query"
    assert query == {
        "minlongitude": 73.446961,
        "minlatitude": 20.22909,
        "maxlongitude": 135.08583,
        "maxlatitude": 53.557926,
        "minmagnitude": 3.9,
        "eventtype": "earthquake",
        "format": "geojson",
        "orderby": "time-asc",
        "limit": 20000,
        "includeallorigins": False,
        "includeallmagnitudes": False,
        "includedeleted": False,
        "includesuperseded": "omit_not_applicable_without_eventid",
        "reviewstatus": "omit_use_default_all",
        "offset": 1,
        "jsonerror": True,
        "nodata": 204,
        "catalog": "omit",
        "contributor": "omit",
    }
    assert comcat["local_model_magnitude_eligibility"] == "magnitude_gte_4.0"
    assert comcat["query_count_preflight_required"] is True
    assert comcat["dynamic_query_partitioning_forbidden"] is True
    assert comcat["allowed_success_http_statuses"] == [200, 204]
    assert comcat["http_204_semantics"] == (
        "legitimate_zero_event_success_with_empty_body_and_zero_rows"
    )
    assert comcat["successful_200_geojson_parse_verified"] is True
    assert comcat["successful_204_geojson_parse_verified"] is False
    assert comcat["local_exact_polygon_filter_required"] is True
    assert comcat["first_seen"]["rule"] == "first_sealed_snapshot_fetch_completed_at_utc"
    assert comcat["first_seen"]["origin_time_as_first_seen_forbidden"] is True
    assert comcat["first_seen"]["provider_updated_as_first_seen_forbidden"] is True

    first_issue = _utc(calendar["first_issue_not_before_utc"])
    cutover = _utc(baseline["cutoff_utc"])
    assert first_issue - timedelta(days=60) > cutover
    assert calendar["query_end_lag_minutes"] == 15
    assert calendar["query_end_rule"] == "T_minus_15_minutes"
    assert calendar["scheduled_issue_sequence"] == {
        "counts": "every_rule_Thursday_including_missed_issue",
        "maximum": 130,
        "maximum_reached_before_104_on_time_action": "evidence_insufficient_and_stop",
    }
    assert calendar["on_time_issue_sequence"] == {
        "counts": "on_time_prediction_records_only",
        "maximum": 104,
        "missed_issue_value": None,
    }
    assert comcat["query_profiles"]["issue_input"] == {
        "query_start_utc": "frozen_cutover_utc",
        "query_end_utc": "Q_equals_T_minus_15_minutes",
        "local_origin_window": "(cutover,Q]",
        "source_role": "prospective_increment",
    }
    mature_profile = comcat["query_profiles"]["mature_truth"]
    assert mature_profile["query_start_utc"] == "issue_T"
    assert mature_profile["query_end_utc"] == "T_plus_h"
    assert mature_profile["local_origin_window"] == "(T,T+h]"
    assert mature_profile["issue_cutover_or_Q_window_reuse_forbidden"] is True
    formal_profile = comcat["query_profiles"]["formal_freeze_full_cohort"]
    assert formal_profile["query_start_utc"] == "minimum_selected_exposure_issue_T"
    assert formal_profile["query_end_utc"] == "maximum_selected_exposure_T_plus_h"
    assert formal_profile["exactly_one_query_response"] is True
    assert formal_profile["all_exposure_windows_derived_locally_from_this_response"] is True
    assert comcat["query_profiles"]["truth_revision"] == {
        "remote_query_forbidden": True,
        "derivation_source": "formal_freeze_full_cohort_single_response",
    }
    assert snapshot["composite_snapshot"]["R30"]["origin_window"] == ("(query_end-30d,query_end]")
    assert snapshot["composite_snapshot"]["RP30"]["origin_window"] == (
        "(query_end-60d,query_end-30d]"
    )
    assert snapshot["composite_snapshot"]["R30"]["exact_duration_days"] == 30
    assert snapshot["composite_snapshot"]["RP30"]["exact_duration_days"] == 30
    assert snapshot["composite_snapshot"]["P0"]["rebuilt_each_issue"] is True
    assert snapshot["missed_issue_installs_source_snapshot_or_prediction"] is False
    assert snapshot["prediction_installed_true_only_for_on_time"] is True

    models = config["models"]
    assert models["support_id"] == "local-support-f6816ab6c6581306"
    assert models["training_start_utc"] == "1970-01-01T00:00:00Z"
    assert models["local_Mc"] == 4.0
    assert models["gaussian_kde_bandwidth_km"] == 75.0
    assert models["P1"]["P0_weight"] == models["P1"]["R30_weight"] == 0.5
    assert models["PP"]["P0_weight"] == models["PP"]["RP30_weight"] == 0.5
    assert models["P1"]["empty_R30_action"] == "byte_identical_to_P0"
    assert models["PP"]["empty_RP30_action"] == "byte_identical_to_P0"


def test_timestamp_license_deduplication_and_truth_revision_are_frozen() -> None:
    config = _load_config()
    timestamp = config["remote_timestamp"]
    comcat = config["comcat"]
    dedup = comcat["deduplication"]
    cohort = config["target_cohort"]

    assert timestamp["authority_order"] == [
        "http://timestamp.digicert.com",
        "http://timestamp.sectigo.com",
    ]
    assert timestamp["method"] == "POST"
    assert timestamp["request_content_type"] == "application/timestamp-query"
    assert timestamp["response_content_type"] == "application/timestamp-reply"
    assert timestamp["request"] == {
        "message_imprint_digest": "SHA256",
        "nonce_required": True,
        "certReq": True,
    }
    assert timestamp["attempt_policy"]["every_attempt_recorded"] is True
    assert (
        timestamp["attempt_policy"]["every_attempt_has_nonnull_attempt_completed_at_utc"]
        is True
    )
    assert (
        timestamp["attempt_policy"]["selection"]
        == "first_offline_trust_path_valid_response_with_genTime_before_"
        "record_specific_deadline"
    )
    assert timestamp["attempt_policy"]["all_authorities_failed_action"] == (
        "fail_closed_record_not_installed"
    )
    assert timestamp["preimage_profile"] == "stage2p_rfc3161_core_v1"
    assert timestamp["preimage"] == (
        "SHA256_of_seismoflux_canonical_json_v1_record_omitting_exactly_the_three"
        "_top_level_fields_timestamp_attempt_evidence_remote_timestamp_content_sha256"
    )
    assert timestamp["nested_RFC3161_proof_or_token_fields_forbidden"] is True
    assert timestamp["proof_preimage_sha256_must_equal_recomputed_record_core_sha256"] is True
    assert timestamp["selected_response_sha256_must_equal_selected_attempt_response_sha256"]
    assert timestamp["deadline_and_attempt_checks_apply_when_remote_timestamp_is_null"] is True
    assert timestamp["response_attempt_completion_rule"] == (
        "response_received_at_utc_equals_attempt_completed_at_utc_for_every_response_outcome;"
        " network_failure_has_null_response_received_at_utc_and_nonnull_attempt_completed_at_utc"
    )
    assert timestamp["authority_sequence_time_order"] == (
        "attempt_1_request_started_at_utc_gte_attempt_0_attempt_completed_at_utc"
    )

    license_policy = comcat["license"]
    assert license_policy["status"] == (
        "USGS_authored_data_public_domain_with_non_USGS_partner_exceptions"
    )
    assert license_policy["anss_data_and_products_policy_url"] == (
        "https://www.usgs.gov/media/files/anss-data-and-products-policy"
    )
    assert license_policy["usgs_data_licensing_url"] == (
        "https://www.usgs.gov/data-management/data-licensing"
    )
    assert license_policy["non_USGS_partner_content_requires_item_level_review"] is True
    assert license_policy["raw_and_row_level_artifacts_remain_local_restricted"] is True

    assert dedup["graph_nodes"] == "preferred_feature_id_and_associated_ids"
    assert dedup["component_rule"] == "connected_components"
    assert dedup["same_identifier_revision_selection"] == {
        "primary": "maximum_provider_updated",
        "tie_break": "minimum_UTF8_raw_feature_bytes_sha256",
    }
    cross_source = dedup["cutover_cross_source_rule"]
    assert cross_source["maximum_time_delta_seconds"] == 300
    assert cross_source["maximum_distance_km"] == 50
    assert cross_source["maximum_magnitude_delta"] == 0.5
    assert cross_source["matched_anchor"] == "frozen_local_baseline"
    assert cross_source["matched_comcat_event_counted_again"] is False
    assert cross_source["candidate_pairs_require_all_three_thresholds"] is True
    assert cross_source["missing_or_nonfinite_magnitude_action"] == "not_a_candidate_pair"
    assert cross_source["distance_implementation"] == "pyproj.Geod_ellps_WGS84"
    assert cross_source["assignment_algorithm"] == "deterministic_greedy_one_to_one"
    assert cross_source["candidate_order"] == [
        "absolute_origin_time_delta_ascending",
        "WGS84_geodesic_distance_ascending",
        "absolute_magnitude_delta_ascending",
        "unsigned_UTF8_comcat_event_id_ascending",
        "unsigned_UTF8_local_event_id_ascending",
    ]
    assert cross_source["used_local_or_comcat_event_cannot_match_again"] is True
    assert comcat["revision"]["later_revision_backfill_into_prior_issue_forbidden"] is True
    assert comcat["coverage_diagnostics"]["arbitrary_China_M4_coverage_threshold_forbidden"]
    assert comcat["coverage_diagnostics"]["target_conditioned_source_switch_forbidden"]

    assert cohort["definition_contains_future_event_ids_coordinates_or_counts"] is False
    assert cohort["realized_target_set_bound_only_by_evaluation_freeze"] is True
    assert cohort["maturity"]["retry_offsets_hours"] == [0, 6, 24, 72, 168]
    revision = cohort["formal_freeze_revision_policy"]
    assert revision["preferred_magnitude_field"] == "preferred_mag_numeric"
    assert revision["magnitude_must_be_finite"] is True
    assert revision["missing_or_nonfinite_magnitude_action"] == "exclude"
    assert revision["M5_6_membership_rule"] == "5.0_lte_preferred_mag_lt_6.0"
    assert revision["location_basis"] == (
        "preferred_location_in_same_single_formal_freeze_response"
    )
    assert revision["identity_basis"] == (
        "preferred_and_associated_ids_in_same_single_formal_freeze_response"
    )
    assert revision["origin_time_field"] == "preferred_origin_time_utc"
    assert revision["origin_time_basis"] == (
        "preferred_origin_in_same_single_formal_freeze_response"
    )
    assert revision["origin_time_window_membership_rule"] == (
        "strict_T_lt_origin_time_lte_T_plus_h"
    )
    assert revision["origin_time_cross_boundary_change_action"] == (
        "append_TruthRevisionRecord_and_reassign_membership"
    )
    assert revision["revision_cutoff"] == "evaluation_freeze_source_snapshot"


def test_statistical_publication_and_compute_gates_are_exact() -> None:
    config = _load_config()
    evaluation = config["evaluation"]
    inference = evaluation["simultaneous_inference"]
    alarm = evaluation["alarm_area"]

    assert alarm["target_budget_km2"] == 600000
    assert alarm["selection"] == "complete_prefix_with_cumulative_exact_area_lte_budget"
    assert alarm["partial_cell_skip_oversized_cell_or_budget_expansion"] == "forbidden"
    assert alarm["actual_area_must_be_recorded"] is True
    assert alarm["ordered_full_ranking_sha256_required"] is True
    assert alarm["selected_complete_prefix_sha256_required"] is True
    assert alarm["next_cell_would_exceed_budget_or_domain_exhausted_required"] is True
    assert evaluation["formal_exposures"]["selection"] == (
        "earliest_issue_greedy_nonoverlap_separately_within_each_horizon"
    )
    assert evaluation["formal_exposures"]["zero_event_exposures_retained"] is True
    assert (
        evaluation["sample_gate"]["horizon_union_minimum_unique_deduplicated_supported_M5_6_events"]
        == 20
    )
    assert evaluation["sample_gate"]["minimum_unique_target_clusters"] == 10
    assert evaluation["sample_gate"]["every_horizon_must_be_evaluable"] is True
    assert evaluation["information_gain"]["target_scope"] == ("frozen_G1_LS_supported_targets_only")
    assert evaluation["information_gain"]["event_density_operator"] == (
        "continuous_boundary_normalized_75km_KDE_at_projected_event_coordinate"
    )
    assert evaluation["strict_recall"]["target_scope"] == (
        "full_study_area_all_unique_M5_6_targets"
    )
    assert evaluation["strict_recall"]["unsupported_target_action"] == (
        "common_not_hit_for_P0_P1_PP"
    )
    assert evaluation["strict_recall"]["hit_rule"] == (
        "target_projected_25km_cell_is_in_complete_alarm_prefix"
    )
    assert evaluation["macro_aggregation"]["weights"] == [1 / 3, 1 / 3, 1 / 3]
    assert evaluation["target_clusters"]["temporal_link_days"] == 30
    assert evaluation["target_clusters"]["spatial_link_km"] == 75
    assert evaluation["bootstrap"]["replications"] == 2000
    assert evaluation["bootstrap"]["generator"] == "numpy.random.PCG64"
    assert evaluation["bootstrap"]["root_seed"] == 147
    assert inference["endpoint_count"] == 4
    assert inference["lower_quantile"] == 0.05 / (2 * 4)
    assert inference["upper_quantile"] == 1 - 0.05 / (2 * 4)
    assert len(inference["endpoints"]) == 4
    assert evaluation["pass_gate"]["all_four_familywise_lower_bounds_gt_zero"] is True
    assert evaluation["pass_gate"]["P1_minus_P0_macro_recall_point_gain_minimum"] == 0.05
    assert evaluation["regional_robustness"]["region_count"] == 39
    assert evaluation["formal_looks"]["count_basis"] == "on_time_issue_sequence_only"
    assert evaluation["formal_looks"]["allowed_on_time_issue_counts"] == [52, 104]
    assert evaluation["formal_looks"]["evaluation_record_phases"] == [
        "input_freeze",
        "result_seal",
    ]
    assert evaluation["formal_looks"]["maximum_confirmatory_result_seals"] == 1
    assert evaluation["formal_looks"]["second_confirmatory_look_forbidden"] is True
    assert (
        evaluation["formal_looks"]["intermediate_confirmatory_effect_display_or_testing_forbidden"]
        is True
    )
    assert config["calendar"]["scheduled_issue_sequence"]["maximum"] == 130

    publication = config["storage_and_publication"]
    assert publication["local_restricted"]["git_ignored_required"] is True
    assert publication["local_restricted"]["tracked_by_git_forbidden"] is True
    assert publication["public"]["aggregate_spatial_publication_authorized_now"] is False
    assert publication["public"]["aggregate_layer_requires_explicit_license"] is True
    assert publication["public"]["exact_rows_coordinates_event_ids_and_local_paths_forbidden"]

    compute = config["compute"]
    assert compute["physical_core_reserve"] == 2
    assert compute["max_workers_formula"] == "min(8, physical_cores - 2)"
    assert compute["blas_threads"] == 1
    assert compute["gpu"]["enabled_by_default"] is False
    assert compute["gpu"]["future_use"] == "numerically_equivalent_acceleration_only"


def test_record_schema_is_a_strict_five_record_union() -> None:
    schema = _load_schema()
    definitions = schema["$defs"]
    expected = {
        "TargetCohortDefinition",
        "IssueInputSnapshotRecord",
        "MatureTruthSnapshotRecord",
        "TruthRevisionRecord",
        "EvaluationFreezeRecord",
    }

    refs = {entry["$ref"].rsplit("/", 1)[-1] for entry in schema["oneOf"]}
    assert refs == expected
    for name in expected:
        definition = definitions[name]
        assert definition["type"] == "object"
        assert definition["additionalProperties"] is False
        assert "content_sha256" in definition["required"]
    for definition in definitions.values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False

    assert definitions["Sha256"]["pattern"] == SHA256_PATTERN
    assert definitions["TargetCohortDefinition"]["properties"]["protocol_version"]["const"] == (
        "0.2.4"
    )
    assert definitions["TargetCohortDefinition"]["properties"]["truth_retry_offsets_hours"][
        "const"
    ] == [0, 6, 24, 72, 168]
    assert definitions["SourceQueryPolicy"]["properties"]["minmagnitude"]["const"] == 3.9
    assert (
        definitions["SourceQueryPolicy"]["properties"]["local_model_magnitude_lower_inclusive"][
            "const"
        ]
        == 4.0
    )
    assert definitions["RemoteTimestampPolicy"]["properties"]["ordered_authorities"]["const"] == [
        "http://timestamp.digicert.com",
        "http://timestamp.sectigo.com",
    ]
    assert definitions["SourceAcquisition"]["properties"]["http_status"]["enum"] == [200, 204]


def test_issue_schema_separates_on_time_from_missed_and_binds_prediction_seal() -> None:
    definitions = _load_schema()["$defs"]
    issue = definitions["IssueInputSnapshotRecord"]
    issue_required = set(issue["required"])
    assert {"scheduled_issue_sequence", "on_time_issue_sequence"} <= issue_required
    assert "issue_sequence" not in issue_required
    branches = {
        clause["if"]["properties"]["status"]["const"]: clause["then"]["properties"]
        for clause in issue["allOf"]
        if set(clause["if"]["properties"]) == {"status"}
    }

    assert branches["on_time"]["source_snapshot"]["$ref"] == "#/$defs/SourceSnapshot"
    assert branches["on_time"]["prediction_seal"]["$ref"] == "#/$defs/PredictionSeal"
    assert branches["on_time"]["prediction_generated"]["const"] is True
    assert branches["on_time"]["prediction_installed"]["const"] is True
    assert branches["on_time"]["on_time_issue_sequence"]["maximum"] == 104
    assert branches["missed_issue"]["source_snapshot"]["type"] == "null"
    assert branches["missed_issue"]["prediction_seal"]["type"] == "null"
    assert branches["missed_issue"]["prediction_installed"]["const"] is False
    assert branches["missed_issue"]["on_time_issue_sequence"]["type"] == "null"
    assert "timestamp_attempt_evidence" in issue["required"]
    assert branches["on_time"]["timestamp_attempt_evidence"]["$ref"] == (
        "#/$defs/VerifiedTsaAttemptEvidence"
    )
    assert branches["missed_issue"]["timestamp_attempt_evidence"]["$ref"] == (
        "#/$defs/FailedOrNotAttemptedTsaEvidence"
    )
    timestamp_failure_clause = issue["allOf"][2]
    assert timestamp_failure_clause["if"]["properties"]["failure_code"]["const"] == (
        "timestamp_failure"
    )
    timestamp_failure_properties = timestamp_failure_clause["then"]["properties"]
    assert timestamp_failure_properties["prediction_generated"]["const"] is True
    assert timestamp_failure_properties["failed_candidate_on_time_core"]["$ref"] == (
        "#/$defs/FailedCandidateOnTimeCoreIdentity"
    )
    assert (
        timestamp_failure_clause["else"]["then"]["properties"]["prediction_generated"]["const"]
        is False
    )

    prediction_required = set(definitions["PredictionSeal"]["required"])
    assert {
        "source_snapshot_sha256",
        "issue_time_utc",
        "query_end_utc",
        "valid_from_utc",
        "protocol_config_sha256",
        "source_query_policy_sha256",
        "deduplication_policy_sha256",
        "revision_policy_sha256",
        "model_formula_identity",
        "P0_event_set_sha256",
        "R30_event_set_sha256",
        "RP30_event_set_sha256",
        "P0",
        "P1",
        "PP",
        "forecast_bundle_manifest_sha256",
        "prediction_seal_sha256",
    } <= prediction_required
    assert "remote_timestamp" not in prediction_required
    assert "timestamp_attempt_evidence" not in prediction_required
    assert "remote_timestamp" not in definitions["PredictionSeal"]["properties"]
    assert "timestamp_attempt_evidence" not in definitions["PredictionSeal"]["properties"]
    assert "visualized_forecast_bundle_manifest_sha256" in set(
        definitions["VisualizationEvidence"]["required"]
    )

    assert "previous_definition_sha256" in definitions["TargetCohortDefinition"]["required"]
    assert "previous_issue_record_sha256" in issue["required"]
    assert "previous_truth_record_sha256" in definitions["MatureTruthSnapshotRecord"]["required"]
    assert "previous_truth_record_sha256" in definitions["TruthRevisionRecord"]["required"]
    assert "previous_evaluation_freeze_sha256" in definitions["EvaluationFreezeRecord"]["required"]
    mature_statuses = definitions["MatureTruthSnapshotRecord"]["properties"]["status"]["enum"]
    assert mature_statuses == ["mature_truth_sealed", "truth_snapshot_unavailable"]

    evaluation = definitions["EvaluationFreezeRecord"]
    evaluation_required = set(evaluation["required"])
    assert {
        "checkpoint_number",
        "trigger_on_time_issue_count",
        "sample_gate_met",
        "bootstrap_preflight",
    } <= evaluation_required
    assert "look_number" not in evaluation_required
    assert "trigger_issue_count" not in evaluation_required
    horizon_prefixes = evaluation["properties"]["horizon_evaluability"]["prefixItems"]
    assert [
        item["allOf"][1]["properties"]["horizon_days"]["const"] for item in horizon_prefixes
    ] == [7, 30, 90]


def test_schema_passes_draft_2020_12_metaschema_and_uses_format_checker() -> None:
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)

    invalid_metaschema = copy.deepcopy(schema)
    invalid_metaschema["$defs"]["TableArtifactIdentity"]["properties"]["row_count"][
        "type"
    ] = "not_a_json_schema_type"
    with pytest.raises(SchemaError):
        Draft202012Validator.check_schema(invalid_metaschema)

    valid = _missed_issue_record()
    _assert_valid_def("IssueInputSnapshotRecord", valid)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(valid)

    invalid = copy.deepcopy(valid)
    invalid["issue_time_utc"] = "2026-13-40T25:61:61Z"
    _assert_invalid_def("IssueInputSnapshotRecord", invalid)


def test_artifact_registry_hashes_and_typed_identities_are_mechanical() -> None:
    config = _load_config()
    definitions = _load_schema()["$defs"]
    artifact_registry = config["artifact_profile_registry"]
    table_identity = definitions["TableArtifactIdentity"]

    assert set(table_identity["required"]) == TABLE_ARTIFACT_IDENTITY_FIELDS
    assert set(table_identity["properties"]) == TABLE_ARTIFACT_IDENTITY_FIELDS
    assert table_identity["additionalProperties"] is False

    for identity_spec in artifact_registry["identity_schemas"].values():
        schema_name = cast(str, identity_spec["schema_ref"]).rsplit("/", 1)[-1]
        assert schema_name in definitions
        assert (
            identity_spec["expected_schema_sha256"]
            == _schema_definition_sha256(schema_name)
        )

    for registry_group in (
        artifact_registry["embedded_manifests"],
        artifact_registry["manifests"],
    ):
        for manifest_spec in registry_group.values():
            schema_name = cast(str, manifest_spec["schema_ref"]).rsplit("/", 1)[-1]
            assert schema_name in definitions
            assert (
                manifest_spec["expected_schema_sha256"]
                == _schema_definition_sha256(schema_name)
            )

    for table_name, table_spec in artifact_registry["tables"].items():
        row_schema_name = cast(str, table_spec["row_schema_ref"]).rsplit("/", 1)[
            -1
        ]
        assert (
            table_spec["expected_row_schema_sha256"]
            == _schema_definition_sha256(row_schema_name)
        )
        assert table_spec["expected_sort_order_sha256"] == hashlib.sha256(
            canonical_json_bytes(table_spec["sort_spec"])
        ).hexdigest()

        table_roles = table_spec.get("table_roles")
        roles = table_roles or [table_spec["table_role"]]
        identity_name = cast(str, table_spec["identity_schema_ref"]).rsplit("/", 1)[
            -1
        ]
        for table_role in roles:
            artifact = _table_artifact(
                f"registry-{table_name}-{table_role}",
                table_name=table_name,
                table_role=cast(str, table_role),
                row_count=1,
            )
            _assert_valid_def(identity_name, artifact)
            assert artifact["row_schema_ref"] == table_spec["row_schema_ref"]
            assert artifact["schema_sha256"] == table_spec[
                "expected_row_schema_sha256"
            ]
            assert artifact["sort_profile"] == table_spec["sort_profile"]
            assert artifact["sort_order_sha256"] == table_spec[
                "expected_sort_order_sha256"
            ]

        wrong_schema_hash = copy.deepcopy(artifact)
        wrong_schema_hash["schema_sha256"] = _sha256(f"wrong-{table_name}-schema")
        _assert_invalid_def(identity_name, wrong_schema_hash)

        wrong_row_schema = copy.deepcopy(artifact)
        wrong_row_schema["row_schema_ref"] = "#/$defs/TableArtifactIdentity"
        _assert_invalid_def(identity_name, wrong_row_schema)

    formal_registry = config["target_cohort"]["formal_freeze_source_manifest"][
        "derived_table_registry"
    ]
    formal_identity_names = {
        "normalized_rows": "FormalNormalizedRowsArtifactIdentity",
        "deduplicated_rows": "FormalDeduplicatedRowsArtifactIdentity",
        "preferred_field_rows": "FormalPreferredFieldRowsArtifactIdentity",
        "window_membership_rows": "FormalWindowMembershipRowsArtifactIdentity",
        "formal_window_target_bindings": (
            "FormalWindowTargetBindingRowsArtifactIdentity"
        ),
    }
    for table_name, table_spec in formal_registry["tables"].items():
        row_schema_name = cast(str, table_spec["row_schema_ref"]).rsplit("/", 1)[
            -1
        ]
        assert (
            table_spec["expected_row_schema_sha256"]
            == _schema_definition_sha256(row_schema_name)
        )
        assert table_spec["expected_sort_order_sha256"] == hashlib.sha256(
            canonical_json_bytes(table_spec["sort_spec"])
        ).hexdigest()
        artifact = _table_artifact(
            f"formal-registry-{table_name}",
            table_name=table_name,
            row_count=1,
        )
        _assert_valid_def(formal_identity_names[table_name], artifact)

    for schema_name, expected_sha256 in EXPECTED_SCHEMA_HASHES.items():
        assert _schema_definition_sha256(schema_name) == expected_sha256

    assert (
        formal_registry["tables"]["preferred_field_rows"][
            "expected_row_schema_sha256"
        ]
        == EXPECTED_SCHEMA_HASHES["FormalPreferredFieldRow"]
    )
    assert (
        artifact_registry["tables"]["scientific_target_rows"][
            "expected_row_schema_sha256"
        ]
        == EXPECTED_SCHEMA_HASHES["FormalScientificTargetRow"]
    )
    assert (
        artifact_registry["manifests"]["effect_rows"]["expected_schema_sha256"]
        == EXPECTED_SCHEMA_HASHES["EffectRowsManifest"]
    )
    assert (
        artifact_registry["manifests"]["result_bundle"]["expected_schema_sha256"]
        == EXPECTED_SCHEMA_HASHES["ResultBundleManifest"]
    )

    tampered_registry = copy.deepcopy(artifact_registry)
    tampered_registry["identity_schemas"]["TableArtifactIdentity"][
        "expected_schema_sha256"
    ] = _sha256("wrong-table-identity-schema")
    assert (
        tampered_registry["identity_schemas"]["TableArtifactIdentity"][
            "expected_schema_sha256"
        ]
        != _schema_definition_sha256("TableArtifactIdentity")
    )
    tampered_registry["manifests"]["effect_rows"]["expected_schema_sha256"] = (
        _sha256("wrong-effect-manifest-schema")
    )
    assert (
        tampered_registry["manifests"]["effect_rows"]["expected_schema_sha256"]
        != _schema_definition_sha256("EffectRowsManifest")
    )


def test_source_truth_event_and_target_identities_require_typed_tables() -> None:
    source_snapshot = _source_snapshot()
    truth_snapshot = _truth_snapshot()
    target_set = _target_set_identity()

    _assert_valid_def("SourceSnapshot", source_snapshot)
    _assert_valid_def("TruthSnapshot", truth_snapshot)
    _assert_valid_def("TargetSetIdentity", target_set)
    for component_role in ("P0", "R30", "RP30"):
        _assert_valid_def(
            f"{component_role}EventSetIdentity",
            source_snapshot[f"{component_role}_event_set"],
        )

    missing_target_rows = copy.deepcopy(target_set)
    missing_target_rows.pop("target_rows")
    _assert_invalid_def("TargetSetIdentity", missing_target_rows)

    wrong_target_role = copy.deepcopy(target_set)
    wrong_target_role["target_rows"]["table_role"] = "truth_normalized_rows"
    _assert_invalid_def("TargetSetIdentity", wrong_target_role)

    missing_cutover_rows = copy.deepcopy(source_snapshot)
    missing_cutover_rows.pop("cutover_cross_source_match_rows")
    _assert_invalid_def("SourceSnapshot", missing_cutover_rows)

    wrong_event_role = copy.deepcopy(source_snapshot["P0_event_set"])
    wrong_event_role["component_role"] = "R30"
    _assert_invalid_def("P0EventSetIdentity", wrong_event_role)

    missing_truth_window_rows = copy.deepcopy(truth_snapshot)
    missing_truth_window_rows.pop("window_membership_rows")
    _assert_invalid_def("TruthSnapshot", missing_truth_window_rows)


def test_effect_and_result_manifests_require_complete_typed_slots() -> None:
    input_freeze_sha256 = _sha256("manifest-input-freeze")
    effect_rows_manifest = _effect_rows_manifest(input_freeze_sha256)
    valid_result_bundle = _result_bundle_manifest(input_freeze_sha256)

    _assert_valid_def("EffectRowsManifest", effect_rows_manifest)
    _assert_valid_def("ResultBundleManifest", valid_result_bundle)
    assert set(
        cast(Mapping[str, object], effect_rows_manifest["point_contribution_rows"])
    ) == set(ENDPOINT_IDS)
    assert set(
        cast(Mapping[str, object], valid_result_bundle["bootstrap_distribution_rows"])
    ) == set(ENDPOINT_IDS)

    missing_effect_table = copy.deepcopy(effect_rows_manifest)
    missing_effect_table.pop("cluster_membership_rows")
    _assert_invalid_def("EffectRowsManifest", missing_effect_table)

    for failure_stage in (
        "effect_rows_open",
        "alarm_area_comparison",
        "bootstrap",
        "endpoint_evaluation",
        "robustness_evaluation",
        "result_bundle_install",
        "result_seal",
    ):
        invalid_result_bundle = _result_bundle_manifest(
            input_freeze_sha256,
            failure_stage=failure_stage,
        )
        _assert_valid_def("ResultBundleManifest", invalid_result_bundle)

    no_audit_evidence = _result_bundle_manifest(
        input_freeze_sha256,
        failure_stage="effect_rows_open",
    )
    no_audit_evidence["available_audit_artifact_sha256"] = []
    _assert_invalid_def("ResultBundleManifest", no_audit_evidence)

    premature_effect_slot = _result_bundle_manifest(
        input_freeze_sha256,
        failure_stage="effect_rows_open",
    )
    premature_effect_slot["effect_rows_manifest_sha256"] = _sha256(
        "premature-effect-slot"
    )
    _assert_invalid_def("ResultBundleManifest", premature_effect_slot)


def test_forecast_schema_enforces_600k_complete_prefix_and_actual_area() -> None:
    forecast = _forecast_identity()
    _assert_valid_def("ForecastIdentity", forecast)

    over_budget = copy.deepcopy(forecast)
    over_budget["actual_alarm_area_km2"] = 600000.1
    over_budget["remaining_budget_km2"] = 0
    _assert_invalid_def("ForecastIdentity", over_budget)

    wrong_budget = copy.deepcopy(forecast)
    wrong_budget["alarm_area_budget_km2"] = 600001
    _assert_invalid_def("ForecastIdentity", wrong_budget)

    partial_cell = copy.deepcopy(forecast)
    partial_cell["complete_cells_only"] = False
    _assert_invalid_def("ForecastIdentity", partial_cell)

    nonmaximal_prefix = copy.deepcopy(forecast)
    nonmaximal_prefix["prefix_maximal_under_budget"] = False
    _assert_invalid_def("ForecastIdentity", nonmaximal_prefix)


def test_source_roles_and_204_empty_success_are_schema_enforced() -> None:
    prospective = _source_acquisition(role="prospective_increment")
    mature_truth = _source_acquisition(role="temporally_independent_mature_truth")
    formal_freeze = _source_acquisition(
        role="formal_freeze_full_cohort",
        http_status=200,
        query_count=1,
        feature_count=1,
        response_body_byte_count=100,
    )
    _assert_valid_def("ProspectiveSourceAcquisition", prospective)
    _assert_valid_def("MatureTruthSourceAcquisition", mature_truth)
    _assert_valid_def("FormalFreezeSourceAcquisition", formal_freeze)
    assert prospective["geojson_parse_verified"] is False

    false_parse_claim_for_empty_204 = copy.deepcopy(prospective)
    false_parse_claim_for_empty_204["geojson_parse_verified"] = True
    _assert_invalid_def(
        "ProspectiveSourceAcquisition",
        false_parse_claim_for_empty_204,
    )

    nonempty_200 = _source_acquisition(
        role="prospective_increment",
        http_status=200,
        query_count=1,
        feature_count=1,
        response_body_byte_count=100,
    )
    _assert_valid_def("ProspectiveSourceAcquisition", nonempty_200)
    assert nonempty_200["geojson_parse_verified"] is True
    unparsed_200 = copy.deepcopy(nonempty_200)
    unparsed_200["geojson_parse_verified"] = False
    _assert_invalid_def("ProspectiveSourceAcquisition", unparsed_200)

    empty_count = _count_preflight(
        start="2026-07-09T04:25:56Z",
        end="2026-09-09T15:45:00Z",
        label="empty-204-count",
        parsed_count=0,
    )
    empty_count.update(
        {
            "http_status": 204,
            "response_content_type": None,
            "response_body_byte_count": 0,
            "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
            "geojson_parse_verified": False,
        }
    )
    _assert_valid_def("FdsnCountPreflightEvidence", empty_count)
    empty_count["geojson_parse_verified"] = True
    _assert_invalid_def("FdsnCountPreflightEvidence", empty_count)

    wrong_role = copy.deepcopy(mature_truth)
    wrong_role["source_role"] = "prospective_increment"
    _assert_invalid_def("MatureTruthSourceAcquisition", wrong_role)

    nonempty_204 = _source_acquisition(
        role="temporally_independent_mature_truth",
        query_count=1,
        feature_count=1,
        response_body_byte_count=100,
    )
    _assert_invalid_def("MatureTruthSourceAcquisition", nonempty_204)

    invalid_truth_window_reuse = copy.deepcopy(mature_truth)
    invalid_truth_window_reuse["query_start_utc"] = "2026-07-09T04:25:56Z"
    invalid_truth_window_reuse["local_starttime_filter"] = (
        "origin_time_utc_strictly_greater_than_cutover"
    )
    _assert_invalid_def("MatureTruthSourceAcquisition", invalid_truth_window_reuse)


def test_truth_retry_schema_accepts_only_first_success_or_all_five_failures() -> None:
    for selected_index in range(len(TRUTH_RETRY_OFFSETS_HOURS)):
        attempts = [
            _truth_attempt(index=index, succeeded=index == selected_index)
            for index in range(selected_index + 1)
        ]
        _assert_valid_def("SuccessfulMatureTruthAttemptSequence", attempts)
        assert not _truth_attempt_semantic_errors(attempts, expect_success=True)

    all_failed = [
        _truth_attempt(index=index, succeeded=False)
        for index in range(len(TRUTH_RETRY_OFFSETS_HOURS))
    ]
    _assert_valid_def("FailedMatureTruthAttemptSequence", all_failed)
    assert not _truth_attempt_semantic_errors(all_failed, expect_success=False)

    later_attempt_after_success = [
        _truth_attempt(index=0, succeeded=True),
        _truth_attempt(index=1, succeeded=True),
    ]
    _assert_invalid_def("SuccessfulMatureTruthAttemptSequence", later_attempt_after_success)
    assert "first_success_must_be_last_attempt" in _truth_attempt_semantic_errors(
        later_attempt_after_success,
        expect_success=True,
    )

    wrong_order = [
        _truth_attempt(index=0, succeeded=False),
        _truth_attempt(index=2, succeeded=True),
    ]
    _assert_invalid_def("SuccessfulMatureTruthAttemptSequence", wrong_order)
    assert "retry_offset_wrong_order" in _truth_attempt_semantic_errors(
        wrong_order,
        expect_success=True,
    )

    wrong_role = [_truth_attempt(index=0, succeeded=True)]
    wrong_role[0]["query_role"] = "truth_revision"
    _assert_invalid_def("SuccessfulMatureTruthAttemptSequence", wrong_role)
    assert "wrong_truth_query_role" in _truth_attempt_semantic_errors(
        wrong_role,
        expect_success=True,
    )

    only_four_failures = all_failed[:-1]
    _assert_invalid_def("FailedMatureTruthAttemptSequence", only_four_failures)
    assert "unavailable_requires_all_five_failures" in _truth_attempt_semantic_errors(
        only_four_failures,
        expect_success=False,
    )


def test_rfc3161_proof_is_top_level_and_uses_exact_three_field_preimage() -> None:
    definitions = _load_schema()["$defs"]
    policy = definitions["RemoteTimestampPolicy"]
    assert policy["properties"]["preimage_excluded_top_level_fields"]["const"] == list(
        RFC3161_EXCLUDED_TOP_LEVEL_FIELDS
    )
    assert policy["properties"]["nested_timestamp_proof_fields_forbidden"]["const"] is True
    assert definitions["RecordContentHash"]["description"].endswith(
        "the final hash therefore includes timestamp_attempt_evidence and remote_timestamp."
    )

    payload: dict[str, object] = {
        "record_type": "EvaluationFreezeRecord",
        "stage_id": "Stage2P-1A",
        "payload": {"frozen_input": _sha256("frozen-input")},
    }
    record = _seal_timestamped_record(
        payload,
        subject_type="EvaluationFreezeRecord",
    )
    _assert_valid_def("EvaluationTimestampProof", record["remote_timestamp"])
    _assert_valid_def("VerifiedTsaAttemptEvidence", record["timestamp_attempt_evidence"])
    assert not _timestamp_semantic_errors(record)
    assert "attempt_completed_at_utc" in definitions["TsaAttemptEvidence"]["required"]

    missing_completion = copy.deepcopy(record["timestamp_attempt_evidence"][0])
    missing_completion.pop("attempt_completed_at_utc")
    _assert_invalid_def("TsaAttemptEvidence", missing_completion)

    delayed_completion = copy.deepcopy(record)
    delayed_completion["timestamp_attempt_evidence"][0][  # type: ignore[index]
        "attempt_completed_at_utc"
    ] = "2026-09-09T15:59:00Z"
    assert "response_received_must_equal_attempt_completed" in _timestamp_semantic_errors(
        delayed_completion
    )

    manual_core = dict(record)
    for field_name in RFC3161_EXCLUDED_TOP_LEVEL_FIELDS:
        manual_core.pop(field_name)
    assert (
        record["remote_timestamp"]["preimage_sha256"]
        == hashlib.sha256(  # type: ignore[index]
            canonical_json_bytes(manual_core)
        ).hexdigest()
    )

    extra_omission = dict(manual_core)
    extra_omission.pop("stage_id")
    assert (
        hashlib.sha256(canonical_json_bytes(extra_omission)).hexdigest()
        != (
            record["remote_timestamp"]["preimage_sha256"]  # type: ignore[index]
        )
    )

    tampered = copy.deepcopy(record)
    tampered["payload"]["frozen_input"] = _sha256("tampered")  # type: ignore[index]
    assert {"preimage_sha256_mismatch", "content_sha256_mismatch"} <= set(
        _timestamp_semantic_errors(tampered)
    )

    wrong_selected_response = copy.deepcopy(record)
    wrong_selected_response["remote_timestamp"]["selected_response_sha256"] = _sha256(  # type: ignore[index]
        "wrong-response"
    )
    assert "selected_response_sha256_mismatch" in _timestamp_semantic_errors(
        wrong_selected_response
    )

    nested_proof = copy.deepcopy(record)
    nested_proof["payload"]["remote_timestamp"] = copy.deepcopy(  # type: ignore[index]
        record["remote_timestamp"]
    )
    assert "nested_timestamp_proof_forbidden" in _timestamp_semantic_errors(nested_proof)

    wrong_attempt_order = _seal_timestamped_record(
        payload,
        subject_type="EvaluationFreezeRecord",
        selected_attempt_index=1,
    )
    wrong_attempt_order["timestamp_attempt_evidence"].reverse()  # type: ignore[union-attr]
    _assert_invalid_def(
        "VerifiedTsaAttemptEvidence",
        wrong_attempt_order["timestamp_attempt_evidence"],
    )
    assert _timestamp_semantic_errors(wrong_attempt_order)

    overlapping_authorities = _seal_timestamped_record(
        payload,
        subject_type="EvaluationFreezeRecord",
        selected_attempt_index=1,
    )
    overlapping_authorities["timestamp_attempt_evidence"][0][  # type: ignore[index]
        "attempt_completed_at_utc"
    ] = "2026-09-09T15:55:30Z"
    assert "tsa_authority_attempts_overlap_or_reverse" in _timestamp_semantic_errors(
        overlapping_authorities
    )


def test_issue_schema_and_semantic_oracle_reject_bad_weekday_Q_chain_and_tamper() -> None:
    first = _missed_issue_record()
    _assert_valid_def("IssueInputSnapshotRecord", first)
    assert not _issue_semantic_errors(
        first,
        previous_record=None,
        previous_on_time_sequence=0,
    )

    timestamp_failure = _timestamp_failure_missed_issue_record()
    _assert_valid_def("IssueInputSnapshotRecord", timestamp_failure)
    assert not _issue_semantic_errors(
        timestamp_failure,
        previous_record=None,
        previous_on_time_sequence=0,
    )

    wrong_candidate_preimage = copy.deepcopy(timestamp_failure)
    wrong_candidate_preimage.pop("content_sha256")
    wrong_candidate_preimage["timestamp_attempt_evidence"][0][  # type: ignore[index]
        "attempt_preimage_sha256"
    ] = _sha256("wrong-candidate-preimage")
    wrong_candidate_preimage = _seal_missed_audit_record(
        wrong_candidate_preimage,
        deadline_utc=cast(str, wrong_candidate_preimage["issue_time_utc"]),
    )
    assert "candidate_tsa_attempt_preimage_mismatch" in _issue_semantic_errors(
        wrong_candidate_preimage,
        previous_record=None,
        previous_on_time_sequence=0,
    )

    delayed_missed_audit_core = copy.deepcopy(timestamp_failure)
    delayed_missed_audit_core.pop("content_sha256")
    delayed_missed_audit_core["record_core_frozen_at_utc"] = "2026-09-09T15:57:00Z"
    delayed_missed_audit_core = _seal_missed_audit_record(
        delayed_missed_audit_core,
        deadline_utc=cast(str, delayed_missed_audit_core["issue_time_utc"]),
    )
    assert "missed_audit_core_must_freeze_by_T_minus_4_minutes" in (
        _issue_semantic_errors(
            delayed_missed_audit_core,
            previous_record=None,
            previous_on_time_sequence=0,
        )
    )

    second = _missed_issue_record(
        scheduled_sequence=2,
        local_time="2026-09-17T00:00:00+08:00",
        previous_sha256=cast(str, first["content_sha256"]),
    )
    _assert_valid_def("IssueInputSnapshotRecord", second)
    assert not _issue_semantic_errors(
        second,
        previous_record=first,
        previous_on_time_sequence=0,
    )

    friday = _missed_issue_record(local_time="2026-09-11T00:00:00+08:00")
    _assert_valid_def("IssueInputSnapshotRecord", friday)
    assert "issue_not_Thursday_midnight_Asia_Shanghai" in _issue_semantic_errors(
        friday,
        previous_record=None,
        previous_on_time_sequence=0,
    )

    wrong_q = copy.deepcopy(first)
    wrong_q["query_end_utc"] = "2026-09-09T15:44:59Z"
    wrong_q = with_content_sha256(wrong_q)
    _assert_valid_def("IssueInputSnapshotRecord", wrong_q)
    assert "Q_must_equal_T_minus_15_minutes" in _issue_semantic_errors(
        wrong_q,
        previous_record=None,
        previous_on_time_sequence=0,
    )

    broken_sequence = copy.deepcopy(second)
    broken_sequence["scheduled_issue_sequence"] = 3
    broken_sequence["previous_issue_record_sha256"] = _sha256("wrong-previous")
    broken_sequence = with_content_sha256(broken_sequence)
    errors = _issue_semantic_errors(
        broken_sequence,
        previous_record=first,
        previous_on_time_sequence=0,
    )
    assert {
        "scheduled_issue_sequence_not_contiguous",
        "previous_issue_hash_chain_broken",
    } <= set(errors)

    bad_on_time_sequence = copy.deepcopy(first)
    bad_on_time_sequence["status"] = "on_time"
    bad_on_time_sequence["on_time_issue_sequence"] = 2
    bad_on_time_sequence = with_content_sha256(bad_on_time_sequence)
    assert "on_time_issue_sequence_not_contiguous" in _issue_semantic_errors(
        bad_on_time_sequence,
        previous_record=None,
        previous_on_time_sequence=0,
    )

    tampered = copy.deepcopy(first)
    tampered["failure_code"] = "different_failure"
    assert "content_sha256_mismatch" in _issue_semantic_errors(
        tampered,
        previous_record=None,
        previous_on_time_sequence=0,
    )

    skipped_week = _missed_issue_record(
        scheduled_sequence=2,
        local_time="2026-09-24T00:00:00+08:00",
        previous_sha256=cast(str, first["content_sha256"]),
    )
    assert "issue_T_not_exactly_previous_T_plus_7_days" in _issue_semantic_errors(
        skipped_week,
        previous_record=first,
        previous_on_time_sequence=0,
    )

    duplicate_week = _missed_issue_record(
        scheduled_sequence=2,
        local_time="2026-09-10T00:00:00+08:00",
        previous_sha256=cast(str, first["content_sha256"]),
    )
    assert "issue_T_not_exactly_previous_T_plus_7_days" in _issue_semantic_errors(
        duplicate_week,
        previous_record=first,
        previous_on_time_sequence=0,
    )

    wrong_issue_id = copy.deepcopy(first)
    wrong_issue_id["issue_id"] = "stage2p-issue-20991231T000000+0800"
    wrong_issue_id = with_content_sha256(wrong_issue_id)
    assert "issue_id_not_derived_from_local_T" in _issue_semantic_errors(
        wrong_issue_id,
        previous_record=None,
        previous_on_time_sequence=0,
    )


def test_evaluation_schema_enforces_52_104_130_and_ten_cluster_gate() -> None:
    first_continue = _evaluation_input_freeze(trigger=52, sample_gate_met=False)
    _assert_valid_def("EvaluationFreezeRecord", first_continue)
    assert first_continue["selected_exposure_manifest_union_sha256"] == (
        _canonical_file_sha256(
            cast(
                Mapping[str, object],
                first_continue["selected_exposure_manifest"],
            )
        )
    )
    assert first_continue["alarm_area_manifest_sha256"] == _canonical_file_sha256(
        cast(Mapping[str, object], first_continue["alarm_area_manifest"])
    )

    first_authorized = _evaluation_input_freeze(trigger=52, sample_gate_met=True)
    _assert_valid_def("EvaluationFreezeRecord", first_authorized)
    assert first_authorized["unique_target_cluster_count"] == 10
    result = _evaluation_result_seal(first_authorized)
    _assert_valid_def("EvaluationFreezeRecord", result)

    final_authorized = _evaluation_input_freeze(trigger=104, sample_gate_met=True)
    _assert_valid_def("EvaluationFreezeRecord", final_authorized)

    scheduled_cap = _evaluation_input_freeze(
        trigger=130,
        sample_gate_met=False,
        scheduled_issue_count=130,
    )
    _assert_valid_def("EvaluationFreezeRecord", scheduled_cap)
    assert scheduled_cap["status"] == "evidence_insufficient"
    assert scheduled_cap["trigger_on_time_issue_count"] < 104
    assert scheduled_cap["realized_target_union"] is None
    assert scheduled_cap["union_unique_supported_M5_6_event_count"] is None
    assert scheduled_cap["union_unique_full_study_area_M5_6_event_count"] is None
    assert scheduled_cap["unique_target_cluster_count"] is None
    assert scheduled_cap["final_window_membership_sha256"] is None
    assert scheduled_cap["cluster_membership_sha256"] is None
    assert all(
        horizon["supported_unique_M5_6_event_count"] is None
        and horizon["all_required_forecast_densities_finite_positive"] is None
        for horizon in cast(
            Sequence[Mapping[str, object]],
            scheduled_cap["horizon_evaluability"],
        )
    )

    no_complete_scope = _evaluation_input_freeze(
        trigger=52,
        sample_gate_met=False,
        formal_freeze_status="not_run_no_complete_scope",
    )
    _assert_valid_def("EvaluationFreezeRecord", no_complete_scope)
    assert no_complete_scope["union_unique_supported_M5_6_event_count"] == 0
    assert no_complete_scope["unique_target_cluster_count"] == 0
    assert (
        cast(
            Mapping[str, object],
            no_complete_scope["realized_target_union"],
        )["target_rows"]["row_count"]  # type: ignore[index]
        == 0
    )
    assert not cast(
        Sequence[object],
        cast(
            Mapping[str, object],
            no_complete_scope["selected_exposure_manifest"],
        )["rows"],
    )

    failed_query = _evaluation_input_freeze(
        trigger=52,
        sample_gate_met=False,
        formal_freeze_status="failed_query_fetch",
    )
    _assert_valid_def("EvaluationFreezeRecord", failed_query)
    assert failed_query["realized_target_union"] is None
    assert failed_query["bootstrap_preflight"]["status"] == (  # type: ignore[index]
        "not_run_formal_freeze_unavailable"
    )

    only_nine_clusters = copy.deepcopy(first_authorized)
    only_nine_clusters["unique_target_cluster_count"] = 9
    _assert_invalid_def("EvaluationFreezeRecord", only_nine_clusters)

    wrong_52_cardinality = copy.deepcopy(first_continue)
    wrong_52_cardinality["ordered_issue_prediction_seal_sha256"].pop()  # type: ignore[union-attr]
    _assert_invalid_def("EvaluationFreezeRecord", wrong_52_cardinality)

    wrong_104_previous = copy.deepcopy(final_authorized)
    wrong_104_previous["previous_evaluation_freeze_sha256"] = None
    _assert_invalid_def("EvaluationFreezeRecord", wrong_104_previous)

    cap_cannot_claim_104 = copy.deepcopy(scheduled_cap)
    cap_cannot_claim_104["trigger_on_time_issue_count"] = 104
    _assert_invalid_def("EvaluationFreezeRecord", cap_cannot_claim_104)

    unavailable_cannot_be_zero = copy.deepcopy(failed_query)
    unavailable_cannot_be_zero["union_unique_supported_M5_6_event_count"] = 0
    _assert_invalid_def("EvaluationFreezeRecord", unavailable_cannot_be_zero)

    missing_selected_manifest = copy.deepcopy(first_continue)
    missing_selected_manifest.pop("selected_exposure_manifest")
    _assert_invalid_def("EvaluationFreezeRecord", missing_selected_manifest)

    missing_alarm_manifest = copy.deepcopy(first_continue)
    missing_alarm_manifest.pop("alarm_area_manifest")
    _assert_invalid_def("EvaluationFreezeRecord", missing_alarm_manifest)


def test_evaluation_result_seal_has_four_endpoints_one_look_and_ordered_horizons() -> None:
    input_freeze = _evaluation_input_freeze(trigger=52, sample_gate_met=True)
    result = _evaluation_result_seal(input_freeze)
    _assert_valid_def("EvaluationFreezeRecord", result)
    confirmatory = cast(dict[str, object], result["confirmatory_result"])
    assert confirmatory["formal_effect_look_number"] == 1
    assert set(cast(Mapping[str, object], confirmatory["endpoint_results"])) == set(ENDPOINT_IDS)
    assert set(cast(Mapping[str, object], confirmatory["robustness_results"])) == set(ENDPOINT_IDS)
    effect_rows_manifest = _effect_rows_manifest(
        cast(str, confirmatory["input_freeze_sha256"])
    )
    _assert_valid_def("EffectRowsManifest", effect_rows_manifest)
    assert confirmatory["effect_rows_manifest_sha256"] == _canonical_file_sha256(
        effect_rows_manifest
    )
    alarm_area_manifest = cast(
        Mapping[str, object],
        result["alarm_area_manifest"],
    )
    alarm_area_comparison = _alarm_area_comparison(
        cast(str, result["alarm_area_manifest_sha256"]),
        entry_count=len(cast(Sequence[object], alarm_area_manifest["entries"])),
    )
    _assert_valid_def("AlarmAreaComparison", alarm_area_comparison)
    assert confirmatory["alarm_area_comparison_sha256"] == _canonical_file_sha256(
        alarm_area_comparison
    )

    second_look = copy.deepcopy(result)
    second_look["confirmatory_result"]["formal_effect_look_number"] = 2  # type: ignore[index]
    _assert_invalid_def("EvaluationFreezeRecord", second_look)

    missing_endpoint = copy.deepcopy(result)
    del missing_endpoint["confirmatory_result"]["endpoint_results"][  # type: ignore[index]
        ENDPOINT_IDS[-1]
    ]
    _assert_invalid_def("EvaluationFreezeRecord", missing_endpoint)

    duplicated_horizon = copy.deepcopy(input_freeze)
    duplicated_horizon["horizon_evaluability"][1]["horizon_days"] = 7  # type: ignore[index]
    _assert_invalid_def("EvaluationFreezeRecord", duplicated_horizon)

    reordered_horizons = copy.deepcopy(input_freeze)
    reordered_horizons["horizon_evaluability"][0], reordered_horizons["horizon_evaluability"][1] = (  # type: ignore[index]
        reordered_horizons["horizon_evaluability"][1],  # type: ignore[index]
        reordered_horizons["horizon_evaluability"][0],  # type: ignore[index]
    )
    _assert_invalid_def("EvaluationFreezeRecord", reordered_horizons)

    broken_result_chain = copy.deepcopy(result)
    broken_result_chain["input_freeze_sha256"] = _sha256("wrong-input-freeze")
    assert (
        broken_result_chain["confirmatory_result"]["input_freeze_sha256"]
        != (  # type: ignore[index]
            broken_result_chain["input_freeze_sha256"]
        )
    )
