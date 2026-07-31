from __future__ import annotations

import copy
import hashlib
import json
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
import yaml
from jsonschema.exceptions import ValidationError

from seismoflux.data.common import canonical_json_bytes
from seismoflux.stage2p.validation import (
    LIFECYCLE_IMPLEMENTATION_STATUS,
    SemanticValidationError,
    _validate_evaluation_policy_hashes,
    _validate_evaluation_sample_gate,
    parse_record_json_bytes,
    validate_evaluation_chain,
    validate_issue_chain,
    validate_prospective_lifecycle,
    validate_record_against_schema,
    validate_record_json_bytes,
    validate_record_semantics,
    validate_truth_chain,
)

ROOT = Path(__file__).resolve().parents[2]
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bind_object_identity(
    value: dict[str, Any],
    *,
    id_field: str,
    hash_field: str,
) -> str:
    body = {
        key: item
        for key, item in value.items()
        if key not in {id_field, hash_field}
    }
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    value[id_field] = digest
    value[hash_field] = digest
    return digest


def _bind_self_hash(value: dict[str, Any], hash_field: str) -> str:
    body = {key: item for key, item in value.items() if key != hash_field}
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    value[hash_field] = digest
    return digest


def _bind_record_content_hash(record: dict[str, Any]) -> str:
    content = {key: value for key, value in record.items() if key != "content_sha256"}
    digest = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    record["content_sha256"] = digest
    return digest


def _bind_record_hashes(record: dict[str, Any]) -> str:
    proof = record.get("remote_timestamp")
    if isinstance(proof, dict):
        core = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "timestamp_attempt_evidence",
                "remote_timestamp",
                "content_sha256",
            }
        }
        proof["preimage_sha256"] = hashlib.sha256(
            json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        attempts = record.get("timestamp_attempt_evidence")
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, dict):
                    attempt["attempt_preimage_sha256"] = proof[
                        "preimage_sha256"
                    ]
    audit_proof = record.get("missed_audit_remote_timestamp")
    if isinstance(audit_proof, dict):
        audit_core = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "missed_audit_timestamp_attempt_evidence",
                "missed_audit_remote_timestamp",
                "content_sha256",
            }
        }
        audit_proof["preimage_sha256"] = hashlib.sha256(
            json.dumps(
                audit_core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        audit_attempts = record.get("missed_audit_timestamp_attempt_evidence")
        if isinstance(audit_attempts, list):
            for attempt in audit_attempts:
                if isinstance(attempt, dict):
                    attempt["attempt_preimage_sha256"] = audit_proof[
                        "preimage_sha256"
                    ]
    return _bind_record_content_hash(record)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _protocol() -> dict[str, object]:
    return {
        "calendar": {
            "issue_weekday": "Thursday",
            "issue_local_time": "00:00:00",
            "issue_utc_offset": "+08:00",
            "first_issue_not_before_local": "2026-09-10T00:00:00+08:00",
            "first_issue_not_before_utc": "2026-09-09T16:00:00Z",
            "query_end_lag_minutes": 15,
        }
    }


def _full_protocol() -> dict[str, object]:
    value = yaml.safe_load(
        (ROOT / "configs" / "prospective_recent_seismicity.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def _install_release_bytes(
    release: dict[str, Any],
    artifacts: dict[str, bytes],
) -> str:
    _bind_self_hash(release, "manifest_sha256")
    raw = canonical_json_bytes(release)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    artifacts[file_sha256] = raw
    return file_sha256


def _replace_release_file_bytes(
    release: dict[str, Any],
    artifacts: dict[str, bytes],
    *,
    path: str,
    raw: bytes,
) -> None:
    file_sha256 = hashlib.sha256(raw).hexdigest()
    artifacts[file_sha256] = raw
    previous = release["files"][path]
    release["files"][path] = {
        "file_sha256": file_sha256,
        "git_blob_sha1": _git_blob_sha1(raw),
        "commit_role": previous["commit_role"],
    }


def _trusted_release_1a() -> tuple[
    dict[str, Any],
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    protocol_path = "configs/prospective_recent_seismicity.yaml"
    schema_path = "data/contracts/stage2p_prospective_records.json"
    protocol_raw = (ROOT / protocol_path).read_bytes()
    schema_raw = (ROOT / schema_path).read_bytes()
    protocol = yaml.safe_load(protocol_raw)
    schema = json.loads(schema_raw)
    assert isinstance(protocol, dict)
    assert isinstance(schema, dict)
    contract = protocol["record_contract"]["trusted_release_manifest_contract"]
    paths = contract["Stage2P_1A_required_existing_protocol_files"]
    path_roles = contract["Stage2P_1A_path_commit_roles"]
    artifacts: dict[str, bytes] = {}
    files: dict[str, dict[str, str]] = {}
    for path in paths:
        raw = (ROOT / path).read_bytes()
        file_sha256 = hashlib.sha256(raw).hexdigest()
        artifacts[file_sha256] = raw
        files[path] = {
            "file_sha256": file_sha256,
            "git_blob_sha1": _git_blob_sha1(raw),
            "commit_role": path_roles[path],
        }
    registry = protocol["remote_timestamp"]["trusted_registry"]
    release: dict[str, Any] = {
        "profile": "stage2p_trusted_release_manifest_v1",
        "implementation_status": LIFECYCLE_IMPLEMENTATION_STATUS,
        "protocol_tag": protocol["protocol"]["protocol_tag"],
        "code_tag": None,
        "protocol_commit": "a" * 40,
        "code_commit": None,
        "files": files,
        "code_commit_mirrored_protocol_files": {},
        "component_paths": contract["fixed_component_paths"],
        "protocol_config_path": protocol_path,
        "record_schema_path": schema_path,
        "timestamp_trust_registry": registry,
        "timestamp_trust_registry_sha256": registry["registry_sha256"],
        "manifest_sha256": _sha256("pending-trusted-release"),
    }
    release_file_sha256 = _install_release_bytes(release, artifacts)
    return release, artifacts, schema, protocol, release_file_sha256


def _trusted_release_accepted() -> tuple[
    dict[str, Any],
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    release, artifacts, schema, protocol, _ = _trusted_release_1a()
    contract = protocol["record_contract"]["trusted_release_manifest_contract"]
    fixed_roles = contract["fixed_path_commit_roles"]
    files: dict[str, dict[str, str]] = {}
    for path, role in fixed_roles.items():
        candidate = ROOT / path
        raw = (
            candidate.read_bytes()
            if candidate.is_file()
            else f"synthetic tagged bytes for {path}\n".encode()
        )
        file_sha256 = hashlib.sha256(raw).hexdigest()
        artifacts[file_sha256] = raw
        files[path] = {
            "file_sha256": file_sha256,
            "git_blob_sha1": _git_blob_sha1(raw),
            "commit_role": role,
        }
    protocol_path = release["protocol_config_path"]
    schema_path = release["record_schema_path"]
    release.update(
        {
            "implementation_status": "stage2p1b_synthetic_accepted",
            "code_tag": protocol["protocol"]["code_tag"],
            "code_commit": "b" * 40,
            "files": files,
            "code_commit_mirrored_protocol_files": {
                path: {
                    **files[path],
                    "commit_role": "code_commit",
                }
                for path in (protocol_path, schema_path)
            },
        }
    )
    code_manifest = {
        field: files[path]["file_sha256"]
        for field, path in release["component_paths"].items()
    }
    _bind_self_hash(code_manifest, "manifest_sha256")
    registry = protocol["remote_timestamp"]["trusted_registry"]
    expected_registry_identities = protocol["remote_timestamp"][
        "trusted_registry_identity"
    ]["expected"]
    cohort = {
        "protocol_config_sha256": files[protocol_path]["file_sha256"],
        "record_schema_sha256": files[schema_path]["file_sha256"],
        "protocol_tag": release["protocol_tag"],
        "code_tag": release["code_tag"],
        "protocol_commit": release["protocol_commit"],
        "code_commit": release["code_commit"],
        "code_manifest": code_manifest,
        "code_manifest_sha256": code_manifest["manifest_sha256"],
        "remote_timestamp_policy": {
            **expected_registry_identities,
            "verification_level": registry["verification_level"],
            "revocation_evaluated": registry["revocation"]["evaluated"],
            "full_non_revocation_claim": False,
        },
    }
    release_file_sha256 = _install_release_bytes(release, artifacts)
    return release, artifacts, schema, protocol, cohort, release_file_sha256


def _forecast_identity(label: str = "P0") -> dict[str, object]:
    return {
        "density_sha256": _sha256(f"{label}-density"),
        "grid_identity_sha256": _sha256("grid"),
        "complete_grid_cell_count": 100,
        "complete_grid_score_rows_sha256": _sha256(f"{label}-complete"),
        "ranked_grid_rows_sha256": _sha256(f"{label}-ranked"),
        "ranking_rule": "mass_per_exact_clipped_area_desc_then_row_column_cell_id_ascending",
        "alarm_area_budget_km2": 600_000,
        "alarm_prefix_cell_count": 60,
        "alarm_prefix_termination_reason": "next_complete_cell_would_exceed_budget",
        "next_unselected_rank_position_1_based": 61,
        "next_unselected_cell_id": "synthetic-next-cell",
        "next_unselected_complete_cell_area_km2": 500.0,
        "next_unselected_ranked_row_sha256": _sha256(f"{label}-next-row"),
        "alarm_prefix_rows_sha256": _sha256(f"{label}-prefix"),
        "selected_alarm_cell_ids_sha256": _sha256(f"{label}-selected"),
        "alarm_mask_sha256": _sha256(f"{label}-mask"),
        "actual_alarm_area_km2": 599_750.0,
        "remaining_budget_km2": 250.0,
        "complete_cells_only": True,
        "prefix_maximal_under_budget": True,
        "local_artifact_bundle_sha256": _sha256(f"{label}-bundle"),
    }


def _prediction_seal(
    issue_utc: datetime,
    query_end: datetime,
    *,
    source_snapshot_sha256: str,
) -> dict[str, object]:
    p0 = _forecast_identity()
    visualization_evidence: dict[str, object] = {
        "static_svg_sha256": _sha256("prediction-svg"),
        "interactive_html_sha256": _sha256("prediction-html"),
        "interactive_script_sha256": _sha256("prediction-script"),
        "external_URL_count": 0,
        "remote_script_count": 0,
        "network_fetch_capability_count": 0,
        "immature_truth_overlay_count": 0,
        "completion_marker_verified": True,
    }
    _bind_self_hash(visualization_evidence, "visualization_evidence_sha256")
    seal: dict[str, object] = {
        "prediction_seal_id": _sha256("pending-prediction-seal"),
        "prediction_seal_sha256": _sha256("pending-prediction-seal"),
        "source_snapshot_sha256": source_snapshot_sha256,
        "issue_time_utc": _utc_text(issue_utc),
        "query_end_utc": _utc_text(query_end),
        "valid_from_utc": _utc_text(issue_utc),
        "grid_identity_sha256": _sha256("grid"),
        "P0": p0,
        "P1": copy.deepcopy(p0),
        "PP": copy.deepcopy(p0),
        "R30_event_count": 0,
        "RP30_event_count": 0,
        "maximum_pairwise_actual_alarm_area_difference_km2": 0.0,
        "runtime_evidence": {
            "physical_core_count": 8,
            "worker_count": 6,
            "host_total_physical_memory_bytes": 64 * 1024**3,
            "host_available_memory_bytes_at_start": 48 * 1024**3,
            "peak_resident_set_bytes": 2 * 1024**3,
            "execution_device": "CPU",
            "gpu_model": None,
            "gpu_driver_version": None,
            "gpu_runtime_version": None,
            "GPU_equivalence_receipt_sha256": None,
        },
        "visualization_evidence": visualization_evidence,
        "sealed_at_utc": _utc_text(issue_utc - timedelta(minutes=9)),
    }
    _bind_object_identity(
        seal,
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    return seal


def _successful_fetch(query_end: datetime, issue_utc: datetime) -> dict[str, object]:
    return {
        "query_end_utc": _utc_text(query_end),
        "fetch_started_at_utc": _utc_text(query_end + timedelta(seconds=1)),
        "fetch_completed_at_utc": _utc_text(query_end + timedelta(minutes=1)),
        "http_status": 200,
        "raw_response_sha256": _sha256("response"),
        "exchange_outcome": "response_received",
        "outcome": "succeeded",
    }


def _on_time_issue(
    local_text: str = "2026-09-10T00:00:00+08:00",
    *,
    scheduled_sequence: int = 1,
    on_time_sequence: int = 1,
    previous_sha256: str | None = None,
    content_sha256: str | None = None,
) -> dict[str, object]:
    local = datetime.fromisoformat(local_text)
    issue_utc = local.astimezone(UTC)
    query_end = issue_utc - timedelta(minutes=15)
    core_frozen = issue_utc - timedelta(minutes=8)
    timestamp_deadline = issue_utc - timedelta(minutes=5)
    response_sha = _sha256(f"tsa-response-{scheduled_sequence}")
    fetch = _successful_fetch(query_end, issue_utc)
    source_snapshot: dict[str, object] = {
        "snapshot_id": _sha256("pending-source-snapshot"),
        "snapshot_sha256": _sha256("pending-source-snapshot"),
        "issue_time_utc": _utc_text(issue_utc),
        "query_end_utc": _utc_text(query_end),
        "acquisition": copy.deepcopy(fetch),
        "seal_completed_at_utc": _utc_text(issue_utc - timedelta(minutes=10)),
    }
    source_snapshot_sha256 = _bind_object_identity(
        source_snapshot,
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    record: dict[str, object] = {
        "record_type": "IssueInputSnapshotRecord",
        "issue_id": f"stage2p-issue-{local:%Y%m%dT000000}+0800",
        "scheduled_issue_sequence": scheduled_sequence,
        "on_time_issue_sequence": on_time_sequence,
        "previous_issue_record_sha256": previous_sha256,
        "issue_time_local": local_text,
        "issue_time_utc": _utc_text(issue_utc),
        "query_end_utc": _utc_text(query_end),
        "record_core_frozen_at_utc": _utc_text(core_frozen),
        "timestamp_deadline_utc": _utc_text(timestamp_deadline),
        "status": "on_time",
        "attempt_evidence": [fetch],
        "source_snapshot": source_snapshot,
        "prediction_seal": _prediction_seal(
            issue_utc,
            query_end,
            source_snapshot_sha256=source_snapshot_sha256,
        ),
        "timestamp_attempt_evidence": [
            {
                "attempt_index": 0,
                "http_status": 200,
                "response_content_type": "application/timestamp-reply",
                "response_byte_count": 512,
                "response_sha256": response_sha,
                "authority_identity_sha256": _sha256("tsa-authority"),
                "trust_chain_sha256": _sha256("tsa-trust-chain"),
                "offline_trust_path_valid": True,
                "request_started_at_utc": _utc_text(
                    issue_utc - timedelta(minutes=7, seconds=30)
                ),
                "genTime_utc": _utc_text(issue_utc - timedelta(minutes=7)),
                "response_received_at_utc": _utc_text(
                    issue_utc - timedelta(minutes=6, seconds=30)
                ),
                "attempt_completed_at_utc": _utc_text(
                    issue_utc - timedelta(minutes=6, seconds=30)
                ),
                "genTime_before_deadline": True,
                "outcome": "selected_valid",
            }
        ],
        "remote_timestamp": {
            "selected_attempt_index": 0,
            "selected_response_sha256": response_sha,
            "deadline_utc": _utc_text(timestamp_deadline),
        },
        "content_sha256": content_sha256 or _sha256(f"issue-{scheduled_sequence}"),
    }
    _bind_record_hashes(record)  # type: ignore[arg-type]
    return record


def _bind_missed_audit(record: dict[str, Any]) -> None:
    issue_time = datetime.fromisoformat(
        str(record["issue_time_utc"]).replace("Z", "+00:00")
    )
    final_core = issue_time - timedelta(minutes=4)
    audit_deadline = issue_time
    response_sha256 = _sha256(
        f"missed-audit-response-{record['scheduled_issue_sequence']}"
    )
    record.update(
        {
            "record_core_frozen_at_utc": _utc_text(final_core),
            "missed_audit_timestamp_deadline_utc": _utc_text(audit_deadline),
            "missed_audit_timestamp_attempt_evidence": [
                {
                    "attempt_index": 0,
                    "authority_url": "http://timestamp.digicert.com",
                    "request_content_type": "application/timestamp-query",
                    "response_content_type": "application/timestamp-reply",
                    "request_byte_count": 128,
                    "request_sha256": _sha256("missed-audit-request"),
                    "attempt_preimage_sha256": _sha256(
                        "pending-missed-audit-core"
                    ),
                    "nonce_sha256": _sha256("missed-audit-nonce"),
                    "certReq": True,
                    "request_started_at_utc": _utc_text(
                        issue_time - timedelta(minutes=3, seconds=30)
                    ),
                    "attempt_completed_at_utc": _utc_text(
                        issue_time - timedelta(minutes=2)
                    ),
                    "response_received_at_utc": _utc_text(
                        issue_time - timedelta(minutes=2)
                    ),
                    "http_status": 200,
                    "response_byte_count": 512,
                    "response_sha256": response_sha256,
                    "authority_identity_sha256": _sha256(
                        "missed-audit-authority"
                    ),
                    "trust_chain_sha256": _sha256(
                        "missed-audit-trust-chain"
                    ),
                    "genTime_utc": _utc_text(
                        issue_time - timedelta(minutes=3)
                    ),
                    "offline_trust_path_valid": True,
                    "genTime_before_deadline": True,
                    "outcome": "selected_valid",
                }
            ],
            "missed_audit_remote_timestamp": {
                "subject_type": "IssueInputSnapshotRecord",
                "proof_field_name": "missed_audit_remote_timestamp",
                "final_hash_field_name": "content_sha256",
                "preimage_profile": "stage2p_missed_audit_rfc3161_core_v1",
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
                "preimage_sha256": _sha256("pending-missed-audit-core"),
                "deadline_utc": _utc_text(audit_deadline),
                "selected_attempt_index": 0,
                "selected_response_sha256": response_sha256,
                "message_imprint_matches_preimage": True,
                "nonce_matches_request": True,
                "tsa_policy_oid": "2.16.840.1.114412.7.1",
                "timestamping_EKU_verified": True,
                "selection_rule": (
                    "first_offline_trust_path_valid_response_with_"
                    "genTime_before_deadline"
                ),
                "verification_code_sha256": _sha256(
                    "missed-audit-verification-code"
                ),
                "verified": True,
            },
        }
    )
    _bind_record_hashes(record)


def _horizon() -> dict[str, object]:
    return {
        "horizon_days": 7,
        "complete_exposure_count": 4,
        "truth_available_complete_exposure_count": 4,
        "selected_truth_snapshot_unavailable_count": 0,
        "supported_unique_M5_6_event_count": 2,
        "full_study_area_unique_M5_6_event_count": 3,
        "unique_target_cluster_count": 2,
        "all_required_forecast_densities_finite_positive": True,
        "evaluable": True,
        "unevaluable_reason": None,
    }


def _request_with_hash(
    *,
    endpoint: str,
    start: str,
    end: str,
    count: bool,
) -> dict[str, object]:
    request: dict[str, object] = {
        "method": "GET",
        "endpoint": endpoint,
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
    }
    if count:
        order = [
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
        ]
        request["excluded_query_only_parameters"] = [
            "orderby",
            "limit",
            "offset",
            "includeallorigins",
            "includeallmagnitudes",
        ]
    else:
        request.update(
            {
                "orderby": "time-asc",
                "limit": 20_000,
                "includeallorigins": False,
                "includeallmagnitudes": False,
                "offset": 1,
            }
        )
        order = [
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
        ]
    request["canonical_parameter_order"] = order
    bbox = request["bbox"]
    values = {
        "starttime": start,
        "endtime": end,
        "minlongitude": bbox[0],  # type: ignore[index]
        "minlatitude": bbox[1],  # type: ignore[index]
        "maxlongitude": bbox[2],  # type: ignore[index]
        "maxlatitude": bbox[3],  # type: ignore[index]
        "minmagnitude": request["minmagnitude"],
        "eventtype": request["eventtype"],
        "format": request["format"],
        "orderby": request.get("orderby"),
        "limit": request.get("limit"),
        "includeallorigins": request.get("includeallorigins"),
        "includeallmagnitudes": request.get("includeallmagnitudes"),
        "includedeleted": request["includedeleted"],
        "includesuperseded": request["includesuperseded"],
        "reviewstatus": request["reviewstatus"],
        "offset": request.get("offset"),
        "jsonerror": request["jsonerror"],
        "nodata": request["nodata"],
    }

    def text(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    query = "&".join(
        f"{quote(name, safe='-._~')}={quote(text(values[name]), safe='-._~')}" for name in order
    )
    request["canonical_url_utf8_sha256"] = hashlib.sha256(
        f"{endpoint}?{query}".encode()
    ).hexdigest()
    return request


def _fdsn_fetch() -> dict[str, object]:
    start = "2026-09-09T16:00:00Z"
    end = "2026-09-16T16:00:00Z"
    query = _request_with_hash(
        endpoint="https://earthquake.usgs.gov/fdsnws/event/1/query",
        start=start,
        end=end,
        count=False,
    )
    count_request = _request_with_hash(
        endpoint="https://earthquake.usgs.gov/fdsnws/event/1/count",
        start=start,
        end=end,
        count=True,
    )
    return {
        "query_start_utc": start,
        "query_end_utc": end,
        "query_request": query,
        "count_preflight": {
            "request": count_request,
            "fetch_started_at_utc": "2026-10-16T16:00:00Z",
            "fetch_completed_at_utc": "2026-10-16T16:00:00Z",
            "http_status": 200,
            "response_content_type": "application/json",
            "response_body_byte_count": 1,
            "response_headers_sha256": _sha256("fdsn-count-headers"),
            "raw_response_sha256": _sha256("fdsn-count-response"),
            "geojson_parse_verified": True,
            "parsed_count": 4,
            "outcome": "succeeded",
        },
        "request_identity_sha256": query["canonical_url_utf8_sha256"],
        "query_count_preflight_request_sha256": count_request["canonical_url_utf8_sha256"],
        "query_count": 4,
        "feature_count": 4,
        "fetch_started_at_utc": "2026-10-16T16:00:01Z",
        "fetch_completed_at_utc": "2026-10-16T16:00:02Z",
        "http_status": 200,
        "raw_response_sha256": _sha256("fdsn-response"),
        "exchange_outcome": "response_received",
        "outcome": "succeeded",
    }


def _mature_truth_record() -> dict[str, object]:
    issue = datetime(2026, 9, 9, 16, tzinfo=UTC)
    target_end = issue + timedelta(days=7)
    due = target_end + timedelta(days=30)
    fetch = _fdsn_fetch()
    attempt = {
        **fetch,
        "attempt_index": 0,
        "issue_time_utc": _utc_text(issue),
        "horizon_days": 7,
        "target_start_exclusive_utc": _utc_text(issue),
        "target_end_inclusive_utc": _utc_text(target_end),
        "maturity_due_at_utc": _utc_text(due),
        "retry_offset_hours": 0,
        "scheduled_at_utc": _utc_text(due),
        "selected_as_truth_snapshot": True,
    }
    truth_snapshot: dict[str, object] = {
        "truth_snapshot_id": _sha256("pending-truth-snapshot"),
        "truth_snapshot_sha256": _sha256("pending-truth-snapshot"),
        "target_start_exclusive_utc": _utc_text(issue),
        "target_end_inclusive_utc": _utc_text(target_end),
        "maturity_due_at_utc": _utc_text(due),
        "retry_offset_hours": 0,
        "acquisition": copy.deepcopy(fetch),
        "seal_completed_at_utc": _utc_text(due + timedelta(seconds=3)),
    }
    _bind_object_identity(
        truth_snapshot,
        id_field="truth_snapshot_id",
        hash_field="truth_snapshot_sha256",
    )
    record: dict[str, object] = {
        "record_type": "MatureTruthSnapshotRecord",
        "issue_id": "stage2p-issue-20260910T000000+0800",
        "horizon_days": 7,
        "maturity_due_at_utc": _utc_text(due),
        "status": "mature_truth_sealed",
        "attempt_evidence": [attempt],
        "truth_snapshot": truth_snapshot,
    }
    _bind_record_content_hash(record)  # type: ignore[arg-type]
    return record


def _code_manifest() -> dict[str, object]:
    manifest = {
        "parser_code_sha256": _sha256("parser"),
        "normalization_config_sha256": _sha256("normalization"),
        "deduplication_code_sha256": _sha256("dedup"),
        "deduplication_config_sha256": _sha256("dedup-config"),
        "revision_policy_sha256": _sha256("revision"),
        "model_code_sha256": _sha256("model"),
        "evaluation_code_sha256": _sha256("evaluation"),
        "visualization_code_sha256": _sha256("visualization"),
        "semantic_validator_code_sha256": _sha256("validator"),
        "environment_lock_file_sha256": _sha256("lock-file"),
        "pyproject_file_sha256": _sha256("pyproject-file"),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return manifest


def _table_artifact_identity(label: str, *, row_count: int) -> dict[str, object]:
    file_sha256 = _sha256(f"{label}-file")
    return {
        "artifact_id": file_sha256,
        "byte_count": max(2, row_count * 64),
        "row_count": row_count,
        "file_sha256": file_sha256,
        "content_sha256": _sha256(f"{label}-content"),
        "schema_sha256": _sha256(f"{label}-schema"),
        "sort_order_sha256": _sha256(f"{label}-sort"),
        "local_restricted": True,
    }


def _typed_table_artifact(
    label: str,
    *,
    role: str,
    row_schema_ref: str,
    sort_profile: str,
    row_count: int = 2,
) -> dict[str, object]:
    file_sha256 = _sha256(f"{label}-file")
    return {
        "artifact_id": file_sha256,
        "byte_count": 128,
        "row_count": row_count,
        "file_sha256": file_sha256,
        "content_sha256": _sha256(f"{label}-content"),
        "schema_sha256": _sha256(f"{label}-schema"),
        "sort_order_sha256": _sha256(f"{label}-sort"),
        "table_role": role,
        "serialization_profile": "seismoflux_canonical_jsonl_v1",
        "row_schema_ref": row_schema_ref,
        "sort_profile": sort_profile,
        "local_restricted": True,
    }


def _formal_freeze_manifest() -> dict[str, object]:
    start = "2026-09-09T16:00:00Z"
    end = "2026-12-16T16:00:00Z"
    query = _request_with_hash(
        endpoint="https://earthquake.usgs.gov/fdsnws/event/1/query",
        start=start,
        end=end,
        count=False,
    )
    count_request = _request_with_hash(
        endpoint="https://earthquake.usgs.gov/fdsnws/event/1/count",
        start=start,
        end=end,
        count=True,
    )
    raw_sha256 = _sha256("formal-raw-response")
    headers_sha256 = _sha256("formal-response-headers")
    acquisition: dict[str, object] = {
        "source_id": "usgs_anss_comcat_fdsn_event_api_v1",
        "query_start_utc": start,
        "query_end_utc": end,
        "query_request": query,
        "count_preflight": {
            "request": count_request,
            "fetch_started_at_utc": "2026-12-20T16:00:00Z",
            "fetch_completed_at_utc": "2026-12-20T16:00:01Z",
            "http_status": 200,
            "response_content_type": "application/json",
            "response_body_byte_count": 1,
            "response_headers_sha256": _sha256("formal-count-headers"),
            "raw_response_sha256": _sha256("formal-count-response"),
            "geojson_parse_verified": True,
            "parsed_count": 4,
            "outcome": "succeeded",
        },
        "request_identity_sha256": query["canonical_url_utf8_sha256"],
        "query_count_preflight_request_sha256": count_request[
            "canonical_url_utf8_sha256"
        ],
        "query_count": 4,
        "feature_count": 4,
        "fetch_started_at_utc": "2026-12-20T16:00:02Z",
        "fetch_completed_at_utc": "2026-12-20T16:00:03Z",
        "http_status": 200,
        "response_content_type": "application/json",
        "response_body_byte_count": 64,
        "geojson_parse_verified": True,
        "response_headers_sha256": headers_sha256,
        "raw_response": {
            "file_sha256": raw_sha256,
        },
    }
    snapshot_preimage = {
        "profile": "stage2p_formal_freeze_source_snapshot_v1",
        "source_id": acquisition["source_id"],
        "canonical_query_request_sha256": acquisition[
            "request_identity_sha256"
        ],
        "response_headers_sha256": headers_sha256,
        "raw_response_sha256": raw_sha256,
        "response_content_type": acquisition["response_content_type"],
        "response_body_byte_count": acquisition["response_body_byte_count"],
        "http_status": acquisition["http_status"],
        "fetch_started_at_utc": acquisition["fetch_started_at_utc"],
        "fetch_completed_at_utc": acquisition["fetch_completed_at_utc"],
    }
    manifest: dict[str, object] = {
        "status": "succeeded_single_full_cohort_response",
        "evaluation_scope_sha256": _sha256("formal-evaluation-scope"),
        "global_target_start_exclusive_utc": start,
        "global_target_end_inclusive_utc": end,
        "source_acquisition": acquisition,
        "source_snapshot_sha256": hashlib.sha256(
            json.dumps(
                snapshot_preimage,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "snapshot_observed_at_utc": "2026-12-20T16:00:03Z",
        "freeze_completed_at_utc": "2026-12-20T16:01:00Z",
        "query_batch_count": 1,
        "ordered_query_request_sha256": [
            query["canonical_url_utf8_sha256"],
        ],
        "ordered_response_headers_sha256": [headers_sha256],
        "ordered_raw_response_sha256": [raw_sha256],
        "normalized_rows": _table_artifact_identity(
            "formal-normalized",
            row_count=4,
        ),
        "deduplicated_rows": _table_artifact_identity(
            "formal-deduplicated",
            row_count=4,
        ),
        "preferred_field_rows": _table_artifact_identity(
            "formal-preferred",
            row_count=4,
        ),
        "window_membership_rows": _table_artifact_identity(
            "formal-window-membership",
            row_count=4,
        ),
        "formal_window_target_bindings": _table_artifact_identity(
            "formal-window-bindings",
            row_count=1,
        ),
        "parser_code_sha256": _sha256("formal-parser"),
        "normalization_config_sha256": _sha256("formal-normalization"),
        "deduplication_code_sha256": _sha256("formal-deduplication"),
        "deduplication_config_sha256": _sha256("formal-dedup-config"),
        "revision_policy_sha256": _sha256("formal-revision"),
        "code_manifest_sha256": _sha256("formal-code-manifest"),
        "local_restricted": True,
    }
    _bind_self_hash(manifest, "manifest_sha256")
    return manifest


def _formal_freeze_manifest_with_registry() -> tuple[
    dict[str, object],
    dict[str, object],
]:
    manifest = _formal_freeze_manifest()
    protocol = _full_protocol()
    schema = json.loads(
        (ROOT / "data" / "contracts" / "stage2p_prospective_records.json").read_text(
            encoding="utf-8"
        )
    )
    registry = protocol["target_cohort"]["formal_freeze_source_manifest"][  # type: ignore[index]
        "derived_table_registry"
    ]
    serialization_profile = registry["serialization"]["profile"]  # type: ignore[index]
    for table_name, contract in registry["tables"].items():  # type: ignore[union-attr]
        identity = manifest[table_name]
        row_schema_name = contract["row_schema_ref"].rsplit("/", 1)[1]
        identity.update(
            {
                "table_role": contract["table_role"],
                "serialization_profile": serialization_profile,
                "row_schema_ref": contract["row_schema_ref"],
                "sort_profile": contract["sort_profile"],
                "schema_sha256": hashlib.sha256(
                    canonical_json_bytes(schema["$defs"][row_schema_name])
                ).hexdigest(),
                "sort_order_sha256": hashlib.sha256(
                    canonical_json_bytes(contract["sort_spec"])
                ).hexdigest(),
            }
        )
    _bind_self_hash(manifest, "manifest_sha256")
    return manifest, protocol


def _formal_failure_record(status: str) -> dict[str, object]:
    start = "2026-09-09T16:00:00Z"
    end = "2026-12-16T16:00:00Z"
    scheduled = "2026-12-20T15:59:00Z"
    count: dict[str, object] | None = None
    stage = "scope_selection"
    artifact: str | None = None
    if status not in {
        "not_run_no_complete_scope",
        "not_run_scheduled_issue_cap_terminal",
    }:
        count = {
            "request": _request_with_hash(
                endpoint="https://earthquake.usgs.gov/fdsnws/event/1/count",
                start=start,
                end=end,
                count=True,
            ),
            "fetch_started_at_utc": "2026-12-20T15:59:01Z",
            "fetch_completed_at_utc": "2026-12-20T15:59:02Z",
            "http_status": 200,
            "response_content_type": "application/json",
            "response_body_byte_count": 5,
            "response_headers_sha256": _sha256("formal-count-headers"),
            "raw_response_sha256": _sha256("formal-count-body"),
            "geojson_parse_verified": True,
            "parsed_count": 12,
            "outcome": "succeeded",
            "failure_code": None,
        }
        stage = "count_preflight"
        if status == "failed_count_preflight":
            count.update(
                {
                    "http_status": None,
                    "response_content_type": None,
                    "response_body_byte_count": None,
                    "response_headers_sha256": None,
                    "raw_response_sha256": None,
                    "geojson_parse_verified": None,
                    "parsed_count": None,
                    "outcome": "network_failure",
                    "failure_code": "count_network_failure",
                }
            )
        elif status == "failed_count_limit":
            count["parsed_count"] = 20_000
        elif status == "failed_query_fetch":
            stage = "query_fetch"
        elif status == "failed_query_parse_or_count_mismatch":
            stage = "query_parse_or_count_consistency"
            artifact = _sha256("query-parse-failure-artifact")
        elif status == "failed_local_derivation_or_freeze":
            stage = "normalization_deduplication_or_window_binding"
            artifact = _sha256("local-derivation-failure-artifact")
    evidence: dict[str, object] = {
        "failure_evidence_sha256": _sha256("pending-formal-failure"),
        "status": status,
        "evaluation_scope_sha256": _sha256("formal-evaluation-scope"),
        "scheduled_at_utc": scheduled,
        "global_target_start_exclusive_utc": (
            None if count is None else start
        ),
        "global_target_end_inclusive_utc": None if count is None else end,
        "count_preflight": count,
        "query_failure_artifact_sha256": artifact,
        "failure_stage": stage,
        "failure_code": status,
        "contains_target_or_effect_rows": False,
        "local_restricted": True,
    }
    _bind_self_hash(evidence, "failure_evidence_sha256")
    unavailable = status != "not_run_no_complete_scope"
    return {
        "formal_freeze_status": status,
        "formal_freeze_source_manifest": None,
        "formal_freeze_source_manifest_sha256": None,
        "formal_freeze_failure_evidence": evidence,
        "formal_freeze_failure_evidence_sha256": evidence[
            "failure_evidence_sha256"
        ],
        "record_core_frozen_at_utc": "2026-12-20T16:00:00Z",
        "effect_rows_opened_at_utc": None,
        "sample_gate_met": False,
        "confirmatory_effects_authorized": False,
        "confirmatory_result": None,
        "union_unique_supported_M5_6_event_count": None if unavailable else 0,
        "union_unique_full_study_area_M5_6_event_count": (
            None if unavailable else 0
        ),
        "unique_target_cluster_count": None if unavailable else 0,
        "realized_target_union": (
            None if unavailable else {"unique_event_count": 0}
        ),
    }


def _no_complete_scope_with_empty_selected_exposure() -> dict[str, object]:
    record = _formal_failure_record("not_run_no_complete_scope")
    empty_array_sha256 = hashlib.sha256(canonical_json_bytes([])).hexdigest()
    selected_manifest = {
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
        "candidate_issue_prediction_seal_sha256": [],
        "rows": [],
    }
    record.update(
        {
            "ordered_issue_prediction_seal_sha256": [],
            "selected_exposure_manifest": selected_manifest,
            "selected_exposure_manifest_union_sha256": hashlib.sha256(
                canonical_json_bytes(selected_manifest)
            ).hexdigest(),
            "truth_availability_manifest_union_sha256": empty_array_sha256,
            "horizon_evaluability": [
                {
                    "horizon_days": days,
                    "complete_exposure_count": 0,
                    "truth_available_complete_exposure_count": 0,
                    "selected_truth_snapshot_unavailable_count": 0,
                    "selected_exposure_manifest_sha256": empty_array_sha256,
                    "truth_availability_manifest_sha256": empty_array_sha256,
                    "supported_unique_M5_6_event_count": 0,
                    "full_study_area_unique_M5_6_event_count": 0,
                    "unique_target_cluster_count": 0,
                    "all_required_forecast_densities_finite_positive": True,
                    "evaluable": False,
                    "unevaluable_reason": "no_complete_exposure",
                }
                for days in (7, 30, 90)
            ],
        }
    )
    return record


def _selected_exposure_record() -> dict[str, object]:
    prediction_seal_sha256 = _sha256("selected-exposure-seal")
    row = {
        "horizon_days": 7,
        "selection_ordinal_1_based": 1,
        "scheduled_issue_sequence": 1,
        "issue_id": "stage2p-issue-20260910T000000+0800",
        "prediction_seal_sha256": prediction_seal_sha256,
        "issue_time_utc": "2026-09-09T16:00:00Z",
        "target_start_exclusive_utc": "2026-09-09T16:00:00Z",
        "target_end_inclusive_utc": "2026-09-16T16:00:00Z",
        "selected_truth_record_sha256": _sha256("selected-exposure-truth"),
        "selected_truth_revision_sequence": 0,
        "truth_record_status": "mature_truth_sealed",
        "truth_available": True,
    }
    selected_manifest = {
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
        "candidate_issue_prediction_seal_sha256": [prediction_seal_sha256],
        "rows": [row],
    }
    projection_fields = (
        "horizon_days",
        "selection_ordinal_1_based",
        "issue_id",
        "prediction_seal_sha256",
        "selected_truth_record_sha256",
        "selected_truth_revision_sequence",
        "truth_record_status",
        "truth_available",
    )
    truth_projection = [{field: row[field] for field in projection_fields}]
    empty_array_sha256 = hashlib.sha256(canonical_json_bytes([])).hexdigest()
    return {
        "ordered_issue_prediction_seal_sha256": [prediction_seal_sha256],
        "selected_exposure_manifest": selected_manifest,
        "selected_exposure_manifest_union_sha256": hashlib.sha256(
            canonical_json_bytes(selected_manifest)
        ).hexdigest(),
        "truth_availability_manifest_union_sha256": hashlib.sha256(
            canonical_json_bytes(truth_projection)
        ).hexdigest(),
        "horizon_evaluability": [
            {
                "horizon_days": days,
                "complete_exposure_count": 1 if days == 7 else 0,
                "truth_available_complete_exposure_count": (
                    1 if days == 7 else 0
                ),
                "selected_truth_snapshot_unavailable_count": 0,
                "selected_exposure_manifest_sha256": (
                    hashlib.sha256(canonical_json_bytes([row])).hexdigest()
                    if days == 7
                    else empty_array_sha256
                ),
                "truth_availability_manifest_sha256": (
                    hashlib.sha256(
                        canonical_json_bytes(truth_projection)
                    ).hexdigest()
                    if days == 7
                    else empty_array_sha256
                ),
                "supported_unique_M5_6_event_count": 1 if days == 7 else 0,
                "full_study_area_unique_M5_6_event_count": (
                    1 if days == 7 else 0
                ),
                "unique_target_cluster_count": 1 if days == 7 else 0,
                "all_required_forecast_densities_finite_positive": True,
                "evaluable": days == 7,
                "unevaluable_reason": (
                    None if days == 7 else "no_complete_exposure"
                ),
            }
            for days in (7, 30, 90)
        ],
    }


def _selected_exposure_record_with_alarm_area() -> dict[str, object]:
    record = _selected_exposure_record()
    row = record["selected_exposure_manifest"]["rows"][0]  # type: ignore[index]
    areas = (1_000.0, 1_300.0, 900.0)
    maximum = max(
        abs(areas[0] - areas[1]),
        abs(areas[0] - areas[2]),
        abs(areas[1] - areas[2]),
    )
    alarm_manifest = {
        "profile": "stage2p_alarm_area_manifest_v1",
        "selected_exposure_manifest_union_sha256": record[
            "selected_exposure_manifest_union_sha256"
        ],
        "maximum_allowed_pairwise_difference_km2_float64_hex": (
            "4083880000000000"
        ),
        "entries": [
            {
                "scheduled_issue_sequence": row["scheduled_issue_sequence"],
                "issue_id": row["issue_id"],
                "prediction_seal_sha256": row["prediction_seal_sha256"],
                "P0_actual_alarm_area_km2_float64_hex": struct.pack(
                    ">d", areas[0]
                ).hex(),
                "P1_actual_alarm_area_km2_float64_hex": struct.pack(
                    ">d", areas[1]
                ).hex(),
                "PP_actual_alarm_area_km2_float64_hex": struct.pack(
                    ">d", areas[2]
                ).hex(),
                (
                    "maximum_pairwise_actual_alarm_area_difference_km2_"
                    "float64_hex"
                ): struct.pack(">d", maximum).hex(),
                "within_maximum_pairwise_difference": True,
            }
        ],
    }
    record.update(
        {
            "alarm_area_manifest": alarm_manifest,
            "alarm_area_manifest_sha256": hashlib.sha256(
                canonical_json_bytes(alarm_manifest)
            ).hexdigest(),
        }
    )
    return record


def _synthetic_schema_value(
    schema: dict[str, Any],
    node: dict[str, Any],
    *,
    path: str,
) -> Any:
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in {"Sha256", "RecordContentHash"}:
            return _sha256(path)
        if name == "NullableSha256":
            return None
        if name == "UtcTimestamp":
            return "2026-09-09T15:00:00Z"
        if name == "NullableUtcTimestamp":
            return None
        if name == "LocalIssueTimestamp":
            return "2026-09-10T00:00:00+08:00"
        if name == "GitCommit":
            return "a" * 40
        return _synthetic_schema_value(schema, schema["$defs"][name], path=f"{path}.{name}")
    if "const" in node:
        return copy.deepcopy(node["const"])
    if "allOf" in node:
        base = {key: value for key, value in node.items() if key != "allOf"}
        merged: Any = (
            _synthetic_schema_value(schema, base, path=f"{path}.base") if base else {}
        )
        for index, part in enumerate(node["allOf"]):
            if "if" in part:
                continue
            value = _synthetic_schema_value(
                schema,
                part,
                path=f"{path}.allOf[{index}]",
            )
            if isinstance(merged, dict) and isinstance(value, dict):
                merged.update(value)
                for name, property_schema in part.get("properties", {}).items():
                    if set(property_schema) <= {"description"}:
                        continue
                    merged[name] = _synthetic_schema_value(
                        schema,
                        property_schema,
                        path=f"{path}.allOf[{index}].{name}",
                    )
            elif value is not None:
                merged = value
        return merged
    for keyword in ("oneOf", "anyOf"):
        if keyword in node:
            return _synthetic_schema_value(
                schema,
                node[keyword][0],
                path=f"{path}.{keyword}[0]",
            )
    if "enum" in node:
        return copy.deepcopy(node["enum"][0])
    node_type = node.get("type")
    if isinstance(node_type, list):
        node_type = next((item for item in node_type if item != "null"), "null")
    if node_type == "object" or "properties" in node:
        properties = node.get("properties", {})
        return {
            name: _synthetic_schema_value(
                schema,
                properties[name],
                path=f"{path}.{name}",
            )
            for name in node.get("required", [])
        }
    if node_type == "array":
        minimum = node.get("minItems", 0)
        prefix = node.get("prefixItems", [])
        items = node.get("items", {})
        return [
            _synthetic_schema_value(
                schema,
                prefix[index] if index < len(prefix) else items,
                path=f"{path}[{index}]",
            )
            for index in range(minimum)
        ]
    if node_type == "boolean":
        return True
    if node_type == "integer":
        return int(node.get("minimum", 0))
    if node_type == "number":
        if "exclusiveMinimum" in node:
            return float(node["exclusiveMinimum"]) + 1.0
        return float(node.get("minimum", 0.0))
    if node_type is None and any(
        keyword in node
        for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
    ):
        if "exclusiveMinimum" in node:
            return float(node["exclusiveMinimum"]) + 1.0
        return float(node.get("minimum", 0.0))
    if node_type == "null":
        return None
    if node_type == "string" or "pattern" in node or "format" in node:
        pattern = node.get("pattern", "")
        if node.get("format") in {"uri", "uri-reference"}:
            return "https://example.test/synthetic"
        if "stage2p-issue-" in pattern:
            return "stage2p-issue-20260910T000000+0800"
        if "stage2p-" in pattern and "-v" in pattern:
            return "stage2p-synthetic-v1"
        if "[0-9a-f]{64}" in pattern:
            return _sha256(path)
        if "[0-9a-f]{16}" in pattern:
            return "0000000000000000"
        if "[0-9a-f]{40}" in pattern:
            return "a" * 40
        if "[0-9]+" in pattern and "\\." in pattern:
            return "1.2"
        return "synthetic"
    raise AssertionError(f"cannot synthesize schema node at {path}: {node}")


def _synthetic_record(schema: dict[str, Any], name: str) -> dict[str, Any]:
    value = _synthetic_schema_value(schema, schema["$defs"][name], path=name)
    assert isinstance(value, dict)

    def bind_artifact_ids(candidate: object) -> None:
        if isinstance(candidate, dict):
            if "artifact_id" in candidate and "file_sha256" in candidate:
                candidate["artifact_id"] = candidate["file_sha256"]
            event_rows = candidate.get("event_rows")
            if isinstance(event_rows, dict):
                candidate["event_count"] = event_rows.get("row_count")
                candidate["event_rows_content_sha256"] = event_rows.get(
                    "content_sha256"
                )
            target_rows = candidate.get("target_rows")
            if isinstance(target_rows, dict):
                candidate["unique_event_count"] = target_rows.get("row_count")
                candidate["target_rows_content_sha256"] = target_rows.get(
                    "content_sha256"
                )
            cutover_rows = candidate.get("cutover_cross_source_match_rows")
            if isinstance(cutover_rows, dict):
                candidate["cutover_cross_source_match_count"] = cutover_rows.get(
                    "row_count"
                )
                candidate["cutover_cross_source_match_sha256"] = cutover_rows.get(
                    "content_sha256"
                )
            window_rows = candidate.get("window_membership_rows")
            if isinstance(window_rows, dict):
                candidate["window_membership_sha256"] = window_rows.get(
                    "content_sha256"
                )
            for nested in candidate.values():
                bind_artifact_ids(nested)
        elif isinstance(candidate, list):
            for nested in candidate:
                bind_artifact_ids(nested)

    bind_artifact_ids(value)
    return value


def _definition_schema(schema: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{name}",
    }


def _bind_complete_successful_fetch(
    fetch: dict[str, Any],
    *,
    start: str,
    end: str,
    count_started: str,
    count_completed: str,
    query_started: str,
    query_completed: str,
    label: str,
) -> None:
    query = _request_with_hash(
        endpoint="https://earthquake.usgs.gov/fdsnws/event/1/query",
        start=start,
        end=end,
        count=False,
    )
    count_request = _request_with_hash(
        endpoint="https://earthquake.usgs.gov/fdsnws/event/1/count",
        start=start,
        end=end,
        count=True,
    )
    captured_headers = {
        "date": "Wed, 09 Sep 2026 16:00:00 GMT",
        "etag": None,
        "last_modified": None,
        "content_type": "application/json",
        "content_length": "64",
    }
    response_headers_sha256 = hashlib.sha256(
        canonical_json_bytes(captured_headers)
    ).hexdigest()
    fetch.update(
        {
            "query_request": query,
            "request_identity_sha256": query["canonical_url_utf8_sha256"],
            "query_count_preflight_request_sha256": count_request[
                "canonical_url_utf8_sha256"
            ],
            "query_count": 4,
            "feature_count": 4,
            "fetch_started_at_utc": query_started,
            "fetch_completed_at_utc": query_completed,
            "http_status": 200,
            "response_content_type": "application/json",
            "response_body_byte_count": 64,
            "geojson_parse_verified": True,
            "response_headers_sha256": response_headers_sha256,
        }
    )
    if "captured_response_headers" in fetch:
        fetch["captured_response_headers"] = captured_headers
    if "query_start_utc" in fetch:
        fetch["query_start_utc"] = start
    if "query_end_utc" in fetch:
        fetch["query_end_utc"] = end
    if "exchange_outcome" in fetch:
        fetch.update(
            {
                "exchange_outcome": "response_received",
                "outcome": "succeeded",
                "selected_as_truth_snapshot": True,
                "failure_code": None,
            }
        )
    if "raw_response_sha256" in fetch:
        fetch["raw_response_sha256"] = _sha256(f"{label}-query-body")
    if "raw_response" in fetch:
        raw_response_sha256 = _sha256(f"{label}-query-body")
        fetch["raw_response"].update(
            {
                "artifact_id": raw_response_sha256,
                "byte_count": 64,
                "file_sha256": raw_response_sha256,
                "local_restricted": True,
            }
        )
    preflight = fetch["count_preflight"]
    preflight.update(
        {
            "request": count_request,
            "fetch_started_at_utc": count_started,
            "fetch_completed_at_utc": count_completed,
            "http_status": 200,
            "response_content_type": "application/json",
            "response_body_byte_count": 64,
            "response_headers_sha256": _sha256(f"{label}-count-headers"),
            "raw_response_sha256": _sha256(f"{label}-count-body"),
            "geojson_parse_verified": True,
            "parsed_count": 4,
            "outcome": "succeeded",
            "failure_code": None,
        }
    )


def _bind_synthetic_formal_manifest(
    manifest: dict[str, Any],
    *,
    start: str,
    end: str,
) -> None:
    acquisition = manifest["source_acquisition"]
    _bind_complete_successful_fetch(
        acquisition,
        start=start,
        end=end,
        count_started="2029-03-15T15:58:00Z",
        count_completed="2029-03-15T15:58:01Z",
        query_started="2029-03-15T15:58:02Z",
        query_completed="2029-03-15T15:58:03Z",
        label="formal-freeze",
    )
    raw_response = acquisition["raw_response"]
    snapshot_preimage = {
        "profile": "stage2p_formal_freeze_source_snapshot_v1",
        "source_id": acquisition["source_id"],
        "canonical_query_request_sha256": acquisition[
            "request_identity_sha256"
        ],
        "response_headers_sha256": acquisition["response_headers_sha256"],
        "raw_response_sha256": raw_response["file_sha256"],
        "response_content_type": acquisition["response_content_type"],
        "response_body_byte_count": acquisition["response_body_byte_count"],
        "http_status": acquisition["http_status"],
        "fetch_started_at_utc": acquisition["fetch_started_at_utc"],
        "fetch_completed_at_utc": acquisition["fetch_completed_at_utc"],
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(
            snapshot_preimage,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest.update(
        {
            "status": "succeeded_single_full_cohort_response",
            "global_target_start_exclusive_utc": start,
            "global_target_end_inclusive_utc": end,
            "source_snapshot_sha256": snapshot_sha256,
            "snapshot_observed_at_utc": acquisition["fetch_completed_at_utc"],
            "freeze_completed_at_utc": "2029-03-15T15:59:00Z",
            "query_batch_count": 1,
            "ordered_query_request_sha256": [
                acquisition["request_identity_sha256"]
            ],
            "ordered_response_headers_sha256": [
                acquisition["response_headers_sha256"]
            ],
            "ordered_raw_response_sha256": [raw_response["file_sha256"]],
            "normalized_row_count": acquisition["feature_count"],
            "deduplicated_row_count": acquisition["feature_count"],
            "preferred_field_row_count": acquisition["feature_count"],
            "formal_window_binding_count": 1,
        }
    )
    _bind_self_hash(manifest, "manifest_sha256")


def _bind_selected_tsa(
    record: dict[str, Any],
    *,
    core_frozen: str,
    request_started: str,
    generated: str,
    response_received: str,
    deadline: str,
    label: str,
) -> None:
    record["record_core_frozen_at_utc"] = core_frozen
    record["timestamp_deadline_utc"] = deadline
    attempt = record["timestamp_attempt_evidence"][0]
    attempt.update(
        {
            "request_started_at_utc": request_started,
            "genTime_utc": generated,
            "response_received_at_utc": response_received,
            "attempt_completed_at_utc": response_received,
            "http_status": 200,
            "response_content_type": "application/timestamp-reply",
            "response_byte_count": 512,
            "response_sha256": _sha256(f"{label}-tsa-response"),
            "authority_identity_sha256": _sha256(f"{label}-tsa-authority"),
            "trust_chain_sha256": _sha256(f"{label}-tsa-chain"),
            "offline_trust_path_valid": True,
            "genTime_before_deadline": True,
            "outcome": "selected_valid",
        }
    )
    proof = record["remote_timestamp"]
    proof.update(
        {
            "deadline_utc": deadline,
            "selected_attempt_index": attempt["attempt_index"],
            "selected_response_sha256": attempt["response_sha256"],
        }
    )


def _evaluation_input(
    *,
    sequence: int,
    checkpoint: int,
    trigger: int,
    passed: bool,
    previous: str | None,
    content: str,
) -> dict[str, object]:
    return {
        "record_type": "EvaluationFreezeRecord",
        "evaluation_sequence": sequence,
        "phase": "input_freeze",
        "checkpoint_number": checkpoint,
        "trigger_reason": f"on_time_checkpoint_{trigger}",
        "scheduled_issue_count": trigger,
        "trigger_on_time_issue_count": trigger,
        "previous_evaluation_freeze_sha256": previous,
        "input_freeze_sha256": None,
        "sample_gate_met": passed,
        "evaluation_code_commit": "a" * 40,
        "evaluation_code_sha256": _sha256("evaluation-code"),
        "environment_lock_file_sha256": _sha256("lock"),
        "pyproject_file_sha256": _sha256("pyproject"),
        "content_sha256": content,
    }


def _evaluation_result(
    input_freeze: dict[str, object],
    *,
    sequence: int,
    content: str,
) -> dict[str, object]:
    input_sha = input_freeze["content_sha256"]
    return {
        **copy.deepcopy(input_freeze),
        "evaluation_sequence": sequence,
        "phase": "result_seal",
        "previous_evaluation_freeze_sha256": input_sha,
        "input_freeze_sha256": input_sha,
        "confirmatory_result": {
            "input_freeze_sha256": input_sha,
            "evaluation_code_commit": input_freeze["evaluation_code_commit"],
            "evaluation_code_sha256": input_freeze["evaluation_code_sha256"],
        },
        "content_sha256": content,
    }


def _evaluation_gate_record(*, passed: bool = True) -> dict[str, Any]:
    horizons = []
    for days in (7, 30, 90):
        horizon = _horizon()
        horizon["horizon_days"] = days
        horizons.append(horizon)
    record: dict[str, Any] = {
        "record_type": "EvaluationFreezeRecord",
        "phase": "input_freeze",
        "trigger_reason": "on_time_checkpoint_52",
        "all_90d_windows_mature": True,
        "union_unique_supported_M5_6_event_count": 20,
        "union_unique_full_study_area_M5_6_event_count": 20,
        "unique_target_cluster_count": 10,
        "realized_target_union": {
            "unique_event_count": 20,
        },
        "horizon_evaluability": horizons,
        "bootstrap_preflight": {
            "status": "passed" if passed else "zero_denominator",
        },
        "sample_gate_met": passed,
        "confirmatory_effects_authorized": passed,
        "status": (
            "confirmatory_effects_authorized"
            if passed
            else "continue_blind_to_104"
        ),
        "effect_rows_opened_at_utc": None,
    }
    _bind_record_content_hash(record)
    return record


def _confirmatory_result_record() -> dict[str, Any]:
    record = _evaluation_gate_record()
    record.update(
        {
            "phase": "result_seal",
            "status": "confirmatory_result_sealed",
            "input_freeze_sha256": _sha256("evaluation-input"),
            "effect_rows_opened_at_utc": "2026-09-09T16:01:00Z",
            "frozen_at_utc": "2026-09-09T16:03:00Z",
            "record_core_frozen_at_utc": "2026-09-09T16:03:00Z",
            "evaluation_code_commit": "a" * 40,
            "evaluation_code_sha256": _sha256("evaluation-code"),
        }
    )
    endpoints: dict[str, Any] = {}
    robustness: dict[str, Any] = {}
    for key in (
        "P1_minus_P0_macro_information_gain",
        "P1_minus_PP_macro_information_gain",
        "P1_minus_P0_macro_recall_gain",
        "P1_minus_PP_macro_recall_gain",
    ):
        endpoints[key] = {
            "endpoint_id": key,
            "point_estimate": 0.10,
            "familywise_lower_bound": 0.01,
            "familywise_upper_bound": 0.20,
            "familywise_lower_bound_gt_zero": True,
        }
        robustness[key] = {
            "endpoint_id": key,
            "target_cluster_count": 10,
            "largest_positive_region_identity_sha256": _sha256(f"{key}-region"),
            "point_estimate_after_largest_positive_region_removal": 0.05,
            "remains_positive_after_largest_positive_region_removal": True,
            "largest_positive_cluster_identity_sha256": _sha256(f"{key}-cluster"),
            "point_estimate_after_largest_positive_cluster_removal": 0.05,
            "remains_positive_after_largest_positive_cluster_removal": True,
        }
    record["confirmatory_result"] = {
        "input_freeze_sha256": record["input_freeze_sha256"],
        "evaluation_code_commit": record["evaluation_code_commit"],
        "evaluation_code_sha256": record["evaluation_code_sha256"],
        "execution_status": "valid",
        "endpoint_results": endpoints,
        "robustness_results": robustness,
        "all_four_familywise_lower_bounds_gt_zero": True,
        "P1_minus_P0_macro_recall_point_gain_gte_0_05": True,
        "all_four_region_removals_remain_positive": True,
        "all_four_cluster_removals_remain_positive": True,
        "formal_gate_passed": True,
        "decision": "pass_direct_improvement",
        "stop_action": "promote_P1_to_next_independent_validation_without_test_tuning",
        "additional_confirmatory_looks_authorized": False,
        "test_tuning_authorized": False,
        "sealed_at_utc": "2026-09-09T16:02:00Z",
        "result_bundle_sha256": _sha256("confirmatory-result-bundle"),
    }
    _bind_record_content_hash(record)
    return record


def _assert_code(code: str, callable_: Any, *args: object, **kwargs: object) -> None:
    with pytest.raises(SemanticValidationError) as caught:
        callable_(*args, **kwargs)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b'{"a":1,"a":2}', "duplicate_json_object_key"),
        (b'{"value":NaN}', "nonfinite_json_number"),
        (b'{"value":Infinity}', "nonfinite_json_number"),
        (b"\xff", "record_json_not_utf8"),
        (b"[]", "record_json_top_level_not_object"),
    ],
)
def test_original_record_bytes_are_parsed_fail_closed(
    raw: bytes,
    code: str,
) -> None:
    _assert_code(code, parse_record_json_bytes, raw)


def test_record_json_bytes_require_the_one_canonical_encoding() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a", "b"],
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
        },
    }
    record = {"a": "中/", "b": 1}
    canonical = canonical_json_bytes(record)
    assert validate_record_json_bytes(schema, canonical, _protocol()) == record

    noncanonical = (
        b'{ "a":"\xe4\xb8\xad/","b":1 }',
        b'{"b":1,"a":"\xe4\xb8\xad/"}',
        b'{"a":"\\u4e2d/","b":1}',
        b'{"a":"\\u4e2d\\/","b":1}',
        canonical + b"\n",
    )
    for raw in noncanonical:
        _assert_code(
            "record_json_bytes_not_canonical",
            validate_record_json_bytes,
            schema,
            raw,
            _protocol(),
        )


@pytest.mark.parametrize(
    "bits",
    [
        "7ff0000000000000",
        "fff0000000000000",
        "7ff8000000000000",
        "7ff0000000000001",
    ],
)
def test_float64_bits_hex_rejects_every_nonfinite_class(bits: str) -> None:
    _assert_code(
        "nonfinite_float64_bits",
        validate_record_semantics,
        {"score_float64_hex": bits},
        {},
    )


@pytest.mark.parametrize(
    "bits",
    [
        "0000000000000000",
        "8000000000000000",
        "3ff0000000000000",
        "bff0000000000000",
        "7fefffffffffffff",
    ],
)
def test_float64_bits_hex_accepts_finite_values(bits: str) -> None:
    validate_record_semantics({"score_float64_hex": bits}, {})


def test_public_stage2p_api_exports_the_lifecycle_gate() -> None:
    from seismoflux import stage2p

    assert stage2p.LIFECYCLE_IMPLEMENTATION_STATUS == "stage2p1b_required"
    assert stage2p.validate_prospective_lifecycle is validate_prospective_lifecycle
    assert stage2p.validate_record_json_bytes is validate_record_json_bytes


def test_trusted_release_1a_reaches_only_the_stage2p1b_required_gate() -> None:
    release, artifacts, schema, protocol, release_file_sha256 = (
        _trusted_release_1a()
    )
    _assert_code(
        "stage2p1b_required",
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )


def test_synthetically_accepted_release_reaches_a_distinct_unimplemented_gate() -> None:
    release, artifacts, schema, protocol, cohort, release_file_sha256 = (
        _trusted_release_accepted()
    )
    _assert_code(
        "stage2p1b_validator_not_implemented",
        validate_prospective_lifecycle,
        cohort,
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )


def test_trusted_release_requires_exact_canonical_manifest_file_bytes() -> None:
    release, artifacts, schema, protocol, release_file_sha256 = (
        _trusted_release_1a()
    )
    missing = dict(artifacts)
    del missing[release_file_sha256]
    _assert_code(
        "lifecycle_artifact_missing",
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=missing,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )

    noncanonical = canonical_json_bytes(release) + b"\n"
    noncanonical_sha256 = hashlib.sha256(noncanonical).hexdigest()
    artifacts[noncanonical_sha256] = noncanonical
    _assert_code(
        "trusted_release_manifest_bytes_not_canonical",
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=noncanonical_sha256,
        schema=schema,
        protocol=protocol,
    )

    duplicate = canonical_json_bytes(release).replace(
        b"{",
        b'{"profile":"stage2p_trusted_release_manifest_v1",',
        1,
    )
    duplicate_sha256 = hashlib.sha256(duplicate).hexdigest()
    artifacts[duplicate_sha256] = duplicate
    _assert_code(
        "duplicate_json_object_key",
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=duplicate_sha256,
        schema=schema,
        protocol=protocol,
    )

    mismatched_mapping = copy.deepcopy(release)
    mismatched_mapping["protocol_commit"] = "b" * 40
    _assert_code(
        "trusted_release_manifest_bytes_mapping_mismatch",
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=mismatched_mapping,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )


def test_trusted_release_recomputes_file_hash_role_and_component_path_contracts() -> None:
    release, artifacts, schema, protocol, _ = _trusted_release_1a()
    uv_identity = release["files"]["uv.lock"]
    uv_sha256 = uv_identity["file_sha256"]
    artifacts[uv_sha256] = b"tampered bytes under a claimed digest"
    release_file_sha256 = _install_release_bytes(release, artifacts)
    _assert_code(
        "lifecycle_artifact_hash_mismatch",
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )

    release, artifacts, schema, protocol, _ = _trusted_release_1a()
    release["files"]["uv.lock"]["commit_role"] = "code_commit"
    release_file_sha256 = _install_release_bytes(release, artifacts)
    _assert_code(
        "trusted_release_file_commit_role_mismatch",
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )

    for mutation in ("wrong_path", "missing_field"):
        release, artifacts, schema, protocol, _ = _trusted_release_1a()
        if mutation == "wrong_path":
            release["component_paths"]["parser_code_sha256"] = (
                "src/seismoflux/stage2p/validation.py"
            )
        else:
            del release["component_paths"]["parser_code_sha256"]
        release_file_sha256 = _install_release_bytes(release, artifacts)
        with pytest.raises(ValidationError):
            validate_prospective_lifecycle(
                {},
                [],
                [],
                [],
                [],
                artifacts_by_sha256=artifacts,
                trusted_release_manifest=release,
                trusted_release_manifest_file_sha256=release_file_sha256,
                schema=schema,
                protocol=protocol,
            )


@pytest.mark.parametrize(
    ("field", "code"),
    [
        (
            "expected_row_schema_sha256",
            "formal_table_registry_schema_identity_mismatch",
        ),
        (
            "expected_sort_order_sha256",
            "formal_table_registry_sort_identity_mismatch",
        ),
    ],
)
def test_trusted_protocol_bytes_freeze_formal_table_schema_and_sort_hashes(
    field: str,
    code: str,
) -> None:
    release, artifacts, schema, protocol, _ = _trusted_release_1a()
    protocol = copy.deepcopy(protocol)
    table = protocol["target_cohort"]["formal_freeze_source_manifest"][  # type: ignore[index]
        "derived_table_registry"
    ]["tables"]["normalized_rows"]
    table[field] = _sha256(f"tampered-{field}")  # type: ignore[index]
    protocol_raw = yaml.safe_dump(
        protocol,
        allow_unicode=True,
        sort_keys=False,
    ).encode()
    protocol_path = release["protocol_config_path"]
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    artifacts[protocol_sha256] = protocol_raw
    release["files"][protocol_path] = {
        "file_sha256": protocol_sha256,
        "git_blob_sha1": _git_blob_sha1(protocol_raw),
        "commit_role": "protocol_commit",
    }
    release_file_sha256 = _install_release_bytes(release, artifacts)
    _assert_code(
        code,
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("table_set", "artifact_profile_registry_table_set_mismatch"),
        (
            "row_schema_hash",
            "artifact_profile_registry_schema_identity_mismatch",
        ),
        (
            "sort_hash",
            "artifact_profile_registry_sort_identity_mismatch",
        ),
        (
            "typed_identity_const",
            "artifact_profile_registry_typed_identity_mismatch",
        ),
        (
            "manifest_schema_hash",
            "artifact_profile_registry_manifest_contract_mismatch",
        ),
        (
            "record_typed_ref",
            "artifact_profile_registry_schema_structure_mismatch",
        ),
        (
            "forecast_slot_ref",
            "artifact_profile_registry_identity_schema_hash_mismatch",
        ),
        (
            "effect_slot_ref",
            "artifact_profile_registry_manifest_contract_mismatch",
        ),
        (
            "result_slot_ref",
            "artifact_profile_registry_manifest_contract_mismatch",
        ),
        (
            "array_wrapper_ref",
            "artifact_profile_registry_identity_schema_hash_mismatch",
        ),
        (
            "identity_schema_set",
            "artifact_profile_registry_identity_schema_set_mismatch",
        ),
        (
            "selected_embedded_schema_hash",
            "artifact_profile_registry_selected_exposure_contract_mismatch",
        ),
    ],
)
def test_trusted_release_freezes_unified_artifact_registry_against_schema(
    mutation: str,
    code: str,
) -> None:
    release, artifacts, schema, protocol, _ = _trusted_release_1a()
    schema = copy.deepcopy(schema)
    protocol = copy.deepcopy(protocol)
    registry = protocol["artifact_profile_registry"]
    tables = registry["tables"]
    if mutation == "table_set":
        del tables["bootstrap_distribution_rows"]
    elif mutation == "row_schema_hash":
        tables["complete_grid_score_rows"]["expected_row_schema_sha256"] = _sha256(
            "tampered-row-schema"
        )
    elif mutation == "sort_hash":
        tables["complete_grid_score_rows"]["expected_sort_order_sha256"] = _sha256(
            "tampered-sort"
        )
    elif mutation == "typed_identity_const":
        schema["$defs"]["GridScoreRowsArtifactIdentity"]["allOf"][1]["properties"][
            "schema_sha256"
        ]["const"] = _sha256("tampered-typed-identity")
    elif mutation == "manifest_schema_hash":
        registry["manifests"]["forecast_bundle"]["expected_schema_sha256"] = (
            _sha256("tampered-manifest-schema")
        )
    elif mutation == "record_typed_ref":
        schema["$defs"]["SourceSnapshot"]["properties"]["normalized_rows"][
            "$ref"
        ] = "#/$defs/TableArtifactIdentity"
    elif mutation == "forecast_slot_ref":
        schema["$defs"]["ForecastArtifactSetIdentity"]["properties"]["score_rows"][
            "$ref"
        ] = "#/$defs/TableArtifactIdentity"
    elif mutation == "effect_slot_ref":
        effect_schema = schema["$defs"]["EffectRowsManifest"]
        effect_schema["properties"]["bootstrap_index_rows"]["$ref"] = (
            "#/$defs/TableArtifactIdentity"
        )
        registry["manifests"]["effect_rows"]["expected_schema_sha256"] = (
            hashlib.sha256(canonical_json_bytes(effect_schema)).hexdigest()
        )
    elif mutation == "result_slot_ref":
        result_schema = schema["$defs"]["ResultBundleManifest"]
        result_schema["properties"]["bootstrap_distribution_rows"]["oneOf"][0][
            "$ref"
        ] = "#/$defs/TableArtifactIdentity"
        registry["manifests"]["result_bundle"]["expected_schema_sha256"] = (
            hashlib.sha256(canonical_json_bytes(result_schema)).hexdigest()
        )
    elif mutation == "array_wrapper_ref":
        schema["$defs"]["MaskArrayArtifactIdentity"]["allOf"][0]["$ref"] = (
            "#/$defs/TableArtifactIdentity"
        )
    elif mutation == "identity_schema_set":
        del registry["identity_schemas"]["AlarmAreaComparison"]
    else:
        registry["embedded_manifests"]["selected_exposure"][
            "expected_schema_sha256"
        ] = _sha256("tampered-selected-exposure-schema")

    protocol_path = release["protocol_config_path"]
    protocol_raw = yaml.safe_dump(
        protocol,
        allow_unicode=True,
        sort_keys=False,
    ).encode()
    _replace_release_file_bytes(
        release,
        artifacts,
        path=protocol_path,
        raw=protocol_raw,
    )
    schema_path = release["record_schema_path"]
    schema_raw = json.dumps(schema, ensure_ascii=False, indent=2).encode() + b"\n"
    _replace_release_file_bytes(
        release,
        artifacts,
        path=schema_path,
        raw=schema_raw,
    )
    release_file_sha256 = _install_release_bytes(release, artifacts)
    _assert_code(
        code,
        validate_prospective_lifecycle,
        {},
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )


def test_accepted_release_protocol_mirror_must_match_exact_protocol_bytes() -> None:
    release, artifacts, schema, protocol, cohort, _ = (
        _trusted_release_accepted()
    )
    protocol_path = release["protocol_config_path"]
    release["code_commit_mirrored_protocol_files"][protocol_path][
        "git_blob_sha1"
    ] = "f" * 40
    release_file_sha256 = _install_release_bytes(release, artifacts)
    _assert_code(
        "trusted_release_protocol_mirror_identity_mismatch",
        validate_prospective_lifecycle,
        cohort,
        [],
        [],
        [],
        [],
        artifacts_by_sha256=artifacts,
        trusted_release_manifest=release,
        trusted_release_manifest_file_sha256=release_file_sha256,
        schema=schema,
        protocol=protocol,
    )


def test_formal_freeze_manifest_recomputes_single_exchange_hashes_and_lengths() -> None:
    manifest, protocol = _formal_freeze_manifest_with_registry()
    record = {
        "formal_freeze_source_manifest": manifest,
        "formal_freeze_source_manifest_sha256": manifest["manifest_sha256"],
        "record_core_frozen_at_utc": "2026-12-20T16:02:00Z",
        "effect_rows_opened_at_utc": "2026-12-20T16:02:00Z",
    }
    validate_record_semantics(record, protocol)

    mismatch = copy.deepcopy(record)
    mismatch["formal_freeze_source_manifest"]["ordered_raw_response_sha256"] = []
    _bind_self_hash(
        mismatch["formal_freeze_source_manifest"],
        "manifest_sha256",
    )
    mismatch["formal_freeze_source_manifest_sha256"] = mismatch[
        "formal_freeze_source_manifest"
    ]["manifest_sha256"]
    _assert_code(
        "formal_freeze_query_batch_count_mismatch",
        validate_record_semantics,
        mismatch,
        protocol,
    )

    wrong_snapshot = copy.deepcopy(record)
    wrong_snapshot["formal_freeze_source_manifest"][
        "source_snapshot_sha256"
    ] = _sha256("wrong-formal-source-snapshot")
    _bind_self_hash(
        wrong_snapshot["formal_freeze_source_manifest"],
        "manifest_sha256",
    )
    wrong_snapshot["formal_freeze_source_manifest_sha256"] = wrong_snapshot[
        "formal_freeze_source_manifest"
    ]["manifest_sha256"]
    _assert_code(
        "formal_freeze_source_snapshot_hash_mismatch",
        validate_record_semantics,
        wrong_snapshot,
        protocol,
    )


def test_formal_freeze_manifest_and_evaluation_timeline_are_hash_bound() -> None:
    manifest, protocol = _formal_freeze_manifest_with_registry()
    record = {
        "formal_freeze_source_manifest": manifest,
        "formal_freeze_source_manifest_sha256": _sha256("mixed-manifest"),
        "record_core_frozen_at_utc": "2026-12-20T16:02:00Z",
        "effect_rows_opened_at_utc": None,
    }
    _assert_code(
        "formal_freeze_manifest_top_level_hash_mismatch",
        validate_record_semantics,
        record,
        protocol,
    )

    record["formal_freeze_source_manifest_sha256"] = manifest["manifest_sha256"]
    record["record_core_frozen_at_utc"] = "2026-12-20T16:00:59Z"
    _assert_code(
        "evaluation_core_frozen_before_formal_freeze_completed",
        validate_record_semantics,
        record,
        protocol,
    )


def test_formal_success_binds_final_window_membership_exactly() -> None:
    manifest, protocol = _formal_freeze_manifest_with_registry()
    window_rows = manifest["window_membership_rows"]
    record = {
        "formal_freeze_status": "succeeded_single_full_cohort_response",
        "formal_freeze_source_manifest": manifest,
        "formal_freeze_source_manifest_sha256": manifest["manifest_sha256"],
        "final_window_membership_sha256": window_rows["content_sha256"],  # type: ignore[index]
        "record_core_frozen_at_utc": "2026-12-20T16:02:00Z",
        "effect_rows_opened_at_utc": None,
    }
    validate_record_semantics(record, protocol)

    mismatch = copy.deepcopy(record)
    mismatch["final_window_membership_sha256"] = _sha256(
        "wrong-final-window-membership"
    )
    _assert_code(
        "final_window_membership_hash_mismatch",
        validate_record_semantics,
        mismatch,
        protocol,
    )


def test_formal_manifest_requires_the_complete_protocol_table_registry() -> None:
    manifest, _ = _formal_freeze_manifest_with_registry()
    record = {
        "formal_freeze_source_manifest": manifest,
        "formal_freeze_source_manifest_sha256": manifest["manifest_sha256"],
        "record_core_frozen_at_utc": "2026-12-20T16:02:00Z",
        "effect_rows_opened_at_utc": None,
    }
    _assert_code(
        "formal_freeze_table_registry_required",
        validate_record_semantics,
        record,
        _protocol(),
    )
    _, incomplete_protocol = _formal_freeze_manifest_with_registry()
    del incomplete_protocol["target_cohort"]["formal_freeze_source_manifest"][  # type: ignore[index]
        "derived_table_registry"
    ]
    _assert_code(
        "formal_freeze_table_registry_required",
        validate_record_semantics,
        record,
        incomplete_protocol,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "normalized_rows",
        "deduplicated_rows",
        "preferred_field_rows",
        "window_membership_rows",
        "formal_window_target_bindings",
    ],
)
@pytest.mark.parametrize(
    ("identity_field", "code"),
    [
        ("artifact_id", "formal_freeze_table_artifact_id_mismatch"),
        ("table_role", "formal_freeze_table_registry_binding_mismatch"),
        ("serialization_profile", "formal_freeze_table_registry_binding_mismatch"),
        ("row_schema_ref", "formal_freeze_table_registry_binding_mismatch"),
        ("sort_profile", "formal_freeze_table_registry_binding_mismatch"),
        ("schema_sha256", "formal_freeze_table_registry_binding_mismatch"),
        ("sort_order_sha256", "formal_freeze_table_registry_binding_mismatch"),
    ],
)
def test_every_formal_table_identity_field_binds_the_protocol_registry(
    table_name: str,
    identity_field: str,
    code: str,
) -> None:
    manifest, protocol = _formal_freeze_manifest_with_registry()
    manifest[table_name][identity_field] = _sha256(  # type: ignore[index]
        f"tampered-{table_name}-{identity_field}"
    )
    _bind_self_hash(manifest, "manifest_sha256")
    record = {
        "formal_freeze_source_manifest": manifest,
        "formal_freeze_source_manifest_sha256": manifest["manifest_sha256"],
        "record_core_frozen_at_utc": "2026-12-20T16:02:00Z",
        "effect_rows_opened_at_utc": None,
    }
    _assert_code(code, validate_record_semantics, record, protocol)


@pytest.mark.parametrize(
    "status",
    [
        "not_run_no_complete_scope",
        "not_run_scheduled_issue_cap_terminal",
        "failed_count_preflight",
        "failed_count_limit",
    ],
)
def test_formal_freeze_failure_evidence_count_only_branches_are_valid(
    status: str,
) -> None:
    validate_record_semantics(_formal_failure_record(status), _protocol())


def test_no_complete_scope_rejects_nonempty_selected_exposure_rows() -> None:
    record = _no_complete_scope_with_empty_selected_exposure()
    validate_record_semantics(record, _protocol())

    nonempty = copy.deepcopy(record)
    prediction_seal_sha256 = _sha256("unexpected-complete-exposure")
    nonempty["ordered_issue_prediction_seal_sha256"] = [
        prediction_seal_sha256
    ]
    selected_manifest = nonempty["selected_exposure_manifest"]
    selected_manifest["candidate_issue_prediction_seal_sha256"] = [  # type: ignore[index]
        prediction_seal_sha256
    ]
    selected_manifest["rows"] = [  # type: ignore[index]
        {
            "horizon_days": 7,
            "selection_ordinal_1_based": 1,
            "scheduled_issue_sequence": 1,
            "issue_id": "stage2p-issue-20260910T000000+0800",
            "prediction_seal_sha256": prediction_seal_sha256,
            "issue_time_utc": "2026-09-09T16:00:00Z",
            "target_start_exclusive_utc": "2026-09-09T16:00:00Z",
            "target_end_inclusive_utc": "2026-09-16T16:00:00Z",
            "selected_truth_record_sha256": _sha256("unexpected-truth"),
            "selected_truth_revision_sequence": 0,
            "truth_record_status": "mature_truth_sealed",
            "truth_available": True,
        }
    ]
    _assert_code(
        "formal_no_complete_scope_contains_selected_exposure",
        validate_record_semantics,
        nonempty,
        _protocol(),
    )


def test_selected_exposure_manifest_rebuilds_axis_windows_and_projections() -> None:
    record = _selected_exposure_record()
    validate_record_semantics(record, _protocol())

    wrong_axis = copy.deepcopy(record)
    wrong_axis["ordered_issue_prediction_seal_sha256"] = [
        _sha256("different-candidate-axis")
    ]
    _assert_code(
        "selected_exposure_candidate_axis_mismatch",
        validate_record_semantics,
        wrong_axis,
        _protocol(),
    )

    overlapping = copy.deepcopy(record)
    first_row = overlapping["selected_exposure_manifest"]["rows"][0]  # type: ignore[index]
    second_seal = _sha256("overlapping-selected-exposure")
    overlapping["ordered_issue_prediction_seal_sha256"].append(  # type: ignore[union-attr]
        second_seal
    )
    overlapping["selected_exposure_manifest"][  # type: ignore[index]
        "candidate_issue_prediction_seal_sha256"
    ].append(second_seal)
    overlapping["selected_exposure_manifest"]["rows"].append(  # type: ignore[index,union-attr]
        {
            **first_row,
            "selection_ordinal_1_based": 2,
            "scheduled_issue_sequence": 2,
            "issue_id": "stage2p-issue-20260911T000000+0800",
            "prediction_seal_sha256": second_seal,
            "issue_time_utc": "2026-09-10T16:00:00Z",
            "target_start_exclusive_utc": "2026-09-10T16:00:00Z",
            "target_end_inclusive_utc": "2026-09-17T16:00:00Z",
        }
    )
    _assert_code(
        "selected_exposure_windows_overlap",
        validate_record_semantics,
        overlapping,
        _protocol(),
    )

    wrong_projection = copy.deepcopy(record)
    wrong_projection["horizon_evaluability"][0][  # type: ignore[index]
        "truth_availability_manifest_sha256"
    ] = _sha256("wrong-truth-projection")
    _assert_code(
        "selected_exposure_horizon_projection_mismatch",
        validate_record_semantics,
        wrong_projection,
        _protocol(),
    )

    wrong_manifest_hash = copy.deepcopy(record)
    wrong_manifest_hash["selected_exposure_manifest_union_sha256"] = _sha256(
        "wrong-selected-manifest"
    )
    _assert_code(
        "selected_exposure_manifest_hash_mismatch",
        validate_record_semantics,
        wrong_manifest_hash,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("entry_axis", "alarm_area_entry_axis_mismatch"),
        ("pairwise_maximum", "alarm_area_pairwise_maximum_mismatch"),
        ("threshold", "alarm_area_threshold_mismatch"),
        ("manifest_hash", "alarm_area_manifest_hash_mismatch"),
    ],
)
def test_alarm_area_manifest_rebuilds_every_selected_issue(
    mutation: str,
    code: str,
) -> None:
    record = _selected_exposure_record_with_alarm_area()
    validate_record_semantics(record, _protocol())
    mutated = copy.deepcopy(record)
    manifest = mutated["alarm_area_manifest"]
    if mutation == "entry_axis":
        manifest["entries"] = []  # type: ignore[index]
    elif mutation == "pairwise_maximum":
        manifest["entries"][0][  # type: ignore[index]
            (
                "maximum_pairwise_actual_alarm_area_difference_km2_"
                "float64_hex"
            )
        ] = struct.pack(">d", 399.0).hex()
    elif mutation == "threshold":
        manifest[  # type: ignore[index]
            "maximum_allowed_pairwise_difference_km2_float64_hex"
        ] = struct.pack(">d", 624.0).hex()
    else:
        mutated["alarm_area_manifest_sha256"] = _sha256(
            "wrong-alarm-area-manifest"
        )
    _assert_code(
        code,
        validate_record_semantics,
        mutated,
        _protocol(),
    )


def test_formal_freeze_failure_self_hash_status_and_fail_closed_state_are_bound() -> None:
    record = _formal_failure_record("failed_count_limit")
    record["formal_freeze_status"] = "failed_count_preflight"
    _assert_code(
        "formal_freeze_failure_status_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _formal_failure_record("failed_count_limit")
    record["sample_gate_met"] = True
    _assert_code(
        "formal_freeze_failure_not_fail_closed",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _formal_failure_record("failed_count_limit")
    record["formal_freeze_failure_evidence"]["failure_code"] = "tampered"  # type: ignore[index]
    _assert_code(
        "formal_freeze_failure_evidence_hash_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _formal_failure_record("failed_count_limit")
    record["union_unique_supported_M5_6_event_count"] = 0
    _assert_code(
        "formal_freeze_failure_contains_target_summary",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_local_derivation_failure_stage_is_closed_and_auditable() -> None:
    record = _formal_failure_record("failed_local_derivation_or_freeze")
    validate_record_semantics(record, _protocol())

    invalid_stage = copy.deepcopy(record)
    evidence = invalid_stage["formal_freeze_failure_evidence"]
    evidence["failure_stage"] = "query_fetch"  # type: ignore[index]
    _bind_self_hash(evidence, "failure_evidence_sha256")  # type: ignore[arg-type]
    invalid_stage["formal_freeze_failure_evidence_sha256"] = evidence[  # type: ignore[index]
        "failure_evidence_sha256"
    ]
    _assert_code(
        "formal_freeze_failure_stage_mismatch",
        validate_record_semantics,
        invalid_stage,
        _protocol(),
    )


def test_source_acquisition_recomputes_headers_body_and_content_length() -> None:
    schema = json.loads(
        (ROOT / "data" / "contracts" / "stage2p_prospective_records.json").read_text(
            encoding="utf-8"
        )
    )
    acquisition = _synthetic_schema_value(
        schema,
        schema["$defs"]["SourceAcquisition"],
        path="SourceAcquisition",
    )
    assert isinstance(acquisition, dict)
    _bind_complete_successful_fetch(
        acquisition,
        start="2026-09-09T16:00:00Z",
        end="2026-09-16T16:00:00Z",
        count_started="2026-10-16T16:00:00Z",
        count_completed="2026-10-16T16:00:01Z",
        query_started="2026-10-16T16:00:02Z",
        query_completed="2026-10-16T16:00:03Z",
        label="provenance",
    )
    validate_record_semantics(acquisition, _protocol())

    bad_headers = copy.deepcopy(acquisition)
    bad_headers["captured_response_headers"]["etag"] = "tampered"  # type: ignore[index]
    _assert_code(
        "captured_response_headers_hash_mismatch",
        validate_record_semantics,
        bad_headers,
        _protocol(),
    )

    bad_length = copy.deepcopy(acquisition)
    bad_length["captured_response_headers"]["content_length"] = "63"  # type: ignore[index]
    bad_length["response_headers_sha256"] = hashlib.sha256(
        canonical_json_bytes(bad_length["captured_response_headers"])
    ).hexdigest()
    _assert_code(
        "captured_content_length_body_mismatch",
        validate_record_semantics,
        bad_length,
        _protocol(),
    )

    bad_raw_size = copy.deepcopy(acquisition)
    bad_raw_size["raw_response"]["byte_count"] = 63  # type: ignore[index]
    _assert_code(
        "raw_response_body_byte_count_mismatch",
        validate_record_semantics,
        bad_raw_size,
        _protocol(),
    )

    wrong_artifact_id = copy.deepcopy(acquisition)
    wrong_artifact_id["raw_response"]["artifact_id"] = _sha256(  # type: ignore[index]
        "different-raw-artifact-id"
    )
    _assert_code(
        "raw_artifact_id_mismatch",
        validate_record_semantics,
        wrong_artifact_id,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("status", "outcome", "verified", "code"),
    [
        (
            204,
            "succeeded",
            True,
            "http_204_marked_geojson_parse_verified",
        ),
        (
            200,
            "succeeded",
            False,
            "http_200_success_without_geojson_parse_verification",
        ),
    ],
)
def test_http_body_parse_evidence_is_not_fabricated(
    status: int,
    outcome: str,
    verified: bool,
    code: str,
) -> None:
    record = {
        "http_status": status,
        "outcome": outcome,
        "geojson_parse_verified": verified,
    }
    _assert_code(code, validate_record_semantics, record, _protocol())


def test_valid_on_time_issue_passes_all_production_semantics() -> None:
    validate_record_semantics(_on_time_issue(), _protocol())


@pytest.mark.parametrize(
    ("record", "code"),
    [
        (_on_time_issue("2026-09-03T00:00:00+08:00"), "issue_before_first_issue_not_before"),
        (_on_time_issue("2026-09-11T00:00:00+08:00"), "issue_not_thursday"),
    ],
)
def test_issue_calendar_rejects_before_threshold_and_non_thursday(
    record: dict[str, object],
    code: str,
) -> None:
    _assert_code(code, validate_record_semantics, record, _protocol())


def test_issue_local_utc_and_q_are_cross_checked() -> None:
    record = _on_time_issue()
    record["issue_time_utc"] = "2026-09-09T16:00:01Z"
    _assert_code("issue_local_utc_mismatch", validate_record_semantics, record, _protocol())

    record = _on_time_issue()
    record["query_end_utc"] = "2026-09-09T15:44:59Z"
    _assert_code(
        "query_end_not_t_minus_15_minutes",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (
            ("attempt_evidence", 0, "query_end_utc"),
            "2026-09-09T16:00:01Z",
            "attempt_query_end_mismatch",
        ),
        (
            ("source_snapshot", "acquisition", "query_end_utc"),
            "2026-09-09T16:00:01Z",
            "acquisition_query_end_mismatch",
        ),
        (
            ("source_snapshot", "acquisition", "fetch_completed_at_utc"),
            "2026-09-09T16:00:00Z",
            "fetch_completed_not_before_issue",
        ),
        (
            ("source_snapshot", "seal_completed_at_utc"),
            "2026-09-09T16:00:00Z",
            "source_snapshot_sealed_not_before_issue",
        ),
        (
            ("prediction_seal", "valid_from_utc"),
            "2026-09-09T15:45:00Z",
            "prediction_valid_from_mismatch",
        ),
        (
            ("prediction_seal", "sealed_at_utc"),
            "2026-09-09T16:00:00Z",
            "prediction_sealed_not_before_issue",
        ),
    ],
)
def test_on_time_issue_rejects_nested_post_q_or_post_t_evidence(
    path: tuple[object, ...],
    value: object,
    code: str,
) -> None:
    record = _on_time_issue()
    target: Any = record
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    _assert_code(code, validate_record_semantics, record, _protocol())


def test_fetch_must_start_at_or_after_q() -> None:
    record = _on_time_issue()
    record["attempt_evidence"][0]["fetch_started_at_utc"] = "2026-09-09T15:44:59Z"  # type: ignore[index]
    _assert_code("fetch_started_before_query_end", validate_record_semantics, record, _protocol())


def test_rfc3161_uses_real_generation_time_not_self_reported_boolean() -> None:
    record = _on_time_issue()
    attempt = record["timestamp_attempt_evidence"][0]  # type: ignore[index]
    attempt["genTime_utc"] = record["timestamp_deadline_utc"]
    attempt["response_received_at_utc"] = record["timestamp_deadline_utc"]
    attempt["attempt_completed_at_utc"] = record["timestamp_deadline_utc"]
    attempt["genTime_before_deadline"] = True  # type: ignore[index]
    _assert_code(
        "tsa_generation_time_not_before_deadline",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_rfc3161_selected_attempt_and_response_are_bound() -> None:
    record = _on_time_issue()
    record["remote_timestamp"]["selected_response_sha256"] = _sha256("wrong")  # type: ignore[index]
    _assert_code("selected_tsa_response_mismatch", validate_record_semantics, record, _protocol())

    record = _on_time_issue()
    record["remote_timestamp"]["selected_attempt_index"] = 1  # type: ignore[index]
    _assert_code("selected_tsa_attempt_not_unique", validate_record_semantics, record, _protocol())


def test_fabricated_first_tsa_failure_cannot_open_second_authority() -> None:
    record = _on_time_issue()
    first = copy.deepcopy(record["timestamp_attempt_evidence"][0])  # type: ignore[index]
    first["outcome"] = "network_failure"
    second = copy.deepcopy(record["timestamp_attempt_evidence"][0])  # type: ignore[index]
    second["attempt_index"] = 1
    second["response_sha256"] = _sha256("second-tsa-response")
    record["timestamp_attempt_evidence"] = [first, second]
    record["remote_timestamp"]["selected_attempt_index"] = 1  # type: ignore[index]
    record["remote_timestamp"]["selected_response_sha256"] = second["response_sha256"]  # type: ignore[index]
    _assert_code(
        "tsa_network_failure_contains_response_evidence",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_all_tsa_attempts_have_bounded_completion_and_serial_authority_order() -> None:
    record = _on_time_issue()
    attempt = record["timestamp_attempt_evidence"][0]  # type: ignore[index]
    attempt["attempt_completed_at_utc"] = "2026-09-09T15:52:29Z"
    _assert_code(
        "tsa_attempt_completed_before_request",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _on_time_issue()
    first = copy.deepcopy(record["timestamp_attempt_evidence"][0])  # type: ignore[index]
    first.update(
        {
            "outcome": "network_failure",
            "http_status": None,
            "response_content_type": None,
            "response_byte_count": 0,
            "response_sha256": None,
            "authority_identity_sha256": None,
            "trust_chain_sha256": None,
            "genTime_utc": None,
            "response_received_at_utc": None,
            "offline_trust_path_valid": False,
            "genTime_before_deadline": False,
            "attempt_completed_at_utc": "2026-09-09T15:53:00Z",
        }
    )
    second = copy.deepcopy(record["timestamp_attempt_evidence"][0])  # type: ignore[index]
    second.update(
        {
            "attempt_index": 1,
            "request_started_at_utc": "2026-09-09T15:52:59Z",
            "response_sha256": _sha256("second-valid-tsa-response"),
        }
    )
    record["timestamp_attempt_evidence"] = [first, second]
    record["remote_timestamp"]["selected_attempt_index"] = 1  # type: ignore[index]
    record["remote_timestamp"]["selected_response_sha256"] = second[  # type: ignore[index]
        "response_sha256"
    ]
    _assert_code(
        "tsa_attempt_started_before_previous_completed",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_tsa_response_receipt_is_exact_attempt_completion() -> None:
    record = _on_time_issue()
    attempt = record["timestamp_attempt_evidence"][0]  # type: ignore[index]
    attempt["attempt_completed_at_utc"] = "2026-09-09T15:53:31Z"
    _assert_code(
        "tsa_response_receipt_not_attempt_completion",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_missed_issue_without_proof_still_validates_deadline_and_failed_attempt() -> None:
    record = _on_time_issue()
    attempt = record["timestamp_attempt_evidence"][0]  # type: ignore[index]
    candidate_sha256 = _sha256("failed-candidate-on-time-core")
    attempt.update(
        {
            "outcome": "network_failure",
            "http_status": None,
            "response_content_type": None,
            "response_byte_count": 0,
            "response_sha256": None,
            "authority_identity_sha256": None,
            "trust_chain_sha256": None,
            "genTime_utc": None,
            "response_received_at_utc": None,
            "offline_trust_path_valid": False,
            "genTime_before_deadline": False,
            "attempt_completed_at_utc": "2026-09-09T15:54:00Z",
            "attempt_preimage_sha256": candidate_sha256,
        }
    )
    source_sha256 = record["source_snapshot"]["snapshot_sha256"]  # type: ignore[index]
    prediction_sha256 = record["prediction_seal"]["prediction_seal_sha256"]  # type: ignore[index]
    record.update(
        {
            "status": "missed_issue",
            "on_time_issue_sequence": None,
            "remote_timestamp": None,
            "source_snapshot": None,
            "prediction_seal": None,
            "failure_code": "timestamp_failure",
            "failed_candidate_on_time_core": {
                "profile": "stage2p_failed_candidate_on_time_core_v1",
                "candidate_core_sha256": candidate_sha256,
                "candidate_core_artifact_sha256": candidate_sha256,
                "issue_id": record["issue_id"],
                "source_snapshot_sha256": source_sha256,
                "prediction_seal_sha256": prediction_sha256,
                "core_frozen_at_utc": "2026-09-09T15:52:00Z",
                "timestamp_deadline_utc": "2026-09-09T15:54:59Z",
                "local_restricted": True,
            },
            "record_core_frozen_at_utc": "2026-09-09T15:56:00Z",
            "timestamp_deadline_utc": "2026-09-09T15:54:59Z",
        }
    )
    _assert_code(
        "issue_tsa_deadline_not_t_minus_five_minutes",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record["timestamp_deadline_utc"] = "2026-09-09T15:55:00Z"
    record["failed_candidate_on_time_core"]["timestamp_deadline_utc"] = record[  # type: ignore[index]
        "timestamp_deadline_utc"
    ]
    attempt["attempt_completed_at_utc"] = "2026-09-09T15:55:01Z"
    _assert_code(
        "tsa_attempt_completed_after_deadline",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_issue_candidate_tsa_deadline_must_equal_t_minus_five_minutes() -> None:
    record = _on_time_issue()
    record["remote_timestamp"]["deadline_utc"] = "2026-09-09T15:59:59Z"  # type: ignore[index]
    _assert_code(
        "tsa_proof_deadline_top_level_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _on_time_issue()
    record["timestamp_deadline_utc"] = "2026-09-09T15:59:59Z"
    record["remote_timestamp"]["deadline_utc"] = "2026-09-09T15:59:59Z"  # type: ignore[index]
    _assert_code(
        "issue_tsa_deadline_not_t_minus_five_minutes",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda record: record.update({"backfill": True}),
            "record_core_preimage_hash_mismatch",
        ),
        (
            lambda record: record.update(
                {"previous_issue_record_sha256": _sha256("other-chain-parent")}
            ),
            "record_core_preimage_hash_mismatch",
        ),
        (
            lambda record: record["remote_timestamp"].update(  # type: ignore[union-attr]
                {"verification_code_sha256": _sha256("tampered-proof")}
            ),
            "record_content_hash_mismatch",
        ),
        (
            lambda record: record.update({"content_sha256": _sha256("tampered-content")}),
            "record_content_hash_mismatch",
        ),
    ],
)
def test_record_core_proof_content_and_chain_hashes_are_recomputed(
    mutation: Any,
    code: str,
) -> None:
    record = _on_time_issue()
    mutation(record)
    _assert_code(code, validate_record_semantics, record, _protocol())


def test_missed_issue_without_timestamp_proof_still_binds_complete_content() -> None:
    record = _on_time_issue()
    record.update(
        {
            "status": "missed_issue",
            "on_time_issue_sequence": None,
            "timestamp_attempt_evidence": [],
            "remote_timestamp": None,
            "source_snapshot": None,
            "prediction_seal": None,
            "failure_code": "timestamp_unavailable",
            "failed_candidate_on_time_core": None,
            "prediction_generated": False,
            "prediction_installed": False,
        }
    )
    _bind_missed_audit(record)
    validate_record_semantics(record, _protocol())

    record["failure_code"] = "tampered_after_seal"
    _assert_code(
        "missed_audit_core_preimage_hash_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    "completion_field",
    [
        "attempt_completed_at_utc",
        "count_completed_at_utc",
        "fetch_completed_at_utc",
        "query_completed_at_utc",
    ],
)
def test_missed_issue_final_core_cannot_precede_any_failed_completion(
    completion_field: str,
) -> None:
    record = _on_time_issue()
    record.update(
        {
            "status": "missed_issue",
            "on_time_issue_sequence": None,
            "timestamp_attempt_evidence": [],
            "remote_timestamp": None,
            "source_snapshot": None,
            "prediction_seal": None,
            "failure_code": "pre_candidate_failure",
            "failed_candidate_on_time_core": None,
            "prediction_generated": False,
            "prediction_installed": False,
            "failed_evidence": {
                completion_field: "2026-09-09T15:56:30Z",
            },
        }
    )
    _bind_missed_audit(record)
    _assert_code(
        "missed_issue_final_core_before_failed_evidence_completed",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda seal: seal["P0"].update({"alarm_prefix_cell_count": 101}),
            "alarm_prefix_exceeds_complete_grid",
        ),
        (
            lambda seal: seal["P0"].update(
                {"alarm_prefix_termination_reason": "domain_exhausted"}
            ),
            "alarm_prefix_termination_reason_inconsistent",
        ),
        (
            lambda seal: seal["P0"].update(
                {"alarm_prefix_termination_reason": "unfrozen_reason"}
            ),
            "alarm_prefix_termination_reason_invalid",
        ),
        (
            lambda seal: seal["P0"].update({"remaining_budget_km2": 249.0}),
            "alarm_area_budget_arithmetic_mismatch",
        ),
        (
            lambda seal: seal["P0"].update(
                {
                    "actual_alarm_area_km2": 600_001.0,
                    "remaining_budget_km2": -1.0,
                }
            ),
            "actual_alarm_area_exceeds_budget",
        ),
        (
            lambda seal: seal["P1"].update({"grid_identity_sha256": _sha256("other-grid")}),
            "forecast_grid_identity_mismatch",
        ),
    ],
)
def test_forecast_identity_semantics_reject_invalid_prefix_area_and_grid(
    mutation: Any,
    code: str,
) -> None:
    record = _on_time_issue()
    mutation(record["prediction_seal"])
    _assert_code(code, validate_record_semantics, record, _protocol())


def test_forecast_identity_accepts_domain_exhaustion_only_at_complete_ranking_end() -> None:
    record = _on_time_issue()
    seal = record["prediction_seal"]
    for model in ("P0", "P1", "PP"):
        seal[model].update(  # type: ignore[index,union-attr]
            {
                "alarm_prefix_cell_count": 100,
                "alarm_prefix_termination_reason": "domain_exhausted",
                "next_unselected_rank_position_1_based": None,
                "next_unselected_cell_id": None,
                "next_unselected_complete_cell_area_km2": None,
                "next_unselected_ranked_row_sha256": None,
            }
        )
    _bind_object_identity(
        seal,  # type: ignore[arg-type]
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _bind_record_hashes(record)  # type: ignore[arg-type]
    validate_record_semantics(record, _protocol())


def test_forecast_prefix_next_cell_and_pairwise_area_are_recomputed() -> None:
    record = _on_time_issue()
    record["prediction_seal"]["P0"][  # type: ignore[index]
        "next_unselected_rank_position_1_based"
    ] = 62
    _assert_code(
        "next_unselected_rank_position_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _on_time_issue()
    record["prediction_seal"]["P0"][  # type: ignore[index]
        "next_unselected_complete_cell_area_km2"
    ] = 200.0
    _assert_code(
        "next_unselected_cell_would_fit_budget",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _on_time_issue()
    seal = record["prediction_seal"]
    seal["RP30_event_count"] = 1  # type: ignore[index]
    seal["PP"].update(  # type: ignore[index,union-attr]
        {
            "actual_alarm_area_km2": 599_600.0,
            "remaining_budget_km2": 400.0,
        }
    )
    _assert_code(
        "maximum_pairwise_alarm_area_difference_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("model", "count", "code"),
    [
        ("P1", "R30_event_count", "empty_r30_forecast_not_byte_identical_to_p0"),
        ("PP", "RP30_event_count", "empty_rp30_forecast_not_byte_identical_to_p0"),
    ],
)
def test_empty_recent_windows_require_complete_forecast_byte_identity(
    model: str,
    count: str,
    code: str,
) -> None:
    record = _on_time_issue()
    seal = record["prediction_seal"]
    seal[count] = 0  # type: ignore[index]
    seal[model]["local_artifact_bundle_sha256"] = _sha256("different")  # type: ignore[index]
    _assert_code(code, validate_record_semantics, record, _protocol())


@pytest.mark.parametrize("worker_count", [5, 7])
def test_runtime_workers_use_the_exact_frozen_default(worker_count: int) -> None:
    record = _on_time_issue()
    record["prediction_seal"]["runtime_evidence"]["worker_count"] = worker_count  # type: ignore[index]
    _assert_code(
        "worker_count_not_frozen_default",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        (
            "host_available_memory_bytes_at_start",
            65 * 1024**3,
            "host_available_memory_exceeds_total",
        ),
        (
            "peak_resident_set_bytes",
            65 * 1024**3,
            "peak_resident_set_exceeds_total_memory",
        ),
    ],
)
def test_runtime_memory_evidence_respects_physical_host_bound(
    field: str,
    value: int,
    code: str,
) -> None:
    record = _on_time_issue()
    record["prediction_seal"]["runtime_evidence"][field] = value  # type: ignore[index]
    _assert_code(code, validate_record_semantics, record, _protocol())


def test_runtime_cpu_and_gpu_identity_branches_are_exclusive() -> None:
    record = _on_time_issue()
    runtime = record["prediction_seal"]["runtime_evidence"]  # type: ignore[index]
    runtime["gpu_model"] = "must-be-null-on-cpu"  # type: ignore[index]
    _assert_code(
        "cpu_runtime_contains_gpu_identity",
        validate_record_semantics,
        record,
        _protocol(),
    )

    record = _on_time_issue()
    runtime = record["prediction_seal"]["runtime_evidence"]  # type: ignore[index]
    runtime.update(  # type: ignore[union-attr]
        {
            "execution_device": "GPU_equivalent",
            "gpu_model": "synthetic-gpu",
            "gpu_driver_version": "1.0",
            "gpu_runtime_version": "1.0",
            "GPU_equivalence_receipt_sha256": _sha256("gpu-receipt"),
        }
    )
    _bind_object_identity(
        record["prediction_seal"],  # type: ignore[arg-type]
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _bind_record_hashes(record)  # type: ignore[arg-type]
    validate_record_semantics(record, _protocol())


def test_horizon_evaluability_is_recomputed_from_all_evidence() -> None:
    validate_record_semantics(_horizon(), _protocol())

    self_reported_false = _horizon()
    self_reported_false["evaluable"] = False
    self_reported_false["unevaluable_reason"] = "no_complete_exposure"
    _assert_code(
        "horizon_evaluable_not_derived_from_evidence",
        validate_record_semantics,
        self_reported_false,
        _protocol(),
    )

    truth_unavailable = _horizon()
    truth_unavailable.update(
        {
            "truth_available_complete_exposure_count": 0,
            "selected_truth_snapshot_unavailable_count": 4,
            "evaluable": False,
            "unevaluable_reason": "truth_unavailable",
        }
    )
    validate_record_semantics(truth_unavailable, _protocol())


@pytest.mark.parametrize(
    "reason",
    [
        "formal_freeze_unavailable",
        "scheduled_issue_cap_terminal_before_formal_freeze",
    ],
)
def test_formal_freeze_unavailable_horizon_keeps_only_exposure_evidence(
    reason: str,
) -> None:
    horizon = {
        **_horizon(),
        "supported_unique_M5_6_event_count": None,
        "full_study_area_unique_M5_6_event_count": None,
        "unique_target_cluster_count": None,
        "all_required_forecast_densities_finite_positive": None,
        "evaluable": False,
        "unevaluable_reason": reason,
    }
    validate_record_semantics(horizon, _protocol())

    false_zero = copy.deepcopy(horizon)
    false_zero["supported_unique_M5_6_event_count"] = 0
    _assert_code(
        "formal_freeze_unavailable_horizon_contains_derived_value",
        validate_record_semantics,
        false_zero,
        _protocol(),
    )


def test_unevaluable_reason_must_exist_and_match_failed_evidence() -> None:
    missing = _horizon()
    missing.update(
        {
            "supported_unique_M5_6_event_count": 0,
            "evaluable": False,
            "unevaluable_reason": None,
        }
    )
    _assert_code(
        "not_evaluable_reason_missing",
        validate_record_semantics,
        missing,
        _protocol(),
    )

    inconsistent = copy.deepcopy(missing)
    inconsistent["unevaluable_reason"] = "truth_unavailable"
    _assert_code(
        "not_evaluable_reason_inconsistent",
        validate_record_semantics,
        inconsistent,
        _protocol(),
    )


def test_result_seal_top_and_nested_input_hash_must_match() -> None:
    record = {
        "record_type": "EvaluationFreezeRecord",
        "phase": "result_seal",
        "input_freeze_sha256": _sha256("input"),
        "confirmatory_result": {
            "input_freeze_sha256": _sha256("other-input"),
        },
    }
    _assert_code(
        "confirmatory_result_input_freeze_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    "field",
    [
        "bootstrap_plan_sha256",
        "statistics_policy_sha256",
        "region_map_sha256",
    ],
)
def test_evaluation_policy_hashes_are_rebuilt_from_trusted_config(
    field: str,
) -> None:
    protocol = _full_protocol()
    evaluation = protocol["evaluation"]
    statistics_policy = {
        "profile": "stage2p_statistics_policy_v1",
        **{
            name: evaluation[name]  # type: ignore[index]
            for name in (
                "information_gain",
                "strict_recall",
                "macro_aggregation",
                "target_clusters",
                "simultaneous_inference",
                "pass_gate",
                "regional_robustness",
            )
        },
    }
    record = {
        "record_type": "EvaluationFreezeRecord",
        "bootstrap_plan_sha256": hashlib.sha256(
            canonical_json_bytes(evaluation["bootstrap"])  # type: ignore[index]
        ).hexdigest(),
        "statistics_policy_sha256": hashlib.sha256(
            canonical_json_bytes(statistics_policy)
        ).hexdigest(),
        "region_map_sha256": evaluation["regional_robustness"][  # type: ignore[index]
            "region_manifest_file_sha256"
        ],
    }
    _validate_evaluation_policy_hashes(record, protocol)
    record[field] = _sha256(f"tampered-{field}")
    _assert_code(
        "evaluation_policy_hash_mismatch",
        _validate_evaluation_policy_hashes,
        record,
        protocol,
    )


def test_valid_evaluation_chains_cover_52_and_104_paths() -> None:
    first = _evaluation_input(
        sequence=1,
        checkpoint=1,
        trigger=52,
        passed=True,
        previous=None,
        content=_sha256("input-52"),
    )
    validate_evaluation_chain(
        [first, _evaluation_result(first, sequence=2, content=_sha256("result-52"))]
    )

    failed_52 = _evaluation_input(
        sequence=1,
        checkpoint=1,
        trigger=52,
        passed=False,
        previous=None,
        content=_sha256("failed-52"),
    )
    passed_104 = _evaluation_input(
        sequence=2,
        checkpoint=2,
        trigger=104,
        passed=True,
        previous=failed_52["content_sha256"],  # type: ignore[arg-type]
        content=_sha256("input-104"),
    )
    validate_evaluation_chain(
        [
            failed_52,
            passed_104,
            _evaluation_result(passed_104, sequence=3, content=_sha256("result-104")),
        ]
    )


def test_104_checkpoint_may_use_new_formal_and_cluster_freezes_without_rewriting_52() -> None:
    failed_52 = _evaluation_input(
        sequence=1,
        checkpoint=1,
        trigger=52,
        passed=False,
        previous=None,
        content=_sha256("immutable-input-52"),
    )
    failed_52.update(
        {
            "formal_freeze_source_manifest_sha256": _sha256("formal-52"),
            "cluster_membership_rows_sha256": _sha256("clusters-52"),
            "ordered_truth_record_sha256": [
                _sha256("truth-before-cluster-merge"),
            ],
        }
    )
    frozen_52 = copy.deepcopy(failed_52)
    passed_104 = _evaluation_input(
        sequence=2,
        checkpoint=2,
        trigger=104,
        passed=True,
        previous=failed_52["content_sha256"],  # type: ignore[arg-type]
        content=_sha256("input-104-after-cluster-merge"),
    )
    passed_104.update(
        {
            "formal_freeze_source_manifest_sha256": _sha256("formal-104"),
            "cluster_membership_rows_sha256": _sha256("clusters-104"),
            "ordered_truth_record_sha256": [
                _sha256("truth-before-cluster-merge"),
            ],
        }
    )
    validate_evaluation_chain(
        [
            failed_52,
            passed_104,
            _evaluation_result(
                passed_104,
                sequence=3,
                content=_sha256("result-104-after-cluster-merge"),
            ),
        ]
    )
    assert failed_52 == frozen_52
    assert (
        failed_52["cluster_membership_rows_sha256"]
        != passed_104["cluster_membership_rows_sha256"]
    )


@pytest.mark.parametrize(
    ("records", "code"),
    [
        (
            [
                _evaluation_input(
                    sequence=2,
                    checkpoint=1,
                    trigger=52,
                    passed=False,
                    previous=None,
                    content=_sha256("a"),
                )
            ],
            "evaluation_sequence_not_contiguous",
        ),
        (
            [
                _evaluation_input(
                    sequence=1,
                    checkpoint=1,
                    trigger=52,
                    passed=False,
                    previous=_sha256("wrong"),
                    content=_sha256("a"),
                )
            ],
            "previous_evaluation_hash_mismatch",
        ),
        (
            [
                _evaluation_input(
                    sequence=1,
                    checkpoint=2,
                    trigger=104,
                    passed=False,
                    previous=None,
                    content=_sha256("a"),
                )
            ],
            "checkpoint_104_without_failed_52_gate",
        ),
    ],
)
def test_evaluation_chain_rejects_sequence_hash_and_unearned_104(
    records: list[dict[str, object]],
    code: str,
) -> None:
    _assert_code(code, validate_evaluation_chain, records)


def test_52_pass_forbids_104_and_result_must_be_immediate() -> None:
    passed_52 = _evaluation_input(
        sequence=1,
        checkpoint=1,
        trigger=52,
        passed=True,
        previous=None,
        content=_sha256("passed-52"),
    )
    checkpoint_104 = _evaluation_input(
        sequence=2,
        checkpoint=2,
        trigger=104,
        passed=True,
        previous=passed_52["content_sha256"],  # type: ignore[arg-type]
        content=_sha256("input-104"),
    )
    _assert_code(
        "checkpoint_104_without_failed_52_gate",
        validate_evaluation_chain,
        [passed_52, checkpoint_104],
    )

    orphan = _evaluation_result(passed_52, sequence=1, content=_sha256("orphan"))
    orphan["previous_evaluation_freeze_sha256"] = None
    _assert_code(
        "result_seal_not_immediately_after_input_freeze",
        validate_evaluation_chain,
        [orphan],
    )


def test_second_result_and_changed_execution_identity_are_forbidden() -> None:
    freeze = _evaluation_input(
        sequence=1,
        checkpoint=1,
        trigger=52,
        passed=True,
        previous=None,
        content=_sha256("input"),
    )
    result = _evaluation_result(freeze, sequence=2, content=_sha256("result"))
    changed = _evaluation_result(freeze, sequence=2, content=_sha256("changed"))
    changed["evaluation_code_sha256"] = _sha256("changed-code")
    _assert_code(
        "result_seal_execution_identity_changed",
        validate_evaluation_chain,
        [freeze, changed],
    )

    second = copy.deepcopy(result)
    second["evaluation_sequence"] = 3
    second["previous_evaluation_freeze_sha256"] = result["content_sha256"]
    _assert_code(
        "second_result_seal_forbidden",
        validate_evaluation_chain,
        [freeze, result, second],
    )


def test_result_seal_cannot_change_any_non_whitelisted_frozen_input() -> None:
    freeze = _evaluation_input(
        sequence=1,
        checkpoint=1,
        trigger=52,
        passed=True,
        previous=None,
        content=_sha256("input"),
    )
    freeze["region_map_sha256"] = _sha256("frozen-region-map")
    result = _evaluation_result(freeze, sequence=2, content=_sha256("result"))
    result["region_map_sha256"] = _sha256("changed-region-map")
    _assert_code(
        "result_seal_frozen_input_changed",
        validate_evaluation_chain,
        [freeze, result],
    )


def test_sample_gate_status_and_authorization_are_mechanically_derived() -> None:
    validate_record_semantics(_evaluation_gate_record(), _protocol())

    fake_not_run = _evaluation_gate_record()
    fake_not_run["bootstrap_preflight"]["status"] = "not_run_basic_gate_failed"
    _assert_code(
        "bootstrap_not_run_despite_satisfied_basic_gate",
        validate_record_semantics,
        fake_not_run,
        _protocol(),
    )

    fake_supported = _evaluation_gate_record()
    fake_supported["union_unique_supported_M5_6_event_count"] = 21
    _assert_code(
        "evaluation_supported_count_exceeds_full_count",
        validate_record_semantics,
        fake_supported,
        _protocol(),
    )

    fake_clusters = _evaluation_gate_record()
    fake_clusters["unique_target_cluster_count"] = 21
    _assert_code(
        "evaluation_cluster_count_exceeds_full_count",
        validate_record_semantics,
        fake_clusters,
        _protocol(),
    )

    false_self_report = _evaluation_gate_record()
    false_self_report["sample_gate_met"] = False
    _assert_code(
        "sample_gate_self_report_mismatch",
        validate_record_semantics,
        false_self_report,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("formal_status", "trigger_reason", "reason", "expected_status"),
    [
        (
            "failed_query_fetch",
            "on_time_checkpoint_52",
            "formal_freeze_unavailable",
            "continue_blind_to_104",
        ),
        (
            "failed_local_derivation_or_freeze",
            "on_time_checkpoint_104",
            "formal_freeze_unavailable",
            "evidence_insufficient",
        ),
        (
            "not_run_scheduled_issue_cap_terminal",
            "scheduled_issue_cap_130",
            "scheduled_issue_cap_terminal_before_formal_freeze",
            "evidence_insufficient",
        ),
    ],
)
def test_formal_freeze_unavailable_sample_gate_is_null_and_fail_closed(
    formal_status: str,
    trigger_reason: str,
    reason: str,
    expected_status: str,
) -> None:
    horizons = [
        {
            **_horizon(),
            "horizon_days": days,
            "supported_unique_M5_6_event_count": None,
            "full_study_area_unique_M5_6_event_count": None,
            "unique_target_cluster_count": None,
            "all_required_forecast_densities_finite_positive": None,
            "evaluable": False,
            "unevaluable_reason": reason,
        }
        for days in (7, 30, 90)
    ]
    record = {
        "record_type": "EvaluationFreezeRecord",
        "formal_freeze_status": formal_status,
        "phase": "input_freeze",
        "trigger_reason": trigger_reason,
        "horizon_evaluability": horizons,
        "union_unique_supported_M5_6_event_count": None,
        "union_unique_full_study_area_M5_6_event_count": None,
        "unique_target_cluster_count": None,
        "realized_target_union": None,
        "final_window_membership_sha256": None,
        "cluster_membership_sha256": None,
        "bootstrap_preflight": {
            "status": "not_run_formal_freeze_unavailable",
            "index_matrix_sha256": None,
            "generated_replication_count": 0,
            "zero_denominator_replication_count": None,
            "frozen_before_effect_rows_open": False,
            "redraw_or_discard_performed": False,
        },
        "sample_gate_met": False,
        "confirmatory_effects_authorized": False,
        "effect_rows_opened_at_utc": None,
        "confirmatory_result": None,
        "input_freeze_sha256": None,
        "status": expected_status,
    }
    _validate_evaluation_sample_gate(record)

    false_zero = copy.deepcopy(record)
    false_zero["union_unique_supported_M5_6_event_count"] = 0
    _assert_code(
        "formal_freeze_unavailable_contains_derived_result",
        _validate_evaluation_sample_gate,
        false_zero,
    )

    wrong_bootstrap = copy.deepcopy(record)
    wrong_bootstrap["bootstrap_preflight"]["status"] = (
        "not_run_basic_gate_failed"
    )
    _assert_code(
        "formal_freeze_unavailable_bootstrap_not_fail_closed",
        _validate_evaluation_sample_gate,
        wrong_bootstrap,
    )


def test_confirmatory_endpoint_intervals_identities_and_booleans_fail_closed() -> None:
    validate_record_semantics(_confirmatory_result_record(), _protocol())

    inverted = _confirmatory_result_record()
    endpoint = inverted["confirmatory_result"]["endpoint_results"][  # type: ignore[index]
        "P1_minus_P0_macro_information_gain"
    ]
    endpoint["familywise_lower_bound"] = 0.15
    _assert_code(
        "confirmatory_endpoint_interval_inverted",
        validate_record_semantics,
        inverted,
        _protocol(),
    )

    missing_identity = _confirmatory_result_record()
    robust = missing_identity["confirmatory_result"]["robustness_results"][  # type: ignore[index]
        "P1_minus_P0_macro_information_gain"
    ]
    robust["largest_positive_region_identity_sha256"] = None
    _assert_code(
        "largest_positive_contribution_identity_missing",
        validate_record_semantics,
        missing_identity,
        _protocol(),
    )

    false_boolean = _confirmatory_result_record()
    false_boolean["confirmatory_result"][  # type: ignore[index]
        "all_four_familywise_lower_bounds_gt_zero"
    ] = False
    _assert_code(
        "confirmatory_summary_boolean_mismatch",
        validate_record_semantics,
        false_boolean,
        _protocol(),
    )

    wrong_cluster_count = _confirmatory_result_record()
    wrong_cluster_count["confirmatory_result"]["robustness_results"][  # type: ignore[index]
        "P1_minus_P0_macro_information_gain"
    ]["target_cluster_count"] = 11
    _assert_code(
        "robustness_target_cluster_count_mismatch",
        validate_record_semantics,
        wrong_cluster_count,
        _protocol(),
    )


def test_issue_chain_is_weekly_hash_linked_and_id_derived() -> None:
    first = _on_time_issue(content_sha256=_sha256("first"))
    second = _on_time_issue(
        "2026-09-17T00:00:00+08:00",
        scheduled_sequence=2,
        on_time_sequence=2,
        previous_sha256=first["content_sha256"],  # type: ignore[arg-type]
        content_sha256=_sha256("second"),
    )
    validate_issue_chain([first, second], _protocol())

    skipped = _on_time_issue(
        "2026-09-24T00:00:00+08:00",
        scheduled_sequence=2,
        on_time_sequence=2,
        previous_sha256=first["content_sha256"],  # type: ignore[arg-type]
    )
    _assert_code("issue_weekly_cadence_broken", validate_issue_chain, [first, skipped], _protocol())

    wrong_id = _on_time_issue()
    wrong_id["issue_id"] = "stage2p-issue-20260917T000000+0800"
    _bind_record_hashes(wrong_id)  # type: ignore[arg-type]
    _assert_code(
        "issue_id_not_derived_from_local_time",
        validate_issue_chain,
        [wrong_id],
        _protocol(),
    )


def _cohort() -> dict[str, object]:
    cohort: dict[str, object] = {
        "record_type": "TargetCohortDefinition",
        "protocol_commit": "a" * 40,
        "code_commit": "b" * 40,
        "definition_frozen_at_utc": "2026-09-02T11:30:00Z",
        "valid_from_utc": "2026-09-09T16:00:00Z",
        "first_issue_not_before_utc": "2026-09-09T16:00:00Z",
        "protocol_tag_remote_receipt": {
            "remote_peeled_commit": "a" * 40,
            "verified_at_utc": "2026-09-02T10:00:00Z",
        },
        "code_tag_remote_receipt": {
            "remote_peeled_commit": "b" * 40,
            "verified_at_utc": "2026-09-02T11:00:00Z",
        },
    }
    for name in ("protocol_tag_remote_receipt", "code_tag_remote_receipt"):
        _bind_self_hash(cohort[name], "receipt_sha256")  # type: ignore[arg-type]
    _bind_record_content_hash(cohort)  # type: ignore[arg-type]
    return cohort


def test_cohort_tag_receipts_bind_commits_and_first_issue_activation() -> None:
    cohort = _cohort()
    validate_record_semantics(cohort, _protocol())
    validate_issue_chain(
        [_on_time_issue()],
        _protocol(),
        cohort_definition=cohort,
    )

    wrong_commit = _cohort()
    wrong_commit["code_tag_remote_receipt"]["remote_peeled_commit"] = "c" * 40  # type: ignore[index]
    _bind_self_hash(
        wrong_commit["code_tag_remote_receipt"],  # type: ignore[arg-type]
        "receipt_sha256",
    )
    _assert_code(
        "code_tag_peeled_commit_mismatch",
        validate_record_semantics,
        wrong_commit,
        _protocol(),
    )

    late = _cohort()
    late["protocol_tag_remote_receipt"]["verified_at_utc"] = late["valid_from_utc"]  # type: ignore[index]
    _bind_self_hash(
        late["protocol_tag_remote_receipt"],  # type: ignore[arg-type]
        "receipt_sha256",
    )
    _assert_code(
        "remote_tag_verified_not_before_valid_from",
        validate_record_semantics,
        late,
        _protocol(),
    )

    tampered_receipt = _cohort()
    tampered_receipt["protocol_tag_remote_receipt"]["verification_response_sha256"] = (  # type: ignore[index]
        _sha256("tampered-receipt")
    )
    _assert_code(
        "remote_tag_receipt_hash_mismatch",
        validate_record_semantics,
        tampered_receipt,
        _protocol(),
    )

    receipt_after_definition = _cohort()
    receipt_after_definition["definition_frozen_at_utc"] = "2026-09-02T10:30:00Z"
    _bind_record_content_hash(receipt_after_definition)  # type: ignore[arg-type]
    _assert_code(
        "target_definition_frozen_before_tag_receipt",
        validate_record_semantics,
        receipt_after_definition,
        _protocol(),
    )


@pytest.mark.parametrize(
    "invalid_valid_from",
    [
        "2026-09-16T16:00:00Z",
        "2026-09-02T16:00:00Z",
        "2026-09-10T12:34:56Z",
    ],
    ids=["plus-seven-days", "minus-seven-days", "arbitrary-delay"],
)
def test_cohort_valid_from_allows_only_the_exact_mechanical_first_issue(
    invalid_valid_from: str,
) -> None:
    cohort = _cohort()
    cohort["valid_from_utc"] = invalid_valid_from
    _bind_record_content_hash(cohort)  # type: ignore[arg-type]
    _assert_code(
        "target_valid_from_not_mechanically_derived",
        validate_record_semantics,
        cohort,
        _protocol(),
    )


def test_first_issue_cannot_skip_the_exact_derived_activation_thursday() -> None:
    cohort = _cohort()
    skipped = _on_time_issue("2026-09-17T00:00:00+08:00")
    _assert_code(
        "first_issue_not_first_rule_after_activation",
        validate_issue_chain,
        [skipped],
        _protocol(),
        cohort_definition=cohort,
    )


def test_fetch_exchange_outcome_must_match_network_http_and_final_outcome() -> None:
    record = _on_time_issue()
    record["attempt_evidence"][0]["exchange_outcome"] = "network_failure"  # type: ignore[index]
    _assert_code("exchange_outcome_inconsistent", validate_record_semantics, record, _protocol())

    record = _on_time_issue()
    record["attempt_evidence"][0]["http_status"] = 503  # type: ignore[index]
    record["source_snapshot"]["acquisition"]["http_status"] = 503  # type: ignore[index]
    _assert_code(
        "successful_fetch_http_status_inconsistent",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_fdsn_requests_are_recomputed_and_count_preflight_binds_query_count() -> None:
    fetch = _fdsn_fetch()
    validate_record_semantics(fetch, _protocol())

    bad_hash = copy.deepcopy(fetch)
    bad_hash["query_request"]["canonical_url_utf8_sha256"] = _sha256("invented")  # type: ignore[index]
    _assert_code(
        "fdsn_canonical_url_hash_mismatch",
        validate_record_semantics,
        bad_hash,
        _protocol(),
    )

    bad_count = copy.deepcopy(fetch)
    bad_count["count_preflight"]["parsed_count"] = 5  # type: ignore[index]
    _assert_code(
        "count_preflight_query_count_mismatch",
        validate_record_semantics,
        bad_count,
        _protocol(),
    )


def test_fdsn_count_limit_prevents_query_exchange() -> None:
    fetch = _fdsn_fetch()
    fetch["count_preflight"]["parsed_count"] = 20_000  # type: ignore[index]
    fetch["query_count"] = 20_000
    _assert_code(
        "query_executed_after_count_limit_reached",
        validate_record_semantics,
        fetch,
        _protocol(),
    )


def test_mature_truth_timeline_binds_issue_horizon_due_attempt_and_snapshot() -> None:
    record = _mature_truth_record()
    validate_record_semantics(record, _protocol())

    wrong_due = copy.deepcopy(record)
    wrong_due["maturity_due_at_utc"] = "2026-10-16T16:00:01Z"
    _assert_code(
        "truth_record_maturity_due_mismatch",
        validate_record_semantics,
        wrong_due,
        _protocol(),
    )

    wrong_window = copy.deepcopy(record)
    wrong_window["attempt_evidence"][0]["target_end_inclusive_utc"] = (  # type: ignore[index]
        "2026-09-16T16:00:01Z"
    )
    _assert_code(
        "truth_attempt_target_end_mismatch",
        validate_record_semantics,
        wrong_window,
        _protocol(),
    )

    early_fetch = copy.deepcopy(record)
    early_fetch["attempt_evidence"][0]["fetch_started_at_utc"] = (  # type: ignore[index]
        "2026-10-16T15:59:59Z"
    )
    _assert_code(
        "truth_fetch_started_before_schedule",
        validate_record_semantics,
        early_fetch,
        _protocol(),
    )

    wrong_snapshot = copy.deepcopy(record)
    wrong_snapshot["truth_snapshot"]["target_start_exclusive_utc"] = (  # type: ignore[index]
        "2026-09-09T15:59:59Z"
    )
    _assert_code(
        "truth_snapshot_target_start_mismatch",
        validate_record_semantics,
        wrong_snapshot,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("record_factory", "object_field", "body_field", "value", "code"),
    [
        (
            _on_time_issue,
            "source_snapshot",
            "seal_completed_at_utc",
            "2026-09-09T15:50:30Z",
            "source_snapshot_hash_mismatch",
        ),
        (
            _on_time_issue,
            "prediction_seal",
            "sealed_at_utc",
            "2026-09-09T15:51:30Z",
            "prediction_seal_hash_mismatch",
        ),
        (
            _mature_truth_record,
            "truth_snapshot",
            "truth_snapshot_id",
            _sha256("different-truth-id"),
            "truth_snapshot_id_hash_mismatch",
        ),
    ],
)
def test_embedded_object_identities_recompute_fixed_exclusion_formulas(
    record_factory: Any,
    object_field: str,
    body_field: str,
    value: object,
    code: str,
) -> None:
    record = record_factory()
    record[object_field][body_field] = value
    _assert_code(code, validate_record_semantics, record, _protocol())


def test_prediction_seal_binds_the_enclosing_source_snapshot_hash() -> None:
    record = _on_time_issue()
    seal = record["prediction_seal"]
    seal["source_snapshot_sha256"] = _sha256("other-source")
    _bind_object_identity(
        seal,
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _assert_code(
        "prediction_source_snapshot_hash_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_prediction_visualization_evidence_hash_is_recomputed() -> None:
    record = _on_time_issue()
    record["prediction_seal"]["visualization_evidence"]["static_svg_sha256"] = (  # type: ignore[index]
        _sha256("tampered-svg")
    )
    _assert_code(
        "visualization_evidence_hash_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )


def _typed_source_snapshot_record() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "snapshot_id": _sha256("pending-typed-source"),
        "snapshot_sha256": _sha256("pending-typed-source"),
        "normalized_rows": _typed_table_artifact(
            "source-normalized",
            role="source_normalized_rows",
            row_schema_ref="#/$defs/FormalNormalizedRow",
            sort_profile="stage2p_source_normalized_rows_sort_v1",
        ),
        "deduplicated_rows": _typed_table_artifact(
            "source-deduplicated",
            role="source_deduplicated_rows",
            row_schema_ref="#/$defs/FormalDeduplicatedRow",
            sort_profile="stage2p_source_deduplicated_rows_sort_v1",
        ),
        "causal_model_view": _typed_table_artifact(
            "causal-view",
            role="causal_model_view_rows",
            row_schema_ref="#/$defs/CausalModelViewRow",
            sort_profile="stage2p_causal_model_view_rows_sort_v1",
        ),
        "cutover_cross_source_match_rows": _typed_table_artifact(
            "cutover-matches",
            role="cutover_match_rows",
            row_schema_ref="#/$defs/CutoverMatchRow",
            sort_profile="stage2p_cutover_match_rows_sort_v1",
        ),
    }
    cutover = snapshot["cutover_cross_source_match_rows"]
    snapshot["cutover_cross_source_match_count"] = cutover["row_count"]
    snapshot["cutover_cross_source_match_sha256"] = cutover["content_sha256"]
    for component in ("P0", "R30", "RP30"):
        rows = _typed_table_artifact(
            f"{component}-events",
            role=f"{component}_event_rows",
            row_schema_ref="#/$defs/EventSetRow",
            sort_profile="stage2p_event_set_rows_sort_v1",
        )
        snapshot[f"{component}_event_set"] = {
            "component_role": component,
            "event_count": rows["row_count"],
            "ordered_event_ids_sha256": _sha256(f"{component}-ordered"),
            "maximum_origin_time_utc": None,
            "maximum_available_at_utc": None,
            "event_rows_content_sha256": rows["content_sha256"],
            "event_rows": rows,
        }
    _bind_object_identity(
        snapshot,
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    return {"source_snapshot": snapshot}


def _typed_truth_snapshot_record() -> dict[str, Any]:
    membership = _typed_table_artifact(
        "truth-membership",
        role="truth_window_membership_rows",
        row_schema_ref="#/$defs/TruthWindowMembershipRow",
        sort_profile="stage2p_truth_window_membership_rows_sort_v1",
    )
    target_rows = _typed_table_artifact(
        "truth-targets",
        role="scientific_target_rows",
        row_schema_ref="#/$defs/FormalScientificTargetRow",
        sort_profile="stage2p_scientific_target_rows_sort_v1",
    )
    snapshot: dict[str, Any] = {
        "truth_snapshot_id": _sha256("pending-typed-truth"),
        "truth_snapshot_sha256": _sha256("pending-typed-truth"),
        "normalized_rows": _typed_table_artifact(
            "truth-normalized",
            role="truth_normalized_rows",
            row_schema_ref="#/$defs/FormalNormalizedRow",
            sort_profile="stage2p_truth_normalized_rows_sort_v1",
        ),
        "deduplicated_rows": _typed_table_artifact(
            "truth-deduplicated",
            role="truth_deduplicated_rows",
            row_schema_ref="#/$defs/FormalDeduplicatedRow",
            sort_profile="stage2p_truth_deduplicated_rows_sort_v1",
        ),
        "window_membership_sha256": membership["content_sha256"],
        "window_membership_rows": membership,
        "realized_target_set": {
            "unique_event_count": target_rows["row_count"],
            "ordered_event_ids_sha256": _sha256("target-order"),
            "target_rows_content_sha256": target_rows["content_sha256"],
            "region_membership_sha256": _sha256("region-membership"),
            "target_rows": target_rows,
        },
    }
    _bind_object_identity(
        snapshot,
        id_field="truth_snapshot_id",
        hash_field="truth_snapshot_sha256",
    )
    return {"truth_snapshot": snapshot}


def test_typed_source_and_truth_artifacts_pass_structural_semantics() -> None:
    validate_record_semantics(_typed_source_snapshot_record(), _protocol())
    validate_record_semantics(_typed_truth_snapshot_record(), _protocol())


def test_typed_artifact_nested_rehashes_cannot_break_summary_bindings() -> None:
    source_record = _typed_source_snapshot_record()
    source = source_record["source_snapshot"]
    source["normalized_rows"]["artifact_id"] = _sha256("substituted-file")  # type: ignore[index]
    _bind_object_identity(
        source,
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    _assert_code(
        "typed_table_artifact_id_mismatch",
        validate_record_semantics,
        source_record,
        _protocol(),
    )

    source_record = _typed_source_snapshot_record()
    source = source_record["source_snapshot"]
    source["R30_event_set"]["event_rows"]["content_sha256"] = _sha256(  # type: ignore[index]
        "substituted-r30-rows"
    )
    _bind_object_identity(
        source,
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    _assert_code(
        "event_set_rows_content_hash_mismatch",
        validate_record_semantics,
        source_record,
        _protocol(),
    )

    source_record = _typed_source_snapshot_record()
    source = source_record["source_snapshot"]
    source["cutover_cross_source_match_rows"]["row_count"] = 3  # type: ignore[index]
    _bind_object_identity(
        source,
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    _assert_code(
        "cutover_match_row_count_mismatch",
        validate_record_semantics,
        source_record,
        _protocol(),
    )

    truth_record = _typed_truth_snapshot_record()
    truth = truth_record["truth_snapshot"]
    truth["window_membership_rows"]["content_sha256"] = _sha256(  # type: ignore[index]
        "substituted-membership"
    )
    _bind_object_identity(
        truth,
        id_field="truth_snapshot_id",
        hash_field="truth_snapshot_sha256",
    )
    _assert_code(
        "truth_window_membership_content_hash_mismatch",
        validate_record_semantics,
        truth_record,
        _protocol(),
    )

    truth_record = _typed_truth_snapshot_record()
    truth = truth_record["truth_snapshot"]
    truth["realized_target_set"]["target_rows"]["row_count"] = 3  # type: ignore[index]
    _bind_object_identity(
        truth,
        id_field="truth_snapshot_id",
        hash_field="truth_snapshot_sha256",
    )
    _assert_code(
        "target_set_row_count_mismatch",
        validate_record_semantics,
        truth_record,
        _protocol(),
    )


def test_prediction_visualization_cannot_rebind_to_a_different_forecast_bundle() -> None:
    record = _on_time_issue()
    seal = record["prediction_seal"]
    visualization = seal["visualization_evidence"]  # type: ignore[index]
    bundle_sha256 = _sha256("forecast-bundle-manifest")
    seal["forecast_bundle_manifest_sha256"] = bundle_sha256  # type: ignore[index]
    visualization["visualized_forecast_bundle_manifest_sha256"] = bundle_sha256
    _bind_self_hash(visualization, "visualization_evidence_sha256")
    _bind_object_identity(
        seal,  # type: ignore[arg-type]
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _bind_record_hashes(record)
    validate_record_semantics(record, _protocol())

    visualization["visualized_forecast_bundle_manifest_sha256"] = _sha256(
        "different-legal-bundle"
    )
    _bind_self_hash(visualization, "visualization_evidence_sha256")
    _bind_object_identity(
        seal,  # type: ignore[arg-type]
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _bind_record_hashes(record)
    _assert_code(
        "visualization_forecast_bundle_manifest_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )


@pytest.mark.parametrize(
    ("event_set_field", "seal_field"),
    [
        ("P0_event_set", "P0_event_set_sha256"),
        ("R30_event_set", "R30_event_set_sha256"),
        ("RP30_event_set", "RP30_event_set_sha256"),
    ],
)
def test_prediction_seal_cannot_rebind_away_from_enclosing_event_sets(
    event_set_field: str,
    seal_field: str,
) -> None:
    record = _on_time_issue()
    snapshot = record["source_snapshot"]
    seal = record["prediction_seal"]
    for source_field, prediction_field in (
        ("P0_event_set", "P0_event_set_sha256"),
        ("R30_event_set", "R30_event_set_sha256"),
        ("RP30_event_set", "RP30_event_set_sha256"),
    ):
        event_rows_sha256 = _sha256(f"{source_field}-rows")
        snapshot[source_field] = {  # type: ignore[index]
            "event_rows_content_sha256": event_rows_sha256,
        }
        seal[prediction_field] = event_rows_sha256  # type: ignore[index]
    snapshot_sha256 = _bind_object_identity(
        snapshot,  # type: ignore[arg-type]
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    seal["source_snapshot_sha256"] = snapshot_sha256  # type: ignore[index]
    _bind_object_identity(
        seal,  # type: ignore[arg-type]
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _bind_record_hashes(record)
    validate_record_semantics(record, _protocol())

    snapshot[event_set_field]["event_rows_content_sha256"] = _sha256(  # type: ignore[index]
        f"tampered-{event_set_field}-rows"
    )
    snapshot_sha256 = _bind_object_identity(
        snapshot,  # type: ignore[arg-type]
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    seal["source_snapshot_sha256"] = snapshot_sha256  # type: ignore[index]
    _bind_object_identity(
        seal,  # type: ignore[arg-type]
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _bind_record_hashes(record)
    assert seal[seal_field] != snapshot[event_set_field][  # type: ignore[index]
        "event_rows_content_sha256"
    ]
    _assert_code(
        "prediction_event_set_hash_mismatch",
        validate_record_semantics,
        record,
        _protocol(),
    )


def test_truth_revision_is_local_and_uses_exact_issue_window_bindings() -> None:
    mature = _mature_truth_record()
    revision = {
        "record_type": "TruthRevisionRecord",
        "issue_id": mature["issue_id"],
        "horizon_days": 7,
        "revision_reasons": ["current_target_added_prior_not_observed"],
        "revision_observed_at_utc": "2026-10-20T16:00:00Z",
        "target_start_exclusive_utc": "2026-09-09T16:00:00Z",
        "target_end_inclusive_utc": "2026-09-16T16:00:00Z",
        "revision_derived_started_at_utc": "2026-10-20T16:01:00Z",
        "revision_derived_completed_at_utc": "2026-10-20T16:01:01Z",
        "seal_completed_at_utc": "2026-10-20T16:01:02Z",
        "record_core_frozen_at_utc": "2026-10-20T16:01:03Z",
    }
    _bind_record_content_hash(revision)
    validate_record_semantics(revision, _protocol())

    revision["target_end_inclusive_utc"] = "2026-09-16T16:00:01Z"
    _assert_code(
        "truth_record_target_end_mismatch",
        validate_record_semantics,
        revision,
        _protocol(),
    )


def test_unavailable_mature_truth_is_a_valid_but_terminal_chain() -> None:
    unavailable = _mature_truth_record()
    unavailable.update(
        {
            "status": "truth_snapshot_unavailable",
            "truth_snapshot": None,
            "replay_visualization": None,
        }
    )
    _bind_record_content_hash(unavailable)  # type: ignore[arg-type]
    validate_truth_chain([unavailable], _protocol())
    _assert_code(
        "truth_revision_after_unavailable_forbidden",
        validate_truth_chain,
        [unavailable, {}],
        _protocol(),
    )


def test_code_manifest_hash_and_component_bindings_are_recomputed() -> None:
    manifest = _code_manifest()
    source_snapshot = {
        "parser_code_sha256": manifest["parser_code_sha256"],
        "normalization_config_sha256": manifest["normalization_config_sha256"],
        "deduplication_code_sha256": manifest["deduplication_code_sha256"],
        "deduplication_config_sha256": manifest["deduplication_config_sha256"],
        "revision_policy_sha256": manifest["revision_policy_sha256"],
    }
    _bind_object_identity(
        source_snapshot,
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    record = {
        "source_snapshot": source_snapshot,
    }
    validate_record_semantics(record, {"code_manifest": manifest})

    wrong_component = copy.deepcopy(record)
    wrong_component["source_snapshot"]["parser_code_sha256"] = _sha256("wrong")  # type: ignore[index]
    _assert_code(
        "record_code_manifest_binding_mismatch",
        validate_record_semantics,
        wrong_component,
        {"code_manifest": manifest},
    )

    wrong_manifest = copy.deepcopy(manifest)
    wrong_manifest["manifest_sha256"] = _sha256("wrong-manifest")
    _assert_code(
        "code_manifest_hash_mismatch",
        validate_record_semantics,
        record,
        {"code_manifest": wrong_manifest},
    )


def test_evaluation_checkpoint_and_scheduled_cap_counts_fail_closed() -> None:
    checkpoint_52 = _evaluation_input(
        sequence=1,
        checkpoint=1,
        trigger=52,
        passed=False,
        previous=None,
        content=_sha256("52-at-cap"),
    )
    checkpoint_52["scheduled_issue_count"] = 130
    _assert_code(
        "checkpoint_52_reached_at_scheduled_cap",
        validate_evaluation_chain,
        [checkpoint_52],
    )

    cap = {
        "record_type": "EvaluationFreezeRecord",
        "evaluation_sequence": 1,
        "phase": "input_freeze",
        "checkpoint_number": 3,
        "trigger_reason": "scheduled_issue_cap_130",
        "scheduled_issue_count": 130,
        "trigger_on_time_issue_count": 80,
        "previous_evaluation_freeze_sha256": None,
        "input_freeze_sha256": None,
        "sample_gate_met": False,
        "status": "evidence_insufficient",
        "confirmatory_effects_authorized": False,
        "content_sha256": _sha256("cap"),
    }
    validate_evaluation_chain([cap])

    invalid_cap = copy.deepcopy(cap)
    invalid_cap["scheduled_issue_count"] = 129
    _assert_code("scheduled_cap_counts_invalid", validate_evaluation_chain, [invalid_cap])


def test_schema_wrapper_runs_draft_2020_format_checker_then_semantics() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": [
            "issue_time_local",
            "issue_time_utc",
            "query_end_utc",
        ],
        "properties": {
            "issue_time_local": {"type": "string", "format": "date-time"},
            "issue_time_utc": {"type": "string", "format": "date-time"},
            "query_end_utc": {"type": "string", "format": "date-time"},
        },
    }
    record = {
        "issue_time_local": "2026-09-10T00:00:00+08:00",
        "issue_time_utc": "2026-09-09T16:00:00Z",
        "query_end_utc": "2026-09-09T15:45:00Z",
    }
    validate_record_against_schema(schema, record, _protocol())

    invalid_format = dict(record)
    invalid_format["query_end_utc"] = "not-a-date"
    with pytest.raises(ValidationError):
        validate_record_against_schema(schema, invalid_format, _protocol())


def test_current_contract_accepts_complete_synthetic_target_cohort() -> None:
    schema = json.loads(
        (ROOT / "data" / "contracts" / "stage2p_prospective_records.json").read_text(
            encoding="utf-8"
        )
    )
    cohort = _synthetic_record(schema, "TargetCohortDefinition")
    cohort["definition_frozen_at_utc"] = "2026-09-02T11:05:00Z"
    cohort["valid_from_utc"] = "2026-09-09T16:00:00Z"
    for name, verified in (
        ("protocol_tag_remote_receipt", "2026-09-02T10:00:00Z"),
        ("code_tag_remote_receipt", "2026-09-02T11:00:00Z"),
    ):
        receipt = cohort[name]
        receipt["verified_at_utc"] = verified
        receipt["remote_peeled_commit"] = (
            cohort["protocol_commit"] if name.startswith("protocol") else cohort["code_commit"]
        )
        _bind_self_hash(receipt, "receipt_sha256")
    manifest = cohort["code_manifest"]
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    cohort["code_manifest_sha256"] = manifest["manifest_sha256"]
    cohort["record_core_frozen_at_utc"] = "2026-09-02T11:10:00Z"
    cohort["timestamp_deadline_utc"] = cohort["valid_from_utc"]
    attempt = cohort["timestamp_attempt_evidence"][0]
    attempt.update(
        {
            "request_started_at_utc": "2026-09-02T11:20:00Z",
            "genTime_utc": "2026-09-02T11:29:00Z",
            "response_received_at_utc": "2026-09-02T11:29:30Z",
            "attempt_completed_at_utc": "2026-09-02T11:29:30Z",
            "http_status": 200,
            "response_content_type": "application/timestamp-reply",
            "response_byte_count": 512,
            "response_sha256": _sha256("cohort-tsa-response"),
            "authority_identity_sha256": _sha256("cohort-tsa-authority"),
            "trust_chain_sha256": _sha256("cohort-tsa-chain"),
            "offline_trust_path_valid": True,
            "genTime_before_deadline": True,
            "outcome": "selected_valid",
        }
    )
    proof = cohort["remote_timestamp"]
    proof["deadline_utc"] = cohort["timestamp_deadline_utc"]
    proof["selected_attempt_index"] = attempt["attempt_index"]
    proof["selected_response_sha256"] = attempt["response_sha256"]
    _bind_record_hashes(cohort)
    validate_record_against_schema(schema, cohort, _protocol())


def test_current_contract_accepts_complete_synthetic_on_time_issue() -> None:
    schema = json.loads(
        (ROOT / "data" / "contracts" / "stage2p_prospective_records.json").read_text(
            encoding="utf-8"
        )
    )
    issue = _synthetic_record(schema, "IssueInputSnapshotRecord")
    issue.update(
        {
            "issue_time_local": "2026-09-10T00:00:00+08:00",
            "issue_time_utc": "2026-09-09T16:00:00Z",
            "query_end_utc": "2026-09-09T15:45:00Z",
            "scheduled_issue_sequence": 1,
            "on_time_issue_sequence": 1,
            "previous_issue_record_sha256": None,
            "status": "on_time",
            "failure_code": None,
            "failed_candidate_on_time_core": None,
            "missed_audit_timestamp_deadline_utc": None,
            "missed_audit_timestamp_attempt_evidence": [],
            "missed_audit_remote_timestamp": None,
            "prediction_generated": True,
            "prediction_installed": True,
        }
    )

    def bind_fetch(
        fetch: dict[str, Any],
        *,
        start: str,
        end: str,
        started: str,
        completed: str,
    ) -> None:
        has_exchange_outcome = "exchange_outcome" in fetch
        query = _request_with_hash(
            endpoint="https://earthquake.usgs.gov/fdsnws/event/1/query",
            start=start,
            end=end,
            count=False,
        )
        count_request = _request_with_hash(
            endpoint="https://earthquake.usgs.gov/fdsnws/event/1/count",
            start=start,
            end=end,
            count=True,
        )
        captured_headers = {
            "date": "Wed, 09 Sep 2026 15:46:00 GMT",
            "etag": None,
            "last_modified": None,
            "content_type": None,
            "content_length": "0",
        }
        response_headers_sha256 = hashlib.sha256(
            canonical_json_bytes(captured_headers)
        ).hexdigest()
        fetch.update(
            {
                "query_start_utc": start,
                "query_end_utc": end,
                "query_request": query,
                "request_identity_sha256": query["canonical_url_utf8_sha256"],
                "query_count_preflight_request_sha256": count_request[
                    "canonical_url_utf8_sha256"
                ],
                "query_count": 0,
                "feature_count": 0,
                "fetch_started_at_utc": started,
                "fetch_completed_at_utc": completed,
                "http_status": 204,
                "response_content_type": None,
                "response_body_byte_count": 0,
                "geojson_parse_verified": False,
                "response_headers_sha256": response_headers_sha256,
                "exchange_outcome": "response_received",
                "outcome": "succeeded",
                "failure_code": None,
            }
        )
        if "captured_response_headers" in fetch:
            fetch["captured_response_headers"] = captured_headers
        if "raw_response_sha256" in fetch:
            fetch["raw_response_sha256"] = _EMPTY_SHA256
        else:
                fetch["raw_response"].update(  # type: ignore[union-attr]
                    {
                        "artifact_id": _EMPTY_SHA256,
                    "byte_count": 0,
                    "file_sha256": _EMPTY_SHA256,
                    "local_restricted": True,
                }
            )
        if not has_exchange_outcome:
            fetch.pop("exchange_outcome", None)
            fetch.pop("outcome", None)
            fetch.pop("failure_code", None)
        preflight = fetch["count_preflight"]
        query_end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
        preflight.update(
            {
                "request": count_request,
                "fetch_started_at_utc": _utc_text(query_end_time + timedelta(seconds=1)),
                "fetch_completed_at_utc": _utc_text(query_end_time + timedelta(seconds=15)),
                "http_status": 200,
                "response_content_type": "application/json",
                "response_body_byte_count": 1,
                "response_headers_sha256": _sha256(f"count-headers-{started}"),
                "raw_response_sha256": _sha256(f"count-raw-{started}"),
                "geojson_parse_verified": True,
                "parsed_count": 0,
                "outcome": "succeeded",
                "failure_code": None,
            }
        )

    attempt = issue["attempt_evidence"][0]
    bind_fetch(
        attempt,
        start="2026-07-09T04:25:56Z",
        end="2026-09-09T15:45:00Z",
        started="2026-09-09T15:45:31Z",
        completed="2026-09-09T15:46:00Z",
    )
    snapshot = issue["source_snapshot"]
    snapshot.update(
        {
            "issue_time_utc": "2026-09-09T16:00:00Z",
            "query_start_utc": "2026-07-09T04:25:56Z",
            "query_end_utc": "2026-09-09T15:45:00Z",
                "seal_completed_at_utc": "2026-09-09T15:50:00Z",
        }
    )
    bind_fetch(
        snapshot["acquisition"],
        start="2026-07-09T04:25:56Z",
        end="2026-09-09T15:45:00Z",
        started="2026-09-09T15:45:31Z",
        completed="2026-09-09T15:46:00Z",
    )
    p0 = _forecast_identity()
    seal = issue["prediction_seal"]
    seal.update(
        {
            "issue_time_utc": "2026-09-09T16:00:00Z",
            "query_end_utc": "2026-09-09T15:45:00Z",
            "valid_from_utc": "2026-09-09T16:00:00Z",
            "grid_identity_sha256": p0["grid_identity_sha256"],
            "P0": p0,
            "P1": copy.deepcopy(p0),
            "PP": copy.deepcopy(p0),
            "R30_event_count": 0,
            "RP30_event_count": 0,
            "empty_R30_byte_identity_verified": True,
            "empty_RP30_byte_identity_verified": True,
                "sealed_at_utc": "2026-09-09T15:51:00Z",
        }
    )
    tsa_attempt = issue["timestamp_attempt_evidence"][0]
    tsa_attempt.update(
        {
            "request_started_at_utc": "2026-09-09T15:52:30Z",
            "response_received_at_utc": "2026-09-09T15:54:00Z",
            "attempt_completed_at_utc": "2026-09-09T15:54:00Z",
            "http_status": 200,
            "response_content_type": "application/timestamp-reply",
            "response_byte_count": 512,
            "response_sha256": _sha256("issue-tsa-response"),
            "authority_identity_sha256": _sha256("issue-tsa-authority"),
            "trust_chain_sha256": _sha256("issue-tsa-chain"),
            "genTime_utc": "2026-09-09T15:53:00Z",
            "offline_trust_path_valid": True,
            "genTime_before_deadline": True,
            "outcome": "selected_valid",
        }
    )
    issue["record_core_frozen_at_utc"] = "2026-09-09T15:52:00Z"
    issue["timestamp_deadline_utc"] = "2026-09-09T15:55:00Z"
    issue["remote_timestamp"].update(
        {
            "deadline_utc": issue["timestamp_deadline_utc"],
            "selected_attempt_index": tsa_attempt["attempt_index"],
            "selected_response_sha256": tsa_attempt["response_sha256"],
        }
    )
    manifest = _code_manifest()
    issue["code_commit"] = "b" * 40
    issue["code_manifest_sha256"] = manifest["manifest_sha256"]
    for field in (
        "parser_code_sha256",
        "normalization_config_sha256",
        "deduplication_code_sha256",
        "deduplication_config_sha256",
        "revision_policy_sha256",
    ):
        snapshot[field] = manifest[field]
    seal["revision_policy_sha256"] = manifest["revision_policy_sha256"]
    for event_set_name in ("P0_event_set", "R30_event_set", "RP30_event_set"):
        seal[f"{event_set_name}_sha256"] = snapshot[event_set_name][  # type: ignore[index]
            "event_rows_content_sha256"
        ]
    runtime = seal["runtime_evidence"]
    runtime.update(
        {
            "gpu_model": None,
            "gpu_driver_version": None,
            "gpu_runtime_version": None,
            "GPU_equivalence_receipt_sha256": None,
        }
    )
    source_snapshot_sha256 = _bind_object_identity(
        snapshot,
        id_field="snapshot_id",
        hash_field="snapshot_sha256",
    )
    seal["source_snapshot_sha256"] = source_snapshot_sha256
    seal["visualization_evidence"][
        "visualized_forecast_bundle_manifest_sha256"
    ] = seal["forecast_bundle_manifest_sha256"]
    _bind_self_hash(
        seal["visualization_evidence"],
        "visualization_evidence_sha256",
    )
    _bind_object_identity(
        seal,
        id_field="prediction_seal_id",
        hash_field="prediction_seal_sha256",
    )
    _bind_record_hashes(issue)
    context = {
        **_protocol(),
        "cohort_definition": {
            "code_commit": issue["code_commit"],
            "code_manifest": manifest,
        },
    }
    definition_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/IssueInputSnapshotRecord",
    }
    validate_record_against_schema(definition_schema, issue, context)


def test_current_contract_accepts_complete_synthetic_mature_truth() -> None:
    schema = json.loads(
        (ROOT / "data" / "contracts" / "stage2p_prospective_records.json").read_text(
            encoding="utf-8"
        )
    )
    record = _synthetic_record(schema, "MatureTruthSnapshotRecord")
    issue = "2026-09-09T16:00:00Z"
    target_end = "2026-09-16T16:00:00Z"
    due = "2026-10-16T16:00:00Z"
    record.update(
        {
            "issue_id": "stage2p-issue-20260910T000000+0800",
            "horizon_days": 7,
            "maturity_due_at_utc": due,
            "status": "mature_truth_sealed",
            "failure_code": None,
            "previous_truth_record_sha256": None,
            "revision_sequence": 0,
        }
    )
    attempt = record["attempt_evidence"][0]
    attempt.update(
        {
            "issue_time_utc": issue,
            "horizon_days": 7,
            "target_start_exclusive_utc": issue,
            "target_end_inclusive_utc": target_end,
            "maturity_due_at_utc": due,
            "retry_offset_hours": 0,
            "scheduled_at_utc": due,
        }
    )
    _bind_complete_successful_fetch(
        attempt,
        start=issue,
        end=target_end,
        count_started="2026-10-16T16:00:01Z",
        count_completed="2026-10-16T16:00:02Z",
        query_started="2026-10-16T16:00:03Z",
        query_completed="2026-10-16T16:00:04Z",
        label="mature-fetch",
    )
    snapshot = record["truth_snapshot"]
    snapshot.update(
        {
            "target_start_exclusive_utc": issue,
            "target_end_inclusive_utc": target_end,
            "maturity_due_at_utc": due,
            "retry_offset_hours": 0,
            "seal_completed_at_utc": "2026-10-16T16:00:05Z",
        }
    )
    _bind_complete_successful_fetch(
        snapshot["acquisition"],
        start=issue,
        end=target_end,
        count_started="2026-10-16T16:00:01Z",
        count_completed="2026-10-16T16:00:02Z",
        query_started="2026-10-16T16:00:03Z",
        query_completed="2026-10-16T16:00:04Z",
        label="mature-fetch",
    )
    _bind_object_identity(
        snapshot,
        id_field="truth_snapshot_id",
        hash_field="truth_snapshot_sha256",
    )
    replay = record["replay_visualization"]
    replay["previous_replay_visualization_sha256"] = None
    replay["truth_target_set_sha256"] = snapshot["realized_target_set"][
        "target_rows_content_sha256"
    ]
    _bind_self_hash(replay, "replay_visualization_sha256")
    _bind_selected_tsa(
        record,
        core_frozen="2026-10-16T16:00:10Z",
        request_started="2026-10-16T16:00:20Z",
        generated="2026-10-16T16:00:30Z",
        response_received="2026-10-16T16:00:40Z",
        deadline="2026-10-16T16:05:10Z",
        label="mature",
    )
    _bind_record_hashes(record)
    validate_record_against_schema(
        _definition_schema(schema, "MatureTruthSnapshotRecord"),
        record,
        _protocol(),
    )


def test_current_contract_accepts_complete_synthetic_truth_revision() -> None:
    schema = json.loads(
        (ROOT / "data" / "contracts" / "stage2p_prospective_records.json").read_text(
            encoding="utf-8"
        )
    )
    record = _synthetic_record(schema, "TruthRevisionRecord")
    issue = "2026-09-09T16:00:00Z"
    target_end = "2026-09-16T16:00:00Z"
    record.update(
        {
            "issue_id": "stage2p-issue-20260910T000000+0800",
            "horizon_days": 7,
            "revision_sequence": 1,
            "revision_observed_at_utc": "2026-10-20T16:00:00Z",
            "target_start_exclusive_utc": issue,
            "target_end_inclusive_utc": target_end,
            "revision_derived_started_at_utc": "2026-10-20T16:01:00Z",
            "revision_derived_completed_at_utc": "2026-10-20T16:01:01Z",
            "seal_completed_at_utc": "2026-10-20T16:01:02Z",
        }
    )
    replay = record["replay_visualization"]
    replay["truth_target_set_sha256"] = record["revised_target_set"][
        "target_rows_content_sha256"
    ]
    _bind_self_hash(replay, "replay_visualization_sha256")
    _bind_selected_tsa(
        record,
        core_frozen="2026-10-20T16:01:10Z",
        request_started="2026-10-20T16:01:20Z",
        generated="2026-10-20T16:01:30Z",
        response_received="2026-10-20T16:01:40Z",
        deadline="2026-10-20T16:06:10Z",
        label="revision",
    )
    _bind_record_hashes(record)
    validate_record_against_schema(
        _definition_schema(schema, "TruthRevisionRecord"),
        record,
        _protocol(),
    )


def test_current_contract_accepts_complete_synthetic_evaluation_cap() -> None:
    schema = json.loads(
        (ROOT / "data" / "contracts" / "stage2p_prospective_records.json").read_text(
            encoding="utf-8"
        )
    )
    record = _synthetic_record(schema, "EvaluationFreezeRecord")
    record.update(
        {
            "evaluation_sequence": 1,
            "phase": "input_freeze",
            "checkpoint_number": 3,
            "trigger_reason": "scheduled_issue_cap_130",
            "scheduled_issue_count": 130,
            "trigger_on_time_issue_count": 0,
            "previous_evaluation_freeze_sha256": None,
            "input_freeze_sha256": None,
            "frozen_at_utc": "2029-03-15T16:00:00Z",
            "record_core_frozen_at_utc": "2029-03-15T16:00:00Z",
            "timestamp_deadline_utc": "2029-03-15T16:05:00Z",
            "effect_rows_opened_at_utc": None,
            "all_90d_windows_mature": False,
            "ordered_issue_prediction_seal_sha256": [],
            "ordered_truth_record_sha256": [],
            "union_unique_supported_M5_6_event_count": None,
            "union_unique_full_study_area_M5_6_event_count": None,
            "unique_target_cluster_count": None,
            "realized_target_union": None,
            "final_window_membership_sha256": None,
            "cluster_membership_sha256": None,
            "sample_gate_met": False,
            "status": "evidence_insufficient",
            "confirmatory_effects_authorized": False,
            "confirmatory_result": None,
        }
    )
    empty_projection_sha256 = hashlib.sha256(canonical_json_bytes([])).hexdigest()
    selected_manifest = record["selected_exposure_manifest"]
    selected_manifest["candidate_issue_prediction_seal_sha256"] = []
    selected_manifest["rows"] = []
    record["selected_exposure_manifest_union_sha256"] = hashlib.sha256(
        canonical_json_bytes(selected_manifest)
    ).hexdigest()
    record["truth_availability_manifest_union_sha256"] = empty_projection_sha256
    alarm_manifest = record["alarm_area_manifest"]
    alarm_manifest.update(
        {
            "selected_exposure_manifest_union_sha256": record[
                "selected_exposure_manifest_union_sha256"
            ],
            "maximum_allowed_pairwise_difference_km2_float64_hex": (
                "4083880000000000"
            ),
            "entries": [],
        }
    )
    record["alarm_area_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(alarm_manifest)
    ).hexdigest()
    for horizon in record["horizon_evaluability"]:
        horizon.update(
            {
                "complete_exposure_count": 0,
                "truth_available_complete_exposure_count": 0,
                "selected_truth_snapshot_unavailable_count": 0,
                "selected_exposure_manifest_sha256": empty_projection_sha256,
                "truth_availability_manifest_sha256": empty_projection_sha256,
                "supported_unique_M5_6_event_count": None,
                "full_study_area_unique_M5_6_event_count": None,
                "unique_target_cluster_count": None,
                "all_required_forecast_densities_finite_positive": None,
                "evaluable": False,
                "unevaluable_reason": (
                    "scheduled_issue_cap_terminal_before_formal_freeze"
                ),
            }
        )
    record["bootstrap_preflight"].update(
        {
            "status": "not_run_formal_freeze_unavailable",
            "index_matrix_sha256": None,
            "generated_replication_count": 0,
            "zero_denominator_replication_count": None,
            "frozen_before_effect_rows_open": False,
            "redraw_or_discard_performed": False,
        }
    )
    protocol = _full_protocol()
    evaluation = protocol["evaluation"]
    record["bootstrap_plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(evaluation["bootstrap"])
    ).hexdigest()
    statistics_policy = {
        "profile": "stage2p_statistics_policy_v1",
        **{
            field: evaluation[field]
            for field in (
                "information_gain",
                "strict_recall",
                "macro_aggregation",
                "target_clusters",
                "simultaneous_inference",
                "pass_gate",
                "regional_robustness",
            )
        },
    }
    record["statistics_policy_sha256"] = hashlib.sha256(
        canonical_json_bytes(statistics_policy)
    ).hexdigest()
    record["region_map_sha256"] = evaluation["regional_robustness"][
        "region_manifest_file_sha256"
    ]
    failure_evidence = {
        "failure_evidence_sha256": _sha256("pending-formal-failure"),
        "status": "not_run_scheduled_issue_cap_terminal",
        "evaluation_scope_sha256": record[
            "selected_exposure_manifest_union_sha256"
        ],
        "scheduled_at_utc": "2029-03-15T15:59:00Z",
        "global_target_start_exclusive_utc": None,
        "global_target_end_inclusive_utc": None,
        "count_preflight": None,
        "query_failure_artifact_sha256": None,
        "failure_stage": "scope_selection",
        "failure_code": "scheduled_issue_cap_terminal",
        "contains_target_or_effect_rows": False,
        "local_restricted": True,
    }
    _bind_self_hash(failure_evidence, "failure_evidence_sha256")
    record.update(
        {
            "formal_freeze_status": "not_run_scheduled_issue_cap_terminal",
            "formal_freeze_source_manifest": None,
            "formal_freeze_source_manifest_sha256": None,
            "formal_freeze_failure_evidence": failure_evidence,
            "formal_freeze_failure_evidence_sha256": failure_evidence[
                "failure_evidence_sha256"
            ],
        }
    )
    _bind_selected_tsa(
        record,
        core_frozen="2029-03-15T16:00:00Z",
        request_started="2029-03-15T16:01:00Z",
        generated="2029-03-15T16:02:00Z",
        response_received="2029-03-15T16:03:00Z",
        deadline="2029-03-15T16:05:00Z",
        label="evaluation-cap",
    )
    _bind_record_hashes(record)
    validate_record_against_schema(
        _definition_schema(schema, "EvaluationFreezeRecord"),
        record,
        protocol,
    )
