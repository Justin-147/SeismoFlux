"""Build or verify stage-4 target-blind scoring qualification from real receipts.

This entrypoint never runs a caller-supplied Boolean gate and never touches the
earthquake target.  It parses two actual zero-failure JUnit documents, freshly
observes the allowed score-blind inputs, strictly loads the canonical 153-issue
formal-preflight receipt, and records the user's GPU request.  The frozen project
has no accepted GPU backend, so the only authorized formal backend remains CPU
float64 and that fallback is explicit in the resulting evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from seismoflux.anomaly_increment.config import (
    STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH,
    STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH,
    STAGE4_JUNIT_RELATIVE_PATH,
    STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH,
    STAGE4_PROTOCOL_PATH,
    STAGE4_QUALIFICATION_RELATIVE_PATH,
    stage4_scoring_freeze_relative_path,
    validate_stage4_r1_execution_contract,
)
from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_PREFLIGHT_RECEIPT_PATH,
    load_formal_preflight_receipt,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
)
from seismoflux.anomaly_increment.preregistration import verify_content_sha256
from seismoflux.anomaly_increment.qualification import (
    Stage4QualificationEvidence,
    build_stage4_qualification_evidence,
    load_stage4_qualification_evidence,
    observe_score_blind_inputs,
    parse_pytest_junit_evidence,
    write_stage4_qualification_evidence_atomic,
)
from seismoflux.anomaly_increment.score_blind_path import (
    require_score_blind_project_path,
)
from seismoflux.background.execution import (
    CommandResult,
    GitCommandRunner,
    subprocess_git_runner,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / STAGE4_PROTOCOL_PATH
QUALIFICATION_PATH = PROJECT_ROOT.joinpath(*STAGE4_QUALIFICATION_RELATIVE_PATH.parts)
PREFLIGHT_RECEIPT_PATH = PROJECT_ROOT.joinpath(*STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH.parts)
STAGE4_JUNIT_PATH = PROJECT_ROOT.joinpath(*STAGE4_JUNIT_RELATIVE_PATH.parts)
FULL_JUNIT_PATH = PROJECT_ROOT.joinpath(*STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH.parts)
LOGICAL_REPLAY_AUDIT_PATH = PROJECT_ROOT.joinpath(*STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH.parts)

_GIT_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(dict[str, Any], dict(value))


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = _read_stable_bytes(path, label="stage-4 protocol")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("stage-4 protocol is not valid UTF-8") from exc
    protocol = _mapping(yaml.safe_load(text), label="protocol")
    validate_stage4_r1_execution_contract(protocol)
    return protocol


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    target = Path(path)
    try:
        require_existing_real_directory_tree(
            Path(target.anchor) if target.anchor else Path.cwd(),
            target.parent,
            label=f"{label} parent directory",
        )
        return read_existing_immutable_bytes(target, label=label)
    except UnsafeImmutableFileError as exc:
        raise ValueError(f"{label} cannot be read safely") from exc


def _current_commit(project_root: Path, git_runner: GitCommandRunner) -> str:
    result = git_runner(("git", "rev-parse", "HEAD"), project_root)
    if not isinstance(result, CommandResult):
        raise TypeError("Git runner must return CommandResult")
    commit = result.stdout.strip()
    if result.returncode != 0 or _GIT_OID.fullmatch(commit) is None:
        raise ValueError("cannot derive the current scoring-code commit from Git")
    return commit


def _logical_replay_audit_sha256(
    payload: bytes,
    *,
    protocol_design_sha256: str,
    random_input_seal_sha256: str,
    score_blind_input_evidence_sha256: str,
    scoring_code_commit: str,
) -> str:
    try:
        document = _mapping(json.loads(payload.decode("utf-8")), label="logical replay audit")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("logical replay audit is not canonical UTF-8 JSON") from exc
    expected = {
        "audit",
        "content_sha256",
        "locked_test_run",
        "protocol_design_sha256",
        "random_input_seal_sha256",
        "schema_version",
        "scoring_code_commit",
        "target_bytes_read",
        "target_path_observed",
        "worktree_clean_before_and_after",
    }
    if set(document) != expected or not verify_content_sha256(document):
        raise ValueError("logical replay audit hash or schema is invalid")
    if (
        document.get("schema_version") != 1
        or document.get("protocol_design_sha256") != protocol_design_sha256
        or document.get("random_input_seal_sha256") != random_input_seal_sha256
        or document.get("scoring_code_commit") != scoring_code_commit
        or document.get("locked_test_run") is not False
        or document.get("target_bytes_read") is not False
        or document.get("target_path_observed") is not False
        or document.get("worktree_clean_before_and_after") is not True
    ):
        raise ValueError("logical replay audit belongs to another or non-target-blind execution")
    audit = _mapping(document.get("audit"), label="logical replay audit payload")
    audit_expected = {
        "content_sha256",
        "grid_id",
        "identity_method",
        "issue_count",
        "query_chunk_size",
        "reproduction_identity_sha256",
        "role",
        "source_columns",
        "source_input_sha256",
        "target_bytes_read",
        "target_path_observed",
        "worker_replays",
    }
    if set(audit) != audit_expected or not verify_content_sha256(audit):
        raise ValueError("logical replay payload hash or schema is invalid")
    replays = audit.get("worker_replays")
    if (
        audit.get("identity_method") != "arrow_ipc_selected_table_logical_identity_r1"
        or audit.get("issue_count") != 153
        or audit.get("role") != "stage4_r1_primary_grid_logical_identity_worker_replay"
        or audit.get("source_input_sha256") != score_blind_input_evidence_sha256
        or audit.get("target_bytes_read") is not False
        or audit.get("target_path_observed") is not False
        or not isinstance(replays, list)
        or len(replays) != 2
    ):
        raise ValueError("logical replay payload is incomplete or crossed the target boundary")
    parsed_replays = [_mapping(item, label="logical replay worker") for item in replays]
    reproduction = audit.get("reproduction_identity_sha256")
    if not isinstance(reproduction, str) or _SHA256.fullmatch(reproduction) is None:
        raise ValueError("logical replay reproduction identity is not a SHA-256")
    receipt_lists: list[list[object]] = []
    for expected_workers, replay in zip((1, 2), parsed_replays, strict=True):
        if set(replay) != {
            "receipts",
            "reproduction_identity_sha256",
            "spatial_workers",
        }:
            raise ValueError("logical replay worker schema changed")
        receipts = replay.get("receipts")
        if (
            replay.get("spatial_workers") != expected_workers
            or replay.get("reproduction_identity_sha256") != reproduction
            or not isinstance(receipts, list)
            or len(receipts) != 153
        ):
            raise ValueError("logical replay worker result is incomplete or not invariant")
        issue_ids: list[str] = []
        for expected_index, raw_receipt in enumerate(receipts):
            receipt = _mapping(raw_receipt, label="logical replay issue receipt")
            if set(receipt) != {
                "accepted_table_sha256",
                "issue_id",
                "issue_index",
                "issue_report_id",
                "recomputed_table_sha256",
            }:
                raise ValueError("logical replay issue receipt schema changed")
            if receipt.get("issue_index") != expected_index or receipt.get(
                "accepted_table_sha256"
            ) != receipt.get("recomputed_table_sha256"):
                raise ValueError("logical replay issue identity differs from accepted input")
            accepted_sha256 = receipt.get("accepted_table_sha256")
            issue_id = receipt.get("issue_id")
            issue_report_id = receipt.get("issue_report_id")
            if (
                not isinstance(accepted_sha256, str)
                or _SHA256.fullmatch(accepted_sha256) is None
                or not isinstance(issue_id, str)
                or not issue_id
                or not isinstance(issue_report_id, str)
                or not issue_report_id
            ):
                raise ValueError("logical replay issue receipt identity is malformed")
            issue_ids.append(issue_id)
        if len(set(issue_ids)) != 153:
            raise ValueError("logical replay issue IDs are not unique")
        receipt_lists.append(receipts)
    if receipt_lists[0] != receipt_lists[1]:
        raise ValueError("logical replay receipts differ between 1 and 2 workers")
    content_sha256 = document.get("content_sha256")
    if not isinstance(content_sha256, str):
        raise ValueError("logical replay audit content SHA-256 is missing")
    return content_sha256


def _build(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    stage4_junit_path: Path,
    full_junit_path: Path,
    preflight_receipt_path: Path,
    logical_replay_audit_path: Path,
    git_runner: GitCommandRunner,
) -> Stage4QualificationEvidence:
    root = Path(project_root).resolve()
    safe_stage4_junit = require_score_blind_project_path(
        root,
        protocol,
        stage4_junit_path,
        label="stage4 JUnit",
    )
    safe_full_junit = require_score_blind_project_path(
        root,
        protocol,
        full_junit_path,
        label="full JUnit",
    )
    safe_preflight = require_score_blind_project_path(
        root,
        protocol,
        preflight_receipt_path,
        label="formal preflight receipt",
    )
    safe_logical_replay = require_score_blind_project_path(
        root,
        protocol,
        logical_replay_audit_path,
        label="R1 logical replay audit",
    )
    # Compare the frozen lexical paths without following a component that could
    # be swapped after the score-blind guard has inspected it.
    canonical_preflight = root.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    if safe_preflight != canonical_preflight:
        raise ValueError("formal preflight receipt must use its canonical local path")
    for key, safe_path, label in (
        ("stage4_junit_path", safe_stage4_junit, "stage-4 JUnit"),
        ("full_non_target_junit_path", safe_full_junit, "full non-target JUnit"),
    ):
        relative = stage4_scoring_freeze_relative_path(protocol, key)
        if safe_path != root.joinpath(*relative.parts):
            raise ValueError(f"{label} must use its frozen R1 path")
    stage4_junit = parse_pytest_junit_evidence(
        _read_stable_bytes(safe_stage4_junit, label="stage-4 JUnit evidence")
    )
    full_junit = parse_pytest_junit_evidence(
        _read_stable_bytes(safe_full_junit, label="full JUnit evidence")
    )
    receipt = load_formal_preflight_receipt(canonical_preflight)
    score_blind_inputs = observe_score_blind_inputs(root, protocol)
    scoring_code_commit = _current_commit(root, git_runner)
    canonical_logical_replay = root.joinpath(*STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH.parts)
    if safe_logical_replay != canonical_logical_replay:
        raise ValueError("logical replay audit must use its canonical R1 local path")
    logical_replay_sha256 = _logical_replay_audit_sha256(
        _read_stable_bytes(safe_logical_replay, label="R1 logical replay audit"),
        protocol_design_sha256=score_blind_inputs.protocol_design_sha256,
        random_input_seal_sha256=score_blind_inputs.random_input_seal_sha256,
        score_blind_input_evidence_sha256=score_blind_inputs.content_sha256,
        scoring_code_commit=scoring_code_commit,
    )
    evidence = build_stage4_qualification_evidence(
        protocol,
        scoring_code_commit=scoring_code_commit,
        score_blind_input_evidence=score_blind_inputs,
        formal_preflight_receipt=receipt,
        logical_identity_replay_audit_sha256=logical_replay_sha256,
        stage4_pytest=stage4_junit,
        full_pytest=full_junit,
        gpu_requested=True,
        gpu_equivalence=None,
    )
    if (
        evidence.gpu_status != "blocked_no_frozen_backend"
        or evidence.formal_backend != "cpu_float64"
    ):
        raise ValueError(
            "current stage-4 freeze requires requested GPU to be blocked and formal CPU retained"
        )
    return evidence


def _summary(action: str, evidence: Stage4QualificationEvidence) -> dict[str, object]:
    return {
        "action": action,
        "formal_attempt_count": 0,
        "formal_backend": evidence.formal_backend,
        "formal_preflight_receipt_sha256": (evidence.formal_preflight_receipt_sha256),
        "full_non_target_test_count": evidence.full_pytest.test_count,
        "gpu_requested": evidence.gpu_requested,
        "gpu_status": evidence.gpu_status,
        "logical_identity_replay_audit_sha256": (evidence.logical_identity_replay_audit_sha256),
        "qualification_evidence_sha256": evidence.content_sha256,
        "scoring_code_commit": evidence.scoring_code_commit,
        "space_placebo_resource_observation_sha256": (
            evidence.space_placebo_resource_observation_sha256
        ),
        "stage4_test_count": evidence.stage4_pytest.test_count,
        "target_path_observed": False,
        "target_read_count": 0,
    }


def generate(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    stage4_junit_path: Path,
    full_junit_path: Path,
    preflight_receipt_path: Path,
    qualification_path: Path,
    logical_replay_audit_path: Path = LOGICAL_REPLAY_AUDIT_PATH,
    git_runner: GitCommandRunner = subprocess_git_runner,
) -> dict[str, object]:
    output = require_score_blind_project_path(
        project_root,
        protocol,
        qualification_path,
        label="qualification output",
    )
    canonical_output = project_root.resolve().joinpath(
        *stage4_scoring_freeze_relative_path(protocol, "qualification_path").parts
    )
    if output != canonical_output:
        raise ValueError("qualification output must use its frozen R1 path")
    evidence = _build(
        project_root,
        protocol,
        stage4_junit_path=stage4_junit_path,
        full_junit_path=full_junit_path,
        preflight_receipt_path=preflight_receipt_path,
        logical_replay_audit_path=logical_replay_audit_path,
        git_runner=git_runner,
    )
    if output.is_file():
        if load_stage4_qualification_evidence(output).as_mapping() != evidence.as_mapping():
            raise ValueError("existing stage-4 qualification differs; refusing to overwrite")
    else:
        write_stage4_qualification_evidence_atomic(output, evidence)
    return _summary("generate", evidence)


def check(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    stage4_junit_path: Path,
    full_junit_path: Path,
    preflight_receipt_path: Path,
    qualification_path: Path,
    logical_replay_audit_path: Path = LOGICAL_REPLAY_AUDIT_PATH,
    git_runner: GitCommandRunner = subprocess_git_runner,
) -> dict[str, object]:
    output = require_score_blind_project_path(
        project_root,
        protocol,
        qualification_path,
        label="qualification output",
    )
    canonical_output = project_root.resolve().joinpath(
        *stage4_scoring_freeze_relative_path(protocol, "qualification_path").parts
    )
    if output != canonical_output:
        raise ValueError("qualification output must use its frozen R1 path")
    expected = _build(
        project_root,
        protocol,
        stage4_junit_path=stage4_junit_path,
        full_junit_path=full_junit_path,
        preflight_receipt_path=preflight_receipt_path,
        logical_replay_audit_path=logical_replay_audit_path,
        git_runner=git_runner,
    )
    if load_stage4_qualification_evidence(output).as_mapping() != expected.as_mapping():
        raise ValueError("stage-4 qualification is stale or altered")
    result = _summary("check", expected)
    result["verified"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "check"))
    parser.add_argument("--stage4-junit", type=Path, default=STAGE4_JUNIT_PATH)
    parser.add_argument("--full-junit", type=Path, default=FULL_JUNIT_PATH)
    parser.add_argument(
        "--formal-preflight-receipt",
        type=Path,
        default=PREFLIGHT_RECEIPT_PATH,
    )
    parser.add_argument("--qualification", type=Path, default=QUALIFICATION_PATH)
    parser.add_argument(
        "--logical-replay-audit",
        type=Path,
        default=LOGICAL_REPLAY_AUDIT_PATH,
    )
    args = parser.parse_args()
    protocol = _load_protocol(PROTOCOL_PATH)
    if protocol.get("protocol_version") != "0.4.0":
        raise ValueError("stage-4 qualification requires protocol_version 0.4.0")
    function = generate if args.action == "generate" else check
    result = function(
        PROJECT_ROOT,
        protocol,
        stage4_junit_path=args.stage4_junit,
        full_junit_path=args.full_junit,
        preflight_receipt_path=args.formal_preflight_receipt,
        qualification_path=args.qualification,
        logical_replay_audit_path=args.logical_replay_audit,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
