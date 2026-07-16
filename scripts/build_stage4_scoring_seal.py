"""Generate or verify the target-unread stage-4 scoring-code execution seal."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml

from seismoflux.anomaly_increment.attempt_ledger import (
    initialize_stage4_ledger,
    read_stage4_ledger,
)
from seismoflux.anomaly_increment.authorization import (
    STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
    STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
    GitStage4RepositoryAdapter,
    Stage4RepositoryAdapter,
    build_stage4_scoring_seal,
    load_stage4_scoring_seal,
    stage4_execution_binding_id,
    write_stage4_scoring_seal_atomic,
)
from seismoflux.anomaly_increment.config import (
    STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH,
    STAGE4_PROTOCOL_PATH,
    STAGE4_PROTOCOL_TAG,
    STAGE4_QUALIFICATION_RELATIVE_PATH,
    STAGE4_SCORING_CODE_TAG,
    stage4_scoring_freeze_relative_path,
    validate_stage4_r2_execution_contract,
)
from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_PREFLIGHT_RECEIPT_PATH,
    load_formal_preflight_receipt,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
    sha256_existing_immutable_file,
)
from seismoflux.anomaly_increment.qualification import (
    load_stage4_qualification_evidence,
    observe_score_blind_inputs,
)
from seismoflux.anomaly_increment.score_blind_path import (
    require_score_blind_project_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / STAGE4_PROTOCOL_PATH
QUALIFICATION_PATH = PROJECT_ROOT.joinpath(*STAGE4_QUALIFICATION_RELATIVE_PATH.parts)
ATTEMPT_LEDGER_PATH = PROJECT_ROOT.joinpath(
    *PurePosixPath(STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH).parts
)
TARGET_READ_LEDGER_PATH = PROJECT_ROOT.joinpath(
    *PurePosixPath(STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH).parts
)
PREFLIGHT_RECEIPT_PATH = PROJECT_ROOT.joinpath(*STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH.parts)


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(dict[str, Any], dict(value))


def _load_protocol(path: Path) -> dict[str, Any]:
    target = Path(path)
    try:
        require_existing_real_directory_tree(
            Path(target.anchor) if target.anchor else Path.cwd(),
            target.parent,
            label="stage-4 protocol parent directory",
        )
        payload = read_existing_immutable_bytes(target, label="stage-4 protocol")
    except UnsafeImmutableFileError as exc:
        raise ValueError("stage-4 protocol cannot be read safely") from exc
    try:
        protocol = _mapping(yaml.safe_load(payload.decode("utf-8")), label="protocol")
    except UnicodeError as exc:
        raise ValueError("stage-4 protocol is not valid UTF-8") from exc
    validate_stage4_r2_execution_contract(protocol)
    return protocol


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_output_path(project_root: Path, protocol: Mapping[str, object]) -> Path:
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    scoring = _mapping(freeze.get("scoring_code_freeze"), label="scoring_code_freeze")
    raw = scoring.get("required_seal_path")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError("required stage-4 scoring seal path is invalid")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
        raise ValueError("required stage-4 scoring seal path is invalid")
    root = Path(project_root).resolve()
    output = root.joinpath(*relative.parts)
    if not output.is_relative_to(root):
        raise ValueError("required stage-4 scoring seal path escapes the project root")
    return output


def _freeze_tags(protocol: Mapping[str, object]) -> tuple[str, str]:
    validate_stage4_r2_execution_contract(protocol)
    return STAGE4_PROTOCOL_TAG, STAGE4_SCORING_CODE_TAG


def _require_canonical_ledgers(
    root: Path,
    attempt_ledger_path: Path,
    target_read_ledger_path: Path,
) -> tuple[Path, Path]:
    expected_attempt = _lexical_absolute(
        root.joinpath(*PurePosixPath(STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH).parts)
    )
    expected_target = _lexical_absolute(
        root.joinpath(*PurePosixPath(STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH).parts)
    )
    if _lexical_absolute(attempt_ledger_path) != expected_attempt:
        raise ValueError("stage-4 attempt ledger path is not the frozen repository path")
    if _lexical_absolute(target_read_ledger_path) != expected_target:
        raise ValueError("stage-4 target-read ledger path is not the frozen repository path")
    return expected_attempt, expected_target


def generate(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    qualification_path: Path,
    preflight_receipt_path: Path,
    attempt_ledger_path: Path,
    target_read_ledger_path: Path,
    repository_adapter: Stage4RepositoryAdapter,
) -> dict[str, object]:
    """Generate the seal from target-blind evidence and empty ledgers only."""

    root = Path(project_root).resolve()
    safe_qualification_path = require_score_blind_project_path(
        root,
        protocol,
        qualification_path,
        label="qualification evidence",
    )
    safe_preflight_receipt_path = require_score_blind_project_path(
        root,
        protocol,
        preflight_receipt_path,
        label="formal preflight receipt",
    )
    safe_attempt_ledger_path = require_score_blind_project_path(
        root,
        protocol,
        attempt_ledger_path,
        label="formal-attempt ledger",
    )
    safe_target_read_ledger_path = require_score_blind_project_path(
        root,
        protocol,
        target_read_ledger_path,
        label="target-read audit ledger",
    )
    canonical_preflight = _lexical_absolute(root.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts))
    if safe_preflight_receipt_path != canonical_preflight:
        raise ValueError("formal preflight receipt must use its canonical local path")
    output_path = require_score_blind_project_path(
        root,
        protocol,
        _relative_output_path(root, protocol),
        label="stage-4 scoring seal output",
    )
    canonical_qualification = _lexical_absolute(
        root.joinpath(*stage4_scoring_freeze_relative_path(protocol, "qualification_path").parts)
    )
    if safe_qualification_path != canonical_qualification:
        raise ValueError("qualification evidence must use its frozen R2 path")
    attempt_ledger_path, target_read_ledger_path = _require_canonical_ledgers(
        root,
        safe_attempt_ledger_path,
        safe_target_read_ledger_path,
    )
    protocol_tag, scoring_tag = _freeze_tags(protocol)
    score_blind_inputs = observe_score_blind_inputs(root, protocol)
    qualification = load_stage4_qualification_evidence(safe_qualification_path)
    preflight_receipt = load_formal_preflight_receipt(safe_preflight_receipt_path)
    repository = repository_adapter.observe(
        root,
        protocol_tag=protocol_tag,
        scoring_code_tag=scoring_tag,
        allowed_untracked_paths=(output_path.relative_to(root).as_posix(),),
    )
    binding = stage4_execution_binding_id(repository, score_blind_inputs, qualification)
    attempt_ledger = initialize_stage4_ledger(
        attempt_ledger_path,
        kind="formal_attempt",
        execution_binding_id=binding,
    )
    target_read_ledger = initialize_stage4_ledger(
        target_read_ledger_path,
        kind="target_read",
        execution_binding_id=binding,
    )
    seal = build_stage4_scoring_seal(
        protocol,
        repository=repository,
        score_blind_inputs=score_blind_inputs,
        qualification=qualification,
        formal_preflight_receipt=preflight_receipt,
        attempt_ledger=attempt_ledger,
        target_read_ledger=target_read_ledger,
    )
    file_sha256 = write_stage4_scoring_seal_atomic(output_path, seal)
    return {
        "action": "generate",
        "formal_attempt_count": 0,
        "formal_backend": qualification.formal_backend,
        "formal_preflight_receipt_sha256": preflight_receipt.content_sha256,
        "gpu_requested": qualification.gpu_requested,
        "gpu_status": qualification.gpu_status,
        "scoring_code_commit": repository.code_commit,
        "scoring_seal_file_sha256": file_sha256,
        "scoring_seal_id": seal.seal_id,
        "target_path_observed": False,
        "target_read_count": 0,
    }


def check(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    qualification_path: Path,
    preflight_receipt_path: Path,
    attempt_ledger_path: Path,
    target_read_ledger_path: Path,
    repository_adapter: Stage4RepositoryAdapter,
) -> dict[str, object]:
    """Rebuild the target-unread seal and require byte-for-byte identity."""

    root = Path(project_root).resolve()
    safe_qualification_path = require_score_blind_project_path(
        root,
        protocol,
        qualification_path,
        label="qualification evidence",
    )
    safe_preflight_receipt_path = require_score_blind_project_path(
        root,
        protocol,
        preflight_receipt_path,
        label="formal preflight receipt",
    )
    safe_attempt_ledger_path = require_score_blind_project_path(
        root,
        protocol,
        attempt_ledger_path,
        label="formal-attempt ledger",
    )
    safe_target_read_ledger_path = require_score_blind_project_path(
        root,
        protocol,
        target_read_ledger_path,
        label="target-read audit ledger",
    )
    canonical_preflight = _lexical_absolute(root.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts))
    if safe_preflight_receipt_path != canonical_preflight:
        raise ValueError("formal preflight receipt must use its canonical local path")
    output_path = require_score_blind_project_path(
        root,
        protocol,
        _relative_output_path(root, protocol),
        label="stage-4 scoring seal output",
    )
    canonical_qualification = _lexical_absolute(
        root.joinpath(*stage4_scoring_freeze_relative_path(protocol, "qualification_path").parts)
    )
    if safe_qualification_path != canonical_qualification:
        raise ValueError("qualification evidence must use its frozen R2 path")
    attempt_ledger_path, target_read_ledger_path = _require_canonical_ledgers(
        root,
        safe_attempt_ledger_path,
        safe_target_read_ledger_path,
    )
    loaded = load_stage4_scoring_seal(output_path)
    protocol_tag, scoring_tag = _freeze_tags(protocol)
    score_blind_inputs = observe_score_blind_inputs(root, protocol)
    qualification = load_stage4_qualification_evidence(safe_qualification_path)
    preflight_receipt = load_formal_preflight_receipt(safe_preflight_receipt_path)
    repository = repository_adapter.observe(
        root,
        protocol_tag=protocol_tag,
        scoring_code_tag=scoring_tag,
        allowed_untracked_paths=(output_path.relative_to(root).as_posix(),),
    )
    if qualification != loaded.qualification:
        raise ValueError("qualification evidence differs from the local scoring seal")
    binding = stage4_execution_binding_id(repository, score_blind_inputs, qualification)
    attempt_ledger = read_stage4_ledger(
        attempt_ledger_path,
        expected_kind="formal_attempt",
        expected_binding_id=binding,
    )
    target_read_ledger = read_stage4_ledger(
        target_read_ledger_path,
        expected_kind="target_read",
        expected_binding_id=binding,
    )
    rebuilt = build_stage4_scoring_seal(
        protocol,
        repository=repository,
        score_blind_inputs=score_blind_inputs,
        qualification=qualification,
        formal_preflight_receipt=preflight_receipt,
        attempt_ledger=attempt_ledger,
        target_read_ledger=target_read_ledger,
    )
    if rebuilt.as_mapping() != loaded.as_mapping():
        raise ValueError("stage-4 scoring seal is stale or altered")
    return {
        "action": "check",
        "formal_attempt_count": 0,
        "formal_backend": qualification.formal_backend,
        "formal_preflight_receipt_sha256": preflight_receipt.content_sha256,
        "gpu_requested": qualification.gpu_requested,
        "gpu_status": qualification.gpu_status,
        "scoring_seal_file_sha256": sha256_existing_immutable_file(
            output_path,
            label="stage-4 scoring seal",
        ),
        "scoring_seal_id": loaded.seal_id,
        "target_path_observed": False,
        "target_read_count": 0,
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "check"))
    parser.add_argument("--qualification-evidence", type=Path, default=QUALIFICATION_PATH)
    parser.add_argument(
        "--formal-preflight-receipt",
        type=Path,
        default=PREFLIGHT_RECEIPT_PATH,
    )
    args = parser.parse_args()
    protocol = _load_protocol(PROTOCOL_PATH)
    if protocol.get("protocol_version") != "0.4.1":
        raise ValueError("stage-4 scoring seal requires protocol_version 0.4.1")
    function = generate if args.action == "generate" else check
    result = function(
        PROJECT_ROOT,
        protocol,
        qualification_path=args.qualification_evidence,
        preflight_receipt_path=args.formal_preflight_receipt,
        attempt_ledger_path=ATTEMPT_LEDGER_PATH,
        target_read_ledger_path=TARGET_READ_LEDGER_PATH,
        repository_adapter=GitStage4RepositoryAdapter(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
