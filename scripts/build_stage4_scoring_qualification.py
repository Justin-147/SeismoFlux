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

from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_PREFLIGHT_RECEIPT_PATH,
    load_formal_preflight_receipt,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
)
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
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "anomaly_increment.yaml"
QUALIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "stage4"
    / "anomaly_increment"
    / "scoring_qualification.json"
)
PREFLIGHT_RECEIPT_PATH = PROJECT_ROOT.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
STAGE4_JUNIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "stage4"
    / "anomaly_increment"
    / "qualification_stage4.junit.xml"
)
FULL_JUNIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "stage4"
    / "anomaly_increment"
    / "qualification_full_non_target.junit.xml"
)

_GIT_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


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
    return _mapping(yaml.safe_load(text), label="protocol")


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


def _build(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    stage4_junit_path: Path,
    full_junit_path: Path,
    preflight_receipt_path: Path,
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
    stage4_junit = parse_pytest_junit_evidence(
        _read_stable_bytes(safe_stage4_junit, label="stage-4 JUnit evidence")
    )
    full_junit = parse_pytest_junit_evidence(
        _read_stable_bytes(safe_full_junit, label="full JUnit evidence")
    )
    canonical_preflight = root.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts).resolve()
    if safe_preflight != canonical_preflight:
        raise ValueError("formal preflight receipt must use its canonical local path")
    receipt = load_formal_preflight_receipt(canonical_preflight)
    score_blind_inputs = observe_score_blind_inputs(root, protocol)
    evidence = build_stage4_qualification_evidence(
        protocol,
        scoring_code_commit=_current_commit(root, git_runner),
        score_blind_input_evidence=score_blind_inputs,
        formal_preflight_receipt=receipt,
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
    git_runner: GitCommandRunner = subprocess_git_runner,
) -> dict[str, object]:
    output = require_score_blind_project_path(
        project_root,
        protocol,
        qualification_path,
        label="qualification output",
    )
    evidence = _build(
        project_root,
        protocol,
        stage4_junit_path=stage4_junit_path,
        full_junit_path=full_junit_path,
        preflight_receipt_path=preflight_receipt_path,
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
    git_runner: GitCommandRunner = subprocess_git_runner,
) -> dict[str, object]:
    output = require_score_blind_project_path(
        project_root,
        protocol,
        qualification_path,
        label="qualification output",
    )
    expected = _build(
        project_root,
        protocol,
        stage4_junit_path=stage4_junit_path,
        full_junit_path=full_junit_path,
        preflight_receipt_path=preflight_receipt_path,
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
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
