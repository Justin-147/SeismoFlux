"""Frozen protocol identities for the Stage 2S one-shot development screen.

This module only reads the public target-blind protocol files and Git metadata.
It never opens, stats, or otherwise probes the study area, cell mapping, or
earthquake catalogue.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

PROTOCOL_CONFIG_RELATIVE_PATH = Path("configs/causal_seismicity_screen.yaml")
FOLD_MANIFEST_RELATIVE_PATH = Path("data/manifests/causal_seismicity_screen_fold_manifest.json")
INPUT_CONTRACT_RELATIVE_PATH = Path(
    "data/manifests/causal_seismicity_screen_target_blind_input_contract.json"
)

PROTOCOL_CONFIG_SHA256 = "a85df78348c0f033444db4c9e3edc81b70ef436da3b108139feab39cd49d8c42"
FOLD_MANIFEST_SHA256 = "c3e2444e8892addd03d4c57526c007e2a861137dac50d5abe2e53bac004456e6"
INPUT_CONTRACT_SHA256 = "50117a0c0cda0d14bd467b8f0d1032855cb5afab0aa2d968370313a933a95ff6"
PROTOCOL_COMMIT = "98e21573057d9a73d552b0cbac7a64f5206b3546"
PROTOCOL_TAG = "v0.2.3-causal-seismicity-screen-protocol"
CODE_TAG = "v0.2.3-causal-seismicity-screen-code"
EXPERIMENT_ID = "stage2s-causal-seismicity-development-v1"
ATTEMPT_ID = f"{EXPERIMENT_ID}-attempt-1"


class Stage2SProtocolError(RuntimeError):
    """Raised when a frozen Stage 2S identity no longer matches the protocol."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_frozen_bytes(repository_root: Path, relative_path: Path) -> bytes:
    if repository_root.is_absolute() is False:
        raise ValueError("repository_root must be absolute")
    path = repository_root / relative_path
    try:
        return path.read_bytes()
    except OSError as exc:
        raise Stage2SProtocolError(f"cannot read frozen protocol file: {relative_path}") from exc


def _load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2SProtocolError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Stage2SProtocolError(f"{label} root must be a mapping")
    return cast(dict[str, Any], value)


def _load_yaml(payload: bytes) -> dict[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise Stage2SProtocolError("Stage 2S config is not valid UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise Stage2SProtocolError("Stage 2S config root must be a mapping")
    return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class Stage2SProtocolBundle:
    """Verified target-blind configuration and calendar identities."""

    repository_root: Path
    config: dict[str, Any]
    fold_manifest: dict[str, Any]
    input_contract: dict[str, Any]
    config_sha256: str
    fold_manifest_sha256: str
    input_contract_sha256: str

    @property
    def protocol_tag(self) -> str:
        return PROTOCOL_TAG

    @property
    def expected_code_tag(self) -> str:
        return CODE_TAG

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_ID

    @property
    def attempt_id(self) -> str:
        return ATTEMPT_ID

    def identity_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "experiment_id": self.experiment_id,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_tag": self.protocol_tag,
            "protocol_config_sha256": self.config_sha256,
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "target_blind_input_contract_sha256": self.input_contract_sha256,
        }


def load_protocol_bundle(repository_root: Path) -> Stage2SProtocolBundle:
    """Load and verify only the three public Stage 2S protocol files."""

    root = repository_root.resolve()
    config_bytes = _read_frozen_bytes(root, PROTOCOL_CONFIG_RELATIVE_PATH)
    fold_bytes = _read_frozen_bytes(root, FOLD_MANIFEST_RELATIVE_PATH)
    input_bytes = _read_frozen_bytes(root, INPUT_CONTRACT_RELATIVE_PATH)
    observed = (
        _sha256(config_bytes),
        _sha256(fold_bytes),
        _sha256(input_bytes),
    )
    expected = (
        PROTOCOL_CONFIG_SHA256,
        FOLD_MANIFEST_SHA256,
        INPUT_CONTRACT_SHA256,
    )
    if observed != expected:
        labels = ("protocol config", "fold manifest", "target-blind input contract")
        mismatches = [
            f"{label}: expected {wanted}, observed {actual}"
            for label, wanted, actual in zip(labels, expected, observed, strict=True)
            if wanted != actual
        ]
        raise Stage2SProtocolError("; ".join(mismatches))
    config = _load_yaml(config_bytes)
    folds = _load_json(fold_bytes, label="fold manifest")
    inputs = _load_json(input_bytes, label="target-blind input contract")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise Stage2SProtocolError("config experiment_id changed")
    if folds.get("experiment_id") != EXPERIMENT_ID:
        raise Stage2SProtocolError("fold manifest experiment_id changed")
    if inputs.get("experiment_id") != EXPERIMENT_ID:
        raise Stage2SProtocolError("input contract experiment_id changed")
    governance = config.get("governance")
    if not isinstance(governance, dict):
        raise Stage2SProtocolError("config governance block is missing")
    if governance.get("protocol_tag") != PROTOCOL_TAG:
        raise Stage2SProtocolError("protocol tag changed")
    if governance.get("expected_code_tag") != CODE_TAG:
        raise Stage2SProtocolError("code tag changed")
    if governance.get("attempt_id") != ATTEMPT_ID:
        raise Stage2SProtocolError("attempt identity changed")
    if config.get("allowed_models", {}).get("exact_order") != ["S0", "S1", "SP"]:
        raise Stage2SProtocolError("frozen model order changed")
    if folds.get("security", {}).get("development_target_read_authorized") is not False:
        raise Stage2SProtocolError("fold manifest unexpectedly authorizes target reads")
    if inputs.get("security", {}).get("real_catalog_bytes_read") is not False:
        raise Stage2SProtocolError("input contract unexpectedly records a catalog read")
    return Stage2SProtocolBundle(
        repository_root=root,
        config=config,
        fold_manifest=folds,
        input_contract=inputs,
        config_sha256=observed[0],
        fold_manifest_sha256=observed[1],
        input_contract_sha256=observed[2],
    )


def _git_output(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise Stage2SProtocolError(f"Git identity check failed: {' '.join(arguments)}") from exc
    return completed.stdout.strip()


def verify_local_protocol_tag(repository_root: Path) -> str:
    """Verify that the frozen protocol tag peels to its preregistered commit."""

    peeled = _git_output(repository_root, "rev-parse", f"{PROTOCOL_TAG}^{{commit}}")
    if peeled != PROTOCOL_COMMIT:
        raise Stage2SProtocolError("local protocol tag does not peel to the frozen commit")
    return peeled


def verify_local_code_tag(repository_root: Path) -> str:
    """Return the commit peeled from the immutable Stage 2S code tag."""

    peeled = _git_output(repository_root, "rev-parse", f"{CODE_TAG}^{{commit}}")
    head = _git_output(repository_root, "rev-parse", "HEAD")
    if peeled != head:
        raise Stage2SProtocolError("formal Stage 2S execution requires HEAD at the code tag")
    return peeled


__all__ = [
    "ATTEMPT_ID",
    "CODE_TAG",
    "EXPERIMENT_ID",
    "FOLD_MANIFEST_RELATIVE_PATH",
    "FOLD_MANIFEST_SHA256",
    "INPUT_CONTRACT_RELATIVE_PATH",
    "INPUT_CONTRACT_SHA256",
    "PROTOCOL_COMMIT",
    "PROTOCOL_CONFIG_RELATIVE_PATH",
    "PROTOCOL_CONFIG_SHA256",
    "PROTOCOL_TAG",
    "Stage2SProtocolBundle",
    "Stage2SProtocolError",
    "load_protocol_bundle",
    "verify_local_code_tag",
    "verify_local_protocol_tag",
]
