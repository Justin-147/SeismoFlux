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

STAGE4_EXECUTION_REVISION: Final = "r1"
STAGE4_PROTOCOL_PATH: Final = Path("configs/anomaly_increment_r1.yaml")
STAGE4_PROTOCOL_TAG: Final = "v0.3.0-anomaly-increment-protocol-r1"
STAGE4_SCORING_CODE_TAG: Final = "v0.3.0-anomaly-increment-scoring-code-r1"
STAGE4_RESULT_TAG: Final = "v0.3.0-anomaly-increment-r1"
STAGE4_SCORING_SEAL_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r1_scoring_seal.json"
)
STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r1/formal_preflight_receipt.json"
)
STAGE4_QUALIFICATION_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r1/scoring_qualification.json"
)
STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r1/logical_identity_worker_replay.json"
)
STAGE4_JUNIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r1/qualification_stage4.junit.xml"
)
STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r1/qualification_full_non_target.junit.xml"
)
STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r1_attempt_ledger.json"
)
STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r1_target_read_ledger.json"
)
STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r1/checkpoints"
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


def validate_stage4_r1_execution_contract(protocol: Mapping[str, object]) -> None:
    """Require the exact R1 execution namespace before any scoring authorization."""

    if protocol.get("protocol_version") != "0.4.0":
        raise ValueError("stage-4 scientific protocol_version must remain 0.4.0")
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    if freeze.get("execution_revision") != STAGE4_EXECUTION_REVISION:
        raise ValueError("stage-4 execution revision must be r1")
    if freeze.get("corrects_execution_revision") != "r0":
        raise ValueError("stage-4 R1 must identify the corrected r0 execution")
    if freeze.get("execution_revision_document") != "docs/anomaly_increment_protocol_r1.md":
        raise ValueError("stage-4 R1 execution-revision document path changed")
    if freeze.get("readiness_incident_document") != (
        "docs/phase4_scoring_readiness_incident_r0.md"
    ):
        raise ValueError("stage-4 R1 readiness-incident document path changed")
    if freeze.get("pre_score_tag") != STAGE4_PROTOCOL_TAG:
        raise ValueError("stage-4 R1 protocol tag changed")
    if freeze.get("results_tag") != STAGE4_RESULT_TAG:
        raise ValueError("stage-4 R1 results tag changed")

    scoring = _mapping(
        freeze.get("scoring_code_freeze"),
        label="freeze.scoring_code_freeze",
    )
    if scoring.get("expected_tag") != STAGE4_SCORING_CODE_TAG:
        raise ValueError("stage-4 R1 scoring-code tag changed")
    for key, expected in _STAGE4_SCORING_FREEZE_PATHS:
        if scoring.get(key) != expected.as_posix():
            raise ValueError(f"stage-4 R1 scoring freeze path changed: {key}")
    logical_identity = _mapping(
        scoring.get("selected_table_logical_identity"),
        label="freeze.scoring_code_freeze.selected_table_logical_identity",
    )
    if logical_identity != _STAGE4_LOGICAL_IDENTITY_CONTRACT:
        raise ValueError("stage-4 R1 selected-table logical identity contract changed")


def stage4_scoring_freeze_relative_path(
    protocol: Mapping[str, object],
    key: str,
) -> PurePosixPath:
    """Return one validated R1 scoring path from the frozen machine contract."""

    validate_stage4_r1_execution_contract(protocol)
    paths = dict(_STAGE4_SCORING_FREEZE_PATHS)
    try:
        expected = paths[key]
    except KeyError as exc:
        raise ValueError(f"unknown stage-4 R1 scoring freeze path: {key}") from exc
    scoring = _mapping(
        _mapping(protocol.get("freeze"), label="freeze").get("scoring_code_freeze"),
        label="freeze.scoring_code_freeze",
    )
    raw = _string(scoring.get(key), label=f"freeze.scoring_code_freeze.{key}")
    relative = PurePosixPath(raw)
    if relative != expected or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"stage-4 R1 scoring freeze path changed: {key}")
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
        raise ValueError("stage-4 execution must use the sole R1 protocol path")
    protocol_path = root / protocol_relative_path
    require_existing_real_directory_tree(
        root,
        protocol_path.parent,
        label="stage-4 protocol directory",
    )
    protocol = _read_yaml_mapping(protocol_path)
    validate_stage4_r1_execution_contract(protocol)

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
    "validate_stage4_r1_execution_contract",
]
