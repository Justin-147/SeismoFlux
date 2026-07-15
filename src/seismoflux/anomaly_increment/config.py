"""Strict, target-blind loading of the frozen stage-4 protocol bundle.

This module deliberately has no target-catalog loader.  The earthquake target is
represented only by the identity strings that were frozen in the protocol.  In
particular, loading or validating a :class:`Stage4ProtocolBundle` must not call
``exists()``, ``stat()``, ``open()`` or a hashing function on the target path.
The only target byte access lives in :mod:`seismoflux.anomaly_increment.target_access`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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

STAGE4_PROTOCOL_PATH: Final = Path("configs/anomaly_increment.yaml")
STAGE4_PROTOCOL_TAG: Final = "v0.3.0-anomaly-increment-protocol"
STAGE4_SCORING_CODE_TAG: Final = "v0.3.0-anomaly-increment-scoring-code"
STAGE4_RESULT_TAG: Final = "v0.3.0-anomaly-increment"

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
    protocol_path = root / protocol_relative_path
    require_existing_real_directory_tree(
        root,
        protocol_path.parent,
        label="stage-4 protocol directory",
    )
    protocol = _read_yaml_mapping(protocol_path)
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    scoring_freeze = _mapping(freeze.get("scoring_code_freeze"), label="freeze.scoring_code_freeze")
    if freeze.get("pre_score_tag") != STAGE4_PROTOCOL_TAG:
        raise ValueError("stage-4 protocol tag changed")
    if freeze.get("results_tag") != STAGE4_RESULT_TAG:
        raise ValueError("stage-4 results tag changed")
    if scoring_freeze.get("expected_tag") != STAGE4_SCORING_CODE_TAG:
        raise ValueError("stage-4 scoring-code tag changed")

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
    "STAGE4_PROTOCOL_PATH",
    "STAGE4_PROTOCOL_TAG",
    "STAGE4_RESULT_TAG",
    "STAGE4_SCORING_CODE_TAG",
    "ExpectedTargetIdentity",
    "FrozenManifest",
    "Stage4ProtocolBundle",
    "load_stage4_protocol_bundle",
]
