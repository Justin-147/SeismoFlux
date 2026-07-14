"""Deterministic publication and registry contracts for stage-2 background results.

The module accepts only already-computed scientific documents.  It performs no catalog
loading, fitting, scoring, or model selection.  Four explicitly typed bundle families are
published through the frozen content-address constructor and immutable artifact store;
the fixed registry and report are create-once projections over those bundles.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import ClassVar, Literal, Self, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from seismoflux.background.artifacts import (
    CANONICAL_JSON_VERSION,
    ArtifactFile,
    ArtifactPublication,
    ProjectRelativePath,
    canonical_json_bytes,
    publish_artifact,
)
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.execution import (
    ExecutionSeal,
    GitCommandRunner,
    RepositoryIdentity,
    build_address_inputs,
    require_execution_seal_unchanged,
    subprocess_git_runner,
)
from seismoflux.background.scoring_authorization import require_background_scoring_authorized

BundleKind: TypeAlias = Literal["processed", "model", "backtest", "experiment"]
ModelId: TypeAlias = Literal["uniform_poisson", "spatial_poisson", "etas"]
SnapshotId: TypeAlias = Literal[
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "final_validation",
]
AttemptStatus: TypeAlias = Literal["succeeded", "failed", "not_run"]
GateStatus: TypeAlias = Literal["passed", "failed", "not_applicable"]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

_BUNDLE_ORDER: tuple[BundleKind, ...] = ("processed", "model", "backtest", "experiment")
_MODEL_ORDER: tuple[ModelId, ...] = ("uniform_poisson", "spatial_poisson", "etas")
_SNAPSHOT_ORDER: tuple[SnapshotId, ...] = (
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "final_validation",
)
_REQUIRED_INPUT_HASH_KEYS = {
    "environment_lock",
    "data_catalog",
    "earthquake_dataset",
    "study_area",
    "issue_manifest",
    "production_fixture",
    "oracle_metadata",
}
_VOLATILE_KEYS = {
    "created_at",
    "created_at_utc",
    "generated_at",
    "generated_at_utc",
    "pid",
    "process_id",
    "run_id",
    "timestamp",
    "updated_at",
    "updated_at_utc",
    "uuid",
}
_RESERVED_CANONICAL_KEY = "$seismoflux_type"
_SHA256_LENGTH = 64
_ARTIFACT_ID_LENGTH = 16
_FORBIDDEN_REPORT_TERMS = ("probability", "概率")


class FixedFileError(RuntimeError):
    """Base class for immutable fixed-file failures."""


class FixedFileConflictError(FixedFileError):
    """A fixed path already exists but cannot be verified byte-for-byte."""


class FixedFilePublicationError(FixedFileError):
    """A fixed file could not be atomically created."""


class RegistryValidationError(ValueError):
    """Registry bytes do not satisfy the complete frozen schema."""


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _validated_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized != normalized.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    if any(ord(character) < 0x20 for character in normalized):
        raise ValueError(f"{name} must not contain control characters")
    return normalized


def _normalize_json(value: object, location: str = "$") -> JsonValue:
    """Copy a value into deterministic JSON while excluding volatile metadata."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite scientific value is forbidden at {location}")
        return value
    if isinstance(value, ProjectRelativePath):
        return value.value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"scientific document keys must be strings at {location}")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key == _RESERVED_CANONICAL_KEY:
                raise ValueError(f"reserved canonical JSON key at {location}")
            if normalized_key.casefold() in _VOLATILE_KEYS:
                raise ValueError(f"volatile scientific metadata is forbidden: {normalized_key}")
            if normalized_key in normalized:
                raise ValueError(f"scientific keys collide after NFC normalization at {location}")
            normalized[normalized_key] = _normalize_json(
                item,
                f"{location}.{normalized_key}",
            )
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, list | tuple):
        return [_normalize_json(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, PurePath):
        raise TypeError(f"paths require ProjectRelativePath at {location}")
    raise TypeError(f"unsupported scientific value at {location}: {type(value).__name__}")


def _normalized_mapping(value: object, *, name: str, require_nonempty: bool) -> dict[str, object]:
    normalized = _normalize_json(value, name)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must be a mapping")
    if require_nonempty and not normalized:
        raise ValueError(f"{name} must not be empty")
    return cast(dict[str, object], normalized)


class BundleDocument:
    """One canonical-JSON scientific file with no runtime metadata surface."""

    __slots__ = ("_document_json", "relative_path")
    relative_path: ProjectRelativePath
    _document_json: bytes

    def __init__(self, relative_path: str | ProjectRelativePath, document: Mapping[str, object]):
        path = (
            relative_path
            if isinstance(relative_path, ProjectRelativePath)
            else ProjectRelativePath(relative_path)
        )
        if Path(path.value).suffix.casefold() != ".json":
            raise ValueError("bundle documents must use a .json extension")
        normalized = _normalized_mapping(document, name="bundle document", require_nonempty=True)
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "_document_json", payload)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("BundleDocument is immutable")

    def artifact_file(self, bundle_kind: BundleKind) -> ArtifactFile:
        content = cast(dict[str, object], json.loads(self._document_json))
        payload = canonical_json_bytes(
            {
                "bundle_kind": bundle_kind,
                "content": content,
                "schema_version": 1,
            }
        )
        return ArtifactFile(
            relative_path=self.relative_path,
            payload=payload,
            media_type="application/vnd.seismoflux.scientific+json",
        )


class BundleBinary:
    """One immutable non-JSON payload, used for deterministic rendered artifacts."""

    __slots__ = ("_payload", "media_type", "relative_path")
    relative_path: ProjectRelativePath
    media_type: str
    _payload: bytes

    def __init__(
        self,
        relative_path: str | ProjectRelativePath,
        payload: bytes,
        *,
        media_type: str,
    ) -> None:
        path = (
            relative_path
            if isinstance(relative_path, ProjectRelativePath)
            else ProjectRelativePath(relative_path)
        )
        if Path(path.value).suffix.casefold() == ".json":
            raise ValueError("JSON payloads must use BundleDocument")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("bundle binary payload must be non-empty bytes")
        normalized_media_type = _validated_text(media_type, name="bundle binary media type")
        if normalized_media_type == "image/png" and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("image/png bundle payload must contain a PNG signature")
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "_payload", bytes(payload))
        object.__setattr__(self, "media_type", normalized_media_type)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("BundleBinary is immutable")

    def artifact_file(self, bundle_kind: BundleKind) -> ArtifactFile:
        del bundle_kind
        return ArtifactFile(
            relative_path=self.relative_path,
            payload=self._payload,
            media_type=self.media_type,
        )


BundleFile: TypeAlias = BundleDocument | BundleBinary


class _ScientificBundle:
    __slots__ = ("_documents", "_parameter_identity_json")

    bundle_kind: ClassVar[BundleKind]
    _documents: tuple[BundleFile, ...]
    _parameter_identity_json: bytes

    def __init__(
        self,
        parameter_identity: Mapping[str, object],
        documents: Sequence[BundleFile],
    ) -> None:
        normalized = _normalized_mapping(
            parameter_identity,
            name="bundle parameter identity",
            require_nonempty=True,
        )
        materialized = tuple(documents)
        if not materialized or any(
            not isinstance(item, BundleDocument | BundleBinary) for item in materialized
        ):
            raise ValueError("bundle files must contain BundleDocument or BundleBinary values")
        paths = tuple(item.relative_path.value for item in materialized)
        if len(set(paths)) != len(paths):
            raise ValueError("bundle document paths must be unique")
        object.__setattr__(
            self,
            "_parameter_identity_json",
            json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )
        object.__setattr__(
            self,
            "_documents",
            tuple(sorted(materialized, key=lambda item: item.relative_path.value)),
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("scientific bundles are immutable")

    def address_parameters(self) -> dict[str, object]:
        parameters = cast(dict[str, object], json.loads(self._parameter_identity_json))
        return {
            "artifact_role": self.bundle_kind,
            "scientific_parameters": parameters,
        }

    def artifact_files(self) -> tuple[ArtifactFile, ...]:
        return tuple(item.artifact_file(self.bundle_kind) for item in self._documents)


class ProcessedBundle(_ScientificBundle):
    __slots__ = ()
    bundle_kind: ClassVar[BundleKind] = "processed"


class ModelBundle(_ScientificBundle):
    __slots__ = ()
    bundle_kind: ClassVar[BundleKind] = "model"


class BacktestBundle(_ScientificBundle):
    __slots__ = ()
    bundle_kind: ClassVar[BundleKind] = "backtest"


class ExperimentBundle(_ScientificBundle):
    __slots__ = ()
    bundle_kind: ClassVar[BundleKind] = "experiment"


@dataclass(frozen=True, slots=True)
class BundlePublication:
    bundle_kind: BundleKind
    root: ProjectRelativePath
    artifact: ArtifactPublication
    code_commit: str
    freeze_tag: str

    def __post_init__(self) -> None:
        if self.bundle_kind not in _BUNDLE_ORDER:
            raise ValueError("unknown background bundle kind")
        if not isinstance(self.root, ProjectRelativePath):
            raise TypeError("bundle root must be ProjectRelativePath")
        if not isinstance(self.artifact, ArtifactPublication):
            raise TypeError("artifact must be ArtifactPublication")
        if not _is_lower_hex(self.artifact.artifact_id, _ARTIFACT_ID_LENGTH):
            raise ValueError("bundle artifact ID must be 16 lowercase hexadecimal characters")
        if not _is_lower_hex(self.artifact.manifest_sha256, _SHA256_LENGTH):
            raise ValueError("bundle manifest SHA-256 must be lowercase hexadecimal")
        if not _is_lower_hex(self.code_commit, 40) and not _is_lower_hex(
            self.code_commit, _SHA256_LENGTH
        ):
            raise ValueError("bundle code commit must be a lowercase Git object ID")
        _validated_text(self.freeze_tag, name="bundle freeze tag")


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    metadata = path.stat(follow_symlinks=False)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _validated_project_destination(project_root: Path, reference: str) -> tuple[Path, Path]:
    root_input = Path(project_root)
    if not _lexists(root_input) or _is_reparse_point(root_input) or not root_input.is_dir():
        raise ValueError("project root must be an existing plain directory")
    root = root_input.resolve()
    relative = ProjectRelativePath(reference)
    destination = root.joinpath(*relative.value.split("/"))
    resolved_destination = destination.resolve(strict=False)
    if not resolved_destination.is_relative_to(root):
        raise ValueError("project publication path escapes the project root")

    current = root
    for part in relative.value.split("/"):
        current /= part
        if not _lexists(current):
            continue
        if _is_reparse_point(current):
            raise ValueError("project publication path contains a symlink or reparse point")
        if current != destination and not current.is_dir():
            raise ValueError("project publication parent contains a non-directory entry")
    return root, destination


def _ensure_plain_directory(root: Path, directory: Path) -> None:
    if directory == root:
        return
    relative = directory.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if not _lexists(current):
            with suppress(FileExistsError):
                current.mkdir()
        if _is_reparse_point(current) or not current.is_dir():
            raise ValueError("project publication parent must remain a plain directory")


def _publish_bundle(
    project_root: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    bundle: _ScientificBundle,
    *,
    expected_type: type[_ScientificBundle],
    root_reference: str,
    uv_lock_sha256: str,
) -> BundlePublication:
    if type(bundle) is not expected_type:
        raise TypeError(f"expected {expected_type.__name__}")
    root, parent = _validated_project_destination(project_root, root_reference)
    _ensure_plain_directory(root, parent)
    address_inputs = build_address_inputs(
        background,
        identity,
        bundle.address_parameters(),
        uv_lock_sha256=uv_lock_sha256,
    )
    artifact = publish_artifact(
        parent,
        address_inputs=address_inputs,
        files=bundle.artifact_files(),
    )
    return BundlePublication(
        bundle_kind=bundle.bundle_kind,
        root=ProjectRelativePath(root_reference),
        artifact=artifact,
        code_commit=identity.code_commit,
        freeze_tag=identity.freeze_tag,
    )


def publish_processed_bundle(
    project_root: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    bundle: ProcessedBundle,
    *,
    uv_lock_sha256: str,
) -> BundlePublication:
    return _publish_bundle(
        project_root,
        background,
        identity,
        bundle,
        expected_type=ProcessedBundle,
        root_reference=background.outputs.processed_root,
        uv_lock_sha256=uv_lock_sha256,
    )


def publish_model_bundle(
    project_root: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    bundle: ModelBundle,
    *,
    uv_lock_sha256: str,
) -> BundlePublication:
    require_background_scoring_authorized(background)
    return _publish_bundle(
        project_root,
        background,
        identity,
        bundle,
        expected_type=ModelBundle,
        root_reference=background.outputs.model_root,
        uv_lock_sha256=uv_lock_sha256,
    )


def publish_backtest_bundle(
    project_root: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    bundle: BacktestBundle,
    *,
    uv_lock_sha256: str,
) -> BundlePublication:
    require_background_scoring_authorized(background)
    return _publish_bundle(
        project_root,
        background,
        identity,
        bundle,
        expected_type=BacktestBundle,
        root_reference=background.outputs.backtest_root,
        uv_lock_sha256=uv_lock_sha256,
    )


def publish_experiment_bundle(
    project_root: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    bundle: ExperimentBundle,
    *,
    uv_lock_sha256: str,
) -> BundlePublication:
    require_background_scoring_authorized(background)
    return _publish_bundle(
        project_root,
        background,
        identity,
        bundle,
        expected_type=ExperimentBundle,
        root_reference=background.outputs.experiment_root,
        uv_lock_sha256=uv_lock_sha256,
    )


class _StrictRegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BundleReference(_StrictRegistryModel):
    bundle_kind: BundleKind
    artifact_id: str = Field(min_length=16, max_length=16)
    manifest_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if not _is_lower_hex(value, _ARTIFACT_ID_LENGTH):
            raise ValueError("artifact_id must be lowercase hexadecimal")
        return value

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str) -> str:
        if not _is_lower_hex(value, _SHA256_LENGTH):
            raise ValueError("manifest_sha256 must be lowercase hexadecimal")
        return value


class GateOutcome(_StrictRegistryModel):
    gate_id: str
    status: GateStatus
    evidence_id: str

    @field_validator("gate_id", "evidence_id")
    @classmethod
    def validate_identifier(cls, value: str, info: object) -> str:
        del info
        return _validated_text(value, name="gate identifier")


class ModelAttemptRecord(_StrictRegistryModel):
    model_id: ModelId
    snapshot_id: SnapshotId
    status: AttemptStatus
    failure_reasons: tuple[str, ...]
    variant: str
    parameter_identity: dict[str, object]
    gates: tuple[GateOutcome, ...]
    score_ids: tuple[str, ...]

    @field_validator("failure_reasons", "score_ids")
    @classmethod
    def validate_ordered_text(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validated_text(value, name="attempt text") for value in values)
        if len(set(normalized)) != len(normalized) or normalized != tuple(sorted(normalized)):
            raise ValueError("attempt text tuples must be sorted and unique")
        return normalized

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, value: str) -> str:
        return _validated_text(value, name="model variant")

    @field_validator("parameter_identity", mode="before")
    @classmethod
    def validate_parameter_identity(cls, value: object) -> dict[str, object]:
        return _normalized_mapping(
            value,
            name="attempt parameter identity",
            require_nonempty=True,
        )

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if not self.gates:
            raise ValueError("every model attempt must record at least one gate")
        gate_ids = tuple(item.gate_id for item in self.gates)
        if len(set(gate_ids)) != len(gate_ids) or gate_ids != tuple(sorted(gate_ids)):
            raise ValueError("attempt gates must be sorted by unique gate_id")
        failed_gate = any(item.status == "failed" for item in self.gates)
        if self.status == "succeeded":
            if self.failure_reasons or failed_gate or not self.score_ids:
                raise ValueError("successful attempts need scores and no failures")
        elif self.status == "failed" and (
            not self.failure_reasons or not failed_gate or self.score_ids
        ):
            raise ValueError("failed attempts need reasons and at least one failed gate")
        elif self.status == "not_run" and (
            not self.failure_reasons
            or self.score_ids
            or failed_gate
            or not any(item.status == "not_applicable" for item in self.gates)
        ):
            raise ValueError("not-run attempts need reasons and an inapplicable upstream gate")
        return self


class G1Conclusion(_StrictRegistryModel):
    status: Literal["evaluated", "not_evaluable"] = "evaluated"
    passed: bool
    passing_models: tuple[Literal["spatial_poisson", "etas"], ...]
    evidence_ids: tuple[str, ...]

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validated_text(value, name="G1 evidence ID") for value in values)
        if (
            not normalized
            or len(set(normalized)) != len(normalized)
            or normalized != tuple(sorted(normalized))
        ):
            raise ValueError("G1 evidence IDs must be non-empty, sorted, and unique")
        return normalized

    @model_validator(mode="after")
    def validate_g1(self) -> Self:
        expected = tuple(
            model for model in ("spatial_poisson", "etas") if model in self.passing_models
        )
        if self.passing_models != expected or len(set(self.passing_models)) != len(
            self.passing_models
        ):
            raise ValueError("G1 passing models must follow frozen model order")
        if self.status == "not_evaluable" and (self.passed or self.passing_models):
            raise ValueError("unevaluable G1 cannot pass or name passing models")
        if self.status == "evaluated" and self.passed != bool(self.passing_models):
            raise ValueError("G1 passed must exactly reflect passing_models")
        return self


class SelectionConclusion(_StrictRegistryModel):
    status: Literal["evaluated", "not_evaluable"] = "evaluated"
    selected_model_id: ModelId | None
    validation_best_model_id: ModelId | None
    eligible_model_ids: tuple[ModelId, ...]
    evidence_id: str

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        return _validated_text(value, name="selection evidence ID")

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.status == "not_evaluable":
            if (
                self.selected_model_id is not None
                or self.validation_best_model_id is not None
                or self.eligible_model_ids
            ):
                raise ValueError("unevaluable selection must remain empty")
            return self
        expected = tuple(model for model in _MODEL_ORDER if model in self.eligible_model_ids)
        if self.eligible_model_ids != expected or len(set(self.eligible_model_ids)) != len(
            self.eligible_model_ids
        ):
            raise ValueError("eligible models must follow frozen simplicity order")
        if self.eligible_model_ids:
            if self.selected_model_id not in self.eligible_model_ids:
                raise ValueError("selected model must be eligible")
            if self.validation_best_model_id not in self.eligible_model_ids:
                raise ValueError("validation-best model must be eligible")
        elif self.selected_model_id is not None or self.validation_best_model_id is not None:
            raise ValueError("selection must be empty when no model is eligible")
        return self


class ModelSnapshotScientificSummary(_StrictRegistryModel):
    """Small auditable score projection for one formal model snapshot."""

    model_id: ModelId
    snapshot_id: SnapshotId
    status: AttemptStatus
    target_event_count: int | None = Field(ge=0)
    information_gain_nats_per_event: float | None
    score_id: str | None

    @model_validator(mode="after")
    def validate_score_projection(self) -> Self:
        information_gain = self.information_gain_nats_per_event
        if information_gain is not None and not math.isfinite(information_gain):
            raise ValueError("snapshot information gain must be finite when present")
        if self.score_id is not None:
            _validated_text(self.score_id, name="snapshot summary score ID")
        if self.status != "succeeded":
            if information_gain is not None or self.score_id is not None:
                raise ValueError("unscored snapshot summaries cannot expose a score")
            if self.status == "not_run" and self.target_event_count is not None:
                raise ValueError("not-run snapshot summaries cannot claim a target count")
        elif self.target_event_count is None:
            raise ValueError("successful snapshot summaries require a target count")
        elif self.score_id is None:
            raise ValueError("successful snapshot summaries require a score ID")
        elif self.target_event_count == 0 and information_gain is not None:
            raise ValueError("zero-target snapshots cannot define per-event information gain")
        elif self.target_event_count > 0 and information_gain is None:
            raise ValueError("successful nonempty snapshots require per-event information gain")
        return self


class ValidationBootstrapScientificSummary(_StrictRegistryModel):
    """Validation information-gain interval or one explicit skip."""

    model_id: Literal["spatial_poisson", "etas"]
    status: Literal["completed", "skipped"]
    point_estimate: float | None
    lower: float | None
    upper: float | None
    replications: Literal[2000] | None
    confidence_level: float | None
    not_run_reason: str | None

    @model_validator(mode="after")
    def validate_bootstrap(self) -> Self:
        numeric = (self.point_estimate, self.lower, self.upper)
        if self.status == "completed":
            if (
                any(value is None for value in numeric)
                or self.replications != 2000
                or self.confidence_level != 0.95
                or self.not_run_reason is not None
            ):
                raise ValueError("completed bootstrap summaries require the frozen interval")
            values = cast(tuple[float, float, float], numeric)
            if any(not math.isfinite(value) for value in values) or values[1] > values[2]:
                raise ValueError("bootstrap summary interval must be finite and ordered")
        elif (
            any(value is not None for value in numeric)
            or self.replications is not None
            or self.confidence_level is not None
            or self.not_run_reason is None
        ):
            raise ValueError("skipped bootstrap summaries require only a reason")
        else:
            _validated_text(self.not_run_reason, name="bootstrap skip reason")
        return self


class HorizonScientificSummary(_StrictRegistryModel):
    """Fixed 0/1/7-day by 7/30/90/180/365-day comparison coverage."""

    model_id: Literal["spatial_poisson", "etas"]
    status: Literal["completed", "skipped"]
    comparison_count: int = Field(ge=0)
    not_run_reason: str | None

    @model_validator(mode="after")
    def validate_horizons(self) -> Self:
        if self.status == "completed":
            if self.comparison_count != 15 or self.not_run_reason is not None:
                raise ValueError("completed horizon summaries require all 15 comparisons")
        elif self.comparison_count != 0 or self.not_run_reason is None:
            raise ValueError("skipped horizon summaries require zero comparisons and a reason")
        else:
            _validated_text(self.not_run_reason, name="horizon skip reason")
        return self


class FutureScientificSummary(_StrictRegistryModel):
    status: Literal["completed", "skipped"]
    issue_count: int = Field(ge=0)
    not_run_reason: str | None

    @model_validator(mode="after")
    def validate_future(self) -> Self:
        if self.status == "completed":
            if self.issue_count <= 0 or self.not_run_reason is not None:
                raise ValueError("completed future summaries require at least one issue")
        elif self.issue_count != 0 or self.not_run_reason is None:
            raise ValueError("skipped future summaries require zero issues and a reason")
        else:
            _validated_text(self.not_run_reason, name="future skip reason")
        return self


class RepresentativeScientificSummary(_StrictRegistryModel):
    issue_date_local: Literal["2025-06-26"]
    grid_cell_size_km: float
    selected_model_id: ModelId

    @model_validator(mode="after")
    def validate_representative(self) -> Self:
        if self.grid_cell_size_km != 25.0:
            raise ValueError("representative summary must use the frozen 25-km grid")
        return self


class ScientificFailureSummary(_StrictRegistryModel):
    """One deterministic expected scientific gate failure that stops stage 2."""

    failure_stage: Literal["completeness", "poisson_kde"]
    failure_reason_code: Literal[
        "all_bandwidths_failed_numerical_gate",
        "estimate_above_frozen_maximum",
        "no_eligible_spatial_stratum",
        "no_eligible_temporal_stratum",
        "selected_mc_has_no_events",
        "zero_target_snapshot",
        "zero_training_events",
    ]
    failure_reasons: tuple[str, ...]
    evidence_id: str

    @field_validator("failure_reasons")
    @classmethod
    def validate_failure_reasons(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _validated_text(value, name="scientific failure reason") for value in values
        )
        if (
            not normalized
            or len(set(normalized)) != len(normalized)
            or normalized != tuple(sorted(normalized))
        ):
            raise ValueError("scientific failure reasons must be non-empty, sorted, and unique")
        return normalized

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not _is_lower_hex(value, _SHA256_LENGTH):
            raise ValueError("scientific failure evidence ID must be SHA-256")
        return value

    @model_validator(mode="after")
    def validate_stage_reason_code(self) -> Self:
        completeness_codes = {
            "estimate_above_frozen_maximum",
            "no_eligible_spatial_stratum",
            "no_eligible_temporal_stratum",
            "selected_mc_has_no_events",
        }
        if (self.failure_reason_code in completeness_codes) != (
            self.failure_stage == "completeness"
        ):
            raise ValueError("scientific failure stage differs from its reason code")
        return self


class BackgroundScientificSummary(_StrictRegistryModel):
    """Small, tracked stage-2 results needed to audit the frozen G1 decision."""

    outcome_status: Literal["completed", "scientific_gate_failed"] = "completed"
    failure: ScientificFailureSummary | None = None
    final_selected_mc: float | None
    selected_kde_bandwidth_km: float | None
    snapshots: tuple[ModelSnapshotScientificSummary, ...]
    validation_bootstrap: tuple[ValidationBootstrapScientificSummary, ...]
    horizons: tuple[HorizonScientificSummary, ...]
    future: FutureScientificSummary
    representative: RepresentativeScientificSummary | None

    @model_validator(mode="after")
    def validate_complete_summary(self) -> Self:
        expected_snapshots = tuple(
            (model_id, snapshot_id) for model_id in _MODEL_ORDER for snapshot_id in _SNAPSHOT_ORDER
        )
        if tuple((item.model_id, item.snapshot_id) for item in self.snapshots) != (
            expected_snapshots
        ):
            raise ValueError("scientific summary must contain the fixed 3x5 snapshots")
        for snapshot_index, snapshot_id in enumerate(_SNAPSHOT_ORDER):
            target_counts = {
                self.snapshots[
                    model_index * len(_SNAPSHOT_ORDER) + snapshot_index
                ].target_event_count
                for model_index in range(len(_MODEL_ORDER))
            }
            if len(target_counts) != 1:
                raise ValueError(
                    f"scientific summary target counts differ across models for {snapshot_id}"
                )
        for uniform in self.snapshots[: len(_SNAPSHOT_ORDER)]:
            expected_uniform_gain = (
                0.0
                if uniform.target_event_count is not None and uniform.target_event_count > 0
                else None
            )
            if (
                uniform.status == "succeeded"
                and uniform.information_gain_nats_per_event != expected_uniform_gain
            ):
                raise ValueError("uniform baseline information gain must be zero or undefined")
        expected_candidates = ("spatial_poisson", "etas")
        if tuple(item.model_id for item in self.validation_bootstrap) != expected_candidates:
            raise ValueError("bootstrap summaries must follow frozen candidate order")
        if tuple(item.model_id for item in self.horizons) != expected_candidates:
            raise ValueError("horizon summaries must follow frozen candidate order")
        if self.outcome_status == "completed":
            if self.failure is not None:
                raise ValueError("completed scientific summaries cannot retain a failure")
            if (
                self.final_selected_mc is None
                or not math.isfinite(self.final_selected_mc)
                or self.final_selected_mc <= 0.0
            ):
                raise ValueError("final selected Mc must be finite and positive")
            if (
                self.selected_kde_bandwidth_km is None
                or not math.isfinite(self.selected_kde_bandwidth_km)
                or self.selected_kde_bandwidth_km <= 0.0
            ):
                raise ValueError("selected KDE bandwidth must be finite and positive")
            if self.representative is None:
                raise ValueError("completed scientific summaries require a representative map")
        else:
            if self.failure is None:
                raise ValueError("failed scientific summaries require failure evidence")
            if (
                self.final_selected_mc is not None
                or self.selected_kde_bandwidth_km is not None
                or self.representative is not None
            ):
                raise ValueError("failed scientific summaries cannot claim fitted outputs")
            if any(
                item.status != "not_run" or item.target_event_count is not None
                for item in self.snapshots
            ):
                raise ValueError("failed scientific summaries require 15 not-run attempts")
            if any(item.status != "skipped" for item in self.validation_bootstrap):
                raise ValueError("failed scientific summaries must skip bootstrap")
            if any(item.status != "skipped" for item in self.horizons):
                raise ValueError("failed scientific summaries must skip horizons")
            if self.future.status != "skipped":
                raise ValueError("failed scientific summaries must skip future ensembles")
        return self


class BackgroundRegistry(_StrictRegistryModel):
    schema_version: Literal[1] = 1
    protocol_version: Literal["0.2.0"]
    protocol_fingerprint_sha256: str = Field(min_length=64, max_length=64)
    freeze_tag: Literal["v0.2.0-background-protocol"]
    code_commit: str
    input_hashes: dict[str, str]
    bundles: tuple[BundleReference, ...]
    model_attempts: tuple[ModelAttemptRecord, ...]
    scientific_summary: BackgroundScientificSummary
    g1: G1Conclusion
    selection: SelectionConclusion
    stage3_allowed: bool

    @field_validator("protocol_fingerprint_sha256")
    @classmethod
    def validate_protocol_fingerprint(cls, value: str) -> str:
        if not _is_lower_hex(value, _SHA256_LENGTH):
            raise ValueError("protocol fingerprint must be lowercase SHA-256")
        return value

    @field_validator("code_commit")
    @classmethod
    def validate_code_commit(cls, value: str) -> str:
        if not _is_lower_hex(value, 40) and not _is_lower_hex(value, _SHA256_LENGTH):
            raise ValueError("code_commit must be a lowercase Git object ID")
        return value

    @field_validator("input_hashes")
    @classmethod
    def validate_input_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != _REQUIRED_INPUT_HASH_KEYS:
            raise ValueError("registry input hashes must contain the frozen seven keys")
        if any(not _is_lower_hex(digest, _SHA256_LENGTH) for digest in value.values()):
            raise ValueError("registry input hashes must be lowercase SHA-256")
        return {key: value[key] for key in sorted(value)}

    @model_validator(mode="after")
    def validate_complete_registry(self) -> Self:
        if tuple(item.bundle_kind for item in self.bundles) != _BUNDLE_ORDER:
            raise ValueError("registry must contain four bundles in frozen order")
        artifact_ids = tuple(item.artifact_id for item in self.bundles)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("registry bundle artifact IDs must be unique")

        expected_attempts = tuple(
            (model_id, snapshot_id) for model_id in _MODEL_ORDER for snapshot_id in _SNAPSHOT_ORDER
        )
        actual_attempts = tuple(
            (attempt.model_id, attempt.snapshot_id) for attempt in self.model_attempts
        )
        if actual_attempts != expected_attempts:
            raise ValueError("registry must contain every model and snapshot exactly once")

        summary_pairs = tuple(
            (item.model_id, item.snapshot_id) for item in self.scientific_summary.snapshots
        )
        if summary_pairs != expected_attempts:
            raise ValueError("registry scientific summary does not match the 3x5 attempt grid")
        for attempt, summary in zip(
            self.model_attempts,
            self.scientific_summary.snapshots,
            strict=True,
        ):
            if summary.status != attempt.status:
                raise ValueError("scientific summary status differs from its model attempt")
            if attempt.status == "succeeded":
                if attempt.score_ids != (summary.score_id,):
                    raise ValueError("scientific summary score ID differs from its model attempt")
            elif summary.score_id is not None:
                raise ValueError("failed model attempts cannot have a summary score ID")

        complete_success = {
            model_id: all(
                attempt.status == "succeeded"
                for attempt in self.model_attempts
                if attempt.model_id == model_id
            )
            for model_id in _MODEL_ORDER
        }
        if any(not complete_success[model] for model in self.g1.passing_models):
            raise ValueError("G1 passing models must have five successful snapshot attempts")
        if any(not complete_success[model] for model in self.selection.eligible_model_ids):
            raise ValueError("eligible models must have five successful snapshot attempts")
        summary_by_model = {
            model_id: tuple(
                item for item in self.scientific_summary.snapshots if item.model_id == model_id
            )
            for model_id in _MODEL_ORDER
        }
        expected_eligible = tuple(
            model_id
            for model_id in _MODEL_ORDER
            if all(item.status == "succeeded" for item in summary_by_model[model_id])
            and all(
                item.information_gain_nats_per_event is not None
                for item in summary_by_model[model_id]
            )
        )
        if self.selection.eligible_model_ids != expected_eligible:
            raise ValueError("selection eligibility differs from the tracked scientific summary")
        expected_g1: list[Literal["spatial_poisson", "etas"]] = []
        for model_id in ("spatial_poisson", "etas"):
            summaries = summary_by_model[model_id]
            validation_gain = summaries[-1].information_gain_nats_per_event
            positive_development_folds = sum(
                gain is not None and gain > 0.0
                for gain in (item.information_gain_nats_per_event for item in summaries[:-1])
            )
            if (
                all(item.status == "succeeded" for item in summaries)
                and validation_gain is not None
                and validation_gain > 0.0
                and positive_development_folds >= 3
            ):
                expected_g1.append(model_id)
        expected_g1_models = tuple(expected_g1)
        if self.g1.passing_models != expected_g1_models:
            raise ValueError("G1 conclusion differs from the tracked scientific summary")
        final_success = {
            model_id: summary_by_model[model_id][-1].status == "succeeded"
            for model_id in _MODEL_ORDER
        }
        for bootstrap in self.scientific_summary.validation_bootstrap:
            final_summary = summary_by_model[bootstrap.model_id][-1]
            expected_completed = (
                final_summary.status == "succeeded"
                and final_summary.target_event_count is not None
                and final_summary.target_event_count > 0
            )
            if (bootstrap.status == "completed") != expected_completed:
                raise ValueError("bootstrap status differs from the final snapshot attempt")
            if (
                bootstrap.status == "completed"
                and bootstrap.point_estimate != final_summary.information_gain_nats_per_event
            ):
                raise ValueError(
                    "bootstrap point estimate differs from final snapshot information gain"
                )
        for horizons in self.scientific_summary.horizons:
            if (horizons.status == "completed") != final_success[horizons.model_id]:
                raise ValueError("horizon status differs from the final snapshot attempt")
        if (self.scientific_summary.future.status == "completed") != final_success["etas"]:
            raise ValueError("future status differs from the final ETAS attempt")
        representative = self.scientific_summary.representative
        if representative is None:
            if self.selection.selected_model_id is not None:
                raise ValueError("missing representative summary requires empty selection")
        elif representative.selected_model_id != self.selection.selected_model_id:
            raise ValueError("representative summary differs from selected model")
        failure = self.scientific_summary.failure
        if failure is not None:
            if self.g1.status != "not_evaluable" or self.selection.status != "not_evaluable":
                raise ValueError("scientific failure requires unevaluable G1 and selection")
            if self.g1.evidence_ids != (failure.evidence_id,):
                raise ValueError("failed G1 evidence differs from scientific failure")
            if self.selection.evidence_id != failure.evidence_id:
                raise ValueError("unevaluable selection differs from scientific failure")
            if any(
                attempt.failure_reasons != failure.failure_reasons
                for attempt in self.model_attempts
            ):
                raise ValueError("failed model attempts differ from scientific failure")
        elif self.g1.status != "evaluated" or self.selection.status != "evaluated":
            raise ValueError("completed science requires evaluated G1 and selection")
        if self.stage3_allowed != self.g1.passed:
            raise ValueError("stage3_allowed must exactly equal the G1 conclusion")
        return self


def _bundle_reference(publication: BundlePublication) -> BundleReference:
    return BundleReference(
        bundle_kind=publication.bundle_kind,
        artifact_id=publication.artifact.artifact_id,
        manifest_sha256=publication.artifact.manifest_sha256,
    )


def build_background_registry(
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    bundles: Sequence[BundlePublication],
    model_attempts: Sequence[ModelAttemptRecord],
    *,
    scientific_summary: BackgroundScientificSummary,
    g1: G1Conclusion,
    selection: SelectionConclusion,
    stage3_allowed: bool,
    uv_lock_sha256: str,
) -> BackgroundRegistry:
    """Build the complete registry from trusted bundle publications."""

    require_background_scoring_authorized(background)
    materialized_bundles = tuple(bundles)
    if tuple(item.bundle_kind for item in materialized_bundles) != _BUNDLE_ORDER:
        raise ValueError("bundle publications must contain four kinds in frozen order")
    if any(item.code_commit != identity.code_commit for item in materialized_bundles):
        raise ValueError("bundle code commits differ from the registry identity")
    if any(item.freeze_tag != identity.freeze_tag for item in materialized_bundles):
        raise ValueError("bundle freeze tags differ from the registry identity")

    address_inputs = build_address_inputs(
        background,
        identity,
        {"artifact_role": "background_registry", "schema_version": 1},
        uv_lock_sha256=uv_lock_sha256,
    )
    protocol = cast(Mapping[str, object], address_inputs["protocol"])
    input_hashes = cast(dict[str, str], address_inputs["input_hashes"])
    protocol_fingerprint = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    return BackgroundRegistry(
        protocol_version=background.protocol_version,
        protocol_fingerprint_sha256=protocol_fingerprint,
        freeze_tag=background.freeze_tag,
        code_commit=identity.code_commit,
        input_hashes=input_hashes,
        bundles=tuple(_bundle_reference(item) for item in materialized_bundles),
        model_attempts=tuple(model_attempts),
        scientific_summary=scientific_summary,
        g1=g1,
        selection=selection,
        stage3_allowed=stage3_allowed,
    )


def registry_payload_bytes(registry: BackgroundRegistry) -> bytes:
    """Serialize one validated registry deterministically, without volatile fields."""

    if not isinstance(registry, BackgroundRegistry):
        raise TypeError("registry must be BackgroundRegistry")
    validated = BackgroundRegistry.model_validate(registry.model_dump(mode="python"))
    document = validated.model_dump(mode="json")
    _normalize_json(document, "registry")
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_registry_payload(payload: bytes) -> BackgroundRegistry:
    """Parse and fully validate canonical registry bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("registry payload must be bytes")
    try:
        registry = BackgroundRegistry.model_validate_json(payload)
    except (ValidationError, ValueError) as exc:
        raise RegistryValidationError("invalid background registry payload") from exc
    if registry_payload_bytes(registry) != payload:
        raise RegistryValidationError("registry payload is not in deterministic canonical form")
    return registry


def render_report_from_registry_payload(payload: bytes) -> bytes:
    """Render the human report from validated registry bytes and no other input."""

    registry = validate_registry_payload(payload)
    summary = registry.scientific_summary

    def number(value: float | None) -> str:
        return "未定义" if value is None else format(value, ".12g")

    def reason(value: str | None) -> str:
        return "—" if value is None else value.replace("|", "\\|")

    def count(value: int | None) -> str:
        return "—" if value is None else str(value)

    representative = summary.representative
    representative_text = (
        "未生成 (科学门控失败)"
        if representative is None
        else (
            f"{representative.issue_date_local}, "
            f"{number(representative.grid_cell_size_km)} km, "
            f"{representative.selected_model_id}"
        )
    )
    failure_lines: list[str] = []
    if summary.failure is not None:
        failure_lines = [
            f"- 失败阶段: `{summary.failure.failure_stage}`",
            f"- 失败代码: `{summary.failure.failure_reason_code}`",
            f"- 失败原因: {'; '.join(summary.failure.failure_reasons)}",
            f"- 失败证据: `{summary.failure.evidence_id}`",
        ]

    lines = [
        "# SeismoFlux 阶段2背景基线报告",
        "",
        (
            "本报告由已验证的不可变注册表确定性渲染。"
            "所有数值术语仅表示条件强度, 相对强度和信息增益, 不作绝对风险解释。"
        ),
        "",
        "## 科学身份",
        "",
        f"- 协议版本: `{registry.protocol_version}`",
        f"- 协议指纹: `{registry.protocol_fingerprint_sha256}`",
        f"- 代码提交: `{registry.code_commit}`",
        "",
        "## 冻结科学摘要",
        "",
        (
            "- 科学结果状态: "
            + ("完成" if summary.outcome_status == "completed" else "科学门控失败")
        ),
        *failure_lines,
        f"- 最终选择的 Mc: {number(summary.final_selected_mc)}",
        f"- 空间 KDE 带宽: {number(summary.selected_kde_bandwidth_km)} km",
        f"- 代表日条件强度: {representative_text}",
        "",
        "### 固定快照评分",
        "",
        "| 模型 | 快照 | 状态 | 目标事件数 | 信息增益 (nats/event) | Score ID |",
        "|---|---|---|---:|---:|---|",
    ]
    snapshot_status_labels = {
        "succeeded": "完成",
        "failed": "失败",
        "not_run": "未运行",
    }
    for snapshot_summary in summary.snapshots:
        status = snapshot_status_labels[snapshot_summary.status]
        score_id = (
            f"`{snapshot_summary.score_id}`" if snapshot_summary.score_id is not None else "—"
        )
        lines.append(
            f"| {snapshot_summary.model_id} | {snapshot_summary.snapshot_id} | {status} | "
            f"{count(snapshot_summary.target_event_count)} | "
            f"{number(snapshot_summary.information_gain_nats_per_event)} | {score_id} |"
        )
    lines.extend(
        [
            "",
            "### 验证段 bootstrap",
            "",
            "| 模型 | 状态 | 点估计 | 下限 | 上限 | 重采样次数 | 置信水平 | 跳过原因 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for bootstrap_summary in summary.validation_bootstrap:
        status = "完成" if bootstrap_summary.status == "completed" else "跳过"
        replications = (
            str(bootstrap_summary.replications)
            if bootstrap_summary.replications is not None
            else "—"
        )
        confidence = number(bootstrap_summary.confidence_level)
        lines.append(
            f"| {bootstrap_summary.model_id} | {status} | "
            f"{number(bootstrap_summary.point_estimate)} | "
            f"{number(bootstrap_summary.lower)} | {number(bootstrap_summary.upper)} | "
            f"{replications} | {confidence} | {reason(bootstrap_summary.not_run_reason)} |"
        )
    lines.extend(
        [
            "",
            "### 延迟与时窗覆盖",
            "",
            "| 模型 | 状态 | 比较数 | 跳过原因 |",
            "|---|---|---:|---|",
        ]
    )
    for horizon_summary in summary.horizons:
        status = "完成" if horizon_summary.status == "completed" else "跳过"
        lines.append(
            f"| {horizon_summary.model_id} | {status} | {horizon_summary.comparison_count} | "
            f"{reason(horizon_summary.not_run_reason)} |"
        )
    future_status = "完成" if summary.future.status == "completed" else "跳过"
    lines.extend(
        [
            "",
            "### 未来集合覆盖",
            "",
            f"- 状态: {future_status}",
            f"- 起报日数量: {summary.future.issue_count}",
            f"- 跳过原因: {reason(summary.future.not_run_reason)}",
            "",
            "## 不可变产物",
            "",
            "| 类型 | Artifact ID | Manifest SHA-256 |",
            "|---|---|---|",
        ]
    )
    lines.extend(
        f"| {item.bundle_kind} | `{item.artifact_id}` | `{item.manifest_sha256}` |"
        for item in registry.bundles
    )
    lines.extend(
        [
            "",
            "## 正式模型尝试",
            "",
            "| 模型 | 快照 | 状态 | 失败原因数 | 门控数 | Score ID 数 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for attempt in registry.model_attempts:
        status = {
            "failed": "失败",
            "not_run": "未运行",
            "succeeded": "成功",
        }[attempt.status]
        lines.append(
            f"| {attempt.model_id} | {attempt.snapshot_id} | {status} | "
            f"{len(attempt.failure_reasons)} | {len(attempt.gates)} | {len(attempt.score_ids)} |"
        )
    passing_models = ", ".join(registry.g1.passing_models) or "无"
    selected = registry.selection.selected_model_id or "无"
    validation_best = registry.selection.validation_best_model_id or "无"
    g1_status = (
        "未评估 (上游科学门控失败)"
        if registry.g1.status == "not_evaluable"
        else ("通过" if registry.g1.passed else "未通过")
    )
    selection_status = (
        "未评估 (上游科学门控失败)" if registry.selection.status == "not_evaluable" else "已评估"
    )
    lines.extend(
        [
            "",
            "## G1 与选择结论",
            "",
            f"- G1: {g1_status}",
            f"- G1 通过模型: {passing_models}",
            f"- 模型选择状态: {selection_status}",
            f"- 验证段最佳模型: {validation_best}",
            f"- 最终选择模型: {selected}",
            f"- 阶段3: {'允许进入' if registry.stage3_allowed else '停止'}",
            "",
        ]
    )
    report = "\n".join(lines).encode("utf-8")
    lowered = report.decode("utf-8").casefold()
    if any(term in lowered for term in _FORBIDDEN_REPORT_TERMS):
        raise RegistryValidationError("report contains forbidden absolute-risk terminology")
    return report


@dataclass(frozen=True, slots=True)
class FixedFilePublication:
    path: Path
    sha256: str
    created: bool


def _verify_fixed_file(path: Path, payload: bytes) -> None:
    if _is_reparse_point(path) or not path.is_file():
        raise FixedFileConflictError("fixed publication destination is not a plain file")
    if path.read_bytes() != payload:
        raise FixedFileConflictError("fixed publication destination has different bytes")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_fixed_project_file(
    project_root: Path,
    project_relative_path: str,
    payload: bytes,
) -> FixedFilePublication:
    """Atomically create a fixed project file or verify identical existing bytes."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("fixed publication payload must be non-empty bytes")
    root, destination = _validated_project_destination(project_root, project_relative_path)
    _ensure_plain_directory(root, destination.parent)
    if _lexists(destination):
        _verify_fixed_file(destination, payload)
        return FixedFilePublication(
            path=destination,
            sha256=hashlib.sha256(payload).hexdigest(),
            created=False,
        )

    temporary_path: Path | None = None
    created = False
    try:
        with tempfile.NamedTemporaryFile(
            "xb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination)
            created = True
        except FileExistsError:
            _verify_fixed_file(destination, payload)
        except OSError as exc:
            if _lexists(destination):
                _verify_fixed_file(destination, payload)
            else:
                raise FixedFilePublicationError(
                    "unable to atomically create fixed publication file"
                ) from exc
        if created:
            _fsync_directory(destination.parent)
            _verify_fixed_file(destination, payload)
        return FixedFilePublication(
            path=destination,
            sha256=hashlib.sha256(payload).hexdigest(),
            created=created,
        )
    except Exception:
        if created and _lexists(destination):
            try:
                destination.unlink()
                _fsync_directory(destination.parent)
            except OSError as rollback_error:
                raise FixedFilePublicationError(
                    "unable to roll back incomplete fixed publication"
                ) from rollback_error
        raise
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class RegistryReportPublication:
    registry: FixedFilePublication
    report: FixedFilePublication


def publish_registry_and_report(
    project_root: Path,
    background: BackgroundConfig,
    registry: BackgroundRegistry,
) -> RegistryReportPublication:
    """Publish fixed registry/report projections after deterministic in-memory rendering."""

    require_background_scoring_authorized(background)
    if registry.protocol_version != background.protocol_version:
        raise ValueError("registry protocol version differs from background configuration")
    expected_fingerprint = hashlib.sha256(
        canonical_json_bytes(background.model_dump(mode="python"))
    ).hexdigest()
    if registry.protocol_fingerprint_sha256 != expected_fingerprint:
        raise ValueError("registry protocol fingerprint differs from background configuration")
    registry_payload = registry_payload_bytes(registry)
    report_payload = render_report_from_registry_payload(registry_payload)
    registry_publication = publish_fixed_project_file(
        project_root,
        background.outputs.registry,
        registry_payload,
    )
    try:
        report_publication = publish_fixed_project_file(
            project_root,
            background.outputs.report,
            report_payload,
        )
    except Exception:
        if registry_publication.created and _lexists(registry_publication.path):
            try:
                registry_publication.path.unlink()
                _fsync_directory(registry_publication.path.parent)
            except OSError as rollback_error:
                raise FixedFilePublicationError(
                    "unable to roll back registry after report publication failure"
                ) from rollback_error
        raise
    return RegistryReportPublication(
        registry=registry_publication,
        report=report_publication,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_loaded_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FixedFilePublicationError("registry bundle manifest is not canonical JSON") from error


def _expected_bundle_directories(paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative_path in paths:
        parts = relative_path.split("/")
        directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    return directories


def _registry_scientific_summary_sha256(registry: BackgroundRegistry) -> str:
    return hashlib.sha256(
        canonical_json_bytes(registry.scientific_summary.model_dump(mode="python"))
    ).hexdigest()


def _plain_bundle_tree(directory: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            raise FixedFilePublicationError("registry bundle directory is unreadable") from error
        for entry in entries:
            child = Path(entry.path)
            relative_path = child.relative_to(directory).as_posix()
            try:
                reparse = _is_reparse_point(child)
            except OSError as error:
                raise FixedFilePublicationError("registry bundle entry is unreadable") from error
            if reparse:
                raise FixedFilePublicationError(
                    f"registry bundle contains a symlink or reparse point: {relative_path}"
                )
            if entry.is_file(follow_symlinks=False):
                files.add(relative_path)
            elif entry.is_dir(follow_symlinks=False):
                directories.add(relative_path)
                pending.append(child)
            else:
                raise FixedFilePublicationError(
                    f"registry bundle contains a non-file entry: {relative_path}"
                )
    return files, directories


def _validated_bundle_manifest(
    manifest_payload: bytes,
    *,
    reference: BundleReference,
    registry: BackgroundRegistry,
) -> tuple[dict[str, object], list[object]]:
    try:
        raw_manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixedFilePublicationError("registry bundle manifest is unreadable") from error
    if not isinstance(raw_manifest, dict):
        raise FixedFilePublicationError("registry bundle manifest root is invalid")
    manifest = cast(dict[str, object], raw_manifest)
    if set(manifest) != {
        "artifact_id",
        "canonicalization",
        "content_address",
        "files",
        "schema_version",
    }:
        raise FixedFilePublicationError("registry bundle manifest schema differs")
    if _canonical_loaded_json_bytes(manifest) != manifest_payload:
        raise FixedFilePublicationError("registry bundle manifest is not canonical")
    if manifest.get("schema_version") != 1:
        raise FixedFilePublicationError("registry bundle manifest schema version differs")
    if manifest.get("canonicalization") != CANONICAL_JSON_VERSION:
        raise FixedFilePublicationError("registry bundle canonicalization differs")
    if manifest.get("artifact_id") != reference.artifact_id:
        raise FixedFilePublicationError("registry bundle manifest identity differs")

    raw_address = manifest.get("content_address")
    if not isinstance(raw_address, dict) or set(raw_address) != {
        "algorithm",
        "first_hex_characters",
        "inputs",
        "sha256",
    }:
        raise FixedFilePublicationError("registry bundle content address is invalid")
    address = cast(dict[str, object], raw_address)
    if address.get("algorithm") != "sha256" or address.get("first_hex_characters") != 16:
        raise FixedFilePublicationError("registry bundle content-address algorithm differs")
    inputs = address.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "code_commit",
        "input_hashes",
        "model_parameters",
        "protocol",
        "uv_lock_sha256",
    }:
        raise FixedFilePublicationError("registry bundle address inputs are invalid")
    address_sha256 = hashlib.sha256(_canonical_loaded_json_bytes(inputs)).hexdigest()
    if address.get("sha256") != address_sha256 or reference.artifact_id != address_sha256[:16]:
        raise FixedFilePublicationError("registry bundle content address differs")
    if inputs.get("code_commit") != registry.code_commit:
        raise FixedFilePublicationError("registry bundle commit differs")
    if inputs.get("input_hashes") != registry.input_hashes:
        raise FixedFilePublicationError("registry bundle input hashes differ")
    if inputs.get("uv_lock_sha256") != registry.input_hashes["environment_lock"]:
        raise FixedFilePublicationError("registry bundle environment lock differs")
    protocol_sha256 = hashlib.sha256(
        _canonical_loaded_json_bytes(inputs.get("protocol"))
    ).hexdigest()
    if protocol_sha256 != registry.protocol_fingerprint_sha256:
        raise FixedFilePublicationError("registry bundle protocol differs")
    model_parameters = inputs.get("model_parameters")
    if (
        not isinstance(model_parameters, dict)
        or model_parameters.get("artifact_role") != reference.bundle_kind
    ):
        raise FixedFilePublicationError("registry bundle role differs")
    if reference.bundle_kind == "backtest":
        scientific_parameters = model_parameters.get("scientific_parameters")
        if not isinstance(scientific_parameters, dict) or scientific_parameters.get(
            "scientific_summary_sha256"
        ) != _registry_scientific_summary_sha256(registry):
            raise FixedFilePublicationError(
                "registry scientific summary differs from the backtest address"
            )
    scientific_failure = registry.scientific_summary.failure
    if scientific_failure is not None and (
        model_parameters.get("scientific_parameters") is None
        or not isinstance(model_parameters["scientific_parameters"], dict)
    ):
        raise FixedFilePublicationError("scientific failure bundle parameters are missing")
    if scientific_failure is not None:
        failure_parameters = cast(dict[str, object], model_parameters["scientific_parameters"])
        expected_failure_parameters = {
            "failure_evidence_sha256": scientific_failure.evidence_id,
            "failure_reason_code": scientific_failure.failure_reason_code,
            "failure_stage": scientific_failure.failure_stage,
        }
        if any(
            failure_parameters.get(key) != value
            for key, value in expected_failure_parameters.items()
        ):
            raise FixedFilePublicationError(
                "registry scientific failure differs from a bundle address"
            )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise FixedFilePublicationError("registry bundle manifest has no payload files")
    return manifest, cast(list[object], raw_files)


def _verify_registry_bundle_directories(
    project_root: Path,
    background: BackgroundConfig,
    registry: BackgroundRegistry,
) -> None:
    roots = {
        "processed": background.outputs.processed_root,
        "model": background.outputs.model_root,
        "backtest": background.outputs.backtest_root,
        "experiment": background.outputs.experiment_root,
    }
    for reference in registry.bundles:
        try:
            _, bundle_root = _validated_project_destination(
                project_root,
                roots[reference.bundle_kind],
            )
        except (OSError, ValueError) as error:
            raise FixedFilePublicationError("registry bundle root is unsafe") from error
        directory = bundle_root / reference.artifact_id
        if not _lexists(directory) or _is_reparse_point(directory) or not directory.is_dir():
            raise FixedFilePublicationError("registry bundle directory is missing or unsafe")
        manifest_path = directory / "manifest.json"
        if (
            not _lexists(manifest_path)
            or _is_reparse_point(manifest_path)
            or not manifest_path.is_file()
        ):
            raise FixedFilePublicationError("registry bundle manifest is missing or unsafe")
        manifest_payload = manifest_path.read_bytes()
        if hashlib.sha256(manifest_payload).hexdigest() != reference.manifest_sha256:
            raise FixedFilePublicationError("registry bundle manifest hash differs before publish")
        _, raw_files = _validated_bundle_manifest(
            manifest_payload,
            reference=reference,
            registry=registry,
        )
        expected_paths = {"manifest.json"}
        seen_casefold_paths = {"manifest.json"}
        validated_entries: list[tuple[ProjectRelativePath, int, str]] = []
        for raw_entry in raw_files:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {
                "byte_count",
                "media_type",
                "relative_path",
                "sha256",
            }:
                raise FixedFilePublicationError("registry bundle file entry is invalid")
            raw_relative = raw_entry.get("relative_path")
            if not isinstance(raw_relative, str):
                raise FixedFilePublicationError("registry bundle file path is invalid")
            try:
                relative = ProjectRelativePath(raw_relative)
            except (TypeError, ValueError) as error:
                raise FixedFilePublicationError("registry bundle file path is invalid") from error
            if relative.value != raw_relative or relative.value.casefold() in seen_casefold_paths:
                raise FixedFilePublicationError(
                    "registry bundle file paths are not unique canonical paths"
                )
            seen_casefold_paths.add(relative.value.casefold())
            byte_count = raw_entry.get("byte_count")
            payload_sha256 = raw_entry.get("sha256")
            if (
                type(byte_count) is not int
                or byte_count < 0
                or not isinstance(payload_sha256, str)
                or not _is_lower_hex(payload_sha256, _SHA256_LENGTH)
            ):
                raise FixedFilePublicationError("registry bundle payload metadata is invalid")
            media_type = raw_entry.get("media_type")
            if (
                not isinstance(media_type, str)
                or not media_type
                or media_type != media_type.strip()
            ):
                raise FixedFilePublicationError("registry bundle payload media type is invalid")
            expected_paths.add(relative.value)
            validated_entries.append((relative, byte_count, payload_sha256))
        actual_paths, actual_directories = _plain_bundle_tree(directory)
        if actual_paths != expected_paths:
            raise FixedFilePublicationError("registry bundle file set differs before publish")
        if actual_directories != _expected_bundle_directories(expected_paths):
            raise FixedFilePublicationError("registry bundle directory set differs before publish")
        for relative, byte_count, payload_sha256 in validated_entries:
            payload_path = directory.joinpath(*relative.value.split("/"))
            if not payload_path.is_file() or _is_reparse_point(payload_path):
                raise FixedFilePublicationError("registry bundle payload is missing or unsafe")
            if byte_count != payload_path.stat().st_size:
                raise FixedFilePublicationError("registry bundle payload byte count differs")
            if payload_sha256 != _sha256_path(payload_path):
                raise FixedFilePublicationError("registry bundle payload hash differs")
        if reference.bundle_kind == "backtest":
            summary_path = directory / "scientific_summary.json"
            expected_summary_payload = canonical_json_bytes(
                {
                    "bundle_kind": "backtest",
                    "content": registry.scientific_summary.model_dump(mode="python"),
                    "schema_version": 1,
                }
            )
            if (
                "scientific_summary.json" not in expected_paths
                or summary_path.read_bytes() != expected_summary_payload
            ):
                raise FixedFilePublicationError(
                    "registry scientific summary differs from the backtest payload"
                )
        scientific_failure = registry.scientific_summary.failure
        if scientific_failure is not None:
            failure_path = directory / "failure.json"
            if "failure.json" not in expected_paths:
                raise FixedFilePublicationError("scientific failure bundle payload is missing")
            try:
                failure_payload = json.loads(failure_path.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FixedFilePublicationError(
                    "scientific failure bundle payload is unreadable"
                ) from error
            if (
                not isinstance(failure_payload, dict)
                or set(failure_payload) != {"bundle_kind", "content", "schema_version"}
                or failure_payload.get("bundle_kind") != reference.bundle_kind
                or failure_payload.get("schema_version") != 1
                or not isinstance(failure_payload.get("content"), dict)
            ):
                raise FixedFilePublicationError("scientific failure bundle payload schema differs")
            content_sha256 = hashlib.sha256(
                _canonical_loaded_json_bytes(failure_payload["content"])
            ).hexdigest()
            if content_sha256 != scientific_failure.evidence_id:
                raise FixedFilePublicationError(
                    "registry scientific failure differs from a bundle payload"
                )
            failure_content = cast(dict[str, object], failure_payload["content"])
            raw_reasons = failure_content.get("failure_reasons")
            if (
                failure_content.get("failure_stage") != scientific_failure.failure_stage
                or failure_content.get("failure_reason_code")
                != scientific_failure.failure_reason_code
                or not isinstance(raw_reasons, list)
                or tuple(raw_reasons) != scientific_failure.failure_reasons
            ):
                raise FixedFilePublicationError(
                    "registry scientific failure semantics differ from a bundle payload"
                )


def publish_registry_and_report_sealed(
    project_root: Path,
    background: BackgroundConfig,
    registry: BackgroundRegistry,
    execution_seal: ExecutionSeal,
    *,
    runner: GitCommandRunner = subprocess_git_runner,
) -> RegistryReportPublication:
    """Re-seal all inputs, verify four bundles, then publish the fixed projections."""

    require_background_scoring_authorized(background)
    current = require_execution_seal_unchanged(
        project_root,
        background,
        execution_seal,
        runner=runner,
    )
    if registry.code_commit != current.repository.code_commit:
        raise FixedFilePublicationError("registry commit differs from the execution seal")
    if registry.input_hashes != current.input_hash_mapping():
        raise FixedFilePublicationError("registry input hashes differ from the execution seal")
    _verify_registry_bundle_directories(project_root, background, registry)
    final = require_execution_seal_unchanged(
        project_root,
        background,
        execution_seal,
        runner=runner,
    )
    if (
        final.repository.code_commit != registry.code_commit
        or final.input_hash_mapping() != registry.input_hashes
    ):
        raise FixedFilePublicationError("registry identity changed during bundle verification")
    return publish_registry_and_report(project_root, background, registry)


__all__ = [
    "BackgroundRegistry",
    "BackgroundScientificSummary",
    "BacktestBundle",
    "BundleBinary",
    "BundleDocument",
    "BundlePublication",
    "BundleReference",
    "ExperimentBundle",
    "FixedFileConflictError",
    "FixedFileError",
    "FixedFilePublication",
    "FixedFilePublicationError",
    "FutureScientificSummary",
    "G1Conclusion",
    "GateOutcome",
    "HorizonScientificSummary",
    "ModelAttemptRecord",
    "ModelBundle",
    "ModelSnapshotScientificSummary",
    "ProcessedBundle",
    "RegistryReportPublication",
    "RegistryValidationError",
    "RepresentativeScientificSummary",
    "ScientificFailureSummary",
    "SelectionConclusion",
    "ValidationBootstrapScientificSummary",
    "build_background_registry",
    "publish_backtest_bundle",
    "publish_experiment_bundle",
    "publish_fixed_project_file",
    "publish_model_bundle",
    "publish_processed_bundle",
    "publish_registry_and_report",
    "publish_registry_and_report_sealed",
    "registry_payload_bytes",
    "render_report_from_registry_payload",
    "validate_registry_payload",
]
