"""Immutable, content-addressed artifact publication for background models.

The address is derived only from explicitly supplied semantic inputs.  Output files are
bound to that address by a byte-level manifest, so reusing an address with different
bytes is always an error rather than an overwrite.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import TypeAlias, cast

CANONICAL_JSON_VERSION = "seismoflux_canonical_json_v1"
MANIFEST_FILENAME = "manifest.json"
_FLOAT_TYPE_KEY = "$seismoflux_type"
_HASH_CHUNK_BYTES = 1024 * 1024
_REQUIRED_ADDRESS_KEYS = {
    "protocol",
    "input_hashes",
    "model_parameters",
    "code_commit",
    "uv_lock_sha256",
}
_BASE_REQUIRED_INPUT_HASH_KEYS = frozenset(
    {
        "environment_lock",
        "data_catalog",
        "earthquake_dataset",
        "study_area",
        "issue_manifest",
        "production_fixture",
        "oracle_metadata",
    }
)
_SUPPORT_MANIFEST_INPUT_HASH_KEY = "support_manifest"
_SUPPORT_MANIFEST_PROTOCOL_FIELDS = (
    "support_manifest",
    "support_manifest_sha256",
)
_SUPPORT_MANIFEST_REQUIRED_INPUT_HASH_KEYS = _BASE_REQUIRED_INPUT_HASH_KEYS | {
    _SUPPORT_MANIFEST_INPUT_HASH_KEY
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")

CanonicalJson: TypeAlias = (
    None | bool | int | str | list["CanonicalJson"] | dict[str, "CanonicalJson"]
)


class ArtifactError(RuntimeError):
    """Base class for immutable artifact storage failures."""


class ArtifactConflictError(ArtifactError):
    """An artifact ID already exists but its exact bytes cannot be verified."""


class ArtifactPublicationError(ArtifactError):
    """An artifact could not be durably staged and atomically published."""


def _normalize_project_relative_path(value: str) -> str:
    normalized_text = unicodedata.normalize("NFC", value).replace("\\", "/")
    posix_path = PurePosixPath(normalized_text)
    windows_path = PureWindowsPath(normalized_text)
    if (
        not normalized_text
        or "\x00" in normalized_text
        or posix_path.is_absolute()
        or bool(windows_path.anchor)
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in posix_path.parts
    ):
        raise ValueError("project artifact paths must be POSIX-relative without traversal")
    normalized = posix_path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("project artifact paths must identify a relative file")
    return normalized


@dataclass(frozen=True, slots=True)
class ProjectRelativePath:
    """An explicitly declared and normalized POSIX project-relative path."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("ProjectRelativePath requires a string")
        object.__setattr__(self, "value", _normalize_project_relative_path(self.value))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """One immutable file to include in an artifact directory."""

    relative_path: ProjectRelativePath
    payload: bytes
    media_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, ProjectRelativePath):
            raise TypeError("artifact file paths must be explicitly wrapped as ProjectRelativePath")
        if not isinstance(self.payload, bytes):
            raise TypeError("artifact payloads must be immutable bytes")
        if not isinstance(self.media_type, str):
            raise TypeError("artifact media_type must be a string")
        normalized_media_type = unicodedata.normalize("NFC", self.media_type)
        if not normalized_media_type or normalized_media_type != normalized_media_type.strip():
            raise ValueError("artifact media_type must be a non-empty trimmed string")
        if any(ord(character) < 0x20 for character in normalized_media_type):
            raise ValueError("artifact media_type must not contain control characters")
        object.__setattr__(self, "media_type", normalized_media_type)


@dataclass(frozen=True, slots=True)
class ArtifactPublication:
    """Result of publishing or verifying a content-addressed directory."""

    artifact_id: str
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    created: bool


def _canonicalize(value: object, location: str = "$") -> CanonicalJson:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, ProjectRelativePath):
        return value.value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is forbidden at {location}")
        return {_FLOAT_TYPE_KEY: "float", "hex": value.hex()}
    if isinstance(value, Mapping):
        if _FLOAT_TYPE_KEY in value:
            raise ValueError(f"reserved canonical JSON key at {location}: {_FLOAT_TYPE_KEY}")
        normalized_items: dict[str, CanonicalJson] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON mapping keys must be strings at {location}")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized_items:
                raise ValueError(f"mapping keys collide after NFC normalization at {location}")
            normalized_items[normalized_key] = _canonicalize(item, f"{location}.{normalized_key}")
        return {key: normalized_items[key] for key in sorted(normalized_items)}
    if isinstance(value, list | tuple):
        return [_canonicalize(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, PurePath):
        raise TypeError(f"paths must be explicitly wrapped as ProjectRelativePath at {location}")
    raise TypeError(f"unsupported canonical JSON value at {location}: {type(value).__name__}")


def _encode_canonical(value: CanonicalJson) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    """Encode ``value`` using ``seismoflux_canonical_json_v1``.

    Strings and mapping keys use Unicode NFC. Mapping keys are sorted by Unicode
    codepoint. Finite floats become typed objects containing ``float.hex()`` output;
    non-finite floats are rejected. Paths require :class:`ProjectRelativePath` so a
    caller cannot accidentally fingerprint a platform-specific absolute path.
    """

    return _encode_canonical(_canonicalize(value))


def _required_input_hash_keys(protocol: Mapping[object, object]) -> frozenset[str]:
    """Select the frozen input-hash schema declared by the protocol payload.

    The original v0.2.0 payload keeps its exact seven-key schema and forbids support
    fields. Version 0.2.1 requires the project-relative support manifest and its
    expected digest, and therefore uses the exact eight-key schema.
    """

    protocol_version = protocol.get("protocol_version")
    protocol_inputs = protocol.get("inputs")
    if protocol_version == "0.2.0":
        if isinstance(protocol_inputs, Mapping) and any(
            field in protocol_inputs for field in _SUPPORT_MANIFEST_PROTOCOL_FIELDS
        ):
            raise ValueError("background protocol 0.2.0 forbids support_manifest fields")
        return _BASE_REQUIRED_INPUT_HASH_KEYS
    if protocol_version != "0.2.1":
        raise ValueError("unsupported background protocol_version for content addressing")
    if not isinstance(protocol_inputs, Mapping):
        raise ValueError("background protocol 0.2.1 must declare an inputs mapping")

    declared = tuple(field in protocol_inputs for field in _SUPPORT_MANIFEST_PROTOCOL_FIELDS)
    if declared != (True, True):
        raise ValueError(
            "background protocol 0.2.1 must declare support_manifest and "
            "support_manifest_sha256 together"
        )

    reference = protocol_inputs["support_manifest"]
    digest = protocol_inputs["support_manifest_sha256"]
    if not isinstance(reference, str) or not reference.strip() or reference != reference.strip():
        raise ValueError("background protocol support_manifest must be a non-empty path")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(
            "background protocol support_manifest_sha256 must be a lowercase SHA-256 string"
        )
    return _SUPPORT_MANIFEST_REQUIRED_INPUT_HASH_KEYS


def _validated_address_inputs(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _REQUIRED_ADDRESS_KEYS:
        raise ValueError("background address inputs must contain the complete frozen key set")
    protocol = value["protocol"]
    if not isinstance(protocol, Mapping):
        raise TypeError("background address protocol must be a canonicalizable mapping")
    if not isinstance(value["model_parameters"], Mapping):
        raise TypeError("background model_parameters must be a canonicalizable mapping")
    input_hashes = value["input_hashes"]
    required_input_hash_keys = _required_input_hash_keys(protocol)
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != required_input_hash_keys:
        raise ValueError("background input_hashes must contain the complete frozen key set")
    for name, digest in input_hashes.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not _SHA256_PATTERN.fullmatch(digest)
        ):
            raise ValueError("background input hashes must be lowercase SHA-256 strings")
    if _SUPPORT_MANIFEST_INPUT_HASH_KEY in required_input_hash_keys:
        protocol_inputs = cast(Mapping[str, object], protocol["inputs"])
        if (
            input_hashes[_SUPPORT_MANIFEST_INPUT_HASH_KEY]
            != protocol_inputs["support_manifest_sha256"]
        ):
            raise ValueError("input_hashes.support_manifest must equal the frozen protocol digest")
    lock_digest = value["uv_lock_sha256"]
    if not isinstance(lock_digest, str) or not _SHA256_PATTERN.fullmatch(lock_digest):
        raise ValueError("uv_lock_sha256 must be a lowercase SHA-256 string")
    if lock_digest != input_hashes["environment_lock"]:
        raise ValueError("uv_lock_sha256 must equal input_hashes.environment_lock")
    code_commit = value["code_commit"]
    if not isinstance(code_commit, str) or not _GIT_COMMIT_PATTERN.fullmatch(code_commit):
        raise ValueError("code_commit must be a lowercase Git object ID")
    return cast(Mapping[str, object], value)


def content_address_id(address_inputs: object) -> str:
    """Return the first 16 hexadecimal characters of the canonical SHA-256 digest."""

    validated = _validated_address_inputs(address_inputs)
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()[:16]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_files(files: Iterable[ArtifactFile]) -> tuple[ArtifactFile, ...]:
    materialized = tuple(files)
    if not materialized:
        raise ValueError("an artifact must contain at least one payload file")

    by_path: dict[str, ArtifactFile] = {}
    by_casefold: dict[str, str] = {}
    for artifact_file in materialized:
        if not isinstance(artifact_file, ArtifactFile):
            raise TypeError("files must contain only ArtifactFile instances")
        relative_path = artifact_file.relative_path.value
        if relative_path.casefold() == MANIFEST_FILENAME.casefold():
            raise ValueError(f"{MANIFEST_FILENAME} is reserved for the artifact manifest")
        if relative_path in by_path:
            raise ValueError(f"duplicate artifact file path: {relative_path}")
        folded_path = relative_path.casefold()
        if folded_path in by_casefold:
            raise ValueError(
                "artifact file paths must also be unique on case-insensitive filesystems: "
                f"{by_casefold[folded_path]} and {relative_path}"
            )
        by_path[relative_path] = artifact_file
        by_casefold[folded_path] = relative_path

    names = set(by_path)
    folded_names = set(by_casefold)
    for relative_path in names:
        parts = relative_path.split("/")
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index])
            folded_prefix = prefix.casefold()
            if folded_prefix in folded_names:
                existing = by_casefold[folded_prefix]
                raise ValueError(
                    "artifact path is both a file and a directory on a "
                    f"case-insensitive filesystem: {existing}"
                )
        if parts[0].casefold() == MANIFEST_FILENAME.casefold():
            raise ValueError(f"{MANIFEST_FILENAME} cannot be used as a payload directory")
    return tuple(by_path[key] for key in sorted(by_path))


def _file_manifest_entries(files: tuple[ArtifactFile, ...]) -> list[CanonicalJson]:
    return [
        {
            "byte_count": len(artifact_file.payload),
            "media_type": artifact_file.media_type,
            "relative_path": artifact_file.relative_path.value,
            "sha256": _sha256_bytes(artifact_file.payload),
        }
        for artifact_file in files
    ]


def _manifest_bytes(
    *,
    artifact_id: str,
    address_sha256: str,
    canonical_inputs: CanonicalJson,
    files: tuple[ArtifactFile, ...],
) -> bytes:
    document: CanonicalJson = {
        "artifact_id": artifact_id,
        "canonicalization": CANONICAL_JSON_VERSION,
        "content_address": {
            "algorithm": "sha256",
            "first_hex_characters": 16,
            "inputs": canonical_inputs,
            "sha256": address_sha256,
        },
        "files": _file_manifest_entries(files),
        "schema_version": 1,
    }
    return _encode_canonical(document)


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _expected_directories(expected_files: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative_path in expected_files:
        parts = relative_path.split("/")
        directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    return directories


def _verify_directory(
    directory: Path,
    *,
    expected_manifest: bytes,
    files: tuple[ArtifactFile, ...],
) -> None:
    if _is_reparse_point(directory) or not directory.is_dir():
        raise ArtifactConflictError(f"artifact destination is not a plain directory: {directory}")

    expected_file_paths = {MANIFEST_FILENAME}
    expected_file_paths.update(item.relative_path.value for item in files)
    expected_directory_paths = _expected_directories(expected_file_paths)
    actual_file_paths: set[str] = set()
    actual_directory_paths: set[str] = set()

    for child in directory.rglob("*"):
        relative_path = child.relative_to(directory).as_posix()
        if _is_reparse_point(child):
            raise ArtifactConflictError(
                f"artifact contains a symlink or reparse point: {relative_path}"
            )
        if child.is_file():
            actual_file_paths.add(relative_path)
        elif child.is_dir():
            actual_directory_paths.add(relative_path)
        else:
            raise ArtifactConflictError(f"artifact contains a non-file entry: {relative_path}")

    if actual_file_paths != expected_file_paths:
        raise ArtifactConflictError("artifact file set differs from the expected manifest")
    if actual_directory_paths != expected_directory_paths:
        raise ArtifactConflictError("artifact directory set differs from the expected manifest")

    manifest_path = directory / MANIFEST_FILENAME
    if manifest_path.read_bytes() != expected_manifest:
        raise ArtifactConflictError("artifact manifest bytes differ for an existing ID")

    for artifact_file in files:
        path = directory.joinpath(*artifact_file.relative_path.value.split("/"))
        metadata = path.stat()
        expected_size = len(artifact_file.payload)
        if metadata.st_size != expected_size or _sha256_file(path) != _sha256_bytes(
            artifact_file.payload
        ):
            raise ArtifactConflictError(
                "artifact payload bytes differ for an existing ID: "
                f"{artifact_file.relative_path.value}"
            )


def _write_durable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _publication_result(
    destination: Path, artifact_id: str, manifest_payload: bytes, *, created: bool
) -> ArtifactPublication:
    return ArtifactPublication(
        artifact_id=artifact_id,
        directory=destination,
        manifest_path=destination / MANIFEST_FILENAME,
        manifest_sha256=_sha256_bytes(manifest_payload),
        created=created,
    )


def publish_artifact(
    parent: Path,
    *,
    address_inputs: object,
    files: Iterable[ArtifactFile],
) -> ArtifactPublication:
    """Durably stage and atomically publish one immutable artifact directory.

    The destination is ``parent / content_address_id(address_inputs)``. If that ID
    already exists, every manifest and payload byte is verified and nothing inside the
    artifact is rewritten. Any difference raises :class:`ArtifactConflictError`.
    """

    canonical_inputs = _canonicalize(_validated_address_inputs(address_inputs))
    address_payload = _encode_canonical(canonical_inputs)
    address_sha256 = _sha256_bytes(address_payload)
    artifact_id = address_sha256[:16]
    validated_files = _validated_files(files)
    manifest_payload = _manifest_bytes(
        artifact_id=artifact_id,
        address_sha256=address_sha256,
        canonical_inputs=canonical_inputs,
        files=validated_files,
    )

    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / artifact_id
    if _destination_exists(destination):
        _verify_directory(destination, expected_manifest=manifest_payload, files=validated_files)
        return _publication_result(destination, artifact_id, manifest_payload, created=False)

    lock_path = parent / f".{artifact_id}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ArtifactPublicationError(
            f"artifact publication is already in progress: {artifact_id}"
        ) from exc

    staging: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "wb") as lock_handle:
            lock_handle.write((artifact_id + "\n").encode("ascii"))
            lock_handle.flush()
            os.fsync(lock_handle.fileno())

        if _destination_exists(destination):
            _verify_directory(
                destination, expected_manifest=manifest_payload, files=validated_files
            )
            return _publication_result(destination, artifact_id, manifest_payload, created=False)

        staging = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", suffix=".tmp", dir=parent))
        for artifact_file in validated_files:
            output_path = staging.joinpath(*artifact_file.relative_path.value.split("/"))
            _write_durable(output_path, artifact_file.payload)
        _write_durable(staging / MANIFEST_FILENAME, manifest_payload)
        _verify_directory(staging, expected_manifest=manifest_payload, files=validated_files)
        _fsync_directory(staging)

        try:
            os.rename(staging, destination)
        except OSError as exc:
            if _destination_exists(destination):
                _verify_directory(
                    destination, expected_manifest=manifest_payload, files=validated_files
                )
                return _publication_result(
                    destination, artifact_id, manifest_payload, created=False
                )
            raise ArtifactPublicationError(
                f"unable to atomically publish artifact: {artifact_id}"
            ) from exc
        staging = None
        _fsync_directory(parent)
        _verify_directory(destination, expected_manifest=manifest_payload, files=validated_files)
        return _publication_result(destination, artifact_id, manifest_payload, created=True)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
