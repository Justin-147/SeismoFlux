"""Target-blind, read-only access-control evidence for restricted stage-4 inputs.

The observer in this module never changes permissions.  It records native numeric
security identities and fails closed whenever the host cannot prove the frozen
owner-only contract.  In particular, it never shells out to ``icacls`` and never
parses localized account names.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    open_existing_immutable_file,
    require_existing_real_directory_tree,
)
from seismoflux.anomaly_increment.preregistration import (
    protocol_design_sha256,
    verify_content_sha256,
    with_content_sha256,
)
from seismoflux.data.common import canonical_json_bytes

RESTRICTED_ACCESS_SCHEMA_VERSION: Final[int] = 3
RESTRICTED_ARTIFACT_IDS: Final[tuple[str, ...]] = (
    "cell_mapping",
    "connectors",
    "entity_mapping",
    "zone_geometry",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SID_PATTERN = re.compile(r"S-\d+(?:-\d+)+")
_POSIX_PRINCIPAL_PATTERN = re.compile(r"uid:(0|[1-9]\d*)")
_WINDOWS_PRINCIPAL_PATTERN = re.compile(r"sid:(S-\d+(?:-\d+)+)")
_WINDOWS_SYSTEM_SID: Final[str] = "S-1-5-18"
_WINDOWS_ADMINISTRATORS_SID: Final[str] = "S-1-5-32-544"
_WINDOWS_FULL_CONTROL_MASK: Final[int] = 0x001F01FF
_FILE_ATTRIBUTE_REPARSE_POINT: Final[int] = 0x400
_INHERIT_ONLY_ACE: Final[int] = 0x08
_INHERITED_ACE: Final[int] = 0x10
_SE_DACL_PROTECTED: Final[int] = 0x1000
_WINDOWS_DESCRIPTOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ace_count",
        "aces_in_native_order",
        "dacl_defaulted",
        "dacl_present",
        "dacl_protected",
        "inherited_ace_count",
        "owner_sid",
        "queried_by_handle",
        "security_descriptor_control",
        "security_descriptor_revision",
    }
)
_WINDOWS_ACE_KEYS: Final[frozenset[str]] = frozenset(
    {"ace_flags", "ace_type", "access_mask", "principal_sid"}
)
_POSIX_DESCRIPTOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "acl_model",
        "acl_related_xattrs",
        "filesystem_locality",
        "filesystem_magic_hex",
        "filesystem_type",
        "mode_octal",
        "owner_uid",
        "platform_family",
        "queried_by_handle",
    }
)
_LINUX_LOCAL_CLASSIC_FILESYSTEMS: Final[dict[int, str]] = {
    0x0000EF53: "ext2_ext3_ext4",
    0x01021994: "tmpfs",
    0x58465342: "xfs",
    0x9123683E: "btrfs",
    0xF2F52010: "f2fs",
}


class RestrictedAccessError(RuntimeError):
    """Raised when restricted local access cannot be proven without mutation."""


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RestrictedAccessError(f"{label} must be a lowercase SHA-256 string")
    return value


def _relative_posix(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RestrictedAccessError(f"{label} must be a normalized project-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise RestrictedAccessError(f"{label} must be a normalized project-relative path")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RestrictedAccessError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def canonical_permission_descriptor(value: Mapping[str, object]) -> str:
    """Return the one canonical JSON representation accepted by observations."""

    return canonical_json_bytes(dict(value)).decode("utf-8")


def _descriptor_mapping(value: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise RestrictedAccessError("permission descriptor must be canonical JSON")
    try:
        parsed = _mapping(json.loads(value), label="permission descriptor")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RestrictedAccessError("permission descriptor must be canonical JSON") from exc
    if canonical_permission_descriptor(parsed) != value:
        raise RestrictedAccessError("permission descriptor is not canonical JSON")
    return parsed


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise RestrictedAccessError(f"{label} must be an integer >= {minimum}{upper}")
    return value


def _validate_posix_descriptor(
    observation: RestrictedPathAccessObservation,
    descriptor: Mapping[str, object],
    *,
    current_principal: str,
) -> None:
    if set(descriptor) != _POSIX_DESCRIPTOR_KEYS:
        raise RestrictedAccessError("restricted POSIX permission descriptor schema changed")
    match = _POSIX_PRINCIPAL_PATTERN.fullmatch(current_principal)
    if match is None:
        raise RestrictedAccessError("restricted POSIX current principal is malformed")
    uid = int(match.group(1))
    expected_mode = "0700" if observation.path_kind == "directory" else "0600"
    if descriptor.get("mode_octal") != expected_mode:
        raise RestrictedAccessError("restricted POSIX mode differs from the owner-only contract")
    owner_uid = _plain_int(
        descriptor.get("owner_uid"),
        label="restricted POSIX owner UID",
        maximum=0xFFFFFFFF,
    )
    if owner_uid != uid or observation.owner_principal != current_principal:
        raise RestrictedAccessError("restricted POSIX owner differs from the effective UID")
    if descriptor.get("platform_family") != "linux":
        raise RestrictedAccessError("restricted POSIX platform family is not provable")
    filesystem_type = descriptor.get("filesystem_type")
    filesystem_magic = descriptor.get("filesystem_magic_hex")
    if not isinstance(filesystem_type, str) or not isinstance(filesystem_magic, str):
        raise RestrictedAccessError("restricted POSIX filesystem identity is malformed")
    expected_filesystem_pairs = {
        (name, f"0x{magic:08x}") for magic, name in _LINUX_LOCAL_CLASSIC_FILESYSTEMS.items()
    }
    if (filesystem_type, filesystem_magic) not in expected_filesystem_pairs:
        raise RestrictedAccessError("restricted POSIX filesystem is not an allowed local model")
    if descriptor.get("filesystem_locality") != "local":
        raise RestrictedAccessError("restricted POSIX filesystem locality is not proven")
    if descriptor.get("acl_model") != "classic_mode_bits_without_acl_xattrs":
        raise RestrictedAccessError("restricted POSIX ACL model is not the classic mode model")
    if descriptor.get("queried_by_handle") is not True:
        raise RestrictedAccessError("restricted POSIX permissions were not queried by handle")
    acl_related_xattrs = descriptor.get("acl_related_xattrs")
    if not isinstance(acl_related_xattrs, list) or acl_related_xattrs:
        raise RestrictedAccessError("restricted POSIX path carries an ACL-related xattr")


def _validate_windows_descriptor(
    observation: RestrictedPathAccessObservation,
    descriptor: Mapping[str, object],
    *,
    current_principal: str,
) -> None:
    if set(descriptor) != _WINDOWS_DESCRIPTOR_KEYS:
        raise RestrictedAccessError("restricted Windows permission descriptor schema changed")
    match = _WINDOWS_PRINCIPAL_PATTERN.fullmatch(current_principal)
    if match is None:
        raise RestrictedAccessError("restricted Windows current principal is malformed")
    current_sid = match.group(1)
    allowed = {current_sid, _WINDOWS_SYSTEM_SID, _WINDOWS_ADMINISTRATORS_SID}
    owner_sid = descriptor.get("owner_sid")
    if not isinstance(owner_sid, str) or owner_sid not in allowed:
        raise RestrictedAccessError("restricted Windows owner is outside the allowed SID set")
    if observation.owner_principal != f"sid:{owner_sid}":
        raise RestrictedAccessError("restricted Windows owner observation is inconsistent")
    if (
        descriptor.get("queried_by_handle") is not True
        or descriptor.get("dacl_present") is not True
        or descriptor.get("dacl_protected") is not True
        or descriptor.get("dacl_defaulted") is not False
        or descriptor.get("inherited_ace_count") != 0
    ):
        raise RestrictedAccessError("restricted Windows DACL is not explicit and protected")
    control = _plain_int(
        descriptor.get("security_descriptor_control"),
        label="Windows security descriptor control",
        maximum=0xFFFF,
    )
    if not control & _SE_DACL_PROTECTED:
        raise RestrictedAccessError("restricted Windows DACL protection bit is absent")
    revision = _plain_int(
        descriptor.get("security_descriptor_revision"),
        label="Windows security descriptor revision",
        maximum=0xFF,
    )
    if revision != 1:
        raise RestrictedAccessError("restricted Windows security descriptor revision changed")
    entries = descriptor.get("aces_in_native_order")
    if not isinstance(entries, list):
        raise RestrictedAccessError("restricted Windows ACE sequence is not a list")
    ace_count = _plain_int(
        descriptor.get("ace_count"),
        label="restricted Windows ACE count",
        maximum=0xFFFF,
    )
    if ace_count != len(entries):
        raise RestrictedAccessError("restricted Windows ACE count is inconsistent")
    declared_inherited_count = _plain_int(
        descriptor.get("inherited_ace_count"),
        label="restricted Windows inherited ACE count",
        maximum=0xFFFF,
    )
    if declared_inherited_count != 0:
        raise RestrictedAccessError("restricted Windows DACL contains inherited ACEs")
    inherited_count = 0
    current_user_has_full_control = False
    for raw_entry in entries:
        entry = _mapping(raw_entry, label="restricted Windows ACE")
        if set(entry) != _WINDOWS_ACE_KEYS:
            raise RestrictedAccessError("restricted Windows ACE schema changed")
        flags = _plain_int(
            entry.get("ace_flags"),
            label="Windows ACE flags",
            maximum=0xFF,
        )
        access_mask = _plain_int(
            entry.get("access_mask"),
            label="Windows ACE access mask",
            maximum=0xFFFFFFFF,
        )
        if entry.get("ace_type") != "allow":
            raise RestrictedAccessError("restricted Windows deny or unknown ACE is unsupported")
        sid = entry.get("principal_sid")
        if not isinstance(sid, str) or sid not in allowed:
            raise RestrictedAccessError("restricted Windows ACE uses an unauthorized SID")
        if flags & _INHERITED_ACE:
            inherited_count += 1
        if (
            sid == current_sid
            and not flags & _INHERIT_ONLY_ACE
            and access_mask & _WINDOWS_FULL_CONTROL_MASK == _WINDOWS_FULL_CONTROL_MASK
        ):
            current_user_has_full_control = True
    if inherited_count != 0:
        raise RestrictedAccessError("restricted Windows DACL contains inherited ACEs")
    if not current_user_has_full_control:
        raise RestrictedAccessError("restricted Windows DACL lacks current-user full control")


def _validate_observation_permissions(
    observation: RestrictedPathAccessObservation,
    *,
    platform: Literal["windows", "posix"],
    current_principal: str,
) -> None:
    descriptor = _descriptor_mapping(observation.permission_descriptor_json)
    if platform == "windows":
        _validate_windows_descriptor(
            observation,
            descriptor,
            current_principal=current_principal,
        )
    else:
        _validate_posix_descriptor(
            observation,
            descriptor,
            current_principal=current_principal,
        )


@dataclass(frozen=True, slots=True)
class RestrictedPathAccessObservation:
    """Native, non-mutating type and permission observation for one frozen path."""

    relative_path: str
    path_kind: Literal["directory", "regular_file"]
    verified_sha256: str | None
    owner_principal: str
    link_count: int
    reparse_point: bool
    permission_descriptor_json: str

    def __post_init__(self) -> None:
        _relative_posix(self.relative_path, label="restricted path")
        if self.path_kind not in {"directory", "regular_file"}:
            raise ValueError("restricted path type is unsupported")
        if not isinstance(self.link_count, int) or isinstance(self.link_count, bool):
            raise ValueError("restricted path link count is invalid")
        if self.link_count < 1 or self.link_count > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("restricted path link count is invalid")
        if self.reparse_point is not False:
            raise ValueError("restricted path must not be a reparse point")
        if self.path_kind == "regular_file":
            _sha256(self.verified_sha256, label="restricted file verified SHA-256")
            if self.link_count != 1:
                raise ValueError("restricted file must be single-link")
        elif self.verified_sha256 is not None:
            raise ValueError("restricted directory must not carry a file hash")
        _descriptor_mapping(self.permission_descriptor_json)
        if not isinstance(self.owner_principal, str) or not self.owner_principal:
            raise ValueError("restricted path owner principal is missing")

    def as_mapping(self) -> dict[str, object]:
        return {
            "link_count": self.link_count,
            "owner_principal": self.owner_principal,
            "path_kind": self.path_kind,
            "permission_descriptor": dict(_descriptor_mapping(self.permission_descriptor_json)),
            "relative_path": self.relative_path,
            "reparse_point": False,
            "single_link_verified": self.link_count == 1
            if self.path_kind == "regular_file"
            else None,
            "verified_sha256": self.verified_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RestrictedPathAccessObservation:
        expected = {
            "link_count",
            "owner_principal",
            "path_kind",
            "permission_descriptor",
            "relative_path",
            "reparse_point",
            "single_link_verified",
            "verified_sha256",
        }
        if set(value) != expected:
            raise RestrictedAccessError("restricted path observation schema changed")
        kind = value.get("path_kind")
        expected_single = True if kind == "regular_file" else None
        if value.get("single_link_verified") is not expected_single:
            raise RestrictedAccessError("restricted path single-link observation changed")
        try:
            result = cls(
                relative_path=cast(str, value["relative_path"]),
                path_kind=cast(Any, kind),
                verified_sha256=cast(str | None, value["verified_sha256"]),
                owner_principal=cast(str, value["owner_principal"]),
                link_count=cast(int, value["link_count"]),
                reparse_point=cast(bool, value["reparse_point"]),
                permission_descriptor_json=canonical_permission_descriptor(
                    _mapping(value.get("permission_descriptor"), label="permission descriptor")
                ),
            )
        except (TypeError, ValueError, RestrictedAccessError) as exc:
            raise RestrictedAccessError("restricted path observation is invalid") from exc
        if result.as_mapping() != dict(value):
            raise RestrictedAccessError("restricted path observation did not canonicalize")
        return result


@dataclass(frozen=True, slots=True)
class RestrictedLocalArtifactAccessControlEvidence:
    """Typed evidence that all five target-blind local paths are owner restricted."""

    protocol_design_sha256: str
    access_contract_sha256: str
    platform: Literal["windows", "posix"]
    current_principal: str
    directory_relative_path: str
    file_artifacts: tuple[tuple[str, str, str], ...]
    observations: tuple[RestrictedPathAccessObservation, ...]

    def __post_init__(self) -> None:
        _sha256(self.protocol_design_sha256, label="restricted access protocol design")
        _sha256(self.access_contract_sha256, label="restricted access contract")
        if self.platform not in {"windows", "posix"}:
            raise ValueError("restricted access platform is unsupported")
        principal_pattern = (
            _WINDOWS_PRINCIPAL_PATTERN if self.platform == "windows" else _POSIX_PRINCIPAL_PATTERN
        )
        if principal_pattern.fullmatch(self.current_principal) is None:
            raise ValueError("restricted access current principal is not numeric")
        directory = _relative_posix(
            self.directory_relative_path,
            label="restricted artifact directory",
        )
        ids = tuple(item[0] for item in self.file_artifacts)
        if ids != RESTRICTED_ARTIFACT_IDS:
            raise ValueError("restricted artifact IDs differ from the frozen four-file set")
        file_paths: dict[str, tuple[str, str]] = {}
        for artifact_id, relative_path, digest in self.file_artifacts:
            path = _relative_posix(relative_path, label=f"restricted artifact {artifact_id}")
            _sha256(digest, label=f"restricted artifact {artifact_id}")
            pure_path = PurePosixPath(path)
            pure_directory = PurePosixPath(directory)
            if (
                pure_path == pure_directory
                or not pure_path.is_relative_to(pure_directory)
                or pure_path.parent != pure_directory
            ):
                raise ValueError("restricted artifact escapes its frozen directory")
            file_paths[path] = (artifact_id, digest)
        if len(file_paths) != len(RESTRICTED_ARTIFACT_IDS):
            raise ValueError("restricted artifact paths are not unique")
        if len(self.observations) != len(RESTRICTED_ARTIFACT_IDS) + 1:
            raise ValueError("restricted access evidence must contain exactly five paths")
        observed_paths = tuple(item.relative_path for item in self.observations)
        if observed_paths != tuple(sorted({directory, *file_paths})):
            raise ValueError("restricted access observations differ from the exact five paths")
        for item in self.observations:
            if item.relative_path == directory:
                if item.path_kind != "directory" or item.verified_sha256 is not None:
                    raise ValueError("restricted artifact directory observation changed")
            else:
                expected = file_paths.get(item.relative_path)
                if (
                    expected is None
                    or item.path_kind != "regular_file"
                    or item.verified_sha256 != expected[1]
                ):
                    raise ValueError("restricted file observation hash or type changed")
            _validate_observation_permissions(
                item,
                platform=self.platform,
                current_principal=self.current_principal,
            )

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "access_contract_sha256": self.access_contract_sha256,
                "current_principal": self.current_principal,
                "directory_relative_path": self.directory_relative_path,
                "file_artifacts": {
                    artifact_id: {"relative_path": path, "verified_sha256": digest}
                    for artifact_id, path, digest in self.file_artifacts
                },
                "observations": [item.as_mapping() for item in self.observations],
                "platform": self.platform,
                "protocol_design_sha256": self.protocol_design_sha256,
                "schema_version": RESTRICTED_ACCESS_SCHEMA_VERSION,
                "target_bytes_read": False,
                "target_path_observed": False,
            }
        )

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> RestrictedLocalArtifactAccessControlEvidence:
        expected = {
            "access_contract_sha256",
            "content_sha256",
            "current_principal",
            "directory_relative_path",
            "file_artifacts",
            "observations",
            "platform",
            "protocol_design_sha256",
            "schema_version",
            "target_bytes_read",
            "target_path_observed",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise RestrictedAccessError("restricted access evidence hash or schema is invalid")
        schema_version = _plain_int(
            value.get("schema_version"),
            label="restricted access schema version",
            maximum=RESTRICTED_ACCESS_SCHEMA_VERSION,
        )
        if (
            schema_version != RESTRICTED_ACCESS_SCHEMA_VERSION
            or value.get("target_bytes_read") is not False
            or value.get("target_path_observed") is not False
        ):
            raise RestrictedAccessError("restricted access evidence crossed the target boundary")
        raw_files = _mapping(value.get("file_artifacts"), label="restricted file artifacts")
        if set(raw_files) != set(RESTRICTED_ARTIFACT_IDS):
            raise RestrictedAccessError("restricted access file set changed")
        files: list[tuple[str, str, str]] = []
        for artifact_id in RESTRICTED_ARTIFACT_IDS:
            entry = _mapping(raw_files[artifact_id], label=f"restricted file {artifact_id}")
            if set(entry) != {"relative_path", "verified_sha256"}:
                raise RestrictedAccessError("restricted file identity schema changed")
            files.append(
                (
                    artifact_id,
                    cast(str, entry["relative_path"]),
                    cast(str, entry["verified_sha256"]),
                )
            )
        raw_observations = value.get("observations")
        if not isinstance(raw_observations, list):
            raise RestrictedAccessError("restricted path observations must be a list")
        try:
            result = cls(
                protocol_design_sha256=cast(str, value["protocol_design_sha256"]),
                access_contract_sha256=cast(str, value["access_contract_sha256"]),
                platform=cast(Any, value["platform"]),
                current_principal=cast(str, value["current_principal"]),
                directory_relative_path=cast(str, value["directory_relative_path"]),
                file_artifacts=tuple(files),
                observations=tuple(
                    RestrictedPathAccessObservation.from_mapping(
                        _mapping(item, label="restricted path observation")
                    )
                    for item in raw_observations
                ),
            )
        except (TypeError, ValueError, RestrictedAccessError) as exc:
            raise RestrictedAccessError("restricted access evidence invariants failed") from exc
        if result.as_mapping() != dict(value):
            raise RestrictedAccessError("restricted access evidence did not canonicalize")
        return result


def restricted_access_contract_sha256(protocol: Mapping[str, object]) -> str:
    """Hash and semantically validate the one frozen platform access contract."""

    topology = _mapping(
        protocol.get("spatial_permutation_topology"),
        label="spatial_permutation_topology",
    )
    local = _mapping(topology.get("local_restricted_artifacts"), label="local_restricted_artifacts")
    contract = _mapping(local.get("access_control"), label="local access_control")
    expected: dict[str, object] = {
        "schema_version": RESTRICTED_ACCESS_SCHEMA_VERSION,
        "policy_id": "stage4_restricted_local_artifact_access_v3",
        "required_before_target_read": True,
        "receipt_bound_to_scoring_qualification_and_execution_seal": True,
        "retained_handle_verification": {
            "exact_path_count": 5,
            "directory_and_four_frozen_artifacts_required": True,
            "no_follow_open_required": True,
            "regular_single_link_files_and_non_reparse_paths_required": True,
            "directory_handle_retained_until_all_file_checks_complete": True,
            "all_four_file_hashes_computed_on_retained_handles": True,
            "entry_and_exit_owner_permission_samples_on_same_retained_handle_required": True,
            "entry_and_exit_canonical_permission_descriptor_bytes_must_match": True,
            "handle_and_path_identity_and_state_reverified_before_release": True,
        },
        "windows": {
            "descriptor_query": "GetSecurityInfo_on_retained_handle",
            "queried_by_handle_required": True,
            "allowed_principal_roles": [
                "current_process_user",
                "local_system",
                "builtin_administrators",
            ],
            "owner_must_be_in_allowed_principal_roles": True,
            "current_process_user_explicit_full_control_required": True,
            "inherit_only_ace_may_not_satisfy_current_object_full_control": True,
            "full_control_access_mask_hex": "0x001f01ff",
            "dacl_present_required": True,
            "dacl_protected_required": True,
            "dacl_defaulted_required": False,
            "inherited_ace_count_required": 0,
            "security_descriptor_revision_required": 1,
            "security_descriptor_control_protected_bit_hex": "0x1000",
            "deny_or_unknown_ace_forbidden": True,
            "ace_for_unauthorized_principal_forbidden": True,
        },
        "posix": {
            "platform_family_required": "linux",
            "queried_by_handle_required": True,
            "owner_must_equal_effective_uid": True,
            "owner_only_file_mode": "0600",
            "owner_only_directory_mode": "0700",
            "filesystem_locality_required": "local",
            "acl_model_required": "classic_mode_bits_without_acl_xattrs",
            "acl_related_xattrs_must_equal": [],
            "allowed_local_classic_filesystems": [
                {
                    "filesystem_type": "ext2_ext3_ext4",
                    "filesystem_magic_hex": "0x0000ef53",
                },
                {"filesystem_type": "tmpfs", "filesystem_magic_hex": "0x01021994"},
                {"filesystem_type": "xfs", "filesystem_magic_hex": "0x58465342"},
                {"filesystem_type": "btrfs", "filesystem_magic_hex": "0x9123683e"},
                {"filesystem_type": "f2fs", "filesystem_magic_hex": "0xf2f52010"},
            ],
        },
        "unsupported_or_unverifiable_platform_action": "fail_closed",
    }
    if canonical_json_bytes(dict(contract)) != canonical_json_bytes(expected):
        raise RestrictedAccessError("restricted local access-control contract changed")
    return hashlib.sha256(canonical_json_bytes(dict(contract))).hexdigest()


def validate_restricted_access_against_protocol(
    protocol: Mapping[str, object],
    evidence: RestrictedLocalArtifactAccessControlEvidence,
) -> None:
    """Recompute the frozen target-blind path and access contract bindings."""

    if not isinstance(evidence, RestrictedLocalArtifactAccessControlEvidence):
        raise TypeError("evidence must be RestrictedLocalArtifactAccessControlEvidence")
    if evidence.protocol_design_sha256 != protocol_design_sha256(protocol):
        raise RestrictedAccessError("restricted access evidence uses another protocol design")
    if evidence.access_contract_sha256 != restricted_access_contract_sha256(protocol):
        raise RestrictedAccessError("restricted access evidence uses another access contract")
    topology = _mapping(
        protocol.get("spatial_permutation_topology"),
        label="spatial_permutation_topology",
    )
    local = _mapping(topology.get("local_restricted_artifacts"), label="local_restricted_artifacts")
    expected_directory = _relative_posix(
        local.get("directory"),
        label="restricted artifact directory",
    )
    if evidence.directory_relative_path != expected_directory:
        raise RestrictedAccessError("restricted access evidence uses another directory")
    expected_paths = tuple(
        (
            artifact_id,
            _relative_posix(
                local.get(artifact_id),
                label=f"restricted artifact {artifact_id}",
            ),
        )
        for artifact_id in RESTRICTED_ARTIFACT_IDS
    )
    observed_paths = tuple((artifact_id, path) for artifact_id, path, _ in evidence.file_artifacts)
    if observed_paths != expected_paths:
        raise RestrictedAccessError("restricted access evidence uses another four-file path set")


def _sid_to_string(pointer: int) -> str:
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi32.ConvertSidToStringSidW
    convert.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    convert.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    text = wintypes.LPWSTR()
    if not convert(ctypes.c_void_p(pointer), ctypes.byref(text)):
        raise RestrictedAccessError("Windows SID cannot be converted to numeric form")
    try:
        result = cast(str, text.value)
    finally:
        kernel32.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    if _SID_PATTERN.fullmatch(result) is None:
        raise RestrictedAccessError("Windows returned a malformed numeric SID")
    return result


def _current_windows_sid() -> str:
    from ctypes import wintypes

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]  # noqa: RUF012

    class TOKEN_USER(ctypes.Structure):
        _fields_ = [("User", SID_AND_ATTRIBUTES)]  # noqa: RUF012

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise RestrictedAccessError("current Windows process token cannot be opened")
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        if needed.value == 0:
            raise RestrictedAccessError("current Windows user SID size cannot be observed")
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token,
            1,
            buffer,
            needed,
            ctypes.byref(needed),
        ):
            raise RestrictedAccessError("current Windows user SID cannot be observed")
        token_user = TOKEN_USER.from_buffer(buffer)
        return _sid_to_string(cast(int, token_user.User.Sid))
    finally:
        kernel32.CloseHandle(token)


def _windows_permission_descriptor(
    file_descriptor: int,
    current_sid: str,
) -> tuple[str, Mapping[str, object]]:
    """Read a native Windows descriptor from the retained no-follow handle."""

    import msvcrt
    from ctypes import wintypes

    class ACL(ctypes.Structure):
        _fields_ = [  # noqa: RUF012
            ("AclRevision", ctypes.c_ubyte),
            ("Sbz1", ctypes.c_ubyte),
            ("AclSize", ctypes.c_ushort),
            ("AceCount", ctypes.c_ushort),
            ("Sbz2", ctypes.c_ushort),
        ]

    class ACE_HEADER(ctypes.Structure):
        _fields_ = [  # noqa: RUF012
            ("AceType", ctypes.c_ubyte),
            ("AceFlags", ctypes.c_ubyte),
            ("AceSize", ctypes.c_ushort),
        ]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetSecurityInfo.restype = wintypes.DWORD
    advapi32.GetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ushort),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetAce.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    try:
        native_handle = wintypes.HANDLE(msvcrt.get_osfhandle(file_descriptor))
    except OSError as exc:
        raise RestrictedAccessError("Windows retained handle cannot be inspected") from exc
    status = advapi32.GetSecurityInfo(
        native_handle,
        1,
        0x00000001 | 0x00000004,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if status != 0 or not descriptor.value:
        raise RestrictedAccessError("Windows security descriptor cannot be observed")
    try:
        control = ctypes.c_ushort()
        revision = wintypes.DWORD()
        if not advapi32.GetSecurityDescriptorControl(
            descriptor,
            ctypes.byref(control),
            ctypes.byref(revision),
        ):
            raise RestrictedAccessError("Windows security descriptor control cannot be observed")
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        observed_dacl = ctypes.c_void_p()
        if not advapi32.GetSecurityDescriptorDacl(
            descriptor,
            ctypes.byref(present),
            ctypes.byref(observed_dacl),
            ctypes.byref(defaulted),
        ):
            raise RestrictedAccessError("Windows DACL cannot be observed")
        if not present.value or not observed_dacl.value:
            raise RestrictedAccessError("Windows DACL is absent or null")
        if defaulted.value:
            raise RestrictedAccessError("Windows DACL is marked as defaulted")
        if not control.value & _SE_DACL_PROTECTED:
            raise RestrictedAccessError("Windows DACL still inherits access entries")
        owner_sid = _sid_to_string(cast(int, owner.value))
        allowed = {current_sid, _WINDOWS_SYSTEM_SID, _WINDOWS_ADMINISTRATORS_SID}
        if owner_sid not in allowed:
            raise RestrictedAccessError("Windows owner SID is outside the frozen principal set")
        acl = ACL.from_address(observed_dacl.value)
        entries: list[dict[str, object]] = []
        inherited_count = 0
        for index in range(acl.AceCount):
            ace_pointer = ctypes.c_void_p()
            if not advapi32.GetAce(observed_dacl, index, ctypes.byref(ace_pointer)):
                raise RestrictedAccessError("Windows ACE cannot be observed")
            address = cast(int, ace_pointer.value)
            header = ACE_HEADER.from_address(address)
            if header.AceType not in {0, 1} or header.AceSize < 12:
                raise RestrictedAccessError("Windows DACL contains an unsupported ACE")
            if header.AceFlags & _INHERITED_ACE:
                inherited_count += 1
            mask = ctypes.c_uint32.from_address(address + 4).value
            sid = _sid_to_string(address + 8)
            if sid not in allowed:
                raise RestrictedAccessError(
                    "Windows DACL grants or records access for an unauthorized principal"
                )
            entries.append(
                {
                    "ace_flags": int(header.AceFlags),
                    "ace_type": "allow" if header.AceType == 0 else "deny",
                    "access_mask": int(mask),
                    "principal_sid": sid,
                }
            )
        if inherited_count != 0:
            raise RestrictedAccessError("Windows DACL contains inherited access entries")
        return f"sid:{owner_sid}", {
            "ace_count": len(entries),
            "aces_in_native_order": entries,
            "dacl_defaulted": bool(defaulted.value),
            "dacl_present": True,
            "dacl_protected": True,
            "inherited_ace_count": 0,
            "owner_sid": owner_sid,
            "queried_by_handle": True,
            "security_descriptor_control": int(control.value),
            "security_descriptor_revision": int(revision.value),
        }
    finally:
        kernel32.LocalFree(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_state(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_uid,
        value.st_gid,
    )


def _status_is_reparse(value: os.stat_result) -> bool:
    return bool(cast(int, getattr(value, "st_file_attributes", 0)) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _require_restricted_status(
    value: os.stat_result,
    *,
    path_kind: Literal["directory", "regular_file"],
) -> None:
    if _status_is_reparse(value) or stat.S_ISLNK(value.st_mode):
        raise RestrictedAccessError("restricted artifact path is a link or reparse point")
    if path_kind == "directory":
        if not stat.S_ISDIR(value.st_mode):
            raise RestrictedAccessError("restricted artifact directory type changed")
        return
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise RestrictedAccessError(
            "restricted artifact must be a regular non-reparse single-link file"
        )


def _open_windows_directory_no_follow(path: Path) -> int:
    if sys.platform != "win32":
        raise RestrictedAccessError("Windows directory handle requested on another platform")
    import msvcrt
    from ctypes import wintypes

    read_control = 0x00020000
    file_read_attributes = 0x00000080
    file_share_read = 0x00000001
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        read_control | file_read_attributes,
        file_share_read,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise RestrictedAccessError("restricted Windows directory cannot be opened safely")
    raw_handle = cast(int, handle)
    try:
        return msvcrt.open_osfhandle(raw_handle, os.O_RDONLY | os.O_BINARY)
    except Exception as exc:
        close_handle(handle)
        raise RestrictedAccessError(
            "restricted Windows directory handle cannot be retained"
        ) from exc


def _open_directory_no_follow(path: Path) -> int:
    if sys.platform == "win32":
        return _open_windows_directory_no_follow(path)
    no_follow = getattr(os, "O_NOFOLLOW", None)  # type: ignore[unreachable]
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or not isinstance(directory_flag, int):
        raise RestrictedAccessError("platform has no race-safe no-follow directory open")
    close_on_exec = cast(int, getattr(os, "O_CLOEXEC", 0))
    try:
        return os.open(path, os.O_RDONLY | no_follow | directory_flag | close_on_exec)
    except OSError as exc:
        raise RestrictedAccessError("restricted directory cannot be opened safely") from exc


@contextmanager
def _open_restricted_directory(
    root: Path,
    path: Path,
) -> Iterator[tuple[int, os.stat_result]]:
    descriptor: int | None = None
    try:
        require_existing_real_directory_tree(
            root,
            path,
            label="restricted artifact directory",
        )
        before = os.lstat(path)
        _require_restricted_status(before, path_kind="directory")
        descriptor = _open_directory_no_follow(path)
        opened = os.fstat(descriptor)
        _require_restricted_status(opened, path_kind="directory")
        if _stat_identity(opened) != _stat_identity(before) or _stat_state(opened) != _stat_state(
            before
        ):
            raise RestrictedAccessError("restricted directory changed during no-follow open")
        yield descriptor, opened
        after_handle = os.fstat(descriptor)
        after_path = os.lstat(path)
        _require_restricted_status(after_handle, path_kind="directory")
        _require_restricted_status(after_path, path_kind="directory")
        if (
            _stat_identity(after_handle) != _stat_identity(opened)
            or _stat_identity(after_path) != _stat_identity(opened)
            or _stat_state(after_handle) != _stat_state(opened)
            or _stat_state(after_path) != _stat_state(opened)
        ):
            raise RestrictedAccessError("restricted directory changed during access observation")
        require_existing_real_directory_tree(
            root,
            path,
            label="restricted artifact directory",
        )
    except RestrictedAccessError:
        raise
    except (OSError, UnsafeImmutableFileError) as exc:
        raise RestrictedAccessError("restricted directory identity cannot be proven") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _open_posix_file_at(
    directory_descriptor: int,
    name: str,
) -> Iterator[BufferedReader]:
    handle: BufferedReader | None = None
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        _require_restricted_status(before, path_kind="regular_file")
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if not isinstance(no_follow, int):
            raise RestrictedAccessError("POSIX platform has no no-follow file open")
        close_on_exec = cast(int, getattr(os, "O_CLOEXEC", 0))
        descriptor = os.open(
            name,
            os.O_RDONLY | no_follow | close_on_exec,
            dir_fd=directory_descriptor,
        )
        handle = _fdopen_binary_owned(descriptor)
        opened = os.fstat(handle.fileno())
        _require_restricted_status(opened, path_kind="regular_file")
        if _stat_identity(opened) != _stat_identity(before) or _stat_state(opened) != _stat_state(
            before
        ):
            raise RestrictedAccessError("restricted POSIX file changed during no-follow open")
        yield handle
        after_handle = os.fstat(handle.fileno())
        after_entry = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        _require_restricted_status(after_handle, path_kind="regular_file")
        _require_restricted_status(after_entry, path_kind="regular_file")
        if (
            _stat_identity(after_handle) != _stat_identity(opened)
            or _stat_identity(after_entry) != _stat_identity(opened)
            or _stat_state(after_handle) != _stat_state(opened)
            or _stat_state(after_entry) != _stat_state(opened)
        ):
            raise RestrictedAccessError("restricted POSIX file changed while observed")
    except RestrictedAccessError:
        raise
    except OSError as exc:
        raise RestrictedAccessError("restricted POSIX file cannot be opened safely") from exc
    finally:
        if handle is not None:
            handle.close()


def _fdopen_binary_owned(descriptor: int) -> BufferedReader:
    """Transfer one raw descriptor to a binary stream without a failure leak."""

    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def _platform_context() -> tuple[Literal["windows", "posix"], str]:
    if os.name == "nt":
        return "windows", f"sid:{_current_windows_sid()}"
    if sys.platform.startswith("linux") and os.name == "posix" and hasattr(os, "geteuid"):
        return "posix", f"uid:{os.geteuid()}"
    raise RestrictedAccessError("host platform cannot prove the restricted access contract")


def _runtime_platform_identity() -> tuple[str, str]:
    return sys.platform, os.name


def _linux_fstatfs_magic(file_descriptor: int) -> int:
    """Return Linux ``f_type`` for a retained descriptor without path lookup."""

    platform_name, os_name = _runtime_platform_identity()
    if not platform_name.startswith("linux") or os_name != "posix":
        raise RestrictedAccessError("Linux filesystem identity requested on another platform")
    _plain_int(file_descriptor, label="Linux filesystem descriptor")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fstatfs = libc.fstatfs
        fstatfs.argtypes = (ctypes.c_int, ctypes.c_void_p)
        fstatfs.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(256)
        if fstatfs(file_descriptor, ctypes.byref(buffer)) != 0:
            raise OSError(ctypes.get_errno(), "fstatfs failed")
        return int(ctypes.c_ulong.from_buffer(buffer).value & 0xFFFFFFFF)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise RestrictedAccessError("restricted Linux filesystem identity is unverifiable") from exc


def _linux_local_filesystem_identity(file_descriptor: int) -> tuple[str, str]:
    magic = _linux_fstatfs_magic(file_descriptor)
    filesystem_type = _LINUX_LOCAL_CLASSIC_FILESYSTEMS.get(magic)
    if filesystem_type is None:
        raise RestrictedAccessError(
            f"restricted Linux filesystem 0x{magic:08x} is not an allowed local classic model"
        )
    return filesystem_type, f"0x{magic:08x}"


def _acl_related_xattr_names(file_descriptor: int) -> tuple[str, ...]:
    try:
        raw_names = cast(Any, os).listxattr(file_descriptor)
    except (AttributeError, OSError, TypeError) as exc:
        raise RestrictedAccessError("restricted POSIX extended ACL state is unverifiable") from exc
    if not isinstance(raw_names, list) or any(not isinstance(name, str) for name in raw_names):
        raise RestrictedAccessError("restricted POSIX xattr names are unverifiable")
    return tuple(sorted(name for name in raw_names if "acl" in name.casefold()))


def _permission_descriptor(
    file_descriptor: int,
    *,
    platform: Literal["windows", "posix"],
    current_principal: str,
    path_kind: Literal["directory", "regular_file"],
    status: os.stat_result,
) -> tuple[str, Mapping[str, object]]:
    if platform == "windows":
        match = _WINDOWS_PRINCIPAL_PATTERN.fullmatch(current_principal)
        if match is None:
            raise RestrictedAccessError("current Windows principal is malformed")
        return _windows_permission_descriptor(file_descriptor, match.group(1))
    uid = int(current_principal.removeprefix("uid:"))
    mode = stat.S_IMODE(status.st_mode)
    expected_mode = 0o700 if path_kind == "directory" else 0o600
    if status.st_uid != uid:
        raise RestrictedAccessError("restricted POSIX path is not owned by the effective UID")
    if mode != expected_mode:
        raise RestrictedAccessError("restricted POSIX path mode differs from the frozen contract")
    filesystem_type, filesystem_magic = _linux_local_filesystem_identity(file_descriptor)
    acl_related_xattrs = _acl_related_xattr_names(file_descriptor)
    if acl_related_xattrs:
        raise RestrictedAccessError("restricted POSIX path carries an ACL-related xattr")
    return f"uid:{status.st_uid}", {
        "acl_model": "classic_mode_bits_without_acl_xattrs",
        "acl_related_xattrs": [],
        "filesystem_locality": "local",
        "filesystem_magic_hex": filesystem_magic,
        "filesystem_type": filesystem_type,
        "mode_octal": format(mode, "04o"),
        "owner_uid": int(status.st_uid),
        "platform_family": "linux",
        "queried_by_handle": True,
    }


@dataclass(frozen=True, slots=True)
class _RetainedPermissionSample:
    file_descriptor: int
    path: Path
    path_kind: Literal["directory", "regular_file"]
    initial_status: os.stat_result
    owner_principal: str
    permission_descriptor_bytes: bytes
    directory_descriptor: int | None = None
    entry_name: str | None = None


def _retained_path_status(sample: _RetainedPermissionSample) -> os.stat_result:
    if sample.directory_descriptor is not None:
        if sample.path_kind != "regular_file" or not sample.entry_name:
            raise RestrictedAccessError("retained POSIX entry binding is inconsistent")
        return os.stat(
            sample.entry_name,
            dir_fd=sample.directory_descriptor,
            follow_symlinks=False,
        )
    return os.lstat(sample.path)


def _require_retained_sample_unchanged(
    root: Path,
    sample: _RetainedPermissionSample,
) -> os.stat_result:
    label = (
        "restricted artifact directory"
        if sample.path_kind == "directory"
        else "restricted artifact parent directory"
    )
    tree_path = sample.path if sample.path_kind == "directory" else sample.path.parent
    require_existing_real_directory_tree(root, tree_path, label=label)
    handle_status = os.fstat(sample.file_descriptor)
    path_status = _retained_path_status(sample)
    _require_restricted_status(handle_status, path_kind=sample.path_kind)
    _require_restricted_status(path_status, path_kind=sample.path_kind)
    if (
        _stat_identity(handle_status) != _stat_identity(sample.initial_status)
        or _stat_identity(path_status) != _stat_identity(sample.initial_status)
        or _stat_state(handle_status) != _stat_state(sample.initial_status)
        or _stat_state(path_status) != _stat_state(sample.initial_status)
    ):
        raise RestrictedAccessError("restricted retained handle or path changed during observation")
    return handle_status


def _resample_retained_permissions(
    root: Path,
    samples: tuple[_RetainedPermissionSample, ...],
    *,
    platform: Literal["windows", "posix"],
    current_principal: str,
) -> None:
    """Re-read owner/ACL on each original handle immediately before release."""

    for sample in samples:
        before = _require_retained_sample_unchanged(root, sample)
        owner, descriptor = _permission_descriptor(
            sample.file_descriptor,
            platform=platform,
            current_principal=current_principal,
            path_kind=sample.path_kind,
            status=before,
        )
        after = _require_retained_sample_unchanged(root, sample)
        if _stat_identity(after) != _stat_identity(before) or _stat_state(after) != _stat_state(
            before
        ):
            raise RestrictedAccessError("restricted retained handle changed during ACL resampling")
        descriptor_bytes = canonical_json_bytes(dict(descriptor))
        if (
            owner != sample.owner_principal
            or descriptor_bytes != sample.permission_descriptor_bytes
        ):
            raise RestrictedAccessError("restricted owner or permission descriptor changed")


def observe_restricted_local_artifact_access_control(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    expected_file_hashes: Mapping[str, str],
) -> RestrictedLocalArtifactAccessControlEvidence:
    """Hash and inspect the exact five paths through retained no-follow handles."""

    root = Path(project_root).resolve()
    topology = _mapping(
        protocol.get("spatial_permutation_topology"),
        label="spatial_permutation_topology",
    )
    local = _mapping(topology.get("local_restricted_artifacts"), label="local_restricted_artifacts")
    directory = _relative_posix(local.get("directory"), label="restricted artifact directory")
    expected_paths = {
        artifact_id: _relative_posix(
            local.get(artifact_id),
            label=f"restricted artifact {artifact_id}",
        )
        for artifact_id in RESTRICTED_ARTIFACT_IDS
    }
    pure_directory = PurePosixPath(directory)
    if any(
        PurePosixPath(relative).parent != pure_directory for relative in expected_paths.values()
    ):
        raise RestrictedAccessError("restricted artifact path escapes the frozen directory")
    if set(expected_file_hashes) != set(RESTRICTED_ARTIFACT_IDS):
        raise RestrictedAccessError("expected restricted artifact hash set changed")
    expected_artifacts = tuple(
        (
            artifact_id,
            expected_paths[artifact_id],
            _sha256(
                expected_file_hashes[artifact_id],
                label=f"expected restricted artifact {artifact_id}",
            ),
        )
        for artifact_id in RESTRICTED_ARTIFACT_IDS
    )
    platform, current_principal = _platform_context()
    try:
        directory_path = Path(os.path.abspath(os.fspath(root.joinpath(*pure_directory.parts))))
        if not directory_path.is_relative_to(root):
            raise RestrictedAccessError("restricted artifact directory escapes the project root")
        with ExitStack() as stack:
            directory_descriptor, directory_status = stack.enter_context(
                _open_restricted_directory(root, directory_path)
            )
            directory_owner, directory_permissions = _permission_descriptor(
                directory_descriptor,
                platform=platform,
                current_principal=current_principal,
                path_kind="directory",
                status=directory_status,
            )
            directory_permission_bytes = canonical_json_bytes(dict(directory_permissions))
            retained_samples = [
                _RetainedPermissionSample(
                    file_descriptor=directory_descriptor,
                    path=directory_path,
                    path_kind="directory",
                    initial_status=directory_status,
                    owner_principal=directory_owner,
                    permission_descriptor_bytes=directory_permission_bytes,
                )
            ]
            observations = [
                RestrictedPathAccessObservation(
                    relative_path=directory,
                    path_kind="directory",
                    verified_sha256=None,
                    owner_principal=directory_owner,
                    link_count=int(directory_status.st_nlink),
                    reparse_point=False,
                    permission_descriptor_json=directory_permission_bytes.decode("utf-8"),
                )
            ]
            observed_artifacts: list[tuple[str, str, str]] = []
            for artifact_id, relative, expected_digest in expected_artifacts:
                pure_relative = PurePosixPath(relative)
                file_path = Path(os.path.abspath(os.fspath(root.joinpath(*pure_relative.parts))))
                if not file_path.is_relative_to(root) or file_path.parent != directory_path:
                    raise RestrictedAccessError(
                        "restricted artifact path escapes its retained directory"
                    )
                if platform == "posix":
                    handle = stack.enter_context(
                        _open_posix_file_at(directory_descriptor, pure_relative.name)
                    )
                else:
                    require_existing_real_directory_tree(
                        root,
                        file_path.parent,
                        label="restricted artifact parent directory",
                    )
                    handle = stack.enter_context(
                        open_existing_immutable_file(
                            file_path,
                            label=f"restricted artifact {artifact_id}",
                        )
                    )
                file_status = os.fstat(handle.fileno())
                _require_restricted_status(file_status, path_kind="regular_file")
                digest = hashlib.sha256()
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                actual_digest = digest.hexdigest()
                if actual_digest != expected_digest:
                    raise RestrictedAccessError(f"restricted artifact hash changed: {artifact_id}")
                permission_status = os.fstat(handle.fileno())
                _require_restricted_status(permission_status, path_kind="regular_file")
                permission_path_status = (
                    os.stat(
                        pure_relative.name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if platform == "posix"
                    else os.lstat(file_path)
                )
                _require_restricted_status(permission_path_status, path_kind="regular_file")
                if (
                    _stat_identity(permission_status) != _stat_identity(file_status)
                    or _stat_identity(permission_path_status) != _stat_identity(file_status)
                    or _stat_state(permission_status) != _stat_state(file_status)
                    or _stat_state(permission_path_status) != _stat_state(file_status)
                ):
                    raise RestrictedAccessError(
                        f"restricted artifact changed before ACL observation: {artifact_id}"
                    )
                owner, permissions = _permission_descriptor(
                    handle.fileno(),
                    platform=platform,
                    current_principal=current_principal,
                    path_kind="regular_file",
                    status=permission_status,
                )
                permission_bytes = canonical_json_bytes(dict(permissions))
                retained_samples.append(
                    _RetainedPermissionSample(
                        file_descriptor=handle.fileno(),
                        path=file_path,
                        path_kind="regular_file",
                        initial_status=permission_status,
                        owner_principal=owner,
                        permission_descriptor_bytes=permission_bytes,
                        directory_descriptor=(
                            directory_descriptor if platform == "posix" else None
                        ),
                        entry_name=pure_relative.name if platform == "posix" else None,
                    )
                )
                observed_artifacts.append((artifact_id, relative, actual_digest))
                observations.append(
                    RestrictedPathAccessObservation(
                        relative_path=relative,
                        path_kind="regular_file",
                        verified_sha256=actual_digest,
                        owner_principal=owner,
                        link_count=int(permission_status.st_nlink),
                        reparse_point=False,
                        permission_descriptor_json=permission_bytes.decode("utf-8"),
                    )
                )
            result = RestrictedLocalArtifactAccessControlEvidence(
                protocol_design_sha256=protocol_design_sha256(protocol),
                access_contract_sha256=restricted_access_contract_sha256(protocol),
                platform=platform,
                current_principal=current_principal,
                directory_relative_path=directory,
                file_artifacts=tuple(observed_artifacts),
                observations=tuple(sorted(observations, key=lambda item: item.relative_path)),
            )
            validate_restricted_access_against_protocol(protocol, result)
            _resample_retained_permissions(
                root,
                tuple(retained_samples),
                platform=platform,
                current_principal=current_principal,
            )
        return result
    except (OSError, UnsafeImmutableFileError, ValueError) as exc:
        raise RestrictedAccessError("restricted access observation is inconsistent") from exc


__all__ = [
    "RESTRICTED_ACCESS_SCHEMA_VERSION",
    "RESTRICTED_ARTIFACT_IDS",
    "RestrictedAccessError",
    "RestrictedLocalArtifactAccessControlEvidence",
    "RestrictedPathAccessObservation",
    "canonical_permission_descriptor",
    "observe_restricted_local_artifact_access_control",
    "restricted_access_contract_sha256",
    "validate_restricted_access_against_protocol",
]
