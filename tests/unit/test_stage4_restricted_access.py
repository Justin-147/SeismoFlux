from __future__ import annotations

import hashlib
import os
import stat
from collections import Counter
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import seismoflux.anomaly_increment.qualification as qualification_module
import seismoflux.anomaly_increment.restricted_access as restricted_access_module
from seismoflux.anomaly_increment.preregistration import (
    protocol_design_sha256,
    with_content_sha256,
)
from seismoflux.anomaly_increment.qualification import QualificationCheckReceipt
from seismoflux.anomaly_increment.restricted_access import (
    RESTRICTED_ARTIFACT_IDS,
    RestrictedAccessError,
    RestrictedLocalArtifactAccessControlEvidence,
    RestrictedPathAccessObservation,
    canonical_permission_descriptor,
    observe_restricted_local_artifact_access_control,
    restricted_access_contract_sha256,
    validate_restricted_access_against_protocol,
)

_CURRENT_UID = 1000
_CURRENT_SID = "S-1-5-21-111-222-333-1001"
_OTHER_SID = "S-1-5-21-111-222-333-1002"
_UNAUTHORIZED_SID = "S-1-5-11"
_WINDOWS_FULL_CONTROL_MASK = 0x001F01FF
_FROZEN_COUNT_GATE = "frozen_full_non_target_junit_count_matches_actual_scoring_freeze_suite"


def _effective_uid() -> int:
    return int(cast(Any, os).geteuid())


def _access_contract() -> dict[str, object]:
    return {
        "schema_version": 3,
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


def _protocol() -> dict[str, object]:
    directory = "data/private/stage4"
    local: dict[str, object] = {
        "access_control": _access_contract(),
        "directory": directory,
    }
    local.update(
        {artifact_id: f"{directory}/{artifact_id}.bin" for artifact_id in RESTRICTED_ARTIFACT_IDS}
    )
    return {
        "generated_manifests": {},
        "spatial_permutation_topology": {"local_restricted_artifacts": local},
    }


def _windows_descriptor(
    *,
    owner_sid: str = _CURRENT_SID,
    ace_sid: str = _CURRENT_SID,
) -> dict[str, object]:
    return {
        "ace_count": 1,
        "aces_in_native_order": [
            {
                "ace_flags": 0,
                "ace_type": "allow",
                "access_mask": _WINDOWS_FULL_CONTROL_MASK,
                "principal_sid": ace_sid,
            }
        ],
        "dacl_defaulted": False,
        "dacl_present": True,
        "dacl_protected": True,
        "inherited_ace_count": 0,
        "owner_sid": owner_sid,
        "queried_by_handle": True,
        "security_descriptor_control": 0x1000,
        "security_descriptor_revision": 1,
    }


def _posix_descriptor(
    *,
    mode: str,
    owner_uid: int | float | bool = _CURRENT_UID,
) -> dict[str, object]:
    return {
        "acl_model": "classic_mode_bits_without_acl_xattrs",
        "acl_related_xattrs": [],
        "filesystem_locality": "local",
        "filesystem_magic_hex": "0x0000ef53",
        "filesystem_type": "ext2_ext3_ext4",
        "mode_octal": mode,
        "owner_uid": owner_uid,
        "platform_family": "linux",
        "queried_by_handle": True,
    }


def _evidence(
    *,
    platform: str,
    access_contract_sha256: str | None = None,
) -> RestrictedLocalArtifactAccessControlEvidence:
    protocol = _protocol()
    local = cast(
        dict[str, object],
        cast(dict[str, object], protocol["spatial_permutation_topology"])[
            "local_restricted_artifacts"
        ],
    )
    directory = cast(str, local["directory"])
    if platform == "posix":
        current_principal = f"uid:{_CURRENT_UID}"
        directory_descriptor = _posix_descriptor(mode="0700")
        file_descriptor = _posix_descriptor(mode="0600")
    else:
        current_principal = f"sid:{_CURRENT_SID}"
        directory_descriptor = _windows_descriptor()
        file_descriptor = _windows_descriptor()

    file_artifacts = tuple(
        (
            artifact_id,
            cast(str, local[artifact_id]),
            f"{index:x}" * 64,
        )
        for index, artifact_id in enumerate(RESTRICTED_ARTIFACT_IDS, start=1)
    )
    observations = [
        RestrictedPathAccessObservation(
            relative_path=directory,
            path_kind="directory",
            verified_sha256=None,
            owner_principal=current_principal,
            link_count=1,
            reparse_point=False,
            permission_descriptor_json=canonical_permission_descriptor(directory_descriptor),
        )
    ]
    observations.extend(
        RestrictedPathAccessObservation(
            relative_path=relative_path,
            path_kind="regular_file",
            verified_sha256=digest,
            owner_principal=current_principal,
            link_count=1,
            reparse_point=False,
            permission_descriptor_json=canonical_permission_descriptor(file_descriptor),
        )
        for _, relative_path, digest in file_artifacts
    )
    return RestrictedLocalArtifactAccessControlEvidence(
        protocol_design_sha256=protocol_design_sha256(protocol),
        access_contract_sha256=(
            access_contract_sha256
            if access_contract_sha256 is not None
            else restricted_access_contract_sha256(protocol)
        ),
        platform=cast(Any, platform),
        current_principal=current_principal,
        directory_relative_path=directory,
        file_artifacts=file_artifacts,
        observations=tuple(sorted(observations, key=lambda item: item.relative_path)),
    )


def _rehash(mapping: dict[str, object]) -> dict[str, object]:
    return with_content_sha256(mapping)


def _first_observation(mapping: dict[str, object]) -> dict[str, object]:
    observations = cast(list[dict[str, object]], mapping["observations"])
    return observations[0]


@pytest.mark.parametrize(
    ("mode", "owner_uid", "owner_principal"),
    (
        ("0777", _CURRENT_UID, f"uid:{_CURRENT_UID}"),
        ("0700", _CURRENT_UID + 1, f"uid:{_CURRENT_UID + 1}"),
    ),
    ids=("world-writable-mode", "owner-mismatch"),
)
def test_posix_from_mapping_rejects_rehashed_permission_forgery(
    mode: str,
    owner_uid: int,
    owner_principal: str,
) -> None:
    mapping = deepcopy(_evidence(platform="posix").as_mapping())
    observation = _first_observation(mapping)
    observation["owner_principal"] = owner_principal
    observation["permission_descriptor"] = _posix_descriptor(
        mode=mode,
        owner_uid=owner_uid,
    )

    with pytest.raises(RestrictedAccessError):
        RestrictedLocalArtifactAccessControlEvidence.from_mapping(_rehash(mapping))


@pytest.mark.parametrize(
    "mutation",
    (
        "unprotected_dacl",
        "inherited_ace",
        "unauthorized_principal",
        "current_sid_mismatch",
        "missing_current_user_full_control",
        "inherit_only_current_user_full_control",
        "deny_ace",
        "defaulted_dacl",
    ),
)
def test_windows_from_mapping_rejects_rehashed_permission_forgery(
    mutation: str,
) -> None:
    mapping = deepcopy(_evidence(platform="windows").as_mapping())
    observation = _first_observation(mapping)
    descriptor = cast(dict[str, object], observation["permission_descriptor"])

    if mutation == "unprotected_dacl":
        descriptor["dacl_protected"] = False
        descriptor["security_descriptor_control"] = 0
    elif mutation == "inherited_ace":
        descriptor["inherited_ace_count"] = 1
        aces = cast(list[dict[str, object]], descriptor["aces_in_native_order"])
        aces[0]["ace_flags"] = 0x10
    elif mutation == "unauthorized_principal":
        aces = cast(list[dict[str, object]], descriptor["aces_in_native_order"])
        aces.append(
            {
                "ace_flags": 0,
                "ace_type": "allow",
                "access_mask": 1,
                "principal_sid": _UNAUTHORIZED_SID,
            }
        )
        descriptor["ace_count"] = 2
    elif mutation == "current_sid_mismatch":
        mapping["current_principal"] = f"sid:{_OTHER_SID}"
    elif mutation == "missing_current_user_full_control":
        aces = cast(list[dict[str, object]], descriptor["aces_in_native_order"])
        aces[0]["access_mask"] = 0x00020089
    elif mutation == "inherit_only_current_user_full_control":
        aces = cast(list[dict[str, object]], descriptor["aces_in_native_order"])
        aces[0]["ace_flags"] = 0x08
    elif mutation == "deny_ace":
        aces = cast(list[dict[str, object]], descriptor["aces_in_native_order"])
        aces[0]["ace_type"] = "deny"
    else:
        descriptor["dacl_defaulted"] = True

    with pytest.raises(RestrictedAccessError):
        RestrictedLocalArtifactAccessControlEvidence.from_mapping(_rehash(mapping))


def test_protocol_validator_rejects_access_contract_sha_mismatch() -> None:
    protocol = _protocol()
    evidence = _evidence(platform="posix", access_contract_sha256="f" * 64)

    with pytest.raises(RestrictedAccessError, match="contract"):
        validate_restricted_access_against_protocol(protocol, evidence)


def test_access_contract_rejects_integer_equal_to_boolean_forgery() -> None:
    protocol = _protocol()
    topology = cast(dict[str, object], protocol["spatial_permutation_topology"])
    local = cast(dict[str, object], topology["local_restricted_artifacts"])
    contract = cast(dict[str, object], local["access_control"])
    contract["required_before_target_read"] = 1

    with pytest.raises(RestrictedAccessError, match="contract changed"):
        restricted_access_contract_sha256(protocol)


def test_production_unfrozen_test_count_makes_qualification_gate_fail_closed() -> None:
    assert qualification_module.FROZEN_FULL_NON_TARGET_TEST_COUNT is None

    with pytest.raises(ValueError, match="not finally frozen"):
        QualificationCheckReceipt(
            name=_FROZEN_COUNT_GATE,
            evidence_kind="full_pytest",
            evidence_sha256="a" * 64,
            expected_test_count=128,
        )


def _materialize_restricted_artifacts(
    root: Path,
) -> tuple[dict[str, object], dict[str, str], Path]:
    protocol = _protocol()
    local = cast(
        dict[str, object],
        cast(dict[str, object], protocol["spatial_permutation_topology"])[
            "local_restricted_artifacts"
        ],
    )
    directory = root.joinpath(*Path(cast(str, local["directory"])).parts)
    directory.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for index, artifact_id in enumerate(RESTRICTED_ARTIFACT_IDS, start=1):
        payload = f"artifact-{artifact_id}-{index}".encode()
        root.joinpath(*Path(cast(str, local[artifact_id])).parts).write_bytes(payload)
        hashes[artifact_id] = hashlib.sha256(payload).hexdigest()
    return protocol, hashes, directory


def _synthetic_platform_permissions(
    file_descriptor: int,
    *,
    platform: str,
    current_principal: str,
    path_kind: str,
    status: os.stat_result,
) -> tuple[str, dict[str, object]]:
    assert os.fstat(file_descriptor) == status
    if path_kind == "regular_file":
        assert os.lseek(file_descriptor, 0, os.SEEK_CUR) == status.st_size
    if platform == "windows":
        sid = current_principal.removeprefix("sid:")
        return current_principal, _windows_descriptor(owner_sid=sid, ace_sid=sid)
    uid = int(current_principal.removeprefix("uid:"))
    return current_principal, _posix_descriptor(
        mode="0700" if path_kind == "directory" else "0600",
        owner_uid=uid,
    )


def test_observer_hashes_and_queries_permissions_on_the_same_retained_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, hashes, _ = _materialize_restricted_artifacts(tmp_path)
    platform = "windows" if os.name == "nt" else "posix"
    principal = f"sid:{_CURRENT_SID}" if platform == "windows" else f"uid:{_effective_uid()}"
    observed_descriptors: list[int] = []

    def observe_permissions(
        file_descriptor: int,
        **kwargs: object,
    ) -> tuple[str, dict[str, object]]:
        observed_descriptors.append(file_descriptor)
        return _synthetic_platform_permissions(file_descriptor, **cast(Any, kwargs))

    monkeypatch.setattr(
        restricted_access_module,
        "_platform_context",
        lambda: (platform, principal),
    )
    monkeypatch.setattr(
        restricted_access_module,
        "_permission_descriptor",
        observe_permissions,
    )
    evidence = observe_restricted_local_artifact_access_control(
        tmp_path,
        protocol,
        expected_file_hashes=hashes,
    )

    assert len(observed_descriptors) == 10
    counts = Counter(observed_descriptors)
    assert len(counts) == 5
    assert set(counts.values()) == {2}
    assert (
        dict((artifact_id, digest) for artifact_id, _, digest in evidence.file_artifacts) == hashes
    )
    for descriptor in counts:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_observer_fails_if_file_path_is_replaced_during_handle_acl_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, hashes, directory = _materialize_restricted_artifacts(tmp_path)
    platform = "windows" if os.name == "nt" else "posix"
    principal = f"sid:{_CURRENT_SID}" if platform == "windows" else f"uid:{_effective_uid()}"
    first_path = directory / f"{RESTRICTED_ARTIFACT_IDS[0]}.bin"
    replacement = directory / "replacement.bin"
    replacement.write_bytes(b"replacement")
    replacement_attempted = False

    def replace_during_permissions(
        file_descriptor: int,
        **kwargs: object,
    ) -> tuple[str, dict[str, object]]:
        nonlocal replacement_attempted
        if kwargs.get("path_kind") == "regular_file" and not replacement_attempted:
            replacement_attempted = True
            os.replace(replacement, first_path)
        return _synthetic_platform_permissions(file_descriptor, **cast(Any, kwargs))

    monkeypatch.setattr(
        restricted_access_module,
        "_platform_context",
        lambda: (platform, principal),
    )
    monkeypatch.setattr(
        restricted_access_module,
        "_permission_descriptor",
        replace_during_permissions,
    )
    with pytest.raises(RestrictedAccessError):
        observe_restricted_local_artifact_access_control(
            tmp_path,
            protocol,
            expected_file_hashes=hashes,
        )
    assert replacement_attempted is True


def test_observer_fails_if_retained_directory_entry_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, hashes, directory = _materialize_restricted_artifacts(tmp_path)
    platform = "windows" if os.name == "nt" else "posix"
    principal = f"sid:{_CURRENT_SID}" if platform == "windows" else f"uid:{_effective_uid()}"
    renamed = directory.with_name("stage4-renamed")
    replacement_attempted = False

    def replace_directory_during_permissions(
        file_descriptor: int,
        **kwargs: object,
    ) -> tuple[str, dict[str, object]]:
        nonlocal replacement_attempted
        if kwargs.get("path_kind") == "regular_file" and not replacement_attempted:
            replacement_attempted = True
            os.replace(directory, renamed)
            directory.symlink_to(renamed, target_is_directory=True)
        return _synthetic_platform_permissions(file_descriptor, **cast(Any, kwargs))

    monkeypatch.setattr(
        restricted_access_module,
        "_platform_context",
        lambda: (platform, principal),
    )
    monkeypatch.setattr(
        restricted_access_module,
        "_permission_descriptor",
        replace_directory_during_permissions,
    )
    with pytest.raises(RestrictedAccessError):
        observe_restricted_local_artifact_access_control(
            tmp_path,
            protocol,
            expected_file_hashes=hashes,
        )
    assert replacement_attempted is True


@pytest.mark.parametrize("drift_kind", ("owner", "descriptor"))
def test_observer_rejects_exit_resample_owner_or_acl_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    protocol, hashes, _ = _materialize_restricted_artifacts(tmp_path)
    platform = "windows" if os.name == "nt" else "posix"
    principal = f"sid:{_CURRENT_SID}" if platform == "windows" else f"uid:{_effective_uid()}"
    calls: Counter[int] = Counter()

    def drift_on_second_sample(
        file_descriptor: int,
        **kwargs: object,
    ) -> tuple[str, dict[str, object]]:
        calls[file_descriptor] += 1
        owner, descriptor = _synthetic_platform_permissions(
            file_descriptor,
            **cast(Any, kwargs),
        )
        if kwargs.get("path_kind") == "directory" and calls[file_descriptor] == 2:
            if drift_kind == "owner":
                owner = f"sid:{_OTHER_SID}" if platform == "windows" else "uid:9999"
            else:
                descriptor = dict(descriptor)
                descriptor["queried_by_handle"] = False
        return owner, descriptor

    monkeypatch.setattr(
        restricted_access_module,
        "_platform_context",
        lambda: (platform, principal),
    )
    monkeypatch.setattr(
        restricted_access_module,
        "_permission_descriptor",
        drift_on_second_sample,
    )

    with pytest.raises(RestrictedAccessError, match="owner or permission descriptor changed"):
        observe_restricted_local_artifact_access_control(
            tmp_path,
            protocol,
            expected_file_hashes=hashes,
        )
    assert 2 in calls.values()


@pytest.mark.parametrize(
    "filesystem_magic",
    (
        0x00006969,  # NFS / NFSv4
        0xFF534D42,  # CIFS
        0x65735546,  # FUSE cannot prove its backing filesystem
        0x794C7630,  # overlay cannot prove its lower filesystem
        0xDEADBEEF,  # unknown
    ),
)
def test_posix_filesystem_allowlist_rejects_network_layered_and_unknown_models(
    monkeypatch: pytest.MonkeyPatch,
    filesystem_magic: int,
) -> None:
    monkeypatch.setattr(
        restricted_access_module,
        "_linux_fstatfs_magic",
        lambda _descriptor: filesystem_magic,
    )

    with pytest.raises(RestrictedAccessError, match="not an allowed local classic model"):
        restricted_access_module._linux_local_filesystem_identity(7)


def test_posix_filesystem_descriptor_binds_allowed_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        restricted_access_module,
        "_linux_fstatfs_magic",
        lambda _descriptor: 0x0000EF53,
    )

    assert restricted_access_module._linux_local_filesystem_identity(7) == (
        "ext2_ext3_ext4",
        "0x0000ef53",
    )


@pytest.mark.parametrize(
    "acl_xattr",
    (
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.richacl",
        "system.nfs4_acl",
        "security.NFSv4ACL",
        "user.acl_note",
    ),
)
def test_posix_permission_descriptor_rejects_every_acl_related_xattr(
    monkeypatch: pytest.MonkeyPatch,
    acl_xattr: str,
) -> None:
    status = cast(
        os.stat_result,
        cast(object, SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=_CURRENT_UID)),
    )
    monkeypatch.setattr(
        restricted_access_module,
        "_linux_local_filesystem_identity",
        lambda _descriptor: ("ext2_ext3_ext4", "0x0000ef53"),
    )
    monkeypatch.setattr(os, "listxattr", lambda _descriptor: [acl_xattr], raising=False)

    with pytest.raises(RestrictedAccessError, match="ACL-related xattr"):
        restricted_access_module._permission_descriptor(
            7,
            platform="posix",
            current_principal=f"uid:{_CURRENT_UID}",
            path_kind="regular_file",
            status=status,
        )


@pytest.mark.parametrize(
    ("platform", "mutation"),
    (
        ("posix", "schema_float"),
        ("posix", "owner_float"),
        ("posix", "owner_bool"),
        ("posix", "owner_out_of_range"),
        ("posix", "link_count_out_of_range"),
        ("windows", "ace_count_float"),
        ("windows", "ace_count_out_of_range"),
        ("windows", "inherited_count_bool"),
        ("windows", "revision_bool"),
        ("windows", "control_out_of_range"),
        ("windows", "flags_float"),
        ("windows", "flags_out_of_range"),
        ("windows", "access_mask_float"),
        ("windows", "access_mask_out_of_range"),
    ),
)
def test_from_mapping_rejects_bool_and_float_integer_forgery(
    platform: str,
    mutation: str,
) -> None:
    mapping = deepcopy(_evidence(platform=platform).as_mapping())
    observation = _first_observation(mapping)
    descriptor = cast(dict[str, object], observation["permission_descriptor"])
    if mutation == "schema_float":
        mapping["schema_version"] = 3.0
    elif mutation == "owner_float":
        descriptor["owner_uid"] = float(_CURRENT_UID)
    elif mutation == "owner_bool":
        descriptor["owner_uid"] = True
    elif mutation == "owner_out_of_range":
        descriptor["owner_uid"] = 0x100000000
    elif mutation == "link_count_out_of_range":
        observation["link_count"] = 0x10000000000000000
    elif mutation == "ace_count_float":
        descriptor["ace_count"] = 1.0
    elif mutation == "ace_count_out_of_range":
        descriptor["ace_count"] = 0x10000
    elif mutation == "inherited_count_bool":
        descriptor["inherited_ace_count"] = False
    elif mutation == "revision_bool":
        descriptor["security_descriptor_revision"] = True
    elif mutation == "control_out_of_range":
        descriptor["security_descriptor_control"] = 0x10000
    else:
        aces = cast(list[dict[str, object]], descriptor["aces_in_native_order"])
        if mutation == "flags_float":
            aces[0]["ace_flags"] = 0.0
        elif mutation == "flags_out_of_range":
            aces[0]["ace_flags"] = 0x100
        elif mutation == "access_mask_out_of_range":
            aces[0]["access_mask"] = 0x100000000
        else:
            aces[0]["access_mask"] = float(_WINDOWS_FULL_CONTROL_MASK)

    with pytest.raises(RestrictedAccessError):
        RestrictedLocalArtifactAccessControlEvidence.from_mapping(_rehash(mapping))


def test_fdopen_failure_closes_transferred_raw_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_descriptor, write_descriptor = os.pipe()

    def fail_fdopen(_descriptor: int, _mode: str) -> Any:
        raise OSError("synthetic fdopen failure")

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    try:
        with pytest.raises(OSError, match="synthetic fdopen failure"):
            restricted_access_module._fdopen_binary_owned(read_descriptor)
        with pytest.raises(OSError):
            os.fstat(read_descriptor)
    finally:
        for descriptor in (read_descriptor, write_descriptor):
            with suppress(OSError):
                os.close(descriptor)
