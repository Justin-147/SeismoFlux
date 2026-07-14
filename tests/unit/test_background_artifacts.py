from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import cast

import pytest

from seismoflux.background.artifacts import (
    ArtifactConflictError,
    ArtifactFile,
    ArtifactPublicationError,
    ProjectRelativePath,
    canonical_json_bytes,
    content_address_id,
    publish_artifact,
)


def _inputs() -> dict[str, object]:
    lock_hash = "c" * 64
    return {
        "protocol": {"version": "0.2.0"},
        "input_hashes": {
            "environment_lock": lock_hash,
            "data_catalog": "a" * 64,
            "earthquake_dataset": "b" * 64,
            "study_area": "d" * 64,
            "issue_manifest": "e" * 64,
            "production_fixture": "f" * 64,
            "oracle_metadata": "1" * 64,
        },
        "model_parameters": {"bandwidth_km": 75.0},
        "code_commit": "b" * 40,
        "uv_lock_sha256": lock_hash,
    }


def _files(payload: bytes = b"metrics") -> tuple[ArtifactFile, ...]:
    return (
        ArtifactFile(
            ProjectRelativePath(r"nested\metrics.bin"),
            payload,
            "application/octet-stream",
        ),
        ArtifactFile(ProjectRelativePath("说明.txt"), "通过".encode(), "text/plain"),
    )


def test_canonical_json_v1_has_exact_nfc_float_and_path_bytes() -> None:
    value = {
        "e\u0301": "e\u0301",
        "path": ProjectRelativePath(r"nested\.\file.txt"),
        "a": 1.5,
        "zero": -0.0,
    }

    payload = canonical_json_bytes(value)

    assert (
        payload
        == (
            '{"a":{"$seismoflux_type":"float","hex":"0x1.8000000000000p+0"},'
            '"path":"nested/file.txt","zero":{"$seismoflux_type":"float",'
            '"hex":"-0x0.0p+0"},"é":"é"}'
        ).encode()
    )
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert not payload.endswith(b"\n")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_nonfinite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"value": value})


def test_canonical_json_rejects_nfc_key_collisions_and_reserved_type_key() -> None:
    with pytest.raises(ValueError, match="collide"):
        canonical_json_bytes({"é": 1, "e\u0301": 2})
    with pytest.raises(ValueError, match="reserved"):
        canonical_json_bytes({"$seismoflux_type": "other", "hex": "0x0.0p+0"})
    with pytest.raises(ValueError, match="reserved"):
        canonical_json_bytes({"$seismoflux_type": "float", "hex": "0x1.8000000000000p+0"})


def test_project_paths_require_explicit_safe_wrapper() -> None:
    with pytest.raises(TypeError, match="explicitly wrapped"):
        canonical_json_bytes({"path": Path("relative/file.txt")})
    with pytest.raises(ValueError, match="POSIX-relative"):
        ProjectRelativePath("../outside.txt")
    with pytest.raises(ValueError, match="POSIX-relative"):
        ProjectRelativePath("C:/outside.txt")
    with pytest.raises(ValueError, match="POSIX-relative"):
        ProjectRelativePath("/outside.txt")


def test_content_address_is_first_16_hex_of_canonical_sha256() -> None:
    inputs = _inputs()
    expected = hashlib.sha256(canonical_json_bytes(inputs)).hexdigest()

    assert content_address_id(inputs) == expected[:16]
    assert len(content_address_id(inputs)) == 16


def test_content_address_rejects_missing_or_malformed_required_inputs() -> None:
    missing = _inputs()
    del missing["code_commit"]
    with pytest.raises(ValueError, match="complete frozen key set"):
        content_address_id(missing)

    missing_hash = _inputs()
    input_hashes = cast(dict[str, str], missing_hash["input_hashes"])
    del input_hashes["earthquake_dataset"]
    with pytest.raises(ValueError, match="complete frozen key set"):
        content_address_id(missing_hash)

    wrong_lock = _inputs()
    wrong_lock["uv_lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="environment_lock"):
        content_address_id(wrong_lock)

    dirty_commit = _inputs()
    dirty_commit["code_commit"] = "not-a-commit"
    with pytest.raises(ValueError, match="Git object ID"):
        content_address_id(dirty_commit)

    extra = _inputs()
    extra["unexpected"] = "value"
    with pytest.raises(ValueError, match="complete frozen key set"):
        content_address_id(extra)

    extra_hash = _inputs()
    cast(dict[str, str], extra_hash["input_hashes"])["unexpected"] = "2" * 64
    with pytest.raises(ValueError, match="complete frozen key set"):
        content_address_id(extra_hash)

    uppercase_hash = _inputs()
    cast(dict[str, str], uppercase_hash["input_hashes"])["data_catalog"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        content_address_id(uppercase_hash)

    short_hash = _inputs()
    cast(dict[str, str], short_hash["input_hashes"])["data_catalog"] = "a" * 63
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        content_address_id(short_hash)

    non_mapping_protocol = _inputs()
    non_mapping_protocol["protocol"] = ["0.2.0"]
    with pytest.raises(TypeError, match="protocol must be"):
        content_address_id(non_mapping_protocol)

    non_mapping_parameters = _inputs()
    non_mapping_parameters["model_parameters"] = [75.0]
    with pytest.raises(TypeError, match="model_parameters must be"):
        content_address_id(non_mapping_parameters)


def test_publish_writes_sorted_manifest_and_verifies_without_rewrite(tmp_path: Path) -> None:
    publication = publish_artifact(tmp_path / "store", address_inputs=_inputs(), files=_files())

    assert publication.created is True
    assert publication.directory.name == content_address_id(_inputs())
    assert publication.manifest_path == publication.directory / "manifest.json"
    manifest_payload = publication.manifest_path.read_bytes()
    manifest = cast(dict[str, object], json.loads(manifest_payload))
    entries = cast(list[dict[str, object]], manifest["files"])
    assert [entry["relative_path"] for entry in entries] == ["nested/metrics.bin", "说明.txt"]
    assert all(
        set(entry) == {"relative_path", "byte_count", "sha256", "media_type"} for entry in entries
    )
    assert entries[0] == {
        "byte_count": 7,
        "media_type": "application/octet-stream",
        "relative_path": "nested/metrics.bin",
        "sha256": hashlib.sha256(b"metrics").hexdigest(),
    }
    assert publication.manifest_sha256 == hashlib.sha256(manifest_payload).hexdigest()
    assert manifest_payload == json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert not manifest_payload.endswith(b"\n")
    assert (publication.directory / "nested" / "metrics.bin").read_bytes() == b"metrics"

    fixed_ns = 1_700_000_000_000_000_000
    os.utime(publication.manifest_path, ns=(fixed_ns, fixed_ns))
    second = publish_artifact(tmp_path / "store", address_inputs=_inputs(), files=_files())

    assert second.created is False
    assert second.directory == publication.directory
    assert second.manifest_path.stat().st_mtime_ns == fixed_ns
    assert not list((tmp_path / "store").glob(".*.tmp"))
    assert not list((tmp_path / "store").glob(".*.lock"))


def test_same_id_with_different_payload_is_a_hard_conflict(tmp_path: Path) -> None:
    store = tmp_path / "store"
    first = publish_artifact(store, address_inputs=_inputs(), files=_files())
    original_manifest = first.manifest_path.read_bytes()

    with pytest.raises(ArtifactConflictError, match="manifest bytes differ"):
        publish_artifact(store, address_inputs=_inputs(), files=_files(b"different"))

    assert first.manifest_path.read_bytes() == original_manifest
    assert (first.directory / "nested" / "metrics.bin").read_bytes() == b"metrics"


def test_existing_payload_tampering_and_untracked_files_are_rejected(tmp_path: Path) -> None:
    store = tmp_path / "store"
    first = publish_artifact(store, address_inputs=_inputs(), files=_files())
    payload_path = first.directory / "nested" / "metrics.bin"
    payload_path.write_bytes(b"tampered")

    with pytest.raises(ArtifactConflictError, match="payload bytes differ"):
        publish_artifact(store, address_inputs=_inputs(), files=_files())

    payload_path.write_bytes(b"metrics")
    (first.directory / "extra.txt").write_bytes(b"extra")
    with pytest.raises(ArtifactConflictError, match="file set differs"):
        publish_artifact(store, address_inputs=_inputs(), files=_files())


def test_atomic_rename_failure_leaves_no_destination_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"

    def fail_rename(source: str | bytes | Path, destination: str | bytes | Path) -> None:
        del source, destination
        raise OSError("synthetic rename failure")

    monkeypatch.setattr("seismoflux.background.artifacts.os.rename", fail_rename)

    with pytest.raises(ArtifactPublicationError, match="atomically publish"):
        publish_artifact(store, address_inputs=_inputs(), files=_files())

    assert not (store / content_address_id(_inputs())).exists()
    assert not list(store.glob(".*.tmp"))
    assert not list(store.glob(".*.lock"))


def test_file_paths_are_unique_and_manifest_name_is_reserved() -> None:
    duplicate = ArtifactFile(ProjectRelativePath("A.txt"), b"one", "text/plain")
    case_duplicate = ArtifactFile(ProjectRelativePath("a.txt"), b"two", "text/plain")
    with pytest.raises(ValueError, match="case-insensitive"):
        publish_artifact(
            Path("unused"), address_inputs=_inputs(), files=[duplicate, case_duplicate]
        )

    reserved = ArtifactFile(ProjectRelativePath("manifest.json"), b"{}", "application/json")
    with pytest.raises(ValueError, match="reserved"):
        publish_artifact(Path("unused"), address_inputs=_inputs(), files=[reserved])


def test_file_directory_conflicts_are_rejected_case_insensitively() -> None:
    parent = ArtifactFile(ProjectRelativePath("A"), b"parent", "application/octet-stream")
    child = ArtifactFile(ProjectRelativePath("a/child.bin"), b"child", "application/octet-stream")

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        publish_artifact(Path("unused"), address_inputs=_inputs(), files=[parent, child])
