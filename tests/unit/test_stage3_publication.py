from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from seismoflux.features.anomaly import publication as publication_module
from seismoflux.features.anomaly.publication import (
    Stage3PublicationError,
    build_stage3_manifest,
    load_and_verify_stage3_bundle,
    stage3_bundle_id,
    stage3_bundle_workspace,
    write_stage3_manifest,
)


def _identity() -> dict[str, object]:
    return {
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "data_catalog_sha256": "c" * 64,
        "protocol_version": "0.3.0",
    }


def _populate(path: Path, bundle_id: str, identity: dict[str, object]) -> None:
    (path / "payload.bin").write_bytes(b"stage-3")
    manifest = build_stage3_manifest(
        bundle_id=bundle_id,
        identity=identity,
        directory=path,
        payload_paths=["payload.bin"],
        datasets={},
        audit={"future_source_reference_count": 0},
    )
    write_stage3_manifest(path, manifest)


@pytest.mark.parametrize(
    ("last_error", "expected"),
    [
        (publication_module._ERROR_INVALID_PARAMETER, False),
        (5, True),
    ],
)
def test_windows_process_probe_treats_invalid_pid_as_dead_and_denied_as_live(
    monkeypatch: pytest.MonkeyPatch,
    last_error: int,
    expected: bool,
) -> None:
    api = publication_module._WindowsProcessApi(
        open_process=lambda _access, _inherit, _pid: None,
        wait_for_single_object=lambda _handle, _timeout: pytest.fail("no handle to wait on"),
        close_handle=lambda _handle: pytest.fail("no handle to close"),
        get_last_error=lambda: last_error,
    )
    monkeypatch.setattr(publication_module, "_load_windows_process_api", lambda: api)

    assert publication_module._windows_process_exists(123) is expected


@pytest.mark.parametrize(
    ("wait_status", "expected"),
    [
        (publication_module._WAIT_OBJECT_0, False),
        (publication_module._WAIT_TIMEOUT, True),
        (0xFFFFFFFF, True),
    ],
)
def test_windows_process_probe_always_closes_open_handle(
    monkeypatch: pytest.MonkeyPatch,
    wait_status: int,
    expected: bool,
) -> None:
    closed: list[int] = []

    def close_handle(handle: int) -> int:
        closed.append(handle)
        return 1

    api = publication_module._WindowsProcessApi(
        open_process=lambda _access, _inherit, _pid: 42,
        wait_for_single_object=lambda _handle, _timeout: wait_status,
        close_handle=close_handle,
        get_last_error=lambda: 0,
    )
    monkeypatch.setattr(publication_module, "_load_windows_process_api", lambda: api)

    assert publication_module._windows_process_exists(123) is expected
    assert closed == [42]


def test_process_exists_windows_branch_never_sends_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[int] = []

    def forbidden_kill(_pid: int, _signal: int) -> None:
        raise AssertionError("Windows process probing must not call os.kill")

    def windows_probe(pid: int) -> bool:
        probed.append(pid)
        return True

    monkeypatch.setattr(publication_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(publication_module, "_windows_process_exists", windows_probe)
    monkeypatch.setattr(os, "kill", forbidden_kill)

    assert publication_module._process_exists(123) is True
    assert probed == [123]


def test_process_exists_posix_branch_uses_signal_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    def record_signal(pid: int, signal: int) -> None:
        signals.append((pid, signal))

    monkeypatch.setattr(publication_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(os, "kill", record_signal)

    assert publication_module._process_exists(123) is True
    assert signals == [(123, 0)]


@pytest.mark.skipif(os.name != "nt", reason="native Windows process API test")
def test_native_windows_process_probe_is_non_destructive_for_current_process() -> None:
    assert publication_module._windows_process_exists(os.getpid()) is True


def test_stage3_bundle_is_atomically_published_and_reused(tmp_path: Path) -> None:
    identity = _identity()
    bundle_id = stage3_bundle_id(identity)

    with stage3_bundle_workspace(tmp_path, bundle_id=bundle_id) as workspace:
        assert workspace.created is True
        _populate(workspace.path, bundle_id, identity)

    destination = tmp_path / bundle_id
    manifest = load_and_verify_stage3_bundle(destination, expected_bundle_id=bundle_id)
    assert manifest["bundle_id"] == bundle_id

    with stage3_bundle_workspace(tmp_path, bundle_id=bundle_id) as workspace:
        assert workspace.created is False
        assert workspace.path == destination


def test_stage3_bundle_tampering_is_detected(tmp_path: Path) -> None:
    identity = _identity()
    bundle_id = stage3_bundle_id(identity)
    with stage3_bundle_workspace(tmp_path, bundle_id=bundle_id) as workspace:
        _populate(workspace.path, bundle_id, identity)
    (tmp_path / bundle_id / "payload.bin").write_bytes(b"changed")

    with pytest.raises(Stage3PublicationError, match="payload hash mismatch"):
        load_and_verify_stage3_bundle(tmp_path / bundle_id, expected_bundle_id=bundle_id)


def test_failed_stage3_bundle_build_leaves_no_destination_or_lock(tmp_path: Path) -> None:
    identity = _identity()
    bundle_id = stage3_bundle_id(identity)

    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        stage3_bundle_workspace(tmp_path, bundle_id=bundle_id),
    ):
        raise RuntimeError("synthetic failure")

    assert not (tmp_path / bundle_id).exists()
    assert not (tmp_path / f".{bundle_id}.lock").exists()
    assert not list(tmp_path.glob(f".{bundle_id}.*.tmp"))


def test_stale_dead_process_lock_and_staging_are_safely_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    bundle_id = stage3_bundle_id(identity)
    lock_path = tmp_path / f".{bundle_id}.lock"
    lock_path.write_text(
        json.dumps({"bundle_id": bundle_id, "pid": 999_999, "schema_version": 1}),
        encoding="utf-8",
    )
    old = time.time() - 120.0
    os.utime(lock_path, (old, old))
    orphan = tmp_path / f".{bundle_id}.orphan.tmp"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"partial")
    monkeypatch.setattr(publication_module, "_process_exists", lambda _pid: False)

    with stage3_bundle_workspace(tmp_path, bundle_id=bundle_id) as workspace:
        assert not orphan.exists()
        _populate(workspace.path, bundle_id, identity)

    assert (tmp_path / bundle_id / "manifest.json").is_file()
    assert not lock_path.exists()


def test_live_process_lock_is_never_reclaimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_id = stage3_bundle_id(_identity())
    lock_path = tmp_path / f".{bundle_id}.lock"
    lock_path.write_text(
        json.dumps({"bundle_id": bundle_id, "pid": 123, "schema_version": 1}),
        encoding="utf-8",
    )
    old = time.time() - 120.0
    os.utime(lock_path, (old, old))
    monkeypatch.setattr(publication_module, "_process_exists", lambda _pid: True)

    with (
        pytest.raises(Stage3PublicationError, match="already in progress"),
        stage3_bundle_workspace(tmp_path, bundle_id=bundle_id),
    ):
        raise AssertionError("live lock must prevent entry")

    assert lock_path.exists()
