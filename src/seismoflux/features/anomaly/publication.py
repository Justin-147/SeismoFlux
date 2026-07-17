"""Immutable local bundle publication for stage-3 anomaly features."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.config import sha256_file
from seismoflux.data.common import write_json_atomic

STAGE3_MANIFEST_FILENAME = "manifest.json"
_BUNDLE_PREFIX = "anomaly-feature-bundle-"
_BUNDLE_PATTERN = re.compile(r"anomaly-feature-bundle-[0-9a-f]{16}")
_LOCK_SCHEMA_VERSION = 1
_STALE_LOCK_GRACE_SECONDS = 60.0
_IS_WINDOWS = os.name == "nt"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_ERROR_INVALID_PARAMETER = 87
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258


class Stage3PublicationError(RuntimeError):
    """Raised when a stage-3 content-addressed bundle cannot be verified or published."""


@dataclass(frozen=True, slots=True)
class Stage3BundleWorkspace:
    """A verified existing bundle or a private staging directory to populate."""

    bundle_id: str
    path: Path
    destination: Path
    created: bool


@dataclass(frozen=True, slots=True)
class _WindowsProcessApi:
    open_process: Callable[[int, int, int], int | None]
    wait_for_single_object: Callable[[int, int], int]
    close_handle: Callable[[int], int]
    get_last_error: Callable[[], int]


def _safe_relative_path(value: str) -> str:
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if (
        not value
        or "\x00" in value
        or posix.is_absolute()
        or bool(windows.anchor)
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
    ):
        raise ValueError("stage-3 bundle paths must be safe relative POSIX paths")
    normalized = posix.as_posix()
    if normalized in {"", ".", STAGE3_MANIFEST_FILENAME}:
        raise ValueError("invalid or reserved stage-3 bundle path")
    return normalized


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _load_windows_process_api() -> _WindowsProcessApi | None:
    try:
        import ctypes
        from ctypes import wintypes

        win_dll = getattr(ctypes, "WinDLL", None)
        get_last_error = getattr(ctypes, "get_last_error", None)
        if win_dll is None or get_last_error is None:
            return None
        kernel32 = win_dll("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        wait_for_single_object.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
    except (AttributeError, OSError, ValueError):
        return None
    return _WindowsProcessApi(
        open_process=cast(Callable[[int, int, int], int | None], open_process),
        wait_for_single_object=cast(Callable[[int, int], int], wait_for_single_object),
        close_handle=cast(Callable[[int], int], close_handle),
        get_last_error=cast(Callable[[], int], get_last_error),
    )


def _windows_process_exists(pid: int) -> bool:
    """Probe a Windows PID without sending a signal or mutating the process."""

    api = _load_windows_process_api()
    if api is None:
        return True
    try:
        handle = api.open_process(
            _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE,
            0,
            pid,
        )
    except (OSError, ValueError):
        return True
    if not handle:
        return api.get_last_error() != _ERROR_INVALID_PARAMETER
    try:
        try:
            wait_status = api.wait_for_single_object(handle, 0)
        except (OSError, ValueError):
            return True
        return wait_status != _WAIT_OBJECT_0
    finally:
        api.close_handle(handle)


def _process_exists(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if _IS_WINDOWS:
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _stale_lock_can_be_reclaimed(lock_path: Path, *, bundle_id: str) -> bool:
    try:
        stat_before = lock_path.stat(follow_symlinks=False)
        payload_bytes = lock_path.read_bytes()
    except FileNotFoundError:
        return True
    if _is_reparse_point(lock_path):
        raise Stage3PublicationError("stage-3 publication lock is a reparse point")
    age_seconds = max(0.0, time.time() - stat_before.st_mtime)
    if age_seconds < _STALE_LOCK_GRACE_SECONDS:
        return False
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload != {
        "bundle_id": bundle_id,
        "pid": payload.get("pid"),
        "schema_version": _LOCK_SCHEMA_VERSION,
    }:
        return False
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or _process_exists(pid):
        return False
    try:
        stat_after = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True
    if (
        stat_after.st_ino != stat_before.st_ino
        or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        or stat_after.st_size != stat_before.st_size
        or lock_path.read_bytes() != payload_bytes
    ):
        return False
    with suppress(FileNotFoundError):
        lock_path.unlink()
    return True


def _acquire_publication_lock(lock_path: Path, *, bundle_id: str) -> tuple[int, bool]:
    reclaimed = False
    for _attempt in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            return descriptor, reclaimed
        except FileExistsError as exc:
            if reclaimed or not _stale_lock_can_be_reclaimed(lock_path, bundle_id=bundle_id):
                raise Stage3PublicationError(
                    f"stage-3 publication already in progress: {bundle_id}"
                ) from exc
            reclaimed = True
    raise AssertionError("stage-3 publication lock acquisition loop changed")


def _remove_stale_staging_directories(parent: Path, *, bundle_id: str) -> None:
    root = parent.resolve()
    prefix = f".{bundle_id}."
    for candidate in parent.iterdir():
        if not (candidate.name.startswith(prefix) and candidate.name.endswith(".tmp")):
            continue
        if _is_reparse_point(candidate) or not candidate.is_dir():
            raise Stage3PublicationError("stale stage-3 staging entry is unsafe")
        resolved = candidate.resolve()
        if resolved.parent != root:
            raise Stage3PublicationError("stale stage-3 staging directory escapes its root")
        shutil.rmtree(resolved)


def stage3_identity_sha256(identity: Mapping[str, object]) -> str:
    """Hash the complete, score-free semantic identity with canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def stage3_bundle_id(identity: Mapping[str, object]) -> str:
    """Return the public-safe prefix plus the first 16 identity digest characters."""

    return _BUNDLE_PREFIX + stage3_identity_sha256(identity)[:16]


def build_stage3_manifest(
    *,
    bundle_id: str,
    identity: Mapping[str, object],
    directory: Path,
    payload_paths: Sequence[str],
    datasets: Mapping[str, Mapping[str, object]],
    audit: Mapping[str, object],
) -> dict[str, object]:
    """Build a byte-bound manifest after every local payload has been written."""

    if _BUNDLE_PATTERN.fullmatch(bundle_id) is None:
        raise ValueError("invalid stage-3 bundle ID")
    identity_digest = stage3_identity_sha256(identity)
    if bundle_id != _BUNDLE_PREFIX + identity_digest[:16]:
        raise ValueError("stage-3 bundle ID does not match its identity")
    normalized_paths = tuple(sorted(_safe_relative_path(path) for path in payload_paths))
    if len(normalized_paths) != len(set(normalized_paths)) or not normalized_paths:
        raise ValueError("stage-3 bundle payload paths must be unique and non-empty")
    files: list[dict[str, object]] = []
    for relative_path in normalized_paths:
        path = directory.joinpath(*relative_path.split("/"))
        if not path.is_file() or _is_reparse_point(path):
            raise Stage3PublicationError(f"missing or unsafe stage-3 payload: {relative_path}")
        files.append(
            {
                "byte_count": path.stat().st_size,
                "relative_path": relative_path,
                "sha256": sha256_file(path),
            }
        )
    return {
        "audit": dict(audit),
        "bundle_id": bundle_id,
        "canonicalization": "seismoflux_canonical_json_v1",
        "datasets": {key: dict(value) for key, value in sorted(datasets.items())},
        "files": files,
        "identity": dict(identity),
        "identity_sha256": identity_digest,
        "locked_test": {
            "artifact_ids": [],
            "result": None,
            "run": False,
            "score_ids": [],
            "target_count": None,
            "target_ids": [],
        },
        "protocol_version": "0.3.0",
        "schema_version": 1,
    }


def write_stage3_manifest(directory: Path, manifest: Mapping[str, object]) -> Path:
    path = directory / STAGE3_MANIFEST_FILENAME
    if path.exists():
        raise Stage3PublicationError("stage-3 bundle manifest already exists in staging")
    write_json_atomic(path, dict(manifest))
    return path


def load_and_verify_stage3_bundle(
    directory: Path,
    *,
    expected_bundle_id: str | None = None,
    require_directory_name: bool = True,
) -> dict[str, object]:
    """Verify the exact file set, identity and bytes of one immutable bundle."""

    if not directory.is_dir() or _is_reparse_point(directory):
        raise Stage3PublicationError(f"stage-3 bundle is not a plain directory: {directory}")
    manifest_path = directory / STAGE3_MANIFEST_FILENAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3PublicationError("stage-3 bundle has no valid manifest") from exc
    if not isinstance(raw, dict):
        raise Stage3PublicationError("stage-3 bundle manifest must be a mapping")
    manifest = cast(dict[str, object], raw)
    if manifest.get("schema_version") != 1 or manifest.get("protocol_version") != "0.3.0":
        raise Stage3PublicationError("unsupported stage-3 bundle manifest version")
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or _BUNDLE_PATTERN.fullmatch(bundle_id) is None:
        raise Stage3PublicationError("invalid stage-3 bundle manifest ID")
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise Stage3PublicationError("stage-3 bundle ID differs from the expected identity")
    if require_directory_name and directory.name != bundle_id:
        raise Stage3PublicationError("stage-3 bundle directory name differs from its manifest")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise Stage3PublicationError("stage-3 bundle has no identity mapping")
    identity_digest = stage3_identity_sha256(cast(dict[str, object], identity))
    if manifest.get("identity_sha256") != identity_digest:
        raise Stage3PublicationError("stage-3 bundle identity digest mismatch")
    if bundle_id != _BUNDLE_PREFIX + identity_digest[:16]:
        raise Stage3PublicationError("stage-3 bundle address mismatch")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise Stage3PublicationError("stage-3 bundle file manifest is empty")
    expected_files = {STAGE3_MANIFEST_FILENAME}
    for entry in files:
        if not isinstance(entry, dict):
            raise Stage3PublicationError("invalid stage-3 bundle file entry")
        relative_path = _safe_relative_path(str(entry.get("relative_path", "")))
        if relative_path in expected_files:
            raise Stage3PublicationError("duplicate stage-3 bundle file entry")
        expected_files.add(relative_path)
        path = directory.joinpath(*relative_path.split("/"))
        if not path.is_file() or _is_reparse_point(path):
            raise Stage3PublicationError(f"missing or unsafe stage-3 payload: {relative_path}")
        if path.stat().st_size != entry.get("byte_count") or sha256_file(path) != entry.get(
            "sha256"
        ):
            raise Stage3PublicationError(f"stage-3 payload hash mismatch: {relative_path}")

    actual_files: set[str] = set()
    for child in directory.rglob("*"):
        if _is_reparse_point(child):
            raise Stage3PublicationError("stage-3 bundle contains a reparse point")
        if child.is_file():
            actual_files.add(child.relative_to(directory).as_posix())
        elif not child.is_dir():
            raise Stage3PublicationError("stage-3 bundle contains an unsupported entry")
    if actual_files != expected_files:
        raise Stage3PublicationError("stage-3 bundle file set differs from its manifest")
    return manifest


@contextmanager
def stage3_bundle_workspace(
    parent: Path,
    *,
    bundle_id: str,
) -> Iterator[Stage3BundleWorkspace]:
    """Yield an existing verified bundle or atomically publish a populated staging directory."""

    if _BUNDLE_PATTERN.fullmatch(bundle_id) is None:
        raise ValueError("invalid stage-3 bundle ID")
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / bundle_id
    if destination.exists():
        load_and_verify_stage3_bundle(destination, expected_bundle_id=bundle_id)
        yield Stage3BundleWorkspace(bundle_id, destination, destination, False)
        return

    lock_path = parent / f".{bundle_id}.lock"
    descriptor, reclaimed = _acquire_publication_lock(lock_path, bundle_id=bundle_id)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            lock_payload = {
                "bundle_id": bundle_id,
                "pid": os.getpid(),
                "schema_version": _LOCK_SCHEMA_VERSION,
            }
            handle.write(canonical_json_bytes(lock_payload))
            handle.flush()
            os.fsync(handle.fileno())
        if reclaimed:
            _remove_stale_staging_directories(parent, bundle_id=bundle_id)
        if destination.exists():
            load_and_verify_stage3_bundle(destination, expected_bundle_id=bundle_id)
            yield Stage3BundleWorkspace(bundle_id, destination, destination, False)
            return
        staging = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", suffix=".tmp", dir=parent))
        published = False
        try:
            yield Stage3BundleWorkspace(bundle_id, staging, destination, True)
            if not (staging / STAGE3_MANIFEST_FILENAME).is_file():
                raise Stage3PublicationError("stage-3 staging completed without a manifest")
            load_and_verify_stage3_bundle(
                staging,
                expected_bundle_id=bundle_id,
                require_directory_name=False,
            )
            temporary_named = staging.with_name(bundle_id)
            if temporary_named != destination:
                raise AssertionError("stage-3 destination derivation changed")
            os.rename(staging, destination)
            published = True
            load_and_verify_stage3_bundle(destination, expected_bundle_id=bundle_id)
        finally:
            if not published:
                shutil.rmtree(staging, ignore_errors=True)
    finally:
        lock_path.unlink(missing_ok=True)


__all__ = [
    "STAGE3_MANIFEST_FILENAME",
    "Stage3BundleWorkspace",
    "Stage3PublicationError",
    "build_stage3_manifest",
    "load_and_verify_stage3_bundle",
    "stage3_bundle_id",
    "stage3_bundle_workspace",
    "stage3_identity_sha256",
    "write_stage3_manifest",
]
