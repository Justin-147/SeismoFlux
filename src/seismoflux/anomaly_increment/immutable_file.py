"""Race-resistant reads for immutable, create-only governance artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import Final, cast

_FILE_ATTRIBUTE_REPARSE_POINT: Final[int] = 0x0400


class UnsafeImmutableFileError(RuntimeError):
    """Raised when an existing governance artifact is not safe to replay."""


@dataclass(frozen=True, slots=True)
class ImmutableFileSnapshot:
    """Identity and state bound to one verified ordinary file."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _state(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = cast(int, getattr(value, "st_file_attributes", 0))
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _require_regular_single_link(value: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(value.st_mode) or _is_reparse_point(value) or value.st_nlink != 1:
        raise UnsafeImmutableFileError(f"{label} must be a regular, non-reparse, single-link file")


def require_existing_real_directory(path: Path, *, label: str) -> os.stat_result:
    """Reject a missing, linked, or Windows-reparse directory entry."""

    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise UnsafeImmutableFileError(f"cannot inspect {label}") from exc
    if not stat.S_ISDIR(observed.st_mode) or _is_reparse_point(observed):
        raise UnsafeImmutableFileError(f"{label} must be a real, non-reparse directory")
    return observed


def require_existing_real_directory_tree(
    root: Path,
    directory: Path,
    *,
    label: str,
) -> None:
    """Reject link/reparse traversal in every existing directory component."""

    anchor = Path(root).resolve()
    candidate = Path(os.path.abspath(os.fspath(directory)))
    if not candidate.is_relative_to(anchor):
        raise UnsafeImmutableFileError(f"{label} escapes its trusted root")
    require_existing_real_directory(anchor, label=f"{label} root")
    cursor = anchor
    for part in candidate.relative_to(anchor).parts:
        cursor /= part
        require_existing_real_directory(cursor, label=label)


def ensure_real_directory_tree(
    root: Path,
    directory: Path,
    *,
    label: str,
) -> None:
    """Create missing directories while rejecting every linked/reparse component."""

    anchor = Path(root).resolve()
    candidate = Path(os.path.abspath(os.fspath(directory)))
    if not candidate.is_relative_to(anchor):
        raise UnsafeImmutableFileError(f"{label} escapes its trusted root")
    require_existing_real_directory(anchor, label=f"{label} root")
    cursor = anchor
    for part in candidate.relative_to(anchor).parts:
        cursor /= part
        try:
            os.mkdir(cursor)
        except FileExistsError:
            require_existing_real_directory(cursor, label=label)


if sys.platform == "win32":

    def _open_windows_reparse_point(path: Path, flags: int = os.O_RDONLY) -> int:
        """Open the final path component itself, never a Windows reparse target."""

        import ctypes
        import msvcrt
        from ctypes import wintypes

        generic_read = 0x80000000
        generic_write = 0x40000000
        file_share_read = 0x00000001
        open_existing = 3
        file_attribute_normal = 0x00000080
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
            generic_read | (generic_write if (flags & os.O_RDWR) or (flags & os.O_WRONLY) else 0),
            file_share_read,
            None,
            open_existing,
            file_attribute_normal | file_flag_open_reparse_point,
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        raw_handle = cast(int, handle)
        try:
            return msvcrt.open_osfhandle(raw_handle, flags | os.O_BINARY)
        except Exception:
            close_handle(handle)
            raise

else:

    def _open_windows_reparse_point(path: Path, flags: int = os.O_RDONLY) -> int:
        """Fail closed if a Windows-only primitive is reached on another platform."""

        raise UnsafeImmutableFileError(
            f"Windows reparse-point open is unavailable for {path} with flags {flags}"
        )


def _open_no_follow(path: Path, flags: int = os.O_RDONLY) -> int:
    if sys.platform == "win32":
        return _open_windows_reparse_point(path, flags)
    else:
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if not isinstance(no_follow, int):
            raise UnsafeImmutableFileError("platform has no race-safe no-follow open")
        close_on_exec = cast(int, getattr(os, "O_CLOEXEC", 0))
        return os.open(path, flags | no_follow | close_on_exec)


def open_existing_single_link_descriptor(
    path: Path,
    *,
    flags: int,
    label: str,
) -> int:
    """Open an existing ordinary single-link file without following aliases."""

    target = Path(path)
    descriptor: int | None = None
    try:
        before = os.lstat(target)
        _require_regular_single_link(before, label=label)
        descriptor = _open_no_follow(target, flags)
        opened = os.fstat(descriptor)
        _require_regular_single_link(opened, label=label)
        after = os.lstat(target)
        _require_regular_single_link(after, label=label)
        if (
            _identity(opened) != _identity(before)
            or _identity(after) != _identity(opened)
            or _state(opened) != _state(before)
            or _state(after) != _state(opened)
        ):
            raise UnsafeImmutableFileError(f"{label} changed during no-follow open")
    except UnsafeImmutableFileError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise UnsafeImmutableFileError(f"cannot safely open {label}") from exc
    assert descriptor is not None
    return descriptor


def verify_opened_single_link_descriptor(
    path: Path,
    descriptor: int,
    *,
    label: str,
) -> None:
    """Bind an already-open descriptor to its current safe directory entry.

    This is used for process locks after the operating-system lock has been
    acquired.  Reopening a write-capable locked file is not portable on Windows,
    so verification deliberately inspects the retained descriptor and the path
    entry without reading or reopening either file.
    """

    try:
        opened = os.fstat(descriptor)
        _require_regular_single_link(opened, label=label)
        observed = os.lstat(path)
        _require_regular_single_link(observed, label=label)
        if _identity(observed) != _identity(opened) or _state(observed) != _state(opened):
            raise UnsafeImmutableFileError(f"{label} changed after its no-follow open")
    except UnsafeImmutableFileError:
        raise
    except OSError as exc:
        raise UnsafeImmutableFileError(f"cannot safely verify {label}") from exc


@contextmanager
def open_existing_immutable_file(
    path: Path,
    *,
    label: str,
) -> Iterator[BufferedReader]:
    """Yield a stable seekable reader while retaining no-follow identity checks."""

    target = Path(path)
    handle: BufferedReader | None = None
    try:
        before = os.lstat(target)
        _require_regular_single_link(before, label=label)
        descriptor = _open_no_follow(target)
        handle = os.fdopen(descriptor, "rb")
        opened = os.fstat(handle.fileno())
        _require_regular_single_link(opened, label=label)
        if _identity(opened) != _identity(before) or _state(opened) != _state(before):
            raise UnsafeImmutableFileError(f"{label} changed between lstat and no-follow open")
        yield handle
        after_read = os.fstat(handle.fileno())
        _require_regular_single_link(after_read, label=label)
        if _identity(after_read) != _identity(opened) or _state(after_read) != _state(opened):
            raise UnsafeImmutableFileError(f"{label} changed while being read")
        after_path = os.lstat(target)
        _require_regular_single_link(after_path, label=label)
        if _identity(after_path) != _identity(after_read) or _state(after_path) != _state(
            after_read
        ):
            raise UnsafeImmutableFileError(f"{label} path changed after it was read")
    except UnsafeImmutableFileError:
        raise
    except OSError as exc:
        raise UnsafeImmutableFileError(f"cannot safely open {label}") from exc
    finally:
        if handle is not None:
            handle.close()


def sha256_existing_immutable_file(path: Path, *, label: str) -> str:
    """Hash one potentially large file through a stable no-follow handle."""

    digest = hashlib.sha256()
    with open_existing_immutable_file(path, label=label) as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_existing_immutable_file(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, ImmutableFileSnapshot]:
    """Read and snapshot one stable file without following its final link.

    The initial ``lstat`` rejects links before opening.  The no-follow handle,
    inode comparison, and post-read checks close the replacement window and
    fail closed if the directory entry or file metadata changes while read.
    """

    target = Path(path)
    try:
        before = os.lstat(target)
        _require_regular_single_link(before, label=label)
        descriptor = _open_no_follow(target)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            _require_regular_single_link(opened, label=label)
            if _identity(opened) != _identity(before) or _state(opened) != _state(before):
                raise UnsafeImmutableFileError(f"{label} changed between lstat and no-follow open")
            payload = handle.read()
            after_read = os.fstat(handle.fileno())
            _require_regular_single_link(after_read, label=label)
            if _identity(after_read) != _identity(opened) or _state(after_read) != _state(opened):
                raise UnsafeImmutableFileError(f"{label} changed while being read")
        after_path = os.lstat(target)
        _require_regular_single_link(after_path, label=label)
        if _identity(after_path) != _identity(after_read) or _state(after_path) != _state(
            after_read
        ):
            raise UnsafeImmutableFileError(f"{label} path changed after it was read")
    except UnsafeImmutableFileError:
        raise
    except OSError as exc:
        raise UnsafeImmutableFileError(f"cannot safely read {label}") from exc
    return payload, ImmutableFileSnapshot(
        device=after_path.st_dev,
        inode=after_path.st_ino,
        size=after_path.st_size,
        mtime_ns=after_path.st_mtime_ns,
        ctime_ns=after_path.st_ctime_ns,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def read_existing_immutable_bytes(path: Path, *, label: str) -> bytes:
    """Read one stable ordinary file without following its final link."""

    payload, _ = read_existing_immutable_file(path, label=label)
    return payload


def unlink_existing_immutable_file(
    path: Path,
    *,
    expected: ImmutableFileSnapshot,
    label: str,
) -> None:
    """Remove only the same ordinary file previously captured by this process."""

    _, observed = read_existing_immutable_file(path, label=label)
    if observed != expected:
        raise UnsafeImmutableFileError(f"{label} changed before rollback")
    try:
        os.unlink(path)
    except OSError as exc:
        raise UnsafeImmutableFileError(f"cannot roll back {label}") from exc


__all__ = [
    "ImmutableFileSnapshot",
    "UnsafeImmutableFileError",
    "ensure_real_directory_tree",
    "open_existing_immutable_file",
    "open_existing_single_link_descriptor",
    "read_existing_immutable_bytes",
    "read_existing_immutable_file",
    "require_existing_real_directory",
    "require_existing_real_directory_tree",
    "sha256_existing_immutable_file",
    "unlink_existing_immutable_file",
]
