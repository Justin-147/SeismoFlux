"""Lexical path guard for every target-blind stage-4 CLI input and output."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


class ScoreBlindPathError(ValueError):
    """Raised before filesystem I/O when a pre-target path is unsafe."""


def _target_relative_path(protocol: Mapping[str, object]) -> PurePosixPath:
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ScoreBlindPathError("protocol inputs are missing")
    target = inputs.get("earthquake_target")
    if not isinstance(target, Mapping):
        raise ScoreBlindPathError("protocol earthquake target identity is missing")
    raw = target.get("path")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ScoreBlindPathError("protocol earthquake target path is invalid")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
        raise ScoreBlindPathError("protocol earthquake target path is not normalized")
    return relative


def require_score_blind_project_path(
    project_root: Path,
    protocol: Mapping[str, object],
    path: Path,
    *,
    label: str,
) -> Path:
    """Reject project escapes, target aliases, links, and reparse traversal before I/O.

    The exact frozen target is compared lexically before any candidate-path stat.
    Existing candidate components are then inspected with ``lstat`` only: symbolic
    links and Windows reparse points are rejected without following them.  A final
    regular file with multiple hard links is rejected as an unverifiable alias.
    """

    root = Path(project_root).resolve()
    candidate = Path(os.path.abspath(os.fspath(path)))
    if not candidate.is_relative_to(root):
        raise ScoreBlindPathError(f"{label} must stay inside the project root")
    target = root.joinpath(*_target_relative_path(protocol).parts)
    normalized_candidate = os.path.normcase(os.fspath(candidate))
    normalized_target = os.path.normcase(os.fspath(target))
    if normalized_candidate == normalized_target:
        raise ScoreBlindPathError(
            f"{label} is the frozen earthquake target and is forbidden before authorization"
        )

    relative_parts = candidate.relative_to(root).parts
    cursor = root
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for index, part in enumerate(relative_parts):
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ScoreBlindPathError(
                f"{label} cannot be safely inspected without following links"
            ) from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag:
            raise ScoreBlindPathError(
                f"{label} must not traverse a symbolic link or junction before authorization"
            )
        is_final = index == len(relative_parts) - 1
        if is_final and stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
            raise ScoreBlindPathError(
                f"{label} must not use a multiply-linked file before authorization"
            )
    return candidate


__all__ = ["ScoreBlindPathError", "require_score_blind_project_path"]
