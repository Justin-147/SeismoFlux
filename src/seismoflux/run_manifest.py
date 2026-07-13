"""Machine-readable execution manifests shared by all CLI commands."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seismoflux.config import SeismoFluxConfig, project_root_for, sha256_file


def _git_metadata(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    dirty = bool(status_result.stdout) if status_result.returncode == 0 else None
    return commit, dirty


def build_run_manifest(
    *,
    command: str,
    dry_run: bool,
    implementation_stage: int,
    implementation_status: str,
    status: str,
    arguments: dict[str, Any],
    config_path: Path,
    config: SeismoFluxConfig,
    planned_inputs: list[str],
    planned_outputs: list[str],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a traceable manifest without exposing migration-only absolute paths."""

    project_root = project_root_for(config_path).resolve()
    resolved_config_path = config_path.resolve()
    try:
        recorded_config_path = resolved_config_path.relative_to(project_root).as_posix()
    except ValueError:
        recorded_config_path = resolved_config_path.name
    lock_path = project_root / "uv.lock"
    git_commit, git_worktree_dirty = _git_metadata(project_root)
    return {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "command": command,
        "mode": "dry_run" if dry_run else "execute",
        "implementation_stage": implementation_stage,
        "implementation_status": implementation_status,
        "status": status,
        "arguments": arguments,
        "config": {
            "path": recorded_config_path,
            "sha256": sha256_file(config_path),
            "random_seed": config.project.random_seed,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "git_commit": git_commit,
            "git_worktree_dirty": git_worktree_dirty,
            "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        },
        "resources": {
            "reserve_physical_cores": config.parallel.reserve_physical_cores,
            "max_workers": config.parallel.max_workers,
            "nested_parallelism": config.parallel.nested_parallelism,
        },
        "planned_inputs": planned_inputs,
        "planned_outputs": planned_outputs,
        "details": details or {},
    }


def emit_run_manifest(manifest: dict[str, Any], destination: str | None) -> None:
    """Print JSON to stdout and optionally persist an explicitly requested copy."""

    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(payload)
    if destination is None or destination == "-":
        return

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
        os.replace(temporary_name, output_path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
