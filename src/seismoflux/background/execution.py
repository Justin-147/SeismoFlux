"""Scoring-free execution preflight for the frozen stage-2 background protocol.

This module deliberately contains no catalog loading, fitting, scoring, or artifact
publication.  It establishes the repository identity that may later be bound to
scientific artifacts, materializes a machine-readable execution plan from validated
configuration objects, and provides the sole constructor for background content-address
inputs.
"""

from __future__ import annotations

import copy
import hashlib
import os
import platform
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from seismoflux.background.artifacts import canonical_json_bytes, content_address_id
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.workflow import build_snapshot_definitions
from seismoflux.config import SeismoFluxConfig, sha256_file

_GIT_OID_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BACKGROUND_MODELS = ("uniform_poisson", "spatial_poisson", "etas")
_EXECUTION_INPUT_KEYS = (
    "data_catalog",
    "earthquake_dataset",
    "environment_lock",
    "issue_manifest",
    "oracle_metadata",
    "production_fixture",
    "study_area",
)


class RepositoryIdentityError(RuntimeError):
    """The repository cannot provide a safe, pushed scoring identity."""


class ExecutionSealError(RuntimeError):
    """The scoring inputs or repository identity changed across a run boundary."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Minimal subprocess result used by the injectable Git command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


GitCommandRunner: TypeAlias = Callable[[tuple[str, ...], Path], CommandResult]
PhysicalCoreProbe: TypeAlias = Callable[[], int | None]


def subprocess_git_runner(command: tuple[str, ...], cwd: Path) -> CommandResult:
    """Run one argument-vector Git command without a shell."""

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepositoryIdentityError("Git is unavailable for repository identity") from exc
    return CommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _git(
    runner: GitCommandRunner,
    project_root: Path,
    *arguments: str,
    description: str,
) -> CommandResult:
    command = ("git", *arguments)
    try:
        result = runner(command, project_root)
    except RepositoryIdentityError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepositoryIdentityError("Git is unavailable for repository identity") from exc
    if not isinstance(result, CommandResult):
        raise TypeError("Git command runners must return CommandResult")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise RepositoryIdentityError(f"unable to {description}{suffix}")
    return result


def _validated_oid(value: str, *, description: str) -> str:
    candidate = value.strip()
    if _GIT_OID_PATTERN.fullmatch(candidate) is None:
        raise RepositoryIdentityError(f"{description} must be a lowercase Git object ID")
    return candidate


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    """A local, clean, pushed commit proven to descend from the protocol tag."""

    code_commit: str
    branch: str
    upstream: str
    upstream_commit: str
    freeze_tag: str
    freeze_tag_commit: str
    git_available: bool
    worktree_clean: bool
    tag_is_ancestor: bool
    upstream_matches_head: bool

    def __post_init__(self) -> None:
        for name in ("code_commit", "upstream_commit", "freeze_tag_commit"):
            value = getattr(self, name)
            if _GIT_OID_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase Git object ID")
        for name in ("branch", "upstream", "freeze_tag"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string")
        for name in (
            "git_available",
            "worktree_clean",
            "tag_is_ancestor",
            "upstream_matches_head",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

    @property
    def ready(self) -> bool:
        """Whether every repository identity gate is satisfied."""

        return (
            self.git_available
            and self.worktree_clean
            and self.tag_is_ancestor
            and self.upstream_matches_head
            and self.upstream_commit == self.code_commit
        )


@dataclass(frozen=True, slots=True)
class ExecutionSeal:
    """A clean pushed Git identity bound to all seven frozen input bytes."""

    repository: RepositoryIdentity
    protocol_sha256: str
    input_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.repository.ready:
            raise ValueError("execution seal repository identity must be scoring-ready")
        if _SHA256_PATTERN.fullmatch(self.protocol_sha256) is None:
            raise ValueError("execution seal protocol fingerprint must be SHA-256")
        observed_keys = tuple(key for key, _ in self.input_hashes)
        if observed_keys != _EXECUTION_INPUT_KEYS:
            raise ValueError("execution seal must contain the frozen seven input hashes")
        if any(_SHA256_PATTERN.fullmatch(value) is None for _, value in self.input_hashes):
            raise ValueError("execution seal input hashes must be lowercase SHA-256")

    @property
    def seal_id(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "repository": {
                        "code_commit": self.repository.code_commit,
                        "branch": self.repository.branch,
                        "upstream": self.repository.upstream,
                        "upstream_commit": self.repository.upstream_commit,
                        "freeze_tag": self.repository.freeze_tag,
                        "freeze_tag_commit": self.repository.freeze_tag_commit,
                    },
                    "protocol_sha256": self.protocol_sha256,
                    "input_hashes": dict(self.input_hashes),
                }
            )
        ).hexdigest()

    def input_hash_mapping(self) -> dict[str, str]:
        return dict(self.input_hashes)


def _execution_input_references(background: BackgroundConfig) -> dict[str, tuple[str, str]]:
    return {
        "environment_lock": (
            background.inputs.environment_lock,
            background.inputs.environment_lock_sha256,
        ),
        "data_catalog": (
            background.inputs.data_catalog,
            background.inputs.data_catalog_sha256,
        ),
        "earthquake_dataset": (
            background.inputs.earthquake_dataset_path,
            background.inputs.earthquake_dataset_sha256,
        ),
        "study_area": (
            background.inputs.study_area,
            background.inputs.study_area_sha256,
        ),
        "issue_manifest": (
            background.inputs.issue_manifest,
            background.inputs.issue_manifest_sha256,
        ),
        "production_fixture": (
            background.numerical_regression.production_fixture,
            background.numerical_regression.production_fixture_sha256,
        ),
        "oracle_metadata": (
            background.numerical_regression.oracle_metadata,
            background.numerical_regression.oracle_metadata_sha256,
        ),
    }


def _observe_execution_input_hashes(
    project_root: Path,
    background: BackgroundConfig,
) -> tuple[tuple[str, str], ...]:
    root = Path(project_root).resolve()
    observed: list[tuple[str, str]] = []
    for key, (reference, expected) in sorted(_execution_input_references(background).items()):
        path = root.joinpath(*reference.split("/"))
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise ExecutionSealError(f"frozen input path escapes project root: {key}")
        if not resolved.is_file():
            raise ExecutionSealError(f"frozen input is missing before scoring: {key}")
        actual = sha256_file(resolved)
        if actual != expected:
            raise ExecutionSealError(f"frozen input hash differs before scoring: {key}")
        observed.append((key, actual))
    return tuple(observed)


def create_execution_seal(
    project_root: Path,
    background: BackgroundConfig,
    *,
    runner: GitCommandRunner = subprocess_git_runner,
) -> ExecutionSeal:
    """Observe the Git and seven-file identity immediately before scientific work."""

    repository = require_repository_identity(
        project_root,
        freeze_tag=background.freeze_tag,
        runner=runner,
    )
    protocol_sha256 = hashlib.sha256(
        canonical_json_bytes(background.model_dump(mode="python"))
    ).hexdigest()
    return ExecutionSeal(
        repository=repository,
        protocol_sha256=protocol_sha256,
        input_hashes=_observe_execution_input_hashes(project_root, background),
    )


def require_execution_seal_unchanged(
    project_root: Path,
    background: BackgroundConfig,
    expected: ExecutionSeal,
    *,
    runner: GitCommandRunner = subprocess_git_runner,
) -> ExecutionSeal:
    """Re-observe every sealed byte and Git identity before any permanent publication."""

    if not isinstance(expected, ExecutionSeal):
        raise TypeError("expected must be an ExecutionSeal")
    current = create_execution_seal(project_root, background, runner=runner)
    if current.repository != expected.repository:
        raise ExecutionSealError("repository identity changed after scoring began")
    if current.protocol_sha256 != expected.protocol_sha256:
        raise ExecutionSealError("background protocol changed after scoring began")
    if current.input_hashes != expected.input_hashes:
        raise ExecutionSealError("frozen input bytes changed after scoring began")
    if current.seal_id != expected.seal_id:
        raise ExecutionSealError("execution seal changed after scoring began")
    return current


def require_repository_identity(
    project_root: Path,
    *,
    freeze_tag: str,
    runner: GitCommandRunner = subprocess_git_runner,
) -> RepositoryIdentity:
    """Return a scoring-ready identity or fail before any scientific work starts.

    Readiness requires a normal branch (not detached HEAD), a clean worktree including
    untracked files, an existing protocol tag that is an ancestor of HEAD, and a configured
    upstream whose local remote-tracking object ID exactly equals HEAD.  HEAD and worktree
    state are sampled again at the end to detect changes during inspection.
    """

    root = Path(project_root).resolve()
    if not freeze_tag or freeze_tag != freeze_tag.strip():
        raise ValueError("freeze_tag must be a non-empty trimmed string")

    inside = _git(
        runner,
        root,
        "rev-parse",
        "--is-inside-work-tree",
        description="verify the Git worktree",
    ).stdout.strip()
    if inside != "true":
        raise RepositoryIdentityError("project root is not inside a Git worktree")

    head = _validated_oid(
        _git(
            runner,
            root,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            description="resolve HEAD",
        ).stdout,
        description="HEAD",
    )
    status = _git(
        runner,
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        description="inspect tracked and untracked worktree state",
    ).stdout
    if status:
        raise RepositoryIdentityError(
            "repository worktree must be clean, including untracked files"
        )

    tag_ref = f"refs/tags/{freeze_tag}^{{commit}}"
    tag_commit = _validated_oid(
        _git(
            runner,
            root,
            "rev-parse",
            "--verify",
            tag_ref,
            description=f"resolve protocol tag {freeze_tag}",
        ).stdout,
        description="protocol tag commit",
    )
    _git(
        runner,
        root,
        "merge-base",
        "--is-ancestor",
        tag_commit,
        head,
        description=f"verify protocol tag {freeze_tag} is an ancestor of HEAD",
    )

    branch = _git(
        runner,
        root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
        description="resolve the current branch",
    ).stdout.strip()
    if not branch:
        raise RepositoryIdentityError("current branch name is empty")
    upstream = _git(
        runner,
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        description="resolve the current branch upstream",
    ).stdout.strip()
    if not upstream:
        raise RepositoryIdentityError("current branch has no named upstream")
    upstream_commit = _validated_oid(
        _git(
            runner,
            root,
            "rev-parse",
            "--verify",
            "@{upstream}^{commit}",
            description="resolve the upstream commit",
        ).stdout,
        description="upstream commit",
    )
    if upstream_commit != head:
        raise RepositoryIdentityError("upstream commit must exactly equal HEAD before scoring")

    final_head = _validated_oid(
        _git(
            runner,
            root,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            description="recheck HEAD",
        ).stdout,
        description="rechecked HEAD",
    )
    if final_head != head:
        raise RepositoryIdentityError("HEAD changed while repository identity was being checked")
    final_status = _git(
        runner,
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        description="recheck tracked and untracked worktree state",
    ).stdout
    if final_status:
        raise RepositoryIdentityError(
            "repository worktree changed while repository identity was being checked"
        )

    identity = RepositoryIdentity(
        code_commit=head,
        branch=branch,
        upstream=upstream,
        upstream_commit=upstream_commit,
        freeze_tag=freeze_tag,
        freeze_tag_commit=tag_commit,
        git_available=True,
        worktree_clean=True,
        tag_is_ancestor=True,
        upstream_matches_head=True,
    )
    if not identity.ready:
        raise RepositoryIdentityError("repository identity is not ready for scoring")
    return identity


def _linux_physical_core_count() -> int | None:
    affinity_function = getattr(os, "sched_getaffinity", None)
    if affinity_function is None:
        return None
    try:
        logical_processors = tuple(sorted(int(value) for value in affinity_function(0)))
    except (OSError, TypeError, ValueError):
        return None
    physical_cores: set[tuple[int, int]] = set()
    for processor in logical_processors:
        topology = Path(f"/sys/devices/system/cpu/cpu{processor}/topology")
        try:
            package_id = int((topology / "physical_package_id").read_text().strip())
            core_id = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError):
            return None
        physical_cores.add((package_id, core_id))
    return len(physical_cores) or None


def _windows_physical_core_count() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            return None
        kernel32 = win_dll("kernel32", use_last_error=True)
        query = kernel32.GetLogicalProcessorInformationEx
        byte_count = wintypes.DWORD(0)
        query(0, None, ctypes.byref(byte_count))
        if byte_count.value < 8:
            return None
        buffer = ctypes.create_string_buffer(byte_count.value)
        if not query(0, ctypes.byref(buffer), ctypes.byref(byte_count)):
            return None
        payload = buffer.raw[: byte_count.value]
    except (AttributeError, OSError, ValueError):
        return None

    offset = 0
    count = 0
    while offset < len(payload):
        if offset + 8 > len(payload):
            return None
        relationship = int.from_bytes(payload[offset : offset + 4], "little")
        record_size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        if record_size < 8 or offset + record_size > len(payload):
            return None
        if relationship == 0:  # RelationProcessorCore
            count += 1
        offset += record_size
    return count or None


def _darwin_physical_core_count() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = int(result.stdout.strip()) if result.returncode == 0 else 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return value if value > 0 else None


def detect_physical_core_count() -> int | None:
    """Return a reliably identified physical-core count, never a logical fallback."""

    system = platform.system()
    if system == "Linux":
        return _linux_physical_core_count()
    if system == "Windows":
        return _windows_physical_core_count()
    if system == "Darwin":
        return _darwin_physical_core_count()
    return None


@dataclass(frozen=True, slots=True)
class SnapshotPlan:
    snapshot_id: str
    fit_end_utc: str
    assessment_start_utc: str
    assessment_end_utc: str
    optimizer_model_id: str
    is_validation: bool


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    detected_physical_cores: int | None
    reserve_physical_cores: int
    configured_max_workers: int
    effective_workers: int
    nested_parallelism: bool


@dataclass(frozen=True, slots=True)
class BackgroundPlan:
    """A deterministic, score-free stage-2 execution plan."""

    protocol_version: str
    freeze_tag: str
    snapshots: tuple[SnapshotPlan, ...]
    models: tuple[str, str, str]
    bandwidth_candidates_km: tuple[float, ...]
    grid_cells_km: tuple[float, ...]
    horizons_days: tuple[int, ...]
    output_paths: tuple[tuple[str, str], ...]
    resources: ResourcePlan

    def to_manifest_details(self) -> dict[str, object]:
        """Return a JSON-compatible plan without absolute paths or volatile values."""

        return {
            "protocol_version": self.protocol_version,
            "freeze_tag": self.freeze_tag,
            "snapshots": [
                {
                    "snapshot_id": item.snapshot_id,
                    "fit_end_utc": item.fit_end_utc,
                    "assessment_start_utc": item.assessment_start_utc,
                    "assessment_end_utc": item.assessment_end_utc,
                    "optimizer_model_id": item.optimizer_model_id,
                    "is_validation": item.is_validation,
                }
                for item in self.snapshots
            ],
            "models": list(self.models),
            "bandwidth_candidates_km": list(self.bandwidth_candidates_km),
            "grid_cells_km": list(self.grid_cells_km),
            "horizons_days": list(self.horizons_days),
            "output_paths": dict(self.output_paths),
            "resources": {
                "detected_physical_cores": self.resources.detected_physical_cores,
                "reserve_physical_cores": self.resources.reserve_physical_cores,
                "configured_max_workers": self.resources.configured_max_workers,
                "effective_workers": self.resources.effective_workers,
                "nested_parallelism": self.resources.nested_parallelism,
            },
        }


def _validate_project_alignment(
    background: BackgroundConfig,
    project: SeismoFluxConfig,
) -> None:
    checks = (
        (
            background.randomness.root_seed,
            project.project.random_seed,
            "background root seed",
        ),
        (
            background.integration.equal_area_crs,
            project.study_area.equal_area_crs,
            "background equal-area CRS",
        ),
        (
            background.inputs.include_external_trigger_buffer_km,
            project.study_area.include_external_trigger_buffer_km,
            "background external trigger buffer",
        ),
        (
            background.time.horizons_days,
            project.forecast.horizons_days,
            "background forecast horizons",
        ),
        (
            background.integration.grid_cells_km,
            project.integration.convergence_cells_km,
            "background integration grids",
        ),
    )
    for actual, expected, description in checks:
        if actual != expected:
            raise ValueError(f"{description} does not match the project configuration")


def build_background_plan(
    background: BackgroundConfig,
    project: SeismoFluxConfig,
    *,
    physical_core_probe: PhysicalCoreProbe = detect_physical_core_count,
) -> BackgroundPlan:
    """Build the complete stage-2 plan without reading any earthquake rows."""

    if not isinstance(background, BackgroundConfig):
        raise TypeError("background must be a validated BackgroundConfig")
    if not isinstance(project, SeismoFluxConfig):
        raise TypeError("project must be a validated SeismoFluxConfig")
    _validate_project_alignment(background, project)

    detected = physical_core_probe()
    if detected is not None and (
        not isinstance(detected, int) or isinstance(detected, bool) or detected <= 0
    ):
        raise ValueError("physical core probe must return a positive integer or None")
    effective_workers = 1
    if detected is not None:
        effective_workers = min(
            project.parallel.max_workers,
            max(1, detected - project.parallel.reserve_physical_cores),
        )

    snapshots = tuple(
        SnapshotPlan(
            snapshot_id=item.snapshot_id,
            fit_end_utc=item.fit_end_utc,
            assessment_start_utc=item.assessment_start_utc,
            assessment_end_utc=item.assessment_end_utc,
            optimizer_model_id=item.optimizer_model_id,
            is_validation=item.is_validation,
        )
        for item in build_snapshot_definitions(background)
    )
    outputs = background.outputs
    output_paths = (
        ("processed_root", outputs.processed_root),
        ("backtest_root", outputs.backtest_root),
        ("experiment_root", outputs.experiment_root),
        ("model_root", outputs.model_root),
        ("registry", outputs.registry),
        ("report", outputs.report),
    )
    return BackgroundPlan(
        protocol_version=background.protocol_version,
        freeze_tag=background.freeze_tag,
        snapshots=snapshots,
        models=_BACKGROUND_MODELS,
        bandwidth_candidates_km=background.spatial_poisson.bandwidth_candidates_km,
        grid_cells_km=background.integration.grid_cells_km,
        horizons_days=background.time.horizons_days,
        output_paths=output_paths,
        resources=ResourcePlan(
            detected_physical_cores=detected,
            reserve_physical_cores=project.parallel.reserve_physical_cores,
            configured_max_workers=project.parallel.max_workers,
            effective_workers=effective_workers,
            nested_parallelism=project.parallel.nested_parallelism,
        ),
    )


def _validated_background(background: BackgroundConfig) -> BackgroundConfig:
    if not isinstance(background, BackgroundConfig):
        raise TypeError("background must be a validated BackgroundConfig")
    return BackgroundConfig.model_validate(background.model_dump(mode="python"))


def build_address_inputs(
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    model_parameters: Mapping[str, object],
    *,
    uv_lock_sha256: str,
) -> dict[str, object]:
    """Construct the sole complete address-input mapping for stage-2 artifacts."""

    validated = _validated_background(background)
    if not isinstance(identity, RepositoryIdentity):
        raise TypeError("identity must be RepositoryIdentity")
    if not identity.ready:
        raise ValueError("repository identity is not ready for content addressing")
    if identity.freeze_tag != validated.freeze_tag:
        raise ValueError("repository identity freeze tag differs from the background protocol")
    if not isinstance(model_parameters, Mapping) or not model_parameters:
        raise ValueError("model_parameters must be a non-empty mapping")
    if any(not isinstance(key, str) for key in model_parameters):
        raise TypeError("model parameter keys must be strings")
    if not isinstance(uv_lock_sha256, str) or _SHA256_PATTERN.fullmatch(uv_lock_sha256) is None:
        raise ValueError("uv_lock_sha256 must be a lowercase SHA-256 string")
    if uv_lock_sha256 != validated.inputs.environment_lock_sha256:
        raise ValueError("observed uv.lock SHA-256 differs from the frozen environment lock")

    input_hashes = {
        "environment_lock": validated.inputs.environment_lock_sha256,
        "data_catalog": validated.inputs.data_catalog_sha256,
        "earthquake_dataset": validated.inputs.earthquake_dataset_sha256,
        "study_area": validated.inputs.study_area_sha256,
        "issue_manifest": validated.inputs.issue_manifest_sha256,
        "production_fixture": validated.numerical_regression.production_fixture_sha256,
        "oracle_metadata": validated.numerical_regression.oracle_metadata_sha256,
    }
    address_inputs: dict[str, object] = {
        "protocol": validated.model_dump(mode="python"),
        "input_hashes": input_hashes,
        "model_parameters": copy.deepcopy(dict(model_parameters)),
        "code_commit": identity.code_commit,
        "uv_lock_sha256": uv_lock_sha256,
    }
    content_address_id(address_inputs)
    return address_inputs


__all__ = [
    "BackgroundPlan",
    "CommandResult",
    "ExecutionSeal",
    "ExecutionSealError",
    "GitCommandRunner",
    "PhysicalCoreProbe",
    "RepositoryIdentity",
    "RepositoryIdentityError",
    "ResourcePlan",
    "SnapshotPlan",
    "build_address_inputs",
    "build_background_plan",
    "create_execution_seal",
    "detect_physical_core_count",
    "require_execution_seal_unchanged",
    "require_repository_identity",
    "subprocess_git_runner",
]
