"""Sealed production entry point for the frozen stage-2 background workflow."""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from seismoflux.background.catalog import load_earthquake_catalog, load_study_area
from seismoflux.background.config import BackgroundConfig, load_project_background_config
from seismoflux.background.deliverables import (
    PublishedBackgroundDeliverables,
    build_background_deliverables,
    publish_background_deliverables,
)
from seismoflux.background.execution import (
    ExecutionSeal,
    GitCommandRunner,
    create_execution_seal,
    require_execution_seal_unchanged,
    subprocess_git_runner,
)
from seismoflux.background.grid import build_grid_family
from seismoflux.background.issues import load_frozen_issue_calendar
from seismoflux.background.pipeline import BackgroundPipelineOutcome, run_background_pipeline
from seismoflux.background.publication import (
    RegistryReportPublication,
    publish_registry_and_report_sealed,
)
from seismoflux.config import load_config, project_root_for, resolve_project_path

ProgressCallback = Callable[[str], None]
MemoryProbe = Callable[[], int | None]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    )


def process_peak_working_set_bytes() -> int | None:
    """Return peak resident process memory when the operating-system API is available."""

    if os.name == "posix":
        try:
            import resource
        except ImportError:
            return None
        resource_module = cast(Any, resource)
        peak = int(resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss)
        if peak <= 0:
            return None
        return peak if sys.platform == "darwin" else peak * 1_024
    if os.name != "nt":
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        loader = ctypes.windll
        get_current_process = loader.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = loader.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        succeeded = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
    except (AttributeError, OSError):
        return None
    if not succeeded:
        return None
    return int(counters.peak_working_set_size)


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


@dataclass(frozen=True, slots=True)
class BackgroundRunTelemetry:
    """Runtime-only measurements kept out of scientific content addresses."""

    elapsed_seconds: float
    cpu_seconds: float
    process_peak_working_set_bytes: int | None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.elapsed_seconds)
            or not math.isfinite(self.cpu_seconds)
            or self.elapsed_seconds < 0.0
            or self.cpu_seconds < 0.0
        ):
            raise ValueError("background runtime measurements must be non-negative")
        if (
            self.process_peak_working_set_bytes is not None
            and self.process_peak_working_set_bytes <= 0
        ):
            raise ValueError("background peak working set must be positive when available")


@dataclass(frozen=True, slots=True)
class BackgroundRunResult:
    """One fully sealed run after four bundles and both fixed projections exist."""

    execution_seal: ExecutionSeal
    science: BackgroundPipelineOutcome
    deliverables: PublishedBackgroundDeliverables
    fixed_publication: RegistryReportPublication
    telemetry: BackgroundRunTelemetry

    def to_manifest_details(self) -> dict[str, object]:
        """Return concise runtime metadata without duplicating scientific score payloads."""

        attempts = self.deliverables.registry.model_attempts
        succeeded = sum(item.status == "succeeded" for item in attempts)
        failed = sum(item.status == "failed" for item in attempts)
        not_run = sum(item.status == "not_run" for item in attempts)
        scientific_summary = self.deliverables.registry.scientific_summary
        scientific_failure = scientific_summary.failure
        resources = self.science.resources
        return {
            "execution_seal_id": self.execution_seal.seal_id,
            "code_commit": self.execution_seal.repository.code_commit,
            "bundle_artifacts": {
                item.bundle_kind: {
                    "artifact_id": item.artifact.artifact_id,
                    "manifest_sha256": item.artifact.manifest_sha256,
                }
                for item in self.deliverables.bundle_publications
            },
            "registry_sha256": self.fixed_publication.registry.sha256,
            "report_sha256": self.fixed_publication.report.sha256,
            "fixed_delivery_confirmed": True,
            "g1_passed": self.deliverables.registry.g1.passed,
            "g1_status": self.deliverables.registry.g1.status,
            "g1_passing_models": list(self.deliverables.registry.g1.passing_models),
            "selected_model_id": self.deliverables.registry.selection.selected_model_id,
            "stage3_allowed": self.deliverables.registry.stage3_allowed,
            "model_attempts": {
                "total": len(attempts),
                "succeeded": succeeded,
                "failed": failed,
                "not_run": not_run,
            },
            "scientific_outcome_status": scientific_summary.outcome_status,
            "scientific_failure": (
                scientific_failure.model_dump(mode="json")
                if scientific_failure is not None
                else None
            ),
            "resources": {
                "detected_physical_cores": resources.detected_physical_cores,
                "reserve_physical_cores": resources.reserve_physical_cores,
                "configured_max_workers": resources.configured_max_workers,
                "effective_workers": resources.effective_workers,
            },
            "telemetry": {
                "elapsed_seconds": self.telemetry.elapsed_seconds,
                "cpu_seconds": self.telemetry.cpu_seconds,
                "process_peak_working_set_bytes": (self.telemetry.process_peak_working_set_bytes),
            },
        }


def run_background_stage2(
    config_path: str | Path = Path("configs/base.yaml"),
    *,
    progress: ProgressCallback | None = None,
    git_runner: GitCommandRunner = subprocess_git_runner,
    memory_probe: MemoryProbe = process_peak_working_set_bytes,
) -> BackgroundRunResult:
    """Run, re-seal, and publish the sole production stage-2 workflow."""

    main_path = Path(config_path).resolve(strict=True)
    project_root = project_root_for(main_path).resolve()
    project = load_config(main_path)
    background: BackgroundConfig = load_project_background_config(main_path)

    _notify(progress, "execution_seal:start")
    seal = create_execution_seal(project_root, background, runner=git_runner)
    _notify(progress, f"execution_seal:ready:{seal.seal_id}")

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    _notify(progress, "inputs:study_area:start")
    study_area = load_study_area(
        resolve_project_path(main_path, background.inputs.study_area),
        background.integration.equal_area_crs,
    )
    _notify(progress, "inputs:study_area:done")
    _notify(progress, "inputs:earthquake_catalog:start")
    catalog = load_earthquake_catalog(
        resolve_project_path(main_path, background.inputs.earthquake_dataset_path),
        study_area=study_area,
        external_buffer_km=background.inputs.include_external_trigger_buffer_km,
        maximum_event_time_utc=datetime.combine(
            date.fromisoformat(background.time.validation_maturity_end_local),
            datetime_time.min,
            tzinfo=ZoneInfo(background.time.timezone),
        ).astimezone(UTC),
    )
    _notify(progress, f"inputs:earthquake_catalog:done:rows={len(catalog)}")
    _notify(progress, "inputs:issue_calendar:start")
    issue_calendar = load_frozen_issue_calendar(
        resolve_project_path(main_path, background.inputs.issue_manifest),
        config=background,
    )
    _notify(progress, "inputs:issue_calendar:done")
    _notify(progress, "integration_grids:start")
    grid_family = build_grid_family(study_area.geographic)
    _notify(progress, "integration_grids:done")

    science = run_background_pipeline(
        background,
        catalog,
        study_area,
        grid_family,
        issue_calendar,
        production_fixture_path=resolve_project_path(
            main_path,
            background.numerical_regression.production_fixture,
        ),
        max_workers=project.parallel.max_workers,
        reserve_physical_cores=project.parallel.reserve_physical_cores,
        progress=progress,
    )
    _notify(progress, "deliverables:adapt:start")
    bundle_inputs = build_background_deliverables(background, science)
    _notify(progress, "deliverables:adapt:done")

    _notify(progress, "execution_seal:pre_bundle_recheck:start")
    require_execution_seal_unchanged(
        project_root,
        background,
        seal,
        runner=git_runner,
    )
    _notify(progress, "execution_seal:pre_bundle_recheck:done")
    _notify(progress, "deliverables:bundle_publish:start")
    published = publish_background_deliverables(
        project_root,
        background,
        seal,
        bundle_inputs,
        runner=git_runner,
    )
    _notify(progress, "deliverables:bundle_publish:done")

    telemetry = BackgroundRunTelemetry(
        elapsed_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        process_peak_working_set_bytes=memory_probe(),
    )
    _notify(progress, "deliverables:fixed_publish:start")
    fixed = publish_registry_and_report_sealed(
        project_root,
        background,
        published.registry,
        seal,
        runner=git_runner,
    )
    return BackgroundRunResult(
        execution_seal=seal,
        science=science,
        deliverables=published,
        fixed_publication=fixed,
        telemetry=telemetry,
    )


__all__ = [
    "BackgroundRunResult",
    "BackgroundRunTelemetry",
    "MemoryProbe",
    "ProgressCallback",
    "process_peak_working_set_bytes",
    "run_background_stage2",
]
