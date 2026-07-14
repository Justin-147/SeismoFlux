"""Sealed production entry point for the frozen stage-2R-1 local-support workflow."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from pathlib import Path
from zoneinfo import ZoneInfo

from seismoflux.background.catalog import load_earthquake_catalog, load_study_area
from seismoflux.background.config import (
    BackgroundConfig,
    load_background_protocol,
    load_project_background_config,
)
from seismoflux.background.execution import (
    ExecutionSeal,
    GitCommandRunner,
    create_execution_seal,
    require_execution_seal_unchanged,
    subprocess_git_runner,
)
from seismoflux.background.issues import load_frozen_issue_calendar
from seismoflux.background.local_support_deliverables import (
    LocalSupportStage2R1Outcome,
    PublishedLocalSupportDeliverables,
    credible_negative_from_error,
    publish_local_support_deliverables,
)
from seismoflux.background.local_support_manifest import (
    load_background_local_support_manifest,
)
from seismoflux.background.local_support_runtime import (
    LocalSupportRuntime,
    build_local_support_runtime,
)
from seismoflux.background.pipeline_poisson import PoissonKDEScientificInability
from seismoflux.background.runner import (
    BackgroundRunTelemetry,
    MemoryProbe,
    process_peak_working_set_bytes,
)
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    create_authorized_execution,
    require_background_scoring_protocol_eligible,
)
from seismoflux.background.stage2r1 import (
    LocalSupportStageGateError,
    run_local_support_stage2r1,
)
from seismoflux.background.workflow import catalog_completeness_events
from seismoflux.config import load_config, project_root_for, resolve_project_path

ProgressCallback = Callable[[str], None]


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


@dataclass(frozen=True, slots=True)
class LocalSupportBackgroundRunResult:
    """One fully sealed 2R-1 run after independent immutable publication."""

    execution_seal: ExecutionSeal
    authorized_execution: AuthorizedExecution
    runtime: LocalSupportRuntime
    science: LocalSupportStage2R1Outcome
    deliverables: PublishedLocalSupportDeliverables
    telemetry: BackgroundRunTelemetry

    def to_manifest_details(self) -> dict[str, object]:
        registry = self.deliverables.registry_payload
        registry_science = registry["science"]
        if not isinstance(registry_science, dict):
            raise TypeError("local-support registry science must be a mapping")
        ledger = registry_science["score_ledger"]
        if not isinstance(ledger, dict):
            raise TypeError("local-support registry ledger must be a mapping")
        return {
            "execution_seal_id": self.execution_seal.seal_id,
            "authorization_id": self.authorized_execution.authorization_id,
            "authorized_public_repository": (
                self.authorized_execution.scoring_authorization.remote_repository
            ),
            "code_commit": self.execution_seal.repository.code_commit,
            "outcome_status": registry_science["outcome_status"],
            "g1_ls_passed": self.science.g1_ls_passed,
            "stage3_allowed": self.science.stage3_allowed,
            "support_manifest_id": self.runtime.manifest_id,
            "score_ledger_id": ledger["ledger_id"],
            "score_ledger_sha256": ledger["ledger_sha256"],
            "score_count": ledger["score_count"],
            "locked_test_run": False,
            "bundle_artifacts": {
                item.bundle_kind: {
                    "artifact_id": item.artifact.artifact_id,
                    "manifest_sha256": item.artifact.manifest_sha256,
                }
                for item in self.deliverables.bundle_publications
            },
            "registry_sha256": self.deliverables.registry.sha256,
            "report_sha256": self.deliverables.report.sha256,
            "results_figure_sha256": self.deliverables.results_figure.sha256,
            "telemetry": {
                "elapsed_seconds": self.telemetry.elapsed_seconds,
                "cpu_seconds": self.telemetry.cpu_seconds,
                "process_peak_working_set_bytes": (self.telemetry.process_peak_working_set_bytes),
            },
        }


def _maximum_catalog_time(background: BackgroundConfig) -> datetime:
    return datetime.combine(
        date.fromisoformat(background.time.validation_maturity_end_local),
        datetime_time.min,
        tzinfo=ZoneInfo(background.time.timezone),
    ).astimezone(UTC)


def run_background_stage2_local_support(
    config_path: str | Path = Path("configs/base_local_support.yaml"),
    *,
    progress: ProgressCallback | None = None,
    git_runner: GitCommandRunner = subprocess_git_runner,
    memory_probe: MemoryProbe = process_peak_working_set_bytes,
) -> LocalSupportBackgroundRunResult:
    """Authorize, run, and publish the sole production 0.2.1 scoring workflow."""

    main_path = Path(config_path).resolve(strict=True)
    project_root = project_root_for(main_path).resolve()
    project = load_config(main_path)
    background_path = resolve_project_path(main_path, project.config_files.background)
    preflight = load_background_protocol(background_path)
    require_background_scoring_protocol_eligible(preflight)
    background: BackgroundConfig = load_project_background_config(main_path)
    require_background_scoring_protocol_eligible(background)
    if str(background.protocol_version) != "0.2.1":
        raise ValueError("local-support runner requires protocol version 0.2.1")

    _notify(progress, "local_support_execution_seal:start")
    seal = create_execution_seal(project_root, background, runner=git_runner)
    _notify(progress, f"local_support_execution_seal:ready:{seal.seal_id}")
    _notify(progress, "local_support_scoring_authorization:start")
    authorized = create_authorized_execution(
        project_root,
        background,
        seal,
        runner=git_runner,
    )
    _notify(progress, f"local_support_scoring_authorization:ready:{authorized.authorization_id}")
    require_execution_seal_unchanged(project_root, background, seal, runner=git_runner)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    _notify(progress, "local_support_inputs:study_area:start")
    study_area = load_study_area(
        resolve_project_path(main_path, background.inputs.study_area),
        background.integration.equal_area_crs,
    )
    _notify(progress, "local_support_inputs:study_area:done")
    _notify(progress, "local_support_inputs:earthquake_catalog:start")
    catalog = load_earthquake_catalog(
        resolve_project_path(main_path, background.inputs.earthquake_dataset_path),
        study_area=study_area,
        external_buffer_km=background.inputs.include_external_trigger_buffer_km,
        maximum_event_time_utc=_maximum_catalog_time(background),
    )
    _notify(progress, f"local_support_inputs:earthquake_catalog:done:rows={len(catalog)}")
    _notify(progress, "local_support_inputs:issue_calendar:start")
    calendar = load_frozen_issue_calendar(
        resolve_project_path(main_path, background.inputs.issue_manifest),
        config=background,
    )
    _notify(progress, "local_support_inputs:issue_calendar:done")
    support_reference = getattr(background.inputs, "support_manifest", None)
    if not isinstance(support_reference, str):
        raise ValueError("local-support runner requires a frozen support manifest")
    _notify(progress, "local_support_runtime:reconstruct:start")
    support_manifest = load_background_local_support_manifest(
        resolve_project_path(main_path, support_reference)
    )
    runtime = build_local_support_runtime(
        support_manifest,
        catalog_completeness_events(catalog),
        study_area_equal_area=study_area.projected,
    )
    _notify(progress, f"local_support_runtime:ready:{runtime.manifest_id}")

    fixture_path = resolve_project_path(
        main_path,
        background.numerical_regression.production_fixture,
    ).resolve(strict=True)
    _notify(progress, "execution_seal:pre_scoring_recheck:start")
    require_execution_seal_unchanged(project_root, background, seal, runner=git_runner)
    _notify(progress, "execution_seal:pre_scoring_recheck:done")
    try:
        science: LocalSupportStage2R1Outcome = run_local_support_stage2r1(
            background,
            catalog,
            calendar,
            runtime,
            authorized,
            production_fixture_path=fixture_path,
            max_workers=project.parallel.max_workers,
            reserve_physical_cores=project.parallel.reserve_physical_cores,
            progress=progress,
        )
    except (LocalSupportStageGateError, PoissonKDEScientificInability) as error:
        science = credible_negative_from_error(
            background,
            calendar,
            runtime,
            authorized,
            error,
        )
        _notify(
            progress,
            f"local_support_science:credible_negative:{science.failure_stage}:"
            f"{science.failure_code}",
        )

    require_execution_seal_unchanged(project_root, background, seal, runner=git_runner)
    _notify(progress, "local_support_deliverables:publish:start")
    published = publish_local_support_deliverables(
        project_root,
        background,
        runtime,
        science,
        authorized,
        runner=git_runner,
    )
    _notify(progress, "local_support_deliverables:publish:done")
    telemetry = BackgroundRunTelemetry(
        elapsed_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        process_peak_working_set_bytes=memory_probe(),
    )
    return LocalSupportBackgroundRunResult(
        execution_seal=seal,
        authorized_execution=authorized,
        runtime=runtime,
        science=science,
        deliverables=published,
        telemetry=telemetry,
    )


__all__ = [
    "LocalSupportBackgroundRunResult",
    "ProgressCallback",
    "run_background_stage2_local_support",
]
