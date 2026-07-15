"""Complete non-locked stage-2R-1 scientific execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from seismoflux.background.adapters import build_analytic_simulation_expectations
from seismoflux.background.catalog import EarthquakeCatalog
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.execution import detect_physical_core_count
from seismoflux.background.future_local_support import (
    LocalSupportFutureOutcome,
    run_local_support_future_ensembles,
)
from seismoflux.background.horizons_local_support import (
    LocalSupportHorizonBacktests,
    run_local_support_issue_horizon_backtests,
)
from seismoflux.background.issues import FrozenIssueCalendar
from seismoflux.background.local_support_runtime import LocalSupportRuntime
from seismoflux.background.pipeline import NumericalRegressionEvidence
from seismoflux.background.pipeline_local_support import (
    LocalSupportPrimaryPipelineResult,
    run_local_support_primary_pipeline,
)
from seismoflux.background.regression import (
    run_analytic_simulation_regression,
    run_production_fixture_regression,
)
from seismoflux.background.score_ledger import (
    ScoreLedger,
    validate_generated_score_collection,
)
from seismoflux.background.scoring_authorization import AuthorizedExecution
from seismoflux.config import sha256_file

ProgressCallback = Callable[[str], None]


class LocalSupportStageGateError(RuntimeError):
    """A preregistered numerical gate failed before or during G1-LS."""


@dataclass(frozen=True, slots=True)
class LocalSupportResourceEvidence:
    detected_physical_cores: int | None
    reserve_physical_cores: int
    configured_max_workers: int
    effective_workers: int

    def __post_init__(self) -> None:
        if self.reserve_physical_cores < 2:
            raise ValueError("stage-2R-1 must reserve at least two physical cores")
        if self.configured_max_workers <= 0 or self.effective_workers <= 0:
            raise ValueError("stage-2R-1 worker counts must be positive")
        expected = (
            1
            if self.detected_physical_cores is None
            else min(
                self.configured_max_workers,
                max(1, self.detected_physical_cores - self.reserve_physical_cores),
            )
        )
        if self.effective_workers != expected:
            raise ValueError("stage-2R-1 worker allocation violates the reserve rule")


@dataclass(frozen=True, slots=True)
class LocalSupportStage2R1Result:
    protocol_sha256: str
    authorization_id: str
    numerical_regression: NumericalRegressionEvidence
    resources: LocalSupportResourceEvidence
    primary: LocalSupportPrimaryPipelineResult
    horizons: LocalSupportHorizonBacktests
    future: LocalSupportFutureOutcome
    score_ledger: ScoreLedger

    def __post_init__(self) -> None:
        if self.primary.protocol_sha256 != self.protocol_sha256:
            raise ValueError("stage-2R-1 primary result uses another protocol")
        if self.primary.authorization_id != self.authorization_id:
            raise ValueError("stage-2R-1 primary result uses another authorization")
        if self.score_ledger.protocol_sha256 != self.protocol_sha256:
            raise ValueError("stage-2R-1 score ledger uses another protocol")
        if self.score_ledger.authorization_id != self.authorization_id:
            raise ValueError("stage-2R-1 score ledger uses another authorization")
        if self.score_ledger.coverage != "complete":
            raise ValueError("stage-2R-1 result requires a complete score ledger")
        if self.future.status == "succeeded":
            if self.future.ensembles is None:
                raise ValueError("successful stage-2R-1 future result has no ensembles")
            future_resources = self.future.ensembles.resources
            if (
                future_resources.detected_physical_cores != self.resources.detected_physical_cores
                or future_resources.reserve_physical_cores != self.resources.reserve_physical_cores
                or future_resources.configured_max_workers != self.resources.configured_max_workers
                or future_resources.effective_workers != self.resources.effective_workers
            ):
                raise ValueError(
                    "future worker resources differ from the stage-2R-1 resource evidence"
                )
        self.score_ledger.assert_locked_test_not_run()

    @property
    def g1_ls_passed(self) -> bool:
        return self.primary.g1.passed

    @property
    def stage3_allowed(self) -> bool:
        return self.g1_ls_passed


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _run_regressions(
    config: BackgroundConfig,
    fixture_path: Path,
    progress: ProgressCallback | None,
) -> NumericalRegressionEvidence:
    _notify(progress, "local_support_regression:production:start")
    production = run_production_fixture_regression(fixture_path)
    if not production.passed:
        failed = ", ".join(name for name, passed in production.checks if not passed)
        raise LocalSupportStageGateError(f"production ETAS numerical regression failed: {failed}")
    _notify(progress, "local_support_regression:production:done")
    analytic_config = config.numerical_regression.analytic_simulation
    _notify(progress, "local_support_regression:analytic_8192:start")
    analytic = run_analytic_simulation_regression(
        replicate_count=analytic_config.replicate_count,
        maximum_events_per_replicate=(
            config.randomness.future_simulation.maximum_events_per_replicate
        ),
        expectations=build_analytic_simulation_expectations(config),
    )
    if not analytic.passed:
        failed = ", ".join(name for name, passed in analytic.checks if not passed)
        raise LocalSupportStageGateError(f"analytic ETAS simulation regression failed: {failed}")
    _notify(progress, "local_support_regression:analytic_8192:done")
    return NumericalRegressionEvidence(
        production_fixture=production,
        analytic_simulation=analytic,
    )


def _resource_evidence(
    *,
    max_workers: int,
    reserve_physical_cores: int,
) -> LocalSupportResourceEvidence:
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise TypeError("max_workers must be an integer")
    if not isinstance(reserve_physical_cores, int) or isinstance(reserve_physical_cores, bool):
        raise TypeError("reserve_physical_cores must be an integer")
    detected = detect_physical_core_count()
    effective = (
        1 if detected is None else min(max_workers, max(1, detected - reserve_physical_cores))
    )
    return LocalSupportResourceEvidence(
        detected_physical_cores=detected,
        reserve_physical_cores=reserve_physical_cores,
        configured_max_workers=max_workers,
        effective_workers=effective,
    )


def run_local_support_stage2r1(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    runtime: LocalSupportRuntime,
    authorized_execution: AuthorizedExecution,
    *,
    production_fixture_path: Path,
    max_workers: int,
    reserve_physical_cores: int,
    progress: ProgressCallback | None = None,
) -> LocalSupportStage2R1Result:
    """Run every preregistered non-locked 2R-1 calculation and seal its ledger."""

    if not production_fixture_path.is_absolute():
        raise ValueError("production fixture path must be absolute")
    expected_fixture_sha256 = config.numerical_regression.production_fixture_sha256
    if sha256_file(production_fixture_path) != expected_fixture_sha256:
        raise ValueError("production fixture bytes differ from the frozen protocol")
    numerical = _run_regressions(config, production_fixture_path, progress)
    resources = _resource_evidence(
        max_workers=max_workers,
        reserve_physical_cores=reserve_physical_cores,
    )
    _notify(progress, "local_support_primary:start")
    primary = run_local_support_primary_pipeline(
        config,
        catalog,
        runtime,
        authorized_execution,
        progress=progress,
    )
    _notify(progress, "local_support_primary:done")
    _notify(progress, "local_support_horizons:start")
    horizons = run_local_support_issue_horizon_backtests(
        config,
        catalog,
        calendar,
        runtime,
        primary,
        authorized_execution,
    )
    _notify(progress, "local_support_horizons:done")
    _notify(progress, "local_support_future:start")
    future = run_local_support_future_ensembles(
        config,
        catalog,
        calendar,
        runtime,
        primary,
        authorized_execution,
        detected_physical_cores=resources.detected_physical_cores,
        max_workers=max_workers,
        reserve_physical_cores=reserve_physical_cores,
        progress=progress,
    )
    _notify(progress, f"local_support_future:{future.status}")
    ledger = ScoreLedger(
        protocol_sha256=primary.protocol_sha256,
        authorization_id=primary.authorization_id,
        issue_manifest_sha256=config.inputs.issue_manifest_sha256,
        calendar=calendar,
        entries=(*primary.score_entries, *horizons.score_entries),
    )
    generated = (*primary.generated_scores, *horizons.generated_scores)
    validate_generated_score_collection(ledger, generated)
    return LocalSupportStage2R1Result(
        protocol_sha256=primary.protocol_sha256,
        authorization_id=primary.authorization_id,
        numerical_regression=numerical,
        resources=resources,
        primary=primary,
        horizons=horizons,
        future=future,
        score_ledger=ledger,
    )


__all__ = [
    "LocalSupportResourceEvidence",
    "LocalSupportStage2R1Result",
    "LocalSupportStageGateError",
    "run_local_support_stage2r1",
]
