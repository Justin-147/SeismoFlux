from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from seismoflux.background import stage2r1
from seismoflux.background.config import BackgroundConfig, load_background_protocol
from seismoflux.background.future import FutureWorkerResources
from seismoflux.background.issues import FrozenIssueCalendar, load_frozen_issue_calendar
from seismoflux.background.score_ledger import (
    FROZEN_PRIMARY_INTERVALS,
    ScoreLedger,
    ScoreLedgerEntry,
)

CONFIG = Path("configs/background_local_support.yaml")
ISSUE_MANIFEST = Path("data/manifests/background_local_support_fold_manifest.json")
PROTOCOL = "a" * 64
AUTHORIZATION = "b" * 64


@pytest.fixture(scope="module")
def config() -> BackgroundConfig:
    return load_background_protocol(CONFIG)


@pytest.fixture(scope="module")
def calendar(config: BackgroundConfig) -> FrozenIssueCalendar:
    return load_frozen_issue_calendar(ISSUE_MANIFEST, config=config)


def _unused_fixture_path() -> Path:
    return (Path.cwd() / "tests/fixtures/background/etas_micro_reference.json").resolve()


def _utc_plus_days(value: str, days: int) -> str:
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    return (parsed + timedelta(days=days)).isoformat(timespec="seconds").replace("+00:00", "Z")


def _failed_primary_entry() -> ScoreLedgerEntry:
    fit_end, assessment_start, assessment_end = FROZEN_PRIMARY_INTERVALS["fold_1"]
    return ScoreLedgerEntry(
        scope="primary_snapshot",
        status="failed",
        protocol_sha256=PROTOCOL,
        authorization_id=AUTHORIZATION,
        support_id="local-support-1111111111111111",
        supported_area_km2=9_500_000.0,
        compensator_domain_id="1" * 64,
        model_id="uniform_poisson",
        model_variant_id="uniform/test",
        parameter_snapshot_id="parameters/uniform/fold-1",
        snapshot_id="fold_1",
        fit_end_utc=fit_end,
        assessment_start_utc=assessment_start,
        assessment_end_utc=assessment_end,
        selected_mc=4.0,
        score=None,
        failure_reasons=("synthetic primary failure",),
    )


def _not_run_horizon_entry(calendar: FrozenIssueCalendar) -> ScoreLedgerEntry:
    exposure = calendar.validation.exposures(7)[0]
    return ScoreLedgerEntry(
        scope="secondary_validation_horizon",
        status="not_run",
        protocol_sha256=PROTOCOL,
        authorization_id=AUTHORIZATION,
        support_id="local-support-5555555555555555",
        supported_area_km2=9_600_000.0,
        compensator_domain_id="5" * 64,
        model_id="etas",
        model_variant_id="etas/test",
        parameter_snapshot_id="parameters/etas/final",
        snapshot_id="final_validation",
        fit_end_utc=FROZEN_PRIMARY_INTERVALS["final_validation"][0],
        assessment_start_utc=exposure.issue_time_utc,
        assessment_end_utc=_utc_plus_days(exposure.issue_time_utc, 7),
        selected_mc=4.0,
        score=None,
        failure_reasons=("synthetic ETAS unavailability",),
        partition_id="validation",
        issue_date_local=exposure.issue_date_local,
        horizon_days=7,
        publication_delay_days=0,
    )


def _placeholder() -> Any:
    return cast(Any, object())


def test_numerical_regression_failure_short_circuits_before_primary(
    monkeypatch: pytest.MonkeyPatch,
    config: BackgroundConfig,
    calendar: FrozenIssueCalendar,
) -> None:
    calls: list[str] = []

    def failed_production(_: Path) -> SimpleNamespace:
        calls.append("production_regression")
        return SimpleNamespace(passed=False, checks=(("objective", False),))

    def forbidden_analytic(**_: object) -> None:
        calls.append("analytic_regression")
        pytest.fail("analytic regression must not run after production regression failure")

    def forbidden_primary(*_: object, **__: object) -> None:
        calls.append("primary")
        pytest.fail("primary scoring must not run after numerical regression failure")

    monkeypatch.setattr(stage2r1, "run_production_fixture_regression", failed_production)
    monkeypatch.setattr(stage2r1, "run_analytic_simulation_regression", forbidden_analytic)
    monkeypatch.setattr(stage2r1, "run_local_support_primary_pipeline", forbidden_primary)

    with pytest.raises(stage2r1.LocalSupportStageGateError, match="objective"):
        stage2r1.run_local_support_stage2r1(
            config,
            _placeholder(),
            calendar,
            _placeholder(),
            _placeholder(),
            production_fixture_path=_unused_fixture_path(),
            max_workers=4,
            reserve_physical_cores=2,
        )

    assert calls == ["production_regression"]


def test_resource_gate_rejects_reserving_fewer_than_two_cores_before_primary(
    monkeypatch: pytest.MonkeyPatch,
    config: BackgroundConfig,
    calendar: FrozenIssueCalendar,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(stage2r1, "_run_regressions", lambda *_: object())
    monkeypatch.setattr(stage2r1, "detect_physical_core_count", lambda: 8)

    def forbidden_primary(*_: object, **__: object) -> None:
        calls.append("primary")
        pytest.fail("primary scoring must not run when the resource gate fails")

    monkeypatch.setattr(stage2r1, "run_local_support_primary_pipeline", forbidden_primary)

    with pytest.raises(ValueError, match="reserve at least two physical cores"):
        stage2r1.run_local_support_stage2r1(
            config,
            _placeholder(),
            calendar,
            _placeholder(),
            _placeholder(),
            production_fixture_path=_unused_fixture_path(),
            max_workers=4,
            reserve_physical_cores=1,
        )

    assert calls == []


@pytest.mark.parametrize("g1_passed", [False, True])
def test_stage2r1_merges_ledgers_preserves_lock_and_gates_stage3_strictly_on_g1(
    monkeypatch: pytest.MonkeyPatch,
    config: BackgroundConfig,
    calendar: FrozenIssueCalendar,
    g1_passed: bool,
) -> None:
    primary_entry = _failed_primary_entry()
    horizon_entry = _not_run_horizon_entry(calendar)
    numerical = cast(Any, SimpleNamespace(passed=True))
    primary = cast(
        Any,
        SimpleNamespace(
            protocol_sha256=PROTOCOL,
            authorization_id=AUTHORIZATION,
            score_entries=(primary_entry,),
            generated_scores=(),
            g1=SimpleNamespace(passed=g1_passed),
        ),
    )
    horizons = cast(
        Any,
        SimpleNamespace(score_entries=(horizon_entry,), generated_scores=()),
    )
    future = cast(
        Any,
        SimpleNamespace(
            status="succeeded",
            ensembles=SimpleNamespace(
                resources=FutureWorkerResources(
                    detected_physical_cores=8,
                    reserve_physical_cores=2,
                    configured_max_workers=12,
                    effective_workers=6,
                )
            ),
        ),
    )
    calls: list[str] = []
    core_probe_calls = 0

    def regressions(*_: object) -> Any:
        calls.append("regressions")
        return numerical

    def run_primary(*_: object, **__: object) -> Any:
        calls.append("primary")
        return primary

    def run_horizons(*_: object, **__: object) -> Any:
        calls.append("horizons")
        return horizons

    def run_future(
        *_: object,
        detected_physical_cores: int | None,
        **__: object,
    ) -> Any:
        calls.append("future")
        assert detected_physical_cores == 8
        return future

    def detect_cores() -> int:
        nonlocal core_probe_calls
        core_probe_calls += 1
        return 8

    monkeypatch.setattr(stage2r1, "_run_regressions", regressions)
    monkeypatch.setattr(stage2r1, "detect_physical_core_count", detect_cores)
    monkeypatch.setattr(stage2r1, "run_local_support_primary_pipeline", run_primary)
    monkeypatch.setattr(stage2r1, "run_local_support_issue_horizon_backtests", run_horizons)
    monkeypatch.setattr(stage2r1, "run_local_support_future_ensembles", run_future)

    def fragment_ledger_as_complete(**values: object) -> ScoreLedger:
        ledger = ScoreLedger(**values, coverage="fragment")  # type: ignore[arg-type]
        object.__setattr__(ledger, "coverage", "complete")
        return ledger

    monkeypatch.setattr(
        stage2r1,
        "ScoreLedger",
        fragment_ledger_as_complete,
    )

    result = stage2r1.run_local_support_stage2r1(
        config,
        _placeholder(),
        calendar,
        _placeholder(),
        _placeholder(),
        production_fixture_path=_unused_fixture_path(),
        max_workers=12,
        reserve_physical_cores=2,
    )

    assert calls == ["regressions", "primary", "horizons", "future"]
    assert core_probe_calls == 1
    assert result.numerical_regression is numerical
    assert result.primary is primary
    assert result.horizons is horizons
    assert result.future is future
    assert result.resources.detected_physical_cores == 8
    assert result.resources.reserve_physical_cores == 2
    assert result.resources.configured_max_workers == 12
    assert result.resources.effective_workers == 6
    assert result.score_ledger.entries == (primary_entry, horizon_entry)
    assert result.score_ledger.score_ids == ()
    assert result.score_ledger.locked_test_run is False
    assert result.score_ledger.locked_test_score_ids == ()
    assert result.score_ledger.locked_test_result is None
    assert result.g1_ls_passed is g1_passed
    assert result.stage3_allowed is g1_passed
