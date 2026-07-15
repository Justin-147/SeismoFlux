from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from shapely.geometry import box

import seismoflux.background.pipeline_poisson as poisson_pipeline
from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.catalog import EarthquakeCatalog
from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.config import BackgroundConfig, load_background_protocol
from seismoflux.background.evidence import PairedInformationGainEvidence
from seismoflux.background.execution import ExecutionSeal, RepositoryIdentity
from seismoflux.background.future_local_support import run_local_support_future_ensembles
from seismoflux.background.horizons_local_support import (
    run_local_support_issue_horizon_backtests,
)
from seismoflux.background.issues import load_frozen_issue_calendar
from seismoflux.background.local_support_deliverables import (
    build_local_support_deliverables,
    credible_negative_from_error,
)
from seismoflux.background.local_support_manifest import (
    LocalSupportSourceFile,
    LocalSupportSources,
    build_background_local_support_manifest,
)
from seismoflux.background.local_support_runtime import (
    LocalSupportRuntime,
    build_local_support_runtime,
)
from seismoflux.background.pipeline_poisson import (
    PoissonKDEScientificInability,
    run_local_support_poisson_kde_pipeline,
)
from seismoflux.background.poisson import (
    FROZEN_BANDWIDTHS_KM,
    BandwidthPreScoreGateEvidence,
    BandwidthPreScoreGateItem,
    fit_uniform_poisson,
)
from seismoflux.background.score_ledger import (
    ScoreLedger,
    validate_generated_score_collection,
)
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    BackgroundScoringAuthorization,
)
from seismoflux.background.workflow import build_snapshot_definitions, physical_target_mask

STUDY_AREA = box(0.0, 0.0, 520_000.0, 500_000.0)
PROTOCOL_SHA256 = "c7d6488bd97f0017867573c8b99230d79091412322652af25badfe732606e76a"
FREEZE_COMMIT = "966fb4e84c36aba373d90b81fe9e1350ffe349b6"
CODE_COMMIT = "2" * 40


def _magnitudes(peak: float, upper: float, *, count: int) -> list[float]:
    peak_count = count * 3 // 4
    return [peak] * peak_count + [upper] * (count - peak_count)


def _events(
    prefix: str,
    magnitudes: Sequence[float],
    *,
    start: datetime,
    x_m: float,
) -> list[CompletenessEvent]:
    return [
        CompletenessEvent(
            event_id=f"{prefix}-{index:04d}",
            origin_time_utc=start + timedelta(minutes=index),
            available_at=start + timedelta(minutes=index),
            magnitude=magnitude,
            inside_study_area=True,
            x_m=x_m,
            y_m=1_000.0,
        )
        for index, magnitude in enumerate(magnitudes)
    ]


def _target(event_id: str, when: datetime, *, x_m: float = 1_000.0) -> CompletenessEvent:
    return CompletenessEvent(
        event_id=event_id,
        origin_time_utc=when,
        available_at=when,
        magnitude=5.0,
        inside_study_area=True,
        x_m=x_m,
        y_m=1_000.0,
    )


def _catalog_events() -> tuple[CompletenessEvent, ...]:
    return tuple(
        [
            *_events(
                "initial-main",
                _magnitudes(3.0, 3.2, count=400),
                start=datetime(1971, 1, 1, tzinfo=UTC),
                x_m=1_000.0,
            ),
            *_events(
                "later-main",
                _magnitudes(3.0, 3.2, count=400),
                start=datetime(2006, 1, 1, tzinfo=UTC),
                x_m=1_000.0,
            ),
            *_events(
                "later-high",
                _magnitudes(4.2, 4.4, count=200),
                start=datetime(2006, 6, 1, tzinfo=UTC),
                x_m=510_000.0,
            ),
            _target("target-fold-2-main", datetime(2011, 6, 1, tzinfo=UTC)),
            _target(
                "target-fold-2-unsupported",
                datetime(2011, 7, 1, tzinfo=UTC),
                x_m=510_000.0,
            ),
            _target("target-fold-3-main", datetime(2016, 6, 1, tzinfo=UTC)),
            _target("target-fold-4-main", datetime(2021, 6, 1, tzinfo=UTC)),
            _target("target-final-main", datetime(2025, 6, 1, tzinfo=UTC)),
        ]
    )


def _earthquake_catalog(events: tuple[CompletenessEvent, ...]) -> EarthquakeCatalog:
    count = len(events)
    return EarthquakeCatalog(
        event_id=np.asarray([event.event_id for event in events], dtype=np.str_),
        origin_day=np.asarray([event.origin_time_utc.timestamp() / 86_400.0 for event in events]),
        available_day=np.asarray([event.available_at.timestamp() / 86_400.0 for event in events]),
        longitude=np.full(count, 105.0),
        latitude=np.full(count, 35.0),
        x_km=np.asarray([event.x_m / 1_000.0 for event in events]),
        y_km=np.asarray([event.y_m / 1_000.0 for event in events]),
        magnitude=np.asarray([event.magnitude for event in events]),
        inside_study_area=np.asarray([event.inside_study_area for event in events]),
        inside_external_buffer=np.ones(count, dtype=np.bool_),
    )


def _with_ineligible_target(
    catalog: EarthquakeCatalog,
    event_id: str,
) -> EarthquakeCatalog:
    magnitude = catalog.magnitude.copy()
    matches = np.flatnonzero(catalog.event_id == event_id)
    assert matches.size == 1
    magnitude[int(matches[0])] = 2.0
    return dataclasses.replace(catalog, magnitude=magnitude)


def _authorized_execution() -> AuthorizedExecution:
    repository = RepositoryIdentity(
        code_commit=CODE_COMMIT,
        branch="codex/stage2-local-support",
        upstream="origin/codex/stage2-local-support",
        upstream_commit=CODE_COMMIT,
        freeze_tag="v0.2.1-background-local-support-protocol",
        freeze_tag_commit=FREEZE_COMMIT,
        git_available=True,
        worktree_clean=True,
        tag_is_ancestor=True,
        upstream_matches_head=True,
    )
    seal = ExecutionSeal(
        repository=repository,
        protocol_sha256=PROTOCOL_SHA256,
        input_hashes=tuple(
            (name, value)
            for name, value in (
                ("data_catalog", "a" * 64),
                ("earthquake_dataset", "b" * 64),
                ("environment_lock", "c" * 64),
                ("issue_manifest", "d" * 64),
                ("oracle_metadata", "e" * 64),
                ("production_fixture", "f" * 64),
                ("study_area", "1" * 64),
                ("support_manifest", "2" * 64),
            )
        ),
    )
    authorization = BackgroundScoringAuthorization(
        execution_seal_id=seal.seal_id,
        protocol_sha256=PROTOCOL_SHA256,
        freeze_tag="v0.2.1-background-local-support-protocol",
        freeze_tag_object="06136e22bb8c6e2606a9debd5e00d53b500f758d",
        freeze_tag_commit=FREEZE_COMMIT,
        scoring_code_tag="v0.2.1-background-local-support-scoring-code-r1",
        scoring_code_tag_object="3" * 40,
        scoring_code_tag_commit=CODE_COMMIT,
        code_commit=CODE_COMMIT,
        remote="origin",
        remote_repository="github.com/Justin-147/SeismoFlux",
        remote_branch_ref="refs/heads/codex/stage2-local-support",
        remote_branch_commit=CODE_COMMIT,
        frozen_blob_ids=(
            (
                "configs/background_local_support.yaml",
                "d12bf40de8f5814e3e33b988f106ed8621538487",
            ),
            (
                "data/manifests/background_local_support_fold_manifest.json",
                "0ff9be3c5b4b330569fcf549616745910c734ecf",
            ),
            (
                "data/manifests/background_local_support_manifest.json",
                "1e93b6a0e76825bea0482bc25540c3782202b2aa",
            ),
        ),
    )
    return AuthorizedExecution(execution_seal=seal, scoring_authorization=authorization)


@pytest.fixture(scope="module")
def local_inputs() -> tuple[
    BackgroundConfig,
    EarthquakeCatalog,
    LocalSupportRuntime,
    AuthorizedExecution,
]:
    config = load_background_protocol("configs/background_local_support.yaml")
    assert (
        hashlib.sha256(canonical_json_bytes(config.model_dump(mode="python"))).hexdigest()
        == PROTOCOL_SHA256
    )
    events = _catalog_events()
    manifest = build_background_local_support_manifest(
        events,
        study_area_equal_area=STUDY_AREA,
        sources=LocalSupportSources(
            earthquake_dataset=LocalSupportSourceFile(
                path="data/processed/earthquake.parquet",
                sha256="a" * 64,
            ),
            study_area=LocalSupportSourceFile(
                path="data/processed/study.geojson",
                sha256="b" * 64,
            ),
        ),
    )
    runtime = build_local_support_runtime(
        manifest,
        events,
        study_area_equal_area=STUDY_AREA,
    )
    return config, _earthquake_catalog(events), runtime, _authorized_execution()


def test_local_pipeline_is_locally_masked_and_final_target_opens_after_selection(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
) -> None:
    config, catalog, runtime, authorized = local_inputs
    trace: list[str] = []
    result = run_local_support_poisson_kde_pipeline(
        config,
        catalog,
        runtime,
        authorized,
        chunk_size=128,
        progress=lambda message: trace.append(f"progress:{message}"),
        target_access_observer=lambda snapshot_id: trace.append(f"access:{snapshot_id}"),
    )

    assert result.scoreability_gate_evidence.passed
    assert tuple(item.snapshot_id for item in result.scoreability_gate_evidence.snapshots) == (
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    )
    assert all(
        item.training_event_count > 0
        and item.target_event_count is not None
        and item.target_event_count > 0
        for item in result.scoreability_gate_evidence.snapshots[:4]
    )
    assert result.scoreability_gate_evidence.snapshot("final_validation").target_event_count is None
    assert result.pre_score_gate_evidence.passed_bandwidths_km
    assert tuple(item.bandwidth_km for item in result.bandwidth_fold_audits) == (
        result.pre_score_gate_evidence.passed_bandwidths_km
    )
    assert all(len(item.development_folds) == 4 for item in result.bandwidth_fold_audits)
    assert result.selected_bandwidth_km in FROZEN_BANDWIDTHS_KM
    assert trace.index("progress:local_poisson_kde:bandwidth_selection:done") < trace.index(
        "access:final_validation"
    )
    assert [item for item in trace if item.startswith("access:")] == [
        "access:fold_1",
        "access:fold_2",
        "access:fold_3",
        "access:fold_4",
        "access:final_validation",
    ]

    fold_2 = result.snapshot("fold_2")
    assert "later-high-0150" not in fold_2.training_event_ids
    assert "later-main-0300" in fold_2.training_event_ids
    fold_2_pair = result.spatial_evidence.development_folds[1]
    assert "target-fold-2-main" in fold_2_pair.candidate.target_event_ids
    assert "target-fold-2-unsupported" not in fold_2_pair.candidate.target_event_ids
    for pair, snapshot in zip(
        (
            *result.spatial_evidence.development_folds,
            result.spatial_evidence.validation,
        ),
        result.snapshots,
        strict=True,
    ):
        assert pair is not None
        assert pair.candidate.support_id == snapshot.support_id
        assert pair.candidate.supported_area_km2 == snapshot.supported_area_km2
        assert pair.candidate.compensator_domain_id == snapshot.compensator_domain_id
        assert pair.candidate.authorization_id == authorized.authorization_id


def test_local_scoreability_preflight_rejects_development_zero_target_before_any_score(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, catalog, runtime, authorized = local_inputs
    changed = _with_ineligible_target(catalog, "target-fold-3-main")
    fit_trace: list[str] = []
    score_trace: list[str] = []
    target_access_trace: list[str] = []

    def forbidden_fit(*args: object, **kwargs: object) -> Any:
        fit_trace.append("fit")
        raise AssertionError("model fit began before the scoreability preflight failed")

    def forbidden_score(*args: object, **kwargs: object) -> Any:
        score_trace.append("score")
        raise AssertionError("formal score creation began before preflight failed")

    monkeypatch.setattr(poisson_pipeline, "fit_uniform_poisson", forbidden_fit)
    monkeypatch.setattr(poisson_pipeline, "_local_uniform_score_evidence", forbidden_score)
    monkeypatch.setattr(poisson_pipeline, "_local_spatial_score_evidence", forbidden_score)

    with pytest.raises(PoissonKDEScientificInability) as raised:
        run_local_support_poisson_kde_pipeline(
            config,
            changed,
            runtime,
            authorized,
            target_access_observer=target_access_trace.append,
        )

    assert raised.value.reason_code == "zero_target_snapshot"
    assert raised.value.scores_started is False
    assert raised.value.partial_failure_evidence is None
    assert raised.value.gate_evidence is None
    gate = raised.value.scoreability_gate_evidence
    assert gate is not None
    assert not gate.passed
    assert gate.zero_training_snapshots == ()
    assert gate.zero_target_snapshots == ("fold_3",)
    assert tuple(item.snapshot_id for item in gate.snapshots) == (
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    )
    assert gate.snapshot("fold_3").target_event_count == 0
    assert gate.snapshot("final_validation").target_event_count is None
    assert "event_id" not in repr(dataclasses.asdict(gate)).casefold()
    assert fit_trace == []
    assert score_trace == []
    assert target_access_trace == []


def test_final_zero_target_retains_all_development_scores_after_selection(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, catalog, runtime, authorized = local_inputs
    changed = _with_ineligible_target(catalog, "target-final-main")
    trace: list[str] = []
    fit_trace: list[str] = []
    generated_score_trace: list[Any] = []
    original_fit = fit_uniform_poisson
    original_uniform_score = poisson_pipeline._local_uniform_score_evidence
    original_spatial_score = poisson_pipeline._local_spatial_score_evidence
    final_definition = build_snapshot_definitions(config)[-1]

    def observed_fit(*args: Any, **kwargs: Any) -> Any:
        fit_trace.append("fit")
        return original_fit(*args, **kwargs)

    def observed_uniform_score(*args: Any, **kwargs: Any) -> Any:
        score = original_uniform_score(*args, **kwargs)
        generated_score_trace.append(score)
        return score

    def observed_spatial_score(*args: Any, **kwargs: Any) -> Any:
        score = original_spatial_score(*args, **kwargs)
        generated_score_trace.append(score)
        return score

    def observed_target_mask(*args: Any, **kwargs: Any) -> Any:
        if kwargs["origin_after_day"] == final_definition.assessment_start_day:
            trace.append("mask:final_validation")
        return physical_target_mask(*args, **kwargs)

    monkeypatch.setattr(poisson_pipeline, "fit_uniform_poisson", observed_fit)
    monkeypatch.setattr(
        poisson_pipeline,
        "_local_uniform_score_evidence",
        observed_uniform_score,
    )
    monkeypatch.setattr(
        poisson_pipeline,
        "_local_spatial_score_evidence",
        observed_spatial_score,
    )
    monkeypatch.setattr(poisson_pipeline, "physical_target_mask", observed_target_mask)

    with pytest.raises(PoissonKDEScientificInability) as raised:
        run_local_support_poisson_kde_pipeline(
            config,
            changed,
            runtime,
            authorized,
            chunk_size=128,
            progress=lambda message: trace.append(f"progress:{message}"),
            target_access_observer=lambda snapshot_id: trace.append(f"access:{snapshot_id}"),
        )

    assert raised.value.reason_code == "zero_target_snapshot"
    assert raised.value.scores_started is True
    partial = raised.value.partial_failure_evidence
    assert partial is not None
    assert raised.value.fitted_snapshots is partial.snapshots
    assert raised.value.gate_evidence is partial.pre_score_gate_evidence
    assert raised.value.scoreability_gate_evidence is partial.scoreability_gate_evidence
    assert partial.failed_snapshot_id == "final_validation"
    assert partial.scoreability_gate_evidence.passed
    assert (
        partial.scoreability_gate_evidence.snapshot("final_validation").target_event_count is None
    )
    assert tuple(score.snapshot_id for score in partial.development_uniform_scores) == (
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    )
    assert tuple(item.bandwidth_km for item in partial.bandwidth_fold_audits) == (
        partial.pre_score_gate_evidence.passed_bandwidths_km
    )
    assert all(len(item.development_folds) == 4 for item in partial.bandwidth_fold_audits)
    assert len(partial.generated_scores) == 4 + 4 * len(partial.bandwidth_fold_audits)
    assert {score.score_id for score in partial.generated_scores} == {
        score.score_id for score in generated_score_trace
    }
    assert all(score.snapshot_id != "final_validation" for score in partial.generated_scores)
    assert partial.failed_bandwidth_gate_items == tuple(
        item for item in partial.pre_score_gate_evidence.candidates if not item.passed
    )
    assert len(fit_trace) == 5
    assert trace.index("progress:local_poisson_kde:bandwidth_selection:done") < trace.index(
        "access:final_validation"
    )
    assert trace.index("progress:local_poisson_kde:bandwidth_selection:done") < trace.index(
        "mask:final_validation"
    )
    assert trace.count("mask:final_validation") == 1
    assert [item for item in trace if item.startswith("access:")] == [
        "access:fold_1",
        "access:fold_2",
        "access:fold_3",
        "access:fold_4",
        "access:final_validation",
    ]

    calendar = load_frozen_issue_calendar(
        Path("data/manifests/background_local_support_fold_manifest.json"),
        config=config,
    )
    negative = credible_negative_from_error(
        config,
        calendar,
        runtime,
        authorized,
        raised.value,
    )
    assert negative.score_ledger.coverage == "fragment"
    assert negative.score_ledger.score_ids == tuple(
        entry.score_id for entry in negative.score_ledger.entries if entry.score_id is not None
    )
    assert set(negative.score_ledger.score_ids) == {
        score.score_id for score in partial.generated_scores
    }
    assert len(negative.score_ledger.entries) == 4 + 4 * len(FROZEN_BANDWIDTHS_KM)
    assert negative.stage3_allowed is False

    deliverables = build_local_support_deliverables(
        config,
        runtime,
        negative,
        authorized,
    )
    support_summary = cast(dict[str, object], deliverables.registry_science["support"])
    support_snapshots = cast(list[dict[str, object]], support_summary["snapshots"])
    assert all(
        isinstance(snapshot["fit_end_utc"], str) and snapshot["fit_end_utc"].endswith("Z")
        for snapshot in support_snapshots
    )
    ledger_summary = cast(dict[str, object], deliverables.registry_science["score_ledger"])
    assert ledger_summary["coverage"] == "fragment"
    assert tuple(cast(tuple[str, ...], ledger_summary["score_ids"])) == (
        negative.score_ledger.score_ids
    )
    poisson_summary = cast(dict[str, object], deliverables.registry_science["poisson_kde"])
    assert poisson_summary["status"] == "failed_after_development_scores"
    assert poisson_summary["final_validation"] == "not_run_zero_target"
    assert poisson_summary["etas"] == "not_run"
    assert poisson_summary["secondary_horizons"] == "not_run"


def test_all_bandwidth_failure_retains_five_fitted_snapshot_gates(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, catalog, runtime, authorized = local_inputs
    score_trace: list[str] = []
    target_access_trace: list[str] = []

    def failed_gate(*args: Any, **kwargs: Any) -> BandwidthPreScoreGateEvidence:
        return BandwidthPreScoreGateEvidence(
            candidates=tuple(
                BandwidthPreScoreGateItem(
                    bandwidth_km=bandwidth,
                    passed=False,
                    numerical_evidence_id=hashlib.sha256(
                        f"forced-failure-{bandwidth:g}".encode()
                    ).hexdigest(),
                    failure_reason="forced complete-candidate numerical failure",
                )
                for bandwidth in FROZEN_BANDWIDTHS_KM
            )
        )

    def forbidden_score(*args: Any, **kwargs: Any) -> Any:
        score_trace.append("score")
        raise AssertionError("formal score creation began after all bandwidths failed")

    monkeypatch.setattr(poisson_pipeline, "_local_global_gate_evidence", failed_gate)
    monkeypatch.setattr(poisson_pipeline, "_local_uniform_score_evidence", forbidden_score)

    with pytest.raises(PoissonKDEScientificInability) as raised:
        run_local_support_poisson_kde_pipeline(
            config,
            catalog,
            runtime,
            authorized,
            chunk_size=128,
            target_access_observer=target_access_trace.append,
        )

    assert raised.value.reason_code == "all_bandwidths_failed_numerical_gate"
    assert raised.value.scores_started is False
    assert raised.value.partial_failure_evidence is None
    assert raised.value.gate_evidence is not None
    assert raised.value.gate_evidence.passed_bandwidths_km == ()
    fitted = raised.value.fitted_snapshots
    assert fitted is not None
    assert tuple(item.definition.snapshot_id for item in fitted) == (
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    )
    assert all(len(item.grid_gate_evidence) == 5 for item in fitted)
    assert all(
        tuple(gate.bandwidth_km for gate in item.grid_gate_evidence) == FROZEN_BANDWIDTHS_KM
        for item in fitted
    )
    assert score_trace == []
    assert target_access_trace == []

    calendar = load_frozen_issue_calendar(
        Path("data/manifests/background_local_support_fold_manifest.json"),
        config=config,
    )
    negative = credible_negative_from_error(
        config,
        calendar,
        runtime,
        authorized,
        raised.value,
    )
    assert negative.score_ledger.coverage == "fragment"
    assert negative.score_ledger.score_ids == ()
    assert len(negative.score_ledger.entries) == 4 * len(FROZEN_BANDWIDTHS_KM)
    assert all(entry.status == "not_run" for entry in negative.score_ledger.entries)
    deliverables = build_local_support_deliverables(
        config,
        runtime,
        negative,
        authorized,
    )
    summary = cast(dict[str, object], deliverables.registry_science["poisson_kde"])
    assert summary["status"] == "failed"
    public_snapshots = cast(tuple[dict[str, object], ...], summary["snapshots"])
    assert len(public_snapshots) == 5
    assert all(len(cast(list[object], item["grid_gates"])) == 5 for item in public_snapshots)


def test_local_scoreability_preflight_rejects_zero_training_before_any_fit(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, catalog, runtime, authorized = local_inputs
    magnitude = catalog.magnitude.copy()
    initial_indices = tuple(
        index
        for index, event_id in enumerate(catalog.event_id)
        if str(event_id).startswith("initial-main-")
    )
    magnitude[np.asarray(initial_indices, dtype=np.int64)] = 2.0
    changed = dataclasses.replace(catalog, magnitude=magnitude)
    fit_trace: list[str] = []
    target_access_trace: list[str] = []

    def forbidden_fit(*args: object, **kwargs: object) -> Any:
        fit_trace.append("fit")
        raise AssertionError("model fit began before the scoreability preflight failed")

    monkeypatch.setattr(poisson_pipeline, "fit_uniform_poisson", forbidden_fit)

    with pytest.raises(PoissonKDEScientificInability) as raised:
        run_local_support_poisson_kde_pipeline(
            config,
            changed,
            runtime,
            authorized,
            target_access_observer=target_access_trace.append,
        )

    assert raised.value.reason_code == "zero_training_events"
    gate = raised.value.scoreability_gate_evidence
    assert gate is not None
    assert gate.zero_training_snapshots == ("fold_1",)
    assert gate.snapshot("fold_1").training_event_count == 0
    assert fit_trace == []
    assert target_access_trace == []


def test_local_pair_rejects_mismatched_compensator_domain(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
) -> None:
    config, catalog, runtime, authorized = local_inputs
    result = run_local_support_poisson_kde_pipeline(
        config,
        catalog,
        runtime,
        authorized,
        chunk_size=128,
    )
    pair = result.spatial_evidence.development_folds[0]
    changed = dataclasses.replace(pair.candidate, compensator_domain_id="0" * 64)

    with pytest.raises(ValueError, match="compensator_domain_id"):
        PairedInformationGainEvidence.build(candidate=changed, uniform=pair.uniform)


def test_local_pipeline_rejects_misaligned_event_ids_before_fitting(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
) -> None:
    config, catalog, runtime, authorized = local_inputs
    changed = dataclasses.replace(catalog, event_id=catalog.event_id[::-1])

    with pytest.raises(ValueError, match="event IDs/order"):
        run_local_support_poisson_kde_pipeline(config, changed, runtime, authorized)


def test_local_secondary_horizons_are_manifest_exact_and_audit_etas_unavailability(
    local_inputs: tuple[
        BackgroundConfig,
        EarthquakeCatalog,
        LocalSupportRuntime,
        AuthorizedExecution,
    ],
) -> None:
    config, catalog, runtime, authorized = local_inputs
    poisson = run_local_support_poisson_kde_pipeline(
        config,
        catalog,
        runtime,
        authorized,
        chunk_size=128,
    )
    calendar = load_frozen_issue_calendar(
        Path("data/manifests/background_local_support_fold_manifest.json"),
        config=config,
    )
    unavailable_attempt = SimpleNamespace(
        succeeded=False,
        fit_result=None,
        parameter_snapshot_id=None,
        grid_gate_evidence=None,
        failure_reasons=("synthetic final ETAS instability",),
    )
    fake_etas = SimpleNamespace(
        evidence=SimpleNamespace(validation=None),
        primary=SimpleNamespace(attempt=lambda _: unavailable_attempt),
        model_variant_id="etas/synthetic-unavailable",
    )
    primary = cast(
        Any,
        SimpleNamespace(
            protocol_sha256=PROTOCOL_SHA256,
            authorization_id=authorized.authorization_id,
            poisson=poisson,
            etas=fake_etas,
        ),
    )

    horizons = run_local_support_issue_horizon_backtests(
        config,
        catalog,
        calendar,
        runtime,
        primary,
        authorized,
    )

    assert len(horizons.comparisons) == 15
    assert {item.candidate_model_id for item in horizons.comparisons} == {"spatial_poisson"}
    assert len(horizons.failed_comparisons) == 15
    assert {item.candidate_model_id for item in horizons.failed_comparisons} == {"etas"}
    assert any(
        entry.assessment_end_utc > "2025-06-30T16:00:00Z" for entry in horizons.score_entries
    )
    ledger = ScoreLedger(
        protocol_sha256=PROTOCOL_SHA256,
        authorization_id=authorized.authorization_id,
        issue_manifest_sha256=("d7ae5266c9143ed0a67a9954da52039b2753f108698fb05477466a6d5b934e38"),
        calendar=calendar,
        entries=horizons.score_entries,
        coverage="fragment",
    )
    validate_generated_score_collection(ledger, horizons.generated_scores)
    assert ledger.locked_test_run is False

    future = run_local_support_future_ensembles(
        config,
        catalog,
        calendar,
        runtime,
        primary,
        authorized,
        detected_physical_cores=8,
        max_workers=2,
        reserve_physical_cores=2,
    )
    assert future.status == "not_run"
    assert future.ensembles is None
    assert future.failure_reason == "synthetic final ETAS instability"
