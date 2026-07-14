from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import yaml

import seismoflux.background.deliverables as deliverables_module
from seismoflux.background.artifacts import ArtifactFile, canonical_json_bytes
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.deliverables import (
    build_background_deliverables,
    publish_background_deliverables,
)
from seismoflux.background.evidence import (
    EXPECTED_SNAPSHOTS,
    AuditedBackgroundModelEvidence,
    AuditedG1Assessment,
    AuditedModelSelection,
    ModelId,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
    assess_audited_g1,
    select_audited_background_model,
)
from seismoflux.background.execution import (
    ExecutionSeal,
    GitCommandRunner,
    RepositoryIdentity,
)
from seismoflux.background.future import (
    FUTURE_HORIZONS_DAYS,
    FUTURE_QUANTILE_METHOD,
    FUTURE_QUANTILE_PROBABILITIES,
    FutureCountQuantile,
    FutureHorizonSummary,
    FutureIssueEnsemble,
    FutureWorkerResources,
    SparseExpectedCellCount,
    SparseGridExpectedCounts,
    ValidationFutureEnsembles,
)
from seismoflux.background.pipeline import (
    BackgroundPipelineFailure,
    BackgroundPipelineResult,
    IntegrationGridEvidence,
    NumericalRegressionEvidence,
    PipelineResourceEvidence,
)
from seismoflux.background.pipeline_etas import ETASSnapshotAttempt
from seismoflux.background.publication import registry_payload_bytes
from seismoflux.background.visualization import ConditionalIntensityRender


@pytest.fixture(scope="module")
def background() -> BackgroundConfig:
    raw = yaml.safe_load(Path("configs/background.yaml").read_text(encoding="utf-8"))
    return BackgroundConfig.model_validate(raw)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol_sha(config: BackgroundConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config.model_dump(mode="python"))).hexdigest()


@dataclass(frozen=True, slots=True)
class _Definition:
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class _Completeness:
    definition: _Definition
    selected_mc: float


@dataclass(frozen=True, slots=True)
class _Mixture:
    training_event_count: int


@dataclass(frozen=True, slots=True)
class _KDEModel:
    normalization_mass: float
    rate_per_day: float
    mixture: _Mixture


@dataclass(frozen=True, slots=True)
class _UniformModel:
    rate_per_day: float
    study_area_km2: float


@dataclass(frozen=True, slots=True)
class _PoissonSnapshot:
    definition: _Definition
    grid_gate_evidence: tuple[dict[str, object], ...]
    kde_family: tuple[tuple[float, _KDEModel], ...]
    rate_per_day: float
    selected_mc: float
    target_event_ids: tuple[str, ...]
    training_duration_days: float
    training_event_count: int
    training_event_ids: tuple[str, ...]
    training_evidence_id: str
    uniform_model: _UniformModel


@dataclass(frozen=True, slots=True)
class _GridCell:
    cell_id: str
    row: int
    column: int
    representative_x_km: float
    representative_y_km: float
    clipped_area_km2: float


@dataclass(frozen=True, slots=True)
class _GridResolution:
    cell_size_km: float
    cells: tuple[_GridCell, ...]
    total_clipped_area_km2: float


@dataclass(frozen=True, slots=True)
class _IntegrationGrids:
    study_area_km2: float
    resolutions: tuple[_GridResolution, ...]

    def at(self, cell_size_km: float) -> _GridResolution:
        return next(item for item in self.resolutions if item.cell_size_km == cell_size_km)


@dataclass(frozen=True, slots=True)
class _PoissonResult:
    protocol_sha256: str
    snapshots: tuple[_PoissonSnapshot, ...]
    selected_bandwidth_km: float
    uniform_evidence: AuditedBackgroundModelEvidence
    spatial_evidence: AuditedBackgroundModelEvidence
    bandwidth_fold_audits: tuple[dict[str, object], ...]
    bandwidth_selection: dict[str, object]
    pre_score_gate_evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Stability:
    stable: bool
    failure_reasons: tuple[str, ...]
    evidence_value: float


@dataclass(frozen=True, slots=True)
class _FitResult:
    stability: _Stability


@dataclass(frozen=True, slots=True)
class _GridGate:
    passed: bool
    numerical_evidence_id: str


@dataclass(frozen=True, slots=True)
class _Attempt:
    definition: _Definition
    selected_mc: float
    fit_selection: dict[str, object]
    score_selection: _SelectionAudit
    fit_result: _FitResult | None
    parameter_snapshot_id: str | None
    grid_gate_evidence: _GridGate | None
    paired_evidence: PairedInformationGainEvidence | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ETASResult:
    protocol_sha256: str
    model_variant_id: str
    attempts: tuple[ETASSnapshotAttempt, ...]
    etas_evidence: AuditedBackgroundModelEvidence


@dataclass(frozen=True, slots=True)
class _SelectionAudit:
    snapshot_id: str
    role: str
    target_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BootstrapInterval:
    replications: int
    confidence_level: float
    point_estimate: float
    lower: float
    upper: float
    replicate_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _BootstrapOutcome:
    model_id: ModelId
    interval: _BootstrapInterval | None
    not_run_reason: str | None


@dataclass(frozen=True, slots=True)
class _Bootstrap:
    spatial_poisson: _BootstrapOutcome
    etas: _BootstrapOutcome


@dataclass(frozen=True, slots=True)
class _HorizonOutcome:
    model_id: ModelId
    comparisons: tuple[dict[str, object], ...] | None
    not_run_reason: str | None


@dataclass(frozen=True, slots=True)
class _Horizons:
    spatial_poisson: _HorizonOutcome
    etas: _HorizonOutcome


@dataclass(frozen=True, slots=True)
class _Regressions:
    production_fixture_passed: bool
    analytic_simulation_passed: bool


@dataclass(frozen=True, slots=True)
class _Resources:
    detected_physical_cores: int
    reserve_physical_cores: int
    effective_workers: int


@dataclass(frozen=True, slots=True)
class _SkippedFuture:
    ensembles: None
    not_run_reason: str


@dataclass(frozen=True, slots=True)
class _CompletedFuture:
    ensembles: ValidationFutureEnsembles
    not_run_reason: None


@dataclass(frozen=True, slots=True)
class _Representative:
    issue_date_local: str
    issue_time_utc: str
    selected_model_id: ModelId
    selected_model_variant_id: str
    primary_grid_cell_size_km: float
    cell_ids: tuple[str, ...]
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    representative_x_km: np.ndarray[tuple[int], np.dtype[np.float64]]
    representative_y_km: np.ndarray[tuple[int], np.dtype[np.float64]]
    clipped_area_km2: np.ndarray[tuple[int], np.dtype[np.float64]]
    eligible_parent_event_ids: tuple[str, ...]
    background_intensity: np.ndarray[tuple[int], np.dtype[np.float64]]
    triggering_intensity: np.ndarray[tuple[int], np.dtype[np.float64]]
    total_intensity: np.ndarray[tuple[int], np.dtype[np.float64]]
    render: ConditionalIntensityRender


@dataclass(frozen=True, slots=True)
class _SyntheticResult:
    protocol_sha256: str
    regressions: _Regressions
    completeness: tuple[_Completeness, ...]
    poisson: _PoissonResult
    etas: _ETASResult
    g1: AuditedG1Assessment
    selection: AuditedModelSelection
    bootstrap: _Bootstrap
    horizons: _Horizons
    future: _SkippedFuture | _CompletedFuture
    representative: _Representative
    resources: _Resources
    integration_grids: _IntegrationGrids


def _score(
    *,
    protocol_sha256: str,
    model_id: ModelId,
    variant: str,
    snapshot_id: str,
    event_log_intensity: float,
    numerical_ids: tuple[str, ...],
) -> PointProcessScoreEvidence:
    return PointProcessScoreEvidence(
        protocol_sha256=protocol_sha256,
        model_id=model_id,
        model_variant_id=variant,
        parameter_snapshot_id=_sha(f"parameter/{model_id}/{snapshot_id}"),
        snapshot_id=snapshot_id,
        fit_end_utc="2020-01-01T00:00:00Z",
        assessment_start_utc="2021-01-01T00:00:00Z",
        assessment_end_utc="2022-01-01T00:00:00Z",
        selected_mc=3.0,
        target_event_ids=(f"event/{snapshot_id}",),
        event_log_intensities=np.asarray([event_log_intensity], dtype=np.float64),
        compensator=1.0,
        numerical_gate_evidence_ids=numerical_ids,
    )


def _evidence(
    protocol_sha256: str,
) -> tuple[
    AuditedBackgroundModelEvidence,
    AuditedBackgroundModelEvidence,
    AuditedBackgroundModelEvidence,
    tuple[ETASSnapshotAttempt, ...],
]:
    uniform_pairs: list[PairedInformationGainEvidence] = []
    spatial_pairs: list[PairedInformationGainEvidence] = []
    etas_pairs: dict[str, PairedInformationGainEvidence] = {}
    attempts: list[ETASSnapshotAttempt] = []
    for snapshot_id in EXPECTED_SNAPSHOTS:
        uniform = _score(
            protocol_sha256=protocol_sha256,
            model_id="uniform_poisson",
            variant="uniform_poisson/frozen-v1",
            snapshot_id=snapshot_id,
            event_log_intensity=0.0,
            numerical_ids=(_sha(f"uniform-gate/{snapshot_id}"),),
        )
        uniform_pairs.append(
            PairedInformationGainEvidence.build(candidate=uniform, uniform=uniform)
        )
        spatial = _score(
            protocol_sha256=protocol_sha256,
            model_id="spatial_poisson",
            variant="spatial_poisson/kde-50km",
            snapshot_id=snapshot_id,
            event_log_intensity=0.2,
            numerical_ids=(
                _sha(f"spatial-local/{snapshot_id}"),
                _sha("spatial-global/50km"),
            ),
        )
        spatial_pairs.append(
            PairedInformationGainEvidence.build(candidate=spatial, uniform=uniform)
        )
        failed = snapshot_id == "fold_2"
        paired: PairedInformationGainEvidence | None = None
        if not failed:
            etas = _score(
                protocol_sha256=protocol_sha256,
                model_id="etas",
                variant="etas/frozen-v1",
                snapshot_id=snapshot_id,
                event_log_intensity=0.3,
                numerical_ids=(
                    _sha(f"etas-parameter/{snapshot_id}"),
                    _sha(f"etas-kde/{snapshot_id}"),
                    _sha(f"etas-grid/{snapshot_id}"),
                ),
            )
            paired = PairedInformationGainEvidence.build(candidate=etas, uniform=uniform)
            etas_pairs[snapshot_id] = paired
        synthetic_attempt = _Attempt(
            definition=_Definition(snapshot_id),
            selected_mc=3.0,
            fit_selection={"snapshot_id": snapshot_id, "role": "fit"},
            score_selection=_SelectionAudit(
                snapshot_id=snapshot_id,
                role="score",
                target_event_ids=(f"event/{snapshot_id}",),
            ),
            fit_result=(
                _FitResult(_Stability(False, ("optimizer convergence",), 2.0))
                if failed
                else _FitResult(_Stability(True, (), 1.0))
            ),
            parameter_snapshot_id=(_sha(f"parameter/etas/{snapshot_id}") if not failed else None),
            grid_gate_evidence=(
                None if failed else _GridGate(True, _sha(f"etas-grid-gate/{snapshot_id}"))
            ),
            paired_evidence=paired,
            failure_reasons=("numerical stability: optimizer convergence",) if failed else (),
        )
        attempts.append(cast(ETASSnapshotAttempt, synthetic_attempt))

    uniform_evidence = AuditedBackgroundModelEvidence(
        model_id="uniform_poisson",
        model_variant_id="uniform_poisson/frozen-v1",
        protocol_sha256=protocol_sha256,
        development_folds=tuple(uniform_pairs[:4]),
        validation=uniform_pairs[4],
        failed_snapshot_reasons=(),
    )
    spatial_evidence = AuditedBackgroundModelEvidence(
        model_id="spatial_poisson",
        model_variant_id="spatial_poisson/kde-50km",
        protocol_sha256=protocol_sha256,
        development_folds=tuple(spatial_pairs[:4]),
        validation=spatial_pairs[4],
        failed_snapshot_reasons=(),
    )
    etas_evidence = AuditedBackgroundModelEvidence(
        model_id="etas",
        model_variant_id="etas/frozen-v1",
        protocol_sha256=protocol_sha256,
        development_folds=tuple(
            etas_pairs[snapshot_id]
            for snapshot_id in EXPECTED_SNAPSHOTS[:4]
            if snapshot_id in etas_pairs
        ),
        validation=etas_pairs["final_validation"],
        failed_snapshot_reasons=(("fold_2", ("numerical stability: optimizer convergence",)),),
    )
    return uniform_evidence, spatial_evidence, etas_evidence, tuple(attempts)


def _future(resources: FutureWorkerResources | None = None) -> _CompletedFuture:
    counts = (1,) * 128
    quantiles = tuple(
        FutureCountQuantile(probability=probability, count=1.0)
        for probability in FUTURE_QUANTILE_PROBABILITIES
    )
    horizons = tuple(
        FutureHorizonSummary(
            horizon_days=horizon_days,
            replicate_counts=counts,
            mean_count=1.0,
            quantiles=quantiles,
            grids=tuple(
                SparseGridExpectedCounts(
                    cell_size_km=cell_size_km,
                    cells=(
                        SparseExpectedCellCount(
                            cell_id=f"g{int(cell_size_km * 1000):08d}_r+0000000_c+0000000",
                            expected_count=1.0,
                        ),
                    ),
                    expected_total=1.0,
                )
                for cell_size_km in (50.0, 25.0, 12.5)
            ),
        )
        for horizon_days in FUTURE_HORIZONS_DAYS
    )
    issue = FutureIssueEnsemble(
        issue_date_local="2024-07-01",
        issue_id="validation/2024-07-01",
        replicate_count=128,
        quantile_probabilities=FUTURE_QUANTILE_PROBABILITIES,
        quantile_method=FUTURE_QUANTILE_METHOD,
        horizons=horizons,
    )
    return _CompletedFuture(
        ensembles=ValidationFutureEnsembles(
            issues=(issue,),
            resources=(
                resources
                if resources is not None
                else FutureWorkerResources(
                    detected_physical_cores=8,
                    reserve_physical_cores=2,
                    configured_max_workers=12,
                    effective_workers=6,
                )
            ),
        ),
        not_run_reason=None,
    )


def _result(config: BackgroundConfig) -> _SyntheticResult:
    protocol_sha256 = _protocol_sha(config)
    uniform, spatial, etas, attempts = _evidence(protocol_sha256)
    audited = (uniform, spatial, etas)
    g1 = assess_audited_g1(audited)
    selection = select_audited_background_model(audited)
    completeness = tuple(
        _Completeness(_Definition(snapshot_id), selected_mc=3.0)
        for snapshot_id in EXPECTED_SNAPSHOTS
    )
    poisson_snapshots = tuple(
        _PoissonSnapshot(
            definition=_Definition(snapshot_id),
            grid_gate_evidence=({"snapshot_id": snapshot_id, "passed": True},),
            kde_family=(
                (25.0, _KDEModel(0.8, 0.1, _Mixture(2))),
                (50.0, _KDEModel(0.9, 0.1, _Mixture(2))),
            ),
            rate_per_day=0.1,
            selected_mc=3.0,
            target_event_ids=(f"event/{snapshot_id}",),
            training_duration_days=3650.0,
            training_event_count=2,
            training_event_ids=(f"train/{snapshot_id}/1", f"train/{snapshot_id}/2"),
            training_evidence_id=_sha(f"training/{snapshot_id}"),
            uniform_model=_UniformModel(0.1, 200.0),
        )
        for snapshot_id in EXPECTED_SNAPSHOTS
    )
    grid_50 = _GridResolution(
        50.0,
        (_GridCell("g50000000_r+0000000_c+0000000", 0, 0, 25.0, 25.0, 200.0),),
        200.0,
    )
    grid_25 = _GridResolution(
        25.0,
        (
            _GridCell("g25000000_r+0000000_c+0000000", 0, 0, 12.5, 12.5, 100.0),
            _GridCell("g25000000_r+0000000_c+0000001", 0, 1, 37.5, 12.5, 100.0),
        ),
        200.0,
    )
    grid_12_5 = _GridResolution(
        12.5,
        (
            _GridCell("g12500000_r+0000000_c+0000000", 0, 0, 6.25, 6.25, 50.0),
            _GridCell("g12500000_r+0000000_c+0000001", 0, 1, 18.75, 6.25, 50.0),
            _GridCell("g12500000_r+0000000_c+0000002", 0, 2, 31.25, 6.25, 50.0),
            _GridCell("g12500000_r+0000000_c+0000003", 0, 3, 43.75, 6.25, 50.0),
        ),
        200.0,
    )
    integration_grids = _IntegrationGrids(200.0, (grid_50, grid_25, grid_12_5))
    render = ConditionalIntensityRender(
        png_bytes=b"\x89PNG\r\n\x1a\nsynthetic-stage-2",
        panel_titles=("Background", "Triggering", "Total"),
        colorbar_label="Conditional intensity (events km^-2 day^-1)",
        footer_label="Conditional intensity, not absolute earthquake probability",
        png_metadata=(("Software", "SeismoFlux"),),
        width_px=1000,
        height_px=400,
        dpi=100,
        font_family="DejaVu Sans",
        colormap_name="seismoflux_conditional_intensity",
    )
    return _SyntheticResult(
        protocol_sha256=protocol_sha256,
        regressions=_Regressions(True, True),
        completeness=completeness,
        poisson=_PoissonResult(
            protocol_sha256=protocol_sha256,
            snapshots=poisson_snapshots,
            selected_bandwidth_km=50.0,
            uniform_evidence=uniform,
            spatial_evidence=spatial,
            bandwidth_fold_audits=({"bandwidth_km": 50.0, "passed": True},),
            bandwidth_selection={"selected_bandwidth_km": 50.0},
            pre_score_gate_evidence={"passed_bandwidths_km": (50.0,)},
        ),
        etas=_ETASResult(
            protocol_sha256=protocol_sha256,
            model_variant_id="etas/frozen-v1",
            attempts=attempts,
            etas_evidence=etas,
        ),
        g1=g1,
        selection=selection,
        bootstrap=_Bootstrap(
            spatial_poisson=_BootstrapOutcome(
                model_id="spatial_poisson",
                interval=_BootstrapInterval(
                    replications=2000,
                    confidence_level=0.95,
                    point_estimate=0.2,
                    lower=0.1,
                    upper=0.3,
                    replicate_values=(0.2,) * 2000,
                ),
                not_run_reason=None,
            ),
            etas=_BootstrapOutcome(
                model_id="etas",
                interval=_BootstrapInterval(
                    replications=2000,
                    confidence_level=0.95,
                    point_estimate=0.3,
                    lower=0.2,
                    upper=0.4,
                    replicate_values=(0.3,) * 2000,
                ),
                not_run_reason=None,
            ),
        ),
        horizons=_Horizons(
            spatial_poisson=_HorizonOutcome(
                model_id="spatial_poisson",
                comparisons=tuple({"comparison_index": index} for index in range(15)),
                not_run_reason=None,
            ),
            etas=_HorizonOutcome(
                model_id="etas",
                comparisons=tuple({"comparison_index": index} for index in range(15)),
                not_run_reason=None,
            ),
        ),
        future=_future(),
        representative=_Representative(
            issue_date_local="2025-06-26",
            issue_time_utc="2025-06-25T16:00:00Z",
            selected_model_id="spatial_poisson",
            selected_model_variant_id="spatial_poisson/kde-50km",
            primary_grid_cell_size_km=25.0,
            cell_ids=tuple(cell.cell_id for cell in grid_25.cells),
            rows=tuple(cell.row for cell in grid_25.cells),
            columns=tuple(cell.column for cell in grid_25.cells),
            representative_x_km=np.asarray(
                [cell.representative_x_km for cell in grid_25.cells],
                dtype=np.float64,
            ),
            representative_y_km=np.asarray(
                [cell.representative_y_km for cell in grid_25.cells],
                dtype=np.float64,
            ),
            clipped_area_km2=np.asarray(
                [cell.clipped_area_km2 for cell in grid_25.cells],
                dtype=np.float64,
            ),
            eligible_parent_event_ids=(),
            background_intensity=np.asarray([0.1, 0.2], dtype=np.float64),
            triggering_intensity=np.asarray([0.0, 0.0], dtype=np.float64),
            total_intensity=np.asarray([0.1, 0.2], dtype=np.float64),
            render=render,
        ),
        resources=_Resources(8, 2, 6),
        integration_grids=integration_grids,
    )


class _Bundle(Protocol):
    def address_parameters(self) -> dict[str, object]: ...

    def artifact_files(self) -> tuple[ArtifactFile, ...]: ...


def _bundle_bytes(bundle: _Bundle) -> tuple[tuple[str, bytes], ...]:
    return tuple((item.relative_path.value, item.payload) for item in bundle.artifact_files())


def _arrow_table(bundle: _Bundle, relative_path: str) -> pa.Table:
    payload = next(
        item.payload
        for item in bundle.artifact_files()
        if item.relative_path.value == relative_path
    )
    with ipc.open_file(pa.BufferReader(payload)) as reader:
        return reader.read_all()


def _json_content(bundle: _Bundle, relative_path: str) -> dict[str, object]:
    payload = next(
        item.payload
        for item in bundle.artifact_files()
        if item.relative_path.value == relative_path
    )
    document = json.loads(payload)
    return cast(dict[str, object], document["content"])


def _identity() -> RepositoryIdentity:
    commit = "a" * 40
    return RepositoryIdentity(
        code_commit=commit,
        branch="codex/phase-2",
        upstream="origin/codex/phase-2",
        upstream_commit=commit,
        freeze_tag="v0.2.0-background-protocol",
        freeze_tag_commit="1" * 40,
        git_available=True,
        worktree_clean=True,
        tag_is_ancestor=True,
        upstream_matches_head=True,
    )


def _seal(config: BackgroundConfig) -> ExecutionSeal:
    return ExecutionSeal(
        repository=_identity(),
        protocol_sha256=_protocol_sha(config),
        input_hashes=tuple(
            sorted(
                {
                    "environment_lock": config.inputs.environment_lock_sha256,
                    "data_catalog": config.inputs.data_catalog_sha256,
                    "earthquake_dataset": config.inputs.earthquake_dataset_sha256,
                    "study_area": config.inputs.study_area_sha256,
                    "issue_manifest": config.inputs.issue_manifest_sha256,
                    "production_fixture": (config.numerical_regression.production_fixture_sha256),
                    "oracle_metadata": config.numerical_regression.oracle_metadata_sha256,
                }.items()
            )
        ),
    )


def test_build_retains_all_attempts_failure_and_complete_four_bundles(
    background: BackgroundConfig,
) -> None:
    result = _result(background)
    deliverables = build_background_deliverables(
        background,
        cast(BackgroundPipelineResult, result),
    )

    assert len(deliverables.model_attempts) == 15
    assert tuple(
        (attempt.model_id, attempt.snapshot_id) for attempt in deliverables.model_attempts
    ) == tuple(
        (model_id, snapshot_id)
        for model_id in ("uniform_poisson", "spatial_poisson", "etas")
        for snapshot_id in EXPECTED_SNAPSHOTS
    )
    failed = next(
        attempt
        for attempt in deliverables.model_attempts
        if attempt.model_id == "etas" and attempt.snapshot_id == "fold_2"
    )
    assert failed.status == "failed"
    assert failed.failure_reasons == ("numerical stability: optimizer convergence",)
    assert any(
        gate.gate_id == "numerical_stability" and gate.status == "failed" for gate in failed.gates
    )
    assert deliverables.g1.passed is True
    assert deliverables.g1.passing_models == ("spatial_poisson",)
    assert deliverables.selection.selected_model_id == "spatial_poisson"
    assert deliverables.selection.eligible_model_ids == (
        "uniform_poisson",
        "spatial_poisson",
    )
    assert deliverables.stage3_allowed is True
    assert deliverables.scientific_summary.final_selected_mc == 3.0
    assert deliverables.scientific_summary.selected_kde_bandwidth_km == 50.0
    assert len(deliverables.scientific_summary.snapshots) == 15
    assert deliverables.scientific_summary.validation_bootstrap[0].replications == 2000

    assert tuple(path for path, _ in _bundle_bytes(deliverables.processed)) == (
        "completeness.json",
        "integration_grid.json",
        "integration_grids.arrow",
        "numerical_regressions.json",
    )
    assert tuple(path for path, _ in _bundle_bytes(deliverables.model)) == (
        "attempts.json",
        "etas.json",
        "poisson.json",
    )
    assert tuple(path for path, _ in _bundle_bytes(deliverables.backtest)) == (
        "audited_scores.json",
        "bootstrap.json",
        "g1.json",
        "horizons.json",
        "scientific_summary.json",
        "selection.json",
    )
    tracked_summary = _json_content(deliverables.backtest, "scientific_summary.json")
    tracked_summary_text = json.dumps(tracked_summary, sort_keys=True)
    tracked_snapshots = cast(list[dict[str, object]], tracked_summary["snapshots"])
    assert len(tracked_snapshots) == 15
    assert set(tracked_snapshots[0]) == {
        "information_gain_nats_per_event",
        "model_id",
        "score_id",
        "snapshot_id",
        "status",
        "target_event_count",
    }
    assert "replicate_values" not in tracked_summary_text
    assert "event_log_intensities" not in tracked_summary_text
    assert tuple(path for path, _ in _bundle_bytes(deliverables.experiment)) == (
        "conditional_intensity/conditional_intensity.json",
        "conditional_intensity/conditional_intensity.png",
        "conditional_intensity/representative_intensity.arrow",
        "future/future_sparse_counts.arrow",
        "future/future_summary.json",
    )
    integration = _arrow_table(deliverables.processed, "integration_grids.arrow")
    assert integration.num_rows == 7
    assert integration.column_names == [
        "cell_size_km",
        "cell_id",
        "row",
        "column",
        "representative_x_km",
        "representative_y_km",
        "clipped_area_km2",
    ]
    integration_summary = _json_content(deliverables.processed, "integration_grid.json")
    assert integration_summary["equal_area_crs"] == background.integration.equal_area_crs
    assert cast(dict[str, object], integration_summary["units"])["clipped_area_km2"] == "km2"
    representative = _arrow_table(
        deliverables.experiment,
        "conditional_intensity/representative_intensity.arrow",
    )
    assert representative.num_rows == 2
    assert representative.column("cell_id").to_pylist() == list(result.representative.cell_ids)
    representative_summary = _json_content(
        deliverables.experiment,
        "conditional_intensity/conditional_intensity.json",
    )
    assert cast(dict[str, object], representative_summary["units"])["intensity"] == (
        "expected_events_per_day_per_km2"
    )
    future = _arrow_table(deliverables.experiment, "future/future_sparse_counts.arrow")
    assert future.num_rows == 5 * (128 + 3)
    assert future.column("record_kind").to_pylist().count("replicate_count") == 5 * 128


def test_g1_and_selection_cannot_be_declared_independently_of_evidence(
    background: BackgroundConfig,
) -> None:
    result = _result(background)
    false_g1 = AuditedG1Assessment(
        passed=False,
        passing_model_variants=(),
        model_pass=(
            ("spatial_poisson", "spatial_poisson/kde-50km", False),
            ("etas", "etas/frozen-v1", False),
        ),
    )
    with pytest.raises(ValueError, match="G1 conclusion differs"):
        build_background_deliverables(
            background,
            cast(BackgroundPipelineResult, replace(result, g1=false_g1)),
        )

    false_selection = AuditedModelSelection(
        validation_best_model_variant_id="uniform_poisson/frozen-v1",
        selected_model_variant_id="uniform_poisson/frozen-v1",
        paired_standard_errors=(),
        excluded_model_variants=(),
    )
    with pytest.raises(ValueError, match="selection differs"):
        build_background_deliverables(
            background,
            cast(
                BackgroundPipelineResult,
                replace(result, selection=false_selection),
            ),
        )


def test_explicit_future_skip_still_delivers_failed_etas_audit(
    background: BackgroundConfig,
) -> None:
    result = replace(
        _result(background),
        future=_SkippedFuture(None, "final ETAS snapshot did not pass its frozen gates"),
    )
    deliverables = build_background_deliverables(
        background,
        cast(BackgroundPipelineResult, result),
    )

    experiment_paths = tuple(path for path, _ in _bundle_bytes(deliverables.experiment))
    assert "future/future_sparse_counts.arrow" not in experiment_paths
    summary = _json_content(deliverables.experiment, "future/future_summary.json")
    assert summary == {
        "arrow": None,
        "not_run_reason": "final ETAS snapshot did not pass its frozen gates",
        "status": "skipped",
    }
    assert deliverables.scientific_summary.future.status == "skipped"
    assert deliverables.scientific_summary.future.issue_count == 0
    assert any(
        attempt.model_id == "etas" and attempt.status == "failed"
        for attempt in deliverables.model_attempts
    )
    assert "etas" not in deliverables.selection.eligible_model_ids


def test_bundle_bytes_addresses_and_registry_are_deterministic(
    tmp_path: Path,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = cast(BackgroundPipelineResult, _result(background))
    first = build_background_deliverables(background, result)
    different_machine = replace(
        _result(background),
        future=_future(
            FutureWorkerResources(
                detected_physical_cores=16,
                reserve_physical_cores=2,
                configured_max_workers=12,
                effective_workers=12,
            )
        ),
        resources=_Resources(16, 2, 12),
    )
    second = build_background_deliverables(
        background,
        cast(BackgroundPipelineResult, different_machine),
    )
    for first_bundle, second_bundle in zip(
        (first.processed, first.model, first.backtest, first.experiment),
        (second.processed, second.model, second.backtest, second.experiment),
        strict=True,
    ):
        assert _bundle_bytes(first_bundle) == _bundle_bytes(second_bundle)
        assert first_bundle.address_parameters() == second_bundle.address_parameters()

    seal = _seal(background)
    revalidated: list[ExecutionSeal] = []

    def revalidate(
        project_root: Path,
        config: BackgroundConfig,
        expected: ExecutionSeal,
        *,
        runner: GitCommandRunner,
    ) -> ExecutionSeal:
        del project_root, config, runner
        revalidated.append(expected)
        return expected

    monkeypatch.setattr(deliverables_module, "require_execution_seal_unchanged", revalidate)
    published = publish_background_deliverables(tmp_path, background, seal, first)
    again = publish_background_deliverables(tmp_path, background, seal, second)

    assert tuple(item.bundle_kind for item in published.bundle_publications) == (
        "processed",
        "model",
        "backtest",
        "experiment",
    )
    assert tuple(item.artifact.artifact_id for item in published.bundle_publications) == tuple(
        item.artifact.artifact_id for item in again.bundle_publications
    )
    assert all(not item.artifact.created for item in again.bundle_publications)
    assert registry_payload_bytes(published.registry) == registry_payload_bytes(again.registry)
    assert published.registry.scientific_summary == first.scientific_summary
    assert published.registry.code_commit == seal.repository.code_commit
    assert published.registry.input_hashes == seal.input_hash_mapping()
    assert revalidated == [seal, seal]
    assert not (tmp_path / background.outputs.registry).exists()
    assert not (tmp_path / background.outputs.report).exists()


def test_publish_rejects_a_seal_not_bound_to_the_frozen_inputs(
    tmp_path: Path,
    background: BackgroundConfig,
) -> None:
    deliverables = build_background_deliverables(
        background,
        cast(BackgroundPipelineResult, _result(background)),
    )
    seal = _seal(background)
    changed_hashes = dict(seal.input_hashes)
    changed_hashes["earthquake_dataset"] = "f" * 64
    changed = ExecutionSeal(
        repository=seal.repository,
        protocol_sha256=seal.protocol_sha256,
        input_hashes=tuple(sorted(changed_hashes.items())),
    )

    with pytest.raises(ValueError, match="input hashes differ"):
        publish_background_deliverables(tmp_path, background, changed, deliverables)


def test_expected_scientific_failure_publishes_four_auditable_negative_bundles(
    tmp_path: Path,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = _result(background)
    failure = BackgroundPipelineFailure(
        protocol_sha256=complete.protocol_sha256,
        failure_stage="completeness",
        failure_reason_code="no_eligible_temporal_stratum",
        failure_reasons=("no eligible temporal completeness stratum",),
        regressions=cast(NumericalRegressionEvidence, complete.regressions),
        completeness=(),
        pre_score_gate_evidence=None,
        resources=cast(PipelineResourceEvidence, complete.resources),
        integration_grids=cast(IntegrationGridEvidence, complete.integration_grids),
    )

    deliverables = build_background_deliverables(background, failure)

    assert len(deliverables.model_attempts) == 15
    assert all(attempt.status == "not_run" for attempt in deliverables.model_attempts)
    assert all(
        attempt.gates[0].status == "not_applicable" for attempt in deliverables.model_attempts
    )
    assert deliverables.scientific_summary.outcome_status == "scientific_gate_failed"
    assert deliverables.scientific_summary.failure is not None
    assert (
        deliverables.scientific_summary.failure.failure_reason_code
        == "no_eligible_temporal_stratum"
    )
    assert deliverables.g1.status == "not_evaluable"
    assert deliverables.selection.status == "not_evaluable"
    assert not deliverables.stage3_allowed
    experiment_paths = tuple(path for path, _ in _bundle_bytes(deliverables.experiment))
    assert experiment_paths == (
        "conditional_intensity/status.json",
        "failure.json",
        "future/status.json",
    )

    different_resources = replace(
        failure,
        resources=cast(PipelineResourceEvidence, _Resources(16, 2, 12)),
    )
    different_deliverables = build_background_deliverables(
        background,
        different_resources,
    )
    for first_bundle, second_bundle in zip(
        (
            deliverables.processed,
            deliverables.model,
            deliverables.backtest,
            deliverables.experiment,
        ),
        (
            different_deliverables.processed,
            different_deliverables.model,
            different_deliverables.backtest,
            different_deliverables.experiment,
        ),
        strict=True,
    ):
        assert _bundle_bytes(first_bundle) == _bundle_bytes(second_bundle)
        assert first_bundle.address_parameters() == second_bundle.address_parameters()
    assert (
        deliverables.scientific_summary.failure == different_deliverables.scientific_summary.failure
    )

    monkeypatch.setattr(
        deliverables_module,
        "require_execution_seal_unchanged",
        lambda *_args, **_kwargs: _seal(background),
    )
    published = publish_background_deliverables(
        tmp_path,
        background,
        _seal(background),
        deliverables,
    )

    assert len(published.bundle_publications) == 4
    assert published.registry.scientific_summary == deliverables.scientific_summary
    assert published.registry.g1.status == "not_evaluable"
    assert published.registry.selection.status == "not_evaluable"
    assert not published.registry.stage3_allowed
