from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from shapely.geometry import box

import seismoflux.background.pipeline_local_support_etas as local_etas
from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.config import load_background_protocol
from seismoflux.background.etas_fit import (
    ETASFitResult,
    ETASLikelihoodProblem,
    ETASModelSpec,
    ETASParameters,
    HessianAudit,
    StabilityAudit,
    observed_hessian_delta_uncertainty,
)
from seismoflux.background.evidence import (
    AuditedBackgroundModelEvidence,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
)
from seismoflux.background.grid import (
    GRID_CELL_SIZES_KM,
    EqualAreaGridFamily,
    build_clipped_grid,
)
from seismoflux.background.poisson import (
    SpatialPoissonModel,
    SpatialQuadrature,
    fit_spatial_poisson_family,
)
from seismoflux.background.workflow import SnapshotDefinition, build_snapshot_definitions

PROTOCOL = "c7d6488bd97f0017867573c8b99230d79091412322652af25badfe732606e76a"
AUTHORIZATION = "d" * 64
DOMAIN = "e" * 64
SUPPORT = "local-support-0123456789abcdef"


def _grid_family() -> EqualAreaGridFamily:
    geometry = box(1_000.0, 1_000.0, 2_000.0, 2_000.0)
    return EqualAreaGridFamily(
        study_area_equal_area=geometry,
        grids=tuple(
            build_clipped_grid(geometry, cell_size_km=cell_size) for cell_size in GRID_CELL_SIZES_KM
        ),
    )


def _catalog() -> EarthquakeCatalog:
    rows = (
        ("training", "2001-01-01T00:00:00Z", 3.5, True, True, 1.50, 1.50),
        ("unsupported-high", "2002-01-01T00:00:00Z", 4.5, True, True, 1.55, 1.50),
        ("unsupported-low", "2003-01-01T00:00:00Z", 4.1, True, True, 1.55, 1.50),
        ("external", "2003-06-01T00:00:00Z", 3.5, False, True, 2.50, 1.50),
        ("target-fold-1", "2006-06-01T00:00:00Z", 3.5, True, True, 1.20, 1.20),
        ("target-fold-2", "2011-06-01T00:00:00Z", 3.5, True, True, 1.80, 1.20),
        ("target-fold-3", "2016-06-01T00:00:00Z", 3.5, True, True, 1.20, 1.80),
        ("target-fold-4", "2021-06-01T00:00:00Z", 3.5, True, True, 1.80, 1.80),
        ("target-final", "2025-01-01T00:00:00Z", 3.5, True, True, 1.50, 1.50),
    )
    count = len(rows)
    days = np.asarray([utc_timestamp_to_day(row[1]) for row in rows])
    return EarthquakeCatalog(
        event_id=np.asarray([row[0] for row in rows], dtype=np.str_),
        origin_day=days,
        available_day=days,
        longitude=np.full(count, 105.0),
        latitude=np.full(count, 35.0),
        x_km=np.asarray([row[5] for row in rows]),
        y_km=np.asarray([row[6] for row in rows]),
        magnitude=np.asarray([row[2] for row in rows]),
        inside_study_area=np.asarray([row[3] for row in rows]),
        inside_external_buffer=np.asarray([row[4] for row in rows]),
    )


def _stable_fit(
    problem: ETASLikelihoodProblem,
    spec: ETASModelSpec,
    **_: object,
) -> ETASFitResult:
    del problem
    parameters = ETASParameters(
        background_rate_per_day=0.02,
        productivity_k=0.005,
        alpha=0.5,
        c_days=1.0,
        p=1.2,
    )
    identity = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(5)) for row in range(5)
    )
    stability = StabilityAudit(
        stable=True,
        converged_start_count=5,
        best_three_relative_objective_range=0.0,
        best_three_transformed_parameter_range=0.0,
        hessian=HessianAudit(
            success=True,
            minimum_eigenvalue=1.0,
            condition_number=1.0,
            matrix=identity,
            failure_reason=None,
        ),
        failure_reasons=(),
    )
    return ETASFitResult(
        best_parameters=parameters,
        best_objective=10.0,
        start_results=(),
        stability=stability,
        uncertainty=observed_hessian_delta_uncertainty(parameters, stability, spec),
    )


@dataclass(frozen=True)
class _FakePoissonSnapshot:
    definition: SnapshotDefinition
    support_id: str
    selected_mc: float
    supported_area_km2: float
    compensator_domain_id: str
    authorization_id: str
    training_event_ids: tuple[str, ...]
    training_evidence_id: str
    model: SpatialPoissonModel

    def kde_model(self, bandwidth_km: float) -> SpatialPoissonModel:
        if bandwidth_km != self.model.bandwidth_km:
            raise KeyError(bandwidth_km)
        return self.model

    def gate_for(self, bandwidth_km: float) -> Any:
        if bandwidth_km != self.model.bandwidth_km:
            raise KeyError(bandwidth_km)
        return SimpleNamespace(numerical_evidence_id="f" * 64)


def _score(definition: SnapshotDefinition, target_id: str) -> PointProcessScoreEvidence:
    return PointProcessScoreEvidence(
        protocol_sha256=PROTOCOL,
        model_id="uniform_poisson",
        model_variant_id="uniform/local",
        parameter_snapshot_id="a" * 64,
        snapshot_id=definition.snapshot_id,
        fit_end_utc=definition.fit_end_utc,
        assessment_start_utc=definition.assessment_start_utc,
        assessment_end_utc=definition.assessment_end_utc,
        selected_mc=3.0,
        target_event_ids=(target_id,),
        event_log_intensities=np.asarray([-4.0]),
        compensator=1.0,
        numerical_gate_evidence_ids=("b" * 64,),
        support_id=SUPPORT,
        supported_area_km2=1.0,
        compensator_domain_id=DOMAIN,
        authorization_id=AUTHORIZATION,
    )


def _inputs() -> tuple[EarthquakeCatalog, Any, Any]:
    config = load_background_protocol("configs/background_local_support.yaml")
    definitions = build_snapshot_definitions(config)
    catalog = _catalog()
    family = _grid_family()
    quadrature = SpatialQuadrature.from_grid(family.at(12.5))
    model = fit_spatial_poisson_family(
        np.asarray([1.5]),
        np.asarray([1.5]),
        training_duration_days=1_000.0,
        normalization_quadrature=quadrature,
        bandwidths_km=(75.0,),
    )[75.0]
    target_ids = (*tuple(f"target-fold-{index}" for index in range(1, 5)), "target-final")
    scores = tuple(
        _score(definition, target_id)
        for definition, target_id in zip(definitions, target_ids, strict=True)
    )
    pairs = tuple(
        PairedInformationGainEvidence.build(candidate=score, uniform=score) for score in scores
    )
    uniform = AuditedBackgroundModelEvidence(
        model_id="uniform_poisson",
        model_variant_id="uniform/local",
        protocol_sha256=PROTOCOL,
        development_folds=pairs[:4],
        validation=pairs[4],
        failed_snapshot_reasons=(),
    )
    snapshots = tuple(
        _FakePoissonSnapshot(
            definition=definition,
            support_id=SUPPORT,
            selected_mc=3.0,
            supported_area_km2=1.0,
            compensator_domain_id=DOMAIN,
            authorization_id=AUTHORIZATION,
            training_event_ids=("training",),
            training_evidence_id="c" * 64,
            model=model,
        )
        for definition in definitions
    )
    poisson = SimpleNamespace(
        protocol_sha256=PROTOCOL,
        snapshots=snapshots,
        uniform_evidence=uniform,
        selected_bandwidth_km=75.0,
    )
    runtime_snapshots = []
    for definition in definitions:
        supported = np.asarray(catalog.inside_study_area, dtype=np.bool_).copy()
        has_unsupported = definition.snapshot_id in {"fold_1", "fold_3"}
        if has_unsupported:
            supported[1:3] = False
        primary = np.asarray(supported & (catalog.magnitude >= 3.0), dtype=np.bool_)
        if has_unsupported:
            primary[1] = True
        support = SimpleNamespace(
            support_id=SUPPORT,
            common_mc=3.0,
            retained_area_m2=1_000_000.0,
            retained_selected_aki_b_value=1.0,
            cells=(SimpleNamespace(status="unsupported" if has_unsupported else "supported"),),
        )
        runtime_snapshots.append(
            SimpleNamespace(
                snapshot_id=definition.snapshot_id,
                support=support,
                grid_family=family,
                compensator_domain_id=DOMAIN,
                supported_mask=supported,
                etas_primary_parent_role_mask=primary,
            )
        )
    runtime = SimpleNamespace(
        event_ids=tuple(str(value) for value in catalog.event_id),
        snapshots=tuple(runtime_snapshots),
    )
    return catalog, cast(Any, runtime), cast(Any, poisson)


def test_local_etas_separates_parent_roles_and_refits_required_sensitivity(
    monkeypatch: Any,
) -> None:
    config = load_background_protocol("configs/background_local_support.yaml")
    catalog, runtime, poisson = _inputs()
    monkeypatch.setattr(local_etas, "require_background_scoring_authorized", lambda *_: None)
    authorization = cast(Any, SimpleNamespace(authorization_id=AUTHORIZATION))

    result = local_etas.run_local_support_etas_pipeline(
        config,
        catalog,
        runtime,
        poisson,
        authorization,
        fit_function=_stable_fit,
    )

    assert result.primary.failed_snapshot_reasons == ()
    assert result.primary.evidence.eligible_for_selection is True
    assert tuple(item.definition.snapshot_id for item in result.sensitivity_attempts) == (
        "fold_1",
        "fold_3",
    )
    assert tuple(item[0] for item in result.sensitivity_not_applicable) == (
        "fold_2",
        "fold_4",
        "final_validation",
    )

    fold_1 = result.primary.attempt("fold_1")
    sensitivity = result.sensitivity_attempts[0]
    assert "unsupported-high" in fold_1.fit_selection.parent_event_ids
    assert "unsupported-low" not in fold_1.fit_selection.parent_event_ids
    assert "external" in fold_1.fit_selection.parent_event_ids
    assert "unsupported-high" not in sensitivity.fit_selection.parent_event_ids
    assert "external" in sensitivity.fit_selection.parent_event_ids
    assert sensitivity.model_variant_id.endswith("/exclude_unsupported_conditional_parents")
    for attempt in (*result.primary.attempts, *result.sensitivity_attempts):
        assert attempt.succeeded
        assert attempt.paired_evidence is not None
        candidate = attempt.paired_evidence.candidate
        assert candidate.support_id == SUPPORT
        assert candidate.compensator_domain_id == DOMAIN
        assert candidate.authorization_id == AUTHORIZATION


def test_late_reported_physical_target_remains_a_target_but_never_an_early_parent(
    monkeypatch: Any,
) -> None:
    config = load_background_protocol("configs/background_local_support.yaml")
    catalog, runtime, poisson = _inputs()
    final_index = int(np.flatnonzero(catalog.event_id == "target-final")[0])
    delayed_availability = np.asarray(catalog.available_day, dtype=np.float64).copy()
    delayed_availability[final_index] = utc_timestamp_to_day("2025-07-15T00:00:00Z")
    delayed_catalog = replace(catalog, available_day=delayed_availability)
    monkeypatch.setattr(local_etas, "require_background_scoring_authorized", lambda *_: None)
    authorization = cast(Any, SimpleNamespace(authorization_id=AUTHORIZATION))

    result = local_etas.run_local_support_etas_pipeline(
        config,
        delayed_catalog,
        runtime,
        poisson,
        authorization,
        fit_function=_stable_fit,
    )

    final = result.primary.attempt("final_validation")
    assert final.succeeded
    assert final.score_selection.target_event_ids == ("target-final",)
    assert "target-final" in final.score_selection.parent_event_ids
    assert final.paired_evidence is not None
    assert final.paired_evidence.candidate.target_event_ids == ("target-final",)
