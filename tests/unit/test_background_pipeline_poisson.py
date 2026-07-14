from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime

import numpy as np
import pytest
from shapely.geometry import box

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.completeness import (
    CompletenessAnalysis,
    CompletenessAudit,
)
from seismoflux.background.config import BackgroundConfig, load_background_config
from seismoflux.background.evidence import EXPECTED_SNAPSHOTS
from seismoflux.background.grid import (
    GRID_CELL_SIZES_KM,
    EqualAreaGridFamily,
    build_clipped_grid,
)
from seismoflux.background.pipeline_poisson import (
    PoissonKDEPipelineError,
    run_poisson_kde_pipeline,
)
from seismoflux.background.poisson import FROZEN_BANDWIDTHS_KM
from seismoflux.background.workflow import (
    CompletenessSnapshot,
    SnapshotDefinition,
    build_snapshot_definitions,
)


def _grid_family() -> EqualAreaGridFamily:
    study_area = box(1_000.0, 1_000.0, 2_000.0, 2_000.0)
    grids = tuple(
        build_clipped_grid(study_area, cell_size_km=cell_size_km)
        for cell_size_km in GRID_CELL_SIZES_KM
    )
    return EqualAreaGridFamily(study_area_equal_area=study_area, grids=grids)


def _catalog(
    *,
    final_target_x_km: float = 1.2,
    omitted_target_id: str | None = None,
    include_training_event: bool = True,
) -> EarthquakeCatalog:
    rows = [
        ("training", "2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", 1.5, 1.5),
        ("target-fold-1", "2006-06-01T00:00:00Z", "2006-06-01T00:00:00Z", 1.2, 1.2),
        ("target-fold-2", "2011-06-01T00:00:00Z", "2011-06-01T00:00:00Z", 1.8, 1.2),
        ("target-fold-3", "2016-06-01T00:00:00Z", "2016-06-01T00:00:00Z", 1.2, 1.8),
        ("target-fold-4", "2021-06-01T00:00:00Z", "2021-06-01T00:00:00Z", 1.8, 1.8),
        (
            "delayed-final",
            "2025-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            final_target_x_km,
            1.5,
        ),
    ]
    selected = tuple(
        row
        for row in rows
        if row[0] != omitted_target_id and (include_training_event or row[0] != "training")
    )
    count = len(selected)
    return EarthquakeCatalog(
        event_id=np.asarray([row[0] for row in selected], dtype=np.str_),
        origin_day=np.asarray([utc_timestamp_to_day(row[1]) for row in selected]),
        available_day=np.asarray([utc_timestamp_to_day(row[2]) for row in selected]),
        longitude=np.full(count, 105.0),
        latitude=np.full(count, 35.0),
        x_km=np.asarray([row[3] for row in selected]),
        y_km=np.asarray([row[4] for row in selected]),
        magnitude=np.full(count, 3.5),
        inside_study_area=np.ones(count, dtype=np.bool_),
        inside_external_buffer=np.ones(count, dtype=np.bool_),
    )


def _analysis(definition: SnapshotDefinition) -> CompletenessAnalysis:
    cutoff = datetime.fromisoformat(definition.fit_end_utc.replace("Z", "+00:00"))
    return CompletenessAnalysis(
        cutoff_utc=cutoff,
        audit=CompletenessAudit(
            input_event_count=0,
            included_historical_inside_count=0,
            excluded_outside_count=0,
            excluded_pre_1970_count=0,
            excluded_future_origin_count=0,
            excluded_unavailable_count=0,
        ),
        temporal_blocks=(),
        spatial_strata=(),
        sparse_cell_resolutions=(),
        regime_changes=(),
        maximum_eligible_estimate=3.0,
        selected_mc=3.0,
        selected_event_count=1,
        selected_aki_b_value=1.0,
        sensitivities=(),
    )


def _completeness(config: BackgroundConfig) -> tuple[CompletenessSnapshot, ...]:
    return tuple(
        CompletenessSnapshot(definition=definition, analysis=_analysis(definition))
        for definition in build_snapshot_definitions(config)
    )


def test_five_snapshot_pipeline_builds_complete_causal_evidence() -> None:
    config = load_background_config("configs/background.yaml")
    catalog = _catalog()
    result = run_poisson_kde_pipeline(
        config,
        catalog,
        _grid_family(),
        _completeness(config),
        chunk_size=2,
    )

    expected_protocol = hashlib.sha256(
        canonical_json_bytes(config.model_dump(mode="python"))
    ).hexdigest()
    assert result.protocol_sha256 == expected_protocol
    assert tuple(item.definition.snapshot_id for item in result.snapshots) == EXPECTED_SNAPSHOTS
    assert tuple(item.training_event_count for item in result.snapshots) == (1, 2, 3, 4, 5)
    assert all(item.rate_per_day > 0.0 for item in result.snapshots)
    assert all(
        tuple(bandwidth for bandwidth, _ in item.kde_family) == FROZEN_BANDWIDTHS_KM
        for item in result.snapshots
    )
    assert all(
        tuple(gate.bandwidth_km for gate in item.grid_gate_evidence) == FROZEN_BANDWIDTHS_KM
        for item in result.snapshots
    )
    assert all(
        gate.passed
        and tuple(
            (pair.coarse_cell_size_km, pair.fine_cell_size_km)
            for pair in gate.convergence.comparisons
        )
        == ((50.0, 25.0), (25.0, 12.5))
        for snapshot in result.snapshots
        for gate in snapshot.grid_gate_evidence
    )
    assert result.pre_score_gate_evidence.passed_bandwidths_km == FROZEN_BANDWIDTHS_KM
    assert result.selected_bandwidth_km in FROZEN_BANDWIDTHS_KM
    assert f"bw{result.selected_bandwidth_km:g}km" in result.spatial_evidence.model_variant_id
    assert result.selected_kde_model("fold_1") is result.snapshot("fold_1").kde_model(
        result.selected_bandwidth_km
    )

    final_pair = result.spatial_evidence.validation
    assert final_pair is not None
    assert final_pair.candidate.target_event_ids == ("delayed-final",)
    delayed_index = int(np.flatnonzero(catalog.event_id == "delayed-final")[0])
    assert catalog.available_day[delayed_index] > result.snapshots[4].definition.assessment_end_day
    for pair, snapshot in zip(
        (*result.spatial_evidence.development_folds, final_pair),
        result.snapshots,
        strict=True,
    ):
        assert pair.candidate.target_event_ids == pair.uniform.target_event_ids
        assert pair.candidate.target_event_ids == snapshot.target_event_ids
        expected_compensator = snapshot.rate_per_day * (
            snapshot.definition.assessment_duration_days
        )
        assert pair.candidate.compensator == expected_compensator
        assert pair.uniform.compensator == expected_compensator
        assert not pair.candidate.event_log_intensities.flags.writeable
        assert re.fullmatch(r"[0-9a-f]{64}", pair.candidate.parameter_snapshot_id)


def test_final_validation_location_cannot_change_grid_gates_or_bandwidth_selection() -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    completeness = _completeness(config)

    first = run_poisson_kde_pipeline(
        config,
        _catalog(final_target_x_km=1.1),
        family,
        completeness,
        chunk_size=2,
    )
    second = run_poisson_kde_pipeline(
        config,
        _catalog(final_target_x_km=1.9),
        family,
        completeness,
        chunk_size=2,
    )

    assert first.pre_score_gate_evidence == second.pre_score_gate_evidence
    assert first.bandwidth_selection == second.bandwidth_selection
    assert first.selected_bandwidth_km == second.selected_bandwidth_km
    assert tuple(grid.cell_ids for grid in family.grids) == tuple(
        grid.cell_ids for grid in family.grids
    )
    first_validation = first.spatial_evidence.validation
    second_validation = second.spatial_evidence.validation
    assert first_validation is not None and second_validation is not None
    assert not math.isclose(
        first_validation.candidate.event_log_intensity_sum,
        second_validation.candidate.event_log_intensity_sum,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def test_zero_target_fold_and_zero_training_snapshot_are_explicit_hard_failures() -> None:
    config = load_background_config("configs/background.yaml")
    completeness = _completeness(config)
    family = _grid_family()

    with pytest.raises(PoissonKDEPipelineError, match="zero-target.*fold_2") as zero_target:
        run_poisson_kde_pipeline(
            config,
            _catalog(omitted_target_id="target-fold-2"),
            family,
            completeness,
            chunk_size=2,
        )
    assert zero_target.value.gate_evidence is not None

    with pytest.raises(PoissonKDEPipelineError, match="zero eligible training.*fold_1"):
        run_poisson_kde_pipeline(
            config,
            _catalog(include_training_event=False),
            family,
            completeness,
            chunk_size=2,
        )


def test_completeness_input_must_be_exactly_four_folds_plus_final() -> None:
    config = load_background_config("configs/background.yaml")
    with pytest.raises(ValueError, match="exactly four folds plus final"):
        run_poisson_kde_pipeline(
            config,
            _catalog(),
            _grid_family(),
            _completeness(config)[:-1],
        )


def test_progress_callback_is_observational_and_reports_every_snapshot() -> None:
    config = load_background_config("configs/background.yaml")
    catalog = _catalog()
    family = _grid_family()
    completeness = _completeness(config)

    without_callback = run_poisson_kde_pipeline(
        config,
        catalog,
        family,
        completeness,
        chunk_size=2,
    )
    messages: list[str] = []
    with_callback = run_poisson_kde_pipeline(
        config,
        catalog,
        family,
        completeness,
        chunk_size=2,
        progress=messages.append,
    )

    assert messages == [
        message
        for snapshot_id in EXPECTED_SNAPSHOTS
        for message in (
            f"poisson_kde:{snapshot_id}:start",
            f"poisson_kde:{snapshot_id}:done",
        )
    ]
    assert with_callback.protocol_sha256 == without_callback.protocol_sha256
    assert with_callback.selected_bandwidth_km == without_callback.selected_bandwidth_km
    assert with_callback.pre_score_gate_evidence == without_callback.pre_score_gate_evidence
    assert with_callback.bandwidth_selection == without_callback.bandwidth_selection
    assert with_callback.spatial_evidence.validation is not None
    assert without_callback.spatial_evidence.validation is not None
    observed_scores = (
        *with_callback.spatial_evidence.development_folds,
        with_callback.spatial_evidence.validation,
    )
    expected_scores = (
        *without_callback.spatial_evidence.development_folds,
        without_callback.spatial_evidence.validation,
    )
    for observed, expected in zip(observed_scores, expected_scores, strict=True):
        assert observed.candidate.score_id == expected.candidate.score_id
        assert observed.uniform.score_id == expected.uniform.score_id
