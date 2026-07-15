from __future__ import annotations

import dataclasses
import hashlib
import inspect
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest

import seismoflux.anomaly_increment.convergence as convergence_module
from seismoflux.anomaly_increment.contracts import FeatureColumnContract, FrozenTargetRateHead
from seismoflux.anomaly_increment.convergence import (
    CONVERGENCE_FAILURE_ACTION,
    CONVERGENCE_PASS_ACTION,
    EVENT_LOG_TERM_POLICY,
    CompensatorConvergenceResult,
    FrozenConvergenceModel,
    FrozenTargetBlindConvergenceInputs,
    FrozenVariantCoefficients,
    RecomputedGridInputs,
    audit_compensator_convergence,
    audit_primary_grid_logical_replay_r1,
    build_target_blind_convergence_inputs,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4GridFamily,
    Stage4IntegrationGrid,
)
from seismoflux.anomaly_increment.model import fit_frozen_target_rate_head
from seismoflux.anomaly_increment.preprocessing import (
    FrozenPreprocessor,
    fit_frozen_preprocessor,
)
from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot


def _preprocessor(*, source_column: str = "signal") -> FrozenPreprocessor:
    contract = FeatureColumnContract(
        source_column=source_column,
        logical_feature=source_column,
        value_output_column=f"value__{source_column}",
        missing_output_column=f"missing__{source_column}",
        transform="identity_finite",
    )
    return fit_frozen_preprocessor(
        (contract,),
        {source_column: np.asarray([0.0, 1.0, 2.0, 3.0])},
    )


def _rate_head(*, tiny: bool = False) -> FrozenTargetRateHead:
    exposure = 1.0e15 if tiny else 4.0
    count = 1 if tiny else 2
    return fit_frozen_target_rate_head(
        training_event_counts={"M5_6": count, "M6_plus": count},
        background_exposures={"M5_6": exposure, "M6_plus": exposure},
    )


def _variants(*, dynamic_beta: float = 0.3) -> tuple[FrozenVariantCoefficients, ...]:
    return (
        FrozenVariantCoefficients(
            variant="background_no_increment",
            design_column_indices=(),
            beta=np.asarray([], dtype=np.float64),
        ),
        FrozenVariantCoefficients(
            variant="coverage_only",
            design_column_indices=(0, 1),
            beta=np.asarray([0.1, 0.0]),
        ),
        FrozenVariantCoefficients(
            variant="snapshot",
            design_column_indices=(0, 1),
            beta=np.asarray([0.2, 0.0]),
        ),
        FrozenVariantCoefficients(
            variant="dynamic",
            design_column_indices=(0, 1),
            beta=np.asarray([dynamic_beta, 0.0]),
        ),
    )


def _model(
    *,
    dynamic_beta: float = 0.3,
    tiny_rate: bool = False,
    preprocessor: FrozenPreprocessor | None = None,
) -> FrozenConvergenceModel:
    return FrozenConvergenceModel(
        preprocessor=preprocessor or _preprocessor(),
        rate_head=_rate_head(tiny=tiny_rate),
        variants=_variants(dynamic_beta=dynamic_beta),
    )


def _grid(cell_size_km: float, *, mass_scale: float = 1.0) -> RecomputedGridInputs:
    return RecomputedGridInputs(
        grid_id=f"synthetic-{cell_size_km:g}km",
        cell_size_km=cell_size_km,
        background_spatial_mass_by_cell_and_bin=(
            np.asarray([[0.4, 0.3], [0.6, 0.7]], dtype=np.float64) * mass_scale
        ),
        feature_columns={"signal": np.asarray([3.0, 3.0])},
    )


def _grids(
    *,
    scale_50: float = 1.0,
    scale_25: float = 1.0,
    scale_12_5: float = 1.0,
) -> tuple[RecomputedGridInputs, ...]:
    return (
        _grid(50.0, mass_scale=scale_50),
        _grid(25.0, mass_scale=scale_25),
        _grid(12.5, mass_scale=scale_12_5),
    )


def test_all_variant_bin_horizon_results_pass_and_hash_deterministically() -> None:
    model = _model()
    grids = _grids(scale_50=1.5)
    first = audit_compensator_convergence(model=model, grids=grids)
    second = audit_compensator_convergence(model=model, grids=grids)

    assert first.passed
    assert first.status == "passed"
    assert first.publication_action == CONVERGENCE_PASS_ACTION
    assert len(first.results) == 4 * 2 * 5
    assert all(item.passed for item in first.results)
    assert any(not item.coarse_50_vs_25.passed for item in first.results)
    assert all(len(item.content_sha256) == 64 for item in first.results)
    assert all(item.sha256 == item.content_sha256 for item in first.results)
    assert len(first.content_sha256) == 64
    assert first.as_mapping() == second.as_mapping()
    assert first.content_sha256 == second.content_sha256
    assert first.as_mapping()["content_sha256"] == first.content_sha256

    coarse = cast(dict[str, object], first.results[0].as_mapping()["coarse_50_vs_25"])
    assert coarse["gate_applied"] is False
    assert coarse["role"] == "required_reported_coarse_trend_diagnostic_not_gate_reference"


def test_spatial_failure_forbids_spatial_publication_but_time_gate_stays_separate() -> None:
    audit = audit_compensator_convergence(
        model=_model(),
        grids=_grids(scale_25=1.01, scale_12_5=1.0),
    )

    assert not audit.passed
    assert audit.status == "failed"
    assert audit.publication_action == CONVERGENCE_FAILURE_ACTION
    assert any(not item.spatial_25_vs_12_5.passed for item in audit.results)
    assert all(item.temporal_1d_vs_0_5d.passed for item in audit.results)


def test_temporal_failure_forbids_spatial_publication_without_spatial_failure() -> None:
    audit = audit_compensator_convergence(
        model=_model(dynamic_beta=100.0),
        grids=_grids(),
    )

    dynamic = tuple(item for item in audit.results if item.variant == "dynamic")
    assert dynamic
    assert all(item.spatial_25_vs_12_5.passed for item in audit.results)
    assert any(not item.temporal_1d_vs_0_5d.passed for item in dynamic)
    assert audit.publication_action == CONVERGENCE_FAILURE_ACTION


def test_near_zero_absolute_branch_passes_large_relative_difference() -> None:
    audit = audit_compensator_convergence(
        model=_model(tiny_rate=True),
        grids=_grids(scale_25=1.1, scale_12_5=1.0),
    )
    selected = next(
        item
        for item in audit.results
        if item.variant == "background_no_increment"
        and item.magnitude_bin == "M5_6"
        and item.horizon_days == 365
    )

    assert selected.spatial_25_vs_12_5.relative_difference > 0.005
    assert selected.spatial_25_vs_12_5.absolute_difference <= 1.0e-10
    assert selected.spatial_25_vs_12_5.passed
    assert audit.passed


def test_grid_inputs_have_no_cross_grid_refit_or_event_term_fields() -> None:
    grid_fields = {field.name for field in dataclasses.fields(RecomputedGridInputs)}
    result_fields = {field.name for field in dataclasses.fields(CompensatorConvergenceResult)}
    assert grid_fields == {
        "grid_id",
        "cell_size_km",
        "background_spatial_mass_by_cell_and_bin",
        "feature_columns",
    }
    assert not any(
        token in name
        for name in grid_fields
        for token in ("fit", "preprocessor", "beta", "rate_head", "training")
    )
    assert not any(
        token in name
        for name in result_fields
        for token in ("event_log", "target", "epicenter", "hypocenter")
    )
    assert not any(name.startswith("fit_") for name in vars(convergence_module))

    signature = inspect.signature(audit_compensator_convergence)
    assert not any(
        token in parameter
        for parameter in signature.parameters
        for token in ("event", "target", "epicenter", "hypocenter")
    )


def test_target_epicenter_fields_are_absent_or_rejected_fail_closed() -> None:
    field_names = {field.name.casefold() for field in dataclasses.fields(RecomputedGridInputs)}
    assert not any(
        token in name
        for name in field_names
        for token in ("target", "epicenter", "hypocenter", "longitude", "latitude")
    )
    with pytest.raises(ValueError, match="forbidden target/epicenter"):
        dataclasses.replace(
            _grid(50.0),
            feature_columns={
                "signal": np.asarray([3.0, 3.0]),
                "target_epicenter_longitude": np.asarray([100.0, 101.0]),
            },
        )

    target_preprocessor = _preprocessor(source_column="target_epicenter_distance")
    with pytest.raises(ValueError, match="forbidden target/epicenter"):
        _model(preprocessor=target_preprocessor)


def test_event_log_term_is_explicitly_excluded_and_unchanged() -> None:
    audit = audit_compensator_convergence(model=_model(), grids=_grids())
    event_term = cast(dict[str, object], audit.as_mapping()["event_log_term"])

    assert audit.event_log_term_policy == EVENT_LOG_TERM_POLICY
    assert audit.event_log_term_included is False
    assert audit.event_log_term_change == 0.0
    assert event_term == {
        "change": 0.0,
        "included": False,
        "policy": "excluded_from_compensator_convergence_and_unchanged",
    }


def test_frozen_grids_and_time_steps_cannot_drift() -> None:
    with pytest.raises(ValueError, match="ordered exactly 50, 25, and 12.5"):
        audit_compensator_convergence(
            model=_model(),
            grids=tuple(reversed(_grids())),
        )
    with pytest.raises(ValueError, match="exactly 1 day"):
        audit_compensator_convergence(
            model=_model(),
            grids=_grids(),
            primary_time_step_days=2.0,
        )
    with pytest.raises(ValueError, match="exactly 0.5 day"):
        audit_compensator_convergence(
            model=_model(),
            grids=_grids(),
            reference_time_step_days=0.25,
        )


def _one_cell_grid(cell_size_km: float) -> Stage4IntegrationGrid:
    token = f"official-convergence-grid:{cell_size_km:g}".encode()
    return Stage4IntegrationGrid(
        grid_id=hashlib.sha256(token).hexdigest(),
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=cell_size_km,
        cell_ids=(f"cell-{cell_size_km:g}",),
        rows=np.asarray([0], dtype=np.int64),
        columns=np.asarray([0], dtype=np.int64),
        query_xy_m=np.asarray([[cell_size_km * 1_000.0, 0.0]], dtype=np.float64),
        clipped_area_km2=np.asarray([1.0], dtype=np.float64),
    )


def _one_cell_issue_table(
    snapshot: Stage3IssueSnapshot,
    grid: Stage4IntegrationGrid,
    *,
    signal: float | None,
) -> pa.Table:
    return pa.table(
        {
            "issue_time_utc": pa.array(
                [snapshot.issue_time_utc],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "issue_report_id": pa.array([snapshot.summary.issue_report_id]),
            "grid_id": pa.array([grid.grid_id]),
            "cell_id": pa.array([grid.cell_ids[0]]),
            "cell_row": pa.array([0], type=pa.int64()),
            "cell_column": pa.array([0], type=pa.int64()),
            "query_x_m": pa.array([grid.query_xy_m[0, 0]], type=pa.float64()),
            "query_y_m": pa.array([grid.query_xy_m[0, 1]], type=pa.float64()),
            "clipped_area_km2": pa.array([1.0], type=pa.float64()),
            "signal": pa.array([signal], type=pa.float64()),
        }
    )


def test_pretarget_builder_proves_all_153_primary_issues_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grids = Stage4GridFamily(
        _one_cell_grid(50.0),
        _one_cell_grid(25.0),
        _one_cell_grid(12.5),
    )
    start = datetime(2022, 7, 20, 16, tzinfo=UTC)
    issue_ids = tuple(f"anomaly-issue-{index:03d}" for index in range(153))
    snapshots = tuple(
        Stage3IssueSnapshot(
            issue_index=index,
            issue_time_utc=start + timedelta(days=7 * index),
            summary=cast(
                Any,
                SimpleNamespace(issue_report_id=f"report-{index:03d}"),
            ),
            entities=(),
            state_snapshot_id=hashlib.sha256(f"state:{index}".encode()).hexdigest(),
            lineage_digest=hashlib.sha256(f"lineage:{index}".encode()).hexdigest(),
        )
        for index in range(153)
    )

    def signal(index: int) -> float | None:
        return None if index == 50 else float(index)

    accepted = {
        issue_id: _one_cell_issue_table(
            snapshot,
            grids.primary_25km,
            signal=signal(index),
        )
        for index, (issue_id, snapshot) in enumerate(zip(issue_ids, snapshots, strict=True))
    }
    observed_worker_counts: list[int] = []

    class FakeEngine:
        def __init__(
            self,
            history: Sequence[Stage3IssueSnapshot],
            query_grid: Stage3QueryGrid,
            **kwargs: object,
        ) -> None:
            assert tuple(history) == snapshots
            observed_worker_counts.append(cast(int, kwargs["spatial_workers"]))
            self.grid = next(item for item in grids.grids() if item.grid_id == query_grid.grid_id)
            self.index = 0

        def build_next_issue(self) -> object:
            index = self.index
            self.index += 1
            return SimpleNamespace(
                table=_one_cell_issue_table(
                    snapshots[index],
                    self.grid,
                    signal=signal(index),
                )
            )

    monkeypatch.setattr(convergence_module, "Stage3FeatureEngine", FakeEngine)
    frozen = build_target_blind_convergence_inputs(
        issue_ids=issue_ids,
        snapshots=snapshots,
        grid_family=grids,
        accepted_primary_issue_tables=accepted,
        source_columns=("signal",),
        source_input_sha256="a" * 64,
    )

    assert isinstance(frozen, FrozenTargetBlindConvergenceInputs)
    assert len(frozen.primary_reproduction_receipts) == 153
    assert all(
        item.accepted_table_sha256 == item.recomputed_table_sha256
        for item in frozen.primary_reproduction_receipts
    )
    assert frozen.issue_id == issue_ids[-1]
    assert frozen.grids[1].feature_columns["signal"].tolist() == [152.0]
    assert frozen.target_bytes_read is False
    assert frozen.target_path_observed is False

    replay = audit_primary_grid_logical_replay_r1(
        issue_ids=issue_ids,
        snapshots=snapshots,
        grid_family=grids,
        accepted_primary_issue_tables=accepted,
        source_columns=("signal",),
        source_input_sha256="a" * 64,
    )
    assert replay.worker_counts == (1, 2)
    assert observed_worker_counts[-2:] == [1, 2]
    assert len(replay.receipts_by_worker[0]) == 153
    assert replay.receipts_by_worker[0] == replay.receipts_by_worker[1]
    assert replay.target_bytes_read is False
    assert replay.target_path_observed is False
    assert replay.as_mapping()["reproduction_identity_sha256"] == (
        replay.reproduction_identity_sha256
    )

    changed = dict(accepted)
    changed[issue_ids[75]] = _one_cell_issue_table(
        snapshots[75],
        grids.primary_25km,
        signal=-75.0,
    )
    with pytest.raises(ValueError, match="identity reconstruction differs"):
        build_target_blind_convergence_inputs(
            issue_ids=issue_ids,
            snapshots=snapshots,
            grid_family=grids,
            accepted_primary_issue_tables=changed,
            source_columns=("signal",),
            source_input_sha256="a" * 64,
        )
