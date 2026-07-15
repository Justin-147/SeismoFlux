from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pytest
from pyproj import CRS, Transformer
from shapely.geometry import box

import seismoflux.anomaly_increment.formal_assembly as formal_assembly
from seismoflux.anomaly_increment.background_adapter import (
    BackgroundGridDensity,
    FrozenBackgroundSnapshot,
)
from seismoflux.anomaly_increment.formal_assembly import (
    BoundTargetCellAssignments,
    FrozenBackgroundField,
    FrozenCellSupport,
    ProspectiveIssuePlan,
    VerifiedStage3Issue,
    assemble_evaluation_scope,
    assemble_prospective_issue_inputs,
    assemble_stage4_formal_inputs,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4IntegrationGrid,
    build_stage4_integration_grid,
)
from seismoflux.anomaly_increment.integration import lead_decay
from seismoflux.anomaly_increment.runner import (
    ExposurePlan,
    FitScopePlan,
    HorizonAssessmentPlan,
    Stage4ScoringPlan,
    TimePermutationPools,
)
from seismoflux.anomaly_increment.scoring_pipeline import FeatureLayout
from seismoflux.anomaly_increment.targets import (
    Stage4TargetCatalog,
    TargetCellAssignments,
    map_targets_to_frozen_primary_grid,
)
from seismoflux.background.catalog import StudyArea
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    build_clipped_grid,
    project_study_area_to_equal_area,
)
from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.state import AnomalyState

SOURCE_COLUMNS = ("coverage", "snapshot", "dynamic")
GRID_ID = "1" * 64
DOMAIN_ID = "2" * 64
STUDY_AREA_SHA256 = "3" * 64
SUPPORT_ID = "synthetic-fixed-cell-support"


def _grid() -> Stage4IntegrationGrid:
    return Stage4IntegrationGrid(
        grid_id=GRID_ID,
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=25.0,
        cell_ids=("cell-r0-c0", "cell-r0-c1", "cell-r1-c0"),
        rows=np.asarray([0, 0, 1], dtype=np.int64),
        columns=np.asarray([0, 1, 0], dtype=np.int64),
        query_xy_m=np.asarray(
            [[12_500.0, 12_500.0], [37_500.0, 12_500.0], [12_500.0, 37_500.0]],
            dtype=np.float64,
        ),
        clipped_area_km2=np.asarray([625.0, 625.0, 625.0], dtype=np.float64),
    )


def _layout() -> FeatureLayout:
    return FeatureLayout(
        coverage_sources=("coverage",),
        snapshot_sources=("coverage", "snapshot"),
        dynamic_sources=SOURCE_COLUMNS,
    )


def _background(
    evaluation_id: str,
    *,
    grid: Stage4IntegrationGrid,
) -> FrozenBackgroundField:
    role: Literal["development", "validation"] = (
        "validation" if evaluation_id == "formal-validation" else "development"
    )
    density = np.asarray([0.0002, 0.0003, 0.0004], dtype=np.float64)
    mass = density * grid.clipped_area_km2
    snapshot = FrozenBackgroundSnapshot(
        evaluation_id=cast(
            formal_assembly.EvaluationId,
            evaluation_id,
        ),
        role=role,
        snapshot_id=f"synthetic-{evaluation_id}",
        parameter_snapshot_id="4" * 64,
        fit_end_utc=datetime(2023, 12, 1, tzinfo=UTC),
        support_id=SUPPORT_ID,
        compensator_domain_id=DOMAIN_ID,
        common_mc=4.0,
        bandwidth_km=75.0,
        supported_area_fraction=1.0,
        study_area_sha256=STUDY_AREA_SHA256,
    )
    return FrozenBackgroundField(
        snapshot=snapshot,
        primary_grid_density=BackgroundGridDensity(
            grid_id=grid.grid_id,
            cell_size_km=25.0,
            cell_ids=grid.cell_ids,
            spatial_density_per_km2=density,
            spatial_cell_mass=mass,
            expected_cell_count_per_day=mass,
        ),
    )


def _support(
    evaluation_id: str,
    *,
    grid: Stage4IntegrationGrid,
) -> FrozenCellSupport:
    return FrozenCellSupport(
        evaluation_id=cast(formal_assembly.EvaluationId, evaluation_id),
        grid_id=grid.grid_id,
        cell_ids=grid.cell_ids,
        support_id=SUPPORT_ID,
        compensator_domain_id=DOMAIN_ID,
        supported_cell_mask=np.asarray([True, False, True], dtype=np.bool_),
    )


def _issue_time(issue_date: date) -> datetime:
    return datetime(issue_date.year, issue_date.month, issue_date.day, tzinfo=UTC) - timedelta(
        hours=8
    )


def _issue_table(
    issue_date: date,
    *,
    grid: Stage4IntegrationGrid,
    ordinal: float,
) -> pa.Table:
    issue_time = _issue_time(issue_date)
    report_id = f"report-{issue_date.isoformat()}"
    return pa.table(
        {
            "issue_time_utc": pa.array(
                [issue_time] * grid.cell_count,
                pa.timestamp("us", tz="UTC"),
            ),
            "issue_report_id": pa.array([report_id] * grid.cell_count, pa.string()),
            "grid_id": pa.array([grid.grid_id] * grid.cell_count, pa.string()),
            "cell_id": pa.array(grid.cell_ids, pa.string()),
            "cell_row": pa.array(grid.rows, pa.int64()),
            "cell_column": pa.array(grid.columns, pa.int64()),
            "query_x_m": pa.array(grid.query_xy_m[:, 0], pa.float64()),
            "query_y_m": pa.array(grid.query_xy_m[:, 1], pa.float64()),
            "clipped_area_km2": pa.array(grid.clipped_area_km2, pa.float64()),
            "coverage": pa.array([ordinal + 1.0, ordinal + 2.0, ordinal + 3.0]),
            "snapshot": pa.array([ordinal + 10.0, ordinal + 20.0, ordinal + 30.0]),
            "dynamic": pa.array([ordinal + 100.0, ordinal + 200.0, ordinal + 300.0]),
        }
    )


def _verified_issue(
    issue_date: date,
    *,
    grid: Stage4IntegrationGrid,
    ordinal: float,
) -> VerifiedStage3Issue:
    issue_time = _issue_time(issue_date)
    report_id = f"report-{issue_date.isoformat()}"
    summary = cast(
        AnomalyState,
        SimpleNamespace(issue_report_id=report_id, issue_time_utc=issue_time),
    )
    snapshot = Stage3IssueSnapshot(
        issue_index=int(ordinal),
        issue_time_utc=issue_time,
        summary=summary,
        entities=(),
        state_snapshot_id="5" * 64,
        lineage_digest="6" * 64,
    )
    return VerifiedStage3Issue.bind(
        _issue_table(issue_date, grid=grid, ordinal=ordinal),
        snapshot,
        primary_grid=grid,
        source_columns=SOURCE_COLUMNS,
    )


def _catalog() -> tuple[Stage4TargetCatalog, TargetCellAssignments]:
    fit_start = _issue_time(date(2024, 1, 1))
    assessment_start = _issue_time(date(2024, 1, 9))
    rows = (
        ("at-fit-start", fit_start, fit_start, 5.2, True, 0),
        (
            "fit-supported",
            fit_start + timedelta(days=0.5),
            fit_start + timedelta(days=0.6),
            5.5,
            True,
            2,
        ),
        (
            "fit-unsupported",
            fit_start + timedelta(days=1.0),
            fit_start + timedelta(days=1.1),
            5.7,
            True,
            1,
        ),
        (
            "fit-late-availability",
            fit_start + timedelta(days=2.0),
            assessment_start + timedelta(hours=1),
            5.4,
            True,
            0,
        ),
        ("fit-end", fit_start + timedelta(days=7.0), fit_start + timedelta(days=7.0), 5.8, True, 0),
        ("at-assessment-start", assessment_start, assessment_start, 5.1, True, 0),
        (
            "assessment-supported",
            assessment_start + timedelta(days=0.5),
            assessment_start + timedelta(days=0.6),
            5.4,
            True,
            2,
        ),
        (
            "assessment-outside",
            assessment_start + timedelta(days=0.75),
            assessment_start + timedelta(days=0.8),
            5.6,
            False,
            0,
        ),
        (
            "assessment-unsupported",
            assessment_start + timedelta(days=1.0),
            assessment_start + timedelta(days=1.1),
            5.5,
            True,
            1,
        ),
        (
            "assessment-m6-end",
            assessment_start + timedelta(days=7.0),
            assessment_start + timedelta(days=7.0),
            6.2,
            True,
            0,
        ),
        (
            "after-seven-days",
            assessment_start + timedelta(days=7.1),
            assessment_start + timedelta(days=7.2),
            5.9,
            True,
            2,
        ),
    )
    event_ids = np.asarray([item[0] for item in rows], dtype=np.str_)
    inside = np.asarray([item[4] for item in rows], dtype=np.bool_)
    catalog = Stage4TargetCatalog(
        event_id=event_ids,
        origin_time_utc=tuple(item[1] for item in rows),
        available_at_utc=tuple(item[2] for item in rows),
        longitude=np.asarray([105.0 + index * 0.001 for index in range(len(rows))]),
        latitude=np.asarray([34.0 + index * 0.001 for index in range(len(rows))]),
        x_m=np.asarray([float(item[5]) * 25_000.0 for item in rows]),
        y_m=np.asarray([0.0] * len(rows)),
        magnitude=np.asarray([item[3] for item in rows], dtype=np.float64),
        inside_study_area=inside,
        source_content_sha256="7" * 64,
        source_schema_sha256="8" * 64,
    )
    inside_rows = tuple(item for item in rows if item[4])
    cell_indices = np.asarray([item[5] for item in inside_rows], dtype=np.int64)
    grid = _grid()
    assignments = TargetCellAssignments(
        event_ids=tuple(item[0] for item in inside_rows),
        cell_indices=cell_indices,
        cell_ids=tuple(grid.cell_ids[int(index)] for index in cell_indices),
        grid_id=grid.grid_id,
    )
    return catalog, assignments


def _scope(evaluation_id: str) -> FitScopePlan:
    formal = evaluation_id == "formal-validation"
    horizons = (7, 30, 90, 180, 365) if formal else (7, 30, 90)
    partition: Literal["development", "validation"] = "validation" if formal else "development"
    return FitScopePlan(
        fit_scope_id=("full-development-before-validation" if formal else evaluation_id),
        role="formal_validation_once" if formal else "development_joint_macro",
        fit_exposures_7d=(ExposurePlan("development", 7, date(2024, 1, 1)),),
        assessments=tuple(
            HorizonAssessmentPlan(
                horizon_days=horizon,
                exposures=(ExposurePlan(partition, horizon, date(2024, 1, 9)),),
            )
            for horizon in horizons
        ),
        time_permutation_pools=TimePermutationPools(
            fit_issue_ids=("fit-issue",),
            assessment_issue_ids=("assessment-issue",),
        ),
        validation_refit_forbidden=formal,
    )


def _inputs() -> tuple[
    Stage4IntegrationGrid,
    Stage4TargetCatalog,
    BoundTargetCellAssignments,
    tuple[VerifiedStage3Issue, ...],
]:
    grid = _grid()
    catalog, assignments = _catalog()
    bound = BoundTargetCellAssignments.bind(catalog, assignments, primary_grid=grid)
    issues = (
        _verified_issue(date(2024, 1, 1), grid=grid, ordinal=1.0),
        _verified_issue(date(2024, 1, 9), grid=grid, ordinal=2.0),
        _verified_issue(date(2024, 2, 1), grid=grid, ordinal=3.0),
    )
    return grid, catalog, bound, issues


def test_formal_assembler_is_pure_memory_and_prospective_api_is_target_free() -> None:
    source = inspect.getsource(formal_assembly)
    tree = ast.parse(source)
    forbidden_names = {"open", "Path", "read_bytes", "read_text", "read_table"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_names
    parameters = inspect.signature(assemble_prospective_issue_inputs).parameters
    assert "catalog" not in parameters
    assert "target_assignments" not in parameters
    assert not any(
        "event" in name for name in formal_assembly.ProspectiveIssueInput.__dataclass_fields__
    )


def test_assignment_and_stage3_bindings_reject_identity_tampering() -> None:
    grid, catalog, bound, issues = _inputs()
    bound.verify(catalog, primary_grid=grid)

    bad_assignments = dataclasses.replace(
        bound.assignments,
        cell_ids=("cell-r0-c1", *bound.assignments.cell_ids[1:]),
    )
    with pytest.raises(ValueError, match="row/column/cell"):
        BoundTargetCellAssignments.bind(catalog, bad_assignments, primary_grid=grid)

    issue = issues[0]
    changed = issue.table.set_column(
        issue.table.schema.get_field_index("issue_report_id"),
        "issue_report_id",
        pa.array(["other-report"] * grid.cell_count),
    )
    forged = dataclasses.replace(issue, table=changed)
    with pytest.raises(ValueError, match="report identity"):
        forged.verify(primary_grid=grid, source_columns=SOURCE_COLUMNS)


def test_formal_scope_uses_strict_windows_cell_center_terms_and_local_support() -> None:
    grid, catalog, bound, issues = _inputs()
    bundle = assemble_evaluation_scope(
        _scope("formal-validation"),
        catalog=catalog,
        target_assignments=bound,
        verified_issues=issues,
        primary_grid=grid,
        background=_background("formal-validation", grid=grid),
        support=_support("formal-validation", grid=grid),
        feature_layout=_layout(),
    )

    fit = bundle.scope.fit
    assert fit.training_event_counts == {"M5_6": 2, "M6_plus": 0}
    assert fit.event_magnitude_bin_ids == ("M5_6", "M5_6")
    assert np.array_equal(
        np.asarray(fit.event_feature_columns["dynamic"], dtype=np.float64),
        [301.0, 101.0],
    )
    assert np.allclose(
        fit.event_decay,
        [cast(float, lead_decay(0.5)), cast(float, lead_decay(7.0))],
    )
    assert np.array_equal(fit.event_background_intensity, [0.0004, 0.0002])
    assert np.array_equal(
        np.asarray(fit.issue_cell_feature_columns["dynamic"], dtype=np.float64),
        [101.0, 301.0],
    )
    assert bundle.audit.fit_unavailable_event_ids == ("fit-late-availability",)
    assert bundle.audit.zero_event_magnitude_bins == ("M6_plus",)
    assert set(bundle.audit.fit_supported_event_ids).isdisjoint(
        bundle.audit.assessment_all_study_area_event_ids
    )

    exposure_m5 = next(
        item
        for item in bundle.scope.exposures
        if item.horizon_days == 7 and item.magnitude_bin == "M5_6"
    )
    assert exposure_m5.cell_order_ids == ("cell-r0-c0", "cell-r1-c0")
    assert np.array_equal(
        np.asarray(exposure_m5.cell_feature_columns["dynamic"], dtype=np.float64),
        [102.0, 302.0],
    )
    assert np.array_equal(exposure_m5.background_spatial_mass, [0.125, 0.25])
    assert exposure_m5.all_study_area_event_ids == (
        "assessment-supported",
        "assessment-unsupported",
    )
    assert exposure_m5.supported_event_ids == ("assessment-supported",)
    assert exposure_m5.event_cell_indices == (1,)
    assert np.array_equal(exposure_m5.event_lead_days, [0.5])
    assert "at-assessment-start" not in exposure_m5.all_study_area_event_ids

    exposure_m6 = next(
        item
        for item in bundle.scope.exposures
        if item.horizon_days == 7 and item.magnitude_bin == "M6_plus"
    )
    assert exposure_m6.supported_event_ids == ("assessment-m6-end",)
    assert np.array_equal(exposure_m6.event_lead_days, [7.0])


def test_prospective_assembly_has_no_target_dependency_or_unsupported_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid, _, _, issues = _inputs()

    def forbidden_target_access(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"prospective assembly consulted target data: {args}/{kwargs}")

    monkeypatch.setattr(formal_assembly, "exposure_target_view", forbidden_target_access)
    prospective = assemble_prospective_issue_inputs(
        (ProspectiveIssuePlan(date(2024, 2, 1), 30, "M5_6"),),
        verified_issues=issues,
        primary_grid=grid,
        background=_background("formal-validation", grid=grid),
        support=_support("formal-validation", grid=grid),
        feature_layout=_layout(),
    )
    assert len(prospective) == 1
    item = prospective[0]
    assert item.cell_order_ids == ("cell-r0-c0", "cell-r1-c0")
    assert np.array_equal(
        np.asarray(item.cell_feature_columns["dynamic"], dtype=np.float64),
        [103.0, 303.0],
    )
    assert not hasattr(item, "event_ids")


def test_top_level_assembles_three_development_folds_and_formal_validation() -> None:
    grid, catalog, bound, issues = _inputs()
    evaluation_ids = (
        "development-fold-1",
        "development-fold-2",
        "development-fold-3",
        "formal-validation",
    )
    scoring_plan = cast(
        Stage4ScoringPlan,
        SimpleNamespace(fit_scopes=tuple(_scope(item) for item in evaluation_ids)),
    )
    assembled = assemble_stage4_formal_inputs(
        scoring_plan,
        catalog=catalog,
        target_assignments=bound,
        verified_issues=issues,
        primary_grid=grid,
        backgrounds=tuple(_background(item, grid=grid) for item in evaluation_ids),
        cell_supports=tuple(_support(item, grid=grid) for item in evaluation_ids),
        feature_layout=_layout(),
        prospective_plans=(ProspectiveIssuePlan(date(2024, 2, 1), 7, "M6_plus"),),
    )
    assert tuple(item.fit.evaluation_id for item in assembled.development_scopes) == (
        "development-fold-1",
        "development-fold-2",
        "development-fold-3",
    )
    assert assembled.formal_scope.fit.evaluation_id == "formal-validation"
    assert tuple(len(item.exposures) for item in assembled.development_scopes) == (6, 6, 6)
    assert len(assembled.formal_scope.exposures) == 10
    assert len(assembled.fit_scopes) == 4
    assert len(assembled.evaluation_exposures) == 28
    assert len(assembled.prospective_issues) == 1


def test_existing_covers_mapping_resolves_boundary_by_frozen_row_column_cell_id() -> None:
    geographic = box(104.0, 33.0, 105.0, 34.0)
    projected = project_study_area_to_equal_area(geographic)
    study_area = StudyArea(
        geographic=geographic,
        projected=projected,
        equal_area_crs=EQUAL_AREA_CRS,
        area_km2=float(projected.area) / 1_000_000.0,
    )
    grid = build_stage4_integration_grid(geographic, cell_size_km=25.0)
    rebuilt = build_clipped_grid(projected, cell_size_km=25.0)
    pair = next(
        (left, right)
        for left in rebuilt.cells
        for right in rebuilt.cells
        if (left.row, left.column) < (right.row, right.column)
        and abs(left.row - right.row) + abs(left.column - right.column) == 1
        and left.clipped_geometry.boundary.intersection(right.clipped_geometry.boundary).length
        > 0.0
    )
    boundary = (
        pair[0]
        .clipped_geometry.boundary.intersection(pair[1].clipped_geometry.boundary)
        .representative_point()
    )
    inverse = Transformer.from_crs(
        CRS.from_user_input(EQUAL_AREA_CRS),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    longitude, latitude = inverse.transform(boundary.x, boundary.y)
    catalog = Stage4TargetCatalog(
        event_id=np.asarray(["boundary-event"], dtype=np.str_),
        origin_time_utc=(datetime(2024, 1, 2, tzinfo=UTC),),
        available_at_utc=(datetime(2024, 1, 2, tzinfo=UTC),),
        longitude=np.asarray([longitude]),
        latitude=np.asarray([latitude]),
        x_m=np.asarray([boundary.x]),
        y_m=np.asarray([boundary.y]),
        magnitude=np.asarray([5.5]),
        inside_study_area=np.asarray([True], dtype=np.bool_),
        source_content_sha256="9" * 64,
        source_schema_sha256="a" * 64,
    )
    assignments = map_targets_to_frozen_primary_grid(
        catalog,
        study_area=study_area,
        primary_grid=grid,
    )
    selected = min(pair, key=lambda cell: (cell.row, cell.column, cell.id))
    selected_index = grid.cell_ids.index(selected.id)
    assert assignments.cell_indices.tolist() == [selected_index]
    assert assignments.cell_ids == (selected.id,)
    assert (int(grid.rows[selected_index]), int(grid.columns[selected_index])) == (
        selected.row,
        selected.column,
    )
    receipt = BoundTargetCellAssignments.bind(catalog, assignments, primary_grid=grid)
    assert receipt.assignment_sha256 == canonical_json_sha256_for_receipt(receipt)


def canonical_json_sha256_for_receipt(receipt: BoundTargetCellAssignments) -> str:
    payload = {
        "schema_version": 1,
        "catalog_event_order_sha256": receipt.catalog_event_order_sha256,
        "grid_id": receipt.assignments.grid_id,
        "event_ids": list(receipt.assignments.event_ids),
        "cell_indices": [int(value) for value in receipt.assignments.cell_indices],
        "cell_ids": list(receipt.assignments.cell_ids),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
