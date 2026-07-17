from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from pyproj import CRS, Transformer

from seismoflux.anomaly_increment.contracts import FeatureColumnContract
from seismoflux.anomaly_increment.evaluation import frozen_same_recall_budget_grid
from seismoflux.anomaly_increment.formal_assembly import BoundTargetCellAssignments
from seismoflux.anomaly_increment.grid_features import Stage4IntegrationGrid
from seismoflux.anomaly_increment.integration import composite_midpoint_quadrature, lead_decay
from seismoflux.anomaly_increment.model import fit_frozen_target_rate_head
from seismoflux.anomaly_increment.preprocessing import fit_frozen_preprocessor
from seismoflux.anomaly_increment.scoring_pipeline import (
    AssembledEvaluationExposure,
    AssembledFitScope,
    AssembledProspectiveIssue,
    EvaluationRegionBinding,
    EvaluationScope,
    ExposureVariantScore,
    FeatureLayout,
    FittedScope,
    FittedVariant,
    PipelineResult,
    Stage4InMemoryPlan,
)
from seismoflux.anomaly_increment.spatial_dashboard import (
    DisplayContextLayer,
    DisplayStudyArea,
    build_local_spatial_dashboard_html,
)
from seismoflux.anomaly_increment.spatial_results import (
    GenuineProspectiveArchive,
    Stage4SpatialResults,
    build_retrospective_forecast_frames,
    build_retrospective_target_overlays,
    build_score_blind_calendar_selection,
    build_static_spatial_svg,
    build_target_blind_forecast_frames,
    genuine_prospective_archive_content_sha256,
    target_blind_frame_bundle_sha256,
)
from seismoflux.anomaly_increment.targets import (
    Stage4TargetCatalog,
    TargetCellAssignments,
)
from seismoflux.background.grid import EQUAL_AREA_CRS

CELL_IDS = ("cell-a", "cell-b", "cell-c")
ROWS = (0, 0, 0)
COLUMNS = (0, 1, 2)
AREA = np.asarray([300_000.0, 300_000.0, 300_000.0], dtype=np.float64)
MASS = np.asarray([1.0 / 3.0] * 3, dtype=np.float64)
FEATURES = {
    "coverage": np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
    "snapshot": np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
    "trend": np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
}
ISSUE = "2025-01-01"
PROSPECTIVE_ISSUE = "2026-01-01"


def _contract() -> tuple[FeatureColumnContract, ...]:
    return tuple(
        FeatureColumnContract(
            source_column=name,
            logical_feature=name,
            value_output_column=f"{name}_value",
            missing_output_column=f"{name}_missing",
            transform="identity_finite",
        )
        for name in ("coverage", "snapshot", "trend")
    )


def _layout() -> FeatureLayout:
    return FeatureLayout(
        coverage_sources=("coverage",),
        snapshot_sources=("coverage", "snapshot"),
        dynamic_sources=("coverage", "snapshot", "trend"),
    )


def _fit_input(evaluation_id: str) -> AssembledFitScope:
    quadrature = composite_midpoint_quadrature(7.0)
    return AssembledFitScope(
        evaluation_id=evaluation_id,
        preprocessing_fit_columns=FEATURES,
        issue_cell_feature_columns=FEATURES,
        event_feature_columns={name: np.asarray([1.0], dtype=np.float64) for name in FEATURES},
        background_spatial_mass_by_row_and_bin=np.asarray(
            [[1.0 / 3.0, 1.0 / 3.0]] * 3,
            dtype=np.float64,
        ),
        midpoint_widths_days=quadrature.widths_days,
        midpoint_decays=np.asarray(lead_decay(quadrature.lead_midpoints_days)),
        event_background_intensity=np.asarray([(1.0 / 3.0) / 300_000.0]),
        event_decay=np.asarray([lead_decay(1.0)]),
        event_magnitude_bin_ids=("M5_6",),
        training_event_counts={"M5_6": 1, "M6_plus": 0},
        background_exposures={"M5_6": 7.0, "M6_plus": 7.0},
    )


def _exposure(
    evaluation_id: str,
    horizon: int,
    *,
    targets: bool,
) -> AssembledEvaluationExposure:
    event_ids = ("event-1",) if targets else ()
    return AssembledEvaluationExposure(
        evaluation_id=evaluation_id,
        issue_date=ISSUE,
        horizon_days=horizon,
        magnitude_bin="M5_6",
        support_id="support-v1",
        compensator_domain_id="domain-v1",
        cell_order_ids=CELL_IDS,
        cell_rows=ROWS,
        cell_columns=COLUMNS,
        cell_area_km2=AREA,
        background_spatial_mass=MASS,
        cell_feature_columns=FEATURES,
        supported_event_ids=event_ids,
        all_study_area_event_ids=event_ids,
        event_cell_indices=(2,) if targets else (),
        event_lead_days=np.asarray([1.0] if targets else [], dtype=np.float64),
    )


def _prospective() -> tuple[AssembledProspectiveIssue, ...]:
    return tuple(
        AssembledProspectiveIssue(
            issue_date=PROSPECTIVE_ISSUE,
            horizon_days=horizon,
            magnitude_bin="M5_6",
            cell_order_ids=CELL_IDS,
            cell_area_km2=AREA,
            background_spatial_mass=MASS,
            cell_feature_columns=FEATURES,
        )
        for horizon in (7, 30, 90)
    )


def _plan() -> Stage4InMemoryPlan:
    development = tuple(
        EvaluationScope(
            fit=_fit_input(f"development-fold-{index}"),
            exposures=(_exposure(f"development-fold-{index}", 7, targets=False),),
        )
        for index in (1, 2, 3)
    )
    return Stage4InMemoryPlan(
        feature_contracts=_contract(),
        feature_layout=_layout(),
        development_scopes=development,
        formal_scope=EvaluationScope(
            fit=_fit_input("formal-validation"),
            exposures=tuple(
                _exposure("formal-validation", horizon, targets=True) for horizon in (7, 30, 90)
            ),
        ),
        prospective_issues=_prospective(),
        evaluation_region_binding=EvaluationRegionBinding(
            all_construction_zone_ids=tuple(
                f"construction-zone-{index:02d}" for index in range(1, 40)
            ),
            cell_ids=CELL_IDS,
            cell_construction_zone_ids=(
                "construction-zone-01",
                "construction-zone-02",
                "construction-zone-03",
            ),
            event_ids=("event-1",),
            event_construction_zone_ids=("construction-zone-03",),
            cell_mapping_sha256="a" * 64,
            event_mapping_sha256="b" * 64,
        ),
        frozen_input_seal_sha256="c" * 64,
        model_version="stage4-spatial-test-v1",
    )


def _plan_with_display_calendar() -> Stage4InMemoryPlan:
    base = _plan()
    formal_dates = (
        "2025-01-01",
        "2025-01-08",
        "2025-01-15",
        "2025-01-22",
        "2025-01-29",
    )
    prospective_dates = ("2025-12-18", "2025-12-25", "2026-01-01")
    formal = tuple(
        dataclasses.replace(_exposure("formal-validation", horizon, targets=True), issue_date=day)
        for horizon in (7, 30, 90)
        for day in formal_dates
    )
    prospective = tuple(
        dataclasses.replace(item, issue_date=day)
        for item in _prospective()
        for day in prospective_dates
    )
    return dataclasses.replace(
        base,
        formal_scope=EvaluationScope(
            fit=base.formal_scope.fit,
            exposures=formal,
        ),
        prospective_issues=prospective,
    )


def _fitted() -> FittedScope:
    preprocessor = fit_frozen_preprocessor(_contract(), FEATURES)
    rate_head = fit_frozen_target_rate_head(
        training_event_counts={"M5_6": 1, "M6_plus": 0},
        background_exposures={"M5_6": 7.0, "M6_plus": 7.0},
    )
    return FittedScope(
        evaluation_id="formal-validation",
        preprocessor=preprocessor,
        rate_head=rate_head,
        variants=(
            FittedVariant(
                variant="background_no_increment",
                design_column_indices=(),
                beta=np.asarray([], dtype=np.float64),
                fit_result=None,
            ),
            FittedVariant(
                variant="coverage_only",
                design_column_indices=(0,),
                beta=np.asarray([0.0]),
                fit_result=None,
            ),
            FittedVariant(
                variant="snapshot",
                design_column_indices=(0, 2),
                beta=np.asarray([0.0, 0.0]),
                fit_result=None,
            ),
            FittedVariant(
                variant="dynamic",
                design_column_indices=(0, 2, 4),
                beta=np.asarray([0.0, 0.0, 1.0]),
                fit_result=None,
            ),
        ),
        primary_fit_evidence_insufficient=False,
        optimizer_run_count=0,
    )


def _score(exposure: AssembledEvaluationExposure) -> ExposureVariantScore:
    intensities = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    order = np.asarray([2, 1, 0], dtype=np.int64)
    cumulative = np.cumsum(AREA[order], dtype=np.float64)
    budgets = np.asarray(frozen_same_recall_budget_grid(), dtype=np.float64)
    counts = np.searchsorted(cumulative, budgets + 1.0e-9, side="right")
    exact = np.zeros(budgets.size, dtype=np.float64)
    positive = counts > 0
    exact[positive] = cumulative[counts[positive] - 1]
    return ExposureVariantScore(
        evaluation_id="formal-validation",
        issue_date=exposure.issue_date,
        horizon_days=exposure.horizon_days,
        magnitude_bin="M5_6",
        variant="dynamic",
        supported_event_ids=("event-1",),
        all_study_area_event_ids=("event-1",),
        event_log_intensities=(0.0,),
        integrated_intensity=6.0,
        cell_integrated_intensities=tuple(float(value) for value in intensities),
        alarm_exact_selected_area_km2=exact,
        alarm_prefix_cell_counts=tuple(int(value) for value in counts),
        supported_event_alarm_ranks=(0,),
        selected_cell_ids_at_600000_km2=("cell-c", "cell-b"),
        strict_hit_event_ids_at_600000_km2=("event-1",),
    )


def _result(plan: Stage4InMemoryPlan) -> PipelineResult:
    scores = tuple(_score(exposure) for exposure in plan.formal_scope.exposures)
    return cast(
        PipelineResult,
        SimpleNamespace(
            fitted_scopes=(_fitted(),),
            exposure_scores=scores,
            dynamic_g2=SimpleNamespace(status="passed"),
            result_fingerprint_sha256="d" * 64,
        ),
    )


def _grid() -> Stage4IntegrationGrid:
    lon = np.asarray([100.0, 101.0, 102.0], dtype=np.float64)
    lat = np.asarray([30.0, 30.0, 30.0], dtype=np.float64)
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_user_input(EQUAL_AREA_CRS),
        always_xy=True,
    )
    x_m, y_m = transformer.transform(lon, lat)
    return Stage4IntegrationGrid(
        grid_id="frozen-primary-grid-v1",
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=25.0,
        cell_ids=CELL_IDS,
        rows=np.asarray(ROWS, dtype=np.int64),
        columns=np.asarray(COLUMNS, dtype=np.int64),
        query_xy_m=np.column_stack((x_m, y_m)).astype(np.float64),
        clipped_area_km2=AREA,
    )


def _catalog(grid: Stage4IntegrationGrid) -> tuple[Stage4TargetCatalog, BoundTargetCellAssignments]:
    origin = datetime.fromisoformat("2025-01-02T00:00:00+08:00").astimezone(UTC)
    x_m, y_m = grid.query_xy_m[2]
    catalog = Stage4TargetCatalog(
        event_id=np.asarray(["event-1"], dtype=np.str_),
        origin_time_utc=(origin,),
        available_at_utc=(origin,),
        longitude=np.asarray([102.0]),
        latitude=np.asarray([30.0]),
        x_m=np.asarray([x_m]),
        y_m=np.asarray([y_m]),
        magnitude=np.asarray([5.4]),
        inside_study_area=np.asarray([True], dtype=np.bool_),
        source_content_sha256="e" * 64,
        source_schema_sha256="f" * 64,
    )
    assignments = BoundTargetCellAssignments.bind(
        catalog,
        TargetCellAssignments(
            event_ids=("event-1",),
            cell_indices=np.asarray([2], dtype=np.int64),
            cell_ids=("cell-c",),
            grid_id=grid.grid_id,
        ),
        primary_grid=grid,
    )
    return catalog, assignments


def _context() -> DisplayContextLayer:
    return DisplayContextLayer(
        layer_id="cross-boundary-fault",
        label="显示断层",
        layer_kind="fault",
        geometry_geojson={
            "type": "LineString",
            "coordinates": [[90.0, 25.0], [110.0, 35.0]],
        },
        source_content_sha256="1" * 64,
    )


def _study_area() -> DisplayStudyArea:
    return DisplayStudyArea(
        study_area_id="frozen-study-area-v1",
        geometry_geojson={
            "type": "Polygon",
            "coordinates": [[[99.0, 29.0], [103.0, 29.0], [103.0, 31.0], [99.0, 29.0]]],
        },
        source_content_sha256="9" * 64,
    )


def test_real_scores_and_target_blind_fit_generate_complete_grid_frames() -> None:
    plan = _plan()
    result = _result(plan)
    grid = _grid()
    selection = build_score_blind_calendar_selection(plan)
    assert selection.retrospective_keys == ((ISSUE, 7), (ISSUE, 30), (ISSUE, 90))
    assert selection.prospective_keys == (
        (PROSPECTIVE_ISSUE, 7),
        (PROSPECTIVE_ISSUE, 30),
        (PROSPECTIVE_ISSUE, 90),
    )
    assert selection.retrospective_support_keys == ()
    assert selection.prospective_support_keys == ()
    assert selection.score_or_target_consulted is False

    retrospective = build_retrospective_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
    )
    frame = retrospective[0]
    assert frame.cell_ids == CELL_IDS
    assert frame.relative_strength == pytest.approx((0.5, 1.0, 1.5))
    assert frame.rank_percentile == pytest.approx((100.0 / 3.0, 200.0 / 3.0, 100.0))
    assert frame.alarm_selected.tolist() == [False, True, True]
    assert frame.selected_alarm_area_km2 == 600_000.0
    assert frame.forecast_status == "retrospective_evaluation"

    target_blind = build_target_blind_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
    )
    assert len(target_blind) == 3
    assert all(
        item.forecast_status == "retrospective_generated_target_blind_shadow"
        for item in target_blind
    )
    assert all(item.evidence_status == "not_evaluated_target_blind" for item in target_blind)
    assert all(item.source_result_fingerprint_sha256 is None for item in target_blind)
    assert {item.source_model_fingerprint_sha256 for item in target_blind} == {
        retrospective[0].source_model_fingerprint_sha256
    }
    assert all(item.cell_ids == CELL_IDS for item in target_blind)
    assert all(item.alarm_selected.shape == (3,) for item in target_blind)


def test_previous_alarm_context_uses_immediate_frozen_calendar_predecessor() -> None:
    plan = _plan_with_display_calendar()
    result = _result(plan)
    selection = build_score_blind_calendar_selection(plan)
    assert selection.retrospective_keys[:3] == (
        ("2025-01-01", 7),
        ("2025-01-15", 7),
        ("2025-01-29", 7),
    )
    assert selection.retrospective_support_keys[:2] == (
        ("2025-01-08", 7),
        ("2025-01-22", 7),
    )
    assert selection.prospective_keys[0] == ("2026-01-01", 7)
    assert selection.prospective_support_keys[0] == ("2025-12-25", 7)

    retrospective = build_retrospective_forecast_frames(
        result,
        plan,
        _grid(),
        selection=selection,
    )
    roles = {(item.issue_date, item.horizon_days): item.display_role for item in retrospective}
    assert roles[("2025-01-29", 7)] == "primary"
    assert roles[("2025-01-22", 7)] == "previous_context"

    target_blind = build_target_blind_forecast_frames(
        result,
        plan,
        _grid(),
        selection=selection,
    )
    assert {(item.issue_date, item.horizon_days, item.display_role) for item in target_blind} >= {
        ("2026-01-01", 7, "primary"),
        ("2025-12-25", 7, "previous_context"),
    }


def test_adapter_rejects_fabricated_score_grid_mismatch_and_future_target_field() -> None:
    plan = _plan()
    result = _result(plan)
    grid = _grid()
    selection = build_score_blind_calendar_selection(plan)
    scores = list(result.exposure_scores)
    scores[0] = dataclasses.replace(
        scores[0],
        selected_cell_ids_at_600000_km2=("cell-a", "cell-b"),
    )
    forged_result = cast(
        PipelineResult,
        SimpleNamespace(
            fitted_scopes=result.fitted_scopes,
            exposure_scores=tuple(scores),
            dynamic_g2=result.dynamic_g2,
            result_fingerprint_sha256=result.result_fingerprint_sha256,
        ),
    )
    with pytest.raises(ValueError, match="fabricated or altered"):
        build_retrospective_forecast_frames(
            forged_result,
            plan,
            grid,
            selection=selection,
        )

    bad_exposures = list(plan.formal_scope.exposures)
    bad_exposures[0] = dataclasses.replace(bad_exposures[0], cell_rows=(1, 1, 1))
    bad_plan = dataclasses.replace(
        plan,
        formal_scope=EvaluationScope(
            fit=plan.formal_scope.fit,
            exposures=tuple(bad_exposures),
        ),
    )
    with pytest.raises(ValueError, match="cell rows differ"):
        build_retrospective_forecast_frames(
            result,
            bad_plan,
            grid,
            selection=build_score_blind_calendar_selection(bad_plan),
        )

    prospective = list(plan.prospective_issues)
    prospective[0] = dataclasses.replace(
        prospective[0],
        cell_feature_columns={"future_target": np.asarray([0.0, 0.0, 0.0])},
    )
    future_plan = dataclasses.replace(plan, prospective_issues=tuple(prospective))
    with pytest.raises(ValueError, match="forbidden field"):
        build_target_blind_forecast_frames(
            result,
            future_plan,
            grid,
            selection=build_score_blind_calendar_selection(future_plan),
        )


def test_authorized_overlay_uses_real_catalog_coordinates_and_real_hit_ids() -> None:
    plan = _plan()
    result = _result(plan)
    grid = _grid()
    catalog, assignments = _catalog(grid)
    overlays = build_retrospective_target_overlays(
        result,
        plan,
        grid,
        catalog,
        assignments,
        selection=build_score_blind_calendar_selection(plan),
    )
    assert len(overlays) == 3
    assert all(item.event_ids == ("event-1",) for item in overlays)
    assert all(item.hit_event_ids_at_600000_km2 == ("event-1",) for item in overlays)
    assert all(item.longitude.tolist() == [102.0] for item in overlays)
    assert all(item.covered_by_600000_km2 == (True,) for item in overlays)

    foreign_grid = dataclasses.replace(grid, grid_id="foreign-grid")
    with pytest.raises(ValueError, match="another frozen grid"):
        build_retrospective_target_overlays(
            result,
            plan,
            foreign_grid,
            catalog,
            assignments,
            selection=build_score_blind_calendar_selection(plan),
        )


def test_true_archive_requires_exact_frame_receipt_and_preissue_time() -> None:
    plan = _plan()
    result = _result(plan)
    grid = _grid()
    selection = build_score_blind_calendar_selection(plan)
    shadow = build_target_blind_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
    )
    archived_at = datetime.fromisoformat("2026-01-01T00:00:00+08:00").astimezone(UTC)
    bundle_sha256 = target_blind_frame_bundle_sha256(shadow)
    receipt = GenuineProspectiveArchive(
        archive_id="archive-1",
        archive_content_sha256=genuine_prospective_archive_content_sha256(
            archive_id="archive-1",
            target_blind_frame_bundle_sha256=bundle_sha256,
            model_version=plan.model_version,
            issue_dates=(PROSPECTIVE_ISSUE,),
            archived_at_utc=archived_at,
        ),
        target_blind_frame_bundle_sha256=bundle_sha256,
        model_version=plan.model_version,
        issue_dates=(PROSPECTIVE_ISSUE,),
        archived_at_utc=archived_at,
    )
    genuine = build_target_blind_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
        genuine_prospective_archive=receipt,
    )
    assert all(item.forecast_status == "genuine_prospective_archive" for item in genuine)

    with pytest.raises(ValueError, match="does not bind"):
        dataclasses.replace(receipt, archive_content_sha256="2" * 64)

    with pytest.raises(ValueError, match="target-derived result metadata"):
        target_blind_frame_bundle_sha256(
            (
                dataclasses.replace(
                    shadow[0],
                    forecast_status="retrospective_evaluation",
                    evidence_status="passed",
                    source_result_fingerprint_sha256="d" * 64,
                ),
            )
        )

    late_time = receipt.archived_at_utc + timedelta(seconds=1)
    late = dataclasses.replace(
        receipt,
        archive_content_sha256=genuine_prospective_archive_content_sha256(
            archive_id=receipt.archive_id,
            target_blind_frame_bundle_sha256=receipt.target_blind_frame_bundle_sha256,
            model_version=receipt.model_version,
            issue_dates=receipt.issue_dates,
            archived_at_utc=late_time,
        ),
        archived_at_utc=late_time,
    )
    with pytest.raises(ValueError, match="after its issue time"):
        build_target_blind_forecast_frames(
            result,
            plan,
            grid,
            selection=selection,
            genuine_prospective_archive=late,
        )


def test_static_spatial_svg_is_local_complete_and_context_clipped() -> None:
    plan = _plan()
    result = _result(plan)
    grid = _grid()
    selection = build_score_blind_calendar_selection(plan)
    retrospective = build_retrospective_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
    )
    target_blind = build_target_blind_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
    )
    catalog, assignments = _catalog(grid)
    targets = build_retrospective_target_overlays(
        result,
        plan,
        grid,
        catalog,
        assignments,
        selection=selection,
    )
    previous_h7 = dataclasses.replace(
        retrospective[0],
        issue_date="2024-12-25",
        knowledge_cutoff_utc=datetime.fromisoformat("2024-12-25T00:00:00+08:00").astimezone(UTC),
    )
    svg = build_static_spatial_svg(
        (previous_h7, *retrospective),
        target_blind,
        targets,
        study_area=_study_area(),
        display_context_layers=(_context(),),
    )
    assert svg.startswith("<svg")
    assert 'data-classification="local_restricted_target_bearing_spatial_visualization"' in svg
    for value in (7, 30, 90):
        assert f'data-horizon-days="{value}"' in svg
    assert "回溯生成的目标盲影子预测" in svg
    assert "不能解释为绝对发生率估计" in svg
    assert "概率" not in svg
    assert "报警格全量" in svg
    assert "clip-path=" in svg
    assert 'clip-rule="evenodd"' in svg
    assert "data-study-area-geometry-sha256=" in svg
    assert 'stroke-dasharray="3 2"' in svg
    assert "显示专用上下文" in svg


def test_self_contained_interactive_wiring_has_byte_receipt_and_local_only_classification() -> None:
    plan = _plan()
    result = _result(plan)
    grid = _grid()
    selection = build_score_blind_calendar_selection(plan)
    retrospective = build_retrospective_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
    )
    target_blind = build_target_blind_forecast_frames(
        result,
        plan,
        grid,
        selection=selection,
    )
    catalog, assignments = _catalog(grid)
    targets = build_retrospective_target_overlays(
        result,
        plan,
        grid,
        catalog,
        assignments,
        selection=selection,
    )
    contexts = (_context(),)
    svg = build_static_spatial_svg(
        retrospective,
        target_blind,
        targets,
        study_area=_study_area(),
        display_context_layers=contexts,
    )
    html = build_local_spatial_dashboard_html(
        study_area=_study_area(),
        retrospective_fields=retrospective,
        prospective_fields=target_blind,
        retrospective_targets=targets,
        display_context_layers=contexts,
    )
    wired = Stage4SpatialResults(
        selection=selection,
        retrospective_frames=retrospective,
        target_blind_frames=target_blind,
        retrospective_targets=targets,
        static_svg=svg,
        interactive_html=html,
        interactive_html_bytes=len(html.encode("utf-8")),
        target_blind_frame_bundle_sha256=target_blind_frame_bundle_sha256(target_blind),
    )
    assert wired.local_only_classification == (
        "local_restricted_target_bearing_spatial_visualization"
    )
    assert wired.interactive_html_bytes < 64 * 1024 * 1024
    assert 'id="prospective-spatial-payload"' in wired.interactive_html
    assert 'id="retrospective-target-overlay-payload"' in wired.interactive_html
    assert "fetch(" not in wired.interactive_html
