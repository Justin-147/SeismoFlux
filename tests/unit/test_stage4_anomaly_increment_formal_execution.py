from __future__ import annotations

import ast
import dataclasses
import inspect
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pytest
from pyproj import CRS, Transformer
from shapely.geometry import box

import seismoflux.anomaly_increment.formal_execution as formal_execution
from seismoflux.anomaly_increment.background_adapter import (
    FROZEN_COMPENSATOR_DOMAIN_ID,
    FROZEN_STUDY_AREA_SHA256,
    BackgroundDomainBinding,
    Stage4BackgroundFit,
    resolve_frozen_background_snapshot,
)
from seismoflux.anomaly_increment.contracts import FeatureColumnContract, FloatArray
from seismoflux.anomaly_increment.formal_assembly import (
    FrozenCellSupport,
    ProspectiveIssuePlan,
    VerifiedStage3Issue,
)
from seismoflux.anomaly_increment.formal_execution import (
    AuthorizedFormalMaterialization,
    EvaluationZoneReceipt,
    FrozenCellZoneMapping,
    TargetBlindFormalContext,
    VerifiedFormalProtocol,
    build_formal_placebo_source_wiring,
    build_stage4_in_memory_plan,
    materialize_after_authorized_target,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4GridFamily,
    Stage4IntegrationGrid,
    build_stage4_grid_family,
)
from seismoflux.anomaly_increment.runner import (
    ExposurePlan,
    FitScopePlan,
    HorizonAssessmentPlan,
    Stage4ScoringPlan,
    TimePermutationPools,
)
from seismoflux.anomaly_increment.scoring_pipeline import (
    AssembledEvaluationExposure,
    AssembledFitScope,
    AssembledProspectiveIssue,
    FeatureLayout,
)
from seismoflux.anomaly_increment.targets import Stage4TargetCatalog
from seismoflux.background.catalog import StudyArea
from seismoflux.background.grid import EQUAL_AREA_CRS, project_study_area_to_equal_area
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.state import AnomalyState

DESIGN_SHA256 = "b" * 64
RANDOM_SEAL_SHA256 = "c" * 64
ZONE_MANIFEST_SHA256 = "d" * 64
SOURCE_COLUMNS = ("coverage", "snapshot", "dynamic")
EVALUATION_IDS = (
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
)


def _background_protocol() -> dict[str, object]:
    return {
        "inputs": {"study_area": {"sha256": FROZEN_STUDY_AREA_SHA256}},
        "background": {
            "background_variant_id": "spatial_poisson/gaussian_kde_bw75km",
            "family": "spatial_poisson",
            "bandwidth_km": 75.0,
            "model_reselection_forbidden": True,
            "development": {
                "snapshot_id": "fold_4",
                "parameter_snapshot_id": (
                    "83a0c60d4b62ba6a6e849ac2d5f430001d054b7aec3af40f76193180a18bf4c5"
                ),
                "fit_end_utc": "2019-12-31T16:00:00Z",
                "support_id": "local-support-788851371baf0e3b",
                "compensator_domain_id": FROZEN_COMPENSATOR_DOMAIN_ID,
                "common_mc": 4.0,
                "supported_area_fraction": 1.0,
            },
            "validation": {
                "snapshot_id": "final_validation",
                "parameter_snapshot_id": (
                    "252f14cad07205b10c1a605fdd21613044bc4072c98bcaa74cf357b7d766ed02"
                ),
                "fit_end_utc": "2023-06-30T16:00:00Z",
                "support_id": "local-support-f6816ab6c6581306",
                "compensator_domain_id": FROZEN_COMPENSATOR_DOMAIN_ID,
                "common_mc": 4.0,
                "supported_area_fraction": 1.0,
            },
        },
    }


def _protocol_projection() -> VerifiedFormalProtocol:
    protocol = _background_protocol()
    return VerifiedFormalProtocol(
        protocol_design_sha256=DESIGN_SHA256,
        random_input_seal_sha256=RANDOM_SEAL_SHA256,
        development_snapshot=resolve_frozen_background_snapshot(
            protocol,
            evaluation_id="development-fold-1",
        ),
        formal_validation_snapshot=resolve_frozen_background_snapshot(
            protocol,
            evaluation_id="formal-validation",
        ),
    )


def _study_and_grids() -> tuple[StudyArea, Stage4GridFamily]:
    geographic = box(103.0, 28.0, 106.0, 31.0)
    projected = project_study_area_to_equal_area(geographic)
    study_area = StudyArea(
        geographic=geographic,
        projected=projected,
        equal_area_crs=EQUAL_AREA_CRS,
        area_km2=float(projected.area) / 1_000_000.0,
    )
    grid_family = build_stage4_grid_family(geographic)
    if grid_family.primary_25km.cell_count < 39:
        raise AssertionError("synthetic formal execution grid must cover all 39 strata")
    return study_area, grid_family


def _layout() -> FeatureLayout:
    return FeatureLayout(
        coverage_sources=("coverage",),
        snapshot_sources=("coverage", "snapshot"),
        dynamic_sources=SOURCE_COLUMNS,
    )


def _contracts() -> tuple[FeatureColumnContract, ...]:
    return tuple(
        FeatureColumnContract(
            source_column=name,
            logical_feature=name,
            value_output_column=f"{name}_value",
            missing_output_column=f"{name}_missing",
            transform="identity_finite",
        )
        for name in SOURCE_COLUMNS
    )


def _issue_time(issue_date: date) -> datetime:
    return datetime(issue_date.year, issue_date.month, issue_date.day, tzinfo=UTC) - timedelta(
        hours=8
    )


def _feature_table(
    issue_date: date,
    *,
    grid: Stage4IntegrationGrid,
    ordinal: int,
) -> pa.Table:
    issue_time = _issue_time(issue_date)
    report_id = f"formal-report-{issue_date.isoformat()}"
    index = np.arange(grid.cell_count, dtype=np.float64)
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
            "coverage": pa.array(ordinal + index, pa.float64()),
            "snapshot": pa.array(10.0 * ordinal + index, pa.float64()),
            "dynamic": pa.array(100.0 * ordinal + index, pa.float64()),
        }
    )


def _verified_issue(
    issue_date: date,
    *,
    grid: Stage4IntegrationGrid,
    ordinal: int,
) -> VerifiedStage3Issue:
    issue_time = _issue_time(issue_date)
    summary = cast(
        AnomalyState,
        SimpleNamespace(
            issue_report_id=f"formal-report-{issue_date.isoformat()}",
            issue_time_utc=issue_time,
        ),
    )
    snapshot = Stage3IssueSnapshot(
        issue_index=ordinal,
        issue_time_utc=issue_time,
        summary=summary,
        entities=(),
        state_snapshot_id=f"{ordinal:x}" * 64,
        lineage_digest=f"{ordinal + 3:x}" * 64,
    )
    return VerifiedStage3Issue.bind(
        _feature_table(issue_date, grid=grid, ordinal=ordinal),
        snapshot,
        primary_grid=grid,
        source_columns=SOURCE_COLUMNS,
    )


def _scope(evaluation_id: str) -> FitScopePlan:
    formal = evaluation_id == "formal-validation"
    horizons = (7, 30, 90, 180, 365) if formal else (7, 30, 90)
    partition: Literal["development", "validation"] = "validation" if formal else "development"
    fit_issue_id = "anomaly-issue-2024-01-01" if formal else "formal-fit-issue"
    assessment_issue_id = "anomaly-issue-2024-01-09" if formal else "formal-assessment-issue"
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
            fit_issue_ids=(fit_issue_id,),
            assessment_issue_ids=(assessment_issue_id,),
        ),
        validation_refit_forbidden=formal,
    )


def _scoring_plan(scopes: tuple[FitScopePlan, ...] | None = None) -> Stage4ScoringPlan:
    return cast(
        Stage4ScoringPlan,
        SimpleNamespace(
            protocol_design_sha256=DESIGN_SHA256,
            random_input_seal_sha256=RANDOM_SEAL_SHA256,
            fit_scopes=(
                tuple(_scope(item) for item in EVALUATION_IDS) if scopes is None else scopes
            ),
        ),
    )


def _catalog(grid: Stage4IntegrationGrid) -> Stage4TargetCatalog:
    fit_start = _issue_time(date(2024, 1, 1))
    assessment_start = _issue_time(date(2024, 1, 9))
    rows = (
        ("at-anchor", datetime(1970, 1, 1, tzinfo=UTC), 4.0, 0),
        ("known-2018", datetime(2018, 1, 1, tzinfo=UTC), 4.2, 0),
        ("at-dev-cutoff", datetime(2019, 12, 31, 16, tzinfo=UTC), 4.0, 1),
        ("after-dev", datetime(2020, 1, 2, tzinfo=UTC), 4.5, 0),
        ("fit-supported", fit_start + timedelta(days=0.5), 5.5, 0),
        ("fit-unsupported", fit_start + timedelta(days=1.0), 5.6, 1),
        ("assessment-supported", assessment_start + timedelta(days=0.5), 5.5, 0),
        ("assessment-unsupported", assessment_start + timedelta(days=1.0), 5.7, 1),
        ("assessment-m6-end", assessment_start + timedelta(days=7.0), 6.2, 0),
    )
    xy = np.asarray([grid.query_xy_m[item[3]] for item in rows], dtype=np.float64)
    inverse = Transformer.from_crs(
        CRS.from_user_input(EQUAL_AREA_CRS),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    longitude, latitude = inverse.transform(xy[:, 0], xy[:, 1])
    available = tuple(
        origin if event_id in {"at-anchor", "at-dev-cutoff"} else origin + timedelta(hours=1)
        for event_id, origin, _, _ in rows
    )
    return Stage4TargetCatalog(
        event_id=np.asarray([item[0] for item in rows], dtype=np.str_),
        origin_time_utc=tuple(item[1] for item in rows),
        available_at_utc=available,
        longitude=np.asarray(longitude, dtype=np.float64),
        latitude=np.asarray(latitude, dtype=np.float64),
        x_m=xy[:, 0],
        y_m=xy[:, 1],
        magnitude=np.asarray([item[2] for item in rows], dtype=np.float64),
        inside_study_area=np.ones(len(rows), dtype=np.bool_),
        source_content_sha256="a" * 64,
        source_schema_sha256="e" * 64,
    )


def _context_and_catalog() -> tuple[TargetBlindFormalContext, Stage4TargetCatalog]:
    study_area, grid_family = _study_and_grids()
    primary = grid_family.primary_25km
    protocol = _protocol_projection()
    domain = BackgroundDomainBinding.from_verified_grid_family(
        grid_family,
        study_area_sha256=FROZEN_STUDY_AREA_SHA256,
        compensator_domain_id=FROZEN_COMPENSATOR_DOMAIN_ID,
    )
    mask = np.ones(primary.cell_count, dtype=np.bool_)
    mask[1] = False
    supports = tuple(
        FrozenCellSupport(
            evaluation_id=cast(formal_execution.EvaluationId, evaluation_id),
            grid_id=primary.grid_id,
            cell_ids=primary.cell_ids,
            support_id=protocol.snapshot_for(
                cast(formal_execution.EvaluationId, evaluation_id)
            ).support_id,
            compensator_domain_id=FROZEN_COMPENSATOR_DOMAIN_ID,
            supported_cell_mask=mask,
        )
        for evaluation_id in EVALUATION_IDS
    )
    all_zones = tuple(f"construction-zone-{index:02d}" for index in range(1, 40))
    zones = tuple(all_zones[index % len(all_zones)] for index in range(primary.cell_count))
    zone_mapping = FrozenCellZoneMapping.bind(
        grid_id=primary.grid_id,
        cell_ids=primary.cell_ids,
        construction_zone_ids=zones,
        all_construction_zone_ids=all_zones,
        manifest_sha256=ZONE_MANIFEST_SHA256,
    )
    issues = (
        _verified_issue(date(2024, 1, 1), grid=primary, ordinal=1),
        _verified_issue(date(2024, 1, 9), grid=primary, ordinal=2),
        _verified_issue(date(2024, 2, 1), grid=primary, ordinal=3),
    )
    context = TargetBlindFormalContext(
        protocol=protocol,
        scoring_plan=_scoring_plan(),
        study_area=study_area,
        grid_family=grid_family,
        background_domain=domain,
        verified_issues=issues,
        feature_layout=_layout(),
        cell_supports=supports,
        prospective_plans=(ProspectiveIssuePlan(date(2024, 2, 1), 30, "M5_6"),),
        cell_zone_mapping=zone_mapping,
    )
    return context, _catalog(primary)


@pytest.fixture(scope="module")
def formal_inputs() -> tuple[TargetBlindFormalContext, Stage4TargetCatalog]:
    return _context_and_catalog()


@pytest.fixture(scope="module")
def materialized(
    formal_inputs: tuple[TargetBlindFormalContext, Stage4TargetCatalog],
) -> AuthorizedFormalMaterialization:
    context, catalog = formal_inputs
    return materialize_after_authorized_target(context, catalog)


def test_target_blind_context_and_bridge_source_have_no_file_or_locked_test_capability() -> None:
    field_contract = {
        field.name: str(field.type) for field in dataclasses.fields(TargetBlindFormalContext)
    }
    assert tuple(field_contract) == (
        "protocol",
        "scoring_plan",
        "study_area",
        "grid_family",
        "background_domain",
        "verified_issues",
        "feature_layout",
        "cell_supports",
        "prospective_plans",
        "cell_zone_mapping",
    )
    forbidden_context_tokens = ("catalog", "targetcell", "boundtarget", "path", "bytes")
    assert not any(
        token in f"{name}:{annotation}".casefold()
        for name, annotation in field_contract.items()
        for token in forbidden_context_tokens
    )
    assert tuple(inspect.signature(materialize_after_authorized_target).parameters) == (
        "context",
        "catalog",
    )
    source = inspect.getsource(formal_execution)
    assert "locked_test" not in source.casefold()
    tree = ast.parse(source)
    forbidden_calls = {
        "open",
        "Path",
        "read_bytes",
        "read_text",
        "stat",
        "lstat",
        "sha256_file",
        "hash_file",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls
        elif isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_calls


def test_context_rejects_scope_reordering_and_cell_zone_receipt_tampering(
    formal_inputs: tuple[TargetBlindFormalContext, Stage4TargetCatalog],
) -> None:
    context, _ = formal_inputs
    scopes = tuple(context.scoring_plan.fit_scopes)
    wrong_plan = _scoring_plan((scopes[1], scopes[0], scopes[2], scopes[3]))
    with pytest.raises(ValueError, match="four frozen scopes in order"):
        dataclasses.replace(context, scoring_plan=wrong_plan)

    changed_zones = (
        context.cell_zone_mapping.construction_zone_ids[1],
        *context.cell_zone_mapping.construction_zone_ids[1:],
    )
    with pytest.raises(ValueError, match="score-blind receipt"):
        dataclasses.replace(
            context.cell_zone_mapping,
            construction_zone_ids=changed_zones,
        )


def test_authorized_bridge_builds_causal_backgrounds_and_local_support_only(
    materialized: AuthorizedFormalMaterialization,
) -> None:
    backgrounds = materialized.background_fits
    assert tuple(item.snapshot.evaluation_id for item in backgrounds) == EVALUATION_IDS
    assert len({item.scientific_identity_sha256 for item in backgrounds[:3]}) == 1
    assert all("after-dev" not in item.training_event_ids for item in backgrounds[:3])
    assert "after-dev" in backgrounds[3].training_event_ids
    assert all("fit-supported" not in item.training_event_ids for item in backgrounds)
    assert backgrounds[0].snapshot.fit_end_utc < backgrounds[3].snapshot.fit_end_utc
    assert backgrounds[0].snapshot.parameter_snapshot_id != (
        backgrounds[3].snapshot.parameter_snapshot_id
    )

    assert tuple(item.fit.evaluation_id for item in materialized.assembly.development_scopes) == (
        "development-fold-1",
        "development-fold-2",
        "development-fold-3",
    )
    formal = materialized.assembly.formal_scope
    exposure = next(
        item for item in formal.exposures if item.horizon_days == 7 and item.magnitude_bin == "M5_6"
    )
    assert exposure.all_study_area_event_ids == (
        "assessment-supported",
        "assessment-unsupported",
    )
    assert exposure.supported_event_ids == ("assessment-supported",)
    assert "assessment-unsupported" in exposure.all_study_area_event_ids
    assert len(exposure.cell_order_ids) + 1 == (
        materialized.background_fits[3].grid(25.0).spatial_cell_mass.size
    )
    prospective = materialized.assembly.prospective_issues[0]
    assert not hasattr(prospective, "event_ids")


def test_authorized_placebo_scope_assembler_rebinds_recipient_identity_in_memory(
    materialized: AuthorizedFormalMaterialization,
) -> None:
    assembler = materialized.placebo_scope_assembler
    assert materialized.assembly_context_sha256 == assembler.assembly_context_sha256
    assert len(assembler.assembly_context_sha256) == 64
    observed_tables = {
        issue_id: recipient.table for issue_id, recipient in assembler.recipient_issues
    }
    observed = assembler(observed_tables)
    assert observed.fit.evaluation_id == "formal-validation"
    assert tuple(item.all_study_area_event_ids for item in observed.exposures) == tuple(
        item.all_study_area_event_ids for item in materialized.assembly.formal_scope.exposures
    )

    assessment_id, assessment_recipient = assembler.recipient_issues[-1]
    source_index = assessment_recipient.table.schema.get_field_index("dynamic")
    original_values = np.asarray(
        assessment_recipient.table["dynamic"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    rebuilt_table = assessment_recipient.table.set_column(
        source_index,
        "dynamic",
        pa.array(original_values + 1_000.0, pa.float64()),
    )
    rebuilt_tables = dict(observed_tables)
    rebuilt_tables[assessment_id] = rebuilt_table
    rebound = assembler.bind_recipient_issues(rebuilt_tables)
    rebound_by_time = {item.issue_time_utc: item for item in rebound}
    changed = rebound_by_time[assessment_recipient.issue_time_utc]
    assert changed.state_snapshot_id == assessment_recipient.state_snapshot_id
    assert changed.lineage_digest == assessment_recipient.lineage_digest
    assert changed.issue_report_id == assessment_recipient.issue_report_id
    assert changed.table_identity_sha256 != assessment_recipient.table_identity_sha256
    rebuilt = assembler(rebuilt_tables)
    assert not np.array_equal(
        cast(FloatArray, rebuilt.exposures[0].cell_feature_columns["dynamic"]),
        cast(FloatArray, observed.exposures[0].cell_feature_columns["dynamic"]),
    )
    for observed_exposure, rebuilt_exposure in zip(
        observed.exposures,
        rebuilt.exposures,
        strict=True,
    ):
        assert rebuilt_exposure.supported_event_ids == observed_exposure.supported_event_ids
        assert rebuilt_exposure.all_study_area_event_ids == (
            observed_exposure.all_study_area_event_ids
        )
        assert rebuilt_exposure.cell_order_ids == observed_exposure.cell_order_ids
        np.testing.assert_array_equal(
            rebuilt_exposure.background_spatial_mass,
            observed_exposure.background_spatial_mass,
        )

    with pytest.raises(ValueError, match="exactly cover"):
        assembler({assessment_id: rebuilt_table})
    report_index = rebuilt_table.schema.get_field_index("issue_report_id")
    wrong_identity = rebuilt_table.set_column(
        report_index,
        "issue_report_id",
        pa.array(["wrong-recipient"] * rebuilt_table.num_rows, pa.string()),
    )
    with pytest.raises(ValueError, match="recipient report identity"):
        assembler({**observed_tables, assessment_id: wrong_identity})


def test_authorized_materialization_builds_pipeline_plan_and_placebo_wiring(
    materialized: AuthorizedFormalMaterialization,
) -> None:
    plan = build_stage4_in_memory_plan(
        materialized,
        feature_contracts=_contracts(),
        feature_layout=_layout(),
        frozen_input_seal_sha256=RANDOM_SEAL_SHA256,
        model_version="synthetic-stage4-formal-bridge",
    )
    assert plan.development_scopes == materialized.assembly.development_scopes
    assert plan.formal_scope == materialized.assembly.formal_scope
    assert plan.prospective_issues == materialized.assembly.prospective_issues
    assert plan.evaluation_region_binding is materialized.evaluation_region_binding
    assert {(item.horizon_days, item.magnitude_bin) for item in plan.formal_scope.exposures} == {
        (horizon, magnitude)
        for horizon in (7, 30, 90, 180, 365)
        for magnitude in ("M5_6", "M6_plus")
    }
    formal_cells = {
        cell_id for exposure in plan.formal_scope.exposures for cell_id in exposure.cell_order_ids
    }
    formal_events = {
        event_id
        for exposure in plan.formal_scope.exposures
        for event_id in exposure.all_study_area_event_ids
    }
    assert formal_cells <= set(plan.evaluation_region_binding.cell_ids)
    assert formal_events <= set(plan.evaluation_region_binding.event_ids)
    assert plan.prospective_issues

    wiring = build_formal_placebo_source_wiring(materialized)
    pools = materialized.placebo_scope_assembler.scope_plan.time_permutation_pools
    assert wiring.fit_issue_ids == pools.fit_issue_ids
    assert wiring.assessment_issue_ids == pools.assessment_issue_ids
    assert wiring.assembly_context_sha256 == materialized.assembly_context_sha256
    assert wiring.scope_assembler is materialized.placebo_scope_assembler

    with pytest.raises(ValueError, match="frozen input seal"):
        build_stage4_in_memory_plan(
            materialized,
            feature_contracts=_contracts(),
            feature_layout=_layout(),
            frozen_input_seal_sha256="f" * 64,
            model_version="synthetic-stage4-formal-bridge",
        )


def test_cell_and_event_zone_receipts_are_complete_local_and_evaluation_only(
    materialized: AuthorizedFormalMaterialization,
) -> None:
    cell_receipt = materialized.evaluation_cell_zone_mapping
    event_receipt = materialized.evaluation_zone_receipt
    region_binding = materialized.evaluation_region_binding
    assert len(cell_receipt.all_construction_zone_ids) == 39
    assert set(cell_receipt.construction_zone_ids) == set(cell_receipt.all_construction_zone_ids)
    assert event_receipt.all_construction_zone_ids == (cell_receipt.all_construction_zone_ids)
    expected_zone_by_cell = dict(
        zip(cell_receipt.cell_ids, cell_receipt.construction_zone_ids, strict=True)
    )
    expected_event_zones = tuple(
        expected_zone_by_cell[cell_id]
        for cell_id in materialized.cell_assignments.assignments.cell_ids
    )
    assert event_receipt.event_ids == materialized.cell_assignments.assignments.event_ids
    assert event_receipt.construction_zone_ids == expected_event_zones
    assert region_binding.all_construction_zone_ids == cell_receipt.all_construction_zone_ids
    assert region_binding.cell_ids == cell_receipt.cell_ids
    assert region_binding.cell_construction_zone_ids == cell_receipt.construction_zone_ids
    assert region_binding.event_ids == event_receipt.event_ids
    assert region_binding.event_construction_zone_ids == event_receipt.construction_zone_ids
    assert region_binding.cell_mapping_sha256 == cell_receipt.mapping_sha256
    assert region_binding.event_mapping_sha256 == event_receipt.receipt_sha256
    with pytest.raises(ValueError, match="event receipt"):
        dataclasses.replace(
            materialized,
            evaluation_region_binding=dataclasses.replace(
                region_binding,
                event_mapping_sha256="f" * 64,
            ),
        )
    assert cell_receipt.public_export_forbidden is True
    assert event_receipt.public_export_forbidden is True
    assert region_binding.public_export_forbidden is True
    assert not hasattr(cell_receipt, "as_mapping")
    assert not hasattr(event_receipt, "as_mapping")
    event_receipt_fields = {field.name for field in dataclasses.fields(EvaluationZoneReceipt)}
    assert not event_receipt_fields & {
        "cell_ids",
        "longitude",
        "latitude",
        "geometry",
        "wkb",
        "geojson",
    }
    model_field_names = {
        field.name
        for cls in (
            AssembledFitScope,
            AssembledEvaluationExposure,
            AssembledProspectiveIssue,
        )
        for field in dataclasses.fields(cls)
    }
    assert not any("construction_zone" in name for name in model_field_names)


def test_formal_background_verifier_rejects_validation_backfill(
    formal_inputs: tuple[TargetBlindFormalContext, Stage4TargetCatalog],
    materialized: AuthorizedFormalMaterialization,
) -> None:
    context, catalog = formal_inputs
    validation = context.protocol.formal_validation_snapshot
    forged_snapshot = dataclasses.replace(
        validation,
        evaluation_id="development-fold-1",
        role="development",
    )
    forged_development = dataclasses.replace(
        materialized.background_fits[0],
        snapshot=forged_snapshot,
    )
    forged = cast(
        tuple[Stage4BackgroundFit, ...],
        (
            forged_development,
            materialized.background_fits[1],
            materialized.background_fits[2],
            materialized.background_fits[3],
        ),
    )
    with pytest.raises(ValueError, match="verified protocol"):
        formal_execution._verify_rebuilt_backgrounds(
            forged,
            context=context,
            catalog=catalog,
        )
