from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Literal

import numpy as np
import pyarrow as pa
import pytest
from shapely.geometry import box

from seismoflux.anomaly_increment.integration import (
    composite_midpoint_quadrature,
    lead_decay,
)
from seismoflux.anomaly_increment.placebo import SpaceBijection, TimeBijection
from seismoflux.anomaly_increment.placebo_features import (
    COVERAGE_COLUMNS,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_VALUE_COLUMNS,
    TRAJECTORY_BASE_COLUMNS,
    TRAJECTORY_COLUMNS,
    TRAJECTORY_QUALITY_COLUMNS,
    TRAJECTORY_VALUE_COLUMNS,
    rebuild_space_placebo_features,
    rebuild_time_placebo_features,
)
from seismoflux.anomaly_increment.placebo_source import (
    PlaceboSourceUniverse,
    build_placebo_source,
)
from seismoflux.anomaly_increment.scoring_pipeline import (
    AssembledEvaluationExposure,
    AssembledFitScope,
    EvaluationScope,
    PlaceboRequest,
)
from seismoflux.features.anomaly.engine import Stage3FeatureEngine
from seismoflux.features.anomaly.grid import Stage3QueryGrid, build_stage3_query_grid
from seismoflux.features.anomaly.snapshot import (
    Stage3IssueSnapshot,
    build_issue_snapshots,
    spatial_entity_arrays,
)
from seismoflux.features.anomaly.spatial import SPATIAL_SCALES_KM, compute_spatial_features
from seismoflux.features.anomaly.state import build_anomaly_state_history
from seismoflux.features.anomaly.trajectory import compute_trajectory_features

LOCAL = timezone(timedelta(hours=8))
STRATUM_ID = "synthetic-zone:inside"


@dataclass(frozen=True, slots=True)
class _Fixture:
    issue_ids: tuple[str, ...]
    snapshots_by_issue_id: dict[str, Stage3IssueSnapshot]
    tables: dict[str, pa.Table]
    grid: Stage3QueryGrid


def _period(day: date, number: int, *, row_count: int) -> dict[str, object]:
    issue = datetime.combine(day + timedelta(days=1), time.min, LOCAL).astimezone(UTC)
    return {
        "report_id": f"report-{number}",
        "source_file": f"anomaly/report-{number}.xls",
        "report_year": day.year,
        "report_period": number,
        "report_date": day,
        "available_at": issue,
        "row_count": row_count,
        "row_report_date_mismatch_count": 0,
        "row_report_date_before_count": 0,
        "row_report_date_after_count": 0,
        "deformation_row_count": row_count,
        "fluid_row_count": 0,
        "electromagnetic_row_count": 0,
        "cross_fault_row_count": 0,
    }


def _observation(
    period: dict[str, object],
    number: int,
    *,
    entity: str,
    longitude: float,
    reliability_flags: tuple[str, ...] = (),
) -> dict[str, object]:
    report_date = period["report_date"]
    assert isinstance(report_date, date) and not isinstance(report_date, datetime)
    return {
        "observation_id": f"obs-{number}-{entity}",
        "anomaly_id": entity,
        "identity_complete": True,
        "report_date": report_date,
        "source_file": period["source_file"],
        "available_at": period["available_at"],
        "station_id": f"station-{entity}",
        "longitude": longitude,
        "latitude": 35.0,
        "discipline": "\u5f62\u53d8",
        "measurement": f"measurement-{entity}",
        "start_time": datetime.combine(report_date - timedelta(days=30), time.min, LOCAL),
        "is_listed": True,
        "report_state": "\u6301\u7eed",
        "reported_end_time": None,
        "right_censored": True,
        "reliability_flags": reliability_flags,
    }


@pytest.fixture(scope="module")
def synthetic() -> _Fixture:
    periods: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    start = date(2024, 1, 1)
    for issue_index in range(6):
        include_b = issue_index % 2 == 0
        period = _period(
            start + timedelta(weeks=issue_index),
            issue_index + 1,
            row_count=2 if include_b else 1,
        )
        periods.append(period)
        observations.append(
            _observation(
                period,
                issue_index + 1,
                entity="a",
                longitude=110.0,
            )
        )
        if include_b:
            observations.append(
                _observation(
                    period,
                    issue_index + 1,
                    entity="b",
                    longitude=111.0,
                    reliability_flags=("end_time_revised",),
                )
            )

    snapshots = build_issue_snapshots(
        build_anomaly_state_history(observations, periods),
        expected_issue_count=6,
    )
    grid = build_stage3_query_grid(box(109.75, 34.75, 111.25, 35.25))
    engine = Stage3FeatureEngine(snapshots, grid, spatial_workers=1)
    issue_ids = tuple(
        f"anomaly-issue-{snapshot.summary.issue_report_date.isoformat()}" for snapshot in snapshots
    )
    tables = {issue_id: engine.build_next_issue().table for issue_id in issue_ids}
    return _Fixture(
        issue_ids=issue_ids,
        snapshots_by_issue_id=dict(zip(issue_ids, snapshots, strict=True)),
        tables=tables,
        grid=grid,
    )


def _time_bijection(
    recipients: tuple[str, ...],
    donors: tuple[str, ...] | None = None,
) -> TimeBijection:
    donor_ids = recipients if donors is None else donors
    digest = hashlib.sha256(repr((recipients, donor_ids)).encode("utf-8")).hexdigest()
    return TimeBijection(
        recipient_issue_ids=recipients,
        donor_issue_ids=donor_ids,
        mapping_sha256=digest,
        fixed_point_count=sum(
            recipient == donor for recipient, donor in zip(recipients, donor_ids, strict=True)
        ),
    )


def _arrow_buffers(column: pa.ChunkedArray) -> tuple[tuple[bytes | None, ...], ...]:
    return tuple(
        tuple(None if buffer is None else buffer.to_pybytes() for buffer in chunk.buffers())
        for chunk in column.chunks
    )


def _assert_columns_byte_equal(
    observed: pa.Table,
    expected: pa.Table,
    columns: tuple[str, ...],
) -> None:
    for column in columns:
        assert observed.schema.field(column) == expected.schema.field(column)
        assert _arrow_buffers(observed[column]) == _arrow_buffers(expected[column]), column


def _space_inputs(
    synthetic: _Fixture,
    *,
    reverse: bool,
) -> tuple[dict[str, str], dict[tuple[str, str], SpaceBijection]]:
    strata: dict[str, str] = {}
    mappings: dict[tuple[str, str], SpaceBijection] = {}
    for issue_id, snapshot in synthetic.snapshots_by_issue_id.items():
        states = tuple(
            sorted(
                (state for state in snapshot.entities if state.spatial_eligible),
                key=lambda state: state.state_id,
            )
        )
        state_ids = tuple(state.state_id for state in states)
        coordinates = tuple(
            (float(state.longitude), float(state.latitude))
            for state in states
            if state.longitude is not None and state.latitude is not None
        )
        assert len(coordinates) == len(state_ids)
        donor_ids = tuple(reversed(state_ids)) if reverse else state_ids
        coordinate_by_state = dict(zip(state_ids, coordinates, strict=True))
        permuted = tuple(coordinate_by_state[state_id] for state_id in donor_ids)
        moved = sum(
            original != replacement
            for original, replacement in zip(coordinates, permuted, strict=True)
        )
        mappings[(issue_id, STRATUM_ID)] = SpaceBijection(
            entity_state_ids=state_ids,
            donor_state_ids=donor_ids,
            permuted_coordinate_pairs=permuted,
            mapping_sha256=hashlib.sha256(repr((issue_id, donor_ids)).encode("utf-8")).hexdigest(),
            fixed_point_count=sum(
                state_id == donor_id
                for state_id, donor_id in zip(state_ids, donor_ids, strict=True)
            ),
            moved_entity_row_count=moved,
            no_effect=moved == 0,
            coordinate_multiset_verified=True,
        )
        strata.update({state_id: STRATUM_ID for state_id in state_ids})
    return strata, mappings


def _time_permutation(synthetic: _Fixture) -> tuple[TimeBijection, TimeBijection]:
    fit = synthetic.issue_ids[:4]
    assessment = synthetic.issue_ids[4:]
    return (
        _time_bijection(fit, (*fit[1:], fit[0])),
        _time_bijection(assessment, tuple(reversed(assessment))),
    )


def test_identity_time_mapping_reproduces_accepted_snapshot_and_trajectory_columns(
    synthetic: _Fixture,
) -> None:
    fit = synthetic.issue_ids[:4]
    assessment = synthetic.issue_ids[4:]
    observed = rebuild_time_placebo_features(
        synthetic.tables,
        fit_bijection=_time_bijection(fit),
        assessment_bijection=_time_bijection(assessment),
    )

    for issue_id in synthetic.issue_ids:
        _assert_columns_byte_equal(
            observed[issue_id],
            synthetic.tables[issue_id],
            (*SNAPSHOT_COLUMNS, *TRAJECTORY_BASE_COLUMNS, *TRAJECTORY_COLUMNS),
        )


def test_identity_space_mapping_reproduces_accepted_200km_scientific_columns(
    synthetic: _Fixture,
) -> None:
    strata, mappings = _space_inputs(synthetic, reverse=False)
    observed = rebuild_space_placebo_features(
        synthetic.tables,
        synthetic.snapshots_by_issue_id,
        synthetic.grid,
        verified_construction_stratum_by_state_id=strata,
        bijections_by_issue_stratum=mappings,
    )

    for issue_id in synthetic.issue_ids:
        _assert_columns_byte_equal(
            observed[issue_id],
            synthetic.tables[issue_id],
            (*SNAPSHOT_COLUMNS, *TRAJECTORY_BASE_COLUMNS, *TRAJECTORY_COLUMNS),
        )


def test_placebo_coverage_values_and_null_bitmap_exactly_unchanged(
    synthetic: _Fixture,
) -> None:
    fit_mapping, assessment_mapping = _time_permutation(synthetic)
    time_placebo = rebuild_time_placebo_features(
        synthetic.tables,
        fit_bijection=fit_mapping,
        assessment_bijection=assessment_mapping,
    )
    strata, mappings = _space_inputs(synthetic, reverse=True)
    space_placebo = rebuild_space_placebo_features(
        synthetic.tables,
        synthetic.snapshots_by_issue_id,
        synthetic.grid,
        verified_construction_stratum_by_state_id=strata,
        bijections_by_issue_stratum=mappings,
    )

    assert any(
        synthetic.tables[issue_id][column].null_count > 0
        for issue_id in synthetic.issue_ids
        for column in COVERAGE_COLUMNS
    )
    for issue_id in synthetic.issue_ids:
        for placebo in (time_placebo, space_placebo):
            _assert_columns_byte_equal(
                placebo[issue_id],
                synthetic.tables[issue_id],
                COVERAGE_COLUMNS,
            )


def test_fixed_small_permutations_match_stage3_low_level_and_are_deterministic(
    synthetic: _Fixture,
) -> None:
    fit_mapping, assessment_mapping = _time_permutation(synthetic)
    first_time = rebuild_time_placebo_features(
        synthetic.tables,
        fit_bijection=fit_mapping,
        assessment_bijection=assessment_mapping,
    )
    second_time = rebuild_time_placebo_features(
        synthetic.tables,
        fit_bijection=fit_mapping,
        assessment_bijection=assessment_mapping,
    )
    donor_by_recipient = dict((*fit_mapping.pairs, *assessment_mapping.pairs))
    ordered_ids = synthetic.issue_ids
    for recipient_id, donor_id in donor_by_recipient.items():
        _assert_columns_byte_equal(
            first_time[recipient_id],
            synthetic.tables[donor_id],
            SNAPSHOT_COLUMNS,
        )

    poisoned_tables: dict[str, pa.Table] = {}
    for issue_id, original_table in synthetic.tables.items():
        table = original_table
        for column in TRAJECTORY_VALUE_COLUMNS:
            index = table.schema.get_field_index(column)
            field = table.schema.field(index)
            table = table.set_column(
                index,
                field,
                pa.array(np.full(table.num_rows, 12345.0), type=field.type),
            )
        for column in TRAJECTORY_QUALITY_COLUMNS:
            index = table.schema.get_field_index(column)
            field = table.schema.field(index)
            if pa.types.is_boolean(field.type):
                values = pa.array(np.zeros(table.num_rows, dtype=np.bool_), type=field.type)
            else:
                values = pa.array(np.zeros(table.num_rows, dtype=np.int64), type=field.type)
            table = table.set_column(index, field, values)
        poisoned_tables[issue_id] = table
    poison_result = rebuild_time_placebo_features(
        poisoned_tables,
        fit_bijection=fit_mapping,
        assessment_bijection=assessment_mapping,
    )
    for issue_id in ordered_ids:
        _assert_columns_byte_equal(
            poison_result[issue_id],
            first_time[issue_id],
            TRAJECTORY_COLUMNS,
        )

    issue_times = np.asarray(
        [
            np.datetime64(
                synthetic.snapshots_by_issue_id[issue_id].issue_time_utc.replace(tzinfo=None),
                "ns",
            )
            for issue_id in ordered_ids
        ]
    )
    donor_listed = np.stack(
        [
            np.asarray(
                synthetic.tables[donor_by_recipient[issue_id]]["radius_200km__listed_count"]
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
            for issue_id in ordered_ids
        ],
        axis=0,
    )
    time_reference = compute_trajectory_features(issue_times, donor_listed)
    for issue_index, issue_id in enumerate(ordered_ids):
        column = "radius_200km__listed_count__slope_4w_per_week"
        valid = time_reference.valid_masks["slope_4w_per_week"][issue_index]
        expected = pa.chunked_array(
            [
                pa.array(
                    time_reference.features["slope_4w_per_week"][issue_index],
                    type=pa.float64(),
                    mask=~valid,
                )
            ]
        )
        assert _arrow_buffers(first_time[issue_id][column]) == _arrow_buffers(expected)
        assert first_time[issue_id].equals(second_time[issue_id], check_metadata=True)

    strata, mappings = _space_inputs(synthetic, reverse=True)
    first_space = rebuild_space_placebo_features(
        synthetic.tables,
        synthetic.snapshots_by_issue_id,
        synthetic.grid,
        verified_construction_stratum_by_state_id=strata,
        bijections_by_issue_stratum=mappings,
    )
    second_space = rebuild_space_placebo_features(
        synthetic.tables,
        synthetic.snapshots_by_issue_id,
        synthetic.grid,
        verified_construction_stratum_by_state_id=strata,
        bijections_by_issue_stratum=mappings,
    )
    scale_index = SPATIAL_SCALES_KM.index(200.0)
    space_listed_history: list[np.ndarray] = []
    for issue_id, snapshot in synthetic.snapshots_by_issue_id.items():
        mapping = mappings[(issue_id, STRATUM_ID)]
        coordinate_by_state = dict(
            zip(mapping.entity_state_ids, mapping.permuted_coordinate_pairs, strict=True)
        )
        permuted = replace(
            snapshot,
            entities=tuple(
                replace(
                    state,
                    longitude=coordinate_by_state[state.state_id][0],
                    latitude=coordinate_by_state[state.state_id][1],
                )
                if state.spatial_eligible
                else state
                for state in snapshot.entities
            ),
        )
        reference = compute_spatial_features(
            synthetic.grid.query_xy_m,
            spatial_entity_arrays(permuted),
        )
        space_listed_history.append(reference.radius_features["listed_count"][:, scale_index])
        expected = reference.gaussian_features["reliability_weighted_listed_count"][:, scale_index]
        observed = np.asarray(
            first_space[issue_id][SNAPSHOT_VALUE_COLUMNS[0]]
            .combine_chunks()
            .to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        np.testing.assert_array_equal(observed, expected)
        assert first_space[issue_id].equals(second_space[issue_id], check_metadata=True)

    space_trajectory_reference = compute_trajectory_features(
        issue_times,
        np.stack(space_listed_history, axis=0),
    )
    for issue_index, issue_id in enumerate(ordered_ids):
        column = "radius_200km__listed_count__slope_13w_per_week"
        valid = space_trajectory_reference.valid_masks["slope_13w_per_week"][issue_index]
        expected = pa.chunked_array(
            [
                pa.array(
                    space_trajectory_reference.features["slope_13w_per_week"][issue_index],
                    type=pa.float64(),
                    mask=~valid,
                )
            ]
        )
        assert _arrow_buffers(first_space[issue_id][column]) == _arrow_buffers(expected)


def test_space_placebo_keeps_spatially_ineligible_entities_fixed_and_excluded(
    synthetic: _Fixture,
) -> None:
    first_issue = synthetic.issue_ids[0]
    first_snapshot = synthetic.snapshots_by_issue_id[first_issue]
    ineligible_state_id = first_snapshot.entities[-1].state_id
    altered_snapshot = replace(
        first_snapshot,
        entities=tuple(
            replace(
                state,
                spatial_eligible=False,
                spatial_exclusion_reason="synthetic_test_exclusion",
            )
            if state.state_id == ineligible_state_id
            else state
            for state in first_snapshot.entities
        ),
    )
    altered_snapshots = dict(synthetic.snapshots_by_issue_id)
    altered_snapshots[first_issue] = altered_snapshot
    altered = replace(synthetic, snapshots_by_issue_id=altered_snapshots)
    strata, mappings = _space_inputs(altered, reverse=True)

    observed = rebuild_space_placebo_features(
        synthetic.tables,
        altered_snapshots,
        synthetic.grid,
        verified_construction_stratum_by_state_id=strata,
        bijections_by_issue_stratum=mappings,
    )
    reference = compute_spatial_features(
        synthetic.grid.query_xy_m,
        spatial_entity_arrays(altered_snapshot),
    )
    expected = reference.gaussian_features["reliability_weighted_listed_count"][
        :, SPATIAL_SCALES_KM.index(200.0)
    ]
    actual = np.asarray(
        observed[first_issue][SNAPSHOT_VALUE_COLUMNS[0]]
        .combine_chunks()
        .to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_invalid_time_and_space_mappings_fail_closed(synthetic: _Fixture) -> None:
    fit = synthetic.issue_ids[:4]
    with pytest.raises(ValueError, match="disjoint"):
        rebuild_time_placebo_features(
            synthetic.tables,
            fit_bijection=_time_bijection(fit),
            assessment_bijection=_time_bijection(fit),
        )

    strata, mappings = _space_inputs(synthetic, reverse=False)
    strata.pop(next(iter(strata)))
    with pytest.raises(ValueError, match="verified construction stratum"):
        rebuild_space_placebo_features(
            synthetic.tables,
            synthetic.snapshots_by_issue_id,
            synthetic.grid,
            verified_construction_stratum_by_state_id=strata,
            bijections_by_issue_stratum=mappings,
        )

    strata, mappings = _space_inputs(synthetic, reverse=False)
    key = next(iter(mappings))
    original = mappings[key]
    forged_donors = tuple(reversed(original.donor_state_ids))
    mappings[key] = SpaceBijection(
        entity_state_ids=original.entity_state_ids,
        donor_state_ids=forged_donors,
        permuted_coordinate_pairs=original.permuted_coordinate_pairs,
        mapping_sha256="f" * 64,
        fixed_point_count=sum(
            state_id == donor_id
            for state_id, donor_id in zip(
                original.entity_state_ids,
                forged_donors,
                strict=True,
            )
        ),
        moved_entity_row_count=0,
        no_effect=True,
        coordinate_multiset_verified=True,
    )
    with pytest.raises(ValueError, match="disagree with donor states"):
        rebuild_space_placebo_features(
            synthetic.tables,
            synthetic.snapshots_by_issue_id,
            synthetic.grid,
            verified_construction_stratum_by_state_id=strata,
            bijections_by_issue_stratum=mappings,
        )


def test_placebo_feature_api_is_memory_only_and_has_no_trajectory_donor_argument() -> None:
    time_parameters = inspect.signature(rebuild_time_placebo_features).parameters
    space_parameters = inspect.signature(rebuild_space_placebo_features).parameters
    assert "path" not in time_parameters and "path" not in space_parameters
    assert "trajectory_columns" not in time_parameters
    assert tuple(time_parameters) == (
        "issue_tables",
        "fit_bijection",
        "assessment_bijection",
    )


def _assembled_formal_scope(issue_tables: Mapping[str, pa.Table]) -> EvaluationScope:
    assert issue_tables
    quadrature = composite_midpoint_quadrature(7.0)
    columns = {"synthetic": np.asarray([0.0], dtype=np.float64)}
    fit = AssembledFitScope(
        evaluation_id="formal-validation",
        preprocessing_fit_columns=columns,
        issue_cell_feature_columns=columns,
        event_feature_columns=columns,
        background_spatial_mass_by_row_and_bin=np.asarray([[1.0, 1.0]], dtype=np.float64),
        midpoint_widths_days=quadrature.widths_days,
        midpoint_decays=np.asarray(lead_decay(quadrature.lead_midpoints_days)),
        event_background_intensity=np.asarray([1.0], dtype=np.float64),
        event_decay=np.asarray([1.0], dtype=np.float64),
        event_magnitude_bin_ids=("M5_6",),
        training_event_counts={"M5_6": 1, "M6_plus": 0},
        background_exposures={"M5_6": 7.0, "M6_plus": 7.0},
    )
    exposure = AssembledEvaluationExposure(
        evaluation_id="formal-validation",
        issue_date="2025-01-01",
        horizon_days=7,
        magnitude_bin="M5_6",
        support_id="synthetic-support",
        compensator_domain_id="synthetic-domain",
        cell_order_ids=("synthetic-cell",),
        cell_rows=(-1,),
        cell_columns=(-1,),
        cell_area_km2=np.asarray([1.0], dtype=np.float64),
        background_spatial_mass=np.asarray([1.0], dtype=np.float64),
        cell_feature_columns=columns,
        supported_event_ids=("synthetic-event",),
        all_study_area_event_ids=("synthetic-event",),
        event_cell_indices=(0,),
        event_lead_days=np.asarray([1.0], dtype=np.float64),
    )
    return EvaluationScope(fit=fit, exposures=(exposure,))


def _source_universe(synthetic: _Fixture) -> PlaceboSourceUniverse:
    strata, _mappings = _space_inputs(synthetic, reverse=False)
    return PlaceboSourceUniverse(
        fit_issue_ids=synthetic.issue_ids[:4],
        assessment_issue_ids=synthetic.issue_ids[4:],
        issue_tables=synthetic.tables,
        snapshots_by_issue_id=synthetic.snapshots_by_issue_id,
        query_grid=synthetic.grid,
        construction_stratum_by_state_id=strata,
        frozen_input_seal_sha256="a" * 64,
        source_input_sha256="b" * 64,
        assembly_context_sha256="c" * 64,
        scope_assembler=_assembled_formal_scope,
    )


def _source_request(
    kind: Literal["time", "space"],
    *,
    variant: Literal["snapshot", "dynamic"],
    observed: float,
) -> PlaceboRequest:
    return PlaceboRequest(
        kind=kind,
        evaluation_id="formal-validation",
        model_variant=variant,
        observed_statistic=observed,
        frozen_rate_head_sha256="d" * 64,
    )


def test_lazy_time_source_is_paired_across_variants_and_reproducible(
    synthetic: _Fixture,
) -> None:
    universe = _source_universe(synthetic)
    dynamic = build_placebo_source(
        universe,
        _source_request("time", variant="dynamic", observed=0.2),
    )
    snapshot = build_placebo_source(
        universe,
        _source_request("time", variant="snapshot", observed=-0.1),
    )

    first = dynamic.build(
        _source_request("time", variant="dynamic", observed=0.2),
        0,
    )
    repeated = snapshot.build(
        _source_request("time", variant="snapshot", observed=-0.1),
        0,
    )

    assert dynamic.source_id_sha256 == snapshot.source_id_sha256
    assert first.mapping_sha256 == repeated.mapping_sha256
    assert first.replication_index == 0
    assert first.fit_scope.evaluation_id == "formal-validation"


def test_lazy_space_source_rebuilds_the_same_mapping_for_the_same_index(
    synthetic: _Fixture,
) -> None:
    universe = _source_universe(synthetic)
    request = _source_request("space", variant="dynamic", observed=0.2)
    source = build_placebo_source(universe, request)

    first = source.build(request, 1)
    repeated = source.build(request, 1)

    assert first.mapping_sha256 == repeated.mapping_sha256
    assert first.mapping_sha256 != source.build(request, 0).mapping_sha256


def test_lightweight_mapping_identity_does_not_assemble_the_scope(
    synthetic: _Fixture,
) -> None:
    strata, _mappings = _space_inputs(synthetic, reverse=False)
    assembly_calls = 0

    def counted_assembler(issue_tables: Mapping[str, pa.Table]) -> EvaluationScope:
        nonlocal assembly_calls
        assembly_calls += 1
        return _assembled_formal_scope(issue_tables)

    universe = PlaceboSourceUniverse(
        fit_issue_ids=synthetic.issue_ids[:4],
        assessment_issue_ids=synthetic.issue_ids[4:],
        issue_tables=synthetic.tables,
        snapshots_by_issue_id=synthetic.snapshots_by_issue_id,
        query_grid=synthetic.grid,
        construction_stratum_by_state_id=strata,
        frozen_input_seal_sha256="a" * 64,
        source_input_sha256="b" * 64,
        assembly_context_sha256="c" * 64,
        scope_assembler=counted_assembler,
    )
    request = _source_request("time", variant="dynamic", observed=0.2)
    source = build_placebo_source(universe, request)

    lightweight_sha256 = source.mapping_sha256(request, 999)

    assert assembly_calls == 0
    assert source.build(request, 999).mapping_sha256 == lightweight_sha256
    assert assembly_calls == 1


def test_source_universe_rejects_pool_or_stratum_drift(synthetic: _Fixture) -> None:
    strata, _mappings = _space_inputs(synthetic, reverse=False)
    strata.pop(next(iter(strata)))
    with pytest.raises(ValueError, match="frozen construction stratum"):
        PlaceboSourceUniverse(
            fit_issue_ids=synthetic.issue_ids[:4],
            assessment_issue_ids=synthetic.issue_ids[4:],
            issue_tables=synthetic.tables,
            snapshots_by_issue_id=synthetic.snapshots_by_issue_id,
            query_grid=synthetic.grid,
            construction_stratum_by_state_id=strata,
            frozen_input_seal_sha256="a" * 64,
            source_input_sha256="b" * 64,
            assembly_context_sha256="c" * 64,
            scope_assembler=_assembled_formal_scope,
        )
