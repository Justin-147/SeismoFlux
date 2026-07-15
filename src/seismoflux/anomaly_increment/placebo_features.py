"""In-memory feature reconstruction for the frozen stage-4 placebos.

The time placebo moves only the preregistered 200 km snapshot fields (including
their missingness) and the two radius-series inputs.  It then rebuilds trajectory
columns from the destination pseudo-history.  The space placebo changes only
eligible entity coordinate pairs inside a verified issue/stratum bijection and
recomputes the same snapshot and trajectory fields with stage-3 low-level APIs.

Neither entry point accepts a filesystem path, catalogue, event label, or score.
Accepted reporting-coverage columns are always retained from the recipient table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import TypeAlias

import numpy as np
import pyarrow as pa

from seismoflux.anomaly_increment.placebo import SpaceBijection, TimeBijection
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.nulls import (
    spatial_null_reason_codes,
    trajectory_null_reason_codes,
)
from seismoflux.features.anomaly.snapshot import (
    Stage3IssueSnapshot,
    spatial_entity_arrays,
)
from seismoflux.features.anomaly.spatial import (
    compute_stage4_placebo_spatial_features,
)
from seismoflux.features.anomaly.state import AnomalyState
from seismoflux.features.anomaly.trajectory import (
    LatestTrajectoryFeatureResult,
    compute_trajectory_features,
)

SpaceStratumKey: TypeAlias = tuple[str, str]

PLACEBO_SCALE_KM = 200.0
SPATIAL_RECOMPUTATION_COST = (
    "stage3_stage4_projection_recomputes_only_the_ten_preregistered_200km_fields;"
    "accepted_radius_gaussian_entropy_age_and_geometry_formulas_are_unchanged"
)

GRID_IDENTITY_COLUMNS = (
    "grid_id",
    "equal_area_crs",
    "cell_size_km",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "clipped_area_km2",
)

COVERAGE_VALUE_COLUMNS = (
    "gaussian_200km__current_to_trailing_station_reporting_coverage_proxy",
    "gaussian_200km__current_to_trailing_measurement_reporting_coverage_proxy",
    "gaussian_200km__distinct_reporting_station_count_reporting_coverage_proxy",
    "gaussian_200km__distinct_reporting_measurement_count_reporting_coverage_proxy",
    "days_since_previous_actual_report_reporting_coverage_proxy",
    "missing_expected_period_count_13w_reporting_coverage_proxy",
    "missing_expected_period_count_52w_reporting_coverage_proxy",
    "report_present_reporting_coverage_proxy",
    "report_row_count_reporting_coverage_proxy",
)
COVERAGE_QUALITY_COLUMNS = (
    "gaussian_200km__current_to_trailing_station_reporting_coverage_proxy__null_reason_code",
    "gaussian_200km__current_to_trailing_measurement_reporting_coverage_proxy__null_reason_code",
    "days_since_previous_actual_report_reporting_coverage_proxy__null_reason_code",
)
COVERAGE_COLUMNS = (*COVERAGE_VALUE_COLUMNS, *COVERAGE_QUALITY_COLUMNS)

_SNAPSHOT_SOURCE_BY_COLUMN = {
    "gaussian_200km__reliability_weighted_listed_count": ("reliability_weighted_listed_count"),
    "gaussian_200km__first_seen_weighted_count": "first_seen_weighted_count",
    "gaussian_200km__not_continued_weighted_count": "not_continued_weighted_count",
    "gaussian_200km__age_mean_days": "age_mean_days",
    "gaussian_200km__discipline_shannon_normalized": "discipline_shannon_normalized",
    "gaussian_200km__multidisciplinary_entity_weighted_fraction": (
        "multidisciplinary_entity_weighted_fraction"
    ),
    "gaussian_200km__concentration": "concentration",
    "gaussian_200km__diffusion_radius_km": "diffusion_radius_km",
}
SNAPSHOT_VALUE_COLUMNS = tuple(_SNAPSHOT_SOURCE_BY_COLUMN)
SNAPSHOT_QUALITY_COLUMNS = tuple(
    f"{column}__null_reason_code"
    for column in (
        "gaussian_200km__age_mean_days",
        "gaussian_200km__discipline_shannon_normalized",
        "gaussian_200km__multidisciplinary_entity_weighted_fraction",
        "gaussian_200km__concentration",
        "gaussian_200km__diffusion_radius_km",
    )
)
SNAPSHOT_COLUMNS = (*SNAPSHOT_VALUE_COLUMNS, *SNAPSHOT_QUALITY_COLUMNS)

TRAJECTORY_BASE_FIELDS = ("listed_count", "first_seen_count")
TRAJECTORY_BASE_COLUMNS = tuple(
    f"radius_200km__{base_field}" for base_field in TRAJECTORY_BASE_FIELDS
)
TRAJECTORY_FEATURES = (
    "slope_4w_per_week",
    "slope_13w_per_week",
    "acceleration_4v13_per_week2",
    "surge_z_13w",
    "peak_drop_52w",
)
TRAJECTORY_VALUE_COLUMNS = tuple(
    f"radius_200km__{base_field}__{feature}"
    for base_field in TRAJECTORY_BASE_FIELDS
    for feature in TRAJECTORY_FEATURES
)
TRAJECTORY_QUALITY_COLUMNS = tuple(
    f"{column}{suffix}"
    for column in TRAJECTORY_VALUE_COLUMNS
    for suffix in ("__valid", "__sample_count", "__null_reason_code")
)
TRAJECTORY_COLUMNS = (*TRAJECTORY_VALUE_COLUMNS, *TRAJECTORY_QUALITY_COLUMNS)

_REQUIRED_COLUMNS = (
    "issue_index",
    "issue_time_utc",
    "issue_report_id",
    *GRID_IDENTITY_COLUMNS,
    *COVERAGE_COLUMNS,
    *SNAPSHOT_COLUMNS,
    *TRAJECTORY_BASE_COLUMNS,
    *TRAJECTORY_COLUMNS,
)


def _one_utc_issue_time(table: pa.Table, *, issue_id: str) -> datetime:
    values = table["issue_time_utc"].combine_chunks().unique().to_pylist()
    if len(values) != 1 or not isinstance(values[0], datetime) or values[0].tzinfo is None:
        raise ValueError(f"issue {issue_id!r} must contain one timezone-aware issue time")
    return values[0].astimezone(UTC)


def _one_text(table: pa.Table, column: str, *, issue_id: str) -> str:
    values = table[column].combine_chunks().unique().to_pylist()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"issue {issue_id!r} must contain one non-empty {column}")
    return values[0]


def _validate_issue_tables(issue_tables: Mapping[str, pa.Table]) -> dict[str, datetime]:
    if not issue_tables:
        raise ValueError("placebo feature reconstruction requires issue tables")
    keys = tuple(issue_tables)
    if any(not isinstance(key, str) or not key or key != key.strip() for key in keys):
        raise ValueError("issue table keys must be non-empty trimmed strings")
    if len(keys) != len(set(keys)):
        raise ValueError("issue table keys must be unique")

    first = issue_tables[keys[0]]
    if not isinstance(first, pa.Table) or first.num_rows <= 0:
        raise ValueError("every issue feature table must be a non-empty Arrow table")
    if len(first.column_names) != len(set(first.column_names)):
        raise ValueError("issue feature schemas cannot contain duplicate columns")
    missing = sorted(set(_REQUIRED_COLUMNS) - set(first.column_names))
    if missing:
        raise ValueError(f"issue feature table omitted frozen columns: {missing}")

    times: dict[str, datetime] = {}
    for issue_id, table in issue_tables.items():
        if not isinstance(table, pa.Table) or table.num_rows != first.num_rows:
            raise ValueError("all issue feature tables must share one positive row count")
        if not table.schema.equals(first.schema, check_metadata=True):
            raise ValueError("all issue feature tables must share the exact Arrow schema")
        for column in GRID_IDENTITY_COLUMNS:
            if not table[column].equals(first[column]):
                raise ValueError(f"issue {issue_id!r} changed frozen grid column {column}")
        times[issue_id] = _one_utc_issue_time(table, issue_id=issue_id)
    if len(set(times.values())) != len(times):
        raise ValueError("issue feature tables must have unique issue times")
    return times


def _replace_column(
    table: pa.Table,
    column: str,
    values: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    index = table.schema.get_field_index(column)
    if index < 0:
        raise ValueError(f"cannot replace absent frozen column: {column}")
    field = table.schema.field(index)
    if values.type != field.type or len(values) != table.num_rows:
        raise ValueError(f"replacement column {column} changed Arrow type or row count")
    return table.set_column(index, field, values)


def _floating_column(table: pa.Table, column: str) -> np.ndarray:
    array = table[column].combine_chunks()
    if not pa.types.is_floating(array.type):
        raise TypeError(f"trajectory base column must be floating point: {column}")
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64).copy()
    missing = np.asarray(array.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
    values[missing] = np.nan
    return values


def _trajectory_column(base_field: str, feature: str) -> str:
    return f"radius_200km__{base_field}__{feature}"


def _rebuild_destination_trajectory(
    issue_ids: Sequence[str],
    tables: Mapping[str, pa.Table],
) -> dict[str, pa.Table]:
    ordered_ids = tuple(issue_ids)
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("destination pseudo-history issue IDs must be non-empty and unique")
    row_count = tables[ordered_ids[0]].num_rows
    issue_times = np.asarray(
        [
            np.datetime64(
                _one_utc_issue_time(tables[issue_id], issue_id=issue_id).replace(tzinfo=None),
                "ns",
            )
            for issue_id in ordered_ids
        ],
        dtype="datetime64[ns]",
    )
    histories = [
        np.stack(
            [
                _floating_column(tables[issue_id], f"radius_200km__{base_field}")
                for issue_id in ordered_ids
            ],
            axis=0,
        )
        for base_field in TRAJECTORY_BASE_FIELDS
    ]
    packed = np.concatenate(histories, axis=1)
    trajectory = compute_trajectory_features(issue_times, packed)

    rebuilt: dict[str, pa.Table] = {}
    for issue_index, issue_id in enumerate(ordered_ids):
        table = tables[issue_id]
        latest = LatestTrajectoryFeatureResult(
            issue_time_utc=trajectory.issue_times_utc[issue_index],
            features={name: values[issue_index] for name, values in trajectory.features.items()},
            valid_masks={
                name: values[issue_index] for name, values in trajectory.valid_masks.items()
            },
            sample_counts={
                name: values[issue_index] for name, values in trajectory.sample_counts.items()
            },
        )
        for feature in TRAJECTORY_FEATURES:
            reasons = trajectory_null_reason_codes(latest, feature)
            for base_index, base_field in enumerate(TRAJECTORY_BASE_FIELDS):
                start = base_index * row_count
                stop = start + row_count
                column = _trajectory_column(base_field, feature)
                field = table.schema.field(column)
                valid = latest.valid_masks[feature][start:stop]
                value_array = pa.array(
                    latest.features[feature][start:stop],
                    type=field.type,
                    mask=~valid,
                )
                table = _replace_column(table, column, value_array)
                table = _replace_column(
                    table,
                    f"{column}__valid",
                    pa.array(valid, type=pa.bool_()),
                )
                table = _replace_column(
                    table,
                    f"{column}__sample_count",
                    pa.array(
                        latest.sample_counts[feature][start:stop],
                        type=pa.int64(),
                    ),
                )
                table = _replace_column(
                    table,
                    f"{column}__null_reason_code",
                    pa.array(reasons[start:stop], type=pa.int8()),
                )
        rebuilt[issue_id] = table
    return rebuilt


def rebuild_time_placebo_features(
    issue_tables: Mapping[str, pa.Table],
    *,
    fit_bijection: TimeBijection,
    assessment_bijection: TimeBijection,
) -> dict[str, pa.Table]:
    """Build one time-placebo feature history without moving stored trajectories.

    ``issue_tables`` is keyed by the same protocol issue IDs used by the two injected
    bijections.  The pools must be disjoint and their union must equal the input.  Fit
    recipient times must precede assessment recipient times.  Precomputed trajectory
    values are never read as donor values; all selected trajectory values, masks,
    sample counts, and reason codes are regenerated from the mapped radius series.
    """

    times = _validate_issue_tables(issue_tables)
    fit_ids = set(fit_bijection.recipient_issue_ids)
    assessment_ids = set(assessment_bijection.recipient_issue_ids)
    if fit_ids & assessment_ids:
        raise ValueError("time-placebo fit and assessment pools must be disjoint")
    if fit_ids | assessment_ids != set(issue_tables):
        raise ValueError("time-placebo pools must exactly cover the supplied issue tables")
    if max(times[issue_id] for issue_id in fit_ids) >= min(
        times[issue_id] for issue_id in assessment_ids
    ):
        raise ValueError("every fit-pool recipient must precede every assessment recipient")

    mapping = dict(fit_bijection.pairs)
    mapping.update(dict(assessment_bijection.pairs))
    if set(mapping) != set(issue_tables) or set(mapping.values()) != set(issue_tables):
        raise ValueError("injected time mappings must remain bijective within their pools")
    ordered_fit = tuple(sorted(fit_ids, key=times.__getitem__))
    ordered_assessment = tuple(sorted(assessment_ids, key=times.__getitem__))
    ordered_ids = (*ordered_fit, *ordered_assessment)

    pseudo_tables: dict[str, pa.Table] = {}
    donor_columns = (*SNAPSHOT_COLUMNS, *TRAJECTORY_BASE_COLUMNS)
    for recipient_id in ordered_ids:
        recipient = issue_tables[recipient_id]
        donor = issue_tables[mapping[recipient_id]]
        table = recipient
        for column in donor_columns:
            table = _replace_column(table, column, donor[column])
        pseudo_tables[recipient_id] = table
    return _rebuild_destination_trajectory(ordered_ids, pseudo_tables)


def _validate_stage3_grid(table: pa.Table, grid: Stage3QueryGrid, *, issue_id: str) -> None:
    if table.num_rows != grid.cell_count:
        raise ValueError(f"issue {issue_id!r} row count differs from the stage-3 grid")
    if _one_text(table, "grid_id", issue_id=issue_id) != grid.grid_id:
        raise ValueError(f"issue {issue_id!r} grid_id differs from the stage-3 grid")
    if _one_text(table, "equal_area_crs", issue_id=issue_id) != grid.equal_area_crs:
        raise ValueError(f"issue {issue_id!r} CRS differs from the stage-3 grid")
    if tuple(table["cell_id"].combine_chunks().to_pylist()) != grid.cell_ids:
        raise ValueError(f"issue {issue_id!r} cell ordering differs from the stage-3 grid")
    expected_arrays = (
        ("cell_row", grid.rows, np.int64),
        ("cell_column", grid.columns, np.int64),
        ("query_x_m", grid.query_xy_m[:, 0], np.float64),
        ("query_y_m", grid.query_xy_m[:, 1], np.float64),
        ("clipped_area_km2", grid.clipped_area_km2, np.float64),
    )
    for column, expected, dtype in expected_arrays:
        observed = np.asarray(
            table[column].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=dtype,
        )
        if not np.array_equal(observed, np.asarray(expected, dtype=dtype)):
            raise ValueError(f"issue {issue_id!r} changed frozen grid column {column}")


def _permuted_snapshots(
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot],
    *,
    verified_construction_stratum_by_state_id: Mapping[str, str],
    bijections_by_issue_stratum: Mapping[SpaceStratumKey, SpaceBijection],
) -> tuple[tuple[str, Stage3IssueSnapshot], ...]:
    eligible_by_group: dict[SpaceStratumKey, list[AnomalyState]] = {}
    eligible_ids: set[str] = set()
    all_state_ids: set[str] = set()
    original_coordinate_by_state_id: dict[str, tuple[float, float]] = {}
    for issue_id, snapshot in snapshots_by_issue_id.items():
        for state in snapshot.entities:
            if state.state_id in all_state_ids:
                raise ValueError("entity state IDs must be unique across issue snapshots")
            all_state_ids.add(state.state_id)
            if not state.spatial_eligible:
                continue
            if (
                state.longitude is None
                or state.latitude is None
                or not math.isfinite(state.longitude)
                or not math.isfinite(state.latitude)
            ):
                raise ValueError("spatially eligible entity must have a finite coordinate pair")
            eligible_ids.add(state.state_id)
            original_coordinate_by_state_id[state.state_id] = (
                float(state.longitude),
                float(state.latitude),
            )
            stratum = verified_construction_stratum_by_state_id.get(state.state_id)
            if not isinstance(stratum, str) or not stratum or stratum != stratum.strip():
                raise ValueError("every eligible state needs one verified construction stratum")
            eligible_by_group.setdefault((issue_id, stratum), []).append(state)

    if set(verified_construction_stratum_by_state_id) != eligible_ids:
        raise ValueError("construction-stratum mapping must cover exactly eligible entity states")
    if set(bijections_by_issue_stratum) != set(eligible_by_group):
        raise ValueError("space bijections must cover every issue/stratum group exactly once")

    replacement_by_state_id: dict[str, tuple[float, float]] = {}
    for key, raw_states in eligible_by_group.items():
        states = sorted(raw_states, key=lambda item: item.state_id)
        state_ids = tuple(state.state_id for state in states)
        coordinates = tuple(original_coordinate_by_state_id[state_id] for state_id in state_ids)
        mapping = bijections_by_issue_stratum[key]
        if mapping.entity_state_ids != state_ids:
            raise ValueError("space bijection entity order must be stable state_id ascending")
        expected_permuted = tuple(
            original_coordinate_by_state_id[donor_id] for donor_id in mapping.donor_state_ids
        )
        if mapping.permuted_coordinate_pairs != expected_permuted:
            raise ValueError("space bijection coordinate pairs disagree with donor states")
        if sorted(coordinates) != sorted(mapping.permuted_coordinate_pairs):
            raise ValueError("space bijection changed the issue/stratum coordinate multiset")
        moved = sum(
            original != permuted
            for original, permuted in zip(
                coordinates,
                mapping.permuted_coordinate_pairs,
                strict=True,
            )
        )
        if moved != mapping.moved_entity_row_count or mapping.no_effect != (moved == 0):
            raise ValueError("space bijection movement audit disagrees with coordinate pairs")
        replacement_by_state_id.update(
            zip(state_ids, mapping.permuted_coordinate_pairs, strict=True)
        )

    output: list[tuple[str, Stage3IssueSnapshot]] = []
    for issue_id, snapshot in snapshots_by_issue_id.items():
        entities = tuple(
            replace(
                state,
                longitude=replacement_by_state_id[state.state_id][0],
                latitude=replacement_by_state_id[state.state_id][1],
            )
            if state.spatial_eligible
            else state
            for state in snapshot.entities
        )
        output.append((issue_id, replace(snapshot, entities=entities)))
    output.sort(key=lambda item: item[1].issue_index)
    if tuple(item[1].issue_index for item in output) != tuple(range(len(output))):
        raise ValueError("space-placebo snapshots must have contiguous causal issue indices")
    return tuple(output)


def rebuild_space_placebo_features(
    issue_tables: Mapping[str, pa.Table],
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot],
    query_grid: Stage3QueryGrid,
    *,
    verified_construction_stratum_by_state_id: Mapping[str, str],
    bijections_by_issue_stratum: Mapping[SpaceStratumKey, SpaceBijection],
    query_chunk_size: int = 256,
) -> dict[str, pa.Table]:
    """Recompute the frozen space-placebo science fields and rejoin coverage.

    Only spatially eligible entity longitude/latitude pairs are replaced.  All
    non-coordinate entity fields and all ineligible entities remain unchanged.
    The stage-3 low-level projection computes only the ten preregistered 200 km
    Gaussian-snapshot and closed-ball trajectory-base fields.  It omits unused
    identifier and geometry outputs without changing any retained formula.
    """

    times = _validate_issue_tables(issue_tables)
    if set(snapshots_by_issue_id) != set(issue_tables):
        raise ValueError("space-placebo snapshots must exactly match issue table keys")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    for issue_id, snapshot in snapshots_by_issue_id.items():
        table = issue_tables[issue_id]
        _validate_stage3_grid(table, query_grid, issue_id=issue_id)
        if times[issue_id] != snapshot.issue_time_utc.astimezone(UTC):
            raise ValueError("snapshot issue time differs from its accepted feature table")
        if _one_text(table, "issue_report_id", issue_id=issue_id) != (
            snapshot.summary.issue_report_id
        ):
            raise ValueError("snapshot report identity differs from its accepted feature table")
        issue_indices = table["issue_index"].combine_chunks().unique().to_pylist()
        if issue_indices != [snapshot.issue_index]:
            raise ValueError("snapshot issue index differs from its accepted feature table")

    ordered = _permuted_snapshots(
        snapshots_by_issue_id,
        verified_construction_stratum_by_state_id=(verified_construction_stratum_by_state_id),
        bijections_by_issue_stratum=bijections_by_issue_stratum,
    )
    scale_index = 0
    rebuilt_snapshot_tables: dict[str, pa.Table] = {}
    for issue_id, snapshot in ordered:
        table = issue_tables[issue_id]
        spatial = compute_stage4_placebo_spatial_features(
            query_grid.query_xy_m,
            spatial_entity_arrays(snapshot),
            query_chunk_size=query_chunk_size,
        )
        for column, source_field in _SNAPSHOT_SOURCE_BY_COLUMN.items():
            values = spatial.gaussian_features[source_field][:, scale_index]
            field = table.schema.field(column)
            invalid = ~np.isfinite(values)
            if np.any(invalid) and not field.nullable:
                raise ValueError(f"non-nullable recomputed spatial feature is invalid: {column}")
            table = _replace_column(
                table,
                column,
                pa.array(
                    values,
                    type=field.type,
                    mask=invalid if field.nullable else None,
                ),
            )
            reason_column = f"{column}__null_reason_code"
            if reason_column in SNAPSHOT_QUALITY_COLUMNS:
                reasons = spatial_null_reason_codes(
                    source_field,
                    spatial.gaussian_features,
                )[:, scale_index]
                table = _replace_column(
                    table,
                    reason_column,
                    pa.array(reasons, type=pa.int8()),
                )
        for base_field, column in zip(
            TRAJECTORY_BASE_FIELDS,
            TRAJECTORY_BASE_COLUMNS,
            strict=True,
        ):
            values = spatial.radius_features[base_field][:, scale_index]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"recomputed trajectory base is nonfinite: {column}")
            table = _replace_column(
                table,
                column,
                pa.array(values, type=table.schema.field(column).type),
            )
        rebuilt_snapshot_tables[issue_id] = table

    ordered_ids = tuple(issue_id for issue_id, _snapshot in ordered)
    return _rebuild_destination_trajectory(ordered_ids, rebuilt_snapshot_tables)


__all__ = [
    "COVERAGE_COLUMNS",
    "COVERAGE_QUALITY_COLUMNS",
    "COVERAGE_VALUE_COLUMNS",
    "PLACEBO_SCALE_KM",
    "SNAPSHOT_COLUMNS",
    "SNAPSHOT_QUALITY_COLUMNS",
    "SNAPSHOT_VALUE_COLUMNS",
    "SPATIAL_RECOMPUTATION_COST",
    "TRAJECTORY_BASE_COLUMNS",
    "TRAJECTORY_BASE_FIELDS",
    "TRAJECTORY_COLUMNS",
    "TRAJECTORY_FEATURES",
    "TRAJECTORY_QUALITY_COLUMNS",
    "TRAJECTORY_VALUE_COLUMNS",
    "SpaceStratumKey",
    "rebuild_space_placebo_features",
    "rebuild_time_placebo_features",
]
