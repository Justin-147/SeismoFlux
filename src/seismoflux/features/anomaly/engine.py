"""Sequential, target-blind construction of the local stage-3 feature store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from seismoflux.features.anomaly.coverage import (
    CoverageEntityBatch,
    LocalCoverageFeatures,
    coverage_batch_from_spatial_arrays,
    global_reporting_coverage_proxy,
    local_coverage_features,
    select_trailing_coverage_batches,
    trailing_coverage_entity_arrays,
)
from seismoflux.features.anomaly.dictionary import (
    DEFAULT_FEATURE_DICTIONARY,
    FEATURE_STORE_SORT_KEYS,
    TRAJECTORY_BASE_SOURCE_FIELDS,
    FeatureDefinition,
    FeatureDictionary,
    build_feature_store_schema,
)
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
    SPATIAL_SCALES_KM,
    SpatialEntityArrays,
    SpatialFeatureResult,
    compute_spatial_features,
)
from seismoflux.features.anomaly.trajectory import (
    LatestTrajectoryFeatureResult,
    compute_latest_trajectory_features,
)


@dataclass(frozen=True, slots=True)
class Stage3IssueFeatureAudit:
    """Aggregate, non-spatial diagnostics for one generated issue table."""

    issue_index: int
    issue_report_id: str
    row_count: int
    entity_state_count: int
    spatial_entity_count: int
    missing_coordinate_count: int
    trailing_reporting_record_count: int
    nullable_value_count: int
    null_value_count: int


@dataclass(frozen=True, slots=True)
class Stage3IssueFeatureTable:
    """One deterministic issue row group and its aggregate audit."""

    table: pa.Table
    audit: Stage3IssueFeatureAudit


def _constant_string(value: str, size: int) -> pa.Array:
    return pa.array([value] * size, type=pa.string())


def _value_array(
    values: NDArray[np.generic],
    *,
    field: pa.Field,
) -> pa.Array:
    if pa.types.is_floating(field.type):
        numeric = np.asarray(values, dtype=np.float64)
        invalid = ~np.isfinite(numeric)
        if np.any(invalid) and not field.nullable:
            raise ValueError(f"non-nullable stage-3 feature is nonfinite: {field.name}")
        return pa.array(numeric, type=field.type, mask=invalid if field.nullable else None)
    if pa.types.is_integer(field.type):
        return pa.array(values, type=field.type)
    if pa.types.is_boolean(field.type):
        return pa.array(values, type=field.type)
    raise TypeError(f"unsupported feature-store value type: {field.type}")


class Stage3FeatureEngine:
    """Build issue tables strictly in causal order with bounded issue-row memory."""

    def __init__(
        self,
        snapshots: tuple[Stage3IssueSnapshot, ...],
        query_grid: Stage3QueryGrid,
        *,
        dictionary: FeatureDictionary = DEFAULT_FEATURE_DICTIONARY,
        query_chunk_size: int = 256,
        spatial_workers: int = 1,
    ) -> None:
        if not snapshots:
            raise ValueError("stage-3 feature engine requires at least one issue snapshot")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if isinstance(spatial_workers, bool) or spatial_workers not in (1, 2):
            raise ValueError("stage-3 spatial workers must be one or two")
        self.snapshots = snapshots
        self.query_grid = query_grid
        self.dictionary = dictionary
        self.schema = build_feature_store_schema(dictionary)
        self.query_chunk_size = query_chunk_size
        self.spatial_workers = spatial_workers
        self._next_issue_index = 0
        self._coverage_batches: list[CoverageEntityBatch] = []
        series_count = (
            len(TRAJECTORY_BASE_SOURCE_FIELDS) * len(SPATIAL_SCALES_KM) * query_grid.cell_count
        )
        self._trajectory_history = np.empty(
            (len(snapshots), series_count),
            dtype=np.float64,
        )
        self._issue_times = np.empty(len(snapshots), dtype="datetime64[ns]")

    @property
    def next_issue_index(self) -> int:
        return self._next_issue_index

    def _fill_trajectory_history(
        self,
        issue_index: int,
        spatial: SpatialFeatureResult,
    ) -> LatestTrajectoryFeatureResult:
        cell_count = self.query_grid.cell_count
        offset = 0
        for base_field in TRAJECTORY_BASE_SOURCE_FIELDS:
            values = spatial.radius_features[base_field]
            for scale_index in range(len(SPATIAL_SCALES_KM)):
                self._trajectory_history[
                    issue_index,
                    offset : offset + cell_count,
                ] = values[:, scale_index]
                offset += cell_count
        if offset != self._trajectory_history.shape[1]:
            raise AssertionError("trajectory series packing changed")
        self._issue_times[issue_index] = np.datetime64(
            self.snapshots[issue_index].issue_time_utc.replace(tzinfo=None),
            "ns",
        )
        return compute_latest_trajectory_features(
            self._issue_times[: issue_index + 1],
            self._trajectory_history[: issue_index + 1],
        )

    def _base_columns(self, snapshot: Stage3IssueSnapshot) -> dict[str, pa.Array]:
        size = self.query_grid.cell_count
        summary = snapshot.summary
        return {
            "issue_index": pa.array(
                np.full(size, snapshot.issue_index, dtype=np.int16),
                type=pa.int16(),
            ),
            "issue_time_utc": pa.array(
                [snapshot.issue_time_utc] * size,
                type=pa.timestamp("us", tz="UTC"),
            ),
            "issue_report_id": _constant_string(summary.issue_report_id, size),
            "issue_report_date": pa.array([summary.issue_report_date] * size, type=pa.date32()),
            "issue_report_year": pa.array(
                np.full(size, summary.issue_report_year, dtype=np.int16),
                type=pa.int16(),
            ),
            "issue_report_period": pa.array(
                np.full(size, summary.issue_report_period, dtype=np.int16),
                type=pa.int16(),
            ),
            "state_snapshot_id": _constant_string(snapshot.state_snapshot_id, size),
            "lineage_digest": _constant_string(snapshot.lineage_digest, size),
            "feature_dictionary_sha256": _constant_string(self.dictionary.sha256, size),
            "grid_id": _constant_string(self.query_grid.grid_id, size),
            "equal_area_crs": _constant_string(self.query_grid.equal_area_crs, size),
            "cell_size_km": pa.array(
                np.full(size, self.query_grid.cell_size_km, dtype=np.float64),
                type=pa.float64(),
            ),
            "cell_id": pa.array(self.query_grid.cell_ids, type=pa.string()),
            "cell_row": pa.array(self.query_grid.rows, type=pa.int32()),
            "cell_column": pa.array(self.query_grid.columns, type=pa.int32()),
            "query_x_m": pa.array(self.query_grid.query_xy_m[:, 0], type=pa.float64()),
            "query_y_m": pa.array(self.query_grid.query_xy_m[:, 1], type=pa.float64()),
            "clipped_area_km2": pa.array(
                self.query_grid.clipped_area_km2,
                type=pa.float64(),
            ),
        }

    def _emit_spatial_definition(
        self,
        columns: dict[str, pa.Array],
        definition: FeatureDefinition,
        spatial: SpatialFeatureResult,
    ) -> tuple[int, int]:
        if definition.source_output_field is None:
            raise AssertionError("spatial definition has no source field")
        nullable = 0
        nulls = 0
        value_columns = iter(definition.storage_value_columns())
        feature_families = {
            "closed_ball": spatial.radius_features,
            "gaussian": spatial.gaussian_features,
        }
        for kernel in definition.kernels:
            features = feature_families[kernel]
            reasons = (
                spatial_null_reason_codes(definition.source_output_field, features)
                if definition.nullable
                else None
            )
            for scale_index in range(len(definition.scales_km)):
                column_name = next(value_columns)
                field = self.schema.field(column_name)
                values = features[definition.source_output_field][:, scale_index]
                columns[column_name] = _value_array(values, field=field)
                if definition.nullable:
                    if reasons is None:
                        raise AssertionError("nullable spatial feature has no reasons")
                    reason_name = definition.null_reason_code_field(column_name)
                    if reason_name is None:
                        raise AssertionError("nullable spatial feature has no reason column")
                    columns[reason_name] = pa.array(reasons[:, scale_index], type=pa.int8())
                    nullable += values.size
                    nulls += int(np.count_nonzero(~np.isfinite(values)))
        try:
            next(value_columns)
        except StopIteration:
            return nullable, nulls
        raise AssertionError("spatial definition storage-column count changed")

    def _emit_local_coverage_definition(
        self,
        columns: dict[str, pa.Array],
        definition: FeatureDefinition,
        local: LocalCoverageFeatures,
    ) -> tuple[int, int]:
        nullable = 0
        nulls = 0
        value_columns = iter(definition.storage_value_columns())
        families = {
            "closed_ball": (local.radius, local.radius_null_reason),
            "gaussian": (local.gaussian, local.gaussian_null_reason),
        }
        for kernel in definition.kernels:
            features, reasons = families[kernel]
            for scale_index in range(len(definition.scales_km)):
                column_name = next(value_columns)
                field = self.schema.field(column_name)
                values = features[definition.name][:, scale_index]
                columns[column_name] = _value_array(values, field=field)
                if definition.nullable:
                    reason_name = definition.null_reason_code_field(column_name)
                    if reason_name is None:
                        raise AssertionError("nullable local proxy has no reason column")
                    columns[reason_name] = pa.array(
                        reasons[definition.name][:, scale_index],
                        type=pa.int8(),
                    )
                    nullable += values.size
                    nulls += int(np.count_nonzero(~np.isfinite(values)))
        try:
            next(value_columns)
        except StopIteration:
            return nullable, nulls
        raise AssertionError("local coverage storage-column count changed")

    def _emit_trajectory_definition(
        self,
        columns: dict[str, pa.Array],
        definition: FeatureDefinition,
        result: LatestTrajectoryFeatureResult,
    ) -> tuple[int, int]:
        if definition.source_output_field is None:
            raise AssertionError("trajectory definition has no source field")
        cell_count = self.query_grid.cell_count
        feature_name = definition.source_output_field
        values = result.features[feature_name]
        valid = result.valid_masks[feature_name]
        sample_counts = result.sample_counts[feature_name]
        reasons = trajectory_null_reason_codes(result, feature_name)
        nullable = 0
        nulls = 0
        value_columns = iter(definition.storage_value_columns())
        for base_index in range(len(TRAJECTORY_BASE_SOURCE_FIELDS)):
            for scale_index in range(len(SPATIAL_SCALES_KM)):
                series_index = base_index * len(SPATIAL_SCALES_KM) + scale_index
                start = series_index * cell_count
                stop = start + cell_count
                column_name = next(value_columns)
                field = self.schema.field(column_name)
                column_values = values[start:stop]
                column_valid = valid[start:stop]
                columns[column_name] = pa.array(
                    column_values,
                    type=field.type,
                    mask=~column_valid,
                )
                validity_name = definition.validity_field(column_name)
                sample_name = definition.sample_count_field(column_name)
                reason_name = definition.null_reason_code_field(column_name)
                if validity_name is None or sample_name is None or reason_name is None:
                    raise AssertionError("trajectory quality companions changed")
                columns[validity_name] = pa.array(column_valid, type=pa.bool_())
                columns[sample_name] = pa.array(sample_counts[start:stop], type=pa.int64())
                columns[reason_name] = pa.array(reasons[start:stop], type=pa.int8())
                nullable += cell_count
                nulls += int(np.count_nonzero(~column_valid))
        try:
            next(value_columns)
        except StopIteration:
            return nullable, nulls
        raise AssertionError("trajectory definition storage-column count changed")

    def _emit_protocol_definition(
        self,
        columns: dict[str, pa.Array],
        definition: FeatureDefinition,
        proxy: dict[str, object],
    ) -> tuple[int, int]:
        (column_name,) = definition.storage_value_columns()
        value = proxy[column_name]
        size = self.query_grid.cell_count
        field = self.schema.field(column_name)
        if value is None:
            columns[column_name] = pa.nulls(size, type=field.type)
            null_count = size
        else:
            columns[column_name] = pa.array([value] * size, type=field.type)
            null_count = 0
        if definition.nullable:
            reason_name = definition.null_reason_code_field(column_name)
            if reason_name is None:
                raise AssertionError("nullable protocol proxy has no reason column")
            reason = proxy[reason_name]
            if isinstance(reason, bool) or not isinstance(reason, int):
                raise TypeError("protocol null-reason code must be an integer")
            columns[reason_name] = pa.array(
                np.full(size, reason, dtype=np.int8),
                type=pa.int8(),
            )
            return size, null_count
        return 0, 0

    def build_next_issue(self) -> Stage3IssueFeatureTable:
        """Build the next actual issue; out-of-order or repeated generation is forbidden."""

        issue_index = self._next_issue_index
        if issue_index >= len(self.snapshots):
            raise StopIteration("all stage-3 issue tables have already been built")
        snapshot = self.snapshots[issue_index]
        if snapshot.issue_index != issue_index:
            raise ValueError("stage-3 issue snapshots must use contiguous causal indices")

        entity_arrays = spatial_entity_arrays(snapshot)
        current_batch = coverage_batch_from_spatial_arrays(
            snapshot.issue_time_utc,
            entity_arrays,
        )
        self._coverage_batches.append(current_batch)
        trailing_batches = select_trailing_coverage_batches(
            tuple(self._coverage_batches),
            snapshot.issue_time_utc,
        )
        trailing_arrays = trailing_coverage_entity_arrays(trailing_batches)
        spatial_arguments = (
            (entity_arrays, "current"),
            (trailing_arrays, "trailing"),
        )

        def compute_one(arguments: tuple[SpatialEntityArrays, str]) -> SpatialFeatureResult:
            arrays, _role = arguments
            return compute_spatial_features(
                self.query_grid.query_xy_m,
                arrays,
                query_chunk_size=self.query_chunk_size,
            )

        if self.spatial_workers == 1:
            spatial, trailing_spatial = tuple(map(compute_one, spatial_arguments))
        else:
            # The two independent spatial aggregates are the only outer parallel work.
            # Formal execution derives this bounded worker count after reserving at
            # least two physical cores; stable ``map`` order preserves determinism.
            with ThreadPoolExecutor(max_workers=self.spatial_workers) as executor:
                spatial, trailing_spatial = tuple(executor.map(compute_one, spatial_arguments))
        local_coverage = local_coverage_features(spatial, trailing_spatial)
        trajectory = self._fill_trajectory_history(issue_index, spatial)
        protocol_proxy = global_reporting_coverage_proxy(self.snapshots, issue_index)

        columns = self._base_columns(snapshot)
        nullable_value_count = 0
        null_value_count = 0
        for definition in self.dictionary.definitions:
            if definition.producer == "spatial_v1":
                nullable, nulls = self._emit_spatial_definition(columns, definition, spatial)
            elif definition.producer == "local_coverage_v1":
                nullable, nulls = self._emit_local_coverage_definition(
                    columns,
                    definition,
                    local_coverage,
                )
            elif definition.producer == "trajectory_v1":
                nullable, nulls = self._emit_trajectory_definition(
                    columns,
                    definition,
                    trajectory,
                )
            elif definition.producer == "protocol_v1":
                nullable, nulls = self._emit_protocol_definition(
                    columns,
                    definition,
                    protocol_proxy,
                )
            else:
                raise AssertionError(f"unsupported feature producer: {definition.producer}")
            nullable_value_count += nullable
            null_value_count += nulls

        missing = [name for name in self.schema.names if name not in columns]
        extras = sorted(set(columns) - set(self.schema.names))
        if missing or extras:
            raise ValueError(f"feature-store schema mismatch: missing={missing}, extras={extras}")
        table = pa.Table.from_arrays(
            [columns[name] for name in self.schema.names],
            schema=self.schema,
        )
        if table.num_rows != self.query_grid.cell_count:
            raise AssertionError("one stage-3 issue must emit exactly one row per fixed query cell")
        self._next_issue_index += 1
        return Stage3IssueFeatureTable(
            table=table,
            audit=Stage3IssueFeatureAudit(
                issue_index=issue_index,
                issue_report_id=snapshot.summary.issue_report_id,
                row_count=table.num_rows,
                entity_state_count=len(snapshot.entities),
                spatial_entity_count=spatial.spatial_entity_count,
                missing_coordinate_count=spatial.missing_coordinate_count,
                trailing_reporting_record_count=trailing_spatial.input_entity_count,
                nullable_value_count=nullable_value_count,
                null_value_count=null_value_count,
            ),
        )


__all__ = [
    "FEATURE_STORE_SORT_KEYS",
    "Stage3FeatureEngine",
    "Stage3IssueFeatureAudit",
    "Stage3IssueFeatureTable",
]
