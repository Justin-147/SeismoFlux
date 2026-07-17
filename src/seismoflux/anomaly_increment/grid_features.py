"""Target-independent multigrid anomaly feature reconstruction for stage 4.

Only the frozen study-area geometry and accepted anomaly snapshots may influence
grid construction or feature values.  Earthquake locations are intentionally not
accepted by any function in this module.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from seismoflux.anomaly_increment.config import Stage4ProtocolBundle
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    GRID_CELL_SIZES_KM,
    build_clipped_grid,
    project_study_area_to_equal_area,
)
from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.engine import (
    Stage3FeatureEngine,
    Stage3IssueFeatureTable,
)
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot

FeatureVariant = Literal["coverage_only", "snapshot", "dynamic"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1: Final[str] = (
    "arrow_ipc_selected_table_logical_identity_r1"
)
SELECTED_TABLE_LOGICAL_IDENTITY_DOMAIN_R1: Final[bytes] = (
    b"seismoflux.selected-table-logical-identity.r1\x00"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise TypeError(f"{label} must contain non-empty strings")
    return cast(tuple[str, ...], result)


def _readonly(array: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class Stage4IntegrationGrid:
    """One frozen-origin integration grid derived from the study area only."""

    grid_id: str
    equal_area_crs: str
    cell_size_km: float
    cell_ids: tuple[str, ...]
    rows: IntArray
    columns: IntArray
    query_xy_m: FloatArray
    clipped_area_km2: FloatArray

    def __post_init__(self) -> None:
        if self.equal_area_crs != EQUAL_AREA_CRS:
            raise ValueError("stage-4 integration CRS changed")
        if float(self.cell_size_km) not in GRID_CELL_SIZES_KM:
            raise ValueError("stage-4 grid must be exactly 50, 25, or 12.5 km")
        count = len(self.cell_ids)
        if not self.grid_id or len(set(self.cell_ids)) != count:
            raise ValueError("stage-4 grid identities must be non-empty and unique")
        if (
            self.rows.shape != (count,)
            or self.columns.shape != (count,)
            or self.query_xy_m.shape != (count, 2)
            or self.clipped_area_km2.shape != (count,)
        ):
            raise ValueError("stage-4 grid array shapes disagree")
        if count == 0 or not np.all(np.isfinite(self.query_xy_m)):
            raise ValueError("stage-4 grid must have finite query centers")
        if not np.all(np.isfinite(self.clipped_area_km2)) or np.any(self.clipped_area_km2 <= 0.0):
            raise ValueError("stage-4 clipped grid areas must be finite and positive")
        object.__setattr__(self, "rows", cast(IntArray, _readonly(self.rows)))
        object.__setattr__(self, "columns", cast(IntArray, _readonly(self.columns)))
        object.__setattr__(self, "query_xy_m", cast(FloatArray, _readonly(self.query_xy_m)))
        object.__setattr__(
            self,
            "clipped_area_km2",
            cast(FloatArray, _readonly(self.clipped_area_km2)),
        )

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)

    @property
    def total_area_km2(self) -> float:
        return float(np.sum(self.clipped_area_km2, dtype=np.float64))

    def as_stage3_query_grid(self) -> Stage3QueryGrid:
        """Adapt the accepted engine to a newly recomputed grid, never interpolate."""

        return Stage3QueryGrid(
            grid_id=self.grid_id,
            equal_area_crs=self.equal_area_crs,
            cell_size_km=self.cell_size_km,
            cell_ids=self.cell_ids,
            rows=self.rows,
            columns=self.columns,
            query_xy_m=self.query_xy_m,
            clipped_area_km2=self.clipped_area_km2,
        )


@dataclass(frozen=True, slots=True)
class Stage4GridFamily:
    coarse_50km: Stage4IntegrationGrid
    primary_25km: Stage4IntegrationGrid
    reference_12_5km: Stage4IntegrationGrid

    def __post_init__(self) -> None:
        if (
            self.coarse_50km.cell_size_km,
            self.primary_25km.cell_size_km,
            self.reference_12_5km.cell_size_km,
        ) != GRID_CELL_SIZES_KM:
            raise ValueError("stage-4 integration grid order changed")
        areas = np.asarray(
            [
                self.coarse_50km.total_area_km2,
                self.primary_25km.total_area_km2,
                self.reference_12_5km.total_area_km2,
            ],
            dtype=np.float64,
        )
        if float(np.max(areas) - np.min(areas)) > max(1e-6, float(np.max(areas)) * 1e-12):
            raise ValueError("stage-4 exact clipped grid areas disagree across resolutions")

    def grids(self) -> tuple[Stage4IntegrationGrid, ...]:
        return (self.coarse_50km, self.primary_25km, self.reference_12_5km)


def _grid_identity_payload(
    *,
    cell_size_km: float,
    cell_ids: tuple[str, ...],
    rows: IntArray,
    columns: IntArray,
    query_xy_m: FloatArray,
    clipped_area_km2: FloatArray,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "stage4_target_independent_integration_grid",
        "equal_area_crs": EQUAL_AREA_CRS,
        "cell_size_km_hex": float(cell_size_km).hex(),
        "cells": [
            {
                "cell_id": cell_id,
                "row": int(row),
                "column": int(column),
                "query_x_m_hex": float(x_m).hex(),
                "query_y_m_hex": float(y_m).hex(),
                "clipped_area_km2_hex": float(area).hex(),
            }
            for cell_id, row, column, (x_m, y_m), area in zip(
                cell_ids,
                rows,
                columns,
                query_xy_m,
                clipped_area_km2,
                strict=True,
            )
        ],
    }


def build_stage4_integration_grid(
    study_area_wgs84: BaseGeometry,
    *,
    cell_size_km: float,
) -> Stage4IntegrationGrid:
    """Build one grid from the study area, with no target-dependent argument."""

    size = float(cell_size_km)
    if size not in GRID_CELL_SIZES_KM:
        raise ValueError("stage-4 grid must be exactly 50, 25, or 12.5 km")
    projected = project_study_area_to_equal_area(study_area_wgs84)
    grid = build_clipped_grid(projected, cell_size_km=size)
    cell_ids = grid.cell_ids
    rows = np.asarray([cell.row for cell in grid.cells], dtype=np.int64)
    columns = np.asarray([cell.column for cell in grid.cells], dtype=np.int64)
    query_xy_m = np.asarray(
        [(cell.representative_point.x, cell.representative_point.y) for cell in grid.cells],
        dtype=np.float64,
    )
    clipped_area_km2 = np.asarray(
        [cell.clipped_area_m2 / 1_000_000.0 for cell in grid.cells],
        dtype=np.float64,
    )
    payload = _grid_identity_payload(
        cell_size_km=size,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=clipped_area_km2,
    )
    return Stage4IntegrationGrid(
        grid_id=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=size,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=clipped_area_km2,
    )


def build_stage4_grid_family(study_area_wgs84: BaseGeometry) -> Stage4GridFamily:
    """Build all three frozen grids from the same target-independent geometry."""

    grids = tuple(
        build_stage4_integration_grid(study_area_wgs84, cell_size_km=size)
        for size in GRID_CELL_SIZES_KM
    )
    return Stage4GridFamily(*grids)


def recompute_stage4_features(
    snapshots: tuple[Stage3IssueSnapshot, ...],
    grid: Stage4IntegrationGrid,
    *,
    query_chunk_size: int = 256,
    spatial_workers: int = 1,
) -> tuple[Stage3IssueFeatureTable, ...]:
    """Recompute accepted feature definitions at this grid's cell centers."""

    engine = Stage3FeatureEngine(
        snapshots,
        grid.as_stage3_query_grid(),
        query_chunk_size=query_chunk_size,
        spatial_workers=spatial_workers,
    )
    return tuple(engine.build_next_issue() for _ in snapshots)


def source_columns_for_variant(
    protocol: Stage4ProtocolBundle,
    variant: FeatureVariant,
) -> tuple[str, ...]:
    feature_sets = _mapping(protocol.feature_set.document.get("feature_sets"), label="feature_sets")
    definition = _mapping(feature_sets.get(variant), label=f"feature_sets.{variant}")
    source_columns = _string_sequence(
        definition.get("source_value_columns"),
        label=f"feature_sets.{variant}.source_value_columns",
    )
    audit_only = set(
        _string_sequence(
            definition.get("audit_only_quality_columns"),
            label=f"feature_sets.{variant}.audit_only_quality_columns",
        )
    )
    if set(source_columns) & audit_only:
        raise ValueError("audit-only quality columns escaped into the design sources")
    if len(set(source_columns)) != len(source_columns):
        raise ValueError("stage-4 source columns are duplicated")
    return source_columns


@dataclass(frozen=True, slots=True)
class RawFeatureMatrix:
    """Raw selected feature values and exact null mask before preprocessing."""

    source_columns: tuple[str, ...]
    values: FloatArray
    missing: BoolArray
    positive_area: BoolArray

    def __post_init__(self) -> None:
        rows, columns = self.values.shape
        if columns != len(self.source_columns) or self.missing.shape != (rows, columns):
            raise ValueError("raw stage-4 feature matrix shapes disagree")
        if self.positive_area.shape != (rows,) or not np.all(self.positive_area):
            raise ValueError("stage-4 fitting rows must all have positive clipped area")
        if np.any(self.missing != ~np.isfinite(self.values)):
            raise ValueError("raw stage-4 missing mask differs from nonfinite values")
        object.__setattr__(self, "values", cast(FloatArray, _readonly(self.values)))
        object.__setattr__(self, "missing", cast(BoolArray, _readonly(self.missing)))
        object.__setattr__(
            self,
            "positive_area",
            cast(BoolArray, _readonly(self.positive_area)),
        )


def extract_raw_feature_matrix(
    table: pa.Table,
    *,
    source_columns: Sequence[str],
) -> RawFeatureMatrix:
    """Extract frozen source columns; quality companions cannot be requested here."""

    names = tuple(source_columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("source column selection must be non-empty and unique")
    missing_columns = sorted(set((*names, "clipped_area_km2")) - set(table.column_names))
    if missing_columns:
        raise ValueError(f"stage-4 feature table is missing columns: {missing_columns}")
    columns: list[FloatArray] = []
    masks: list[BoolArray] = []
    for name in names:
        array = table[name].combine_chunks().cast(pa.float64())
        values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
        missing = np.asarray(array.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
        values = values.copy()
        values[missing] = np.nan
        columns.append(values)
        masks.append(missing)
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    missing_matrix = np.column_stack(masks).astype(np.bool_, copy=False)
    areas = np.asarray(
        table["clipped_area_km2"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    positive_area = np.asarray(np.isfinite(areas) & (areas > 0.0), dtype=np.bool_)
    if not np.all(positive_area):
        raise ValueError("stage-4 feature tables may contain only positive-area grid rows")
    return RawFeatureMatrix(
        source_columns=names,
        values=matrix,
        missing=missing_matrix,
        positive_area=positive_area,
    )


def selected_table_identity_sha256(table: pa.Table, columns: Sequence[str]) -> str:
    """Legacy R0 physical IPC identity; retained only for historical replay."""

    names = tuple(columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("identity columns must be non-empty and unique")
    if missing := sorted(set(names) - set(table.column_names)):
        raise ValueError(f"identity table is missing columns: {missing}")
    selected = table.select(list(names)).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, selected.schema) as writer:
        writer.write_table(selected)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _logical_identity_column_names(
    table: pa.Table,
    columns: Sequence[str],
) -> tuple[str, ...]:
    if not isinstance(table, pa.Table):
        raise TypeError("logical identity input must be an Arrow table")
    if isinstance(columns, str | bytes):
        raise TypeError("logical identity columns must be a sequence of names")
    names = tuple(columns)
    if (
        not names
        or not all(isinstance(name, str) and name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("logical identity columns must be non-empty unique names")
    if missing := sorted(set(names) - set(table.column_names)):
        raise ValueError(f"logical identity table is missing columns: {missing}")
    return names


def _logical_identity_type_width(data_type: pa.DataType) -> int | None:
    """Return the fixed-width payload size, or ``None`` for supported variable types."""

    if pa.types.is_boolean(data_type) or pa.types.is_string(data_type):
        return None
    if (
        pa.types.is_signed_integer(data_type)
        or pa.types.is_unsigned_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_timestamp(data_type)
    ):
        bit_width = getattr(data_type, "bit_width", None)
        if not isinstance(bit_width, int) or bit_width <= 0 or bit_width % 8:
            raise TypeError(f"logical identity has invalid fixed-width type {data_type}")
        return bit_width // 8
    raise TypeError(f"logical identity does not support Arrow type {data_type}")


def _buffer_view(buffer: pa.Buffer | None, *, label: str) -> memoryview:
    if buffer is None:
        raise ValueError(f"logical identity array is missing its {label} buffer")
    return memoryview(buffer)


def _logical_validity_bits_r1(chunk: pa.Array) -> BoolArray:
    source = chunk.buffers()[0]
    if source is None:
        return np.ones(len(chunk), dtype=np.bool_)
    raw = np.frombuffer(source, dtype=np.uint8)
    required_bits = chunk.offset + len(chunk)
    if raw.size * 8 < required_bits:
        raise ValueError("logical identity validity bitmap is truncated")
    unpacked = np.unpackbits(raw, bitorder="little")
    return np.asarray(
        unpacked[chunk.offset : required_bits],
        dtype=np.bool_,
    )


def _canonical_validity_buffer_r1(validity: BoolArray) -> pa.Buffer | None:
    if bool(np.all(validity)):
        return None
    return pa.py_buffer(np.packbits(validity, bitorder="little").tobytes())


def _canonical_fixed_width_array_r1(
    array: pa.ChunkedArray,
    *,
    byte_width: int,
) -> pa.Array:
    length = len(array)
    validity = np.empty(length, dtype=np.bool_)
    values = np.zeros((length, byte_width), dtype=np.uint8)
    output_index = 0
    for chunk in array.chunks:
        chunk_length = len(chunk)
        chunk_validity = _logical_validity_bits_r1(chunk)
        validity[output_index : output_index + chunk_length] = chunk_validity
        value_source = _buffer_view(chunk.buffers()[1], label="value")
        required = (chunk.offset + len(chunk)) * byte_width
        if len(value_source) < required:
            raise ValueError("logical identity fixed-width value buffer is truncated")
        source_start = chunk.offset * byte_width
        source_end = source_start + chunk_length * byte_width
        block = values[output_index : output_index + chunk_length]
        block[:, :] = np.frombuffer(
            value_source[source_start:source_end],
            dtype=np.uint8,
        ).reshape(chunk_length, byte_width)
        block[~chunk_validity, :] = 0
        output_index += chunk_length
    null_count = int(np.count_nonzero(~validity))
    return pa.Array.from_buffers(
        array.type,
        length,
        [_canonical_validity_buffer_r1(validity), pa.py_buffer(values.tobytes())],
        null_count=null_count,
        offset=0,
    )


def _canonical_boolean_array_r1(array: pa.ChunkedArray) -> pa.Array:
    length = len(array)
    validity = np.empty(length, dtype=np.bool_)
    values = np.zeros(length, dtype=np.bool_)
    output_index = 0
    for chunk in array.chunks:
        chunk_length = len(chunk)
        chunk_validity = _logical_validity_bits_r1(chunk)
        validity[output_index : output_index + chunk_length] = chunk_validity
        value_source = _buffer_view(chunk.buffers()[1], label="boolean value")
        required_bits = chunk.offset + len(chunk)
        if len(value_source) * 8 < required_bits:
            raise ValueError("logical identity boolean value buffer is truncated")
        source_values = np.unpackbits(
            np.frombuffer(value_source, dtype=np.uint8),
            bitorder="little",
        )[chunk.offset : required_bits].astype(np.bool_, copy=False)
        values[output_index : output_index + chunk_length] = source_values & chunk_validity
        output_index += chunk_length
    null_count = int(np.count_nonzero(~validity))
    return pa.Array.from_buffers(
        array.type,
        length,
        [
            _canonical_validity_buffer_r1(validity),
            pa.py_buffer(np.packbits(values, bitorder="little").tobytes()),
        ],
        null_count=null_count,
        offset=0,
    )


def _canonical_string_array_r1(array: pa.ChunkedArray) -> pa.Array:
    length = len(array)
    validity = np.empty(length, dtype=np.bool_)
    offsets = bytearray((length + 1) * 4)
    data = bytearray()
    output_index = 0
    struct.pack_into("<i", offsets, 0, 0)
    for chunk in array.chunks:
        buffers = chunk.buffers()
        chunk_length = len(chunk)
        chunk_validity = _logical_validity_bits_r1(chunk)
        validity[output_index : output_index + chunk_length] = chunk_validity
        offset_source = _buffer_view(buffers[1], label="string offset")
        data_source = memoryview(buffers[2]) if buffers[2] is not None else memoryview(b"")
        required_offsets = (chunk.offset + len(chunk) + 1) * 4
        if len(offset_source) < required_offsets:
            raise ValueError("logical identity string offset buffer is truncated")
        for local_index in range(len(chunk)):
            source_index = chunk.offset + local_index
            if bool(chunk_validity[local_index]):
                start = struct.unpack_from("<i", offset_source, source_index * 4)[0]
                end = struct.unpack_from("<i", offset_source, (source_index + 1) * 4)[0]
                if start < 0 or end < start or end > len(data_source):
                    raise ValueError("logical identity string offsets are invalid")
                data.extend(data_source[start:end])
            output_index += 1
            if len(data) > 2_147_483_647:
                raise OverflowError("logical identity utf8 payload exceeds int32 offsets")
            struct.pack_into("<i", offsets, output_index * 4, len(data))
    null_count = int(np.count_nonzero(~validity))
    return pa.Array.from_buffers(
        array.type,
        length,
        [
            _canonical_validity_buffer_r1(validity),
            pa.py_buffer(bytes(offsets)),
            pa.py_buffer(bytes(data)),
        ],
        null_count=null_count,
        offset=0,
    )


def _canonical_logical_array_r1(array: pa.ChunkedArray) -> pa.Array:
    byte_width = _logical_identity_type_width(array.type)
    if pa.types.is_boolean(array.type):
        return _canonical_boolean_array_r1(array)
    if pa.types.is_string(array.type):
        return _canonical_string_array_r1(array)
    if byte_width is None:  # pragma: no cover - supported variable types return above
        raise AssertionError("logical identity type classification is incomplete")
    return _canonical_fixed_width_array_r1(array, byte_width=byte_width)


def _canonical_logical_field_r1(field: pa.Field) -> pa.Field:
    _logical_identity_type_width(field.type)
    metadata = field.metadata
    ordered_metadata = (
        None if metadata is None else {key: metadata[key] for key in sorted(metadata)}
    )
    return pa.field(
        field.name,
        field.type,
        nullable=field.nullable,
        metadata=ordered_metadata,
    )


def _canonical_selected_logical_table_r1(
    table: pa.Table,
    columns: Sequence[str],
) -> pa.Table:
    names = _logical_identity_column_names(table, columns)
    fields = tuple(_canonical_logical_field_r1(table.schema.field(name)) for name in names)
    schema = pa.schema(fields, metadata=None)
    arrays = tuple(_canonical_logical_array_r1(table[name]) for name in names)
    return pa.Table.from_arrays(arrays, schema=schema)


def selected_table_logical_identity_sha256_r1(
    table: pa.Table,
    columns: Sequence[str],
) -> str:
    """Hash the frozen R1 logical Arrow identity under its explicit SHA-256 domain."""

    canonical = _canonical_selected_logical_table_r1(table, columns)
    return _canonical_logical_table_identity_sha256_r1(canonical)


def _canonical_logical_table_identity_sha256_r1(canonical: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(metadata_version=pa.MetadataVersion.V5)
    with pa.ipc.new_stream(sink, canonical.schema, options=options) as writer:
        writer.write_table(canonical)
    digest = hashlib.sha256()
    digest.update(SELECTED_TABLE_LOGICAL_IDENTITY_DOMAIN_R1)
    digest.update(sink.getvalue().to_pybytes())
    return digest.hexdigest()


def assert_selected_columns_logically_exact_r1(
    accepted: pa.Table,
    recomputed: pa.Table,
    *,
    columns: Sequence[str],
) -> str:
    """Fail unless every R1-preserved field, validity bit, and valid payload bit matches."""

    accepted_canonical = _canonical_selected_logical_table_r1(accepted, columns)
    recomputed_canonical = _canonical_selected_logical_table_r1(recomputed, columns)
    if not accepted_canonical.schema.equals(recomputed_canonical.schema, check_metadata=True):
        raise ValueError(
            "identity reconstruction differs: fields, types, nullability, metadata, or "
            "column order changed"
        )
    accepted_hash = _canonical_logical_table_identity_sha256_r1(accepted_canonical)
    recomputed_hash = _canonical_logical_table_identity_sha256_r1(recomputed_canonical)
    if accepted_hash != recomputed_hash:
        raise ValueError(
            "identity reconstruction differs: values, validity, or valid payload bits changed"
        )
    return accepted_hash


def assert_selected_columns_exact(
    accepted: pa.Table,
    recomputed: pa.Table,
    *,
    columns: Sequence[str],
) -> str:
    """Fail unless values and Arrow validity bitmaps are exactly identical."""

    accepted_hash = selected_table_identity_sha256(accepted, columns)
    recomputed_hash = selected_table_identity_sha256(recomputed, columns)
    if accepted_hash != recomputed_hash:
        raise ValueError("identity reconstruction differs from the accepted stage-3 columns")
    return accepted_hash


def relative_total_intensity_error(candidate: float, reference: float) -> float:
    """Frozen spatial/time integration convergence denominator."""

    left = float(candidate)
    right = float(reference)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("integrated intensities must be finite")
    return abs(left - right) / max(abs(right), 1.0e-12)


__all__ = [
    "SELECTED_TABLE_LOGICAL_IDENTITY_DOMAIN_R1",
    "SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1",
    "FeatureVariant",
    "RawFeatureMatrix",
    "Stage4GridFamily",
    "Stage4IntegrationGrid",
    "assert_selected_columns_exact",
    "assert_selected_columns_logically_exact_r1",
    "build_stage4_grid_family",
    "build_stage4_integration_grid",
    "extract_raw_feature_matrix",
    "recompute_stage4_features",
    "relative_total_intensity_error",
    "selected_table_identity_sha256",
    "selected_table_logical_identity_sha256_r1",
    "source_columns_for_variant",
]
