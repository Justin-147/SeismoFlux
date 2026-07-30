"""Single-buffer, immutable earthquake-catalog parsing for Stage 2S.

Physical path opening and streaming deliberately live outside this module.  The
only accepted payload is an already frozen ``bytes`` object, and the Parquet
table is parsed exactly once through :class:`pyarrow.BufferReader`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from seismoflux.data.contracts import CONTRACTS
from seismoflux.data.parquet import schema_sha256, table_content_sha256

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

FROZEN_CATALOG_ROW_COUNT = 40_898
FROZEN_CATALOG_FILE_SHA256 = "2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347"
FROZEN_CATALOG_CONTENT_SHA256 = "2005f0ec465978829d0832e7228f22ecd34f1a7e9f268598979de72a5295e404"
FROZEN_CATALOG_SCHEMA_SHA256 = "c88d20b32a2cb599f22d1921764757b617fe19b030a0f58585652e9f7c74f3c5"
REQUIRED_RAW_FIELDS = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "longitude",
    "latitude",
    "magnitude",
    "inside_study_area",
)


def _sha256_hex(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


@dataclass(frozen=True, slots=True)
class ArrowFieldContract:
    """One exact Arrow field declaration, including position and nullability."""

    name: str
    type_name: str
    nullable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not (name := self.name.strip()):
            raise ValueError("Arrow field name must be non-empty")
        if not isinstance(self.type_name, str) or not (type_name := self.type_name.strip()):
            raise ValueError("Arrow field type_name must be non-empty")
        if not isinstance(self.nullable, bool):
            raise TypeError("Arrow field nullable must be bool")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type_name", type_name)

    @classmethod
    def from_arrow(cls, value: pa.Field) -> ArrowFieldContract:
        return cls(
            name=value.name,
            type_name=str(value.type),
            nullable=value.nullable,
        )


@dataclass(frozen=True, slots=True)
class CatalogByteContract:
    """Exact identities required before catalog rows may be exposed."""

    row_count: int
    file_sha256: str
    content_sha256: str
    schema_sha256: str
    fields: tuple[ArrowFieldContract, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count <= 0
        ):
            raise ValueError("row_count must be a positive integer")
        fields = tuple(self.fields)
        if not fields or any(not isinstance(item, ArrowFieldContract) for item in fields):
            raise ValueError("fields must contain ArrowFieldContract values")
        names = tuple(item.name for item in fields)
        if len(set(names)) != len(names):
            raise ValueError("catalog field names must be unique")
        if not set(REQUIRED_RAW_FIELDS).issubset(names):
            raise ValueError("catalog contract is missing a required Stage 2S raw field")
        object.__setattr__(self, "file_sha256", _sha256_hex("file_sha256", self.file_sha256))
        object.__setattr__(
            self,
            "content_sha256",
            _sha256_hex("content_sha256", self.content_sha256),
        )
        object.__setattr__(
            self,
            "schema_sha256",
            _sha256_hex("schema_sha256", self.schema_sha256),
        )
        object.__setattr__(self, "fields", fields)


_EARTHQUAKE_SCHEMA = CONTRACTS["earthquake_event"].schema
FROZEN_EARTHQUAKE_CATALOG_CONTRACT = CatalogByteContract(
    row_count=FROZEN_CATALOG_ROW_COUNT,
    file_sha256=FROZEN_CATALOG_FILE_SHA256,
    content_sha256=FROZEN_CATALOG_CONTENT_SHA256,
    schema_sha256=FROZEN_CATALOG_SCHEMA_SHA256,
    fields=tuple(ArrowFieldContract.from_arrow(field) for field in _EARTHQUAKE_SCHEMA),
)


def _read_only_int64(name: str, value: object, *, length: int) -> IntArray:
    result = np.asarray(value, dtype=np.int64)
    if result.ndim != 1 or result.size != length:
        raise ValueError(f"{name} must be a one-dimensional catalog-length array")
    owned = np.array(result, dtype=np.int64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_only_float64(name: str, value: object, *, length: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size != length:
        raise ValueError(f"{name} must be a one-dimensional catalog-length array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    owned = np.array(result, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_only_bool(name: str, value: object, *, length: int) -> BoolArray:
    result = np.asarray(value)
    if result.ndim != 1 or result.size != length or result.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must be a one-dimensional catalog-length boolean array")
    owned = np.array(result, dtype=np.bool_, copy=True, order="C")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class CatalogIdentity:
    """Verified identities of the exact in-memory bytes and parsed Arrow table."""

    row_count: int
    file_sha256: str
    content_sha256: str
    schema_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count <= 0
        ):
            raise ValueError("row_count must be a positive integer")
        object.__setattr__(self, "file_sha256", _sha256_hex("file_sha256", self.file_sha256))
        object.__setattr__(
            self,
            "content_sha256",
            _sha256_hex("content_sha256", self.content_sha256),
        )
        object.__setattr__(
            self,
            "schema_sha256",
            _sha256_hex("schema_sha256", self.schema_sha256),
        )


@dataclass(frozen=True, slots=True)
class Stage2SEarthquakeCatalog:
    """Read-only raw fields shared by all Stage 2S role views."""

    identity: CatalogIdentity
    event_ids: tuple[str, ...]
    origin_time_us: IntArray
    available_at_us: IntArray
    longitude: FloatArray
    latitude: FloatArray
    magnitude: FloatArray
    inside_study_area: BoolArray
    _table: pa.Table = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        row_count = self.identity.row_count
        event_ids = tuple(self.event_ids)
        if len(event_ids) != row_count:
            raise ValueError("event_ids must match the verified row count")
        if any(not isinstance(value, str) or not value for value in event_ids):
            raise ValueError("event_ids must contain only non-empty strings")
        if len(set(event_ids)) != row_count:
            raise ValueError("event_ids must be unique physical-event identifiers")
        origin = _read_only_int64("origin_time_us", self.origin_time_us, length=row_count)
        available = _read_only_int64("available_at_us", self.available_at_us, length=row_count)
        if np.any(available < origin):
            raise ValueError("available_at cannot precede origin_time_utc")
        longitude = _read_only_float64("longitude", self.longitude, length=row_count)
        latitude = _read_only_float64("latitude", self.latitude, length=row_count)
        magnitude = _read_only_float64("magnitude", self.magnitude, length=row_count)
        if np.any((longitude < -180.0) | (longitude > 180.0)):
            raise ValueError("catalog longitude lies outside [-180, 180]")
        if np.any((latitude < -90.0) | (latitude > 90.0)):
            raise ValueError("catalog latitude lies outside [-90, 90]")
        inside = _read_only_bool(
            "inside_study_area",
            self.inside_study_area,
            length=row_count,
        )
        if not isinstance(self._table, pa.Table) or self._table.num_rows != row_count:
            raise ValueError("_table must be the verified catalog-length Arrow table")
        fixed_order = tuple(
            sorted(
                range(row_count),
                key=lambda index: (
                    int(origin[index]),
                    event_ids[index].encode("utf-8"),
                ),
            )
        )
        if fixed_order != tuple(range(row_count)):
            raise ValueError("catalog rows must remain sorted by origin_time_utc then event_id")
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "origin_time_us", origin)
        object.__setattr__(self, "available_at_us", available)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "inside_study_area", inside)

    @property
    def row_count(self) -> int:
        return self.identity.row_count

    @property
    def table(self) -> pa.Table:
        """Return the same immutable Arrow table parsed from the one byte buffer."""

        return self._table


def _timestamp_us(table: pa.Table, name: str) -> IntArray:
    column = table.column(name)
    if column.null_count:
        raise ValueError(f"{name} must not contain null values")
    converted = column.combine_chunks().cast(pa.int64()).to_numpy(zero_copy_only=False)
    return np.asarray(converted, dtype=np.int64)


def _float_column(table: pa.Table, name: str) -> FloatArray:
    column = table.column(name)
    if column.null_count:
        raise ValueError(f"{name} must not contain null values")
    return np.asarray(
        column.combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


def _bool_column(table: pa.Table, name: str) -> BoolArray:
    column = table.column(name)
    if column.null_count:
        raise ValueError(f"{name} must not contain null values")
    return np.asarray(
        column.combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    )


def _event_ids(table: pa.Table) -> tuple[str, ...]:
    column = table.column("event_id")
    if column.null_count:
        raise ValueError("event_id must not contain null values")
    values = tuple(column.combine_chunks().to_pylist())
    if any(not isinstance(value, str) for value in values):
        raise ValueError("event_id must decode as strings")
    return values


def parse_catalog_bytes(
    payload: bytes,
    *,
    contract: CatalogByteContract,
) -> Stage2SEarthquakeCatalog:
    """Verify and parse one immutable Parquet payload without any path access."""

    if type(payload) is not bytes:
        raise TypeError("payload must be one immutable bytes object")
    if not isinstance(contract, CatalogByteContract):
        raise TypeError("contract must be a CatalogByteContract")
    actual_file_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_file_sha256 != contract.file_sha256:
        raise ValueError("catalog file_sha256 mismatch")
    try:
        with pa.BufferReader(payload) as reader:
            table = pq.read_table(reader, use_threads=False)
        table.validate(full=True)
    except (pa.ArrowInvalid, pa.ArrowIOError, OSError) as error:
        raise ValueError("catalog bytes are not a valid Parquet table") from error
    if table.num_rows != contract.row_count:
        raise ValueError("catalog row_count mismatch")
    actual_fields = tuple(ArrowFieldContract.from_arrow(field) for field in table.schema)
    if actual_fields != contract.fields:
        raise ValueError("catalog field order, type, or nullability mismatch")
    for arrow_field in table.schema:
        if not arrow_field.nullable and table.column(arrow_field.name).null_count:
            raise ValueError(f"non-nullable catalog field contains nulls: {arrow_field.name}")
    actual_content_sha256 = table_content_sha256(table)
    if actual_content_sha256 != contract.content_sha256:
        raise ValueError("catalog content_sha256 mismatch")
    actual_schema_sha256 = schema_sha256(table.schema)
    if actual_schema_sha256 != contract.schema_sha256:
        raise ValueError("catalog schema_sha256 mismatch")
    identity = CatalogIdentity(
        row_count=table.num_rows,
        file_sha256=actual_file_sha256,
        content_sha256=actual_content_sha256,
        schema_sha256=actual_schema_sha256,
    )
    catalog = Stage2SEarthquakeCatalog(
        identity=identity,
        event_ids=_event_ids(table),
        origin_time_us=_timestamp_us(table, "origin_time_utc"),
        available_at_us=_timestamp_us(table, "available_at"),
        longitude=_float_column(table, "longitude"),
        latitude=_float_column(table, "latitude"),
        magnitude=_float_column(table, "magnitude"),
        inside_study_area=_bool_column(table, "inside_study_area"),
        _table=table,
    )
    if any(
        not math.isfinite(float(value))
        for values in (catalog.longitude, catalog.latitude, catalog.magnitude)
        for value in values
    ):
        raise ValueError("catalog numerical fields must remain finite")
    return catalog


def parse_frozen_catalog_bytes(payload: bytes) -> Stage2SEarthquakeCatalog:
    """Parse bytes only against the preregistered Stage 2S catalog identities."""

    return parse_catalog_bytes(
        payload,
        contract=FROZEN_EARTHQUAKE_CATALOG_CONTRACT,
    )


__all__ = [
    "FROZEN_CATALOG_CONTENT_SHA256",
    "FROZEN_CATALOG_FILE_SHA256",
    "FROZEN_CATALOG_ROW_COUNT",
    "FROZEN_CATALOG_SCHEMA_SHA256",
    "FROZEN_EARTHQUAKE_CATALOG_CONTRACT",
    "REQUIRED_RAW_FIELDS",
    "ArrowFieldContract",
    "CatalogByteContract",
    "CatalogIdentity",
    "Stage2SEarthquakeCatalog",
    "parse_catalog_bytes",
    "parse_frozen_catalog_bytes",
]
