"""Deterministic Arrow and Parquet serialization for standardized datasets."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from seismoflux.config import sha256_file
from seismoflux.data.settings import ParquetSettings


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    name: str
    path: str
    row_count: int
    file_sha256: str
    content_sha256: str
    schema_sha256: str
    sort_keys: tuple[str, ...]

    def as_catalog_entry(self, schema: pa.Schema) -> dict[str, Any]:
        return {
            "path": self.path,
            "row_count": self.row_count,
            "file_sha256": self.file_sha256,
            "content_sha256": self.content_sha256,
            "schema_sha256": self.schema_sha256,
            "sort_keys": list(self.sort_keys),
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in schema
            ],
        }


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def table_content_sha256(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def table_from_records(
    records: Sequence[dict[str, Any]],
    schema: pa.Schema,
    sort_keys: tuple[str, ...],
) -> pa.Table:
    """Build a strict table and apply a deterministic total ordering."""

    table = pa.Table.from_pylist(list(records), schema=schema)
    if sort_keys and table.num_rows:
        table = table.sort_by([(key, "ascending") for key in sort_keys])
    metadata = {
        b"seismoflux_contract": b"0.1.0",
        b"seismoflux_sort_keys": ",".join(sort_keys).encode("utf-8"),
    }
    return table.replace_schema_metadata(metadata)


def write_parquet_atomic(
    *,
    name: str,
    table: pa.Table,
    output_path: Path,
    project_root: Path,
    sort_keys: tuple[str, ...],
    settings: ParquetSettings,
) -> DatasetArtifact:
    """Write a Parquet file atomically with all writer options frozen."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        pq.write_table(
            table,
            temporary_name,
            compression=settings.compression,
            compression_level=settings.compression_level,
            version=settings.version,
            data_page_version=settings.data_page_version,
            use_dictionary=settings.use_dictionary,
            write_statistics=True,
            coerce_timestamps=settings.timestamp_unit,
            allow_truncated_timestamps=False,
        )
        os.replace(temporary_name, output_path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise

    persisted_table = pq.read_table(output_path)
    if persisted_table.num_rows != table.num_rows:
        raise RuntimeError(f"Parquet round-trip changed the row count for {name}")
    relative_path = output_path.resolve().relative_to(project_root.resolve()).as_posix()
    return DatasetArtifact(
        name=name,
        path=relative_path,
        row_count=persisted_table.num_rows,
        file_sha256=sha256_file(output_path),
        content_sha256=table_content_sha256(persisted_table),
        schema_sha256=schema_sha256(persisted_table.schema),
        sort_keys=sort_keys,
    )


def verify_parquet_artifact(project_root: Path, entry: dict[str, Any]) -> list[str]:
    """Return validation errors for one catalog entry without mutating it."""

    errors: list[str] = []
    path = (project_root / str(entry["path"])).resolve()
    if not path.is_relative_to(project_root.resolve()) or not path.is_file():
        return [f"missing or unsafe dataset path: {entry['path']}"]
    if sha256_file(path) != entry["file_sha256"]:
        errors.append(f"file hash mismatch: {entry['path']}")
        return errors
    table = pq.read_table(path)
    if table.num_rows != entry["row_count"]:
        errors.append(f"row count mismatch: {entry['path']}")
    if table_content_sha256(table) != entry["content_sha256"]:
        errors.append(f"content hash mismatch: {entry['path']}")
    if schema_sha256(table.schema) != entry["schema_sha256"]:
        errors.append(f"schema hash mismatch: {entry['path']}")
    actual_fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in table.schema
    ]
    if actual_fields != entry.get("fields"):
        errors.append(f"field schema mismatch: {entry['path']}")
    sort_keys = entry.get("sort_keys")
    if not isinstance(sort_keys, list) or not all(isinstance(key, str) for key in sort_keys):
        errors.append(f"invalid sort keys: {entry['path']}")
    elif table.num_rows:
        sorted_table = table.sort_by([(key, "ascending") for key in sort_keys])
        if not table.equals(sorted_table, check_metadata=True):
            errors.append(f"row order mismatch: {entry['path']}")
    return errors
