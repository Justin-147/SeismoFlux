"""Bounded-memory, deterministic Parquet storage for stage-3 feature artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from seismoflux.config import sha256_file
from seismoflux.data.parquet import schema_sha256, table_content_sha256

_CONTENT_HASH_DOMAIN = b"seismoflux_stage3_persisted_row_groups_v2\0"
_CONTENT_HASH_ALGORITHM = b"stage3_persisted_row_groups_v2"


class _HashLike(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Stage3DatasetArtifact:
    """Identity and shape of one locally restricted stage-3 Parquet dataset."""

    name: str
    path: str
    row_count: int
    row_group_count: int
    file_size_bytes: int
    file_sha256: str
    content_sha256: str
    schema_sha256: str
    sort_keys: tuple[str, ...]

    def as_manifest_entry(self, schema: pa.Schema) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "fields": [
                {
                    "name": field.name,
                    "nullable": field.nullable,
                    "type": str(field.type),
                }
                for field in schema
            ],
            "file_sha256": self.file_sha256,
            "file_size_bytes": self.file_size_bytes,
            "path": self.path,
            "row_count": self.row_count,
            "row_group_count": self.row_group_count,
            "schema_sha256": self.schema_sha256,
            "sort_keys": list(self.sort_keys),
        }


def _field_schema(schema: pa.Schema) -> pa.Schema:
    return schema.remove_metadata()


def _ordered_value(value: object) -> tuple[bool, Any]:
    return value is not None, value if value is not None else ""


def _row_key(table: pa.Table, index: int, sort_keys: Sequence[str]) -> tuple[Any, ...]:
    return tuple(_ordered_value(table[key][index].as_py()) for key in sort_keys)


def _require_sorted(
    table: pa.Table,
    *,
    sort_keys: tuple[str, ...],
    previous_last_key: tuple[Any, ...] | None,
) -> tuple[Any, ...] | None:
    if not sort_keys or table.num_rows == 0:
        return previous_last_key
    missing = [key for key in sort_keys if key not in table.column_names]
    if missing:
        raise ValueError(f"stage-3 table is missing sort keys: {missing}")
    sorted_table = table.sort_by([(key, "ascending") for key in sort_keys])
    if not table.equals(sorted_table, check_metadata=False):
        raise ValueError("stage-3 Parquet row group is not deterministically sorted")
    first_key = _row_key(table, 0, sort_keys)
    last_key = _row_key(table, table.num_rows - 1, sort_keys)
    if previous_last_key is not None and first_key < previous_last_key:
        raise ValueError("stage-3 Parquet row groups are not globally sorted")
    return last_key


def _content_digest_start(schema: pa.Schema) -> _HashLike:
    digest = hashlib.sha256()
    digest.update(_CONTENT_HASH_DOMAIN)
    serialized = schema.serialize().to_pybytes()
    digest.update(len(serialized).to_bytes(8, "big"))
    digest.update(serialized)
    return digest


def _update_content_digest(digest: _HashLike, table: pa.Table) -> None:
    group_digest = bytes.fromhex(table_content_sha256(table))
    digest.update(table.num_rows.to_bytes(8, "big"))
    digest.update(group_digest)


def _persisted_content_sha256(parquet_file: Any) -> str:
    """Hash the exact Arrow view reconstructed from persisted Parquet row groups."""

    digest = _content_digest_start(parquet_file.schema_arrow)
    for index in range(parquet_file.metadata.num_row_groups):
        _update_content_digest(digest, parquet_file.read_row_group(index))
    return digest.hexdigest()


def write_parquet_row_groups_atomic(
    *,
    name: str,
    row_groups: Iterable[pa.Table],
    schema: pa.Schema,
    output_path: Path,
    project_root: Path,
    sort_keys: tuple[str, ...],
    compression: str = "zstd",
    compression_level: int = 9,
    version: str = "2.6",
    data_page_version: str = "1.0",
    use_dictionary: bool = False,
    timestamp_unit: str = "us",
) -> Stage3DatasetArtifact:
    """Write already ordered row groups without materializing the full dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"seismoflux_contract": b"0.3.0",
        b"seismoflux_content_hash": _CONTENT_HASH_ALGORITHM,
        b"seismoflux_sort_keys": ",".join(sort_keys).encode("utf-8"),
    }
    storage_schema = schema.with_metadata(metadata)
    expected_fields = _field_schema(storage_schema)
    row_count = 0
    row_group_count = 0
    previous_last_key: tuple[Any, ...] | None = None
    temporary_name: str | None = None
    writer: pq.ParquetWriter | None = None
    persisted_schema: pa.Schema | None = None
    content_sha256: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        writer = pq.ParquetWriter(
            temporary_name,
            storage_schema,
            compression=compression,
            compression_level=compression_level,
            version=version,
            data_page_version=data_page_version,
            use_dictionary=use_dictionary,
        )
        for raw_table in row_groups:
            if raw_table.num_rows == 0:
                continue
            if _field_schema(raw_table.schema) != expected_fields:
                raise ValueError(f"stage-3 row-group schema mismatch for {name}")
            table = raw_table.replace_schema_metadata(metadata)
            previous_last_key = _require_sorted(
                table,
                sort_keys=sort_keys,
                previous_last_key=previous_last_key,
            )
            writer.write_table(table, row_group_size=table.num_rows)
            row_count += table.num_rows
            row_group_count += 1
        writer.close()
        writer = None
        if row_group_count == 0:
            raise ValueError(f"stage-3 dataset must contain at least one row: {name}")
        with pq.ParquetFile(temporary_name) as parquet_file:
            if parquet_file.metadata.num_rows != row_count:
                raise RuntimeError(f"Parquet round-trip changed row count for {name}")
            if parquet_file.metadata.num_row_groups != row_group_count:
                raise RuntimeError(f"Parquet round-trip changed row-group count for {name}")
            persisted_schema = parquet_file.schema_arrow
            if _field_schema(persisted_schema) != expected_fields:
                raise RuntimeError(f"Parquet round-trip changed schema for {name}")
            content_sha256 = _persisted_content_sha256(parquet_file)
        os.replace(temporary_name, output_path)
        temporary_name = None
    except Exception:
        if writer is not None:
            writer.close()
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise

    if persisted_schema is None or content_sha256 is None:
        raise AssertionError("stage-3 persisted Parquet verification did not complete")
    relative_path = output_path.resolve().relative_to(project_root.resolve()).as_posix()
    return Stage3DatasetArtifact(
        name=name,
        path=relative_path,
        row_count=row_count,
        row_group_count=row_group_count,
        file_size_bytes=output_path.stat().st_size,
        file_sha256=sha256_file(output_path),
        content_sha256=content_sha256,
        schema_sha256=schema_sha256(persisted_schema),
        sort_keys=sort_keys,
    )


def verify_stage3_parquet_artifact(
    *,
    project_root: Path,
    entry: dict[str, Any],
    schema: pa.Schema,
) -> list[str]:
    """Verify a manifest entry by file hash and deterministic row-group content."""

    errors: list[str] = []
    path = (project_root / str(entry.get("path", ""))).resolve()
    if not path.is_relative_to(project_root.resolve()) or not path.is_file():
        return [f"missing or unsafe stage-3 dataset: {entry.get('path')}"]
    if sha256_file(path) != entry.get("file_sha256"):
        return [f"stage-3 file hash mismatch: {entry.get('path')}"]
    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows != entry.get("row_count"):
        errors.append(f"stage-3 row count mismatch: {entry.get('path')}")
    if parquet_file.metadata.num_row_groups != entry.get("row_group_count"):
        errors.append(f"stage-3 row-group count mismatch: {entry.get('path')}")
    if path.stat().st_size != entry.get("file_size_bytes"):
        errors.append(f"stage-3 file size mismatch: {entry.get('path')}")
    if _field_schema(parquet_file.schema_arrow) != _field_schema(schema):
        errors.append(f"stage-3 schema mismatch: {entry.get('path')}")
        return errors
    if schema_sha256(parquet_file.schema_arrow) != entry.get("schema_sha256"):
        errors.append(f"stage-3 schema hash mismatch: {entry.get('path')}")

    if _persisted_content_sha256(parquet_file) != entry.get("content_sha256"):
        errors.append(f"stage-3 content hash mismatch: {entry.get('path')}")
    return errors


__all__ = [
    "Stage3DatasetArtifact",
    "verify_stage3_parquet_artifact",
    "write_parquet_row_groups_atomic",
]
