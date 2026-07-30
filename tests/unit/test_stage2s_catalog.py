"""Pure in-memory tests for the Stage 2S single-buffer catalog contract."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.data.contracts import CONTRACTS
from seismoflux.data.parquet import (
    schema_sha256,
    table_content_sha256,
    table_from_records,
)
from seismoflux.stage2s.catalog import (
    ArrowFieldContract,
    CatalogByteContract,
    parse_catalog_bytes,
)

_LOCAL_OFFSET = timezone(timedelta(hours=8))


def _record(
    event_id: str,
    origin: datetime,
    *,
    available_delay_hours: int = 0,
    longitude: float = 105.0,
    latitude: float = 35.0,
    magnitude: float = 5.2,
    inside: bool = True,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "origin_time_utc": origin,
        "available_at": origin + timedelta(hours=available_delay_hours),
        "origin_time_local": origin.astimezone(_LOCAL_OFFSET),
        "longitude": longitude,
        "latitude": latitude,
        "depth_km": 10.0,
        "magnitude": magnitude,
        "magnitude_type": "M",
        "place": "synthetic",
        "catalog_sources": ["synthetic-source"],
        "inside_study_area": inside,
        "dedup_confidence": "exact",
        "anchor_source_record_id": f"source-{event_id}",
        "quality_flags": [],
    }


def _payload_and_contract(
    records: list[dict[str, object]],
    *,
    column_order: tuple[str, ...] | None = None,
) -> tuple[bytes, CatalogByteContract, pa.Table]:
    stage1 = CONTRACTS["earthquake_event"]
    table = table_from_records(records, stage1.schema, stage1.sort_keys)
    if column_order is not None:
        table = table.select(column_order)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        version="2.6",
        data_page_version="1.0",
        use_dictionary=False,
        write_statistics=True,
        coerce_timestamps="us",
        allow_truncated_timestamps=False,
    )
    payload = sink.getvalue().to_pybytes()
    persisted = pq.read_table(pa.BufferReader(payload), use_threads=False)
    contract = CatalogByteContract(
        row_count=persisted.num_rows,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        content_sha256=table_content_sha256(persisted),
        schema_sha256=schema_sha256(persisted.schema),
        fields=tuple(ArrowFieldContract.from_arrow(field) for field in persisted.schema),
    )
    return payload, contract, persisted


def _records() -> list[dict[str, object]]:
    anchor = datetime(2022, 1, 1, tzinfo=UTC)
    return [
        _record("event-b", anchor + timedelta(days=1), magnitude=4.1),
        _record("event-a", anchor, available_delay_hours=3, magnitude=5.5),
        _record(
            "event-c",
            anchor + timedelta(days=2),
            longitude=120.0,
            latitude=20.0,
            magnitude=5.9,
            inside=False,
        ),
    ]


def test_parse_catalog_bytes_verifies_all_hashes_and_returns_read_only_raw_arrays() -> None:
    payload, contract, persisted = _payload_and_contract(_records())
    catalog = parse_catalog_bytes(payload, contract=contract)
    assert catalog.identity.file_sha256 == hashlib.sha256(payload).hexdigest()
    assert catalog.identity.content_sha256 == table_content_sha256(persisted)
    assert catalog.identity.schema_sha256 == schema_sha256(persisted.schema)
    assert catalog.event_ids == ("event-a", "event-b", "event-c")
    assert catalog.table is catalog.table
    assert catalog.table.equals(persisted, check_metadata=True)
    assert catalog.origin_time_us.tolist() == sorted(catalog.origin_time_us.tolist())
    assert catalog.available_at_us[0] > catalog.origin_time_us[0]
    assert catalog.inside_study_area.tolist() == [True, True, False]
    for values in (
        catalog.origin_time_us,
        catalog.available_at_us,
        catalog.longitude,
        catalog.latitude,
        catalog.magnitude,
        catalog.inside_study_area,
    ):
        assert not values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        catalog.magnitude[0] = 9.0


def test_file_hash_is_checked_before_invalid_parquet_is_parsed() -> None:
    payload, contract, _ = _payload_and_contract(_records())
    corrupted = payload[:-1] + bytes([payload[-1] ^ 0x01])
    with pytest.raises(ValueError, match="file_sha256 mismatch"):
        parse_catalog_bytes(corrupted, contract=contract)
    with pytest.raises(TypeError, match="immutable bytes"):
        parse_catalog_bytes(bytearray(payload), contract=contract)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("row_count", "row_count mismatch"),
        ("content_sha256", "content_sha256 mismatch"),
        ("schema_sha256", "schema_sha256 mismatch"),
    ],
)
def test_declared_catalog_identity_mismatch_fails_closed(field: str, message: str) -> None:
    payload, contract, _ = _payload_and_contract(_records())
    values: dict[str, object] = {
        "row_count": contract.row_count,
        "file_sha256": contract.file_sha256,
        "content_sha256": contract.content_sha256,
        "schema_sha256": contract.schema_sha256,
        "fields": contract.fields,
    }
    values[field] = contract.row_count + 1 if field == "row_count" else "0" * 64
    mismatched = CatalogByteContract(**values)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        parse_catalog_bytes(payload, contract=mismatched)


def test_exact_arrow_field_order_type_and_nullability_are_required() -> None:
    original_names = tuple(CONTRACTS["earthquake_event"].schema.names)
    reordered_names = (original_names[1], original_names[0], *original_names[2:])
    payload, actual_contract, _ = _payload_and_contract(
        _records(),
        column_order=reordered_names,
    )
    frozen_fields = tuple(
        ArrowFieldContract.from_arrow(field) for field in CONTRACTS["earthquake_event"].schema
    )
    mismatched = CatalogByteContract(
        row_count=actual_contract.row_count,
        file_sha256=actual_contract.file_sha256,
        content_sha256=actual_contract.content_sha256,
        schema_sha256=actual_contract.schema_sha256,
        fields=frozen_fields,
    )
    with pytest.raises(ValueError, match="field order, type, or nullability"):
        parse_catalog_bytes(payload, contract=mismatched)


def test_catalog_rejects_duplicate_physical_event_ids_after_hash_verification() -> None:
    anchor = datetime(2022, 1, 1, tzinfo=UTC)
    payload, contract, _ = _payload_and_contract(
        [
            _record("duplicate", anchor),
            _record("duplicate", anchor + timedelta(days=1)),
        ]
    )
    with pytest.raises(ValueError, match="unique physical-event"):
        parse_catalog_bytes(payload, contract=contract)
