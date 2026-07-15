from __future__ import annotations

import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.features.anomaly.storage import (
    verify_stage3_parquet_artifact,
    write_parquet_row_groups_atomic,
)

SCHEMA = pa.schema(
    [
        pa.field("issue_index", pa.int16(), nullable=False),
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("value", pa.float32(), nullable=True),
    ]
)


def _table(records: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(records, schema=SCHEMA)


def _table_with_hidden_null_payload(hidden_value: float) -> pa.Table:
    validity = pa.py_buffer(b"\x05")
    values = pa.py_buffer(struct.pack("<fff", 1.0, hidden_value, 2.0))
    nullable_values = pa.Array.from_buffers(
        pa.float32(),
        3,
        [validity, values],
        null_count=1,
    )
    return pa.Table.from_arrays(
        [
            pa.array([0, 0, 0], type=pa.int16()),
            pa.array(["c0", "c1", "c2"], type=pa.string()),
            nullable_values,
        ],
        schema=SCHEMA,
    )


def test_stage3_streaming_parquet_round_trips_and_verifies(tmp_path: Path) -> None:
    output = tmp_path / "data" / "processed" / "stage3" / "bundle" / "features.parquet"
    groups = [
        _table(
            [
                {"issue_index": 0, "cell_id": "c0", "value": 1.0},
                {"issue_index": 0, "cell_id": "c1", "value": None},
            ]
        ),
        _table([{"issue_index": 1, "cell_id": "c0", "value": 2.0}]),
    ]

    artifact = write_parquet_row_groups_atomic(
        name="features",
        row_groups=groups,
        schema=SCHEMA,
        output_path=output,
        project_root=tmp_path,
        sort_keys=("issue_index", "cell_id"),
    )
    entry = artifact.as_manifest_entry(output_schema := pq.read_schema(output))

    assert artifact.row_count == 3
    assert artifact.row_group_count == 2
    assert artifact.path == "data/processed/stage3/bundle/features.parquet"
    assert (
        verify_stage3_parquet_artifact(
            project_root=tmp_path,
            entry=entry,
            schema=output_schema,
        )
        == []
    )


def test_persisted_content_hash_ignores_invisible_null_slot_payload(tmp_path: Path) -> None:
    left = _table_with_hidden_null_payload(17.0)
    right = _table_with_hidden_null_payload(999.0)
    assert left.equals(right, check_metadata=True)
    assert left["value"].chunk(0).buffers()[1] != right["value"].chunk(0).buffers()[1]

    artifacts = []
    for name, table in (("left", left), ("right", right)):
        output = tmp_path / f"{name}.parquet"
        artifact = write_parquet_row_groups_atomic(
            name=name,
            row_groups=[table],
            schema=SCHEMA,
            output_path=output,
            project_root=tmp_path,
            sort_keys=("issue_index", "cell_id"),
        )
        output_schema = pq.read_schema(output)
        assert output_schema.metadata is not None
        assert (
            output_schema.metadata[b"seismoflux_content_hash"] == b"stage3_persisted_row_groups_v2"
        )
        assert (
            verify_stage3_parquet_artifact(
                project_root=tmp_path,
                entry=artifact.as_manifest_entry(output_schema),
                schema=output_schema,
            )
            == []
        )
        artifacts.append(artifact)

    assert artifacts[0].content_sha256 == artifacts[1].content_sha256


def test_persisted_content_hash_ignores_arrow_chunk_layout(tmp_path: Path) -> None:
    records = [
        {"issue_index": 0, "cell_id": "c0", "value": 1.0},
        {"issue_index": 0, "cell_id": "c1", "value": None},
        {"issue_index": 0, "cell_id": "c2", "value": 2.0},
    ]
    single_chunk = _table(records)
    multi_chunk = pa.concat_tables([_table(records[:1]), _table(records[1:])])
    assert single_chunk.equals(multi_chunk, check_metadata=True)
    assert multi_chunk["value"].num_chunks == 2

    artifacts = []
    for name, table in (("single", single_chunk), ("multi", multi_chunk)):
        output = tmp_path / f"{name}.parquet"
        artifact = write_parquet_row_groups_atomic(
            name=name,
            row_groups=[table],
            schema=SCHEMA,
            output_path=output,
            project_root=tmp_path,
            sort_keys=("issue_index", "cell_id"),
        )
        output_schema = pq.read_schema(output)
        assert (
            verify_stage3_parquet_artifact(
                project_root=tmp_path,
                entry=artifact.as_manifest_entry(output_schema),
                schema=output_schema,
            )
            == []
        )
        artifacts.append(artifact)

    assert artifacts[0].content_sha256 == artifacts[1].content_sha256


def test_stage3_streaming_parquet_rejects_unsorted_rows(tmp_path: Path) -> None:
    output = tmp_path / "features.parquet"
    group = _table(
        [
            {"issue_index": 0, "cell_id": "c1", "value": 1.0},
            {"issue_index": 0, "cell_id": "c0", "value": 2.0},
        ]
    )

    with pytest.raises(ValueError, match="not deterministically sorted"):
        write_parquet_row_groups_atomic(
            name="features",
            row_groups=[group],
            schema=SCHEMA,
            output_path=output,
            project_root=tmp_path,
            sort_keys=("issue_index", "cell_id"),
        )

    assert not output.exists()


def test_stage3_streaming_parquet_verifier_detects_mutation(tmp_path: Path) -> None:
    output = tmp_path / "features.parquet"
    artifact = write_parquet_row_groups_atomic(
        name="features",
        row_groups=[_table([{"issue_index": 0, "cell_id": "c0", "value": 1.0}])],
        schema=SCHEMA,
        output_path=output,
        project_root=tmp_path,
        sort_keys=("issue_index", "cell_id"),
    )
    schema = pq.read_schema(output)
    entry = artifact.as_manifest_entry(schema)
    output.write_bytes(output.read_bytes() + b"tamper")

    errors = verify_stage3_parquet_artifact(
        project_root=tmp_path,
        entry=entry,
        schema=schema,
    )

    assert errors == ["stage-3 file hash mismatch: features.parquet"]
