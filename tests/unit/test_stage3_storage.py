from __future__ import annotations

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
