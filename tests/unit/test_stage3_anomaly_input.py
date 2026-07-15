from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.config import sha256_file
from seismoflux.data.parquet import schema_sha256, table_content_sha256
from seismoflux.features.anomaly.config import load_anomaly_history_config
from seismoflux.features.anomaly.input import (
    RegisteredParquetInput,
    RegisteredStudyAreaInput,
    Stage3InputSpec,
    load_stage3_inputs,
    require_stage3_dataset_boundary,
    select_allowlisted_columns,
    stage3_input_spec_from_config,
    validate_stage3_input_tables,
)

UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.string(), nullable=False),
        pa.field("anomaly_id", pa.string(), nullable=False),
        pa.field("report_date", pa.date32(), nullable=False),
        pa.field("available_at", UTC_TIMESTAMP, nullable=False),
        pa.field("longitude", pa.float64()),
        pa.field("latitude", pa.float64()),
        pa.field("measurement", pa.string()),
        pa.field("report_state", pa.string(), nullable=False),
        pa.field("is_listed", pa.bool_(), nullable=False),
        pa.field("right_censored", pa.bool_(), nullable=False),
        pa.field("identity_complete", pa.bool_(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        # This registered stage-1 column is intentionally not authorized downstream.
        pa.field("forecast_efficacy", pa.string()),
    ]
)
REPORT_SCHEMA = pa.schema(
    [
        pa.field("report_id", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("report_year", pa.int64(), nullable=False),
        pa.field("report_period", pa.int64(), nullable=False),
        pa.field("report_date", pa.date32(), nullable=False),
        pa.field("available_at", UTC_TIMESTAMP, nullable=False),
    ]
)
OBSERVATION_ALLOWLIST = tuple(
    name for name in OBSERVATION_SCHEMA.names if name != "forecast_efficacy"
)
REPORT_ALLOWLIST = tuple(REPORT_SCHEMA.names)


@dataclass(frozen=True, slots=True)
class SyntheticInputs:
    spec: Stage3InputSpec
    observation_path: Path
    catalog_path: Path


def _records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report_rows = [
        {
            "report_id": "report-1",
            "source_file": "raw/report-1.xlsx",
            "report_year": 2024,
            "report_period": 1,
            "report_date": date(2024, 1, 1),
            "available_at": datetime(2024, 1, 2, tzinfo=UTC),
        },
        {
            "report_id": "report-2",
            "source_file": "raw/report-2.xlsx",
            "report_year": 2024,
            "report_period": 2,
            "report_date": date(2024, 1, 8),
            "available_at": datetime(2024, 1, 9, tzinfo=UTC),
        },
    ]
    observation_rows = [
        {
            "observation_id": "observation-1",
            "anomaly_id": "anomaly-a",
            "report_date": date(2024, 1, 1),
            "available_at": datetime(2024, 1, 2, tzinfo=UTC),
            "longitude": 105.0,
            "latitude": 32.0,
            "measurement": "groundwater level",
            "report_state": "新增",
            "is_listed": True,
            "right_censored": True,
            "identity_complete": True,
            "source_file": "raw/report-1.xlsx",
            "forecast_efficacy": "disabled free text",
        },
        {
            "observation_id": "observation-2",
            "anomaly_id": "anomaly-a",
            "report_date": date(2024, 1, 8),
            "available_at": datetime(2024, 1, 9, tzinfo=UTC),
            "longitude": 105.0,
            "latitude": 32.0,
            "measurement": "groundwater level",
            "report_state": "持续",
            "is_listed": True,
            "right_censored": True,
            "identity_complete": True,
            "source_file": "raw/report-2.xlsx",
            "forecast_efficacy": "disabled free text",
        },
        {
            "observation_id": "observation-3",
            "anomaly_id": "temporary-b",
            "report_date": date(2024, 1, 8),
            "available_at": datetime(2024, 1, 9, tzinfo=UTC),
            "longitude": None,
            "latitude": None,
            "measurement": None,
            "report_state": "新增",
            "is_listed": True,
            "right_censored": True,
            "identity_complete": False,
            "source_file": "raw/report-2.xlsx",
            "forecast_efficacy": "disabled free text",
        },
    ]
    return observation_rows, report_rows


def _field_documents(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _write_parquet_entry(
    path: Path,
    table: pa.Table,
    *,
    relative_path: str,
    sort_keys: tuple[str, ...],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    persisted = pq.read_table(path)
    return {
        "path": relative_path,
        "row_count": persisted.num_rows,
        "file_sha256": sha256_file(path),
        "content_sha256": table_content_sha256(persisted),
        "schema_sha256": schema_sha256(persisted.schema),
        "sort_keys": list(sort_keys),
        "fields": _field_documents(persisted.schema),
    }


def _registered_parquet(
    name: str,
    entry: dict[str, object],
    allowlist: tuple[str, ...],
) -> RegisteredParquetInput:
    return RegisteredParquetInput(
        dataset_name=name,
        path=str(entry["path"]),
        row_count=cast(int, entry["row_count"]),
        file_sha256=str(entry["file_sha256"]),
        content_sha256=str(entry["content_sha256"]),
        schema_sha256=str(entry["schema_sha256"]),
        source_column_allowlist=allowlist,
    )


def _synthetic_inputs(
    tmp_path: Path,
    *,
    unsorted_observations: bool = False,
    target_independent: bool = True,
    predictor_feature: bool = False,
    catalog_field_drift: bool = False,
) -> SyntheticInputs:
    observation_rows, report_rows = _records()
    if unsorted_observations:
        observation_rows[0], observation_rows[1] = observation_rows[1], observation_rows[0]
    observation_table = pa.Table.from_pylist(observation_rows, schema=OBSERVATION_SCHEMA)
    report_table = pa.Table.from_pylist(report_rows, schema=REPORT_SCHEMA)

    observation_relative = "data/observation.parquet"
    report_relative = "data/report-period.parquet"
    observation_path = tmp_path / observation_relative
    observation_entry = _write_parquet_entry(
        observation_path,
        observation_table,
        relative_path=observation_relative,
        sort_keys=("report_date", "source_file", "observation_id"),
    )
    report_entry = _write_parquet_entry(
        tmp_path / report_relative,
        report_table,
        relative_path=report_relative,
        sort_keys=("report_date", "source_file"),
    )
    if catalog_field_drift:
        fields = list(cast(list[dict[str, object]], observation_entry["fields"]))
        fields[0] = {**fields[0], "nullable": True}
        observation_entry["fields"] = fields

    study_relative = "data/study-area.geojson"
    study_path = tmp_path / study_relative
    study_document = {
        "type": "Feature",
        "properties": {
            "target_independent": target_independent,
            "predictor_feature": predictor_feature,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[100.0, 30.0], [101.0, 30.0], [101.0, 31.0], [100.0, 30.0]]],
        },
    }
    study_path.write_text(
        json.dumps(study_document, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    study_hash = sha256_file(study_path)

    snapshot_id = "synthetic-stage1"
    catalog = {
        "schema_version": 1,
        "contract_version": "0.1.0",
        "snapshot_id": snapshot_id,
        "standardized_data_committed_to_git": False,
        "datasets": {
            "anomaly_observation": observation_entry,
            "anomaly_report_period": report_entry,
            "earthquake_event": {"path": "forbidden-and-never-opened.parquet"},
        },
        "study_area": {
            "path": study_relative,
            "sha256": study_hash,
            "properties": {
                "target_independent": target_independent,
                "predictor_feature": predictor_feature,
            },
        },
    }
    catalog_relative = "data/catalog.json"
    catalog_path = tmp_path / catalog_relative
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    spec = Stage3InputSpec(
        data_catalog_path=catalog_relative,
        data_catalog_sha256=sha256_file(catalog_path),
        expected_stage1_snapshot_id=snapshot_id,
        observation=_registered_parquet(
            "anomaly_observation",
            observation_entry,
            OBSERVATION_ALLOWLIST,
        ),
        report_period=_registered_parquet(
            "anomaly_report_period",
            report_entry,
            REPORT_ALLOWLIST,
        ),
        study_area=RegisteredStudyAreaInput(path=study_relative, sha256=study_hash),
        expected_report_period_count=2,
        query_grid_cell_km=25.0,
    )
    return SyntheticInputs(
        spec=spec,
        observation_path=observation_path,
        catalog_path=catalog_path,
    )


def _valid_projected_tables() -> tuple[pa.Table, pa.Table]:
    observation_rows, report_rows = _records()
    observation_table = pa.Table.from_pylist(
        observation_rows,
        schema=OBSERVATION_SCHEMA,
    ).select(list(OBSERVATION_ALLOWLIST))
    report_table = pa.Table.from_pylist(report_rows, schema=REPORT_SCHEMA)
    return observation_table, report_table


def test_frozen_config_authorizes_only_two_anomaly_inputs() -> None:
    config = load_anomaly_history_config("configs/anomaly_history.yaml")
    spec = stage3_input_spec_from_config(config)

    assert spec.expected_report_period_count == 205
    assert (
        spec.observation.dataset_name,
        spec.report_period.dataset_name,
    ) == ("anomaly_observation", "anomaly_report_period")
    assert "forecast_efficacy" not in spec.observation.source_column_allowlist
    assert "anomaly_description" not in spec.observation.source_column_allowlist
    assert "unit" not in spec.observation.source_column_allowlist
    assert "trend_time" not in spec.observation.source_column_allowlist
    assert spec.study_area.path == "data/processed/china_mainland.geojson"
    assert spec.query_grid_cell_km == 25.0


@pytest.mark.parametrize(
    "names",
    [
        ("anomaly_report_period", "anomaly_observation"),
        ("anomaly_observation", "anomaly_report_period", "earthquake_event"),
        ("anomaly_observation", "background_local_support_manifest"),
        ("anomaly_observation", "completeness_mc"),
    ],
)
def test_scientific_boundary_rejects_every_non_anomaly_input(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="earthquake catalogs/labels, G1 support masks"):
        require_stage3_dataset_boundary(names)


def test_loader_verifies_full_identity_then_discards_disabled_columns(tmp_path: Path) -> None:
    synthetic = _synthetic_inputs(tmp_path)

    loaded = load_stage3_inputs(synthetic.spec, tmp_path)

    assert tuple(loaded.observation_table.column_names) == OBSERVATION_ALLOWLIST
    assert "forecast_efficacy" not in loaded.observation_table.column_names
    assert tuple(loaded.report_period_table.column_names) == REPORT_ALLOWLIST
    assert loaded.stage1_snapshot_id == "synthetic-stage1"
    assert loaded.study_area_document["type"] == "Feature"
    assert not (tmp_path / "forbidden-and-never-opened.parquet").exists()


def test_downstream_column_selection_rechecks_frozen_allowlist(tmp_path: Path) -> None:
    synthetic = _synthetic_inputs(tmp_path)
    loaded = load_stage3_inputs(synthetic.spec, tmp_path)

    selected = select_allowlisted_columns(
        loaded.observation_table,
        ("observation_id", "available_at"),
        identity=synthetic.spec.observation,
    )
    assert selected.column_names == ["observation_id", "available_at"]
    with pytest.raises(ValueError, match="not authorized"):
        select_allowlisted_columns(
            loaded.observation_table,
            ("forecast_efficacy",),
            identity=synthetic.spec.observation,
        )


def test_loader_rejects_file_tampering_before_arrow_load(tmp_path: Path) -> None:
    synthetic = _synthetic_inputs(tmp_path)
    with synthetic.observation_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="file hash mismatch"):
        load_stage3_inputs(synthetic.spec, tmp_path)


def test_loader_rejects_catalog_hash_and_snapshot_drift(tmp_path: Path) -> None:
    synthetic = _synthetic_inputs(tmp_path)

    with pytest.raises(ValueError, match="data catalog hash mismatch"):
        load_stage3_inputs(
            replace(synthetic.spec, data_catalog_sha256="0" * 64),
            tmp_path,
        )
    with pytest.raises(ValueError, match="snapshot identity mismatch"):
        load_stage3_inputs(
            replace(synthetic.spec, expected_stage1_snapshot_id="different-snapshot"),
            tmp_path,
        )


def test_loader_rejects_catalog_schema_document_drift(tmp_path: Path) -> None:
    synthetic = _synthetic_inputs(tmp_path, catalog_field_drift=True)

    with pytest.raises(ValueError, match="catalog field schema mismatch"):
        load_stage3_inputs(synthetic.spec, tmp_path)


def test_loader_rejects_unsorted_registered_rows(tmp_path: Path) -> None:
    synthetic = _synthetic_inputs(tmp_path, unsorted_observations=True)

    with pytest.raises(ValueError, match="row order mismatch"):
        load_stage3_inputs(synthetic.spec, tmp_path)


@pytest.mark.parametrize(
    ("target_independent", "predictor_feature", "message"),
    [
        (False, False, "target-independent"),
        (True, True, "predictor feature"),
    ],
)
def test_loader_rejects_target_derived_or_predictor_study_area(
    tmp_path: Path,
    target_independent: bool,
    predictor_feature: bool,
    message: str,
) -> None:
    synthetic = _synthetic_inputs(
        tmp_path,
        target_independent=target_independent,
        predictor_feature=predictor_feature,
    )

    with pytest.raises(ValueError, match=message):
        load_stage3_inputs(synthetic.spec, tmp_path)


def test_semantic_validator_accepts_causal_complete_synthetic_history() -> None:
    observation_table, report_table = _valid_projected_tables()

    validate_stage3_input_tables(
        observation_table,
        report_table,
        expected_report_period_count=2,
    )


def test_semantic_validator_rejects_period_count_and_order_drift() -> None:
    observation_table, report_table = _valid_projected_tables()

    with pytest.raises(ValueError, match="report-period count mismatch"):
        validate_stage3_input_tables(
            observation_table,
            report_table,
            expected_report_period_count=3,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_stage3_input_tables(
            observation_table,
            pa.Table.from_pylist(
                list(reversed(report_table.to_pylist())),
                schema=report_table.schema,
            ),
            expected_report_period_count=2,
        )


def test_semantic_validator_rejects_duplicate_observation_identity() -> None:
    observation_table, report_table = _valid_projected_tables()
    rows = observation_table.to_pylist()
    rows[1]["observation_id"] = rows[0]["observation_id"]

    with pytest.raises(ValueError, match="observation_id identities must be unique"):
        validate_stage3_input_tables(
            pa.Table.from_pylist(rows, schema=observation_table.schema),
            report_table,
            expected_report_period_count=2,
        )


def test_semantic_validator_rejects_linked_incomplete_entities() -> None:
    observation_table, report_table = _valid_projected_tables()
    rows = observation_table.to_pylist()
    for index in (0, 2):
        rows[index]["anomaly_id"] = "shared-temporary-id"
        rows[index]["identity_complete"] = False
        rows[index]["longitude"] = None
        rows[index]["latitude"] = None
        rows[index]["measurement"] = None

    with pytest.raises(ValueError, match="incomplete anomaly identities may not link"):
        validate_stage3_input_tables(
            pa.Table.from_pylist(rows, schema=observation_table.schema),
            report_table,
            expected_report_period_count=2,
        )


def test_semantic_validator_rejects_future_report_use() -> None:
    observation_table, report_table = _valid_projected_tables()
    rows = observation_table.to_pylist()
    rows[0]["available_at"] = datetime(2024, 1, 1, tzinfo=UTC)

    with pytest.raises(ValueError, match="precedes its report-period availability"):
        validate_stage3_input_tables(
            pa.Table.from_pylist(rows, schema=observation_table.schema),
            report_table,
            expected_report_period_count=2,
        )


@pytest.mark.parametrize(
    ("longitude", "latitude", "message"),
    [
        (105.0, None, "jointly present"),
        (181.0, 32.0, "outside WGS84 bounds"),
        (105.0, 91.0, "outside WGS84 bounds"),
    ],
)
def test_semantic_validator_rejects_invalid_coordinates(
    longitude: float | None,
    latitude: float | None,
    message: str,
) -> None:
    observation_table, report_table = _valid_projected_tables()
    rows = observation_table.to_pylist()
    rows[0]["longitude"] = longitude
    rows[0]["latitude"] = latitude

    with pytest.raises(ValueError, match=message):
        validate_stage3_input_tables(
            pa.Table.from_pylist(rows, schema=observation_table.schema),
            report_table,
            expected_report_period_count=2,
        )
