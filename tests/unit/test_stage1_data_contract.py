from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from seismoflux.config import load_yaml_mapping
from seismoflux.data.contracts import CONTRACTS, contract_document
from seismoflux.data.parquet import (
    schema_sha256,
    table_content_sha256,
    table_from_records,
    verify_parquet_artifact,
    write_parquet_atomic,
)
from seismoflux.data.pipeline import _validate_record_shape
from seismoflux.data.settings import IngestionSettings, ParquetSettings

INGESTION_CONFIG = Path("configs/ingestion.yaml")
EXPECTED_DATASETS = {
    "anomaly_entity_audit",
    "anomaly_observation",
    "anomaly_report_period",
    "basemap_feature",
    "earthquake_dedup_candidate",
    "earthquake_event",
    "earthquake_source_record",
    "fault_point_raw",
    "fault_segment",
    "fault_trace",
    "fault_trace_crosswalk_audit",
}


def _ingestion_mapping() -> dict[str, Any]:
    return deepcopy(load_yaml_mapping(INGESTION_CONFIG))


def _parquet_settings() -> ParquetSettings:
    return ParquetSettings.model_validate(_ingestion_mapping()["parquet"])


def test_frozen_contracts_have_unique_fields_and_valid_total_sort_keys() -> None:
    assert set(CONTRACTS) == EXPECTED_DATASETS

    for name, contract in CONTRACTS.items():
        assert len(contract.schema.names) == len(set(contract.schema.names)), name
        assert set(contract.sort_keys) <= set(contract.schema.names), name
        assert all(not contract.schema.field(key).nullable for key in contract.sort_keys), name

        document = contract_document(name, contract)
        assert document["dataset"] == name
        assert document["sort_keys"] == list(contract.sort_keys)
        assert [field["name"] for field in document["fields"]] == contract.schema.names


def test_pipeline_record_shape_rejects_missing_and_undocumented_fields() -> None:
    name = "anomaly_entity_audit"
    exact_shape = dict.fromkeys(CONTRACTS[name].schema.names)

    _validate_record_shape(name, [exact_shape])

    missing = dict(exact_shape)
    missing.pop("audit_id")
    with pytest.raises(ValueError, match=r"missing=\['audit_id'\], extra=\[\]"):
        _validate_record_shape(name, [missing])

    extra = {**exact_shape, "undocumented": "forbidden"}
    with pytest.raises(ValueError, match=r"missing=\[\], extra=\['undocumented'\]"):
        _validate_record_shape(name, [extra])


def test_parquet_list_fields_round_trip_with_stable_file_and_content_hashes(
    tmp_path: Path,
) -> None:
    contract = CONTRACTS["anomaly_entity_audit"]
    records = [
        {
            "audit_id": "audit-b",
            "audit_type": "identity_review",
            "status": "unreviewed",
            "source_file": "周报.xlsx",
            "source_sheet": "形变",
            "report_date": None,
            "anomaly_ids": ["anomaly-2", "anomaly-1"],
            "observation_ids": ["observation-2"],
            "reliability_flags": ["second", "first"],
            "previous_reported_end_time": None,
            "current_reported_end_time": None,
        },
        {
            "audit_id": "audit-a",
            "audit_type": "identity_review",
            "status": "unreviewed",
            "source_file": None,
            "source_sheet": None,
            "report_date": None,
            "anomaly_ids": [],
            "observation_ids": ["observation-1"],
            "reliability_flags": ["only"],
            "previous_reported_end_time": None,
            "current_reported_end_time": None,
        },
    ]
    table = table_from_records(records, contract.schema, contract.sort_keys)

    artifacts = [
        write_parquet_atomic(
            name="anomaly_entity_audit",
            table=table,
            output_path=tmp_path / "data" / f"audit-{index}.parquet",
            project_root=tmp_path,
            sort_keys=contract.sort_keys,
            settings=_parquet_settings(),
        )
        for index in range(2)
    ]
    persisted = pq.read_table(tmp_path / artifacts[0].path)

    assert persisted.schema.remove_metadata() == contract.schema
    assert persisted.column("audit_id").to_pylist() == ["audit-a", "audit-b"]
    assert persisted.column("anomaly_ids").to_pylist() == [[], ["anomaly-2", "anomaly-1"]]
    assert persisted.column("reliability_flags").to_pylist() == [
        ["only"],
        ["second", "first"],
    ]
    assert table_content_sha256(persisted) == artifacts[0].content_sha256
    assert schema_sha256(persisted.schema) == artifacts[0].schema_sha256
    assert artifacts[0].file_sha256 == artifacts[1].file_sha256
    assert artifacts[0].content_sha256 == artifacts[1].content_sha256
    assert artifacts[0].schema_sha256 == artifacts[1].schema_sha256

    entry = artifacts[0].as_catalog_entry(persisted.schema)
    assert verify_parquet_artifact(tmp_path, entry) == []

    entry["content_sha256"] = "0" * 64
    assert verify_parquet_artifact(tmp_path, entry) == [
        f"content hash mismatch: {artifacts[0].path}"
    ]


def test_ingestion_settings_match_the_frozen_stage1_contract() -> None:
    settings = IngestionSettings.model_validate(_ingestion_mapping())

    assert settings.contract_version == "0.1.0"
    assert settings.parquet.use_dictionary is False
    assert settings.time_assumptions.geology_snapshot_available_at.utcoffset() is not None
    assert settings.earthquake_deduplication.canonical_source_priority == (
        settings.source_roles.earthquake_m5_plus,
        settings.source_roles.earthquake_m3_plus,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["outputs"].__setitem__("processed_root", "../outside"),
            "project-relative",
        ),
        (
            lambda raw: raw["source_roles"].__setitem__(
                "earthquake_m5_plus", raw["source_roles"]["earthquake_m3_plus"]
            ),
            "source roles must be unique",
        ),
        (
            lambda raw: raw["earthquake_deduplication"]["auto_merge"].__setitem__(
                "require_unique_degree_both_sides", False
            ),
            "must require unique degree",
        ),
        (
            lambda raw: raw["earthquake_deduplication"]["manual_review"].__setitem__(
                "max_distance_km", 4
            ),
            "must contain auto-merge thresholds",
        ),
        (
            lambda raw: raw["earthquake_deduplication"].__setitem__(
                "canonical_source_priority",
                list(reversed(raw["earthquake_deduplication"]["canonical_source_priority"])),
            ),
            "canonical source priority",
        ),
        (
            lambda raw: raw["time_assumptions"].__setitem__(
                "geology_snapshot_available_at", "2026-07-13T00:00:00"
            ),
            "must include a timezone",
        ),
        (
            lambda raw: raw["parquet"].__setitem__("use_dictionary", True),
            "Input should be False",
        ),
    ],
)
def test_ingestion_settings_reject_contract_drift(mutate: Any, message: str) -> None:
    raw = _ingestion_mapping()
    mutate(raw)

    with pytest.raises(ValidationError, match=message):
        IngestionSettings.model_validate(raw)
