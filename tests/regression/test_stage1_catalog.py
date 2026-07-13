from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from seismoflux.config import load_yaml_mapping, sha256_file
from seismoflux.data.catalog import build_data_catalog, validate_data_catalog
from seismoflux.data.common import write_json_atomic
from seismoflux.data.contracts import CONTRACTS, contract_document
from seismoflux.data.parquet import DatasetArtifact, table_from_records, write_parquet_atomic
from seismoflux.data.pipeline import validate_stage1_data
from seismoflux.data.quality import count_leakage_violations, write_text_atomic
from seismoflux.data.settings import IngestionSettings

INGESTION_CONFIG = Path("configs/ingestion.yaml")


@dataclass(frozen=True, slots=True)
class SyntheticStage1:
    root: Path
    config_path: Path
    ingestion_path: Path
    catalog_path: Path
    quality_path: Path
    dataset_path: Path
    settings: IngestionSettings
    catalog: dict[str, Any]


def _build_synthetic_stage1(root: Path) -> SyntheticStage1:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "base.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    ingestion_path = config_dir / "ingestion.yaml"
    ingestion_path.write_bytes(INGESTION_CONFIG.read_bytes())
    settings = IngestionSettings.model_validate(load_yaml_mapping(ingestion_path))

    inventory_path = root / settings.source_inventory
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text("synthetic inventory\n", encoding="utf-8")

    datasets: dict[str, tuple[DatasetArtifact, Any]] = {}
    contract_entries: dict[str, tuple[str, str]] = {}
    dataset_path: Path | None = None
    for name, contract in sorted(CONTRACTS.items()):
        records: list[dict[str, Any]] = []
        if name == "anomaly_entity_audit":
            records = [
                {
                    "audit_id": "audit-1",
                    "audit_type": "identity_review",
                    "status": "unreviewed",
                    "source_file": None,
                    "source_sheet": None,
                    "report_date": None,
                    "anomaly_ids": ["anomaly-1"],
                    "observation_ids": ["observation-1"],
                    "reliability_flags": ["synthetic"],
                    "previous_reported_end_time": None,
                    "current_reported_end_time": None,
                }
            ]
        table = table_from_records(records, contract.schema, contract.sort_keys)
        path = root / settings.outputs.processed_root / "synthetic" / f"{name}.parquet"
        artifact = write_parquet_atomic(
            name=name,
            table=table,
            output_path=path,
            project_root=root,
            sort_keys=contract.sort_keys,
            settings=settings.parquet,
        )
        datasets[name] = (artifact, table.schema)
        contract_path = root / settings.outputs.contracts_root / f"{name}.json"
        write_json_atomic(contract_path, contract_document(name, contract))
        contract_entries[name] = (
            contract_path.relative_to(root).as_posix(),
            sha256_file(contract_path),
        )
        if name == "anomaly_entity_audit":
            dataset_path = path
    assert dataset_path is not None
    study_area_path = root / settings.outputs.study_area_geojson
    write_json_atomic(study_area_path, {"type": "FeatureCollection", "features": []})
    quality_path = root / settings.outputs.quality_json
    _, empty_leakage_checks = count_leakage_violations([], [], [], [])
    write_json_atomic(
        quality_path,
        {
            "schema_version": 1,
            "contract_version": "0.1.0",
            "status": "pass_with_documented_warnings",
            "leakage": {"violations": 0, "checks": empty_leakage_checks},
        },
    )
    quality_markdown_path = root / settings.outputs.quality_markdown
    write_text_atomic(quality_markdown_path, "# Synthetic quality report\n")

    catalog = build_data_catalog(
        source_inventory_path=inventory_path.relative_to(root).as_posix(),
        source_inventory_sha256=sha256_file(inventory_path),
        ingestion_config_path=ingestion_path.relative_to(root).as_posix(),
        ingestion_config_sha256=sha256_file(ingestion_path),
        snapshot_id="synthetic-snapshot",
        datasets=datasets,
        contracts=contract_entries,
        study_area_path=study_area_path.relative_to(root).as_posix(),
        study_area_sha256=sha256_file(study_area_path),
        study_area_properties={"target_independent": True},
        quality_json_path=quality_path.relative_to(root).as_posix(),
        quality_json_sha256=sha256_file(quality_path),
        quality_markdown_path=quality_markdown_path.relative_to(root).as_posix(),
        quality_markdown_sha256=sha256_file(quality_markdown_path),
        dependency_versions={"pyarrow": "synthetic"},
    )
    catalog_path = root / settings.outputs.data_catalog
    write_json_atomic(catalog_path, catalog)
    return SyntheticStage1(
        root=root,
        config_path=config_path,
        ingestion_path=ingestion_path,
        catalog_path=catalog_path,
        quality_path=quality_path,
        dataset_path=dataset_path,
        settings=settings,
        catalog=catalog,
    )


def test_pristine_synthetic_catalog_and_stage1_validation_pass(tmp_path: Path) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)

    assert validate_data_catalog(stage1.catalog, stage1.root) == []
    result = validate_stage1_data(
        config_path=stage1.config_path,
        settings=stage1.settings,
    )

    assert result["status"] == "passed"
    assert result["snapshot_id"] == "synthetic-snapshot"
    assert result["dataset_rows"] == {
        name: 1 if name == "anomaly_entity_audit" else 0 for name in sorted(CONTRACTS)
    }
    assert result["leakage_violations"] == 0


def test_catalog_detects_ingestion_configuration_tampering(tmp_path: Path) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    stage1.ingestion_path.write_text("tampered: true\n", encoding="utf-8")

    assert validate_data_catalog(stage1.catalog, stage1.root) == [
        "hash mismatch for ingestion config: configs/ingestion.yaml"
    ]


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("datasets", "data catalog dataset names do not match the frozen contracts"),
        ("contracts", "data catalog contract names do not match the frozen contracts"),
    ],
)
def test_catalog_rejects_an_incomplete_frozen_contract_set(
    tmp_path: Path, section: str, expected: str
) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    cast(dict[str, Any], stage1.catalog[section]).pop("earthquake_event")

    errors = validate_data_catalog(stage1.catalog, stage1.root)

    assert expected in errors


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        ("row_count", 2, "row count mismatch"),
        ("content_sha256", "0" * 64, "content hash mismatch"),
        ("schema_sha256", "0" * 64, "schema hash mismatch"),
    ],
)
def test_catalog_detects_tampered_dataset_claims(
    tmp_path: Path, field: str, replacement: object, expected: str
) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    entry = cast(dict[str, Any], stage1.catalog["datasets"]["anomaly_entity_audit"])
    entry[field] = replacement

    errors = validate_data_catalog(stage1.catalog, stage1.root)

    assert any(error.startswith(expected) for error in errors)


def test_catalog_rejects_dataset_path_escape(tmp_path: Path) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    entry = cast(dict[str, Any], stage1.catalog["datasets"]["anomaly_entity_audit"])
    entry["path"] = "../outside.parquet"

    assert validate_data_catalog(stage1.catalog, stage1.root) == [
        "missing or unsafe dataset path: ../outside.parquet"
    ]


def test_catalog_detects_dataset_file_tampering_before_parquet_read(tmp_path: Path) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    with stage1.dataset_path.open("ab") as handle:
        handle.write(b"tampered")

    assert validate_data_catalog(stage1.catalog, stage1.root) == [
        f"file hash mismatch: {stage1.dataset_path.relative_to(stage1.root).as_posix()}"
    ]


def test_validate_stage1_data_rejects_missing_quality_report(tmp_path: Path) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    stage1.quality_path.unlink()

    with pytest.raises(ValueError, match="unable to load the stage-1 quality report"):
        validate_stage1_data(config_path=stage1.config_path, settings=stage1.settings)


def test_validate_stage1_data_rejects_nonzero_leakage_and_failing_status(tmp_path: Path) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    quality = cast(
        dict[str, Any],
        json.loads(stage1.quality_path.read_text(encoding="utf-8")),
    )
    quality["status"] = "fail"
    quality["leakage"] = {"violations": 1, "checks": {"synthetic_violation": 1}}
    write_json_atomic(stage1.quality_path, quality)

    with pytest.raises(ValueError) as error:
        validate_stage1_data(config_path=stage1.config_path, settings=stage1.settings)

    message = str(error.value)
    assert "hash mismatch for quality report json" in message
    assert "quality report does not prove zero leakage violations" in message
    assert "quality report status is not passing" in message


def test_validate_stage1_data_recomputes_leakage_from_parquet(tmp_path: Path) -> None:
    stage1 = _build_synthetic_stage1(tmp_path)
    contract = CONTRACTS["basemap_feature"]
    table = table_from_records(
        [
            {
                "basemap_feature_id": "basemap-1",
                "role": "visualization_only",
                "source_segment_number": 1,
                "geometry_wkb": b"synthetic",
                "raw_point_count": 2,
                "geometry_point_count": 2,
                "is_closed": False,
                "geometry_hash": "geometry-1",
                "duplicate_group_id": None,
                "duplicate_geometry": False,
                "quality_flags": [],
                "source_file": "synthetic.gmt",
                "delimiter_source_line": 1,
                "source_comments": [],
                "source_available_at": datetime(2026, 7, 12, 16, tzinfo=UTC),
                "model_feature_eligible": True,
            }
        ],
        contract.schema,
        contract.sort_keys,
    )
    dataset_entry = cast(dict[str, Any], stage1.catalog["datasets"]["basemap_feature"])
    artifact = write_parquet_atomic(
        name="basemap_feature",
        table=table,
        output_path=stage1.root / dataset_entry["path"],
        project_root=stage1.root,
        sort_keys=contract.sort_keys,
        settings=stage1.settings.parquet,
    )
    stage1.catalog["datasets"]["basemap_feature"] = artifact.as_catalog_entry(table.schema)
    write_json_atomic(stage1.catalog_path, stage1.catalog)

    with pytest.raises(ValueError, match="recomputed leakage checks found 1 violations"):
        validate_stage1_data(config_path=stage1.config_path, settings=stage1.settings)
