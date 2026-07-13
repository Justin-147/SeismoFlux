"""End-to-end stage-1 ingestion and validation pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import shape

from seismoflux.config import (
    SeismoFluxConfig,
    project_root_for,
    resolve_project_path,
    sha256_file,
)
from seismoflux.data.anomaly import parse_anomaly_files
from seismoflux.data.catalog import build_data_catalog, validate_data_catalog
from seismoflux.data.common import (
    SourceFile,
    canonical_json_bytes,
    sha256_bytes,
    write_json_atomic,
)
from seismoflux.data.contracts import CONTRACTS, contract_document, write_contract_documents
from seismoflux.data.earthquake import DedupThresholds, parse_earthquake_catalogs
from seismoflux.data.geology import parse_geology_sources
from seismoflux.data.parquet import DatasetArtifact, table_from_records, write_parquet_atomic
from seismoflux.data.quality import (
    build_quality_report,
    count_leakage_violations,
    write_quality_outputs,
)
from seismoflux.data.settings import IngestionSettings
from seismoflux.data.source import load_inventory_sources, require_single_source
from seismoflux.inventory import DataSourcesConfig, build_inventory, write_inventory


def _progress(message: str) -> None:
    sys.stderr.write(f"seismoflux ingest: {message}\n")
    sys.stderr.flush()


def _record_dict(record: object) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    raise TypeError(f"unsupported standardized record: {type(record).__name__}")


def _records(items: Sequence[object]) -> list[dict[str, Any]]:
    return [_record_dict(item) for item in items]


def _validate_record_shape(name: str, records: Sequence[Mapping[str, Any]]) -> None:
    expected = set(CONTRACTS[name].schema.names)
    for index, record in enumerate(records):
        actual = set(record)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{name} record {index} violates the contract; missing={missing}, extra={extra}"
            )


def _flatten_sources(
    source_map: Mapping[str, tuple[SourceFile, ...]], source_ids: Sequence[str]
) -> dict[str, SourceFile]:
    flattened: dict[str, SourceFile] = {}
    for source_id in source_ids:
        for source in source_map[source_id]:
            if source.relative_path in flattened:
                raise ValueError(f"duplicate raw relative path: {source.relative_path}")
            flattened[source.relative_path] = source
    return flattened


def _recheck_inventory(
    source_config: DataSourcesConfig, inventory_path: Path, project_root: Path
) -> None:
    with tempfile.TemporaryDirectory(prefix=".stage1-inventory-", dir=project_root) as directory:
        check_path = Path(directory) / "source_inventory.csv"
        write_inventory(build_inventory(source_config), check_path)
        if sha256_file(check_path) != sha256_file(inventory_path):
            raise ValueError("raw source inventory changed during stage-1 ingestion")


def _validate_study_area(
    study_area: Mapping[str, Any], settings: IngestionSettings, equal_area_crs: str
) -> None:
    properties = study_area.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("study area has no properties")
    expected = settings.study_area
    checks = {
        "source_sha256": expected.expected_source_sha256,
        "source_segment_number": expected.expected_segment_number_1based,
        "source_coordinate_count": expected.expected_coordinate_count,
        "selection_rule": expected.selection_rule,
        "island_policy": expected.island_policy,
        "equal_area_crs": equal_area_crs,
        "target_independent": True,
        "predictor_feature": False,
    }
    for key, value in checks.items():
        if properties.get(key) != value:
            raise ValueError(f"study-area contract mismatch for {key}: {properties.get(key)!r}")
    measured_area = float(properties["geodesic_area_km2"])
    if abs(measured_area - expected.expected_geodesic_area_km2) > expected.area_tolerance_km2:
        raise ValueError("study-area geodesic area is outside the frozen tolerance")


def ingest_stage1(
    *,
    config_path: Path,
    config: SeismoFluxConfig,
    ingestion_path: Path,
    settings: IngestionSettings,
    source_config: DataSourcesConfig,
) -> dict[str, Any]:
    """Parse all raw inputs, write deterministic artifacts, and return manifest details."""

    project_root = project_root_for(config_path).resolve()
    inventory_path = resolve_project_path(config_path, settings.source_inventory)
    source_inventory_sha256 = sha256_file(inventory_path)
    ingestion_config_sha256 = sha256_file(ingestion_path)
    source_map = load_inventory_sources(source_config, inventory_path)
    configured_role_ids = set(settings.source_roles.model_dump().values())
    if configured_role_ids != set(source_map):
        raise ValueError("ingestion source roles do not exactly cover the source inventory")
    if config.study_area.polygon != settings.outputs.study_area_geojson:
        raise ValueError("base and ingestion configurations disagree on the study-area path")

    _progress("解析断层、长期危险性和底图")
    geology_source_ids = (
        settings.source_roles.fault_coordinates,
        settings.source_roles.fault_attributes,
        settings.source_roles.long_term_hazard,
        settings.source_roles.basemap_and_fault_trace,
    )
    geology = parse_geology_sources(
        _flatten_sources(source_map, geology_source_ids),
        config.study_area.equal_area_crs,
        settings.time_assumptions.geology_snapshot_available_at,
    )
    _validate_study_area(geology.study_area, settings, config.study_area.equal_area_crs)
    study_area_geometry = shape(geology.study_area["geometry"])

    _progress("解析 205 份异常周报")
    anomaly = parse_anomaly_files(list(source_map[settings.source_roles.anomaly]))

    _progress("解析并去重两个地震目录")
    auto_settings = settings.earthquake_deduplication.auto_merge
    review_settings = settings.earthquake_deduplication.manual_review
    earthquakes = parse_earthquake_catalogs(
        require_single_source(source_map, settings.source_roles.earthquake_m3_plus),
        require_single_source(source_map, settings.source_roles.earthquake_m5_plus),
        study_area_geometry,
        auto=DedupThresholds(
            auto_settings.max_time_delta_seconds,
            auto_settings.max_distance_km,
            auto_settings.max_magnitude_delta,
        ),
        review=DedupThresholds(
            review_settings.max_time_delta_seconds,
            review_settings.max_distance_km,
            review_settings.max_magnitude_delta,
        ),
    )

    datasets: dict[str, list[dict[str, Any]]] = {
        "anomaly_observation": _records(anomaly.observations),
        "anomaly_entity_audit": _records(anomaly.entity_audit),
        "anomaly_report_period": _records(anomaly.report_periods),
        "earthquake_event": _records(earthquakes.events),
        "earthquake_source_record": _records(earthquakes.source_records),
        "earthquake_dedup_candidate": _records(earthquakes.dedup_candidates),
        "fault_point_raw": _records(geology.fault_points),
        "fault_segment": _records(geology.fault_segments),
        "fault_trace": _records(geology.fault_traces),
        "basemap_feature": _records(geology.basemap_features),
        "fault_trace_crosswalk_audit": _records(geology.crosswalk_audit),
    }
    if set(datasets) != set(CONTRACTS):
        raise ValueError("pipeline datasets and frozen contracts differ")

    dependency_versions = {
        package: version(package)
        for package in ("openpyxl", "pandas", "pyarrow", "pyproj", "shapely", "xlrd")
    }
    contract_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                name: contract_document(name, contract)
                for name, contract in sorted(CONTRACTS.items())
            }
        )
    )
    snapshot_id = sha256_bytes(
        canonical_json_bytes(
            {
                "contract_fingerprint": contract_fingerprint,
                "dependency_versions": dependency_versions,
                "ingestion_config_sha256": ingestion_config_sha256,
                "source_inventory_sha256": source_inventory_sha256,
            }
        )
    )[:16]
    processed_root = resolve_project_path(config_path, settings.outputs.processed_root)
    snapshot_root = processed_root / snapshot_id
    tables: dict[str, pa.Table] = {}
    for name, records in sorted(datasets.items()):
        _validate_record_shape(name, records)
        contract = CONTRACTS[name]
        tables[name] = table_from_records(records, contract.schema, contract.sort_keys)

    geology_records = [
        *datasets["fault_point_raw"],
        *datasets["fault_segment"],
        *datasets["fault_trace"],
        *datasets["fault_trace_crosswalk_audit"],
    ]
    earthquake_records = [
        *datasets["earthquake_event"],
        *datasets["earthquake_source_record"],
    ]
    leakage_violations, leakage_checks = count_leakage_violations(
        datasets["anomaly_observation"],
        geology_records,
        earthquake_records,
        datasets["basemap_feature"],
    )
    quality_report = build_quality_report(
        source_inventory_sha256=source_inventory_sha256,
        anomaly_quality=anomaly.quality,
        earthquake_quality=earthquakes.quality,
        geology_quality=geology.quality,
        leakage_checks=leakage_checks,
        leakage_violations=leakage_violations,
    )
    if leakage_violations:
        raise ValueError(f"stage-1 leakage checks found {leakage_violations} violations")

    _progress("写入前复核原始清单在解析期间未变化")
    _recheck_inventory(source_config, inventory_path, project_root)

    artifact_entries: dict[str, tuple[DatasetArtifact, pa.Schema]] = {}
    _progress("写入固定排序的 Parquet")
    for name, table in sorted(tables.items()):
        contract = CONTRACTS[name]
        artifact = write_parquet_atomic(
            name=name,
            table=table,
            output_path=snapshot_root / f"{name}.parquet",
            project_root=project_root,
            sort_keys=contract.sort_keys,
            settings=settings.parquet,
        )
        artifact_entries[name] = (artifact, table.schema)

    study_area_path = resolve_project_path(config_path, settings.outputs.study_area_geojson)
    quality_json_path = resolve_project_path(config_path, settings.outputs.quality_json)
    quality_markdown_path = resolve_project_path(config_path, settings.outputs.quality_markdown)
    contracts_root = resolve_project_path(config_path, settings.outputs.contracts_root)
    write_json_atomic(study_area_path, geology.study_area)
    write_quality_outputs(quality_report, quality_json_path, quality_markdown_path)
    contract_names = write_contract_documents(contracts_root)

    contract_entries = {
        name: (
            (contracts_root / filename).relative_to(project_root).as_posix(),
            sha256_file(contracts_root / filename),
        )
        for name, filename in contract_names.items()
    }
    catalog = build_data_catalog(
        source_inventory_path=inventory_path.relative_to(project_root).as_posix(),
        source_inventory_sha256=source_inventory_sha256,
        ingestion_config_path=ingestion_path.relative_to(project_root).as_posix(),
        ingestion_config_sha256=ingestion_config_sha256,
        snapshot_id=snapshot_id,
        datasets=artifact_entries,
        contracts=contract_entries,
        study_area_path=study_area_path.relative_to(project_root).as_posix(),
        study_area_sha256=sha256_file(study_area_path),
        study_area_properties=geology.study_area["properties"],
        quality_json_path=quality_json_path.relative_to(project_root).as_posix(),
        quality_json_sha256=sha256_file(quality_json_path),
        quality_markdown_path=quality_markdown_path.relative_to(project_root).as_posix(),
        quality_markdown_sha256=sha256_file(quality_markdown_path),
        dependency_versions=dependency_versions,
    )
    catalog_path = resolve_project_path(config_path, settings.outputs.data_catalog)
    catalog_errors = validate_data_catalog(catalog, project_root)
    if catalog_errors:
        raise ValueError("data catalog validation failed: " + "; ".join(catalog_errors))
    write_json_atomic(catalog_path, catalog)
    return {
        "snapshot_id": snapshot_id,
        "data_catalog": catalog_path.relative_to(project_root).as_posix(),
        "data_catalog_sha256": sha256_file(catalog_path),
        "quality_status": quality_report["status"],
        "leakage_violations": leakage_violations,
        "dataset_rows": {
            name: artifact.row_count for name, (artifact, _) in sorted(artifact_entries.items())
        },
        "dataset_file_sha256": {
            name: artifact.file_sha256 for name, (artifact, _) in sorted(artifact_entries.items())
        },
    }


def validate_stage1_data(*, config_path: Path, settings: IngestionSettings) -> dict[str, Any]:
    project_root = project_root_for(config_path).resolve()
    catalog_path = resolve_project_path(config_path, settings.outputs.data_catalog)
    try:
        catalog_raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unable to load the stage-1 data catalog") from exc
    if not isinstance(catalog_raw, dict):
        raise ValueError("stage-1 data catalog must be a JSON object")
    errors = validate_data_catalog(catalog_raw, project_root)
    quality_path = resolve_project_path(config_path, settings.outputs.quality_json)
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unable to load the stage-1 quality report") from exc
    if not isinstance(quality, dict):
        errors.append("quality report must be a JSON object")
    else:
        leakage = quality.get("leakage")
        if not isinstance(leakage, dict) or leakage.get("violations") != 0:
            errors.append("quality report does not prove zero leakage violations")
        if quality.get("status") != "pass_with_documented_warnings":
            errors.append("quality report status is not passing")
    if not errors:
        dataset_entries = cast(dict[str, dict[str, Any]], catalog_raw["datasets"])

        def records(name: str) -> list[dict[str, Any]]:
            dataset_path = project_root / dataset_entries[name]["path"]
            return cast(list[dict[str, Any]], pq.read_table(dataset_path).to_pylist())

        recomputed_violations, recomputed_checks = count_leakage_violations(
            records("anomaly_observation"),
            [
                *records("fault_point_raw"),
                *records("fault_segment"),
                *records("fault_trace"),
                *records("fault_trace_crosswalk_audit"),
            ],
            [*records("earthquake_event"), *records("earthquake_source_record")],
            records("basemap_feature"),
        )
        if recomputed_violations:
            errors.append(f"recomputed leakage checks found {recomputed_violations} violations")
        if isinstance(quality, dict):
            reported_leakage = quality.get("leakage")
            if not isinstance(reported_leakage, dict) or reported_leakage.get("checks") != dict(
                sorted(recomputed_checks.items())
            ):
                errors.append("quality report leakage checks do not match the Parquet data")
    if errors:
        raise ValueError("stage-1 data validation failed: " + "; ".join(errors))
    return {
        "status": "passed",
        "data_catalog": catalog_path.relative_to(project_root).as_posix(),
        "data_catalog_sha256": sha256_file(catalog_path),
        "snapshot_id": catalog_raw.get("snapshot_id"),
        "dataset_rows": {
            name: entry["row_count"] for name, entry in catalog_raw["datasets"].items()
        },
        "leakage_violations": 0,
    }
