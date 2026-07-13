"""Deterministic data catalog creation and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seismoflux.config import sha256_file
from seismoflux.data.contracts import CONTRACTS, contract_document
from seismoflux.data.parquet import DatasetArtifact, verify_parquet_artifact


def build_data_catalog(
    *,
    source_inventory_path: str,
    source_inventory_sha256: str,
    ingestion_config_path: str,
    ingestion_config_sha256: str,
    snapshot_id: str,
    datasets: Mapping[str, tuple[DatasetArtifact, Any]],
    contracts: Mapping[str, tuple[str, str]],
    study_area_path: str,
    study_area_sha256: str,
    study_area_properties: Mapping[str, Any],
    quality_json_path: str,
    quality_json_sha256: str,
    quality_markdown_path: str,
    quality_markdown_sha256: str,
    dependency_versions: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_version": "0.1.0",
        "snapshot_id": snapshot_id,
        "source_inventory": {
            "path": source_inventory_path,
            "sha256": source_inventory_sha256,
        },
        "ingestion_config": {
            "path": ingestion_config_path,
            "sha256": ingestion_config_sha256,
        },
        "license_status": "unknown_no_redistribution",
        "standardized_data_committed_to_git": False,
        "time_semantics": {
            "anomaly_available_at": "max(file_report_date,row_report_date)+1d@00:00_fixed_UTC+08",
            "anomaly_occurrence_time_precision": "source_date_only_as_00:00_fixed_UTC+08_flagged",
            "earthquake_origin_timezone": "fixed_UTC+08_assumption",
            "earthquake_publication_time": "origin_time_optimistic_assumption_flagged",
            "earthquake_available_at": "origin_time_utc",
            "forecast_history_rule": "available_at_or_origin_time_lte_issue_time",
            "forecast_target_rule": "(T,T+h]",
        },
        "study_area": {
            "path": study_area_path,
            "sha256": study_area_sha256,
            "properties": dict(sorted(study_area_properties.items())),
        },
        "datasets": {
            name: artifact.as_catalog_entry(schema)
            for name, (artifact, schema) in sorted(datasets.items())
        },
        "contracts": {
            name: {"path": path, "sha256": digest}
            for name, (path, digest) in sorted(contracts.items())
        },
        "quality_report": {
            "json": {"path": quality_json_path, "sha256": quality_json_sha256},
            "markdown": {
                "path": quality_markdown_path,
                "sha256": quality_markdown_sha256,
            },
        },
        "dependency_versions": dict(sorted(dependency_versions.items())),
        "determinism": {
            "source_order": "inventory_relative_path_casefold_then_path",
            "row_order": "dataset_contract_sort_keys",
            "parquet_writer_options": "configs/ingestion.yaml#parquet",
            "volatile_generation_timestamp_forbidden": True,
        },
    }


def _verify_file(project_root: Path, entry: Mapping[str, Any], label: str) -> list[str]:
    path = (project_root / str(entry["path"])).resolve()
    if not path.is_relative_to(project_root.resolve()) or not path.is_file():
        return [f"missing or unsafe {label}: {entry['path']}"]
    if sha256_file(path) != entry["sha256"]:
        return [f"hash mismatch for {label}: {entry['path']}"]
    return []


def validate_data_catalog(catalog: Mapping[str, Any], project_root: Path) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema_version") != 1 or catalog.get("contract_version") != "0.1.0":
        errors.append("unsupported data catalog version")
    source_inventory = catalog.get("source_inventory")
    if isinstance(source_inventory, Mapping):
        errors.extend(_verify_file(project_root, source_inventory, "source inventory"))
    else:
        errors.append("data catalog has no source inventory")

    ingestion_config = catalog.get("ingestion_config")
    if isinstance(ingestion_config, Mapping):
        errors.extend(_verify_file(project_root, ingestion_config, "ingestion config"))
    else:
        errors.append("data catalog has no ingestion config")

    datasets = catalog.get("datasets")
    if not isinstance(datasets, Mapping):
        errors.append("data catalog has no datasets")
    else:
        if set(datasets) != set(CONTRACTS):
            errors.append("data catalog dataset names do not match the frozen contracts")
        for name, entry in datasets.items():
            if not isinstance(entry, dict):
                errors.append(f"invalid dataset entry: {name}")
                continue
            errors.extend(verify_parquet_artifact(project_root, entry))

    contracts = catalog.get("contracts")
    if not isinstance(contracts, Mapping):
        errors.append("data catalog has no contracts")
    else:
        if set(contracts) != set(CONTRACTS):
            errors.append("data catalog contract names do not match the frozen contracts")
        for name, entry in contracts.items():
            if isinstance(entry, Mapping):
                errors.extend(_verify_file(project_root, entry, f"contract {name}"))
                if name not in CONTRACTS:
                    continue
                path = (project_root / str(entry["path"])).resolve()
                if not path.is_relative_to(project_root.resolve()) or not path.is_file():
                    continue
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    errors.append(f"invalid contract document: {entry['path']}")
                    continue
                if document != contract_document(name, CONTRACTS[name]):
                    errors.append(f"contract content mismatch: {entry['path']}")
                dataset_entry = datasets.get(name) if isinstance(datasets, Mapping) else None
                if isinstance(dataset_entry, Mapping) and (
                    dataset_entry.get("fields") != document.get("fields")
                    or dataset_entry.get("sort_keys") != document.get("sort_keys")
                ):
                    errors.append(f"catalog and contract schema mismatch: {name}")
            else:
                errors.append(f"invalid contract entry: {name}")

    study_area = catalog.get("study_area")
    if isinstance(study_area, Mapping):
        errors.extend(_verify_file(project_root, study_area, "study area"))
    else:
        errors.append("data catalog has no study area")
    quality = catalog.get("quality_report")
    if isinstance(quality, Mapping):
        for label in ("json", "markdown"):
            entry = quality.get(label)
            if isinstance(entry, Mapping):
                errors.extend(_verify_file(project_root, entry, f"quality report {label}"))
            else:
                errors.append(f"data catalog has no quality report {label}")
    else:
        errors.append("data catalog has no quality report")
    return errors
