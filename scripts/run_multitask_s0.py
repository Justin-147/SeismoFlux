# ruff: noqa: E402, RUF001
"""Build deterministic, score-blind S0 sample ledgers and scientific figures.

This entry point deliberately stops before model fitting or scoring.  It reads the
frozen S0 contract, verifies local input identities, counts causal samples, and
renders three figures that describe only coverage and sample water levels.
"""

from __future__ import annotations

import os

NUMERIC_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _name in NUMERIC_THREAD_ENVIRONMENT:
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from matplotlib import font_manager
from shapely.geometry import shape

from seismoflux.d1_replay.spatial import D1SpatialDomain, build_d1_spatial_domain
from seismoflux.multitask_s0 import (
    build_episodes,
    build_s0_ledger,
    filter_catalog,
    load_catalog_frame,
)

CATALOG_START_LOCAL: Final[str] = "1970-01-01T00:00:00+08:00"
EXPECTED_FORMAL_HORIZONS: Final[tuple[int, ...]] = (7, 30, 90, 180, 365)
EXPECTED_CATALOG_FILENAME: Final[str] = "earthquake_event.parquet"
SPATIAL_PENDING_STATUS: Final[str] = "deferred_target_blind"
ATOMIC_BLOCK_LEDGER_FIELDS: Final[tuple[str, ...]] = (
    "atomic_block_id",
    "time_block_id",
    "time_block_role",
    "time_block_start_utc",
    "time_block_end_rule",
    "grid_cell_count",
    "clipped_area_km2",
    "all_event_count",
    "m4_plus_event_count",
    "m5_6_event_count",
    "m6_plus_event_count",
    "m5_6_episode_anchor_count",
    "m6_plus_episode_anchor_count",
)
FORBIDDEN_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_model_score",
        "model_scores",
        "hit_count",
        "miss_count",
        "recall",
        "information_gain",
        "information_gain_nats_per_event",
        "candidate_ranking",
        "selected_operating_point",
    }
)
PANEL_DATASETS: Final[dict[str, tuple[str, ...]]] = {
    "catalog_modern": ("earthquake_event",),
    "catalog_historical_large": ("earthquake_event", "earthquake_source_record"),
    "anomaly_reports": ("anomaly_report_period", "anomaly_observation"),
    "simplified_fault_geometry": ("fault_point_raw", "fault_segment"),
    "fault_attributes": ("fault_segment",),
    "long_term_hazard": ("fault_segment",),
    "true_fault_traces": ("fault_trace", "fault_trace_crosswalk_audit"),
    "tectonic_and_map_layers": ("basemap_feature",),
}
MEDIA_TYPES: Final[dict[str, str]] = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".png": "image/png",
}


class S0RunnerError(ValueError):
    """Raised when the frozen score-blind S0 contract cannot be honoured."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise S0RunnerError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise S0RunnerError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise S0RunnerError(f"value is not canonical-JSON serializable: {exc}") from exc
    return text.encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise S0RunnerError(f"cannot write an empty ledger: {path.name}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise S0RunnerError(f"CSV rows disagree on field order: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _content_hash(document: Mapping[str, object], *, domain: str) -> str:
    body = dict(document)
    body.pop("content_sha256", None)
    return _sha256_bytes(_canonical_json_bytes({"domain": domain, "document": body}))


def _assert_score_blind(value: object, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_RESULT_KEYS:
                raise S0RunnerError(f"forbidden model-result key at {path}.{key_text}")
            _assert_score_blind(child, path=f"{path}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _assert_score_blind(child, path=f"{path}[{index}]")


def _load_contract(config_path: Path) -> tuple[Mapping[str, Any], bytes]:
    payload = config_path.read_bytes()
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise S0RunnerError(f"cannot read S0 YAML: {exc}") from exc
    config = _mapping(document, "config")
    scope = _mapping(config.get("scope"), "scope")
    required = {
        "stage": "S0",
        "score_access": "forbidden",
        "model_fitting": "forbidden",
        "locked_test": "forbidden",
        "network_access": "forbidden",
    }
    for key, expected in required.items():
        if scope.get(key) != expected:
            raise S0RunnerError(f"scope.{key} must remain {expected!r}")
    gate = _mapping(config.get("method_adoption_gate"), "method_adoption_gate")
    if gate.get("review_required_before_implementation_or_score_access") is not True:
        raise S0RunnerError("the literature-and-method adoption gate is not frozen")
    horizons = tuple(
        int(value)
        for value in _sequence(
            _mapping(config.get("time_semantics"), "time_semantics").get("formal_horizons_days"),
            "time_semantics.formal_horizons_days",
        )
    )
    if horizons != EXPECTED_FORMAL_HORIZONS:
        raise S0RunnerError("formal horizons changed from 7/30/90/180/365 days")
    return config, payload


def _magnitude_bins(config: Mapping[str, Any]) -> dict[str, tuple[float, float | None]]:
    tasks = _mapping(config.get("magnitude_tasks"), "magnitude_tasks")
    formal = _mapping(tasks.get("formal"), "magnitude_tasks.formal")
    diagnostic = _mapping(tasks.get("training_diagnostic"), "magnitude_tasks.training_diagnostic")

    def bounds(raw: object, label: str) -> tuple[float, float | None]:
        item = _mapping(raw, label)
        lower = float(item.get("lower_inclusive"))
        upper_raw = item.get("upper_exclusive")
        upper = None if upper_raw is None else float(upper_raw)
        if not math.isfinite(lower) or (
            upper is not None and (not math.isfinite(upper) or upper <= lower)
        ):
            raise S0RunnerError(f"invalid bounds in {label}")
        return lower, upper

    result = {
        "m4_plus": bounds(diagnostic.get("M4_plus"), "training_diagnostic.M4_plus"),
        "m5_6": bounds(formal.get("M5_6"), "formal.M5_6"),
        "m6_plus": bounds(formal.get("M6_plus"), "formal.M6_plus"),
    }
    if result != {"m4_plus": (4.0, None), "m5_6": (5.0, 6.0), "m6_plus": (6.0, None)}:
        raise S0RunnerError("formal or diagnostic magnitude bins changed")
    return result


def _catalog_folds(config: Mapping[str, Any]) -> list[dict[str, object]]:
    section = _mapping(config.get("catalog_time_folds"), "catalog_time_folds")
    folds = _sequence(section.get("outer_folds"), "catalog_time_folds.outer_folds")
    result: list[dict[str, object]] = []
    for index, raw in enumerate(folds):
        fold = dict(_mapping(raw, f"catalog_time_folds.outer_folds[{index}]"))
        evaluation_start = pd.Timestamp(fold["target_block_start"])
        if evaluation_start.tzinfo is None:
            raise S0RunnerError("fold target times must be timezone-aware")
        fold["train_start_utc"] = CATALOG_START_LOCAL
        fold["embargo_days"] = 30
        fold["parameter_selection_end_utc"] = (evaluation_start - pd.Timedelta(days=30)).isoformat()
        result.append(fold)
    if len(result) != 6:
        raise S0RunnerError("the six frozen catalog folds are required")
    return result


def _relative_processed_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    prefix = "data/processed/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return Path(normalized)


def _parquet_identity(
    *,
    dataset_name: str,
    metadata: Mapping[str, Any],
    data_root: Path,
    cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    if dataset_name in cache:
        return dict(cache[dataset_name])
    relative = _relative_processed_path(str(metadata.get("path")))
    path = data_root / relative
    expected_sha256 = str(metadata.get("file_sha256"))
    expected_rows = int(metadata.get("row_count"))
    record: dict[str, object] = {
        "dataset_name": dataset_name,
        "path_relative_to_data_root": relative.as_posix(),
        "expected_file_sha256": expected_sha256,
        "expected_row_count": expected_rows,
        "status": "missing",
    }
    if path.is_file():
        parquet = pq.ParquetFile(path)
        actual_rows = int(parquet.metadata.num_rows)
        actual_sha256 = _sha256_file(path)
        schema = parquet.schema_arrow
        record.update(
            {
                "byte_count": path.stat().st_size,
                "actual_file_sha256": actual_sha256,
                "actual_row_count": actual_rows,
                "schema": [
                    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                    for field in schema
                ],
                "hash_matches": actual_sha256 == expected_sha256,
                "row_count_matches": actual_rows == expected_rows,
                "status": (
                    "verified"
                    if actual_sha256 == expected_sha256 and actual_rows == expected_rows
                    else "identity_mismatch"
                ),
            }
        )
    cache[dataset_name] = record
    return dict(record)


def _source_time_range(path: Path, column: str) -> tuple[str | None, str | None]:
    if not path.is_file() or column not in pq.read_schema(path).names:
        return None, None
    values = pd.read_parquet(path, columns=[column])[column]
    values = pd.to_datetime(values, utc=True, errors="coerce").dropna()
    if values.empty:
        return None, None
    return values.min().isoformat().replace("+00:00", "Z"), values.max().isoformat().replace(
        "+00:00", "Z"
    )


def _coverage_rows(
    *,
    catalog: pd.DataFrame,
    data_root: Path,
    data_catalog: Mapping[str, Any],
    anomaly_bundle: Mapping[str, Any] | None,
) -> list[dict[str, object]]:
    datasets = _mapping(data_catalog.get("datasets"), "data_catalog.datasets")

    def dataset_path(name: str) -> Path:
        metadata = _mapping(datasets.get(name), f"data_catalog.datasets.{name}")
        return data_root / _relative_processed_path(str(metadata.get("path")))

    cutoff = cast(pd.Timestamp, catalog["origin_time_utc"].max())
    catalog_min = cast(pd.Timestamp, catalog["origin_time_utc"].min())
    rows: list[dict[str, object]] = [
        {
            "data_family": "catalog_historical_large",
            "display_name": "长期大震目录（并入总目录）",
            "coverage_start": catalog_min.date().isoformat(),
            "coverage_end": cutoff.date().isoformat(),
            "coverage_kind": "event_history",
            "scientific_role": "1900+大震尾部敏感性",
        },
        {
            "data_family": "catalog_modern",
            "display_name": "1970+地震目录",
            "coverage_start": "1970-01-01",
            "coverage_end": cutoff.date().isoformat(),
            "coverage_kind": "event_history",
            "scientific_role": "时间、地点、震级与联合任务主干",
        },
    ]
    anomaly_path = dataset_path("anomaly_report_period")
    anomaly_dates = pd.read_parquet(anomaly_path, columns=["report_date"])["report_date"]
    rows.append(
        {
            "data_family": "anomaly_reports",
            "display_name": "异常周报/因果异常特征",
            "coverage_start": str(anomaly_dates.min()),
            "coverage_end": str(anomaly_dates.max()),
            "coverage_kind": "report_history",
            "scientific_role": "短覆盖增量；只在共同issue配对比较",
        }
    )
    if anomaly_bundle is not None:
        audit = _mapping(anomaly_bundle.get("audit"), "anomaly_bundle.audit")
        if int(audit.get("actual_snapshot_count", -1)) != len(anomaly_dates):
            raise S0RunnerError("anomaly bundle/report snapshot counts disagree")

    static_specs = (
        ("simplified_fault_geometry", "简化断层与属性快照", "fault_segment"),
        ("true_fault_traces", "真实断层迹线快照", "fault_trace"),
        ("tectonic_and_map_layers", "构造/底图快照", "basemap_feature"),
    )
    for family, label, dataset in static_specs:
        start, end = _source_time_range(dataset_path(dataset), "source_available_at")
        if start is None or end is None:
            continue
        rows.append(
            {
                "data_family": family,
                "display_name": label,
                "coverage_start": start[:10],
                "coverage_end": end[:10],
                "coverage_kind": "current_snapshot",
                "scientific_role": "当前快照；历史回顾仅作描述性结构敏感性",
            }
        )
    return rows


def build_authoritative_input_ledger(
    *,
    config: Mapping[str, Any],
    config_payload: bytes,
    project_root: Path,
    data_root: Path,
    catalog: pd.DataFrame,
) -> dict[str, object]:
    data_catalog_path = project_root / "data" / "manifests" / "data_catalog.json"
    source_inventory_path = project_root / "data" / "manifests" / "source_inventory.csv"
    data_catalog = _mapping(
        json.loads(data_catalog_path.read_text(encoding="utf-8")), "data_catalog"
    )
    datasets = _mapping(data_catalog.get("datasets"), "data_catalog.datasets")
    panels_config = _mapping(config.get("data_panels"), "data_panels")
    cache: dict[str, dict[str, object]] = {}
    panels: dict[str, object] = {}
    for panel_name, raw_panel in panels_config.items():
        panel = _mapping(raw_panel, f"data_panels.{panel_name}")
        dataset_records = [
            _parquet_identity(
                dataset_name=dataset,
                metadata=_mapping(datasets.get(dataset), f"datasets.{dataset}"),
                data_root=data_root,
                cache=cache,
            )
            for dataset in PANEL_DATASETS.get(panel_name, ())
        ]
        if panel_name == "public_candidates":
            status = "candidate_only_not_downloaded_or_used_in_s0"
        elif panel_name == "human_forecast_baseline":
            status = "independent_baseline_not_materialized_in_s0"
        elif panel_name == "catalog_historical_large":
            status = "verified_embedded_in_merged_catalog_not_a_separate_model_panel"
        elif dataset_records and all(item["status"] == "verified" for item in dataset_records):
            status = "verified"
        elif dataset_records:
            status = "input_identity_incomplete"
        else:
            status = "described_without_separate_local_dataset"
        panels[panel_name] = {
            "family": panel.get("family"),
            "expected_native_coverage": panel.get("expected_native_coverage"),
            "tasks": panel.get("tasks"),
            "causal_or_historical_status": panel.get("historical_status", panel.get("role")),
            "status": status,
            "datasets": dataset_records,
        }

    anomaly_panel = _mapping(panels_config.get("anomaly_reports"), "anomaly_reports")
    anomaly_relative = Path(str(anomaly_panel.get("authoritative_processed_dataset")))
    anomaly_directory = data_root / anomaly_relative
    anomaly_manifest_path = anomaly_directory / "manifest.json"
    anomaly_bundle: Mapping[str, Any] | None = None
    anomaly_identity: dict[str, object] = {
        "path_relative_to_data_root": anomaly_relative.as_posix(),
        "status": "missing",
    }
    if anomaly_manifest_path.is_file():
        anomaly_bundle = _mapping(
            json.loads(anomaly_manifest_path.read_text(encoding="utf-8")), "anomaly_bundle"
        )
        if anomaly_bundle.get("bundle_id") != anomaly_directory.name:
            raise S0RunnerError("anomaly bundle ID disagrees with its content-addressed directory")
        audit = _mapping(anomaly_bundle.get("audit"), "anomaly_bundle.audit")
        locked = _mapping(audit.get("locked_test"), "anomaly_bundle.audit.locked_test")
        if locked.get("run") is not False:
            raise S0RunnerError("S0 cannot use an anomaly bundle with a locked-test run")
        anomaly_identity.update(
            {
                "status": "manifest_verified_without_rehashing_4GB_feature_store",
                "bundle_id": anomaly_bundle.get("bundle_id"),
                "manifest_file_sha256": _sha256_file(anomaly_manifest_path),
                "bundle_identity_sha256": anomaly_bundle.get("identity_sha256"),
                "actual_snapshot_count": audit.get("actual_snapshot_count"),
                "feature_row_count": audit.get("feature_row_count"),
                "first_report_date": audit.get("first_report_date"),
                "last_report_date": audit.get("last_report_date"),
                "target_or_earthquake_label_read_count": audit.get(
                    "target_or_earthquake_label_read_count"
                ),
                "locked_test_run": locked.get("run"),
                "identity_note": (
                    "S0 verifies the small bundle manifest and its content-addressed ID; "
                    "it does not spend hours rehashing the 4GB feature store."
                ),
            }
        )

    study_area_meta = _mapping(data_catalog.get("study_area"), "data_catalog.study_area")
    study_area_path = data_root / _relative_processed_path(str(study_area_meta.get("path")))
    expected_study_hash = str(study_area_meta.get("sha256"))
    actual_study_hash = _sha256_file(study_area_path)
    if actual_study_hash != expected_study_hash:
        raise S0RunnerError("study-area identity differs from the authoritative manifest")

    coverage = _coverage_rows(
        catalog=catalog,
        data_root=data_root,
        data_catalog=data_catalog,
        anomaly_bundle=anomaly_bundle,
    )
    ledger: dict[str, object] = {
        "schema_version": 1,
        "ledger_type": "multitask_s0_authoritative_input_ledger",
        "score_blind": True,
        "config_sha256": _sha256_bytes(config_payload),
        "data_root_semantics": "explicit_read_only_processed_root_not_serialized_as_machine_path",
        "data_catalog": {
            "path_relative_to_project": "data/manifests/data_catalog.json",
            "file_sha256": _sha256_file(data_catalog_path),
            "snapshot_id": data_catalog.get("snapshot_id"),
        },
        "source_inventory": {
            "path_relative_to_project": "data/manifests/source_inventory.csv",
            "file_sha256": _sha256_file(source_inventory_path),
        },
        "study_area": {
            "path_relative_to_data_root": _relative_processed_path(
                str(study_area_meta.get("path"))
            ).as_posix(),
            "expected_sha256": expected_study_hash,
            "actual_sha256": actual_study_hash,
            "hash_matches": True,
            "properties": study_area_meta.get("properties"),
        },
        "anomaly_feature_bundle": anomaly_identity,
        "panels": panels,
        "coverage_timeline": coverage,
        "interpretation": {
            "native_coverage": "each family uses its longest causally legal history",
            "paired_overlap": "new-data contribution later requires identical issues and targets",
            "s0_effect_evidence": "none; these are input and sample identities only",
        },
    }
    ledger["content_sha256"] = _content_hash(
        ledger, domain="seismoflux.multitask-s0-authoritative-input-ledger.v1"
    )
    _assert_score_blind(ledger)
    return ledger


def build_atomic_block_sample_water_levels(
    *,
    config: Mapping[str, Any],
    catalog: pd.DataFrame,
    catalog_cutoff: pd.Timestamp,
    magnitude_bins: Mapping[str, tuple[float, float | None]],
    spatial_domain: D1SpatialDomain,
    cell_mapping_path: Path,
    cell_mapping_sha256: str,
) -> dict[str, object]:
    """Count target-blind sample water levels on 39 frozen anonymous blocks.

    Raw construction-zone identifiers are used only in memory.  Public rows expose
    deterministic aliases derived from UTF-8 byte ordering, never the alias map,
    cell identifiers, coordinates, geometry, or per-cell assignments.
    """

    mapping = pd.read_parquet(
        cell_mapping_path,
        columns=["cell_id", "construction_zone_id"],
    )
    if len(mapping) != spatial_domain.operational_grid.cell_count:
        raise S0RunnerError("25 km cell-to-block mapping row count changed")
    if mapping.isna().any().any():
        raise S0RunnerError("25 km cell-to-block mapping contains missing identities")
    mapping["cell_id"] = mapping["cell_id"].astype("string")
    mapping["construction_zone_id"] = mapping["construction_zone_id"].astype("string")
    if mapping["cell_id"].duplicated().any():
        raise S0RunnerError("25 km cell-to-block mapping contains duplicate cells")

    operational_grid = spatial_domain.operational_grid
    mapped_cell_ids = set(mapping["cell_id"].tolist())
    if mapped_cell_ids != set(operational_grid.cell_ids):
        raise S0RunnerError("D1 target-blind grid and frozen cell-to-block mapping differ")
    raw_zone_ids = sorted(
        set(mapping["construction_zone_id"].tolist()),
        key=lambda value: value.encode("utf-8"),
    )
    if len(raw_zone_ids) != 39:
        raise S0RunnerError("frozen nonempty atomic-block count changed")
    alias_by_zone = {
        zone_id: f"atomic_block_{index:02d}" for index, zone_id in enumerate(raw_zone_ids, start=1)
    }
    zone_by_cell = dict(zip(mapping["cell_id"], mapping["construction_zone_id"], strict=True))
    alias_by_cell = {
        cell_id: alias_by_zone[str(zone_by_cell[cell_id])] for cell_id in operational_grid.cell_ids
    }

    aliases = tuple(f"atomic_block_{index:02d}" for index in range(1, 40))
    cell_count_by_alias = {alias: 0 for alias in aliases}
    area_by_alias: dict[str, list[float]] = {alias: [] for alias in aliases}
    for cell_id, area in zip(
        operational_grid.cell_ids,
        operational_grid.clipped_area_km2,
        strict=True,
    ):
        alias = alias_by_cell[cell_id]
        cell_count_by_alias[alias] += 1
        area_by_alias[alias].append(float(area))
    if any(count <= 0 for count in cell_count_by_alias.values()):
        raise S0RunnerError("every frozen atomic block must contain at least one 25 km cell")

    eligible = filter_catalog(
        catalog,
        origin_start=CATALOG_START_LOCAL,
        origin_end=catalog_cutoff + pd.Timedelta(nanoseconds=1),
        available_by=catalog_cutoff,
        study_area_only=True,
    ).copy()
    event_aliases: list[str | None] = []
    for row in eligible.itertuples(index=False):
        cell_index = spatial_domain.locator.locate_lonlat(
            float(row.longitude),
            float(row.latitude),
        )
        event_aliases.append(
            None if cell_index is None else alias_by_cell[operational_grid.cell_ids[cell_index]]
        )
    unlocated_count = sum(alias is None for alias in event_aliases)
    if unlocated_count:
        raise S0RunnerError(
            f"{unlocated_count} study-area events since 1970 were not located on the frozen grid"
        )
    eligible["_atomic_block_id"] = cast(list[str], event_aliases)

    anchor_ids_by_bin: dict[str, set[str]] = {}
    for magnitude_bin in ("m5_6", "m6_plus"):
        minimum, maximum = magnitude_bins[magnitude_bin]
        formal_events = eligible[eligible["magnitude"] >= minimum]
        if maximum is not None:
            formal_events = formal_events[formal_events["magnitude"] < maximum]
        anchor_ids_by_bin[magnitude_bin] = {
            str(episode["anchor_event_id"])
            for episode in build_episodes(formal_events.reset_index(drop=True))
        }

    catalog_fold_section = _mapping(config.get("catalog_time_folds"), "catalog_time_folds")
    raw_folds = _sequence(catalog_fold_section.get("outer_folds"), "catalog_time_folds.outer_folds")
    if len(raw_folds) != 6:
        raise S0RunnerError("the six frozen catalog time blocks are required")
    catalog_start_utc = pd.Timestamp(CATALOG_START_LOCAL).tz_convert("UTC")
    time_blocks: list[dict[str, object]] = [
        {
            "id": "ALL_1970_PLUS",
            "role": "all_catalog_1970_plus",
            "start": catalog_start_utc,
            "end": catalog_cutoff + pd.Timedelta(nanoseconds=1),
            "end_rule": "catalog_truth_cutoff_inclusive",
        }
    ]
    for index, raw_fold in enumerate(raw_folds):
        fold = _mapping(raw_fold, f"catalog_time_folds.outer_folds[{index}]")
        start = pd.Timestamp(fold.get("target_block_start"))
        if start.tzinfo is None:
            raise S0RunnerError("catalog time-block starts must be timezone-aware")
        start = start.tz_convert("UTC")
        raw_end = fold.get("target_block_end_exclusive")
        if raw_end == "derived_from_catalog_truth_cutoff":
            end = catalog_cutoff + pd.Timedelta(nanoseconds=1)
            end_rule = "catalog_truth_cutoff_inclusive"
        else:
            end = pd.Timestamp(raw_end)
            if end.tzinfo is None:
                raise S0RunnerError("catalog time-block ends must be timezone-aware")
            end = end.tz_convert("UTC")
            end_rule = end.isoformat().replace("+00:00", "Z") + "_exclusive"
        if end <= start:
            raise S0RunnerError("catalog time block must have positive duration")
        time_blocks.append(
            {
                "id": str(fold.get("id")),
                "role": str(fold.get("role")),
                "start": start,
                "end": end,
                "end_rule": end_rule,
            }
        )

    rows: list[dict[str, object]] = []
    for time_block in time_blocks:
        start = cast(pd.Timestamp, time_block["start"])
        end = cast(pd.Timestamp, time_block["end"])
        window = eligible[
            (eligible["origin_time_utc"] >= start) & (eligible["origin_time_utc"] < end)
        ]
        for alias in aliases:
            block = window[window["_atomic_block_id"] == alias]
            m4_plus = block["magnitude"] >= magnitude_bins["m4_plus"][0]
            m5_6 = (block["magnitude"] >= magnitude_bins["m5_6"][0]) & (
                block["magnitude"] < cast(float, magnitude_bins["m5_6"][1])
            )
            m6_plus = block["magnitude"] >= magnitude_bins["m6_plus"][0]
            event_ids = set(block["event_id"].astype(str).tolist())
            row = {
                "atomic_block_id": alias,
                "time_block_id": time_block["id"],
                "time_block_role": time_block["role"],
                "time_block_start_utc": start.isoformat().replace("+00:00", "Z"),
                "time_block_end_rule": time_block["end_rule"],
                "grid_cell_count": cell_count_by_alias[alias],
                "clipped_area_km2": math.fsum(area_by_alias[alias]),
                "all_event_count": len(block),
                "m4_plus_event_count": int(m4_plus.sum()),
                "m5_6_event_count": int(m5_6.sum()),
                "m6_plus_event_count": int(m6_plus.sum()),
                "m5_6_episode_anchor_count": len(event_ids & anchor_ids_by_bin["m5_6"]),
                "m6_plus_episode_anchor_count": len(event_ids & anchor_ids_by_bin["m6_plus"]),
            }
            if tuple(row) != ATOMIC_BLOCK_LEDGER_FIELDS:
                raise S0RunnerError("atomic-block public ledger field order changed")
            rows.append(row)

    row_content_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            {
                "domain": "seismoflux.multitask-s0-atomic-block-water-levels.v1",
                "rows": rows,
            }
        )
    )
    ledger: dict[str, object] = {
        "schema_version": 1,
        "ledger_type": "multitask_s0_anonymous_atomic_block_sample_water_levels",
        "score_blind": True,
        "scientific_role": "sample_size_and_transfer_power_audit_only",
        "block_definition": (
            "39 frozen nonempty construction-linework atomic blocks; aliases follow "
            "internal UTF-8 ordering and the private alias map is not emitted"
        ),
        "event_assignment_method": (
            "existing D1 target-independent 25 km locator, then frozen private "
            "cell-to-construction-block mapping"
        ),
        "episode_definition": "30d/75km causal fixed-first-event anchor before time blocks",
        "source_cell_mapping_file_sha256": cell_mapping_sha256,
        "public_row_content_sha256": row_content_sha256,
        "atomic_block_count": len(aliases),
        "time_block_count_including_all_1970_plus": len(time_blocks),
        "public_row_count": len(rows),
        "inside_study_area_1970_plus_event_count": len(eligible),
        "unlocated_event_count": unlocated_count,
        "target_counts_did_not_define_or_adjust_blocks": True,
        "raw_zone_ids_emitted": False,
        "cell_ids_or_coordinates_emitted": False,
        "alias_to_zone_mapping_emitted": False,
        "rows": rows,
    }
    _assert_score_blind(ledger)
    return ledger


def build_spatial_identity_ledger(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    data_root: Path,
    input_ledger: Mapping[str, object],
    catalog: pd.DataFrame,
    catalog_cutoff: pd.Timestamp,
    magnitude_bins: Mapping[str, tuple[float, float | None]],
) -> dict[str, object]:
    spatial = _mapping(config.get("spatial_extrapolation"), "spatial_extrapolation")
    manifest_relative = Path(str(spatial.get("frozen_zone_manifest")))
    manifest_path = project_root / manifest_relative
    expected_hash = str(spatial.get("expected_manifest_sha256"))
    actual_hash = _sha256_file(manifest_path)
    if actual_hash != expected_hash:
        raise S0RunnerError("frozen spatial source manifest hash changed")
    source = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "spatial_manifest")
    aggregate = _mapping(source.get("aggregate"), "spatial_manifest.aggregate")
    topology = _mapping(source.get("topology_gate"), "spatial_manifest.topology_gate")
    security = _mapping(source.get("security"), "spatial_manifest.security")
    manifest_inputs = _mapping(source.get("input_hashes"), "spatial_manifest.input_hashes")
    manifest_local = _mapping(source.get("local_artifacts"), "spatial_manifest.local_artifacts")
    if security.get("target_or_score_read") is not False:
        raise S0RunnerError("spatial source manifest is not target/score blind")
    expected_linework = _mapping(
        spatial.get("source_linework_sha256"), "spatial_extrapolation.source_linework_sha256"
    )
    if manifest_inputs.get("construction_linework_l1") != expected_linework.get(
        "L1"
    ) or manifest_inputs.get("construction_linework_l2") != expected_linework.get("L2"):
        raise S0RunnerError("construction-linework identities differ from the S0 contract")
    expected_zone_set = str(spatial.get("frozen_zone_set_sha256"))
    if aggregate.get("zone_set_sha256") != expected_zone_set:
        raise S0RunnerError("construction-zone set identity differs from the S0 contract")

    restricted_directory = data_root.parent / "interim" / "stage4" / "anomaly_increment_r2"
    restricted_names = {
        "cell_mapping": "construction_zone_cell_mapping.parquet",
        "entity_mapping": "construction_zone_entity_mapping.parquet",
        "connectors": "construction_zone_connectors.json",
        "zone_geometry": "construction_zones.parquet",
    }
    restricted_records: dict[str, object] = {}
    for artifact_name, filename in restricted_names.items():
        artifact_meta = _mapping(
            manifest_local.get(artifact_name), f"spatial_manifest.local_artifacts.{artifact_name}"
        )
        artifact_path = restricted_directory / filename
        if not artifact_path.is_file():
            raise S0RunnerError(f"missing restricted spatial identity source: {filename}")
        actual_artifact_hash = _sha256_file(artifact_path)
        if actual_artifact_hash != artifact_meta.get("sha256"):
            raise S0RunnerError(f"restricted spatial artifact hash changed: {filename}")
        record: dict[str, object] = {
            "path_relative_to_data_directory": (
                Path("interim") / "stage4" / "anomaly_increment_r2" / filename
            ).as_posix(),
            "byte_count": artifact_path.stat().st_size,
            "sha256": actual_artifact_hash,
            "hash_matches_public_manifest": True,
        }
        if artifact_path.suffix == ".parquet":
            record["row_count"] = int(pq.ParquetFile(artifact_path).metadata.num_rows)
        restricted_records[artifact_name] = record

    study = _mapping(input_ledger.get("study_area"), "input_ledger.study_area")
    study_path = data_root / Path(str(study.get("path_relative_to_data_root")))
    study_document = _mapping(
        json.loads(study_path.read_text(encoding="utf-8")), "study_area_geojson"
    )
    study_geometry = shape(study_document.get("geometry"))
    spatial_domain = build_d1_spatial_domain(study_geometry)
    grid_records = [
        {
            "cell_size_km": grid.cell_size_km,
            "grid_id": grid.grid_id,
            "cell_count": grid.cell_count,
            "clipped_area_km2": math.fsum(float(area) for area in grid.clipped_area_km2),
            "target_independent": True,
        }
        for grid in spatial_domain.quadrature_family.grids
    ]
    primary_grid = next(item for item in grid_records if item["cell_size_km"] == 25.0)
    source_query_cells = int(aggregate.get("assigned_query_cell_count"))
    grid_count_matches = int(primary_grid["cell_count"]) == source_query_cells
    if not grid_count_matches:
        raise S0RunnerError("recomputed 25 km grid count differs from the frozen spatial source")
    atomic_block_count = int(spatial.get("atomic_block_count"))
    if (
        int(aggregate.get("zone_count")) != 65
        or int(aggregate.get("assigned_nonempty_zone_count")) != atomic_block_count
        or atomic_block_count != 39
        or source_query_cells != 15_697
    ):
        raise S0RunnerError("frozen 65/39/15697 construction-zone water levels changed")
    configured_cell_mapping_hash = str(spatial.get("frozen_25km_cell_mapping_file_sha256"))
    if configured_cell_mapping_hash != _mapping(
        manifest_local.get("cell_mapping"), "cell_mapping"
    ).get("sha256"):
        raise S0RunnerError("25 km cell-to-zone mapping identity changed")
    selection = _mapping(
        spatial.get("model_selection_geometry_folds"),
        "spatial_extrapolation.model_selection_geometry_folds",
    )
    if selection.get("current_status") != SPATIAL_PENDING_STATUS:
        raise S0RunnerError("model-selection spatial fold status changed")
    atomic_block_water_levels = build_atomic_block_sample_water_levels(
        config=config,
        catalog=catalog,
        catalog_cutoff=catalog_cutoff,
        magnitude_bins=magnitude_bins,
        spatial_domain=spatial_domain,
        cell_mapping_path=restricted_directory / restricted_names["cell_mapping"],
        cell_mapping_sha256=configured_cell_mapping_hash,
    )

    ledger: dict[str, object] = {
        "schema_version": 2,
        "ledger_type": "multitask_s0_spatial_identity_ledger",
        "score_blind": True,
        "study_area": {
            "sha256": study.get("actual_sha256"),
            "target_independent": True,
        },
        "integration_grid_family": {
            "method": "existing_target_independent_China_Albers_clipped_grid_builder",
            "grids": grid_records,
            "primary_25km_count_matches_spatial_source": grid_count_matches,
        },
        "construction_zone_source": {
            "manifest_path_relative_to_project": manifest_relative.as_posix(),
            "expected_manifest_sha256": expected_hash,
            "actual_manifest_sha256": actual_hash,
            "manifest_hash_matches": True,
            "zone_count": aggregate.get("zone_count"),
            "nonempty_zone_count": aggregate.get("assigned_nonempty_zone_count"),
            "zone_set_sha256": expected_zone_set,
            "query_cell_count": source_query_cells,
            "topology_gate_passed": topology.get("passed"),
            "target_or_score_read": security.get("target_or_score_read"),
            "restricted_artifacts": restricted_records,
        },
        "validation_tracks": {
            "primary_scientific_track": spatial.get("primary_scientific_track"),
            "secondary_transfer_stress_test": {
                "status": "source_identity_and_score_blind_sample_water_levels_verified",
                "scientific_role": spatial.get("scientific_role"),
                "source_interpretation": spatial.get("source_interpretation"),
                "atomic_block_count": atomic_block_count,
                "outer_scheme": spatial.get("outer_scheme"),
                "outer_fit_count": spatial.get("outer_fit_count"),
                "pooled_oof_alarm_rule": spatial.get("pooled_oof_alarm_rule"),
                "primary_training_buffer_km": spatial.get("primary_training_buffer_km"),
                "buffer_sensitivity_km": spatial.get("buffer_sensitivity_km"),
            },
        },
        "anonymous_atomic_block_sample_water_levels": atomic_block_water_levels,
        "model_selection_geometry_folds": {
            "status": selection.get("current_status"),
            "materialization_stage": selection.get("materialization_stage"),
            "atomic_inputs": selection.get("atomic_inputs"),
            "algorithm": selection.get("algorithm"),
            "candidate_k_in_order": selection.get("candidate_k_in_order"),
            "random_seed": selection.get("random_seed"),
            "n_init": selection.get("n_init"),
            "zone_to_fold_mapping": None,
            "scientific_reason": (
                "Time-forward CSEP-like evaluation is primary and 39-block LOBO is a "
                "secondary transfer stress test. The 39-to-k selection map is "
                "intentionally deferred until S1 before any model score."
            ),
            "next_action": (
                "run the preregistered target-blind geometry/power procedure at the start "
                "of S1 and publish the exact zone-to-fold table and SHA-256 before scores"
            ),
        },
        "interpretation": (
            "Study area, grids, 65 source zones, 39 nonempty atomic blocks, 15697 cells, "
            "all four restricted artifacts, and public-safe block sample water levels "
            "are verified. Target counts did not define the blocks. The optional 39-to-k "
            "map is deliberately not claimed complete in S0."
        ),
    }
    ledger["content_sha256"] = _content_hash(
        ledger, domain="seismoflux.multitask-s0-spatial-identity-ledger.v2"
    )
    _assert_score_blind(ledger)
    return ledger


def build_episode_rows(
    *,
    catalog: pd.DataFrame,
    catalog_cutoff: pd.Timestamp,
    magnitude_bins: Mapping[str, tuple[float, float | None]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episode_rows: list[dict[str, object]] = []
    size_rows: list[dict[str, object]] = []
    classes: tuple[tuple[str, int, int | None], ...] = (
        ("1", 1, 1),
        ("2", 2, 2),
        ("3–5", 3, 5),
        ("6–10", 6, 10),
        (">10", 11, None),
    )
    for magnitude_bin in ("m5_6", "m6_plus"):
        minimum, maximum = magnitude_bins[magnitude_bin]
        eligible = filter_catalog(
            catalog,
            origin_start=CATALOG_START_LOCAL,
            origin_end=catalog_cutoff + pd.Timedelta(nanoseconds=1),
            available_by=catalog_cutoff,
            magnitude_minimum=minimum,
            magnitude_maximum_exclusive=maximum,
            study_area_only=True,
        )
        episodes = build_episodes(eligible)
        for episode in episodes:
            member_ids = [str(value) for value in episode["member_event_ids"]]
            episode_rows.append(
                {
                    "magnitude_bin": magnitude_bin,
                    "episode_id": episode["episode_id"],
                    "anchor_event_id": episode["anchor_event_id"],
                    "anchor_time_utc": episode["anchor_time_utc"],
                    "anchor_magnitude": episode["anchor_magnitude"],
                    "maximum_magnitude": episode["maximum_magnitude"],
                    "member_count": episode["member_count"],
                    "member_time_min_utc": episode["member_time_min_utc"],
                    "member_time_max_utc": episode["member_time_max_utc"],
                    "member_event_ids_sha256": _sha256_bytes(_canonical_json_bytes(member_ids)),
                    "member_event_ids": "|".join(member_ids),
                }
            )
        member_counts = np.asarray([int(item["member_count"]) for item in episodes], dtype=int)
        for label, lower, upper in classes:
            mask = member_counts >= lower
            if upper is not None:
                mask &= member_counts <= upper
            selected = member_counts[mask]
            size_rows.append(
                {
                    "magnitude_bin": magnitude_bin,
                    "episode_size_class": label,
                    "minimum_members": lower,
                    "maximum_members": upper,
                    "episode_count": int(mask.sum()),
                    "event_count": int(selected.sum()),
                }
            )
    return episode_rows, size_rows


def flatten_fold_horizon_ledger(catalog_ledger: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    folds = cast(Sequence[Mapping[str, object]], catalog_ledger["fold_maturity"])
    for fold in folds:
        horizons = cast(Mapping[str, Mapping[str, object]], fold["horizons"])
        for horizon_text, horizon in horizons.items():
            for axis_name in ("operational_weekly", "primary_exposure"):
                axis = cast(Mapping[str, object], horizon[axis_name])
                bins = cast(Mapping[str, Mapping[str, object]], axis["magnitude_bins"])
                for magnitude_bin, values in bins.items():
                    evaluable = bool(axis["evaluable"])
                    rows.append(
                        {
                            "fold_id": fold["fold_id"],
                            "role": fold["role"],
                            "horizon_days": int(horizon_text),
                            "axis": axis_name,
                            "statistical_status": axis["statistical_status"],
                            "availability_status": axis["availability_status"],
                            "evaluable": evaluable,
                            "issue_count": axis["issue_count"],
                            "magnitude_bin": magnitude_bin,
                            "issue_target_pair_count": (
                                values.get("issue_target_pair_count") if evaluable else None
                            ),
                            "unique_event_count": (
                                values.get("unique_event_count") if evaluable else None
                            ),
                            "episode_sampling_status": values.get("episode_sampling_status"),
                            "touched_episode_count": (
                                values.get("touched_episode_count") if evaluable else None
                            ),
                            "anchor_target_count": (
                                values.get("anchor_target_count") if evaluable else None
                            ),
                            "subsequent_target_event_count": (
                                values.get("subsequent_target_event_count") if evaluable else None
                            ),
                            "episode_balanced_total_weight": (
                                values.get("episode_balanced_total_weight") if evaluable else None
                            ),
                        }
                    )
    return rows


def flatten_issue_ledger(catalog_ledger: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    folds = cast(Sequence[Mapping[str, object]], catalog_ledger["fold_maturity"])
    for fold in folds:
        calendar = cast(Mapping[str, object], fold["catalog_issue_calendar"])
        issues = [pd.Timestamp(value) for value in cast(Sequence[str], calendar["issue_times_utc"])]
        horizons = cast(Mapping[str, Mapping[str, object]], fold["horizons"])
        for horizon_text, horizon in horizons.items():
            horizon_days = int(horizon_text)
            maturity_limit = pd.Timestamp(horizon["maturity_limit_utc"])
            primary = cast(Mapping[str, object], horizon["primary_exposure"])
            primary_times = set(cast(Sequence[str], primary["issue_times_utc"]))
            for issue in issues:
                target_end = issue + pd.Timedelta(days=horizon_days)
                issue_text = issue.isoformat().replace("+00:00", "Z")
                mature = target_end <= maturity_limit
                rows.append(
                    {
                        "fold_id": fold["fold_id"],
                        "role": fold["role"],
                        "issue_time_utc": issue_text,
                        "horizon_days": horizon_days,
                        "target_interval": "(T,T+h]",
                        "target_end_utc": target_end.isoformat().replace("+00:00", "Z"),
                        "maturity_status": "mature" if mature else "unavailable_not_mature",
                        "primary_exposure_selected": mature and issue_text in primary_times,
                    }
                )
    return rows


def _configure_plot_style() -> str:
    installed = {item.name for item in font_manager.fontManager.ttflist}
    preferred = (
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    )
    font_name = next((name for name in preferred if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c7cbd1",
            "axes.labelcolor": "#27303f",
            "text.color": "#27303f",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "axes.titleweight": "bold",
        }
    )
    return font_name


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        metadata={"Software": "SeismoFlux S0 score-blind renderer"},
    )
    plt.close(figure)


def render_coverage_timeline(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.5, 6.8))
    colors = {
        "event_history": "#31688e",
        "report_history": "#35b779",
        "current_snapshot": "#f59e0b",
    }
    for index, row in enumerate(rows):
        start = pd.Timestamp(row["coverage_start"])
        end = pd.Timestamp(row["coverage_end"])
        kind = str(row["coverage_kind"])
        if kind == "current_snapshot" or start == end:
            axis.scatter(end, index, marker="D", s=85, color=colors[kind], zorder=3)
        else:
            axis.barh(
                index,
                (end - start).days,
                left=mdates.date2num(start),
                height=0.48,
                color=colors[kind],
                alpha=0.92,
            )
        axis.text(
            end + pd.Timedelta(days=180),
            index,
            str(row["scientific_role"]),
            va="center",
            fontsize=8.5,
            color="#4b5563",
        )
    axis.set_yticks(range(len(rows)), [str(row["display_name"]) for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(pd.Timestamp("1895-01-01"), pd.Timestamp("2038-01-01"))
    axis.xaxis.set_major_locator(mdates.YearLocator(10))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    axis.set_title("SeismoFlux S0｜不同数据的实际时间覆盖")
    axis.set_xlabel("年份（长条=历史覆盖；菱形=当前静态快照）")
    figure.text(
        0.5,
        0.015,
        "样本水位，无模型效果｜短覆盖数据只在共同期评价增量，不截短1970+目录",
        ha="center",
        fontsize=10,
        weight="bold",
        color="#9a3412",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _save_figure(figure, path)


def render_sample_funnel_and_episode_sizes(
    *,
    catalog_ledger: Mapping[str, object],
    size_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    funnel = cast(Mapping[str, object], catalog_ledger["sample_funnel"])
    bins = cast(Mapping[str, int], funnel["magnitude_bin_counts"])
    summaries = cast(
        Mapping[str, Mapping[str, object]], catalog_ledger["episode_summary_by_magnitude_bin"]
    )
    labels = (
        "全部去重事件",
        "研究区内",
        "研究区+1970起",
        "M4+训练/诊断",
        "M5–6事件",
        "M5–6固定锚点episode",
        "M6+事件",
        "M6+固定锚点episode",
    )
    values = np.asarray(
        [
            funnel["all_catalog_rows"],
            funnel["inside_study_area_rows"],
            funnel["inside_origin_range_and_available_rows"],
            bins["m4_plus"],
            bins["m5_6"],
            summaries["m5_6"]["episode_count"],
            bins["m6_plus"],
            summaries["m6_plus"]["episode_count"],
        ],
        dtype=float,
    )
    colors = (
        "#355f8d",
        "#3b7a8d",
        "#33977c",
        "#43b36b",
        "#f59e0b",
        "#d97706",
        "#ef6c57",
        "#c2413b",
    )
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14.8, 7.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    positions = np.arange(len(labels))
    left.barh(positions, values, color=colors, height=0.68)
    left.set_yticks(positions, labels)
    left.invert_yaxis()
    left.set_xscale("log")
    left.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    left.set_xlabel("样本数（对数坐标）")
    left.set_title("A｜事件到正式评价episode的样本漏斗")
    for position, value in zip(positions, values, strict=True):
        left.text(value * 1.07, position, f"{int(value):,}", va="center", fontsize=9)

    classes = ["1", "2", "3–5", "6–10", ">10"]
    x = np.arange(len(classes))
    width = 0.36
    for offset, magnitude_bin, label, color in (
        (-width / 2, "m5_6", "M5–6", "#d97706"),
        (width / 2, "m6_plus", "M6+", "#c2413b"),
    ):
        lookup = {
            str(row["episode_size_class"]): int(row["episode_count"])
            for row in size_rows
            if row["magnitude_bin"] == magnitude_bin
        }
        heights = [lookup[item] for item in classes]
        bars = right.bar(x + offset, heights, width, label=label, color=color)
        right.bar_label(bars, padding=3, fontsize=8)
    right.set_xticks(x, classes)
    right.set_xlabel("每个episode包含的事件数")
    right.set_ylabel("episode数")
    right.set_title("B｜固定锚点episode大小分布")
    right.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    right.legend(frameon=False)
    figure.suptitle("SeismoFlux S0｜目录样本怎样变成独立评价单位", fontsize=16, weight="bold")
    figure.text(
        0.5,
        0.02,
        "30天/75 km因果固定首事件锚点；成员不扩张边界｜样本水位，无模型效果",
        ha="center",
        fontsize=10,
        weight="bold",
        color="#9a3412",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save_figure(figure, path)


def render_fold_horizon_heatmap(
    *,
    fold_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    primary = [row for row in fold_rows if row["axis"] == "primary_exposure"]
    folds = list(dict.fromkeys(str(row["fold_id"]) for row in primary))
    columns = [
        (horizon, magnitude_bin)
        for horizon in EXPECTED_FORMAL_HORIZONS
        for magnitude_bin in ("m5_6", "m6_plus")
    ]
    event_values = np.full((len(folds), len(columns)), np.nan)
    episode_values = np.full_like(event_values, np.nan)
    lookup = {
        (str(row["fold_id"]), int(row["horizon_days"]), str(row["magnitude_bin"])): row
        for row in primary
    }
    for row_index, fold in enumerate(folds):
        for column_index, (horizon, magnitude_bin) in enumerate(columns):
            row = lookup[(fold, horizon, magnitude_bin)]
            if bool(row["evaluable"]):
                event_values[row_index, column_index] = float(row["unique_event_count"])
                episode = row["anchor_target_count"]
                episode_values[row_index, column_index] = (
                    np.nan if episode is None else float(episode)
                )

    figure, axes = plt.subplots(2, 1, figsize=(15.8, 9.2), sharex=True)
    labels = [f"{h}d\n{'M5–6' if m == 'm5_6' else 'M6+'}" for h, m in columns]
    for axis, values, title, colour_map in (
        (axes[0], event_values, "A｜唯一目标事件水位", "YlGnBu"),
        (axes[1], episode_values, "B｜固定锚点episode锚点目标水位", "YlOrRd"),
    ):
        masked = np.ma.masked_invalid(values)
        image = axis.imshow(masked, aspect="auto", cmap=colour_map, interpolation="nearest")
        image.cmap.set_bad("#e5e7eb")
        axis.set_yticks(range(len(folds)), folds)
        axis.set_title(title, loc="left")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                text = "NA" if np.isnan(value) else str(int(value))
                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111827",
                    weight="bold" if not np.isnan(value) else "normal",
                )
        figure.colorbar(image, ax=axis, shrink=0.72, pad=0.015, label="样本数")
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_xlabel("预测时长 × 正式震级档")
    figure.suptitle("SeismoFlux S0｜各时间外推折的独立主暴露样本水位", fontsize=16, weight="bold")
    figure.text(
        0.5,
        0.015,
        "按时间贪心选择不重叠目标窗并加30天保护；NA=窗口尚未成熟｜样本水位，无模型效果",
        ha="center",
        fontsize=10,
        weight="bold",
        color="#9a3412",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.95))
    _save_figure(figure, path)


def _readme_text(
    *,
    catalog_ledger: Mapping[str, object],
    spatial_ledger: Mapping[str, object],
    font_name: str,
) -> str:
    funnel = cast(Mapping[str, object], catalog_ledger["sample_funnel"])
    bins = cast(Mapping[str, int], funnel["magnitude_bin_counts"])
    episodes = cast(
        Mapping[str, Mapping[str, object]], catalog_ledger["episode_summary_by_magnitude_bin"]
    )
    spatial_folds = cast(Mapping[str, object], spatial_ledger["model_selection_geometry_folds"])
    block_water = cast(
        Mapping[str, object],
        spatial_ledger["anonymous_atomic_block_sample_water_levels"],
    )
    return f"""# SeismoFlux S0 无成绩样本包

## 一句话说明

这一步只回答“现有数据覆盖多久、真正有多少可独立评价的地震样本、每个历史外推折是否有足够水位”，
不训练模型，也不包含命中、召回、信息增益或候选排名。因此这些图不能证明预测已经变好。

## 当前目录水位

- 全部去重事件：{int(funnel["all_catalog_rows"]):,}
- 研究区内事件：{int(funnel["inside_study_area_rows"]):,}
- 1970年以来研究区内事件：{int(funnel["inside_origin_range_and_available_rows"]):,}
- M4+训练/诊断事件：{int(bins["m4_plus"]):,}
- M5–6正式事件 / 固定锚点episode：{int(bins["m5_6"]):,} /
  {int(episodes["m5_6"]["episode_count"]):,}
- M6+正式事件 / 固定锚点episode：{int(bins["m6_plus"]):,} /
  {int(episodes["m6_plus"]["episode_count"]):,}

episode采用已经完成国内外经典方法复核后的“30天/75 km因果固定首事件锚点”：每个新事件搜索全部
既有锚点，后续成员不会把边界一路拉长。它只用于衡量“是否发现新地区”，完整目录仍保留后续事件。

## 图件怎么看

1. `figure_01_data_coverage_timeline.png`：各类数据实际覆盖的年代。它说明异常只有短覆盖，不能据此
   截断1970+地震目录；当前断层快照也不能假装在历史起报时已经可用。
2. `figure_02_catalog_sample_funnel_episode_sizes.png`：目录经研究区、时间、震级和episode口径后还剩
   多少样本，以及一个episode通常包含几个事件。
3. `figure_03_fold_horizon_sample_waterlevels.png`：各外推年份、预测时长和震级档的事件/episode水位。
   `NA`表示目标窗尚未成熟，不能当成0。

## 尚未完成的科学阻点

空间源、研究区、50/25/12.5 km目标无关网格、39个有格原子块及受限文件身份已经核验；另以既有
D1目标盲定位器统计了39块在1970+全期和6个时间块的事件/固定锚点episode水位，共
{int(block_water["public_row_count"]):,}行，研究区内未定位事件为
{int(block_water["unlocated_event_count"]):,}。原始分块ID、坐标、cell_id和匿名映射均未公开。主科学证据
是时间向前；39块逐块留出并池化为全国OOF面是次级迁移压力测试。39→k模型选择映射状态仍为
`{spatial_folds["status"]}`，按经典方法复审结论延至S1任何模型成绩出现之前生成和封存。本脚本没有
为了凑齐产物自行发明分组。

## 产物边界

- JSON/CSV是可机器复核的输入、episode、issue、折次和空间身份账本；
- PNG使用本机常用字体 `{font_name}`；
- `artifact_manifest.json`记录每个产物的字节数和SHA-256；
- 所有数值库线程被固定为1，执行单进程，不使用GPU、不联网、不运行锁定测试。
"""


def build_artifact_manifest(output_root: Path, artifact_paths: Sequence[Path]) -> dict[str, object]:
    artifacts = []
    for path in sorted(artifact_paths, key=lambda item: item.relative_to(output_root).as_posix()):
        relative = path.relative_to(output_root).as_posix()
        artifacts.append(
            {
                "path": relative,
                "media_type": MEDIA_TYPES.get(path.suffix.casefold(), "application/octet-stream"),
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": "multitask_s0_score_blind_artifacts",
        "score_blind": True,
        "model_fitting_run": False,
        "locked_test_run": False,
        "network_accessed": False,
        "gpu_used": False,
        "process_policy": "single_process_numeric_libraries_single_thread",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "manifest_hash_scope": "canonical_manifest_content_excluding_content_sha256_and_self_file",
    }
    manifest["content_sha256"] = _content_hash(
        manifest, domain="seismoflux.multitask-s0-artifact-manifest.v1"
    )
    _assert_score_blind(manifest)
    return manifest


def run(*, config_path: Path, data_root: Path, output_root: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    if output_root == data_root or data_root in output_root.parents:
        raise S0RunnerError("output_root cannot be the processed-data root or one of its children")
    project_root = config_path.parents[1]
    config, config_payload = _load_contract(config_path)
    data_panels = _mapping(config.get("data_panels"), "data_panels")
    catalog_panel = _mapping(data_panels.get("catalog_modern"), "catalog_modern")
    catalog_relative = Path(str(catalog_panel.get("authoritative_processed_dataset")))
    catalog_path = data_root / catalog_relative
    if catalog_path.name != EXPECTED_CATALOG_FILENAME:
        raise S0RunnerError("the authoritative catalog filename changed")
    catalog = load_catalog_frame(catalog_path)
    maximum_origin = cast(pd.Timestamp, catalog["origin_time_utc"].max())
    maximum_available = cast(pd.Timestamp, catalog["available_at"].max())
    if maximum_origin != maximum_available:
        raise S0RunnerError(
            "the frozen truth cutoff is ambiguous because maximum "
            "origin/availability timestamps differ"
        )
    catalog_cutoff = maximum_origin
    horizons = tuple(
        int(value)
        for value in cast(
            Sequence[object],
            _mapping(config["time_semantics"], "time_semantics")["formal_horizons_days"],
        )
    )
    magnitude_bins = _magnitude_bins(config)
    folds = _catalog_folds(config)

    catalog_ledger = build_s0_ledger(
        catalog_path,
        catalog_start=CATALOG_START_LOCAL,
        catalog_cutoff=catalog_cutoff,
        folds=folds,
        horizons_days=horizons,
        magnitude_bins=magnitude_bins,
    )
    catalog_ledger["truth_cutoff_derivation"] = {
        "rule": "maximum_origin_time_equals_maximum_available_at_in_frozen_catalog",
        "maximum_origin_time_utc": maximum_origin.isoformat().replace("+00:00", "Z"),
        "maximum_available_at_utc": maximum_available.isoformat().replace("+00:00", "Z"),
    }
    catalog_ledger["content_sha256"] = _content_hash(
        catalog_ledger, domain="seismoflux.multitask-s0-catalog-runner-ledger.v1"
    )
    _assert_score_blind(catalog_ledger)

    input_ledger = build_authoritative_input_ledger(
        config=config,
        config_payload=config_payload,
        project_root=project_root,
        data_root=data_root,
        catalog=catalog,
    )
    spatial_ledger = build_spatial_identity_ledger(
        config=config,
        project_root=project_root,
        data_root=data_root,
        input_ledger=input_ledger,
        catalog=catalog,
        catalog_cutoff=catalog_cutoff,
        magnitude_bins=magnitude_bins,
    )
    episode_rows, size_rows = build_episode_rows(
        catalog=catalog,
        catalog_cutoff=catalog_cutoff,
        magnitude_bins=magnitude_bins,
    )
    fold_rows = flatten_fold_horizon_ledger(catalog_ledger)
    issue_rows = flatten_issue_ledger(catalog_ledger)
    coverage_rows = cast(Sequence[Mapping[str, object]], input_ledger["coverage_timeline"])
    atomic_block_rows = cast(
        Sequence[Mapping[str, object]],
        _mapping(
            spatial_ledger["anonymous_atomic_block_sample_water_levels"],
            "anonymous_atomic_block_sample_water_levels",
        )["rows"],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = [
        output_root / "authoritative_input_ledger.json",
        output_root / "catalog_sample_ledger.json",
        output_root / "spatial_identity_ledger.json",
        output_root / "coverage_timeline_ledger.csv",
        output_root / "episode_ledger.csv",
        output_root / "episode_size_distribution.csv",
        output_root / "fold_horizon_sample_ledger.csv",
        output_root / "issue_maturity_ledger.csv",
        output_root / "figure_01_data_coverage_timeline.png",
        output_root / "figure_02_catalog_sample_funnel_episode_sizes.png",
        output_root / "figure_03_fold_horizon_sample_waterlevels.png",
        output_root / "README.md",
        output_root / "atomic_block_sample_waterlevels.csv",
    ]
    _write_json(artifact_paths[0], input_ledger)
    _write_json(artifact_paths[1], catalog_ledger)
    _write_json(artifact_paths[2], spatial_ledger)
    _write_csv(artifact_paths[3], coverage_rows)
    _write_csv(artifact_paths[4], episode_rows)
    _write_csv(artifact_paths[5], size_rows)
    _write_csv(artifact_paths[6], fold_rows)
    _write_csv(artifact_paths[7], issue_rows)
    font_name = _configure_plot_style()
    render_coverage_timeline(coverage_rows, artifact_paths[8])
    render_sample_funnel_and_episode_sizes(
        catalog_ledger=catalog_ledger,
        size_rows=size_rows,
        path=artifact_paths[9],
    )
    render_fold_horizon_heatmap(fold_rows=fold_rows, path=artifact_paths[10])
    artifact_paths[11].write_text(
        _readme_text(
            catalog_ledger=catalog_ledger,
            spatial_ledger=spatial_ledger,
            font_name=font_name,
        ),
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(artifact_paths[12], atomic_block_rows)
    manifest = build_artifact_manifest(output_root, artifact_paths)
    manifest_path = output_root / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_root": str(output_root),
        "artifact_count_including_manifest": len(artifact_paths) + 1,
        "catalog_ledger_sha256": catalog_ledger["content_sha256"],
        "input_ledger_sha256": input_ledger["content_sha256"],
        "spatial_ledger_sha256": spatial_ledger["content_sha256"],
        "artifact_manifest_sha256": manifest["content_sha256"],
        "spatial_fold_status": SPATIAL_PENDING_STATUS,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build score-blind S0 input/sample ledgers and three static figures."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = run(
            config_path=arguments.config,
            data_root=arguments.data_root,
            output_root=arguments.output_root,
        )
    except (FileNotFoundError, NotADirectoryError, S0RunnerError, ValueError) as exc:
        print(f"S0 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
