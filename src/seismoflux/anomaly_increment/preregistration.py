"""Score-blind stage-4 preregistration builders.

This module deliberately has no earthquake-catalog dependency.  It materialises the
calendar, feature-column, random-stream, and construction-strata design that must be
frozen before stage 4 is allowed to read a target.  Construction geometry and cell
assignments are restricted local artifacts; only aggregate, coordinate-free evidence
is returned for public publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely import get_parts, normalize, set_precision, to_wkb
from shapely.geometry import LineString, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, polygonize_full, transform, unary_union
from shapely.strtree import STRtree

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.config import sha256_file
from seismoflux.data.common import canonical_json_bytes

STAGE4_PROTOCOL_VERSION: Final[str] = "0.4.1"
STAGE4_ROOT_SEED: Final[int] = 147
STAGE4_HORIZONS_DAYS: Final[tuple[int, ...]] = (7, 30, 90, 180, 365)
PRIMARY_MACRO_HORIZONS_DAYS: Final[tuple[int, ...]] = (7, 30, 90)
ROLLING_FOLD_COUNT: Final[int] = 3
STAGE3_QUERY_CELL_SIZE_KM: Final[float] = 25.0
STAGE3_QUERY_CELL_COUNT: Final[int] = 15_697
CONSTRUCTION_CONNECTOR_MAXIMUM_M: Final[float] = 100_000.0
CONSTRUCTION_TOPOLOGY_PRECISION_M: Final[float] = 1.0
CONSTRUCTION_PRECISION_SENSITIVITY_M: Final[tuple[float, ...]] = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
)
CONSTRUCTION_SOURCE_LICENSE: Final[str] = "unknown_no_redistribution"
LOCAL_MAPPING_FILENAME: Final[str] = "construction_zone_cell_mapping.parquet"
LOCAL_ENTITY_MAPPING_FILENAME: Final[str] = "construction_zone_entity_mapping.parquet"
LOCAL_CONNECTORS_FILENAME: Final[str] = "construction_zone_connectors.json"
LOCAL_ZONES_FILENAME: Final[str] = "construction_zones.parquet"
CHINA_STANDARD_TIME: Final[timezone] = timezone(timedelta(hours=8))

JsonObject: TypeAlias = dict[str, object]
PartitionName: TypeAlias = Literal["development", "validation"]
RandomPurpose: TypeAlias = Literal["bootstrap", "space_permutation", "time_permutation"]
RandomPartitionRole: TypeAlias = Literal["assessment", "fit", "joint"]

STAGE4_RANDOM_EVALUATION_IDS: Final[tuple[str, ...]] = (
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
)
STAGE4_RANDOM_PAIRING_SCOPE: Final[str] = (
    "shared_across_model_variants_magnitude_bins_all_five_horizons_and_metrics"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _string_sequence(value: object, *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence of strings")
    result = tuple(_string(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(payload: Mapping[str, object]) -> JsonObject:
    """Return a public JSON object with a self-excluding canonical content hash."""

    body = dict(payload)
    body.pop("content_sha256", None)
    body["content_sha256"] = _sha256_json(body)
    return body


def verify_content_sha256(payload: Mapping[str, object]) -> bool:
    """Verify the self-excluding canonical hash used by public stage-4 manifests."""

    expected = payload.get("content_sha256")
    if not isinstance(expected, str):
        return False
    body = dict(payload)
    del body["content_sha256"]
    return expected == _sha256_json(body)


def protocol_design_sha256(protocol: Mapping[str, object]) -> str:
    """Hash the protocol design while excluding generated-manifest digest back-references."""

    normalized = json.loads(json.dumps(dict(protocol), ensure_ascii=False, allow_nan=False))
    generated = _mapping(normalized.get("generated_manifests"), label="generated_manifests")
    for manifest_id, raw_entry in generated.items():
        entry = dict(_mapping(raw_entry, label=f"generated_manifests.{manifest_id}"))
        entry.pop("sha256", None)
        cast(dict[str, object], normalized["generated_manifests"])[manifest_id] = entry
    return _sha256_json(normalized)


def build_random_input_seal(
    protocol: Mapping[str, object],
    *,
    fold_manifest: Mapping[str, object],
    feature_manifest: Mapping[str, object],
    spatial_manifest: Mapping[str, object],
) -> JsonObject:
    """Bind typed RNG streams to every non-circular frozen stage-4 design component."""

    for label, manifest in (
        ("fold", fold_manifest),
        ("feature", feature_manifest),
        ("spatial", spatial_manifest),
    ):
        if not verify_content_sha256(manifest):
            raise ValueError(f"{label} manifest content hash is invalid")
    inputs = _mapping(protocol.get("inputs"), label="inputs")
    background_model = _mapping(
        inputs.get("background_model_manifest"),
        label="inputs.background_model_manifest",
    )
    stage3_store = _mapping(
        inputs.get("stage3_feature_store"),
        label="inputs.stage3_feature_store",
    )
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    scoring_freeze = _mapping(
        freeze.get("scoring_code_freeze"),
        label="freeze.scoring_code_freeze",
    )
    components: JsonObject = {
        "accepted_background_artifact_id": background_model.get("artifact_id"),
        "accepted_stage3_bundle_identity_sha256": stage3_store.get("bundle_identity_sha256"),
        "expected_scoring_code_tag": scoring_freeze.get("expected_tag"),
        "feature_manifest_content_sha256": feature_manifest.get("content_sha256"),
        "fold_manifest_content_sha256": fold_manifest.get("content_sha256"),
        "normalized_protocol_design_sha256": protocol_design_sha256(protocol),
        "protocol_version": STAGE4_PROTOCOL_VERSION,
        "spatial_local_artifacts": spatial_manifest.get("local_artifacts"),
        "spatial_manifest_content_sha256": spatial_manifest.get("content_sha256"),
    }
    return with_content_sha256(components)


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(cast(str, key) for key in value),
            *(nested for child in value.values() for nested in _nested_keys(child)),
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return {nested for child in value for nested in _nested_keys(child)}
    return set()


def validate_stage4_protocol_bundle(
    protocol: Mapping[str, object],
    *,
    fold_manifest: Mapping[str, object],
    feature_manifest: Mapping[str, object],
    randomness_manifest: Mapping[str, object],
    spatial_manifest: Mapping[str, object],
) -> JsonObject:
    """Fail closed on cross-file stage-4 preregistration inconsistencies."""

    expected_root_keys = {
        "alarm_area_diagnostic",
        "background",
        "blueprint",
        "compute",
        "evaluation",
        "experiment_family",
        "features",
        "forbidden_result_fields_before_freeze",
        "forecast",
        "freeze",
        "frozen_on",
        "gates",
        "generated_manifests",
        "hypotheses",
        "inputs",
        "integration",
        "locked_test",
        "magnitude",
        "model",
        "preprocessing",
        "protocol_version",
        "publication",
        "randomness",
        "schema_version",
        "spatial_permutation_topology",
        "split",
        "stage",
        "status",
    }
    if set(protocol) != expected_root_keys:
        raise ValueError("stage-4 protocol root keys changed without a protocol revision")
    if protocol.get("protocol_version") != STAGE4_PROTOCOL_VERSION:
        raise ValueError("stage-4 protocol version is not 0.4.1")
    if protocol.get("status") != "preregistered_before_any_stage4_target_score":
        raise ValueError("stage-4 protocol is not score-blind preregistration")

    manifests = {
        "feature": feature_manifest,
        "fold": fold_manifest,
        "randomness": randomness_manifest,
        "spatial": spatial_manifest,
    }
    for label, manifest in manifests.items():
        if not verify_content_sha256(manifest):
            raise ValueError(f"{label} manifest content hash is invalid")
    forbidden_result_fields = set(
        _string_sequence(
            protocol.get("forbidden_result_fields_before_freeze"),
            label="forbidden_result_fields_before_freeze",
        )
    )
    for label, manifest in manifests.items():
        escaped = forbidden_result_fields & _nested_keys(manifest)
        if escaped:
            raise ValueError(f"{label} manifest contains pre-score result fields: {escaped}")

    fold_horizons = _mapping(fold_manifest.get("horizons"), label="fold.horizons")
    if set(fold_horizons) != {str(value) for value in STAGE4_HORIZONS_DAYS}:
        raise ValueError("fold manifest horizon set changed")
    if fold_manifest.get("model_fit_authority") != (
        "joint_macro_rolling_folds_and_formal_validation_fit_only"
    ):
        raise ValueError("fold manifest model-fit authority is ambiguous")
    raw_folds = fold_manifest.get("joint_macro_rolling_folds")
    if not isinstance(raw_folds, Sequence) or isinstance(raw_folds, str | bytes):
        raise TypeError("joint macro folds must be a sequence")
    folds = tuple(
        _mapping(item, label=f"joint_macro_rolling_folds[{index}]")
        for index, item in enumerate(raw_folds)
    )
    if len(folds) != ROLLING_FOLD_COUNT:
        raise ValueError("stage 4 requires exactly three joint macro folds")
    previous_end: str | None = None
    for index, fold in enumerate(folds, start=1):
        if fold.get("fold_index") != index:
            raise ValueError("joint macro fold indices must be 1, 2, 3")
        if fold.get("model_fit_scope") != "one_per_variant_shared_across_all_horizons":
            raise ValueError("a joint macro fold permits per-horizon refitting")
        assessment = _mapping(fold.get("assessment_band"), label="assessment_band")
        start = _string(assessment.get("start_exclusive_utc"), label="assessment start")
        end = _string(assessment.get("end_inclusive_utc"), label="assessment end")
        if previous_end is not None and start <= previous_end:
            raise ValueError("joint macro assessment target bands overlap")
        previous_end = end
        pools = _mapping(
            fold.get("time_permutation_feature_pools"),
            label="time_permutation_feature_pools",
        )
        fit_pool = set(_string_sequence(pools.get("fit"), label="time fit pool"))
        assessment_pool = set(
            _string_sequence(pools.get("assessment"), label="time assessment pool")
        )
        if fit_pool & assessment_pool or pools.get("pool_crossing_forbidden") is not True:
            raise ValueError("time-placebo fit and assessment pools are not isolated")
    for horizon in fold_horizons.values():
        if _mapping(horizon, label="horizon").get("assessment_only_not_model_fit") is not True:
            raise ValueError("a horizon node is incorrectly authorized to fit a model")

    feature_sets = _mapping(feature_manifest.get("feature_sets"), label="feature_sets")
    if set(feature_sets) != {"coverage_only", "dynamic", "snapshot"}:
        raise ValueError("stage-4 feature-set variants changed")
    if feature_manifest.get("preprocessing_contract") != protocol.get("preprocessing"):
        raise ValueError("feature manifest preprocessing contract differs from the protocol")
    design_outputs: dict[str, set[str]] = {}
    for variant, raw_definition in feature_sets.items():
        definition = _mapping(raw_definition, label=f"feature_sets.{variant}")
        if definition.get("null_reason_validity_and_sample_count_predictors_forbidden") is not True:
            raise ValueError("quality companions escaped into the design matrix")
        raw_design = definition.get("design_columns")
        if not isinstance(raw_design, Sequence) or isinstance(raw_design, str | bytes):
            raise TypeError("design_columns must be a sequence")
        outputs: set[str] = set()
        for raw_column in raw_design:
            column = _mapping(raw_column, label="design column")
            role = column.get("role")
            if role not in {"value", "missing_indicator"}:
                raise ValueError("design column role is not frozen")
            output = _string(column.get("output_column"), label="design output column")
            if output in outputs:
                raise ValueError("design output columns are duplicated")
            outputs.add(output)
            if column.get("zero_scale_action") != (
                "coefficient_fixed_zero_only_when_training_min_equals_max"
            ):
                raise ValueError("design column may silently drop a nonconstant predictor")
        design_outputs[variant] = outputs
    if not (
        design_outputs["coverage_only"] < design_outputs["snapshot"] < design_outputs["dynamic"]
    ):
        raise ValueError("feature designs are not strict nested variants")

    seal = build_random_input_seal(
        protocol,
        fold_manifest=fold_manifest,
        feature_manifest=feature_manifest,
        spatial_manifest=spatial_manifest,
    )
    if randomness_manifest.get("frozen_input_seal_sha256") != seal.get("content_sha256"):
        raise ValueError("randomness is not bound to the frozen protocol and manifests")
    families = _mapping(randomness_manifest.get("families"), label="random families")
    if set(families) != {"bootstrap", "space_permutation", "time_permutation"}:
        raise ValueError("random family set changed")
    forbidden_seed_fields = {
        "horizon",
        "magnitude_bin",
        "metric",
        "model_variant",
        "worker_count",
    }
    for family_id, raw_family in families.items():
        family = _mapping(raw_family, label=f"random families.{family_id}")
        context_fields = set(_string_sequence(family.get("context_fields"), label="context_fields"))
        if context_fields & forbidden_seed_fields:
            raise ValueError("paired result dimensions escaped into a seed context")
        references = family.get("reference_vectors")
        if not isinstance(references, Sequence) or isinstance(references, str | bytes):
            raise TypeError("random reference vectors must be a sequence")
        expected_count = 12 if family_id == "bootstrap" else 24
        if len(references) != expected_count:
            raise ValueError(f"{family_id} reference-vector coverage is incomplete")

    input_gate = _mapping(spatial_manifest.get("input_hash_gate"), label="input_hash_gate")
    entity_gate = _mapping(
        spatial_manifest.get("entity_stratification_gate"),
        label="entity_stratification_gate",
    )
    local_artifacts = _mapping(
        spatial_manifest.get("local_artifacts"),
        label="local_artifacts",
    )
    if input_gate.get("passed") is not True or entity_gate.get("passed") is not True:
        raise ValueError("spatial input or entity-stratification gate failed")
    if spatial_manifest.get("spatial_placebo_implementation_authorized") is not True:
        raise ValueError("spatial manifest does not authorize score-blind implementation")
    if spatial_manifest.get("stage4_target_read_authorized") is not False:
        raise ValueError("spatial protocol manifest may not authorize target reads")
    if set(local_artifacts) != {
        "cell_mapping",
        "connectors",
        "entity_mapping",
        "zone_geometry",
    }:
        raise ValueError("spatial local artifact set changed")

    locked = _mapping(protocol.get("locked_test"), label="locked_test")
    if (
        locked.get("run") is not False
        or locked.get("result") is not None
        or locked.get("target_count") is not None
        or locked.get("outcomes_must_remain_unread") is not True
    ):
        raise ValueError("locked-test prohibition changed")
    return with_content_sha256(
        {
            "manifest_count": len(manifests),
            "protocol_design_sha256": protocol_design_sha256(protocol),
            "random_input_seal_sha256": seal.get("content_sha256"),
            "stage4_target_read_authorized": False,
            "target_read_count": 0,
            "validated": True,
        }
    )


def load_json_mapping(path: Path) -> JsonObject:
    """Load one JSON mapping from an explicitly injected path."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return dict(_mapping(value, label=str(path)))


def write_public_manifest_atomic(path: Path, payload: Mapping[str, object]) -> str:
    """Atomically write a canonical, content-hashed public manifest and return its file hash."""

    if not verify_content_sha256(payload):
        raise ValueError("public manifest has a missing or invalid content_sha256")
    serialized = canonical_json_bytes(dict(payload)) + b"\n"
    _write_bytes_atomic(path, serialized)
    return sha256_file(path)


@dataclass(frozen=True, slots=True)
class Exposure:
    """One target-window exposure defined only by an issue date and horizon."""

    partition: PartitionName
    horizon_days: int
    issue_date_local: date

    def __post_init__(self) -> None:
        if self.partition not in {"development", "validation"}:
            raise ValueError("exposure partition must be development or validation")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("exposure horizon is not frozen for stage 4")

    @property
    def identifier(self) -> str:
        return f"{self.partition}-h{self.horizon_days:03d}-{self.issue_date_local.isoformat()}"

    @property
    def target_end_inclusive_local(self) -> date:
        return self.issue_date_local + timedelta(days=self.horizon_days)

    @property
    def issue_time_local(self) -> datetime:
        return datetime.combine(
            self.issue_date_local,
            time.min,
            tzinfo=CHINA_STANDARD_TIME,
        )

    @property
    def issue_time_utc(self) -> datetime:
        return self.issue_time_local.astimezone(UTC)

    @property
    def target_end_utc(self) -> datetime:
        return self.issue_time_utc + timedelta(days=self.horizon_days)

    def as_mapping(self) -> JsonObject:
        return {
            "id": self.identifier,
            "issue_date_local": self.issue_date_local.isoformat(),
            "issue_time_local": self.issue_time_local.isoformat(),
            "issue_time_utc": self.issue_time_utc.isoformat().replace("+00:00", "Z"),
            "target_end_inclusive_local": self.target_end_inclusive_local.isoformat(),
            "target_end_inclusive_time_local": (
                self.issue_time_local + timedelta(days=self.horizon_days)
            ).isoformat(),
            "target_end_inclusive_utc": self.target_end_utc.isoformat().replace("+00:00", "Z"),
            "target_start_exclusive_local": self.issue_date_local.isoformat(),
            "target_start_exclusive_utc": self.issue_time_utc.isoformat().replace("+00:00", "Z"),
        }


def _parse_dates(value: object, *, label: str) -> tuple[date, ...]:
    if isinstance(value, str):
        raw_dates = value.split()
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        raw_dates = list(value)
    else:
        raise TypeError(f"{label} must be a date list or whitespace-delimited date string")
    parsed: list[date] = []
    for index, raw in enumerate(raw_dates):
        text = _string(raw, label=f"{label}[{index}]")
        try:
            parsed.append(date.fromisoformat(text))
        except ValueError as exc:
            raise ValueError(f"{label}[{index}] is not an ISO local date") from exc
    result = tuple(parsed)
    if not result or result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique, increasing dates")
    return result


def _partition_exposures(
    background_manifest: Mapping[str, object],
    *,
    partition: PartitionName,
    horizon_days: int,
) -> tuple[Exposure, ...]:
    partitions = _mapping(background_manifest.get("partitions"), label="partitions")
    partition_payload = _mapping(partitions.get(partition), label=f"partitions.{partition}")
    actual_dates = _parse_dates(
        partition_payload.get("actual_issue_dates_local"),
        label=f"partitions.{partition}.actual_issue_dates_local",
    )
    declared_actual_count = _integer(
        partition_payload.get("actual_issue_date_count"),
        label=f"partitions.{partition}.actual_issue_date_count",
        minimum=1,
    )
    if declared_actual_count != len(actual_dates):
        raise ValueError(f"{partition} actual issue-date count mismatch")
    non_overlapping = _mapping(
        partition_payload.get("non_overlapping_exposures"),
        label=f"partitions.{partition}.non_overlapping_exposures",
    )
    horizon_payload = _mapping(
        non_overlapping.get(str(horizon_days)),
        label=f"partitions.{partition}.non_overlapping_exposures.{horizon_days}",
    )
    dates = _parse_dates(
        horizon_payload.get("issue_dates_local"),
        label=(
            f"partitions.{partition}.non_overlapping_exposures.{horizon_days}.issue_dates_local"
        ),
    )
    count = _integer(
        horizon_payload.get("count"),
        label=f"partitions.{partition}.non_overlapping_exposures.{horizon_days}.count",
        minimum=1,
    )
    if count != len(dates):
        raise ValueError(f"{partition} {horizon_days}-day exposure count mismatch")
    if not set(dates).issubset(actual_dates):
        raise ValueError(f"{partition} {horizon_days}-day exposures are not actual issue dates")
    exposures = tuple(Exposure(partition, horizon_days, item) for item in dates)
    for left, right in pairwise(exposures):
        if right.issue_date_local < left.target_end_inclusive_local:
            raise ValueError(
                f"{partition} {horizon_days}-day exposures overlap contrary to the source manifest"
            )
    return exposures


def _balanced_block_sizes(item_count: int, block_count: int) -> tuple[int, ...]:
    if item_count < block_count or block_count < 1:
        raise ValueError("balanced blocks require at least one item per block")
    quotient, remainder = divmod(item_count, block_count)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(block_count))


def _rolling_folds(exposures: tuple[Exposure, ...]) -> tuple[JsonObject, ...]:
    if len(exposures) < 2:
        return ()
    assessment_fold_count = min(ROLLING_FOLD_COUNT, len(exposures) - 1)
    sizes = _balanced_block_sizes(len(exposures), assessment_fold_count + 1)
    blocks: list[tuple[Exposure, ...]] = []
    start = 0
    for size in sizes:
        blocks.append(exposures[start : start + size])
        start += size

    folds: list[JsonObject] = []
    for fold_index, assessment in enumerate(blocks[1:], start=1):
        assessment_start = assessment[0].issue_date_local
        prior = tuple(item for block in blocks[:fold_index] for item in block)
        training = tuple(
            item for item in prior if item.target_end_inclusive_local < assessment_start
        )
        if not training:
            continue
        train_end = max(item.target_end_inclusive_local for item in training)
        if train_end >= assessment_start:
            raise AssertionError("rolling-fold purge failed its strict temporal boundary")
        folds.append(
            {
                "assessment_exposure_ids": [item.identifier for item in assessment],
                "assessment_target_start_local": assessment_start.isoformat(),
                "fold_index": fold_index,
                "training_exposure_ids": [item.identifier for item in training],
                "train_target_end_local": train_end.isoformat(),
            }
        )
    return tuple(folds)


def _joint_macro_fit_folds(
    development_by_horizon: Mapping[int, tuple[Exposure, ...]],
    anomaly_issue_dates_local: tuple[date, ...],
) -> tuple[JsonObject, ...]:
    """Build three disjoint 90-day target bands with one shared 7-day fit per band."""

    seven_day = development_by_horizon[7]
    ninety_day = development_by_horizon[90]
    if len(ninety_day) != ROLLING_FOLD_COUNT + 1:
        raise ValueError("the frozen development calendar must expose four 90-day blocks")

    folds: list[JsonObject] = []
    previous_band_end: date | None = None
    for fold_index, band_exposure in enumerate(ninety_day[1:], start=1):
        band_start = band_exposure.issue_date_local
        band_end = band_exposure.target_end_inclusive_local
        if previous_band_end is not None and band_start <= previous_band_end:
            raise ValueError("joint macro assessment bands must be strictly disjoint")
        previous_band_end = band_end

        fit = tuple(
            exposure for exposure in seven_day if exposure.target_end_inclusive_local < band_start
        )
        if not fit:
            raise ValueError("each joint macro fold requires a non-empty shared 7-day fit")

        assessment_by_horizon: JsonObject = {}
        for horizon_days in PRIMARY_MACRO_HORIZONS_DAYS:
            contained = tuple(
                exposure
                for exposure in development_by_horizon[horizon_days]
                if exposure.issue_date_local >= band_start
                and exposure.target_end_inclusive_local <= band_end
            )
            if not contained:
                raise ValueError(
                    f"joint macro fold {fold_index} has no {horizon_days}-day assessment"
                )
            assessment_by_horizon[str(horizon_days)] = [
                exposure.identifier for exposure in contained
            ]

        fit_end = max(exposure.target_end_utc for exposure in fit)
        fit_last_issue = max(exposure.issue_date_local for exposure in fit)
        assessment_last_issue = max(
            exposure.issue_date_local
            for horizon_days in PRIMARY_MACRO_HORIZONS_DAYS
            for exposure in development_by_horizon[horizon_days]
            if exposure.issue_date_local >= band_start
            and exposure.target_end_inclusive_local <= band_end
        )
        fit_feature_issues = tuple(
            item for item in anomaly_issue_dates_local if item <= fit_last_issue
        )
        assessment_feature_issues = tuple(
            item
            for item in anomaly_issue_dates_local
            if fit_last_issue < item <= assessment_last_issue
        )
        if not fit_feature_issues or not assessment_feature_issues:
            raise ValueError("time-placebo feature pools must both be non-empty")
        band_start_utc = band_exposure.issue_time_utc
        band_end_utc = band_exposure.target_end_utc
        if fit_end >= band_start_utc:
            raise AssertionError("shared 7-day fit overlaps its joint macro assessment band")
        folds.append(
            {
                "assessment_band": {
                    "end_inclusive_local": band_end.isoformat(),
                    "end_inclusive_utc": band_end_utc.isoformat().replace("+00:00", "Z"),
                    "start_exclusive_local": band_start.isoformat(),
                    "start_exclusive_utc": band_start_utc.isoformat().replace("+00:00", "Z"),
                },
                "assessment_exposure_ids_by_horizon": assessment_by_horizon,
                "assessment_horizons_days": list(PRIMARY_MACRO_HORIZONS_DAYS),
                "fold_index": fold_index,
                "fit_exposure_ids_7d": [exposure.identifier for exposure in fit],
                "fit_scope_id": f"development-fold-{fold_index}",
                "fit_target_end_inclusive_utc": fit_end.isoformat().replace("+00:00", "Z"),
                "model_fit_id_pattern": (f"stage4-development-fold-{fold_index}/{{model_variant}}"),
                "model_fit_scope": "one_per_variant_shared_across_all_horizons",
                "preprocessor_fit_scope": "same_7d_fit_exposures_as_model",
                "time_permutation_feature_pools": {
                    "assessment": [
                        f"anomaly-issue-{item.isoformat()}" for item in assessment_feature_issues
                    ],
                    "fit": [f"anomaly-issue-{item.isoformat()}" for item in fit_feature_issues],
                    "pool_crossing_forbidden": True,
                    "pseudo_history_concatenation": "permuted_fit_then_permuted_assessment",
                    "trajectory_lookback_before_first_available_issue": "unavailable_not_zero",
                },
            }
        )
    return tuple(folds)


def build_exposure_preregistration(
    background_manifest: Mapping[str, object],
    *,
    anomaly_issue_dates_local: tuple[date, ...],
    horizons_days: tuple[int, ...] = STAGE4_HORIZONS_DAYS,
    rolling_fold_count: int = ROLLING_FOLD_COUNT,
) -> JsonObject:
    """Build explicit score-blind development/validation exposures and rolling folds."""

    if rolling_fold_count != ROLLING_FOLD_COUNT:
        raise ValueError("stage 4 freezes exactly three requested development rolling folds")
    if horizons_days != STAGE4_HORIZONS_DAYS:
        raise ValueError("stage-4 horizons must remain exactly 7, 30, 90, 180, and 365 days")
    if (
        not anomaly_issue_dates_local
        or anomaly_issue_dates_local != tuple(sorted(anomaly_issue_dates_local))
        or len(anomaly_issue_dates_local) != len(set(anomaly_issue_dates_local))
    ):
        raise ValueError("anomaly issue dates must be a non-empty unique increasing tuple")

    development_by_horizon: dict[int, tuple[Exposure, ...]] = {}
    validation_by_horizon: dict[int, tuple[Exposure, ...]] = {}
    for horizon_days in horizons_days:
        development_by_horizon[horizon_days] = _partition_exposures(
            background_manifest,
            partition="development",
            horizon_days=horizon_days,
        )
        validation_by_horizon[horizon_days] = _partition_exposures(
            background_manifest,
            partition="validation",
            horizon_days=horizon_days,
        )

    all_scored_issue_dates = {
        exposure.issue_date_local
        for partition in (development_by_horizon, validation_by_horizon)
        for exposures in partition.values()
        for exposure in exposures
    }
    if not all_scored_issue_dates.issubset(set(anomaly_issue_dates_local)):
        raise ValueError("every scored issue must exist in the accepted anomaly issue calendar")

    joint_folds = _joint_macro_fit_folds(
        development_by_horizon,
        anomaly_issue_dates_local,
    )
    horizons: JsonObject = {}
    for horizon_days in horizons_days:
        development = development_by_horizon[horizon_days]
        validation = validation_by_horizon[horizon_days]
        primary_horizon = horizon_days in PRIMARY_MACRO_HORIZONS_DAYS
        horizons[str(horizon_days)] = {
            "assessment_only_not_model_fit": True,
            "development_exposures": [item.as_mapping() for item in development],
            "role": "primary" if primary_horizon else "evidence_insufficient",
            "status": (
                "candidate_calendar_mature_pending_label_sufficiency"
                if primary_horizon
                else "evidence_insufficient_no_random_split"
            ),
            "validation_exposures": [item.as_mapping() for item in validation],
        }

    full_development_fit = development_by_horizon[7]
    full_development_fit_end = max(item.target_end_utc for item in full_development_fit)
    validation_start = min(
        item.issue_time_utc
        for horizon_days in STAGE4_HORIZONS_DAYS
        for item in validation_by_horizon[horizon_days]
    )
    if full_development_fit_end >= validation_start:
        raise AssertionError("full development fit overlaps formal validation")
    formal_fit_last_issue = max(item.issue_date_local for item in full_development_fit)
    formal_assessment_last_issue = max(
        item.issue_date_local
        for horizon_days in STAGE4_HORIZONS_DAYS
        for item in validation_by_horizon[horizon_days]
    )
    formal_fit_feature_issues = tuple(
        item for item in anomaly_issue_dates_local if item <= formal_fit_last_issue
    )
    formal_assessment_feature_issues = tuple(
        item
        for item in anomaly_issue_dates_local
        if formal_fit_last_issue < item <= formal_assessment_last_issue
    )
    if not formal_fit_feature_issues or not formal_assessment_feature_issues:
        raise ValueError("formal time-placebo feature pools must both be non-empty")

    payload: JsonObject = {
        "formal_validation_fit": {
            "assessment_exposure_ids_by_horizon": {
                str(horizon_days): [item.identifier for item in validation_by_horizon[horizon_days]]
                for horizon_days in STAGE4_HORIZONS_DAYS
            },
            "fit_exposure_ids_7d": [item.identifier for item in full_development_fit],
            "fit_scope_id": "full-development-before-validation",
            "fit_target_end_inclusive_utc": full_development_fit_end.isoformat().replace(
                "+00:00", "Z"
            ),
            "model_fit_id_pattern": "stage4-formal-validation/{model_variant}",
            "model_fit_scope": "one_per_variant_shared_across_all_five_horizons",
            "preprocessor_fit_scope": "same_7d_fit_exposures_as_model",
            "time_permutation_feature_pools": {
                "assessment": [
                    f"anomaly-issue-{item.isoformat()}" for item in formal_assessment_feature_issues
                ],
                "fit": [f"anomaly-issue-{item.isoformat()}" for item in formal_fit_feature_issues],
                "pool_crossing_forbidden": True,
                "pseudo_history_concatenation": "permuted_fit_then_permuted_assessment",
                "trajectory_lookback_before_first_available_issue": "unavailable_not_zero",
            },
            "validation_refit_forbidden": True,
        },
        "horizons": horizons,
        "joint_macro_rolling_folds": list(joint_folds),
        "joint_macro_target_bands_mutually_disjoint": True,
        "model_fit_authority": "joint_macro_rolling_folds_and_formal_validation_fit_only",
        "protocol_version": STAGE4_PROTOCOL_VERSION,
        "random_split_forbidden": True,
        "rolling_fold_rule": "three_disjoint_90d_target_bands_with_expanding_shared_7d_fit",
        "schema_version": 1,
        "shared_model_fit_exposure_horizon_days": 7,
        "single_parameter_vector_per_variant_and_fit_scope": True,
        "target_window_rule": "(issue_time,issue_time+h]",
        "training_target_end_must_be_strictly_before_assessment_target_start": True,
    }
    return with_content_sha256(payload)


def _feature_definitions(
    feature_dictionary: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    raw = feature_dictionary.get("features")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or not raw:
        raise ValueError("feature dictionary must contain a non-empty features list")
    result = tuple(
        _mapping(item, label=f"feature_dictionary.features[{index}]")
        for index, item in enumerate(raw)
    )
    names = tuple(
        _string(item.get("name"), label=f"feature_dictionary.features[{index}].name")
        for index, item in enumerate(result)
    )
    if len(names) != len(set(names)):
        raise ValueError("feature dictionary contains duplicate logical feature names")
    return result


def _storage_columns(definition: Mapping[str, object]) -> tuple[str, ...]:
    return _string_sequence(
        definition.get("storage_value_columns"),
        label=f"feature {definition.get('name')}.storage_value_columns",
    )


def _selected_value_columns(
    definition: Mapping[str, object],
    *,
    logical_name: str,
    feature_group: str,
    features_config: Mapping[str, object],
) -> tuple[str, ...]:
    columns = _storage_columns(definition)
    producer = _string(definition.get("producer"), label=f"feature {logical_name}.producer")
    if producer == "protocol_v1":
        if len(columns) != 1:
            raise ValueError(
                f"protocol feature {logical_name} must have exactly one storage column"
            )
        return columns

    if feature_group == "trajectory_signal":
        kernel = _string(
            features_config.get("trajectory_kernel"),
            label="features.trajectory_kernel",
        )
        if kernel != "closed_ball":
            raise ValueError("stage-4 trajectory kernel must remain closed_ball")
        scale = _integer(
            features_config.get("trajectory_scale_km"),
            label="features.trajectory_scale_km",
            minimum=1,
        )
        sources = _string_sequence(
            features_config.get("trajectory_source_series"),
            label="features.trajectory_source_series",
        )
        expected = tuple(f"radius_{scale}km__{source}__{logical_name}" for source in sources)
    else:
        kernel = _string(
            features_config.get("confirmatory_kernel"),
            label="features.confirmatory_kernel",
        )
        scale = _integer(
            features_config.get("confirmatory_scale_km"),
            label="features.confirmatory_scale_km",
            minimum=1,
        )
        prefix = "radius" if kernel == "closed_ball" else kernel
        expected = (f"{prefix}_{scale}km__{logical_name}",)

    missing = [column for column in expected if column not in columns]
    if missing:
        raise ValueError(f"configured storage columns are absent for {logical_name}: {missing}")
    return expected


def _quality_columns(definition: Mapping[str, object], value_column: str) -> tuple[str, ...]:
    quality = _mapping(
        definition.get("quality_companions"),
        label=f"feature {definition.get('name')}.quality_companions",
    )
    ordered_suffix_keys = (
        "validity_suffix",
        "sample_count_suffix",
        "null_reason_code_suffix",
    )
    result: list[str] = []
    for key in ordered_suffix_keys:
        raw_suffix = quality.get(key)
        if raw_suffix is None:
            continue
        suffix = _string(raw_suffix, label=f"feature {definition.get('name')}.{key}")
        if not suffix.startswith("__"):
            raise ValueError(f"feature quality suffix must begin with '__': {key}")
        result.append(f"{value_column}{suffix}")
    return tuple(result)


def select_feature_storage_columns(
    feature_dictionary: Mapping[str, object],
    selection_config: Mapping[str, object],
) -> JsonObject:
    """Resolve frozen logical groups to an exact, typed design-matrix contract."""

    definitions = _feature_definitions(feature_dictionary)
    by_name = {_string(item.get("name"), label="feature.name"): item for item in definitions}
    raw_features = selection_config.get("features", selection_config)
    features_config = _mapping(raw_features, label="features")
    group_names = (
        "coverage_controls",
        "snapshot_signal",
        "trajectory_signal",
    )
    selected_by_group: dict[str, tuple[str, ...]] = {}
    group_value_columns: dict[str, tuple[str, ...]] = {}
    group_value_pairs: dict[str, tuple[tuple[str, str], ...]] = {}
    group_all_columns: dict[str, tuple[str, ...]] = {}
    for group_name in group_names:
        logical_names = _string_sequence(
            features_config.get(group_name),
            label=f"features.{group_name}",
        )
        selected_by_group[group_name] = logical_names
        values: list[str] = []
        value_pairs: list[tuple[str, str]] = []
        all_columns: list[str] = []
        for logical_name in logical_names:
            if logical_name not in by_name:
                raise ValueError(f"configured logical feature is absent: {logical_name}")
            definition = by_name[logical_name]
            selected_values = _selected_value_columns(
                definition,
                logical_name=logical_name,
                feature_group=group_name,
                features_config=features_config,
            )
            for value_column in selected_values:
                values.append(value_column)
                value_pairs.append((logical_name, value_column))
                all_columns.append(value_column)
                all_columns.extend(_quality_columns(definition, value_column))
        if len(all_columns) != len(set(all_columns)):
            raise ValueError(f"feature group {group_name} resolves duplicate storage columns")
        group_value_columns[group_name] = tuple(values)
        group_value_pairs[group_name] = tuple(value_pairs)
        group_all_columns[group_name] = tuple(all_columns)

    preprocessing = _mapping(
        selection_config.get("preprocessing"),
        label="preprocessing",
    )
    transform_by_logical = _mapping(
        preprocessing.get("transform_by_logical_feature"),
        label="preprocessing.transform_by_logical_feature",
    )
    allowed_transforms = {
        "asinh_signed",
        "identity_binary",
        "identity_finite",
        "log1p_nonnegative",
    }

    selected_logical_names = tuple(
        name for group_name in group_names for name in selected_by_group[group_name]
    )
    if set(transform_by_logical) != set(selected_logical_names):
        missing = sorted(set(selected_logical_names) - set(transform_by_logical))
        extra = sorted(set(transform_by_logical) - set(selected_logical_names))
        raise ValueError(
            "preprocessing transform map must exactly cover selected logical features: "
            f"missing={missing}, extra={extra}"
        )
    normalized_transforms: dict[str, str] = {}
    for logical_name in selected_logical_names:
        transform_name = _string(
            transform_by_logical[logical_name],
            label=f"preprocessing.transform_by_logical_feature.{logical_name}",
        )
        if transform_name not in allowed_transforms:
            raise ValueError(f"unsupported frozen transform for {logical_name}: {transform_name}")
        normalized_transforms[logical_name] = transform_name

    model_sets = _mapping(features_config.get("model_sets"), label="features.model_sets")
    expected_model_sets = ("coverage_only", "snapshot", "dynamic")
    feature_sets: JsonObject = {}
    resolved_model_columns: dict[str, tuple[str, ...]] = {}
    for model_set in expected_model_sets:
        groups = _string_sequence(
            model_sets.get(model_set),
            label=f"features.model_sets.{model_set}",
        )
        unknown = [group for group in groups if group not in group_all_columns]
        if unknown:
            raise ValueError(f"model set {model_set} references unknown groups: {unknown}")
        logical_names = tuple(name for group in groups for name in selected_by_group[group])
        value_columns = tuple(column for group in groups for column in group_value_columns[group])
        model_value_pairs = tuple(pair for group in groups for pair in group_value_pairs[group])
        audit_columns = tuple(
            sorted(
                column
                for group in groups
                for column in group_all_columns[group]
                if column not in set(value_columns)
            )
        )
        if len(logical_names) != len(set(logical_names)):
            raise ValueError(f"model set {model_set} repeats logical features")
        if len(value_columns) != len(set(value_columns)):
            raise ValueError(f"model set {model_set} repeats value columns")

        design_columns: list[JsonObject] = []
        for logical_name, source_column in model_value_pairs:
            transform_name = normalized_transforms[logical_name]
            is_binary_value = transform_name == "identity_binary"
            design_columns.extend(
                (
                    {
                        "categorical_encoding": None,
                        "clipping": (
                            "none" if is_binary_value else "training_q0.005_q0.995_numpy_linear"
                        ),
                        "imputation": "training_median_after_transform",
                        "logical_feature": logical_name,
                        "output_column": f"value__{source_column}",
                        "penalty_factor": 1.0,
                        "role": "value",
                        "source_column": source_column,
                        "standardization": (
                            "none_binary"
                            if is_binary_value
                            else (
                                "training_median_center_scale_fallback_"
                                "1.4826_MAD_then_IQR_div_1.349_then_population_SD"
                            )
                        ),
                        "statistics_row_weighting": (
                            "unweighted_equal_rows_over_all_fit_issue_x_positive_area_25km_cells"
                        ),
                        "transform": transform_name,
                        "zero_scale_action": (
                            "coefficient_fixed_zero_only_when_training_min_equals_max"
                        ),
                    },
                    {
                        "categorical_encoding": None,
                        "clipping": "none",
                        "imputation": "none",
                        "logical_feature": logical_name,
                        "output_column": f"missing__{source_column}",
                        "penalty_factor": 1.0,
                        "role": "missing_indicator",
                        "source_column": source_column,
                        "standardization": "none",
                        "statistics_row_weighting": "not_applicable",
                        "transform": "is_null_boolean",
                        "zero_scale_action": (
                            "coefficient_fixed_zero_only_when_training_min_equals_max"
                        ),
                    },
                )
            )
        design_outputs = [cast(str, item["output_column"]) for item in design_columns]
        if len(design_outputs) != len(set(design_outputs)):
            raise ValueError(f"model set {model_set} repeats design output columns")
        feature_sets[model_set] = {
            "audit_only_quality_columns": list(audit_columns),
            "audit_only_quality_columns_sha256": _sha256_json(list(audit_columns)),
            "design_columns": design_columns,
            "design_columns_sha256": _sha256_json(design_columns),
            "design_output_columns": design_outputs,
            "design_output_columns_sha256": _sha256_json(design_outputs),
            "logical_features": list(logical_names),
            "null_reason_validity_and_sample_count_predictors_forbidden": True,
            "source_value_columns": list(value_columns),
            "source_value_columns_sha256": _sha256_json(list(value_columns)),
        }
        resolved_model_columns[model_set] = tuple(design_outputs)

    coverage_columns = set(resolved_model_columns["coverage_only"])
    snapshot_columns = set(resolved_model_columns["snapshot"])
    dynamic_columns = set(resolved_model_columns["dynamic"])
    if not coverage_columns < snapshot_columns < dynamic_columns:
        raise ValueError("feature sets must be strict coverage < snapshot < dynamic nests")

    configured_forbidden = _string_sequence(
        features_config.get("forbidden"),
        label="features.forbidden",
    )
    forbidden = tuple(
        sorted(
            {
                *configured_forbidden,
                "earthquake_target",
                "epicenter",
                "fault",
                "future",
                "label",
                "manual",
                "manual_prediction",
                "source_duration",
            }
        )
    )
    fault = _mapping(features_config.get("fault_interaction"), label="features.fault_interaction")
    if fault.get("status") not in {"deferred_to_stage5", "deferred_to_stage_5"}:
        raise ValueError("fault interaction must remain deferred to stage 5")
    if fault.get("forbidden_in_stage4_model") is not True:
        raise ValueError("fault interaction must be forbidden in the stage-4 model")

    allowed_producers = {"local_coverage_v1", "protocol_v1", "spatial_v1", "trajectory_v1"}
    forbidden_source_fragments = {
        "earthquake",
        "epicenter",
        "fault",
        "future",
        "human prediction",
        "manual",
        "source_duration",
        "target",
    }
    verified_definition_count = 0
    for logical_name in selected_logical_names:
        definition = by_name[logical_name]
        producer = _string(
            definition.get("producer"),
            label=f"feature {logical_name}.producer",
        )
        if producer not in allowed_producers:
            raise ValueError(f"feature {logical_name} uses a non-whitelisted producer: {producer}")
        causal_sources = _string_sequence(
            definition.get("causal_sources"),
            label=f"feature {logical_name}.causal_sources",
        )
        raw_source_output_field = definition.get("source_output_field")
        source_output_field = (
            ""
            if raw_source_output_field is None
            else _string(
                raw_source_output_field,
                label=f"feature {logical_name}.source_output_field",
            )
        )
        inspected_values = (
            logical_name,
            producer,
            source_output_field,
            *causal_sources,
            *_storage_columns(definition),
        )
        folded = "\n".join(inspected_values).casefold()
        matched = sorted(token for token in forbidden_source_fragments if token in folded)
        if matched:
            raise ValueError(
                f"feature {logical_name} matches forbidden causal-source tokens: {matched}"
            )
        verified_definition_count += 1

    payload: JsonObject = {
        "fault_features": {
            "forbidden_in_stage4_model": True,
            "status": "deferred_to_stage_5",
        },
        "feature_dictionary_identity_sha256": _sha256_json(dict(feature_dictionary)),
        "feature_sets": feature_sets,
        "forbidden_source_match_count": 0,
        "forbidden_tokens": list(forbidden),
        "protocol_version": STAGE4_PROTOCOL_VERSION,
        "preprocessing_contract": dict(preprocessing),
        "schema_version": 1,
        "manual_prediction_feature_use_count": 0,
        "selection": {
            "confirmatory_kernel": features_config.get("confirmatory_kernel"),
            "confirmatory_scale_km": features_config.get("confirmatory_scale_km"),
            "trajectory_kernel": features_config.get("trajectory_kernel"),
            "trajectory_scale_km": features_config.get("trajectory_scale_km"),
            "trajectory_source_series": list(
                _string_sequence(
                    features_config.get("trajectory_source_series"),
                    label="features.trajectory_source_series",
                )
            ),
        },
        "source_duration_feature_use_count": 0,
        "verified_selected_definition_count": verified_definition_count,
    }
    return with_content_sha256(payload)


@dataclass(frozen=True, slots=True)
class Stage4SeedContext:
    """A typed, worker-count-independent stage-4 random stream identity."""

    purpose: RandomPurpose
    evaluation_id: str
    partition_role: RandomPartitionRole
    replicate_index: int
    frozen_input_seal_sha256: str
    issue_id: str | None = None
    construction_stratum_id: str | None = None
    root_seed: int = STAGE4_ROOT_SEED

    def __post_init__(self) -> None:
        if self.root_seed != STAGE4_ROOT_SEED:
            raise ValueError("stage-4 root_seed is frozen at 147")
        if self.purpose not in {"bootstrap", "space_permutation", "time_permutation"}:
            raise ValueError("unsupported stage-4 random purpose")
        if len(self.frozen_input_seal_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.frozen_input_seal_sha256
        ):
            raise ValueError("frozen_input_seal_sha256 must be a lowercase SHA-256")
        if self.evaluation_id not in STAGE4_RANDOM_EVALUATION_IDS:
            raise ValueError("evaluation_id is not a frozen stage-4 evaluation scope")
        maximum = 1_999 if self.purpose == "bootstrap" else 999
        index = _integer(self.replicate_index, label="replicate_index")
        if index > maximum:
            raise ValueError(f"{self.purpose} replicate_index must be <= {maximum}")

        if self.purpose == "bootstrap":
            if self.partition_role != "joint":
                raise ValueError("bootstrap partition_role must be joint")
            if self.issue_id is not None or self.construction_stratum_id is not None:
                raise ValueError("bootstrap context may not contain issue or zone fields")
        elif self.purpose == "time_permutation":
            if self.partition_role not in {"fit", "assessment"}:
                raise ValueError("time permutation role must be fit or assessment")
            if self.issue_id is not None or self.construction_stratum_id is not None:
                raise ValueError("time permutation context may not contain issue or zone fields")
        else:
            if self.partition_role not in {"fit", "assessment"}:
                raise ValueError("space permutation role must be fit or assessment")
            _string(self.issue_id, label="issue_id")
            _string(self.construction_stratum_id, label="construction_stratum_id")

    def as_mapping(self) -> JsonObject:
        payload: JsonObject = {
            "evaluation_id": self.evaluation_id,
            "frozen_input_seal_sha256": self.frozen_input_seal_sha256,
            "pairing_scope": STAGE4_RANDOM_PAIRING_SCOPE,
            "partition_role": self.partition_role,
            "protocol_version": STAGE4_PROTOCOL_VERSION,
            "purpose": self.purpose,
            "replication_index": self.replicate_index,
            "schema_version": 1,
        }
        if self.purpose == "space_permutation":
            payload["construction_stratum_id"] = cast(str, self.construction_stratum_id)
            payload["issue_id"] = cast(str, self.issue_id)
        return payload

    def digest(self) -> bytes:
        payload = b"\x00".join(
            (
                b"seismoflux-stage4",
                str(self.root_seed).encode("ascii"),
                canonical_json_bytes(self.as_mapping()),
            )
        )
        return hashlib.sha256(payload).digest()

    def entropy(self) -> int:
        return int.from_bytes(self.digest()[:16], byteorder="big", signed=False)

    def generator(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.entropy()))

    def reference_uint64(self) -> int:
        value = self.generator().integers(0, 2**64, dtype=np.uint64)
        return int(value)


def _reference_vector(context: Stage4SeedContext) -> JsonObject:
    return {
        "context": context.as_mapping(),
        "context_sha256": hashlib.sha256(canonical_json_bytes(context.as_mapping())).hexdigest(),
        "reference_permutation_n5": context.generator().permutation(5).tolist(),
        "reference_permutation_n7": context.generator().permutation(7).tolist(),
        "reference_resample_indices_n5": context.generator()
        .integers(0, 5, size=5, dtype=np.int64)
        .tolist(),
        "reference_uint64": context.reference_uint64(),
    }


def build_randomness_manifest(*, frozen_input_seal_sha256: str) -> JsonObject:
    """Freeze typed random families without caller-defined namespace strings."""

    bootstrap_references = [
        _reference_vector(
            Stage4SeedContext(
                purpose="bootstrap",
                evaluation_id=evaluation_id,
                partition_role="joint",
                replicate_index=index,
                frozen_input_seal_sha256=frozen_input_seal_sha256,
            )
        )
        for evaluation_id in STAGE4_RANDOM_EVALUATION_IDS
        for index in (0, 1, 1_999)
    ]
    time_references = [
        _reference_vector(
            Stage4SeedContext(
                purpose="time_permutation",
                evaluation_id=evaluation_id,
                partition_role=role,
                replicate_index=index,
                frozen_input_seal_sha256=frozen_input_seal_sha256,
            )
        )
        for evaluation_id in STAGE4_RANDOM_EVALUATION_IDS
        for role in cast(tuple[RandomPartitionRole, ...], ("fit", "assessment"))
        for index in (0, 1, 999)
    ]
    space_references = [
        _reference_vector(
            Stage4SeedContext(
                purpose="space_permutation",
                evaluation_id=evaluation_id,
                partition_role=role,
                replicate_index=index,
                frozen_input_seal_sha256=frozen_input_seal_sha256,
                issue_id="reference-issue-only",
                construction_stratum_id="reference-zone-inside-only",
            )
        )
        for evaluation_id in STAGE4_RANDOM_EVALUATION_IDS
        for role in cast(tuple[RandomPartitionRole, ...], ("fit", "assessment"))
        for index in (0, 1, 999)
    ]
    payload: JsonObject = {
        "context_canonicalization": "canonical_json_utf8_sort_keys_no_nan",
        "derivation": (
            "sha256('seismoflux-stage4' NUL root_seed NUL canonical_context_json)"
            "_then_first_16_bytes_big_endian_entropy"
        ),
        "direct_integer_subseeds_forbidden": True,
        "frozen_input_seal_sha256": frozen_input_seal_sha256,
        "generator": "numpy.random.PCG64",
        "paired_result_dimensions_not_seed_fields": [
            "model_variant",
            "magnitude_bin",
            "horizon",
            "metric",
        ],
        "pairing_scope": STAGE4_RANDOM_PAIRING_SCOPE,
        "protocol_version": STAGE4_PROTOCOL_VERSION,
        "root_seed": STAGE4_ROOT_SEED,
        "schema_version": 1,
        "families": {
            "bootstrap": {
                "context_fields": [
                    "schema_version",
                    "protocol_version",
                    "frozen_input_seal_sha256",
                    "purpose",
                    "evaluation_id",
                    "partition_role",
                    "pairing_scope",
                    "replication_index",
                ],
                "event_membership_horizons_days": list(STAGE4_HORIZONS_DAYS),
                "full_model_refit_each_replication": False,
                "strata": "magnitude_bin_x_nonempty_five_horizon_membership_signature",
                "within_stratum_sample_size": "same_as_original_unique_physical_event_count",
                "mapping_algorithm": (
                    "sort_strata_by_magnitude_then_five_bit_signature_sort_events_by_event_id_"
                    "then_Generator_PCG64_integers_0_n_size_n_as_donor_indices"
                ),
                "per_horizon_marginal_event_count_preserved": True,
                "singleton_stratum_action": "deterministic_self_resample",
                "physical_event_cluster_weights_shared_across_all_dimensions": True,
                "reference_vectors": bootstrap_references,
                "replications": 2_000,
            },
            "space_permutation": {
                "context_fields": [
                    "schema_version",
                    "protocol_version",
                    "frozen_input_seal_sha256",
                    "purpose",
                    "evaluation_id",
                    "partition_role",
                    "issue_id",
                    "construction_stratum_id",
                    "pairing_scope",
                    "replication_index",
                ],
                "anomaly_increment_coefficients_and_preprocessor_refit_each_replication": True,
                "frozen_background_and_target_rate_head_reused_each_replication": True,
                "mapping_algorithm": (
                    "sort_recipient_and_donor_rows_by_state_id_then_"
                    "recipient_i_gets_donor_Generator_PCG64_permutation_n_i"
                ),
                "mapping_shared_across_all_dimensions": True,
                "reference_vectors": space_references,
                "replications": 1_000,
            },
            "time_permutation": {
                "context_fields": [
                    "schema_version",
                    "protocol_version",
                    "frozen_input_seal_sha256",
                    "purpose",
                    "evaluation_id",
                    "partition_role",
                    "pairing_scope",
                    "replication_index",
                ],
                "fit_and_assessment_mappings_separate": True,
                "anomaly_increment_coefficients_and_preprocessor_refit_each_replication": True,
                "frozen_background_and_target_rate_head_reused_each_replication": True,
                "mapping_algorithm": (
                    "sort_recipient_and_donor_issues_by_fold_manifest_issue_id_then_"
                    "recipient_i_gets_donor_Generator_PCG64_permutation_n_i"
                ),
                "mapping_shared_across_all_dimensions": True,
                "reference_vectors": time_references,
                "replications": 1_000,
            },
        },
        "worker_count_invariant": True,
    }
    return with_content_sha256(payload)


def _geometry_wkb(geometry: BaseGeometry) -> bytes:
    canonical = cast(BaseGeometry, normalize(geometry))
    return cast(
        bytes,
        to_wkb(
            canonical,
            hex=False,
            output_dimension=2,
            byte_order=1,
            include_srid=False,
        ),
    )


def _geometry_sha256(geometry: BaseGeometry) -> str:
    return hashlib.sha256(_geometry_wkb(geometry)).hexdigest()


def _canonical_lines(lines: Sequence[LineString]) -> tuple[LineString, ...]:
    by_wkb: dict[bytes, LineString] = {}
    for line in lines:
        if line.is_empty or not line.is_valid or line.length <= 0.0:
            continue
        canonical = cast(LineString, normalize(line))
        by_wkb[_geometry_wkb(canonical)] = canonical
    if not by_wkb:
        raise ValueError("construction linework is empty after projection and clipping")
    return tuple(by_wkb[key] for key in sorted(by_wkb))


def _line_parts(geometry: BaseGeometry) -> tuple[LineString, ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type in {"LineString", "LinearRing"}:
        line = LineString(geometry.coords)
        return (line,) if line.length > 0.0 else ()
    result: list[LineString] = []
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            result.extend(_line_parts(cast(BaseGeometry, part)))
    return tuple(result)


def _read_gmt_segments(path: Path) -> tuple[LineString, ...]:
    segments: list[LineString] = []
    coordinates: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal coordinates
        deduplicated: list[tuple[float, float]] = []
        for coordinate in coordinates:
            if not deduplicated or coordinate != deduplicated[-1]:
                deduplicated.append(coordinate)
        if len(deduplicated) >= 2:
            line = LineString(deduplicated)
            if line.is_empty or not line.is_valid or line.length <= 0.0:
                raise ValueError(f"invalid GMT line segment in {path}")
            segments.append(line)
        elif coordinates:
            raise ValueError(f"GMT segment has fewer than two distinct coordinates: {path}")
        coordinates = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith(">"):
                flush()
                continue
            fields = text.split()
            if len(fields) < 2:
                raise ValueError(f"invalid GMT coordinate at {path}:{line_number}")
            try:
                longitude = float(fields[0])
                latitude = float(fields[1])
            except ValueError as exc:
                raise ValueError(f"invalid GMT coordinate at {path}:{line_number}") from exc
            if (
                not math.isfinite(longitude)
                or not math.isfinite(latitude)
                or not -180.0 <= longitude <= 180.0
                or not -90.0 <= latitude <= 90.0
            ):
                raise ValueError(f"out-of-range GMT coordinate at {path}:{line_number}")
            coordinates.append((longitude, latitude))
    flush()
    if not segments:
        raise ValueError(f"GMT source contains no valid segments: {path}")
    return tuple(segments)


def _load_study_area(path: Path) -> BaseGeometry:
    payload = load_json_mapping(path)
    geometry_type = payload.get("type")
    if geometry_type == "Feature":
        raw_geometry = payload.get("geometry")
        geometry = cast(BaseGeometry, shape(raw_geometry))
    elif geometry_type == "FeatureCollection":
        raw_features = payload.get("features")
        if not isinstance(raw_features, Sequence) or isinstance(raw_features, str | bytes):
            raise ValueError("study-area FeatureCollection has no feature list")
        geometries = [
            cast(
                BaseGeometry,
                shape(_mapping(item, label="study-area feature").get("geometry")),
            )
            for item in raw_features
        ]
        geometry = cast(BaseGeometry, unary_union(geometries))
    else:
        geometry = cast(BaseGeometry, shape(payload))
    if (
        geometry.geom_type not in {"Polygon", "MultiPolygon"}
        or geometry.is_empty
        or not geometry.is_valid
        or geometry.area <= 0.0
    ):
        raise ValueError("study-area geometry must be a non-empty valid WGS84 polygon")
    return geometry


def _project_and_clip_lines(
    lines_wgs84: Sequence[LineString],
    *,
    study_area_projected: BaseGeometry,
) -> tuple[LineString, ...]:
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_user_input(EQUAL_AREA_CRS),
        always_xy=True,
    )
    projected: list[LineString] = []
    for line in lines_wgs84:
        projected_line = cast(BaseGeometry, transform(transformer.transform, line))
        clipped = cast(BaseGeometry, projected_line.intersection(study_area_projected))
        projected.extend(_line_parts(clipped))
    return _canonical_lines(projected)


@dataclass(frozen=True, slots=True)
class _Connector:
    geometry: LineString
    source_line_sha256: str
    endpoint_index: int
    target_kind: str

    def as_mapping(self) -> JsonObject:
        coordinates = tuple((float(x), float(y)) for x, y in self.geometry.coords)
        return {
            "coordinates_equal_area_m": [[x, y] for x, y in coordinates],
            "endpoint_index": self.endpoint_index,
            "length_m": float(self.geometry.length),
            "source_line_sha256": self.source_line_sha256,
            "target_kind": self.target_kind,
        }


def _endpoint_connectors(
    lines: tuple[LineString, ...],
    *,
    study_area_projected: BaseGeometry,
    maximum_distance_m: float,
) -> tuple[_Connector, ...]:
    boundary = cast(BaseGeometry, study_area_projected.boundary)
    connectors: dict[bytes, _Connector] = {}
    boundary_tolerance = 1.0e-7
    for line_index, line in enumerate(lines):
        line_hash = _geometry_sha256(line)
        endpoint_coordinates = (line.coords[0], line.coords[-1])
        other_lines = tuple(item for index, item in enumerate(lines) if index != line_index)
        other_union = cast(BaseGeometry, unary_union(other_lines)) if other_lines else None
        for endpoint_index, coordinate in enumerate(endpoint_coordinates):
            endpoint = Point(float(coordinate[0]), float(coordinate[1]))
            if endpoint.distance(boundary) <= boundary_tolerance:
                continue
            candidates: list[tuple[float, str, BaseGeometry]] = [
                (float(endpoint.distance(boundary)), "study_area_boundary", boundary)
            ]
            if other_union is not None and not other_union.is_empty:
                candidates.append(
                    (float(endpoint.distance(other_union)), "other_construction_line", other_union)
                )
            distance_m, target_kind, target = min(
                candidates,
                key=lambda item: (item[0], item[1], _geometry_sha256(item[2])),
            )
            if distance_m <= boundary_tolerance:
                continue
            if not math.isfinite(distance_m) or distance_m > maximum_distance_m:
                raise ValueError(
                    "internal construction-line endpoint has no other line or boundary "
                    f"within {maximum_distance_m / 1_000.0:g} km"
                )
            _, nearest = nearest_points(endpoint, target)
            connector_line = LineString((endpoint.coords[0], nearest.coords[0]))
            if connector_line.length <= boundary_tolerance:
                continue
            connector = _Connector(
                geometry=cast(LineString, normalize(connector_line)),
                source_line_sha256=line_hash,
                endpoint_index=endpoint_index,
                target_kind=target_kind,
            )
            connectors[_geometry_wkb(connector.geometry)] = connector
    return tuple(connectors[key] for key in sorted(connectors))


@dataclass(frozen=True, slots=True)
class _Topology:
    precision_m: float
    zones: tuple[BaseGeometry, ...]
    cuts_count: int
    dangles_count: int
    invalid_count: int
    zone_set_sha256: str


def _part_count(geometry: BaseGeometry) -> int:
    if geometry.is_empty:
        return 0
    return len(get_parts(geometry))


def _polygonize_network(
    lines: tuple[LineString, ...],
    connectors: tuple[_Connector, ...],
    *,
    study_area_projected: BaseGeometry,
    precision_m: float,
) -> _Topology:
    exact_network = cast(
        BaseGeometry,
        unary_union(
            [
                cast(BaseGeometry, study_area_projected.boundary),
                *lines,
                *(item.geometry for item in connectors),
            ]
        ),
    )
    precision_network = cast(BaseGeometry, set_precision(exact_network, precision_m))
    polygons, cuts, dangles, invalid = polygonize_full(list(get_parts(precision_network)))
    polygon_parts = tuple(
        cast(BaseGeometry, item)
        for item in get_parts(polygons)
        if not cast(BaseGeometry, item).is_empty and cast(BaseGeometry, item).area > 0.0
    )
    precision_study_area = cast(BaseGeometry, set_precision(study_area_projected, precision_m))
    zones = tuple(
        zone for zone in polygon_parts if precision_study_area.covers(zone.representative_point())
    )
    if not zones:
        raise ValueError("construction topology did not create any in-study-area zones")
    zone_union = cast(BaseGeometry, unary_union(zones))
    uncovered_area = float(precision_study_area.symmetric_difference(zone_union).area)
    coverage_tolerance_m2 = max(1.0, float(precision_study_area.area) * 1.0e-10)
    if uncovered_area > coverage_tolerance_m2:
        raise ValueError(
            "construction zones do not exactly cover the precision-reduced study area: "
            f"difference={uncovered_area:g} m2"
        )
    zone_ids = tuple(sorted(_geometry_sha256(zone) for zone in zones))
    if len(zone_ids) != len(set(zone_ids)):
        raise ValueError("construction topology produced duplicate normalized zone geometries")
    return _Topology(
        precision_m=precision_m,
        zones=tuple(sorted(zones, key=_geometry_sha256)),
        cuts_count=_part_count(cast(BaseGeometry, cuts)),
        dangles_count=_part_count(cast(BaseGeometry, dangles)),
        invalid_count=_part_count(cast(BaseGeometry, invalid)),
        zone_set_sha256=_sha256_json(list(zone_ids)),
    )


@dataclass(frozen=True, slots=True)
class _Stage3GridRows:
    grid_id: str
    cell_ids: tuple[str, ...]
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    query_x_m: NDArray[np.float64]
    query_y_m: NDArray[np.float64]


def _first_stage3_grid_rows(path: Path, *, expected_cell_count: int) -> _Stage3GridRows:
    columns = (
        "issue_index",
        "grid_id",
        "equal_area_crs",
        "cell_size_km",
        "cell_id",
        "cell_row",
        "cell_column",
        "query_x_m",
        "query_y_m",
    )
    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_row_groups < 1:
        raise ValueError("stage-3 feature store has no row groups")
    table = parquet_file.read_row_group(0, columns=list(columns))
    if table.num_rows != expected_cell_count:
        raise ValueError(
            f"stage-3 first row group has {table.num_rows} cells; expected {expected_cell_count}"
        )
    issue_indices = table["issue_index"].to_pylist()
    if set(issue_indices) != {0}:
        raise ValueError("stage-3 first row group must be issue_index zero")
    grid_ids = tuple(cast(str, item) for item in table["grid_id"].to_pylist())
    if len(set(grid_ids)) != 1:
        raise ValueError("stage-3 first row group must have one fixed grid_id")
    crs_values = tuple(cast(str, item) for item in table["equal_area_crs"].to_pylist())
    if set(crs_values) != {EQUAL_AREA_CRS}:
        raise ValueError("stage-3 first row group does not use the frozen Albers CRS")
    cell_sizes = np.asarray(table["cell_size_km"].to_numpy(), dtype=np.float64)
    if not np.all(cell_sizes == STAGE3_QUERY_CELL_SIZE_KM):
        raise ValueError("stage-3 first row group is not the frozen 25 km grid")
    cell_ids = tuple(cast(str, item) for item in table["cell_id"].to_pylist())
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("stage-3 first row group contains duplicate cell IDs")
    rows = np.asarray(table["cell_row"].to_numpy(), dtype=np.int64)
    cell_columns = np.asarray(table["cell_column"].to_numpy(), dtype=np.int64)
    x_m = np.asarray(table["query_x_m"].to_numpy(), dtype=np.float64)
    y_m = np.asarray(table["query_y_m"].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(x_m)) or not np.all(np.isfinite(y_m)):
        raise ValueError("stage-3 query coordinates must be finite")
    order = np.lexsort((np.asarray(cell_ids), cell_columns, rows))
    if not np.array_equal(order, np.arange(expected_cell_count)):
        raise ValueError("stage-3 first row group is not in frozen row/column/cell order")
    return _Stage3GridRows(grid_ids[0], cell_ids, rows, cell_columns, x_m, y_m)


@dataclass(frozen=True, slots=True)
class _CellAssignments:
    zone_ids: tuple[str, ...]
    boundary_tie_count: int
    precision_snap_count: int


def _deterministic_nearest_zone(
    point: Point,
    zones: tuple[BaseGeometry, ...],
    zone_ids: tuple[str, ...],
) -> tuple[int, int]:
    distances = np.asarray([zone.distance(point) for zone in zones], dtype=np.float64)
    if not np.all(np.isfinite(distances)):
        raise ValueError("construction-zone nearest distances must be finite")
    minimum = float(np.min(distances))
    tolerance = max(1.0e-9, abs(minimum) * 1.0e-12)
    tied = [
        index
        for index, distance_m in enumerate(distances)
        if abs(float(distance_m) - minimum) <= tolerance
    ]
    chosen = min(tied, key=lambda index: zone_ids[index])
    return chosen, len(tied) - 1


def _assign_cells(
    grid: _Stage3GridRows,
    zones: tuple[BaseGeometry, ...],
    *,
    precision_m: float,
) -> _CellAssignments:
    zone_ids = tuple(_geometry_sha256(zone) for zone in zones)
    tree = STRtree(zones)
    assignments: list[str] = []
    boundary_ties = 0
    precision_snaps = 0
    maximum_snap_m = math.sqrt(2.0) * precision_m + 1.0e-9
    for x_m, y_m in zip(grid.query_x_m, grid.query_y_m, strict=True):
        point = Point(float(x_m), float(y_m))
        candidate_indices = cast(NDArray[np.int64], tree.query(point))
        covering = sorted(
            int(index) for index in candidate_indices if zones[int(index)].covers(point)
        )
        if covering:
            if len(covering) > 1:
                boundary_ties += 1
            assignments.append(min(zone_ids[index] for index in covering))
            continue
        nearest_index, _ = _deterministic_nearest_zone(point, zones, zone_ids)
        distance_m = float(zones[nearest_index].distance(point))
        if not math.isfinite(distance_m) or distance_m > maximum_snap_m:
            raise ValueError("a fixed stage-3 query cell cannot be assigned to a construction zone")
        precision_snaps += 1
        assignments.append(zone_ids[nearest_index])
    if len(assignments) != len(grid.cell_ids):
        raise AssertionError("construction-zone assignment changed the fixed grid size")
    return _CellAssignments(tuple(assignments), boundary_ties, precision_snaps)


@dataclass(frozen=True, slots=True)
class _EntityAssignments:
    state_ids: tuple[str, ...]
    anomaly_ids: tuple[str, ...]
    issue_times_utc: tuple[datetime, ...]
    construction_stratum_ids: tuple[str, ...]
    coordinate_pair_sha256: tuple[str, ...]
    outside_study_area: tuple[bool, ...]
    total_entity_state_count: int
    spatially_ineligible_count: int
    boundary_tie_count: int
    precision_snap_count: int
    nearest_distance_tie_count: int


def _assign_spatial_entity_states(
    state_history_path: Path,
    zones: tuple[BaseGeometry, ...],
    *,
    study_area_projected: BaseGeometry,
    transformer: Transformer,
    precision_m: float,
) -> _EntityAssignments:
    columns = (
        "state_row_kind",
        "state_id",
        "issue_time_utc",
        "anomaly_id",
        "spatial_eligible",
        "longitude",
        "latitude",
    )
    table = pq.read_table(state_history_path, columns=list(columns))
    records = table.to_pylist()
    entity_records = [record for record in records if record["state_row_kind"] == "entity_state"]
    eligible = [record for record in entity_records if record["spatial_eligible"] is True]
    ineligible_count = len(entity_records) - len(eligible)
    if not eligible:
        raise ValueError("stage-3 state history has no spatially eligible anomaly entities")

    longitudes = np.asarray([record["longitude"] for record in eligible], dtype=np.float64)
    latitudes = np.asarray([record["latitude"] for record in eligible], dtype=np.float64)
    if not np.all(np.isfinite(longitudes)) or not np.all(np.isfinite(latitudes)):
        raise ValueError("spatially eligible anomaly entity has missing or non-finite coordinates")
    x_values, y_values = transformer.transform(longitudes, latitudes)
    x_m = np.asarray(x_values, dtype=np.float64)
    y_m = np.asarray(y_values, dtype=np.float64)
    if not np.all(np.isfinite(x_m)) or not np.all(np.isfinite(y_m)):
        raise ValueError("projected anomaly entity coordinates must be finite")

    zone_ids = tuple(_geometry_sha256(zone) for zone in zones)
    tree = STRtree(zones)
    stratum_ids: list[str] = []
    outside_flags: list[bool] = []
    boundary_ties = 0
    precision_snaps = 0
    nearest_distance_ties = 0
    coordinate_hashes: list[str] = []
    maximum_snap_m = math.sqrt(2.0) * precision_m + 1.0e-9
    for x_value, y_value in zip(x_m, y_m, strict=True):
        point = Point(float(x_value), float(y_value))
        outside = not study_area_projected.covers(point)
        candidate_indices = cast(NDArray[np.int64], tree.query(point))
        covering = sorted(
            int(index) for index in candidate_indices if zones[int(index)].covers(point)
        )
        if covering:
            if len(covering) > 1:
                boundary_ties += 1
            zone_id = min(zone_ids[index] for index in covering)
        else:
            nearest_index, tie_count = _deterministic_nearest_zone(point, zones, zone_ids)
            nearest_distance_ties += tie_count
            distance_m = float(zones[nearest_index].distance(point))
            if not outside and (not math.isfinite(distance_m) or distance_m > maximum_snap_m):
                raise ValueError(
                    "an inside-study anomaly entity cannot be assigned to a construction zone"
                )
            if not math.isfinite(distance_m):
                raise ValueError("anomaly entity has no finite nearest construction zone")
            if not outside:
                precision_snaps += 1
            zone_id = zone_ids[nearest_index]
        outside_flags.append(outside)
        stratum_ids.append(f"{zone_id}:{'outside' if outside else 'inside'}")
        coordinate_hashes.append(
            hashlib.sha256(
                np.asarray([x_value, y_value], dtype="<f8").tobytes(order="C")
            ).hexdigest()
        )

    state_ids = tuple(_string(record["state_id"], label="state_id") for record in eligible)
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("stage-3 spatial entity state IDs must be unique")
    anomaly_ids = tuple(_string(record["anomaly_id"], label="anomaly_id") for record in eligible)
    issue_times: list[datetime] = []
    for record in eligible:
        value = record["issue_time_utc"]
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("stage-3 entity issue_time_utc must be timezone-aware")
        issue_times.append(value.astimezone(UTC))
    return _EntityAssignments(
        state_ids=state_ids,
        anomaly_ids=anomaly_ids,
        issue_times_utc=tuple(issue_times),
        construction_stratum_ids=tuple(stratum_ids),
        coordinate_pair_sha256=tuple(coordinate_hashes),
        outside_study_area=tuple(outside_flags),
        total_entity_state_count=len(entity_records),
        spatially_ineligible_count=ineligible_count,
        boundary_tie_count=boundary_ties,
        precision_snap_count=precision_snaps,
        nearest_distance_tie_count=nearest_distance_ties,
    )


def _require_local_ignored_output(output_dir: Path, *, project_root: Path) -> Path:
    root = project_root.resolve()
    resolved = output_dir.resolve()
    allowed = (root / "data" / "interim").resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError("coordinate-bearing stage-4 artifacts must stay under data/interim")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_mapping_parquet(
    path: Path,
    *,
    grid: _Stage3GridRows,
    assignments: _CellAssignments,
) -> None:
    metadata = {
        b"seismoflux_contract": b"0.4.1-local-construction-zone-cell-mapping",
        b"seismoflux_license": CONSTRUCTION_SOURCE_LICENSE.encode("ascii"),
        b"seismoflux_publication": b"forbidden_contains_coordinates_and_per_cell_mapping",
    }
    table = pa.table(
        {
            "grid_id": pa.array([grid.grid_id] * len(grid.cell_ids), type=pa.string()),
            "cell_id": pa.array(grid.cell_ids, type=pa.string()),
            "cell_row": pa.array(grid.rows, type=pa.int64()),
            "cell_column": pa.array(grid.columns, type=pa.int64()),
            "query_x_m": pa.array(grid.query_x_m, type=pa.float64()),
            "query_y_m": pa.array(grid.query_y_m, type=pa.float64()),
            "construction_zone_id": pa.array(assignments.zone_ids, type=pa.string()),
        },
        metadata=metadata,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
        pq.write_table(
            table,
            temporary_name,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            version="2.6",
            data_page_version="1.0",
            row_group_size=table.num_rows,
        )
        check = pq.read_table(temporary_name)
        if not check.equals(table, check_metadata=True):
            raise RuntimeError("local construction-zone mapping changed during Parquet round trip")
        os.replace(temporary_name, path)
        temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_entity_mapping_parquet(path: Path, assignments: _EntityAssignments) -> None:
    metadata = {
        b"seismoflux_contract": b"0.4.1-local-construction-zone-entity-mapping",
        b"seismoflux_license": CONSTRUCTION_SOURCE_LICENSE.encode("ascii"),
        b"seismoflux_publication": b"forbidden_contains_per-entity-stratification",
    }
    table = pa.table(
        {
            "state_id": pa.array(assignments.state_ids, type=pa.string()),
            "anomaly_id": pa.array(assignments.anomaly_ids, type=pa.string()),
            "issue_time_utc": pa.array(
                assignments.issue_times_utc,
                type=pa.timestamp("us", tz="UTC"),
            ),
            "construction_stratum_id": pa.array(
                assignments.construction_stratum_ids,
                type=pa.string(),
            ),
            "coordinate_pair_sha256": pa.array(
                assignments.coordinate_pair_sha256,
                type=pa.string(),
            ),
            "outside_study_area": pa.array(assignments.outside_study_area, type=pa.bool_()),
        },
        metadata=metadata,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
        pq.write_table(
            table,
            temporary_name,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            version="2.6",
            data_page_version="1.0",
            row_group_size=table.num_rows,
        )
        check = pq.read_table(temporary_name)
        if not check.equals(table, check_metadata=True):
            raise RuntimeError("local entity-zone mapping changed during Parquet round trip")
        os.replace(temporary_name, path)
        temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_zones_parquet(path: Path, zones: Sequence[BaseGeometry]) -> None:
    ordered = tuple(sorted(((_geometry_sha256(zone), zone) for zone in zones), key=lambda x: x[0]))
    metadata = {
        b"seismoflux_contract": b"0.4.1-local-construction-zone-geometry",
        b"seismoflux_license": CONSTRUCTION_SOURCE_LICENSE.encode("ascii"),
        b"seismoflux_publication": b"forbidden_contains_restricted_geometry",
    }
    table = pa.table(
        {
            "construction_zone_id": pa.array([item[0] for item in ordered], type=pa.string()),
            "geometry_wkb_equal_area_m": pa.array(
                [_geometry_wkb(item[1]) for item in ordered],
                type=pa.binary(),
            ),
        },
        metadata=metadata,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
        pq.write_table(
            table,
            temporary_name,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            version="2.6",
            data_page_version="1.0",
            row_group_size=table.num_rows,
        )
        check = pq.read_table(temporary_name)
        if not check.equals(table, check_metadata=True):
            raise RuntimeError("local construction-zone geometry changed during Parquet round trip")
        os.replace(temporary_name, path)
        temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class ConstructionStrataResult:
    """Local restricted artifact paths plus the only public-safe aggregate summary."""

    mapping_parquet_path: Path
    entity_mapping_parquet_path: Path
    connectors_json_path: Path
    zones_parquet_path: Path
    public_summary: JsonObject


def build_construction_strata(
    *,
    l1_gmt_path: Path,
    l2_gmt_path: Path,
    study_area_path: Path,
    stage3_feature_store_path: Path,
    stage3_state_history_path: Path,
    local_output_dir: Path,
    project_root: Path,
    expected_l1_sha256: str,
    expected_l2_sha256: str,
    expected_study_area_sha256: str,
    expected_stage3_feature_store_sha256: str,
    expected_stage3_state_history_sha256: str,
    expected_cell_count: int = STAGE3_QUERY_CELL_COUNT,
    maximum_connector_distance_m: float = CONSTRUCTION_CONNECTOR_MAXIMUM_M,
) -> ConstructionStrataResult:
    """Build score-blind local construction strata and a coordinate-free public summary."""

    if expected_cell_count < 1:
        raise ValueError("expected_cell_count must be positive")
    if maximum_connector_distance_m != CONSTRUCTION_CONNECTOR_MAXIMUM_M:
        raise ValueError("stage-4 connector distance is frozen at 100 km")
    local_dir = _require_local_ignored_output(local_output_dir, project_root=project_root)
    source_paths = (l1_gmt_path, l2_gmt_path)
    for source in (
        *source_paths,
        study_area_path,
        stage3_feature_store_path,
        stage3_state_history_path,
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
    actual_hashes = {
        "construction_linework_l1": sha256_file(l1_gmt_path),
        "construction_linework_l2": sha256_file(l2_gmt_path),
        "stage3_feature_store": sha256_file(stage3_feature_store_path),
        "stage3_state_history": sha256_file(stage3_state_history_path),
        "study_area": sha256_file(study_area_path),
    }
    expected_hashes = {
        "construction_linework_l1": expected_l1_sha256,
        "construction_linework_l2": expected_l2_sha256,
        "stage3_feature_store": expected_stage3_feature_store_sha256,
        "stage3_state_history": expected_stage3_state_history_sha256,
        "study_area": expected_study_area_sha256,
    }
    for key, expected in expected_hashes.items():
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"{key} expected SHA-256 is invalid")
        if actual_hashes[key] != expected:
            raise ValueError(f"{key} SHA-256 does not match the frozen preregistration")
    all_hashes_match = actual_hashes == expected_hashes
    if not all_hashes_match:
        raise AssertionError("construction input hash gate did not fail closed")

    study_wgs84 = _load_study_area(study_area_path)
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(EQUAL_AREA_CRS), always_xy=True
    )
    study_projected = cast(BaseGeometry, transform(transformer.transform, study_wgs84))
    if study_projected.is_empty or not study_projected.is_valid or study_projected.area <= 0.0:
        raise ValueError("projected study-area geometry is invalid")
    l1_segments = _read_gmt_segments(l1_gmt_path)
    l2_segments = _read_gmt_segments(l2_gmt_path)
    source_segments = (*l1_segments, *l2_segments)
    lines = _project_and_clip_lines(source_segments, study_area_projected=study_projected)
    connectors = _endpoint_connectors(
        lines,
        study_area_projected=study_projected,
        maximum_distance_m=maximum_connector_distance_m,
    )
    sensitivity = tuple(
        _polygonize_network(
            lines,
            connectors,
            study_area_projected=study_projected,
            precision_m=precision_m,
        )
        for precision_m in CONSTRUCTION_PRECISION_SENSITIVITY_M
    )
    zone_counts = tuple(len(item.zones) for item in sensitivity)
    if len(set(zone_counts)) != 1:
        raise ValueError(
            f"construction-zone count changes across precision sensitivity: {zone_counts}"
        )
    primary = next(
        item for item in sensitivity if item.precision_m == CONSTRUCTION_TOPOLOGY_PRECISION_M
    )
    if primary.cuts_count or primary.dangles_count or primary.invalid_count:
        raise ValueError(
            "construction topology gate failed: "
            f"cuts={primary.cuts_count}, dangles={primary.dangles_count}, "
            f"invalid={primary.invalid_count}"
        )

    reordered_lines = tuple(reversed(lines))
    reordered_connectors = _endpoint_connectors(
        _canonical_lines(reordered_lines),
        study_area_projected=study_projected,
        maximum_distance_m=maximum_connector_distance_m,
    )
    reordered = _polygonize_network(
        _canonical_lines(reordered_lines),
        reordered_connectors,
        study_area_projected=study_projected,
        precision_m=CONSTRUCTION_TOPOLOGY_PRECISION_M,
    )
    reorder_stable = reordered.zone_set_sha256 == primary.zone_set_sha256
    if not reorder_stable:
        raise ValueError("construction-zone identity changes when input line order is reversed")

    grid = _first_stage3_grid_rows(
        stage3_feature_store_path,
        expected_cell_count=expected_cell_count,
    )
    assignments = _assign_cells(
        grid,
        primary.zones,
        precision_m=CONSTRUCTION_TOPOLOGY_PRECISION_M,
    )
    entity_assignments = _assign_spatial_entity_states(
        stage3_state_history_path,
        primary.zones,
        study_area_projected=study_projected,
        transformer=transformer,
        precision_m=CONSTRUCTION_TOPOLOGY_PRECISION_M,
    )
    mapping_path = local_dir / LOCAL_MAPPING_FILENAME
    entity_mapping_path = local_dir / LOCAL_ENTITY_MAPPING_FILENAME
    connectors_path = local_dir / LOCAL_CONNECTORS_FILENAME
    zones_path = local_dir / LOCAL_ZONES_FILENAME
    _write_mapping_parquet(mapping_path, grid=grid, assignments=assignments)
    _write_entity_mapping_parquet(entity_mapping_path, entity_assignments)
    _write_zones_parquet(zones_path, primary.zones)
    connector_payload = {
        "connector_count": len(connectors),
        "connectors": [item.as_mapping() for item in connectors],
        "coordinate_crs": EQUAL_AREA_CRS,
        "license_status": CONSTRUCTION_SOURCE_LICENSE,
        "maximum_connector_distance_m": maximum_connector_distance_m,
        "protocol_version": STAGE4_PROTOCOL_VERSION,
        "publication": "forbidden_contains_restricted_coordinates",
        "schema_version": 1,
    }
    _write_bytes_atomic(connectors_path, canonical_json_bytes(connector_payload) + b"\n")

    mapping_sha = sha256_file(mapping_path)
    entity_mapping_sha = sha256_file(entity_mapping_path)
    connectors_sha = sha256_file(connectors_path)
    zones_sha = sha256_file(zones_path)
    assignment_counts = Counter(assignments.zone_ids)
    entity_group_coordinates: dict[tuple[datetime, str], list[str]] = {}
    for issue_time, stratum_id, coordinate_hash in zip(
        entity_assignments.issue_times_utc,
        entity_assignments.construction_stratum_ids,
        entity_assignments.coordinate_pair_sha256,
        strict=True,
    ):
        entity_group_coordinates.setdefault((issue_time, stratum_id), []).append(coordinate_hash)
    entity_singleton_group_count = sum(
        len(coordinates) == 1 for coordinates in entity_group_coordinates.values()
    )
    entity_identical_coordinate_group_count = sum(
        len(coordinates) > 1 and len(set(coordinates)) == 1
        for coordinates in entity_group_coordinates.values()
    )
    entity_nontrivial_group_count = sum(
        len(coordinates) > 1 and len(set(coordinates)) > 1
        for coordinates in entity_group_coordinates.values()
    )
    entity_no_effect_state_count = sum(
        len(coordinates)
        for coordinates in entity_group_coordinates.values()
        if len(coordinates) == 1 or len(set(coordinates)) == 1
    )
    topology_passed = (
        primary.cuts_count == 0
        and primary.dangles_count == 0
        and primary.invalid_count == 0
        and len(set(zone_counts)) == 1
        and reorder_stable
        and len(assignments.zone_ids) == expected_cell_count
        and all_hashes_match
        and len(entity_assignments.state_ids) + entity_assignments.spatially_ineligible_count
        == entity_assignments.total_entity_state_count
    )
    singleton_grid_zone_count = sum(count == 1 for count in assignment_counts.values())
    fixed_grid_assignment = {
        "ambiguous_count": assignments.boundary_tie_count,
        "assigned_once_count": len(assignments.zone_ids),
        "query_cell_count": expected_cell_count,
        "unassigned_count": 0,
    }
    public_payload: JsonObject = {
        "aggregate": {
            "assigned_nonempty_zone_count": len(assignment_counts),
            "assigned_query_cell_count": len(assignments.zone_ids),
            "assigned_spatial_entity_state_count": len(entity_assignments.state_ids),
            "boundary_tie_count": assignments.boundary_tie_count,
            "connector_count": len(connectors),
            "input_gmt_segment_count": len(source_segments),
            "maximum_cells_per_nonempty_zone": max(assignment_counts.values()),
            "minimum_cells_per_nonempty_zone": min(assignment_counts.values()),
            "precision_snap_count": assignments.precision_snap_count,
            "entity_boundary_tie_count": entity_assignments.boundary_tie_count,
            "entity_precision_snap_count": entity_assignments.precision_snap_count,
            "outside_study_area_entity_state_count": sum(entity_assignments.outside_study_area),
            "outside_study_area_unique_anomaly_count": len(
                {
                    anomaly_id
                    for anomaly_id, outside in zip(
                        entity_assignments.anomaly_ids,
                        entity_assignments.outside_study_area,
                        strict=True,
                    )
                    if outside
                }
            ),
            "projected_clipped_line_count": len(lines),
            "singleton_nonempty_grid_zone_count": singleton_grid_zone_count,
            "spatially_ineligible_entity_state_count": (
                entity_assignments.spatially_ineligible_count
            ),
            "total_entity_state_count": entity_assignments.total_entity_state_count,
            "entity_permutation_group_count": len(entity_group_coordinates),
            "entity_permutation_singleton_group_count": entity_singleton_group_count,
            "entity_permutation_identical_coordinate_group_count": (
                entity_identical_coordinate_group_count
            ),
            "entity_permutation_nontrivial_group_count": entity_nontrivial_group_count,
            "entity_permutation_no_effect_state_count": entity_no_effect_state_count,
            "entity_nearest_distance_tie_count": (entity_assignments.nearest_distance_tie_count),
            "zone_count": len(primary.zones),
            "zone_set_sha256": primary.zone_set_sha256,
            "zero_cell_zone_count": len(primary.zones) - len(assignment_counts),
        },
        "input_hashes": actual_hashes,
        "local_artifacts": {
            "cell_mapping": {
                "byte_count": mapping_path.stat().st_size,
                "media_type": "application/vnd.apache.parquet",
                "sha256": mapping_sha,
            },
            "entity_mapping": {
                "byte_count": entity_mapping_path.stat().st_size,
                "media_type": "application/vnd.apache.parquet",
                "sha256": entity_mapping_sha,
            },
            "connectors": {
                "byte_count": connectors_path.stat().st_size,
                "media_type": "application/json",
                "sha256": connectors_sha,
            },
            "zone_geometry": {
                "byte_count": zones_path.stat().st_size,
                "media_type": "application/vnd.apache.parquet",
                "sha256": zones_sha,
            },
        },
        "nonempty_stratum_count": len(assignment_counts),
        "publication_safety": {
            "contains_coordinates": False,
            "contains_geometry": False,
            "contains_per_cell_mapping": False,
            "contains_wkb_or_geojson": False,
            "raw_linework_redistributed": False,
        },
        "protocol_version": STAGE4_PROTOCOL_VERSION,
        "schema_version": 1,
        "security": {
            "coordinate_bearing_artifacts_local_only": True,
            "earthquake_catalog_read": False,
            "geometry_used_as_model_feature": False,
            "license_status": CONSTRUCTION_SOURCE_LICENSE,
            "public_contains_coordinates": False,
            "public_contains_geometry": False,
            "public_contains_per_cell_mapping": False,
            "target_or_score_read": False,
        },
        "entity_permutation_group_policy": (
            "issue_time_x_construction_stratum_coordinate_bijection_"
            "singleton_or_identical_coordinate_group_identity_disclosed"
        ),
        "spatial_placebo_implementation_authorized": topology_passed,
        "stage4_target_read_authorized": False,
        "entity_stratification_gate": {
            "all_entity_states_accounted_for": (
                len(entity_assignments.state_ids) + entity_assignments.spatially_ineligible_count
                == entity_assignments.total_entity_state_count
            ),
            "all_spatially_eligible_states_assigned_once": True,
            "boundary_tie_rule": "lexicographically_smallest_zone_id",
            "outside_rule": "nearest_zone_id_with_separate_outside_flag_stratum",
            "spatially_ineligible_rule": "fixed_exclusion_never_permuted",
            "passed": True,
            "required_for_scoring_authorization": True,
        },
        "topology_gate": {
            "cut_count": primary.cuts_count,
            "cuts_count": primary.cuts_count,
            "dangle_count": primary.dangles_count,
            "dangles_count": primary.dangles_count,
            "endpoint_connection_max_km": int(maximum_connector_distance_m / 1_000.0),
            "every_fixed_query_cell_exactly_one_zone": (
                len(assignments.zone_ids) == expected_cell_count
            ),
            "exact_noding_before_precision_reduction": True,
            "fail_closed": True,
            "fixed_query_grid_assignment": fixed_grid_assignment,
            "input_segment_order_identity_stable": reorder_stable,
            "invalid_ring_count": primary.invalid_count,
            "invalid_count": primary.invalid_count,
            "passed": topology_passed,
            "polygon_count": len(primary.zones),
            "precision_m": int(CONSTRUCTION_TOPOLOGY_PRECISION_M),
            "polygonize_precision_m": CONSTRUCTION_TOPOLOGY_PRECISION_M,
            "precision_sensitivity": [
                {"precision_m": item.precision_m, "zone_count": len(item.zones)}
                for item in sensitivity
            ],
            "source_linework_levels": ["L1", "L2"],
            "stable_polygon_count": len(primary.zones),
            "stable_precision_range_m": list(CONSTRUCTION_PRECISION_SENSITIVITY_M),
            "zone_count_invariant_across_precision": len(set(zone_counts)) == 1,
        },
        "input_hash_gate": {
            "all_hashes_match": all_hashes_match,
            "passed": all_hashes_match,
            "required_for_scoring_authorization": True,
        },
    }
    public_summary = with_content_sha256(public_payload)
    serialized_public = canonical_json_bytes(public_summary).decode("utf-8").casefold()
    forbidden_public_tokens = (
        "coordinates_equal_area_m",
        "query_x_m",
        "query_y_m",
        "cell_id",
        '"wkb"',
        '"wkt"',
    )
    if any(token in serialized_public for token in forbidden_public_tokens):
        raise AssertionError(
            "coordinate or per-cell construction detail escaped into public summary"
        )
    return ConstructionStrataResult(
        mapping_path,
        entity_mapping_path,
        connectors_path,
        zones_path,
        public_summary,
    )


__all__ = [
    "CONSTRUCTION_CONNECTOR_MAXIMUM_M",
    "CONSTRUCTION_PRECISION_SENSITIVITY_M",
    "CONSTRUCTION_SOURCE_LICENSE",
    "CONSTRUCTION_TOPOLOGY_PRECISION_M",
    "PRIMARY_MACRO_HORIZONS_DAYS",
    "ROLLING_FOLD_COUNT",
    "STAGE3_QUERY_CELL_COUNT",
    "STAGE3_QUERY_CELL_SIZE_KM",
    "STAGE4_HORIZONS_DAYS",
    "STAGE4_PROTOCOL_VERSION",
    "STAGE4_ROOT_SEED",
    "ConstructionStrataResult",
    "Exposure",
    "Stage4SeedContext",
    "build_construction_strata",
    "build_exposure_preregistration",
    "build_random_input_seal",
    "build_randomness_manifest",
    "load_json_mapping",
    "protocol_design_sha256",
    "select_feature_storage_columns",
    "validate_stage4_protocol_bundle",
    "verify_content_sha256",
    "with_content_sha256",
    "write_public_manifest_atomic",
]
