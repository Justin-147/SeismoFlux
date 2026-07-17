"""Frozen, target-free manifest for the stage-2R local-support surfaces.

The manifest is deliberately limited to historical completeness evidence,
geometry-free fixed-grid identities, and source identities.  Assessment targets, model scores,
information gain, hits, and model-selection results are rejected before model
validation so they cannot be hidden in an otherwise content-addressed file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shapely.geometry.base import BaseGeometry

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.completeness import (
    CATALOG_ANCHOR_UTC,
    COMPLETENESS_CANDIDATES,
    FINAL_PARTIAL_BLOCK_MINIMUM_YEARS,
    SPATIAL_MINIMUM_EVENTS,
    TEMPORAL_BLOCK_YEARS,
    TEMPORAL_MINIMUM_EVENTS,
    CompletenessEvent,
    select_candidate_magnitude,
)
from seismoflux.background.local_support import (
    LOCAL_SUPPORT_PROTOCOL_VERSION,
    MAXIMUM_SUPPORTED_RAW_MC,
    MINIMUM_RETAINED_AREA_FRACTION,
    LocalSupportManifest,
    build_local_support_manifest,
    build_local_support_snapshot,
    build_local_support_study_area_identity,
)
from seismoflux.background.scientific import scientific_mapping

BACKGROUND_LOCAL_SUPPORT_PROTOCOL_VERSION = "0.2.1"
BACKGROUND_LOCAL_SUPPORT_FREEZE_TAG = "v0.2.1-background-local-support-protocol"
BACKGROUND_LOCAL_SUPPORT_GATE_NAME = "G1-LS"
BACKGROUND_LOCAL_SUPPORT_FROZEN_ON = "2026-07-14"
PARENT_PROTOCOL_EXECUTION_FINGERPRINT = (
    "f386f0d6abd5b7ca0e31e073ce0f74da812fb561052639d45227e8f339ff9032"
)

EXPECTED_LOCAL_SUPPORT_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    ("fold_1", "2004-12-31T16:00:00.000000Z"),
    ("fold_2", "2009-12-31T16:00:00.000000Z"),
    ("fold_3", "2014-12-31T16:00:00.000000Z"),
    ("fold_4", "2019-12-31T16:00:00.000000Z"),
    ("final_validation", "2023-06-30T16:00:00.000000Z"),
)

FORBIDDEN_SUPPORT_MANIFEST_FIELDS = frozenset(
    {
        "assessment_target_count",
        "assessment_target_locations",
        "validation_target_count",
        "validation_target_locations",
        "model_score",
        "information_gain",
        "hit_result",
        "model_selection",
        "score_id",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SUPPORT_ID_RE = re.compile(r"local-support-[0-9a-f]{16}")
_BUNDLE_ID_RE = re.compile(r"local-support-bundle-[0-9a-f]{16}")
_COMPARISON_TOLERANCE = 1.0e-12
_SECONDS_PER_TROPICAL_YEAR = 365.2425 * 24.0 * 60.0 * 60.0


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _parse_utc(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    canonical = parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if value != canonical:
        raise ValueError(f"{field_name} must use canonical microsecond UTC text")
    return parsed


def _fixed_cell_id(cell_size_km: float, row: int, column: int) -> str:
    cell_size_mm = int(cell_size_km * 1_000_000.0)
    return f"g{cell_size_mm:08d}_r{row:+08d}_c{column:+08d}"


def _add_years(value: datetime, years: int) -> datetime:
    return value.replace(year=value.year + years)


def _require_finite_mc(name: str, value: float | None) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_candidate_mapping(raw_mc: float, candidate_mc: float) -> None:
    expected = select_candidate_magnitude(raw_mc)
    if candidate_mc != expected:
        raise ValueError("candidate Mc must be the frozen upward mapping of raw Mc")


def _forbid_result_fields(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"support-manifest key must be a string at {location}")
            if key.casefold() in FORBIDDEN_SUPPORT_MANIFEST_FIELDS:
                raise ValueError(f"forbidden result field in support manifest: {location}.{key}")
            _forbid_result_fields(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _forbid_result_fields(item, location=f"{location}[{index}]")


class LocalSupportSourceFile(_StrictFrozenModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        path = Path(self.path)
        if (
            not self.path
            or "\\" in self.path
            or path.is_absolute()
            or ".." in path.parts
            or self.path != path.as_posix()
        ):
            raise ValueError("local-support source path must be normalized and project-relative")
        return self


class LocalSupportSources(_StrictFrozenModel):
    earthquake_dataset: LocalSupportSourceFile
    study_area: LocalSupportSourceFile


class LocalSupportTemporalRecord(_StrictFrozenModel):
    block_id: str
    start_utc: str
    end_utc: str
    duration_years: float = Field(gt=0.0)
    is_partial: bool
    event_count: int = Field(ge=0)
    eligible: bool
    raw_mc: float | None
    candidate_mc: float | None

    @model_validator(mode="after")
    def validate_temporal_evidence(self) -> Self:
        start = _parse_utc(self.start_utc, field_name="temporal start_utc")
        end = _parse_utc(self.end_utc, field_name="temporal end_utc")
        if end <= start or not math.isfinite(self.duration_years):
            raise ValueError("temporal completeness block must have positive duration")
        if self.eligible:
            if self.raw_mc is None or self.candidate_mc is None:
                raise ValueError("eligible temporal block requires raw and candidate Mc")
            _require_finite_mc("temporal raw Mc", self.raw_mc)
            _require_finite_mc("temporal candidate Mc", self.candidate_mc)
            if self.raw_mc > MAXIMUM_SUPPORTED_RAW_MC + _COMPARISON_TOLERANCE:
                raise ValueError("temporal raw Mc above 4.0 is a global hard failure")
            _require_candidate_mapping(self.raw_mc, self.candidate_mc)
        elif self.raw_mc is not None or self.candidate_mc is not None:
            raise ValueError("ineligible temporal block cannot carry an Mc estimate")
        return self


SupportStatus = Literal["supported", "indeterminate", "unsupported"]
SupportSource = Literal["base_500km", "fixed_1000km_parent"]


class LocalSupportCellRecord(_StrictFrozenModel):
    cell_id: str
    row: int
    column: int
    clipped_area_m2: float = Field(gt=0.0)
    status: SupportStatus
    source: SupportSource
    base_event_count: int = Field(ge=0)
    source_event_count: int = Field(ge=0)
    parent_cell_id: str | None
    raw_mc: float | None
    candidate_mc: float | None
    applied_mc: float | None

    @model_validator(mode="after")
    def validate_cell(self) -> Self:
        if self.cell_id != _fixed_cell_id(500.0, self.row, self.column):
            raise ValueError("local-support cell ID does not match its fixed 500 km index")
        if not math.isfinite(self.clipped_area_m2):
            raise ValueError("local-support clipped area must be finite")
        if self.source_event_count < self.base_event_count:
            raise ValueError("local-support source count cannot be below base-cell count")
        if self.source == "base_500km":
            if (
                self.parent_cell_id is not None
                or self.source_event_count != self.base_event_count
                or self.base_event_count < SPATIAL_MINIMUM_EVENTS
                or self.status == "indeterminate"
            ):
                raise ValueError("dense base support cannot declare a parent source")
        else:
            expected_parent = _fixed_cell_id(1000.0, self.row // 2, self.column // 2)
            if (
                self.parent_cell_id != expected_parent
                or self.base_event_count >= SPATIAL_MINIMUM_EVENTS
                or (
                    self.status == "indeterminate"
                    and self.source_event_count >= SPATIAL_MINIMUM_EVENTS
                )
                or (
                    self.status != "indeterminate"
                    and self.source_event_count < SPATIAL_MINIMUM_EVENTS
                )
            ):
                raise ValueError("sparse support is inconsistent with its fixed 1000 km parent")

        _require_finite_mc("raw Mc", self.raw_mc)
        _require_finite_mc("candidate Mc", self.candidate_mc)
        _require_finite_mc("applied Mc", self.applied_mc)

        if self.status == "supported":
            if self.raw_mc is None or self.candidate_mc is None or self.applied_mc is None:
                raise ValueError("supported cell requires raw, candidate, and applied Mc")
            if self.raw_mc > MAXIMUM_SUPPORTED_RAW_MC + _COMPARISON_TOLERANCE:
                raise ValueError("supported cell raw Mc exceeds 4.0")
            _require_candidate_mapping(self.raw_mc, self.candidate_mc)
            if self.applied_mc not in COMPLETENESS_CANDIDATES:
                raise ValueError("supported cell applied Mc is not frozen")
        elif self.status == "indeterminate":
            if (
                self.raw_mc is not None
                or self.candidate_mc is not None
                or self.applied_mc not in COMPLETENESS_CANDIDATES
            ):
                raise ValueError("indeterminate cell must carry only the common applied Mc")
        elif (
            self.raw_mc is None
            or self.raw_mc <= MAXIMUM_SUPPORTED_RAW_MC + _COMPARISON_TOLERANCE
            or self.candidate_mc is not None
            or self.applied_mc is not None
        ):
            raise ValueError("unsupported cell requires only a raw Mc above 4.0")
        return self


class LocalSupportFixedCellRecord(_StrictFrozenModel):
    """One geometry-free fixed-cell identity shared by all five snapshots."""

    cell_id: str
    row: int
    column: int
    clipped_area_m2: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_fixed_cell(self) -> Self:
        if self.cell_id != _fixed_cell_id(500.0, self.row, self.column):
            raise ValueError("fixed cell ID does not match its 500 km index")
        if not math.isfinite(self.clipped_area_m2):
            raise ValueError("fixed-cell clipped area must be finite")
        return self


class LocalSupportCellDecisionRecord(_StrictFrozenModel):
    """Snapshot-specific causal decision; fixed geometry is stored at bundle level."""

    cell_id: str
    status: SupportStatus
    source: SupportSource
    base_event_count: int = Field(ge=0)
    source_event_count: int = Field(ge=0)
    parent_cell_id: str | None
    raw_mc: float | None
    candidate_mc: float | None
    applied_mc: float | None


class LocalSupportSnapshotRecord(_StrictFrozenModel):
    protocol_version: Literal["seismoflux_local_support_v1"]
    support_id: str
    fit_end_utc: str
    historical_event_count: int = Field(gt=0)
    historical_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    study_area_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_cell_size_km: float
    parent_cell_size_km: float
    minimum_events_per_stratum: int
    maximum_supported_raw_mc: float
    minimum_retained_area_fraction: float
    common_mc: float
    retained_selected_event_count: int = Field(gt=0)
    retained_selected_aki_b_value: float
    total_area_m2: float = Field(gt=0.0)
    retained_area_m2: float = Field(gt=0.0)
    retained_area_fraction: float = Field(gt=0.0, le=1.0)
    temporal_blocks: tuple[LocalSupportTemporalRecord, ...]
    cells: tuple[LocalSupportCellDecisionRecord, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        fit_end = _parse_utc(self.fit_end_utc, field_name="fit_end_utc")
        if _SUPPORT_ID_RE.fullmatch(self.support_id) is None:
            raise ValueError("support_id must be a deterministic local-support address")
        if (
            self.protocol_version != LOCAL_SUPPORT_PROTOCOL_VERSION
            or self.base_cell_size_km != 500.0
            or self.parent_cell_size_km != 1000.0
            or self.minimum_events_per_stratum != 200
            or self.maximum_supported_raw_mc != MAXIMUM_SUPPORTED_RAW_MC
            or self.minimum_retained_area_fraction != MINIMUM_RETAINED_AREA_FRACTION
            or self.common_mc not in COMPLETENESS_CANDIDATES
        ):
            raise ValueError("local-support snapshot policy differs from the frozen protocol")
        if not math.isfinite(self.retained_selected_aki_b_value):
            raise ValueError("local-support Aki b-value must be finite")
        if not self.temporal_blocks or not any(block.eligible for block in self.temporal_blocks):
            raise ValueError("local-support snapshot requires eligible temporal evidence")
        if fit_end <= CATALOG_ANCHOR_UTC:
            raise ValueError("local-support fit cutoff must follow the 1970 catalog anchor")

        expected_start = CATALOG_ANCHOR_UTC
        for block in self.temporal_blocks:
            start = _parse_utc(block.start_utc, field_name="temporal start_utc")
            end = _parse_utc(block.end_utc, field_name="temporal end_utc")
            if start != expected_start or start >= fit_end:
                raise ValueError("temporal blocks must be contiguous from the 1970 anchor")
            nominal_end = _add_years(start, TEMPORAL_BLOCK_YEARS)
            expected_end = min(nominal_end, fit_end)
            expected_partial = nominal_end > fit_end
            expected_block_id = (
                f"{start.year:04d}-partial-{fit_end.date().isoformat()}"
                if expected_partial
                else f"{start.year:04d}-{nominal_end.year - 1:04d}"
            )
            expected_duration = (expected_end - start).total_seconds() / _SECONDS_PER_TROPICAL_YEAR
            if (
                end != expected_end
                or block.is_partial != expected_partial
                or block.block_id != expected_block_id
                or block.duration_years != expected_duration
            ):
                raise ValueError("temporal block identity differs from the frozen calendar")
            duration_eligible = not expected_partial or fit_end >= _add_years(
                start,
                FINAL_PARTIAL_BLOCK_MINIMUM_YEARS,
            )
            expected_eligible = block.event_count >= TEMPORAL_MINIMUM_EVENTS and duration_eligible
            if block.eligible != expected_eligible:
                raise ValueError("temporal block eligibility differs from count and duration rules")
            expected_start = expected_end
        if expected_start != fit_end:
            raise ValueError("final temporal block must end at fit_end_utc")
        if sum(block.event_count for block in self.temporal_blocks) != self.historical_event_count:
            raise ValueError("temporal block counts must cover every historical event exactly once")

        if self.retained_area_fraction < MINIMUM_RETAINED_AREA_FRACTION:
            raise ValueError("local-support snapshot violates the frozen 95% area gate")

        candidates = [
            block.candidate_mc
            for block in self.temporal_blocks
            if block.eligible and block.candidate_mc is not None
        ]
        candidates.extend(
            cell.candidate_mc
            for cell in self.cells
            if cell.status == "supported" and cell.candidate_mc is not None
        )
        if not candidates or max(candidates) != self.common_mc:
            raise ValueError("local-support common Mc is not the maximum retained candidate")
        if any(
            cell.applied_mc is not None and cell.applied_mc != self.common_mc for cell in self.cells
        ):
            raise ValueError("all retained cells must share the snapshot common Mc")

        return self


SnapshotId = Literal["fold_1", "fold_2", "fold_3", "fold_4", "final_validation"]


class LocalSupportSnapshotEntry(_StrictFrozenModel):
    snapshot_id: SnapshotId
    support: LocalSupportSnapshotRecord


class BackgroundLocalSupportManifest(_StrictFrozenModel):
    schema_version: Literal["1.0.0"]
    protocol_version: Literal["0.2.1"]
    local_support_protocol_version: Literal["seismoflux_local_support_v1"]
    gate_name: Literal["G1-LS"]
    frozen_on: Literal["2026-07-14"]
    freeze_tag: Literal["v0.2.1-background-local-support-protocol"]
    parent_protocol_execution_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: LocalSupportSources
    fixed_cells: tuple[LocalSupportFixedCellRecord, ...]
    snapshots: tuple[LocalSupportSnapshotEntry, ...]
    manifest_id: str

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        if (
            self.local_support_protocol_version != LOCAL_SUPPORT_PROTOCOL_VERSION
            or self.parent_protocol_execution_fingerprint != PARENT_PROTOCOL_EXECUTION_FINGERPRINT
        ):
            raise ValueError("local-support bundle identity differs from preregistration")
        observed = tuple((entry.snapshot_id, entry.support.fit_end_utc) for entry in self.snapshots)
        if observed != EXPECTED_LOCAL_SUPPORT_SNAPSHOTS:
            raise ValueError("local-support bundle must contain the five frozen snapshots in order")

        fixed_order = tuple((cell.row, cell.column) for cell in self.fixed_cells)
        if not self.fixed_cells or fixed_order != tuple(sorted(fixed_order)):
            raise ValueError("fixed local-support cells must be row/column ordered")
        if len(fixed_order) != len(set(fixed_order)):
            raise ValueError("fixed local-support cells must be unique")
        fixed_ids = tuple(cell.cell_id for cell in self.fixed_cells)
        total_area = math.fsum(cell.clipped_area_m2 for cell in self.fixed_cells)
        first_study_hash = self.snapshots[0].support.study_area_sha256
        first_total_area = self.snapshots[0].support.total_area_m2
        if total_area != first_total_area:
            raise ValueError("fixed-cell areas must sum to the frozen study area")
        for entry in self.snapshots:
            if (
                entry.support.study_area_sha256 != first_study_hash
                or entry.support.total_area_m2 != first_total_area
            ):
                raise ValueError("all snapshots must share one fixed study-area geometry")
            if tuple(cell.cell_id for cell in entry.support.cells) != fixed_ids:
                raise ValueError("every snapshot must cover all fixed cells in the same order")

            expanded_cells: list[dict[str, object]] = []
            retained_area = 0.0
            for fixed, decision in zip(self.fixed_cells, entry.support.cells, strict=True):
                expanded = {
                    **fixed.model_dump(mode="python"),
                    **decision.model_dump(mode="python"),
                }
                LocalSupportCellRecord.model_validate(expanded)
                expanded_cells.append(expanded)
                if decision.status != "unsupported":
                    retained_area += fixed.clipped_area_m2
            if entry.support.retained_area_m2 != math.fsum(
                fixed.clipped_area_m2
                for fixed, decision in zip(
                    self.fixed_cells,
                    entry.support.cells,
                    strict=True,
                )
                if decision.status != "unsupported"
            ):
                raise ValueError("snapshot retained area is not an exact fixed-cell sum")
            if entry.support.retained_area_fraction != retained_area / total_area:
                raise ValueError("snapshot retained-area fraction is inconsistent")

            support_payload = entry.support.model_dump(
                mode="python",
                exclude={"support_id", "cells"},
            )
            support_payload["cells"] = expanded_cells
            expected_support_id = (
                "local-support-"
                + hashlib.sha256(canonical_json_bytes(support_payload)).hexdigest()[:16]
            )
            if entry.support.support_id != expected_support_id:
                raise ValueError("support_id does not match the complete causal snapshot payload")

        if _BUNDLE_ID_RE.fullmatch(self.manifest_id) is None:
            raise ValueError("manifest_id must use the deterministic bundle address format")
        payload = self.model_dump(mode="python", exclude={"manifest_id"})
        expected_id = (
            "local-support-bundle-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:16]
        )
        if self.manifest_id != expected_id:
            raise ValueError("manifest_id does not match the complete local-support bundle")
        return self


def _fixed_cell_records(
    manifest: LocalSupportManifest,
) -> tuple[LocalSupportFixedCellRecord, ...]:
    return tuple(
        LocalSupportFixedCellRecord(
            cell_id=cell.cell_id,
            row=cell.row,
            column=cell.column,
            clipped_area_m2=cell.clipped_area_m2,
        )
        for cell in manifest.cells
    )


def _snapshot_record(manifest: LocalSupportManifest) -> LocalSupportSnapshotRecord:
    payload = scientific_mapping(manifest)
    cell_values = payload.pop("cells")
    if not isinstance(cell_values, list):
        raise TypeError("local-support snapshot cells must be a list")
    payload["cells"] = [
        {
            key: value
            for key, value in cell.items()
            if key
            not in {
                "row",
                "column",
                "clipped_area_m2",
            }
        }
        for cell in cell_values
        if isinstance(cell, dict)
    ]
    if len(cast(list[object], payload["cells"])) != len(cell_values):
        raise TypeError("local-support snapshot cells must be mappings")
    return LocalSupportSnapshotRecord.model_validate(payload)


def build_background_local_support_manifest(
    events: Iterable[CompletenessEvent],
    *,
    study_area_equal_area: BaseGeometry,
    sources: LocalSupportSources,
) -> BackgroundLocalSupportManifest:
    """Build all five causal support surfaces without opening any assessment target."""

    supplied_events = tuple(events)
    snapshots: list[dict[str, object]] = []
    fixed_cells: tuple[LocalSupportFixedCellRecord, ...] | None = None
    for snapshot_id, fit_end_text in EXPECTED_LOCAL_SUPPORT_SNAPSHOTS:
        support = build_local_support_snapshot(
            supplied_events,
            fit_end_utc=_parse_utc(fit_end_text, field_name="fit_end_utc"),
            study_area_equal_area=study_area_equal_area,
        )
        support_manifest = build_local_support_manifest(support)
        observed_fixed_cells = _fixed_cell_records(support_manifest)
        if fixed_cells is None:
            fixed_cells = observed_fixed_cells
        elif observed_fixed_cells != fixed_cells:
            raise ValueError("causal snapshots produced different fixed support geometries")
        record = _snapshot_record(support_manifest)
        snapshots.append(
            {
                "snapshot_id": snapshot_id,
                "support": record.model_dump(mode="python"),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "protocol_version": BACKGROUND_LOCAL_SUPPORT_PROTOCOL_VERSION,
        "local_support_protocol_version": LOCAL_SUPPORT_PROTOCOL_VERSION,
        "gate_name": BACKGROUND_LOCAL_SUPPORT_GATE_NAME,
        "frozen_on": BACKGROUND_LOCAL_SUPPORT_FROZEN_ON,
        "freeze_tag": BACKGROUND_LOCAL_SUPPORT_FREEZE_TAG,
        "parent_protocol_execution_fingerprint": PARENT_PROTOCOL_EXECUTION_FINGERPRINT,
        "sources": sources.model_dump(mode="python"),
        "fixed_cells": [cell.model_dump(mode="python") for cell in (fixed_cells or ())],
        "snapshots": snapshots,
    }
    payload["manifest_id"] = (
        "local-support-bundle-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:16]
    )
    return BackgroundLocalSupportManifest.model_validate(payload)


def background_local_support_manifest_bytes(
    manifest: BackgroundLocalSupportManifest,
) -> bytes:
    """Render deterministic, readable UTF-8 JSON for the frozen input file."""

    validated = BackgroundLocalSupportManifest.model_validate(manifest.model_dump(mode="python"))
    payload = validated.model_dump(mode="json")
    _forbid_result_fields(payload)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_background_local_support_manifest(
    path: str | Path,
) -> BackgroundLocalSupportManifest:
    """Read and strictly validate a frozen target-free local-support bundle."""

    manifest_path = Path(path)
    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read local-support manifest: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("local-support manifest root must be a mapping")
    _forbid_result_fields(raw)
    return BackgroundLocalSupportManifest.model_validate(raw)


def validate_background_local_support_study_area(
    manifest: BackgroundLocalSupportManifest,
    study_area_equal_area: BaseGeometry,
) -> None:
    """Bind a public manifest to its sealed local projected study geometry.

    The fixed 500 km cells are rebuilt locally.  Their IDs, indices, and exact
    clipped areas, along with the total area and normalized study-area digest,
    must match every corresponding identity recorded in the public bundle.
    Geometry bytes are neither returned nor added to the manifest.
    """

    if not isinstance(manifest, BackgroundLocalSupportManifest):
        raise TypeError("manifest must be a BackgroundLocalSupportManifest")
    identity = build_local_support_study_area_identity(study_area_equal_area)
    observed_cells = tuple(
        (cell.cell_id, cell.row, cell.column, cell.clipped_area_m2) for cell in manifest.fixed_cells
    )
    expected_cells = tuple(
        (cell.cell_id, cell.row, cell.column, cell.clipped_area_m2) for cell in identity.fixed_cells
    )
    if observed_cells != expected_cells:
        raise ValueError(
            "local-support fixed-cell IDs, indices, or clipped areas do not match "
            "the sealed study area"
        )
    if math.fsum(cell.clipped_area_m2 for cell in manifest.fixed_cells) != (identity.total_area_m2):
        raise ValueError("local-support total area does not match the sealed study area")
    for entry in manifest.snapshots:
        if entry.support.total_area_m2 != identity.total_area_m2:
            raise ValueError(
                "local-support snapshot total area does not match the sealed study area"
            )
        if entry.support.study_area_sha256 != identity.study_area_sha256:
            raise ValueError("local-support study-area digest does not match the sealed study area")


__all__ = [
    "BACKGROUND_LOCAL_SUPPORT_FREEZE_TAG",
    "BACKGROUND_LOCAL_SUPPORT_FROZEN_ON",
    "BACKGROUND_LOCAL_SUPPORT_GATE_NAME",
    "BACKGROUND_LOCAL_SUPPORT_PROTOCOL_VERSION",
    "EXPECTED_LOCAL_SUPPORT_SNAPSHOTS",
    "FORBIDDEN_SUPPORT_MANIFEST_FIELDS",
    "PARENT_PROTOCOL_EXECUTION_FINGERPRINT",
    "BackgroundLocalSupportManifest",
    "LocalSupportCellDecisionRecord",
    "LocalSupportCellRecord",
    "LocalSupportFixedCellRecord",
    "LocalSupportSnapshotEntry",
    "LocalSupportSnapshotRecord",
    "LocalSupportSourceFile",
    "LocalSupportSources",
    "LocalSupportTemporalRecord",
    "background_local_support_manifest_bytes",
    "build_background_local_support_manifest",
    "load_background_local_support_manifest",
    "validate_background_local_support_study_area",
]
