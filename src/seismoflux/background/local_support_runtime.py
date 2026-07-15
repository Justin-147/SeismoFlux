"""Score-free runtime reconstruction for the frozen stage-2R local support.

This module binds the public geometry-free support manifest to the sealed local
study geometry and earthquake catalog.  It reconstructs only fit-time support,
fixed integration grids, and spatial/magnitude ETAS parent roles.  Assessment
targets, model fitting, likelihoods, and scores deliberately do not appear here.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.grid import (
    EqualAreaGridFamily,
    build_equal_area_grid_family,
)
from seismoflux.background.local_support import (
    LocalSupportCellLocator,
    LocalSupportCellManifest,
    LocalSupportManifest,
    LocalSupportSnapshot,
    LocalSupportTemporalManifest,
    build_local_support_manifest,
    build_local_support_snapshot,
)
from seismoflux.background.local_support_manifest import (
    EXPECTED_LOCAL_SUPPORT_SNAPSHOTS,
    BackgroundLocalSupportManifest,
    LocalSupportSnapshotEntry,
    validate_background_local_support_study_area,
)

BoolArray = NDArray[np.bool_]
_COMPARISON_TOLERANCE = 1.0e-12
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class LocalSupportRuntimeError(RuntimeError):
    """Raised when sealed runtime inputs differ from the frozen support record."""


def _readonly_bool(values: object) -> BoolArray:
    result = np.array(values, dtype=np.bool_, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError("local support runtime masks must be one-dimensional")
    result.setflags(write=False)
    return result


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("local support runtime cutoff must be timezone-aware")
    return parsed.astimezone(UTC)


def _frozen_manifest_for_entry(
    manifest: BackgroundLocalSupportManifest,
    entry: LocalSupportSnapshotEntry,
) -> LocalSupportManifest:
    record = entry.support
    temporal_blocks = tuple(
        LocalSupportTemporalManifest(**block.model_dump(mode="python"))
        for block in record.temporal_blocks
    )
    cells = tuple(
        LocalSupportCellManifest(
            cell_id=fixed.cell_id,
            row=fixed.row,
            column=fixed.column,
            clipped_area_m2=fixed.clipped_area_m2,
            status=decision.status,
            source=decision.source,
            base_event_count=decision.base_event_count,
            source_event_count=decision.source_event_count,
            parent_cell_id=decision.parent_cell_id,
            raw_mc=decision.raw_mc,
            candidate_mc=decision.candidate_mc,
            applied_mc=decision.applied_mc,
        )
        for fixed, decision in zip(manifest.fixed_cells, record.cells, strict=True)
    )
    return LocalSupportManifest(
        protocol_version=record.protocol_version,
        support_id=record.support_id,
        fit_end_utc=record.fit_end_utc,
        historical_event_count=record.historical_event_count,
        historical_event_sha256=record.historical_event_sha256,
        study_area_sha256=record.study_area_sha256,
        base_cell_size_km=record.base_cell_size_km,
        parent_cell_size_km=record.parent_cell_size_km,
        minimum_events_per_stratum=record.minimum_events_per_stratum,
        maximum_supported_raw_mc=record.maximum_supported_raw_mc,
        minimum_retained_area_fraction=record.minimum_retained_area_fraction,
        common_mc=record.common_mc,
        retained_selected_event_count=record.retained_selected_event_count,
        retained_selected_aki_b_value=record.retained_selected_aki_b_value,
        total_area_m2=record.total_area_m2,
        retained_area_m2=record.retained_area_m2,
        retained_area_fraction=record.retained_area_fraction,
        temporal_blocks=temporal_blocks,
        cells=cells,
    )


def _require_frozen_equal(actual: Any, expected: Any, *, path: str) -> None:
    """Report the first exact field mismatch in two frozen manifest values."""

    if is_dataclass(actual) and is_dataclass(expected):
        if type(actual) is not type(expected):
            raise LocalSupportRuntimeError(f"{path} has a different frozen value type")
        for item in fields(actual):
            _require_frozen_equal(
                getattr(actual, item.name),
                getattr(expected, item.name),
                path=f"{path}.{item.name}",
            )
        return
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        if len(actual) != len(expected):
            raise LocalSupportRuntimeError(f"{path} has a different frozen tuple length")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _require_frozen_equal(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if actual != expected:
        raise LocalSupportRuntimeError(f"{path} differs from the frozen support manifest")


def _validate_grid_family_area(
    grid_family: EqualAreaGridFamily,
    support: LocalSupportSnapshot,
) -> None:
    if not grid_family.study_area_equal_area.equals(support.retained_geometry):
        raise LocalSupportRuntimeError("integration grids use a different retained geometry")
    tolerance_m2 = max(1.0e-6, support.retained_area_m2 * 1.0e-12)
    for grid in grid_family.grids:
        area_m2 = math.fsum(cell.clipped_area_m2 for cell in grid.cells)
        if abs(area_m2 - support.retained_area_m2) > tolerance_m2:
            raise LocalSupportRuntimeError(
                f"{grid.spec.cell_size_km:g} km grid area differs from retained support"
            )


def _compensator_domain_id(
    grid_family: EqualAreaGridFamily,
    support: LocalSupportSnapshot,
) -> str:
    payload = {
        "schema": "seismoflux_local_support_compensator_domain_v1",
        "study_area_sha256": support.study_area_sha256,
        "retained_fixed_cell_ids": tuple(
            cell.cell_id for cell in support.cells if cell.status != "unsupported"
        ),
        "grids": tuple(
            {
                "cell_size_km": grid.spec.cell_size_km,
                "cells": tuple(
                    {
                        "cell_id": cell.id,
                        "representative_x_m": float(cell.representative_point.x),
                        "representative_y_m": float(cell.representative_point.y),
                        "clipped_area_m2": cell.clipped_area_m2,
                    }
                    for cell in grid.cells
                ),
            }
            for grid in grid_family.grids
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalSupportRuntimeSnapshot:
    """One score-free support snapshot aligned to the supplied event order.

    Parent-role masks encode only local-support spatial and magnitude roles.
    Later ETAS code must still intersect them with its causal origin-time and
    availability masks and add separately identified external-buffer parents.
    """

    snapshot_id: str
    support: LocalSupportSnapshot
    grid_family: EqualAreaGridFamily
    compensator_domain_id: str
    event_cell_ids: tuple[str | None, ...]
    supported_mask: BoolArray
    etas_primary_parent_role_mask: BoolArray
    etas_sensitivity_parent_role_mask: BoolArray

    def __post_init__(self) -> None:
        expected_cutoffs = dict(EXPECTED_LOCAL_SUPPORT_SNAPSHOTS)
        if self.snapshot_id not in expected_cutoffs:
            raise ValueError("unknown local support runtime snapshot_id")
        if self.support.fit_end_utc != _parse_utc(expected_cutoffs[self.snapshot_id]):
            raise ValueError("local support runtime snapshot uses the wrong fit cutoff")
        if _SHA256_RE.fullmatch(self.compensator_domain_id) is None:
            raise ValueError("compensator_domain_id must be a lowercase SHA-256")
        event_cell_ids = tuple(self.event_cell_ids)
        supported = _readonly_bool(self.supported_mask)
        primary = _readonly_bool(self.etas_primary_parent_role_mask)
        sensitivity = _readonly_bool(self.etas_sensitivity_parent_role_mask)
        expected_shape = (len(event_cell_ids),)
        if any(mask.shape != expected_shape for mask in (supported, primary, sensitivity)):
            raise ValueError("local support runtime masks must align with event_cell_ids")
        if np.any(sensitivity & ~supported):
            raise ValueError("sensitivity ETAS parents must remain inside supported cells")
        if np.any(sensitivity & ~primary):
            raise ValueError("primary ETAS parents must include sensitivity parents")
        object.__setattr__(self, "event_cell_ids", event_cell_ids)
        object.__setattr__(self, "supported_mask", supported)
        object.__setattr__(self, "etas_primary_parent_role_mask", primary)
        object.__setattr__(self, "etas_sensitivity_parent_role_mask", sensitivity)


@dataclass(frozen=True, slots=True)
class LocalSupportRuntime:
    """All five reconstructed support snapshots without assessment targets."""

    manifest_id: str
    event_ids: tuple[str, ...]
    snapshots: tuple[LocalSupportRuntimeSnapshot, ...]

    def __post_init__(self) -> None:
        event_ids = tuple(self.event_ids)
        if not event_ids or any(not event_id for event_id in event_ids):
            raise ValueError("local support runtime requires non-empty event IDs")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("local support runtime event IDs must be unique")
        observed = tuple(snapshot.snapshot_id for snapshot in self.snapshots)
        expected = tuple(item[0] for item in EXPECTED_LOCAL_SUPPORT_SNAPSHOTS)
        if observed != expected:
            raise ValueError("local support runtime snapshots must use frozen order")
        if any(len(snapshot.event_cell_ids) != len(event_ids) for snapshot in self.snapshots):
            raise ValueError("every local support snapshot must align with all input events")
        object.__setattr__(self, "event_ids", event_ids)

    def snapshot(self, snapshot_id: str) -> LocalSupportRuntimeSnapshot:
        for snapshot in self.snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        raise KeyError(f"local support runtime has no snapshot {snapshot_id!r}")


def _event_roles(
    events: tuple[CompletenessEvent, ...],
    support: LocalSupportSnapshot,
) -> tuple[tuple[str | None, ...], BoolArray, BoolArray, BoolArray]:
    locator = LocalSupportCellLocator(support.cells)
    event_cell_ids: list[str | None] = []
    supported = np.zeros(len(events), dtype=np.bool_)
    primary = np.zeros(len(events), dtype=np.bool_)
    sensitivity = np.zeros(len(events), dtype=np.bool_)

    for index, event in enumerate(events):
        if not event.inside_study_area:
            event_cell_ids.append(None)
            continue
        cell = locator.resolve(x_m=event.x_m, y_m=event.y_m)
        if cell is None:
            raise LocalSupportRuntimeError(
                f"inside-study event {event.event_id!r} has no fixed support cell"
            )
        event_cell_ids.append(cell.cell_id)
        if cell.status != "unsupported":
            supported[index] = True
            eligible = event.magnitude >= support.common_mc - _COMPARISON_TOLERANCE
            primary[index] = eligible
            sensitivity[index] = eligible
            continue
        if cell.raw_mc is None:
            raise LocalSupportRuntimeError("unsupported support cell has no frozen raw Mc")
        primary[index] = event.magnitude >= cell.raw_mc - _COMPARISON_TOLERANCE

    return tuple(event_cell_ids), supported, primary, sensitivity


def build_local_support_runtime(
    manifest: BackgroundLocalSupportManifest,
    events: Iterable[CompletenessEvent],
    *,
    study_area_equal_area: BaseGeometry,
) -> LocalSupportRuntime:
    """Reconstruct and bind all score-free local-support runtime inputs.

    The supplied event order is preserved in every mask.  No assessment date,
    target definition, likelihood, model score, or model-selection value is
    accepted or produced by this function.
    """

    if not isinstance(manifest, BackgroundLocalSupportManifest):
        raise TypeError("manifest must be a BackgroundLocalSupportManifest")
    manifest = BackgroundLocalSupportManifest.model_validate(manifest.model_dump(mode="python"))
    supplied_events = tuple(events)
    if not supplied_events or any(
        not isinstance(event, CompletenessEvent) for event in supplied_events
    ):
        raise ValueError("local support runtime requires CompletenessEvent inputs")
    validate_background_local_support_study_area(manifest, study_area_equal_area)

    grid_cache: dict[
        tuple[str, ...],
        tuple[EqualAreaGridFamily, str],
    ] = {}
    runtime_snapshots: list[LocalSupportRuntimeSnapshot] = []
    for entry in manifest.snapshots:
        support = build_local_support_snapshot(
            supplied_events,
            fit_end_utc=_parse_utc(entry.support.fit_end_utc),
            study_area_equal_area=study_area_equal_area,
        )
        actual_manifest = build_local_support_manifest(support)
        frozen_manifest = _frozen_manifest_for_entry(manifest, entry)
        _require_frozen_equal(
            actual_manifest,
            frozen_manifest,
            path=f"snapshots.{entry.snapshot_id}.support",
        )

        domain_key = tuple(cell.cell_id for cell in support.cells if cell.status != "unsupported")
        cached = grid_cache.get(domain_key)
        if cached is None:
            grid_family = build_equal_area_grid_family(support.retained_geometry)
            _validate_grid_family_area(grid_family, support)
            domain_id = _compensator_domain_id(grid_family, support)
            cached = (grid_family, domain_id)
            grid_cache[domain_key] = cached
        else:
            grid_family, _ = cached
            _validate_grid_family_area(grid_family, support)
        grid_family, domain_id = cached
        event_cell_ids, supported, primary, sensitivity = _event_roles(
            supplied_events,
            support,
        )
        runtime_snapshots.append(
            LocalSupportRuntimeSnapshot(
                snapshot_id=entry.snapshot_id,
                support=support,
                grid_family=grid_family,
                compensator_domain_id=domain_id,
                event_cell_ids=event_cell_ids,
                supported_mask=supported,
                etas_primary_parent_role_mask=primary,
                etas_sensitivity_parent_role_mask=sensitivity,
            )
        )

    return LocalSupportRuntime(
        manifest_id=manifest.manifest_id,
        event_ids=tuple(event.event_id for event in supplied_events),
        snapshots=tuple(runtime_snapshots),
    )


__all__ = [
    "LocalSupportRuntime",
    "LocalSupportRuntimeError",
    "LocalSupportRuntimeSnapshot",
    "build_local_support_runtime",
]
