"""Score-blind causal local-completeness masks for the frozen S1-C1 test.

This module does one scientific job: at each preregistered inner-block or
outer-fold start, estimate which fixed 500 km cells have enough causal catalog
support for M4 training centres.  It never changes the national prediction
domain and contains no target, prediction, parameter-selection, or scoring API.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal, cast

from seismoflux.background.completeness import maximum_curvature_estimate
from seismoflux.background.local_support import LocalSupportBasePartition
from seismoflux.data.common import canonical_json_bytes

C1_HISTORY_START_UTC: Final = datetime(1969, 12, 31, 16, tzinfo=UTC)
C1_SNAPSHOT_DELAY: Final = timedelta(hours=24)
C1_MINIMUM_EVENTS: Final = 200
C1_MAXIMUM_SUPPORTED_RAW_MC: Final = 4.0
C1_SUPPORT_GATE_FRACTION: Final = 0.95

SnapshotRole = Literal["inner_block_start", "outer_fold_start"]
CompletenessStatus = Literal["supported", "indeterminate", "unsupported"]
EstimateSource = Literal["base_500km", "parent_1000km", "not_estimable"]


class LocalCompletenessError(ValueError):
    """Raised when a score-blind C1 completeness invariant is violated."""


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise LocalCompletenessError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return _utc(value, label="timestamp").isoformat(timespec="microseconds").replace("+00:00", "Z")


def _local_contract_time(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LocalCompletenessError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LocalCompletenessError(f"{label} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(hours=8):
        raise LocalCompletenessError(f"{label} must use the frozen +08:00 offset")
    if parsed.time().replace(tzinfo=None) != datetime.min.time():
        raise LocalCompletenessError(f"{label} must be local midnight")
    return parsed.astimezone(UTC)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalCompletenessError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise LocalCompletenessError(f"{label} must be a sequence")
    return value


@dataclass(frozen=True, slots=True)
class CompletenessSnapshotAnchor:
    """One frozen mask identity and its causal catalog cutoff."""

    snapshot_id: str
    fold_id: str
    role: SnapshotRole
    block_id: str | None
    anchor_utc: datetime
    cutoff_utc: datetime

    def __post_init__(self) -> None:
        anchor = _utc(self.anchor_utc, label="anchor_utc")
        cutoff = _utc(self.cutoff_utc, label="cutoff_utc")
        if not self.snapshot_id or not self.fold_id or cutoff != anchor - C1_SNAPSHOT_DELAY:
            raise LocalCompletenessError("snapshot identity or anchor-minus-24h cutoff changed")
        if self.role == "inner_block_start":
            if self.block_id not in {"I1", "I2", "I3"}:
                raise LocalCompletenessError("inner snapshot must name I1, I2, or I3")
        elif self.role == "outer_fold_start":
            if self.block_id is not None:
                raise LocalCompletenessError("outer snapshot cannot name an inner block")
        else:
            raise LocalCompletenessError("unknown snapshot role")
        object.__setattr__(self, "anchor_utc", anchor)
        object.__setattr__(self, "cutoff_utc", cutoff)


@dataclass(frozen=True, slots=True)
class LocalCompletenessEvent:
    """One authenticated inside-study-area catalog row in equal-area metres."""

    event_id: str
    origin_time_utc: datetime
    available_at_utc: datetime
    magnitude: float
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        origin = _utc(self.origin_time_utc, label="origin_time_utc")
        available = _utc(self.available_at_utc, label="available_at_utc")
        if not self.event_id:
            raise LocalCompletenessError("event_id must not be empty")
        if available < origin:
            raise LocalCompletenessError("available_at cannot precede origin time")
        if not all(math.isfinite(float(value)) for value in (self.magnitude, self.x_m, self.y_m)):
            raise LocalCompletenessError("event magnitude and coordinates must be finite")
        if self.magnitude < 0.0:
            raise LocalCompletenessError("event magnitude must be non-negative")
        object.__setattr__(self, "origin_time_utc", origin)
        object.__setattr__(self, "available_at_utc", available)


@dataclass(frozen=True, slots=True)
class LocatedCompletenessEvent:
    """A catalog event assigned once to the target-independent base partition."""

    event: LocalCompletenessEvent
    cell_id: str
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class TemporalCoverageDiagnostic:
    """Descriptive time coverage; it never changes or hard-fails the spatial mask."""

    visible_event_count: int
    first_origin_utc: datetime | None
    last_origin_utc: datetime | None
    origin_year_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class LocalCompletenessCell:
    """One fixed-cell support decision and the two frozen training masks."""

    cell_id: str
    row: int
    column: int
    clipped_area_m2: float
    base_event_count: int
    parent_row: int
    parent_column: int
    parent_event_count: int
    estimate_source: EstimateSource
    status: CompletenessStatus
    raw_mc: float | None
    main_common_mc4_training_allowed: bool
    exclude_indeterminate_training_allowed: bool
    supported_area_contributor: bool

    def __post_init__(self) -> None:
        if self.status == "supported":
            expected = (True, True, True)
            if self.raw_mc is None or self.raw_mc > C1_MAXIMUM_SUPPORTED_RAW_MC:
                raise LocalCompletenessError("supported cell must have raw Mc <= 4")
        elif self.status == "indeterminate":
            expected = (True, False, False)
            if self.raw_mc is not None or self.estimate_source != "not_estimable":
                raise LocalCompletenessError("indeterminate cell cannot claim an Mc estimate")
        elif self.status == "unsupported":
            expected = (False, False, False)
            if self.raw_mc is None or self.raw_mc <= C1_MAXIMUM_SUPPORTED_RAW_MC:
                raise LocalCompletenessError("unsupported cell must have raw Mc > 4")
        else:
            raise LocalCompletenessError("unknown completeness status")
        observed = (
            self.main_common_mc4_training_allowed,
            self.exclude_indeterminate_training_allowed,
            self.supported_area_contributor,
        )
        if observed != expected:
            raise LocalCompletenessError("cell masks do not match the frozen status rules")


@dataclass(frozen=True, slots=True)
class LocalCompletenessSnapshot:
    """A complete score-blind C1 support diagnostic for one frozen anchor."""

    anchor: CompletenessSnapshotAnchor
    visible_event_count: int
    visible_event_sha256: str
    temporal_coverage: TemporalCoverageDiagnostic
    cells: tuple[LocalCompletenessCell, ...]
    study_area_sha256: str
    total_area_m2: float
    supported_area_m2: float
    supported_area_fraction: float
    support_gate_passed: bool

    def __post_init__(self) -> None:
        order = tuple((cell.row, cell.column) for cell in self.cells)
        if not order or order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise LocalCompletenessError("snapshot cells must be unique and sorted")
        total = math.fsum(cell.clipped_area_m2 for cell in self.cells)
        supported = math.fsum(
            cell.clipped_area_m2 for cell in self.cells if cell.supported_area_contributor
        )
        if self.total_area_m2 != total or self.supported_area_m2 != supported:
            raise LocalCompletenessError("snapshot area totals are inconsistent")
        fraction = supported / total
        if self.supported_area_fraction != fraction:
            raise LocalCompletenessError("supported area fraction is inconsistent")
        if self.support_gate_passed != (fraction >= C1_SUPPORT_GATE_FRACTION):
            raise LocalCompletenessError("support gate flag is inconsistent")


def build_completeness_snapshot_anchors(
    contract: Mapping[str, Any],
) -> tuple[CompletenessSnapshotAnchor, ...]:
    """Return the exact 12 inner and four outer mask anchors in contract order."""

    anchors: list[CompletenessSnapshotAnchor] = []
    folds = _sequence(contract.get("outer_folds"), label="outer_folds")
    for raw_fold in folds:
        fold = _mapping(raw_fold, label="outer_fold")
        fold_id = str(fold.get("id"))
        if not fold_id:
            raise LocalCompletenessError("outer fold ID is empty")
        blocks = _sequence(fold.get("inner_blocks"), label=f"{fold_id}.inner_blocks")
        if len(blocks) != 3:
            raise LocalCompletenessError("each fold must contain exactly three inner blocks")
        for raw_block in blocks:
            block = _mapping(raw_block, label=f"{fold_id}.inner_block")
            block_id = str(block.get("id"))
            anchor = _local_contract_time(block.get("start"), label=f"{fold_id}.{block_id}.start")
            anchors.append(
                CompletenessSnapshotAnchor(
                    snapshot_id=f"{fold_id}__{block_id}",
                    fold_id=fold_id,
                    role="inner_block_start",
                    block_id=block_id,
                    anchor_utc=anchor,
                    cutoff_utc=anchor - C1_SNAPSHOT_DELAY,
                )
            )
        outer_anchor = _local_contract_time(fold.get("outer_start"), label=f"{fold_id}.outer_start")
        anchors.append(
            CompletenessSnapshotAnchor(
                snapshot_id=f"{fold_id}__OUTER",
                fold_id=fold_id,
                role="outer_fold_start",
                block_id=None,
                anchor_utc=outer_anchor,
                cutoff_utc=outer_anchor - C1_SNAPSHOT_DELAY,
            )
        )
    if len(anchors) != 16 or len({item.snapshot_id for item in anchors}) != 16:
        raise LocalCompletenessError("C1 requires exactly 12 inner and four outer snapshots")
    return tuple(anchors)


def locate_completeness_events(
    events: Iterable[LocalCompletenessEvent],
    partition: LocalSupportBasePartition,
) -> tuple[LocatedCompletenessEvent, ...]:
    """Assign authenticated inside-domain events once, before causal snapshots."""

    supplied = tuple(events)
    seen: set[str] = set()
    located: list[LocatedCompletenessEvent] = []
    for event in supplied:
        if not isinstance(event, LocalCompletenessEvent):
            raise TypeError("events must contain LocalCompletenessEvent instances")
        if event.event_id in seen:
            raise LocalCompletenessError(f"duplicate physical event_id: {event.event_id}")
        seen.add(event.event_id)
        cell = partition.resolve(x_m=event.x_m, y_m=event.y_m)
        if cell is None:
            raise LocalCompletenessError(
                "an authenticated inside-study-area event did not resolve to the fixed partition"
            )
        located.append(LocatedCompletenessEvent(event, cell.cell_id, cell.row, cell.column))
    located.sort(key=lambda item: (item.event.origin_time_utc, item.event.event_id.encode("utf-8")))
    return tuple(located)


def _visible_events(
    events: Sequence[LocatedCompletenessEvent], cutoff_utc: datetime
) -> tuple[LocatedCompletenessEvent, ...]:
    cutoff = _utc(cutoff_utc, label="cutoff_utc")
    return tuple(
        item
        for item in events
        if item.event.origin_time_utc >= C1_HISTORY_START_UTC
        and item.event.origin_time_utc <= cutoff
        and item.event.available_at_utc <= cutoff
    )


def _visible_digest(events: Sequence[LocatedCompletenessEvent]) -> str:
    payload = [
        {
            "event_id": item.event.event_id,
            "origin_time_utc": _utc_text(item.event.origin_time_utc),
            "available_at_utc": _utc_text(item.event.available_at_utc),
            "magnitude": item.event.magnitude,
            "cell_id": item.cell_id,
        }
        for item in events
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _temporal_coverage(events: Sequence[LocatedCompletenessEvent]) -> TemporalCoverageDiagnostic:
    year_counts = Counter(item.event.origin_time_utc.year for item in events)
    return TemporalCoverageDiagnostic(
        visible_event_count=len(events),
        first_origin_utc=(events[0].event.origin_time_utc if events else None),
        last_origin_utc=(events[-1].event.origin_time_utc if events else None),
        origin_year_counts=tuple(sorted(year_counts.items())),
    )


def _raw_mc(events: Sequence[LocatedCompletenessEvent]) -> float:
    if len(events) < C1_MINIMUM_EVENTS:
        raise LocalCompletenessError("Mc estimation requires at least 200 events")
    return maximum_curvature_estimate(item.event.magnitude for item in events).corrected_magnitude


def build_local_completeness_snapshot(
    located_events: Sequence[LocatedCompletenessEvent],
    *,
    anchor: CompletenessSnapshotAnchor,
    partition: LocalSupportBasePartition,
) -> LocalCompletenessSnapshot:
    """Build one causal mask without changing or hard-failing the national domain."""

    visible = _visible_events(located_events, anchor.cutoff_utc)
    by_base: dict[tuple[int, int], list[LocatedCompletenessEvent]] = defaultdict(list)
    by_parent: dict[tuple[int, int], list[LocatedCompletenessEvent]] = defaultdict(list)
    for item in visible:
        base_key = (item.row, item.column)
        parent_key = (item.row // 2, item.column // 2)
        by_base[base_key].append(item)
        by_parent[parent_key].append(item)

    parent_mc: dict[tuple[int, int], float] = {}
    cells: list[LocalCompletenessCell] = []
    for base in partition.cells:
        key = (base.row, base.column)
        parent_key = (base.row // 2, base.column // 2)
        base_events = by_base.get(key, [])
        parent_events = by_parent.get(parent_key, [])
        if len(base_events) >= C1_MINIMUM_EVENTS:
            raw_mc = _raw_mc(base_events)
            source: EstimateSource = "base_500km"
        elif len(parent_events) >= C1_MINIMUM_EVENTS:
            if parent_key not in parent_mc:
                parent_mc[parent_key] = _raw_mc(parent_events)
            raw_mc = parent_mc[parent_key]
            source = "parent_1000km"
        else:
            raw_mc = None
            source = "not_estimable"

        if raw_mc is None:
            status: CompletenessStatus = "indeterminate"
        elif raw_mc > C1_MAXIMUM_SUPPORTED_RAW_MC:
            status = "unsupported"
        else:
            status = "supported"
        cells.append(
            LocalCompletenessCell(
                cell_id=base.cell_id,
                row=base.row,
                column=base.column,
                clipped_area_m2=base.clipped_area_m2,
                base_event_count=len(base_events),
                parent_row=parent_key[0],
                parent_column=parent_key[1],
                parent_event_count=len(parent_events),
                estimate_source=source,
                status=status,
                raw_mc=raw_mc,
                main_common_mc4_training_allowed=status != "unsupported",
                exclude_indeterminate_training_allowed=status == "supported",
                supported_area_contributor=status == "supported",
            )
        )

    cell_tuple = tuple(cells)
    supported_area = math.fsum(
        cell.clipped_area_m2 for cell in cell_tuple if cell.supported_area_contributor
    )
    fraction = supported_area / partition.total_area_m2
    return LocalCompletenessSnapshot(
        anchor=anchor,
        visible_event_count=len(visible),
        visible_event_sha256=_visible_digest(visible),
        temporal_coverage=_temporal_coverage(visible),
        cells=cell_tuple,
        study_area_sha256=partition.study_area_sha256,
        total_area_m2=partition.total_area_m2,
        supported_area_m2=supported_area,
        supported_area_fraction=fraction,
        support_gate_passed=fraction >= C1_SUPPORT_GATE_FRACTION,
    )


def snapshot_summary_record(snapshot: LocalCompletenessSnapshot) -> dict[str, object]:
    """Return the deterministic JSON-safe, result-free snapshot summary."""

    counts = Counter(cell.status for cell in snapshot.cells)
    temporal = snapshot.temporal_coverage
    return {
        "snapshot_id": snapshot.anchor.snapshot_id,
        "fold_id": snapshot.anchor.fold_id,
        "anchor_role": snapshot.anchor.role,
        "block_id": snapshot.anchor.block_id,
        "anchor_utc": _utc_text(snapshot.anchor.anchor_utc),
        "cutoff_utc": _utc_text(snapshot.anchor.cutoff_utc),
        "visible_catalog_event_count": snapshot.visible_event_count,
        "visible_catalog_event_sha256": snapshot.visible_event_sha256,
        "temporal_coverage_diagnostic_only": {
            "changes_spatial_training_mask": False,
            "hard_failure": False,
            "first_origin_utc": (
                _utc_text(temporal.first_origin_utc) if temporal.first_origin_utc else None
            ),
            "last_origin_utc": (
                _utc_text(temporal.last_origin_utc) if temporal.last_origin_utc else None
            ),
            "origin_year_counts": [
                {"year": year, "event_count": count} for year, count in temporal.origin_year_counts
            ],
        },
        "cell_status_counts": {
            "supported": counts["supported"],
            "indeterminate": counts["indeterminate"],
            "unsupported": counts["unsupported"],
        },
        "total_area_m2": snapshot.total_area_m2,
        "supported_area_m2": snapshot.supported_area_m2,
        "supported_area_fraction": snapshot.supported_area_fraction,
        "minimum_required_supported_area_fraction": C1_SUPPORT_GATE_FRACTION,
        "support_gate_passed": snapshot.support_gate_passed,
    }


def snapshot_cell_records(snapshot: LocalCompletenessSnapshot) -> tuple[dict[str, object], ...]:
    """Return fixed-cell CSV records containing both frozen training masks."""

    prefix = {
        "snapshot_id": snapshot.anchor.snapshot_id,
        "fold_id": snapshot.anchor.fold_id,
        "anchor_role": snapshot.anchor.role,
        "block_id": snapshot.anchor.block_id or "",
        "anchor_utc": _utc_text(snapshot.anchor.anchor_utc),
        "cutoff_utc": _utc_text(snapshot.anchor.cutoff_utc),
    }
    return tuple(
        {
            **prefix,
            "cell_id": cell.cell_id,
            "row": cell.row,
            "column": cell.column,
            "clipped_area_m2": cell.clipped_area_m2,
            "base_event_count": cell.base_event_count,
            "parent_row": cell.parent_row,
            "parent_column": cell.parent_column,
            "parent_event_count": cell.parent_event_count,
            "estimate_source": cell.estimate_source,
            "status": cell.status,
            "raw_mc": "" if cell.raw_mc is None else cell.raw_mc,
            "main_common_mc4_training_allowed": cell.main_common_mc4_training_allowed,
            "exclude_indeterminate_training_allowed": (cell.exclude_indeterminate_training_allowed),
            "supported_area_contributor": cell.supported_area_contributor,
        }
        for cell in snapshot.cells
    )


__all__ = [
    "C1_HISTORY_START_UTC",
    "C1_MAXIMUM_SUPPORTED_RAW_MC",
    "C1_MINIMUM_EVENTS",
    "C1_SNAPSHOT_DELAY",
    "C1_SUPPORT_GATE_FRACTION",
    "CompletenessSnapshotAnchor",
    "LocalCompletenessCell",
    "LocalCompletenessError",
    "LocalCompletenessEvent",
    "LocalCompletenessSnapshot",
    "LocatedCompletenessEvent",
    "TemporalCoverageDiagnostic",
    "build_completeness_snapshot_anchors",
    "build_local_completeness_snapshot",
    "locate_completeness_events",
    "snapshot_cell_records",
    "snapshot_summary_record",
]
