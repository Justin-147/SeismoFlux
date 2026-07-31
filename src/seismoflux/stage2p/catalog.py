"""Pure in-memory synthetic catalog windows for the Stage 2P science MVP."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

QUERY_LEAD = timedelta(minutes=15)
RECENT_WINDOW = timedelta(days=30)
RECENT_MAGNITUDE_MINIMUM = 4.0


def _utc(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    result = value.astimezone(UTC)
    if result.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must resolve to UTC")
    return result


@dataclass(frozen=True, slots=True)
class SyntheticEvent:
    """One synthetic event known to the forecaster at a recorded time."""

    id: str
    origin_time: datetime
    first_seen: datetime
    x_km: float
    y_km: float
    magnitude: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not (event_id := self.id.strip()):
            raise ValueError("id must be a non-empty string")
        origin = _utc("origin_time", self.origin_time)
        first_seen = _utc("first_seen", self.first_seen)
        if first_seen < origin:
            raise ValueError("first_seen cannot precede origin_time")
        coordinates_and_magnitude = (
            float(self.x_km),
            float(self.y_km),
            float(self.magnitude),
        )
        if any(not math.isfinite(value) for value in coordinates_and_magnitude):
            raise ValueError("coordinates and magnitude must be finite")
        object.__setattr__(self, "id", event_id)
        object.__setattr__(self, "origin_time", origin)
        object.__setattr__(self, "first_seen", first_seen)
        object.__setattr__(self, "x_km", coordinates_and_magnitude[0])
        object.__setattr__(self, "y_km", coordinates_and_magnitude[1])
        object.__setattr__(self, "magnitude", coordinates_and_magnitude[2])


Event = SyntheticEvent


@dataclass(frozen=True, slots=True)
class CausalCatalogWindows:
    """The three deterministic, target-free event views for one issue."""

    issue_time: datetime
    query_cutoff: datetime
    training_start: datetime
    p0_events: tuple[SyntheticEvent, ...]
    r30_events: tuple[SyntheticEvent, ...]
    rp30_events: tuple[SyntheticEvent, ...]

    def __post_init__(self) -> None:
        issue_time = _utc("issue_time", self.issue_time)
        query_cutoff = _utc("query_cutoff", self.query_cutoff)
        training_start = _utc("training_start", self.training_start)
        if query_cutoff != issue_time - QUERY_LEAD:
            raise ValueError("query_cutoff must equal issue_time minus 15 minutes")
        if training_start > query_cutoff:
            raise ValueError("training_start cannot follow query_cutoff")
        object.__setattr__(self, "issue_time", issue_time)
        object.__setattr__(self, "query_cutoff", query_cutoff)
        object.__setattr__(self, "training_start", training_start)
        normalized: dict[str, tuple[SyntheticEvent, ...]] = {}
        for name in ("p0_events", "r30_events", "rp30_events"):
            events = tuple(getattr(self, name))
            if any(not isinstance(event, SyntheticEvent) for event in events):
                raise TypeError(f"{name} must contain only SyntheticEvent values")
            normalized[name] = events
            object.__setattr__(self, name, events)
        p0 = normalized["p0_events"]
        p0_ids = tuple(event.id for event in p0)
        if len(set(p0_ids)) != len(p0_ids):
            raise ValueError("p0_events must contain unique event ids")
        if p0 != tuple(sorted(p0, key=_fixed_order)):
            raise ValueError("p0_events must use fixed origin-time/event-id order")
        for event in p0:
            if not training_start <= event.origin_time <= query_cutoff:
                raise ValueError("every P0 event must lie within training_start through Q")
            if event.first_seen >= issue_time:
                raise ValueError("every P0 event must be first seen before T")
        expected_r30, expected_rp30 = _derive_recent_windows(p0, query_cutoff)
        if normalized["r30_events"] != expected_r30:
            raise ValueError("r30_events must be the complete ordered R30 subset of P0")
        if normalized["rp30_events"] != expected_rp30:
            raise ValueError("rp30_events must be the complete ordered RP30 subset of P0")


def _fixed_order(event: SyntheticEvent) -> tuple[datetime, bytes]:
    return (event.origin_time, event.id.encode("utf-8"))


def _derive_recent_windows(
    p0_events: tuple[SyntheticEvent, ...],
    query_cutoff: datetime,
) -> tuple[tuple[SyntheticEvent, ...], tuple[SyntheticEvent, ...]]:
    recent_start = query_cutoff - RECENT_WINDOW
    preceding_start = query_cutoff - 2 * RECENT_WINDOW
    r30 = tuple(
        event
        for event in p0_events
        if recent_start < event.origin_time <= query_cutoff
        and event.magnitude >= RECENT_MAGNITUDE_MINIMUM
    )
    rp30 = tuple(
        event
        for event in p0_events
        if preceding_start < event.origin_time <= recent_start
        and event.magnitude >= RECENT_MAGNITUDE_MINIMUM
    )
    return r30, rp30


def select_causal_windows(
    events: Iterable[SyntheticEvent],
    *,
    issue_time: datetime,
    query_cutoff: datetime,
    training_start: datetime,
) -> CausalCatalogWindows:
    """Select P0, R30 and RP30 without accepting any future target row.

    The input is deliberately only an in-memory iterable.  Passing an event
    that was not visible before ``T`` or whose origin follows ``Q`` fails
    closed instead of silently discarding a future row.
    """

    issue = _utc("issue_time", issue_time)
    cutoff = _utc("query_cutoff", query_cutoff)
    start = _utc("training_start", training_start)
    if cutoff != issue - QUERY_LEAD:
        raise ValueError("query_cutoff must equal issue_time minus 15 minutes")
    if start > cutoff:
        raise ValueError("training_start cannot follow query_cutoff")

    selected = tuple(events)
    if any(not isinstance(event, SyntheticEvent) for event in selected):
        raise TypeError("events must contain only SyntheticEvent values")
    event_ids = tuple(event.id for event in selected)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("synthetic event ids must be unique")
    for event in selected:
        if event.origin_time > cutoff:
            raise ValueError("an event originating after Q cannot enter a forecast")
        if event.first_seen >= issue:
            raise ValueError("an event first seen at or after T cannot enter a forecast")

    ordered = tuple(sorted(selected, key=_fixed_order))
    p0 = tuple(event for event in ordered if start <= event.origin_time <= cutoff)
    r30, rp30 = _derive_recent_windows(p0, cutoff)
    return CausalCatalogWindows(
        issue_time=issue,
        query_cutoff=cutoff,
        training_start=start,
        p0_events=p0,
        r30_events=r30,
        rp30_events=rp30,
    )


__all__ = [
    "QUERY_LEAD",
    "RECENT_MAGNITUDE_MINIMUM",
    "RECENT_WINDOW",
    "CausalCatalogWindows",
    "Event",
    "SyntheticEvent",
    "select_causal_windows",
]
