"""Validated issue snapshots and spatial arrays derived from stage-3 state rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pyproj import CRS, Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.spatial import (
    DISCIPLINE_NAMES,
    SpatialEntityArrays,
)
from seismoflux.features.anomaly.state import AnomalyState

_DISCIPLINE_CODE = {
    "形变": 0,
    "流体": 1,
    "电磁": 2,
    "跨断层": 3,
}
if tuple(DISCIPLINE_NAMES) != (
    "deformation",
    "fluid",
    "electromagnetic",
    "cross_fault",
):
    raise AssertionError("spatial discipline order changed")


@dataclass(frozen=True, slots=True)
class Stage3IssueSnapshot:
    """One actual issue with one report summary and deterministic entity states."""

    issue_index: int
    issue_time_utc: datetime
    summary: AnomalyState
    entities: tuple[AnomalyState, ...]
    state_snapshot_id: str
    lineage_digest: str


def _snapshot_id(states: tuple[AnomalyState, ...], *, lineage: bool) -> str:
    if not states:
        raise ValueError("a stage-3 issue snapshot cannot be empty")
    summary = next(
        (state for state in states if state.state_row_kind == "report_period_summary"), None
    )
    if summary is None:
        raise ValueError("a stage-3 issue snapshot requires a report summary row")
    payload: dict[str, object] = {
        "contract_version": summary.contract_version,
        "issue_time_utc": summary.issue_time_utc.isoformat(),
        "issue_report_id": summary.issue_report_id,
    }
    if lineage:
        payload["state_lineage"] = [
            {
                "state_id": state.state_id,
                "lineage_sha256": state.lineage_sha256,
                "lineage_max_available_at_utc": state.lineage_max_available_at_utc.isoformat(),
            }
            for state in states
        ]
    else:
        payload["state_ids"] = [state.state_id for state in states]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_issue_snapshots(
    states: tuple[AnomalyState, ...],
    *,
    expected_issue_count: int | None = None,
) -> tuple[Stage3IssueSnapshot, ...]:
    """Group the total-sorted state table and enforce causal lineage boundaries."""

    if not states:
        raise ValueError("stage-3 state history must not be empty")
    grouped: dict[datetime, list[AnomalyState]] = {}
    for state in states:
        if state.lineage_max_available_at_utc > state.issue_time_utc:
            raise ValueError("state lineage contains a source later than its issue")
        grouped.setdefault(state.issue_time_utc, []).append(state)
    times = tuple(sorted(grouped))
    if expected_issue_count is not None and len(times) != expected_issue_count:
        raise ValueError(
            "stage-3 issue snapshot count mismatch: "
            f"expected={expected_issue_count}, observed={len(times)}"
        )

    output: list[Stage3IssueSnapshot] = []
    state_ids: set[str] = set()
    for issue_index, issue_time in enumerate(times):
        issue_states = tuple(
            sorted(
                grouped[issue_time],
                key=lambda state: (
                    state.state_row_kind,
                    state.anomaly_id,
                    state.entity_scope,
                    state.state_id,
                ),
            )
        )
        summaries = tuple(
            state for state in issue_states if state.state_row_kind == "report_period_summary"
        )
        if len(summaries) != 1:
            raise ValueError("every actual issue must have exactly one report summary row")
        summary = summaries[0]
        entities = tuple(state for state in issue_states if state.state_row_kind == "entity_state")
        if any(
            state.issue_report_id != summary.issue_report_id
            or state.issue_report_date != summary.issue_report_date
            or state.issue_time_utc != summary.issue_time_utc
            for state in entities
        ):
            raise ValueError("entity and report-summary issue identities disagree")
        duplicate_ids = state_ids.intersection(state.state_id for state in issue_states)
        if duplicate_ids:
            raise ValueError(f"duplicate state IDs across issue snapshots: {sorted(duplicate_ids)}")
        state_ids.update(state.state_id for state in issue_states)
        output.append(
            Stage3IssueSnapshot(
                issue_index=issue_index,
                issue_time_utc=issue_time,
                summary=summary,
                entities=entities,
                state_snapshot_id=_snapshot_id(issue_states, lineage=False),
                lineage_digest=_snapshot_id(issue_states, lineage=True),
            )
        )
    return tuple(output)


def _entity_disciplines(state: AnomalyState) -> tuple[str, ...]:
    current = getattr(state, "current_reporting_disciplines", ())
    known = getattr(state, "known_disciplines", ())
    candidates = current if state.current_report_listed and current else known
    if candidates:
        return tuple(sorted(set(candidates)))
    if state.discipline is not None:
        return (state.discipline,)
    raise ValueError(f"entity state has no replayable discipline: {state.state_id}")


def spatial_entity_arrays(snapshot: Stage3IssueSnapshot) -> SpatialEntityArrays:
    """Convert entity states to the exact arrays consumed by spatial aggregation."""

    entities = snapshot.entities
    size = len(entities)
    lon = np.full(size, np.nan, dtype=np.float64)
    lat = np.full(size, np.nan, dtype=np.float64)
    for index, state in enumerate(entities):
        if state.spatial_eligible:
            if state.longitude is None or state.latitude is None:
                raise ValueError("spatial-eligible state has no finite coordinate pair")
            lon[index] = state.longitude
            lat[index] = state.latitude
    xy_m = np.full((size, 2), np.nan, dtype=np.float64)
    coordinate_mask = np.isfinite(lon) & np.isfinite(lat)
    if np.any(coordinate_mask):
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326),
            CRS.from_user_input(EQUAL_AREA_CRS),
            always_xy=True,
        )
        x_m, y_m = transformer.transform(lon[coordinate_mask], lat[coordinate_mask])
        xy_m[coordinate_mask, 0] = np.asarray(x_m, dtype=np.float64)
        xy_m[coordinate_mask, 1] = np.asarray(y_m, dtype=np.float64)

    complete = np.asarray([state.identity_complete for state in entities], dtype=np.bool_)
    current = np.asarray([state.current_report_listed for state in entities], dtype=np.bool_)
    left_truncated = np.asarray([state.left_truncated for state in entities], dtype=np.bool_)
    disciplines = tuple(_entity_disciplines(state) for state in entities)
    discipline_membership = np.zeros((size, len(DISCIPLINE_NAMES)), dtype=np.bool_)
    discipline_code = np.zeros(size, dtype=np.int64)
    for index, values in enumerate(disciplines):
        codes: list[int] = []
        for value in values:
            try:
                codes.append(_DISCIPLINE_CODE[value])
            except KeyError as exc:
                raise ValueError(f"unsupported anomaly discipline: {value}") from exc
        discipline_code[index] = min(codes)
        discipline_membership[index, codes] = True

    return SpatialEntityArrays(
        xy_m=xy_m,
        listed=complete & current,
        source_new=np.asarray(
            [
                state.identity_complete and state.current_report_listed and state.source_new
                for state in entities
            ],
            dtype=np.bool_,
        ),
        first_seen=np.asarray(
            [
                state.identity_complete and state.system_first_seen and not state.left_truncated
                for state in entities
            ],
            dtype=np.bool_,
        ),
        explicit_end=np.asarray(
            [state.identity_complete and state.explicit_end_known for state in entities],
            dtype=np.bool_,
        ),
        not_continued=np.asarray(
            [state.identity_complete and state.system_not_continued for state in entities],
            dtype=np.bool_,
        ),
        relisted=np.asarray(
            [state.identity_complete and state.system_relisted for state in entities],
            dtype=np.bool_,
        ),
        right_censored=np.asarray(
            [
                state.identity_complete and state.current_report_listed and state.right_censored
                for state in entities
            ],
            dtype=np.bool_,
        ),
        reliability_high=np.asarray(
            [state.reliability_grade == "high" for state in entities],
            dtype=np.bool_,
        ),
        reliability_cautious=np.asarray(
            [state.reliability_grade == "cautious" for state in entities],
            dtype=np.bool_,
        ),
        station_id=np.asarray(
            [state.station_id or "" for state in entities],
            dtype=object,
        ),
        measurement_id=np.asarray(
            [state.measurement or "" for state in entities],
            dtype=object,
        ),
        discipline_code=discipline_code,
        age_days=np.asarray(
            [np.nan if state.age_days is None else state.age_days for state in entities],
            dtype=np.float64,
        ),
        known_duration_days=np.asarray(
            [
                np.nan if state.known_duration_days is None else state.known_duration_days
                for state in entities
            ],
            dtype=np.float64,
        ),
        left_truncated=complete & current & left_truncated,
        late_entry=np.asarray(
            [
                state.identity_complete
                and state.current_report_listed
                and state.late_entry_or_gap_unknown
                for state in entities
            ],
            dtype=np.bool_,
        ),
        temporary=~complete & current,
        multidisciplinary=np.asarray([len(values) > 1 for values in disciplines], dtype=np.bool_),
        discipline_membership=discipline_membership,
        reporting_listed=current,
    )


__all__ = [
    "Stage3IssueSnapshot",
    "build_issue_snapshots",
    "spatial_entity_arrays",
]
