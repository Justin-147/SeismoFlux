from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Any

import pyarrow as pa
import pytest

from seismoflux.features.anomaly.contracts import ANOMALY_STATE_HISTORY_CONTRACT
from seismoflux.features.anomaly.state import (
    AnomalyState,
    build_anomaly_state_history,
    state_records,
    states_from_records,
)

LOCAL = timezone(timedelta(hours=8))


def _period(
    report_date: date,
    number: int,
    *,
    row_count: int = 1,
    mismatch_count: int = 0,
    before_count: int = 0,
    after_count: int = 0,
    deformation_count: int | None = None,
    fluid_count: int = 0,
    electromagnetic_count: int = 0,
    cross_fault_count: int = 0,
) -> dict[str, object]:
    available_local = datetime.combine(report_date + timedelta(days=1), time.min, LOCAL)
    resolved_deformation_count = (
        row_count - fluid_count - electromagnetic_count - cross_fault_count
        if deformation_count is None
        else deformation_count
    )
    return {
        "report_id": f"report-{number}",
        "source_file": f"anomaly/report-{number}.xls",
        "report_year": report_date.year,
        "report_period": number,
        "report_date": report_date,
        "available_at": available_local.astimezone(UTC),
        "row_count": row_count,
        "row_report_date_mismatch_count": mismatch_count,
        "row_report_date_before_count": before_count,
        "row_report_date_after_count": after_count,
        "deformation_row_count": resolved_deformation_count,
        "fluid_row_count": fluid_count,
        "electromagnetic_row_count": electromagnetic_count,
        "cross_fault_row_count": cross_fault_count,
    }


def _observation(
    period: dict[str, object],
    observation_id: str,
    anomaly_id: str,
    *,
    identity_complete: bool = True,
    report_state: str = "持续",
    available_at: datetime | None = None,
    station_id: str = "station-a",
    longitude: float | None = 110.0,
    latitude: float | None = 35.0,
    discipline: str = "deformation",
    measurement: str | None = "measurement-a",
    start_time: datetime | None = None,
    is_listed: bool = True,
    reported_end_time: datetime | None = None,
    right_censored: bool = True,
    reliability_flags: tuple[str, ...] = (),
) -> dict[str, object]:
    period_available_at = period["available_at"]
    report_date = period["report_date"]
    source_file = period["source_file"]
    assert isinstance(period_available_at, datetime)
    assert isinstance(report_date, date)
    assert not isinstance(report_date, datetime)
    assert isinstance(source_file, str)
    return {
        "observation_id": observation_id,
        "anomaly_id": anomaly_id,
        "identity_complete": identity_complete,
        "report_date": report_date,
        "source_file": source_file,
        "available_at": period_available_at if available_at is None else available_at,
        "station_id": station_id,
        "longitude": longitude,
        "latitude": latitude,
        "discipline": discipline,
        "measurement": measurement,
        "start_time": (
            datetime.combine(report_date - timedelta(days=30), time.min, LOCAL)
            if start_time is None
            else start_time
        ),
        "is_listed": is_listed,
        "report_state": report_state,
        "reported_end_time": reported_end_time,
        "right_censored": right_censored,
        "reliability_flags": reliability_flags,
    }


def _states_for(states: tuple[AnomalyState, ...], anomaly_id: str) -> tuple[AnomalyState, ...]:
    return tuple(state for state in states if state.anomaly_id == anomaly_id)


def test_later_end_confirmation_does_not_backfill_an_earlier_issue() -> None:
    first = _period(date(2024, 1, 1), 1)
    second = _period(date(2024, 1, 8), 2)
    observations = [
        _observation(first, "obs-1", "entity-a"),
        _observation(
            second,
            "obs-2",
            "entity-a",
            report_state="取消",
            reported_end_time=datetime(2024, 1, 5, tzinfo=LOCAL),
            right_censored=False,
        ),
    ]

    entity_states = _states_for(
        build_anomaly_state_history(observations, [second, first]), "entity-a"
    )

    assert len(entity_states) == 2
    assert entity_states[0].explicit_end_known is False
    assert entity_states[0].right_censored is True
    assert entity_states[0].lineage_observation_count == 1
    assert entity_states[0].latest_observation_id == "obs-1"
    assert entity_states[1].explicit_end_known is True
    assert entity_states[1].source_cancelled is True
    assert entity_states[1].lineage_observation_count == 2
    assert entity_states[1].latest_observation_id == "obs-2"


def test_future_end_never_changes_state_from_calendar_time_alone() -> None:
    periods = [
        _period(date(2024, 1, 1), 1),
        _period(date(2024, 1, 8), 2),
        _period(date(2024, 1, 15), 3),
    ]
    observations = [
        _observation(
            periods[0],
            "obs-future-end",
            "entity-a",
            reported_end_time=datetime(2024, 1, 10, tzinfo=LOCAL),
            right_censored=True,
        )
    ]

    entity_states = _states_for(build_anomaly_state_history(observations, periods), "entity-a")

    assert len(entity_states) == 3
    assert [state.explicit_end_known for state in entity_states] == [False, False, False]
    assert [state.right_censored for state in entity_states] == [True, True, True]
    assert entity_states[2].issue_report_date > date(2024, 1, 10)
    assert entity_states[2].latest_observation_id == "obs-future-end"


def test_missing_whole_period_does_not_trigger_not_continued() -> None:
    first = _period(date(2024, 1, 1), 1)
    after_gap = _period(date(2024, 1, 15), 3)

    entity_states = _states_for(
        build_anomaly_state_history(
            [_observation(first, "obs-1", "entity-a")],
            [first, after_gap],
        ),
        "entity-a",
    )

    assert entity_states[1].previous_period_consecutive is False
    assert entity_states[1].current_report_listed is False
    assert entity_states[1].system_not_continued is False


def test_source_states_and_system_transitions_remain_separate() -> None:
    periods = [
        _period(date(2024, 1, 1), 1),
        _period(date(2024, 1, 8), 2),
        _period(date(2024, 1, 15), 3),
    ]
    observations = [
        _observation(periods[0], "obs-1", "entity-a", report_state="持续"),
        _observation(periods[2], "obs-3", "entity-a", report_state="持续"),
    ]

    entity_states = _states_for(build_anomaly_state_history(observations, periods), "entity-a")

    assert entity_states[0].system_first_seen is True
    assert entity_states[0].source_new is False
    assert entity_states[0].source_continued is True
    assert entity_states[1].system_not_continued is True
    assert entity_states[1].current_source_report_states == ()
    assert entity_states[2].system_relisted is True
    assert entity_states[2].source_new is False
    assert entity_states[2].source_continued is True


def test_incomplete_identity_is_report_local_even_when_id_is_reused() -> None:
    periods = [
        _period(date(2024, 1, 1), 1),
        _period(date(2024, 1, 8), 2),
        _period(date(2024, 1, 15), 3),
    ]
    observations = [
        _observation(periods[0], "temporary-1", "provisional", identity_complete=False),
        _observation(periods[1], "temporary-2", "provisional", identity_complete=False),
    ]

    temporary_states = _states_for(
        build_anomaly_state_history(observations, periods), "provisional"
    )

    assert len(temporary_states) == 2
    assert all(not state.identity_complete for state in temporary_states)
    assert all(state.entity_scope == "report_local_incomplete_entity" for state in temporary_states)
    assert [state.lineage_observation_count for state in temporary_states] == [1, 1]
    assert [state.current_observation_ids for state in temporary_states] == [
        ("temporary-1",),
        ("temporary-2",),
    ]
    assert all(state.system_first_seen for state in temporary_states)
    assert all(not state.system_not_continued for state in temporary_states)


def test_same_entity_report_is_deduplicated_but_all_observation_ids_are_retained() -> None:
    period = _period(date(2024, 1, 1), 1)
    observations = [
        _observation(period, "duplicate-b", "entity-a", report_state="持续"),
        _observation(period, "duplicate-a", "entity-a", report_state="新增"),
    ]

    forward = build_anomaly_state_history(observations, [period])
    reverse = build_anomaly_state_history(list(reversed(observations)), [period])

    assert forward == reverse
    entity_states = _states_for(forward, "entity-a")
    assert len(entity_states) == 1
    state = entity_states[0]
    assert state.current_observation_ids == ("duplicate-a", "duplicate-b")
    assert state.latest_observation_ids == ("duplicate-a", "duplicate-b")
    assert state.lineage_observation_ids == ("duplicate-a", "duplicate-b")
    assert state.lineage_source_report_ids == ("report-1",)
    assert state.lineage_observation_count == 2
    assert state.source_new is True
    assert state.source_continued is True
    assert state.current_source_report_states == ("新增", "持续")


def test_conflicting_duplicate_end_evidence_is_conservatively_right_censored() -> None:
    period = _period(date(2024, 1, 1), 1)
    observations = [
        _observation(
            period,
            "end-a",
            "entity-a",
            reported_end_time=datetime(2023, 12, 30, tzinfo=LOCAL),
            right_censored=False,
        ),
        _observation(
            period,
            "end-b",
            "entity-a",
            reported_end_time=datetime(2023, 12, 31, tzinfo=LOCAL),
            right_censored=False,
        ),
    ]

    state = build_anomaly_state_history(observations, [period])[0]

    assert state.explicit_end_known is False
    assert state.right_censored is True
    assert state.current_observation_ids == ("end-a", "end-b")


def test_each_incomplete_observation_remains_its_own_report_local_entity() -> None:
    period = _period(date(2024, 1, 1), 1)
    observations = [
        _observation(period, "temp-a", "same-provisional", identity_complete=False),
        _observation(period, "temp-b", "same-provisional", identity_complete=False),
    ]

    states = _states_for(
        build_anomaly_state_history(observations, [period]),
        "same-provisional",
    )

    assert len(states) == 2
    assert len({state.state_id for state in states}) == 2
    assert [state.current_observation_ids for state in states] == [
        ("temp-a",),
        ("temp-b",),
    ]


def test_delayed_row_enters_only_at_the_first_closed_issue_boundary() -> None:
    first = _period(date(2024, 1, 1), 1)
    second = _period(date(2024, 1, 8), 2)
    second_issue = second["available_at"]
    assert isinstance(second_issue, datetime)
    delayed = _observation(
        first,
        "delayed",
        "entity-a",
        available_at=second_issue,
    )
    future = _observation(
        first,
        "future",
        "entity-b",
        available_at=second_issue + timedelta(microseconds=1),
    )

    states = build_anomaly_state_history([future, delayed], [first, second])

    assert _states_for(states, "entity-a")[0].issue_report_id == "report-2"
    delayed_state = _states_for(states, "entity-a")[0]
    assert delayed_state.current_report_listed is False
    assert delayed_state.system_first_seen is True
    assert _states_for(states, "entity-b") == ()


def test_future_record_mutation_cannot_change_prior_state_or_lineage() -> None:
    first = _period(date(2024, 1, 1), 1)
    second = _period(date(2024, 1, 8), 2)
    observations = [
        _observation(first, "obs-1", "entity-a"),
        _observation(second, "obs-2", "entity-a"),
    ]
    mutated = deepcopy(observations)
    mutated[1]["report_state"] = "取消"
    mutated[1]["reported_end_time"] = datetime(2024, 1, 2, tzinfo=LOCAL)
    mutated[1]["right_censored"] = False
    mutated[1]["longitude"] = 179.0
    mutated[1]["latitude"] = -80.0
    mutated[1]["measurement"] = "future-mutated-measurement"
    mutated[1]["start_time"] = datetime(2023, 1, 1, tzinfo=LOCAL)
    mutated[1]["reliability_flags"] = ("identity_incomplete",)

    original_first = _states_for(
        build_anomaly_state_history(observations, [first, second]), "entity-a"
    )[0]
    mutated_first = _states_for(build_anomaly_state_history(mutated, [first, second]), "entity-a")[
        0
    ]

    assert original_first == mutated_first


def test_future_report_metadata_cannot_change_prior_snapshot() -> None:
    first = _period(date(2024, 1, 1), 1)
    second = _period(date(2024, 1, 8), 2, row_count=0)
    mutated_second = deepcopy(second)
    mutated_second["row_count"] = 3
    mutated_second["deformation_row_count"] = 1
    mutated_second["fluid_row_count"] = 1
    mutated_second["electromagnetic_row_count"] = 1
    observation = _observation(first, "obs-1", "entity-a")

    prefix = build_anomaly_state_history([observation], [first])
    full = build_anomaly_state_history([observation], [first, mutated_second])
    full_first_issue = tuple(state for state in full if state.issue_report_id == "report-1")

    assert full_first_issue == prefix
    assert all(state.max_source_available_at <= state.issue_time_utc for state in full)


def test_state_records_round_trip_through_the_stage3_arrow_schema() -> None:
    period = _period(date(2024, 1, 1), 1)
    states = build_anomaly_state_history([_observation(period, "obs-1", "entity-a")], [period])

    table = pa.Table.from_pylist(
        list(state_records(states)),
        schema=ANOMALY_STATE_HISTORY_CONTRACT.schema,
    )

    assert table.schema == ANOMALY_STATE_HISTORY_CONTRACT.schema
    assert table.num_rows == 2
    assert table.column("state_id").to_pylist() == [state.state_id for state in states]
    assert states_from_records(table.to_pylist()) == states


def test_state_history_contains_all_fields_needed_for_causal_feature_replay() -> None:
    first = _period(
        date(2024, 1, 1),
        1,
        row_count=4,
        mismatch_count=2,
        before_count=1,
        after_count=1,
        deformation_count=1,
        fluid_count=1,
        electromagnetic_count=1,
        cross_fault_count=1,
    )
    second = _period(date(2024, 1, 8), 2, row_count=0)
    observation = _observation(
        first,
        "obs-1",
        "entity-a",
        station_id="station-147",
        longitude=103.25,
        latitude=31.75,
        discipline="fluid",
        measurement="groundwater-level",
        start_time=datetime(2023, 12, 20, tzinfo=LOCAL),
    )

    states = build_anomaly_state_history([observation], [first, second])
    summaries = tuple(state for state in states if state.state_row_kind == "report_period_summary")
    entity_states = _states_for(states, "entity-a")

    assert len(summaries) == 2
    assert summaries[0].row_count == 4
    assert summaries[0].row_report_date_mismatch_count == 2
    assert summaries[0].row_report_date_before_count == 1
    assert summaries[0].row_report_date_after_count == 1
    assert summaries[0].deformation_row_count == 1
    assert summaries[0].fluid_row_count == 1
    assert summaries[0].electromagnetic_row_count == 1
    assert summaries[0].cross_fault_row_count == 1
    assert summaries[1].row_count == 0

    latest = entity_states[-1]
    assert latest.current_report_listed is False
    assert latest.station_id == "station-147"
    assert latest.longitude == pytest.approx(103.25)
    assert latest.latitude == pytest.approx(31.75)
    assert latest.discipline == "fluid"
    assert latest.measurement == "groundwater-level"
    assert latest.start_time == datetime(2023, 12, 20, tzinfo=LOCAL)
    assert latest.current_reporting_station_ids == ()
    assert latest.current_reporting_disciplines == ()
    assert latest.current_reporting_measurements == ()
    assert latest.known_station_ids == ("station-147",)
    assert latest.known_disciplines == ("fluid",)
    assert latest.known_measurements == ("groundwater-level",)
    assert latest.source_observation_ids == ("obs-1",)
    assert latest.source_report_ids == ("report-1",)
    assert latest.max_source_available_at <= latest.issue_time_utc

    schema_fields = set(ANOMALY_STATE_HISTORY_CONTRACT.schema.names)
    assert {
        "left_truncated",
        "late_entry_or_gap_unknown",
        "age_days",
        "known_duration_days",
        "right_censored",
        "reliability_grade",
        "reliability_weight",
        "source_observation_ids",
        "source_report_ids",
        "max_source_available_at",
        "station_id",
        "longitude",
        "latitude",
        "discipline",
        "measurement",
        "start_time",
        "current_reporting_station_ids",
        "current_reporting_disciplines",
        "current_reporting_measurements",
        "known_station_ids",
        "known_disciplines",
        "known_measurements",
        "row_count",
        "row_report_date_mismatch_count",
        "row_report_date_before_count",
        "row_report_date_after_count",
        "deformation_row_count",
        "fluid_row_count",
        "electromagnetic_row_count",
        "cross_fault_row_count",
    } <= schema_fields
    assert {
        "source_duration",
        "duration",
        "anomaly_description",
        "predicted_place",
        "predicted_time",
    }.isdisjoint(schema_fields)

    nullable_pairs = (
        ("station_id", "station_id_null_reason"),
        ("longitude", "longitude_null_reason"),
        ("latitude", "latitude_null_reason"),
        ("discipline", "discipline_null_reason"),
        ("measurement", "measurement_null_reason"),
        ("start_time", "start_time_null_reason"),
        ("reported_end_time", "reported_end_time_null_reason"),
        ("age_days", "age_days_null_reason"),
        ("known_duration_days", "known_duration_days_null_reason"),
    )
    for state in states:
        for value_field, reason_field in nullable_pairs:
            assert (getattr(state, value_field) is None) == (
                getattr(state, reason_field) is not None
            )
        assert state.spatial_eligible == (state.spatial_exclusion_reason is None)


def test_zero_entity_report_period_still_has_one_replayable_summary_row() -> None:
    period = _period(date(2024, 1, 1), 1, row_count=0)

    states = build_anomaly_state_history([], [period])

    assert len(states) == 1
    summary = states[0]
    assert summary.state_row_kind == "report_period_summary"
    assert summary.entity_scope == "report_period_summary"
    assert summary.row_count == 0
    assert summary.source_observation_ids == ()
    assert summary.source_report_ids == ("report-1",)
    assert summary.max_source_available_at == summary.issue_time_utc
    assert summary.spatial_eligible is False

    table = pa.Table.from_pylist(
        list(state_records(states)),
        schema=ANOMALY_STATE_HISTORY_CONTRACT.schema,
    )
    assert table.num_rows == 1
    assert table.column("state_row_kind").to_pylist() == ["report_period_summary"]


def test_left_truncation_late_entry_age_and_known_duration_are_causal() -> None:
    first = _period(date(2024, 1, 1), 1)
    second = _period(date(2024, 1, 8), 2)
    observations = [
        _observation(
            first,
            "left-observation",
            "left-entity",
            start_time=datetime(2023, 12, 20, tzinfo=LOCAL),
        ),
        _observation(
            second,
            "late-observation",
            "late-entity",
            start_time=datetime(2023, 12, 20, tzinfo=LOCAL),
            reported_end_time=datetime(2024, 1, 5, tzinfo=LOCAL),
            right_censored=False,
        ),
    ]

    states = build_anomaly_state_history(observations, [first, second])
    left_states = _states_for(states, "left-entity")
    late_state = _states_for(states, "late-entity")[0]

    assert all(state.left_truncated for state in left_states)
    assert all(not state.late_entry_or_gap_unknown for state in left_states)
    assert left_states[0].age_days == pytest.approx(13.0)
    assert left_states[0].age_days_null_reason is None
    assert late_state.left_truncated is False
    assert late_state.late_entry_or_gap_unknown is True
    assert late_state.previous_period_consecutive is True
    assert late_state.explicit_end_known is True
    assert late_state.reported_end_time == datetime(2024, 1, 5, tzinfo=LOCAL)
    assert late_state.known_duration_days == pytest.approx(16.0)
    assert late_state.known_duration_days_null_reason is None
    assert late_state.max_source_available_at <= late_state.issue_time_utc


def test_end_before_start_is_not_promoted_to_an_explicit_known_end() -> None:
    period = _period(date(2024, 1, 1), 1)
    observation = _observation(
        period,
        "invalid-end",
        "entity-a",
        start_time=datetime(2024, 1, 1, tzinfo=LOCAL),
        reported_end_time=datetime(2023, 12, 31, tzinfo=LOCAL),
        right_censored=False,
        reliability_flags=("end_before_start",),
    )

    state = _states_for(
        build_anomaly_state_history([observation], [period]),
        "entity-a",
    )[0]

    assert state.explicit_end_known is False
    assert state.right_censored is True
    assert state.reported_end_time is None
    assert state.reported_end_time_null_reason == "end_before_start_time"
    assert state.known_duration_days is None
    assert state.known_duration_days_null_reason == "end_before_start_time"


def test_reliability_precedence_is_excluded_then_cautious_then_high() -> None:
    period = _period(date(2024, 1, 1), 1, row_count=4)
    observations = [
        _observation(
            period,
            "high",
            "entity-high",
            reliability_flags=("source_duration_ignored",),
        ),
        _observation(
            period,
            "cautious",
            "entity-cautious",
            reliability_flags=("end_time_revised",),
        ),
        _observation(
            period,
            "excluded",
            "entity-excluded",
            reliability_flags=("end_time_revised", "missing_measurement"),
        ),
        _observation(
            period,
            "temporary",
            "entity-temporary",
            identity_complete=False,
        ),
    ]

    states = build_anomaly_state_history(observations, [period])
    high = _states_for(states, "entity-high")[0]
    cautious = _states_for(states, "entity-cautious")[0]
    excluded = _states_for(states, "entity-excluded")[0]
    temporary = _states_for(states, "entity-temporary")[0]

    assert (high.reliability_grade, high.reliability_weight) == ("high", 1.0)
    assert (cautious.reliability_grade, cautious.reliability_weight) == (
        "cautious",
        0.5,
    )
    assert (excluded.reliability_grade, excluded.reliability_weight) == (
        "excluded",
        0.0,
    )
    assert (temporary.reliability_grade, temporary.reliability_weight) == (
        "excluded",
        0.0,
    )
    assert temporary.age_days is None
    assert temporary.age_days_null_reason == "temporary_entity_longitudinal_excluded"


def test_conflicting_duplicate_structure_is_not_arbitrarily_spatialized() -> None:
    period = _period(date(2024, 1, 1), 1, row_count=2)
    observations = [
        _observation(period, "duplicate-a", "entity-a", longitude=110.0),
        _observation(
            period,
            "duplicate-b",
            "entity-a",
            longitude=111.0,
            discipline="fluid",
            measurement="measurement-b",
        ),
    ]

    forward = _states_for(build_anomaly_state_history(observations, [period]), "entity-a")[0]
    reverse = _states_for(
        build_anomaly_state_history(list(reversed(observations)), [period]),
        "entity-a",
    )[0]

    assert forward == reverse
    assert forward.longitude is None
    assert forward.longitude_null_reason == "conflicting_duplicate_values"
    assert forward.latitude == pytest.approx(35.0)
    assert forward.discipline is None
    assert forward.discipline_null_reason == "conflicting_duplicate_values"
    assert forward.measurement is None
    assert forward.measurement_null_reason == "conflicting_duplicate_values"
    assert forward.current_reporting_station_ids == ("station-a",)
    assert forward.current_reporting_disciplines == ("deformation", "fluid")
    assert forward.known_disciplines == ("deformation", "fluid")
    assert forward.current_reporting_measurements == (
        "measurement-a",
        "measurement-b",
    )
    assert forward.spatial_eligible is False
    assert forward.spatial_exclusion_reason == "longitude_unavailable"
    assert forward.reliability_grade == "cautious"
    assert forward.reliability_weight == pytest.approx(0.5)
    assert forward.source_observation_ids == ("duplicate-a", "duplicate-b")


@pytest.mark.parametrize(
    "forbidden_field",
    ["source_duration", "duration", "anomaly_description", "predicted_place"],
)
def test_forbidden_or_free_text_source_columns_are_rejected(forbidden_field: str) -> None:
    period = _period(date(2024, 1, 1), 1)
    observation = _observation(period, "obs-1", "entity-a")
    observation[forbidden_field] = "must never enter stage-3 state"

    with pytest.raises(ValueError, match="forbidden or unexpected"):
        build_anomaly_state_history([observation], [period])


def test_frozen_actual_period_count_can_be_enforced_without_breaking_synthetic_api() -> None:
    period = _period(date(2024, 1, 1), 1)

    with pytest.raises(ValueError, match="expected=205, observed=1"):
        build_anomaly_state_history([], [period], expected_report_period_count=205)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[0].__setitem__("identity_complete", "yes"), "boolean"),
        (lambda rows: rows[0].__setitem__("report_state", "未知"), "unsupported"),
        (
            lambda rows: rows[0].__setitem__("available_at", datetime(2024, 1, 1, tzinfo=UTC)),
            "cannot precede",
        ),
    ],
)
def test_state_builder_rejects_invalid_stage1_boundaries(mutation: Any, message: str) -> None:
    period = _period(date(2024, 1, 1), 1)
    rows = [_observation(period, "obs-1", "entity-a")]
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        build_anomaly_state_history(rows, [period])
