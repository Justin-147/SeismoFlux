"""Sample-count bookkeeping tests; no prediction effects or real data."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from seismoflux.multitask_s3.input_waterlevel import summarize_windows


def test_unique_events_do_not_multiply_with_overlapping_operational_windows():
    t = datetime(2023, 8, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "event_id": ["at_issue", "five", "six", "delayed"],
            "origin_time_utc": [
                t,
                t + timedelta(days=2),
                t + timedelta(days=6),
                t + timedelta(days=3),
            ],
            "available_at": [
                t,
                t + timedelta(days=2),
                t + timedelta(days=6),
                t + timedelta(days=20),
            ],
            "magnitude": [5.0, 5.2, 6.1, 5.4],
        }
    )
    result = summarize_windows(
        frame,
        (t, t + timedelta(days=1)),
        7,
        t + timedelta(days=10),
        {"Ms5_6": {"five"}, "Ms6_plus": {"six"}},
    )
    assert result["unique_events"] == {"Ms4_plus": 2, "Ms5_6": 1, "Ms6_plus": 1}
    assert result["event_occurrences"] == {"Ms4_plus": 4, "Ms5_6": 2, "Ms6_plus": 2}
    assert result["fixed_first_anchor_events"] == {"Ms5_6": 1, "Ms6_plus": 1}


def test_no_eligible_issues_is_not_a_score_or_fabricated_window():
    frame = pd.DataFrame(
        {"event_id": [], "origin_time_utc": [], "available_at": [], "magnitude": []}
    )
    result = summarize_windows(frame, (), 365, datetime(2024, 1, 1, tzinfo=UTC), {"Ms5_6": set()})
    assert result["issue_count"] == 0
    assert result["first_issue_utc"] is result["last_issue_utc"] is None
    assert "recall" not in result
