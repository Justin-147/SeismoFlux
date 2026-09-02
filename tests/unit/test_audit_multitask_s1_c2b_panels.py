from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[2] / "scripts/audit_multitask_s1_c2b_panels.py"
    spec = importlib.util.spec_from_file_location("c2b_panel_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(event_specs, source_specs, *, unit="us"):
    events = pd.DataFrame(event_specs)
    sources = pd.DataFrame(source_specs)
    for frame in (events, sources):
        for column in ("origin_time_utc", "available_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601").dt.as_unit(
                unit
            )
    return events, sources


def _event(identifier, time, members, *, mag=5.0, inside=True, available=None):
    return {
        "event_id": identifier,
        "origin_time_utc": time,
        "available_at": available or time,
        "magnitude": mag,
        "inside_study_area": inside,
        "catalog_sources": members,
        "anchor_source_record_id": members[0],
    }


def _source(identifier, source, time, *, available=None):
    return {
        "source_record_id": identifier,
        "source_id": source,
        "origin_time_utc": time,
        "available_at": available or time,
    }


def test_any_source_member_not_anchor_and_no_double_count(audit):
    time = "1981-01-01T00:00:00Z"
    events, sources = _frames(
        [_event("canonical", time, ["m5", "m3a", "m3b"])],
        [
            _source("m5", "earthquake_catalog_m5_plus", time),
            _source("m3a", "earthquake_catalog_m3_plus", time),
            _source("m3b", "earthquake_catalog_m3_plus", time),
        ],
    )
    rows = audit.build_panel_rows(events, sources)[:3]
    assert [row["training_event_count"] for row in rows] == [1, 1, 1]


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_exact_local_cutoff_origin_availability_and_microseconds(audit, unit):
    cutoff = "1999-12-30T16:00:00Z"
    before = "1999-12-30T15:59:59.999999Z"
    after = "1999-12-30T16:00:00.000001Z"
    specs = [
        _event("exact", cutoff, ["s"]),
        _event("before", before, ["s"]),
        _event("after", after, ["s"]),
        _event("delayed", before, ["s"], available=after),
        _event("outside", before, ["s"], inside=False),
    ]
    events, sources = _frames(
        specs, [_source("s", "earthquake_catalog_m3_plus", before)], unit=unit
    )
    rows = [
        row for row in audit.build_panel_rows(events, sources) if row["cutoff_year_local"] == 2000
    ]
    assert [row["training_event_count"] for row in rows] == [2, 2, 0]
    assert rows[0]["training_cutoff_inclusive_utc"] == cutoff
    assert rows[0]["observed_origin_max_utc"] == cutoff


def test_start_and_magnitude_boundaries_are_canonical_and_inclusive(audit):
    exact = "1979-12-31T16:00:00Z"
    before = "1979-12-31T15:59:59.999999Z"
    events, sources = _frames(
        [
            _event("exact", exact, ["s"], mag=4.0),
            _event("early", before, ["s"], mag=4.0),
            _event("low", exact, ["s"], mag=3.9999),
        ],
        [_source("s", "earthquake_catalog_m3_plus", before)],
    )
    rows = audit.build_panel_rows(events, sources)[:3]
    assert [row["training_event_count"] for row in rows] == [2, 1, 0]


def test_source_membership_itself_must_be_available(audit):
    cutoff = "1999-12-30T16:00:00Z"
    earlier = "1999-12-01T00:00:00Z"
    events, sources = _frames(
        [_event("canonical", earlier, ["m5", "m3"])],
        [
            _source("m5", "earthquake_catalog_m5_plus", earlier),
            _source(
                "m3", "earthquake_catalog_m3_plus", earlier, available="1999-12-30T16:00:00.000001Z"
            ),
        ],
    )
    rows = [
        row for row in audit.build_panel_rows(events, sources) if row["cutoff_year_local"] == 2000
    ]
    assert rows[0]["training_cutoff_inclusive_utc"] == cutoff
    assert [row["training_event_count"] for row in rows] == [1, 0, 1]


def test_ledger_has_only_declared_training_nodes_and_no_event_identifiers(audit):
    time = "1960-01-01T00:00:00Z"
    events, sources = _frames(
        [_event("secret-event-id", time, ["s"])],
        [_source("s", "earthquake_catalog_m5_plus", time)],
    )
    rows = audit.build_panel_rows(events, sources)
    assert len(rows) == 24
    assert {row["cutoff_year_local"] for row in rows} == set(audit.YEARS)
    assert "secret-event-id" not in str(rows)
    assert all(row["effective_magnitude_type"] == "Ms" for row in rows)
