from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seismoflux.multitask_s1.c2b_inputs import (
    NO_SOURCE,
    c2b_catalog_from_frames,
    load_c2b_catalog,
    panel_indices,
)

M3 = "earthquake_catalog_m3_plus"
M5 = "earthquake_catalog_m5_plus"
PANEL = {
    "start_local": "1980-01-01T00:00:00+08:00",
    "magnitude_minimum": 4.0,
    "required_source": M3,
}


def _event(identifier, time, members, *, magnitude=5.0, available=None, inside=True):
    return {
        "event_id": identifier,
        "origin_time_utc": time,
        "available_at": available or time,
        "longitude": 100.125,
        "latitude": 30.25,
        "magnitude": magnitude,
        "inside_study_area": inside,
        "catalog_sources": members,
    }


def _source(identifier, source, time, *, available=None):
    return {
        "source_record_id": identifier,
        "source_id": source,
        "origin_time_utc": time,
        "available_at": available or time,
    }


def _frames(event_rows, source_rows, unit="us"):
    events, sources = pd.DataFrame(event_rows), pd.DataFrame(source_rows)
    for frame in (events, sources):
        for name in ("origin_time_utc", "available_at"):
            frame[name] = pd.to_datetime(frame[name], utc=True, format="ISO8601").dt.as_unit(unit)
    return events, sources


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_alignment_any_member_minimum_visible_time_and_units(unit):
    early, late = "1999-01-01T00:00:00Z", "1999-02-01T00:00:00Z"
    events, sources = _frames(
        [_event("z", late, ["m5", "slow", "fast"]), _event("a", early, ["missing"])],
        [
            _source("m5", M5, early),
            _source("slow", M3, early, available=late),
            _source("fast", M3, early),
        ],
        unit,
    )
    catalog = c2b_catalog_from_frames(events, sources)
    assert catalog.table.event_ids == ("a", "z")
    assert catalog.source_visible_us[M3].tolist() == [NO_SOURCE, 915148800000000]
    assert catalog.source_visible_us[M5].tolist() == [NO_SOURCE, 915148800000000]
    assert catalog.table.longitude.tolist() == [100.125, 100.125]
    assert not catalog.source_visible_us[M3].flags.writeable


def test_canonical_and_source_delays_and_inclusive_bounds():
    start = "1979-12-31T16:00:00Z"
    cutoff = "1999-12-30T16:00:00Z"
    after = "1999-12-30T16:00:00.000001Z"
    events, sources = _frames(
        [
            _event("start", start, ["s"], magnitude=4.0),
            _event("exact", cutoff, ["s"]),
            _event("delayed_event", cutoff, ["s"], available=after),
            _event("delayed_member", cutoff, ["slow"]),
            _event("low", cutoff, ["s"], magnitude=3.9999),
            _event("outside", cutoff, ["s"], inside=False),
            _event("early", "1979-12-31T15:59:59.999999Z", ["s"]),
        ],
        [_source("s", M3, start), _source("slow", M3, start, available=after)],
    )
    catalog = c2b_catalog_from_frames(events, sources)
    selected = panel_indices(catalog, PANEL, datetime(1999, 12, 31, 16, tzinfo=UTC))
    assert [catalog.table.event_ids[i] for i in selected] == ["start", "exact"]
    all_source = dict(PANEL, required_source=None)
    selected = panel_indices(catalog, all_source, datetime(1999, 12, 31, 16, tzinfo=UTC))
    assert [catalog.table.event_ids[i] for i in selected] == ["start", "delayed_member", "exact"]


def test_recent_lower_bound_is_issue_relative_and_strict():
    events, sources = _frames(
        [
            _event("too_old", "1999-12-31T12:00:00Z", ["s"]),
            _event("exact_lower", "2000-01-01T00:00:00Z", ["s"]),
            _event("after_lower", "2000-01-01T00:00:00.000001Z", ["s"]),
            _event("at_cutoff", "2000-01-30T00:00:00Z", ["s"]),
        ],
        [_source("s", M3, "1990-01-01T00:00:00Z")],
    )
    catalog = c2b_catalog_from_frames(events, sources)
    selected = panel_indices(catalog, PANEL, datetime(2000, 1, 31, tzinfo=UTC), recent_days=30)
    assert [catalog.table.event_ids[i] for i in selected] == ["after_lower", "at_cutoff"]
    with pytest.raises(ValueError, match="positive integer"):
        panel_indices(catalog, PANEL, datetime(2000, 1, 31, tzinfo=UTC), recent_days=0)


def test_loader_authenticates_and_filters_synthetic_pre2020_rows(tmp_path: Path):
    events, sources = _frames(
        [
            _event("past", "2019-12-30T16:00:00Z", ["past"]),
            _event("future", "2020-01-01T00:00:00Z", ["future"]),
        ],
        [
            _source("past", M3, "2019-12-30T16:00:00Z"),
            _source("future", M3, "2020-01-01T00:00:00Z"),
        ],
    )
    event_path, source_path = tmp_path / "events.parquet", tmp_path / "sources.parquet"
    events.to_parquet(event_path, index=False)
    sources.to_parquet(source_path, index=False)
    protocol = {
        "inputs": {
            "canonical_catalog": event_path.name,
            "canonical_catalog_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
            "canonical_catalog_rows": 2,
            "source_records": source_path.name,
            "source_records_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_record_rows": 2,
        }
    }
    catalog = load_c2b_catalog(data_root=tmp_path, protocol=protocol)
    assert catalog.table.event_ids == ("past",)
    protocol["inputs"]["canonical_catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash or row count"):
        load_c2b_catalog(data_root=tmp_path, protocol=protocol)


def test_submicrosecond_source_is_not_silently_rounded():
    events, sources = _frames(
        [_event("e", "2000-01-01T00:00:00Z", ["s"])],
        [_source("s", M3, "2000-01-01T00:00:00.000000001Z")],
        unit="ns",
    )
    with pytest.raises(ValueError, match="microseconds"):
        c2b_catalog_from_frames(events, sources)
    sources["origin_time_utc"] = sources["origin_time_utc"].dt.floor("us")
    sources["available_at"] = sources["available_at"].dt.floor("us")
    catalog = c2b_catalog_from_frames(events, sources)
    assert catalog.source_visible_us[M3].dtype == np.int64
