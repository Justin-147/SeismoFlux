"""Synthetic replay joins only: no stored real predictions or targets are read."""

from __future__ import annotations

import copy

import pytest
from shapely.geometry import GeometryCollection, Polygon

from seismoflux.multitask_s3.case_ledger import VARIANTS
from seismoflux.multitask_s3.replay_runner import (
    frame_from_saved_diagnostics,
    serialize_geometry,
)
from seismoflux.multitask_s3.runner import BANDS, COUNT_VARIANTS


def fixture_records(*, empty: bool = False):
    ids = [] if empty else ["synthetic-a", "synthetic-b"]
    alarms = {
        v: [{"area_budget_km2": 600000, "actual_area_km2": 599990, "selected": [0, 2]}]
        for v in VARIANTS
    }
    records = []
    for band in BANDS:
        hits = [] if empty else [True, False]
        records.append(
            {
                "fold_id": "A_DEV_2023_2024",
                "horizon_days": 30,
                "issue_time_utc": "2023-07-06T00:00:00+08:00",
                "primary_nonoverlap": True,
                "magnitude_band": band,
                "target_event_ids": ids,
                "anchor_mask": [] if empty else [True, False],
                "spatial": {
                    v: {
                        "alarms": [
                            {
                                "area_budget_km2": 600000,
                                "actual_area_km2": 599990,
                                "secondary_70km": {"status": "scored"},
                                "_local": {
                                    "strict_hits": hits.copy(),
                                    "secondary_70km_hits": [] if empty else [True, True],
                                },
                            }
                        ]
                    }
                    for v in VARIANTS
                },
                "count": {
                    v: {"observed_count": len(ids), "expected_count": 1.3} for v in COUNT_VARIANTS
                },
            }
        )
    return records, alarms


def test_replay_preserves_bands_costs_saved_hits_and_count_models_without_mutation():
    records, alarms = fixture_records()
    old_records, old_alarms = copy.deepcopy(records), copy.deepcopy(alarms)
    frame = frame_from_saved_diagnostics(records, model_alarms=alarms)
    assert frame["target_end_utc"] == "2023-08-05T00:00:00+08:00"
    assert frame["primary_nonoverlap"] is True
    assert frame["models"]["CAT_DYN"]["alarms"][0]["selected"] == [0, 2]
    for band in BANDS:
        assert frame["bands"][band]["outcomes"]["CAT_DYN"][0]["strict_hits"] == [True, False]
        assert frame["bands"][band]["outcomes"]["CAT_DYN"][0]["secondary_70km_hits"] == [True, True]
        assert set(frame["bands"][band]["counts"]) == set(COUNT_VARIANTS)
    assert records == old_records and alarms == old_alarms


def test_zero_earthquake_windows_remain_visible_with_their_expected_counts():
    records, alarms = fixture_records(empty=True)
    frame = frame_from_saved_diagnostics(records, model_alarms=alarms)
    assert frame["bands"]["Ms5_6"]["event_ids"] == []
    assert frame["bands"]["Ms5_6"]["counts"]["T0_CAL_DYN"]["expected_count"] == 1.3


@pytest.mark.parametrize("problem", ["band", "axis", "cost", "budget", "hits", "count", "mode"])
def test_mismatched_or_missing_existing_evidence_is_not_silently_displayed(problem):
    records, alarms = fixture_records()
    if problem == "band":
        records.pop()
    elif problem == "axis":
        records[1]["primary_nonoverlap"] = False
    elif problem == "cost":
        alarms["CAT_DYN"][0]["actual_area_km2"] = 600000
    elif problem == "budget":
        alarms["CAT_DYN"][0]["area_budget_km2"] = 750000
    elif problem == "hits":
        records[0]["spatial"]["CAT_DYN"]["alarms"][0]["_local"]["strict_hits"] = [True]
    elif problem == "count":
        del records[0]["count"]["T0_CAL_DYN"]
    else:
        records[0]["spatial"]["CAT_DYN"]["alarms"][0]["secondary_70km"]["status"] = "not_scored"
    with pytest.raises(ValueError):
        frame_from_saved_diagnostics(records, model_alarms=alarms)


def test_display_geometry_keeps_holes_and_exact_frozen_cell_order():
    first = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]])
    second = Polygon([(20, 0), (21, 0), (21, 1), (20, 1)])
    result = serialize_geometry([first, GeometryCollection([second])])
    assert result["bounds"] == [0, 0, 21, 10]
    assert len(result["cells"]) == 2
    assert len(result["cells"][0][0]) == 2
    assert result["cells"][1][0][0][0] == [20, 0]
    with pytest.raises(ValueError, match="empty"):
        serialize_geometry([])
