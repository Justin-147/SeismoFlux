"""Small synthetic saved-diagnostic ledgers, without model/score computation."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from seismoflux.multitask_s3.case_ledger import VARIANTS, build_case_ledger


def _record(issue: str = "2023-07-01T00:00:00+00:00", primary: bool = True) -> dict[str, Any]:
    ids, cells, anchor = ["a", "b", "c", "d"], [0, 1, 2, 3], [True, True, False, False]
    digest = hashlib.sha256(
        json.dumps(
            list(zip(ids, cells, anchor, strict=True)), ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    spatial = {}
    for variant in VARIANTS:
        hits = [True, False, True, False] if variant == "CAT_DYN" else [False, True, True, False]
        spatial[variant] = {
            "grid_id": "synthetic_grid",
            "event_count": 4,
            "anchor_count": 2,
            "_local": {"target_order_sha256": digest, "anchor_mask": anchor.copy()},
            "alarms": [
                {
                    "area_budget_km2": budget,
                    "actual_area_km2": budget - 10,
                    "secondary_70km": {"status": "scored"},
                    "_local": {"strict_hits": hits.copy(), "secondary_70km_hits": [True] * 4},
                }
                for budget in (100.0, 200.0)
            ],
        }
    return {
        "fold_id": "A_DEV_2023_2024",
        "horizon_days": 30,
        "magnitude_band": "Ms5_6",
        "issue_time_utc": issue,
        "primary_nonoverlap": primary,
        "target_event_ids": ids,
        "target_cell_indices": cells,
        "anchor_mask": anchor,
        "spatial": spatial,
    }


def _metadata() -> dict[str, Any]:
    return {
        identifier: {
            "origin_time_utc": "2023-07-30T00:00:00.123456789+00:00",
            "available_at": "2023-07-31T00:00:00+00:00",
            "magnitude": 5.2,
            "longitude": 105.0,
            "latitude": 30.0,
        }
        for identifier in ("a", "b", "c", "d")
    }


def test_preserves_every_contrast_budget_mode_and_positive_and_negative_case() -> None:
    result = build_case_ledger([_record()], catalog_metadata=_metadata())
    assert result["local_only"] is True
    assert result["status"] == "ledger_complete_no_case_selection"
    assert len(result["rows"]) == 4 * 6 * 2 * 2
    rows = [
        row
        for row in result["rows"]
        if row["comparison"] == "CAT_DYN_minus_CATALOG"
        and row["mode"] == "strict"
        and row["area_budget_km2"] == 100.0
    ]
    assert [row["classification"] for row in rows] == ["gained", "lost", "both_hit", "both_miss"]
    assert [row["event_view"] for row in rows] == ["anchor", "anchor", "subsequent", "subsequent"]
    assert all(row["candidate_actual_area_km2"] == 90 for row in rows)
    assert all(row["primary_nonoverlap"] for row in rows)
    assert result["events"]["a"]["origin_time_utc"].endswith("123456789+00:00")
    json.dumps(result, allow_nan=False)


def test_repeated_windows_remain_exposures_not_new_events_and_unused_metadata_is_ignored() -> None:
    records = [_record(), _record("2023-07-08T00:00:00+00:00", primary=False)]
    metadata = _metadata()
    metadata["unused_future_event"] = {"bad": object()}
    before = copy.deepcopy(records)
    result = build_case_ledger(records, catalog_metadata=metadata)
    assert records == before
    assert result["summary"]["unique_event_count"] == 4
    assert result["summary"]["event_exposure_count"] == 8
    assert result["summary"]["primary_event_exposure_count"] == 4
    assert result["summary"]["exposure_count_fold_horizon_band_issue"] == 2
    assert "unused_future_event" not in result["events"]


@pytest.mark.parametrize(
    "corruption", ["fingerprint", "order", "anchor", "budget", "hits", "70km", "variant", "grid"]
)
def test_unaligned_or_missing_saved_diagnostics_fail_instead_of_dropping_cases(
    corruption: str,
) -> None:
    record = _record()
    score = record["spatial"]["CAT_DYN"]
    if corruption == "fingerprint":
        score["_local"]["target_order_sha256"] = "wrong"
    elif corruption == "order":
        record["target_event_ids"].reverse()
    elif corruption == "anchor":
        score["_local"]["anchor_mask"][0] = False
    elif corruption == "budget":
        score["alarms"].pop()
    elif corruption == "hits":
        score["alarms"][0]["_local"]["strict_hits"] = [True]
    elif corruption == "70km":
        score["alarms"][0]["_local"]["secondary_70km_hits"] = None
    elif corruption == "variant":
        del record["spatial"]["CAT_SNAP"]
    else:
        score["grid_id"] = "another_grid"
    with pytest.raises(ValueError):
        build_case_ledger([record], catalog_metadata=_metadata())


def test_missing_metadata_and_duplicate_exposure_are_explicit_errors() -> None:
    metadata = _metadata()
    del metadata["b"]
    with pytest.raises(ValueError, match="missing metadata"):
        build_case_ledger([_record()], catalog_metadata=metadata)
    with pytest.raises(ValueError, match="duplicate"):
        build_case_ledger([_record(), _record()], catalog_metadata=_metadata())


def test_empty_input_has_empty_inventory_not_fake_prediction_result() -> None:
    result = build_case_ledger([], catalog_metadata={})
    assert result["rows"] == [] and result["events"] == {}
    assert result["summary"]["unique_event_count"] == 0
    assert result["summary"]["event_exposure_count"] == 0
