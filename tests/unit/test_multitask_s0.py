from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from seismoflux.multitask_s0 import (
    FoldSpec,
    build_episodes,
    build_s0_ledger,
    filter_catalog,
    stable_ledger_sha256,
    summarize_episode_samples,
    summarize_fold_maturity,
    summarize_sample_funnel,
)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).assign(
        origin_time_utc=lambda value: pd.to_datetime(value["origin_time_utc"], utc=True),
        available_at=lambda value: pd.to_datetime(value["available_at"], utc=True),
    )


def _row(
    event_id: str,
    origin: str,
    magnitude: float,
    *,
    available: str | None = None,
    longitude: float = 100.0,
    latitude: float = 30.0,
    inside: bool = True,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "origin_time_utc": origin,
        "available_at": available or origin,
        "longitude": longitude,
        "latitude": latitude,
        "magnitude": magnitude,
        "inside_study_area": inside,
    }


def test_magnitude_and_time_boundaries_are_half_open_and_causal() -> None:
    catalog = _frame(
        [
            _row("before", "1999-12-31T23:59:59Z", 5.5),
            _row("m4", "2000-01-01T00:00:00Z", 4.0),
            _row("m5", "2000-01-02T00:00:00Z", 5.0),
            _row("m6", "2000-01-03T00:00:00Z", 6.0),
            _row("end", "2000-01-04T00:00:00Z", 5.5),
            _row(
                "late",
                "2000-01-02T12:00:00Z",
                5.5,
                available="2000-01-05T00:00:00Z",
            ),
            _row("outside", "2000-01-02T00:00:00Z", 5.5, inside=False),
        ]
    )

    m5_6 = filter_catalog(
        catalog,
        origin_start="2000-01-01T00:00:00Z",
        origin_end="2000-01-04T00:00:00Z",
        available_by="2000-01-04T00:00:00Z",
        magnitude_minimum=5.0,
        magnitude_maximum_exclusive=6.0,
    )
    assert m5_6["event_id"].tolist() == ["m5"]

    funnel = summarize_sample_funnel(
        catalog,
        catalog_start="2000-01-01T00:00:00Z",
        catalog_cutoff="2000-01-04T00:00:00Z",
    )
    assert funnel["magnitude_bin_counts"] == {"m4_plus": 4, "m5_6": 2, "m6_plus": 1}
    assert funnel["excluded_late_availability_rows"] == 1


def test_episode_uses_fixed_anchor_without_transitive_chain_extension() -> None:
    catalog = _frame(
        [
            _row("a", "2020-01-01T00:00:00Z", 5.0, longitude=100.0),
            _row("b", "2020-01-20T00:00:00Z", 5.2, longitude=100.5),
            # Linked to b but >30 days from a: b must not pull c into a's episode.
            _row("c", "2020-02-10T00:00:00Z", 6.0, longitude=101.0),
            _row("far", "2020-01-20T00:00:00Z", 5.5, longitude=110.0),
        ]
    )

    episodes = build_episodes(catalog)
    members = [set(item["member_event_ids"]) for item in episodes]
    assert {"a", "b"} in members
    assert {"c"} in members
    assert {"far"} in members
    times = catalog.set_index("event_id")["origin_time_utc"]
    for episode in episodes:
        anchor = times.loc[episode["anchor_event_id"]]
        assert all(
            times.loc[event_id] - anchor <= pd.Timedelta(days=30)
            for event_id in episode["member_event_ids"]
        )
    reordered = catalog.iloc[::-1].reset_index(drop=True)
    assert build_episodes(reordered) == episodes
    summary = summarize_episode_samples(episodes)
    assert summary["maximum_anchor_span_days"] <= 30.0
    assert summary["episode_member_count_histogram"] == {"1": 2, "2": 1}


def test_episode_multi_anchor_tie_break_uses_anchor_event_id() -> None:
    catalog = _frame(
        [
            _row("z_anchor", "2020-01-01T00:00:00Z", 5.0, longitude=-0.5, latitude=0.0),
            _row("a_anchor", "2020-01-01T00:00:00Z", 5.0, longitude=0.5, latitude=0.0),
            _row("candidate", "2020-01-02T00:00:00Z", 5.1, longitude=0.0, latitude=0.0),
        ]
    )
    episodes = build_episodes(catalog)
    members_by_anchor = {
        item["anchor_event_id"]: set(item["member_event_ids"]) for item in episodes
    }
    assert members_by_anchor["a_anchor"] == {"a_anchor", "candidate"}
    assert members_by_anchor["z_anchor"] == {"z_anchor"}


def test_cross_fold_episode_is_owned_by_anchor_fold_and_never_split() -> None:
    catalog = _frame(
        [
            _row("anchor", "2020-02-12T00:00:00Z", 5.3),
            _row("later", "2020-02-20T00:00:00Z", 5.4, longitude=100.1),
            _row("independent", "2020-03-05T00:00:00Z", 5.5, longitude=120.0),
        ]
    )
    folds = [
        {
            "fold_id": "f1",
            "role": "development",
            "train_start_utc": "2019-01-01T00:00:00Z",
            "train_end_utc": "2019-12-20T00:00:00Z",
            "assessment_start_utc": "2020-01-01T00:00:00Z",
            "assessment_end_utc": "2020-02-15T00:00:00Z",
            "embargo_days": 12,
            "issue_frequency_days": 7,
            "horizon_days": [7],
        },
        {
            "fold_id": "f2",
            "role": "least_exposed_retrospective_holdout",
            "train_start_utc": "2019-01-01T00:00:00Z",
            "train_end_utc": "2020-02-01T00:00:00Z",
            "assessment_start_utc": "2020-02-15T00:00:00Z",
            "assessment_end_utc": "2020-04-01T00:00:00Z",
            "embargo_days": 12,
            "issue_frequency_days": 7,
            "horizon_days": [7],
        },
    ]
    records = summarize_fold_maturity(
        catalog,
        folds=folds,
        truth_cutoff="2020-04-01T00:00:00Z",
        horizons_days=[7],
    )
    first = records[0]["horizons"]["7"]["operational_weekly"]["magnitude_bins"]["m5_6"]
    second = records[1]["horizons"]["7"]["operational_weekly"]["magnitude_bins"]["m5_6"]
    assert first["touched_episode_count"] == 1
    assert first["anchor_target_count"] == 1
    assert first["subsequent_target_event_count"] == 0
    assert first["episode_balanced_total_weight"] == pytest.approx(0.5)
    # The Feb-20 event is visible in f2's event targets, but its physical episode
    # remains wholly owned by the Feb-12 anchor in f1.
    assert second["unique_event_count"] >= 1
    assert second["touched_episode_count"] == 1  # only the independent Mar-05 episode


def test_maturity_boundary_and_embargo_are_inclusive_without_dropping_targets() -> None:
    catalog = _frame(
        [
            _row("at_issue", "2020-04-01T16:00:00Z", 5.2),
            _row("at_horizon", "2020-04-08T16:00:00Z", 5.3),
            _row("after_horizon", "2020-04-08T16:00:01Z", 5.4),
        ]
    )
    fold = {
        "fold_id": "boundary",
        "role": "transparent_stability_audit",
        "train_start_utc": "2019-01-01T00:00:00Z",
        "train_end_utc": "2020-04-01T00:00:00Z",
        "parameter_selection_end_utc": "2020-03-20T00:00:00Z",
        "assessment_start_utc": "2020-04-01T00:00:00Z",
        "assessment_end_utc": "2020-04-09T00:00:00Z",
        "embargo_days": 12,
        "issue_frequency_days": 7,
        "horizon_days": [7, 30],
    }
    record = summarize_fold_maturity(
        catalog,
        folds=[fold],
        truth_cutoff="2020-04-08T16:00:00Z",
        horizons_days=[7, 30],
    )[0]
    horizon = record["horizons"]["7"]
    assert horizon["mature_issue_count"] == 1
    assert horizon["immature_issue_count"] == 1
    assert (
        horizon["operational_weekly"]["magnitude_bins"]["m5_6"]["unique_event_count"]
        == 1
    )
    issues = record["catalog_issue_calendar"]["issue_times_utc"]
    for issue in issues:
        local = pd.Timestamp(issue).tz_convert("Asia/Shanghai")
        assert local.weekday() == 3
        assert (local.hour, local.minute, local.second) == (0, 0, 0)
    unavailable = record["horizons"]["30"]["primary_exposure"]
    assert unavailable["evaluable"] is False
    assert unavailable["magnitude_bins"]["m5_6"]["unique_event_count"] is None

    invalid = dict(fold, parameter_selection_end_utc="2020-03-20T00:00:01Z")
    with pytest.raises(ValueError, match="embargo"):
        FoldSpec.from_mapping(invalid)


def test_ledger_hash_is_order_independent_and_build_is_repeatable(tmp_path: Path) -> None:
    left = OrderedDict([("z", 1), ("a", {"b": 2, "a": 1})])
    right = OrderedDict([("a", {"a": 1, "b": 2}), ("z", 1)])
    assert stable_ledger_sha256(left) == stable_ledger_sha256(right)

    path = tmp_path / "earthquake_event.parquet"
    synthetic = _frame(
        [
            _row("train", "2019-01-01T00:00:00Z", 4.0),
            _row("target", "2020-01-03T00:00:00Z", 5.0),
            _row("large", "2020-01-04T00:00:00Z", 6.0),
        ]
    )
    synthetic["depth_km"] = [10.0, None, 20.0]
    synthetic["magnitude_type"] = [None, "Mw", None]
    synthetic.to_parquet(path, index=False)
    fold = {
        "fold_id": "f1",
        "role": "development",
        "train_start_utc": "2018-01-01T00:00:00Z",
        "train_end_utc": "2019-12-20T00:00:00Z",
        "assessment_start_utc": "2020-01-01T00:00:00Z",
        "assessment_end_utc": "2020-01-10T00:00:00Z",
        "embargo_days": 12,
        "issue_frequency_days": 7,
        "horizon_days": [7],
    }
    first = build_s0_ledger(
        path,
        catalog_start="2018-01-01T00:00:00Z",
        catalog_cutoff="2020-02-01T00:00:00Z",
        folds=[fold],
        horizons_days=[7],
        require_authoritative_identity=False,
    )
    second = build_s0_ledger(
        path,
        catalog_start="2018-01-01T00:00:00Z",
        catalog_cutoff="2020-02-01T00:00:00Z",
        folds=[fold],
        horizons_days=[7],
        require_authoritative_identity=False,
    )
    assert first == second
    assert first["content_sha256"] == stable_ledger_sha256(first)
    assert first["episode_summary_by_magnitude_bin"]["m5_6"]["event_count"] == 1
    assert first["episode_summary_by_magnitude_bin"]["m6_plus"]["event_count"] == 1
    assert first["episode_summary_by_magnitude_bin"]["m4_plus"]["episode_count"] is None
    assert first["episode_summary_by_magnitude_bin"]["m5_6"]["maximum_anchor_span_days"] == 0
    assert first["episode_summary_by_magnitude_bin"]["m5_6"][
        "episode_member_count_histogram"
    ] == {"1": 1}
    quality = first["quality_by_catalog_panel"]
    assert quality["full_catalog_all_eras_and_domains"]["depth_km"]["non_null_count"] == 2
    assert quality["m4_plus"]["magnitude_type"]["non_null_count"] == 1
    assert quality["m5_6"]["depth_km"]["non_null_fraction"] == 0.0
    assert quality["m6_plus"]["depth_km"]["non_null_fraction"] == 1.0

    with pytest.raises(ValueError, match="SHA-256"):
        build_s0_ledger(
            path,
            catalog_start="2018-01-01T00:00:00Z",
            catalog_cutoff="2020-02-01T00:00:00Z",
            folds=[fold],
            horizons_days=[7],
        )


def test_fold_spec_accepts_frozen_yaml_field_names_and_derived_end() -> None:
    frozen = {
        "id": "C_DEV_2000_2004",
        "role": "development",
        "fit_history_end_exclusive": "2000-01-01T00:00:00+08:00",
        "target_block_start": "2000-01-01T00:00:00+08:00",
        "target_block_end_exclusive": "2005-01-01T00:00:00+08:00",
    }
    spec = FoldSpec.from_mapping(frozen, default_horizon_days=[7, 30])
    assert spec.fold_id == "C_DEV_2000_2004"
    assert spec.horizon_days == (7, 30)
    assert spec.evaluation_start == pd.Timestamp("1999-12-31T16:00:00Z")

    derived = dict(frozen, target_block_end_exclusive="derived_from_catalog_truth_cutoff")
    spec_derived = FoldSpec.from_mapping(
        derived,
        derived_assessment_end="2026-07-09T04:25:56Z",
    )
    assert spec_derived.evaluation_end == pd.Timestamp("2026-07-09T04:25:56Z")
