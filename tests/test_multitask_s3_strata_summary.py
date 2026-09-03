"""Synthetic paired strata accounting; no stored predictions or targets are read."""

import math
from typing import Any

import pytest

from seismoflux.multitask_s3.strata_summary import summarize_paired_group


def _row(
    event: str,
    episode: str,
    candidate: bool,
    reference: bool,
    *,
    weight: float = 1.0,
    fold: str = "fold_a",
    issue: str = "2024-01-04T00:00:00Z",
) -> dict[str, Any]:
    return {
        "event_id": event,
        "episode_id": episode,
        "fold_id": fold,
        "issue_time_utc": issue,
        "weight": weight,
        "candidate_hit": candidate,
        "reference_hit": reference,
    }


def test_all_four_paired_outcomes_preserve_weighted_numerators_and_denominator() -> None:
    result = summarize_paired_group(
        [
            _row("e1", "p1", True, True, weight=0.5),
            _row("e2", "p1", True, False, weight=0.5),
            _row("e3", "p2", False, True, weight=1.0),
            _row("e4", "p3", False, False, weight=2.0),
        ]
    )
    assert result["status"] == "available"
    assert result["total_weight"] == 4.0
    assert result["candidate_weighted_hits"] == 1.0
    assert result["reference_weighted_hits"] == 1.5
    assert result["candidate_weighted_recall"] == 0.25
    assert result["reference_weighted_recall"] == 0.375
    assert result["gained_weight"] == result["shared_hit_weight"] == 0.5
    assert result["lost_weight"] == 1.0
    assert result["shared_miss_weight"] == 2.0
    assert result["delta_weighted_hits"] == -0.5
    assert result["delta_recall_pp"] == -12.5
    assert result["event_exposure_count"] == result["unique_event_count"] == 4
    assert result["unique_episode_count"] == 3


def test_frozen_full_episode_weights_are_not_renormalized_within_a_subgroup() -> None:
    result = summarize_paired_group(
        [
            _row("e1", "episode_of_four", True, False, weight=0.25),
            _row("e2", "episode_of_four", False, True, weight=0.25),
            _row("e3", "single_event_episode", True, False, weight=1.0),
        ]
    )
    assert result["total_weight"] == 1.5
    assert result["candidate_weighted_hits"] == 1.25
    assert result["reference_weighted_hits"] == 0.25
    assert result["candidate_weighted_recall"] == pytest.approx(5 / 6)
    assert result["delta_recall_pp"] == pytest.approx(100 * 2 / 3)
    assert result["_local"]["episode_clusters"]["episode_of_four"]["total_weight"] == 0.5


def test_repeated_windows_do_not_inflate_unique_event_or_episode_counts() -> None:
    result = summarize_paired_group(
        [
            _row("e1", "p1", True, False),
            _row("e1", "p1", False, True, issue="2024-01-11T00:00:00Z"),
        ]
    )
    assert result["event_exposure_count"] == 2
    assert result["unique_event_count"] == result["unique_episode_count"] == 1
    assert result["total_weight"] == 2.0
    assert len(result["_local"]["issue_clusters"]) == 2


def test_empty_group_is_na_not_a_zero_recall() -> None:
    result = summarize_paired_group([])
    assert result["status"] == "empty_group_NA"
    assert result["candidate_weighted_recall"] is None
    assert result["reference_weighted_recall"] is None
    assert result["delta_recall_pp"] is None
    assert result["event_exposure_count"] == result["unique_episode_count"] == 0
    assert result["total_weight"] == result["candidate_weighted_hits"] == 0.0
    assert result["_local"] == {"episode_clusters": {}, "issue_clusters": {}}


@pytest.mark.parametrize("weight", [0.0, -1.0, math.nan, math.inf, -math.inf, True, "1"])
def test_invalid_weight_is_rejected(weight: Any) -> None:
    row = _row("e1", "p1", True, False)
    row["weight"] = weight
    with pytest.raises(ValueError, match="weight"):
        summarize_paired_group([row])


@pytest.mark.parametrize("field", ["candidate_hit", "reference_hit"])
@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_hit_values_are_not_silently_coerced_to_booleans(field: str, value: Any) -> None:
    row = _row("e1", "p1", True, False)
    row[field] = value
    with pytest.raises(ValueError, match="booleans"):
        summarize_paired_group([row])


def test_local_episode_and_fold_issue_clusters_sum_the_same_paired_quantities() -> None:
    result = summarize_paired_group(
        [
            _row("e1", "p1", True, False, weight=0.25),
            _row("e2", "p1", False, True, weight=0.25),
            _row("e3", "p2", True, False, weight=0.5),
            _row("e1", "p1", True, False, weight=0.25, fold="fold_b"),
        ]
    )
    local = result["_local"]
    assert local["episode_clusters"]["p1"] == {
        "candidate_minus_reference_weighted_hit_sum": 0.25,
        "total_weight": 0.75,
        "candidate_weighted_hits": 0.5,
        "reference_weighted_hits": 0.25,
    }
    assert local["issue_clusters"]["fold_a|2024-01-04T00:00:00Z"] == {
        "candidate_minus_reference_weighted_hit_sum": 0.5,
        "total_weight": 1.0,
        "candidate_weighted_hits": 0.75,
        "reference_weighted_hits": 0.25,
    }
    assert local["issue_clusters"]["fold_b|2024-01-04T00:00:00Z"] == {
        "candidate_minus_reference_weighted_hit_sum": 0.25,
        "total_weight": 0.25,
        "candidate_weighted_hits": 0.25,
        "reference_weighted_hits": 0.0,
    }
    for clusters in local.values():
        assert math.fsum(row["total_weight"] for row in clusters.values()) == 1.25
        assert math.fsum(
            row["candidate_minus_reference_weighted_hit_sum"] for row in clusters.values()
        ) == result["delta_weighted_hits"]
        assert math.fsum(row["candidate_weighted_hits"] for row in clusters.values()) == 1.0
        assert math.fsum(row["reference_weighted_hits"] for row in clusters.values()) == 0.25


def test_paired_difference_uses_fsum_instead_of_subtracting_large_hit_totals() -> None:
    result = summarize_paired_group(
        [
            _row("e1", "p1", True, False, weight=1e16),
            _row("e2", "p2", True, False, weight=1.0),
            _row("e3", "p3", False, True, weight=1e16),
        ]
    )
    assert result["delta_weighted_hits"] == 1.0


def test_cluster_separator_collision_is_rejected() -> None:
    with pytest.raises(ValueError, match="separator"):
        summarize_paired_group([_row("e1", "p1", True, False, fold="fold_a|extra")])
