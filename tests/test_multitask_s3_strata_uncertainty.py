"""Synthetic bootstrap adaptation only; no stored real targets or scores are read."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from seismoflux.multitask_s1.development_summary import _EpisodeUnit
from seismoflux.multitask_s3 import strata_uncertainty
from seismoflux.multitask_s3.strata_summary import summarize_paired_group
from seismoflux.multitask_s3.strata_uncertainty import paired_uncertainty

ISSUE_A = "fold_a|2024-01-04T00:00:00Z"
ISSUE_B = "fold_b|2025-01-02T00:00:00Z"
ISSUE_EMPTY = "fold_b|2025-02-06T00:00:00Z"


def _row(
    event: str,
    episode: str,
    issue_key: str,
    candidate: bool,
    reference: bool,
    weight: float = 1.0,
) -> dict[str, Any]:
    fold, issue = issue_key.split("|")
    return {
        "event_id": event,
        "episode_id": episode,
        "fold_id": fold,
        "issue_time_utc": issue,
        "weight": weight,
        "candidate_hit": candidate,
        "reference_hit": reference,
    }


def test_existing_components_receive_full_episode_and_zero_filled_time_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = summarize_paired_group(
        [
            _row("e1", "p1", ISSUE_A, True, False, 0.25),
            _row("e2", "p1", ISSUE_B, False, True, 0.25),
            _row("e3", "p2", ISSUE_B, True, True),
        ]
    )
    observed_units: list[_EpisodeUnit] = []

    def episode_mock(units: Sequence[_EpisodeUnit], *, label: str) -> dict[str, Any]:
        observed_units.extend(units)
        assert label == "frozen-label"
        return {
            "status": "evaluable",
            "root_seed": 147,
            "replicates_requested": 2000,
            "replicates_evaluable": 2000,
            "confidence_interval_95": [-0.2, 0.3],
        }

    def time_mock(delta_hits: np.ndarray, totals: np.ndarray) -> list[float]:
        np.testing.assert_array_equal(delta_hits, [0.25, -0.25, 0.0])
        np.testing.assert_array_equal(totals, [0.25, 1.25, 0.0])
        return [-20.0, 30.0]

    monkeypatch.setattr(strata_uncertainty, "_bootstrap_episode_ratio", episode_mock)
    monkeypatch.setattr(strata_uncertainty, "exposure_bootstrap", time_mock)
    result = paired_uncertainty(
        summary,
        issue_keys=[ISSUE_A, ISSUE_B, ISSUE_EMPTY],
        label="frozen-label",
        global_member_counts={"p1": 4, "p2": 1},
    )
    assert len(observed_units) == 2
    assert observed_units[0].episode_id == observed_units[0].unit_id == "p1"
    assert observed_units[0].global_member_count == 4
    assert observed_units[0].candidate_hits == observed_units[0].baseline_hits == 0.25
    assert observed_units[0].total_weight == 0.5
    assert result["episode"]["confidence_interval_95_pp"] == [-20.0, 30.0]
    assert result["time_block"]["confidence_interval_95_pp"] == [-20.0, 30.0]
    assert result["time_block"]["issue_count"] == 3
    assert result["time_block"]["empty_issue_count"] == 1
    assert result["time_block"]["replicates_evaluable"] is None


def test_real_2000_draw_components_are_repeatable_and_intervals_have_pp_units() -> None:
    summary = summarize_paired_group(
        [
            _row("e1", "p1", ISSUE_A, True, False, 0.5),
            _row("e2", "p2", ISSUE_A, False, True),
            _row("e3", "p3", ISSUE_B, True, False),
        ]
    )
    arguments: dict[str, Any] = {
        "issue_keys": [ISSUE_A, ISSUE_B, ISSUE_EMPTY],
        "label": "synthetic-fixed-task",
        "global_member_counts": {"p1": 2, "p2": 1, "p3": 1},
    }
    first = paired_uncertainty(summary, **arguments)
    second = paired_uncertainty(summary, **arguments)
    assert first == second
    assert first["point_estimate_delta_recall_pp"] == 20.0
    assert first["episode"]["replicates_evaluable"] == 2000
    for name in ("episode", "time_block"):
        assert first[name]["root_seed"] == 147
        assert first[name]["replicates_requested"] == 2000
        assert first[name]["effect_unit"] == "percentage_points"
        assert first[name]["status"] == "evaluable"
        lower, upper = first[name]["confidence_interval_95_pp"]
        assert -100 <= lower <= upper <= 100
    assert first["adoption_threshold"] is None


def test_single_primary_issue_interval_is_na_without_discarding_point_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = summarize_paired_group([_row("e1", "p1", ISSUE_A, True, False)])

    def forbidden_call(*args: Any) -> None:
        raise AssertionError("one-issue time bootstrap must not run")

    monkeypatch.setattr(strata_uncertainty, "exposure_bootstrap", forbidden_call)
    result = paired_uncertainty(
        summary, issue_keys=[ISSUE_A], label="one", global_member_counts={"p1": 1}
    )
    assert result["point_estimate_delta_recall_pp"] == 100.0
    assert result["episode"]["confidence_interval_95_pp"] is None
    assert result["time_block"]["confidence_interval_95_pp"] is None
    assert result["time_block"]["reason"] == "fewer_than_two_primary_issues"


def test_zero_total_keeps_all_empty_primary_issues_and_returns_na() -> None:
    result = paired_uncertainty(
        summarize_paired_group([]),
        issue_keys=[ISSUE_A, ISSUE_EMPTY],
        label="empty",
        global_member_counts={},
    )
    assert result["point_estimate_delta_recall_pp"] is None
    assert result["episode"]["confidence_interval_95_pp"] is None
    assert result["time_block"]["confidence_interval_95_pp"] is None
    assert result["time_block"]["issue_count"] == result["time_block"]["empty_issue_count"] == 2
    assert result["time_block"]["reason"] == "zero_total_weight"


@pytest.mark.parametrize("issues", [[ISSUE_A, ISSUE_A], [ISSUE_B], [""], ["fold_a|"]])
def test_duplicate_missing_or_invalid_issue_identifiers_raise(issues: list[str]) -> None:
    summary = summarize_paired_group([_row("e1", "p1", ISSUE_A, True, False)])
    with pytest.raises(ValueError, match="issue"):
        paired_uncertainty(summary, issue_keys=issues, label="bad", global_member_counts={"p1": 1})


@pytest.mark.parametrize("counts", [{}, {"p1": 0}, {"p1": True}, {"p1": 1.5}])
def test_missing_or_invalid_frozen_global_membership_raises(counts: dict[str, Any]) -> None:
    summary = summarize_paired_group([_row("e1", "p1", ISSUE_A, True, False)])
    with pytest.raises(ValueError, match="member"):
        paired_uncertainty(
            summary, issue_keys=[ISSUE_A], label="bad", global_member_counts=counts
        )


def test_cluster_denominator_mismatch_cannot_silently_discard_exposures() -> None:
    summary = summarize_paired_group([_row("e1", "p1", ISSUE_A, True, False)])
    summary["total_weight"] = 2.0
    with pytest.raises(ValueError, match="denominators"):
        paired_uncertainty(
            summary, issue_keys=[ISSUE_A], label="bad", global_member_counts={"p1": 1}
        )
