"""Synthetic S3 scoring checks, without file loading or outer outcome access."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp  # type: ignore[import-untyped]

from seismoflux.multitask_s1.c2b_score import log_alarm_prefixes
from seismoflux.multitask_s3.scoring import (
    pair_spatial,
    score_count,
    score_spatial,
    summarize_spatial,
)
from seismoflux.multitask_s3.targets import S3BandTargets
from seismoflux.stage2s.contracts import SpatialGrid


def _grid(areas: tuple[float, ...] = (100.0, 200.0, 300.0)) -> SpatialGrid:
    n = len(areas)
    return SpatialGrid(
        grid_id="synthetic_only",
        cell_size_km=25.0,
        cell_ids=tuple(f"c{index}" for index in range(n)),
        rows=np.zeros(n, dtype=np.int64),
        columns=np.arange(n, dtype=np.int64),
        query_xy_km=np.column_stack((np.arange(n) * 25.0, np.zeros(n))),
        clipped_area_km2=np.asarray(areas),
    )


def _targets(
    cells: tuple[int, ...] = (0, 1, 2), anchors: tuple[bool, ...] = (True, True, False)
) -> S3BandTargets:
    return S3BandTargets(
        tuple(f"synthetic_event_{index}" for index in range(len(cells))),
        np.asarray(cells, dtype=np.int64),
        np.asarray(anchors, dtype=bool),
    )


def _score(
    mass: tuple[float, ...] = (0.2, 0.6, 0.2),
    *,
    targets: S3BandTargets | None = None,
    near_cells: list[set[int]] | None = None,
    budgets: tuple[float, ...] = (250.0, 600.0),
) -> dict[str, Any]:
    return score_spatial(
        np.log(mass),
        targets=_targets() if targets is None else targets,
        grid=_grid(),
        budgets_km2=budgets,
        near_cells=near_cells,
    )


def test_actual_area_and_no_skip_prefix_exactly_reuse_existing_rule() -> None:
    grid = _grid()
    logs = np.log([0.2, 0.6, 0.2])
    old = log_alarm_prefixes(logs, grid, [250.0, 600.0])
    result = _score()
    assert old[0]["selected"] == [1]
    assert result["alarms"][0]["actual_area_km2"] == old[0]["actual_area_km2"] == 200.0
    assert result["alarms"][1]["actual_area_km2"] == 600.0
    alarm = result["alarms"][0]
    assert alarm["strict"]["all"] == {"hits": 1, "total": 3, "recall": 1 / 3}
    assert alarm["strict"]["anchor"] == {"hits": 1, "total": 2, "recall": 0.5}
    assert alarm["strict"]["subsequent"] == {"hits": 0, "total": 1, "recall": 0.0}
    assert alarm["_local"]["strict_hits"] == [False, True, False]
    expected = logs - np.log(grid.clipped_area_km2)
    assert result["log_density_per_km2"]["all"]["sum"] == pytest.approx(expected.sum())
    assert result["log_density_per_km2"]["anchor"]["mean"] == pytest.approx(expected[:2].mean())


def test_density_ranking_differs_from_mass_and_equal_density_uses_frozen_order() -> None:
    result = score_spatial(
        np.log([0.3, 0.7]),
        targets=_targets((0, 1), (True, False)),
        grid=_grid((100.0, 300.0)),
        budgets_km2=[100.0],
    )
    assert result["alarms"][0]["_local"]["strict_hits"] == [True, False]
    tied = score_spatial(
        np.log([0.5, 0.5]),
        targets=_targets((0, 1), (True, False)),
        grid=_grid((100.0, 100.0)),
        budgets_km2=[100.0],
    )
    assert tied["alarms"][0]["_local"]["strict_hits"] == [True, False]


def test_secondary_70km_does_not_expand_paid_area_and_missing_is_not_zero() -> None:
    missing = _score()["alarms"][0]["secondary_70km"]
    assert missing == {"status": "not_provided", "views": None}
    score = _score(near_cells=[{0, 1}, {1}, {1, 2}])
    alarm = score["alarms"][0]
    assert alarm["actual_area_km2"] == 200.0
    assert alarm["secondary_70km"]["views"]["all"] == {"hits": 3, "total": 3, "recall": 1.0}
    assert alarm["_local"]["secondary_70km_hits"] == [True, True, True]


@pytest.mark.parametrize("nearby", [[{0}], [{0}, {1}, {3}], [{1}, {1}, {2}]])
def test_invalid_or_unaligned_neighborhoods_raise(nearby: list[set[int]]) -> None:
    with pytest.raises(ValueError, match="70-km"):
        _score(near_cells=nearby)


def test_empty_exposure_has_none_recall_and_log_mean_not_zero_skill() -> None:
    result = _score(targets=_targets((), ()))
    assert result["event_count"] == result["anchor_count"] == 0
    assert result["log_density_per_km2"]["all"] == {"sum": 0.0, "total": 0, "mean": None}
    for view in ("all", "anchor", "subsequent"):
        assert result["alarms"][0]["strict"][view] == {"hits": 0, "total": 0, "recall": None}
    assert result["alarms"][0]["actual_area_km2"] == 200.0
    assert result["alarms"][0]["secondary_70km"]["status"] == "not_provided"
    provided = _score(targets=_targets((), ()), near_cells=[])
    assert provided["alarms"][0]["secondary_70km"]["views"]["all"]["recall"] is None


def test_extreme_log_mass_tail_is_scored_without_exp_then_log() -> None:
    logs = np.array([0.0, -1000.0, -1001.0])
    logs -= logsumexp(logs)
    result = score_spatial(logs, targets=_targets(), grid=_grid(), budgets_km2=[100.0])
    assert result["log_density_per_km2"]["all"]["sum"] == pytest.approx(
        float((logs - np.log([100, 200, 300])).sum())
    )
    assert math.isfinite(result["log_density_per_km2"]["all"]["sum"])


def test_spatial_return_is_strict_json_without_event_ids_or_coordinates() -> None:
    serialized = json.dumps(_score(), allow_nan=False)
    assert "synthetic_event" not in serialized
    assert "longitude" not in serialized and "latitude" not in serialized
    assert "_local" in serialized


@pytest.mark.parametrize("budgets", [(), (-1.0,), (float("nan"),), (100.0, 100.0)])
def test_invalid_alarm_budgets_raise(budgets: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="budgets"):
        _score(budgets=budgets)


def test_zero_budget_has_real_zero_paid_area_and_zero_hits() -> None:
    alarm = _score(budgets=(0.0,))["alarms"][0]
    assert alarm["actual_area_km2"] == 0.0
    assert alarm["strict"]["all"]["hits"] == 0


@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [-math.inf, 0.0, -1.0], [0.0]])
def test_unnormalized_or_invalid_mass_is_not_renormalized_during_score(values: list[float]) -> None:
    with pytest.raises(ValueError, match="log mass"):
        score_spatial(np.asarray(values), targets=_targets(), grid=_grid(), budgets_km2=[100.0])


def test_unmapped_targets_raise_instead_of_shrinking_score_denominator() -> None:
    with pytest.raises(ValueError, match="target cells"):
        _score(targets=_targets((-1,), (True,)))


@pytest.mark.parametrize("count", [0, 1, 4])
def test_count_score_matches_poisson_formula_and_probability(count: int) -> None:
    mean = 2.3
    row = score_count(math.log(mean), count)
    assert row["poisson_log_score"] == pytest.approx(
        count * math.log(mean) - mean - math.lgamma(count + 1)
    )
    p = 1 - math.exp(-mean)
    assert row["poisson_probability_at_least_one"] == pytest.approx(p)
    assert row["brier_at_least_one"] == pytest.approx((p - int(count > 0)) ** 2)
    assert row["count_bias_expected_minus_observed"] == pytest.approx(mean - count)
    json.dumps(row, allow_nan=False)


def test_very_negative_log_mean_retains_finite_positive_event_score() -> None:
    row = score_count(-1000.0, 2)
    assert row["expected_count"] == 0.0
    assert row["expected_count_underflow"] is True
    assert row["poisson_log_score"] == pytest.approx(-2000.0 - math.log(2))
    assert row["poisson_log_score_status"] == "finite"
    assert row["log_mean"] == -1000.0
    assert score_count(-1000.0, 0)["poisson_log_score"] == 0.0


def test_true_zero_mean_is_explicit_and_never_probability_clipped() -> None:
    zero = score_count(-math.inf, 0)
    impossible = score_count(-math.inf, 1)
    assert zero["poisson_log_score"] == 0.0
    assert zero["brier_at_least_one"] == 0.0
    assert impossible["poisson_log_score"] is None
    assert impossible["poisson_log_score_status"] == "negative_infinity"
    assert impossible["brier_at_least_one"] == 1.0
    assert impossible["expected_count_underflow"] is False
    json.dumps(impossible, allow_nan=False)


@pytest.mark.parametrize("count", [-1, 0.5, True])
def test_invalid_count_targets_raise(count: Any) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        score_count(0.0, count)


@pytest.mark.parametrize("logged", [math.nan, math.inf])
def test_invalid_count_predictions_raise(logged: float) -> None:
    with pytest.raises(ValueError, match="log_mean"):
        score_count(logged, 0)


def test_count_numeric_overflow_is_explicit_not_clipped() -> None:
    with pytest.raises(FloatingPointError, match="finite numeric range"):
        score_count(1000.0, 1)


def test_summary_pools_event_counts_and_keeps_empty_issue_costs() -> None:
    scores = {
        "one": _score(targets=_targets((1,), (True,))),
        "two": _score(),
        "empty": _score(targets=_targets((), ())),
    }
    summary = summarize_spatial(scores)
    assert summary["issue_count"] == 3
    assert summary["empty_issue_count"] == 1
    assert summary["event_occurrences"] == 4
    assert summary["alarms"][0]["strict"]["all"] == {"hits": 2, "total": 4, "recall": 0.5}
    assert summary["alarms"][0]["actual_area_mean_km2"] == 200.0
    assert summary["alarms"][0]["secondary_70km"]["views"] is None
    assert summary["log_density_per_km2"]["all"]["total"] == 4
    json.dumps(summary, allow_nan=False)


def test_pairing_reports_simultaneous_gains_and_losses_without_adoption_gate() -> None:
    reference = {"issue": _score((0.7, 0.2, 0.1))}
    candidate = {"issue": _score((0.1, 0.8, 0.1))}
    paired = pair_spatial(candidate, reference, bootstrap_nonoverlapping_issues=True)
    alarm = paired["alarms"][0]
    anchors = alarm["strict"]["views"]["anchor"]
    assert anchors["gained"] == anchors["lost"] == 1
    assert anchors["net_hits"] == 0
    assert anchors["total"] == 2
    assert alarm["candidate_actual_area_mean_km2"] == 200.0
    assert alarm["reference_actual_area_mean_km2"] == 100.0
    assert paired["alarms"][0]["secondary_70km"]["status"] == "not_provided_for_all_issues"
    assert not any(key in json.dumps(paired) for key in ("passed", "significant", "rejected"))
    json.dumps(paired, allow_nan=False)


def test_single_issue_improvement_is_retained_even_without_significance_requirement() -> None:
    target = _targets((1,), (True,))
    paired = pair_spatial(
        {"only": _score((0.1, 0.8, 0.1), targets=target)},
        {"only": _score((0.8, 0.1, 0.1), targets=target)},
    )
    anchors = paired["alarms"][0]["strict"]["views"]["anchor"]
    assert anchors["gained"] == anchors["net_hits"] == 1
    assert anchors["lost"] == 0
    assert anchors["delta_recall_pp"] == 100.0
    assert anchors["descriptive_ci95_recall_pp"] is None


def test_pairing_rejects_mismatched_issue_target_or_budget_without_intersection_filter() -> None:
    reference = {"issue": _score()}
    with pytest.raises(ValueError, match="same issues"):
        pair_spatial({}, reference)
    with pytest.raises(ValueError, match="target identity"):
        pair_spatial({"issue": _score(targets=_targets((2, 1, 0)))}, reference)
    with pytest.raises(ValueError, match="identical alarm budgets"):
        pair_spatial({"issue": _score(budgets=(100.0,))}, reference)


def test_empty_mappings_are_no_issues_not_zero_skill() -> None:
    assert summarize_spatial({})["status"] == "no_issues"
    assert pair_spatial({}, {})["status"] == "no_issues"
