"""Pure S3 scores and paired descriptive summaries; no outcome-reading entrypoint.

Call only after the runner has saved every outer prediction.  The spatial
ranking and whole-cell alarm cost are inherited unchanged.  ``_local`` values
are aligned event-level diagnostics for local pairing, not public artifacts;
no event identifiers or coordinates are returned.  No score or interval here
defines a scientific adoption threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence, Set
from typing import Any

import numpy as np
from numpy.typing import NDArray

from seismoflux.multitask_s1.c2b_score import exposure_bootstrap, log_alarm_prefixes
from seismoflux.multitask_s3.targets import S3BandTargets
from seismoflux.stage2s.contracts import SpatialGrid

_VIEWS = ("all", "anchor", "subsequent")


def _target_arrays(
    targets: S3BandTargets, cells: int
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    if not isinstance(targets, S3BandTargets):
        raise TypeError("targets must be S3BandTargets")
    size = targets.event_count
    if len(set(targets.event_ids)) != size or any(
        not isinstance(value, str) or not value for value in targets.event_ids
    ):
        raise ValueError("target event identifiers must be non-empty and unique")
    positions = np.asarray(targets.cell_indices)
    anchor = np.asarray(targets.anchor_mask)
    if (
        positions.shape != (size,)
        or not np.issubdtype(positions.dtype, np.integer)
        or np.any(positions < 0)
        or np.any(positions >= cells)
    ):
        raise ValueError("target cells must be valid integer positions aligned to events")
    if anchor.shape != (size,) or anchor.dtype != np.dtype(bool):
        raise ValueError("anchor_mask must be boolean and aligned to events")
    return np.asarray(positions, dtype=np.int64), anchor


def _masks(anchor: NDArray[np.bool_]) -> dict[str, NDArray[np.bool_]]:
    return {"all": np.ones(anchor.size, dtype=bool), "anchor": anchor, "subsequent": ~anchor}


def _recalls(hit: NDArray[np.bool_], anchor: NDArray[np.bool_]) -> dict[str, Any]:
    result = {}
    for name, mask in _masks(anchor).items():
        hits, total = int(np.count_nonzero(hit & mask)), int(np.count_nonzero(mask))
        result[name] = {"hits": hits, "total": total, "recall": hits / total if total else None}
    return result


def score_spatial(
    log_mass: NDArray[np.float64],
    *,
    targets: S3BandTargets,
    grid: SpatialGrid,
    budgets_km2: Sequence[float],
    near_cells: Sequence[Set[int]] | None = None,
) -> dict[str, Any]:
    """Score three event views at fixed paid area, with optional 70-km hits.

    ``near_cells`` must be the caller's 70-km polygon neighborhoods in the exact
    target-event order.  It cannot change the paid alarm area.  Missing secondary
    neighborhoods are explicitly uncomputed, including for an empty exposure.
    The conditional log score is log density per km2, never an event probability.
    """
    if not isinstance(grid, SpatialGrid):
        raise TypeError("grid must be the independently frozen SpatialGrid")
    budgets = [float(value) for value in budgets_km2]
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(not math.isfinite(value) or value < 0 for value in budgets)
    ):
        raise ValueError("alarm budgets must be unique finite nonnegative areas")
    cells, anchor = _target_arrays(targets, grid.cell_count)
    neighborhoods: list[set[int]] | None = None
    if near_cells is not None:
        if len(near_cells) != len(cells):
            raise ValueError("70-km neighborhoods must align with target events")
        neighborhoods = []
        for cell, nearby in zip(cells, near_cells, strict=True):
            if not isinstance(nearby, Set) or any(
                isinstance(value, bool | np.bool_)
                or not isinstance(value, int | np.integer)
                or not 0 <= value < grid.cell_count
                for value in nearby
            ):
                raise ValueError("70-km neighborhoods must contain valid frozen-grid positions")
            if int(cell) not in nearby:
                raise ValueError("70-km neighborhood must include the target's own cell")
            neighborhoods.append({int(value) for value in nearby})
    values = np.asarray(log_mass, dtype=np.float64)
    prefixes = log_alarm_prefixes(values, grid, budgets)
    event_logs = values[cells] - np.log(grid.clipped_area_km2[cells])
    log_density = {}
    for name, mask in _masks(anchor).items():
        total = int(np.count_nonzero(mask))
        summed = math.fsum(float(value) for value in event_logs[mask])
        log_density[name] = {
            "sum": summed,
            "total": total,
            "mean": summed / total if total else None,
        }
    alarms = []
    for prefix in prefixes:
        selected = set(prefix["selected"])
        strict = np.array([int(cell) in selected for cell in cells], dtype=bool)
        secondary = (
            np.array([bool(selected & nearby) for nearby in neighborhoods], dtype=bool)
            if neighborhoods is not None
            else None
        )
        alarms.append(
            {
                "area_budget_km2": prefix["area_budget_km2"],
                "actual_area_km2": prefix["actual_area_km2"],
                "strict": _recalls(strict, anchor),
                "secondary_70km": {
                    "status": "scored" if secondary is not None else "not_provided",
                    "views": _recalls(secondary, anchor) if secondary is not None else None,
                },
                "_local": {
                    "strict_hits": strict.tolist(),
                    "secondary_70km_hits": secondary.tolist() if secondary is not None else None,
                },
            }
        )
    identity = json.dumps(
        list(zip(targets.event_ids, cells.tolist(), anchor.tolist(), strict=True)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "status": "scored",
        "grid_id": grid.grid_id,
        "event_count": targets.event_count,
        "anchor_count": targets.anchor_count,
        "log_density_per_km2": log_density,
        "alarms": alarms,
        "_local": {
            "target_order_sha256": hashlib.sha256(identity).hexdigest(),
            "anchor_mask": anchor.tolist(),
        },
    }


def score_count(log_mean: float, count: int) -> dict[str, Any]:
    """Score a complete count exposure directly in log-mean space, including N=0.

    Finite very negative log means retain finite positive-event log scores even
    when their displayed expectation underflows to zero.  An exact zero mean
    (``log_mean=-inf``) with a positive count has log score ``-inf``: represented
    by ``None`` and an explicit status for strict JSON, never a clipped floor.
    """
    if isinstance(count, bool | np.bool_) or not isinstance(count, int | np.integer) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    observed, logged = int(count), float(log_mean)
    if math.isnan(logged) or logged == math.inf:
        raise ValueError("log_mean must be finite or negative infinity for an exact zero rate")
    if logged == -math.inf:
        mean = 0.0
        score = 0.0 if observed == 0 else None
        log_status = "finite" if observed == 0 else "negative_infinity"
    else:
        try:
            mean = math.exp(logged)
            # Do not replace logged with log(mean): exp(-1000) is already zero.
            score = observed * logged - mean - math.lgamma(observed + 1.0)
        except OverflowError as exc:
            raise FloatingPointError("Poisson score exceeds finite numeric range") from exc
        if not math.isfinite(score):
            raise FloatingPointError("Poisson score exceeds finite numeric range")
        log_status = "finite"
    probability = -math.expm1(-mean)
    return {
        "status": "scored" if score is not None else "impossible_positive_count_at_exact_zero_rate",
        "observed_count": observed,
        "log_mean": logged if math.isfinite(logged) else None,
        "log_mean_status": "finite" if math.isfinite(logged) else "negative_infinity_exact_zero",
        "expected_count": mean,
        "expected_count_underflow": math.isfinite(logged) and mean == 0.0,
        "poisson_log_score": score,
        "poisson_log_score_status": log_status,
        "poisson_probability_at_least_one": probability,
        "brier_at_least_one": (probability - float(observed > 0)) ** 2,
        "count_bias_expected_minus_observed": mean - observed,
    }


def _alarm_map(score: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    result = {float(row["area_budget_km2"]): row for row in score["alarms"]}
    if len(result) != len(score["alarms"]):
        raise ValueError("duplicate alarm budget in a spatial score")
    return result


def summarize_spatial(scores_by_issue: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pool numerators/denominators, not issue recalls; counts are occurrences.

    The caller groups one fixed horizon, magnitude band and reporting axis.
    Overlapping replay windows remain descriptive, not independent earthquakes.
    Empty exposures remain in the issue and paid-area summaries.
    """
    scores = list(scores_by_issue.values())
    if not scores:
        return {"status": "no_issues", "issue_count": 0, "alarms": [], "log_density_per_km2": None}
    alarms = [_alarm_map(score) for score in scores]
    budgets = list(alarms[0])
    if any(
        score["grid_id"] != scores[0]["grid_id"] or set(alarm) != set(budgets)
        for score, alarm in zip(scores, alarms, strict=True)
    ):
        raise ValueError("spatial summaries require the same frozen grid and budget set")
    log_density = {}
    for view in _VIEWS:
        total = sum(int(score["log_density_per_km2"][view]["total"]) for score in scores)
        summed = math.fsum(float(score["log_density_per_km2"][view]["sum"]) for score in scores)
        log_density[view] = {
            "sum": summed,
            "total": total,
            "mean": summed / total if total else None,
        }
    combined = []
    for budget in budgets:
        parts = [alarm[budget] for alarm in alarms]
        area = [float(part["actual_area_km2"]) for part in parts]
        row: dict[str, Any] = {
            "area_budget_km2": budget,
            "actual_area_mean_km2": math.fsum(area) / len(area),
            "actual_area_min_km2": min(area),
            "actual_area_max_km2": max(area),
        }
        for mode in ("strict", "secondary_70km"):
            present = mode == "strict" or all(
                part["secondary_70km"]["status"] == "scored" for part in parts
            )
            views: dict[str, Any] = {}
            if present:
                for view in _VIEWS:
                    values = [
                        part["strict"][view]
                        if mode == "strict"
                        else part["secondary_70km"]["views"][view]
                        for part in parts
                    ]
                    hits = sum(int(value["hits"]) for value in values)
                    total = sum(int(value["total"]) for value in values)
                    views[view] = {
                        "hits": hits,
                        "total": total,
                        "recall": hits / total if total else None,
                    }
            row[mode] = (
                views
                if mode == "strict"
                else {
                    "status": "scored" if present else "not_provided_for_all_issues",
                    "views": views if present else None,
                }
            )
        combined.append(row)
    return {
        "status": "summarized",
        "issue_count": len(scores),
        "empty_issue_count": sum(int(score["event_count"] == 0) for score in scores),
        "event_occurrences": sum(int(score["event_count"]) for score in scores),
        "anchor_occurrences": sum(int(score["anchor_count"]) for score in scores),
        "log_density_per_km2": log_density,
        "alarms": combined,
    }


def pair_spatial(
    candidate_by_issue: Mapping[str, Mapping[str, Any]],
    reference_by_issue: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_nonoverlapping_issues: bool = False,
) -> dict[str, Any]:
    """Compare exact issue/target pairs and report gains AND losses at each budget.

    Only enable the optional fixed 2000-draw/seed-147 descriptive bootstrap for
    the caller's non-overlapping primary issue axis.  Its interval is not an
    adoption gate or a guarantee that different earthquakes are independent.
    Actual paid areas can differ under the same inherited whole-cell budget;
    both are reported, never silently called exactly equal spent area.
    """
    if set(candidate_by_issue) != set(reference_by_issue):
        raise ValueError("paired spatial scores must contain exactly the same issues")
    if not candidate_by_issue:
        return {"status": "no_issues", "issue_count": 0, "alarms": []}
    pairs = [
        (candidate, reference_by_issue[issue]) for issue, candidate in candidate_by_issue.items()
    ]
    for candidate, reference in pairs:
        if (
            candidate["grid_id"] != reference["grid_id"]
            or candidate["grid_id"] != pairs[0][0]["grid_id"]
            or candidate["_local"] != reference["_local"]
            or candidate["event_count"] != reference["event_count"]
        ):
            raise ValueError("paired spatial target identity/order or frozen grid differs")
    c_maps = [_alarm_map(candidate) for candidate, _ in pairs]
    r_maps = [_alarm_map(reference) for _, reference in pairs]
    budgets = list(c_maps[0])
    if any(set(mapping) != set(budgets) for mapping in [*c_maps, *r_maps]):
        raise ValueError("paired spatial scores must use identical alarm budgets")
    result = []
    for budget in budgets:
        candidate_area = [float(mapping[budget]["actual_area_km2"]) for mapping in c_maps]
        reference_area = [float(mapping[budget]["actual_area_km2"]) for mapping in r_maps]
        row: dict[str, Any] = {
            "area_budget_km2": budget,
            "candidate_actual_area_mean_km2": math.fsum(candidate_area) / len(pairs),
            "reference_actual_area_mean_km2": math.fsum(reference_area) / len(pairs),
        }
        for mode, hit_key in (("strict", "strict_hits"), ("secondary_70km", "secondary_70km_hits")):
            if any(mapping[budget]["_local"][hit_key] is None for mapping in [*c_maps, *r_maps]):
                row[mode] = {"status": "not_provided_for_all_issues", "views": None}
                continue
            views = {}
            for view in _VIEWS:
                gained = lost = candidate_hits = reference_hits = 0
                totals, deltas = [], []
                for (candidate, _), c_map, r_map in zip(pairs, c_maps, r_maps, strict=True):
                    anchor = np.asarray(candidate["_local"]["anchor_mask"], dtype=bool)
                    mask = _masks(anchor)[view]
                    c_hit = np.asarray(c_map[budget]["_local"][hit_key], dtype=bool)
                    r_hit = np.asarray(r_map[budget]["_local"][hit_key], dtype=bool)
                    if c_hit.shape != anchor.shape or r_hit.shape != anchor.shape:
                        raise ValueError("paired event hit arrays are not target aligned")
                    c_count, r_count = (
                        int(np.count_nonzero(c_hit & mask)),
                        int(np.count_nonzero(r_hit & mask)),
                    )
                    candidate_hits += c_count
                    reference_hits += r_count
                    gained += int(np.count_nonzero(c_hit & ~r_hit & mask))
                    lost += int(np.count_nonzero(~c_hit & r_hit & mask))
                    totals.append(int(np.count_nonzero(mask)))
                    deltas.append(c_count - r_count)
                total = sum(totals)
                views[view] = {
                    "candidate_hits": candidate_hits,
                    "reference_hits": reference_hits,
                    "total": total,
                    "gained": gained,
                    "lost": lost,
                    "net_hits": gained - lost,
                    "delta_recall_pp": 100.0 * (gained - lost) / total if total else None,
                    "descriptive_ci95_recall_pp": exposure_bootstrap(
                        np.asarray(deltas, dtype=float), np.asarray(totals, dtype=float)
                    )
                    if bootstrap_nonoverlapping_issues
                    else None,
                }
            row[mode] = {"status": "paired", "views": views}
        result.append(row)
    return {
        "status": "paired",
        "issue_count": len(pairs),
        "bootstrap": "paired_primary_issue_2000_seed147_descriptive"
        if bootstrap_nonoverlapping_issues
        else "not_requested",
        "alarm_cost_comparison": "same_budget_whole_cell_prefix_actual_paid_areas_reported",
        "alarms": result,
    }
