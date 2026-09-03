"""Descriptive null-effect summaries only; no effect selection or adoption gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from typing import Any, Literal

import numpy as np

Direction = Literal["higher_better", "lower_better"]
Metric = Literal[
    "delta_recall_pp",
    "spatial_log_density_delta_mean",
    "delta_poisson_log_score_mean",
    "delta_brier_at_least_one_mean",
]


def _number(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("effects must be finite numbers or explicit None for NA")
    return float(value)


def summarize_null_effect(
    observed: float | None,
    replicate_values: Mapping[int, float | None],
    failures: Set[int],
    *,
    total: int = 200,
    direction: Direction = "higher_better",
) -> dict[str, Any]:
    """Describe the same fixed effect across 200 registered replicas (IDs 0..199).

    Caller must supply one identical fold/horizon/band/axis/contrast/budget/view
    in every value. Missing replicas, failed runs and evaluated NA are distinct.
    Failures never become replacement draws or silently vanish from the planned
    denominator. Conditional valid-only fractions are separately named, not
    probabilities, exact randomization p-values, or scientific adoption rules.
    """
    if isinstance(total, bool) or total != 200:
        raise ValueError("the frozen trial contains exactly 200 registered replicas")
    if direction not in ("higher_better", "lower_better"):
        raise ValueError("direction must be higher_better or lower_better")
    ids = set(replicate_values) | set(failures)
    if any(
        isinstance(key, bool) or not isinstance(key, int) or not 0 <= key < total for key in ids
    ):
        raise ValueError("replica identities must be integer IDs 0 through 199")
    values = {key: _number(value) for key, value in replicate_values.items()}
    if any(values.get(key) is not None for key in failures):
        raise ValueError("a failed replica cannot simultaneously supply an effect value")
    actual = _number(observed)
    valid = [value for key, value in values.items() if key not in failures and value is not None]
    na_count = sum(value is None and key not in failures for key, value in values.items())
    missing_count = total - len(ids)
    distribution = None
    if valid:
        quantiles = np.quantile(np.asarray(valid), [0.05, 0.25, 0.5, 0.75, 0.95])
        distribution = {
            "min": min(valid),
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
            "max": max(valid),
            "quantile_method": "linear_on_valid_values_only",
        }
    comparisons = None
    if actual is not None and valid:
        above = sum(actual > value for value in valid)
        equal = sum(actual == value for value in valid)
        below = sum(actual < value for value in valid)
        better = above if direction == "higher_better" else below
        comparisons = {
            "observed_above_null_count": above,
            "observed_equal_null_count": equal,
            "observed_below_null_count": below,
            "observed_better_than_null_count": better,
            "observed_worse_than_null_count": below if direction == "higher_better" else above,
            "registered_denominator": total,
            "valid_denominator": len(valid),
            "better_fraction_of_registered": better / total,
            "better_fraction_among_valid_only": better / len(valid),
            "ties": "exact_saved_numeric_equality_no_tie_jitter",
        }
    return {
        "status": "observed_NA"
        if actual is None
        else "no_valid_replicates_yet"
        if not valid
        else "complete_descriptive_summary"
        if len(valid) == total
        else "partial_descriptive_summary",
        "observed_effect": actual,
        "direction": direction,
        "registered_replicates": total,
        "valid_replicates": len(valid),
        "failed_replicates": len(failures),
        "NA_replicates": na_count,
        "not_provided_replicates": missing_count,
        "failed_replica_ids": sorted(failures),
        "NA_replica_ids": sorted(
            key for key, value in values.items() if value is None and key not in failures
        ),
        "distribution": distribution,
        "comparisons": comparisons,
        "interpretation": "offline_attribution_description_not_exact_p_value_or_adoption_threshold",
        "adoption_threshold": None,
    }


def extract_axis_effect(
    axis_summary: Mapping[str, Any],
    *,
    contrast: str,
    metric: Metric,
    area_budget_km2: float | None = None,
    mode: Literal["strict", "secondary_70km"] = "strict",
    view: Literal["anchor", "all", "subsequent"] = "anchor",
) -> dict[str, Any]:
    """Extract one specified effect; never select the best budget/task/contrast.

    Spatial log-density is area-independent, so an alarm budget must not be
    supplied for it or count scores. The returned NA does not manufacture zero.
    Matching horizon, magnitude band and axis across inputs remains the caller's
    responsibility because ``summarize_axis`` carries no horizon/band identity.
    """
    if mode not in ("strict", "secondary_70km") or view not in ("anchor", "all", "subsequent"):
        raise ValueError("unregistered spatial mode or view")
    if metric not in (
        "delta_recall_pp",
        "spatial_log_density_delta_mean",
        "delta_poisson_log_score_mean",
        "delta_brier_at_least_one_mean",
    ):
        raise ValueError("metric is not one of the existing S3 effects")
    if metric != "delta_recall_pp" and area_budget_km2 is not None:
        raise ValueError("this metric does not depend on an alarm-area budget")
    if metric == "delta_recall_pp" and (
        area_budget_km2 is None or _number(area_budget_km2) is None
    ):
        raise ValueError("recall extraction needs the exact registered alarm budget")
    value = None
    try:
        if metric == "delta_recall_pp":
            matches = [
                row
                for row in axis_summary["spatial_contrasts"][contrast]["alarms"]
                if row["area_budget_km2"] == area_budget_km2
            ]
            if len(matches) > 1:
                raise ValueError("duplicate alarm budget in axis summary")
            if matches:
                value = matches[0][mode]["views"][view]["delta_recall_pp"]
        elif metric == "spatial_log_density_delta_mean":
            candidate, reference = contrast.split("_minus_")
            c_value = axis_summary["spatial"][candidate]["log_density_per_km2"][view]["mean"]
            r_value = axis_summary["spatial"][reference]["log_density_per_km2"][view]["mean"]
            if c_value is not None and r_value is not None:
                value = float(c_value) - float(r_value)
        else:
            value = axis_summary["count_contrasts"][contrast][metric]
    except (KeyError, TypeError):
        value = None
    value = _number(value)
    return {
        "status": "available" if value is not None else "metric_NA_or_not_in_schema",
        "value": value,
        "contrast": contrast,
        "metric": metric,
        "axis": axis_summary.get("axis"),
        "area_budget_km2": area_budget_km2,
        "mode": mode if metric == "delta_recall_pp" else None,
        "view": view if metric in ("delta_recall_pp", "spatial_log_density_delta_mean") else None,
        "direction": "lower_better"
        if metric == "delta_brier_at_least_one_mean"
        else "higher_better",
    }
