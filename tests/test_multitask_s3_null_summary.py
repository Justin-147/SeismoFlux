"""Synthetic descriptive comparisons; no stored real effects are read."""

import math

import pytest

from seismoflux.multitask_s3.null_summary import extract_axis_effect, summarize_null_effect


def test_failures_na_and_missing_stay_in_registered_200_denominator() -> None:
    result = summarize_null_effect(2.0, {0: 1.0, 1: 2.0, 2: 3.0, 3: None}, {4})
    assert result["registered_replicates"] == 200
    assert result["valid_replicates"] == 3
    assert result["failed_replicates"] == result["NA_replicates"] == 1
    assert result["not_provided_replicates"] == 195
    assert result["comparisons"]["observed_above_null_count"] == 1
    assert result["comparisons"]["observed_equal_null_count"] == 1
    assert result["comparisons"]["observed_below_null_count"] == 1
    assert result["comparisons"]["better_fraction_of_registered"] == 1 / 200
    assert result["comparisons"]["better_fraction_among_valid_only"] == 1 / 3
    assert result["distribution"]["median"] == 2
    assert result["adoption_threshold"] is None


def test_lower_better_direction_is_not_confused_with_numerically_higher() -> None:
    result = summarize_null_effect(
        -1.0, {index: 0.0 for index in range(200)}, set(), direction="lower_better"
    )
    assert result["status"] == "complete_descriptive_summary"
    assert result["comparisons"]["observed_above_null_count"] == 0
    assert result["comparisons"]["observed_better_than_null_count"] == 200
    assert result["distribution"]["min"] == result["distribution"]["max"] == 0


def test_observed_na_empty_replica_set_and_failed_values_are_explicit() -> None:
    result = summarize_null_effect(None, {0: None}, {1})
    assert result["observed_effect"] is None and result["comparisons"] is None
    assert result["distribution"] is None
    assert summarize_null_effect(1.0, {}, set())["status"] == "no_valid_replicates_yet"
    with pytest.raises(ValueError):
        summarize_null_effect(1.0, {0: 1.0}, {0})
    for values in ({200: 1.0}, {-1: 1.0}, {0: math.nan}):
        with pytest.raises(ValueError):
            summarize_null_effect(1.0, values, set())
    with pytest.raises(ValueError, match="exactly 200"):
        summarize_null_effect(1.0, {}, set(), total=80)


def test_schema_extracts_only_requested_effect_and_missing_metric_is_na() -> None:
    axis = {
        "axis": "primary_nonoverlap",
        "spatial_contrasts": {
            "CAT_DYN_minus_CAT_COV": {
                "alarms": [
                    {
                        "area_budget_km2": 600000.0,
                        "strict": {"views": {"anchor": {"delta_recall_pp": 1.5}}},
                    },
                    {
                        "area_budget_km2": 750000.0,
                        "strict": {"views": {"anchor": {"delta_recall_pp": 9.0}}},
                    },
                ]
            }
        },
        "spatial": {
            "CAT_DYN": {"log_density_per_km2": {"anchor": {"mean": -2.0}}},
            "CAT_COV": {"log_density_per_km2": {"anchor": {"mean": -3.0}}},
        },
        "count_contrasts": {
            "T0_CAL_DYN_minus_T0_CAL_COV": {"delta_brier_at_least_one_mean": -0.01}
        },
    }
    recall = extract_axis_effect(
        axis, contrast="CAT_DYN_minus_CAT_COV", metric="delta_recall_pp", area_budget_km2=600000
    )
    assert recall["value"] == 1.5
    logscore = extract_axis_effect(
        axis, contrast="CAT_DYN_minus_CAT_COV", metric="spatial_log_density_delta_mean"
    )
    assert logscore["value"] == 1.0
    brier = extract_axis_effect(
        axis, contrast="T0_CAL_DYN_minus_T0_CAL_COV", metric="delta_brier_at_least_one_mean"
    )
    assert brier["value"] == -0.01 and brier["direction"] == "lower_better"
    missing = extract_axis_effect(
        axis,
        contrast="CAT_DYN_minus_CAT_COV",
        metric="delta_recall_pp",
        area_budget_km2=600000,
        mode="secondary_70km",
    )
    assert missing["value"] is None and missing["status"] == "metric_NA_or_not_in_schema"
