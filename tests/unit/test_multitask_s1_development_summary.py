from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Literal

import pytest

from seismoflux.multitask_s1.development_score import (
    FIXED_EPISODE_DEFINITION,
    MAIN_SCIENTIFIC_ANCHOR,
    DevelopmentRawScores,
    JointRawScoreRow,
    LocationRawScoreRow,
    MagnitudeRawScoreRow,
    TimeRawScoreRow,
)
from seismoflux.multitask_s1.development_summary import (
    BOOTSTRAP_REPLICATES,
    DevelopmentSummaryError,
    summarize_development_scores,
)

MagnitudeBin = Literal["M5_6", "M6_plus"]

_FOLDS = (
    "C_DEV_2000_2004",
    "C_DEV_2005_2009",
    "C_DEV_2010_2014",
    "C_DEV_2015_2019",
)
_LOCATION_MODELS = (
    "L0_UNIFORM",
    "L1_REGIONAL_CONSTANT",
    "L2_KDE_CAUSAL",
    "L2_KDE75_LEGACY",
    "L3_B0_R30_CAUSAL",
)
_JOINT_MODELS = (
    "J0_U_P_GR",
    "J1_R_P_GR",
    "J2_KDE_P_GR",
    "J3_R30_P_GR",
    "J4_KDE_NB_GR",
)


def _issue(index: int) -> datetime:
    return datetime(2000 + index * 5, 1, 6, 16, tzinfo=UTC)


def _location_recall(
    *,
    fold_id: str,
    issue: datetime,
    model_id: str,
    event_id: str,
    episode_id: str,
    global_member_count: int,
    hit: bool,
    horizon_days: int = 30,
    magnitude_bin: MagnitudeBin = "M5_6",
    area_budget_km2: float = 600_000.0,
) -> LocationRawScoreRow:
    is_main = horizon_days == 30 and magnitude_bin == "M5_6" and area_budget_km2 == 600_000.0
    return LocationRawScoreRow(
        fold_id=fold_id,
        issue_time_utc=issue,
        horizon_days=horizon_days,
        magnitude_bin=magnitude_bin,
        model_id=model_id,
        metric="strict_recall",
        basis="anchor",
        area_budget_km2=area_budget_km2,
        actual_area_km2=area_budget_km2,
        value=1.0 if hit else 0.0,
        event_count=1,
        hit_weight=1.0 if hit else 0.0,
        total_weight=1.0,
        status="evaluable",
        is_main_scientific_anchor=is_main,
        scientific_anchor_id=MAIN_SCIENTIFIC_ANCHOR if is_main else None,
        event_ids=(event_id,),
        event_log_densities_per_km2=(-10.0,),
        episode_ids=(episode_id,),
        global_episode_member_counts=(global_member_count,),
        is_episode_anchor=(True,),
        event_cell_indices=(0,),
        event_longitudes=(100.0,),
        event_latitudes=(30.0,),
        event_weights=(1.0,),
        hit_flags=(hit,),
        episode_definition=FIXED_EPISODE_DEFINITION,
    )


def _location_density(
    *,
    fold_id: str,
    issue: datetime,
    model_id: str,
    event_id: str,
    episode_id: str,
    log_density: float,
) -> LocationRawScoreRow:
    return LocationRawScoreRow(
        fold_id=fold_id,
        issue_time_utc=issue,
        horizon_days=30,
        magnitude_bin="M5_6",
        model_id=model_id,
        metric="spatial_log_density",
        basis="all",
        area_budget_km2=None,
        actual_area_km2=None,
        value=log_density,
        event_count=1,
        hit_weight=None,
        total_weight=None,
        status="evaluable",
        is_main_scientific_anchor=False,
        scientific_anchor_id=None,
        event_ids=(event_id,),
        event_log_densities_per_km2=(log_density,),
        episode_ids=(episode_id,),
        global_episode_member_counts=(1,),
        is_episode_anchor=(True,),
        event_cell_indices=(0,),
        event_longitudes=(100.0,),
        event_latitudes=(30.0,),
        event_weights=(1.0,),
        hit_flags=None,
        episode_definition=FIXED_EPISODE_DEFINITION,
    )


def _time_row(
    *,
    fold_id: str,
    issue: datetime,
    horizon_days: int,
    model_id: str,
    observed_count: int,
    log_score: float,
) -> TimeRawScoreRow:
    is_t1 = model_id == "T1_NEGATIVE_BINOMIAL"
    return TimeRawScoreRow(
        fold_id=fold_id,
        issue_time_utc=issue,
        horizon_days=horizon_days,
        magnitude_band="M5_6",
        model_id=model_id,
        distribution="nb2" if is_t1 else "poisson",
        observed_count=observed_count,
        expected_count=0.5,
        count_log_score=log_score,
        occurrence_brier=0.15 if is_t1 else 0.2,
        count_bias=0.5 - observed_count,
        status="evaluable",
        reason="synthetic_evaluable",
    )


def _magnitude_rows(
    fold_id: str, issue: datetime, event_id: str
) -> tuple[MagnitudeRawScoreRow, ...]:
    return (
        MagnitudeRawScoreRow(
            fold_id=fold_id,
            forecast_issue_time_utc=issue,
            model_id="M0_GR_GLOBAL",
            conditional_support="M>=4 unique physical events",
            event_ids=(event_id,),
            event_log_probabilities=(-2.0,),
            log_probability_sum=-2.0,
            mean_log_probability=-2.0,
            m6_plus_probability=0.1,
            mean_m6_plus_brier=0.1,
            status="evaluable",
        ),
        MagnitudeRawScoreRow(
            fold_id=fold_id,
            forecast_issue_time_utc=issue,
            model_id="M0_GR_GLOBAL",
            conditional_support="M>=5 unique physical events, M0 re-normalized tail",
            event_ids=(event_id,),
            event_log_probabilities=(-1.5,),
            log_probability_sum=-1.5,
            mean_log_probability=-1.5,
            m6_plus_probability=0.2,
            mean_m6_plus_brier=0.12,
            status="evaluable",
        ),
        MagnitudeRawScoreRow(
            fold_id=fold_id,
            forecast_issue_time_utc=issue,
            model_id="M3_GR_LONG_M5",
            conditional_support="M>=5 unique physical events, conditional tail",
            event_ids=(event_id,),
            event_log_probabilities=(-1.4,),
            log_probability_sum=-1.4,
            mean_log_probability=-1.4,
            m6_plus_probability=0.22,
            mean_m6_plus_brier=0.1,
            status="evaluable",
        ),
    )


def _joint_row(
    *,
    fold_id: str,
    issue: datetime,
    model_id: str,
    event_count: int,
    log_score: float,
) -> JointRawScoreRow:
    return JointRawScoreRow(
        fold_id=fold_id,
        issue_time_utc=issue,
        horizon_days=30,
        joint_model_id=model_id,
        event_count=event_count,
        count_distribution="nb2" if model_id == "J4_KDE_NB_GR" else "poisson",
        count_log_score=-0.5,
        conditional_location_log_density_sum=-1.0 if event_count else 0.0,
        conditional_magnitude_log_probability_sum=-0.5 if event_count else 0.0,
        joint_log_score=log_score,
        status="evaluable",
    )


def _synthetic_raw_scores() -> DevelopmentRawScores:
    baseline_hits = (False, True, False, True)
    location: list[LocationRawScoreRow] = []
    time: list[TimeRawScoreRow] = []
    magnitude: list[MagnitudeRawScoreRow] = []
    joint: list[JointRawScoreRow] = []
    for index, fold_id in enumerate(_FOLDS):
        issue = _issue(index)
        event_id = f"event-{index}"
        episode_id = f"episode-{index}"
        model_hits = {
            "L0_UNIFORM": baseline_hits[index],
            "L1_REGIONAL_CONSTANT": True,
            "L2_KDE_CAUSAL": False,
            "L2_KDE75_LEGACY": baseline_hits[index],
            "L3_B0_R30_CAUSAL": index == 0,
        }
        for model_id in _LOCATION_MODELS:
            location.append(
                _location_recall(
                    fold_id=fold_id,
                    issue=issue,
                    model_id=model_id,
                    event_id=event_id,
                    episode_id=episode_id,
                    global_member_count=5 - index,
                    hit=model_hits[model_id],
                )
            )
        location.extend(
            (
                _location_density(
                    fold_id=fold_id,
                    issue=issue,
                    model_id="L0_UNIFORM",
                    event_id=event_id,
                    episode_id=episode_id,
                    log_density=-10.0,
                ),
                _location_density(
                    fold_id=fold_id,
                    issue=issue,
                    model_id="L1_REGIONAL_CONSTANT",
                    event_id=event_id,
                    episode_id=episode_id,
                    log_density=-9.0,
                ),
            )
        )
        if index == 0:
            location.extend(
                (
                    _location_recall(
                        fold_id=fold_id,
                        issue=issue,
                        model_id="L0_UNIFORM",
                        event_id="axis-event",
                        episode_id="axis-episode",
                        global_member_count=1,
                        hit=False,
                        horizon_days=7,
                        magnitude_bin="M6_plus",
                        area_budget_km2=300_000.0,
                    ),
                    _location_recall(
                        fold_id=fold_id,
                        issue=issue,
                        model_id="L1_REGIONAL_CONSTANT",
                        event_id="axis-event",
                        episode_id="axis-episode",
                        global_member_count=1,
                        hit=True,
                        horizon_days=7,
                        magnitude_bin="M6_plus",
                        area_budget_km2=300_000.0,
                    ),
                )
            )

        observed = 0 if index % 2 == 0 else 1
        time.extend(
            (
                _time_row(
                    fold_id=fold_id,
                    issue=issue,
                    horizon_days=30,
                    model_id="T0_POISSON_EXPANDING",
                    observed_count=observed,
                    log_score=-2.0,
                ),
                _time_row(
                    fold_id=fold_id,
                    issue=issue,
                    horizon_days=30,
                    model_id="T1_NEGATIVE_BINOMIAL",
                    observed_count=observed,
                    log_score=(-1.8, -2.1, -1.7, -2.0)[index],
                ),
                _time_row(
                    fold_id=fold_id,
                    issue=issue,
                    horizon_days=7,
                    model_id="T0_POISSON_EXPANDING",
                    observed_count=observed,
                    log_score=-2.0,
                ),
                _time_row(
                    fold_id=fold_id,
                    issue=issue,
                    horizon_days=7,
                    model_id="T1_NEGATIVE_BINOMIAL",
                    observed_count=observed,
                    log_score=-2.2,
                ),
            )
        )
        magnitude.extend(_magnitude_rows(fold_id, issue, f"magnitude-event-{index}"))
        joint_deltas = {
            "J0_U_P_GR": 0.0,
            "J1_R_P_GR": 0.1,
            "J2_KDE_P_GR": -0.1,
            "J3_R30_P_GR": 0.0,
            "J4_KDE_NB_GR": 0.05,
        }
        for model_id in _JOINT_MODELS:
            joint.append(
                _joint_row(
                    fold_id=fold_id,
                    issue=issue,
                    model_id=model_id,
                    event_count=observed,
                    log_score=-3.0 + joint_deltas[model_id],
                )
            )
    return DevelopmentRawScores(
        location=tuple(location),
        time=tuple(time),
        magnitude=tuple(magnitude),
        joint=tuple(joint),
    )


def _comparison(summary: dict[str, object], model_id: str) -> dict[str, object]:
    location = summary["location"]
    assert isinstance(location, dict)
    anchor = location["main_scientific_anchor"]
    assert isinstance(anchor, dict)
    comparisons = anchor["comparisons"]
    assert isinstance(comparisons, list)
    return next(
        item
        for item in comparisons
        if isinstance(item, dict) and item["candidate_model_id"] == model_id
    )


def _metric_value(container: object, key: str) -> float | None:
    assert isinstance(container, dict)
    metric = container[key]
    assert isinstance(metric, dict)
    value = metric["value"]
    assert value is None or isinstance(value, float)
    return value


def test_directional_summary_is_deterministic_and_json_compatible() -> None:
    raw = _synthetic_raw_scores()
    first = summarize_development_scores(raw)
    second = summarize_development_scores(raw)
    assert first == second
    json.dumps(first, allow_nan=False, sort_keys=True)

    l1 = _comparison(first, "L1_REGIONAL_CONSTANT")
    l2 = _comparison(first, "L2_KDE_CAUSAL")
    assert _metric_value(l1, "pooled_difference") == pytest.approx(0.5)
    assert l1["positive_fold_count"] == 2
    assert l1["direction"] == "positive"
    assert _metric_value(l2, "pooled_difference") == pytest.approx(-0.5)
    assert l2["direction"] == "non_positive"
    bootstrap = l1["paired_bootstrap"]
    assert isinstance(bootstrap, dict)
    assert bootstrap["replicates_evaluable"] == BOOTSTRAP_REPLICATES
    region = l1["fixed_origin_500km_region_sensitivity"]
    assert isinstance(region, dict)
    assert region["status"] == "not_evaluable"

    time = first["time"]
    assert isinstance(time, dict)
    time_comparisons = time["comparisons"]
    assert isinstance(time_comparisons, list)
    t30 = next(
        item
        for item in time_comparisons
        if isinstance(item, dict)
        and item["magnitude_band"] == "M5_6"
        and item["horizon_days"] == 30
    )
    assert t30["zero_event_exposure_count"] == 2
    assert _metric_value(t30, "pooled_count_log_score_difference_T1_minus_T0") == pytest.approx(0.1)

    magnitude = first["magnitude"]
    assert isinstance(magnitude, dict)
    tail = magnitude["M3_vs_M0_common_M5_sensitivity"]
    assert isinstance(tail, dict)
    assert tail["unique_event_count"] == 4
    assert tail["direct_comparison_to_M0_M4_support_allowed"] is False
    assert _metric_value(tail, "mean_log_probability_difference_M3_minus_M0") == pytest.approx(0.1)

    joint = first["joint"]
    assert isinstance(joint, dict)
    assert joint["M5_6_and_M6_plus_count_terms_summed"] is False
    joint_comparisons = joint["comparisons"]
    assert isinstance(joint_comparisons, list)
    j1 = next(
        item
        for item in joint_comparisons
        if isinstance(item, dict)
        and item["horizon_days"] == 30
        and item["candidate_model_id"] == "J1_R_P_GR"
    )
    assert _metric_value(j1, "mean_joint_log_score_difference") == pytest.approx(0.1)
    assert j1["zero_event_exposure_count"] == 2
    assert first["champion_selection_allowed"] is False
    assert first["holdout_opening_allowed"] is False
    assert first["mandatory_next_stage_regardless_of_screen_result"] == (
        "S1-C1_causal_local_completeness_main"
    )


def test_horizon_area_and_magnitude_bin_are_never_pooled() -> None:
    summary = summarize_development_scores(_synthetic_raw_scores())
    location = summary["location"]
    assert isinstance(location, dict)
    recall_groups = location["recall_groups"]
    assert isinstance(recall_groups, list)
    axis_rows = [
        item
        for item in recall_groups
        if isinstance(item, dict)
        and item["fold_id"] == _FOLDS[0]
        and item["model_id"] == "L0_UNIFORM"
    ]
    axes = {
        (item["horizon_days"], item["magnitude_bin"], item["area_budget_km2"]) for item in axis_rows
    }
    assert (30, "M5_6", 600_000.0) in axes
    assert (7, "M6_plus", 300_000.0) in axes
    assert all(item["exposure_count"] == 1 for item in axis_rows)

    densities = location["log_density_and_information_gain_groups"]
    assert isinstance(densities, list)
    l1_density = next(
        item
        for item in densities
        if isinstance(item, dict)
        and item["fold_id"] == _FOLDS[0]
        and item["model_id"] == "L1_REGIONAL_CONSTANT"
    )
    assert _metric_value(l1_density, "paired_information_gain_vs_L0_nats_per_event") == 1.0

    time = summary["time"]
    assert isinstance(time, dict)
    comparisons = time["comparisons"]
    assert isinstance(comparisons, list)
    h7 = next(
        item
        for item in comparisons
        if isinstance(item, dict) and item["magnitude_band"] == "M5_6" and item["horizon_days"] == 7
    )
    h30 = next(
        item
        for item in comparisons
        if isinstance(item, dict)
        and item["magnitude_band"] == "M5_6"
        and item["horizon_days"] == 30
    )
    h7_difference = _metric_value(h7, "pooled_count_log_score_difference_T1_minus_T0")
    h30_difference = _metric_value(h30, "pooled_count_log_score_difference_T1_minus_T0")
    assert h7_difference is not None and h7_difference < 0.0
    assert h30_difference is not None and h30_difference > 0.0
    prohibitions = summary["pooling_prohibitions"]
    assert isinstance(prohibitions, dict)
    assert prohibitions == {
        "horizons_pooled": False,
        "alarm_areas_pooled": False,
        "magnitude_bins_pooled": False,
        "grid_cells_treated_as_independent_units": False,
    }


def test_empty_and_single_episode_do_not_fabricate_uncertainty() -> None:
    empty = summarize_development_scores(
        DevelopmentRawScores(location=(), time=(), magnitude=(), joint=())
    )
    json.dumps(empty, allow_nan=False)
    l1_empty = _comparison(empty, "L1_REGIONAL_CONSTANT")
    assert l1_empty["status"] == "not_evaluable"
    empty_bootstrap = l1_empty["paired_bootstrap"]
    assert isinstance(empty_bootstrap, dict)
    assert empty_bootstrap["replicates_evaluable"] == 0

    issue = _issue(0)
    baseline = _location_recall(
        fold_id=_FOLDS[0],
        issue=issue,
        model_id="L0_UNIFORM",
        event_id="one-event",
        episode_id="one-episode",
        global_member_count=12,
        hit=False,
    )
    candidate = _location_recall(
        fold_id=_FOLDS[0],
        issue=issue,
        model_id="L1_REGIONAL_CONSTANT",
        event_id="one-event",
        episode_id="one-episode",
        global_member_count=12,
        hit=True,
    )
    single = summarize_development_scores(
        DevelopmentRawScores(location=(baseline, candidate), time=(), magnitude=(), joint=())
    )
    l1_single = _comparison(single, "L1_REGIONAL_CONSTANT")
    assert l1_single["independent_episode_count"] == 1
    single_bootstrap = l1_single["paired_bootstrap"]
    assert isinstance(single_bootstrap, dict)
    assert single_bootstrap["status"] == "not_evaluable"
    assert single_bootstrap["reason"] == "fewer_than_two_independent_episodes"
    leave_largest = l1_single["leave_largest_episode"]
    assert isinstance(leave_largest, dict)
    leave_effect = leave_largest["pooled_difference_after_removal"]
    assert isinstance(leave_effect, dict)
    assert leave_effect["status"] == "not_evaluable"


def test_paired_target_mismatch_fails_closed() -> None:
    issue = _issue(0)
    baseline = _location_recall(
        fold_id=_FOLDS[0],
        issue=issue,
        model_id="L0_UNIFORM",
        event_id="baseline-event",
        episode_id="episode",
        global_member_count=1,
        hit=False,
    )
    candidate = _location_recall(
        fold_id=_FOLDS[0],
        issue=issue,
        model_id="L1_REGIONAL_CONSTANT",
        event_id="different-event",
        episode_id="episode",
        global_member_count=1,
        hit=True,
    )
    with pytest.raises(DevelopmentSummaryError, match="paired target payload differs"):
        summarize_development_scores(
            DevelopmentRawScores(location=(baseline, candidate), time=(), magnitude=(), joint=())
        )


def test_negative_infinity_is_json_safe_not_nan() -> None:
    issue = _issue(0)
    baseline = _location_density(
        fold_id=_FOLDS[0],
        issue=issue,
        model_id="L0_UNIFORM",
        event_id="event",
        episode_id="episode",
        log_density=-10.0,
    )
    candidate = _location_density(
        fold_id=_FOLDS[0],
        issue=issue,
        model_id="L1_REGIONAL_CONSTANT",
        event_id="event",
        episode_id="episode",
        log_density=-math.inf,
    )
    summary = summarize_development_scores(
        DevelopmentRawScores(location=(baseline, candidate), time=(), magnitude=(), joint=())
    )
    json.dumps(summary, allow_nan=False)
    location = summary["location"]
    assert isinstance(location, dict)
    groups = location["log_density_and_information_gain_groups"]
    assert isinstance(groups, list)
    l1 = next(
        item
        for item in groups
        if isinstance(item, dict) and item["model_id"] == "L1_REGIONAL_CONSTANT"
    )
    mean_density = l1["mean_log_density_per_event"]
    assert isinstance(mean_density, dict)
    assert mean_density["status"] == "negative_infinity"
    assert mean_density["value"] is None
