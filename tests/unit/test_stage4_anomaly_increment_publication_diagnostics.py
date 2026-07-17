from __future__ import annotations

import dataclasses
import inspect
import math

import pytest

import seismoflux.anomaly_increment.publication_diagnostics as diagnostics_module
from seismoflux.anomaly_increment.publication_diagnostics import (
    PUBLICATION_ALARM_BUDGETS_KM2,
    AlarmBudgetRecallCurve,
    AlarmBudgetRecallPoint,
    CoefficientEffectCurve,
    DataMethodFlowDiagnostics,
    DistanceLeadDecayDiagnostics,
    FoldMacroValue,
    IncrementVariant,
    InformationGainInterval,
    PermutationDistribution,
    PublicationDiagnostics,
    RegionHorizonMetric,
    SameRecallAreaReduction,
)


def _null(offset: float, *, include_failure: bool = False) -> tuple[float, ...]:
    values = [offset + index * 0.00035 for index in range(1_000)]
    if include_failure:
        values[-1] = math.inf
    return tuple(values)


def _information_gain() -> tuple[InformationGainInterval, ...]:
    output: list[InformationGainInterval] = []
    for magnitude_bin in ("M5_6", "M6_plus"):
        for horizon in (7, 30, 90, 180, 365):
            count = 12 if magnitude_bin == "M5_6" else (2 if horizon == 7 else 0)
            if horizon in {180, 365} and count > 0:
                point = 0.18 - 0.0008 * horizon
                output.append(
                    InformationGainInterval(
                        magnitude_bin,
                        horizon,
                        count,
                        "evidence_insufficient_no_random_split",
                        point,
                        point - 0.05,
                        point + 0.05,
                    )
                )
            elif count == 0:
                output.append(
                    InformationGainInterval(
                        magnitude_bin,
                        horizon,
                        0,
                        "evidence_insufficient_zero_events",
                        None,
                        None,
                        None,
                    )
                )
            else:
                point = 0.18 - 0.0008 * horizon
                output.append(
                    InformationGainInterval(
                        magnitude_bin,
                        horizon,
                        count,
                        ("exploratory_low_sample" if magnitude_bin == "M6_plus" else "evaluated"),
                        point,
                        point - 0.05,
                        point + 0.05,
                    )
                )
    return tuple(output)


def _curves() -> tuple[AlarmBudgetRecallCurve, ...]:
    budgets = PUBLICATION_ALARM_BUDGETS_KM2
    output: list[AlarmBudgetRecallCurve] = []
    for variant in ("background_no_increment", "dynamic"):
        for horizon in (7, 30, 90):
            advantage = 0.09 if variant == "dynamic" else 0.0
            horizon_offset = {7: 0.04, 30: 0.02, 90: 0.0}[horizon]
            recalls = tuple(
                min(0.95, base + advantage + horizon_offset)
                for base in (0.12, 0.22, 0.35, 0.5, 0.62)
            )
            output.append(
                AlarmBudgetRecallCurve(
                    variant=variant,
                    magnitude_bin="M5_6",
                    horizon_days=horizon,
                    points=tuple(
                        AlarmBudgetRecallPoint(
                            budget_km2=budget,
                            selected_alarm_area_km2=budget - 5_000.0,
                            strict_recall=recall,
                            all_study_area_event_count=12,
                            evidence_status="evaluated",
                        )
                        for budget, recall in zip(budgets, recalls, strict=True)
                    ),
                )
            )
    return tuple(output)


def _diagnostics(*, region_count: int = 39) -> PublicationDiagnostics:
    horizons = (7, 30, 90, 180, 365)
    regions = tuple(f"construction-zone-{index:02d}" for index in range(1, region_count + 1))
    effect_specs: tuple[tuple[IncrementVariant, float], ...] = (
        ("coverage_only", 0.12),
        ("snapshot", 0.21),
        ("dynamic", 0.33),
    )
    return PublicationDiagnostics(
        data_flow=DataMethodFlowDiagnostics(
            training_issue_count=84,
            training_cell_count=12_480,
            fitted_feature_count=18,
            independent_event_count=37,
            study_area_km2=9_600_000.0,
        ),
        coefficient_effects=tuple(
            CoefficientEffectCurve(
                variant=variant,
                coefficient_name=f"{variant}_signal",
                coefficient_estimate=coefficient,
                input_values=(-1.0, 0.0, 1.0),
                effect_values=(-coefficient, 0.0, coefficient),
            )
            for variant, coefficient in effect_specs
        ),
        distance_lead_decay=DistanceLeadDecayDiagnostics(
            distance_km=(0.0, 100.0, 200.0, 400.0),
            spatial_relative_weight=(1.0, 0.78, 0.37, 0.02),
            lead_days=(0.0, 30.0, 90.0, 180.0),
            temporal_relative_weight=(1.0, 0.79, 0.5, 0.25),
        ),
        fold_macro_values=(
            FoldMacroValue(1, (4, 5, 6), "evaluated", 0.08, 0.03),
            FoldMacroValue(2, (5, 5, 7), "evaluated", 0.11, 0.05),
            FoldMacroValue(3, (3, 4, 5), "evaluated", 0.06, 0.02),
        ),
        permutation_distributions=(
            PermutationDistribution("dynamic", "time", 0.15, _null(-0.2)),
            PermutationDistribution(
                "dynamic",
                "space",
                0.15,
                _null(-0.24, include_failure=True),
            ),
            PermutationDistribution("snapshot", "time", 0.09, _null(-0.18)),
        ),
        information_gain_intervals=_information_gain(),
        region_ids=regions,
        region_horizon_metrics=tuple(
            RegionHorizonMetric(
                region_id=region,
                horizon_days=horizon,
                information_gain_nats_per_event=(
                    None
                    if region_index == region_count - 1 and horizon == 365
                    else (region_index % 7 - 3) * 0.015 - horizon_index * 0.005
                ),
                supported_event_count=(
                    0 if region_index == region_count - 1 and horizon == 365 else 2
                ),
                all_study_area_event_count=(
                    0 if region_index == region_count - 1 and horizon == 365 else 3
                ),
                strict_recall=(
                    None
                    if region_index == region_count - 1 and horizon == 365
                    else min(1.0, 0.2 + region_index * 0.01 + horizon_index * 0.03)
                ),
                information_gain_evidence_status=(
                    "evidence_insufficient_zero_supported_events"
                    if region_index == region_count - 1 and horizon == 365
                    else "evaluated"
                ),
                strict_recall_evidence_status=(
                    "evidence_insufficient_zero_all_events"
                    if region_index == region_count - 1 and horizon == 365
                    else "evaluated"
                ),
            )
            for region_index, region in enumerate(regions)
            for horizon_index, horizon in enumerate(horizons)
        ),
        alarm_budget_curves=_curves(),
        same_recall_area_reductions=tuple(
            SameRecallAreaReduction(
                magnitude_bin="M5_6",
                horizon_days=horizon,
                target_recall=0.5,
                comparator_variant="background_no_increment",
                candidate_variant="dynamic",
                comparator_area_km2=750_000.0,
                candidate_area_km2=600_000.0 - horizon_index * 15_000.0,
                area_reduction_lower_95=0.1,
                area_reduction_upper_95=0.35,
            )
            for horizon_index, horizon in enumerate((7, 30, 90))
        ),
    )


def test_complete_publication_diagnostics_are_deterministic_and_content_hashed() -> None:
    first = _diagnostics()
    second = _diagnostics()

    assert first.as_mapping() == second.as_mapping()
    assert first.content_sha256 == second.content_sha256
    assert len(first.content_sha256) == 64
    assert first.as_mapping()["content_sha256"] == first.content_sha256
    assert len(first.permutation_distributions) == 3
    assert all(len(item.null_statistics) == 1_000 for item in first.permutation_distributions)
    assert first.permutation_distributions[1].scientific_failure_count == 1
    assert len(first.information_gain_intervals) == 2 * 5
    m6_primary = next(
        item
        for item in first.information_gain_intervals
        if item.magnitude_bin == "M6_plus" and item.horizon_days == 7
    )
    assert m6_primary.evidence_status == "exploratory_low_sample"
    assert len(first.region_horizon_metrics) == 39 * 5
    assert len(first.alarm_budget_curves) == 2 * 3
    assert all(len(curve.points) == 5 for curve in first.alarm_budget_curves)
    assert first.same_recall_area_reductions[0].area_reduction_fraction == pytest.approx(0.2)


def test_missing_or_incomplete_formal_diagnostics_fail_closed() -> None:
    complete = _diagnostics()
    with pytest.raises(ValueError, match="complete region-by-horizon"):
        dataclasses.replace(
            complete,
            region_horizon_metrics=complete.region_horizon_metrics[:-1],
        )
    with pytest.raises(ValueError, match="1000 values"):
        PermutationDistribution(
            "dynamic",
            "time",
            0.15,
            tuple(float(index) for index in range(999)),
        )
    with pytest.raises(ValueError, match="1% ceiling"):
        PermutationDistribution(
            "dynamic",
            "space",
            0.15,
            (math.inf,) * 11 + tuple(float(index) for index in range(989)),
        )
    with pytest.raises(ValueError, match="at least three real budget points"):
        AlarmBudgetRecallCurve(
            variant="dynamic",
            magnitude_bin="M5_6",
            horizon_days=90,
            points=(
                AlarmBudgetRecallPoint(300_000.0, 295_000.0, 0.2, 4, "evaluated"),
                AlarmBudgetRecallPoint(600_000.0, 595_000.0, 0.5, 4, "evaluated"),
            ),
        )


def test_zero_event_metrics_remain_none_and_scientific_failures_are_json_safe() -> None:
    diagnostics = _diagnostics()
    zero_region = diagnostics.region_horizon_metrics[-1]
    m6_long = next(
        item
        for item in diagnostics.information_gain_intervals
        if item.magnitude_bin == "M6_plus" and item.horizon_days == 365
    )
    serialized = diagnostics.as_mapping()

    assert zero_region.supported_event_count == 0
    assert zero_region.all_study_area_event_count == 0
    assert zero_region.information_gain_nats_per_event is None
    assert zero_region.strict_recall is None
    assert m6_long.independent_event_count == 0
    assert m6_long.point_estimate is None
    permutation = serialized["permutation_distributions"]
    assert isinstance(permutation, list)
    assert "positive_infinity_scientific_failure" in permutation[1]["null_statistics"]


def test_legitimate_evidence_insufficient_branches_do_not_require_fake_zeroes() -> None:
    fold = FoldMacroValue(
        1,
        (2, 0, 3),
        "evidence_insufficient_zero_events",
        None,
        None,
    )
    permutation = PermutationDistribution(
        "dynamic",
        "time",
        None,
        (),
        "evidence_insufficient_zero_events",
    )
    regional = RegionHorizonMetric(
        region_id="construction-zone-01",
        horizon_days=7,
        information_gain_nats_per_event=None,
        supported_event_count=0,
        all_study_area_event_count=2,
        strict_recall=0.0,
        information_gain_evidence_status="evidence_insufficient_zero_supported_events",
        strict_recall_evidence_status="evaluated",
    )
    reduction = SameRecallAreaReduction(
        magnitude_bin="M5_6",
        horizon_days=7,
        target_recall=0.0,
        comparator_variant="background_no_increment",
        candidate_variant="dynamic",
        comparator_area_km2=600_000.0,
        candidate_area_km2=None,
        area_reduction_lower_95=None,
        area_reduction_upper_95=None,
        evidence_status="evidence_insufficient_zero_comparator_recall",
    )

    assert fold.dynamic_macro_information_gain is None
    assert permutation.p_value is None
    assert regional.information_gain_nats_per_event is None
    assert regional.strict_recall == 0.0
    assert reduction.area_reduction_fraction is None


def test_permutation_failure_fraction_above_one_percent_is_published_as_insufficient() -> None:
    nulls = (float("inf"),) * 11 + (0.0,) * 989
    distribution = PermutationDistribution(
        "dynamic",
        "space",
        0.2,
        nulls,
        "evidence_insufficient_scientific_failure_fraction",
    )

    assert distribution.scientific_failure_count == 11
    assert distribution.evidence_status == ("evidence_insufficient_scientific_failure_fraction")
    with pytest.raises(ValueError, match="exceeds the 1% failure ceiling"):
        PermutationDistribution("dynamic", "space", 0.2, nulls, "evaluated")


def test_percentile_intervals_need_not_contain_points_and_area_can_plateau() -> None:
    interval = InformationGainInterval(
        "M5_6",
        7,
        4,
        "evaluated",
        0.5,
        -0.2,
        0.3,
    )
    reduction = SameRecallAreaReduction(
        magnitude_bin="M5_6",
        horizon_days=7,
        target_recall=0.5,
        comparator_variant="background_no_increment",
        candidate_variant="dynamic",
        comparator_area_km2=750_000.0,
        candidate_area_km2=600_000.0,
        area_reduction_lower_95=-0.1,
        area_reduction_upper_95=0.1,
    )
    points = tuple(
        AlarmBudgetRecallPoint(budget, area, recall, 4, "evaluated")
        for budget, area, recall in zip(
            PUBLICATION_ALARM_BUDGETS_KM2,
            (295_000.0, 445_000.0, 595_000.0, 595_000.0, 955_000.0),
            (0.1, 0.2, 0.3, 0.3, 0.5),
            strict=True,
        )
    )
    curve = AlarmBudgetRecallCurve("dynamic", "M5_6", 7, points)

    assert interval.point_estimate == 0.5
    assert reduction.area_reduction_fraction == pytest.approx(0.2)
    assert curve.points[2].selected_alarm_area_km2 == curve.points[3].selected_alarm_area_km2


def test_low_sample_m6_information_gain_cannot_be_mislabeled_confirmatory() -> None:
    with pytest.raises(ValueError, match="must remain exploratory"):
        InformationGainInterval(
            "M6_plus",
            7,
            3,
            "evaluated",
            0.2,
            0.1,
            0.3,
        )

    combined = InformationGainInterval(
        "M6_plus",
        180,
        3,
        "exploratory_low_sample_no_random_split",
        0.15,
        0.02,
        0.31,
    )
    assert combined.independent_event_count == 3


def test_publication_contract_contains_no_geometry_cell_or_loading_capability() -> None:
    field_names = {
        field.name.casefold()
        for cls in (
            PublicationDiagnostics,
            RegionHorizonMetric,
            AlarmBudgetRecallCurve,
            SameRecallAreaReduction,
        )
        for field in dataclasses.fields(cls)
    }
    assert not any(
        token in field
        for field in field_names
        for token in (
            "longitude",
            "latitude",
            "geojson",
            "wkb",
            "cell_id",
            "query_x",
            "query_y",
            "target_id",
        )
    )
    source = inspect.getsource(diagnostics_module).casefold()
    assert "from pathlib" not in source
    assert "open(" not in source
    assert "read_" not in source
    assert "load_" not in source
