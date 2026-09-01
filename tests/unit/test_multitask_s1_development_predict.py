from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pyproj import CRS, Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s1 import development_predict
from seismoflux.multitask_s1.development_predict import (
    LOCATION_MODEL_IDS,
    TIME_BANDS,
    DevelopmentFoldPrediction,
    DevelopmentPredictionError,
    FoldHorizonParameterSelection,
    InnerTimeCountSeries,
    StrictlyEarlierInnerLocationWindow,
    build_primary_issue_prediction,
    build_strictly_earlier_inner_m4_location_block,
    build_strictly_earlier_inner_time_count_series,
    build_weekly_magnitude_snapshot,
    fit_t1_qualifications_from_inner_count_series,
    fold_prediction_npz_arrays,
    select_location_parameters_from_strictly_earlier_inner_blocks,
    validate_frozen_fold_prediction_npz_arrays,
)
from seismoflux.multitask_s1.location import (
    CausalRecent30History,
    CausalSpatialHistory,
    FrozenSpatialGrid,
    l2_gaussian_kde_relative_mass,
    l3_b0_r30_relative_mass,
)
from seismoflux.multitask_s1.runner_inputs import (
    InnerExposure,
    OuterIssueRow,
    catalog_event_table_from_frame,
)

FOLD = "C_DEV_2000_2004"
LON = 105.0
LAT = 35.0


def _row(
    event_id: str,
    origin: datetime,
    magnitude: float,
    *,
    available: datetime | None = None,
    inside: bool = True,
    longitude: float = LON,
    latitude: float = LAT,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "origin_time_utc": origin.isoformat(),
        "available_at": (available or origin).isoformat(),
        "longitude": longitude,
        "latitude": latitude,
        "magnitude": magnitude,
        "inside_study_area": inside,
    }


def _catalog(rows: list[dict[str, object]]) -> Any:
    return catalog_event_table_from_frame(pd.DataFrame(rows))


def _one_cell_grid() -> FrozenSpatialGrid:
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(EQUAL_AREA_CRS), always_xy=True
    )
    x_m, y_m = transformer.transform(LON, LAT)
    return FrozenSpatialGrid(
        x_km=np.asarray([float(x_m) / 1_000.0]),
        y_km=np.asarray([float(y_m) / 1_000.0]),
        area_km2=np.asarray([625.0]),
    )


def _selection_catalog(*, future_event: bool = False) -> Any:
    rows = [
        _row("h4a", datetime(1970, 1, 2, tzinfo=UTC), 4.0),
        _row("h4b", datetime(1971, 1, 2, tzinfo=UTC), 4.4),
        _row("h5a", datetime(1972, 1, 2, tzinfo=UTC), 5.1),
        _row("h5b", datetime(1973, 1, 2, tzinfo=UTC), 5.7),
        _row("h6a", datetime(1974, 1, 2, tzinfo=UTC), 6.1),
        _row("old_m5", datetime(1950, 1, 2, tzinfo=UTC), 5.4),
    ]
    issues = (
        datetime(1985, 1, 3, tzinfo=UTC),
        datetime(1990, 1, 4, tzinfo=UTC),
        datetime(1995, 1, 5, tzinfo=UTC),
    )
    for block_index, issue in enumerate(issues, start=1):
        for event_index, magnitude in enumerate((4.1, 4.5, 5.2, 6.2), start=1):
            rows.append(
                _row(
                    f"i{block_index}_{event_index}",
                    issue + timedelta(days=event_index),
                    magnitude,
                )
            )
    if future_event:
        rows.append(
            _row(
                "future_outlier",
                datetime(2025, 1, 1, tzinfo=UTC),
                8.0,
                longitude=80.0,
                latitude=20.0,
            )
        )
    return _catalog(rows)


def _inner_exposures() -> tuple[InnerExposure, ...]:
    issues = (
        datetime(1985, 1, 3, tzinfo=UTC),
        datetime(1990, 1, 4, tzinfo=UTC),
        datetime(1995, 1, 5, tzinfo=UTC),
    )
    return tuple(
        InnerExposure(
            fold_id=FOLD,
            block_id=f"I{index}",
            horizon_days=7,
            issue_times_utc=(issue,),
        )
        for index, issue in enumerate(issues, start=1)
    )


def _block_ends() -> dict[str, datetime]:
    return {
        exposure.block_id: exposure.issue_times_utc[0] + timedelta(days=7)
        for exposure in _inner_exposures()
    }


def _location_selection(catalog: Any) -> Any:
    exposures = _inner_exposures()
    ends = _block_ends()
    blocks = tuple(
        build_strictly_earlier_inner_m4_location_block(
            catalog,
            exposure,
            block_end_utc=ends[exposure.block_id],
            locate_lonlat=lambda _longitude, _latitude: 0,
        )
        for exposure in exposures
    )
    return select_location_parameters_from_strictly_earlier_inner_blocks(
        catalog,
        _one_cell_grid(),
        blocks,
        outer_start_utc=ends["I3"] + timedelta(days=30),
    )


def test_inner_m4_builder_uses_open_closed_interval_and_rejects_outer_type() -> None:
    issue = datetime(2000, 1, 6, tzinfo=UTC)
    end = issue + timedelta(days=7)
    block_end = end + timedelta(days=1)
    catalog = _catalog(
        [
            _row("at_issue", issue, 4.5),
            _row("inside", issue + timedelta(microseconds=1), 4.0),
            _row("at_end", end, 5.2),
            _row("after_end", end + timedelta(microseconds=1), 5.5),
            _row("below_m4", issue + timedelta(days=2), 3.9),
            _row("outside", issue + timedelta(days=3), 6.0, inside=False),
            _row(
                "not_available_by_block_end",
                issue + timedelta(days=4),
                5.0,
                available=block_end + timedelta(microseconds=1),
            ),
        ]
    )
    exposure = InnerExposure(FOLD, "I1", 7, (issue,))

    block = build_strictly_earlier_inner_m4_location_block(
        catalog,
        exposure,
        block_end_utc=block_end,
        locate_lonlat=lambda _longitude, _latitude: 0,
    )

    assert block.windows[0].event_ids == ("inside", "at_end")
    assert block.windows[0].event_cell_indices == (0, 0)
    outer = OuterIssueRow(FOLD, issue, 7, end, "mature", True)
    with pytest.raises(TypeError, match="never an outer"):
        build_strictly_earlier_inner_m4_location_block(
            catalog,
            cast(Any, outer),
            block_end_utc=block_end,
            locate_lonlat=lambda _longitude, _latitude: 0,
        )


def test_inner_builder_asserts_same_horizon_primary_event_ids_are_unique() -> None:
    first = datetime(2000, 1, 6, tzinfo=UTC)
    second = first + timedelta(days=1)
    catalog = _catalog([_row("overlap", second + timedelta(days=1), 5.0)])
    overlapping = InnerExposure(FOLD, "I1", 7, (first, second))

    with pytest.raises(DevelopmentPredictionError, match="multiple primary inner windows"):
        build_strictly_earlier_inner_m4_location_block(
            catalog,
            overlapping,
            block_end_utc=second + timedelta(days=7),
            locate_lonlat=lambda _longitude, _latitude: 0,
        )


def test_location_selection_uses_three_equal_blocks_and_frozen_tie_rules() -> None:
    catalog = _selection_catalog(future_event=True)

    selection = _location_selection(catalog)

    assert selection.inner_block_event_counts == (4, 4, 4)
    assert selection.inner_event_count == 12
    assert selection.regional_tau_years == 10.0
    assert selection.selected_bandwidth_km == 300.0
    assert selection.recent_alpha == 0.0
    assert len(selection.regional_candidates) == 3
    assert len(selection.kde_candidates) == 5
    assert len(selection.recent_candidates) == 4
    assert all(len(item.block_mean_log_density) == 3 for item in selection.kde_candidates)


def test_cached_kde_mixture_is_exactly_equivalent_and_avoids_repeat_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = FrozenSpatialGrid(
        x_km=np.asarray([0.0, 100.0, 200.0]),
        y_km=np.asarray([0.0, 0.0, 0.0]),
        area_km2=np.asarray([1.0, 2.0, 1.0]),
    )
    long_history = CausalSpatialHistory(np.asarray([0.0, 150.0]), np.asarray([0.0, 0.0]))
    issue_us = 100 * 86_400_000_000
    recent = CausalRecent30History(
        x_km=np.asarray([200.0]),
        y_km=np.asarray([0.0]),
        origin_time_us=np.asarray([issue_us - 2 * 86_400_000_000], dtype=np.int64),
        available_at_us=np.asarray([issue_us - 2 * 86_400_000_000], dtype=np.int64),
        issue_time_us=issue_us,
        data_cutoff_us=issue_us - 86_400_000_000,
    )
    long_surface = l2_gaussian_kde_relative_mass(long_history, grid, bandwidth_km=75.0)
    recent_surface = l2_gaussian_kde_relative_mass(
        recent.as_spatial_history(), grid, bandwidth_km=75.0, model_id="R30_COMPONENT"
    )
    cached = development_predict._l3_from_cached_surfaces(
        long_surface,
        recent_surface,
        recent_event_count=1,
        alpha=0.25,
    )
    direct = l3_b0_r30_relative_mass(
        long_history,
        recent,
        grid,
        bandwidth_km=75.0,
        alpha=0.25,
    )
    np.testing.assert_array_equal(cached.cell_relative_mass, direct.cell_relative_mass)
    assert cached.source_event_count == direct.source_event_count
    assert cached.bandwidth_km == direct.bandwidth_km
    assert cached.alpha == direct.alpha
    assert cached.recent_fallback_to_long == direct.recent_fallback_to_long

    prediction_module = cast(Any, development_predict)
    original = prediction_module.l2_gaussian_kde_relative_mass
    call_count = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(prediction_module, "l2_gaussian_kde_relative_mass", counted)
    catalog = _selection_catalog()
    selection = _location_selection(catalog)
    assert call_count == 15

    t1 = fit_t1_qualifications_from_inner_count_series(
        (
            InnerTimeCountSeries("m5_6", (1, 1, 1), (1.0, 1.0, 1.0)),
            InnerTimeCountSeries("m6_plus", (0, 0, 0), (0.2, 0.2, 0.2)),
            InnerTimeCountSeries("m5_plus_1970_for_joint", (1, 1, 1), (1.2, 1.2, 1.2)),
        ),
        horizon_days=7,
    )
    frozen = FoldHorizonParameterSelection(FOLD, 7, selection, t1)
    before_outer = call_count
    build_primary_issue_prediction(
        catalog,
        _one_cell_grid(),
        frozen,
        issue_time_utc=datetime(2000, 1, 6, tzinfo=UTC),
    )
    assert call_count - before_outer == 2


def test_recent_kde_underflow_zero_is_allowed_only_before_positive_l3_mixture() -> None:
    grid = FrozenSpatialGrid(
        x_km=np.asarray([0.0, 4_000.0]),
        y_km=np.asarray([0.0, 0.0]),
        area_km2=np.asarray([1.0, 1.0]),
    )
    issue = datetime(1985, 1, 3, tzinfo=UTC)
    window = StrictlyEarlierInnerLocationWindow(
        fold_id=FOLD,
        block_id="I1",
        horizon_days=7,
        issue_time_utc=issue,
        interval_end_utc=issue + timedelta(days=7),
        event_ids=("distant_target",),
        event_cell_indices=(1,),
    )
    recent_underflow = l2_gaussian_kde_relative_mass(
        CausalSpatialHistory(np.asarray([0.0]), np.asarray([0.0])),
        grid,
        bandwidth_km=75.0,
        model_id="R30_COMPONENT",
    )
    np.testing.assert_array_equal(recent_underflow.cell_relative_mass, np.asarray([1.0, 0.0]))
    with pytest.raises(DevelopmentPredictionError, match="zero numerical mass"):
        development_predict._target_masses(recent_underflow, window, grid)
    assert development_predict._target_masses(
        recent_underflow,
        window,
        grid,
        allow_numerical_zero=True,
    ) == (0.0,)

    long_surface = l2_gaussian_kde_relative_mass(
        CausalSpatialHistory(np.asarray([0.0, 4_000.0]), np.asarray([0.0, 0.0])),
        grid,
        bandwidth_km=75.0,
    )
    mixed = development_predict._l3_from_cached_surfaces(
        long_surface,
        recent_underflow,
        recent_event_count=1,
        alpha=0.75,
    )
    np.testing.assert_array_equal(mixed.cell_relative_mass, np.asarray([0.875, 0.125]))
    assert mixed.cell_relative_mass[1] > 0.0


def test_t1_m6_plus_7d_shares_m5_6_k_or_uses_poisson_fallback() -> None:
    overdispersed = (0, 0, 1, 9, 0, 10)
    common_means = (20.0 / 6.0,) * 6
    series = (
        InnerTimeCountSeries("m5_6", overdispersed, common_means),
        InnerTimeCountSeries("m6_plus", (0, 0, 0, 7, 0, 8), (2.5,) * 6),
        InnerTimeCountSeries("m5_plus_1970_for_joint", overdispersed, common_means),
    )

    fitted = fit_t1_qualifications_from_inner_count_series(series, horizon_days=7)
    by_band = {item.band: item.qualification for item in fitted}

    assert by_band["m5_6"].status == "evaluable"
    assert by_band["m6_plus"].status == "evaluable"
    assert by_band["m6_plus"].dispersion_k == by_band["m5_6"].dispersion_k
    assert by_band["m6_plus"].reason.startswith("shared_M5_6")
    assert by_band["m6_plus"].sample_mean_count == pytest.approx(2.5)
    assert by_band["m6_plus"].sample_variance_count == pytest.approx(15.1)
    assert by_band["m6_plus"].sample_mean_count != by_band["m5_6"].sample_mean_count

    poisson_source = (
        InnerTimeCountSeries("m5_6", (1, 1, 1), (1.0, 1.0, 1.0)),
        InnerTimeCountSeries("m6_plus", (0, 0, 9), (3.0, 3.0, 3.0)),
        InnerTimeCountSeries("m5_plus_1970_for_joint", (1, 1, 1), (1.0, 1.0, 1.0)),
    )
    fallback = fit_t1_qualifications_from_inner_count_series(poisson_source, horizon_days=7)
    fallback_by_band = {item.band: item.qualification for item in fallback}
    assert fallback_by_band["m6_plus"].status == "poisson_limit"
    assert fallback_by_band["m6_plus"].dispersion_k is None
    assert "no_independent_k_fit" in fallback_by_band["m6_plus"].reason

    mismatched_axis = (
        InnerTimeCountSeries("m5_6", overdispersed, common_means),
        InnerTimeCountSeries("m6_plus", (0, 0, 7), (2.5,) * 3),
        InnerTimeCountSeries("m5_plus_1970_for_joint", overdispersed, common_means),
    )
    with pytest.raises(DevelopmentPredictionError, match="common non-overlapping exposure axis"):
        fit_t1_qualifications_from_inner_count_series(mismatched_axis, horizon_days=7)


def test_inner_time_series_pools_three_blocks_with_matching_elapsed_t0_means() -> None:
    catalog = _selection_catalog()

    series = build_strictly_earlier_inner_time_count_series(
        catalog,
        _inner_exposures(),
        block_end_by_id=_block_ends(),
    )

    assert tuple(item.band for item in series) == TIME_BANDS
    by_band = {item.band: item for item in series}
    assert by_band["m5_6"].counts == (1, 1, 1)
    assert by_band["m6_plus"].counts == (1, 1, 1)
    assert by_band["m5_plus_1970_for_joint"].counts == (2, 2, 2)
    assert all(value > 0.0 for item in series for value in item.poisson_expected_counts)
    assert by_band["m5_plus_1970_for_joint"].poisson_expected_counts[1] > 0.0


def test_primary_prediction_and_npz_schema_are_target_free_and_future_invariant() -> None:
    catalog = _selection_catalog()
    catalog_with_future = _selection_catalog(future_event=True)
    location = _location_selection(catalog)
    t1 = fit_t1_qualifications_from_inner_count_series(
        (
            InnerTimeCountSeries("m5_6", (1, 1, 1), (1.0, 1.0, 1.0)),
            InnerTimeCountSeries("m6_plus", (0, 0, 0), (0.2, 0.2, 0.2)),
            InnerTimeCountSeries("m5_plus_1970_for_joint", (1, 1, 1), (1.2, 1.2, 1.2)),
        ),
        horizon_days=7,
    )
    selection = FoldHorizonParameterSelection(FOLD, 7, location, t1)
    issue = datetime(2000, 1, 6, tzinfo=UTC)

    prediction = build_primary_issue_prediction(
        catalog,
        _one_cell_grid(),
        selection,
        issue_time_utc=issue,
    )
    future_prediction = build_primary_issue_prediction(
        catalog_with_future,
        _one_cell_grid(),
        selection,
        issue_time_utc=issue,
    )

    assert tuple(item.model_id for item in prediction.location_surfaces) == LOCATION_MODEL_IDS
    assert tuple(item.band for item in prediction.count_forecasts) == TIME_BANDS
    for first, second in zip(
        prediction.location_surfaces, future_prediction.location_surfaces, strict=True
    ):
        np.testing.assert_array_equal(first.cell_relative_mass, second.cell_relative_mass)
    assert tuple(item.poisson_expected_count for item in prediction.count_forecasts) == tuple(
        item.poisson_expected_count for item in future_prediction.count_forecasts
    )

    magnitude = build_weekly_magnitude_snapshot(catalog, fold_id=FOLD, issue_time_utc=issue)
    weekly_axis, primary_axis = development_predict._frozen_fold_issue_axes(FOLD)
    primary_predictions = tuple(
        replace(prediction, horizon_days=horizon, issue_time_utc=primary_issue)
        for horizon, primary_issue in primary_axis
    )
    magnitude_snapshots = tuple(
        replace(magnitude, issue_time_utc=weekly_issue) for weekly_issue in weekly_axis
    )
    fold = DevelopmentFoldPrediction(FOLD, primary_predictions, magnitude_snapshots)
    arrays = fold_prediction_npz_arrays(fold, cell_count=1)

    assert not any("target" in key.lower() or "score" in key.lower() for key in arrays)
    assert arrays["primary_fold_index"].tolist() == [0] * 99
    assert arrays["primary_horizon_days"].shape == (99,)
    assert arrays["location_model_index"].tolist() == [0, 1, 2, 3, 4]
    assert arrays["time_band_index"].tolist() == [0, 1, 2]
    assert arrays["location_regional_tau_years"].shape == (99,)
    assert set(arrays["location_regional_tau_years"].tolist()) == {10.0}
    assert arrays["location_relative_mass"].shape == (99, 5, 1)
    assert arrays["t0_expected_count"].shape == (99, 3)
    assert arrays["m0_bin_probability_mass"].shape == (261, 55)
    assert arrays["m3_bin_probability_mass"].shape == (261, 45)
    assert arrays["t1_reason_code"].shape == (99, 3)
    assert arrays["t1_historical_block_count"].shape == (99, 3)
    assert arrays["t1_observed_information_k_applicable"].shape == (99, 3)
    assert arrays["location_relative_mass"].dtype == np.dtype("float64")
    assert arrays["primary_issue_time_us"].dtype == np.dtype("int64")
    assert all(not value.flags.writeable for value in arrays.values())

    bad_source_counts = {key: np.array(value, copy=True) for key, value in arrays.items()}
    bad_source_counts["location_source_event_count"][0, 1] += 1
    with pytest.raises(DevelopmentPredictionError, match="source counts"):
        validate_frozen_fold_prediction_npz_arrays(FOLD, bad_source_counts, cell_count=1)

    bad_tau = {key: np.array(value, copy=True) for key, value in arrays.items()}
    bad_tau["location_regional_tau_years"][0] = 2.0
    with pytest.raises(DevelopmentPredictionError, match="regional tau"):
        validate_frozen_fold_prediction_npz_arrays(FOLD, bad_tau, cell_count=1)

    bad_bandwidth = {key: np.array(value, copy=True) for key, value in arrays.items()}
    bad_bandwidth["location_bandwidth_km"][0, 2] = 125.0
    bad_bandwidth["location_bandwidth_km"][0, 4] = 125.0
    with pytest.raises(DevelopmentPredictionError, match="bandwidth"):
        validate_frozen_fold_prediction_npz_arrays(FOLD, bad_bandwidth, cell_count=1)

    bad_partition = {key: np.array(value, copy=True) for key, value in arrays.items()}
    bad_partition["t0_expected_count"][0, 2] += 1.0
    with pytest.raises(DevelopmentPredictionError, match=r"M5\+"):
        validate_frozen_fold_prediction_npz_arrays(FOLD, bad_partition, cell_count=1)


def test_prediction_boundaries_reject_non_development_fold_and_bad_inner_embargo() -> None:
    issue = datetime(2000, 1, 6, tzinfo=UTC)
    catalog = _catalog([_row("event", issue + timedelta(days=1), 5.0)])
    with pytest.raises(DevelopmentPredictionError, match="only the four"):
        build_strictly_earlier_inner_m4_location_block(
            catalog,
            InnerExposure("C_HOLDOUT_2020_2022", "I1", 7, (issue,)),
            block_end_utc=issue + timedelta(days=7),
            locate_lonlat=lambda _longitude, _latitude: 0,
        )

    valid_catalog = _selection_catalog()
    exposures = _inner_exposures()
    ends = _block_ends()
    blocks = tuple(
        build_strictly_earlier_inner_m4_location_block(
            valid_catalog,
            exposure,
            block_end_utc=ends[exposure.block_id],
            locate_lonlat=lambda _longitude, _latitude: 0,
        )
        for exposure in exposures
    )
    with pytest.raises(ValueError, match="at least 30 days"):
        select_location_parameters_from_strictly_earlier_inner_blocks(
            valid_catalog,
            _one_cell_grid(),
            blocks,
            outer_start_utc=ends["I3"] + timedelta(days=30) - timedelta(microseconds=1),
        )
