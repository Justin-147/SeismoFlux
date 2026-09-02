from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from pyproj import Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s1.c2b_models import gaussian_log_masses
from seismoflux.multitask_s1.location import FrozenSpatialGrid
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_HISTORY_START_UTC,
    CatalogEventTable,
    catalog_event_table_from_frame,
)
from seismoflux.multitask_s3.catalog_background import (
    KERNEL_SCALES_KM,
    SPATIAL_WEIGHTS_BY_HORIZON,
    build_catalog_background,
    build_catalog_background_components,
)

# Synthetic issue instant; no actual report or future target is loaded by these tests.
ISSUE = datetime(2023, 7, 20, 4, tzinfo=UTC)
Q = ISSUE - timedelta(hours=24)


def _row(
    event_id: str,
    origin: datetime,
    magnitude: float,
    *,
    longitude: float = 105.0,
    available: datetime | None = None,
    inside: bool = True,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "origin_time_utc": origin,
        "available_at": origin if available is None else available,
        "longitude": longitude,
        "latitude": 35.0,
        "magnitude": magnitude,
        "inside_study_area": inside,
    }


def _catalog(rows: list[dict[str, object]]) -> CatalogEventTable:
    return catalog_event_table_from_frame(pd.DataFrame(rows))


def _grid() -> FrozenSpatialGrid:
    transformer = Transformer.from_crs(4326, EQUAL_AREA_CRS, always_xy=True)
    x, y = transformer.transform([104.5, 105.0, 105.5], [35.0, 35.0, 35.0])
    return FrozenSpatialGrid(
        np.asarray(x) / 1_000.0,
        np.asarray(y) / 1_000.0,
        np.array([100.0, 200.0, 300.0]),
    )


def test_catalog_parameters_match_the_frozen_s3_protocol() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/multitask_s3_anomaly.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    background = protocol["background"]
    assert list(KERNEL_SCALES_KM) == background["kernel_scales_km"]
    assert {
        horizon: list(weights) for horizon, weights in SPATIAL_WEIGHTS_BY_HORIZON.items()
    } == background["weights_by_horizon"]
    assert background["new_catalog_parameter_search"] is False


def test_future_and_not_yet_available_events_cannot_change_background() -> None:
    historical_rows = [
        _row("old", datetime(2001, 1, 1, tzinfo=UTC), 5.2, longitude=104.7),
        _row("recent", ISSUE - timedelta(days=4), 6.2, longitude=105.4),
    ]
    forbidden_for_this_issue = [
        _row("future", ISSUE + timedelta(days=7), 8.0, longitude=110.0),
        _row("after_Q", Q + timedelta(microseconds=1), 6.5, longitude=110.0),
        _row(
            "delayed",
            ISSUE - timedelta(days=20),
            7.0,
            available=Q + timedelta(microseconds=1),
        ),
    ]
    old = build_catalog_background(_catalog(historical_rows), _grid(), ISSUE, 30)
    appended = build_catalog_background(
        _catalog(historical_rows + forbidden_for_this_issue), _grid(), ISSUE, 30
    )
    np.testing.assert_array_equal(old.primary_log_mass, appended.primary_log_mass)
    np.testing.assert_array_equal(old.r30_reference_log_mass, appended.r30_reference_log_mass)
    assert old.expected_counts == appended.expected_counts
    assert old.waterlevel == appended.waterlevel


def test_d0_boundaries_band_additivity_and_old_2019_cutoff_is_not_reused() -> None:
    rows = [
        _row("pre1970", CATALOG_HISTORY_START_UTC - timedelta(microseconds=1), 5.0),
        _row("at1970", CATALOG_HISTORY_START_UTC, 5.0),
        _row("m4", datetime(2000, 1, 1, tzinfo=UTC), 4.0),
        _row("post2019", datetime(2021, 1, 1, tzinfo=UTC), 5.7),
        _row("at6", datetime(2022, 1, 1, tzinfo=UTC), 6.0),
        _row("outside", datetime(2022, 2, 1, tzinfo=UTC), 7.0, inside=False),
        _row("below4", ISSUE - timedelta(days=10), 3.99),
        _row("at_recent_lower", ISSUE - timedelta(days=30), 5.3),
        _row("inside_recent", ISSUE - timedelta(days=29), 6.1),
        _row("atQ", Q, 5.1),
    ]
    components = build_catalog_background_components(_catalog(rows), _grid(), ISSUE)
    water = components.waterlevel
    assert water.history_counts == {"Ms4_plus": 7, "Ms5_6": 4, "Ms6_plus": 2, "Ms5_plus": 6}
    assert water.recent_m4_event_count == 2
    assert water.availability_basis == "canonical_availability_only"
    assert water.recent_interval == "(T-30d,T-24h]"
    assert water.data_cutoff_utc == Q
    exposure = (Q - CATALOG_HISTORY_START_UTC).total_seconds() / 86_400.0
    assert water.history_exposure_days == exposure
    prediction = components.for_horizon(30)
    for band in ("Ms5_6", "Ms6_plus", "Ms5_plus"):
        assert prediction.expected_counts[band] == pytest.approx(
            30.0 * water.history_counts[band] / exposure
        )
    assert prediction.expected_counts["Ms5_plus"] == pytest.approx(
        prediction.expected_counts["Ms5_6"] + prediction.expected_counts["Ms6_plus"]
    )
    assert prediction.poisson_at_least_one["Ms5_plus"] == pytest.approx(
        -np.expm1(-prediction.expected_counts["Ms5_plus"])
    )


def test_fixed_mixtures_and_count_horizon_scaling() -> None:
    rows = [
        _row("old", datetime(1990, 1, 1, tzinfo=UTC), 5.5, longitude=104.5),
        _row("recent", ISSUE - timedelta(days=4), 6.1, longitude=105.5),
    ]
    components = build_catalog_background_components(_catalog(rows), _grid(), ISSUE)
    for horizon, weights in SPATIAL_WEIGHTS_BY_HORIZON.items():
        prediction = components.for_horizon(horizon)
        expected_mass = sum(
            weight * np.exp(components.kernel_log_masses[scale])
            for weight, scale in zip(weights, KERNEL_SCALES_KM, strict=True)
        )
        np.testing.assert_allclose(np.exp(prediction.primary_log_mass), expected_mass, atol=1e-14)
        assert np.exp(prediction.primary_log_mass).sum() == pytest.approx(1.0)
        for band, mean in prediction.expected_counts.items():
            assert mean / horizon == pytest.approx(components.poisson_rates[band].rate_per_day)


def test_empty_recent_uses_long_kde75_and_empty_d0_uses_area_uniform() -> None:
    grid = _grid()
    old_only = _catalog([_row("old", datetime(2000, 1, 1, tzinfo=UTC), 5.5)])
    components = build_catalog_background_components(old_only, grid, ISSUE)
    assert components.waterlevel.recent_m4_event_count == 0
    np.testing.assert_array_equal(
        components.r30_reference_log_mass, components.kernel_log_masses[75.0]
    )
    no_d0 = _catalog([_row("outside", datetime(2000, 1, 1, tzinfo=UTC), 7.0, inside=False)])
    prediction = build_catalog_background(no_d0, grid, ISSUE, 7)
    np.testing.assert_allclose(
        np.exp(prediction.primary_log_mass), grid.area_km2 / grid.total_area_km2
    )
    assert prediction.waterlevel.earliest_m4_origin_time_us is None
    assert prediction.waterlevel.latest_m4_origin_time_us is None
    assert prediction.waterlevel.latest_m4_available_at_us is None
    assert all(mean == 0.0 for mean in prediction.expected_counts.values())
    assert all(value == 0.0 for value in prediction.poisson_at_least_one.values())


def test_microsecond_and_nanosecond_input_frames_have_identical_causal_counts() -> None:
    frame = pd.DataFrame(
        [
            _row("atQ", Q, 5.1),
            _row("afterQ", Q + timedelta(microseconds=1), 6.0),
        ]
    )
    results = []
    for unit in ("us", "ns"):
        converted = frame.copy()
        for column in ("origin_time_utc", "available_at"):
            converted[column] = pd.to_datetime(converted[column], utc=True).astype(
                f"datetime64[{unit}, UTC]"
            )
        table = catalog_event_table_from_frame(converted)
        results.append(build_catalog_background(table, _grid(), ISSUE, 7))
    assert results[0].waterlevel.history_counts["Ms5_plus"] == 1
    assert results[0].waterlevel == results[1].waterlevel
    np.testing.assert_array_equal(results[0].primary_log_mass, results[1].primary_log_mass)


def test_components_reuse_kernels_across_horizons_and_return_read_only_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seismoflux.multitask_s3 import catalog_background as module

    original = gaussian_log_masses
    calls: list[tuple[float, ...]] = []

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["bandwidths_km"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "gaussian_log_masses", tracked)
    catalog = _catalog([_row("recent", ISSUE - timedelta(days=4), 5.5)])
    components = build_catalog_background_components(catalog, _grid(), ISSUE)
    predictions = [components.for_horizon(h) for h in SPATIAL_WEIGHTS_BY_HORIZON]
    assert calls == [(25.0, 75.0, 150.0), (75.0,)]
    for prediction in predictions:
        assert prediction.primary_log_mass.flags.writeable is False
        assert prediction.r30_reference_log_mass.flags.writeable is False


@pytest.mark.parametrize("bad_horizon", [0, 14, 30.0, True])
def test_unfrozen_horizons_are_rejected(bad_horizon: Any) -> None:
    catalog = _catalog([_row("old", datetime(2000, 1, 1, tzinfo=UTC), 5.5)])
    components = build_catalog_background_components(catalog, _grid(), ISSUE)
    with pytest.raises((TypeError, ValueError), match="horizon_days"):
        components.for_horizon(bad_horizon)


def test_naive_issue_and_nonpositive_history_exposure_are_rejected() -> None:
    catalog = _catalog([_row("old", datetime(2000, 1, 1, tzinfo=UTC), 5.5)])
    with pytest.raises(ValueError, match="timezone-aware"):
        build_catalog_background_components(catalog, _grid(), ISSUE.replace(tzinfo=None))
    with pytest.raises(ValueError, match="history start"):
        build_catalog_background_components(
            catalog, _grid(), CATALOG_HISTORY_START_UTC + timedelta(hours=24)
        )
