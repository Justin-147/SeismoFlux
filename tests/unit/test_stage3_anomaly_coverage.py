from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np

from seismoflux.features.anomaly.coverage import (
    CoverageEntityBatch,
    local_coverage_features,
    select_trailing_coverage_batches,
    trailing_coverage_entity_arrays,
)
from seismoflux.features.anomaly.nulls import NullReasonCode
from seismoflux.features.anomaly.spatial import (
    SpatialFeatureResult,
    compute_spatial_features,
)


def _result(stations: float, measurements: float) -> SpatialFeatureResult:
    shape = (2, 5)
    features = {
        "unique_station_count": np.full(shape, stations, dtype=np.float64),
        "unique_measurement_count": np.full(shape, measurements, dtype=np.float64),
    }
    return SpatialFeatureResult(
        query_xy_m=np.zeros((2, 2), dtype=np.float64),
        scales_km=np.asarray([50.0, 100.0, 200.0, 300.0, 500.0]),
        radius_features=features,
        gaussian_features={key: value.copy() for key, value in features.items()},
        input_entity_count=0,
        spatial_entity_count=0,
        missing_coordinate_count=0,
    )


def test_local_zero_denominator_does_not_mask_unrelated_cells() -> None:
    current = _result(1.0, 1.0)
    trailing = _result(2.0, 4.0)
    trailing.radius_features["unique_station_count"][0, 0] = 0.0
    trailing.gaussian_features["unique_station_count"][0, 0] = 0.0

    features = local_coverage_features(current, trailing)
    station = features.radius["current_to_trailing_station_reporting_coverage_proxy"]
    reason = features.radius_null_reason["current_to_trailing_station_reporting_coverage_proxy"]

    assert np.isnan(station[0, 0])
    assert reason[0, 0] == int(NullReasonCode.ZERO_DENOMINATOR)
    assert station[1, 0] == 0.5
    assert reason[1, 0] == int(NullReasonCode.VALID)


def test_trailing_window_is_closed_and_causal() -> None:
    issue = datetime(2025, 1, 1, 16, tzinfo=UTC)

    def batch(offset_days: int) -> CoverageEntityBatch:
        return CoverageEntityBatch(
            issue_time_utc=issue + timedelta(days=offset_days),
            xy_m=np.empty((0, 2), dtype=np.float64),
            station_id=np.empty(0, dtype=object),
            measurement_id=np.empty(0, dtype=object),
            discipline_code=np.empty(0, dtype=np.int64),
            discipline_membership=np.empty((0, 4), dtype=np.bool_),
        )

    selected = select_trailing_coverage_batches(
        (batch(-365), batch(-364), batch(0), batch(1)),
        issue,
    )

    assert tuple(item.issue_time_utc for item in selected) == (
        issue - timedelta(days=364),
        issue,
    )


def test_trailing_unique_only_fast_path_matches_full_spatial_unique_semantics() -> None:
    issue = datetime(2025, 1, 1, 16, tzinfo=UTC)
    xy_m = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [75_000.0, 0.0], [150_000.0, 0.0]],
        dtype=np.float64,
    )
    batch = CoverageEntityBatch(
        issue_time_utc=issue,
        xy_m=xy_m,
        station_id=np.asarray(["s1", "s1", "s1", "s2"], dtype=object),
        measurement_id=np.asarray(["m1", "m1", "m2", "m1"], dtype=object),
        discipline_code=np.zeros(4, dtype=np.int64),
        discipline_membership=np.tile(
            np.asarray([[True, False, False, False]], dtype=np.bool_),
            (4, 1),
        ),
    )
    arrays = trailing_coverage_entity_arrays((batch, batch))
    queries = np.asarray([[0.0, 0.0], [100_000.0, 0.0]], dtype=np.float64)

    fast = compute_spatial_features(queries, arrays, query_chunk_size=1)
    full = compute_spatial_features(
        queries,
        replace(arrays, coverage_unique_only=False),
        query_chunk_size=1,
    )

    assert arrays.coverage_unique_only is True
    assert fast.input_entity_count == 8
    assert fast.spatial_entity_count == 8
    assert set(fast.radius_features) == {
        "unique_station_count",
        "unique_measurement_count",
    }
    assert set(fast.gaussian_features) == set(fast.radius_features)
    for name in fast.radius_features:
        np.testing.assert_allclose(fast.radius_features[name], full.radius_features[name])
        np.testing.assert_allclose(fast.gaussian_features[name], full.gaussian_features[name])
