from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from seismoflux.features.anomaly.spatial import (
    SpatialEntityArrays,
    _compute_spatial_features_dense_reference,
    compute_spatial_features,
)


def _entities(
    xy_m: NDArray[np.float64],
    *,
    listed: NDArray[np.bool_] | None = None,
    source_new: NDArray[np.bool_] | None = None,
    first_seen: NDArray[np.bool_] | None = None,
    explicit_end: NDArray[np.bool_] | None = None,
    not_continued: NDArray[np.bool_] | None = None,
    relisted: NDArray[np.bool_] | None = None,
    right_censored: NDArray[np.bool_] | None = None,
    reliability_high: NDArray[np.bool_] | None = None,
    reliability_cautious: NDArray[np.bool_] | None = None,
    station_id: NDArray[np.object_] | None = None,
    measurement_id: NDArray[np.object_] | None = None,
    discipline_code: NDArray[np.int64] | None = None,
    age_days: NDArray[np.float64] | None = None,
    known_duration_days: NDArray[np.float64] | None = None,
    left_truncated: NDArray[np.bool_] | None = None,
    late_entry: NDArray[np.bool_] | None = None,
    temporary: NDArray[np.bool_] | None = None,
    multidisciplinary: NDArray[np.bool_] | None = None,
    discipline_membership: NDArray[np.bool_] | None = None,
    reporting_listed: NDArray[np.bool_] | None = None,
) -> SpatialEntityArrays:
    size = xy_m.shape[0]
    false = np.zeros(size, dtype=np.bool_)
    resolved_high = np.ones(size, dtype=np.bool_) if reliability_high is None else reliability_high
    return SpatialEntityArrays(
        xy_m=xy_m,
        listed=np.ones(size, dtype=np.bool_) if listed is None else listed,
        source_new=false if source_new is None else source_new,
        first_seen=false if first_seen is None else first_seen,
        explicit_end=false if explicit_end is None else explicit_end,
        not_continued=false if not_continued is None else not_continued,
        relisted=false if relisted is None else relisted,
        right_censored=false if right_censored is None else right_censored,
        reliability_high=resolved_high,
        reliability_cautious=(
            ~resolved_high if reliability_cautious is None else reliability_cautious
        ),
        station_id=(
            np.asarray([f"station-{index}" for index in range(size)], dtype=object)
            if station_id is None
            else station_id
        ),
        measurement_id=(
            np.asarray([f"measurement-{index}" for index in range(size)], dtype=object)
            if measurement_id is None
            else measurement_id
        ),
        discipline_code=(
            np.zeros(size, dtype=np.int64) if discipline_code is None else discipline_code
        ),
        age_days=np.ones(size, dtype=np.float64) if age_days is None else age_days,
        known_duration_days=(
            np.full(size, np.nan, dtype=np.float64)
            if known_duration_days is None
            else known_duration_days
        ),
        left_truncated=left_truncated,
        late_entry=late_entry,
        temporary=temporary,
        multidisciplinary=multidisciplinary,
        discipline_membership=discipline_membership,
        reporting_listed=reporting_listed,
    )


def test_radius_is_a_closed_ball_at_boundary_plus_or_minus_epsilon() -> None:
    epsilon_m = 0.01
    entities = _entities(
        np.asarray(
            [
                [50_000.0 - epsilon_m, 0.0],
                [50_000.0, 0.0],
                [50_000.0 + epsilon_m, 0.0],
            ],
            dtype=np.float64,
        )
    )

    result = compute_spatial_features(np.asarray([[0.0, 0.0]]), entities)

    assert result.radius_features["listed_count"][0, 0] == 2.0


def test_radius_counts_are_nested_across_frozen_scales() -> None:
    entities = _entities(
        np.asarray(
            [[25_000.0, 0.0], [75_000.0, 0.0], [150_000.0, 0.0], [450_000.0, 0.0]],
            dtype=np.float64,
        )
    )

    counts = compute_spatial_features(np.asarray([[0.0, 0.0]]), entities).radius_features[
        "listed_count"
    ][0]

    assert np.all(np.diff(counts) >= 0.0)
    assert counts.tolist() == [1.0, 2.0, 3.0, 3.0, 4.0]


def test_gaussian_kernel_is_finite_monotone_and_zero_beyond_three_sigma() -> None:
    entities = _entities(np.asarray([[0.0, 0.0]], dtype=np.float64))
    queries = np.asarray(
        [[0.0, 0.0], [25_000.0, 0.0], [50_000.0, 0.0], [150_000.0, 0.0], [150_000.01, 0.0]],
        dtype=np.float64,
    )

    weights = compute_spatial_features(queries, entities).gaussian_features["listed_count"][:, 0]

    assert np.isfinite(weights).all()
    assert np.all(np.diff(weights[:4]) < 0.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[3] == pytest.approx(np.exp(-4.5))
    assert weights[4] == 0.0


def test_core_coverage_discipline_age_and_geometry_features_are_auditable() -> None:
    xy_m = np.asarray(
        [
            [-15_000.0, 0.0],
            [-5_000.0, 0.0],
            [5_000.0, 0.0],
            [15_000.0, 0.0],
            [np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    entities = _entities(
        xy_m,
        source_new=np.asarray([True, True, False, False, True]),
        first_seen=np.asarray([True, False, False, False, True]),
        explicit_end=np.asarray([False, False, True, False, False]),
        not_continued=np.asarray([False, False, False, True, False]),
        relisted=np.asarray([False, True, False, False, False]),
        right_censored=np.asarray([True, True, False, True, True]),
        reliability_high=np.asarray([True, False, True, False, True]),
        station_id=np.asarray(["s1", "s1", "s2", "s3", "s4"], dtype=object),
        measurement_id=np.asarray(["m1", "m2", "m1", "m3", "m4"], dtype=object),
        discipline_code=np.asarray([0, 1, 2, 3, 0], dtype=np.int64),
        age_days=np.asarray([10.0, 20.0, np.nan, 40.0, 50.0]),
        known_duration_days=np.asarray([np.nan, 5.0, 6.0, np.nan, 8.0]),
    )

    result = compute_spatial_features(np.asarray([[0.0, 0.0]]), entities)
    features = result.radius_features

    assert result.input_entity_count == 5
    assert result.spatial_entity_count == 4
    assert result.missing_coordinate_count == 1
    assert features["listed_count"][0, 0] == 4.0
    assert features["listed_weighted_count"][0, 0] == 3.0
    assert features["source_new_count"][0, 0] == 2.0
    assert features["source_new_weighted_count"][0, 0] == 1.5
    assert features["high_reliability_listed_count"][0, 0] == 2.0
    assert features["cautious_reliability_listed_count"][0, 0] == 2.0
    assert features["excluded_reliability_listed_count"][0, 0] == 0.0
    assert features["spatial_weight_mass"][0, 0] == 4.0
    assert features["unique_station_count"][0, 0] == 3.0
    assert features["unique_measurement_count"][0, 0] == 3.0
    assert features["unique_station_measurement_count"][0, 0] == 4.0
    assert features["discipline_count"][0, 0] == 4.0
    assert features["discipline_shannon_normalized"][0, 0] == pytest.approx(1.0)
    assert features["age_known_count"][0, 0] == 3.0
    assert features["age_mean_days"][0, 0] == pytest.approx(70.0 / 3.0)
    assert features["known_duration_count"][0, 0] == 2.0
    assert features["known_duration_mean_days"][0, 0] == pytest.approx(5.5)
    assert features["principal_direction_deg"][0, 0] == pytest.approx(0.0)
    assert features["anisotropy"][0, 0] == pytest.approx(1.0)
    assert 0.0 <= features["concentration"][0, 0] <= 1.0


def test_entity_in_neither_reliability_tier_has_fixed_zero_weight() -> None:
    entities = _entities(
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        reliability_high=np.asarray([False]),
        reliability_cautious=np.asarray([False]),
        source_new=np.asarray([True]),
    )

    features = compute_spatial_features(np.asarray([[0.0, 0.0]]), entities).radius_features

    assert features["listed_count"][0, 0] == 1.0
    assert features["listed_weighted_count"][0, 0] == 0.0
    assert features["source_new_weighted_count"][0, 0] == 0.0
    assert features["excluded_reliability_listed_count"][0, 0] == 1.0


def test_raw_reporting_proxy_can_include_temporary_rows_without_changing_entity_count() -> None:
    entities = _entities(
        np.asarray([[0.0, 0.0], [1_000.0, 0.0]], dtype=np.float64),
        listed=np.asarray([True, False]),
        reporting_listed=np.asarray([True, True]),
        station_id=np.asarray(["complete", "temporary"], dtype=object),
        measurement_id=np.asarray(["m1", "m2"], dtype=object),
    )

    features = compute_spatial_features(np.asarray([[0.0, 0.0]]), entities).radius_features

    assert features["listed_count"][0, 0] == 1.0
    assert features["unique_station_count"][0, 0] == 2.0
    assert features["unique_measurement_count"][0, 0] == 2.0


def test_left_truncation_temporary_and_multidisciplinary_features_remain_separate() -> None:
    entities = _entities(
        np.asarray([[0.0, 0.0], [1_000.0, 0.0], [2_000.0, 0.0]], dtype=np.float64),
        listed=np.asarray([True, True, False]),
        first_seen=np.asarray([False, True, False]),
        right_censored=np.asarray([True, True, False]),
        left_truncated=np.asarray([True, False, False]),
        late_entry=np.asarray([False, True, False]),
        temporary=np.asarray([False, False, True]),
        multidisciplinary=np.asarray([True, False, False]),
        discipline_membership=np.asarray(
            [[True, True, False, False], [True, False, False, False], [True, False, False, False]],
            dtype=np.bool_,
        ),
        reliability_high=np.asarray([True, True, False]),
        reliability_cautious=np.asarray([False, False, False]),
    )

    features = compute_spatial_features(np.asarray([[0.0, 0.0]]), entities).radius_features

    assert features["listed_count"][0, 0] == 2.0
    assert features["temporary_entity_count"][0, 0] == 1.0
    assert features["temporary_entity_weighted_count"][0, 0] == 0.0
    assert features["left_truncated_count"][0, 0] == 1.0
    assert features["late_entry_count"][0, 0] == 1.0
    assert features["first_seen_rate"][0, 0] == pytest.approx(1.0)
    assert features["right_censored_fraction"][0, 0] == pytest.approx(1.0)
    assert features["multidisciplinary_entity_fraction"][0, 0] == pytest.approx(0.5)
    assert features["discipline_count"][0, 0] == 2.0
    assert features["discipline_deformation_count"][0, 0] == 2.0
    assert features["discipline_fluid_count"][0, 0] == 1.0


def test_anomaly_outside_maximum_kernel_support_cannot_change_local_features() -> None:
    local_only = _entities(np.asarray([[0.0, 0.0]], dtype=np.float64))
    separated = _entities(np.asarray([[0.0, 0.0], [4_000_000.0, 0.0]], dtype=np.float64))
    local_query = np.asarray([[0.0, 0.0]], dtype=np.float64)

    local_result = compute_spatial_features(local_query, local_only)
    separated_result = compute_spatial_features(local_query, separated)

    for feature_name, expected in local_result.radius_features.items():
        np.testing.assert_allclose(
            separated_result.radius_features[feature_name],
            expected,
            equal_nan=True,
        )
    for feature_name, expected in local_result.gaussian_features.items():
        np.testing.assert_allclose(
            separated_result.gaussian_features[feature_name],
            expected,
            equal_nan=True,
        )


def _random_entities(seed: int, size: int) -> SpatialEntityArrays:
    generator = np.random.default_rng(seed)
    xy_m = generator.uniform(-1_600_000.0, 1_600_000.0, size=(size, 2))
    xy_m[[3, 17]] = np.nan
    high = generator.random(size) < 0.55
    cautious = ~high & (generator.random(size) < 0.5)
    discipline_code = generator.integers(0, 4, size=size, dtype=np.int64)
    membership = np.zeros((size, 4), dtype=np.bool_)
    membership[np.arange(size), discipline_code] = True
    membership[::11, (discipline_code[::11] + 1) % 4] = True
    age_days = generator.uniform(0.0, 1_000.0, size=size)
    age_days[generator.random(size) < 0.2] = np.nan
    duration_days = generator.uniform(0.0, 400.0, size=size)
    duration_days[generator.random(size) < 0.4] = np.nan

    def flags(probability: float) -> NDArray[np.bool_]:
        return np.asarray(generator.random(size) < probability, dtype=np.bool_)

    return SpatialEntityArrays(
        xy_m=xy_m,
        listed=flags(0.7),
        source_new=flags(0.15),
        first_seen=flags(0.1),
        explicit_end=flags(0.08),
        not_continued=flags(0.06),
        relisted=flags(0.04),
        right_censored=flags(0.5),
        reliability_high=high,
        reliability_cautious=cautious,
        station_id=np.asarray(
            [None if index % 19 == 0 else f"station-{index % 17}" for index in range(size)],
            dtype=object,
        ),
        measurement_id=np.asarray(
            [" " if index % 23 == 0 else f"measurement-{index % 9}" for index in range(size)],
            dtype=object,
        ),
        discipline_code=discipline_code,
        age_days=age_days,
        known_duration_days=duration_days,
        left_truncated=flags(0.1),
        late_entry=flags(0.1),
        temporary=flags(0.05),
        multidisciplinary=np.sum(membership, axis=1) > 1,
        discipline_membership=membership,
        reporting_listed=flags(0.8),
    )


@pytest.mark.parametrize("query_chunk_size", [1, 7, 64])
def test_sparse_backend_matches_dense_reference_on_randomized_all_fields(
    query_chunk_size: int,
) -> None:
    generator = np.random.default_rng(90210)
    queries = generator.uniform(-1_400_000.0, 1_400_000.0, size=(19, 2))
    entities = _random_entities(147, 97)

    expected = _compute_spatial_features_dense_reference(
        queries,
        entities,
        query_chunk_size=8,
    )
    observed = compute_spatial_features(
        queries,
        entities,
        query_chunk_size=query_chunk_size,
    )

    assert observed.radius_features.keys() == expected.radius_features.keys()
    assert observed.gaussian_features.keys() == expected.gaussian_features.keys()
    for name, expected_values in expected.radius_features.items():
        np.testing.assert_allclose(
            observed.radius_features[name],
            expected_values,
            rtol=2e-12,
            atol=1e-9,
            equal_nan=True,
        )
    for name, expected_values in expected.gaussian_features.items():
        np.testing.assert_allclose(
            observed.gaussian_features[name],
            expected_values,
            rtol=2e-12,
            atol=1e-9,
            equal_nan=True,
        )


def test_unique_identifier_gaussian_uses_nearest_maximum_weight_not_sum() -> None:
    entities = _entities(
        np.asarray([[0.0, 0.0], [50_000.0, 0.0], [25_000.0, 0.0]], dtype=np.float64),
        station_id=np.asarray(["same", "same", "other"], dtype=object),
        measurement_id=np.asarray(["m", "m", "m"], dtype=object),
        reporting_listed=np.ones(3, dtype=np.bool_),
    )

    features = compute_spatial_features(np.asarray([[0.0, 0.0]]), entities)

    assert features.radius_features["unique_station_count"][0, 0] == 3.0 - 1.0
    assert features.gaussian_features["unique_station_count"][0, 0] == pytest.approx(
        1.0 + np.exp(-0.5 * (25_000.0 / 50_000.0) ** 2)
    )
    assert features.gaussian_features["unique_measurement_count"][0, 0] == 1.0


def test_sparse_candidate_search_handles_a_chunk_with_no_neighbours() -> None:
    entities = _entities(np.asarray([[4_000_000.0, 4_000_000.0]], dtype=np.float64))

    result = compute_spatial_features(
        np.asarray([[0.0, 0.0], [100.0, 100.0]], dtype=np.float64),
        entities,
        query_chunk_size=2,
    )

    assert np.count_nonzero(result.radius_features["listed_count"]) == 0
    assert np.count_nonzero(result.gaussian_features["listed_count"]) == 0
    assert np.isnan(result.radius_features["age_mean_days"]).all()
