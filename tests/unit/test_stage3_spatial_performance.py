from __future__ import annotations

from time import perf_counter

import numpy as np

from seismoflux.features.anomaly.spatial import (
    SpatialEntityArrays,
    _compute_spatial_features_dense_reference,
    compute_spatial_features,
)


def _benchmark_entities(seed: int, size: int) -> SpatialEntityArrays:
    generator = np.random.default_rng(seed)
    false = np.zeros(size, dtype=np.bool_)
    return SpatialEntityArrays(
        xy_m=generator.uniform(-2_000_000.0, 2_000_000.0, size=(size, 2)),
        listed=generator.random(size) < 0.7,
        source_new=generator.random(size) < 0.1,
        first_seen=generator.random(size) < 0.1,
        explicit_end=generator.random(size) < 0.05,
        not_continued=generator.random(size) < 0.05,
        relisted=generator.random(size) < 0.03,
        right_censored=generator.random(size) < 0.6,
        reliability_high=generator.random(size) < 0.65,
        reliability_cautious=false,
        station_id=np.asarray([f"station-{index % 200}" for index in range(size)], dtype=object),
        measurement_id=np.asarray(
            [f"measurement-{index % 60}" for index in range(size)],
            dtype=object,
        ),
        discipline_code=generator.integers(0, 4, size=size, dtype=np.int64),
        age_days=generator.uniform(0.0, 500.0, size=size),
        known_duration_days=generator.uniform(0.0, 200.0, size=size),
    )


def test_sparse_backend_small_scale_benchmark_is_materially_faster() -> None:
    generator = np.random.default_rng(147)
    queries = generator.uniform(-2_000_000.0, 2_000_000.0, size=(256, 2))
    entities = _benchmark_entities(90210, 800)
    compute_spatial_features(queries[:8], entities, query_chunk_size=64)

    started = perf_counter()
    expected = _compute_spatial_features_dense_reference(
        queries,
        entities,
        query_chunk_size=64,
    )
    dense_seconds = perf_counter() - started
    started = perf_counter()
    observed = compute_spatial_features(queries, entities, query_chunk_size=64)
    sparse_seconds = perf_counter() - started

    np.testing.assert_allclose(
        observed.gaussian_features["listed_count"],
        expected.gaussian_features["listed_count"],
        rtol=2e-12,
        atol=1e-12,
    )
    assert sparse_seconds * 2.0 < dense_seconds
