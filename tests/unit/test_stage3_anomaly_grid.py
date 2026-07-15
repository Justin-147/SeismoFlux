from __future__ import annotations

import numpy as np
from shapely.geometry import box

from seismoflux.features.anomaly.grid import build_stage3_query_grid


def test_stage3_query_grid_is_deterministic_and_target_independent() -> None:
    study_area = box(104.8, 34.8, 105.2, 35.2)

    first = build_stage3_query_grid(study_area)
    second = build_stage3_query_grid(study_area)

    assert first.grid_id == second.grid_id
    assert first.cell_ids == second.cell_ids
    assert first.cell_count > 0
    assert tuple(zip(first.rows, first.columns, strict=True)) == tuple(
        sorted(zip(first.rows, first.columns, strict=True))
    )
    np.testing.assert_array_equal(first.query_xy_m, second.query_xy_m)
    np.testing.assert_array_equal(first.clipped_area_km2, second.clipped_area_km2)
    assert np.all(first.clipped_area_km2 > 0.0)
    assert np.all(first.clipped_area_km2 <= 25.0 * 25.0)


def test_grid_identity_changes_when_the_study_area_changes() -> None:
    first = build_stage3_query_grid(box(104.8, 34.8, 105.2, 35.2))
    second = build_stage3_query_grid(box(104.8, 34.8, 105.3, 35.2))

    assert first.grid_id != second.grid_id
