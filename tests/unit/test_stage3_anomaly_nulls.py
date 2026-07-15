from __future__ import annotations

import numpy as np

from seismoflux.features.anomaly.nulls import (
    NullReasonCode,
    spatial_null_reason_codes,
    trajectory_null_reason_codes,
)
from seismoflux.features.anomaly.trajectory import compute_latest_trajectory_features


def test_spatial_null_reasons_distinguish_empty_from_undefined_direction() -> None:
    features = {
        "principal_direction_deg": np.asarray([[np.nan], [np.nan]], dtype=np.float64),
        "listed_count": np.asarray([[0.0], [1.0]], dtype=np.float64),
    }

    reasons = spatial_null_reason_codes("principal_direction_deg", features)

    assert reasons[:, 0].tolist() == [
        int(NullReasonCode.NO_ELIGIBLE_ENTITY),
        int(NullReasonCode.DIRECTION_UNDEFINED),
    ]


def test_trajectory_null_reasons_cover_insufficient_history_and_zero_variance() -> None:
    times = np.asarray(
        ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
        dtype="datetime64[ns]",
    )
    values = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 0.0, 3.0],
            [1.0, 0.0, np.nan],
        ],
        dtype=np.float64,
    )
    result = compute_latest_trajectory_features(times, values)

    surge_reasons = trajectory_null_reason_codes(result, "surge_z_13w")
    ratio_reasons = trajectory_null_reason_codes(result, "peak_ratio_52w")

    assert surge_reasons.tolist() == [
        int(NullReasonCode.ZERO_OR_UNDEFINED_BASELINE_VARIANCE),
        int(NullReasonCode.ZERO_OR_UNDEFINED_BASELINE_VARIANCE),
        int(NullReasonCode.CURRENT_VALUE_UNAVAILABLE),
    ]
    assert ratio_reasons.tolist() == [
        int(NullReasonCode.VALID),
        int(NullReasonCode.ZERO_CAUSAL_PEAK),
        int(NullReasonCode.CURRENT_VALUE_UNAVAILABLE),
    ]
