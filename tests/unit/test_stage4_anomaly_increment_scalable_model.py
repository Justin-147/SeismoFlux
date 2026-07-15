from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seismoflux.anomaly_increment.contracts import DesignMatrix
from seismoflux.anomaly_increment.integration import (
    composite_midpoint_quadrature,
    expand_midpoint_compensator_terms,
    lead_decay,
)
from seismoflux.anomaly_increment.model import (
    SharedPoissonObjective,
    fit_frozen_target_rate_head,
    fit_shared_ridge_poisson,
)
from seismoflux.anomaly_increment.scalable_model import (
    GroupedMidpointSharedPoissonObjective,
)


def _design(values: NDArray[np.float64]) -> DesignMatrix:
    return DesignMatrix(
        values=values,
        column_names=("x", "missing_x"),
        penalty_factors=np.ones(2, dtype=np.float64),
        active_coefficients=np.ones(2, dtype=np.bool_),
    )


def _objectives() -> tuple[SharedPoissonObjective, GroupedMidpointSharedPoissonObjective]:
    issue = _design(np.asarray([[0.2, 0.0], [-0.4, 1.0], [0.8, 0.0]], dtype=np.float64))
    event = _design(np.asarray([[0.2, 0.0], [0.8, 0.0]], dtype=np.float64))
    mass = np.asarray(
        [[0.2, 0.2], [0.3, 0.3], [0.5, 0.5]],
        dtype=np.float64,
    )
    rate_head = fit_frozen_target_rate_head(
        training_event_counts={"M5_6": 3, "M6_plus": 1},
        background_exposures={"M5_6": 7.0, "M6_plus": 7.0},
    )
    expanded = expand_midpoint_compensator_terms(
        issue_design=issue,
        background_spatial_mass_by_cell_and_bin=mass,
        horizon_days=7.0,
    )
    dense = SharedPoissonObjective(
        quadrature_design=expanded.design,
        quadrature_background_exposure_by_bin=expanded.background_exposure_by_bin,
        quadrature_decay=expanded.decay,
        event_design=event,
        event_background_intensity=np.asarray([0.2, 0.5], dtype=np.float64),
        event_decay=np.asarray(
            lead_decay(np.asarray([2.0, 6.0], dtype=np.float64)), dtype=np.float64
        ),
        event_magnitude_bin_ids=("M5_6", "M6_plus"),
        rate_head=rate_head,
    )
    quadrature = composite_midpoint_quadrature(7.0)
    grouped = GroupedMidpointSharedPoissonObjective(
        issue_design=issue,
        background_spatial_mass_by_row_and_bin=mass,
        midpoint_widths_days=quadrature.widths_days,
        midpoint_decays=np.asarray(lead_decay(quadrature.lead_midpoints_days), dtype=np.float64),
        event_design=event,
        event_background_intensity=np.asarray([0.2, 0.5], dtype=np.float64),
        event_decay=np.asarray(
            lead_decay(np.asarray([2.0, 6.0], dtype=np.float64)), dtype=np.float64
        ),
        event_magnitude_bin_ids=("M5_6", "M6_plus"),
        rate_head=rate_head,
    )
    return dense, grouped


def test_grouped_objective_is_float64_equivalent_to_dense_midpoint_expansion() -> None:
    dense, grouped = _objectives()
    beta = np.asarray([0.31, -0.17], dtype=np.float64)
    dense_value, dense_gradient = dense.value_and_gradient(beta)
    grouped_value, grouped_gradient = grouped.value_and_gradient(beta)

    np.testing.assert_allclose(
        grouped_value,
        dense_value,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(grouped_gradient, dense_gradient, rtol=2.0e-15, atol=2.0e-15)
    assert grouped.evaluate(beta).objective == grouped_value


def test_grouped_objective_uses_one_design_copy_and_shared_optimizer() -> None:
    dense, grouped = _objectives()
    assert dense.quadrature_design.row_count == (
        grouped.midpoint_widths_days.size * grouped.issue_design.row_count
    )
    result = fit_shared_ridge_poisson(grouped)
    assert result.converged
    assert result.beta.shape == (2,)
