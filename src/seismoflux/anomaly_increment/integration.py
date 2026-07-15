"""Target-independent spatial and lead-time quadrature for stage 4."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from seismoflux.anomaly_increment.contracts import (
    DesignMatrix,
    FloatArray,
    readonly_float_matrix,
    readonly_float_vector,
)

ANOMALY_HALF_LIFE_DAYS = 90.0
PRIMARY_TIME_STEP_DAYS = 1.0
REFERENCE_TIME_STEP_DAYS = 0.5
INTEGRATION_RELATIVE_TOLERANCE = 0.005
INTEGRATION_NEAR_ZERO_ABSOLUTE_TOLERANCE = 1.0e-10
SPATIAL_RELATIVE_DENOMINATOR_FLOOR = 1.0e-12


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def lead_decay(
    lead_days: float | FloatArray,
    *,
    half_life_days: float = ANOMALY_HALF_LIFE_DAYS,
) -> float | FloatArray:
    """Return ``2**(-lead/half_life)`` without touching future anomaly updates."""

    half_life = _positive_finite("half_life_days", half_life_days)
    values = np.asarray(lead_days, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("lead_days must be finite and non-negative")
    result = np.exp2(-values / half_life).astype(np.float64, copy=False)
    if result.ndim == 0:
        return float(result)
    owned = np.array(result, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class MidpointQuadrature:
    """Composite midpoint nodes on the open-closed forecast interval."""

    horizon_days: float
    maximum_step_days: float
    lead_midpoints_days: FloatArray
    widths_days: FloatArray

    def __post_init__(self) -> None:
        horizon = _positive_finite("horizon_days", self.horizon_days)
        step = _positive_finite("maximum_step_days", self.maximum_step_days)
        midpoints = readonly_float_vector("lead_midpoints_days", self.lead_midpoints_days)
        widths = readonly_float_vector("widths_days", self.widths_days)
        if midpoints.shape != widths.shape:
            raise ValueError("midpoint leads and widths must align")
        if np.any(widths <= 0.0) or np.any(widths > step):
            raise ValueError("midpoint widths must lie in (0, maximum_step_days]")
        if np.any(midpoints <= 0.0) or np.any(midpoints > horizon):
            raise ValueError("midpoint leads must lie inside (0, horizon_days]")
        if not math.isclose(
            math.fsum(float(value) for value in widths),
            horizon,
            rel_tol=0.0,
            abs_tol=2.0e-14 * max(1.0, horizon),
        ):
            raise ValueError("midpoint widths must exactly partition the horizon")
        object.__setattr__(self, "horizon_days", horizon)
        object.__setattr__(self, "maximum_step_days", step)
        object.__setattr__(self, "lead_midpoints_days", midpoints)
        object.__setattr__(self, "widths_days", widths)


def composite_midpoint_quadrature(
    horizon_days: float,
    *,
    maximum_step_days: float = PRIMARY_TIME_STEP_DAYS,
) -> MidpointQuadrature:
    """Build midpoint nodes, including a midpoint for the final partial step."""

    horizon = _positive_finite("horizon_days", horizon_days)
    step = _positive_finite("maximum_step_days", maximum_step_days)
    full_count = math.floor(horizon / step)
    widths = [step] * full_count
    used = math.fsum(widths)
    remainder = horizon - used
    tolerance = 8.0 * np.finfo(np.float64).eps * max(1.0, horizon, step)
    if remainder > tolerance:
        widths.append(remainder)
    elif widths:
        widths[-1] += remainder
    else:
        widths.append(horizon)
    midpoints: list[float] = []
    cursor = 0.0
    for width in widths:
        midpoints.append(cursor + 0.5 * width)
        cursor += width
    return MidpointQuadrature(
        horizon_days=horizon,
        maximum_step_days=step,
        lead_midpoints_days=np.asarray(midpoints, dtype=np.float64),
        widths_days=np.asarray(widths, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class MidpointCompensatorTerms:
    """Time-expanded, target-blind rows for a shared-bin Poisson compensator."""

    design: DesignMatrix
    background_exposure_by_bin: FloatArray
    decay: FloatArray
    lead_midpoints_days: FloatArray

    def __post_init__(self) -> None:
        exposure = readonly_float_matrix(
            "background_exposure_by_bin",
            self.background_exposure_by_bin,
            allow_empty_rows=False,
            allow_empty_columns=False,
        )
        decay = readonly_float_vector("decay", self.decay)
        leads = readonly_float_vector("lead_midpoints_days", self.lead_midpoints_days)
        if exposure.shape[0] != self.design.row_count:
            raise ValueError("expanded background exposure must align with design rows")
        if decay.size != self.design.row_count or leads.size != self.design.row_count:
            raise ValueError("expanded lead-time columns must align with design rows")
        if np.any(exposure < 0.0):
            raise ValueError("background exposures must be non-negative")
        if not np.any(exposure > 0.0):
            raise ValueError("expanded background exposure must contain positive mass")
        if np.any(decay <= 0.0) or np.any(decay > 1.0):
            raise ValueError("lead decays must lie in (0, 1]")
        if np.any(leads <= 0.0):
            raise ValueError("midpoint leads must be positive")
        object.__setattr__(self, "background_exposure_by_bin", exposure)
        object.__setattr__(self, "decay", decay)
        object.__setattr__(self, "lead_midpoints_days", leads)


def expand_midpoint_compensator_terms(
    *,
    issue_design: DesignMatrix,
    background_spatial_mass_by_cell_and_bin: object,
    horizon_days: float,
    maximum_step_days: float = PRIMARY_TIME_STEP_DAYS,
) -> MidpointCompensatorTerms:
    """Repeat one issue-time field at midpoint leads and attach exact time widths."""

    spatial_mass = readonly_float_matrix(
        "background_spatial_mass_by_cell_and_bin",
        background_spatial_mass_by_cell_and_bin,
        allow_empty_rows=False,
        allow_empty_columns=False,
    )
    if spatial_mass.shape[0] != issue_design.row_count:
        raise ValueError("background spatial masses must align with issue design rows")
    if np.any(spatial_mass < 0.0) or not np.any(spatial_mass > 0.0):
        raise ValueError("background spatial masses must be non-negative with positive total mass")
    quadrature = composite_midpoint_quadrature(
        horizon_days,
        maximum_step_days=maximum_step_days,
    )
    time_count = quadrature.widths_days.size
    expanded_values = np.tile(issue_design.values, (time_count, 1))
    expanded_exposure = np.concatenate(
        [spatial_mass * width for width in quadrature.widths_days],
        axis=0,
    )
    expanded_leads = np.repeat(quadrature.lead_midpoints_days, issue_design.row_count)
    decay = lead_decay(expanded_leads)
    if not isinstance(decay, np.ndarray):  # pragma: no cover - expanded leads are always an array
        raise AssertionError("array lead decay unexpectedly returned a scalar")
    return MidpointCompensatorTerms(
        design=DesignMatrix(
            values=expanded_values,
            column_names=issue_design.column_names,
            penalty_factors=issue_design.penalty_factors,
            active_coefficients=issue_design.active_coefficients,
        ),
        background_exposure_by_bin=expanded_exposure,
        decay=decay,
        lead_midpoints_days=expanded_leads,
    )


def integrate_conditional_intensity(
    *,
    background_spatial_mass: object,
    issue_linear_predictor: object,
    rate_multiplier: float,
    horizon_days: float,
    maximum_step_days: float = PRIMARY_TIME_STEP_DAYS,
) -> float:
    """Integrate one bin's conditional intensity with the frozen midpoint rule."""

    background = readonly_float_vector("background_spatial_mass", background_spatial_mass)
    predictor = readonly_float_vector("issue_linear_predictor", issue_linear_predictor)
    if background.shape != predictor.shape:
        raise ValueError("background spatial mass and linear predictor must align")
    if np.any(background < 0.0) or not np.any(background > 0.0):
        raise ValueError("background spatial mass must be non-negative with positive total mass")
    rate = float(rate_multiplier)
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("rate_multiplier must be finite and non-negative")
    horizon = _positive_finite("horizon_days", horizon_days)
    if rate == 0.0:
        return 0.0
    background_total = math.fsum(float(value) for value in background)
    if not np.any(predictor):
        # This branch is the exact disabled-increment identity used in the code-freeze test.
        return rate * background_total * horizon
    quadrature = composite_midpoint_quadrature(
        horizon,
        maximum_step_days=maximum_step_days,
    )
    total = 0.0
    for midpoint, width in zip(
        quadrature.lead_midpoints_days,
        quadrature.widths_days,
        strict=True,
    ):
        decay = float(lead_decay(float(midpoint)))
        with np.errstate(over="raise", invalid="raise"):
            multiplier = np.exp(decay * predictor)
        spatial = float(np.dot(background, multiplier))
        total += rate * spatial * float(width)
    if not math.isfinite(total) or total < 0.0:
        raise FloatingPointError("conditional-intensity integral is not finite and non-negative")
    return total


@dataclass(frozen=True, slots=True)
class ConvergenceCheck:
    """One frozen relative/near-zero integration comparison."""

    candidate: float
    reference: float
    absolute_difference: float
    relative_difference: float
    relative_tolerance: float
    near_zero_absolute_tolerance: float
    passed: bool

    def as_mapping(self) -> dict[str, object]:
        return {
            "absolute_difference": self.absolute_difference,
            "candidate": self.candidate,
            "near_zero_absolute_tolerance": self.near_zero_absolute_tolerance,
            "passed": self.passed,
            "reference": self.reference,
            "relative_difference": self.relative_difference,
            "relative_tolerance": self.relative_tolerance,
        }


@dataclass(frozen=True, slots=True)
class SpatialGridConvergence:
    """Required 50/25/12.5 km audit with only 25 vs 12.5 km gating."""

    intensity_50km: float
    intensity_25km: float
    intensity_12_5km: float
    primary_vs_reference: ConvergenceCheck

    def __post_init__(self) -> None:
        values = (self.intensity_50km, self.intensity_25km, self.intensity_12_5km)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("all three spatial-grid intensities must be finite and non-negative")
        if self.primary_vs_reference.candidate != self.intensity_25km:
            raise ValueError("the 25 km integral must be the convergence candidate")
        if self.primary_vs_reference.reference != self.intensity_12_5km:
            raise ValueError("the 12.5 km integral must be the convergence reference")

    @property
    def passed(self) -> bool:
        return self.primary_vs_reference.passed

    def as_mapping(self) -> dict[str, object]:
        return {
            "gate": self.primary_vs_reference.as_mapping(),
            "grid_50km_role": "reported_coarse_trend_diagnostic_not_gate_reference",
            "intensity_12_5km": self.intensity_12_5km,
            "intensity_25km": self.intensity_25km,
            "intensity_50km": self.intensity_50km,
            "passed": self.passed,
        }


def compare_integrals(
    candidate: float,
    reference: float,
    *,
    relative_tolerance: float = INTEGRATION_RELATIVE_TOLERANCE,
    near_zero_absolute_tolerance: float = INTEGRATION_NEAR_ZERO_ABSOLUTE_TOLERANCE,
) -> ConvergenceCheck:
    """Apply the preregistered relative gate with its near-zero absolute branch."""

    candidate_value = float(candidate)
    reference_value = float(reference)
    if not math.isfinite(candidate_value) or candidate_value < 0.0:
        raise ValueError("candidate integral must be finite and non-negative")
    if not math.isfinite(reference_value) or reference_value < 0.0:
        raise ValueError("reference integral must be finite and non-negative")
    rel_tol = _positive_finite("relative_tolerance", relative_tolerance)
    abs_tol = _positive_finite("near_zero_absolute_tolerance", near_zero_absolute_tolerance)
    difference = abs(candidate_value - reference_value)
    relative = difference / max(abs(reference_value), SPATIAL_RELATIVE_DENOMINATOR_FLOOR)
    passed = difference <= abs_tol or relative <= rel_tol
    return ConvergenceCheck(
        candidate=candidate_value,
        reference=reference_value,
        absolute_difference=difference,
        relative_difference=relative,
        relative_tolerance=rel_tol,
        near_zero_absolute_tolerance=abs_tol,
        passed=passed,
    )


def spatial_grid_convergence(
    *,
    intensity_50km: float,
    intensity_25km: float,
    intensity_12_5km: float,
) -> SpatialGridConvergence:
    """Apply the preregistered 25 km versus 12.5 km spatial gate."""

    check = compare_integrals(intensity_25km, intensity_12_5km)
    return SpatialGridConvergence(
        intensity_50km=float(intensity_50km),
        intensity_25km=float(intensity_25km),
        intensity_12_5km=float(intensity_12_5km),
        primary_vs_reference=check,
    )


def temporal_midpoint_convergence(
    *,
    background_spatial_mass: object,
    issue_linear_predictor: object,
    rate_multiplier: float,
    horizon_days: float,
) -> tuple[float, float, ConvergenceCheck]:
    """Compare the frozen 1-day primary and 0.5-day reference integrals."""

    primary = integrate_conditional_intensity(
        background_spatial_mass=background_spatial_mass,
        issue_linear_predictor=issue_linear_predictor,
        rate_multiplier=rate_multiplier,
        horizon_days=horizon_days,
        maximum_step_days=PRIMARY_TIME_STEP_DAYS,
    )
    reference = integrate_conditional_intensity(
        background_spatial_mass=background_spatial_mass,
        issue_linear_predictor=issue_linear_predictor,
        rate_multiplier=rate_multiplier,
        horizon_days=horizon_days,
        maximum_step_days=REFERENCE_TIME_STEP_DAYS,
    )
    return primary, reference, compare_integrals(primary, reference)


__all__ = [
    "ANOMALY_HALF_LIFE_DAYS",
    "INTEGRATION_NEAR_ZERO_ABSOLUTE_TOLERANCE",
    "INTEGRATION_RELATIVE_TOLERANCE",
    "PRIMARY_TIME_STEP_DAYS",
    "REFERENCE_TIME_STEP_DAYS",
    "SPATIAL_RELATIVE_DENOMINATOR_FLOOR",
    "ConvergenceCheck",
    "MidpointCompensatorTerms",
    "MidpointQuadrature",
    "SpatialGridConvergence",
    "compare_integrals",
    "composite_midpoint_quadrature",
    "expand_midpoint_compensator_terms",
    "integrate_conditional_intensity",
    "lead_decay",
    "spatial_grid_convergence",
    "temporal_midpoint_convergence",
]
