"""Frozen spatial numerics for the Stage 2S causal-seismicity screen."""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Iterable
from decimal import Decimal, localcontext
from typing import Literal

import numpy as np

from seismoflux.background.grid import GridConvergenceError
from seismoflux.background.poisson import GaussianMixtureFamily, SpatialQuadrature
from seismoflux.stage2s.contracts import (
    ALPHA_BISECTION_ITERATIONS,
    COMPENSATOR_DIFFERENCE_ABSOLUTE_TOLERANCE,
    CONVERGENCE_DENSITY_L1_MAX,
    CONVERGENCE_RELATIVE_COUNT_MAX,
    DERIVATIVE_SIGN_TOLERANCE,
    FROZEN_SPATIAL_BANDWIDTH_KM,
    MASS_SUM_ABSOLUTE_TOLERANCE,
    PRIMARY_ALARM_BUDGET_KM2,
    AlarmMask,
    AlphaFit,
    DerivativeComparison,
    EvidenceInsufficientError,
    FitEventOrder,
    FloatArray,
    GridConvergence,
    ModelId,
    NormalizedSpatialDensity,
    PairedCompensator,
    SharedRate,
    SignedLogDerivative,
    SpatialGrid,
    SpatialQuadratureFamily,
    Stage2SModels,
)

_MODEL_BUILD_IDS = frozenset({"S0", "R", "RP"})
_MIXTURE_MODEL_IDS = frozenset({"S1", "SP"})
_FINE_CELL_SIZE_KM = 12.5
_OPERATIONAL_CELL_SIZE_KM = 25.0
_DIAGNOSTIC_CELL_SIZE_KM = 50.0
_OPERATIONAL_CELL_SIZE_M = 25_000.0


def _read_float_vector(
    name: str,
    value: object,
    *,
    allow_empty: bool = False,
    positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return result


def _read_source_xy(value: object, *, allow_empty: bool) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (2,):
        raise ValueError("source_xy_km must have shape (n, 2)")
    if not allow_empty and result.shape[0] == 0:
        raise ValueError("source_xy_km must not be empty")
    if not np.isfinite(result).all():
        raise ValueError("source_xy_km must contain only finite values")
    return result


def _read_only(values: object) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _mass_total(values: FloatArray) -> float:
    return math.fsum(float(value) for value in values)


def _validate_normalized_mass(name: str, values: FloatArray) -> None:
    total = _mass_total(values)
    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{name} must sum to one within the frozen tolerance")


def _aggregate_aligned_mass(
    child_mass: object,
    *,
    child_grid: SpatialGrid,
    parent_grid: SpatialGrid,
) -> FloatArray:
    """Aggregate one aligned grid to its factor-two parent in fixed child order."""

    if child_grid.cell_size_km * 2.0 != parent_grid.cell_size_km:
        raise ValueError("aligned aggregation requires a factor-two parent grid")
    values = _read_float_vector("child_mass", child_mass, nonnegative=True)
    if values.size != child_grid.cell_count:
        raise ValueError("child_mass must match the child grid")
    parent_lookup = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(
            zip(parent_grid.rows, parent_grid.columns, strict=True)
        )
    }
    grouped: list[list[float]] = [[] for _ in range(parent_grid.cell_count)]
    for mass, child_row, child_column in zip(
        values,
        child_grid.rows,
        child_grid.columns,
        strict=True,
    ):
        parent_key = (int(child_row) // 2, int(child_column) // 2)
        try:
            parent_index = parent_lookup[parent_key]
        except KeyError as error:
            raise ValueError("every child cell must have one aligned clipped parent") from error
        grouped[parent_index].append(float(mass))
    if any(not members for members in grouped):
        raise ValueError("every clipped parent must contain at least one clipped child")
    aggregated = np.asarray(
        [math.fsum(members) for members in grouped],
        dtype=np.float64,
    )
    child_total = _mass_total(values)
    parent_total = _mass_total(aggregated)
    if not math.isclose(
        child_total,
        parent_total,
        rel_tol=0.0,
        abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("aligned aggregation must preserve total mass")
    aggregated.setflags(write=False)
    return aggregated


def aggregate_operational_mass_to_25km(
    mass_12_5km: object,
    quadrature_family: SpatialQuadratureFamily,
) -> FloatArray:
    """Aggregate normalized 12.5 km masses to aligned operational 25 km cells."""

    fine_mass = _read_float_vector(
        "mass_12_5km",
        mass_12_5km,
        nonnegative=True,
    )
    _validate_normalized_mass("mass_12_5km", fine_mass)
    aggregated = _aggregate_aligned_mass(
        fine_mass,
        child_grid=quadrature_family.at(_FINE_CELL_SIZE_KM),
        parent_grid=quadrature_family.at(_OPERATIONAL_CELL_SIZE_KM),
    )
    _validate_normalized_mass("aggregated 25 km mass", aggregated)
    return aggregated


def _direct_cell_mass(
    mixture: GaussianMixtureFamily,
    *,
    bandwidth_km: float,
    normalization_mass: float,
    quadrature: SpatialQuadrature,
) -> FloatArray:
    density = mixture.raw_densities(
        quadrature.x_km,
        quadrature.y_km,
        bandwidths_km=(bandwidth_km,),
    )[bandwidth_km]
    density /= normalization_mass
    masses = np.asarray(density * quadrature.area_km2, dtype=np.float64)
    if not np.isfinite(masses).all() or np.any(masses < 0.0):
        raise ValueError("direct quadrature masses must be finite and non-negative")
    masses.setflags(write=False)
    return masses


def _relative_count_difference(coarse: FloatArray, fine: FloatArray) -> float:
    coarse_total = _mass_total(coarse)
    fine_total = _mass_total(fine)
    if not math.isfinite(coarse_total) or not math.isfinite(fine_total) or fine_total <= 0.0:
        raise ValueError("convergence mass totals must be finite and positive")
    return abs(coarse_total - fine_total) / max(abs(fine_total), 1.0e-12)


def _normalized_l1(left: FloatArray, right: FloatArray) -> float:
    if left.shape != right.shape:
        raise ValueError("L1 convergence arrays must have one common shape")
    left_total = _mass_total(left)
    right_total = _mass_total(right)
    if (
        not math.isfinite(left_total)
        or not math.isfinite(right_total)
        or left_total <= 0.0
        or right_total <= 0.0
    ):
        raise ValueError("L1 convergence mass totals must be finite and positive")
    return math.fsum(
        abs(float(left_value) / left_total - float(right_value) / right_total)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _convergence(
    *,
    quadrature_family: SpatialQuadratureFamily,
    mass_25km: FloatArray,
    direct_mass_25km: FloatArray,
    direct_mass_50km: FloatArray,
) -> GridConvergence:
    aggregated_direct_25_to_50 = _aggregate_aligned_mass(
        direct_mass_25km,
        child_grid=quadrature_family.at(_OPERATIONAL_CELL_SIZE_KM),
        parent_grid=quadrature_family.at(_DIAGNOSTIC_CELL_SIZE_KM),
    )
    primary_relative = _relative_count_difference(direct_mass_25km, mass_25km)
    primary_l1 = _normalized_l1(direct_mass_25km, mass_25km)
    diagnostic_relative = _relative_count_difference(
        direct_mass_50km,
        direct_mass_25km,
    )
    diagnostic_l1 = _normalized_l1(
        direct_mass_50km,
        aggregated_direct_25_to_50,
    )
    passed = (
        primary_relative <= CONVERGENCE_RELATIVE_COUNT_MAX
        and primary_l1 <= CONVERGENCE_DENSITY_L1_MAX
        and diagnostic_relative <= CONVERGENCE_RELATIVE_COUNT_MAX
        and diagnostic_l1 <= CONVERGENCE_DENSITY_L1_MAX
    )
    return GridConvergence(
        primary_relative_count_difference=primary_relative,
        primary_density_l1=primary_l1,
        diagnostic_relative_count_difference=diagnostic_relative,
        diagnostic_density_l1=diagnostic_l1,
        passed=passed,
    )


def _require_convergence(model_id: str, convergence: GridConvergence) -> None:
    if not convergence.passed:
        raise GridConvergenceError(
            f"{model_id} failed the frozen 50/25/12.5 km quadrature convergence gate"
        )


def build_normalized_kde(
    source_xy_km: object,
    quadrature_family: SpatialQuadratureFamily,
    *,
    model_id: ModelId,
    bandwidth_km: float = FROZEN_SPATIAL_BANDWIDTH_KM,
) -> NormalizedSpatialDensity:
    """Build one fixed 75 km equal-weight Gaussian mean-KDE.

    The normalization constant is accumulated with :func:`math.fsum` in the
    exact stored 12.5 km cell order.  No target, score, or alarm information is
    accepted by this function.
    """

    if model_id not in _MODEL_BUILD_IDS:
        raise ValueError("a primitive KDE model_id must be S0, R, or RP")
    bandwidth = float(bandwidth_km)
    if bandwidth != FROZEN_SPATIAL_BANDWIDTH_KM:
        raise ValueError("Stage 2S bandwidth is frozen at exactly 75 km")
    source_xy = _read_source_xy(source_xy_km, allow_empty=False)
    mixture = GaussianMixtureFamily(source_xy[:, 0], source_xy[:, 1])
    fine_grid = quadrature_family.at(_FINE_CELL_SIZE_KM)
    raw_fine_density = mixture.raw_densities(
        fine_grid.quadrature.x_km,
        fine_grid.quadrature.y_km,
        bandwidths_km=(bandwidth,),
    )[bandwidth]
    raw_fine_masses = np.asarray(
        raw_fine_density * fine_grid.quadrature.area_km2,
        dtype=np.float64,
    )
    normalization_mass = math.fsum(float(value) for value in raw_fine_masses)
    if not math.isfinite(normalization_mass) or normalization_mass <= 0.0:
        raise ValueError("KDE normalization mass must be finite and positive")
    mass_12_5km = np.asarray(raw_fine_masses / normalization_mass, dtype=np.float64)
    if not np.isfinite(mass_12_5km).all() or np.any(mass_12_5km < 0.0):
        raise ValueError("normalized 12.5 km masses must be finite and non-negative")
    mass_12_5km.setflags(write=False)
    _validate_normalized_mass("normalized 12.5 km mass", mass_12_5km)
    mass_25km = aggregate_operational_mass_to_25km(
        mass_12_5km,
        quadrature_family,
    )
    direct_mass_25km = _direct_cell_mass(
        mixture,
        bandwidth_km=bandwidth,
        normalization_mass=normalization_mass,
        quadrature=quadrature_family.at(_OPERATIONAL_CELL_SIZE_KM).quadrature,
    )
    direct_mass_50km = _direct_cell_mass(
        mixture,
        bandwidth_km=bandwidth,
        normalization_mass=normalization_mass,
        quadrature=quadrature_family.at(_DIAGNOSTIC_CELL_SIZE_KM).quadrature,
    )
    convergence = _convergence(
        quadrature_family=quadrature_family,
        mass_25km=mass_25km,
        direct_mass_25km=direct_mass_25km,
        direct_mass_50km=direct_mass_50km,
    )
    _require_convergence(model_id, convergence)
    return NormalizedSpatialDensity(
        model_id=model_id,
        grid_family=quadrature_family,
        mass_12_5km=mass_12_5km,
        mass_25km=mass_25km,
        direct_mass_25km=direct_mass_25km,
        direct_mass_50km=direct_mass_50km,
        convergence=convergence,
        bandwidth_km=bandwidth,
        normalization_mass=normalization_mass,
        _kernel=mixture,
    )


def build_recent_component(
    source_xy_km: object,
    quadrature_family: SpatialQuadratureFamily,
    *,
    component_id: Literal["R", "RP"],
    empty_fallback_s0: NormalizedSpatialDensity,
) -> NormalizedSpatialDensity:
    """Build R or RP, returning the exact S0 object for an empty causal window."""

    if component_id not in {"R", "RP"}:
        raise ValueError("component_id must be R or RP")
    if empty_fallback_s0.model_id != "S0":
        raise ValueError("empty_fallback_s0 must be S0")
    if empty_fallback_s0.grid_family is not quadrature_family:
        raise ValueError("empty fallback and recent component must share one grid family object")
    source_xy = _read_source_xy(source_xy_km, allow_empty=True)
    if source_xy.shape[0] == 0:
        return empty_fallback_s0
    return build_normalized_kde(
        source_xy,
        quadrature_family,
        model_id=component_id,
    )


def _convex_mass(
    baseline: FloatArray,
    component: FloatArray,
    *,
    alpha: float,
) -> FloatArray:
    if baseline.shape != component.shape:
        raise ValueError("mixture mass arrays must have one common shape")
    if alpha == 0.0 or baseline is component:
        return _read_only(baseline)
    if alpha == 1.0:
        return _read_only(component)
    values = np.asarray(
        (1.0 - alpha) * baseline + alpha * component,
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("mixture masses must be finite and non-negative")
    values.setflags(write=False)
    return values


def mix_density(
    s0: NormalizedSpatialDensity,
    component: NormalizedSpatialDensity,
    alpha: float,
    *,
    model_id: Literal["S1", "SP"],
) -> NormalizedSpatialDensity:
    """Build S1 or SP as the preregistered convex spatial mixture."""

    if s0.model_id != "S0":
        raise ValueError("the mixture baseline must be S0")
    if model_id not in _MIXTURE_MODEL_IDS:
        raise ValueError("mixture model_id must be S1 or SP")
    allowed_component_ids = {"S0", "R"} if model_id == "S1" else {"S0", "RP"}
    if component.model_id not in allowed_component_ids:
        raise ValueError(f"{model_id} received the wrong recent-history component")
    if component.grid_family is not s0.grid_family:
        raise ValueError("mixture components must share one grid family object")
    weight = float(alpha)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")
    mass_12_5km = _convex_mass(s0.mass_12_5km, component.mass_12_5km, alpha=weight)
    _validate_normalized_mass("mixture 12.5 km mass", mass_12_5km)
    mass_25km = aggregate_operational_mass_to_25km(
        mass_12_5km,
        s0.grid_family,
    )
    direct_mass_25km = _convex_mass(
        s0.direct_mass_25km,
        component.direct_mass_25km,
        alpha=weight,
    )
    direct_mass_50km = _convex_mass(
        s0.direct_mass_50km,
        component.direct_mass_50km,
        alpha=weight,
    )
    convergence = _convergence(
        quadrature_family=s0.grid_family,
        mass_25km=mass_25km,
        direct_mass_25km=direct_mass_25km,
        direct_mass_50km=direct_mass_50km,
    )
    _require_convergence(model_id, convergence)
    return NormalizedSpatialDensity(
        model_id=model_id,
        grid_family=s0.grid_family,
        mass_12_5km=mass_12_5km,
        mass_25km=mass_25km,
        direct_mass_25km=direct_mass_25km,
        direct_mass_50km=direct_mass_50km,
        convergence=convergence,
        _baseline=s0,
        _component=component,
        _alpha=weight,
    )


def build_stage2s_models(
    s0: NormalizedSpatialDensity,
    recent: NormalizedSpatialDensity,
    preceding: NormalizedSpatialDensity,
    *,
    alpha_r: float,
    alpha_p: float,
) -> Stage2SModels:
    """Return the only allowed models in the frozen S0/S1/SP order."""

    return Stage2SModels(
        s0=s0,
        s1=mix_density(s0, recent, alpha_r, model_id="S1"),
        sp=mix_density(s0, preceding, alpha_p, model_id="SP"),
    )


def _ordered_log_fit_vectors(
    log_q0: object,
    log_qx: object,
    event_order: FitEventOrder,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    baseline = _read_float_vector(
        "log_q0",
        log_q0,
        allow_empty=True,
    )
    component = _read_float_vector(
        "log_qx",
        log_qx,
        allow_empty=True,
    )
    if baseline.shape != component.shape:
        raise ValueError("log_q0 and log_qx must have one common shape")
    if baseline.size != event_order.event_count:
        raise ValueError("fit log densities must match the deterministic event keys")
    if baseline.size == 0:
        raise EvidenceInsufficientError("alpha fit requires at least one fit event")
    indices = event_order.deterministic_indices
    return (
        tuple(float(baseline[index]) for index in indices),
        tuple(float(component[index]) for index in indices),
    )


def _finite_derivative_summary(
    *,
    sign: int,
    log_abs_mean: float | None,
) -> SignedLogDerivative:
    if sign == 0:
        return SignedLogDerivative(
            sign=0,
            log_abs_mean=None,
            finite_float_value=0.0,
            comparison="within_tolerance",
        )
    assert log_abs_mean is not None
    log_tolerance = math.log(DERIVATIVE_SIGN_TOLERANCE)
    if log_abs_mean <= log_tolerance:
        comparison: DerivativeComparison = "within_tolerance"
    elif sign < 0:
        comparison = "less_than_negative_tolerance"
    else:
        comparison = "greater_than_positive_tolerance"
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    if math.log(minimum_subnormal) <= log_abs_mean <= math.log(sys.float_info.max):
        absolute = math.exp(log_abs_mean)
        finite_value = float(sign) * absolute if absolute != 0.0 else None
    else:
        finite_value = None
    return SignedLogDerivative(
        sign=sign,
        log_abs_mean=log_abs_mean,
        finite_float_value=finite_value,
        comparison=comparison,
    )


def _endpoint_term_signed_log(
    log_ratio: float,
    *,
    alpha: Literal[0, 1],
) -> tuple[int, float] | None:
    if log_ratio == 0.0:
        return None
    if alpha == 0:
        if log_ratio > 0.0:
            correction = math.log(-math.expm1(-log_ratio))
            return (1, log_ratio + correction)
        return (-1, math.log(-math.expm1(log_ratio)))
    if log_ratio > 0.0:
        return (1, math.log(-math.expm1(-log_ratio)))
    correction = math.log(-math.expm1(log_ratio))
    return (-1, -log_ratio + correction)


def _logsumexp(log_values: list[float]) -> float | None:
    if not log_values:
        return None
    maximum = max(log_values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in log_values))


def _decimal_endpoint_summary(
    log_ratios: tuple[float, ...],
    *,
    alpha: Literal[0, 1],
) -> SignedLogDerivative:
    with localcontext() as context:
        context.prec = 100
        terms: list[Decimal] = []
        one = Decimal(1)
        for value in log_ratios:
            ratio = Decimal(repr(value)).exp()
            terms.append(ratio - one if alpha == 0 else one - (one / ratio))
        mean = sum(terms, Decimal(0)) / Decimal(len(terms))
        if mean == 0:
            return _finite_derivative_summary(sign=0, log_abs_mean=None)
        sign = 1 if mean > 0 else -1
        log_abs_mean = float(abs(mean).ln())
    return _finite_derivative_summary(sign=sign, log_abs_mean=log_abs_mean)


def _endpoint_derivative_summary(
    log_ratios: tuple[float, ...],
    *,
    alpha: Literal[0, 1],
) -> SignedLogDerivative:
    positive: list[float] = []
    negative: list[float] = []
    for value in log_ratios:
        term = _endpoint_term_signed_log(value, alpha=alpha)
        if term is None:
            continue
        sign, log_abs = term
        (positive if sign > 0 else negative).append(log_abs)
    log_positive = _logsumexp(positive)
    log_negative = _logsumexp(negative)
    if log_positive is None and log_negative is None:
        return _finite_derivative_summary(sign=0, log_abs_mean=None)
    if log_positive is None:
        assert log_negative is not None
        return _finite_derivative_summary(
            sign=-1,
            log_abs_mean=log_negative - math.log(len(log_ratios)),
        )
    if log_negative is None:
        return _finite_derivative_summary(
            sign=1,
            log_abs_mean=log_positive - math.log(len(log_ratios)),
        )
    if abs(log_positive - log_negative) <= 1.0e-10:
        return _decimal_endpoint_summary(log_ratios, alpha=alpha)
    if log_positive > log_negative:
        sign = 1
        high, low = log_positive, log_negative
    else:
        sign = -1
        high, low = log_negative, log_positive
    log_abs_sum = high + math.log1p(-math.exp(low - high))
    return _finite_derivative_summary(
        sign=sign,
        log_abs_mean=log_abs_sum - math.log(len(log_ratios)),
    )


def _interior_derivative(
    alpha: float,
    log_ratios: tuple[float, ...],
) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("interior derivative requires 0 < alpha < 1")
    terms: list[float] = []
    for log_ratio in log_ratios:
        if log_ratio >= 0.0:
            exp_negative = math.exp(-log_ratio)
            numerator = -math.expm1(-log_ratio)
            denominator = alpha + (1.0 - alpha) * exp_negative
        else:
            exp_positive = math.exp(log_ratio)
            numerator = math.expm1(log_ratio)
            denominator = (1.0 - alpha) + alpha * exp_positive
        term = numerator / denominator
        terms.append(term)
    total = math.fsum(terms)
    absolute_total = math.fsum(abs(term) for term in terms)
    if absolute_total and abs(total) <= absolute_total * 1.0e-14:
        with localcontext() as context:
            context.prec = 80
            total_decimal = sum(
                (Decimal.from_float(term) for term in terms),
                Decimal(0),
            )
            return float(total_decimal / Decimal(len(terms)))
    return total / len(terms)


def fit_alpha_log_density(
    log_q0: object,
    log_qx: object,
    event_order: FitEventOrder,
) -> AlphaFit:
    """Solve alpha from finite log densities without floors or ratio clipping."""

    baseline, component = _ordered_log_fit_vectors(log_q0, log_qx, event_order)
    log_ratios = tuple(
        component_value - baseline_value
        for baseline_value, component_value in zip(
            baseline,
            component,
            strict=True,
        )
    )
    if not all(math.isfinite(value) for value in log_ratios):
        raise ValueError("fit log-density ratios must remain finite")
    derivative_at_zero = _endpoint_derivative_summary(log_ratios, alpha=0)
    derivative_at_one = _endpoint_derivative_summary(log_ratios, alpha=1)
    event_count = len(log_ratios)
    positive_flat_limit = math.log1p(DERIVATIVE_SIGN_TOLERANCE)
    negative_flat_limit = math.log1p(-DERIVATIVE_SIGN_TOLERANCE)
    if all(
        (0.0 <= value <= positive_flat_limit) or (negative_flat_limit <= value < 0.0)
        for value in log_ratios
    ):
        return AlphaFit(
            alpha=0.0,
            solver_case="flat",
            iterations=0,
            derivative_at_zero=derivative_at_zero,
            derivative_at_one=derivative_at_one,
            fit_event_count=event_count,
        )
    if derivative_at_zero.comparison != "greater_than_positive_tolerance":
        return AlphaFit(
            alpha=0.0,
            solver_case="derivative_at_zero",
            iterations=0,
            derivative_at_zero=derivative_at_zero,
            derivative_at_one=derivative_at_one,
            fit_event_count=event_count,
        )
    if derivative_at_one.comparison != "less_than_negative_tolerance":
        return AlphaFit(
            alpha=1.0,
            solver_case="derivative_at_one",
            iterations=0,
            derivative_at_zero=derivative_at_zero,
            derivative_at_one=derivative_at_one,
            fit_event_count=event_count,
        )
    left = 0.0
    right = 1.0
    for _ in range(ALPHA_BISECTION_ITERATIONS):
        midpoint = (left + right) / 2.0
        if _interior_derivative(midpoint, log_ratios) > 0.0:
            left = midpoint
        else:
            right = midpoint
    return AlphaFit(
        alpha=(left + right) / 2.0,
        solver_case="bisection",
        iterations=ALPHA_BISECTION_ITERATIONS,
        derivative_at_zero=derivative_at_zero,
        derivative_at_one=derivative_at_one,
        fit_event_count=event_count,
    )


def fit_alpha(
    q0: object,
    qx: object,
    event_order: FitEventOrder,
) -> AlphaFit:
    """Backward-compatible density input routed through the stable log solver."""

    baseline = _read_float_vector("q0", q0, allow_empty=True, positive=True)
    component = _read_float_vector("qx", qx, allow_empty=True, positive=True)
    return fit_alpha_log_density(np.log(baseline), np.log(component), event_order)


def stable_mixture_log_density(
    log_q0: object,
    log_qx: object,
    alpha: float,
) -> FloatArray:
    """Apply the exact endpoint rules and interior ``numpy.logaddexp`` formula."""

    baseline = _read_float_vector("log_q0", log_q0, allow_empty=True)
    component = _read_float_vector("log_qx", log_qx, allow_empty=True)
    if baseline.shape != component.shape:
        raise ValueError("log_q0 and log_qx must have one common shape")
    weight = float(alpha)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")
    if weight == 0.0:
        return _read_only(baseline)
    if weight == 1.0:
        return _read_only(component)
    result = np.logaddexp(
        math.log1p(-weight) + baseline,
        math.log(weight) + component,
    )
    if not np.isfinite(result).all():
        raise ValueError("mixture log densities must remain finite")
    result.setflags(write=False)
    return result


def estimate_shared_m5_6_rate(
    assigned_event_ids: Iterable[str],
    *,
    total_exposure_days: float,
) -> SharedRate:
    """Estimate the one fold-level M5--6 daily rate, independently of spatial models."""

    event_ids = tuple(assigned_event_ids)
    if any(not isinstance(event_id, str) or not event_id.strip() for event_id in event_ids):
        raise ValueError("assigned M5--6 event IDs must be non-empty strings")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("each M5--6 event must be assigned exactly once")
    exposure = float(total_exposure_days)
    if not math.isfinite(exposure) or exposure <= 0.0:
        raise ValueError("total_exposure_days must be finite and positive")
    if not event_ids:
        raise EvidenceInsufficientError("a zero M5--6 fit rate is evidence insufficient")
    return SharedRate(
        rate_per_day=len(event_ids) / exposure,
        event_count=len(event_ids),
        exposure_days=exposure,
        assigned_event_ids=event_ids,
    )


def recompute_pair_compensator(
    rate_per_day: float,
    horizon_days: float,
    ordered_issue_mass_a: object,
    ordered_issue_mass_b: object,
) -> PairedCompensator:
    """Independently recompute both paired compensators from issue mass arrays."""

    rate = float(rate_per_day)
    horizon = float(horizon_days)
    if not math.isfinite(rate) or rate <= 0.0:
        raise EvidenceInsufficientError("shared M5--6 rate must be finite and positive")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_days must be finite and positive")
    mass_a = _read_float_vector(
        "ordered_issue_mass_a",
        ordered_issue_mass_a,
        positive=True,
    )
    mass_b = _read_float_vector(
        "ordered_issue_mass_b",
        ordered_issue_mass_b,
        positive=True,
    )
    if mass_a.shape != mass_b.shape:
        raise ValueError("paired issue mass arrays must have one common shape")
    for name, values in (
        ("ordered_issue_mass_a", mass_a),
        ("ordered_issue_mass_b", mass_b),
    ):
        if any(
            not math.isclose(
                float(value),
                1.0,
                rel_tol=0.0,
                abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
            )
            for value in values
        ):
            raise ValueError(f"every {name} value must independently sum to one")
    compensator_a = math.fsum(rate * horizon * float(value) for value in mass_a)
    compensator_b = math.fsum(rate * horizon * float(value) for value in mass_b)
    difference = compensator_a - compensator_b
    if not all(math.isfinite(value) for value in (compensator_a, compensator_b, difference)):
        raise ValueError("paired compensators must remain finite")
    if abs(difference) > COMPENSATOR_DIFFERENCE_ABSOLUTE_TOLERANCE:
        raise ValueError("paired global compensator difference exceeds the frozen tolerance")
    return PairedCompensator(
        rate_per_day=rate,
        horizon_days=horizon,
        issue_count=int(mass_a.size),
        compensator_a=compensator_a,
        compensator_b=compensator_b,
        difference=difference,
    )


def event_cell_index_25km(x_m: float, y_m: float) -> tuple[int, int]:
    """Apply the frozen half-open 25 km cell floor rule with no epsilon."""

    x_value = float(x_m)
    y_value = float(y_m)
    if not math.isfinite(x_value) or not math.isfinite(y_value):
        raise ValueError("projected event coordinates must be finite")
    column = math.floor(x_value / _OPERATIONAL_CELL_SIZE_M)
    row = math.floor(y_value / _OPERATIONAL_CELL_SIZE_M)
    return row, column


def _validate_alarm_grid_identity(
    model_grid: SpatialGrid,
    query_grid: SpatialGrid,
) -> None:
    if query_grid.cell_size_km != _OPERATIONAL_CELL_SIZE_KM:
        raise ValueError("alarm query grid must use frozen 25 km cells")
    if model_grid.cell_ids != query_grid.cell_ids:
        raise ValueError("alarm query-grid cell IDs must match the operational mass grid")
    if not np.array_equal(model_grid.rows, query_grid.rows):
        raise ValueError("alarm query-grid rows must match the operational mass grid")
    if not np.array_equal(model_grid.columns, query_grid.columns):
        raise ValueError("alarm query-grid columns must match the operational mass grid")
    if not np.array_equal(model_grid.query_xy_km, query_grid.query_xy_km):
        raise ValueError("alarm query coordinates must match the operational mass grid bitwise")
    if not np.array_equal(
        model_grid.clipped_area_km2,
        query_grid.clipped_area_km2,
    ):
        raise ValueError("alarm clipped areas must match the operational mass grid bitwise")


def _ranking_digest(
    model_id: str,
    grid: SpatialGrid,
    ranking: tuple[int, ...],
    intensity: FloatArray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"seismoflux-stage2s-alarm-ranking-v1\0")
    digest.update(model_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(grid.grid_id.encode("utf-8"))
    digest.update(b"\0")
    for index in ranking:
        digest.update(grid.cell_ids[index].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(grid.rows[index])).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(grid.columns[index])).encode("ascii"))
        digest.update(b"\0")
        digest.update(float(intensity[index]).hex().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_alarm_prefix(
    model: NormalizedSpatialDensity,
    query_grid: SpatialGrid | None = None,
    *,
    budget_km2: float = PRIMARY_ALARM_BUDGET_KM2,
) -> AlarmMask:
    """Select the complete frozen-area prefix by mass/area and fixed tie breaks."""

    budget = float(budget_km2)
    if not math.isfinite(budget) or budget != PRIMARY_ALARM_BUDGET_KM2:
        raise ValueError("Stage 2S alarm budget is frozen at exactly 600000 km2")
    operational_grid = model.grid_family.at(_OPERATIONAL_CELL_SIZE_KM)
    alarm_grid = operational_grid if query_grid is None else query_grid
    _validate_alarm_grid_identity(operational_grid, alarm_grid)
    mass = _read_float_vector(
        "model.mass_25km",
        model.mass_25km,
        nonnegative=True,
    )
    _validate_normalized_mass("model.mass_25km", mass)
    intensity = np.asarray(mass / alarm_grid.clipped_area_km2, dtype=np.float64)
    if not np.isfinite(intensity).all() or np.any(intensity < 0.0):
        raise ValueError("alarm ranking intensity must be finite and non-negative")
    ranking = tuple(
        sorted(
            range(alarm_grid.cell_count),
            key=lambda index: (
                -float(intensity[index]),
                int(alarm_grid.rows[index]),
                int(alarm_grid.columns[index]),
                alarm_grid.cell_ids[index].encode("utf-8"),
            ),
        )
    )
    selected: list[int] = []
    selected_areas: list[float] = []
    for index in ranking:
        candidate_areas = (*selected_areas, float(alarm_grid.clipped_area_km2[index]))
        candidate_total = math.fsum(candidate_areas)
        if candidate_total > budget:
            break
        selected.append(index)
        selected_areas.append(float(alarm_grid.clipped_area_km2[index]))
    selected_indices = np.asarray(selected, dtype=np.int64)
    selected_indices.setflags(write=False)
    selected_cell_ids = tuple(alarm_grid.cell_ids[index] for index in selected)
    return AlarmMask(
        model_id=model.model_id,
        selected_cell_ids=selected_cell_ids,
        selected_indices=selected_indices,
        actual_area_km2=math.fsum(selected_areas),
        budget_km2=budget,
        grid_id=alarm_grid.grid_id,
        ranking_sha256=_ranking_digest(
            model.model_id,
            alarm_grid,
            ranking,
            intensity,
        ),
    )


__all__ = [
    "aggregate_operational_mass_to_25km",
    "build_normalized_kde",
    "build_recent_component",
    "build_stage2s_models",
    "estimate_shared_m5_6_rate",
    "event_cell_index_25km",
    "fit_alpha",
    "fit_alpha_log_density",
    "mix_density",
    "recompute_pair_compensator",
    "select_alarm_prefix",
    "stable_mixture_log_density",
]
