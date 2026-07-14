"""Dependency-free analytic primitives for the production ETAS model.

All distances are kilometres, areas are square kilometres, and times are
days.  The functions deliberately accept scalars only: array evaluation and
numerical integration belong to higher-level CPU backends, while this module
defines the formulas and their input domain in one auditable place.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

_DEFAULT_EQUALITY_TOLERANCE = 1.0e-12


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_result(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite for the supplied parameters")
    return value


def _exp_finite(name: str, log_value: float) -> float:
    try:
        result = math.exp(log_value)
    except OverflowError as error:
        raise ValueError(f"{name} is not finite for the supplied parameters") from error
    return _finite_result(name, result)


def _validate_magnitude(magnitude: float, mc: float) -> tuple[float, float]:
    magnitude_value = _finite("magnitude", magnitude)
    mc_value = _finite("mc", mc)
    if magnitude_value < mc_value:
        raise ValueError("magnitude must be greater than or equal to mc")
    return magnitude_value, mc_value


def _validate_productivity_parameters(k: float, alpha: float) -> tuple[float, float]:
    return _nonnegative("k", k), _nonnegative("alpha", alpha)


def _validate_omori_parameters(c_days: float, p: float) -> tuple[float, float]:
    c_value = _positive("c_days", c_days)
    p_value = _finite("p", p)
    if p_value <= 1.0:
        raise ValueError("p must be greater than 1")
    return c_value, p_value


def _validate_spatial_parameters(
    d_km2: float,
    q: float,
    gamma: float,
    cutoff_radius_km: float,
) -> tuple[float, float, float, float]:
    d_value = _positive("d_km2", d_km2)
    q_value = _finite("q", q)
    if q_value <= 1.0:
        raise ValueError("q must be greater than 1")
    gamma_value = _nonnegative("gamma", gamma)
    cutoff_value = _positive("cutoff_radius_km", cutoff_radius_km)
    return d_value, q_value, gamma_value, cutoff_value


def _log1p_ratio(numerator: float, denominator: float) -> float:
    """Return log(1 + numerator / denominator) without ratio overflow."""

    if numerator == 0.0:
        return 0.0
    log_ratio = math.log(numerator) - math.log(denominator)
    if log_ratio > 50.0:
        return log_ratio + math.log1p(math.exp(-log_ratio))
    return math.log1p(math.exp(log_ratio))


def _log1p_radius_squared_over_scale(radius_km: float, scale_km2: float) -> float:
    if radius_km == 0.0:
        return 0.0
    log_ratio = 2.0 * math.log(radius_km) - math.log(scale_km2)
    if log_ratio > 50.0:
        return log_ratio + math.log1p(math.exp(-log_ratio))
    return math.log1p(math.exp(log_ratio))


def _untruncated_inverse_power_mass(
    radius_km: float,
    *,
    scale_km2: float,
    q: float,
) -> float:
    log_base = _log1p_radius_squared_over_scale(radius_km, scale_km2)
    return -math.expm1((1.0 - q) * log_base)


def productivity(magnitude: float, *, k: float, alpha: float, mc: float) -> float:
    """Return ``K exp(alpha (m - Mc))`` for a complete parent event."""

    magnitude_value, mc_value = _validate_magnitude(magnitude, mc)
    k_value, alpha_value = _validate_productivity_parameters(k, alpha)
    if k_value == 0.0:
        return 0.0
    exponent = alpha_value * (magnitude_value - mc_value)
    _finite_result("productivity exponent", exponent)
    log_value = math.log(k_value) + exponent
    return _exp_finite("productivity", log_value)


def omori_density(delta_days: float, *, c_days: float, p: float) -> float:
    """Return the normalized infinite-support Omori--Utsu density.

    ``delta_days`` must be strictly positive, enforcing the ETAS history
    predicate ``parent_time < evaluation_time``.
    """

    delta_value = _positive("delta_days", delta_days)
    c_value, p_value = _validate_omori_parameters(c_days, p)
    log_base = _log1p_ratio(delta_value, c_value)
    log_density = math.log(p_value - 1.0) - math.log(c_value) - p_value * log_base
    return _exp_finite("Omori density", log_density)


def omori_cdf(delta_days: float, *, c_days: float, p: float) -> float:
    """Return the normalized Omori--Utsu CDF on non-negative elapsed time."""

    delta_value = _nonnegative("delta_days", delta_days)
    c_value, p_value = _validate_omori_parameters(c_days, p)
    if delta_value == 0.0:
        return 0.0
    log_base = _log1p_ratio(delta_value, c_value)
    return _finite_result("Omori CDF", -math.expm1((1.0 - p_value) * log_base))


def inverse_power_scale(
    magnitude: float,
    *,
    d_km2: float,
    gamma: float,
    mc: float,
) -> float:
    """Return ``D exp(gamma (m - Mc))`` for the isotropic spatial kernel."""

    magnitude_value, mc_value = _validate_magnitude(magnitude, mc)
    d_value = _positive("d_km2", d_km2)
    gamma_value = _nonnegative("gamma", gamma)
    exponent = gamma_value * (magnitude_value - mc_value)
    _finite_result("spatial-scale exponent", exponent)
    log_value = math.log(d_value) + exponent
    return _exp_finite("spatial scale", log_value)


def inverse_power_cutoff_mass(
    magnitude: float,
    *,
    d_km2: float,
    q: float,
    gamma: float,
    mc: float,
    cutoff_radius_km: float,
) -> float:
    """Return untruncated radial probability mass inside the cutoff radius."""

    d_value, q_value, gamma_value, cutoff_value = _validate_spatial_parameters(
        d_km2, q, gamma, cutoff_radius_km
    )
    scale = inverse_power_scale(magnitude, d_km2=d_value, gamma=gamma_value, mc=mc)
    result = _untruncated_inverse_power_mass(
        cutoff_value,
        scale_km2=scale,
        q=q_value,
    )
    if result <= 0.0:
        raise ValueError("spatial cutoff mass is numerically zero")
    return _finite_result("spatial cutoff mass", result)


def inverse_power_density(
    radius_km: float,
    magnitude: float,
    *,
    d_km2: float,
    q: float,
    gamma: float,
    mc: float,
    cutoff_radius_km: float,
) -> float:
    """Return the radially truncated and renormalized 2-D spatial density."""

    radius_value = _nonnegative("radius_km", radius_km)
    d_value, q_value, gamma_value, cutoff_value = _validate_spatial_parameters(
        d_km2, q, gamma, cutoff_radius_km
    )
    scale = inverse_power_scale(magnitude, d_km2=d_value, gamma=gamma_value, mc=mc)
    cutoff_mass = _untruncated_inverse_power_mass(
        cutoff_value,
        scale_km2=scale,
        q=q_value,
    )
    if cutoff_mass <= 0.0:
        raise ValueError("spatial cutoff mass is numerically zero")
    if radius_value > cutoff_value:
        return 0.0
    log_base = _log1p_radius_squared_over_scale(radius_value, scale)
    log_density = (
        math.log(q_value - 1.0)
        - math.log(math.pi)
        - math.log(scale)
        - q_value * log_base
        - math.log(cutoff_mass)
    )
    return _exp_finite("spatial density", log_density)


def inverse_power_mass(
    radius_km: float,
    magnitude: float,
    *,
    d_km2: float,
    q: float,
    gamma: float,
    mc: float,
    cutoff_radius_km: float,
) -> float:
    """Return radial mass of the truncated and renormalized spatial kernel."""

    radius_value = _nonnegative("radius_km", radius_km)
    d_value, q_value, gamma_value, cutoff_value = _validate_spatial_parameters(
        d_km2, q, gamma, cutoff_radius_km
    )
    scale = inverse_power_scale(magnitude, d_km2=d_value, gamma=gamma_value, mc=mc)
    cutoff_mass = _untruncated_inverse_power_mass(
        cutoff_value,
        scale_km2=scale,
        q=q_value,
    )
    if cutoff_mass <= 0.0:
        raise ValueError("spatial cutoff mass is numerically zero")
    if radius_value >= cutoff_value:
        return 1.0
    numerator = _untruncated_inverse_power_mass(
        radius_value,
        scale_km2=scale,
        q=q_value,
    )
    return _finite_result("spatial mass", numerator / cutoff_mass)


def aki_b_value(
    magnitudes: Iterable[float],
    *,
    mc: float,
    bin_width: float = 0.1,
) -> float:
    """Estimate the Aki b value with the half-bin completeness correction."""

    mc_value = _finite("mc", mc)
    bin_width_value = _positive("bin_width", bin_width)
    values: list[float] = []
    for magnitude in magnitudes:
        magnitude_value = _finite("magnitude", magnitude)
        if magnitude_value < mc_value:
            raise ValueError("all magnitudes must be greater than or equal to mc")
        values.append(magnitude_value)
    if not values:
        raise ValueError("at least one magnitude is required")
    try:
        mean_magnitude = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ValueError("mean magnitude is not finite") from error
    denominator = mean_magnitude - (mc_value - bin_width_value / 2.0)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Aki b denominator must be finite and positive")
    return _finite_result("Aki b value", math.log10(math.e) / denominator)


def _log_one_minus_exp_negative(value: float) -> float:
    """Return log(1 - exp(-value)) for a strictly positive value."""

    if math.isinf(value):
        return 0.0
    return math.log(-math.expm1(-value))


def _log_expm1_positive(value: float) -> float:
    """Return log(exp(value) - 1) without overflowing exp(value)."""

    if math.isinf(value):
        return math.inf
    if value > 50.0:
        return value + math.log1p(-math.exp(-value))
    return math.log(math.expm1(value))


def truncated_gr_exp_expectation(
    *,
    alpha: float,
    beta: float,
    magnitude_span: float,
    equality_tolerance: float = _DEFAULT_EQUALITY_TOLERANCE,
) -> float:
    """Return ``E[exp(alpha X)]`` for GR ``X`` truncated to ``[0, L]``.

    The Gutenberg--Richter density is
    ``beta exp(-beta x) / (1 - exp(-beta L))``.  The explicit continuous
    limit is used when ``alpha`` is within ``equality_tolerance`` of ``beta``.
    """

    alpha_value = _nonnegative("alpha", alpha)
    beta_value = _positive("beta", beta)
    span_value = _positive("magnitude_span", magnitude_span)
    tolerance_value = _nonnegative("equality_tolerance", equality_tolerance)
    beta_span = beta_value * span_value
    if not math.isfinite(beta_span):
        beta_span = math.inf
    log_normalizer = _log_one_minus_exp_negative(beta_span)

    if alpha_value == 0.0:
        return 1.0

    difference = alpha_value - beta_value
    if abs(difference) <= tolerance_value:
        log_integral = math.log(span_value)
    else:
        absolute_span = abs(difference) * span_value
        if not math.isfinite(absolute_span):
            absolute_span = math.inf
        if absolute_span == 0.0:
            log_integral = math.log(span_value)
        elif difference > 0.0:
            log_integral = _log_expm1_positive(absolute_span) - math.log(difference)
        else:
            log_integral = _log_one_minus_exp_negative(absolute_span) - math.log(-difference)

    log_result = math.log(beta_value) + log_integral - log_normalizer
    return _exp_finite("truncated GR exponential expectation", log_result)


def branching_ratio(
    *,
    k: float,
    alpha: float,
    beta: float,
    magnitude_span: float,
    equality_tolerance: float = _DEFAULT_EQUALITY_TOLERANCE,
) -> float:
    """Return the ETAS mean direct-offspring count under truncated GR marks."""

    k_value = _nonnegative("k", k)
    expectation = truncated_gr_exp_expectation(
        alpha=alpha,
        beta=beta,
        magnitude_span=magnitude_span,
        equality_tolerance=equality_tolerance,
    )
    return _finite_result("branching ratio", k_value * expectation)


@dataclass(frozen=True, slots=True)
class ETASParent:
    """One causal parent event in local projected coordinates."""

    time_days: float
    x_km: float
    y_km: float
    magnitude: float

    def __post_init__(self) -> None:
        _finite("parent time_days", self.time_days)
        _finite("parent x_km", self.x_km)
        _finite("parent y_km", self.y_km)
        _finite("parent magnitude", self.magnitude)


def conditional_intensity(
    *,
    time_days: float,
    x_km: float,
    y_km: float,
    background_density_per_day_km2: float,
    parents: Iterable[ETASParent],
    mc: float,
    k: float,
    alpha: float,
    c_days: float,
    p: float,
    d_km2: float,
    q: float,
    gamma: float,
    spatial_cutoff_km: float,
) -> float:
    """Evaluate unmarked ETAS conditional intensity at one space-time point.

    Every supplied parent must precede the evaluation time.  Callers are
    responsible for applying the separately frozen 3650-day history cutoff;
    silently accepting future or simultaneous events here would mask leakage.
    """

    time_value = _finite("time_days", time_days)
    x_value = _finite("x_km", x_km)
    y_value = _finite("y_km", y_km)
    background_value = _nonnegative(
        "background_density_per_day_km2", background_density_per_day_km2
    )
    mc_value = _finite("mc", mc)
    k_value, alpha_value = _validate_productivity_parameters(k, alpha)
    c_value, p_value = _validate_omori_parameters(c_days, p)
    d_value, q_value, gamma_value, cutoff_value = _validate_spatial_parameters(
        d_km2,
        q,
        gamma,
        spatial_cutoff_km,
    )

    contributions = [background_value]
    for parent in parents:
        if not isinstance(parent, ETASParent):
            raise TypeError("parents must contain ETASParent instances")
        delta_days = time_value - parent.time_days
        if not math.isfinite(delta_days) or delta_days <= 0.0:
            raise ValueError("every parent must be strictly earlier than the evaluation time")
        delta_x = x_value - parent.x_km
        delta_y = y_value - parent.y_km
        if not math.isfinite(delta_x) or not math.isfinite(delta_y):
            raise ValueError("parent-to-evaluation coordinate difference must be finite")
        radius = math.hypot(delta_x, delta_y)
        if not math.isfinite(radius):
            raise ValueError("parent-to-evaluation radius must be finite")
        contribution = (
            productivity(parent.magnitude, k=k_value, alpha=alpha_value, mc=mc_value)
            * omori_density(delta_days, c_days=c_value, p=p_value)
            * inverse_power_density(
                radius,
                parent.magnitude,
                d_km2=d_value,
                q=q_value,
                gamma=gamma_value,
                mc=mc_value,
                cutoff_radius_km=cutoff_value,
            )
        )
        contributions.append(_finite_result("parent intensity contribution", contribution))
    try:
        result = math.fsum(contributions)
    except OverflowError as error:
        raise ValueError("conditional intensity is not finite") from error
    return _finite_result("conditional intensity", result)
