"""Pure time-count and magnitude baselines frozen for S1-B.

This module deliberately does not construct forecast windows, horizons,
episodes, or catalog subsets.  Callers must pass already causal expanding
history, already non-overlapping historical count blocks, and unique physical
target events.  The scalar scoring functions score exactly the supplied item
once and never duplicate it across horizons.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from numbers import Integral
from typing import Final, Literal

from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import digamma, polygamma  # type: ignore[import-untyped]

from seismoflux.background.etas import aki_b_value

MAIN_GR_MC: Final = 4.0
LONG_M5_GR_MC: Final = 5.0
GR_BIN_WIDTH: Final = 0.1
GR_MAXIMUM_MAGNITUDE: Final = 9.5

_NB2_BRACKET_EXPANSIONS: Final = 128
_NB2_ROOT_MAX_ITERATIONS: Final = 200
_NB2_ROOT_ABSOLUTE_TOLERANCE: Final = 1.0e-12
_NB2_ROOT_RELATIVE_TOLERANCE: Final = 1.0e-12
_GR_MASS_SUM_TOLERANCE: Final = 1.0e-12

NB2Status = Literal["evaluable", "poisson_limit", "not_evaluable"]


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


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class ExpandingPoissonRate:
    """One causal expanding-history Poisson rate estimate."""

    history_event_count: int
    history_exposure_days: float
    rate_per_day: float

    def expected_count(self, horizon_days: float) -> float:
        return poisson_expected_count(self.rate_per_day, horizon_days)

    def at_least_one_probability(self, horizon_days: float) -> float:
        return poisson_at_least_one_probability(self.expected_count(horizon_days))

    def logpmf(self, observed_count: int, horizon_days: float) -> float:
        return poisson_logpmf(observed_count, self.expected_count(horizon_days))


def fit_expanding_poisson(
    *, history_event_count: int, history_exposure_days: float
) -> ExpandingPoissonRate:
    """Fit ``rate = count / exposure`` from caller-supplied causal history."""

    count = _count("history_event_count", history_event_count)
    exposure = _positive("history_exposure_days", history_exposure_days)
    return ExpandingPoissonRate(
        history_event_count=count,
        history_exposure_days=exposure,
        rate_per_day=count / exposure,
    )


def poisson_expected_count(rate_per_day: float, horizon_days: float) -> float:
    """Return the Poisson expectation for one caller-supplied horizon."""

    rate = _nonnegative("rate_per_day", rate_per_day)
    horizon = _positive("horizon_days", horizon_days)
    result = rate * horizon
    if not math.isfinite(result):
        raise ValueError("Poisson expected count must be finite")
    return result


def poisson_at_least_one_probability(expected_count: float) -> float:
    """Return ``P(N >= 1)`` without cancellation for a small mean."""

    mean = _nonnegative("expected_count", expected_count)
    return -math.expm1(-mean)


def poisson_logpmf(observed_count: int, expected_count: float) -> float:
    """Stable Poisson log-PMF, including the exact ``count=mean=0`` case."""

    count = _count("observed_count", observed_count)
    mean = _nonnegative("expected_count", expected_count)
    if mean == 0.0:
        return 0.0 if count == 0 else -math.inf
    return count * math.log(mean) - mean - math.lgamma(count + 1.0)


@dataclass(frozen=True, slots=True)
class NB2DispersionQualification:
    """Qualification of the one extra NB2 dispersion parameter ``k``.

    ``poisson_limit`` means the caller should score with the Poisson PMF.
    ``not_evaluable`` is missing evidence, never a negative model score.
    """

    status: NB2Status
    reason: str
    historical_block_count: int
    sample_mean_count: float
    sample_variance_count: float | None
    dispersion_k: float | None = None
    observed_information_k: float | None = None
    standard_error_k: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"evaluable", "poisson_limit", "not_evaluable"}:
            raise ValueError("unknown NB2 qualification status")
        if not self.reason.strip():
            raise ValueError("NB2 qualification reason must be non-empty")
        if self.status == "evaluable":
            for name, value in (
                ("dispersion_k", self.dispersion_k),
                ("observed_information_k", self.observed_information_k),
                ("standard_error_k", self.standard_error_k),
            ):
                if value is None or not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"evaluable NB2 {name} must be finite and positive")
        elif any(
            value is not None
            for value in (self.dispersion_k, self.observed_information_k, self.standard_error_k)
        ):
            raise ValueError("non-evaluable NB2 results must not contain fitted parameters")


def nb2_logpmf(observed_count: int, expected_count: float, dispersion_k: float) -> float:
    """Return the stable NB2 log-PMF with ``Var(N)=mu+mu^2/k``."""

    count = _count("observed_count", observed_count)
    mean = _nonnegative("expected_count", expected_count)
    k = _positive("dispersion_k", dispersion_k)
    if mean == 0.0:
        return 0.0 if count == 0 else -math.inf
    log_one_plus_mean_over_k = math.log1p(mean / k)
    log_rising_factorial = math.fsum(math.log(k + offset) for offset in range(count))
    return (
        log_rising_factorial
        - math.lgamma(count + 1.0)
        - k * log_one_plus_mean_over_k
        + count * (math.log(mean) - math.log(k) - log_one_plus_mean_over_k)
    )


def nb2_at_least_one_probability(expected_count: float, dispersion_k: float) -> float:
    """Return NB2 ``P(N >= 1)`` stably for small means and large ``k``."""

    mean = _nonnegative("expected_count", expected_count)
    k = _positive("dispersion_k", dispersion_k)
    log_zero_probability = -k * math.log1p(mean / k)
    return -math.expm1(log_zero_probability)


def qualified_nb2_logpmf(
    observed_count: int,
    expected_count: float,
    qualification: NB2DispersionQualification,
) -> float:
    """Score one supplied count, respecting the frozen qualification branch."""

    if qualification.status == "not_evaluable":
        raise ValueError("an NB2 model marked not_evaluable must not be scored")
    if qualification.status == "poisson_limit":
        return poisson_logpmf(observed_count, expected_count)
    assert qualification.dispersion_k is not None
    return nb2_logpmf(observed_count, expected_count, qualification.dispersion_k)


def _nb2_score_k(counts: tuple[int, ...], means: tuple[float, ...], k: float) -> float:
    terms = []
    for count, mean in zip(counts, means, strict=True):
        terms.append(
            float(digamma(count + k) - digamma(k))
            + math.log(k)
            + 1.0
            - math.log(k + mean)
            - (k + count) / (k + mean)
        )
    return math.fsum(terms)


def _nb2_observed_information_k(
    counts: tuple[int, ...], means: tuple[float, ...], k: float
) -> float:
    second_derivative_terms = []
    for count, mean in zip(counts, means, strict=True):
        second_derivative_terms.append(
            float(polygamma(1, count + k) - polygamma(1, k))
            + 1.0 / k
            - 1.0 / (k + mean)
            + (count - mean) / ((k + mean) * (k + mean))
        )
    return -math.fsum(second_derivative_terms)


def _not_evaluable_nb2(
    *,
    reason: str,
    block_count: int,
    sample_mean: float,
    sample_variance: float | None,
) -> NB2DispersionQualification:
    return NB2DispersionQualification(
        status="not_evaluable",
        reason=reason,
        historical_block_count=block_count,
        sample_mean_count=sample_mean,
        sample_variance_count=sample_variance,
    )


def fit_nb2_dispersion(
    non_overlapping_history_counts: Sequence[int],
    poisson_expected_counts: Sequence[float],
) -> NB2DispersionQualification:
    """Fit NB2 ``k`` using only caller-supplied non-overlapping history blocks.

    The function never constructs, overlaps, filters, or selects blocks.  Each
    supplied expected count must come from the same causal Poisson rate used by
    the comparator.  Sample variance no larger than sample mean is the frozen
    ``poisson_limit`` branch.  A finite MLE and positive observed information
    are both required for ``evaluable``.
    """

    counts = tuple(
        _count("historical block count", value) for value in non_overlapping_history_counts
    )
    means = tuple(
        _nonnegative("historical Poisson expected count", value)
        for value in poisson_expected_counts
    )
    if len(counts) != len(means):
        raise ValueError("historical counts and Poisson expectations must have one common length")
    if not counts:
        return _not_evaluable_nb2(
            reason="no_non_overlapping_history_blocks",
            block_count=0,
            sample_mean=0.0,
            sample_variance=None,
        )
    sample_mean = math.fsum(counts) / len(counts)
    if len(counts) < 2:
        return _not_evaluable_nb2(
            reason="fewer_than_two_non_overlapping_history_blocks",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=None,
        )
    sample_variance = math.fsum((value - sample_mean) ** 2 for value in counts) / (len(counts) - 1)
    if any(count > 0 and mean == 0.0 for count, mean in zip(counts, means, strict=True)):
        return _not_evaluable_nb2(
            reason="positive_count_with_zero_poisson_expectation",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
    if sample_variance <= sample_mean:
        return NB2DispersionQualification(
            status="poisson_limit",
            reason="sample_variance_not_greater_than_sample_mean",
            historical_block_count=len(counts),
            sample_mean_count=sample_mean,
            sample_variance_count=sample_variance,
        )

    moment_k = sample_mean * sample_mean / (sample_variance - sample_mean)
    if not math.isfinite(moment_k) or moment_k <= 0.0:
        return _not_evaluable_nb2(
            reason="nonfinite_method_of_moments_start",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )

    lower = moment_k
    upper = moment_k
    try:
        lower_score = _nb2_score_k(counts, means, lower)
        upper_score = lower_score
        for _ in range(_NB2_BRACKET_EXPANSIONS):
            if lower_score > 0.0:
                break
            lower /= 2.0
            if lower == 0.0 or not math.isfinite(lower):
                break
            lower_score = _nb2_score_k(counts, means, lower)
        for _ in range(_NB2_BRACKET_EXPANSIONS):
            if upper_score < 0.0:
                break
            upper *= 2.0
            if not math.isfinite(upper):
                break
            upper_score = _nb2_score_k(counts, means, upper)
    except (OverflowError, ValueError, ZeroDivisionError):
        return _not_evaluable_nb2(
            reason="nonfinite_nb2_score_during_bracketing",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
    if not (
        math.isfinite(lower_score)
        and math.isfinite(upper_score)
        and lower_score > 0.0
        and upper_score < 0.0
        and 0.0 < lower < upper
    ):
        return _not_evaluable_nb2(
            reason="finite_dispersion_mle_not_bracketed",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )

    try:
        fitted_raw, root_result = brentq(
            lambda value: _nb2_score_k(counts, means, value),
            lower,
            upper,
            xtol=_NB2_ROOT_ABSOLUTE_TOLERANCE,
            rtol=_NB2_ROOT_RELATIVE_TOLERANCE,
            maxiter=_NB2_ROOT_MAX_ITERATIONS,
            full_output=True,
            disp=False,
        )
        fitted_k = float(fitted_raw)
    except (OverflowError, RuntimeError, ValueError, ZeroDivisionError):
        return _not_evaluable_nb2(
            reason="nb2_maximum_likelihood_did_not_converge",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
    if not bool(root_result.converged) or not math.isfinite(fitted_k) or fitted_k <= 0.0:
        return _not_evaluable_nb2(
            reason="nb2_maximum_likelihood_did_not_converge",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
    information = _nb2_observed_information_k(counts, means, fitted_k)
    if not math.isfinite(information) or information <= 0.0:
        return _not_evaluable_nb2(
            reason="nb2_observed_information_not_positive_finite",
            block_count=len(counts),
            sample_mean=sample_mean,
            sample_variance=sample_variance,
        )
    return NB2DispersionQualification(
        status="evaluable",
        reason="finite_mle_and_positive_observed_information",
        historical_block_count=len(counts),
        sample_mean_count=sample_mean,
        sample_variance_count=sample_variance,
        dispersion_k=fitted_k,
        observed_information_k=information,
        standard_error_k=1.0 / math.sqrt(information),
    )


@dataclass(frozen=True, slots=True)
class TruncatedGRMagnitudeModel:
    """Bin-normalized truncated Gutenberg--Richter magnitude model."""

    model_id: Literal["M0_GR_GLOBAL", "M3_GR_LONG_M5"]
    training_event_count: int
    mc: float
    maximum_magnitude: float
    bin_width: float
    b_value: float
    beta: float
    bin_lower_edges: tuple[float, ...]
    bin_upper_edges: tuple[float, ...]
    bin_probability_masses: tuple[float, ...]
    m5_6_probability_mass: float
    m6_plus_probability_mass: float

    def log_probability(self, magnitude: float) -> float:
        """Score one caller-supplied unique physical event exactly once."""

        value = _finite("magnitude", magnitude)
        if value < self.mc or value > self.maximum_magnitude:
            raise ValueError("magnitude is outside the frozen truncated GR support")
        if value == self.maximum_magnitude:
            index = len(self.bin_probability_masses) - 1
        else:
            decimal_offset = (Decimal(str(value)) - Decimal(str(self.mc))) / Decimal(
                str(self.bin_width)
            )
            index = int(decimal_offset.to_integral_value(rounding=ROUND_FLOOR))
            index = min(max(index, 0), len(self.bin_probability_masses) - 1)
        return math.log(self.bin_probability_masses[index])


def _truncated_gr_interval_mass(
    *, beta: float, mc: float, maximum_magnitude: float, lower: float, upper: float
) -> float:
    if not mc <= lower < upper <= maximum_magnitude:
        raise ValueError("GR interval must lie inside the frozen support")
    span = maximum_magnitude - mc
    normalizer = -math.expm1(-beta * span)
    lower_survival = math.exp(-beta * (lower - mc))
    width_mass = -math.expm1(-beta * (upper - lower))
    return lower_survival * width_mass / normalizer


def _fit_truncated_gr(
    magnitudes: Iterable[float],
    *,
    model_id: Literal["M0_GR_GLOBAL", "M3_GR_LONG_M5"],
    mc: float,
) -> TruncatedGRMagnitudeModel:
    values = tuple(_finite("magnitude", value) for value in magnitudes)
    if not values:
        raise ValueError("at least one causal training magnitude is required")
    if any(value < mc or value > GR_MAXIMUM_MAGNITUDE for value in values):
        raise ValueError("all training magnitudes must lie inside the frozen GR support")
    b_value = aki_b_value(values, mc=mc, bin_width=GR_BIN_WIDTH)
    beta = b_value * math.log(10.0)
    span_in_bins = (GR_MAXIMUM_MAGNITUDE - mc) / GR_BIN_WIDTH
    bin_count = round(span_in_bins)
    if not math.isclose(span_in_bins, bin_count, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("frozen GR support must contain an integer number of bins")
    lower_edges = tuple(mc + index * GR_BIN_WIDTH for index in range(bin_count))
    upper_edges_list = [mc + (index + 1) * GR_BIN_WIDTH for index in range(bin_count)]
    upper_edges_list[-1] = GR_MAXIMUM_MAGNITUDE
    upper_edges = tuple(upper_edges_list)
    masses = tuple(
        _truncated_gr_interval_mass(
            beta=beta,
            mc=mc,
            maximum_magnitude=GR_MAXIMUM_MAGNITUDE,
            lower=lower,
            upper=upper,
        )
        for lower, upper in zip(lower_edges, upper_edges, strict=True)
    )
    if any(not math.isfinite(mass) or mass <= 0.0 for mass in masses):
        raise ValueError("every frozen GR bin must have positive finite mass")
    if not math.isclose(math.fsum(masses), 1.0, rel_tol=0.0, abs_tol=_GR_MASS_SUM_TOLERANCE):
        raise ValueError("frozen GR bin masses must sum to one")
    m5_6_mass = _truncated_gr_interval_mass(
        beta=beta,
        mc=mc,
        maximum_magnitude=GR_MAXIMUM_MAGNITUDE,
        lower=5.0,
        upper=6.0,
    )
    m6_plus_mass = _truncated_gr_interval_mass(
        beta=beta,
        mc=mc,
        maximum_magnitude=GR_MAXIMUM_MAGNITUDE,
        lower=6.0,
        upper=GR_MAXIMUM_MAGNITUDE,
    )
    return TruncatedGRMagnitudeModel(
        model_id=model_id,
        training_event_count=len(values),
        mc=mc,
        maximum_magnitude=GR_MAXIMUM_MAGNITUDE,
        bin_width=GR_BIN_WIDTH,
        b_value=b_value,
        beta=beta,
        bin_lower_edges=lower_edges,
        bin_upper_edges=upper_edges,
        bin_probability_masses=masses,
        m5_6_probability_mass=m5_6_mass,
        m6_plus_probability_mass=m6_plus_mass,
    )


def fit_m0_gr_global(magnitudes: Iterable[float]) -> TruncatedGRMagnitudeModel:
    """Fit the frozen 1970+ ``Mc=4`` truncated-GR main baseline."""

    return _fit_truncated_gr(magnitudes, model_id="M0_GR_GLOBAL", mc=MAIN_GR_MC)


def fit_m3_gr_long_m5(magnitudes: Iterable[float]) -> TruncatedGRMagnitudeModel:
    """Fit the frozen 1900+ ``Mc=5`` tail sensitivity model."""

    return _fit_truncated_gr(magnitudes, model_id="M3_GR_LONG_M5", mc=LONG_M5_GR_MC)


__all__ = [
    "GR_BIN_WIDTH",
    "GR_MAXIMUM_MAGNITUDE",
    "LONG_M5_GR_MC",
    "MAIN_GR_MC",
    "ExpandingPoissonRate",
    "NB2DispersionQualification",
    "NB2Status",
    "TruncatedGRMagnitudeModel",
    "fit_expanding_poisson",
    "fit_m0_gr_global",
    "fit_m3_gr_long_m5",
    "fit_nb2_dispersion",
    "nb2_at_least_one_probability",
    "nb2_logpmf",
    "poisson_at_least_one_probability",
    "poisson_expected_count",
    "poisson_logpmf",
    "qualified_nb2_logpmf",
]
