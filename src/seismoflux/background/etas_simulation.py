"""Deterministic future ETAS branching simulation with buffer propagation."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np

from seismoflux.background.etas import (
    inverse_power_cutoff_mass,
    inverse_power_scale,
    omori_cdf,
    productivity,
)
from seismoflux.background.etas_fit import ETASEvent, ETASModelSpec, ETASParameters
from seismoflux.background.randomness import SeedContext

DomainClass = Literal["study", "buffer", "outside"]
DomainClassifier = Callable[[float, float], DomainClass]
BackgroundSampler = Callable[[np.random.Generator], tuple[float, float]]


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class SimulationDomain:
    """Injected target-independent study and 300-km-buffer geometry."""

    classify: DomainClassifier
    sample_background_location: BackgroundSampler

    def membership(self, x_km: float, y_km: float) -> DomainClass:
        x_value = _finite("simulation x_km", x_km)
        y_value = _finite("simulation y_km", y_km)
        membership = self.classify(x_value, y_value)
        if membership not in {"study", "buffer", "outside"}:
            raise ValueError("domain classifier must return study, buffer, or outside")
        return membership

    def sample_immigrant_location(self, generator: np.random.Generator) -> tuple[float, float]:
        x_km, y_km = self.sample_background_location(generator)
        x_value = _finite("background immigrant x_km", x_km)
        y_value = _finite("background immigrant y_km", y_km)
        if self.membership(x_value, y_value) != "study":
            raise ValueError("background immigrant sampler must return a study-area location")
        return x_value, y_value


@dataclass(frozen=True, slots=True)
class SimulatedEvent:
    """One future study-area event safe to expose to forecasts and metrics."""

    event_id: str
    time_days: float
    x_km: float
    y_km: float
    magnitude: float
    generation: int
    parent_id: str | None
    within_parent_child_index: int


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """A single longest-horizon catalog; buffer event records are not exposed."""

    horizon_days: float
    events: tuple[SimulatedEvent, ...]

    def through(self, horizon_days: float) -> tuple[SimulatedEvent, ...]:
        """Slice the same catalog for a shorter horizon without resimulation."""

        horizon = _finite("slice horizon_days", horizon_days)
        if horizon < 0.0 or horizon > self.horizon_days:
            raise ValueError("slice horizon must lie within the simulated horizon")
        return tuple(event for event in self.events if event.time_days <= horizon)


class EventCapExceeded(RuntimeError):
    """Raised when a replicate reaches the preregistered hard event cap."""


@dataclass(frozen=True, slots=True)
class _PropagationEvent:
    event_id: str
    time_days: float
    x_km: float
    y_km: float
    magnitude: float
    generation: int
    parent_id: str | None
    within_parent_child_index: int
    membership: Literal["study", "buffer"]


def sample_truncated_gr_magnitude(
    generator: np.random.Generator,
    *,
    mc: float,
    maximum_magnitude: float,
    beta: float,
) -> float:
    """Draw one magnitude from the frozen truncated Gutenberg--Richter law."""

    mc_value = _finite("mc", mc)
    maximum = _finite("maximum_magnitude", maximum_magnitude)
    if maximum <= mc_value:
        raise ValueError("maximum_magnitude must exceed mc")
    beta_value = _positive("beta", beta)
    span = maximum - mc_value
    normalizer = -math.expm1(-beta_value * span)
    uniform = float(generator.random())
    offset = -math.log1p(-uniform * normalizer) / beta_value
    magnitude = mc_value + offset
    return min(maximum, max(mc_value, magnitude))


def sample_omori_elapsed_time(
    generator: np.random.Generator,
    *,
    lower_days: float,
    upper_days: float,
    c_days: float,
    p: float,
) -> float:
    """Draw normalized Omori elapsed time conditional on a finite interval."""

    lower = _finite("lower_days", lower_days)
    upper = _finite("upper_days", upper_days)
    if lower < 0.0 or upper <= lower:
        raise ValueError("Omori sampling interval must satisfy 0 <= lower < upper")
    lower_mass = omori_cdf(lower, c_days=c_days, p=p)
    upper_mass = omori_cdf(upper, c_days=c_days, p=p)
    if not upper_mass > lower_mass:
        raise ValueError("Omori sampling interval has no representable probability mass")
    uniform = max(float(generator.random()), math.nextafter(0.0, 1.0))
    probability = lower_mass + uniform * (upper_mass - lower_mass)
    if probability <= lower_mass:
        probability = math.nextafter(lower_mass, upper_mass)
    probability = min(probability, math.nextafter(1.0, 0.0))
    elapsed = c_days * math.expm1(math.log1p(-probability) / (1.0 - p))
    if elapsed <= lower:
        elapsed = math.nextafter(lower, upper)
    if not lower < elapsed <= upper or not math.isfinite(elapsed):
        raise ValueError("sampled Omori elapsed time is outside its conditioning interval")
    return elapsed


def sample_inverse_power_offset(
    generator: np.random.Generator,
    magnitude: float,
    *,
    spec: ETASModelSpec,
) -> tuple[float, float]:
    """Draw one isotropic offset from the 300-km-renormalized spatial kernel."""

    scale = inverse_power_scale(
        magnitude,
        d_km2=spec.d_km2,
        gamma=spec.gamma,
        mc=spec.mc,
    )
    cutoff_mass = inverse_power_cutoff_mass(
        magnitude,
        d_km2=spec.d_km2,
        q=spec.q,
        gamma=spec.gamma,
        mc=spec.mc,
        cutoff_radius_km=spec.spatial_cutoff_km,
    )
    target_mass = float(generator.random()) * cutoff_mass
    exponent = math.log1p(-target_mass) / (1.0 - spec.q)
    radius = math.sqrt(scale * math.expm1(exponent))
    radius = min(radius, spec.spatial_cutoff_km)
    angle = 2.0 * math.pi * float(generator.random())
    return radius * math.cos(angle), radius * math.sin(angle)


def _stable_key(event: _PropagationEvent) -> tuple[float, int, str, int]:
    return (
        event.time_days,
        event.generation,
        event.parent_id or "",
        event.within_parent_child_index,
    )


def _output_key(event: SimulatedEvent) -> tuple[float, int, str, int]:
    return (
        event.time_days,
        event.generation,
        event.parent_id or "",
        event.within_parent_child_index,
    )


def _history_parent(event: ETASEvent, domain: SimulationDomain) -> _PropagationEvent:
    if event.time_days > 0.0:
        raise ValueError("history parent times must be at or before the issue time")
    if event.available_time_days > 0.0:
        raise ValueError("history parent must be available at or before the issue time")
    membership = domain.membership(event.x_km, event.y_km)
    expected: Literal["study", "buffer"] = "study" if event.inside_study_area else "buffer"
    if membership != expected or not event.inside_parent_domain:
        raise ValueError("history event domain flags disagree with injected simulation geometry")
    return _PropagationEvent(
        event_id=event.event_id,
        time_days=event.time_days,
        x_km=event.x_km,
        y_km=event.y_km,
        magnitude=event.magnitude,
        generation=-1,
        parent_id=None,
        within_parent_child_index=0,
        membership=expected,
    )


def simulate_future_catalog(
    parameters: ETASParameters,
    spec: ETASModelSpec,
    history_events: Sequence[ETASEvent],
    domain: SimulationDomain,
    *,
    horizon_days: float,
    seed_context: SeedContext,
    maximum_events: int = 100_000,
) -> SimulationResult:
    """Simulate one future catalog with absorbing propagation-domain boundary.

    Background immigrants occur only in the study area.  Descendants retained
    in the 300-km buffer continue to trigger, but their records never appear in
    the returned catalog.  Every attempted future event counts toward the hard
    cap, including an event absorbed outside the propagation domain.
    """

    horizon = _positive("horizon_days", horizon_days)
    if (
        not isinstance(maximum_events, int)
        or isinstance(maximum_events, bool)
        or maximum_events <= 0
    ):
        raise ValueError("maximum_events must be a positive integer")
    if seed_context.namespace != "future_simulation":
        raise ValueError("future simulation requires the future_simulation RNG namespace")
    if seed_context.root_seed != 147 or seed_context.protocol_version != "0.2.0":
        raise ValueError("future simulation seed must use the frozen root and protocol version")
    if seed_context.model_id != "etas/final_validation":
        raise ValueError("future simulation model_id must be etas/final_validation")
    if seed_context.issue_id is None or not seed_context.issue_id.startswith("validation/"):
        raise ValueError("future simulation issue_id must use validation/YYYY-MM-DD")
    try:
        date.fromisoformat(seed_context.issue_id.removeprefix("validation/"))
    except ValueError as error:
        raise ValueError("future simulation issue_id must use validation/YYYY-MM-DD") from error
    if not 0 <= seed_context.replicate_index <= 127:
        raise ValueError("future simulation replicate index must be in 0..127")
    if horizon != 365.0:
        raise ValueError("future simulation must run once at the frozen 365-day horizon")
    spec.validate_parameters(parameters)
    for event in history_events:
        spec.validate_event(event)

    generator = seed_context.generator()
    queue: list[tuple[tuple[float, int, str, int], int, _PropagationEvent]] = []
    serial = 0
    for event in sorted(history_events, key=lambda item: (item.time_days, item.event_id)):
        parent = _history_parent(event, domain)
        heapq.heappush(queue, (_stable_key(parent), serial, parent))
        serial += 1

    attempted_events = 0
    immigrant_count = int(generator.poisson(parameters.background_rate_per_day * horizon))
    if immigrant_count > maximum_events:
        raise EventCapExceeded("background immigrants exceed maximum_events")
    for immigrant_index in range(immigrant_count):
        attempted_events += 1
        uniform_time = max(float(generator.random()), math.nextafter(0.0, 1.0))
        time_days = uniform_time * horizon
        x_km, y_km = domain.sample_immigrant_location(generator)
        immigrant = _PropagationEvent(
            event_id=(f"sim/{seed_context.replicate_index:08d}/immigrant/{immigrant_index:08d}"),
            time_days=time_days,
            x_km=x_km,
            y_km=y_km,
            magnitude=sample_truncated_gr_magnitude(
                generator,
                mc=spec.mc,
                maximum_magnitude=spec.maximum_magnitude,
                beta=spec.beta,
            ),
            generation=0,
            parent_id=None,
            within_parent_child_index=immigrant_index,
            membership="study",
        )
        heapq.heappush(queue, (_stable_key(immigrant), serial, immigrant))
        serial += 1

    study_events: list[SimulatedEvent] = []
    while queue:
        _, _, parent = heapq.heappop(queue)
        if parent.generation >= 0 and parent.membership == "study":
            study_events.append(
                SimulatedEvent(
                    event_id=parent.event_id,
                    time_days=parent.time_days,
                    x_km=parent.x_km,
                    y_km=parent.y_km,
                    magnitude=parent.magnitude,
                    generation=parent.generation,
                    parent_id=parent.parent_id,
                    within_parent_child_index=parent.within_parent_child_index,
                )
            )

        lower_elapsed = max(0.0, -parent.time_days)
        upper_elapsed = min(
            spec.history_parent_cutoff_days,
            horizon - parent.time_days,
        )
        if upper_elapsed <= lower_elapsed:
            continue
        lower_mass = omori_cdf(
            lower_elapsed,
            c_days=parameters.c_days,
            p=parameters.p,
        )
        upper_mass = omori_cdf(
            upper_elapsed,
            c_days=parameters.c_days,
            p=parameters.p,
        )
        mean_children = productivity(
            parent.magnitude,
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            mc=spec.mc,
        ) * (upper_mass - lower_mass)
        if mean_children < 0.0 or not math.isfinite(mean_children):
            raise ValueError("future direct-offspring mean must be finite and non-negative")
        child_count = int(generator.poisson(mean_children))
        if attempted_events + child_count > maximum_events:
            raise EventCapExceeded("future branching process reached maximum_events")
        attempted_events += child_count
        for child_index in range(child_count):
            elapsed = sample_omori_elapsed_time(
                generator,
                lower_days=lower_elapsed,
                upper_days=upper_elapsed,
                c_days=parameters.c_days,
                p=parameters.p,
            )
            child_time = parent.time_days + elapsed
            offset_x, offset_y = sample_inverse_power_offset(
                generator,
                parent.magnitude,
                spec=spec,
            )
            child_x = parent.x_km + offset_x
            child_y = parent.y_km + offset_y
            membership = domain.membership(child_x, child_y)
            magnitude = sample_truncated_gr_magnitude(
                generator,
                mc=spec.mc,
                maximum_magnitude=spec.maximum_magnitude,
                beta=spec.beta,
            )
            if membership == "outside":
                continue
            child = _PropagationEvent(
                event_id=f"{parent.event_id}/child/{child_index:08d}",
                time_days=child_time,
                x_km=child_x,
                y_km=child_y,
                magnitude=magnitude,
                generation=parent.generation + 1,
                parent_id=parent.event_id,
                within_parent_child_index=child_index,
                membership=membership,
            )
            heapq.heappush(queue, (_stable_key(child), serial, child))
            serial += 1

    study_events.sort(key=_output_key)
    return SimulationResult(horizon_days=horizon, events=tuple(study_events))
