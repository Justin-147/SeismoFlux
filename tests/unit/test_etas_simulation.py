from __future__ import annotations

import math

import numpy as np
import pytest

from seismoflux.background.etas_fit import ETASEvent, ETASModelSpec, ETASParameters
from seismoflux.background.etas_simulation import (
    EventCapExceeded,
    SimulationDomain,
    sample_inverse_power_offset,
    sample_omori_elapsed_time,
    sample_truncated_gr_magnitude,
    simulate_future_catalog,
)
from seismoflux.background.randomness import SeedContext


def _parameters(*, background_rate: float = 0.2) -> ETASParameters:
    return ETASParameters(background_rate, 0.2, 1.0, 0.05, 1.2)


def _spec() -> ETASModelSpec:
    return ETASModelSpec(mc=3.0, beta=2.0)


def _domain() -> SimulationDomain:
    def classify(x_km: float, y_km: float) -> str:
        radius = math.hypot(x_km, y_km)
        if radius <= 25.0:
            return "study"
        if radius <= 325.0:
            return "buffer"
        return "outside"

    return SimulationDomain(
        classify=classify,  # type: ignore[arg-type]
        sample_background_location=lambda _generator: (0.0, 0.0),
    )


def _context(replicate: int = 0) -> SeedContext:
    return SeedContext(
        root_seed=147,
        protocol_version="0.2.0",
        namespace="future_simulation",
        model_id="etas/final_validation",
        issue_id="validation/2025-06-26",
        replicate_index=replicate,
    )


def test_inverse_cdf_samplers_respect_frozen_supports() -> None:
    generator = np.random.Generator(np.random.PCG64(1234))
    spec = _spec()
    magnitudes = [
        sample_truncated_gr_magnitude(
            generator,
            mc=spec.mc,
            maximum_magnitude=spec.maximum_magnitude,
            beta=spec.beta,
        )
        for _ in range(1000)
    ]
    elapsed = [
        sample_omori_elapsed_time(
            generator,
            lower_days=0.5,
            upper_days=2.0,
            c_days=0.05,
            p=1.2,
        )
        for _ in range(1000)
    ]
    offsets = [sample_inverse_power_offset(generator, 4.0, spec=spec) for _ in range(1000)]

    assert all(spec.mc <= value <= spec.maximum_magnitude for value in magnitudes)
    assert all(0.5 < value <= 2.0 for value in elapsed)
    assert all(math.hypot(x_km, y_km) <= 300.0 for x_km, y_km in offsets)


def test_future_catalog_is_deterministic_study_only_and_reused_across_horizons() -> None:
    history = (
        ETASEvent("inside-history", 0.0, 0.0, 0.0, 0.0, 4.0, True, True),
        ETASEvent("buffer-history", -1.0, -1.0, 100.0, 0.0, 4.0, False, True),
    )
    first = simulate_future_catalog(
        _parameters(),
        _spec(),
        history,
        _domain(),
        horizon_days=365.0,
        seed_context=_context(),
    )
    again = simulate_future_catalog(
        _parameters(),
        _spec(),
        history,
        _domain(),
        horizon_days=365.0,
        seed_context=_context(),
    )

    assert first == again
    assert first.events
    assert all(_domain().membership(event.x_km, event.y_km) == "study" for event in first.events)
    assert all(0.0 < event.time_days <= 365.0 for event in first.events)
    assert all(3.0 <= event.magnitude <= 9.5 for event in first.events)
    event_order = [
        (
            event.time_days,
            event.generation,
            event.parent_id or "",
            event.within_parent_child_index,
        )
        for event in first.events
    ]
    assert event_order == sorted(event_order)
    horizon_counts = [len(first.through(days)) for days in (7.0, 30.0, 90.0, 180.0, 365.0)]
    assert horizon_counts == sorted(horizon_counts)


def test_parent_older_than_3650_days_is_omitted_without_rng_or_kernel_renormalization() -> None:
    no_history = simulate_future_catalog(
        _parameters(background_rate=0.01),
        _spec(),
        (),
        _domain(),
        horizon_days=365.0,
        seed_context=_context(3),
    )
    with_old_history = simulate_future_catalog(
        _parameters(background_rate=0.01),
        _spec(),
        (ETASEvent("old", -4000.0, -4000.0, 0.0, 0.0, 9.0, True, True),),
        _domain(),
        horizon_days=365.0,
        seed_context=_context(3),
    )

    assert with_old_history == no_history


def test_simulated_buffer_descendants_remain_parents_but_are_not_output_rows() -> None:
    parameters = ETASParameters(0.01, 0.4, 1.0, 0.05, 1.2)
    history = (ETASEvent("buffer-root", 0.0, 0.0, 30.0, 0.0, 9.5, False, True),)
    result = simulate_future_catalog(
        parameters,
        _spec(),
        history,
        _domain(),
        horizon_days=365.0,
        seed_context=_context(),
    )
    output_ids = {event.event_id for event in result.events}
    returned_from_hidden_buffer_parent = [
        event
        for event in result.events
        if event.parent_id is not None
        and event.parent_id != "buffer-root"
        and event.parent_id not in output_ids
    ]

    assert returned_from_hidden_buffer_parent
    assert all(_domain().membership(event.x_km, event.y_km) == "study" for event in result.events)


def test_future_simulation_hard_fails_on_event_cap_and_supercritical_parameters() -> None:
    with pytest.raises(EventCapExceeded, match="maximum_events"):
        simulate_future_catalog(
            _parameters(background_rate=10.0),
            _spec(),
            (),
            _domain(),
            horizon_days=365.0,
            seed_context=_context(7),
            maximum_events=1,
        )

    supercritical = ETASParameters(0.1, 0.5, 2.0, 0.05, 1.2)
    with pytest.raises(ValueError, match="branching ratio"):
        simulate_future_catalog(
            supercritical,
            _spec(),
            (),
            _domain(),
            horizon_days=365.0,
            seed_context=_context(8),
        )


def test_future_simulation_rejects_future_history_and_wrong_seed_namespace() -> None:
    future = ETASEvent("future", 0.1, 0.1, 0.0, 0.0, 4.0, True, True)
    with pytest.raises(ValueError, match="at or before"):
        simulate_future_catalog(
            _parameters(),
            _spec(),
            (future,),
            _domain(),
            horizon_days=365.0,
            seed_context=_context(),
        )

    unavailable = ETASEvent("unavailable", -1.0, 0.1, 0.0, 0.0, 4.0, True, True)
    with pytest.raises(ValueError, match="available at or before"):
        simulate_future_catalog(
            _parameters(),
            _spec(),
            (unavailable,),
            _domain(),
            horizon_days=365.0,
            seed_context=_context(),
        )

    available_at_issue = ETASEvent("available-at-issue", -4000.0, 0.0, 0.0, 0.0, 4.0, True, True)
    simulate_future_catalog(
        _parameters(background_rate=0.01),
        _spec(),
        (available_at_issue,),
        _domain(),
        horizon_days=365.0,
        seed_context=_context(2),
    )

    wrong_context = SeedContext(147, "0.2.0", "bootstrap", "etas/final_validation", None, 0)
    with pytest.raises(ValueError, match="namespace"):
        simulate_future_catalog(
            _parameters(),
            _spec(),
            (),
            _domain(),
            horizon_days=365.0,
            seed_context=wrong_context,
        )
