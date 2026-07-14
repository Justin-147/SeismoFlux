"""Deterministic analytic simulation regression for the stage-2 ETAS implementation."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import kstest, poisson  # type: ignore[import-untyped]

from seismoflux.background.etas import (
    ETASParent,
    branching_ratio,
    conditional_intensity,
    inverse_power_cutoff_mass,
    inverse_power_density,
    inverse_power_mass,
    inverse_power_scale,
    omori_cdf,
    omori_density,
    productivity,
)
from seismoflux.background.etas_simulation import sample_truncated_gr_magnitude
from seismoflux.background.randomness import SeedContext


@dataclass(frozen=True, slots=True)
class AnalyticSimulationExpectations:
    expected_generic_branching_ratio: float = 0.3928055160151634
    expected_root_direct_offspring_mean: float = 0.5436563656918091
    expected_root_total_descendants_mean: float = 0.8953578796104242
    expected_root_total_descendants_variance: float = 2.8311915410643453
    expected_root_zero_descendant_probability: float = 0.580621402297363
    descendant_mean_absolute_tolerance: float = 0.08
    descendant_variance_relative_tolerance: float = 0.25
    zero_descendant_probability_absolute_tolerance: float = 0.03
    direct_offspring_pit_ks_maximum: float = 0.035
    minimum_pooled_direct_offspring: int = 3500


@dataclass(frozen=True, slots=True)
class AnalyticSimulationRegression:
    replicate_count: int
    generic_branching_ratio: float
    root_direct_offspring_mean: float
    root_total_descendants_mean: float
    root_total_descendants_variance: float
    root_zero_descendant_probability: float
    direct_offspring_pit_ks: float
    pooled_direct_offspring_count: int
    event_cap_hits: int
    nonfinite_values: int
    checks: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return all(passed for _, passed in self.checks)


@dataclass(frozen=True, slots=True)
class ProductionFixtureRegression:
    """Runtime result for the content-addressed closed-form ETAS fixture."""

    fixture_id: str
    observed_values: tuple[tuple[str, float], ...]
    checks: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return all(passed for _, passed in self.checks)


def _fixture_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"production fixture {label} must be a mapping")
    return cast(dict[str, Any], value)


def run_production_fixture_regression(path: Path) -> ProductionFixtureRegression:
    """Execute the frozen scalar kernel, intensity, and likelihood regression."""

    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read production ETAS fixture: {path}") from exc
    document = _fixture_mapping(root, label="root")
    if document.get("schema_version") != 1:
        raise ValueError("production fixture schema version must be one")
    if document.get("fixture_id") != "etas_inverse_power_cut300_v1":
        raise ValueError("production fixture ID differs from the frozen protocol")
    if document.get("include_mark_likelihood") is not False:
        raise ValueError("production fixture must remain an unmarked likelihood")
    parameters = _fixture_mapping(document.get("parameters"), label="parameters")
    domain = _fixture_mapping(document.get("study_domain"), label="study domain")
    interval = _fixture_mapping(
        document.get("assessment_interval_days"),
        label="assessment interval",
    )
    expected = _fixture_mapping(document.get("expected"), label="expected values")
    tolerances = _fixture_mapping(document.get("tolerances"), label="tolerances")

    mc = float(parameters["Mc"])
    background = float(parameters["background_density_per_day_km2"])
    k = float(parameters["K"])
    alpha = float(parameters["alpha_per_magnitude"])
    c_days = float(parameters["c_days"])
    p = float(parameters["p"])
    d_km2 = float(parameters["D_km2"])
    q = float(parameters["q"])
    gamma = float(parameters["gamma_per_magnitude"])
    cutoff = float(parameters["spatial_cutoff_km"])
    radius = float(domain["radius_km"])
    start = float(interval["left"])
    end = float(interval["right"])
    if interval.get("closure") != "(left,right]" or not start < end:
        raise ValueError("production fixture assessment interval has drifted")

    root_parent = ETASParent(time_days=0.0, x_km=0.0, y_km=0.0, magnitude=4.0)
    observed_parent = ETASParent(time_days=1.0, x_km=0.0, y_km=0.0, magnitude=3.5)
    lambda_event = conditional_intensity(
        time_days=1.0,
        x_km=0.0,
        y_km=0.0,
        background_density_per_day_km2=background,
        parents=(root_parent,),
        mc=mc,
        k=k,
        alpha=alpha,
        c_days=c_days,
        p=p,
        d_km2=d_km2,
        q=q,
        gamma=gamma,
        spatial_cutoff_km=cutoff,
    )
    lambda_probe = conditional_intensity(
        time_days=2.0,
        x_km=3.0,
        y_km=4.0,
        background_density_per_day_km2=background,
        parents=(root_parent, observed_parent),
        mc=mc,
        k=k,
        alpha=alpha,
        c_days=c_days,
        p=p,
        d_km2=d_km2,
        q=q,
        gamma=gamma,
        spatial_cutoff_km=cutoff,
    )

    def cutoff_mass(magnitude: float) -> float:
        return inverse_power_cutoff_mass(
            magnitude,
            d_km2=d_km2,
            q=q,
            gamma=gamma,
            mc=mc,
            cutoff_radius_km=cutoff,
        )

    def spatial_density(radius_km: float, magnitude: float) -> float:
        return inverse_power_density(
            radius_km,
            magnitude,
            d_km2=d_km2,
            q=q,
            gamma=gamma,
            mc=mc,
            cutoff_radius_km=cutoff,
        )

    def spatial_mass(radius_km: float, magnitude: float) -> float:
        return inverse_power_mass(
            radius_km,
            magnitude,
            d_km2=d_km2,
            q=q,
            gamma=gamma,
            mc=mc,
            cutoff_radius_km=cutoff,
        )

    productivity_m4 = productivity(4.0, k=k, alpha=alpha, mc=mc)
    productivity_m3p5 = productivity(3.5, k=k, alpha=alpha, mc=mc)
    temporal_cdf_start = omori_cdf(start, c_days=c_days, p=p)
    temporal_cdf_one = omori_cdf(1.0, c_days=c_days, p=p)
    temporal_cdf_end = omori_cdf(end, c_days=c_days, p=p)
    background_compensator = background * math.pi * radius**2 * (end - start)
    root_compensator = (
        productivity_m4 * (temporal_cdf_end - temporal_cdf_start) * spatial_mass(radius, 4.0)
    )
    observed_compensator = productivity_m3p5 * temporal_cdf_one * spatial_mass(radius, 3.5)
    total_compensator = background_compensator + root_compensator + observed_compensator
    log_likelihood = math.log(lambda_event) - total_compensator

    values = {
        "cutoff_mass_m4": cutoff_mass(4.0),
        "cutoff_mass_m3p5": cutoff_mass(3.5),
        "sigma_m4_km2": inverse_power_scale(4.0, d_km2=d_km2, gamma=gamma, mc=mc),
        "sigma_m3p5_km2": inverse_power_scale(
            3.5,
            d_km2=d_km2,
            gamma=gamma,
            mc=mc,
        ),
        "productivity_m4": productivity_m4,
        "productivity_m3p5": productivity_m3p5,
        "temporal_density_dt1_per_day": omori_density(1.0, c_days=c_days, p=p),
        "temporal_density_dt2_per_day": omori_density(2.0, c_days=c_days, p=p),
        "temporal_cdf_dt0p5": temporal_cdf_start,
        "temporal_cdf_dt1": temporal_cdf_one,
        "temporal_cdf_dt2": temporal_cdf_end,
        "spatial_density_r0_m4_per_km2": spatial_density(0.0, 4.0),
        "spatial_density_r5_m4_per_km2": spatial_density(5.0, 4.0),
        "spatial_density_r5_m3p5_per_km2": spatial_density(5.0, 3.5),
        "spatial_mass_r30_m4": spatial_mass(radius, 4.0),
        "spatial_mass_r30_m3p5": spatial_mass(radius, 3.5),
        "lambda_event_t1_r0_per_day_km2": lambda_event,
        "background_compensator": background_compensator,
        "root_compensator": root_compensator,
        "event1_postevent_compensator": observed_compensator,
        "total_compensator": total_compensator,
        "unmarked_log_likelihood": log_likelihood,
        "lambda_probe_t2_r5_per_day_km2": lambda_probe,
    }
    if set(values) != set(expected):
        raise ValueError("production fixture expected keys differ from the runtime regression")
    relative = float(tolerances["scalar_relative"])
    absolute = float(tolerances["scalar_absolute"])
    likelihood_absolute = float(tolerances["log_likelihood_absolute"])
    checks = tuple(
        (
            name,
            math.isclose(
                actual,
                float(expected[name]),
                rel_tol=(0.0 if name == "unmarked_log_likelihood" else relative),
                abs_tol=(likelihood_absolute if name == "unmarked_log_likelihood" else absolute),
            ),
        )
        for name, actual in values.items()
    )
    return ProductionFixtureRegression(
        fixture_id=cast(str, document["fixture_id"]),
        observed_values=tuple(values.items()),
        checks=checks,
    )


def _randomized_poisson_pit(
    count: int,
    mean: float,
    uniform: float,
) -> float:
    lower = float(poisson.cdf(count - 1, mean)) if count > 0 else 0.0
    mass = float(poisson.pmf(count, mean))
    return lower + uniform * mass


def run_analytic_simulation_regression(
    *,
    replicate_count: int = 8192,
    maximum_events_per_replicate: int = 100_000,
    expectations: AnalyticSimulationExpectations | None = None,
) -> AnalyticSimulationRegression:
    """Simulate the frozen mixed-Poisson branching fixture to extinction."""

    if replicate_count != 8192:
        raise ValueError("analytic simulation regression must use 8192 replicates")
    if maximum_events_per_replicate != 100_000:
        raise ValueError("analytic simulation event cap must remain 100000")
    expected = expectations or AnalyticSimulationExpectations()
    mc = 3.0
    maximum_magnitude = 7.0
    beta = 2.0
    k = 0.2
    alpha = 1.0
    root_magnitude = 4.0
    generic_ratio = branching_ratio(
        k=k,
        alpha=alpha,
        beta=beta,
        magnitude_span=maximum_magnitude - mc,
    )
    total_descendants = np.empty(replicate_count, dtype=np.float64)
    root_direct_counts = np.empty(replicate_count, dtype=np.float64)
    pits: list[float] = []
    cap_hits = 0
    nonfinite_values = 0

    for replicate_index in range(replicate_count):
        generator = SeedContext(
            root_seed=147,
            protocol_version="0.2.0",
            namespace="simulation_regression",
            model_id="etas_inverse_power_cut300_v1",
            issue_id=None,
            replicate_index=replicate_index,
        ).generator()
        queue: deque[float] = deque([root_magnitude])
        descendants = 0
        direct_count: int | None = None
        capped = False
        while queue:
            parent_magnitude = queue.popleft()
            mean = productivity(parent_magnitude, k=k, alpha=alpha, mc=mc)
            count = int(generator.poisson(mean))
            pits.append(_randomized_poisson_pit(count, mean, float(generator.random())))
            if direct_count is None:
                direct_count = count
            if descendants + count > maximum_events_per_replicate:
                cap_hits += 1
                capped = True
                break
            descendants += count
            for _ in range(count):
                queue.append(
                    sample_truncated_gr_magnitude(
                        generator,
                        mc=mc,
                        maximum_magnitude=maximum_magnitude,
                        beta=beta,
                    )
                )
        if capped:
            total_descendants[replicate_index] = math.nan
            root_direct_counts[replicate_index] = math.nan
            continue
        total_descendants[replicate_index] = descendants
        root_direct_counts[replicate_index] = direct_count if direct_count is not None else 0

    finite = np.isfinite(total_descendants) & np.isfinite(root_direct_counts)
    nonfinite_values = int(np.count_nonzero(~finite))
    usable_descendants = total_descendants[finite]
    usable_direct = root_direct_counts[finite]
    if usable_descendants.size == 0:
        raise ValueError("analytic simulation regression produced no finite replicates")
    descendant_mean = float(np.mean(usable_descendants))
    descendant_variance = float(np.var(usable_descendants, ddof=1))
    zero_probability = float(np.mean(usable_descendants == 0.0))
    direct_mean = float(np.mean(usable_direct))
    pit_ks = float(kstest(np.asarray(pits, dtype=np.float64), "uniform").statistic)
    checks = (
        (
            "generic_branching_ratio",
            math.isclose(
                generic_ratio,
                expected.expected_generic_branching_ratio,
                rel_tol=5.0e-12,
                abs_tol=1.0e-14,
            ),
        ),
        (
            "root_direct_offspring_mean",
            abs(direct_mean - expected.expected_root_direct_offspring_mean)
            <= expected.descendant_mean_absolute_tolerance,
        ),
        (
            "root_total_descendants_mean",
            abs(descendant_mean - expected.expected_root_total_descendants_mean)
            <= expected.descendant_mean_absolute_tolerance,
        ),
        (
            "root_total_descendants_variance",
            abs(descendant_variance - expected.expected_root_total_descendants_variance)
            / expected.expected_root_total_descendants_variance
            <= expected.descendant_variance_relative_tolerance,
        ),
        (
            "root_zero_descendant_probability",
            abs(zero_probability - expected.expected_root_zero_descendant_probability)
            <= expected.zero_descendant_probability_absolute_tolerance,
        ),
        ("direct_offspring_pit_ks", pit_ks <= expected.direct_offspring_pit_ks_maximum),
        (
            "minimum_pooled_direct_offspring",
            len(pits) >= expected.minimum_pooled_direct_offspring,
        ),
        ("event_cap_hits", cap_hits == 0),
        ("nonfinite_values", nonfinite_values == 0),
    )
    return AnalyticSimulationRegression(
        replicate_count=replicate_count,
        generic_branching_ratio=generic_ratio,
        root_direct_offspring_mean=direct_mean,
        root_total_descendants_mean=descendant_mean,
        root_total_descendants_variance=descendant_variance,
        root_zero_descendant_probability=zero_probability,
        direct_offspring_pit_ks=pit_ks,
        pooled_direct_offspring_count=len(pits),
        event_cap_hits=cap_hits,
        nonfinite_values=nonfinite_values,
        checks=checks,
    )


__all__ = [
    "AnalyticSimulationExpectations",
    "AnalyticSimulationRegression",
    "ProductionFixtureRegression",
    "run_analytic_simulation_regression",
    "run_production_fixture_regression",
]
