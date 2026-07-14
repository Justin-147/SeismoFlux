"""Causal ETAS likelihood, optimization, and numerical-stability audits."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from seismoflux.background.etas import (
    branching_ratio,
    inverse_power_density,
)
from seismoflux.background.randomness import SeedContext

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Objective = Callable[[FloatArray], float]
BackgroundDensity = Callable[[float, float], float]


class BatchBackgroundDensity(Protocol):
    """Optional vectorized extension for a callable background density."""

    def __call__(self, x_km: float, y_km: float) -> float:
        """Evaluate one location."""

    def density_many(self, x_km: npt.ArrayLike, y_km: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluate aligned location arrays."""


ETAS_PARAMETER_ORDER = (
    "background_rate_per_day",
    "productivity_k",
    "alpha",
    "c_days",
    "p",
)
ETAS_TRANSFORMED_PARAMETER_ORDER = (
    "log_background_rate_per_day",
    "log_productivity_k",
    "log_alpha",
    "log_c_days",
    "log_p_minus_one",
)
DELTA_CONFIDENCE_LEVEL = 0.95
_DELTA_NORMAL_QUANTILE = 1.959963984540054
_MATRIX_SYMMETRY_ABSOLUTE_TOLERANCE = 1.0e-12
_BACKGROUND_DENSITY_BATCH_SIZE = 65_536
ISSUE_FIELD_GRID_SIZES_KM = (50.0, 25.0, 12.5)


def _readonly_float(values: npt.ArrayLike) -> FloatArray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _owned_readonly_float(values: npt.ArrayLike) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_int(values: npt.ArrayLike) -> IntArray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


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


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class ETASEvent:
    """One catalog event with distinct physical-origin and report-availability times."""

    event_id: str
    time_days: float
    available_time_days: float
    x_km: float
    y_km: float
    magnitude: float
    inside_study_area: bool
    inside_parent_domain: bool

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        origin_time = _finite("event time_days", self.time_days)
        available_time = _finite("event available_time_days", self.available_time_days)
        if available_time < origin_time:
            raise ValueError("event available_time_days must not precede time_days")
        _finite("event x_km", self.x_km)
        _finite("event y_km", self.y_km)
        _finite("event magnitude", self.magnitude)
        if not isinstance(self.inside_study_area, bool) or not isinstance(
            self.inside_parent_domain, bool
        ):
            raise TypeError("event domain flags must be bool")
        if self.inside_study_area and not self.inside_parent_domain:
            raise ValueError("study-area events must also be inside the parent domain")


@dataclass(frozen=True, slots=True)
class ETASParameters:
    """Five fitted ETAS parameters in physical space."""

    background_rate_per_day: float
    productivity_k: float
    alpha: float
    c_days: float
    p: float

    def __post_init__(self) -> None:
        _positive("background_rate_per_day", self.background_rate_per_day)
        _nonnegative("productivity_k", self.productivity_k)
        _nonnegative("alpha", self.alpha)
        _positive("c_days", self.c_days)
        if _finite("p", self.p) <= 1.0:
            raise ValueError("p must be greater than 1")


@dataclass(frozen=True, slots=True)
class ETASModelSpec:
    """Fixed catalog, mark, temporal-cutoff, and spatial-kernel parameters."""

    mc: float
    beta: float
    maximum_magnitude: float = 9.5
    d_km2: float = 25.0
    q: float = 1.5
    gamma: float = 1.0
    spatial_cutoff_km: float = 300.0
    history_parent_cutoff_days: float = 3650.0
    branching_ratio_maximum: float = 0.95
    alpha_beta_equality_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        mc = _finite("mc", self.mc)
        maximum = _finite("maximum_magnitude", self.maximum_magnitude)
        if maximum <= mc:
            raise ValueError("maximum_magnitude must exceed mc")
        if maximum != 9.5:
            raise ValueError("maximum_magnitude must remain frozen at 9.5")
        _positive("beta", self.beta)
        d_km2 = _positive("d_km2", self.d_km2)
        if d_km2 not in {25.0, 100.0, 400.0}:
            raise ValueError("d_km2 must use a preregistered main or sensitivity value")
        if _finite("q", self.q) <= 1.0:
            raise ValueError("q must be greater than 1")
        if self.q != 1.5:
            raise ValueError("q must remain frozen at 1.5")
        if _nonnegative("gamma", self.gamma) != 1.0:
            raise ValueError("gamma must remain frozen at 1.0")
        if _positive("spatial_cutoff_km", self.spatial_cutoff_km) != 300.0:
            raise ValueError("spatial_cutoff_km must remain frozen at 300")
        if _positive("history_parent_cutoff_days", self.history_parent_cutoff_days) != 3650.0:
            raise ValueError("history_parent_cutoff_days must remain frozen at 3650")
        maximum_ratio = _positive("branching_ratio_maximum", self.branching_ratio_maximum)
        if maximum_ratio != 0.95:
            raise ValueError("branching_ratio_maximum must remain frozen at 0.95")
        if (
            _nonnegative("alpha_beta_equality_tolerance", self.alpha_beta_equality_tolerance)
            != 1.0e-12
        ):
            raise ValueError("alpha-beta equality tolerance must remain frozen at 1e-12")

    @property
    def magnitude_span(self) -> float:
        return self.maximum_magnitude - self.mc

    def branching_ratio(self, parameters: ETASParameters) -> float:
        return branching_ratio(
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            beta=self.beta,
            magnitude_span=self.magnitude_span,
            equality_tolerance=self.alpha_beta_equality_tolerance,
        )

    def validate_parameters(self, parameters: ETASParameters) -> None:
        ratio = self.branching_ratio(parameters)
        if ratio >= self.branching_ratio_maximum:
            raise ValueError("ETAS branching ratio must be below the frozen maximum")

    def validate_event(self, event: ETASEvent) -> None:
        if event.magnitude < self.mc:
            raise ValueError(f"event {event.event_id!r} is below mc")
        if event.magnitude > self.maximum_magnitude:
            raise ValueError(f"event {event.event_id!r} exceeds maximum_magnitude")


@dataclass(frozen=True, slots=True)
class ETASParameterBounds:
    """Frozen physical bounds and their log-transformed representation."""

    background_rate_per_day: tuple[float, float] = (0.01, 10.0)
    productivity_k: tuple[float, float] = (0.0001, 0.5)
    alpha: tuple[float, float] = (0.05, 2.0)
    c_days: tuple[float, float] = (0.001, 30.0)
    p: tuple[float, float] = (1.01, 2.5)

    def __post_init__(self) -> None:
        for name in ("background_rate_per_day", "productivity_k", "alpha", "c_days"):
            lower, upper = getattr(self, name)
            if _positive(f"{name} lower", lower) >= _positive(f"{name} upper", upper):
                raise ValueError(f"{name} bounds must be increasing")
        p_lower, p_upper = self.p
        if _finite("p lower", p_lower) <= 1.0 or p_lower >= _finite("p upper", p_upper):
            raise ValueError("p bounds must be increasing and greater than one")

    def transformed(self) -> tuple[tuple[float, float], ...]:
        physical = (
            self.background_rate_per_day,
            self.productivity_k,
            self.alpha,
            self.c_days,
            (self.p[0] - 1.0, self.p[1] - 1.0),
        )
        return tuple((math.log(lower), math.log(upper)) for lower, upper in physical)

    def contains(self, parameters: ETASParameters) -> bool:
        values = (
            parameters.background_rate_per_day,
            parameters.productivity_k,
            parameters.alpha,
            parameters.c_days,
            parameters.p,
        )
        limits = (
            self.background_rate_per_day,
            self.productivity_k,
            self.alpha,
            self.c_days,
            self.p,
        )
        return all(
            lower <= value <= upper for value, (lower, upper) in zip(values, limits, strict=True)
        )

    def to_transformed(self, parameters: ETASParameters) -> FloatArray:
        if not self.contains(parameters):
            raise ValueError("ETAS parameters are outside frozen bounds")
        return np.asarray(
            [
                math.log(parameters.background_rate_per_day),
                math.log(parameters.productivity_k),
                math.log(parameters.alpha),
                math.log(parameters.c_days),
                math.log(parameters.p - 1.0),
            ],
            dtype=np.float64,
        )

    def from_transformed(self, transformed: npt.ArrayLike) -> ETASParameters:
        values = np.asarray(transformed, dtype=np.float64)
        if values.ndim != 1 or values.size != 5:
            raise ValueError("transformed ETAS parameter vector must have length five")
        if not np.all(np.isfinite(values)):
            raise ValueError("transformed ETAS parameters must be finite")
        try:
            with np.errstate(over="raise", invalid="raise"):
                physical = np.exp(values)
        except FloatingPointError as error:
            raise ValueError("transformed ETAS parameters overflow physical space") from error
        if not np.all(np.isfinite(physical)):
            raise ValueError("transformed ETAS parameters overflow physical space")
        parameters = ETASParameters(
            background_rate_per_day=float(physical[0]),
            productivity_k=float(physical[1]),
            alpha=float(physical[2]),
            c_days=float(physical[3]),
            p=1.0 + float(physical[4]),
        )
        if not self.contains(parameters):
            raise ValueError("transformed ETAS parameters are outside frozen bounds")
        return parameters


@dataclass(frozen=True, slots=True)
class QuadraturePoint:
    """One clipped-cell representative point and its exact clipped area."""

    x_km: float
    y_km: float
    area_km2: float

    def __post_init__(self) -> None:
        _finite("quadrature x_km", self.x_km)
        _finite("quadrature y_km", self.y_km)
        _positive("quadrature area_km2", self.area_km2)


class SpatialIntegrator(Protocol):
    """Injectable study-area integration boundary."""

    def integrate(self, density: BackgroundDensity) -> float:
        """Integrate a non-negative density over the study area."""


def _vectorized_inverse_power_density_squared(
    distance_squared_km2: FloatArray,
    magnitudes: FloatArray,
    *,
    spec: ETASModelSpec,
) -> FloatArray:
    """Evaluate the frozen spatial density for aligned distance/mark arrays."""

    if distance_squared_km2.shape != magnitudes.shape:
        raise ValueError("distance and magnitude arrays must have identical shapes")
    if np.any(distance_squared_km2 < 0.0) or not np.all(np.isfinite(distance_squared_km2)):
        raise ValueError("squared spatial distances must be finite and non-negative")
    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("parent magnitudes must be finite")
    if np.any(magnitudes < spec.mc) or np.any(magnitudes > spec.maximum_magnitude):
        raise ValueError("parent magnitudes must lie in the frozen truncated-GR support")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scale = spec.d_km2 * np.exp(spec.gamma * (magnitudes - spec.mc))
        cutoff_mass = 1.0 - np.power(
            1.0 + spec.spatial_cutoff_km**2 / scale,
            1.0 - spec.q,
        )
        density = (
            (spec.q - 1.0)
            / (math.pi * scale)
            * np.power(1.0 + distance_squared_km2 / scale, -spec.q)
            / cutoff_mass
        )
    density = np.where(
        distance_squared_km2 <= spec.spatial_cutoff_km**2,
        density,
        0.0,
    )
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("vectorized spatial density must be finite and non-negative")
    return np.asarray(density, dtype=np.float64)


def _query_ball_indices(
    tree: cKDTree,
    coordinates: FloatArray,
    radius_km: float,
) -> tuple[IntArray, ...]:
    raw = cast(
        Sequence[Sequence[int]],
        tree.query_ball_point(
            coordinates,
            r=radius_km,
            workers=1,
            return_sorted=True,
        ),
    )
    return tuple(np.asarray(indices, dtype=np.int64) for indices in raw)


@dataclass(frozen=True, slots=True)
class PointAreaQuadrature:
    """Frozen representative-point-times-exact-area quadrature."""

    points: tuple[QuadraturePoint, ...]
    _coordinates: FloatArray = field(init=False, repr=False, compare=False)
    _areas: FloatArray = field(init=False, repr=False, compare=False)
    _tree: cKDTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("spatial quadrature must contain at least one point")
        coordinates = _readonly_float([(point.x_km, point.y_km) for point in self.points])
        areas = _readonly_float([point.area_km2 for point in self.points])
        object.__setattr__(self, "_coordinates", coordinates)
        object.__setattr__(self, "_areas", areas)
        object.__setattr__(
            self,
            "_tree",
            cKDTree(coordinates, compact_nodes=True, balanced_tree=True, copy_data=True),
        )

    @classmethod
    def from_points(cls, points: Iterable[QuadraturePoint]) -> PointAreaQuadrature:
        return cls(tuple(points))

    def integrate(self, density: BackgroundDensity) -> float:
        values = _evaluate_background_density_many(
            density,
            self._coordinates[:, 0],
            self._coordinates[:, 1],
            label="quadrature density",
        )
        terms = values * self._areas
        if not np.all(np.isfinite(terms)):
            raise ValueError("quadrature contributions must be finite")
        try:
            result = math.fsum(float(value) for value in terms)
        except OverflowError as error:
            raise ValueError("spatial quadrature integral must be finite") from error
        return _nonnegative("spatial quadrature integral", result)

    def inverse_power_masses(
        self,
        parent_x_km: npt.ArrayLike,
        parent_y_km: npt.ArrayLike,
        parent_magnitudes: npt.ArrayLike,
        *,
        spec: ETASModelSpec,
        batch_size: int = 256,
    ) -> FloatArray:
        """Integrate many parent kernels using batched 300-km tree neighborhoods."""

        parent_x = np.asarray(parent_x_km, dtype=np.float64)
        parent_y = np.asarray(parent_y_km, dtype=np.float64)
        magnitudes = np.asarray(parent_magnitudes, dtype=np.float64)
        if (
            parent_x.ndim != 1
            or parent_x.shape != parent_y.shape
            or parent_x.shape != magnitudes.shape
        ):
            raise ValueError("parent coordinate and magnitude arrays must be aligned vectors")
        if not (
            np.all(np.isfinite(parent_x))
            and np.all(np.isfinite(parent_y))
            and np.all(np.isfinite(magnitudes))
        ):
            raise ValueError("parent coordinate and magnitude arrays must be finite")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        masses = np.zeros(parent_x.size, dtype=np.float64)
        for start in range(0, parent_x.size, batch_size):
            stop = min(parent_x.size, start + batch_size)
            parent_coordinates = np.column_stack((parent_x[start:stop], parent_y[start:stop]))
            neighbors = _query_ball_indices(
                self._tree,
                np.asarray(parent_coordinates, dtype=np.float64),
                spec.spatial_cutoff_km,
            )
            lengths = np.fromiter((indices.size for indices in neighbors), dtype=np.int64)
            if int(lengths.sum()) == 0:
                continue
            local_parent_indices = np.repeat(np.arange(stop - start, dtype=np.int64), lengths)
            cell_indices = np.concatenate(neighbors)
            delta = self._coordinates[cell_indices] - parent_coordinates[local_parent_indices]
            distance_squared = np.einsum("ij,ij->i", delta, delta)
            pair_magnitudes = magnitudes[start:stop][local_parent_indices]
            density = _vectorized_inverse_power_density_squared(
                np.asarray(distance_squared, dtype=np.float64),
                np.asarray(pair_magnitudes, dtype=np.float64),
                spec=spec,
            )
            masses[start:stop] = np.bincount(
                local_parent_indices,
                weights=density * self._areas[cell_indices],
                minlength=stop - start,
            )
        if np.any(masses < 0.0) or not np.all(np.isfinite(masses)):
            raise ValueError("vectorized parent spatial masses must be finite and non-negative")
        return _readonly_float(masses)

    def inverse_power_weighted_point_sums(
        self,
        parent_x_km: npt.ArrayLike,
        parent_y_km: npt.ArrayLike,
        parent_magnitudes: npt.ArrayLike,
        parent_weights: npt.ArrayLike,
        *,
        spec: ETASModelSpec,
        batch_size: int = 256,
    ) -> FloatArray:
        """Sum weighted parent spatial densities at every quadrature point."""

        parent_x = np.asarray(parent_x_km, dtype=np.float64)
        parent_y = np.asarray(parent_y_km, dtype=np.float64)
        magnitudes = np.asarray(parent_magnitudes, dtype=np.float64)
        weights = np.asarray(parent_weights, dtype=np.float64)
        if (
            parent_x.ndim != 1
            or parent_x.shape != parent_y.shape
            or parent_x.shape != magnitudes.shape
            or parent_x.shape != weights.shape
        ):
            raise ValueError("weighted parent inputs must be aligned vectors")
        if not (
            np.all(np.isfinite(parent_x))
            and np.all(np.isfinite(parent_y))
            and np.all(np.isfinite(magnitudes))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError("weighted parent inputs must be finite")
        if np.any(weights < 0.0):
            raise ValueError("parent weights must be non-negative")
        if np.any(magnitudes < spec.mc) or np.any(magnitudes > spec.maximum_magnitude):
            raise ValueError("parent magnitudes must lie in the frozen truncated-GR support")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        point_sums = np.zeros(self._coordinates.shape[0], dtype=np.float64)
        for start in range(0, parent_x.size, batch_size):
            stop = min(parent_x.size, start + batch_size)
            parent_coordinates = np.column_stack((parent_x[start:stop], parent_y[start:stop]))
            neighbors = _query_ball_indices(
                self._tree,
                np.asarray(parent_coordinates, dtype=np.float64),
                spec.spatial_cutoff_km,
            )
            lengths = np.fromiter((indices.size for indices in neighbors), dtype=np.int64)
            if int(lengths.sum()) == 0:
                continue
            local_parent_indices = np.repeat(np.arange(stop - start, dtype=np.int64), lengths)
            point_indices = np.concatenate(neighbors)
            delta = self._coordinates[point_indices] - parent_coordinates[local_parent_indices]
            distance_squared = np.einsum("ij,ij->i", delta, delta)
            pair_magnitudes = magnitudes[start:stop][local_parent_indices]
            density = _vectorized_inverse_power_density_squared(
                np.asarray(distance_squared, dtype=np.float64),
                np.asarray(pair_magnitudes, dtype=np.float64),
                spec=spec,
            )
            pair_weights = weights[start:stop][local_parent_indices]
            point_sums += np.bincount(
                point_indices,
                weights=pair_weights * density,
                minlength=self._coordinates.shape[0],
            )
        if np.any(point_sums < 0.0) or not np.all(np.isfinite(point_sums)):
            raise ValueError("weighted point sums must be finite and non-negative")
        return _readonly_float(point_sums)


@dataclass(frozen=True, slots=True)
class ETASLikelihoodProblem:
    """One fixed causal likelihood interval and its availability-gated event history."""

    assessment_start_days: float
    assessment_end_days: float
    target_events: tuple[ETASEvent, ...]
    parent_events: tuple[ETASEvent, ...]
    background_density: BackgroundDensity
    spatial_integrator: SpatialIntegrator

    def __post_init__(self) -> None:
        start = _finite("assessment_start_days", self.assessment_start_days)
        end = _finite("assessment_end_days", self.assessment_end_days)
        if end <= start:
            raise ValueError("assessment interval must have positive duration")
        target_ids: set[str] = set()
        for event in self.target_events:
            if event.event_id in target_ids:
                raise ValueError("target event IDs must be unique")
            target_ids.add(event.event_id)
            if not event.inside_study_area:
                raise ValueError("every target event must be inside the study area")
            if not start < event.time_days <= end:
                raise ValueError("target events must lie in (assessment_start, assessment_end]")
        parent_by_id: dict[str, ETASEvent] = {}
        for event in self.parent_events:
            if event.event_id in parent_by_id:
                raise ValueError("parent event IDs must be unique")
            if not event.inside_parent_domain:
                raise ValueError("every supplied parent must be inside study area plus 300 km")
            if event.time_days > end:
                raise ValueError("parent history must not include events after assessment_end")
            parent_by_id[event.event_id] = event
        expected_target_ids = {
            event.event_id
            for event in self.parent_events
            if event.inside_study_area and start < event.time_days <= end
        }
        if target_ids != expected_target_ids:
            raise ValueError(
                "targets must equal all study-area parent-domain events in the assessment interval"
            )
        for target in self.target_events:
            if parent_by_id.get(target.event_id) != target:
                raise ValueError("each target event must appear identically in parent history")


@dataclass(frozen=True, slots=True)
class ETASLikelihoodResult:
    """Auditable event and compensator decomposition of one log likelihood."""

    target_event_ids: tuple[str, ...]
    event_intensities: tuple[float, ...]
    event_log_intensity_sum: float
    background_compensator: float
    triggering_compensator: float
    total_compensator: float
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class PreparedETASLikelihood:
    """Contiguous geometry cache reused across optimizer evaluations."""

    spec: ETASModelSpec
    duration_days: float
    background_study_area_mass: float
    target_event_ids: tuple[str, ...]
    target_background_spatial_density: FloatArray
    event_parent_target_index: IntArray
    event_parent_delta_days: FloatArray
    event_parent_magnitude: FloatArray
    event_parent_spatial_density: FloatArray
    compensator_lower_age_days: FloatArray
    compensator_upper_age_days: FloatArray
    compensator_parent_magnitude: FloatArray
    compensator_parent_spatial_mass: FloatArray


def _validated_background_density(
    background_density: BackgroundDensity,
    x_km: float,
    y_km: float,
) -> float:
    return _nonnegative("background spatial density", background_density(x_km, y_km))


def _evaluate_background_density_many(
    background_density: BackgroundDensity,
    x_km: npt.ArrayLike,
    y_km: npt.ArrayLike,
    *,
    label: str = "background spatial density",
) -> FloatArray:
    """Evaluate a background density in fixed chunks, with scalar fallback."""

    x_values = np.asarray(x_km, dtype=np.float64)
    y_values = np.asarray(y_km, dtype=np.float64)
    if x_values.ndim != 1 or x_values.shape != y_values.shape:
        raise ValueError("background density coordinates must be aligned vectors")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("background density coordinates must be finite")
    output = np.empty(x_values.size, dtype=np.float64)
    batch_method = getattr(background_density, "density_many", None)
    if callable(batch_method):
        vectorized = cast(Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike], batch_method)
        for start in range(0, x_values.size, _BACKGROUND_DENSITY_BATCH_SIZE):
            stop = min(start + _BACKGROUND_DENSITY_BATCH_SIZE, x_values.size)
            batch = np.asarray(
                vectorized(x_values[start:stop], y_values[start:stop]),
                dtype=np.float64,
            )
            if batch.shape != (stop - start,):
                raise ValueError("density_many must return one value per aligned coordinate")
            output[start:stop] = batch
    else:
        for index, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
            output[index] = background_density(float(x_value), float(y_value))
    if not np.all(np.isfinite(output)) or np.any(output < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    return _readonly_float(output)


def _spatial_parent_mass(
    parent: ETASEvent,
    *,
    spec: ETASModelSpec,
    integrator: SpatialIntegrator,
) -> float:
    def density(x_km: float, y_km: float) -> float:
        return inverse_power_density(
            math.hypot(x_km - parent.x_km, y_km - parent.y_km),
            parent.magnitude,
            d_km2=spec.d_km2,
            q=spec.q,
            gamma=spec.gamma,
            mc=spec.mc,
            cutoff_radius_km=spec.spatial_cutoff_km,
        )

    return integrator.integrate(density)


def _prepare_event_parent_arrays(
    targets: Sequence[ETASEvent],
    parents: Sequence[ETASEvent],
    *,
    spec: ETASModelSpec,
    batch_size: int = 512,
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
    if not targets or not parents:
        return (
            _readonly_int([]),
            _readonly_float([]),
            _readonly_float([]),
            _readonly_float([]),
        )
    parent_coordinates = _readonly_float([(event.x_km, event.y_km) for event in parents])
    parent_times = _readonly_float([event.time_days for event in parents])
    parent_available_times = _readonly_float([event.available_time_days for event in parents])
    parent_magnitudes = _readonly_float([event.magnitude for event in parents])
    tree = cKDTree(
        parent_coordinates,
        compact_nodes=True,
        balanced_tree=True,
        copy_data=True,
    )
    target_index_chunks: list[IntArray] = []
    delta_chunks: list[FloatArray] = []
    magnitude_chunks: list[FloatArray] = []
    density_chunks: list[FloatArray] = []
    for start in range(0, len(targets), batch_size):
        stop = min(len(targets), start + batch_size)
        target_batch = targets[start:stop]
        target_coordinates = _readonly_float([(event.x_km, event.y_km) for event in target_batch])
        target_times = _readonly_float([event.time_days for event in target_batch])
        neighbors = _query_ball_indices(
            tree,
            target_coordinates,
            spec.spatial_cutoff_km,
        )
        lengths = np.fromiter((indices.size for indices in neighbors), dtype=np.int64)
        if int(lengths.sum()) == 0:
            continue
        local_target_indices = np.repeat(np.arange(stop - start, dtype=np.int64), lengths)
        parent_indices = np.concatenate(neighbors)
        delta_days = target_times[local_target_indices] - parent_times[parent_indices]
        eligible = (
            (delta_days > 0.0)
            & (delta_days <= spec.history_parent_cutoff_days)
            & (parent_available_times[parent_indices] <= target_times[local_target_indices])
        )
        if not np.any(eligible):
            continue
        local_target_indices = local_target_indices[eligible]
        parent_indices = parent_indices[eligible]
        delta_days = delta_days[eligible]
        coordinate_delta = (
            target_coordinates[local_target_indices] - parent_coordinates[parent_indices]
        )
        distance_squared = np.einsum("ij,ij->i", coordinate_delta, coordinate_delta)
        pair_magnitudes = parent_magnitudes[parent_indices]
        target_index_chunks.append(np.asarray(local_target_indices + start, dtype=np.int64))
        delta_chunks.append(np.asarray(delta_days, dtype=np.float64))
        magnitude_chunks.append(np.asarray(pair_magnitudes, dtype=np.float64))
        density_chunks.append(
            _vectorized_inverse_power_density_squared(
                np.asarray(distance_squared, dtype=np.float64),
                np.asarray(pair_magnitudes, dtype=np.float64),
                spec=spec,
            )
        )
    if not target_index_chunks:
        return (
            _readonly_int([]),
            _readonly_float([]),
            _readonly_float([]),
            _readonly_float([]),
        )
    return (
        _readonly_int(np.concatenate(target_index_chunks)),
        _readonly_float(np.concatenate(delta_chunks)),
        _readonly_float(np.concatenate(magnitude_chunks)),
        _readonly_float(np.concatenate(density_chunks)),
    )


def _eligible_compensator_parent_windows(
    problem: ETASLikelihoodProblem,
    parents: Sequence[ETASEvent],
    *,
    spec: ETASModelSpec,
) -> tuple[tuple[ETASEvent, ...], FloatArray, FloatArray, FloatArray]:
    eligible_parents: list[ETASEvent] = []
    lower_ages: list[float] = []
    upper_ages: list[float] = []
    for parent in parents:
        lower_age = (
            max(problem.assessment_start_days, parent.available_time_days) - parent.time_days
        )
        upper_age = min(
            spec.history_parent_cutoff_days,
            problem.assessment_end_days - parent.time_days,
        )
        if upper_age <= lower_age:
            continue
        eligible_parents.append(parent)
        lower_ages.append(lower_age)
        upper_ages.append(upper_age)
    parent_tuple = tuple(eligible_parents)
    return (
        parent_tuple,
        _readonly_float(lower_ages),
        _readonly_float(upper_ages),
        _readonly_float([parent.magnitude for parent in parent_tuple]),
    )


def _prepare_compensator_arrays(
    problem: ETASLikelihoodProblem,
    parents: Sequence[ETASEvent],
    *,
    spec: ETASModelSpec,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    eligible_parents, lower_ages, upper_ages, magnitudes = _eligible_compensator_parent_windows(
        problem, parents, spec=spec
    )
    if not eligible_parents:
        return (
            _readonly_float([]),
            _readonly_float([]),
            _readonly_float([]),
            _readonly_float([]),
        )
    if isinstance(problem.spatial_integrator, PointAreaQuadrature):
        spatial_masses = problem.spatial_integrator.inverse_power_masses(
            [parent.x_km for parent in eligible_parents],
            [parent.y_km for parent in eligible_parents],
            magnitudes,
            spec=spec,
        )
    else:
        spatial_masses = _readonly_float(
            [
                _spatial_parent_mass(
                    parent,
                    spec=spec,
                    integrator=problem.spatial_integrator,
                )
                for parent in eligible_parents
            ]
        )
    return (
        _readonly_float(lower_ages),
        _readonly_float(upper_ages),
        magnitudes,
        spatial_masses,
    )


def prepare_etas_likelihood(
    problem: ETASLikelihoodProblem,
    spec: ETASModelSpec,
) -> PreparedETASLikelihood:
    """Validate causal domains and cache fixed spatial terms."""

    for event in problem.parent_events:
        spec.validate_event(event)
    sorted_targets = tuple(
        sorted(problem.target_events, key=lambda item: (item.time_days, item.event_id))
    )
    sorted_parents = tuple(
        sorted(problem.parent_events, key=lambda item: (item.time_days, item.event_id))
    )
    (
        event_parent_target_index,
        event_parent_delta_days,
        event_parent_magnitude,
        event_parent_spatial_density,
    ) = _prepare_event_parent_arrays(sorted_targets, sorted_parents, spec=spec)
    target_background_density = _evaluate_background_density_many(
        problem.background_density,
        [target.x_km for target in sorted_targets],
        [target.y_km for target in sorted_targets],
    )

    if isinstance(problem.spatial_integrator, PointAreaQuadrature):
        background_mass = problem.spatial_integrator.integrate(problem.background_density)
    else:
        background_mass = problem.spatial_integrator.integrate(
            lambda x_km, y_km: _validated_background_density(
                problem.background_density,
                x_km,
                y_km,
            )
        )
    (
        compensator_lower_age_days,
        compensator_upper_age_days,
        compensator_parent_magnitude,
        compensator_parent_spatial_mass,
    ) = _prepare_compensator_arrays(problem, sorted_parents, spec=spec)
    return PreparedETASLikelihood(
        spec=spec,
        duration_days=problem.assessment_end_days - problem.assessment_start_days,
        background_study_area_mass=background_mass,
        target_event_ids=tuple(target.event_id for target in sorted_targets),
        target_background_spatial_density=target_background_density,
        event_parent_target_index=event_parent_target_index,
        event_parent_delta_days=event_parent_delta_days,
        event_parent_magnitude=event_parent_magnitude,
        event_parent_spatial_density=event_parent_spatial_density,
        compensator_lower_age_days=compensator_lower_age_days,
        compensator_upper_age_days=compensator_upper_age_days,
        compensator_parent_magnitude=compensator_parent_magnitude,
        compensator_parent_spatial_mass=compensator_parent_spatial_mass,
    )


def _triggering_window_weights(
    lower_age_days: FloatArray,
    upper_age_days: FloatArray,
    parent_magnitudes: FloatArray,
    parameters: ETASParameters,
    spec: ETASModelSpec,
) -> FloatArray:
    if not (lower_age_days.shape == upper_age_days.shape == parent_magnitudes.shape):
        raise ValueError("triggering-window arrays must be aligned")
    if lower_age_days.size == 0:
        return _readonly_float([])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            upper_cdf = -np.expm1(
                (1.0 - parameters.p) * np.log1p(upper_age_days / parameters.c_days)
            )
            lower_cdf = -np.expm1(
                (1.0 - parameters.p) * np.log1p(lower_age_days / parameters.c_days)
            )
            weights = (
                parameters.productivity_k
                * np.exp(parameters.alpha * (parent_magnitudes - spec.mc))
                * (upper_cdf - lower_cdf)
            )
    except FloatingPointError as error:
        raise ValueError("triggering-window weights must be finite") from error
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("triggering-window weights must be finite and non-negative")
    return _readonly_float(weights)


def evaluate_prepared_likelihood(
    prepared: PreparedETASLikelihood,
    parameters: ETASParameters,
) -> ETASLikelihoodResult:
    """Evaluate the complete unmarked point-process likelihood.

    The event term uses only study-area targets.  Parent history may include
    study-area and 300-km-buffer events, is truncated at 3650 days of age, and
    begins contributing only once each report is available.  Omori age remains
    measured from physical origin, and the infinite-support kernel is not
    renormalized at the 3650-day cutoff.
    Both background and triggering compensators are integrated only over the
    injected study-area quadrature.
    """

    spec = prepared.spec
    spec.validate_parameters(parameters)
    event_intensity_array = (
        parameters.background_rate_per_day
        * prepared.target_background_spatial_density.astype(np.float64, copy=True)
    )
    try:
        if prepared.event_parent_delta_days.size:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                event_productivity = parameters.productivity_k * np.exp(
                    parameters.alpha * (prepared.event_parent_magnitude - spec.mc)
                )
                event_temporal_density = (
                    (parameters.p - 1.0)
                    / parameters.c_days
                    * np.power(
                        1.0 + prepared.event_parent_delta_days / parameters.c_days,
                        -parameters.p,
                    )
                )
                event_contributions = (
                    event_productivity
                    * event_temporal_density
                    * prepared.event_parent_spatial_density
                )
            event_intensity_array += np.bincount(
                prepared.event_parent_target_index,
                weights=event_contributions,
                minlength=len(prepared.target_event_ids),
            )
        if np.any(event_intensity_array <= 0.0) or not np.all(np.isfinite(event_intensity_array)):
            raise ValueError("target conditional intensity must be finite and positive")
        event_intensities = tuple(float(value) for value in event_intensity_array)
        event_log_sum = math.fsum(math.log(value) for value in event_intensities)

        if prepared.compensator_lower_age_days.size:
            triggering_weights = _triggering_window_weights(
                prepared.compensator_lower_age_days,
                prepared.compensator_upper_age_days,
                prepared.compensator_parent_magnitude,
                parameters,
                spec,
            )
            trigger_terms = triggering_weights * prepared.compensator_parent_spatial_mass
            if np.any(trigger_terms < 0.0) or not np.all(np.isfinite(trigger_terms)):
                raise ValueError("triggering compensator terms must be finite and non-negative")
            triggering_compensator = math.fsum(float(value) for value in trigger_terms)
        else:
            triggering_compensator = 0.0
    except (OverflowError, ValueError, FloatingPointError) as error:
        raise ValueError("ETAS likelihood terms must be finite") from error
    background_compensator = (
        parameters.background_rate_per_day
        * prepared.duration_days
        * prepared.background_study_area_mass
    )
    total_compensator = background_compensator + triggering_compensator
    log_likelihood = event_log_sum - total_compensator
    for name, value in (
        ("event_log_intensity_sum", event_log_sum),
        ("background_compensator", background_compensator),
        ("triggering_compensator", triggering_compensator),
        ("total_compensator", total_compensator),
        ("log_likelihood", log_likelihood),
    ):
        _finite(name, value)
    return ETASLikelihoodResult(
        target_event_ids=prepared.target_event_ids,
        event_intensities=event_intensities,
        event_log_intensity_sum=event_log_sum,
        background_compensator=background_compensator,
        triggering_compensator=triggering_compensator,
        total_compensator=total_compensator,
        log_likelihood=log_likelihood,
    )


def etas_log_likelihood(
    problem: ETASLikelihoodProblem,
    parameters: ETASParameters,
    spec: ETASModelSpec,
) -> ETASLikelihoodResult:
    """Prepare and evaluate one complete point-process likelihood."""

    return evaluate_prepared_likelihood(prepare_etas_likelihood(problem, spec), parameters)


def _validated_readonly_components(
    background: npt.ArrayLike,
    triggering: npt.ArrayLike,
    total: npt.ArrayLike,
    *,
    label: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    background_values = _owned_readonly_float(background)
    triggering_values = _owned_readonly_float(triggering)
    total_values = _owned_readonly_float(total)
    if (
        background_values.ndim != 1
        or background_values.shape != triggering_values.shape
        or background_values.shape != total_values.shape
    ):
        raise ValueError(f"{label} component arrays must be aligned vectors")
    if not (
        np.all(np.isfinite(background_values))
        and np.all(np.isfinite(triggering_values))
        and np.all(np.isfinite(total_values))
    ):
        raise ValueError(f"{label} component arrays must be finite")
    if (
        np.any(background_values < 0.0)
        or np.any(triggering_values < 0.0)
        or np.any(total_values < 0.0)
    ):
        raise ValueError(f"{label} component arrays must be non-negative")
    if not np.allclose(
        total_values,
        background_values + triggering_values,
        rtol=5.0e-15,
        atol=1.0e-15,
    ):
        raise ValueError(f"{label} total must equal background plus triggering")
    return background_values, triggering_values, total_values


@dataclass(frozen=True, slots=True)
class ETASExpectedCellMasses:
    """Per-cell expected event masses over one likelihood interval."""

    background_mass: FloatArray
    triggering_mass: FloatArray
    total_mass: FloatArray
    background_total: float
    triggering_total: float
    total: float

    def __post_init__(self) -> None:
        background, triggering, total = _validated_readonly_components(
            self.background_mass,
            self.triggering_mass,
            self.total_mass,
            label="expected cell mass",
        )
        object.__setattr__(self, "background_mass", background)
        object.__setattr__(self, "triggering_mass", triggering)
        object.__setattr__(self, "total_mass", total)
        totals = (
            _nonnegative("background_total", self.background_total),
            _nonnegative("triggering_total", self.triggering_total),
            _nonnegative("total", self.total),
        )
        if not all(math.isfinite(value) for value in totals):
            raise ValueError("expected-mass totals must be finite")
        object.__setattr__(self, "background_total", totals[0])
        object.__setattr__(self, "triggering_total", totals[1])
        object.__setattr__(self, "total", totals[2])
        calculated = (
            math.fsum(float(value) for value in background),
            math.fsum(float(value) for value in triggering),
            math.fsum(float(value) for value in total),
        )
        if any(
            not math.isclose(reported, expected, rel_tol=5.0e-15, abs_tol=1.0e-15)
            for reported, expected in zip(totals, calculated, strict=True)
        ):
            raise ValueError("expected-mass totals must equal their cell sums")
        if not math.isclose(
            self.total,
            self.background_total + self.triggering_total,
            rel_tol=5.0e-15,
            abs_tol=1.0e-15,
        ):
            raise ValueError("expected-mass total must equal component totals")


@dataclass(frozen=True, slots=True)
class ETASGridPointIntensityField:
    """Background, triggering, and total conditional intensities on one fixed grid."""

    cell_size_km: float
    background_intensity: FloatArray
    triggering_intensity: FloatArray
    total_intensity: FloatArray

    def __post_init__(self) -> None:
        cell_size = _positive("cell_size_km", self.cell_size_km)
        if cell_size not in ISSUE_FIELD_GRID_SIZES_KM:
            raise ValueError("intensity field cell_size_km must be 50, 25, or 12.5")
        object.__setattr__(self, "cell_size_km", cell_size)
        background, triggering, total = _validated_readonly_components(
            self.background_intensity,
            self.triggering_intensity,
            self.total_intensity,
            label="conditional intensity",
        )
        object.__setattr__(self, "background_intensity", background)
        object.__setattr__(self, "triggering_intensity", triggering)
        object.__setattr__(self, "total_intensity", total)


@dataclass(frozen=True, slots=True)
class ETASIssueIntensityFields:
    """Issue-time conditional intensity fields for the frozen 50/25/12.5 km grids."""

    issue_time_days: float
    eligible_parent_event_ids: tuple[str, ...]
    fields: tuple[ETASGridPointIntensityField, ...]

    def __post_init__(self) -> None:
        issue_time = _finite("issue_time_days", self.issue_time_days)
        object.__setattr__(self, "issue_time_days", issue_time)
        if not isinstance(self.eligible_parent_event_ids, tuple) or any(
            not isinstance(event_id, str) or not event_id
            for event_id in self.eligible_parent_event_ids
        ):
            raise TypeError("eligible_parent_event_ids must be an immutable string tuple")
        if len(set(self.eligible_parent_event_ids)) != len(self.eligible_parent_event_ids):
            raise ValueError("eligible parent event IDs must be unique")
        if not isinstance(self.fields, tuple) or any(
            not isinstance(field, ETASGridPointIntensityField) for field in self.fields
        ):
            raise TypeError("fields must be an immutable intensity-field tuple")
        if tuple(field.cell_size_km for field in self.fields) != ISSUE_FIELD_GRID_SIZES_KM:
            raise ValueError("issue intensity fields must remain ordered 50, 25, 12.5 km")

    def at(self, cell_size_km: float) -> ETASGridPointIntensityField:
        requested = float(cell_size_km)
        for grid_field in self.fields:
            if grid_field.cell_size_km == requested:
                return grid_field
        raise KeyError(f"issue intensity fields have no {requested:g} km grid")


def evaluate_etas_cell_expected_masses(
    problem: ETASLikelihoodProblem,
    parameters: ETASParameters,
    spec: ETASModelSpec,
    quadrature: PointAreaQuadrature,
) -> ETASExpectedCellMasses:
    """Integrate expected ETAS event mass into each supplied quadrature cell."""

    if not isinstance(problem, ETASLikelihoodProblem):
        raise TypeError("problem must be ETASLikelihoodProblem")
    if not isinstance(parameters, ETASParameters):
        raise TypeError("parameters must be ETASParameters")
    if not isinstance(spec, ETASModelSpec):
        raise TypeError("spec must be ETASModelSpec")
    if not isinstance(quadrature, PointAreaQuadrature):
        raise TypeError("quadrature must be PointAreaQuadrature")
    spec.validate_parameters(parameters)
    sorted_parents = tuple(
        sorted(problem.parent_events, key=lambda event: (event.time_days, event.event_id))
    )
    for event in sorted_parents:
        spec.validate_event(event)

    background_density = _evaluate_background_density_many(
        problem.background_density,
        quadrature._coordinates[:, 0],
        quadrature._coordinates[:, 1],
    )
    background_mass = (
        parameters.background_rate_per_day
        * (problem.assessment_end_days - problem.assessment_start_days)
        * background_density
        * quadrature._areas
    )
    eligible_parents, lower_ages, upper_ages, magnitudes = _eligible_compensator_parent_windows(
        problem, sorted_parents, spec=spec
    )
    triggering_weights = _triggering_window_weights(
        lower_ages,
        upper_ages,
        magnitudes,
        parameters,
        spec,
    )
    triggering_density = quadrature.inverse_power_weighted_point_sums(
        [parent.x_km for parent in eligible_parents],
        [parent.y_km for parent in eligible_parents],
        magnitudes,
        triggering_weights,
        spec=spec,
    )
    triggering_mass = triggering_density * quadrature._areas
    total_mass = background_mass + triggering_mass
    return ETASExpectedCellMasses(
        background_mass=background_mass,
        triggering_mass=triggering_mass,
        total_mass=total_mass,
        background_total=math.fsum(float(value) for value in background_mass),
        triggering_total=math.fsum(float(value) for value in triggering_mass),
        total=math.fsum(float(value) for value in total_mass),
    )


def _instantaneous_triggering_weights(
    ages_days: FloatArray,
    parent_magnitudes: FloatArray,
    parameters: ETASParameters,
    spec: ETASModelSpec,
) -> FloatArray:
    if ages_days.shape != parent_magnitudes.shape:
        raise ValueError("instantaneous parent ages and magnitudes must be aligned")
    if ages_days.size == 0:
        return _readonly_float([])
    if np.any(ages_days <= 0.0) or np.any(ages_days > spec.history_parent_cutoff_days):
        raise ValueError("instantaneous parent ages must lie in (0, 3650]")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            productivity_values = parameters.productivity_k * np.exp(
                parameters.alpha * (parent_magnitudes - spec.mc)
            )
            temporal_density = (
                (parameters.p - 1.0)
                / parameters.c_days
                * np.power(1.0 + ages_days / parameters.c_days, -parameters.p)
            )
            weights = productivity_values * temporal_density
    except FloatingPointError as error:
        raise ValueError("instantaneous triggering weights must be finite") from error
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("instantaneous triggering weights must be finite and non-negative")
    return _readonly_float(weights)


def evaluate_etas_issue_intensity_fields(
    parameters: ETASParameters,
    spec: ETASModelSpec,
    history_events: Sequence[ETASEvent],
    background_density: BackgroundDensity,
    quadratures_by_cell_size_km: Mapping[float, PointAreaQuadrature],
    *,
    issue_time_days: float = 0.0,
) -> ETASIssueIntensityFields:
    """Evaluate one conditional ETAS intensity snapshot on all three frozen grids."""

    if not isinstance(parameters, ETASParameters):
        raise TypeError("parameters must be ETASParameters")
    if not isinstance(spec, ETASModelSpec):
        raise TypeError("spec must be ETASModelSpec")
    issue_time = _finite("issue_time_days", issue_time_days)
    spec.validate_parameters(parameters)
    normalized_quadratures: dict[float, PointAreaQuadrature] = {}
    for raw_cell_size, quadrature in quadratures_by_cell_size_km.items():
        if isinstance(raw_cell_size, bool):
            raise TypeError("grid cell sizes must be numeric")
        cell_size = float(raw_cell_size)
        if not isinstance(quadrature, PointAreaQuadrature):
            raise TypeError("every issue field grid must be PointAreaQuadrature")
        if cell_size in normalized_quadratures:
            raise ValueError("issue field grid sizes must be unique")
        normalized_quadratures[cell_size] = quadrature
    if set(normalized_quadratures) != set(ISSUE_FIELD_GRID_SIZES_KM):
        raise ValueError("issue intensity fields require exactly 50, 25, and 12.5 km grids")

    seen_ids: set[str] = set()
    eligible: list[ETASEvent] = []
    for event in history_events:
        if not isinstance(event, ETASEvent):
            raise TypeError("history_events must contain ETASEvent values")
        if event.event_id in seen_ids:
            raise ValueError("history event IDs must be unique")
        seen_ids.add(event.event_id)
        if not event.inside_parent_domain:
            raise ValueError("history events must lie inside the ETAS parent domain")
        spec.validate_event(event)
        age = issue_time - event.time_days
        if (
            age > 0.0
            and age <= spec.history_parent_cutoff_days
            and event.available_time_days <= issue_time
        ):
            eligible.append(event)
    eligible.sort(key=lambda event: (event.time_days, event.event_id))
    parent_tuple = tuple(eligible)
    parent_x = _readonly_float([event.x_km for event in parent_tuple])
    parent_y = _readonly_float([event.y_km for event in parent_tuple])
    parent_magnitudes = _readonly_float([event.magnitude for event in parent_tuple])
    parent_ages = _readonly_float([issue_time - event.time_days for event in parent_tuple])
    triggering_weights = _instantaneous_triggering_weights(
        parent_ages,
        parent_magnitudes,
        parameters,
        spec,
    )

    fields: list[ETASGridPointIntensityField] = []
    for cell_size in ISSUE_FIELD_GRID_SIZES_KM:
        quadrature = normalized_quadratures[cell_size]
        background_values = parameters.background_rate_per_day * (
            _evaluate_background_density_many(
                background_density,
                quadrature._coordinates[:, 0],
                quadrature._coordinates[:, 1],
            )
        )
        triggering_values = quadrature.inverse_power_weighted_point_sums(
            parent_x,
            parent_y,
            parent_magnitudes,
            triggering_weights,
            spec=spec,
        )
        fields.append(
            ETASGridPointIntensityField(
                cell_size_km=cell_size,
                background_intensity=background_values,
                triggering_intensity=triggering_values,
                total_intensity=background_values + triggering_values,
            )
        )
    return ETASIssueIntensityFields(
        issue_time_days=issue_time,
        eligible_parent_event_ids=tuple(event.event_id for event in parent_tuple),
        fields=tuple(fields),
    )


def etas_objective(
    problem: ETASLikelihoodProblem,
    spec: ETASModelSpec,
    bounds: ETASParameterBounds,
) -> Objective:
    """Build the frozen transformed negative-log-likelihood objective."""

    prepared = prepare_etas_likelihood(problem, spec)

    def objective(transformed: FloatArray) -> float:
        try:
            parameters = bounds.from_transformed(transformed)
            result = evaluate_prepared_likelihood(prepared, parameters)
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return math.inf
        value = -result.log_likelihood
        return value if math.isfinite(value) else math.inf

    return objective


class NumericalStencilError(ValueError):
    """Raised when the frozen finite-difference stencil cannot be evaluated."""


def _inside_bounds(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def three_point_gradient(
    objective: Objective,
    transformed: npt.ArrayLike,
    bounds: Sequence[tuple[float, float]],
    *,
    relative_step: float = 1.0e-6,
) -> FloatArray:
    """Evaluate the frozen central/second-order-one-sided 3-point gradient."""

    point = np.asarray(transformed, dtype=np.float64)
    if point.ndim != 1 or len(point) != len(bounds) or not np.all(np.isfinite(point)):
        raise ValueError("gradient point and bounds must be finite and dimensionally equal")
    step_scale = _positive("relative_step", relative_step)
    base = float(objective(point.copy()))
    if not math.isfinite(base):
        raise NumericalStencilError("objective is not finite at gradient point")
    gradient = np.empty_like(point)
    for index, value in enumerate(point):
        step = step_scale * max(1.0, abs(float(value)))
        lower, upper = bounds[index]

        minus = point.copy()
        plus = point.copy()
        minus[index] -= step
        plus[index] += step
        minus_value = (
            float(objective(minus))
            if _inside_bounds(float(minus[index]), (lower, upper))
            else math.inf
        )
        plus_value = (
            float(objective(plus))
            if _inside_bounds(float(plus[index]), (lower, upper))
            else math.inf
        )
        if math.isfinite(minus_value) and math.isfinite(plus_value):
            gradient[index] = (plus_value - minus_value) / (2.0 * step)
            continue

        forward2 = point.copy()
        forward2[index] += 2.0 * step
        forward2_value = (
            float(objective(forward2))
            if _inside_bounds(float(forward2[index]), (lower, upper))
            else math.inf
        )
        if math.isfinite(plus_value) and math.isfinite(forward2_value):
            gradient[index] = (-3.0 * base + 4.0 * plus_value - forward2_value) / (2.0 * step)
            continue

        backward2 = point.copy()
        backward2[index] -= 2.0 * step
        backward2_value = (
            float(objective(backward2))
            if _inside_bounds(float(backward2[index]), (lower, upper))
            else math.inf
        )
        if math.isfinite(minus_value) and math.isfinite(backward2_value):
            gradient[index] = (3.0 * base - 4.0 * minus_value + backward2_value) / (2.0 * step)
            continue
        raise NumericalStencilError(f"no valid three-point gradient stencil for index {index}")
    if not np.all(np.isfinite(gradient)):
        raise NumericalStencilError("three-point gradient is non-finite")
    return gradient


def central_hessian(
    objective: Objective,
    transformed: npt.ArrayLike,
    bounds: Sequence[tuple[float, float]],
    *,
    relative_step: float = 1.0e-4,
) -> FloatArray:
    """Evaluate the frozen symmetric central second-difference Hessian."""

    point = np.asarray(transformed, dtype=np.float64)
    if point.ndim != 1 or len(point) != len(bounds) or not np.all(np.isfinite(point)):
        raise ValueError("Hessian point and bounds must be finite and dimensionally equal")
    step_scale = _positive("relative_step", relative_step)
    steps = step_scale * np.maximum(1.0, np.abs(point))
    base = float(objective(point.copy()))
    if not math.isfinite(base):
        raise NumericalStencilError("objective is not finite at Hessian point")
    dimension = len(point)
    hessian = np.empty((dimension, dimension), dtype=np.float64)

    def evaluate(candidate: FloatArray) -> float:
        if any(
            not _inside_bounds(float(candidate[index]), bounds[index]) for index in range(dimension)
        ):
            raise NumericalStencilError("central Hessian stencil crosses transformed bounds")
        value = float(objective(candidate))
        if not math.isfinite(value):
            raise NumericalStencilError("central Hessian stencil has non-finite objective")
        return value

    for row in range(dimension):
        plus = point.copy()
        minus = point.copy()
        plus[row] += steps[row]
        minus[row] -= steps[row]
        hessian[row, row] = (evaluate(plus) - 2.0 * base + evaluate(minus)) / steps[row] ** 2
        for column in range(row):
            plus_plus = point.copy()
            plus_minus = point.copy()
            minus_plus = point.copy()
            minus_minus = point.copy()
            plus_plus[row] += steps[row]
            plus_plus[column] += steps[column]
            plus_minus[row] += steps[row]
            plus_minus[column] -= steps[column]
            minus_plus[row] -= steps[row]
            minus_plus[column] += steps[column]
            minus_minus[row] -= steps[row]
            minus_minus[column] -= steps[column]
            value = (
                evaluate(plus_plus)
                - evaluate(plus_minus)
                - evaluate(minus_plus)
                + evaluate(minus_minus)
            ) / (4.0 * steps[row] * steps[column])
            hessian[row, column] = value
            hessian[column, row] = value
    hessian = (hessian + hessian.T) / 2.0
    if not np.all(np.isfinite(hessian)):
        raise NumericalStencilError("central Hessian is non-finite")
    return hessian


@dataclass(frozen=True, slots=True)
class OptimizerOptions:
    """Frozen L-BFGS-B and gradient settings."""

    ftol: float = 1.0e-12
    gtol: float = 1.0e-6
    maxiter: int = 500
    maxfun: int = 100_000
    maxls: int = 20
    gradient_relative_step: float = 1.0e-6

    def __post_init__(self) -> None:
        _positive("optimizer ftol", self.ftol)
        _positive("optimizer gtol", self.gtol)
        _positive("gradient_relative_step", self.gradient_relative_step)
        for name in ("maxiter", "maxfun", "maxls"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class StabilityThresholds:
    """Frozen multistart and Hessian acceptance thresholds."""

    minimum_converged_starts: int = 4
    gradient_infinity_norm_maximum: float = 1.0e-4
    best_three_relative_objective_range_maximum: float = 1.0e-4
    transformed_parameter_maximum_range: float = 0.1
    hessian_minimum_eigenvalue: float = 1.0e-8
    hessian_condition_number_maximum: float = 1.0e10
    hessian_relative_step: float = 1.0e-4

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum_converged_starts, int)
            or isinstance(self.minimum_converged_starts, bool)
            or self.minimum_converged_starts <= 0
        ):
            raise ValueError("minimum_converged_starts must be a positive integer")
        for name in (
            "gradient_infinity_norm_maximum",
            "best_three_relative_objective_range_maximum",
            "transformed_parameter_maximum_range",
            "hessian_minimum_eigenvalue",
            "hessian_condition_number_maximum",
            "hessian_relative_step",
        ):
            _positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ETASStartResult:
    start_index: int
    initial_transformed: tuple[float, ...]
    final_transformed: tuple[float, ...]
    objective: float
    scipy_converged: bool
    gradient_infinity_norm: float
    iterations: int
    function_evaluations: int
    message: str


@dataclass(frozen=True, slots=True)
class HessianAudit:
    success: bool
    minimum_eigenvalue: float | None
    condition_number: float | None
    matrix: tuple[tuple[float, ...], ...] | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class StabilityAudit:
    stable: bool
    converged_start_count: int
    best_three_relative_objective_range: float | None
    best_three_transformed_parameter_range: float | None
    hessian: HessianAudit
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeltaEstimate:
    """One immutable linear Delta-method estimate and confidence interval."""

    name: str
    estimate: float
    standard_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float

    def __post_init__(self) -> None:
        if self.name not in {*ETAS_PARAMETER_ORDER, "branching_ratio"}:
            raise ValueError("unknown Delta-method estimate name")
        for field_name in (
            "estimate",
            "standard_error",
            "confidence_interval_lower",
            "confidence_interval_upper",
        ):
            raw_value = getattr(self, field_name)
            if isinstance(raw_value, bool | np.bool_):
                raise TypeError(f"{field_name} must be a real scalar")
            value = _finite(field_name, raw_value)
            object.__setattr__(self, field_name, value)
        if self.standard_error < 0.0:
            raise ValueError("Delta-method standard_error must be non-negative")
        if not self.confidence_interval_lower <= self.estimate <= self.confidence_interval_upper:
            raise ValueError("Delta-method confidence interval must contain its estimate")
        expected_half_width = _DELTA_NORMAL_QUANTILE * self.standard_error
        if not math.isclose(
            self.confidence_interval_lower,
            self.estimate - expected_half_width,
            rel_tol=1.0e-15,
            abs_tol=1.0e-15,
        ) or not math.isclose(
            self.confidence_interval_upper,
            self.estimate + expected_half_width,
            rel_tol=1.0e-15,
            abs_tol=1.0e-15,
        ):
            raise ValueError("Delta-method confidence interval must use the frozen 95% quantile")


def _strict_positive_definite_matrix(
    value: tuple[tuple[float, ...], ...],
    *,
    name: str,
) -> FloatArray:
    if not isinstance(value, tuple) or len(value) != 5:
        raise TypeError(f"{name} must be an immutable 5-by-5 tuple")
    if any(not isinstance(row, tuple) or len(row) != 5 for row in value):
        raise TypeError(f"{name} must be an immutable 5-by-5 tuple")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (5, 5) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite and 5-by-5")
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=0.0,
        atol=_MATRIX_SYMMETRY_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{name} must be symmetric")
    if float(np.linalg.eigvalsh(matrix)[0]) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return matrix


@dataclass(frozen=True, slots=True)
class ObservedHessianDeltaUncertainty:
    """Frozen 95% observed-Hessian Delta uncertainty on transformed and physical scales."""

    confidence_level: float
    transformed_parameter_order: tuple[str, ...]
    physical_parameter_order: tuple[str, ...]
    transformed_covariance: tuple[tuple[float, ...], ...]
    physical_covariance: tuple[tuple[float, ...], ...]
    parameter_estimates: tuple[DeltaEstimate, ...]
    branching_ratio: DeltaEstimate

    def __post_init__(self) -> None:
        confidence = _finite("confidence_level", self.confidence_level)
        if confidence != DELTA_CONFIDENCE_LEVEL:
            raise ValueError("observed-Hessian Delta confidence level must remain 0.95")
        object.__setattr__(self, "confidence_level", confidence)
        if self.transformed_parameter_order != ETAS_TRANSFORMED_PARAMETER_ORDER:
            raise ValueError("transformed parameter order differs from the frozen order")
        if self.physical_parameter_order != ETAS_PARAMETER_ORDER:
            raise ValueError("physical parameter order differs from the frozen order")
        _strict_positive_definite_matrix(
            self.transformed_covariance,
            name="transformed_covariance",
        )
        _strict_positive_definite_matrix(
            self.physical_covariance,
            name="physical_covariance",
        )
        if not isinstance(self.parameter_estimates, tuple) or len(self.parameter_estimates) != len(
            ETAS_PARAMETER_ORDER
        ):
            raise TypeError("parameter_estimates must be an immutable five-element tuple")
        if any(not isinstance(item, DeltaEstimate) for item in self.parameter_estimates):
            raise TypeError("parameter_estimates must contain DeltaEstimate values")
        if tuple(item.name for item in self.parameter_estimates) != ETAS_PARAMETER_ORDER:
            raise ValueError("Delta-method parameter estimates differ from the frozen order")
        if not isinstance(self.branching_ratio, DeltaEstimate):
            raise TypeError("branching_ratio must be a DeltaEstimate")
        if self.branching_ratio.name != "branching_ratio":
            raise ValueError("branching_ratio estimate has the wrong name")


@dataclass(frozen=True, slots=True)
class ETASFitResult:
    best_parameters: ETASParameters | None
    best_objective: float | None
    start_results: tuple[ETASStartResult, ...]
    stability: StabilityAudit
    uncertainty: ObservedHessianDeltaUncertainty | None = None

    def __post_init__(self) -> None:
        if (self.best_parameters is None) != (self.best_objective is None):
            raise ValueError("best_parameters and best_objective must be present together")
        if self.best_objective is not None:
            _finite("best_objective", self.best_objective)
        if not isinstance(self.start_results, tuple) or any(
            not isinstance(item, ETASStartResult) for item in self.start_results
        ):
            raise TypeError("start_results must be an immutable ETASStartResult tuple")
        if not isinstance(self.stability, StabilityAudit):
            raise TypeError("stability must be a StabilityAudit")
        if self.stability.stable:
            if self.best_parameters is None:
                raise ValueError("a stable ETAS fit must contain best_parameters")
            if self.uncertainty is None:
                raise ValueError("a stable ETAS fit must contain observed-Hessian uncertainty")
        elif self.uncertainty is not None:
            raise ValueError("an unstable ETAS fit must not expose parameter uncertainty")
        if self.uncertainty is not None and not isinstance(
            self.uncertainty,
            ObservedHessianDeltaUncertainty,
        ):
            raise TypeError("uncertainty must be ObservedHessianDeltaUncertainty")
        if self.uncertainty is not None and self.best_parameters is not None:
            physical_values = (
                self.best_parameters.background_rate_per_day,
                self.best_parameters.productivity_k,
                self.best_parameters.alpha,
                self.best_parameters.c_days,
                self.best_parameters.p,
            )
            if (
                tuple(item.estimate for item in self.uncertainty.parameter_estimates)
                != physical_values
            ):
                raise ValueError("uncertainty estimates must match best_parameters exactly")


def _matrix_tuple(matrix: FloatArray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _truncated_gr_expectation_alpha_derivative(spec: ETASModelSpec, alpha: float) -> float:
    """Return d E[exp(alpha (M-Mc))] / d alpha for the frozen truncated GR law."""

    alpha_value = _positive("alpha", alpha)
    beta = spec.beta
    span = spec.magnitude_span
    scaled_difference = (alpha_value - beta) * span
    if abs(scaled_difference) < 1.0e-4:
        term = 0.5
        unit_interval_integral = term
        for index in range(1, 100):
            term *= scaled_difference / index * (index + 1.0) / (index + 2.0)
            updated = unit_interval_integral + term
            if updated == unit_interval_integral:
                break
            unit_interval_integral = updated
        first_moment_integral = span * span * unit_interval_integral
    else:
        exponential = math.exp(scaled_difference)
        first_moment_integral = (
            span
            * span
            * (scaled_difference * exponential - math.expm1(scaled_difference))
            / (scaled_difference * scaled_difference)
        )
    normalizer = -math.expm1(-beta * span)
    derivative = beta * first_moment_integral / normalizer
    return _positive("truncated-GR expectation alpha derivative", derivative)


def _delta_estimate(name: str, estimate: float, variance: float) -> DeltaEstimate:
    variance_value = _finite(f"{name} variance", variance)
    if variance_value < 0.0:
        raise ValueError(f"{name} Delta-method variance must be non-negative")
    standard_error = math.sqrt(variance_value)
    half_width = _DELTA_NORMAL_QUANTILE * standard_error
    return DeltaEstimate(
        name=name,
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval_lower=estimate - half_width,
        confidence_interval_upper=estimate + half_width,
    )


def observed_hessian_delta_uncertainty(
    parameters: ETASParameters | None,
    stability: StabilityAudit,
    spec: ETASModelSpec,
) -> ObservedHessianDeltaUncertainty:
    """Compute the frozen 95% linear Delta uncertainty or fail closed.

    The observed Hessian is on ``log(mu), log(K), log(alpha), log(c),
    log(p-1)``.  This function independently rechecks its positive-definite
    and conditioning gates before inversion; audit flags alone are not trusted.
    """

    if parameters is None:
        raise ValueError("observed-Hessian uncertainty requires fitted parameters")
    if not isinstance(parameters, ETASParameters):
        raise TypeError("parameters must be ETASParameters")
    if not isinstance(stability, StabilityAudit):
        raise TypeError("stability must be StabilityAudit")
    if not isinstance(spec, ETASModelSpec):
        raise TypeError("spec must be ETASModelSpec")
    if not stability.stable:
        raise ValueError("observed-Hessian uncertainty requires a stable ETAS fit")
    if stability.failure_reasons:
        raise ValueError("a stable ETAS audit must not contain failure reasons")
    hessian_audit = stability.hessian
    if not hessian_audit.success or hessian_audit.matrix is None:
        raise ValueError("observed-Hessian uncertainty requires a successful Hessian audit")
    if hessian_audit.minimum_eigenvalue is None or hessian_audit.condition_number is None:
        raise ValueError("successful Hessian audit is missing eigenvalue diagnostics")

    hessian = np.asarray(hessian_audit.matrix, dtype=np.float64)
    if hessian.shape != (5, 5) or not np.all(np.isfinite(hessian)):
        raise ValueError("observed Hessian must be a finite 5-by-5 matrix")
    if not np.allclose(
        hessian,
        hessian.T,
        rtol=0.0,
        atol=_MATRIX_SYMMETRY_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("observed Hessian must be symmetric")
    eigenvalues = np.linalg.eigvalsh(hessian)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    if minimum_eigenvalue <= 0.0:
        raise ValueError("observed Hessian must be positive definite")
    condition_number = maximum_eigenvalue / minimum_eigenvalue
    frozen_thresholds = StabilityThresholds()
    if minimum_eigenvalue < frozen_thresholds.hessian_minimum_eigenvalue:
        raise ValueError("observed Hessian fails the frozen minimum-eigenvalue gate")
    if not math.isfinite(condition_number) or (
        condition_number > frozen_thresholds.hessian_condition_number_maximum
    ):
        raise ValueError("observed Hessian fails the frozen condition-number gate")
    if not math.isclose(
        hessian_audit.minimum_eigenvalue,
        minimum_eigenvalue,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        hessian_audit.condition_number,
        condition_number,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Hessian matrix and audit diagnostics disagree")

    spec.validate_parameters(parameters)
    transformed_jacobian_diagonal = np.asarray(
        [
            _positive("background_rate_per_day", parameters.background_rate_per_day),
            _positive("productivity_k", parameters.productivity_k),
            _positive("alpha", parameters.alpha),
            _positive("c_days", parameters.c_days),
            _positive("p_minus_one", parameters.p - 1.0),
        ],
        dtype=np.float64,
    )
    try:
        transformed_covariance = np.linalg.solve(hessian, np.eye(5, dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise ValueError("observed Hessian inversion failed") from error
    transformed_covariance = (transformed_covariance + transformed_covariance.T) / 2.0
    if not np.all(np.isfinite(transformed_covariance)):
        raise ValueError("transformed covariance must be finite")
    physical_covariance = (
        transformed_jacobian_diagonal[:, None]
        * transformed_covariance
        * transformed_jacobian_diagonal[None, :]
    )
    physical_covariance = (physical_covariance + physical_covariance.T) / 2.0
    if not np.all(np.isfinite(physical_covariance)):
        raise ValueError("physical covariance must be finite")

    physical_values = (
        parameters.background_rate_per_day,
        parameters.productivity_k,
        parameters.alpha,
        parameters.c_days,
        parameters.p,
    )
    parameter_estimates = tuple(
        _delta_estimate(name, float(value), float(physical_covariance[index, index]))
        for index, (name, value) in enumerate(
            zip(ETAS_PARAMETER_ORDER, physical_values, strict=True)
        )
    )

    branching_ratio_value = spec.branching_ratio(parameters)
    productivity_expectation = branching_ratio_value / parameters.productivity_k
    expectation_derivative = _truncated_gr_expectation_alpha_derivative(
        spec,
        parameters.alpha,
    )
    branching_gradient = np.asarray(
        [
            0.0,
            productivity_expectation,
            parameters.productivity_k * expectation_derivative,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    branching_variance = float(branching_gradient @ physical_covariance @ branching_gradient)
    branching_estimate = _delta_estimate(
        "branching_ratio",
        branching_ratio_value,
        branching_variance,
    )
    return ObservedHessianDeltaUncertainty(
        confidence_level=DELTA_CONFIDENCE_LEVEL,
        transformed_parameter_order=ETAS_TRANSFORMED_PARAMETER_ORDER,
        physical_parameter_order=ETAS_PARAMETER_ORDER,
        transformed_covariance=_matrix_tuple(transformed_covariance),
        physical_covariance=_matrix_tuple(physical_covariance),
        parameter_estimates=parameter_estimates,
        branching_ratio=branching_estimate,
    )


def optimizer_start(
    bounds: Sequence[tuple[float, float]],
    *,
    root_seed: int,
    protocol_version: str,
    model_id: str,
    start_index: int,
) -> FloatArray:
    """Generate one transformed start from its namespaced PCG64 stream."""

    generator = SeedContext(
        root_seed=root_seed,
        protocol_version=protocol_version,
        namespace="optimizer_start",
        model_id=model_id,
        issue_id=None,
        replicate_index=start_index,
    ).generator()
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    if lower.ndim != 1 or len(lower) == 0 or not np.all(np.isfinite(lower)):
        raise ValueError("optimizer bounds must be non-empty and finite")
    if not np.all(upper > lower):
        raise ValueError("optimizer bounds must be strictly increasing")
    return np.asarray(generator.uniform(lower, upper), dtype=np.float64)


def _failed_start(index: int, initial: FloatArray, message: str) -> ETASStartResult:
    return ETASStartResult(
        start_index=index,
        initial_transformed=tuple(float(value) for value in initial),
        final_transformed=tuple(float(value) for value in initial),
        objective=math.inf,
        scipy_converged=False,
        gradient_infinity_norm=math.inf,
        iterations=0,
        function_evaluations=1,
        message=message,
    )


def run_five_start_lbfgsb(
    objective: Objective,
    bounds: Sequence[tuple[float, float]],
    *,
    root_seed: int,
    protocol_version: str,
    model_id: str,
    options: OptimizerOptions | None = None,
) -> tuple[ETASStartResult, ...]:
    """Run exactly five independently namespaced L-BFGS-B starts."""

    effective_options = options if options is not None else OptimizerOptions()
    results: list[ETASStartResult] = []
    for start_index in range(5):
        initial = optimizer_start(
            bounds,
            root_seed=root_seed,
            protocol_version=protocol_version,
            model_id=model_id,
            start_index=start_index,
        )
        if not math.isfinite(float(objective(initial.copy()))):
            results.append(_failed_start(start_index, initial, "initial objective is invalid"))
            continue

        def gradient(point: FloatArray) -> FloatArray:
            return three_point_gradient(
                objective,
                point,
                bounds,
                relative_step=effective_options.gradient_relative_step,
            )

        try:
            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=gradient,
                bounds=tuple(bounds),
                tol=None,
                options={
                    "ftol": effective_options.ftol,
                    "gtol": effective_options.gtol,
                    "maxiter": effective_options.maxiter,
                    "maxfun": effective_options.maxfun,
                    "maxls": effective_options.maxls,
                },
            )
            final = np.asarray(result.x, dtype=np.float64)
            final_objective = float(objective(final.copy()))
            final_gradient = three_point_gradient(
                objective,
                final,
                bounds,
                relative_step=effective_options.gradient_relative_step,
            )
            gradient_norm = float(np.linalg.norm(final_gradient, ord=np.inf))
            converged = bool(result.success) and math.isfinite(final_objective)
            results.append(
                ETASStartResult(
                    start_index=start_index,
                    initial_transformed=tuple(float(value) for value in initial),
                    final_transformed=tuple(float(value) for value in final),
                    objective=final_objective,
                    scipy_converged=converged,
                    gradient_infinity_norm=gradient_norm,
                    iterations=int(result.nit),
                    function_evaluations=int(result.nfev),
                    message=str(result.message),
                )
            )
        except (NumericalStencilError, TypeError, ValueError, FloatingPointError) as error:
            results.append(_failed_start(start_index, initial, f"optimizer failed: {error}"))
    return tuple(results)


def audit_stability(
    objective: Objective,
    start_results: Sequence[ETASStartResult],
    bounds: Sequence[tuple[float, float]],
    *,
    thresholds: StabilityThresholds | None = None,
) -> StabilityAudit:
    """Apply the frozen convergence, best-three, and Hessian gates."""

    effective_thresholds = thresholds if thresholds is not None else StabilityThresholds()
    converged = sorted(
        (
            result
            for result in start_results
            if result.scipy_converged
            and math.isfinite(result.objective)
            and math.isfinite(result.gradient_infinity_norm)
        ),
        key=lambda item: (item.objective, item.start_index),
    )
    reasons: list[str] = []
    if len(converged) < effective_thresholds.minimum_converged_starts:
        reasons.append("fewer than four L-BFGS-B starts converged")
    if converged and any(
        result.gradient_infinity_norm > effective_thresholds.gradient_infinity_norm_maximum
        for result in converged
    ):
        reasons.append("a converged start exceeds the gradient infinity-norm threshold")

    objective_range: float | None = None
    parameter_range: float | None = None
    if len(converged) >= 3:
        best_three = converged[:3]
        best_objective = best_three[0].objective
        objective_range = (best_three[-1].objective - best_objective) / max(
            1.0, abs(best_objective)
        )
        matrix = np.asarray(
            [result.final_transformed for result in best_three],
            dtype=np.float64,
        )
        parameter_range = float(np.max(np.ptp(matrix, axis=0)))
        if objective_range > effective_thresholds.best_three_relative_objective_range_maximum:
            reasons.append("best-three relative objective range exceeds threshold")
        if parameter_range > effective_thresholds.transformed_parameter_maximum_range:
            reasons.append("best-three transformed parameter range exceeds threshold")
    else:
        reasons.append("fewer than three converged starts are available for agreement audit")

    hessian_audit: HessianAudit
    if converged:
        best_point = converged[0].final_transformed
        try:
            hessian = central_hessian(
                objective,
                best_point,
                bounds,
                relative_step=effective_thresholds.hessian_relative_step,
            )
            eigenvalues = np.linalg.eigvalsh(hessian)
            minimum_eigenvalue = float(eigenvalues[0])
            maximum_eigenvalue = float(eigenvalues[-1])
            condition_number = (
                maximum_eigenvalue / minimum_eigenvalue if minimum_eigenvalue > 0.0 else math.inf
            )
            hessian_success = (
                minimum_eigenvalue >= effective_thresholds.hessian_minimum_eigenvalue
                and condition_number <= effective_thresholds.hessian_condition_number_maximum
            )
            if not hessian_success:
                reasons.append("observed Hessian fails positive-definiteness or conditioning gates")
            hessian_audit = HessianAudit(
                success=hessian_success,
                minimum_eigenvalue=minimum_eigenvalue,
                condition_number=condition_number,
                matrix=tuple(tuple(float(value) for value in row) for row in hessian),
                failure_reason=None if hessian_success else "eigenvalue or condition-number gate",
            )
        except (NumericalStencilError, ValueError, np.linalg.LinAlgError) as error:
            reasons.append("central Hessian audit could not be evaluated")
            hessian_audit = HessianAudit(
                success=False,
                minimum_eigenvalue=None,
                condition_number=None,
                matrix=None,
                failure_reason=str(error),
            )
    else:
        hessian_audit = HessianAudit(
            success=False,
            minimum_eigenvalue=None,
            condition_number=None,
            matrix=None,
            failure_reason="no converged start",
        )
    return StabilityAudit(
        stable=not reasons,
        converged_start_count=len(converged),
        best_three_relative_objective_range=objective_range,
        best_three_transformed_parameter_range=parameter_range,
        hessian=hessian_audit,
        failure_reasons=tuple(reasons),
    )


def fit_etas(
    problem: ETASLikelihoodProblem,
    spec: ETASModelSpec,
    *,
    root_seed: int,
    protocol_version: str,
    model_id: str,
    bounds: ETASParameterBounds | None = None,
    options: OptimizerOptions | None = None,
    thresholds: StabilityThresholds | None = None,
) -> ETASFitResult:
    """Fit one causal ETAS snapshot and return all pass/fail audit evidence."""

    effective_bounds = bounds if bounds is not None else ETASParameterBounds()
    objective = etas_objective(problem, spec, effective_bounds)
    transformed_bounds = effective_bounds.transformed()
    start_results = run_five_start_lbfgsb(
        objective,
        transformed_bounds,
        root_seed=root_seed,
        protocol_version=protocol_version,
        model_id=model_id,
        options=options,
    )
    stability = audit_stability(
        objective,
        start_results,
        transformed_bounds,
        thresholds=thresholds,
    )
    eligible = [
        result
        for result in start_results
        if result.scipy_converged and math.isfinite(result.objective)
    ]
    if not eligible:
        return ETASFitResult(
            best_parameters=None,
            best_objective=None,
            start_results=start_results,
            stability=stability,
            uncertainty=None,
        )
    best = min(eligible, key=lambda item: (item.objective, item.start_index))
    best_parameters = effective_bounds.from_transformed(best.final_transformed)
    spec.validate_parameters(best_parameters)
    uncertainty = (
        observed_hessian_delta_uncertainty(best_parameters, stability, spec)
        if stability.stable
        else None
    )
    return ETASFitResult(
        best_parameters=best_parameters,
        best_objective=best.objective,
        start_results=start_results,
        stability=stability,
        uncertainty=uncertainty,
    )
