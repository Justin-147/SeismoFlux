"""Uniform and boundary-normalized spatial Poisson background baselines."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.grid import EqualAreaGrid

FloatArray = NDArray[np.float64]
FROZEN_BANDWIDTHS_KM = (75.0, 100.0, 150.0, 200.0, 300.0)
_MAX_DISTANCE_BLOCK_ELEMENTS = 2_000_000


def _vector(name: str, value: object, *, allow_empty: bool = False) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _coordinates(
    x_km: object, y_km: object, *, allow_empty: bool = False
) -> tuple[FloatArray, FloatArray]:
    x = _vector("x_km", x_km, allow_empty=allow_empty)
    y = _vector("y_km", y_km, allow_empty=allow_empty)
    if x.shape != y.shape:
        raise ValueError("x_km and y_km must have the same shape")
    return x, y


def _owned_read_only_vector(name: str, value: object) -> FloatArray:
    result = np.array(_vector(name, value), dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validated_bandwidths(bandwidths_km: tuple[float, ...]) -> tuple[float, ...]:
    bandwidths = tuple(float(value) for value in bandwidths_km)
    if not bandwidths or any(not math.isfinite(value) or value <= 0.0 for value in bandwidths):
        raise ValueError("bandwidths_km must contain positive finite values")
    if len(set(bandwidths)) != len(bandwidths):
        raise ValueError("bandwidths_km must be unique")
    return bandwidths


def _squared_distance_block(
    query_points_km: FloatArray,
    training_points_km: FloatArray,
    training_squared_norms_km2: FloatArray,
) -> FloatArray:
    """Return pairwise squared distances with one query-by-training allocation."""

    squared_distance = cast(
        FloatArray,
        np.matmul(query_points_km, training_points_km.T),
    )
    squared_distance *= -2.0
    squared_distance += training_squared_norms_km2[None, :]
    query_squared_norms = np.sum(query_points_km * query_points_km, axis=1)
    squared_distance += query_squared_norms[:, None]
    # The norm identity can produce tiny negatives for coincident projected points.
    np.maximum(squared_distance, 0.0, out=squared_distance)
    return squared_distance


@dataclass(frozen=True, slots=True)
class SpatialQuadrature:
    """Representative points and exact clipped areas from one frozen grid."""

    cell_ids: tuple[str, ...]
    x_km: FloatArray
    y_km: FloatArray
    area_km2: FloatArray

    def __post_init__(self) -> None:
        x = _owned_read_only_vector("x_km", self.x_km)
        y = _owned_read_only_vector("y_km", self.y_km)
        if x.shape != y.shape:
            raise ValueError("x_km and y_km must have the same shape")
        area = _owned_read_only_vector("area_km2", self.area_km2)
        cell_ids = tuple(self.cell_ids)
        if x.shape != area.shape or len(self.cell_ids) != x.size:
            raise ValueError("quadrature columns must have one common length")
        if np.any(area <= 0.0):
            raise ValueError("quadrature areas must be positive")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("quadrature cell IDs must be unique")
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "x_km", x)
        object.__setattr__(self, "y_km", y)
        object.__setattr__(self, "area_km2", area)

    @classmethod
    def from_grid(cls, grid: EqualAreaGrid) -> SpatialQuadrature:
        return cls(
            cell_ids=grid.cell_ids,
            x_km=np.asarray(
                [cell.representative_point.x / 1_000.0 for cell in grid.cells],
                dtype=np.float64,
            ),
            y_km=np.asarray(
                [cell.representative_point.y / 1_000.0 for cell in grid.cells],
                dtype=np.float64,
            ),
            area_km2=np.asarray(
                [cell.clipped_area_m2 / 1_000_000.0 for cell in grid.cells],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True, slots=True)
class PoissonLogLikelihood:
    event_count: int
    event_log_intensity_sum: float
    compensator: float
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class UniformPoissonModel:
    """Closed-form uniform spatial Poisson model inside the study area."""

    rate_per_day: float
    study_area_km2: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate_per_day) or self.rate_per_day < 0.0:
            raise ValueError("rate_per_day must be finite and non-negative")
        if not math.isfinite(self.study_area_km2) or self.study_area_km2 <= 0.0:
            raise ValueError("study_area_km2 must be finite and positive")

    @property
    def spatial_density_per_km2(self) -> float:
        return 1.0 / self.study_area_km2

    def density(self, x_km: object, y_km: object) -> FloatArray:
        x, _ = _coordinates(x_km, y_km, allow_empty=True)
        return np.full(x.shape, self.spatial_density_per_km2, dtype=np.float64)

    def score(self, *, event_count: int, duration_days: float) -> PoissonLogLikelihood:
        return score_spatial_poisson(
            rate_per_day=self.rate_per_day,
            event_densities=np.full(event_count, self.spatial_density_per_km2),
            duration_days=duration_days,
            spatial_mass=1.0,
        )


def fit_uniform_poisson(
    *,
    training_event_count: int,
    training_duration_days: float,
    study_area_km2: float,
) -> UniformPoissonModel:
    if (
        not isinstance(training_event_count, int)
        or isinstance(training_event_count, bool)
        or training_event_count < 0
    ):
        raise ValueError("training_event_count must be a non-negative integer")
    duration = float(training_duration_days)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("training_duration_days must be finite and positive")
    return UniformPoissonModel(
        rate_per_day=training_event_count / duration,
        study_area_km2=float(study_area_km2),
    )


@dataclass(frozen=True, slots=True)
class GaussianMixtureFamily:
    """Equal-weight isotropic Gaussian mixture evaluated without boundary adjustment."""

    training_x_km: FloatArray
    training_y_km: FloatArray
    chunk_size: int = 256
    _training_points_km: FloatArray = field(init=False, repr=False, compare=False)
    _training_squared_norms_km2: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        x = _owned_read_only_vector("training_x_km", self.training_x_km)
        y = _owned_read_only_vector("training_y_km", self.training_y_km)
        if x.shape != y.shape:
            raise ValueError("training_x_km and training_y_km must have the same shape")
        if (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
            or self.chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")
        training_points = np.empty((x.size, 2), dtype=np.float64)
        training_points[:, 0] = x
        training_points[:, 1] = y
        training_squared_norms = np.sum(training_points * training_points, axis=1)
        training_points.setflags(write=False)
        training_squared_norms.setflags(write=False)
        object.__setattr__(self, "training_x_km", x)
        object.__setattr__(self, "training_y_km", y)
        object.__setattr__(self, "_training_points_km", training_points)
        object.__setattr__(self, "_training_squared_norms_km2", training_squared_norms)

    @property
    def training_event_count(self) -> int:
        return int(self.training_x_km.size)

    def raw_densities(
        self,
        x_km: object,
        y_km: object,
        *,
        bandwidths_km: tuple[float, ...],
    ) -> dict[float, FloatArray]:
        """Evaluate several infinite-support mixtures while sharing distance matrices."""

        x, y = _coordinates(x_km, y_km, allow_empty=True)
        bandwidths = _validated_bandwidths(bandwidths_km)
        outputs = {bandwidth: np.empty(x.size, dtype=np.float64) for bandwidth in bandwidths}
        count = float(self.training_event_count)
        effective_chunk_size = min(
            self.chunk_size,
            max(1, _MAX_DISTANCE_BLOCK_ELEMENTS // self.training_event_count),
        )
        inverse_normalizers = {
            bandwidth: 1.0 / (count * 2.0 * math.pi * bandwidth * bandwidth)
            for bandwidth in bandwidths
        }
        exponent_scales = {bandwidth: -0.5 / (bandwidth * bandwidth) for bandwidth in bandwidths}
        for start in range(0, x.size, effective_chunk_size):
            stop = min(x.size, start + effective_chunk_size)
            query_points = np.empty((stop - start, 2), dtype=np.float64)
            query_points[:, 0] = x[start:stop]
            query_points[:, 1] = y[start:stop]
            squared_distance = _squared_distance_block(
                query_points,
                self._training_points_km,
                self._training_squared_norms_km2,
            )
            kernel_work = np.empty_like(squared_distance)
            for bandwidth in bandwidths:
                np.multiply(squared_distance, exponent_scales[bandwidth], out=kernel_work)
                np.exp(kernel_work, out=kernel_work)
                kernel_sum = np.sum(kernel_work, axis=1, dtype=np.float64)
                kernel_sum *= inverse_normalizers[bandwidth]
                outputs[bandwidth][start:stop] = kernel_sum
        if any(
            not np.isfinite(values).all() or np.any(values < 0.0) for values in outputs.values()
        ):
            raise ValueError("Gaussian mixture density evaluation is non-finite")
        return outputs

    def raw_log_densities(
        self,
        x_km: object,
        y_km: object,
        *,
        bandwidths_km: tuple[float, ...],
    ) -> dict[float, FloatArray]:
        """Evaluate infinite-support mixture logs with a shared stable log-sum-exp pass."""

        x, y = _coordinates(x_km, y_km, allow_empty=True)
        bandwidths = _validated_bandwidths(bandwidths_km)
        outputs = {bandwidth: np.empty(x.size, dtype=np.float64) for bandwidth in bandwidths}
        effective_chunk_size = min(
            self.chunk_size,
            max(1, _MAX_DISTANCE_BLOCK_ELEMENTS // self.training_event_count),
        )
        exponent_scales = {bandwidth: -0.5 / (bandwidth * bandwidth) for bandwidth in bandwidths}
        log_normalizers = {
            bandwidth: math.log(self.training_event_count)
            + math.log(2.0 * math.pi)
            + 2.0 * math.log(bandwidth)
            for bandwidth in bandwidths
        }
        for start in range(0, x.size, effective_chunk_size):
            stop = min(x.size, start + effective_chunk_size)
            query_points = np.empty((stop - start, 2), dtype=np.float64)
            query_points[:, 0] = x[start:stop]
            query_points[:, 1] = y[start:stop]
            squared_distance = _squared_distance_block(
                query_points,
                self._training_points_km,
                self._training_squared_norms_km2,
            )
            exponent_work = np.empty_like(squared_distance)
            for bandwidth in bandwidths:
                np.multiply(squared_distance, exponent_scales[bandwidth], out=exponent_work)
                maximum_exponent = cast(FloatArray, np.max(exponent_work, axis=1))
                exponent_work -= maximum_exponent[:, None]
                np.exp(exponent_work, out=exponent_work)
                log_kernel_sum = cast(
                    FloatArray,
                    np.sum(exponent_work, axis=1, dtype=np.float64),
                )
                np.log(log_kernel_sum, out=log_kernel_sum)
                log_kernel_sum += maximum_exponent
                log_kernel_sum -= log_normalizers[bandwidth]
                outputs[bandwidth][start:stop] = log_kernel_sum
        if any(not np.isfinite(values).all() for values in outputs.values()):
            raise ValueError("Gaussian mixture log-density evaluation is non-finite")
        return outputs


@dataclass(frozen=True, slots=True)
class SpatialPoissonModel:
    """One boundary-normalized KDE with an independently fitted daily rate."""

    mixture: GaussianMixtureFamily
    bandwidth_km: float
    normalization_mass: float
    rate_per_day: float
    normalization_quadrature: SpatialQuadrature | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    normalization_cell_masses: FloatArray | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.bandwidth_km) or self.bandwidth_km <= 0.0:
            raise ValueError("bandwidth_km must be finite and positive")
        if not math.isfinite(self.normalization_mass) or self.normalization_mass <= 0.0:
            raise ValueError("normalization_mass must be finite and positive")
        if not math.isfinite(self.rate_per_day) or self.rate_per_day < 0.0:
            raise ValueError("rate_per_day must be finite and non-negative")
        if (self.normalization_quadrature is None) != (self.normalization_cell_masses is None):
            raise ValueError(
                "normalization_quadrature and normalization_cell_masses must be provided together"
            )
        if self.normalization_quadrature is not None:
            cell_masses = _owned_read_only_vector(
                "normalization_cell_masses",
                self.normalization_cell_masses,
            )
            if cell_masses.shape != self.normalization_quadrature.area_km2.shape:
                raise ValueError("normalization cell masses must match the quadrature")
            if np.any(cell_masses < 0.0):
                raise ValueError("normalization cell masses must be non-negative")
            if not math.isclose(
                float(np.sum(cell_masses, dtype=np.float64)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("normalization cell masses must sum to one")
            object.__setattr__(self, "normalization_cell_masses", cell_masses)

    def density(self, x_km: object, y_km: object) -> FloatArray:
        raw = self.mixture.raw_densities(
            x_km,
            y_km,
            bandwidths_km=(self.bandwidth_km,),
        )[self.bandwidth_km]
        return raw / self.normalization_mass

    def density_scalar(self, x_km: float, y_km: float) -> float:
        return float(self.density(np.asarray([x_km]), np.asarray([y_km]))[0])

    def log_density(self, x_km: object, y_km: object) -> FloatArray:
        """Return stable boundary-normalized logs for event likelihood terms."""

        raw_log = self.mixture.raw_log_densities(
            x_km,
            y_km,
            bandwidths_km=(self.bandwidth_km,),
        )[self.bandwidth_km]
        raw_log -= math.log(self.normalization_mass)
        return raw_log

    def log_density_scalar(self, x_km: float, y_km: float) -> float:
        return float(self.log_density(np.asarray([x_km]), np.asarray([y_km]))[0])

    def cell_masses(self, quadrature: SpatialQuadrature) -> FloatArray:
        """Return cell masses, reusing the exact fitted quadrature evaluation when possible."""

        if (
            quadrature is self.normalization_quadrature
            and self.normalization_cell_masses is not None
        ):
            return self.normalization_cell_masses
        density = self.density(quadrature.x_km, quadrature.y_km)
        masses = np.asarray(density * quadrature.area_km2, dtype=np.float64)
        masses.setflags(write=False)
        return masses

    def grid_masses(self, quadrature: SpatialQuadrature) -> dict[str, float]:
        masses = self.cell_masses(quadrature)
        return {
            identifier: float(mass)
            for identifier, mass in zip(quadrature.cell_ids, masses, strict=True)
        }

    def score(
        self,
        x_km: object,
        y_km: object,
        *,
        duration_days: float,
        spatial_mass: float = 1.0,
    ) -> PoissonLogLikelihood:
        return score_log_densities(
            rate_per_day=self.rate_per_day,
            event_log_densities=self.log_density(x_km, y_km),
            duration_days=duration_days,
            spatial_mass=spatial_mass,
        )


def fit_spatial_poisson_family(
    training_x_km: object,
    training_y_km: object,
    *,
    training_duration_days: float,
    normalization_quadrature: SpatialQuadrature,
    bandwidths_km: tuple[float, ...] = FROZEN_BANDWIDTHS_KM,
    chunk_size: int = 256,
) -> dict[float, SpatialPoissonModel]:
    """Fit several candidates with one 12.5-km normalization evaluation."""

    training_x, training_y = _coordinates(training_x_km, training_y_km)
    duration = float(training_duration_days)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("training_duration_days must be finite and positive")
    mixture = GaussianMixtureFamily(training_x, training_y, chunk_size=chunk_size)
    bandwidths = _validated_bandwidths(bandwidths_km)
    raw = mixture.raw_densities(
        normalization_quadrature.x_km,
        normalization_quadrature.y_km,
        bandwidths_km=bandwidths,
    )
    models: dict[float, SpatialPoissonModel] = {}
    for value in bandwidths:
        cell_masses = raw.pop(value)
        np.multiply(cell_masses, normalization_quadrature.area_km2, out=cell_masses)
        mass = float(np.sum(cell_masses, dtype=np.float64))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("KDE normalization mass must be finite and positive")
        cell_masses /= mass
        models[value] = SpatialPoissonModel(
            mixture=mixture,
            bandwidth_km=value,
            normalization_mass=mass,
            rate_per_day=mixture.training_event_count / duration,
            normalization_quadrature=normalization_quadrature,
            normalization_cell_masses=cell_masses,
        )
    return models


def evaluate_spatial_poisson_family(
    models: Mapping[float, SpatialPoissonModel],
    x_km: object,
    y_km: object,
) -> dict[float, FloatArray]:
    """Evaluate normalized densities for a fitted family with one distance pass."""

    if not models:
        raise ValueError("models must not be empty")
    bandwidths = tuple(float(value) for value in models)
    if len(set(bandwidths)) != len(bandwidths):
        raise ValueError("model bandwidth keys must be unique")
    first_model = next(iter(models.values()))
    mixture = first_model.mixture
    normalized_models: dict[float, SpatialPoissonModel] = {}
    for key, model in models.items():
        bandwidth = float(key)
        if bandwidth != model.bandwidth_km:
            raise ValueError("every model key must match its bandwidth")
        if model.mixture is not mixture:
            raise ValueError("all models must share one fitted Gaussian mixture")
        normalized_models[bandwidth] = model
    raw = mixture.raw_densities(x_km, y_km, bandwidths_km=bandwidths)
    for bandwidth, values in raw.items():
        values /= normalized_models[bandwidth].normalization_mass
    return raw


def evaluate_spatial_poisson_family_log_densities(
    models: Mapping[float, SpatialPoissonModel],
    x_km: object,
    y_km: object,
) -> dict[float, FloatArray]:
    """Evaluate stable normalized log densities for a fitted family in one pass."""

    if not models:
        raise ValueError("models must not be empty")
    bandwidths = tuple(float(value) for value in models)
    if len(set(bandwidths)) != len(bandwidths):
        raise ValueError("model bandwidth keys must be unique")
    first_model = next(iter(models.values()))
    mixture = first_model.mixture
    normalized_models: dict[float, SpatialPoissonModel] = {}
    for key, model in models.items():
        bandwidth = float(key)
        if bandwidth != model.bandwidth_km:
            raise ValueError("every model key must match its bandwidth")
        if model.mixture is not mixture:
            raise ValueError("all models must share one fitted Gaussian mixture")
        normalized_models[bandwidth] = model
    raw_log = mixture.raw_log_densities(x_km, y_km, bandwidths_km=bandwidths)
    for bandwidth, values in raw_log.items():
        values -= math.log(normalized_models[bandwidth].normalization_mass)
    return raw_log


def evaluate_spatial_poisson_family_cell_masses(
    models: Mapping[float, SpatialPoissonModel],
    quadrature: SpatialQuadrature,
) -> dict[float, FloatArray]:
    """Evaluate all candidate cell masses once, reusing the fitted-grid cache."""

    if not models:
        raise ValueError("models must not be empty")
    cached: dict[float, FloatArray] = {}
    first_model = next(iter(models.values()))
    for key, model in models.items():
        bandwidth = float(key)
        if bandwidth != model.bandwidth_km:
            raise ValueError("every model key must match its bandwidth")
        if model.mixture is not first_model.mixture:
            raise ValueError("all models must share one fitted Gaussian mixture")
        if (
            model.normalization_quadrature is not quadrature
            or model.normalization_cell_masses is None
        ):
            cached.clear()
            break
        cached[bandwidth] = model.normalization_cell_masses
    if len(cached) == len(models):
        return cached

    densities = evaluate_spatial_poisson_family(
        models,
        quadrature.x_km,
        quadrature.y_km,
    )
    for values in densities.values():
        np.multiply(values, quadrature.area_km2, out=values)
        values.setflags(write=False)
    return densities


def score_spatial_poisson(
    *,
    rate_per_day: float,
    event_densities: object,
    duration_days: float,
    spatial_mass: float,
) -> PoissonLogLikelihood:
    """Score ordinary densities; event-model code should prefer stable log densities."""

    densities = _vector("event_densities", event_densities, allow_empty=True)
    if densities.size and np.any(densities <= 0.0):
        raise ValueError("positive event densities are required when targets exist")
    log_densities = np.log(densities) if densities.size else densities.copy()
    return score_log_densities(
        rate_per_day=rate_per_day,
        event_log_densities=log_densities,
        duration_days=duration_days,
        spatial_mass=spatial_mass,
    )


def score_log_densities(
    *,
    rate_per_day: float,
    event_log_densities: object,
    duration_days: float,
    spatial_mass: float,
) -> PoissonLogLikelihood:
    """Score finite log densities directly while retaining the Poisson compensator."""

    rate = float(rate_per_day)
    duration = float(duration_days)
    mass = float(spatial_mass)
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("rate_per_day must be finite and non-negative")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_days must be finite and positive")
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("spatial_mass must be finite and positive")
    log_densities = _vector(
        "event_log_densities",
        event_log_densities,
        allow_empty=True,
    )
    if log_densities.size and rate <= 0.0:
        raise ValueError("positive event intensities are required when targets exist")
    try:
        event_log_sum = (
            len(log_densities) * math.log(rate) + math.fsum(float(value) for value in log_densities)
            if log_densities.size
            else 0.0
        )
    except OverflowError as error:
        raise ValueError("Poisson event log-intensity sum must be finite") from error
    compensator = rate * duration * mass
    log_likelihood = event_log_sum - compensator
    if not all(math.isfinite(value) for value in (event_log_sum, compensator, log_likelihood)):
        raise ValueError("Poisson likelihood must be finite")
    return PoissonLogLikelihood(
        event_count=int(log_densities.size),
        event_log_intensity_sum=event_log_sum,
        compensator=compensator,
        log_likelihood=log_likelihood,
    )


@dataclass(frozen=True, slots=True)
class BandwidthCandidateAudit:
    bandwidth_km: float
    fold_scores: tuple[float, float, float, float]
    mean_score: float
    paired_standard_error: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class BandwidthPreScoreGateItem:
    """One frozen bandwidth's auditable normalization/convergence gate result."""

    bandwidth_km: float
    passed: bool
    numerical_evidence_id: str
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        bandwidth = float(self.bandwidth_km)
        if bandwidth not in FROZEN_BANDWIDTHS_KM:
            raise ValueError("gate item bandwidth must be a frozen candidate")
        if not isinstance(self.passed, bool):
            raise TypeError("gate item passed must be bool")
        if not isinstance(self.numerical_evidence_id, str) or not (
            evidence_id := self.numerical_evidence_id.strip()
        ):
            raise ValueError("gate item numerical_evidence_id must be non-empty")
        if self.passed:
            if self.failure_reason is not None:
                raise ValueError("a passed gate item must not contain a failure reason")
        else:
            if not isinstance(self.failure_reason, str) or not (
                failure_reason := self.failure_reason.strip()
            ):
                raise ValueError("a failed gate item must contain a non-empty failure reason")
            object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "bandwidth_km", bandwidth)
        object.__setattr__(self, "numerical_evidence_id", evidence_id)


@dataclass(frozen=True, slots=True)
class BandwidthPreScoreGateEvidence:
    """Complete five-candidate evidence established before any fold scoring."""

    candidates: tuple[BandwidthPreScoreGateItem, ...]

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(item, BandwidthPreScoreGateItem) for item in candidates):
            raise TypeError("pre-score gate candidates must be gate evidence items")
        if tuple(item.bandwidth_km for item in candidates) != FROZEN_BANDWIDTHS_KM:
            raise ValueError(
                "pre-score gate evidence must contain exactly five frozen candidates in fixed order"
            )
        evidence_ids = tuple(item.numerical_evidence_id for item in candidates)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("pre-score numerical evidence IDs must be unique")
        object.__setattr__(self, "candidates", candidates)

    @property
    def passed_bandwidths_km(self) -> tuple[float, ...]:
        return tuple(item.bandwidth_km for item in self.candidates if item.passed)

    @property
    def exclusions(self) -> tuple[BandwidthPreScoreGateItem, ...]:
        return tuple(item for item in self.candidates if not item.passed)


@dataclass(frozen=True, slots=True)
class BandwidthSelection:
    best_mean_bandwidth_km: float
    selected_bandwidth_km: float
    candidates: tuple[BandwidthCandidateAudit, ...]
    pre_score_gate_evidence: BandwidthPreScoreGateEvidence
    gate_exclusions: tuple[BandwidthPreScoreGateItem, ...]

    def __post_init__(self) -> None:
        if self.gate_exclusions != self.pre_score_gate_evidence.exclusions:
            raise ValueError("selection exclusions must exactly preserve failed gate evidence")

    @property
    def gate_excluded_bandwidths_km(self) -> tuple[float, ...]:
        return tuple(item.bandwidth_km for item in self.gate_exclusions)


def select_kde_bandwidth(
    fold_scores_by_bandwidth: Mapping[float, tuple[float, float, float, float]],
    *,
    pre_score_gate_evidence: BandwidthPreScoreGateEvidence,
) -> BandwidthSelection:
    """Apply one-SE only after complete five-candidate gate evidence is supplied."""

    normalized_fold_scores: dict[float, tuple[float, float, float, float]] = {}
    for raw_bandwidth, fold_scores in fold_scores_by_bandwidth.items():
        bandwidth = float(raw_bandwidth)
        if bandwidth in normalized_fold_scores:
            raise ValueError("bandwidth score keys must be unique")
        normalized_fold_scores[bandwidth] = fold_scores
    score_keys = set(normalized_fold_scores)
    unknown = score_keys.difference(FROZEN_BANDWIDTHS_KM)
    if unknown:
        raise ValueError("bandwidth scores contain an unknown frozen candidate")
    included_bandwidths = pre_score_gate_evidence.passed_bandwidths_km
    if not included_bandwidths:
        raise ValueError("at least one gate-passing frozen bandwidth is required")
    if score_keys != set(included_bandwidths):
        raise ValueError(
            "four-fold score keys must exactly match pre-score gate-passing bandwidths"
        )
    normalized: dict[float, FloatArray] = {}
    means: dict[float, float] = {}
    for bandwidth in included_bandwidths:
        score_array = _vector("fold scores", normalized_fold_scores[bandwidth])
        if score_array.shape != (4,):
            raise ValueError("every bandwidth must have exactly four fold scores")
        normalized[bandwidth] = score_array
        means[bandwidth] = float(np.mean(score_array))
    best_mean = max(means.values())
    best_bandwidth = max(
        bandwidth
        for bandwidth, mean_score in means.items()
        if math.isclose(mean_score, best_mean, rel_tol=0.0, abs_tol=1.0e-15)
    )
    best_scores = normalized[best_bandwidth]
    audits: list[BandwidthCandidateAudit] = []
    for bandwidth in included_bandwidths:
        differences = normalized[bandwidth] - best_scores
        paired_se = float(np.std(differences, ddof=1) / math.sqrt(4.0))
        eligible = means[bandwidth] >= best_mean - paired_se
        audits.append(
            BandwidthCandidateAudit(
                bandwidth_km=bandwidth,
                fold_scores=cast(tuple[float, float, float, float], tuple(normalized[bandwidth])),
                mean_score=means[bandwidth],
                paired_standard_error=paired_se,
                eligible=eligible,
            )
        )
    selected = max(item.bandwidth_km for item in audits if item.eligible)
    return BandwidthSelection(
        best_mean_bandwidth_km=best_bandwidth,
        selected_bandwidth_km=selected,
        candidates=tuple(audits),
        pre_score_gate_evidence=pre_score_gate_evidence,
        gate_exclusions=pre_score_gate_evidence.exclusions,
    )


__all__ = [
    "FROZEN_BANDWIDTHS_KM",
    "BandwidthCandidateAudit",
    "BandwidthPreScoreGateEvidence",
    "BandwidthPreScoreGateItem",
    "BandwidthSelection",
    "GaussianMixtureFamily",
    "PoissonLogLikelihood",
    "SpatialPoissonModel",
    "SpatialQuadrature",
    "UniformPoissonModel",
    "evaluate_spatial_poisson_family",
    "evaluate_spatial_poisson_family_cell_masses",
    "evaluate_spatial_poisson_family_log_densities",
    "fit_spatial_poisson_family",
    "fit_uniform_poisson",
    "score_log_densities",
    "score_spatial_poisson",
    "select_kde_bandwidth",
]
