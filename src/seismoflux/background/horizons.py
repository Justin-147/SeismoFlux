"""Frozen validation issue-horizon backtests for the stage-2 background models."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from seismoflux.background.adapters import point_area_quadrature_from_grid
from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.catalog import EarthquakeCatalog
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.etas_fit import (
    BackgroundDensity,
    ETASLikelihoodProblem,
    ETASModelSpec,
    ETASParameters,
    PointAreaQuadrature,
    etas_log_likelihood,
)
from seismoflux.background.grid import EqualAreaGridFamily
from seismoflux.background.issues import FrozenIssueCalendar, IssueExposure
from seismoflux.background.poisson import SpatialPoissonModel, UniformPoissonModel
from seismoflux.background.workflow import catalog_etas_events, physical_target_mask

ModelId = Literal["uniform_poisson", "spatial_poisson", "etas"]
CandidateModelId = Literal["spatial_poisson", "etas"]
EXPECTED_HORIZONS = (7, 30, 90, 180, 365)
EXPECTED_PUBLICATION_DELAYS = (0, 1, 7)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _readonly(values: object) -> NDArray[np.float64]:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class IssuePointProcessScore:
    """One immutable point-process score on a frozen issue exposure."""

    protocol_sha256: str
    model_id: ModelId
    model_variant_id: str
    parameter_snapshot_id: str
    publication_delay_days: int
    issue_date_local: str
    issue_time_utc: str
    horizon_days: int
    target_event_ids: tuple[str, ...]
    event_log_intensities: NDArray[np.float64]
    compensator: float

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.protocol_sha256) is None:
            raise ValueError("protocol_sha256 must be a lowercase SHA-256 string")
        if self.model_id not in {"uniform_poisson", "spatial_poisson", "etas"}:
            raise ValueError("unknown issue score model_id")
        if not self.model_variant_id:
            raise ValueError("model_variant_id must not be empty")
        if _SHA256_PATTERN.fullmatch(self.parameter_snapshot_id) is None:
            raise ValueError("parameter_snapshot_id must be a lowercase SHA-256 string")
        if self.publication_delay_days not in EXPECTED_PUBLICATION_DELAYS:
            raise ValueError("publication delay must be 0, 1, or 7 days")
        if self.horizon_days not in EXPECTED_HORIZONS:
            raise ValueError("issue horizon must be 7, 30, 90, 180, or 365 days")
        if not self.issue_date_local or not self.issue_time_utc:
            raise ValueError("issue identity must not be empty")
        identifiers = tuple(self.target_event_ids)
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("target event IDs must be non-empty strings and unique")
        values = _readonly(self.event_log_intensities)
        if values.shape != (len(identifiers),) or not np.all(np.isfinite(values)):
            raise ValueError("event log intensities must be finite and align with target IDs")
        if not math.isfinite(self.compensator) or self.compensator < 0.0:
            raise ValueError("point-process compensator must be finite and non-negative")
        object.__setattr__(self, "target_event_ids", identifiers)
        object.__setattr__(self, "event_log_intensities", values)

    @property
    def log_likelihood(self) -> float:
        return float(np.sum(self.event_log_intensities, dtype=np.float64)) - self.compensator

    @property
    def score_id(self) -> str:
        return _canonical_sha256(
            {
                "protocol_sha256": self.protocol_sha256,
                "model_id": self.model_id,
                "model_variant_id": self.model_variant_id,
                "parameter_snapshot_id": self.parameter_snapshot_id,
                "publication_delay_days": self.publication_delay_days,
                "issue_date_local": self.issue_date_local,
                "issue_time_utc": self.issue_time_utc,
                "horizon_days": self.horizon_days,
                "target_event_ids": self.target_event_ids,
                "event_log_intensities": tuple(
                    float(value) for value in self.event_log_intensities
                ),
                "compensator": self.compensator,
            }
        )


@dataclass(frozen=True, slots=True)
class PairedHorizonBacktest:
    """All non-overlapping exposures for one model, delay, and horizon."""

    candidate_model_id: CandidateModelId
    publication_delay_days: int
    horizon_days: int
    exposure_pairs: tuple[tuple[IssuePointProcessScore, IssuePointProcessScore], ...]

    def __post_init__(self) -> None:
        if self.candidate_model_id not in {"spatial_poisson", "etas"}:
            raise ValueError("horizon candidate must be spatial_poisson or etas")
        if self.publication_delay_days not in EXPECTED_PUBLICATION_DELAYS:
            raise ValueError("publication delay must be 0, 1, or 7 days")
        if self.horizon_days not in EXPECTED_HORIZONS:
            raise ValueError("unknown frozen horizon")
        if not self.exposure_pairs:
            raise ValueError("horizon evidence must retain at least one exposure")
        seen_events: set[str] = set()
        previous_issue_date = ""
        for candidate, uniform in self.exposure_pairs:
            if candidate.model_id != self.candidate_model_id:
                raise ValueError("candidate score has the wrong model family")
            if uniform.model_id != "uniform_poisson":
                raise ValueError("paired baseline must be uniform_poisson")
            exact_fields = (
                "protocol_sha256",
                "publication_delay_days",
                "issue_date_local",
                "issue_time_utc",
                "horizon_days",
                "target_event_ids",
            )
            if any(getattr(candidate, name) != getattr(uniform, name) for name in exact_fields):
                raise ValueError("candidate and uniform issue scores are not exactly paired")
            if candidate.publication_delay_days != self.publication_delay_days:
                raise ValueError("exposure delay differs from its horizon backtest")
            if candidate.horizon_days != self.horizon_days:
                raise ValueError("exposure horizon differs from its horizon backtest")
            if candidate.issue_date_local <= previous_issue_date:
                raise ValueError("exposures must be in strictly increasing issue-date order")
            previous_issue_date = candidate.issue_date_local
            overlap = seen_events.intersection(candidate.target_event_ids)
            if overlap:
                raise ValueError("a physical target appears in more than one exposure")
            seen_events.update(candidate.target_event_ids)

    @property
    def exposure_count(self) -> int:
        return len(self.exposure_pairs)

    @property
    def target_event_ids(self) -> tuple[str, ...]:
        return tuple(
            event_id
            for candidate, _ in self.exposure_pairs
            for event_id in candidate.target_event_ids
        )

    @property
    def event_log_intensity_differences(self) -> NDArray[np.float64]:
        return _readonly(
            np.concatenate(
                [
                    candidate.event_log_intensities - uniform.event_log_intensities
                    for candidate, uniform in self.exposure_pairs
                ]
            )
            if self.target_event_ids
            else np.asarray([], dtype=np.float64)
        )

    @property
    def compensator_difference(self) -> float:
        return math.fsum(
            candidate.compensator - uniform.compensator
            for candidate, uniform in self.exposure_pairs
        )

    @property
    def information_gain_per_event(self) -> float | None:
        count = len(self.target_event_ids)
        if count == 0:
            return None
        numerator = (
            float(np.sum(self.event_log_intensity_differences, dtype=np.float64))
            - self.compensator_difference
        )
        return numerator / count


@dataclass(frozen=True, slots=True)
class BackgroundHorizonBacktests:
    """Complete validation-only secondary diagnostics without horizon aggregation."""

    comparisons: tuple[PairedHorizonBacktest, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            (model_id, delay, horizon)
            for delay in EXPECTED_PUBLICATION_DELAYS
            for model_id in ("spatial_poisson", "etas")
            for horizon in EXPECTED_HORIZONS
        )
        observed = tuple(
            (item.candidate_model_id, item.publication_delay_days, item.horizon_days)
            for item in self.comparisons
        )
        if observed != expected:
            raise ValueError("horizon backtests must contain the exact model/delay/horizon grid")

    def at(
        self,
        model_id: CandidateModelId,
        publication_delay_days: int,
        horizon_days: int,
    ) -> PairedHorizonBacktest:
        for item in self.comparisons:
            if (
                item.candidate_model_id == model_id
                and item.publication_delay_days == publication_delay_days
                and item.horizon_days == horizon_days
            ):
                return item
        raise KeyError("no matching issue-horizon backtest")


@dataclass(frozen=True, slots=True)
class _SpatialBackgroundDensity:
    model: SpatialPoissonModel

    def __call__(self, x_km: float, y_km: float) -> float:
        return self.model.density_scalar(x_km, y_km)

    def density_many(self, x_km: object, y_km: object) -> NDArray[np.float64]:
        return self.model.density(x_km, y_km)


def _parent_mass_key(x_km: float, y_km: float, magnitude: float) -> tuple[str, str, str]:
    return (float(x_km).hex(), float(y_km).hex(), float(magnitude).hex())


class _CachedPointAreaQuadrature(PointAreaQuadrature):
    """One exact grid with parent masses and KDE mass evaluated only once."""

    __slots__ = ("_background_mass", "_background_model", "_mass_by_parent", "_spec")
    _background_mass: float
    _background_model: SpatialPoissonModel
    _mass_by_parent: dict[tuple[str, str, str], float]
    _spec: ETASModelSpec

    def __init__(
        self,
        base: PointAreaQuadrature,
        *,
        parent_x_km: npt.ArrayLike,
        parent_y_km: npt.ArrayLike,
        parent_magnitudes: npt.ArrayLike,
        spec: ETASModelSpec,
        background_model: SpatialPoissonModel,
    ) -> None:
        super().__init__(base.points)
        x_values = np.asarray(parent_x_km, dtype=np.float64)
        y_values = np.asarray(parent_y_km, dtype=np.float64)
        magnitudes = np.asarray(parent_magnitudes, dtype=np.float64)
        if x_values.ndim != 1 or not (x_values.shape == y_values.shape == magnitudes.shape):
            raise ValueError("cached parent arrays must be aligned vectors")
        masses = base.inverse_power_masses(
            x_values,
            y_values,
            magnitudes,
            spec=spec,
        )
        lookup: dict[tuple[str, str, str], float] = {}
        for x_value, y_value, magnitude, mass in zip(
            x_values,
            y_values,
            magnitudes,
            masses,
            strict=True,
        ):
            key = _parent_mass_key(float(x_value), float(y_value), float(magnitude))
            observed = lookup.get(key)
            if observed is not None and observed != float(mass):
                raise ValueError("duplicate cached parent geometry has inconsistent mass")
            lookup[key] = float(mass)
        background = _SpatialBackgroundDensity(background_model)
        object.__setattr__(self, "_mass_by_parent", lookup)
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_background_model", background_model)
        object.__setattr__(self, "_background_mass", base.integrate(background))

    def integrate(self, density: BackgroundDensity) -> float:
        if (
            isinstance(density, _SpatialBackgroundDensity)
            and density.model is self._background_model
        ):
            return self._background_mass
        return super().integrate(density)

    def inverse_power_masses(
        self,
        parent_x_km: npt.ArrayLike,
        parent_y_km: npt.ArrayLike,
        parent_magnitudes: npt.ArrayLike,
        *,
        spec: ETASModelSpec,
        batch_size: int = 256,
    ) -> NDArray[np.float64]:
        del batch_size
        if spec != self._spec:
            raise ValueError("cached parent masses cannot be reused with another ETAS spec")
        x_values = np.asarray(parent_x_km, dtype=np.float64)
        y_values = np.asarray(parent_y_km, dtype=np.float64)
        magnitudes = np.asarray(parent_magnitudes, dtype=np.float64)
        if x_values.ndim != 1 or not (x_values.shape == y_values.shape == magnitudes.shape):
            raise ValueError("cached parent arrays must be aligned vectors")
        try:
            values = np.asarray(
                [
                    self._mass_by_parent[
                        _parent_mass_key(float(x_value), float(y_value), float(magnitude))
                    ]
                    for x_value, y_value, magnitude in zip(
                        x_values,
                        y_values,
                        magnitudes,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
        except KeyError as error:
            raise ValueError(
                "ETAS exposure requested a parent absent from the frozen cache"
            ) from error
        values.setflags(write=False)
        return values


def _model_snapshot_ids(
    protocol_sha256: str,
    uniform: UniformPoissonModel,
    spatial: SpatialPoissonModel,
) -> tuple[str, str]:
    uniform_id = _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "model_id": "uniform_poisson",
            "rate_per_day": uniform.rate_per_day,
            "study_area_km2": uniform.study_area_km2,
        }
    )
    spatial_id = _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "model_id": "spatial_poisson",
            "rate_per_day": spatial.rate_per_day,
            "bandwidth_km": spatial.bandwidth_km,
            "normalization_mass": spatial.normalization_mass,
            "training_x_km": tuple(float(value) for value in spatial.mixture.training_x_km),
            "training_y_km": tuple(float(value) for value in spatial.mixture.training_y_km),
        }
    )
    return uniform_id, spatial_id


def _target_indices(
    catalog: EarthquakeCatalog,
    exposure: IssueExposure,
    *,
    selected_mc: float,
) -> NDArray[np.int64]:
    mask = physical_target_mask(
        catalog,
        minimum_magnitude=selected_mc,
        origin_after_day=exposure.issue_day,
        origin_through_day=exposure.end_day,
    )
    indices = [int(value) for value in np.flatnonzero(mask)]
    indices.sort(key=lambda index: (float(catalog.origin_day[index]), str(catalog.event_id[index])))
    return np.asarray(indices, dtype=np.int64)


def _poisson_scores(
    *,
    protocol_sha256: str,
    exposure: IssueExposure,
    delay_days: int,
    catalog: EarthquakeCatalog,
    target_indices: NDArray[np.int64],
    uniform: UniformPoissonModel,
    spatial: SpatialPoissonModel,
    uniform_snapshot_id: str,
    spatial_snapshot_id: str,
) -> tuple[IssuePointProcessScore, IssuePointProcessScore]:
    identifiers = tuple(str(catalog.event_id[index]) for index in target_indices)
    uniform_log = math.log(uniform.rate_per_day) - math.log(uniform.study_area_km2)
    uniform_score = IssuePointProcessScore(
        protocol_sha256=protocol_sha256,
        model_id="uniform_poisson",
        model_variant_id="uniform_poisson/spatial_uniform_v1",
        parameter_snapshot_id=uniform_snapshot_id,
        publication_delay_days=delay_days,
        issue_date_local=exposure.issue_date_local,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        target_event_ids=identifiers,
        event_log_intensities=np.full(len(identifiers), uniform_log, dtype=np.float64),
        compensator=uniform.rate_per_day * exposure.horizon_days,
    )
    spatial_score = IssuePointProcessScore(
        protocol_sha256=protocol_sha256,
        model_id="spatial_poisson",
        model_variant_id=f"spatial_poisson/gaussian_kde_bw{spatial.bandwidth_km:g}km",
        parameter_snapshot_id=spatial_snapshot_id,
        publication_delay_days=delay_days,
        issue_date_local=exposure.issue_date_local,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        target_event_ids=identifiers,
        event_log_intensities=(
            spatial.log_density(catalog.x_km[target_indices], catalog.y_km[target_indices])
            + math.log(spatial.rate_per_day)
        ),
        compensator=spatial.rate_per_day * exposure.horizon_days,
    )
    return uniform_score, spatial_score


def _etas_score(
    *,
    protocol_sha256: str,
    exposure: IssueExposure,
    delay_days: int,
    catalog: EarthquakeCatalog,
    target_indices: NDArray[np.int64],
    selected_mc: float,
    spatial: SpatialPoissonModel,
    parameters: ETASParameters,
    spec: ETASModelSpec,
    parameter_snapshot_id: str,
    model_variant_id: str,
    quadrature: PointAreaQuadrature,
) -> IssuePointProcessScore:
    parent_mask = np.asarray(
        catalog.inside_external_buffer
        & (catalog.magnitude >= selected_mc)
        & (catalog.origin_day > exposure.issue_day - spec.history_parent_cutoff_days)
        & (catalog.origin_day <= exposure.end_day),
        dtype=np.bool_,
    )
    parents = catalog_etas_events(
        catalog,
        parent_mask,
        time_origin_day=exposure.issue_day,
        publication_delay_days=float(delay_days),
    )
    target_ids = {str(catalog.event_id[index]) for index in target_indices}
    targets = tuple(event for event in parents if event.event_id in target_ids)
    background = _SpatialBackgroundDensity(spatial)
    result = etas_log_likelihood(
        ETASLikelihoodProblem(
            assessment_start_days=0.0,
            assessment_end_days=float(exposure.horizon_days),
            target_events=targets,
            parent_events=parents,
            background_density=background,
            spatial_integrator=quadrature,
        ),
        parameters,
        spec,
    )
    return IssuePointProcessScore(
        protocol_sha256=protocol_sha256,
        model_id="etas",
        model_variant_id=model_variant_id,
        parameter_snapshot_id=parameter_snapshot_id,
        publication_delay_days=delay_days,
        issue_date_local=exposure.issue_date_local,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        target_event_ids=result.target_event_ids,
        event_log_intensities=np.log(np.asarray(result.event_intensities, dtype=np.float64)),
        compensator=result.total_compensator,
    )


def run_issue_horizon_backtests(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    grid_family: EqualAreaGridFamily,
    *,
    selected_mc: float,
    uniform_model: UniformPoissonModel,
    spatial_model: SpatialPoissonModel,
    etas_parameters: ETASParameters,
    etas_spec: ETASModelSpec,
    etas_parameter_snapshot_id: str,
    etas_model_variant_id: str,
) -> BackgroundHorizonBacktests:
    """Run frozen validation issue diagnostics separately for every horizon and delay."""

    if tuple(config.time.horizons_days) != EXPECTED_HORIZONS:
        raise ValueError("configured horizons differ from the frozen backtest horizons")
    if tuple(config.time.publication_delay_sensitivity_days) != EXPECTED_PUBLICATION_DELAYS:
        raise ValueError("configured publication delays differ from the frozen sensitivity")
    if calendar.validation.partition_id != "validation":
        raise ValueError("issue-horizon backtests may score only the validation partition")
    if tuple(horizon for horizon, _ in calendar.validation.exposures_by_horizon) != (
        EXPECTED_HORIZONS
    ):
        raise ValueError("validation exposure calendar does not contain the frozen horizons")
    if selected_mc != etas_spec.mc:
        raise ValueError("ETAS specification and backtest completeness magnitude differ")
    if uniform_model.rate_per_day != spatial_model.rate_per_day:
        raise ValueError("uniform and spatial Poisson rates must be exactly paired")
    if _SHA256_PATTERN.fullmatch(etas_parameter_snapshot_id) is None:
        raise ValueError("ETAS parameter snapshot ID must be a lowercase SHA-256 string")
    if not etas_model_variant_id:
        raise ValueError("ETAS model variant ID must not be empty")
    etas_spec.validate_parameters(etas_parameters)

    protocol_sha256 = _canonical_sha256(config.model_dump(mode="python"))
    uniform_snapshot_id, spatial_snapshot_id = _model_snapshot_ids(
        protocol_sha256,
        uniform_model,
        spatial_model,
    )
    validation_exposures = tuple(
        exposure
        for horizon_days in EXPECTED_HORIZONS
        for exposure in calendar.validation.exposures(horizon_days)
    )
    minimum_issue_day = min(exposure.issue_day for exposure in validation_exposures)
    maximum_end_day = max(exposure.end_day for exposure in validation_exposures)
    cache_mask = np.asarray(
        catalog.inside_external_buffer
        & (catalog.magnitude >= selected_mc)
        & (catalog.origin_day > minimum_issue_day - etas_spec.history_parent_cutoff_days)
        & (catalog.origin_day <= maximum_end_day),
        dtype=np.bool_,
    )
    cache_indices = np.flatnonzero(cache_mask)
    base_quadrature = point_area_quadrature_from_grid(grid_family.at(12.5))
    quadrature = _CachedPointAreaQuadrature(
        base_quadrature,
        parent_x_km=catalog.x_km[cache_indices],
        parent_y_km=catalog.y_km[cache_indices],
        parent_magnitudes=catalog.magnitude[cache_indices],
        spec=etas_spec,
        background_model=spatial_model,
    )
    comparisons: list[PairedHorizonBacktest] = []
    for delay_days in EXPECTED_PUBLICATION_DELAYS:
        spatial_by_horizon: dict[
            int, tuple[tuple[IssuePointProcessScore, IssuePointProcessScore], ...]
        ] = {}
        etas_by_horizon: dict[
            int, tuple[tuple[IssuePointProcessScore, IssuePointProcessScore], ...]
        ] = {}
        for horizon_days in EXPECTED_HORIZONS:
            spatial_pairs: list[tuple[IssuePointProcessScore, IssuePointProcessScore]] = []
            etas_pairs: list[tuple[IssuePointProcessScore, IssuePointProcessScore]] = []
            for exposure in calendar.validation.exposures(horizon_days):
                indices = _target_indices(catalog, exposure, selected_mc=selected_mc)
                uniform_score, spatial_score = _poisson_scores(
                    protocol_sha256=protocol_sha256,
                    exposure=exposure,
                    delay_days=delay_days,
                    catalog=catalog,
                    target_indices=indices,
                    uniform=uniform_model,
                    spatial=spatial_model,
                    uniform_snapshot_id=uniform_snapshot_id,
                    spatial_snapshot_id=spatial_snapshot_id,
                )
                etas_score = _etas_score(
                    protocol_sha256=protocol_sha256,
                    exposure=exposure,
                    delay_days=delay_days,
                    catalog=catalog,
                    target_indices=indices,
                    selected_mc=selected_mc,
                    spatial=spatial_model,
                    parameters=etas_parameters,
                    spec=etas_spec,
                    parameter_snapshot_id=etas_parameter_snapshot_id,
                    model_variant_id=etas_model_variant_id,
                    quadrature=quadrature,
                )
                spatial_pairs.append((spatial_score, uniform_score))
                etas_pairs.append((etas_score, uniform_score))
            spatial_by_horizon[horizon_days] = tuple(spatial_pairs)
            etas_by_horizon[horizon_days] = tuple(etas_pairs)
        model_groups: tuple[
            tuple[
                CandidateModelId,
                dict[
                    int,
                    tuple[tuple[IssuePointProcessScore, IssuePointProcessScore], ...],
                ],
            ],
            ...,
        ] = (
            ("spatial_poisson", spatial_by_horizon),
            ("etas", etas_by_horizon),
        )
        for model_id, values in model_groups:
            for horizon_days in EXPECTED_HORIZONS:
                comparisons.append(
                    PairedHorizonBacktest(
                        candidate_model_id=model_id,
                        publication_delay_days=delay_days,
                        horizon_days=horizon_days,
                        exposure_pairs=values[horizon_days],
                    )
                )
    return BackgroundHorizonBacktests(comparisons=tuple(comparisons))


__all__ = [
    "BackgroundHorizonBacktests",
    "IssuePointProcessScore",
    "PairedHorizonBacktest",
    "run_issue_horizon_backtests",
]
