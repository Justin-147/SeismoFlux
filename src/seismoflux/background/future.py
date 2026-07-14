"""Frozen validation-issue future ensembles for the stage-2 ETAS baseline.

The public constructors deliberately accept no anomaly, fault, target-event, or
post-issue input.  Background immigrant locations are sampled from the fitted
spatial-Poisson KDE conditional on the target-independent study polygon.  One
365-day catalog is generated for every replicate and then reused for every frozen
horizon and every member of the fixed three-grid family.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Literal, TypeAlias, cast

import numpy as np
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely import from_wkb, prepare, to_wkb
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from seismoflux.background.catalog import StudyArea
from seismoflux.background.etas_fit import ETASEvent, ETASModelSpec, ETASParameters
from seismoflux.background.etas_simulation import (
    DomainClass,
    SimulatedEvent,
    SimulationDomain,
    SimulationResult,
    simulate_future_catalog,
)
from seismoflux.background.execution import detect_physical_core_count
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    GRID_CELL_SIZES_KM,
    EqualAreaGrid,
    EqualAreaGridFamily,
    GridCell,
    cell_id,
    point_cell_index,
)
from seismoflux.background.issues import FrozenIssueCalendar
from seismoflux.background.poisson import SpatialPoissonModel
from seismoflux.background.randomness import SeedContext

FUTURE_HORIZONS_DAYS = (7, 30, 90, 180, 365)
FUTURE_REPLICATE_COUNT = 128
FUTURE_REPLICATE_INDICES = tuple(range(FUTURE_REPLICATE_COUNT))
FUTURE_MAXIMUM_EVENTS = 100_000
FUTURE_ROOT_SEED = 147
FUTURE_PROTOCOL_VERSION = "0.2.0"
FUTURE_SEED_NAMESPACE = "future_simulation"
FUTURE_MODEL_ID = "etas/final_validation"
FUTURE_QUANTILE_PROBABILITIES = (0.025, 0.5, 0.975)
FUTURE_QUANTILE_METHOD: Literal["linear"] = "linear"
FUTURE_PROPAGATION_BUFFER_KM = 300.0
DEFAULT_REJECTION_ATTEMPT_LIMIT = 1_000_000
DEFAULT_MAX_WORKERS = 12
MINIMUM_RESERVED_PHYSICAL_CORES = 2

PhysicalCoreProbe: TypeAlias = Callable[[], int | None]
ProgressCallback: TypeAlias = Callable[[str], None]


class BackgroundRejectionSamplingError(RuntimeError):
    """The conditional KDE sampler could not reach the study polygon."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(cast(float, value))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: object) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_local_date(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a canonical local date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must use YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must use canonical YYYY-MM-DD")
    return value


def _validate_polygonal_geometry(geometry: BaseGeometry, *, label: str) -> None:
    if not isinstance(geometry, BaseGeometry):
        raise TypeError(f"{label} must be a Shapely geometry")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"{label} must be polygonal")
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{label} must be non-empty and valid")
    area = float(geometry.area)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(f"{label} must have finite positive area")
    if not all(math.isfinite(float(value)) for value in geometry.bounds):
        raise ValueError(f"{label} bounds must be finite")


def _validate_study_area(study_area: StudyArea) -> None:
    if not isinstance(study_area, StudyArea):
        raise TypeError("study_area must be StudyArea")
    _validate_polygonal_geometry(study_area.geographic, label="geographic study area")
    _validate_polygonal_geometry(study_area.projected, label="projected study area")
    reported_area = _finite("study_area.area_km2", study_area.area_km2)
    computed_area = float(study_area.projected.area) / 1_000_000.0
    if not math.isclose(reported_area, computed_area, rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError("study-area area must equal its projected geometry area")
    try:
        actual_crs = CRS.from_user_input(study_area.equal_area_crs)
        frozen_crs = CRS.from_user_input(EQUAL_AREA_CRS)
    except CRSError as error:
        raise ValueError("study-area equal-area CRS is invalid") from error
    if not actual_crs.equals(frozen_crs):
        raise ValueError("study area must use the frozen China Albers equal-area CRS")


@dataclass(frozen=True, slots=True)
class _StudyBufferClassifier:
    study_geometry: BaseGeometry
    propagation_geometry: BaseGeometry

    def __call__(self, x_km: float, y_km: float) -> DomainClass:
        point = Point(x_km * 1_000.0, y_km * 1_000.0)
        if self.study_geometry.covers(point):
            return "study"
        if self.propagation_geometry.covers(point):
            return "buffer"
        return "outside"


@dataclass(frozen=True, slots=True)
class _ConditionalKDESampler:
    model: SpatialPoissonModel
    classifier: _StudyBufferClassifier
    maximum_attempts: int

    def __call__(self, generator: np.random.Generator) -> tuple[float, float]:
        if not isinstance(generator, np.random.Generator):
            raise TypeError("conditional KDE sampling requires numpy.random.Generator")
        mixture = self.model.mixture
        for _ in range(self.maximum_attempts):
            parent_index = int(generator.integers(0, mixture.training_event_count))
            x_km = float(
                generator.normal(
                    loc=float(mixture.training_x_km[parent_index]),
                    scale=self.model.bandwidth_km,
                )
            )
            y_km = float(
                generator.normal(
                    loc=float(mixture.training_y_km[parent_index]),
                    scale=self.model.bandwidth_km,
                )
            )
            if not math.isfinite(x_km) or not math.isfinite(y_km):
                raise BackgroundRejectionSamplingError(
                    "conditional KDE sampling produced a non-finite location"
                )
            if self.classifier(x_km, y_km) == "study":
                return x_km, y_km
        raise BackgroundRejectionSamplingError(
            "conditional KDE sampling exhausted the explicit rejection-attempt limit"
        )


def build_kde_study_area_simulation_domain(
    model: SpatialPoissonModel,
    study_area: StudyArea,
    *,
    maximum_rejection_attempts: int = DEFAULT_REJECTION_ATTEMPT_LIMIT,
) -> SimulationDomain:
    """Build the frozen study/buffer domain and conditional KDE immigrant sampler.

    The fitted spatial-Poisson model and the target-independent study geometry are the
    only inputs.  A Gaussian-mixture component is selected uniformly, an isotropic KDE
    draw is made with the fitted bandwidth, and draws outside the study polygon are
    rejected.  This is exact sampling from the fitted infinite-support KDE conditional
    on the study area; the 300-km propagation buffer is never an immigrant domain.
    """

    if not isinstance(model, SpatialPoissonModel):
        raise TypeError("model must be SpatialPoissonModel")
    _validate_study_area(study_area)
    attempt_limit = _positive_integer(
        "maximum_rejection_attempts",
        maximum_rejection_attempts,
    )
    study_geometry = cast(BaseGeometry, from_wkb(to_wkb(study_area.projected)))
    propagation_geometry = cast(
        BaseGeometry,
        study_geometry.buffer(FUTURE_PROPAGATION_BUFFER_KM * 1_000.0),
    )
    _validate_polygonal_geometry(propagation_geometry, label="ETAS propagation domain")
    prepare(study_geometry)
    prepare(propagation_geometry)
    classifier = _StudyBufferClassifier(
        study_geometry=study_geometry,
        propagation_geometry=propagation_geometry,
    )
    sampler = _ConditionalKDESampler(
        model=model,
        classifier=classifier,
        maximum_attempts=attempt_limit,
    )
    return SimulationDomain(
        classify=classifier,
        sample_background_location=sampler,
    )


@dataclass(frozen=True, slots=True)
class FutureCountQuantile:
    """One fixed linear quantile of the 128 replicate event counts."""

    probability: float
    count: float

    def __post_init__(self) -> None:
        probability = _finite("quantile probability", self.probability)
        if probability not in FUTURE_QUANTILE_PROBABILITIES:
            raise ValueError("future count quantiles must use 0.025, 0.5, or 0.975")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "count", _nonnegative("quantile count", self.count))


@dataclass(frozen=True, slots=True)
class SparseExpectedCellCount:
    """One positive ensemble-mean count in a fixed equal-area grid cell."""

    cell_id: str
    expected_count: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cell_id, str)
            or not self.cell_id
            or self.cell_id != self.cell_id.strip()
        ):
            raise ValueError("sparse expected-count cell_id must be a non-empty trimmed string")
        expected = _nonnegative("sparse expected cell count", self.expected_count)
        if expected == 0.0:
            raise ValueError("sparse expected cell counts must omit zero-valued cells")
        object.__setattr__(self, "expected_count", expected)


@dataclass(frozen=True, slots=True)
class SparseGridExpectedCounts:
    """Stable cell-ID-sorted sparse expected counts for one horizon and grid."""

    cell_size_km: float
    cells: tuple[SparseExpectedCellCount, ...]
    expected_total: float

    def __post_init__(self) -> None:
        cell_size = _finite("cell_size_km", self.cell_size_km)
        if cell_size not in GRID_CELL_SIZES_KM:
            raise ValueError("future grid cell size must be 50, 25, or 12.5 km")
        if not isinstance(self.cells, tuple) or any(
            not isinstance(item, SparseExpectedCellCount) for item in self.cells
        ):
            raise TypeError("sparse expected cells must be an immutable cell tuple")
        identifiers = tuple(item.cell_id for item in self.cells)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("sparse expected cells must have unique cell-ID sorting")
        expected_total = _nonnegative("grid expected total", self.expected_total)
        calculated_total = math.fsum(item.expected_count for item in self.cells)
        if not math.isclose(
            expected_total,
            calculated_total,
            rel_tol=5.0e-15,
            abs_tol=1.0e-15,
        ):
            raise ValueError("grid expected total must equal its sparse cell sum")
        object.__setattr__(self, "cell_size_km", cell_size)
        object.__setattr__(self, "expected_total", expected_total)

    def expected_count(self, requested_cell_id: str) -> float:
        """Return a stored expected count, or zero for an omitted sparse cell."""

        for item in self.cells:
            if item.cell_id == requested_cell_id:
                return item.expected_count
        return 0.0


@dataclass(frozen=True, slots=True)
class FutureHorizonSummary:
    """Replicate counts, linear quantiles, and three-grid means for one horizon."""

    horizon_days: int
    replicate_counts: tuple[int, ...]
    mean_count: float
    quantiles: tuple[FutureCountQuantile, ...]
    grids: tuple[SparseGridExpectedCounts, ...]

    def __post_init__(self) -> None:
        if self.horizon_days not in FUTURE_HORIZONS_DAYS:
            raise ValueError("future summary uses an unknown frozen horizon")
        if not isinstance(self.replicate_counts, tuple) or len(self.replicate_counts) != (
            FUTURE_REPLICATE_COUNT
        ):
            raise ValueError("future summary must retain exactly 128 ordered replicate counts")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.replicate_counts
        ):
            raise ValueError("future replicate counts must be non-negative integers")
        mean_count = _nonnegative("future mean count", self.mean_count)
        expected_mean = math.fsum(self.replicate_counts) / FUTURE_REPLICATE_COUNT
        if not math.isclose(mean_count, expected_mean, rel_tol=5.0e-15, abs_tol=1.0e-15):
            raise ValueError("future mean count must equal the mean of all 128 replicates")
        if (
            not isinstance(self.quantiles, tuple)
            or tuple(item.probability for item in self.quantiles) != FUTURE_QUANTILE_PROBABILITIES
        ):
            raise ValueError("future count quantiles must remain ordered 0.025, 0.5, 0.975")
        count_array = np.asarray(self.replicate_counts, dtype=np.float64)
        expected_quantiles = np.quantile(
            count_array,
            FUTURE_QUANTILE_PROBABILITIES,
            method=FUTURE_QUANTILE_METHOD,
        )
        for item, expected in zip(self.quantiles, expected_quantiles, strict=True):
            if not math.isclose(item.count, float(expected), rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError("future count quantile disagrees with linear NumPy quantile")
        if (
            not isinstance(self.grids, tuple)
            or tuple(grid.cell_size_km for grid in self.grids) != GRID_CELL_SIZES_KM
        ):
            raise ValueError("future horizon grids must remain ordered 50, 25, 12.5 km")
        if any(
            not math.isclose(
                grid.expected_total,
                mean_count,
                rel_tol=5.0e-15,
                abs_tol=1.0e-15,
            )
            for grid in self.grids
        ):
            raise ValueError("every grid expected total must equal its horizon mean count")
        object.__setattr__(self, "mean_count", mean_count)

    @property
    def replicate_indices(self) -> tuple[int, ...]:
        """Indices corresponding positionally to ``replicate_counts``."""

        return FUTURE_REPLICATE_INDICES

    def grid(self, cell_size_km: float) -> SparseGridExpectedCounts:
        requested = float(cell_size_km)
        for grid_counts in self.grids:
            if grid_counts.cell_size_km == requested:
                return grid_counts
        raise KeyError(f"future horizon has no {requested:g} km grid")


@dataclass(frozen=True, slots=True)
class FutureIssueEnsemble:
    """Complete deterministic ensemble evidence for one frozen validation issue."""

    issue_date_local: str
    issue_id: str
    replicate_count: int
    quantile_probabilities: tuple[float, ...]
    quantile_method: Literal["linear"]
    horizons: tuple[FutureHorizonSummary, ...]

    def __post_init__(self) -> None:
        issue_date = _canonical_local_date(self.issue_date_local, label="issue_date_local")
        if self.issue_id != f"validation/{issue_date}":
            raise ValueError("future issue_id must use validation/YYYY-MM-DD")
        if self.replicate_count != FUTURE_REPLICATE_COUNT:
            raise ValueError("future issue ensemble must contain exactly 128 replicates")
        if self.quantile_probabilities != FUTURE_QUANTILE_PROBABILITIES:
            raise ValueError("future issue quantile probabilities must remain frozen")
        if self.quantile_method != FUTURE_QUANTILE_METHOD:
            raise ValueError("future issue quantile method must remain linear")
        if (
            not isinstance(self.horizons, tuple)
            or tuple(summary.horizon_days for summary in self.horizons) != FUTURE_HORIZONS_DAYS
        ):
            raise ValueError("future issue horizons must remain ordered 7, 30, 90, 180, 365")

    def horizon(self, horizon_days: int) -> FutureHorizonSummary:
        for summary in self.horizons:
            if summary.horizon_days == horizon_days:
                return summary
        raise KeyError(f"future issue has no {horizon_days}-day horizon")


@dataclass(frozen=True, slots=True)
class FutureWorkerResources:
    """Auditable outer-worker allocation with at least two configured reserve cores."""

    detected_physical_cores: int | None
    reserve_physical_cores: int
    configured_max_workers: int
    effective_workers: int

    def __post_init__(self) -> None:
        detected = self.detected_physical_cores
        if detected is not None:
            _positive_integer("detected_physical_cores", detected)
        reserve = _positive_integer("reserve_physical_cores", self.reserve_physical_cores)
        if reserve < MINIMUM_RESERVED_PHYSICAL_CORES:
            raise ValueError("future simulations must reserve at least two physical cores")
        configured = _positive_integer("configured_max_workers", self.configured_max_workers)
        effective = _positive_integer("effective_workers", self.effective_workers)
        expected = 1 if detected is None else min(configured, max(1, detected - reserve))
        if effective != expected:
            raise ValueError("effective future workers disagree with the reserved-core rule")


@dataclass(frozen=True, slots=True)
class ValidationFutureEnsembles:
    """All validation issue ensembles in frozen calendar order."""

    issues: tuple[FutureIssueEnsemble, ...]
    resources: FutureWorkerResources

    def __post_init__(self) -> None:
        if not isinstance(self.issues, tuple) or not self.issues:
            raise ValueError("validation future ensembles must contain at least one issue")
        issue_dates = tuple(issue.issue_date_local for issue in self.issues)
        if issue_dates != tuple(sorted(set(issue_dates))):
            raise ValueError("validation future ensembles must use unique ascending issue dates")
        if not isinstance(self.resources, FutureWorkerResources):
            raise TypeError("resources must be FutureWorkerResources")


def _validated_validation_dates(calendar: FrozenIssueCalendar) -> tuple[str, ...]:
    if not isinstance(calendar, FrozenIssueCalendar):
        raise TypeError("calendar must be FrozenIssueCalendar")
    validation = calendar.validation
    if validation.partition_id != "validation":
        raise ValueError("calendar validation partition must be named validation")
    issue_dates = tuple(
        _canonical_local_date(value, label="validation issue date")
        for value in validation.actual_issue_dates_local
    )
    if not issue_dates or issue_dates != tuple(sorted(set(issue_dates))):
        raise ValueError("validation issue dates must be non-empty, unique, and sorted")
    if len(validation.actual_issue_days) != len(issue_dates):
        raise ValueError("validation issue dates and issue days must have one common length")
    return issue_dates


def _validated_issue_date(calendar: FrozenIssueCalendar, issue_date_local: object) -> str:
    issue_date = _canonical_local_date(issue_date_local, label="issue_date_local")
    if issue_date not in _validated_validation_dates(calendar):
        raise ValueError("future simulation issue must belong to FrozenIssueCalendar.validation")
    return issue_date


def _validate_grid_family(grid_family: EqualAreaGridFamily, study_area: StudyArea) -> None:
    if not isinstance(grid_family, EqualAreaGridFamily):
        raise TypeError("grid_family must be EqualAreaGridFamily")
    if tuple(grid.spec.cell_size_km for grid in grid_family.grids) != GRID_CELL_SIZES_KM:
        raise ValueError("future simulation requires exactly the 50, 25, and 12.5 km grids")
    if not grid_family.study_area_equal_area.equals(study_area.projected):
        raise ValueError("future grids must use the supplied target-independent study area")
    expected_area = float(study_area.projected.area)
    for grid in grid_family.grids:
        grid_area = math.fsum(cell.clipped_area_m2 for cell in grid.cells)
        if not math.isclose(grid_area, expected_area, rel_tol=1.0e-12, abs_tol=1.0e-3):
            raise ValueError("every future grid must completely partition the study area")


def _validated_history(
    history_events: Sequence[ETASEvent],
    *,
    spec: ETASModelSpec,
    domain: SimulationDomain,
) -> tuple[ETASEvent, ...]:
    seen_ids: set[str] = set()
    validated: list[ETASEvent] = []
    for event in history_events:
        if not isinstance(event, ETASEvent):
            raise TypeError("history_events must contain ETASEvent values")
        if event.event_id in seen_ids:
            raise ValueError("history event IDs must be unique")
        seen_ids.add(event.event_id)
        if event.time_days > 0.0:
            raise ValueError("history event times must be relative to and at or before the issue")
        if event.available_time_days > 0.0:
            raise ValueError("history events must be available at or before the issue")
        spec.validate_event(event)
        if not event.inside_parent_domain:
            raise ValueError("history events must lie inside the ETAS propagation domain")
        expected_membership: DomainClass = "study" if event.inside_study_area else "buffer"
        if domain.membership(event.x_km, event.y_km) != expected_membership:
            raise ValueError("history event domain flags disagree with the frozen geometry")
        validated.append(event)
    return tuple(sorted(validated, key=lambda event: (event.time_days, event.event_id)))


def _simulation_event_key(event: SimulatedEvent) -> tuple[float, int, str, int]:
    return (
        event.time_days,
        event.generation,
        event.parent_id or "",
        event.within_parent_child_index,
    )


def _validate_simulation_result(
    result: SimulationResult,
    *,
    spec: ETASModelSpec,
    domain: SimulationDomain,
) -> tuple[SimulatedEvent, ...]:
    if not isinstance(result, SimulationResult):
        raise TypeError("future simulator must return SimulationResult")
    if result.horizon_days != float(max(FUTURE_HORIZONS_DAYS)):
        raise ValueError("future simulator must return the one 365-day catalog")
    if len(result.events) > FUTURE_MAXIMUM_EVENTS:
        raise ValueError("future simulator returned more events than the hard cap")
    if any(not isinstance(event, SimulatedEvent) for event in result.events):
        raise TypeError("future simulator events must be SimulatedEvent values")
    keys = tuple(_simulation_event_key(event) for event in result.events)
    if keys != tuple(sorted(keys)):
        raise ValueError("future simulator events must retain the frozen stable order")
    seen_ids: set[str] = set()
    for event in result.events:
        if not event.event_id or event.event_id in seen_ids:
            raise ValueError("future simulator event IDs must be non-empty and unique")
        seen_ids.add(event.event_id)
        if not 0.0 < _finite("simulated event time", event.time_days) <= 365.0:
            raise ValueError("simulated event times must lie in (0, 365]")
        _finite("simulated event x_km", event.x_km)
        _finite("simulated event y_km", event.y_km)
        magnitude = _finite("simulated event magnitude", event.magnitude)
        if not spec.mc <= magnitude <= spec.maximum_magnitude:
            raise ValueError("simulated event magnitude lies outside the frozen support")
        if domain.membership(event.x_km, event.y_km) != "study":
            raise ValueError("future outputs and aggregates must contain study events only")
    return result.events


def _grid_cell_lookup(grid: EqualAreaGrid) -> dict[str, GridCell]:
    return {cell.id: cell for cell in grid.cells}


def _event_cell_id(
    event: SimulatedEvent,
    grid: EqualAreaGrid,
    cells_by_id: Mapping[str, GridCell],
) -> str:
    row, column = point_cell_index(event.x_km * 1_000.0, event.y_km * 1_000.0, grid.spec)
    identifier = cell_id(grid.spec, row=row, column=column)
    raw_cell = cells_by_id.get(identifier)
    if raw_cell is None:
        raise ValueError("simulated study event has no cell in a frozen grid")
    if not raw_cell.clipped_geometry.covers(Point(event.x_km * 1_000.0, event.y_km * 1_000.0)):
        raise ValueError("simulated study event is outside its clipped grid cell")
    return identifier


def _future_seed_context(issue_id: str, replicate_index: int) -> SeedContext:
    return SeedContext(
        root_seed=FUTURE_ROOT_SEED,
        protocol_version=FUTURE_PROTOCOL_VERSION,
        namespace=FUTURE_SEED_NAMESPACE,
        model_id=FUTURE_MODEL_ID,
        issue_id=issue_id,
        replicate_index=replicate_index,
    )


def _simulate_validation_issue_ensemble(
    parameters: ETASParameters,
    spec: ETASModelSpec,
    history_events: Sequence[ETASEvent],
    domain: SimulationDomain,
    grid_family: EqualAreaGridFamily,
    calendar: FrozenIssueCalendar,
    *,
    issue_date_local: str,
) -> FutureIssueEnsemble:
    if not isinstance(parameters, ETASParameters):
        raise TypeError("parameters must be ETASParameters")
    if not isinstance(spec, ETASModelSpec):
        raise TypeError("spec must be ETASModelSpec")
    spec.validate_parameters(parameters)
    issue_date = _validated_issue_date(calendar, issue_date_local)
    issue_id = f"validation/{issue_date}"
    history = _validated_history(history_events, spec=spec, domain=domain)
    grids = grid_family.grids
    cell_lookups = tuple(_grid_cell_lookup(grid) for grid in grids)

    replicate_counts = [[0 for _ in FUTURE_REPLICATE_INDICES] for _ in FUTURE_HORIZONS_DAYS]
    aggregate_cell_counts: list[list[dict[str, int]]] = [
        [dict() for _ in GRID_CELL_SIZES_KM] for _ in FUTURE_HORIZONS_DAYS
    ]
    for replicate_index in FUTURE_REPLICATE_INDICES:
        result = simulate_future_catalog(
            parameters,
            spec,
            history,
            domain,
            horizon_days=365.0,
            seed_context=_future_seed_context(issue_id, replicate_index),
            maximum_events=FUTURE_MAXIMUM_EVENTS,
        )
        events = _validate_simulation_result(result, spec=spec, domain=domain)
        for event in events:
            first_horizon_index = bisect.bisect_left(FUTURE_HORIZONS_DAYS, event.time_days)
            if first_horizon_index >= len(FUTURE_HORIZONS_DAYS):
                raise ValueError("simulated event lies beyond every frozen horizon")
            event_cell_ids = tuple(
                _event_cell_id(event, grid, lookup)
                for grid, lookup in zip(grids, cell_lookups, strict=True)
            )
            for horizon_index in range(first_horizon_index, len(FUTURE_HORIZONS_DAYS)):
                replicate_counts[horizon_index][replicate_index] += 1
                for grid_index, identifier in enumerate(event_cell_ids):
                    cell_counts = aggregate_cell_counts[horizon_index][grid_index]
                    cell_counts[identifier] = cell_counts.get(identifier, 0) + 1

    horizon_summaries: list[FutureHorizonSummary] = []
    for horizon_index, horizon_days in enumerate(FUTURE_HORIZONS_DAYS):
        counts = tuple(replicate_counts[horizon_index])
        total_event_count = sum(counts)
        mean_count = total_event_count / FUTURE_REPLICATE_COUNT
        count_quantiles = np.quantile(
            np.asarray(counts, dtype=np.float64),
            FUTURE_QUANTILE_PROBABILITIES,
            method=FUTURE_QUANTILE_METHOD,
        )
        grid_summaries: list[SparseGridExpectedCounts] = []
        for grid_index, cell_size_km in enumerate(GRID_CELL_SIZES_KM):
            cell_counts = aggregate_cell_counts[horizon_index][grid_index]
            if sum(cell_counts.values()) != total_event_count:
                raise ValueError("future grid counts do not preserve the horizon event total")
            sparse_cells = tuple(
                SparseExpectedCellCount(
                    cell_id=identifier,
                    expected_count=cell_counts[identifier] / FUTURE_REPLICATE_COUNT,
                )
                for identifier in sorted(cell_counts)
            )
            grid_summaries.append(
                SparseGridExpectedCounts(
                    cell_size_km=cell_size_km,
                    cells=sparse_cells,
                    expected_total=mean_count,
                )
            )
        horizon_summaries.append(
            FutureHorizonSummary(
                horizon_days=horizon_days,
                replicate_counts=counts,
                mean_count=mean_count,
                quantiles=tuple(
                    FutureCountQuantile(probability=probability, count=float(value))
                    for probability, value in zip(
                        FUTURE_QUANTILE_PROBABILITIES,
                        count_quantiles,
                        strict=True,
                    )
                ),
                grids=tuple(grid_summaries),
            )
        )
    return FutureIssueEnsemble(
        issue_date_local=issue_date,
        issue_id=issue_id,
        replicate_count=FUTURE_REPLICATE_COUNT,
        quantile_probabilities=FUTURE_QUANTILE_PROBABILITIES,
        quantile_method=FUTURE_QUANTILE_METHOD,
        horizons=tuple(horizon_summaries),
    )


def simulate_validation_issue_ensemble(
    parameters: ETASParameters,
    spec: ETASModelSpec,
    history_events: Sequence[ETASEvent],
    spatial_model: SpatialPoissonModel,
    study_area: StudyArea,
    grid_family: EqualAreaGridFamily,
    calendar: FrozenIssueCalendar,
    *,
    issue_date_local: str,
) -> FutureIssueEnsemble:
    """Simulate exactly 128 reusable 365-day catalogs for one validation issue."""

    _validate_study_area(study_area)
    _validate_grid_family(grid_family, study_area)
    domain = build_kde_study_area_simulation_domain(spatial_model, study_area)
    return _simulate_validation_issue_ensemble(
        parameters,
        spec,
        history_events,
        domain,
        grid_family,
        calendar,
        issue_date_local=issue_date_local,
    )


def _future_worker_resources(
    *,
    max_workers: int,
    reserve_physical_cores: int,
    physical_core_probe: PhysicalCoreProbe,
) -> FutureWorkerResources:
    configured = _positive_integer("max_workers", max_workers)
    reserve = _positive_integer("reserve_physical_cores", reserve_physical_cores)
    if reserve < MINIMUM_RESERVED_PHYSICAL_CORES:
        raise ValueError("future simulations must reserve at least two physical cores")
    detected = physical_core_probe()
    if detected is not None:
        _positive_integer("detected physical core count", detected)
    effective = 1 if detected is None else min(configured, max(1, detected - reserve))
    return FutureWorkerResources(
        detected_physical_cores=detected,
        reserve_physical_cores=reserve,
        configured_max_workers=configured,
        effective_workers=effective,
    )


def simulate_all_validation_issue_ensembles(
    parameters: ETASParameters,
    spec: ETASModelSpec,
    histories_by_issue_date: Mapping[str, Sequence[ETASEvent]],
    spatial_model: SpatialPoissonModel,
    study_area: StudyArea,
    grid_family: EqualAreaGridFamily,
    calendar: FrozenIssueCalendar,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    reserve_physical_cores: int = MINIMUM_RESERVED_PHYSICAL_CORES,
    physical_core_probe: PhysicalCoreProbe = detect_physical_core_count,
    progress: ProgressCallback | None = None,
) -> ValidationFutureEnsembles:
    """Run every validation issue with controlled outer workers and stable gathering.

    Histories must be supplied for exactly the validation calendar dates.  Worker-count
    invariance follows from the issue/replicate-specific SHA-256 seed contexts, and
    results are gathered in frozen calendar order rather than completion order.
    """

    if not isinstance(histories_by_issue_date, Mapping):
        raise TypeError("histories_by_issue_date must be a mapping")
    if not isinstance(parameters, ETASParameters):
        raise TypeError("parameters must be ETASParameters")
    if not isinstance(spec, ETASModelSpec):
        raise TypeError("spec must be ETASModelSpec")
    spec.validate_parameters(parameters)
    if not callable(physical_core_probe):
        raise TypeError("physical_core_probe must be callable")
    _validate_study_area(study_area)
    _validate_grid_family(grid_family, study_area)
    issue_dates = _validated_validation_dates(calendar)
    if set(histories_by_issue_date) != set(issue_dates):
        raise ValueError("history mapping keys must exactly match all validation issue dates")
    resources = _future_worker_resources(
        max_workers=max_workers,
        reserve_physical_cores=reserve_physical_cores,
        physical_core_probe=physical_core_probe,
    )
    domain = build_kde_study_area_simulation_domain(spatial_model, study_area)
    validated_histories = {
        issue_date: _validated_history(
            histories_by_issue_date[issue_date],
            spec=spec,
            domain=domain,
        )
        for issue_date in issue_dates
    }

    def run_issue(issue_date: str) -> FutureIssueEnsemble:
        return _simulate_validation_issue_ensemble(
            parameters,
            spec,
            validated_histories[issue_date],
            domain,
            grid_family,
            calendar,
            issue_date_local=issue_date,
        )

    if resources.effective_workers == 1:
        sequential_issues: list[FutureIssueEnsemble] = []
        for issue_date in issue_dates:
            if progress is not None:
                progress(f"future:{issue_date}:start")
            sequential_issues.append(run_issue(issue_date))
            if progress is not None:
                progress(f"future:{issue_date}:done")
        issues = tuple(sequential_issues)
    else:
        with ThreadPoolExecutor(max_workers=resources.effective_workers) as executor:
            futures = []
            for issue_date in issue_dates:
                if progress is not None:
                    progress(f"future:{issue_date}:start")
                futures.append(executor.submit(run_issue, issue_date))
            gathered: list[FutureIssueEnsemble] = []
            for issue_date, future in zip(issue_dates, futures, strict=True):
                gathered.append(future.result())
                if progress is not None:
                    progress(f"future:{issue_date}:done")
            issues = tuple(gathered)
    if tuple(issue.issue_date_local for issue in issues) != issue_dates:
        raise ValueError("validation future ensemble gather order is unstable")
    return ValidationFutureEnsembles(issues=issues, resources=resources)


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_REJECTION_ATTEMPT_LIMIT",
    "FUTURE_HORIZONS_DAYS",
    "FUTURE_MAXIMUM_EVENTS",
    "FUTURE_QUANTILE_METHOD",
    "FUTURE_QUANTILE_PROBABILITIES",
    "FUTURE_REPLICATE_COUNT",
    "FUTURE_REPLICATE_INDICES",
    "MINIMUM_RESERVED_PHYSICAL_CORES",
    "BackgroundRejectionSamplingError",
    "FutureCountQuantile",
    "FutureHorizonSummary",
    "FutureIssueEnsemble",
    "FutureWorkerResources",
    "ProgressCallback",
    "SparseExpectedCellCount",
    "SparseGridExpectedCounts",
    "ValidationFutureEnsembles",
    "build_kde_study_area_simulation_domain",
    "simulate_all_validation_issue_ensembles",
    "simulate_validation_issue_ensemble",
]
