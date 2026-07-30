"""Immutable numerical contracts for the Stage 2S spatial screen.

The objects in this module contain only target-independent grid definitions,
spatial-model state, and fit summaries.  They deliberately have no score,
assessment, hit, gate, or callback fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.poisson import GaussianMixtureFamily, SpatialQuadrature

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ModelId = Literal["S0", "R", "RP", "S1", "SP"]
AlphaSolverCase = Literal["flat", "derivative_at_zero", "derivative_at_one", "bisection"]
DerivativeComparison = Literal[
    "less_than_negative_tolerance",
    "within_tolerance",
    "greater_than_positive_tolerance",
]

FROZEN_SPATIAL_BANDWIDTH_KM = 75.0
FROZEN_GRID_SIZES_KM = (50.0, 25.0, 12.5)
PRIMARY_ALARM_BUDGET_KM2 = 600_000.0
MASS_SUM_ABSOLUTE_TOLERANCE = 1.0e-12
COMPENSATOR_DIFFERENCE_ABSOLUTE_TOLERANCE = 1.0e-10
DERIVATIVE_SIGN_TOLERANCE = 1.0e-12
CONCAVITY_POSITIVE_TOLERANCE = 1.0e-12
ALPHA_BISECTION_ITERATIONS = 64
CONVERGENCE_RELATIVE_COUNT_MAX = 0.02
CONVERGENCE_DENSITY_L1_MAX = 0.05


class EvidenceInsufficientError(RuntimeError):
    """Raised when the frozen protocol does not permit a numerical estimate."""


def _nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _read_only_float_vector(
    name: str,
    value: object,
    *,
    allow_empty: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    owned = np.array(result, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_only_int_vector(name: str, value: object, *, allow_empty: bool = False) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if allow_empty and raw.size == 0:
        owned_empty = np.asarray((), dtype=np.int64)
        owned_empty.setflags(write=False)
        return owned_empty
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain only integers")
    converted = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(raw, converted):
        raise ValueError(f"{name} contains values outside int64")
    owned = np.array(converted, dtype=np.int64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_only_xy(name: str, value: object, *, allow_empty: bool = False) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape (n, 2)")
    if not allow_empty and result.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    owned = np.array(result, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_only_model_mass(name: str, value: object) -> FloatArray:
    return _read_only_float_vector(name, value, nonnegative=True)


def _aggregate_fine_to_operational(
    fine_mass: FloatArray,
    *,
    fine_grid: SpatialGrid,
    operational_grid: SpatialGrid,
) -> FloatArray:
    parent_lookup = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(
            zip(operational_grid.rows, operational_grid.columns, strict=True)
        )
    }
    grouped: list[list[float]] = [[] for _ in range(operational_grid.cell_count)]
    for value, row, column in zip(
        fine_mass,
        fine_grid.rows,
        fine_grid.columns,
        strict=True,
    ):
        try:
            parent_index = parent_lookup[(int(row) // 2, int(column) // 2)]
        except KeyError as error:
            raise ValueError("every 12.5 km cell must have one aligned 25 km parent") from error
        grouped[parent_index].append(float(value))
    if any(not values for values in grouped):
        raise ValueError("every 25 km parent must contain at least one 12.5 km child")
    return np.asarray([math.fsum(values) for values in grouped], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SpatialGrid:
    """One fixed, row/column-ordered clipped quadrature grid."""

    grid_id: str
    cell_size_km: float
    cell_ids: tuple[str, ...]
    rows: IntArray
    columns: IntArray
    query_xy_km: FloatArray
    clipped_area_km2: FloatArray
    _quadrature: SpatialQuadrature = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        grid_id = _nonempty_text("grid_id", self.grid_id)
        cell_size = float(self.cell_size_km)
        if cell_size not in FROZEN_GRID_SIZES_KM:
            raise ValueError("cell_size_km must be exactly 50, 25, or 12.5")
        cell_ids = tuple(_nonempty_text("cell_id", value) for value in self.cell_ids)
        if not cell_ids:
            raise ValueError("cell_ids must not be empty")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("cell_ids must be unique")
        rows = _read_only_int_vector("rows", self.rows)
        columns = _read_only_int_vector("columns", self.columns)
        query_xy = _read_only_xy("query_xy_km", self.query_xy_km)
        areas = _read_only_float_vector("clipped_area_km2", self.clipped_area_km2)
        length = len(cell_ids)
        if not (rows.size == columns.size == query_xy.shape[0] == areas.size == length):
            raise ValueError("all spatial-grid columns must have one common length")
        if np.any(areas <= 0.0):
            raise ValueError("clipped_area_km2 must contain only positive values")
        row_columns = tuple(
            (int(row), int(column)) for row, column in zip(rows, columns, strict=True)
        )
        if row_columns != tuple(sorted(row_columns)):
            raise ValueError("grid rows and columns must be in fixed ascending order")
        if len(set(row_columns)) != length:
            raise ValueError("grid row/column pairs must be unique")
        quadrature = SpatialQuadrature(
            cell_ids=cell_ids,
            x_km=query_xy[:, 0],
            y_km=query_xy[:, 1],
            area_km2=areas,
        )
        object.__setattr__(self, "grid_id", grid_id)
        object.__setattr__(self, "cell_size_km", cell_size)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "query_xy_km", query_xy)
        object.__setattr__(self, "clipped_area_km2", areas)
        object.__setattr__(self, "_quadrature", quadrature)

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)

    @property
    def quadrature(self) -> SpatialQuadrature:
        return self._quadrature


@dataclass(frozen=True, slots=True)
class SpatialQuadratureFamily:
    """The aligned 50/25/12.5 km grids used by every Stage 2S model."""

    grids: tuple[SpatialGrid, SpatialGrid, SpatialGrid]

    def __post_init__(self) -> None:
        grids = tuple(self.grids)
        if len(grids) != 3 or any(not isinstance(grid, SpatialGrid) for grid in grids):
            raise ValueError("grids must contain exactly three SpatialGrid objects")
        if tuple(grid.cell_size_km for grid in grids) != FROZEN_GRID_SIZES_KM:
            raise ValueError("grids must be ordered as 50, 25, and 12.5 km")
        total_areas = tuple(
            math.fsum(float(area) for area in grid.clipped_area_km2) for grid in grids
        )
        reference_area = total_areas[-1]
        area_tolerance = max(1.0e-9, reference_area * 1.0e-12)
        if any(
            not math.isclose(total, reference_area, rel_tol=0.0, abs_tol=area_tolerance)
            for total in total_areas[:-1]
        ):
            raise ValueError("all quadrature grids must cover one common clipped domain")
        object.__setattr__(self, "grids", grids)

    def at(self, cell_size_km: float) -> SpatialGrid:
        requested = float(cell_size_km)
        for grid in self.grids:
            if grid.cell_size_km == requested:
                return grid
        raise KeyError(f"quadrature family has no {requested:g} km grid")


@dataclass(frozen=True, slots=True)
class GridConvergence:
    """Frozen direct-quadrature convergence evidence for one spatial density."""

    primary_relative_count_difference: float
    primary_density_l1: float
    diagnostic_relative_count_difference: float
    diagnostic_density_l1: float
    passed: bool

    def __post_init__(self) -> None:
        values = (
            float(self.primary_relative_count_difference),
            float(self.primary_density_l1),
            float(self.diagnostic_relative_count_difference),
            float(self.diagnostic_density_l1),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("convergence diagnostics must be finite and non-negative")
        expected_passed = (
            values[0] <= CONVERGENCE_RELATIVE_COUNT_MAX
            and values[1] <= CONVERGENCE_DENSITY_L1_MAX
            and values[2] <= CONVERGENCE_RELATIVE_COUNT_MAX
            and values[3] <= CONVERGENCE_DENSITY_L1_MAX
        )
        if not isinstance(self.passed, bool) or self.passed is not expected_passed:
            raise ValueError("passed must exactly reflect the frozen convergence thresholds")
        object.__setattr__(self, "primary_relative_count_difference", values[0])
        object.__setattr__(self, "primary_density_l1", values[1])
        object.__setattr__(self, "diagnostic_relative_count_difference", values[2])
        object.__setattr__(self, "diagnostic_density_l1", values[3])


@dataclass(frozen=True, slots=True)
class NormalizedSpatialDensity:
    """One immutable continuous density and its frozen operational grid masses."""

    model_id: ModelId
    grid_family: SpatialQuadratureFamily
    mass_12_5km: FloatArray
    mass_25km: FloatArray
    direct_mass_25km: FloatArray
    direct_mass_50km: FloatArray
    convergence: GridConvergence
    bandwidth_km: float = FROZEN_SPATIAL_BANDWIDTH_KM
    normalization_mass: float | None = None
    _kernel: GaussianMixtureFamily | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _baseline: NormalizedSpatialDensity | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _component: NormalizedSpatialDensity | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _alpha: float | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model_id not in {"S0", "R", "RP", "S1", "SP"}:
            raise ValueError("model_id must be one of S0, R, RP, S1, or SP")
        bandwidth = float(self.bandwidth_km)
        if bandwidth != FROZEN_SPATIAL_BANDWIDTH_KM:
            raise ValueError("Stage 2S bandwidth must be exactly 75 km")
        fine = _read_only_model_mass("mass_12_5km", self.mass_12_5km)
        operational = _read_only_model_mass("mass_25km", self.mass_25km)
        direct_25 = _read_only_model_mass("direct_mass_25km", self.direct_mass_25km)
        direct_50 = _read_only_model_mass("direct_mass_50km", self.direct_mass_50km)
        expected_lengths = (
            self.grid_family.at(12.5).cell_count,
            self.grid_family.at(25.0).cell_count,
            self.grid_family.at(25.0).cell_count,
            self.grid_family.at(50.0).cell_count,
        )
        if tuple(array.size for array in (fine, operational, direct_25, direct_50)) != (
            expected_lengths
        ):
            raise ValueError("model mass arrays must match their fixed quadrature grids")
        fine_total = math.fsum(float(value) for value in fine)
        operational_total = math.fsum(float(value) for value in operational)
        if not math.isclose(
            fine_total,
            1.0,
            rel_tol=0.0,
            abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError("12.5 km operational masses must sum to one")
        if not math.isclose(
            operational_total,
            1.0,
            rel_tol=0.0,
            abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError("25 km operational masses must sum to one")
        expected_operational = _aggregate_fine_to_operational(
            fine,
            fine_grid=self.grid_family.at(12.5),
            operational_grid=self.grid_family.at(25.0),
        )
        if not np.allclose(
            operational,
            expected_operational,
            rtol=0.0,
            atol=MASS_SUM_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                "25 km operational masses must be the aligned aggregation of 12.5 km masses"
            )
        direct_25_total = math.fsum(float(value) for value in direct_25)
        if not math.isfinite(direct_25_total) or direct_25_total <= 0.0:
            raise ValueError("direct 25 km masses must have finite positive total")
        direct_50_total = math.fsum(float(value) for value in direct_50)
        if not math.isfinite(direct_50_total) or direct_50_total <= 0.0:
            raise ValueError("direct 50 km masses must have finite positive total")
        if not self.convergence.passed:
            raise ValueError("a Stage 2S spatial density must pass frozen grid convergence")

        is_kernel = self._kernel is not None
        is_mixture = self._baseline is not None or self._component is not None
        if is_kernel == is_mixture:
            raise ValueError("a spatial density must be exactly one kernel or one mixture")
        if is_kernel:
            if self._baseline is not None or self._component is not None or self._alpha is not None:
                raise ValueError("a kernel density cannot contain mixture state")
            if self.normalization_mass is None:
                raise ValueError("a kernel density requires a normalization mass")
            normalization = float(self.normalization_mass)
            if not math.isfinite(normalization) or normalization <= 0.0:
                raise ValueError("normalization_mass must be finite and positive")
            object.__setattr__(self, "normalization_mass", normalization)
        else:
            if self._baseline is None or self._component is None or self._alpha is None:
                raise ValueError("a mixture density requires both components and alpha")
            if self.normalization_mass is not None:
                raise ValueError("a normalized mixture has no independent normalization mass")
            alpha = float(self._alpha)
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError("mixture alpha must be finite and within [0, 1]")
            if self._baseline.grid_family is not self.grid_family:
                raise ValueError("mixture baseline must use the same grid family")
            if self._component.grid_family is not self.grid_family:
                raise ValueError("mixture component must use the same grid family")
            object.__setattr__(self, "_alpha", alpha)

        object.__setattr__(self, "bandwidth_km", bandwidth)
        object.__setattr__(self, "mass_12_5km", fine)
        object.__setattr__(self, "mass_25km", operational)
        object.__setattr__(self, "direct_mass_25km", direct_25)
        object.__setattr__(self, "direct_mass_50km", direct_50)

    @property
    def alpha(self) -> float | None:
        return self._alpha

    @property
    def source_event_count(self) -> int | None:
        return self._kernel.training_event_count if self._kernel is not None else None

    @property
    def log_normalization(self) -> float | None:
        return math.log(self.normalization_mass) if self.normalization_mass is not None else None

    def density(self, x_km: object, y_km: object) -> FloatArray:
        """Evaluate the normalized continuous density without grid lookup."""

        if self._kernel is not None:
            assert self.normalization_mass is not None
            values = self._kernel.raw_densities(
                x_km,
                y_km,
                bandwidths_km=(self.bandwidth_km,),
            )[self.bandwidth_km]
            values /= self.normalization_mass
        else:
            assert self._baseline is not None
            assert self._component is not None
            assert self._alpha is not None
            if self._baseline is self._component or self._alpha == 0.0:
                return self._baseline.density(x_km, y_km)
            if self._alpha == 1.0:
                return self._component.density(x_km, y_km)
            baseline = self._baseline.density(x_km, y_km)
            component = self._component.density(x_km, y_km)
            values = np.asarray(
                (1.0 - self._alpha) * baseline + self._alpha * component,
                dtype=np.float64,
            )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("continuous density evaluation must be finite and non-negative")
        values.setflags(write=False)
        return values

    def log_density(self, x_km: object, y_km: object) -> FloatArray:
        """Evaluate stable continuous log density without a density floor."""

        if self._kernel is not None:
            assert self.normalization_mass is not None
            values = self._kernel.raw_log_densities(
                x_km,
                y_km,
                bandwidths_km=(self.bandwidth_km,),
            )[self.bandwidth_km]
            values -= math.log(self.normalization_mass)
        else:
            assert self._baseline is not None
            assert self._component is not None
            assert self._alpha is not None
            if self._baseline is self._component or self._alpha == 0.0:
                return self._baseline.log_density(x_km, y_km)
            if self._alpha == 1.0:
                return self._component.log_density(x_km, y_km)
            baseline = self._baseline.log_density(x_km, y_km)
            component = self._component.log_density(x_km, y_km)
            values = np.logaddexp(
                math.log1p(-self._alpha) + baseline,
                math.log(self._alpha) + component,
            )
        if not np.isfinite(values).all():
            raise ValueError("continuous log-density evaluation must be finite")
        values.setflags(write=False)
        return values


@dataclass(frozen=True, slots=True)
class FitEventOrder:
    """Deterministic event keys paired positionally with alpha-fit densities."""

    origin_time_ns: IntArray
    event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        origin_time = _read_only_int_vector(
            "origin_time_ns",
            self.origin_time_ns,
            allow_empty=True,
        )
        event_ids = tuple(_nonempty_text("event_id", value) for value in self.event_ids)
        if origin_time.size != len(event_ids):
            raise ValueError("origin_time_ns and event_ids must have one common length")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("fit event IDs must be unique")
        object.__setattr__(self, "origin_time_ns", origin_time)
        object.__setattr__(self, "event_ids", event_ids)

    @property
    def event_count(self) -> int:
        return len(self.event_ids)

    @property
    def deterministic_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                range(self.event_count),
                key=lambda index: (
                    int(self.origin_time_ns[index]),
                    self.event_ids[index].encode("utf-8"),
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class SignedLogDerivative:
    """Auditable endpoint derivative without overflow, underflow, or clipping."""

    sign: int
    log_abs_mean: float | None
    finite_float_value: float | None
    comparison: DerivativeComparison

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sign, int)
            or isinstance(self.sign, bool)
            or self.sign not in {-1, 0, 1}
        ):
            raise ValueError("derivative sign must be -1, 0, or 1")
        if self.sign == 0:
            if self.log_abs_mean is not None or self.finite_float_value != 0.0:
                raise ValueError("an exact-zero derivative must use null log_abs_mean and 0.0")
            expected: DerivativeComparison = "within_tolerance"
        else:
            if self.log_abs_mean is None:
                raise ValueError("a nonzero derivative requires log_abs_mean")
            log_abs = float(self.log_abs_mean)
            if not math.isfinite(log_abs):
                raise ValueError("derivative log_abs_mean must be finite")
            object.__setattr__(self, "log_abs_mean", log_abs)
            if self.finite_float_value is not None:
                finite = float(self.finite_float_value)
                if not math.isfinite(finite) or finite == 0.0:
                    raise ValueError("finite_float_value must be finite, nonzero, or null")
                if (finite > 0.0) != (self.sign > 0):
                    raise ValueError("finite_float_value sign differs from derivative sign")
                object.__setattr__(self, "finite_float_value", finite)
            if log_abs <= math.log(DERIVATIVE_SIGN_TOLERANCE):
                expected = "within_tolerance"
            elif self.sign < 0:
                expected = "less_than_negative_tolerance"
            else:
                expected = "greater_than_positive_tolerance"
        if self.comparison != expected:
            raise ValueError("derivative comparison differs from sign/log magnitude")


@dataclass(frozen=True, slots=True)
class AlphaFit:
    """One fold-level deterministic maximum-likelihood mixture weight."""

    alpha: float
    solver_case: AlphaSolverCase
    iterations: int
    derivative_at_zero: SignedLogDerivative
    derivative_at_one: SignedLogDerivative
    fit_event_count: int

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and within [0, 1]")
        if self.solver_case not in {
            "flat",
            "derivative_at_zero",
            "derivative_at_one",
            "bisection",
        }:
            raise ValueError("unknown alpha solver case")
        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations < 0
        ):
            raise ValueError("iterations must be a non-negative integer")
        if self.iterations != (
            ALPHA_BISECTION_ITERATIONS if self.solver_case == "bisection" else 0
        ):
            raise ValueError("iterations must match the frozen solver case")
        if self.solver_case in {"flat", "derivative_at_zero"} and alpha != 0.0:
            raise ValueError("flat and derivative-at-zero cases must return alpha zero")
        if self.solver_case == "derivative_at_one" and alpha != 1.0:
            raise ValueError("the derivative-at-one case must return alpha one")
        if self.solver_case == "bisection" and not 0.0 < alpha < 1.0:
            raise ValueError("the bisection case must return an interior alpha")
        if not isinstance(self.derivative_at_zero, SignedLogDerivative) or not isinstance(
            self.derivative_at_one,
            SignedLogDerivative,
        ):
            raise TypeError("endpoint derivatives must be SignedLogDerivative values")
        if (
            not isinstance(self.fit_event_count, int)
            or isinstance(self.fit_event_count, bool)
            or self.fit_event_count <= 0
        ):
            raise ValueError("fit_event_count must be a positive integer")
        object.__setattr__(self, "alpha", alpha)


@dataclass(frozen=True, slots=True)
class SharedRate:
    """The one M5--6 fit rate shared exactly by S0, S1, and SP."""

    rate_per_day: float
    event_count: int
    exposure_days: float
    assigned_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        event_ids = tuple(_nonempty_text("event_id", value) for value in self.assigned_event_ids)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("assigned M5--6 event IDs must be unique")
        if (
            not isinstance(self.event_count, int)
            or isinstance(self.event_count, bool)
            or self.event_count != len(event_ids)
            or self.event_count <= 0
        ):
            raise ValueError("event_count must equal a positive number of assigned event IDs")
        exposure = float(self.exposure_days)
        if not math.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("exposure_days must be finite and positive")
        rate = float(self.rate_per_day)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("rate_per_day must be finite and positive")
        if rate != self.event_count / exposure:
            raise ValueError("rate_per_day must equal event_count divided by exposure_days")
        object.__setattr__(self, "rate_per_day", rate)
        object.__setattr__(self, "exposure_days", exposure)
        object.__setattr__(self, "assigned_event_ids", event_ids)


@dataclass(frozen=True, slots=True)
class PairedCompensator:
    """Independently recomputed compensators for one paired model comparison."""

    rate_per_day: float
    horizon_days: float
    issue_count: int
    compensator_a: float
    compensator_b: float
    difference: float

    def __post_init__(self) -> None:
        values = (
            float(self.rate_per_day),
            float(self.horizon_days),
            float(self.compensator_a),
            float(self.compensator_b),
            float(self.difference),
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("paired compensator values must be finite")
        if values[0] <= 0.0 or values[1] <= 0.0:
            raise ValueError("paired compensator rate and horizon must be positive")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError("paired compensators must be positive")
        if (
            not isinstance(self.issue_count, int)
            or isinstance(self.issue_count, bool)
            or self.issue_count <= 0
        ):
            raise ValueError("issue_count must be a positive integer")
        if values[4] != values[2] - values[3]:
            raise ValueError("difference must be the independently computed A minus B value")
        object.__setattr__(self, "rate_per_day", values[0])
        object.__setattr__(self, "horizon_days", values[1])
        object.__setattr__(self, "compensator_a", values[2])
        object.__setattr__(self, "compensator_b", values[3])
        object.__setattr__(self, "difference", values[4])


@dataclass(frozen=True, slots=True)
class AlarmMask:
    """A complete fixed-area prefix; no partial or skipped cell is representable."""

    model_id: ModelId
    selected_cell_ids: tuple[str, ...]
    selected_indices: IntArray
    actual_area_km2: float
    budget_km2: float
    grid_id: str
    ranking_sha256: str

    def __post_init__(self) -> None:
        if self.model_id not in {"S0", "R", "RP", "S1", "SP"}:
            raise ValueError("unknown alarm model_id")
        selected_ids = tuple(
            _nonempty_text("selected_cell_id", value) for value in self.selected_cell_ids
        )
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selected alarm cell IDs must be unique")
        indices = _read_only_int_vector(
            "selected_indices",
            self.selected_indices,
            allow_empty=True,
        )
        if indices.size != len(selected_ids) or np.any(indices < 0):
            raise ValueError("selected indices must match the selected alarm cells")
        if len(set(int(index) for index in indices)) != indices.size:
            raise ValueError("selected alarm indices must be unique")
        actual_area = float(self.actual_area_km2)
        budget = float(self.budget_km2)
        if not math.isfinite(actual_area) or actual_area < 0.0:
            raise ValueError("actual_area_km2 must be finite and non-negative")
        if not math.isfinite(budget) or budget != PRIMARY_ALARM_BUDGET_KM2:
            raise ValueError("budget_km2 must equal the frozen 600000 km2 budget")
        if actual_area > budget:
            raise ValueError("actual alarm area cannot exceed the frozen budget")
        grid_id = _nonempty_text("grid_id", self.grid_id)
        ranking_sha256 = _nonempty_text("ranking_sha256", self.ranking_sha256)
        if len(ranking_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in ranking_sha256
        ):
            raise ValueError("ranking_sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "selected_cell_ids", selected_ids)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "actual_area_km2", actual_area)
        object.__setattr__(self, "budget_km2", budget)
        object.__setattr__(self, "grid_id", grid_id)
        object.__setattr__(self, "ranking_sha256", ranking_sha256)


@dataclass(frozen=True, slots=True)
class Stage2SModels:
    """The only three Stage 2S models, in the preregistered order."""

    s0: NormalizedSpatialDensity
    s1: NormalizedSpatialDensity
    sp: NormalizedSpatialDensity

    def __post_init__(self) -> None:
        if (self.s0.model_id, self.s1.model_id, self.sp.model_id) != ("S0", "S1", "SP"):
            raise ValueError("Stage 2S models must be ordered exactly as S0, S1, and SP")
        if not (
            self.s0.grid_family is self.s1.grid_family
            and self.s0.grid_family is self.sp.grid_family
        ):
            raise ValueError("S0, S1, and SP must use one common grid family")


__all__ = [
    "ALPHA_BISECTION_ITERATIONS",
    "COMPENSATOR_DIFFERENCE_ABSOLUTE_TOLERANCE",
    "CONCAVITY_POSITIVE_TOLERANCE",
    "CONVERGENCE_DENSITY_L1_MAX",
    "CONVERGENCE_RELATIVE_COUNT_MAX",
    "DERIVATIVE_SIGN_TOLERANCE",
    "FROZEN_GRID_SIZES_KM",
    "FROZEN_SPATIAL_BANDWIDTH_KM",
    "MASS_SUM_ABSOLUTE_TOLERANCE",
    "PRIMARY_ALARM_BUDGET_KM2",
    "AlarmMask",
    "AlphaFit",
    "AlphaSolverCase",
    "DerivativeComparison",
    "EvidenceInsufficientError",
    "FitEventOrder",
    "FloatArray",
    "GridConvergence",
    "IntArray",
    "ModelId",
    "NormalizedSpatialDensity",
    "PairedCompensator",
    "SharedRate",
    "SignedLogDerivative",
    "SpatialGrid",
    "SpatialQuadratureFamily",
    "Stage2SModels",
]
