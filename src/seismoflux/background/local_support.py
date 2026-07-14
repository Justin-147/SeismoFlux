"""Causal local catalog-completeness support for stage-2 background models.

The support surface is built exclusively from the fixed study geometry and
earthquakes known at a snapshot fit cutoff.  A spatially local completeness
estimate above the frozen maximum removes only its corresponding 500 km cell;
temporal estimates above that maximum remain a hard scientific failure.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from shapely import normalize as normalize_geometry
from shapely import to_wkb
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.completeness import (
    CATALOG_ANCHOR_UTC,
    COMPLETENESS_CANDIDATES,
    FINAL_PARTIAL_BLOCK_MINIMUM_YEARS,
    SPATIAL_BASE_CELL_KM,
    SPATIAL_MINIMUM_EVENTS,
    SPATIAL_PARENT_CELL_KM,
    TEMPORAL_BLOCK_YEARS,
    TEMPORAL_MINIMUM_EVENTS,
    CompletenessEvent,
    CompletenessScientificInability,
    StratumCompletenessEstimate,
    TemporalBlockDiagnostic,
    estimate_aki_b_value,
    maximum_curvature_estimate,
    select_candidate_magnitude,
)

LOCAL_SUPPORT_PROTOCOL_VERSION = "seismoflux_local_support_v1"
MINIMUM_RETAINED_AREA_FRACTION = 0.95
MAXIMUM_SUPPORTED_RAW_MC = 4.0

_SECONDS_PER_TROPICAL_YEAR = 365.2425 * 24.0 * 60.0 * 60.0
_COMPARISON_TOLERANCE = 1.0e-12
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

SupportStatus = Literal["supported", "indeterminate", "unsupported"]
SupportSource = Literal["base_500km", "fixed_1000km_parent"]


class LocalSupportError(RuntimeError):
    """Base class for failures of the local-support protocol."""


class LocalSupportRetentionError(LocalSupportError):
    """Raised when spatial exclusions retain less than 95% of study area."""

    def __init__(
        self,
        retained_area_fraction: float,
        unsupported_cell_ids: tuple[str, ...],
    ) -> None:
        self.retained_area_fraction = retained_area_fraction
        self.unsupported_cell_ids = unsupported_cell_ids
        super().__init__(
            "local completeness support retained "
            f"{retained_area_fraction:.12g} of study area, below frozen minimum "
            f"{MINIMUM_RETAINED_AREA_FRACTION:g}; unsupported cells: "
            + ", ".join(unsupported_cell_ids)
        )


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _utc_text(value: datetime) -> str:
    _require_utc("datetime", value)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _geometry_wkb(geometry: BaseGeometry) -> bytes:
    normalized = normalize_geometry(geometry)
    return cast(
        bytes,
        to_wkb(
            normalized,
            hex=False,
            output_dimension=2,
            byte_order=1,
            include_srid=False,
        ),
    )


def _fixed_cell_id(cell_size_km: float, row: int, column: int) -> str:
    cell_size_mm = int(cell_size_km * 1_000_000.0)
    return f"g{cell_size_mm:08d}_r{row:+08d}_c{column:+08d}"


def _require_finite_mc(name: str, value: float | None) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_candidate_mapping(raw_mc: float, candidate_mc: float) -> None:
    expected = select_candidate_magnitude(raw_mc)
    if candidate_mc != expected:
        raise ValueError("candidate Mc must be the frozen upward mapping of raw Mc")


@dataclass(frozen=True, slots=True)
class LocalSupportAudit:
    """Input filtering counts retained outside the causal support manifest."""

    input_event_count: int
    included_historical_inside_count: int
    excluded_outside_count: int
    excluded_pre_1970_count: int
    excluded_future_origin_count: int
    excluded_unavailable_count: int


@dataclass(frozen=True, slots=True)
class LocalSupportCell:
    """One exactly clipped fixed 500 km support cell."""

    cell_id: str
    row: int
    column: int
    clipped_geometry: BaseGeometry
    clipped_area_m2: float
    status: SupportStatus
    source: SupportSource
    base_event_count: int
    source_event_count: int
    parent_cell_id: str | None
    raw_mc: float | None
    candidate_mc: float | None
    applied_mc: float | None

    def __post_init__(self) -> None:
        expected_id = _fixed_cell_id(SPATIAL_BASE_CELL_KM, self.row, self.column)
        if self.cell_id != expected_id:
            raise ValueError("local support cell_id does not match its fixed grid index")
        area = float(self.clipped_area_m2)
        if (
            self.clipped_geometry.is_empty
            or not self.clipped_geometry.is_valid
            or not math.isfinite(area)
            or area <= 0.0
            or float(self.clipped_geometry.area) != area
        ):
            raise ValueError("local support cell must have exact positive clipped geometry")
        if (
            type(self.base_event_count) is not int
            or type(self.source_event_count) is not int
            or self.base_event_count < 0
            or self.source_event_count < self.base_event_count
        ):
            raise ValueError("local support event counts are inconsistent")
        if self.source == "base_500km":
            if (
                self.parent_cell_id is not None
                or self.source_event_count != self.base_event_count
                or self.base_event_count < SPATIAL_MINIMUM_EVENTS
                or self.status == "indeterminate"
            ):
                raise ValueError("base-cell support cannot declare a parent source")
        elif self.source == "fixed_1000km_parent":
            expected_parent_id = _fixed_cell_id(
                SPATIAL_PARENT_CELL_KM,
                self.row // 2,
                self.column // 2,
            )
            if (
                self.parent_cell_id != expected_parent_id
                or self.base_event_count >= SPATIAL_MINIMUM_EVENTS
                or (
                    self.status == "indeterminate"
                    and self.source_event_count >= SPATIAL_MINIMUM_EVENTS
                )
                or (
                    self.status != "indeterminate"
                    and self.source_event_count < SPATIAL_MINIMUM_EVENTS
                )
            ):
                raise ValueError("sparse-cell support is inconsistent with its fixed parent")
        else:
            raise ValueError("unknown local support source")

        _require_finite_mc("raw Mc", self.raw_mc)
        _require_finite_mc("candidate Mc", self.candidate_mc)
        _require_finite_mc("applied Mc", self.applied_mc)

        if self.status == "supported":
            if self.raw_mc is None or self.candidate_mc is None or self.applied_mc is None:
                raise ValueError("supported cells require raw, candidate, and applied Mc")
            if self.raw_mc > MAXIMUM_SUPPORTED_RAW_MC + _COMPARISON_TOLERANCE:
                raise ValueError("supported cell raw Mc exceeds the frozen maximum")
            _require_candidate_mapping(self.raw_mc, self.candidate_mc)
            if self.applied_mc not in COMPLETENESS_CANDIDATES:
                raise ValueError("supported cell applied Mc is not frozen")
        elif self.status == "indeterminate":
            if (
                self.raw_mc is not None
                or self.candidate_mc is not None
                or self.applied_mc not in COMPLETENESS_CANDIDATES
            ):
                raise ValueError("indeterminate cells must use only the common applied Mc")
        elif self.status == "unsupported":
            if (
                self.raw_mc is None
                or self.raw_mc <= MAXIMUM_SUPPORTED_RAW_MC + _COMPARISON_TOLERANCE
                or self.candidate_mc is not None
                or self.applied_mc is not None
            ):
                raise ValueError("unsupported cells require only a raw Mc above 4.0")
        else:
            raise ValueError("unknown local support status")


@dataclass(frozen=True, slots=True)
class LocalSupportSnapshot:
    """Complete causal support surface for one fit cutoff."""

    support_id: str
    fit_end_utc: datetime
    audit: LocalSupportAudit
    temporal_blocks: tuple[TemporalBlockDiagnostic, ...]
    cells: tuple[LocalSupportCell, ...]
    common_mc: float
    retained_selected_event_count: int
    retained_selected_aki_b_value: float
    total_area_m2: float
    retained_area_m2: float
    retained_area_fraction: float
    retained_geometry: BaseGeometry
    historical_event_sha256: str
    study_area_sha256: str

    def __post_init__(self) -> None:
        _require_utc("fit_end_utc", self.fit_end_utc)
        _validate_temporal_block_sequence(self.temporal_blocks, self.fit_end_utc)
        if (
            sum(block.event_count for block in self.temporal_blocks)
            != self.audit.included_historical_inside_count
        ):
            raise ValueError("temporal blocks must cover every historical event exactly once")
        if not re.fullmatch(r"local-support-[0-9a-f]{16}", self.support_id):
            raise ValueError("support_id must use the deterministic local-support address format")
        if not _SHA256_RE.fullmatch(self.historical_event_sha256):
            raise ValueError("historical_event_sha256 must be a lowercase SHA-256")
        if not _SHA256_RE.fullmatch(self.study_area_sha256):
            raise ValueError("study_area_sha256 must be a lowercase SHA-256")
        if self.common_mc not in COMPLETENESS_CANDIDATES:
            raise ValueError("common_mc must be one of the frozen candidates")
        if self.retained_selected_event_count <= 0:
            raise ValueError("retained support must contain complete events")
        if not math.isfinite(self.retained_selected_aki_b_value):
            raise ValueError("retained selected Aki b-value must be finite")
        order = tuple((cell.row, cell.column) for cell in self.cells)
        if not self.cells or order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise ValueError("local support cells must be unique and row/column sorted")
        if any(
            cell.applied_mc is not None
            and not math.isclose(
                cell.applied_mc,
                self.common_mc,
                rel_tol=0.0,
                abs_tol=_COMPARISON_TOLERANCE,
            )
            for cell in self.cells
        ):
            raise ValueError("every retained cell must use the snapshot common Mc")

        total = math.fsum(cell.clipped_area_m2 for cell in self.cells)
        retained = math.fsum(
            cell.clipped_area_m2 for cell in self.cells if cell.status != "unsupported"
        )
        if total != self.total_area_m2 or retained != self.retained_area_m2:
            raise ValueError("snapshot areas must be exact sums of clipped cells")
        expected_fraction = retained / total
        if self.retained_area_fraction != expected_fraction:
            raise ValueError("retained area fraction is inconsistent")
        if expected_fraction < MINIMUM_RETAINED_AREA_FRACTION:
            raise ValueError("snapshot violates the frozen retained-area gate")
        if self.retained_geometry.is_empty or not self.retained_geometry.is_valid:
            raise ValueError("retained support geometry must be non-empty and valid")
        if not math.isclose(
            float(self.retained_geometry.area),
            retained,
            rel_tol=1.0e-12,
            abs_tol=1.0e-6,
        ):
            raise ValueError("retained geometry area differs from exact clipped-cell sum")


@dataclass(frozen=True, slots=True)
class LocalSupportTemporalManifest:
    """JSON-safe temporal completeness evidence used by the support address."""

    block_id: str
    start_utc: str
    end_utc: str
    duration_years: float
    is_partial: bool
    event_count: int
    eligible: bool
    raw_mc: float | None
    candidate_mc: float | None


@dataclass(frozen=True, slots=True)
class LocalSupportCellManifest:
    """JSON-safe fixed-cell identity and causal support decision.

    The clipped geometry itself is deliberately excluded from the public
    manifest.  It is reconstructed from the sealed local study-area input
    whenever the manifest is loaded for execution.
    """

    cell_id: str
    row: int
    column: int
    clipped_area_m2: float
    status: SupportStatus
    source: SupportSource
    base_event_count: int
    source_event_count: int
    parent_cell_id: str | None
    raw_mc: float | None
    candidate_mc: float | None
    applied_mc: float | None


@dataclass(frozen=True, slots=True)
class LocalSupportManifest:
    """Target-free, score-free manifest for one local support surface."""

    protocol_version: str
    support_id: str
    fit_end_utc: str
    historical_event_count: int
    historical_event_sha256: str
    study_area_sha256: str
    base_cell_size_km: float
    parent_cell_size_km: float
    minimum_events_per_stratum: int
    maximum_supported_raw_mc: float
    minimum_retained_area_fraction: float
    common_mc: float
    retained_selected_event_count: int
    retained_selected_aki_b_value: float
    total_area_m2: float
    retained_area_m2: float
    retained_area_fraction: float
    temporal_blocks: tuple[LocalSupportTemporalManifest, ...]
    cells: tuple[LocalSupportCellManifest, ...]


@dataclass(frozen=True, slots=True)
class LocalSupportFixedCellIdentity:
    """Geometry-free public identity of one fixed 500 km clipped cell."""

    cell_id: str
    row: int
    column: int
    clipped_area_m2: float


@dataclass(frozen=True, slots=True)
class LocalSupportStudyAreaIdentity:
    """Locally reconstructed identity of the sealed projected study area."""

    study_area_sha256: str
    total_area_m2: float
    fixed_cells: tuple[LocalSupportFixedCellIdentity, ...]


@dataclass(frozen=True, slots=True)
class _ClippedCell:
    row: int
    column: int
    geometry: BaseGeometry
    area_m2: float


@dataclass(frozen=True, slots=True)
class _SpatialEstimate:
    raw_mc: float
    candidate_mc: float | None


@dataclass(frozen=True, slots=True)
class _CellDraft:
    clipped: _ClippedCell
    status: SupportStatus
    source: SupportSource
    base_event_count: int
    source_event_count: int
    parent_cell_id: str | None
    raw_mc: float | None
    candidate_mc: float | None


def _validate_study_geometry(study_area_equal_area: BaseGeometry) -> None:
    if not isinstance(study_area_equal_area, BaseGeometry):
        raise TypeError("study_area_equal_area must be a Shapely geometry")
    if study_area_equal_area.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("study_area_equal_area must be polygonal")
    if (
        study_area_equal_area.is_empty
        or not study_area_equal_area.is_valid
        or not math.isfinite(float(study_area_equal_area.area))
        or study_area_equal_area.area <= 0.0
        or not all(math.isfinite(float(value)) for value in study_area_equal_area.bounds)
    ):
        raise ValueError("study_area_equal_area must be non-empty, valid, and positive-area")


def _cell_index(coordinate_m: float, cell_size_km: float) -> int:
    return math.floor(coordinate_m / (cell_size_km * 1_000.0))


def _build_clipped_base_cells(study_area_equal_area: BaseGeometry) -> tuple[_ClippedCell, ...]:
    _validate_study_geometry(study_area_equal_area)
    minimum_x, minimum_y, maximum_x, maximum_y = study_area_equal_area.bounds
    minimum_row = _cell_index(minimum_y, SPATIAL_BASE_CELL_KM)
    maximum_row = _cell_index(maximum_y, SPATIAL_BASE_CELL_KM)
    minimum_column = _cell_index(minimum_x, SPATIAL_BASE_CELL_KM)
    maximum_column = _cell_index(maximum_x, SPATIAL_BASE_CELL_KM)

    cells: list[_ClippedCell] = []
    cell_m = SPATIAL_BASE_CELL_KM * 1_000.0
    for row in range(minimum_row, maximum_row + 1):
        for column in range(minimum_column, maximum_column + 1):
            minimum_cell_x = column * cell_m
            minimum_cell_y = row * cell_m
            clipped = study_area_equal_area.intersection(
                box(
                    minimum_cell_x,
                    minimum_cell_y,
                    minimum_cell_x + cell_m,
                    minimum_cell_y + cell_m,
                )
            )
            area_m2 = float(clipped.area)
            if clipped.is_empty or not math.isfinite(area_m2) or area_m2 <= 0.0:
                continue
            if not clipped.is_valid:
                raise ValueError("clipped local-support geometry is invalid")
            normalized_clipped = cast(BaseGeometry, normalize_geometry(clipped))
            normalized_area_m2 = float(normalized_clipped.area)
            cells.append(
                _ClippedCell(
                    row=row,
                    column=column,
                    geometry=normalized_clipped,
                    area_m2=normalized_area_m2,
                )
            )
    if not cells:
        raise ValueError("study area produced no positive-area 500 km support cells")
    return tuple(cells)


def _resolve_base_cell_key(
    x_m: float,
    y_m: float,
    clipped_by_key: dict[tuple[int, int], _ClippedCell],
) -> tuple[int, int] | None:
    """Assign a covered point using half-open high-side priority.

    A study-area outer boundary can coincide exactly with a grid line.  In that
    measure-zero case the half-open preferred cell can have no positive-area
    study intersection, so the covered adjacent low-side cell is used instead.
    """

    row = _cell_index(y_m, SPATIAL_BASE_CELL_KM)
    column = _cell_index(x_m, SPATIAL_BASE_CELL_KM)
    cell_m = SPATIAL_BASE_CELL_KM * 1_000.0
    on_horizontal_line = y_m == row * cell_m
    on_vertical_line = x_m == column * cell_m
    candidate_keys = [(row, column)]
    if on_vertical_line:
        candidate_keys.append((row, column - 1))
    if on_horizontal_line:
        candidate_keys.append((row - 1, column))
    if on_horizontal_line and on_vertical_line:
        candidate_keys.append((row - 1, column - 1))

    event_point = Point(x_m, y_m)
    for key in candidate_keys:
        clipped = clipped_by_key.get(key)
        if clipped is not None and clipped.geometry.covers(event_point):
            return key
    return None


def build_local_support_study_area_identity(
    study_area_equal_area: BaseGeometry,
) -> LocalSupportStudyAreaIdentity:
    """Rebuild the geometry-free fixed-grid identity from a local study area.

    No geometry bytes leave this function.  The returned digest, row/column
    indices, cell IDs, and exact clipped areas are sufficient to bind a public
    support manifest to the sealed local geometry used for execution.
    """

    clipped_cells = _build_clipped_base_cells(study_area_equal_area)
    fixed_cells = tuple(
        LocalSupportFixedCellIdentity(
            cell_id=_fixed_cell_id(SPATIAL_BASE_CELL_KM, cell.row, cell.column),
            row=cell.row,
            column=cell.column,
            clipped_area_m2=cell.area_m2,
        )
        for cell in clipped_cells
    )
    return LocalSupportStudyAreaIdentity(
        study_area_sha256=hashlib.sha256(_geometry_wkb(study_area_equal_area)).hexdigest(),
        total_area_m2=math.fsum(cell.clipped_area_m2 for cell in fixed_cells),
        fixed_cells=fixed_cells,
    )


def _filter_historical_events(
    events: Iterable[CompletenessEvent], fit_end_utc: datetime
) -> tuple[tuple[CompletenessEvent, ...], LocalSupportAudit]:
    supplied = tuple(events)
    seen_ids: set[str] = set()
    included: list[CompletenessEvent] = []
    outside = 0
    pre_1970 = 0
    future_origin = 0
    unavailable = 0

    for event in supplied:
        if not isinstance(event, CompletenessEvent):
            raise TypeError("events must contain CompletenessEvent instances")
        if event.event_id in seen_ids:
            raise ValueError(f"duplicate physical event_id: {event.event_id}")
        seen_ids.add(event.event_id)
        if not event.inside_study_area:
            outside += 1
        elif event.origin_time_utc < CATALOG_ANCHOR_UTC:
            pre_1970 += 1
        elif event.origin_time_utc > fit_end_utc:
            future_origin += 1
        elif event.available_at > fit_end_utc:
            unavailable += 1
        else:
            included.append(event)

    included.sort(key=lambda event: (event.origin_time_utc, event.available_at, event.event_id))
    return tuple(included), LocalSupportAudit(
        input_event_count=len(supplied),
        included_historical_inside_count=len(included),
        excluded_outside_count=outside,
        excluded_pre_1970_count=pre_1970,
        excluded_future_origin_count=future_origin,
        excluded_unavailable_count=unavailable,
    )


def _stratum_estimate(
    events: Sequence[CompletenessEvent], *, stratum_id: str
) -> StratumCompletenessEstimate:
    maxc = maximum_curvature_estimate(event.magnitude for event in events)
    try:
        candidate_mc = select_candidate_magnitude(maxc.corrected_magnitude)
    except CompletenessScientificInability as error:
        raise CompletenessScientificInability(
            error.reason_code,
            f"{stratum_id}: {error}",
        ) from error
    complete_magnitudes = tuple(
        event.magnitude
        for event in events
        if event.magnitude >= candidate_mc - _COMPARISON_TOLERANCE
    )
    b_value = (
        estimate_aki_b_value(complete_magnitudes, mc=candidate_mc) if complete_magnitudes else None
    )
    return StratumCompletenessEstimate(
        maxc=maxc,
        candidate_mc=candidate_mc,
        complete_event_count=len(complete_magnitudes),
        aki_b_value=b_value,
    )


def _add_years(value: datetime, years: int) -> datetime:
    return value.replace(year=value.year + years)


def _validate_temporal_block_sequence(
    blocks: Sequence[TemporalBlockDiagnostic],
    fit_end_utc: datetime,
) -> None:
    if fit_end_utc <= CATALOG_ANCHOR_UTC or not blocks:
        raise ValueError("temporal blocks require a fit cutoff after the catalog anchor")
    expected_start = CATALOG_ANCHOR_UTC
    for block in blocks:
        _require_utc("temporal start_utc", block.start_utc)
        _require_utc("temporal end_utc", block.end_utc)
        if block.start_utc != expected_start or block.start_utc >= fit_end_utc:
            raise ValueError("temporal blocks must be contiguous from the 1970 anchor")

        nominal_end = _add_years(block.start_utc, TEMPORAL_BLOCK_YEARS)
        expected_end = min(nominal_end, fit_end_utc)
        expected_partial = nominal_end > fit_end_utc
        expected_block_id = (
            f"{block.start_utc.year:04d}-partial-{fit_end_utc.date().isoformat()}"
            if expected_partial
            else f"{block.start_utc.year:04d}-{nominal_end.year - 1:04d}"
        )
        expected_duration = (
            expected_end - block.start_utc
        ).total_seconds() / _SECONDS_PER_TROPICAL_YEAR
        if (
            block.end_utc != expected_end
            or type(block.is_partial) is not bool
            or block.is_partial != expected_partial
            or block.block_id != expected_block_id
            or not math.isfinite(float(block.duration_years))
            or block.duration_years != expected_duration
        ):
            raise ValueError("temporal block identity differs from the frozen calendar")
        if type(block.event_count) is not int or block.event_count < 0:
            raise ValueError("temporal block event count must be a non-negative integer")
        if type(block.eligible) is not bool:
            raise ValueError("temporal block eligibility must be a bool")

        duration_eligible = not expected_partial or fit_end_utc >= _add_years(
            block.start_utc,
            FINAL_PARTIAL_BLOCK_MINIMUM_YEARS,
        )
        expected_eligible = block.event_count >= TEMPORAL_MINIMUM_EVENTS and duration_eligible
        if block.eligible != expected_eligible:
            raise ValueError("temporal block eligibility differs from count and duration rules")
        if not expected_eligible:
            if block.estimate is not None:
                raise ValueError("ineligible temporal block cannot carry an estimate")
        else:
            if block.estimate is None:
                raise ValueError("eligible temporal block requires an estimate")
            raw_mc = block.estimate.maxc.corrected_magnitude
            candidate_mc = block.estimate.candidate_mc
            _require_finite_mc("temporal raw Mc", raw_mc)
            _require_finite_mc("temporal candidate Mc", candidate_mc)
            if raw_mc > MAXIMUM_SUPPORTED_RAW_MC + _COMPARISON_TOLERANCE:
                raise ValueError("temporal raw Mc above 4.0 is a global hard failure")
            _require_candidate_mapping(raw_mc, candidate_mc)
            if block.estimate.maxc.event_count != block.event_count:
                raise ValueError("temporal MAXC count must equal its block event count")
            if not 0 < block.estimate.complete_event_count <= block.event_count:
                raise ValueError("temporal complete-event count is inconsistent")
        expected_start = expected_end

    if expected_start != fit_end_utc:
        raise ValueError("final temporal block must end at fit_end_utc")


def _build_temporal_blocks(
    events: Sequence[CompletenessEvent], fit_end_utc: datetime
) -> tuple[TemporalBlockDiagnostic, ...]:
    blocks: list[TemporalBlockDiagnostic] = []
    start = CATALOG_ANCHOR_UTC
    while start < fit_end_utc:
        nominal_end = _add_years(start, TEMPORAL_BLOCK_YEARS)
        is_partial = nominal_end > fit_end_utc
        end = fit_end_utc if is_partial else nominal_end
        is_final = end == fit_end_utc
        block_events = tuple(
            event
            for event in events
            if event.origin_time_utc >= start
            and (event.origin_time_utc <= end if is_final else event.origin_time_utc < end)
        )
        duration_years = (end - start).total_seconds() / _SECONDS_PER_TROPICAL_YEAR
        duration_eligible = not is_partial or fit_end_utc >= _add_years(
            start, FINAL_PARTIAL_BLOCK_MINIMUM_YEARS
        )
        eligible = len(block_events) >= TEMPORAL_MINIMUM_EVENTS and duration_eligible
        block_id = (
            f"{start.year:04d}-partial-{fit_end_utc.date().isoformat()}"
            if is_partial
            else f"{start.year:04d}-{nominal_end.year - 1:04d}"
        )
        estimate = (
            _stratum_estimate(block_events, stratum_id=f"temporal/{block_id}") if eligible else None
        )
        blocks.append(
            TemporalBlockDiagnostic(
                block_id=block_id,
                start_utc=start,
                end_utc=end,
                duration_years=duration_years,
                is_partial=is_partial,
                event_count=len(block_events),
                eligible=eligible,
                estimate=estimate,
            )
        )
        if is_partial:
            break
        start = nominal_end
    return tuple(blocks)


def _spatial_estimate(events: Sequence[CompletenessEvent]) -> _SpatialEstimate:
    maxc = maximum_curvature_estimate(event.magnitude for event in events)
    raw_mc = maxc.corrected_magnitude
    candidate_mc = (
        None
        if raw_mc > MAXIMUM_SUPPORTED_RAW_MC + _COMPARISON_TOLERANCE
        else select_candidate_magnitude(raw_mc)
    )
    return _SpatialEstimate(raw_mc=raw_mc, candidate_mc=candidate_mc)


def _historical_event_digest(events: Sequence[CompletenessEvent]) -> str:
    payload = [
        {
            "event_id": event.event_id,
            "origin_time_utc": _utc_text(event.origin_time_utc),
            "available_at": _utc_text(event.available_at),
            "magnitude": event.magnitude,
            "x_m": event.x_m,
            "y_m": event.y_m,
        }
        for event in events
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _temporal_manifest(
    temporal_blocks: Sequence[TemporalBlockDiagnostic],
) -> tuple[LocalSupportTemporalManifest, ...]:
    return tuple(
        LocalSupportTemporalManifest(
            block_id=block.block_id,
            start_utc=_utc_text(block.start_utc),
            end_utc=_utc_text(block.end_utc),
            duration_years=block.duration_years,
            is_partial=block.is_partial,
            event_count=block.event_count,
            eligible=block.eligible,
            raw_mc=(
                block.estimate.maxc.corrected_magnitude if block.estimate is not None else None
            ),
            candidate_mc=(block.estimate.candidate_mc if block.estimate is not None else None),
        )
        for block in temporal_blocks
    )


def _cell_manifest(cells: Sequence[LocalSupportCell]) -> tuple[LocalSupportCellManifest, ...]:
    return tuple(
        LocalSupportCellManifest(
            cell_id=cell.cell_id,
            row=cell.row,
            column=cell.column,
            clipped_area_m2=cell.clipped_area_m2,
            status=cell.status,
            source=cell.source,
            base_event_count=cell.base_event_count,
            source_event_count=cell.source_event_count,
            parent_cell_id=cell.parent_cell_id,
            raw_mc=cell.raw_mc,
            candidate_mc=cell.candidate_mc,
            applied_mc=cell.applied_mc,
        )
        for cell in cells
    )


def _manifest_payload(
    *,
    fit_end_utc: datetime,
    historical_event_count: int,
    historical_event_sha256: str,
    study_area_sha256: str,
    common_mc: float,
    retained_selected_event_count: int,
    retained_selected_aki_b_value: float,
    total_area_m2: float,
    retained_area_m2: float,
    retained_area_fraction: float,
    temporal_blocks: tuple[LocalSupportTemporalManifest, ...],
    cells: tuple[LocalSupportCellManifest, ...],
) -> dict[str, object]:
    return {
        "protocol_version": LOCAL_SUPPORT_PROTOCOL_VERSION,
        "fit_end_utc": _utc_text(fit_end_utc),
        "historical_event_count": historical_event_count,
        "historical_event_sha256": historical_event_sha256,
        "study_area_sha256": study_area_sha256,
        "base_cell_size_km": SPATIAL_BASE_CELL_KM,
        "parent_cell_size_km": SPATIAL_PARENT_CELL_KM,
        "minimum_events_per_stratum": SPATIAL_MINIMUM_EVENTS,
        "maximum_supported_raw_mc": MAXIMUM_SUPPORTED_RAW_MC,
        "minimum_retained_area_fraction": MINIMUM_RETAINED_AREA_FRACTION,
        "common_mc": common_mc,
        "retained_selected_event_count": retained_selected_event_count,
        "retained_selected_aki_b_value": retained_selected_aki_b_value,
        "total_area_m2": total_area_m2,
        "retained_area_m2": retained_area_m2,
        "retained_area_fraction": retained_area_fraction,
        "temporal_blocks": [asdict(block) for block in temporal_blocks],
        "cells": [asdict(cell) for cell in cells],
    }


def _support_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"local-support-{digest[:16]}"


def build_local_support_snapshot(
    events: Iterable[CompletenessEvent],
    *,
    fit_end_utc: datetime,
    study_area_equal_area: BaseGeometry,
) -> LocalSupportSnapshot:
    """Build and enforce the frozen local-support protocol for one snapshot.

    All temporal estimates remain global hard gates.  Spatial estimates above
    4.0 mark only their fixed 500 km child cells unsupported.  Sparse children
    inherit the fixed 1000 km parent estimate; a still-sparse parent is retained
    as indeterminate and later receives the common Mc.
    """

    _require_utc("fit_end_utc", fit_end_utc)
    if fit_end_utc <= CATALOG_ANCHOR_UTC:
        raise LocalSupportError("fit_end_utc must be after the 1970 catalog anchor")
    clipped_cells = _build_clipped_base_cells(study_area_equal_area)
    historical_events, audit = _filter_historical_events(events, fit_end_utc)

    temporal_blocks = _build_temporal_blocks(historical_events, fit_end_utc)
    eligible_temporal = tuple(block for block in temporal_blocks if block.estimate is not None)
    if not eligible_temporal:
        raise CompletenessScientificInability(
            "no_eligible_temporal_stratum",
            "no eligible temporal completeness stratum",
        )

    clipped_by_key = {(cell.row, cell.column): cell for cell in clipped_cells}
    by_base: dict[tuple[int, int], list[CompletenessEvent]] = defaultdict(list)
    by_parent: dict[tuple[int, int], list[CompletenessEvent]] = defaultdict(list)
    base_key_by_event_id: dict[str, tuple[int, int]] = {}
    unassigned_keys: set[tuple[int, int]] = set()
    for event in historical_events:
        preferred_key = (
            _cell_index(event.y_m, SPATIAL_BASE_CELL_KM),
            _cell_index(event.x_m, SPATIAL_BASE_CELL_KM),
        )
        base_key = _resolve_base_cell_key(
            event.x_m,
            event.y_m,
            clipped_by_key,
        )
        if base_key is None:
            unassigned_keys.add(preferred_key)
            continue
        parent_key = (base_key[0] // 2, base_key[1] // 2)
        by_base[base_key].append(event)
        by_parent[parent_key].append(event)
        base_key_by_event_id[event.event_id] = base_key

    if unassigned_keys:
        raise ValueError(
            "inside-study events map outside positive-area clipped support cells: "
            + ", ".join(f"r{row:+d}/c{column:+d}" for row, column in sorted(unassigned_keys))
        )

    base_estimates: dict[tuple[int, int], _SpatialEstimate] = {}
    parent_estimates: dict[tuple[int, int], _SpatialEstimate] = {}
    drafts: list[_CellDraft] = []
    spatial_candidates: list[float] = []
    for clipped in clipped_cells:
        base_key = (clipped.row, clipped.column)
        base_events = by_base.get(base_key, [])
        if len(base_events) >= SPATIAL_MINIMUM_EVENTS:
            estimate = base_estimates.setdefault(base_key, _spatial_estimate(base_events))
            status: SupportStatus = "unsupported" if estimate.candidate_mc is None else "supported"
            if estimate.candidate_mc is not None:
                spatial_candidates.append(estimate.candidate_mc)
            drafts.append(
                _CellDraft(
                    clipped=clipped,
                    status=status,
                    source="base_500km",
                    base_event_count=len(base_events),
                    source_event_count=len(base_events),
                    parent_cell_id=None,
                    raw_mc=estimate.raw_mc,
                    candidate_mc=estimate.candidate_mc,
                )
            )
            continue

        parent_key = (clipped.row // 2, clipped.column // 2)
        parent_events = by_parent.get(parent_key, [])
        parent_id = _fixed_cell_id(SPATIAL_PARENT_CELL_KM, *parent_key)
        if len(parent_events) < SPATIAL_MINIMUM_EVENTS:
            drafts.append(
                _CellDraft(
                    clipped=clipped,
                    status="indeterminate",
                    source="fixed_1000km_parent",
                    base_event_count=len(base_events),
                    source_event_count=len(parent_events),
                    parent_cell_id=parent_id,
                    raw_mc=None,
                    candidate_mc=None,
                )
            )
            continue

        parent_estimate = parent_estimates.get(parent_key)
        if parent_estimate is None:
            parent_estimate = _spatial_estimate(parent_events)
            parent_estimates[parent_key] = parent_estimate
        status = "unsupported" if parent_estimate.candidate_mc is None else "supported"
        if parent_estimate.candidate_mc is not None:
            spatial_candidates.append(parent_estimate.candidate_mc)
        drafts.append(
            _CellDraft(
                clipped=clipped,
                status=status,
                source="fixed_1000km_parent",
                base_event_count=len(base_events),
                source_event_count=len(parent_events),
                parent_cell_id=parent_id,
                raw_mc=parent_estimate.raw_mc,
                candidate_mc=parent_estimate.candidate_mc,
            )
        )

    temporal_candidates = [
        block.estimate.candidate_mc for block in eligible_temporal if block.estimate is not None
    ]
    common_mc = max((*temporal_candidates, *spatial_candidates))
    cells = tuple(
        LocalSupportCell(
            cell_id=_fixed_cell_id(
                SPATIAL_BASE_CELL_KM,
                draft.clipped.row,
                draft.clipped.column,
            ),
            row=draft.clipped.row,
            column=draft.clipped.column,
            clipped_geometry=draft.clipped.geometry,
            clipped_area_m2=draft.clipped.area_m2,
            status=draft.status,
            source=draft.source,
            base_event_count=draft.base_event_count,
            source_event_count=draft.source_event_count,
            parent_cell_id=draft.parent_cell_id,
            raw_mc=draft.raw_mc,
            candidate_mc=draft.candidate_mc,
            applied_mc=None if draft.status == "unsupported" else common_mc,
        )
        for draft in drafts
    )

    total_area_m2 = math.fsum(cell.clipped_area_m2 for cell in cells)
    complete_partition = cast(
        BaseGeometry,
        normalize_geometry(unary_union([cell.clipped_geometry for cell in cells])),
    )
    area_tolerance_m2 = max(1.0e-6, float(study_area_equal_area.area) * 1.0e-12)
    if not math.isclose(
        total_area_m2,
        float(study_area_equal_area.area),
        rel_tol=1.0e-12,
        abs_tol=1.0e-6,
    ):
        raise ValueError("fixed 500 km cells do not sum to the original study-area denominator")
    if (
        abs(float(complete_partition.area) - total_area_m2) > area_tolerance_m2
        or float(complete_partition.symmetric_difference(study_area_equal_area).area)
        > area_tolerance_m2
    ):
        raise ValueError("fixed 500 km cells do not form a non-overlapping study-area partition")
    retained_area_m2 = math.fsum(
        cell.clipped_area_m2 for cell in cells if cell.status != "unsupported"
    )
    retained_area_fraction = retained_area_m2 / total_area_m2
    unsupported_ids = tuple(cell.cell_id for cell in cells if cell.status == "unsupported")
    if retained_area_fraction < MINIMUM_RETAINED_AREA_FRACTION:
        raise LocalSupportRetentionError(retained_area_fraction, unsupported_ids)

    status_by_key = {(cell.row, cell.column): cell.status for cell in cells}
    retained_magnitudes = tuple(
        event.magnitude
        for event in historical_events
        if status_by_key[base_key_by_event_id[event.event_id]] != "unsupported"
        and event.magnitude >= common_mc - _COMPARISON_TOLERANCE
    )
    if not retained_magnitudes:
        raise CompletenessScientificInability(
            "selected_mc_has_no_events",
            "common completeness magnitude has no events in retained local support",
        )
    retained_b_value = estimate_aki_b_value(retained_magnitudes, mc=common_mc)
    retained_geometry = cast(
        BaseGeometry,
        normalize_geometry(
            unary_union([cell.clipped_geometry for cell in cells if cell.status != "unsupported"])
        ),
    )

    historical_event_sha256 = _historical_event_digest(historical_events)
    study_area_sha256 = hashlib.sha256(_geometry_wkb(study_area_equal_area)).hexdigest()
    temporal_manifest = _temporal_manifest(temporal_blocks)
    cell_manifest = _cell_manifest(cells)
    payload = _manifest_payload(
        fit_end_utc=fit_end_utc,
        historical_event_count=len(historical_events),
        historical_event_sha256=historical_event_sha256,
        study_area_sha256=study_area_sha256,
        common_mc=common_mc,
        retained_selected_event_count=len(retained_magnitudes),
        retained_selected_aki_b_value=retained_b_value,
        total_area_m2=total_area_m2,
        retained_area_m2=retained_area_m2,
        retained_area_fraction=retained_area_fraction,
        temporal_blocks=temporal_manifest,
        cells=cell_manifest,
    )
    return LocalSupportSnapshot(
        support_id=_support_id(payload),
        fit_end_utc=fit_end_utc,
        audit=audit,
        temporal_blocks=temporal_blocks,
        cells=cells,
        common_mc=common_mc,
        retained_selected_event_count=len(retained_magnitudes),
        retained_selected_aki_b_value=retained_b_value,
        total_area_m2=total_area_m2,
        retained_area_m2=retained_area_m2,
        retained_area_fraction=retained_area_fraction,
        retained_geometry=retained_geometry,
        historical_event_sha256=historical_event_sha256,
        study_area_sha256=study_area_sha256,
    )


def build_local_support_manifest(snapshot: LocalSupportSnapshot) -> LocalSupportManifest:
    """Return the immutable target-free and score-free support manifest."""

    if not isinstance(snapshot, LocalSupportSnapshot):
        raise TypeError("snapshot must be a LocalSupportSnapshot")
    temporal_blocks = _temporal_manifest(snapshot.temporal_blocks)
    cells = _cell_manifest(snapshot.cells)
    payload = _manifest_payload(
        fit_end_utc=snapshot.fit_end_utc,
        historical_event_count=snapshot.audit.included_historical_inside_count,
        historical_event_sha256=snapshot.historical_event_sha256,
        study_area_sha256=snapshot.study_area_sha256,
        common_mc=snapshot.common_mc,
        retained_selected_event_count=snapshot.retained_selected_event_count,
        retained_selected_aki_b_value=snapshot.retained_selected_aki_b_value,
        total_area_m2=snapshot.total_area_m2,
        retained_area_m2=snapshot.retained_area_m2,
        retained_area_fraction=snapshot.retained_area_fraction,
        temporal_blocks=temporal_blocks,
        cells=cells,
    )
    if _support_id(payload) != snapshot.support_id:
        raise ValueError("snapshot support_id does not match its causal manifest payload")
    return LocalSupportManifest(
        protocol_version=LOCAL_SUPPORT_PROTOCOL_VERSION,
        support_id=snapshot.support_id,
        fit_end_utc=_utc_text(snapshot.fit_end_utc),
        historical_event_count=snapshot.audit.included_historical_inside_count,
        historical_event_sha256=snapshot.historical_event_sha256,
        study_area_sha256=snapshot.study_area_sha256,
        base_cell_size_km=SPATIAL_BASE_CELL_KM,
        parent_cell_size_km=SPATIAL_PARENT_CELL_KM,
        minimum_events_per_stratum=SPATIAL_MINIMUM_EVENTS,
        maximum_supported_raw_mc=MAXIMUM_SUPPORTED_RAW_MC,
        minimum_retained_area_fraction=MINIMUM_RETAINED_AREA_FRACTION,
        common_mc=snapshot.common_mc,
        retained_selected_event_count=snapshot.retained_selected_event_count,
        retained_selected_aki_b_value=snapshot.retained_selected_aki_b_value,
        total_area_m2=snapshot.total_area_m2,
        retained_area_m2=snapshot.retained_area_m2,
        retained_area_fraction=snapshot.retained_area_fraction,
        temporal_blocks=temporal_blocks,
        cells=cells,
    )


__all__ = [
    "LOCAL_SUPPORT_PROTOCOL_VERSION",
    "MAXIMUM_SUPPORTED_RAW_MC",
    "MINIMUM_RETAINED_AREA_FRACTION",
    "LocalSupportAudit",
    "LocalSupportCell",
    "LocalSupportCellManifest",
    "LocalSupportError",
    "LocalSupportFixedCellIdentity",
    "LocalSupportManifest",
    "LocalSupportRetentionError",
    "LocalSupportSnapshot",
    "LocalSupportStudyAreaIdentity",
    "LocalSupportTemporalManifest",
    "SupportSource",
    "SupportStatus",
    "build_local_support_manifest",
    "build_local_support_snapshot",
    "build_local_support_study_area_identity",
]
