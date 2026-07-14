"""Deterministic catalog-completeness diagnostics for stage 2.

The implementation is deliberately independent of file formats and model
scores.  Callers provide already-normalized physical earthquake events in the
frozen equal-area coordinate system and an explicit historical cutoff.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise
from typing import Literal

from seismoflux.background.etas import aki_b_value as _aki_b_value

MAGNITUDE_BIN_WIDTH = 0.1
MAXIMUM_CURVATURE_CORRECTION = 0.2
COMPLETENESS_CANDIDATES = (3.0, 3.2, 3.5, 4.0)
CATALOG_ANCHOR_UTC = datetime(1970, 1, 1, tzinfo=UTC)
TEMPORAL_BLOCK_YEARS = 5
TEMPORAL_MINIMUM_EVENTS = 200
FINAL_PARTIAL_BLOCK_MINIMUM_YEARS = 3
SPATIAL_BASE_CELL_KM = 500.0
SPATIAL_PARENT_CELL_KM = 1000.0
SPATIAL_MINIMUM_EVENTS = 200
REGIME_MC_DIFFERENCE = 0.3
REGIME_ANNUAL_COUNT_RATIO = 2.0

_SECONDS_PER_TROPICAL_YEAR = 365.2425 * 24.0 * 60.0 * 60.0
_COMPARISON_TOLERANCE = 1.0e-12
_BIN_WIDTH_DECIMAL = Decimal("0.1")
_CORRECTION_DECIMAL = Decimal("0.2")


class CompletenessError(RuntimeError):
    """Raised when a frozen completeness gate cannot be satisfied."""


CompletenessInabilityCode = Literal[
    "estimate_above_frozen_maximum",
    "no_eligible_spatial_stratum",
    "no_eligible_temporal_stratum",
    "selected_mc_has_no_events",
]


class CompletenessScientificInability(CompletenessError):
    """Expected catalog limitation that makes the frozen completeness gate unevaluable."""

    def __init__(self, reason_code: CompletenessInabilityCode, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class CompletenessEvent:
    """One deduplicated earthquake with frozen availability and coordinates."""

    event_id: str
    origin_time_utc: datetime
    available_at: datetime
    magnitude: float
    inside_study_area: bool
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        _require_utc("origin_time_utc", self.origin_time_utc)
        _require_utc("available_at", self.available_at)
        if self.available_at < self.origin_time_utc:
            raise ValueError("available_at must not precede origin_time_utc")
        if not math.isfinite(self.magnitude) or self.magnitude < 0.0:
            raise ValueError("magnitude must be finite and non-negative")
        if type(self.inside_study_area) is not bool:
            raise TypeError("inside_study_area must be a bool")
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("equal-area coordinates must be finite")


@dataclass(frozen=True, slots=True)
class MagnitudeBin:
    """One occupied 0.1-magnitude histogram bin."""

    magnitude: float
    count: int


@dataclass(frozen=True, slots=True)
class MaximumCurvatureEstimate:
    """MAXC histogram peak and its fixed +0.2 correction."""

    event_count: int
    peak_bin_magnitude: float
    corrected_magnitude: float
    histogram: tuple[MagnitudeBin, ...]


@dataclass(frozen=True, slots=True)
class StratumCompletenessEstimate:
    """Completeness and Aki b-value for one eligible diagnostic stratum."""

    maxc: MaximumCurvatureEstimate
    candidate_mc: float
    complete_event_count: int
    aki_b_value: float | None


@dataclass(frozen=True, slots=True)
class TemporalBlockDiagnostic:
    """One fixed five-year block, including a possible final partial block."""

    block_id: str
    start_utc: datetime
    end_utc: datetime
    duration_years: float
    is_partial: bool
    event_count: int
    eligible: bool
    estimate: StratumCompletenessEstimate | None


SpatialStratumSource = Literal["base_500km", "sparse_parent_1000km"]


@dataclass(frozen=True, slots=True)
class SpatialStratumDiagnostic:
    """One unique fixed-grid stratum used by the spatial diagnostic."""

    stratum_id: str
    source: SpatialStratumSource
    cell_size_km: float
    row: int
    column: int
    event_count: int
    eligible: bool
    indeterminate: bool
    estimate: StratumCompletenessEstimate | None
    applied_mc: float | None


@dataclass(frozen=True, slots=True)
class SparseCellResolution:
    """Mapping from a sparse 500 km cell to its fixed 1000 km parent."""

    base_row: int
    base_column: int
    base_event_count: int
    parent_row: int
    parent_column: int
    parent_event_count: int
    parent_eligible: bool
    indeterminate: bool
    applied_mc: float | None


@dataclass(frozen=True, slots=True)
class RegimeChangeDiagnostic:
    """Non-causal flag comparing two adjacent fixed temporal blocks."""

    earlier_block_id: str
    later_block_id: str
    mc_difference: float | None
    annual_count_ratio: float
    mc_threshold_reached: bool
    count_ratio_threshold_reached: bool
    flagged: bool


@dataclass(frozen=True, slots=True)
class MagnitudeSensitivity:
    """Global event count and Aki b-value at one frozen candidate threshold."""

    magnitude_threshold: float
    event_count: int
    aki_b_value: float | None


@dataclass(frozen=True, slots=True)
class CompletenessAudit:
    """Mutually exclusive input-filter counts for leakage auditing."""

    input_event_count: int
    included_historical_inside_count: int
    excluded_outside_count: int
    excluded_pre_1970_count: int
    excluded_future_origin_count: int
    excluded_unavailable_count: int


@dataclass(frozen=True, slots=True)
class CompletenessAnalysis:
    """Complete frozen MAXC, stratification, diagnostic, and sensitivity result."""

    cutoff_utc: datetime
    audit: CompletenessAudit
    temporal_blocks: tuple[TemporalBlockDiagnostic, ...]
    spatial_strata: tuple[SpatialStratumDiagnostic, ...]
    sparse_cell_resolutions: tuple[SparseCellResolution, ...]
    regime_changes: tuple[RegimeChangeDiagnostic, ...]
    maximum_eligible_estimate: float
    selected_mc: float
    selected_event_count: int
    selected_aki_b_value: float
    sensitivities: tuple[MagnitudeSensitivity, ...]


def maximum_curvature_estimate(magnitudes: Iterable[float]) -> MaximumCurvatureEstimate:
    """Estimate MAXC using fixed 0.1 bins and the frozen +0.2 correction.

    Exact count ties use the higher magnitude bin, a deterministic conservative
    rule.  The returned histogram contains occupied bins in ascending order.
    """

    counts: Counter[int] = Counter()
    event_count = 0
    for magnitude in magnitudes:
        value = float(magnitude)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("magnitudes must be finite and non-negative")
        decimal_value = Decimal(str(value))
        index = int((decimal_value / _BIN_WIDTH_DECIMAL).to_integral_value(rounding=ROUND_HALF_UP))
        counts[index] += 1
        event_count += 1
    if not counts:
        raise ValueError("at least one magnitude is required for MAXC")

    peak_index = max(counts, key=lambda index: (counts[index], index))
    peak_decimal = Decimal(peak_index) * _BIN_WIDTH_DECIMAL
    histogram = tuple(
        MagnitudeBin(
            magnitude=float(Decimal(index) * _BIN_WIDTH_DECIMAL),
            count=counts[index],
        )
        for index in sorted(counts)
    )
    return MaximumCurvatureEstimate(
        event_count=event_count,
        peak_bin_magnitude=float(peak_decimal),
        corrected_magnitude=float(peak_decimal + _CORRECTION_DECIMAL),
        histogram=histogram,
    )


def select_candidate_magnitude(estimate: float) -> float:
    """Round a raw MAXC estimate upward to the frozen candidate set."""

    value = float(estimate)
    if not math.isfinite(value):
        raise ValueError("completeness estimate must be finite")
    for candidate in COMPLETENESS_CANDIDATES:
        if value <= candidate + _COMPARISON_TOLERANCE:
            return candidate
    raise CompletenessScientificInability(
        "estimate_above_frozen_maximum",
        f"completeness estimate {value:.12g} exceeds frozen maximum candidate 4.0",
    )


def estimate_aki_b_value(magnitudes: Iterable[float], *, mc: float) -> float:
    """Return the fixed-bin Aki (1965) b-value at ``mc``."""

    return _aki_b_value(magnitudes, mc=mc, bin_width=MAGNITUDE_BIN_WIDTH)


def _stratum_estimate(
    events: Sequence[CompletenessEvent], *, stratum_id: str
) -> StratumCompletenessEstimate:
    maxc = maximum_curvature_estimate(event.magnitude for event in events)
    try:
        candidate = select_candidate_magnitude(maxc.corrected_magnitude)
    except CompletenessScientificInability as error:
        raise CompletenessScientificInability(
            error.reason_code,
            f"{stratum_id}: {error}",
        ) from error
    complete_magnitudes = tuple(
        event.magnitude for event in events if event.magnitude >= candidate - _COMPARISON_TOLERANCE
    )
    b_value = (
        estimate_aki_b_value(complete_magnitudes, mc=candidate) if complete_magnitudes else None
    )
    return StratumCompletenessEstimate(
        maxc=maxc,
        candidate_mc=candidate,
        complete_event_count=len(complete_magnitudes),
        aki_b_value=b_value,
    )


def _filter_events(
    events: Iterable[CompletenessEvent], cutoff_utc: datetime
) -> tuple[tuple[CompletenessEvent, ...], CompletenessAudit]:
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
        elif event.origin_time_utc > cutoff_utc:
            future_origin += 1
        elif event.available_at > cutoff_utc:
            unavailable += 1
        else:
            included.append(event)

    included.sort(key=lambda event: (event.origin_time_utc, event.available_at, event.event_id))
    audit = CompletenessAudit(
        input_event_count=len(supplied),
        included_historical_inside_count=len(included),
        excluded_outside_count=outside,
        excluded_pre_1970_count=pre_1970,
        excluded_future_origin_count=future_origin,
        excluded_unavailable_count=unavailable,
    )
    return tuple(included), audit


def _add_years(value: datetime, years: int) -> datetime:
    return value.replace(year=value.year + years)


def _build_temporal_blocks(
    events: Sequence[CompletenessEvent], cutoff_utc: datetime
) -> tuple[TemporalBlockDiagnostic, ...]:
    blocks: list[TemporalBlockDiagnostic] = []
    start = CATALOG_ANCHOR_UTC
    while start < cutoff_utc:
        nominal_end = _add_years(start, TEMPORAL_BLOCK_YEARS)
        is_partial = nominal_end > cutoff_utc
        end = cutoff_utc if is_partial else nominal_end
        is_final = end == cutoff_utc
        block_events = tuple(
            event
            for event in events
            if event.origin_time_utc >= start
            and (event.origin_time_utc <= end if is_final else event.origin_time_utc < end)
        )
        duration_years = (end - start).total_seconds() / _SECONDS_PER_TROPICAL_YEAR
        duration_eligible = not is_partial or cutoff_utc >= _add_years(
            start, FINAL_PARTIAL_BLOCK_MINIMUM_YEARS
        )
        eligible = len(block_events) >= TEMPORAL_MINIMUM_EVENTS and duration_eligible
        block_id = (
            f"{start.year:04d}-partial-{cutoff_utc.date().isoformat()}"
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


def _cell_index(coordinate_m: float, cell_km: float) -> int:
    return math.floor(coordinate_m / (cell_km * 1000.0))


def _spatial_id(cell_km: float, row: int, column: int) -> str:
    return f"{int(cell_km)}km/r{row:+d}/c{column:+d}"


def _build_spatial_diagnostics(
    events: Sequence[CompletenessEvent],
) -> tuple[tuple[SpatialStratumDiagnostic, ...], tuple[SparseCellResolution, ...]]:
    base_events: dict[tuple[int, int], list[CompletenessEvent]] = defaultdict(list)
    parent_events: dict[tuple[int, int], list[CompletenessEvent]] = defaultdict(list)
    for event in events:
        base_key = (
            _cell_index(event.y_m, SPATIAL_BASE_CELL_KM),
            _cell_index(event.x_m, SPATIAL_BASE_CELL_KM),
        )
        parent_key = (
            _cell_index(event.y_m, SPATIAL_PARENT_CELL_KM),
            _cell_index(event.x_m, SPATIAL_PARENT_CELL_KM),
        )
        base_events[base_key].append(event)
        parent_events[parent_key].append(event)

    diagnostics: list[SpatialStratumDiagnostic] = []
    sparse_parent_keys: set[tuple[int, int]] = set()
    sparse_base_keys: list[tuple[int, int]] = []
    for (row, column), cell_events in sorted(base_events.items()):
        if len(cell_events) >= SPATIAL_MINIMUM_EVENTS:
            stratum_id = _spatial_id(SPATIAL_BASE_CELL_KM, row, column)
            estimate = _stratum_estimate(cell_events, stratum_id=f"spatial/{stratum_id}")
            diagnostics.append(
                SpatialStratumDiagnostic(
                    stratum_id=stratum_id,
                    source="base_500km",
                    cell_size_km=SPATIAL_BASE_CELL_KM,
                    row=row,
                    column=column,
                    event_count=len(cell_events),
                    eligible=True,
                    indeterminate=False,
                    estimate=estimate,
                    applied_mc=None,
                )
            )
        else:
            sparse_base_keys.append((row, column))
            # Both grids share the zero origin, so floor division preserves
            # correct parentage for negative as well as positive indices.
            sparse_parent_keys.add((row // 2, column // 2))

    parent_lookup: dict[tuple[int, int], SpatialStratumDiagnostic] = {}
    for row, column in sorted(sparse_parent_keys):
        cell_events = parent_events.get((row, column), [])
        eligible = len(cell_events) >= SPATIAL_MINIMUM_EVENTS
        stratum_id = _spatial_id(SPATIAL_PARENT_CELL_KM, row, column)
        parent_estimate = (
            _stratum_estimate(cell_events, stratum_id=f"spatial/{stratum_id}") if eligible else None
        )
        diagnostic = SpatialStratumDiagnostic(
            stratum_id=stratum_id,
            source="sparse_parent_1000km",
            cell_size_km=SPATIAL_PARENT_CELL_KM,
            row=row,
            column=column,
            event_count=len(cell_events),
            eligible=eligible,
            indeterminate=not eligible,
            estimate=parent_estimate,
            applied_mc=None,
        )
        diagnostics.append(diagnostic)
        parent_lookup[(row, column)] = diagnostic

    resolutions: list[SparseCellResolution] = []
    for row, column in sorted(sparse_base_keys):
        parent_key = (row // 2, column // 2)
        parent = parent_lookup[parent_key]
        resolutions.append(
            SparseCellResolution(
                base_row=row,
                base_column=column,
                base_event_count=len(base_events[(row, column)]),
                parent_row=parent.row,
                parent_column=parent.column,
                parent_event_count=parent.event_count,
                parent_eligible=parent.eligible,
                indeterminate=parent.indeterminate,
                applied_mc=None,
            )
        )

    diagnostics.sort(key=lambda value: (value.cell_size_km, value.row, value.column))
    return tuple(diagnostics), tuple(resolutions)


def _annual_count_ratio(earlier: TemporalBlockDiagnostic, later: TemporalBlockDiagnostic) -> float:
    earlier_rate = earlier.event_count / earlier.duration_years
    later_rate = later.event_count / later.duration_years
    low = min(earlier_rate, later_rate)
    high = max(earlier_rate, later_rate)
    if high == 0.0:
        return 1.0
    if low == 0.0:
        return math.inf
    return high / low


def _regime_change_diagnostics(
    blocks: Sequence[TemporalBlockDiagnostic],
) -> tuple[RegimeChangeDiagnostic, ...]:
    diagnostics: list[RegimeChangeDiagnostic] = []
    for earlier, later in pairwise(blocks):
        if earlier.estimate is not None and later.estimate is not None:
            mc_difference: float | None = abs(
                later.estimate.maxc.corrected_magnitude - earlier.estimate.maxc.corrected_magnitude
            )
        else:
            mc_difference = None
        count_ratio = _annual_count_ratio(earlier, later)
        mc_threshold_reached = (
            mc_difference is not None
            and mc_difference >= REGIME_MC_DIFFERENCE - _COMPARISON_TOLERANCE
        )
        count_threshold_reached = count_ratio >= REGIME_ANNUAL_COUNT_RATIO - _COMPARISON_TOLERANCE
        diagnostics.append(
            RegimeChangeDiagnostic(
                earlier_block_id=earlier.block_id,
                later_block_id=later.block_id,
                mc_difference=mc_difference,
                annual_count_ratio=count_ratio,
                mc_threshold_reached=mc_threshold_reached,
                count_ratio_threshold_reached=count_threshold_reached,
                flagged=mc_threshold_reached or count_threshold_reached,
            )
        )
    return tuple(diagnostics)


def _magnitude_sensitivities(
    events: Sequence[CompletenessEvent],
) -> tuple[MagnitudeSensitivity, ...]:
    results: list[MagnitudeSensitivity] = []
    for threshold in COMPLETENESS_CANDIDATES:
        magnitudes = tuple(
            event.magnitude
            for event in events
            if event.magnitude >= threshold - _COMPARISON_TOLERANCE
        )
        b_value = estimate_aki_b_value(magnitudes, mc=threshold) if magnitudes else None
        results.append(
            MagnitudeSensitivity(
                magnitude_threshold=threshold,
                event_count=len(magnitudes),
                aki_b_value=b_value,
            )
        )
    return tuple(results)


def analyze_catalog_completeness(
    events: Iterable[CompletenessEvent], *, cutoff_utc: datetime
) -> CompletenessAnalysis:
    """Run the frozen stage-2 completeness analysis without reading files.

    Only inside-study-area events known by ``cutoff_utc`` and originating from
    1970 onward are eligible.  At least one temporal and one spatial stratum
    must be estimable.  Any eligible MAXC estimate above 4.0 is a hard failure.
    """

    _require_utc("cutoff_utc", cutoff_utc)
    if cutoff_utc <= CATALOG_ANCHOR_UTC:
        raise CompletenessError("cutoff_utc must be after the 1970 catalog anchor")

    historical_events, audit = _filter_events(events, cutoff_utc)
    temporal_blocks = _build_temporal_blocks(historical_events, cutoff_utc)
    eligible_temporal = tuple(block for block in temporal_blocks if block.eligible)
    if not eligible_temporal:
        raise CompletenessScientificInability(
            "no_eligible_temporal_stratum",
            "no eligible temporal completeness stratum",
        )

    spatial_strata, sparse_resolutions = _build_spatial_diagnostics(historical_events)
    eligible_spatial = tuple(stratum for stratum in spatial_strata if stratum.eligible)
    if not eligible_spatial:
        raise CompletenessScientificInability(
            "no_eligible_spatial_stratum",
            "no eligible spatial completeness stratum",
        )

    raw_estimates = tuple(
        block.estimate.maxc.corrected_magnitude
        for block in eligible_temporal
        if block.estimate is not None
    ) + tuple(
        stratum.estimate.maxc.corrected_magnitude
        for stratum in eligible_spatial
        if stratum.estimate is not None
    )
    maximum_estimate = max(raw_estimates)
    selected_mc = select_candidate_magnitude(maximum_estimate)
    sensitivities = _magnitude_sensitivities(historical_events)
    selected_sensitivity = next(
        value
        for value in sensitivities
        if math.isclose(
            value.magnitude_threshold,
            selected_mc,
            rel_tol=0.0,
            abs_tol=_COMPARISON_TOLERANCE,
        )
    )
    if selected_sensitivity.aki_b_value is None:
        raise CompletenessScientificInability(
            "selected_mc_has_no_events",
            "selected completeness magnitude has no complete events",
        )

    final_spatial = tuple(
        replace(
            stratum,
            applied_mc=(
                stratum.estimate.candidate_mc if stratum.estimate is not None else selected_mc
            ),
        )
        for stratum in spatial_strata
    )
    parent_mc = {
        (stratum.row, stratum.column): stratum.applied_mc
        for stratum in final_spatial
        if stratum.source == "sparse_parent_1000km"
    }
    final_resolutions = tuple(
        replace(
            resolution,
            applied_mc=parent_mc[(resolution.parent_row, resolution.parent_column)],
        )
        for resolution in sparse_resolutions
    )

    return CompletenessAnalysis(
        cutoff_utc=cutoff_utc,
        audit=audit,
        temporal_blocks=temporal_blocks,
        spatial_strata=final_spatial,
        sparse_cell_resolutions=final_resolutions,
        regime_changes=_regime_change_diagnostics(temporal_blocks),
        maximum_eligible_estimate=maximum_estimate,
        selected_mc=selected_mc,
        selected_event_count=selected_sensitivity.event_count,
        selected_aki_b_value=selected_sensitivity.aki_b_value,
        sensitivities=sensitivities,
    )


__all__ = [
    "CATALOG_ANCHOR_UTC",
    "COMPLETENESS_CANDIDATES",
    "CompletenessAnalysis",
    "CompletenessAudit",
    "CompletenessError",
    "CompletenessEvent",
    "CompletenessInabilityCode",
    "CompletenessScientificInability",
    "MagnitudeBin",
    "MagnitudeSensitivity",
    "MaximumCurvatureEstimate",
    "RegimeChangeDiagnostic",
    "SparseCellResolution",
    "SpatialStratumDiagnostic",
    "StratumCompletenessEstimate",
    "TemporalBlockDiagnostic",
    "analyze_catalog_completeness",
    "estimate_aki_b_value",
    "maximum_curvature_estimate",
    "select_candidate_magnitude",
]
