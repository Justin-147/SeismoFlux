"""Fail-closed, score-free inputs for the minimal four-fold S1 runner.

This module authenticates the frozen catalog and study support, materializes
only the four development calendars, and builds catalog histories that were
visible by ``issue - 24 hours``.  It deliberately has no target-construction,
model-fit, or scoring API.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import (
    D1SpatialDomain,
    build_d1_spatial_domain_from_bytes,
)
from seismoflux.multitask_s0 import (
    load_catalog_frame,
    verify_authoritative_catalog_identity,
)
from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
    DevelopmentContractSummary,
    load_development_contract,
)
from seismoflux.multitask_s1.location import CausalSpatialHistory, FrozenSpatialGrid
from seismoflux.stage2s.contracts import FROZEN_GRID_SIZES_KM, SpatialGrid

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

CONTRACT_RELATIVE_PATH: Final = Path("configs/multitask_s1_development_contract.yaml")
CATALOG_RELATIVE_PATH: Final = Path("processed/stage1/debc98054172a4a1/earthquake_event.parquet")
STUDY_AREA_RELATIVE_PATH: Final = Path("processed/china_mainland.geojson")
EXPECTED_STUDY_AREA_SHA256: Final = (
    "5e5dcf012e080882161c95bf592a1ee39a0f0fdad7114bcff58d645aeb30bb02"
)
EXPECTED_25KM_GRID_ID: Final = "3aacdbdda04fed652dd5ee3674906f674c127cb735dea5d5e989527b20809763"
EXPECTED_25KM_CELL_COUNT: Final = 15_697
EXPECTED_TOTAL_AREA_KM2: Final = 9_415_305.754432771
MAIN_CATALOG_DELAY: Final = timedelta(hours=24)
CATALOG_HISTORY_START_UTC: Final = datetime(1969, 12, 31, 16, tzinfo=UTC)
HISTORICAL_M5_START_UTC: Final = datetime(1899, 12, 31, 16, tzinfo=UTC)
SHANGHAI_OFFSET: Final = timezone(timedelta(hours=8))
_UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECONDS_PER_SECOND: Final = 1_000_000
_MICROSECONDS_PER_DAY: Final = 86_400 * _MICROSECONDS_PER_SECOND
_EXPECTED_LEDGER_COLUMNS: Final = (
    "fold_id",
    "role",
    "issue_time_utc",
    "horizon_days",
    "target_interval",
    "target_end_utc",
    "maturity_status",
    "primary_exposure_selected",
)
_MAGNITUDE_PANEL_SPECS: Final = MappingProxyType(
    {
        "m4_plus": (4.0, None, CATALOG_HISTORY_START_UTC),
        "m5_6": (5.0, 6.0, CATALOG_HISTORY_START_UTC),
        "m6_plus": (6.0, None, CATALOG_HISTORY_START_UTC),
        "m5_plus_1970_for_joint": (5.0, None, CATALOG_HISTORY_START_UTC),
        "m5_plus_1900_for_m3": (5.0, None, HISTORICAL_M5_START_UTC),
    }
)


class RunnerInputError(ValueError):
    """Raised before any S1 model can see an unauthenticated or illegal input."""


def _readonly_float(values: object, *, label: str) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RunnerInputError(f"{label} must be a finite one-dimensional vector")
    result.setflags(write=False)
    return result


def _readonly_int(values: object, *, label: str) -> IntArray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if result.ndim != 1:
        raise RunnerInputError(f"{label} must be a one-dimensional integer vector")
    result.setflags(write=False)
    return result


def _readonly_bool(values: object, *, label: str) -> BoolArray:
    result = np.array(values, dtype=np.bool_, copy=True, order="C")
    if result.ndim != 1:
        raise RunnerInputError(f"{label} must be a one-dimensional boolean vector")
    result.setflags(write=False)
    return result


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RunnerInputError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _parse_utc(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RunnerInputError(f"{label} must be a canonical UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RunnerInputError(f"{label} is not a valid UTC timestamp") from exc
    return _aware_utc(parsed, label=label)


def _epoch_us(value: datetime, *, label: str) -> int:
    instant = _aware_utc(value, label=label)
    delta = instant - _UTC_EPOCH
    return (delta.days * 86_400 + delta.seconds) * _MICROSECONDS_PER_SECOND + delta.microseconds


def _resolve_directory(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, str | Path):
        raise RunnerInputError(f"{label} must be supplied explicitly")
    path = Path(value).resolve()
    if not path.is_dir():
        raise RunnerInputError(f"{label} is not an existing directory: {path}")
    return path


def _scoped_file(root: Path, relative_path: Path, *, label: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RunnerInputError(f"{label} escaped its explicit root") from exc
    if not path.is_file():
        raise RunnerInputError(f"{label} is missing: {path}")
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _as_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerInputError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _as_sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise RunnerInputError(f"{label} must be a sequence")
    return value


@dataclass(frozen=True, slots=True)
class CatalogEventTable:
    """Immutable columns from the authenticated catalog; no target membership."""

    event_ids: tuple[str, ...]
    origin_time_us: IntArray
    available_at_us: IntArray
    longitude: FloatArray
    latitude: FloatArray
    magnitude: FloatArray
    inside_study_area: BoolArray

    def __post_init__(self) -> None:
        identifiers = tuple(self.event_ids)
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise RunnerInputError("catalog event IDs must be non-empty and unique")
        origin = _readonly_int(self.origin_time_us, label="origin_time_us")
        available = _readonly_int(self.available_at_us, label="available_at_us")
        longitude = _readonly_float(self.longitude, label="longitude")
        latitude = _readonly_float(self.latitude, label="latitude")
        magnitude = _readonly_float(self.magnitude, label="magnitude")
        inside = _readonly_bool(self.inside_study_area, label="inside_study_area")
        lengths = {
            len(identifiers),
            origin.size,
            available.size,
            longitude.size,
            latitude.size,
            magnitude.size,
            inside.size,
        }
        if len(lengths) != 1 or not identifiers:
            raise RunnerInputError("catalog columns must have one non-zero length")
        if np.any(available < origin):
            raise RunnerInputError("catalog available_at cannot precede origin time")
        if np.any((longitude < -180.0) | (longitude > 180.0)):
            raise RunnerInputError("catalog longitude is outside [-180, 180]")
        if np.any((latitude < -90.0) | (latitude > 90.0)):
            raise RunnerInputError("catalog latitude is outside [-90, 90]")
        ordering = tuple(
            sorted(
                range(len(identifiers)),
                key=lambda index: (int(origin[index]), identifiers[index].encode("utf-8")),
            )
        )
        if ordering != tuple(range(len(identifiers))):
            raise RunnerInputError("catalog rows must remain ordered by origin time and event ID")
        object.__setattr__(self, "event_ids", identifiers)
        object.__setattr__(self, "origin_time_us", origin)
        object.__setattr__(self, "available_at_us", available)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "inside_study_area", inside)

    @property
    def row_count(self) -> int:
        return len(self.event_ids)


def catalog_event_table_from_frame(frame: pd.DataFrame) -> CatalogEventTable:
    """Convert a validated synthetic or authenticated frame to immutable columns."""

    required = {
        "event_id",
        "origin_time_utc",
        "available_at",
        "longitude",
        "latitude",
        "magnitude",
        "inside_study_area",
    }
    if not isinstance(frame, pd.DataFrame) or not required <= set(frame.columns):
        raise RunnerInputError("catalog frame is missing one or more required columns")
    ordered = frame.copy()
    ordered["origin_time_utc"] = pd.to_datetime(
        ordered["origin_time_utc"], utc=True, errors="raise", format="mixed"
    )
    ordered["available_at"] = pd.to_datetime(
        ordered["available_at"], utc=True, errors="raise", format="mixed"
    )
    ordered = ordered.sort_values(
        ["origin_time_utc", "event_id"], kind="mergesort", ignore_index=True
    )
    origin = cast(pd.Series, ordered["origin_time_utc"])
    available = cast(pd.Series, ordered["available_at"])
    origin_ns = origin.astype("datetime64[ns, UTC]").astype("int64").to_numpy(dtype=np.int64)
    available_ns = available.astype("datetime64[ns, UTC]").astype("int64").to_numpy(dtype=np.int64)
    if np.any(origin_ns % 1_000 != 0) or np.any(available_ns % 1_000 != 0):
        raise RunnerInputError("catalog timestamps must be exactly representable in microseconds")
    return CatalogEventTable(
        event_ids=tuple(ordered["event_id"].astype(str).tolist()),
        origin_time_us=origin_ns // 1_000,
        available_at_us=available_ns // 1_000,
        longitude=ordered["longitude"].to_numpy(dtype=np.float64),
        latitude=ordered["latitude"].to_numpy(dtype=np.float64),
        magnitude=ordered["magnitude"].to_numpy(dtype=np.float64),
        inside_study_area=ordered["inside_study_area"].to_numpy(dtype=np.bool_),
    )


@dataclass(frozen=True, slots=True)
class CausalMagnitudeHistory:
    """One magnitude panel visible at the frozen T-minus-24-hour cutoff."""

    magnitude_bin: str
    issue_time_utc: datetime
    data_cutoff_utc: datetime
    event_ids: tuple[str, ...]
    origin_time_us: IntArray
    available_at_us: IntArray
    magnitude: FloatArray
    spatial: CausalSpatialHistory

    @property
    def event_count(self) -> int:
        return len(self.event_ids)


def causal_catalog_histories(
    catalog: CatalogEventTable,
    issue_time: datetime,
) -> Mapping[str, CausalMagnitudeHistory]:
    """Return task-explicit catalog histories visible by ``issue - 24h``.

    This is a pure transformation of an already authenticated table.  It does
    not inspect any future interval and cannot construct an outer-fold target.
    The separate ``m5_plus_1900_for_m3`` panel is only for the frozen long-tail
    M3 sensitivity model; it is never mixed into the 1970+ T0/M0 histories.
    """

    if not isinstance(catalog, CatalogEventTable):
        raise TypeError("catalog must be a CatalogEventTable")
    issue_utc = _aware_utc(issue_time, label="issue_time")
    cutoff_utc = issue_utc - MAIN_CATALOG_DELAY
    cutoff_us = _epoch_us(cutoff_utc, label="data cutoff")
    causally_visible = (
        catalog.inside_study_area
        & (catalog.origin_time_us <= cutoff_us)
        & (catalog.available_at_us <= cutoff_us)
    )
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(EQUAL_AREA_CRS), always_xy=True
    )
    result: dict[str, CausalMagnitudeHistory] = {}
    for name, (minimum, maximum, history_start) in _MAGNITUDE_PANEL_SPECS.items():
        history_start_us = _epoch_us(history_start, label=f"{name} history start")
        selected = (
            causally_visible
            & (catalog.origin_time_us >= history_start_us)
            & (catalog.magnitude >= minimum)
        )
        if maximum is not None:
            selected &= catalog.magnitude < maximum
        indices = np.flatnonzero(selected)
        x_m, y_m = transformer.transform(catalog.longitude[indices], catalog.latitude[indices])
        spatial = CausalSpatialHistory(
            np.asarray(x_m, dtype=np.float64) / 1_000.0,
            np.asarray(y_m, dtype=np.float64) / 1_000.0,
        )
        result[name] = CausalMagnitudeHistory(
            magnitude_bin=name,
            issue_time_utc=issue_utc,
            data_cutoff_utc=cutoff_utc,
            event_ids=tuple(catalog.event_ids[int(index)] for index in indices),
            origin_time_us=_readonly_int(
                catalog.origin_time_us[indices], label=f"{name}.origin_time_us"
            ),
            available_at_us=_readonly_int(
                catalog.available_at_us[indices], label=f"{name}.available_at_us"
            ),
            magnitude=_readonly_float(catalog.magnitude[indices], label=f"{name}.magnitude"),
            spatial=spatial,
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class OuterIssueRow:
    fold_id: str
    issue_time_utc: datetime
    horizon_days: int
    target_end_utc: datetime
    maturity_status: str
    primary_exposure_selected: bool


@dataclass(frozen=True, slots=True)
class InnerExposure:
    fold_id: str
    block_id: str
    horizon_days: int
    issue_times_utc: tuple[datetime, ...]

    @property
    def exposure_count(self) -> int:
        return len(self.issue_times_utc)


def _local_contract_time(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise RunnerInputError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RunnerInputError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(hours=8):
        raise RunnerInputError(f"{label} must use the frozen +08:00 offset")
    if parsed.hour or parsed.minute or parsed.second or parsed.microsecond:
        raise RunnerInputError(f"{label} must be local midnight")
    return parsed


def _thursday_axis(start_local: datetime, end_local: datetime) -> tuple[datetime, ...]:
    if start_local >= end_local:
        raise RunnerInputError("calendar start must precede calendar end")
    candidate = start_local + timedelta(days=(3 - start_local.weekday()) % 7)
    issues: list[datetime] = []
    while candidate < end_local:
        if candidate.weekday() != 3 or candidate.time() != datetime.min.time():
            raise AssertionError("generated issue is not Thursday 00:00 local time")
        issues.append(candidate.astimezone(UTC))
        candidate += timedelta(days=7)
    return tuple(issues)


def _greedy_primary(issues: Sequence[datetime], *, horizon_days: int) -> tuple[datetime, ...]:
    selected: list[datetime] = []
    separation = timedelta(days=horizon_days + 30)
    for issue in issues:
        if not selected or issue >= selected[-1] + separation:
            selected.append(issue)
    return tuple(selected)


def _contract_fold_map(
    contract: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw_folds = _as_sequence(contract.get("outer_folds"), label="outer_folds")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_fold in enumerate(raw_folds):
        fold = _as_mapping(raw_fold, label=f"outer_folds[{index}]")
        fold_id = str(fold.get("id"))
        if fold_id in result:
            raise RunnerInputError("contract outer fold IDs must be unique")
        result[fold_id] = fold
    if tuple(result) != DEVELOPMENT_FOLD_IDS:
        raise RunnerInputError("runner contract must contain only the four development folds")
    return result


def build_inner_development_exposures(
    contract: Mapping[str, Any],
) -> tuple[InnerExposure, ...]:
    """Generate all 12 by 5 inner exposures and match the frozen water levels."""

    fold_map = _contract_fold_map(contract)
    raw_waterlevels = _as_sequence(
        contract.get("inner_block_waterlevels"), label="inner_block_waterlevels"
    )
    waterlevels: dict[tuple[str, str], tuple[int, ...]] = {}
    for index, raw_row in enumerate(raw_waterlevels):
        row = _as_mapping(raw_row, label=f"inner_block_waterlevels[{index}]")
        key = (str(row.get("fold")), str(row.get("block")))
        values = tuple(
            int(value) for value in _as_sequence(row.get("exposures"), label=f"{key}.exposures")
        )
        if key in waterlevels or len(values) != len(HORIZONS_DAYS):
            raise RunnerInputError("inner water-level exposure rows changed")
        waterlevels[key] = values

    generated: list[InnerExposure] = []
    seen_blocks: set[tuple[str, str]] = set()
    for fold_id in DEVELOPMENT_FOLD_IDS:
        fold = fold_map[fold_id]
        blocks = _as_sequence(fold.get("inner_blocks"), label=f"{fold_id}.inner_blocks")
        for raw_block in blocks:
            block = _as_mapping(raw_block, label=f"{fold_id}.inner_block")
            block_id = str(block.get("id"))
            key = (fold_id, block_id)
            if key in seen_blocks or key not in waterlevels:
                raise RunnerInputError("inner block identity is duplicated or missing")
            seen_blocks.add(key)
            start = _local_contract_time(block.get("start"), label=f"{key}.start")
            end = _local_contract_time(block.get("end"), label=f"{key}.end")
            weekly = _thursday_axis(start, end)
            for position, horizon in enumerate(HORIZONS_DAYS):
                mature = tuple(issue for issue in weekly if issue + timedelta(days=horizon) <= end)
                selected = _greedy_primary(mature, horizon_days=horizon)
                if len(selected) != waterlevels[key][position]:
                    raise RunnerInputError(
                        f"generated inner exposure count differs from frozen water level: "
                        f"{fold_id}.{block_id}.{horizon}d"
                    )
                generated.append(
                    InnerExposure(
                        fold_id=fold_id,
                        block_id=block_id,
                        horizon_days=horizon,
                        issue_times_utc=selected,
                    )
                )
    if seen_blocks != set(waterlevels) or len(generated) != 12 * len(HORIZONS_DAYS):
        raise RunnerInputError("inner exposure coverage must remain exactly 12 blocks by 5")
    return tuple(generated)


def load_outer_development_issues(
    path: str | Path,
    contract: Mapping[str, Any],
    *,
    requested_fold_ids: Sequence[str] = DEVELOPMENT_FOLD_IDS,
) -> tuple[OuterIssueRow, ...]:
    """Parse the S0 issue ledger while making holdout/audit rows unreachable."""

    requested = tuple(requested_fold_ids)
    if requested != DEVELOPMENT_FOLD_IDS:
        raise RunnerInputError("S1 runner accepts exactly the four frozen development folds")
    fold_map = _contract_fold_map(contract)
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise RunnerInputError(f"issue ledger is missing: {ledger_path}")
    selected_rows: list[OuterIssueRow] = []
    seen_keys: set[tuple[str, int, datetime]] = set()
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _EXPECTED_LEDGER_COLUMNS:
            raise RunnerInputError("issue ledger columns or order changed")
        for row_index, ledger_record in enumerate(reader, start=2):
            fold_id = ledger_record["fold_id"]
            role = ledger_record["role"]
            if fold_id not in DEVELOPMENT_FOLD_IDS:
                if role == "development":
                    raise RunnerInputError("issue ledger introduced an unapproved development fold")
                continue
            if role != "development":
                raise RunnerInputError("a frozen development fold changed to holdout or audit")
            try:
                horizon = int(ledger_record["horizon_days"])
            except ValueError as exc:
                raise RunnerInputError(f"invalid horizon at issue-ledger row {row_index}") from exc
            if horizon not in HORIZONS_DAYS or ledger_record["target_interval"] != "(T,T+h]":
                raise RunnerInputError("issue ledger target semantics changed")
            issue = _parse_utc(ledger_record["issue_time_utc"], label=f"row {row_index} issue")
            target_end = _parse_utc(
                ledger_record["target_end_utc"], label=f"row {row_index} target end"
            )
            if target_end != issue + timedelta(days=horizon):
                raise RunnerInputError("issue ledger target end differs from issue plus horizon")
            local_issue = issue.astimezone(SHANGHAI_OFFSET)
            if local_issue.weekday() != 3 or local_issue.time() != datetime.min.time():
                raise RunnerInputError("outer issue is not Thursday 00:00 Asia/Shanghai")
            maturity = ledger_record["maturity_status"]
            if maturity not in {"mature", "unavailable_not_mature"}:
                raise RunnerInputError("issue ledger maturity status changed")
            primary_text = ledger_record["primary_exposure_selected"]
            if primary_text not in {"True", "False"}:
                raise RunnerInputError("issue ledger primary-exposure flag is not canonical")
            key = (fold_id, horizon, issue)
            if key in seen_keys:
                raise RunnerInputError("issue ledger contains a duplicate development row")
            seen_keys.add(key)
            selected_rows.append(
                OuterIssueRow(
                    fold_id=fold_id,
                    issue_time_utc=issue,
                    horizon_days=horizon,
                    target_end_utc=target_end,
                    maturity_status=maturity,
                    primary_exposure_selected=primary_text == "True",
                )
            )

    order = {fold_id: index for index, fold_id in enumerate(DEVELOPMENT_FOLD_IDS)}
    selected_rows.sort(
        key=lambda item: (
            order[item.fold_id],
            HORIZONS_DAYS.index(item.horizon_days),
            item.issue_time_utc,
        )
    )
    for fold_id in DEVELOPMENT_FOLD_IDS:
        fold = fold_map[fold_id]
        outer_start = _local_contract_time(fold.get("outer_start"), label=f"{fold_id}.start")
        outer_end = _local_contract_time(fold.get("outer_end"), label=f"{fold_id}.end")
        expected_weekly = _thursday_axis(outer_start, outer_end)
        for horizon in HORIZONS_DAYS:
            rows = [
                row
                for row in selected_rows
                if row.fold_id == fold_id and row.horizon_days == horizon
            ]
            if tuple(row.issue_time_utc for row in rows) != expected_weekly:
                raise RunnerInputError("issue ledger does not contain the exact frozen weekly axis")
            mature = tuple(
                row.issue_time_utc
                for row in rows
                if row.issue_time_utc + timedelta(days=horizon) <= outer_end
            )
            primary = set(_greedy_primary(mature, horizon_days=horizon))
            for outer_row in rows:
                expected_maturity = (
                    "mature" if outer_row.issue_time_utc in mature else "unavailable_not_mature"
                )
                if outer_row.maturity_status != expected_maturity:
                    raise RunnerInputError("issue ledger maturity disagrees with the frozen fold")
                if outer_row.primary_exposure_selected != (outer_row.issue_time_utc in primary):
                    raise RunnerInputError("issue ledger primary exposure disagrees with h+30d")
    return tuple(selected_rows)


def adapt_frozen_spatial_grid(grid: SpatialGrid) -> FrozenSpatialGrid:
    """Adapt the authenticated 25 km quadrature to the pure S1 location API."""

    if not isinstance(grid, SpatialGrid) or grid.cell_size_km != 25.0:
        raise RunnerInputError("S1 location models require the frozen 25 km grid")
    return FrozenSpatialGrid(
        x_km=grid.query_xy_km[:, 0],
        y_km=grid.query_xy_km[:, 1],
        area_km2=grid.clipped_area_km2,
    )


def load_verified_spatial_inputs(
    data_root: str | Path,
) -> tuple[D1SpatialDomain, FrozenSpatialGrid, str]:
    """Authenticate the study bytes and rebuild the exact 50/25/12.5 km domain."""

    root = _resolve_directory(data_root, label="data_root")
    path = _scoped_file(root, STUDY_AREA_RELATIVE_PATH, label="study area")
    payload = path.read_bytes()
    observed_hash = _sha256_bytes(payload)
    if observed_hash != EXPECTED_STUDY_AREA_SHA256:
        raise RunnerInputError("authoritative china_mainland.geojson SHA-256 mismatch")
    domain = build_d1_spatial_domain_from_bytes(payload)
    family = domain.quadrature_family
    if tuple(grid.cell_size_km for grid in family.grids) != FROZEN_GRID_SIZES_KM:
        raise RunnerInputError("D1 did not rebuild the exact 50/25/12.5 km grid family")
    grid = domain.operational_grid
    if grid.grid_id != EXPECTED_25KM_GRID_ID or domain.stage3_grid.grid_id != EXPECTED_25KM_GRID_ID:
        raise RunnerInputError("frozen 25 km grid_id changed")
    if grid.cell_count != EXPECTED_25KM_CELL_COUNT:
        raise RunnerInputError("frozen 25 km cell count changed")
    for member in family.grids:
        total = math.fsum(float(value) for value in member.clipped_area_km2)
        if not math.isclose(total, EXPECTED_TOTAL_AREA_KM2, rel_tol=0.0, abs_tol=1.0e-9):
            raise RunnerInputError("frozen national grid area changed")
    frozen = adapt_frozen_spatial_grid(grid)
    if not math.isclose(
        frozen.total_area_km2,
        EXPECTED_TOTAL_AREA_KM2,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RunnerInputError("FrozenSpatialGrid area differs from the D1 grid")
    return domain, frozen, observed_hash


def load_verified_catalog_inputs(
    data_root: str | Path,
) -> tuple[CatalogEventTable, Mapping[str, object]]:
    """Authenticate and load only the frozen catalog allowlist."""

    root = _resolve_directory(data_root, label="data_root")
    path = _scoped_file(root, CATALOG_RELATIVE_PATH, label="earthquake catalog")
    try:
        identity = verify_authoritative_catalog_identity(path)
        frame = load_catalog_frame(path)
    except (OSError, ValueError) as exc:
        raise RunnerInputError("authoritative earthquake catalog verification failed") from exc
    table = catalog_event_table_from_frame(frame)
    if table.row_count != 40_898:
        raise RunnerInputError("authenticated catalog row count changed after allowlist loading")
    return table, MappingProxyType(dict(identity))


@dataclass(frozen=True, slots=True)
class S1RunnerInputs:
    """Authenticated, score-free input state for the next minimal runner layer."""

    project_root: Path
    data_root: Path
    contract: Mapping[str, Any]
    contract_summary: DevelopmentContractSummary
    catalog: CatalogEventTable
    catalog_identity: Mapping[str, object]
    study_area_sha256: str
    spatial_domain: D1SpatialDomain
    location_grid: FrozenSpatialGrid
    outer_issues: tuple[OuterIssueRow, ...]
    inner_exposures: tuple[InnerExposure, ...]


def load_s1_runner_inputs(*, project_root: str | Path, data_root: str | Path) -> S1RunnerInputs:
    """Load the complete no-score S1 input state from two explicit roots.

    ``project_root`` authenticates the frozen contract and S0 ledgers.  The
    independently supplied ``data_root`` points at the main repository's
    authoritative raw/processed data tree; neither root is inferred from CWD.
    """

    project = _resolve_directory(project_root, label="project_root")
    data = _resolve_directory(data_root, label="data_root")
    contract_path = _scoped_file(project, CONTRACT_RELATIVE_PATH, label="S1 contract")
    contract, summary = load_development_contract(contract_path, project_root=project)
    catalog, catalog_identity = load_verified_catalog_inputs(data)
    domain, location_grid, study_hash = load_verified_spatial_inputs(data)
    sources = _as_mapping(contract.get("source_identities"), label="source_identities")
    issue_source = _as_mapping(sources.get("issue_maturity_ledger"), label="issue_maturity_ledger")
    issue_relative = Path(str(issue_source.get("path")))
    issue_path = _scoped_file(project, issue_relative, label="issue maturity ledger")
    outer_issues = load_outer_development_issues(issue_path, contract)
    inner_exposures = build_inner_development_exposures(contract)
    return S1RunnerInputs(
        project_root=project,
        data_root=data,
        contract=contract,
        contract_summary=summary,
        catalog=catalog,
        catalog_identity=catalog_identity,
        study_area_sha256=study_hash,
        spatial_domain=domain,
        location_grid=location_grid,
        outer_issues=outer_issues,
        inner_exposures=inner_exposures,
    )


__all__ = [
    "CATALOG_HISTORY_START_UTC",
    "CATALOG_RELATIVE_PATH",
    "CONTRACT_RELATIVE_PATH",
    "EXPECTED_25KM_CELL_COUNT",
    "EXPECTED_25KM_GRID_ID",
    "EXPECTED_STUDY_AREA_SHA256",
    "EXPECTED_TOTAL_AREA_KM2",
    "HISTORICAL_M5_START_UTC",
    "MAIN_CATALOG_DELAY",
    "STUDY_AREA_RELATIVE_PATH",
    "CatalogEventTable",
    "CausalMagnitudeHistory",
    "InnerExposure",
    "OuterIssueRow",
    "RunnerInputError",
    "S1RunnerInputs",
    "adapt_frozen_spatial_grid",
    "build_inner_development_exposures",
    "catalog_event_table_from_frame",
    "causal_catalog_histories",
    "load_outer_development_issues",
    "load_s1_runner_inputs",
    "load_verified_catalog_inputs",
    "load_verified_spatial_inputs",
]
