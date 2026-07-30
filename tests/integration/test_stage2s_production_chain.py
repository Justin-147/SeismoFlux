# ruff: noqa: E402
from __future__ import annotations

import builtins
import glob as glob_module
import hashlib
import io
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, cast

_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_variable in _THREAD_ENVIRONMENT_VARIABLES:
    os.environ[_thread_variable] = "1"

import numpy as np
import pyarrow as pa
import pytest
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely.geometry import box

import seismoflux.stage2s.production as production
from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    EqualAreaGridFamily,
    build_equal_area_grid_family,
)
from seismoflux.background.local_support import (
    LocalSupportSnapshot,
    build_local_support_snapshot,
)
from seismoflux.background.local_support_manifest import BackgroundLocalSupportManifest
from seismoflux.stage2s.calendar import (
    AssessmentIssue,
    FoldCalendar,
    Stage2SFoldCalendar,
    TargetExposure,
)
from seismoflux.stage2s.catalog import (
    CatalogIdentity,
    Stage2SEarthquakeCatalog,
)
from seismoflux.stage2s.inputs import (
    NonTargetPreflight,
    NonTargetSpatialAdapter,
    Stage2SQueryGrid,
)
from seismoflux.stage2s.production import (
    FormalPreflightContext,
    FormalScienceInputs,
    run_formal_science,
)
from seismoflux.stage2s.protocol import Stage2SProtocolBundle
from seismoflux.stage2s.rendering import ALL_ARTIFACT_NAMES, PROTOCOL_ARTIFACT_NAMES
from seismoflux.stage2s.seals import Stage2SSealExists, write_o_excl_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BEIJING_OFFSET = timezone(timedelta(hours=8))
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_HORIZONS = (7, 30, 90)
_ASSESSMENT_TARGET_OFFSETS_DAYS = (1, 3, 6, 10, 20, 29, 45, 60, 80)

_PathValue: TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
_OpenPathValue: TypeAlias = int | _PathValue


@dataclass(frozen=True, slots=True)
class _SyntheticEvent:
    event_id: str
    origin_time_utc: datetime
    available_at: datetime
    x_m: float
    y_m: float
    magnitude: float


class _CoordinateCanary:
    """Reject reads of selected assessment-only coordinates before master sealing."""

    def __init__(
        self,
        values: NDArray[np.float64],
        *,
        canary_indices: tuple[int, ...],
        allowed: Callable[[], bool],
    ) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._canary_indices = frozenset(canary_indices)
        self._allowed = allowed
        self.accessed_indices: list[int] = []
        self.premaster_attempts: list[int] = []

    def __getitem__(self, key: Any) -> Any:
        selected = np.asarray(np.arange(self._values.size)[key]).reshape(-1)
        indices = [int(value) for value in selected]
        blocked = [index for index in indices if index in self._canary_indices]
        if blocked and not self._allowed():
            self.premaster_attempts.extend(blocked)
            raise AssertionError(
                "synthetic production chain opened assessment-only coordinates before master seal"
            )
        self.accessed_indices.extend(indices)
        return self._values[key]


@dataclass(frozen=True, slots=True)
class _FormalFixture:
    inputs: FormalScienceInputs
    catalog: Stage2SEarthquakeCatalog
    snapshot: LocalSupportSnapshot
    progress: list[str]
    expected_support_event_ids: tuple[str, ...]
    assessment_canary_indices: tuple[int, ...]
    longitude_canary: _CoordinateCanary
    latitude_canary: _CoordinateCanary
    expected_frame_count: int


class _ProcessedPathGuard:
    """Reject every relevant access below the real repository data/processed tree."""

    def __init__(self, repository_root: Path) -> None:
        forbidden_root = self._normalize(repository_root / "data" / "processed")
        assert forbidden_root is not None
        self._forbidden_root = forbidden_root
        self.attempts: list[tuple[str, str]] = []

    @staticmethod
    def _normalize(value: object) -> str | None:
        if isinstance(value, int):
            return None
        if not isinstance(value, str | bytes | os.PathLike):
            return None
        try:
            raw_path = os.fspath(cast(_PathValue, value))
        except TypeError:
            return None
        if isinstance(raw_path, bytes):
            raw_path = os.fsdecode(raw_path)
        return os.path.normcase(os.path.abspath(os.path.normpath(raw_path)))

    def check(self, value: object, *, operation: str) -> None:
        candidate = self._normalize(value)
        if candidate is None:
            return
        try:
            forbidden = (
                os.path.commonpath((candidate, self._forbidden_root)) == self._forbidden_root
            )
        except ValueError:
            forbidden = False
        if forbidden:
            self.attempts.append((operation, candidate))
            raise AssertionError(
                f"production-chain acceptance attempted {operation} under real "
                f"data/processed: {candidate}"
            )


def _install_processed_path_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> _ProcessedPathGuard:
    guard = _ProcessedPathGuard(REPOSITORY_ROOT)

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_os_stat = os.stat
    original_os_lstat = os.lstat
    original_os_listdir = os.listdir
    original_os_scandir = os.scandir
    original_path_open = Path.open
    original_path_stat = Path.stat
    original_path_iterdir = Path.iterdir
    original_path_glob = Path.glob
    original_path_rglob = Path.rglob
    original_glob = glob_module.glob
    original_iglob = glob_module.iglob

    def guarded_builtin_open(file: _OpenPathValue, *args: Any, **kwargs: Any) -> Any:
        guard.check(file, operation="builtins.open")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: _OpenPathValue, *args: Any, **kwargs: Any) -> Any:
        guard.check(file, operation="io.open")
        return original_io_open(file, *args, **kwargs)

    def guarded_os_open(path: _PathValue, *args: Any, **kwargs: Any) -> int:
        guard.check(path, operation="os.open")
        return original_os_open(path, *args, **kwargs)

    def guarded_os_stat(
        path: _OpenPathValue,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        guard.check(path, operation="os.stat")
        return original_os_stat(path, *args, **kwargs)

    def guarded_os_lstat(
        path: _PathValue,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        guard.check(path, operation="os.lstat")
        return original_os_lstat(path, *args, **kwargs)

    def guarded_os_listdir(
        path: int | str | os.PathLike[str] | None = ".",
    ) -> list[str]:
        guard.check(path, operation="os.listdir")
        return original_os_listdir(path)

    def guarded_os_scandir(path: object = ".") -> Any:
        guard.check(path, operation="os.scandir")
        return original_os_scandir(path)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        guard.check(path, operation="Path.open")
        return original_path_open(path, *args, **kwargs)

    def guarded_path_stat(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        guard.check(path, operation="Path.stat")
        return original_path_stat(path, *args, **kwargs)

    def guarded_path_iterdir(path: Path) -> Iterator[Path]:
        guard.check(path, operation="Path.iterdir")
        return original_path_iterdir(path)

    def guarded_path_glob(path: Path, pattern: str, **kwargs: Any) -> Iterator[Path]:
        guard.check(path, operation="Path.glob")
        return original_path_glob(path, pattern, **kwargs)

    def guarded_path_rglob(path: Path, pattern: str, **kwargs: Any) -> Iterator[Path]:
        guard.check(path, operation="Path.rglob")
        return original_path_rglob(path, pattern, **kwargs)

    def guarded_glob(pathname: _PathValue, *args: Any, **kwargs: Any) -> Any:
        root_dir = kwargs.get("root_dir")
        guard.check(pathname, operation="glob.glob")
        if root_dir is not None:
            guard.check(
                Path(cast(str | os.PathLike[str], root_dir)) / os.fsdecode(pathname),
                operation="glob.glob",
            )
        return original_glob(pathname, *args, **kwargs)

    def guarded_iglob(pathname: _PathValue, *args: Any, **kwargs: Any) -> Any:
        root_dir = kwargs.get("root_dir")
        guard.check(pathname, operation="glob.iglob")
        if root_dir is not None:
            guard.check(
                Path(cast(str | os.PathLike[str], root_dir)) / os.fsdecode(pathname),
                operation="glob.iglob",
            )
        return original_iglob(pathname, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(os, "stat", guarded_os_stat)
    monkeypatch.setattr(os, "lstat", guarded_os_lstat)
    monkeypatch.setattr(os, "listdir", guarded_os_listdir)
    monkeypatch.setattr(os, "scandir", guarded_os_scandir)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "stat", guarded_path_stat)
    monkeypatch.setattr(Path, "iterdir", guarded_path_iterdir)
    monkeypatch.setattr(Path, "glob", guarded_path_glob)
    monkeypatch.setattr(Path, "rglob", guarded_path_rglob)
    monkeypatch.setattr(glob_module, "glob", guarded_glob)
    monkeypatch.setattr(glob_module, "iglob", guarded_iglob)
    return guard


def _issue_time(local_date: date) -> datetime:
    return datetime.combine(local_date, time.min, _BEIJING_OFFSET).astimezone(UTC)


def _datetime_us(value: datetime) -> int:
    return int((value.astimezone(UTC) - _EPOCH).total_seconds() * 1_000_000)


def _build_grid_family() -> EqualAreaGridFamily:
    return build_equal_area_grid_family(box(0.0, 0.0, 250_000.0, 100_000.0))


def _build_calendar() -> Stage2SFoldCalendar:
    fit_dates = (date(1975, 3, 1), date(1975, 3, 8), date(1975, 3, 15))
    assessment_dates = (date(1976, 1, 2), date(1976, 5, 2), date(1976, 9, 2))
    folds: list[FoldCalendar] = []
    for fold_index, assessment_date in enumerate(assessment_dates, start=1):
        fit_exposures = tuple(
            TargetExposure(
                fold_index=fold_index,
                role="fit",
                issue_date_local=fit_date,
                issue_time_utc=_issue_time(fit_date),
                horizon_days=7,
                target_end_inclusive_utc=_issue_time(fit_date) + timedelta(days=7),
            )
            for fit_date in fit_dates[:fold_index]
        )
        assessment_issue_time = _issue_time(assessment_date)
        assessment_exposures = tuple(
            TargetExposure(
                fold_index=fold_index,
                role="assessment",
                issue_date_local=assessment_date,
                issue_time_utc=assessment_issue_time,
                horizon_days=horizon,
                target_end_inclusive_utc=assessment_issue_time + timedelta(days=horizon),
            )
            for horizon in _HORIZONS
        )
        folds.append(
            FoldCalendar(
                fold_index=fold_index,
                fit_scope_id=f"stage2s-development-fold-{fold_index}",
                fit_exposures=fit_exposures,
                fit_target_end_inclusive_utc=fit_exposures[-1].target_end_inclusive_utc,
                assessment_start_exclusive_utc=assessment_issue_time,
                assessment_end_inclusive_utc=assessment_issue_time + timedelta(days=90),
                assessment_exposures=assessment_exposures,
                assessment_issues=(
                    AssessmentIssue(
                        fold_index=fold_index,
                        issue_date_local=assessment_date,
                        issue_time_utc=assessment_issue_time,
                        horizons_days=_HORIZONS,
                    ),
                ),
            )
        )
    return Stage2SFoldCalendar(
        manifest_sha256=hashlib.sha256(b"stage2s-small-production-calendar").hexdigest(),
        folds=cast(
            tuple[FoldCalendar, FoldCalendar, FoldCalendar],
            tuple(folds),
        ),
    )


def _build_events(
    grid_family: EqualAreaGridFamily,
    calendar: Stage2SFoldCalendar,
) -> tuple[list[_SyntheticEvent], tuple[CompletenessEvent, ...]]:
    query_points = tuple(
        (float(cell.representative_point.x), float(cell.representative_point.y))
        for cell in grid_family.at(25.0).cells
    )
    assert len(query_points) == 40
    events: list[_SyntheticEvent] = []
    support_events: list[CompletenessEvent] = []
    support_start = datetime(1970, 1, 2, tzinfo=UTC)
    for index in range(200):
        origin = support_start + timedelta(minutes=index)
        magnitude = 3.0 if index < 150 else 3.2
        x_m, y_m = query_points[index % len(query_points)]
        event = _SyntheticEvent(
            event_id=f"support-{index:04d}",
            origin_time_utc=origin,
            available_at=origin,
            x_m=x_m,
            y_m=y_m,
            magnitude=magnitude,
        )
        events.append(event)
        support_events.append(
            CompletenessEvent(
                event_id=event.event_id,
                origin_time_utc=origin,
                available_at=origin,
                magnitude=magnitude,
                inside_study_area=True,
                x_m=x_m,
                y_m=y_m,
            )
        )

    for fit_index, fit_date in enumerate(
        (date(1975, 3, 1), date(1975, 3, 8), date(1975, 3, 15)),
        start=1,
    ):
        issue = _issue_time(fit_date)
        for component, offset_days, point_offset in (
            ("rp", -40, 0),
            ("r", -10, 1),
            ("r-late", -2, 2),
        ):
            origin = issue + timedelta(days=offset_days)
            available = issue - timedelta(hours=12) if component == "r-late" else origin
            x_m, y_m = query_points[(30 + fit_index + point_offset) % len(query_points)]
            events.append(
                _SyntheticEvent(
                    event_id=f"fit-{fit_index}-{component}",
                    origin_time_utc=origin,
                    available_at=available,
                    x_m=x_m,
                    y_m=y_m,
                    magnitude=4.5,
                )
            )
        x_m, y_m = query_points[(34 + fit_index) % len(query_points)]
        events.append(
            _SyntheticEvent(
                event_id=f"fit-{fit_index}-target",
                origin_time_utc=issue + timedelta(days=2),
                available_at=issue + timedelta(days=2),
                x_m=x_m,
                y_m=y_m,
                magnitude=5.5,
            )
        )

    for fold in calendar.folds:
        issue = fold.assessment_issues[0].issue_time_utc
        for component, offset_days, point_offset in (
            ("rp", -40, 0),
            ("r", -10, 1),
            ("r-late", -2, 2),
        ):
            origin = issue + timedelta(days=offset_days)
            available = issue - timedelta(hours=12) if component == "r-late" else origin
            x_m, y_m = query_points[(fold.fold_index * 3 + point_offset) % len(query_points)]
            events.append(
                _SyntheticEvent(
                    event_id=f"assessment-fold-{fold.fold_index}-{component}",
                    origin_time_utc=origin,
                    available_at=available,
                    x_m=x_m,
                    y_m=y_m,
                    magnitude=4.5,
                )
            )
        for target_index, offset_days in enumerate(
            _ASSESSMENT_TARGET_OFFSETS_DAYS,
            start=1,
        ):
            point_index = (fold.fold_index - 1) * 10 + target_index - 1
            x_m, y_m = query_points[point_index]
            target_time = issue + timedelta(days=offset_days)
            events.append(
                _SyntheticEvent(
                    event_id=(f"assessment-fold-{fold.fold_index}-target-{target_index:02d}"),
                    origin_time_utc=target_time,
                    available_at=target_time,
                    x_m=x_m,
                    y_m=y_m,
                    magnitude=5.5,
                )
            )
    return events, tuple(support_events)


def _build_catalog(
    events: list[_SyntheticEvent],
) -> Stage2SEarthquakeCatalog:
    ordered = tuple(
        sorted(
            events,
            key=lambda event: (
                event.origin_time_utc,
                event.event_id.encode("utf-8"),
            ),
        )
    )
    inverse = Transformer.from_crs(
        CRS.from_user_input(EQUAL_AREA_CRS),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    longitude_values: list[float] = []
    latitude_values: list[float] = []
    for event in ordered:
        longitude, latitude = inverse.transform(event.x_m, event.y_m)
        longitude_values.append(float(longitude))
        latitude_values.append(float(latitude))
    event_ids = tuple(event.event_id for event in ordered)
    origin_time_us = np.asarray(
        [_datetime_us(event.origin_time_utc) for event in ordered],
        dtype=np.int64,
    )
    available_at_us = np.asarray(
        [_datetime_us(event.available_at) for event in ordered],
        dtype=np.int64,
    )
    longitude = np.asarray(longitude_values, dtype=np.float64)
    latitude = np.asarray(latitude_values, dtype=np.float64)
    magnitude = np.asarray([event.magnitude for event in ordered], dtype=np.float64)
    inside = np.ones(len(ordered), dtype=np.bool_)
    table = pa.table(
        {
            "event_id": pa.array(event_ids, type=pa.string()),
            "origin_time_utc": pa.array(
                [event.origin_time_utc for event in ordered],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "available_at": pa.array(
                [event.available_at for event in ordered],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "longitude": pa.array(longitude),
            "latitude": pa.array(latitude),
            "magnitude": pa.array(magnitude),
            "inside_study_area": pa.array(inside),
        }
    )
    identity_payload = json.dumps(
        {
            "event_ids": event_ids,
            "origin_time_us": origin_time_us.tolist(),
            "available_at_us": available_at_us.tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Stage2SEarthquakeCatalog(
        identity=CatalogIdentity(
            row_count=len(ordered),
            file_sha256=hashlib.sha256(b"file:" + identity_payload).hexdigest(),
            content_sha256=hashlib.sha256(b"content:" + identity_payload).hexdigest(),
            schema_sha256=hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest(),
        ),
        event_ids=event_ids,
        origin_time_us=origin_time_us,
        available_at_us=available_at_us,
        longitude=longitude,
        latitude=latitude,
        magnitude=magnitude,
        inside_study_area=inside,
        _table=table,
    )


def _build_spatial_preflight(
    grid_family: EqualAreaGridFamily,
) -> NonTargetPreflight:
    source_grid = grid_family.at(25.0)
    assert len(source_grid.cells) == 40
    rows = np.asarray([cell.row for cell in source_grid.cells], dtype=np.int64)
    columns = np.asarray([cell.column for cell in source_grid.cells], dtype=np.int64)
    query_xy_m = np.asarray(
        [(cell.representative_point.x, cell.representative_point.y) for cell in source_grid.cells],
        dtype=np.float64,
    )
    areas = np.asarray(
        [cell.clipped_area_m2 / 1_000_000.0 for cell in source_grid.cells],
        dtype=np.float64,
    )
    query_grid = Stage2SQueryGrid(
        grid_id=hashlib.sha256(b"stage2s-small-query-grid").hexdigest(),
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=25.0,
        cell_ids=source_grid.cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=areas,
    )
    zone_ids = tuple(f"zone-{index:02d}" for index in range(1, 40))
    zone_by_cell = {
        cell_id: zone_ids[min(index, len(zone_ids) - 1)]
        for index, cell_id in enumerate(query_grid.cell_ids)
    }
    adapter = NonTargetSpatialAdapter(
        query_grid=query_grid,
        construction_zone_id_by_cell_id=MappingProxyType(zone_by_cell),
        grid_family=grid_family,
    )
    grid_counts = cast(
        tuple[int, int, int],
        tuple(len(grid.cells) for grid in grid_family.grids),
    )
    grid_areas = cast(
        tuple[float, float, float],
        tuple(
            sum(cell.clipped_area_m2 for cell in grid.cells) / 1_000_000.0
            for grid in grid_family.grids
        ),
    )
    return NonTargetPreflight(
        adapter=adapter,
        study_area_file_sha256=hashlib.sha256(b"synthetic-study-area").hexdigest(),
        projected_geometry_sha256=hashlib.sha256(b"synthetic-projected-geometry").hexdigest(),
        projected_area_m2=float(grid_family.study_area_equal_area.area),
        mapping_file_sha256=hashlib.sha256(b"synthetic-zone-mapping").hexdigest(),
        mapping_schema_sha256=hashlib.sha256(b"synthetic-zone-schema").hexdigest(),
        zone_ids=zone_ids,
        grid_cell_counts=grid_counts,
        grid_total_area_km2=grid_areas,
    )


def _build_fixture(repository_root: Path) -> _FormalFixture:
    root = repository_root.resolve()
    grid_family = _build_grid_family()
    calendar = _build_calendar()
    events, completeness_events = _build_events(grid_family, calendar)
    fit_end = datetime(1975, 1, 1, tzinfo=UTC)
    snapshot = build_local_support_snapshot(
        completeness_events,
        fit_end_utc=fit_end,
        study_area_equal_area=grid_family.study_area_equal_area,
    )
    assert snapshot.retained_area_fraction == 1.0
    catalog = _build_catalog(events)
    spatial = _build_spatial_preflight(grid_family)
    protocol = Stage2SProtocolBundle(
        repository_root=root,
        config={
            "long_term_background": {
                "fit_end_utc": fit_end.isoformat().replace("+00:00", "Z"),
                "support_id": snapshot.support_id,
            }
        },
        fold_manifest={"synthetic": True},
        input_contract={"synthetic": True},
        config_sha256=hashlib.sha256(b"synthetic-config").hexdigest(),
        fold_manifest_sha256=calendar.manifest_sha256,
        input_contract_sha256=hashlib.sha256(b"synthetic-input-contract").hexdigest(),
    )
    fixture_identity_root = (root / "fixture-input-identities").resolve()
    preflight_receipt = write_o_excl_record(
        fixture_identity_root / "preflight.json",
        record_type="stage2s_non_target_preflight_receipt",
        bindings={"synthetic_fixture_only": True},
    )
    attempt = write_o_excl_record(
        fixture_identity_root / "attempt.json",
        record_type="stage2s_formal_attempt_ledger",
        bindings={"synthetic_fixture_only": True},
    )
    target_read = write_o_excl_record(
        fixture_identity_root / "target-read.json",
        record_type="stage2s_target_read_receipt",
        bindings={"synthetic_fixture_only": True},
    )
    preflight = FormalPreflightContext(
        spatial=spatial,
        support_manifest=cast(BackgroundLocalSupportManifest, object()),
        calendar=calendar,
        receipt_bindings=MappingProxyType({"synthetic_fixture_only": True}),
    )
    progress: list[str] = []
    inputs = FormalScienceInputs(
        protocol=protocol,
        code_commit="b" * 40,
        preflight=preflight,
        preflight_receipt=preflight_receipt,
        attempt_ledger=attempt,
        target_read_receipt=target_read,
        progress=progress.append,
    )
    expected_support_event_ids = tuple(
        event_id for event_id in catalog.event_ids if event_id.startswith("support-")
    )
    assessment_canary_indices = tuple(
        index
        for index, event_id in enumerate(catalog.event_ids)
        if event_id.startswith("assessment-fold-3-target-")
    )
    assert len(expected_support_event_ids) == 200
    assert len(assessment_canary_indices) == len(_ASSESSMENT_TARGET_OFFSETS_DAYS)
    master_path = (
        root / "data/interim/stage2s/causal_seismicity_screen/prediction_seal.json"
    ).resolve()
    longitude_canary = _CoordinateCanary(
        np.asarray(catalog.longitude, dtype=np.float64),
        canary_indices=assessment_canary_indices,
        allowed=master_path.is_file,
    )
    latitude_canary = _CoordinateCanary(
        np.asarray(catalog.latitude, dtype=np.float64),
        canary_indices=assessment_canary_indices,
        allowed=master_path.is_file,
    )
    object.__setattr__(catalog, "longitude", longitude_canary)
    object.__setattr__(catalog, "latitude", latitude_canary)
    return _FormalFixture(
        inputs=inputs,
        catalog=catalog,
        snapshot=snapshot,
        progress=progress,
        expected_support_event_ids=expected_support_event_ids,
        assessment_canary_indices=assessment_canary_indices,
        longitude_canary=longitude_canary,
        latitude_canary=latitude_canary,
        expected_frame_count=3 * len(_HORIZONS) * 3,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _embedded_map_data(document: str) -> Mapping[str, object]:
    matched = re.search(
        r"const DATA=(\{.*\});\s*const groupKey=frame=>",
        document,
        flags=re.DOTALL,
    )
    assert matched is not None
    parsed = json.loads(matched.group(1))
    assert isinstance(parsed, dict)
    return cast(Mapping[str, object], parsed)


def test_real_production_science_chain_with_only_synthetic_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert all(os.environ[name] == "1" for name in _THREAD_ENVIRONMENT_VARIABLES)
    guard = _install_processed_path_guard(monkeypatch)
    fixture = _build_fixture(tmp_path / "synthetic-formal-repository")
    rebuild_calls = 0

    def rebuild_fold4_boundary(
        inputs: FormalScienceInputs,
        support_view: Any,
    ) -> LocalSupportSnapshot:
        nonlocal rebuild_calls
        rebuild_calls += 1
        assert inputs is fixture.inputs
        assert support_view.catalog is fixture.catalog
        assert support_view.cutoff_utc == fixture.snapshot.fit_end_utc
        observed_ids = tuple(
            fixture.catalog.event_ids[int(index)] for index in support_view.indices
        )
        assert observed_ids == fixture.expected_support_event_ids
        return fixture.snapshot

    # The only production monkeypatch: the public frozen Fold-4 manifest equality
    # boundary. The snapshot itself is still built by the real completeness code.
    monkeypatch.setattr(production, "_rebuild_fold4", rebuild_fold4_boundary)

    record = run_formal_science(fixture.inputs, fixture.catalog)

    assert rebuild_calls == 1
    assert record.mode == "formal_development"
    assert fixture.progress == [
        "fold4_support_rebuilt",
        "fold_1_sealed",
        "fold_2_sealed",
        "fold_3_sealed",
        "master_prediction_sealed",
        "assessment_scored",
        "artifacts_rendered",
    ]

    expected_cells = tuple(
        (fold_index, horizon) for fold_index in (1, 2, 3) for horizon in _HORIZONS
    )
    assert (
        tuple((cell.get("fold_index"), cell.get("horizon_days")) for cell in record.cell_scores)
        == expected_cells
    )
    assert all(cell.get("event_ids") for cell in record.cell_scores)
    target_union = {
        cast(str, event_id)
        for cell in record.cell_scores
        for event_id in cast(tuple[object, ...], cell["event_ids"])
    }
    assert len(target_union) >= 20

    assert len(record.bootstrap_rows) == 2_000
    assert all(len(row) == 4 for row in record.bootstrap_rows)
    assert tuple(item.get("delay_days") for item in record.latency_evidence) == (1, 7)
    regions = record.regional_evidence.get("regions")
    assert isinstance(regions, tuple)
    assert len(regions) == 39
    assert len({cast(str, region["zone_id"]) for region in regions}) == 39

    seal_chain = record.seal_chain
    assert len(cast(tuple[object, ...], seal_chain["fold_fit_receipt_sha256"])) == 3
    assert len(cast(tuple[object, ...], seal_chain["issue_prediction_seal_sha256"])) == 3
    assert len(cast(tuple[object, ...], seal_chain["fold_prediction_seal_sha256"])) == 3
    seal_root = (
        fixture.inputs.protocol.repository_root / "data/interim/stage2s/causal_seismicity_screen"
    )
    seal_files = tuple(sorted(seal_root.rglob("*.json")))
    assert len(seal_files) == 10
    observed_record_types = [
        json.loads(path.read_text(encoding="utf-8"))["record_type"] for path in seal_files
    ]
    assert observed_record_types.count("stage2s_fold_fit_receipt") == 3
    assert observed_record_types.count("stage2s_issue_prediction_seal") == 3
    assert observed_record_types.count("stage2s_fold_prediction_seal") == 3
    assert observed_record_types.count("stage2s_master_prediction_seal") == 1

    canary_indices = set(fixture.assessment_canary_indices)
    assert fixture.longitude_canary.premaster_attempts == []
    assert fixture.latitude_canary.premaster_attempts == []
    assert canary_indices.issubset(fixture.longitude_canary.accessed_indices)
    assert canary_indices.issubset(fixture.latitude_canary.accessed_indices)

    output_root = (
        fixture.inputs.protocol.repository_root / "outputs/stage2s/causal_seismicity_screen"
    )
    assert set(record.artifact_sha256_by_name) == set(ALL_ARTIFACT_NAMES)
    assert tuple(sorted(path.name for path in output_root.iterdir())) == tuple(
        sorted(ALL_ARTIFACT_NAMES)
    )
    for name in ALL_ARTIFACT_NAMES:
        assert _sha256((output_root / name).read_bytes()) == record.artifact_sha256_by_name[name]

    backtest_document = (output_root / PROTOCOL_ARTIFACT_NAMES[3]).read_text(encoding="utf-8")
    for zone_index in range(1, 40):
        assert f"zone-{zone_index:02d}" in backtest_document

    map_document = (output_root / PROTOCOL_ARTIFACT_NAMES[4]).read_text(encoding="utf-8")
    map_data = _embedded_map_data(map_document)
    frames = map_data.get("frames")
    assert isinstance(frames, list)
    assert len(frames) == fixture.expected_frame_count
    frame_groups: dict[tuple[int, int, str], set[str]] = {}
    for raw_frame in frames:
        assert isinstance(raw_frame, dict)
        key = (
            int(raw_frame["fold_index"]),
            int(raw_frame["horizon_days"]),
            cast(str, raw_frame["issue_time_utc"]),
        )
        frame_groups.setdefault(key, set()).add(cast(str, raw_frame["model_id"]))
    assert len(frame_groups) == 3 * len(_HORIZONS)
    assert all(models == {"S0", "S1", "SP"} for models in frame_groups.values())

    before_retry = _file_snapshot(fixture.inputs.protocol.repository_root)
    with pytest.raises(Stage2SSealExists):
        run_formal_science(fixture.inputs, fixture.catalog)
    assert _file_snapshot(fixture.inputs.protocol.repository_root) == before_retry
    assert rebuild_calls == 2
    assert guard.attempts == []
