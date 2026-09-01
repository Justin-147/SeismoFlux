"""Authorization-gated target construction and raw S1 development scoring.

The prediction phase is deliberately unable to import target rows from this
module.  Every public target/scoring entry point requires the concrete
``DevelopmentScoringAuthorization`` returned only after all four development
prediction seals have been re-verified.  This module performs no file I/O,
bootstrap, model selection, plotting, holdout, audit, or locked-test work.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Protocol, cast, runtime_checkable

import pandas as pd

from seismoflux.d1_replay.spatial import select_alarm_prefixes
from seismoflux.multitask_s0 import build_episodes
from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
)
from seismoflux.multitask_s1.metrics import (
    AlarmRecallScore,
    MagnitudeEvaluation,
    compare_m0_m3_on_common_m5_support,
    score_location_events,
    score_m0_unique_events,
    score_minimal_joint_m5,
    score_nb2_exposure,
    score_poisson_exposure,
)
from seismoflux.multitask_s1.prediction_seal import (
    DevelopmentScoringAuthorization,
    PredictionInputIdentities,
    SealedArtifact,
    authorize_development_scoring,
)
from seismoflux.multitask_s1.runner_inputs import CatalogEventTable, OuterIssueRow
from seismoflux.multitask_s1.time_magnitude import (
    NB2DispersionQualification,
    TruncatedGRMagnitudeModel,
)
from seismoflux.stage2s.contracts import SpatialGrid

MagnitudeBin = Literal["M5_6", "M6_plus"]
CountBand = Literal["M5_6", "M6_plus", "M5_plus_for_joint"]
CountDistribution = Literal["poisson", "nb2"]
ScoreStatus = Literal["evaluable", "not_evaluable"]

MAIN_SCIENTIFIC_ANCHOR: Final = (
    "24h_delay_M5_6_30d_600000km2_strict_0km_fixed_anchor_episode_recall"
)
FIXED_EPISODE_DEFINITION: Final = "full_catalog_fixed_anchor_30d_75km_by_magnitude_bin"
TARGET_INTERVAL: Final = "(issue,issue+horizon]"
CATALOG_DELAY_HOURS: Final = 24
STRICT_HIT_TOLERANCE_KM: Final = 0.0
_SHANGHAI: Final = timezone(timedelta(hours=8))
_SHA256 = re.compile(r"[0-9a-f]{64}")

_FOLD_BOUNDS_UTC: Final = MappingProxyType(
    {
        "C_DEV_2000_2004": (
            datetime(1999, 12, 31, 16, tzinfo=UTC),
            datetime(2004, 12, 31, 16, tzinfo=UTC),
        ),
        "C_DEV_2005_2009": (
            datetime(2004, 12, 31, 16, tzinfo=UTC),
            datetime(2009, 12, 31, 16, tzinfo=UTC),
        ),
        "C_DEV_2010_2014": (
            datetime(2009, 12, 31, 16, tzinfo=UTC),
            datetime(2014, 12, 31, 16, tzinfo=UTC),
        ),
        "C_DEV_2015_2019": (
            datetime(2014, 12, 31, 16, tzinfo=UTC),
            datetime(2019, 12, 31, 16, tzinfo=UTC),
        ),
    }
)
_MAGNITUDE_BOUNDS: Final = MappingProxyType({"M5_6": (5.0, 6.0), "M6_plus": (6.0, None)})


class DevelopmentScoreError(ValueError):
    """Raised when an authorized development scoring invariant fails closed."""


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Filesystem roots needed to re-authorize every public scoring entry point."""

    output_root: Path
    expected_seal_sha256: str
    project_root: Path
    data_root: Path


def _require_authorization(value: object) -> DevelopmentScoringAuthorization:
    """Reject duck-typed or partial stand-ins for the four-fold authorization."""

    if type(value) is not DevelopmentScoringAuthorization:
        raise TypeError("outer development targets/scores require DevelopmentScoringAuthorization")
    authorization = value
    if (
        type(authorization.seal) is not SealedArtifact
        or type(authorization.input_identities) is not PredictionInputIdentities
        or authorization.seal.size_bytes <= 0
        or _SHA256.fullmatch(authorization.seal.sha256) is None
    ):
        raise DevelopmentScoreError("development scoring authorization is malformed")
    ordered = authorization.ordered_fold_sha256
    if tuple(fold_id for fold_id, _ in ordered) != DEVELOPMENT_FOLD_IDS or any(
        _SHA256.fullmatch(digest) is None for _, digest in ordered
    ):
        raise DevelopmentScoreError(
            "development scoring authorization does not bind exactly four development folds"
        )
    return authorization


def _reauthorize(context: ScoringContext) -> DevelopmentScoringAuthorization:
    if type(context) is not ScoringContext:
        raise TypeError("outer development targets/scores require a ScoringContext")
    if _SHA256.fullmatch(context.expected_seal_sha256) is None:
        raise DevelopmentScoreError("scoring context seal SHA-256 is malformed")
    return _require_authorization(
        authorize_development_scoring(
            context.output_root,
            expected_seal_sha256=context.expected_seal_sha256,
            project_root=context.project_root,
            data_root=context.data_root,
        )
    )


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DevelopmentScoreError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _epoch_us_to_utc(value: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=value)


def _validate_fold_id(value: str) -> str:
    if value not in DEVELOPMENT_FOLD_IDS:
        raise DevelopmentScoreError(
            "only the four frozen development folds may construct targets or scores"
        )
    return value


def _weekly_issue_axis(fold_id: str) -> tuple[datetime, ...]:
    start, end = _FOLD_BOUNDS_UTC[_validate_fold_id(fold_id)]
    local_start = start.astimezone(_SHANGHAI)
    candidate = local_start + timedelta(days=(3 - local_start.weekday()) % 7)
    result: list[datetime] = []
    while candidate.astimezone(UTC) < end:
        result.append(candidate.astimezone(UTC))
        candidate += timedelta(days=7)
    return tuple(result)


def _primary_issue_axis(fold_id: str, horizon_days: int) -> tuple[datetime, ...]:
    if horizon_days not in HORIZONS_DAYS:
        raise DevelopmentScoreError("horizon is outside the five frozen development horizons")
    _, fold_end = _FOLD_BOUNDS_UTC[_validate_fold_id(fold_id)]
    mature = tuple(
        issue
        for issue in _weekly_issue_axis(fold_id)
        if issue + timedelta(days=horizon_days) <= fold_end
    )
    selected: list[datetime] = []
    separation = timedelta(days=horizon_days + 30)
    for issue in mature:
        if not selected or issue >= selected[-1] + separation:
            selected.append(issue)
    return tuple(selected)


@runtime_checkable
class EventCellLocator(Protocol):
    """Small adapter implemented by the frozen 25 km D1 cell locator."""

    def locate_lonlat(self, longitude: float, latitude: float) -> int | None: ...


@dataclass(frozen=True, slots=True)
class EpisodeTargetEvent:
    """One physical target with its full-catalog episode metadata."""

    event_id: str
    origin_time_utc: datetime
    magnitude: float
    longitude: float
    latitude: float
    cell_index: int
    episode_id: str
    global_episode_member_count: int
    is_episode_anchor: bool


@dataclass(frozen=True, slots=True)
class PrimaryExposureTargets:
    """One retained non-overlapping outer development exposure, including N=0."""

    fold_id: str
    issue_time_utc: datetime
    horizon_days: int
    target_end_utc: datetime
    magnitude_bin: MagnitudeBin
    target_interval: str
    events: tuple[EpisodeTargetEvent, ...]


@dataclass(frozen=True, slots=True)
class PrimaryDevelopmentTargets:
    """All four-fold primary targets constructed only after prediction sealing."""

    grid_id: str
    exposures: tuple[PrimaryExposureTargets, ...]
    episode_definition: str = FIXED_EPISODE_DEFINITION


@dataclass(frozen=True, slots=True)
class StandaloneMagnitudeEvent:
    """A unique M>=4 event tied to the latest strictly earlier weekly forecast."""

    fold_id: str
    forecast_issue_time_utc: datetime
    event_id: str
    origin_time_utc: datetime
    magnitude: float


@dataclass(frozen=True, slots=True)
class StandaloneMagnitudeTargets:
    """Unique-event populations for M0 and the common M0|M5 versus M3 tail."""

    m0_m4_events: tuple[StandaloneMagnitudeEvent, ...]
    common_m5_events: tuple[StandaloneMagnitudeEvent, ...]


def _catalog_frame(catalog: CatalogEventTable) -> pd.DataFrame:
    if not isinstance(catalog, CatalogEventTable):
        raise TypeError("catalog must be a CatalogEventTable")
    return pd.DataFrame(
        {
            "event_id": catalog.event_ids,
            "origin_time_utc": pd.to_datetime(catalog.origin_time_us, unit="us", utc=True),
            "available_at": pd.to_datetime(catalog.available_at_us, unit="us", utc=True),
            "longitude": catalog.longitude,
            "latitude": catalog.latitude,
            "magnitude": catalog.magnitude,
            "inside_study_area": catalog.inside_study_area,
        }
    )


def _validate_primary_issues(
    issue_rows: Sequence[OuterIssueRow],
) -> tuple[OuterIssueRow, ...]:
    rows = tuple(issue_rows)
    grouped: dict[tuple[str, int], list[datetime]] = defaultdict(list)
    seen: set[tuple[str, int, datetime]] = set()
    for row in rows:
        if not isinstance(row, OuterIssueRow):
            raise TypeError("primary issue rows must contain OuterIssueRow values")
        fold_id = _validate_fold_id(row.fold_id)
        if row.horizon_days not in HORIZONS_DAYS:
            raise DevelopmentScoreError("outer issue uses an unfrozen horizon")
        issue = _aware_utc(row.issue_time_utc, label="outer issue")
        target_end = _aware_utc(row.target_end_utc, label="outer target end")
        if (
            not row.primary_exposure_selected
            or row.maturity_status != "mature"
            or target_end != issue + timedelta(days=row.horizon_days)
        ):
            raise DevelopmentScoreError(
                "outer scoring accepts only mature primary exposures with (issue,issue+h]"
            )
        start, end = _FOLD_BOUNDS_UTC[fold_id]
        local_issue = issue.astimezone(_SHANGHAI)
        if (
            not start <= issue < target_end <= end
            or local_issue.weekday() != 3
            or local_issue.time() != datetime.min.time()
        ):
            raise DevelopmentScoreError("primary issue violates the frozen fold or Thursday axis")
        key = (fold_id, row.horizon_days, issue)
        if key in seen:
            raise DevelopmentScoreError("duplicate primary issue row")
        seen.add(key)
        grouped[(fold_id, row.horizon_days)].append(issue)
    expected_groups = {
        (fold_id, horizon) for fold_id in DEVELOPMENT_FOLD_IDS for horizon in HORIZONS_DAYS
    }
    if set(grouped) != expected_groups:
        raise DevelopmentScoreError(
            "primary target construction requires all four folds and five horizons"
        )
    for group_key, issues in grouped.items():
        actual = tuple(sorted(issues))
        if actual != _primary_issue_axis(*group_key):
            raise DevelopmentScoreError(
                "primary issues differ from the frozen greedy h+30d calendar"
            )
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                DEVELOPMENT_FOLD_IDS.index(item.fold_id),
                HORIZONS_DAYS.index(item.horizon_days),
                item.issue_time_utc,
            ),
        )
    )


def _episode_owner(anchor_time: datetime) -> str | None:
    for fold_id in DEVELOPMENT_FOLD_IDS:
        start, end = _FOLD_BOUNDS_UTC[fold_id]
        if start <= anchor_time < end:
            return fold_id
    return None


def _build_primary_exposure_targets(
    context: ScoringContext,
    *,
    catalog: CatalogEventTable,
    primary_issue_rows: Sequence[OuterIssueRow],
    grid: SpatialGrid,
    locator: EventCellLocator,
) -> PrimaryDevelopmentTargets:
    """Construct only the frozen four-fold primary target populations.

    Episodes are built once per formal magnitude bin from the complete in-study
    catalog, before fold ownership.  The anchor's outer fold owns the entire
    episode; a member cannot migrate into a later fold.
    """

    _reauthorize(context)
    if not isinstance(grid, SpatialGrid) or grid.cell_size_km != 25.0:
        raise TypeError("target construction requires the frozen 25 km SpatialGrid")
    if not isinstance(locator, EventCellLocator):
        raise TypeError("locator must implement the frozen lon/lat cell interface")
    rows = _validate_primary_issues(primary_issue_rows)
    frame = _catalog_frame(catalog)
    catalog_index = {event_id: index for index, event_id in enumerate(catalog.event_ids)}
    located_cells: dict[str, int] = {}
    exposures: list[PrimaryExposureTargets] = []
    seen_by_bin_horizon: dict[tuple[MagnitudeBin, int], set[str]] = defaultdict(set)

    for magnitude_bin in ("M5_6", "M6_plus"):
        minimum, maximum = _MAGNITUDE_BOUNDS[magnitude_bin]
        selected = frame["inside_study_area"] & (frame["magnitude"] >= minimum)
        if maximum is not None:
            selected &= frame["magnitude"] < maximum
        panel = frame.loc[selected].reset_index(drop=True)
        episodes = build_episodes(panel, max_time_days=30, max_distance_km=75.0)
        metadata: dict[str, tuple[str, int, bool, str | None]] = {}
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            anchor_id = str(episode["anchor_event_id"])
            member_ids = tuple(
                str(value) for value in cast(Sequence[object], episode["member_event_ids"])
            )
            member_count = int(cast(int, episode["member_count"]))
            anchor_time = pd.Timestamp(episode["anchor_time_utc"]).to_pydatetime().astimezone(UTC)
            owner = _episode_owner(anchor_time)
            for event_id in member_ids:
                if event_id in metadata:
                    raise AssertionError("one event appeared in two full-catalog episodes")
                metadata[event_id] = (
                    episode_id,
                    member_count,
                    event_id == anchor_id,
                    owner,
                )
        if set(metadata) != set(panel["event_id"].astype(str)):
            raise AssertionError("full-catalog episode registry lost a formal target event")

        for row in rows:
            issue = _aware_utc(row.issue_time_utc, label="outer issue")
            target_end = _aware_utc(row.target_end_utc, label="outer target end")
            event_rows: list[EpisodeTargetEvent] = []
            for event_id, (episode_id, member_count, is_anchor, owner) in metadata.items():
                if owner != row.fold_id:
                    continue
                index = catalog_index[event_id]
                origin = _epoch_us_to_utc(int(catalog.origin_time_us[index]))
                if not issue < origin <= target_end:
                    continue
                cell_index = located_cells.get(event_id)
                if cell_index is None:
                    located = locator.locate_lonlat(
                        float(catalog.longitude[index]), float(catalog.latitude[index])
                    )
                    if isinstance(located, bool) or not isinstance(located, int):
                        raise DevelopmentScoreError(
                            "an in-study target did not map to one frozen 25 km cell"
                        )
                    if not 0 <= located < grid.cell_count:
                        raise DevelopmentScoreError("target cell index is outside the frozen grid")
                    cell_index = located
                    located_cells[event_id] = cell_index
                event_rows.append(
                    EpisodeTargetEvent(
                        event_id=event_id,
                        origin_time_utc=origin,
                        magnitude=float(catalog.magnitude[index]),
                        longitude=float(catalog.longitude[index]),
                        latitude=float(catalog.latitude[index]),
                        cell_index=cell_index,
                        episode_id=episode_id,
                        global_episode_member_count=member_count,
                        is_episode_anchor=is_anchor,
                    )
                )
            event_rows.sort(key=lambda item: (item.origin_time_utc, item.event_id.encode("utf-8")))
            seen = seen_by_bin_horizon[(magnitude_bin, row.horizon_days)]
            duplicates = seen.intersection(item.event_id for item in event_rows)
            if duplicates:
                raise AssertionError(
                    "a physical event appeared twice on one horizon's primary exposure axis"
                )
            seen.update(item.event_id for item in event_rows)
            exposures.append(
                PrimaryExposureTargets(
                    fold_id=row.fold_id,
                    issue_time_utc=issue,
                    horizon_days=row.horizon_days,
                    target_end_utc=target_end,
                    magnitude_bin=magnitude_bin,
                    target_interval=TARGET_INTERVAL,
                    events=tuple(event_rows),
                )
            )
    return PrimaryDevelopmentTargets(
        grid_id=grid.grid_id,
        exposures=tuple(
            sorted(
                exposures,
                key=lambda item: (
                    DEVELOPMENT_FOLD_IDS.index(item.fold_id),
                    HORIZONS_DAYS.index(item.horizon_days),
                    item.issue_time_utc,
                    item.magnitude_bin,
                ),
            )
        ),
    )


def _assign_standalone_magnitude_targets(
    context: ScoringContext,
    *,
    catalog: CatalogEventTable,
    scheduled_issue_times_by_fold: Mapping[str, Sequence[datetime]],
) -> StandaloneMagnitudeTargets:
    """Assign each M>=4 event once to its latest strictly earlier Thursday.

    The weekly axis is used only for unique-event magnitude attribution.  It is
    not an additional location/time exposure sample and cannot enlarge the
    primary independent sample count.
    """

    _reauthorize(context)
    if not isinstance(catalog, CatalogEventTable):
        raise TypeError("catalog must be a CatalogEventTable")
    if set(scheduled_issue_times_by_fold) != set(DEVELOPMENT_FOLD_IDS):
        raise DevelopmentScoreError(
            "standalone magnitude calendar must cover four development folds"
        )
    calendars: dict[str, tuple[datetime, ...]] = {}
    for fold_id in DEVELOPMENT_FOLD_IDS:
        raw = scheduled_issue_times_by_fold[fold_id]
        actual = tuple(_aware_utc(value, label="scheduled issue") for value in raw)
        if actual != _weekly_issue_axis(fold_id):
            raise DevelopmentScoreError(
                "standalone magnitude assignment requires the exact weekly Thursday axis"
            )
        calendars[fold_id] = actual

    assignments: list[StandaloneMagnitudeEvent] = []
    for index, event_id in enumerate(catalog.event_ids):
        if not bool(catalog.inside_study_area[index]) or float(catalog.magnitude[index]) < 4.0:
            continue
        origin = _epoch_us_to_utc(int(catalog.origin_time_us[index]))
        owner = _episode_owner(origin)
        if owner is None:
            continue
        issues = calendars[owner]
        issue_index = bisect.bisect_left(issues, origin) - 1
        if issue_index < 0:
            continue
        assignments.append(
            StandaloneMagnitudeEvent(
                fold_id=owner,
                forecast_issue_time_utc=issues[issue_index],
                event_id=event_id,
                origin_time_utc=origin,
                magnitude=float(catalog.magnitude[index]),
            )
        )
    assignments.sort(
        key=lambda item: (
            DEVELOPMENT_FOLD_IDS.index(item.fold_id),
            item.origin_time_utc,
            item.event_id.encode("utf-8"),
        )
    )
    if len({item.event_id for item in assignments}) != len(assignments):
        raise AssertionError("standalone M0 assignment duplicated a physical event")
    common_m5 = tuple(item for item in assignments if item.magnitude >= 5.0)
    return StandaloneMagnitudeTargets(tuple(assignments), common_m5)


@dataclass(frozen=True, slots=True)
class LocationForecast:
    fold_id: str
    issue_time_utc: datetime
    horizon_days: int
    model_id: str
    cell_relative_mass: object


@dataclass(frozen=True, slots=True)
class CountForecast:
    fold_id: str
    issue_time_utc: datetime
    horizon_days: int
    model_id: str
    magnitude_band: CountBand
    expected_count: float
    distribution: CountDistribution
    nb2_qualification: NB2DispersionQualification | None = None


@dataclass(frozen=True, slots=True)
class MagnitudeForecast:
    fold_id: str
    issue_time_utc: datetime
    model_id: str
    model: TruncatedGRMagnitudeModel


@runtime_checkable
class DevelopmentPredictionSource(Protocol):
    """NPZ-independent forecast lookup used by the authorized score layer."""

    def location_forecast(
        self, *, fold_id: str, issue_time_utc: datetime, horizon_days: int, model_id: str
    ) -> LocationForecast: ...

    def count_forecast(
        self,
        *,
        fold_id: str,
        issue_time_utc: datetime,
        horizon_days: int,
        model_id: str,
        magnitude_band: CountBand,
    ) -> CountForecast: ...

    def magnitude_forecast(
        self, *, fold_id: str, issue_time_utc: datetime, model_id: str
    ) -> MagnitudeForecast: ...


@dataclass(frozen=True, slots=True)
class LocationRawScoreRow:
    fold_id: str
    issue_time_utc: datetime
    horizon_days: int
    magnitude_bin: MagnitudeBin
    model_id: str
    metric: Literal["spatial_log_density", "strict_recall"]
    basis: str | None
    area_budget_km2: float | None
    actual_area_km2: float | None
    value: float | None
    event_count: int
    hit_weight: float | None
    total_weight: float | None
    status: ScoreStatus
    is_main_scientific_anchor: bool
    scientific_anchor_id: str | None
    event_ids: tuple[str, ...]
    event_log_densities_per_km2: tuple[float, ...]
    episode_ids: tuple[str, ...]
    global_episode_member_counts: tuple[int, ...]
    is_episode_anchor: tuple[bool, ...]
    event_cell_indices: tuple[int, ...]
    event_longitudes: tuple[float, ...]
    event_latitudes: tuple[float, ...]
    event_weights: tuple[float, ...]
    hit_flags: tuple[bool, ...] | None
    catalog_delay_hours: int = CATALOG_DELAY_HOURS
    hit_tolerance_km: float = STRICT_HIT_TOLERANCE_KM
    episode_definition: str = FIXED_EPISODE_DEFINITION


@dataclass(frozen=True, slots=True)
class TimeRawScoreRow:
    fold_id: str
    issue_time_utc: datetime
    horizon_days: int
    magnitude_band: CountBand
    model_id: str
    distribution: CountDistribution
    observed_count: int
    expected_count: float
    count_log_score: float | None
    occurrence_brier: float | None
    count_bias: float
    status: ScoreStatus
    reason: str


@dataclass(frozen=True, slots=True)
class MagnitudeRawScoreRow:
    fold_id: str
    forecast_issue_time_utc: datetime
    model_id: str
    conditional_support: str
    event_ids: tuple[str, ...]
    event_log_probabilities: tuple[float, ...]
    log_probability_sum: float | None
    mean_log_probability: float | None
    m6_plus_probability: float | None
    mean_m6_plus_brier: float | None
    status: ScoreStatus


@dataclass(frozen=True, slots=True)
class JointModelSpec:
    joint_model_id: str
    count_model_id: str
    location_model_id: str
    count_distribution: CountDistribution
    magnitude_model_id: str = "M0_GR_GLOBAL"


FROZEN_JOINT_MODELS: Final[tuple[JointModelSpec, ...]] = (
    JointModelSpec("J0_U_P_GR", "T0_POISSON_EXPANDING", "L0_UNIFORM", "poisson"),
    JointModelSpec("J1_R_P_GR", "T0_POISSON_EXPANDING", "L1_REGIONAL_CONSTANT", "poisson"),
    JointModelSpec("J2_KDE_P_GR", "T0_POISSON_EXPANDING", "L2_KDE_CAUSAL", "poisson"),
    JointModelSpec("J3_R30_P_GR", "T0_POISSON_EXPANDING", "L3_B0_R30_CAUSAL", "poisson"),
    JointModelSpec("J4_KDE_NB_GR", "T1_NEGATIVE_BINOMIAL", "L2_KDE_CAUSAL", "nb2"),
)


@dataclass(frozen=True, slots=True)
class JointRawScoreRow:
    fold_id: str
    issue_time_utc: datetime
    horizon_days: int
    joint_model_id: str
    event_count: int
    count_distribution: CountDistribution
    count_log_score: float | None
    conditional_location_log_density_sum: float
    conditional_magnitude_log_probability_sum: float
    joint_log_score: float | None
    status: ScoreStatus


@dataclass(frozen=True, slots=True)
class DevelopmentRawScores:
    location: tuple[LocationRawScoreRow, ...]
    time: tuple[TimeRawScoreRow, ...]
    magnitude: tuple[MagnitudeRawScoreRow, ...]
    joint: tuple[JointRawScoreRow, ...]


@dataclass(frozen=True, slots=True)
class AuthorizedDevelopmentScores:
    """Raw scores returned only by the closed, context-loaded official path."""

    authorization: DevelopmentScoringAuthorization
    scores: DevelopmentRawScores


def _validate_target_set(targets: PrimaryDevelopmentTargets, grid: SpatialGrid) -> None:
    if not isinstance(targets, PrimaryDevelopmentTargets):
        raise TypeError("targets must be PrimaryDevelopmentTargets")
    if not isinstance(grid, SpatialGrid) or grid.grid_id != targets.grid_id:
        raise DevelopmentScoreError("score grid differs from the target-construction grid")
    expected_keys = {
        (fold_id, horizon, issue, magnitude_bin)
        for fold_id in DEVELOPMENT_FOLD_IDS
        for horizon in HORIZONS_DAYS
        for issue in _primary_issue_axis(fold_id, horizon)
        for magnitude_bin in ("M5_6", "M6_plus")
    }
    keys: set[tuple[str, int, datetime, str]] = set()
    seen: dict[tuple[int, str], set[str]] = defaultdict(set)
    for exposure in targets.exposures:
        if not isinstance(exposure, PrimaryExposureTargets):
            raise TypeError("target set contains a non-primary exposure")
        _validate_fold_id(exposure.fold_id)
        issue = _aware_utc(exposure.issue_time_utc, label="target issue")
        if (
            exposure.target_interval != TARGET_INTERVAL
            or exposure.horizon_days not in HORIZONS_DAYS
            or exposure.target_end_utc != issue + timedelta(days=exposure.horizon_days)
            or exposure.magnitude_bin not in _MAGNITUDE_BOUNDS
        ):
            raise DevelopmentScoreError("target exposure identity changed")
        key = (exposure.fold_id, exposure.horizon_days, issue, exposure.magnitude_bin)
        if key in keys:
            raise DevelopmentScoreError("target exposure is duplicated")
        keys.add(key)
        for event in exposure.events:
            if not isinstance(event, EpisodeTargetEvent):
                raise TypeError("target exposure contains an invalid event")
            if (
                not issue < event.origin_time_utc <= exposure.target_end_utc
                or not math.isfinite(event.longitude)
                or not math.isfinite(event.latitude)
                or not -180.0 <= event.longitude <= 180.0
                or not -90.0 <= event.latitude <= 90.0
                or not 0 <= event.cell_index < grid.cell_count
                or event.global_episode_member_count <= 0
            ):
                raise DevelopmentScoreError("target event lies outside (issue,issue+h]")
            horizon_key = (exposure.horizon_days, exposure.magnitude_bin)
            if event.event_id in seen[horizon_key]:
                raise DevelopmentScoreError("physical event duplicated on one primary horizon")
            seen[horizon_key].add(event.event_id)
    if keys != expected_keys:
        raise DevelopmentScoreError("target set omitted or introduced a primary exposure")


def _validate_source(source: DevelopmentPredictionSource) -> None:
    if not isinstance(source, DevelopmentPredictionSource):
        raise TypeError("predictions must implement DevelopmentPredictionSource")


def _location_prediction(
    source: DevelopmentPredictionSource,
    exposure: PrimaryExposureTargets,
    model_id: str,
) -> LocationForecast:
    result = source.location_forecast(
        fold_id=exposure.fold_id,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        model_id=model_id,
    )
    if not isinstance(result, LocationForecast) or (
        result.fold_id,
        result.issue_time_utc,
        result.horizon_days,
        result.model_id,
    ) != (
        exposure.fold_id,
        exposure.issue_time_utc,
        exposure.horizon_days,
        model_id,
    ):
        raise DevelopmentScoreError("location prediction snapshot identity changed")
    return result


def _count_prediction(
    source: DevelopmentPredictionSource,
    exposure: PrimaryExposureTargets,
    model_id: str,
    magnitude_band: CountBand,
) -> CountForecast:
    result = source.count_forecast(
        fold_id=exposure.fold_id,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        model_id=model_id,
        magnitude_band=magnitude_band,
    )
    if not isinstance(result, CountForecast) or (
        result.fold_id,
        result.issue_time_utc,
        result.horizon_days,
        result.model_id,
        result.magnitude_band,
    ) != (
        exposure.fold_id,
        exposure.issue_time_utc,
        exposure.horizon_days,
        model_id,
        magnitude_band,
    ):
        raise DevelopmentScoreError("count prediction snapshot identity changed")
    return result


def _magnitude_prediction(
    source: DevelopmentPredictionSource,
    *,
    fold_id: str,
    issue_time_utc: datetime,
    model_id: str,
) -> MagnitudeForecast:
    result = source.magnitude_forecast(
        fold_id=fold_id,
        issue_time_utc=issue_time_utc,
        model_id=model_id,
    )
    if not isinstance(result, MagnitudeForecast) or (
        result.fold_id,
        result.issue_time_utc,
        result.model_id,
    ) != (fold_id, issue_time_utc, model_id):
        raise DevelopmentScoreError(
            "magnitude prediction snapshot differs from the requested issue"
        )
    return result


def _score_locations(
    context: ScoringContext,
    *,
    targets: PrimaryDevelopmentTargets,
    grid: SpatialGrid,
    predictions: DevelopmentPredictionSource,
    model_ids: Sequence[str],
) -> tuple[LocationRawScoreRow, ...]:
    """Score exact-cell density and every area/view without pooling either axis."""

    _reauthorize(context)
    _validate_target_set(targets, grid)
    _validate_source(predictions)
    models = tuple(model_ids)
    if not models or any(not isinstance(value, str) or not value for value in models):
        raise ValueError("location model_ids must be non-empty strings")
    rows: list[LocationRawScoreRow] = []
    for exposure in targets.exposures:
        event_ids = tuple(item.event_id for item in exposure.events)
        for model_id in models:
            forecast = _location_prediction(predictions, exposure, model_id)
            evaluation = score_location_events(
                forecast.cell_relative_mass,
                grid,
                event_ids=event_ids,
                event_cell_indices=tuple(item.cell_index for item in exposure.events),
                episode_ids=tuple(item.episode_id for item in exposure.events),
                episode_member_counts=tuple(
                    item.global_episode_member_count for item in exposure.events
                ),
                is_episode_anchor=tuple(item.is_episode_anchor for item in exposure.events),
            )
            event_logs = tuple(item.log_density_per_km2 for item in evaluation.event_log_densities)
            if tuple(item.event_id for item in evaluation.event_log_densities) != event_ids:
                raise AssertionError("location metric changed the ordered physical event IDs")
            rows.append(
                LocationRawScoreRow(
                    fold_id=exposure.fold_id,
                    issue_time_utc=exposure.issue_time_utc,
                    horizon_days=exposure.horizon_days,
                    magnitude_bin=exposure.magnitude_bin,
                    model_id=model_id,
                    metric="spatial_log_density",
                    basis="all",
                    area_budget_km2=None,
                    actual_area_km2=None,
                    value=evaluation.mean_log_density_per_event,
                    event_count=len(event_ids),
                    hit_weight=None,
                    total_weight=None,
                    status=("evaluable" if event_ids else "not_evaluable"),
                    is_main_scientific_anchor=False,
                    scientific_anchor_id=None,
                    event_ids=event_ids,
                    event_log_densities_per_km2=event_logs,
                    episode_ids=tuple(item.episode_id for item in exposure.events),
                    global_episode_member_counts=tuple(
                        item.global_episode_member_count for item in exposure.events
                    ),
                    is_episode_anchor=tuple(item.is_episode_anchor for item in exposure.events),
                    event_cell_indices=tuple(item.cell_index for item in exposure.events),
                    event_longitudes=tuple(item.longitude for item in exposure.events),
                    event_latitudes=tuple(item.latitude for item in exposure.events),
                    event_weights=tuple(1.0 for _ in exposure.events),
                    hit_flags=None,
                )
            )
            prefix_by_budget = {
                item.budget_km2: frozenset(int(value) for value in item.selected_indices)
                for item in select_alarm_prefixes(forecast.cell_relative_mass, grid)
            }
            for recall in evaluation.alarm_recall:
                rows.append(
                    _location_recall_row(
                        exposure,
                        model_id,
                        recall,
                        event_logs,
                        prefix_by_budget[recall.area_budget_km2],
                    )
                )
    return tuple(rows)


def _location_recall_row(
    exposure: PrimaryExposureTargets,
    model_id: str,
    recall: AlarmRecallScore,
    event_logs: tuple[float, ...],
    selected_cell_indices: frozenset[int],
) -> LocationRawScoreRow:
    is_main = (
        exposure.magnitude_bin == "M5_6"
        and exposure.horizon_days == 30
        and recall.basis == "anchor"
        and recall.area_budget_km2 == 600_000.0
    )
    if recall.basis == "all":
        weights = tuple(1.0 for _ in exposure.events)
    elif recall.basis == "anchor":
        weights = tuple(1.0 if item.is_episode_anchor else 0.0 for item in exposure.events)
    elif recall.basis == "episode_balanced":
        weights = tuple(1.0 / item.global_episode_member_count for item in exposure.events)
    elif recall.basis == "subsequent":
        weights = tuple(0.0 if item.is_episode_anchor else 1.0 for item in exposure.events)
    else:
        raise AssertionError("location metric returned an unknown recall basis")
    hit_flags = tuple(item.cell_index in selected_cell_indices for item in exposure.events)
    if not math.isclose(math.fsum(weights), recall.total_weight, abs_tol=1.0e-12):
        raise AssertionError("raw event weights disagree with aggregate recall weight")
    if not math.isclose(
        math.fsum(weight for weight, hit in zip(weights, hit_flags, strict=True) if hit),
        recall.hit_weight,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("raw hit flags disagree with aggregate recall hit weight")
    return LocationRawScoreRow(
        fold_id=exposure.fold_id,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        magnitude_bin=exposure.magnitude_bin,
        model_id=model_id,
        metric="strict_recall",
        basis=recall.basis,
        area_budget_km2=recall.area_budget_km2,
        actual_area_km2=recall.actual_area_km2,
        value=recall.recall,
        event_count=len(exposure.events),
        hit_weight=recall.hit_weight,
        total_weight=recall.total_weight,
        status="evaluable" if recall.recall is not None else "not_evaluable",
        is_main_scientific_anchor=is_main,
        scientific_anchor_id=MAIN_SCIENTIFIC_ANCHOR if is_main else None,
        event_ids=tuple(item.event_id for item in exposure.events),
        event_log_densities_per_km2=event_logs,
        episode_ids=tuple(item.episode_id for item in exposure.events),
        global_episode_member_counts=tuple(
            item.global_episode_member_count for item in exposure.events
        ),
        is_episode_anchor=tuple(item.is_episode_anchor for item in exposure.events),
        event_cell_indices=tuple(item.cell_index for item in exposure.events),
        event_longitudes=tuple(item.longitude for item in exposure.events),
        event_latitudes=tuple(item.latitude for item in exposure.events),
        event_weights=weights,
        hit_flags=hit_flags,
    )


def _score_time(
    context: ScoringContext,
    *,
    targets: PrimaryDevelopmentTargets,
    grid: SpatialGrid,
    predictions: DevelopmentPredictionSource,
    model_ids: Sequence[str],
) -> tuple[TimeRawScoreRow, ...]:
    """Score all primary count windows, retaining every observed zero and NA."""

    _reauthorize(context)
    _validate_target_set(targets, grid)
    _validate_source(predictions)
    models = tuple(model_ids)
    if not models:
        raise ValueError("time model_ids must not be empty")
    rows: list[TimeRawScoreRow] = []
    for exposure in targets.exposures:
        for model_id in models:
            forecast = _count_prediction(predictions, exposure, model_id, exposure.magnitude_bin)
            if forecast.distribution == "poisson":
                if forecast.nb2_qualification is not None:
                    raise DevelopmentScoreError(
                        "Poisson time forecast cannot carry an NB2 qualification"
                    )
                result = score_poisson_exposure(
                    observed_count=len(exposure.events),
                    expected_count=forecast.expected_count,
                )
            elif forecast.distribution == "nb2":
                if forecast.nb2_qualification is None:
                    raise DevelopmentScoreError(
                        "NB2 time forecast requires its frozen qualification"
                    )
                result = score_nb2_exposure(
                    observed_count=len(exposure.events),
                    expected_count=forecast.expected_count,
                    qualification=forecast.nb2_qualification,
                )
            else:
                raise DevelopmentScoreError("unknown time count distribution")
            rows.append(
                TimeRawScoreRow(
                    fold_id=exposure.fold_id,
                    issue_time_utc=exposure.issue_time_utc,
                    horizon_days=exposure.horizon_days,
                    magnitude_band=exposure.magnitude_bin,
                    model_id=model_id,
                    distribution=forecast.distribution,
                    observed_count=result.observed_count,
                    expected_count=result.expected_count,
                    count_log_score=result.log_score,
                    occurrence_brier=result.at_least_one_brier,
                    count_bias=result.count_bias,
                    status=result.status,
                    reason=result.reason,
                )
            )
    return tuple(rows)


def _magnitude_row(
    *, fold_id: str, issue: datetime, evaluation: MagnitudeEvaluation
) -> MagnitudeRawScoreRow:
    # Kept local to avoid exposing a second, authorization-free scoring entry point.
    scores = evaluation.event_scores
    return MagnitudeRawScoreRow(
        fold_id=fold_id,
        forecast_issue_time_utc=issue,
        model_id=evaluation.model_id,
        conditional_support=evaluation.conditional_support,
        event_ids=tuple(item.event_id for item in scores),
        event_log_probabilities=tuple(item.log_probability for item in scores),
        log_probability_sum=evaluation.log_probability_sum,
        mean_log_probability=evaluation.mean_log_probability,
        m6_plus_probability=evaluation.m6_plus_probability,
        mean_m6_plus_brier=evaluation.mean_m6_plus_brier,
        status="evaluable" if scores else "not_evaluable",
    )


def _score_magnitudes(
    context: ScoringContext,
    *,
    targets: StandaloneMagnitudeTargets,
    predictions: DevelopmentPredictionSource,
) -> tuple[MagnitudeRawScoreRow, ...]:
    """Score M0 once per M>=4 event and M0|M5/M3 on one exact M>=5 set."""

    _reauthorize(context)
    _validate_source(predictions)
    if not isinstance(targets, StandaloneMagnitudeTargets):
        raise TypeError("targets must be StandaloneMagnitudeTargets")
    if tuple(item for item in targets.m0_m4_events if item.magnitude >= 5.0) != (
        targets.common_m5_events
    ):
        raise DevelopmentScoreError("M0|M5 and M3 common event population changed")
    grouped_m4: dict[tuple[str, datetime], list[StandaloneMagnitudeEvent]] = defaultdict(list)
    grouped_m5: dict[tuple[str, datetime], list[StandaloneMagnitudeEvent]] = defaultdict(list)
    for item in targets.m0_m4_events:
        _validate_fold_id(item.fold_id)
        if not item.forecast_issue_time_utc < item.origin_time_utc or item.magnitude < 4.0:
            raise DevelopmentScoreError("standalone M0 event assignment is not strictly causal")
        grouped_m4[(item.fold_id, item.forecast_issue_time_utc)].append(item)
    for item in targets.common_m5_events:
        if item.magnitude < 5.0:
            raise DevelopmentScoreError("common M5 tail contains a sub-M5 event")
        grouped_m5[(item.fold_id, item.forecast_issue_time_utc)].append(item)

    rows: list[MagnitudeRawScoreRow] = []
    for (fold_id, issue), events in sorted(grouped_m4.items()):
        forecast = _magnitude_prediction(
            predictions, fold_id=fold_id, issue_time_utc=issue, model_id="M0_GR_GLOBAL"
        )
        evaluation = score_m0_unique_events(
            forecast.model,
            event_ids=tuple(item.event_id for item in events),
            magnitudes=tuple(item.magnitude for item in events),
        )
        rows.append(_magnitude_row(fold_id=fold_id, issue=issue, evaluation=evaluation))
    for (fold_id, issue), events in sorted(grouped_m5.items()):
        m0 = _magnitude_prediction(
            predictions, fold_id=fold_id, issue_time_utc=issue, model_id="M0_GR_GLOBAL"
        )
        m3 = _magnitude_prediction(
            predictions, fold_id=fold_id, issue_time_utc=issue, model_id="M3_GR_LONG_M5"
        )
        event_ids = tuple(item.event_id for item in events)
        magnitudes = tuple(item.magnitude for item in events)
        comparison = compare_m0_m3_on_common_m5_support(
            m0.model, m3.model, event_ids=event_ids, magnitudes=magnitudes
        )
        if comparison.event_ids != event_ids:
            raise AssertionError("paired M0|M5 and M3 scoring changed event identities")
        rows.append(
            _magnitude_row(fold_id=fold_id, issue=issue, evaluation=comparison.m0_conditional_m5)
        )
        rows.append(
            _magnitude_row(fold_id=fold_id, issue=issue, evaluation=comparison.m3_conditional_m5)
        )
    return tuple(rows)


def _score_joint(
    context: ScoringContext,
    *,
    targets: PrimaryDevelopmentTargets,
    grid: SpatialGrid,
    predictions: DevelopmentPredictionSource,
    joint_models: Sequence[JointModelSpec] = FROZEN_JOINT_MODELS,
) -> tuple[JointRawScoreRow, ...]:
    """Score one M>=5 count plus conditional location and M0 magnitude once.

    Every component is requested with the exact same primary issue identity.
    Returning a later weekly snapshot from any adapter is mechanically rejected.
    """

    _reauthorize(context)
    _validate_target_set(targets, grid)
    _validate_source(predictions)
    specs = tuple(joint_models)
    if not specs or any(not isinstance(item, JointModelSpec) for item in specs):
        raise TypeError("joint_models must contain JointModelSpec values")
    by_key: dict[tuple[str, datetime, int], dict[MagnitudeBin, PrimaryExposureTargets]] = (
        defaultdict(dict)
    )
    for exposure in targets.exposures:
        key = (exposure.fold_id, exposure.issue_time_utc, exposure.horizon_days)
        if exposure.magnitude_bin in by_key[key]:
            raise DevelopmentScoreError("joint target magnitude bin is duplicated")
        by_key[key][exposure.magnitude_bin] = exposure
    rows: list[JointRawScoreRow] = []
    for (fold_id, issue, horizon), bands in sorted(
        by_key.items(),
        key=lambda item: (
            DEVELOPMENT_FOLD_IDS.index(item[0][0]),
            HORIZONS_DAYS.index(item[0][2]),
            item[0][1],
        ),
    ):
        if set(bands) != {"M5_6", "M6_plus"}:
            raise DevelopmentScoreError("joint target requires both disjoint M>=5 bins")
        events = tuple(
            sorted(
                (*bands["M5_6"].events, *bands["M6_plus"].events),
                key=lambda item: (item.origin_time_utc, item.event_id.encode("utf-8")),
            )
        )
        if len({item.event_id for item in events}) != len(events):
            raise DevelopmentScoreError("joint M>=5 target double-counted a physical event")
        representative = bands["M5_6"]
        for spec in specs:
            location = _location_prediction(predictions, representative, spec.location_model_id)
            count = _count_prediction(
                predictions,
                representative,
                spec.count_model_id,
                "M5_plus_for_joint",
            )
            magnitude = _magnitude_prediction(
                predictions,
                fold_id=fold_id,
                issue_time_utc=issue,
                model_id=spec.magnitude_model_id,
            )
            if count.distribution != spec.count_distribution:
                raise DevelopmentScoreError("joint count distribution differs from frozen J model")
            result = score_minimal_joint_m5(
                location.cell_relative_mass,
                grid,
                magnitude.model,
                event_ids=tuple(item.event_id for item in events),
                event_cell_indices=tuple(item.cell_index for item in events),
                event_magnitudes=tuple(item.magnitude for item in events),
                expected_count=count.expected_count,
                count_distribution=count.distribution,
                nb2_qualification=count.nb2_qualification,
            )
            rows.append(
                JointRawScoreRow(
                    fold_id=fold_id,
                    issue_time_utc=issue,
                    horizon_days=horizon,
                    joint_model_id=spec.joint_model_id,
                    event_count=result.event_count,
                    count_distribution=result.count_distribution,
                    count_log_score=result.count_log_score,
                    conditional_location_log_density_sum=(
                        result.conditional_location_log_density_sum
                    ),
                    conditional_magnitude_log_probability_sum=(
                        result.conditional_magnitude_log_probability_sum
                    ),
                    joint_log_score=result.joint_log_score,
                    status="evaluable" if result.joint_log_score is not None else "not_evaluable",
                )
            )
    return tuple(rows)


def _score_development(
    context: ScoringContext,
    *,
    primary_targets: PrimaryDevelopmentTargets,
    magnitude_targets: StandaloneMagnitudeTargets,
    grid: SpatialGrid,
    predictions: DevelopmentPredictionSource,
    location_model_ids: Sequence[str],
    time_model_ids: Sequence[str],
    joint_models: Sequence[JointModelSpec] = FROZEN_JOINT_MODELS,
) -> DevelopmentRawScores:
    """Return raw, unbootstrapped rows while retaining horizon/area identities."""

    _reauthorize(context)
    return DevelopmentRawScores(
        location=_score_locations(
            context,
            targets=primary_targets,
            grid=grid,
            predictions=predictions,
            model_ids=location_model_ids,
        ),
        time=_score_time(
            context,
            targets=primary_targets,
            grid=grid,
            predictions=predictions,
            model_ids=time_model_ids,
        ),
        magnitude=_score_magnitudes(context, targets=magnitude_targets, predictions=predictions),
        joint=_score_joint(
            context,
            targets=primary_targets,
            grid=grid,
            predictions=predictions,
            joint_models=joint_models,
        ),
    )


def score_authorized_development_from_context(
    context: ScoringContext,
) -> AuthorizedDevelopmentScores:
    """Load every official input internally and score only the sealed four-fold NPZs.

    This is the sole public target/scoring entry point.  It intentionally has no
    catalog, grid, issue-row, target, prediction-source, or model-list argument:
    callers therefore cannot pair a valid master seal with substituted truth or
    predictions.  Re-authorization brackets both loading and scoring so a
    missing, different, or changed payload fails closed.
    """

    authorization_before = _reauthorize(context)

    # Local imports avoid letting the score-blind prediction phase import target
    # construction.  The runtime loader validates the complete numeric-only NPZ
    # schema with allow_pickle=False before constructing its concrete adapter.
    from seismoflux.multitask_s1.development_predict import LOCATION_MODEL_IDS
    from seismoflux.multitask_s1.development_runtime import (
        NPZDevelopmentPredictionSource,
        _load_all_prediction_arrays,
    )
    from seismoflux.multitask_s1.runner_inputs import (
        EXPECTED_25KM_CELL_COUNT,
        load_s1_runner_inputs,
    )

    inputs = load_s1_runner_inputs(
        project_root=context.project_root,
        data_root=context.data_root,
    )
    grid = inputs.spatial_domain.operational_grid
    if (
        inputs.location_grid.cell_count != EXPECTED_25KM_CELL_COUNT
        or grid.cell_count != EXPECTED_25KM_CELL_COUNT
    ):
        raise DevelopmentScoreError("official scoring grid cell count changed")
    arrays = _load_all_prediction_arrays(
        context.output_root,
        cell_count=EXPECTED_25KM_CELL_COUNT,
    )
    source = NPZDevelopmentPredictionSource(
        arrays,
        cell_count=EXPECTED_25KM_CELL_COUNT,
    )
    if source.fold_ids != DEVELOPMENT_FOLD_IDS:
        raise DevelopmentScoreError("official scoring requires four ordered prediction folds")

    authorization_loaded = _reauthorize(context)
    if authorization_loaded != authorization_before:
        raise DevelopmentScoreError("prediction authorization changed while inputs were loaded")
    primary_targets = _build_primary_exposure_targets(
        context,
        catalog=inputs.catalog,
        primary_issue_rows=inputs.outer_issues,
        grid=grid,
        locator=inputs.spatial_domain.locator,
    )
    magnitude_targets = _assign_standalone_magnitude_targets(
        context,
        catalog=inputs.catalog,
        scheduled_issue_times_by_fold=source.weekly_issue_times_by_fold(),
    )
    scores = _score_development(
        context,
        primary_targets=primary_targets,
        magnitude_targets=magnitude_targets,
        grid=grid,
        predictions=source,
        location_model_ids=LOCATION_MODEL_IDS,
        time_model_ids=("T0_POISSON_EXPANDING", "T1_NEGATIVE_BINOMIAL"),
        joint_models=FROZEN_JOINT_MODELS,
    )
    authorization_after = _reauthorize(context)
    if authorization_after != authorization_loaded:
        raise DevelopmentScoreError("prediction authorization changed during scoring")
    return AuthorizedDevelopmentScores(
        authorization=authorization_after,
        scores=scores,
    )


__all__ = [
    "CATALOG_DELAY_HOURS",
    "FIXED_EPISODE_DEFINITION",
    "FROZEN_JOINT_MODELS",
    "MAIN_SCIENTIFIC_ANCHOR",
    "STRICT_HIT_TOLERANCE_KM",
    "TARGET_INTERVAL",
    "AuthorizedDevelopmentScores",
    "CountForecast",
    "DevelopmentPredictionSource",
    "DevelopmentRawScores",
    "DevelopmentScoreError",
    "EpisodeTargetEvent",
    "EventCellLocator",
    "JointModelSpec",
    "JointRawScoreRow",
    "LocationForecast",
    "LocationRawScoreRow",
    "MagnitudeForecast",
    "MagnitudeRawScoreRow",
    "PrimaryDevelopmentTargets",
    "PrimaryExposureTargets",
    "ScoringContext",
    "StandaloneMagnitudeEvent",
    "StandaloneMagnitudeTargets",
    "TimeRawScoreRow",
    "score_authorized_development_from_context",
]
