"""Score-blind sample ledgers for the multi-task, multi-data S0 protocol.

The functions in this module only describe catalog coverage and frozen sample
membership.  They deliberately contain no model, feature score, or test-set
selection logic.  All temporal cutoffs are explicit inputs so a caller cannot
silently substitute the newest catalog state for what was available at an
earlier issue time.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Geod

from seismoflux.data.common import canonical_json_bytes

CATALOG_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "longitude",
    "latitude",
    "magnitude",
    "inside_study_area",
)
OPTIONAL_QUALITY_COLUMNS: Final[tuple[str, ...]] = ("depth_km", "magnitude_type")
AUTHORITATIVE_CATALOG_FILE_SHA256: Final[str] = (
    "2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347"
)
AUTHORITATIVE_CATALOG_ROW_COUNT: Final[int] = 40_898
AUTHORITATIVE_CATALOG_SCHEMA: Final[tuple[tuple[str, str], ...]] = (
    ("event_id", "string"),
    ("origin_time_utc", "timestamp[us, tz=UTC]"),
    ("available_at", "timestamp[us, tz=UTC]"),
    ("origin_time_local", "timestamp[us, tz=+08:00]"),
    ("longitude", "double"),
    ("latitude", "double"),
    ("depth_km", "double"),
    ("magnitude", "double"),
    ("magnitude_type", "string"),
    ("place", "string"),
    ("catalog_sources", "list<element: string>"),
    ("inside_study_area", "bool"),
    ("dedup_confidence", "string"),
    ("anchor_source_record_id", "string"),
    ("quality_flags", "list<element: string>"),
)
DEFAULT_HORIZONS_DAYS: Final[tuple[int, ...]] = (7, 30, 90, 180, 365)
DEFAULT_MAGNITUDE_BINS: Final[dict[str, tuple[float, float | None]]] = {
    "m4_plus": (4.0, None),
    "m5_6": (5.0, 6.0),
    "m6_plus": (6.0, None),
}
EPISODE_MAX_DAYS: Final[int] = 30
EPISODE_MAX_DISTANCE_KM: Final[float] = 75.0
LEDGER_HASH_DOMAIN: Final[str] = "seismoflux.multitask-s0-ledger.v1"
EPISODE_HASH_DOMAIN: Final[str] = "seismoflux.multitask-s0-episode.v1"


def _utc(value: object, *, label: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid timestamp") from exc
    if result.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return result.tz_convert("UTC")


def _iso_utc(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_ledger_sha256(value: Mapping[str, object]) -> str:
    """Return the canonical content hash, excluding a prior hash field."""

    body = dict(value)
    body.pop("content_sha256", None)
    payload = {"domain": LEDGER_HASH_DOMAIN, "ledger": body}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CATALOG_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"earthquake catalog is missing columns: {missing}")
    retained = [*CATALOG_COLUMNS, *(name for name in OPTIONAL_QUALITY_COLUMNS if name in frame)]
    result = frame.loc[:, retained].copy()
    if result.empty:
        raise ValueError("earthquake catalog must not be empty")
    if result.loc[:, CATALOG_COLUMNS].isna().any().any():
        raise ValueError("earthquake catalog ledger columns must not contain missing values")
    result["event_id"] = result["event_id"].astype("string")
    if (result["event_id"].str.len() == 0).any() or result["event_id"].duplicated().any():
        raise ValueError("event_id values must be non-empty and unique")
    result["origin_time_utc"] = pd.to_datetime(result["origin_time_utc"], utc=True, errors="raise")
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True, errors="raise")
    if (result["available_at"] < result["origin_time_utc"]).any():
        raise ValueError("available_at cannot precede origin_time_utc")
    for column in ("longitude", "latitude", "magnitude"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if not np.isfinite(result[column].to_numpy()).all():
            raise ValueError(f"{column} must contain only finite values")
    if ((result["longitude"] < -180.0) | (result["longitude"] > 180.0)).any():
        raise ValueError("longitude is outside [-180, 180]")
    if ((result["latitude"] < -90.0) | (result["latitude"] > 90.0)).any():
        raise ValueError("latitude is outside [-90, 90]")
    if not pd.api.types.is_bool_dtype(result["inside_study_area"]):
        values = set(result["inside_study_area"].unique().tolist())
        if not values <= {True, False}:
            raise ValueError("inside_study_area must be boolean")
        result["inside_study_area"] = result["inside_study_area"].astype(bool)
    if "depth_km" in result:
        result["depth_km"] = pd.to_numeric(result["depth_km"], errors="coerce")
    if "magnitude_type" in result:
        result["magnitude_type"] = result["magnitude_type"].astype("string")
    return result.sort_values(
        ["origin_time_utc", "event_id"], kind="mergesort", ignore_index=True
    )


def load_catalog_frame(path: str | Path) -> pd.DataFrame:
    """Load only the frozen S0 allowlist from an injected parquet path."""

    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)
    available_columns = set(pq.read_schema(catalog_path).names)
    selected = [
        *CATALOG_COLUMNS,
        *(name for name in OPTIONAL_QUALITY_COLUMNS if name in available_columns),
    ]
    frame = pd.read_parquet(catalog_path, columns=selected)
    return _validate_catalog(frame)


def verify_authoritative_catalog_identity(path: str | Path) -> dict[str, object]:
    """Fail closed unless the parquet is the frozen 40,898-row Stage-1 catalog."""

    catalog_path = Path(path)
    file_sha256 = _sha256_file(catalog_path)
    if file_sha256 != AUTHORITATIVE_CATALOG_FILE_SHA256:
        raise ValueError("authoritative earthquake catalog file SHA-256 mismatch")
    parquet_file = pq.ParquetFile(catalog_path)
    row_count = parquet_file.metadata.num_rows
    if row_count != AUTHORITATIVE_CATALOG_ROW_COUNT:
        raise ValueError("authoritative earthquake catalog row-count mismatch")
    schema = tuple((field.name, str(field.type)) for field in parquet_file.schema_arrow)
    if schema != AUTHORITATIVE_CATALOG_SCHEMA:
        raise ValueError("authoritative earthquake catalog Arrow schema mismatch")
    return {
        "status": "verified_authoritative_fail_closed",
        "file_sha256": file_sha256,
        "row_count": row_count,
        "arrow_schema": [{"name": name, "type": kind} for name, kind in schema],
    }


def filter_catalog(
    frame: pd.DataFrame,
    *,
    origin_start: object | None = None,
    origin_end: object | None = None,
    available_by: object | None = None,
    magnitude_minimum: float | None = None,
    magnitude_maximum_exclusive: float | None = None,
    study_area_only: bool = True,
) -> pd.DataFrame:
    """Apply half-open time/magnitude boundaries and an explicit availability cutoff."""

    catalog = _validate_catalog(frame)
    mask = np.ones(len(catalog), dtype=bool)
    if study_area_only:
        mask &= catalog["inside_study_area"].to_numpy(dtype=bool)
    if origin_start is not None:
        mask &= (catalog["origin_time_utc"] >= _utc(origin_start, label="origin_start")).to_numpy()
    if origin_end is not None:
        mask &= (catalog["origin_time_utc"] < _utc(origin_end, label="origin_end")).to_numpy()
    if available_by is not None:
        mask &= (catalog["available_at"] <= _utc(available_by, label="available_by")).to_numpy()
    if magnitude_minimum is not None:
        minimum = float(magnitude_minimum)
        if not math.isfinite(minimum):
            raise ValueError("magnitude_minimum must be finite")
        mask &= catalog["magnitude"].to_numpy() >= minimum
    if magnitude_maximum_exclusive is not None:
        maximum = float(magnitude_maximum_exclusive)
        if not math.isfinite(maximum):
            raise ValueError("magnitude_maximum_exclusive must be finite")
        if magnitude_minimum is not None and maximum <= float(magnitude_minimum):
            raise ValueError("magnitude maximum must exceed the minimum")
        mask &= catalog["magnitude"].to_numpy() < maximum
    return catalog.loc[mask].reset_index(drop=True)


def normalize_magnitude_bins(
    bins: Mapping[str, Sequence[float | None]] | None = None,
) -> dict[str, tuple[float, float | None]]:
    """Validate magnitude bins as ``[minimum, maximum)`` intervals."""

    raw = DEFAULT_MAGNITUDE_BINS if bins is None else bins
    result: dict[str, tuple[float, float | None]] = {}
    for name, bounds in raw.items():
        if not isinstance(name, str) or not name or len(bounds) != 2:
            raise ValueError("magnitude bins require a name and two bounds")
        minimum = float(bounds[0]) if bounds[0] is not None else float("nan")
        maximum = None if bounds[1] is None else float(bounds[1])
        if not math.isfinite(minimum) or (
            maximum is not None and (not math.isfinite(maximum) or maximum <= minimum)
        ):
            raise ValueError(f"invalid magnitude bin: {name}")
        result[name] = (minimum, maximum)
    return result


def summarize_coverage(frame: pd.DataFrame) -> dict[str, object]:
    """Describe the catalog's temporal and spatial coverage without model output."""

    catalog = _validate_catalog(frame)
    inside = catalog["inside_study_area"].to_numpy(dtype=bool)
    quality: dict[str, object] = {}
    for column in OPTIONAL_QUALITY_COLUMNS:
        if column in catalog:
            non_null = int(catalog[column].notna().sum())
            quality[column] = {
                "status": "computed",
                "non_null_count": non_null,
                "non_null_fraction": non_null / len(catalog),
            }
        else:
            quality[column] = {"status": "not_computed_column_absent"}
    return {
        "row_count": len(catalog),
        "inside_study_area_count": int(inside.sum()),
        "outside_study_area_count": int((~inside).sum()),
        "origin_time_min_utc": _iso_utc(catalog["origin_time_utc"].min()),
        "origin_time_max_utc": _iso_utc(catalog["origin_time_utc"].max()),
        "available_at_min_utc": _iso_utc(catalog["available_at"].min()),
        "available_at_max_utc": _iso_utc(catalog["available_at"].max()),
        "magnitude_minimum": float(catalog["magnitude"].min()),
        "magnitude_maximum": float(catalog["magnitude"].max()),
        "quality_completeness": quality,
    }


def summarize_sample_funnel(
    frame: pd.DataFrame,
    *,
    catalog_start: object,
    catalog_cutoff: object,
    magnitude_bins: Mapping[str, Sequence[float | None]] | None = None,
) -> dict[str, object]:
    """Count each deterministic filter step and the formal magnitude bins."""

    catalog = _validate_catalog(frame)
    start = _utc(catalog_start, label="catalog_start")
    cutoff = _utc(catalog_cutoff, label="catalog_cutoff")
    if cutoff < start:
        raise ValueError("catalog_cutoff must not precede catalog_start")
    inside = catalog["inside_study_area"]
    in_time = (catalog["origin_time_utc"] >= start) & (catalog["origin_time_utc"] <= cutoff)
    available = catalog["available_at"] <= cutoff
    eligible = inside & in_time & available
    bin_counts: dict[str, int] = {}
    for name, (minimum, maximum) in normalize_magnitude_bins(magnitude_bins).items():
        selected = eligible & (catalog["magnitude"] >= minimum)
        if maximum is not None:
            selected &= catalog["magnitude"] < maximum
        bin_counts[name] = int(selected.sum())
    return {
        "all_catalog_rows": len(catalog),
        "inside_study_area_rows": int(inside.sum()),
        "inside_and_origin_range_rows": int((inside & in_time).sum()),
        "inside_origin_range_and_available_rows": int(eligible.sum()),
        "excluded_late_availability_rows": int((inside & in_time & ~available).sum()),
        "catalog_start_inclusive_utc": _iso_utc(start),
        "catalog_cutoff_inclusive_utc": _iso_utc(cutoff),
        "magnitude_bin_counts": bin_counts,
    }


def summarize_quality_by_catalog_panel(
    frame: pd.DataFrame,
    *,
    catalog_start: object,
    catalog_cutoff: object,
    magnitude_bins: Mapping[str, Sequence[float | None]] | None = None,
) -> dict[str, object]:
    """Report depth/type completeness on each frozen scientific catalog panel."""

    catalog = _validate_catalog(frame)
    bins = normalize_magnitude_bins(magnitude_bins)

    def quality(panel: pd.DataFrame) -> dict[str, object]:
        result: dict[str, object] = {"row_count": len(panel)}
        for column in OPTIONAL_QUALITY_COLUMNS:
            if column not in panel:
                result[column] = {"status": "not_computed_column_absent"}
                continue
            non_null = int(panel[column].notna().sum())
            result[column] = {
                "status": "computed",
                "non_null_count": non_null,
                "non_null_fraction": non_null / len(panel) if len(panel) else None,
            }
        return result

    study_era = filter_catalog(
        catalog,
        origin_start=catalog_start,
        available_by=catalog_cutoff,
        study_area_only=True,
    )
    panels: dict[str, object] = {
        "full_catalog_all_eras_and_domains": quality(catalog),
        "study_area_1970_plus": quality(study_era),
    }
    for name, (minimum, maximum) in bins.items():
        selected = study_era[study_era["magnitude"] >= minimum]
        if maximum is not None:
            selected = selected[selected["magnitude"] < maximum]
        panels[name] = quality(selected)
    return panels


def build_episodes(
    frame: pd.DataFrame,
    *,
    max_time_days: int = EPISODE_MAX_DAYS,
    max_distance_km: float = EPISODE_MAX_DISTANCE_KM,
) -> list[dict[str, object]]:
    """Build deterministic fixed-anchor WGS84 episodes before fold assignment.

    Events are processed by ``(origin_time, event_id UTF-8)``.  Each event
    searches all existing anchors that directly satisfy both limits.  It opens
    a new episode only when no anchor qualifies; otherwise it joins the minimum
    ``(distance, time difference, anchor event_id UTF-8)`` candidate.  Members
    never extend an episode through a transitive chain.
    """

    if frame.empty:
        missing = [column for column in CATALOG_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"earthquake catalog is missing columns: {missing}")
        return []
    catalog = _validate_catalog(frame)
    if max_time_days < 0 or not math.isfinite(max_distance_km) or max_distance_km < 0.0:
        raise ValueError("episode edge limits must be non-negative")
    origins_ns = pd.DatetimeIndex(catalog["origin_time_utc"]).as_unit("ns").asi8
    maximum_ns = int(timedelta(days=max_time_days).total_seconds() * 1_000_000_000)
    longitude = catalog["longitude"].to_numpy(dtype=float)
    latitude = catalog["latitude"].to_numpy(dtype=float)
    geod = Geod(ellps="WGS84")
    anchor_indices: list[int] = []
    grouped: list[list[int]] = []
    for event_index in range(len(catalog)):
        candidates: list[tuple[float, int, bytes, int]] = []
        for episode_index, anchor_index in enumerate(anchor_indices):
            time_difference_ns = int(origins_ns[event_index] - origins_ns[anchor_index])
            if time_difference_ns < 0 or time_difference_ns > maximum_ns:
                continue
            _, _, raw_distance = geod.inv(
                float(longitude[anchor_index]),
                float(latitude[anchor_index]),
                float(longitude[event_index]),
                float(latitude[event_index]),
            )
            distance_metres = abs(float(raw_distance))
            if distance_metres <= max_distance_km * 1_000.0:
                candidates.append(
                    (
                        distance_metres,
                        time_difference_ns,
                        str(catalog.at[anchor_index, "event_id"]).encode("utf-8"),
                        episode_index,
                    )
                )
        if not candidates:
            anchor_indices.append(event_index)
            grouped.append([event_index])
        else:
            selected_episode = min(candidates)[3]
            grouped[selected_episode].append(event_index)
    episodes: list[dict[str, object]] = []
    for indices in grouped:
        ordered_indices = sorted(
            indices,
            key=lambda index: (
                catalog.at[index, "origin_time_utc"],
                str(catalog.at[index, "event_id"]).encode("utf-8"),
            ),
        )
        member_ids = [str(catalog.at[index, "event_id"]) for index in ordered_indices]
        episode_id = hashlib.sha256(
            canonical_json_bytes(
                {"domain": EPISODE_HASH_DOMAIN, "ordered_member_event_ids": sorted(member_ids)}
            )
        ).hexdigest()
        anchor_index = ordered_indices[0]
        episodes.append(
            {
                "episode_id": episode_id,
                "anchor_event_id": str(catalog.at[anchor_index, "event_id"]),
                "anchor_time_utc": _iso_utc(catalog.at[anchor_index, "origin_time_utc"]),
                "anchor_magnitude": float(catalog.at[anchor_index, "magnitude"]),
                "maximum_magnitude": float(catalog.loc[ordered_indices, "magnitude"].max()),
                "member_count": len(ordered_indices),
                "member_event_ids": member_ids,
                "member_time_min_utc": _iso_utc(catalog.at[anchor_index, "origin_time_utc"]),
                "member_time_max_utc": _iso_utc(
                    max(catalog.at[index, "origin_time_utc"] for index in ordered_indices)
                ),
            }
        )
    return sorted(
        episodes,
        key=lambda item: (str(item["anchor_time_utc"]), str(item["episode_id"])),
    )


def summarize_episode_samples(episodes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Report first-event and episode-balanced sample sizes for one target panel."""

    member_counts = [int(item["member_count"]) for item in episodes]
    total_events = sum(member_counts)
    anchor_spans_days = [
        (
            _utc(item["member_time_max_utc"], label="episode member_time_max")
            - _utc(item["anchor_time_utc"], label="episode anchor_time")
        ).total_seconds()
        / 86_400.0
        for item in episodes
    ]
    if any(span < 0.0 or span > EPISODE_MAX_DAYS for span in anchor_spans_days):
        raise ValueError("episode member span violates the frozen fixed-anchor time limit")
    member_count_histogram = {
        str(size): member_counts.count(size) for size in sorted(set(member_counts))
    }
    return {
        "event_count": total_events,
        "episode_count": len(episodes),
        "first_event_count": len(episodes),
        "subsequent_event_count": total_events - len(episodes),
        "episode_balanced_total_weight": len(episodes),
        "maximum_episode_member_count": max(member_counts, default=0),
        "maximum_anchor_span_days": max(anchor_spans_days) if anchor_spans_days else None,
        "episode_member_count_histogram": member_count_histogram,
    }


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One expanding-time fold with an explicit leakage embargo."""

    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    embargo_days: int
    issue_frequency_days: int = 7
    role: str = "development"
    horizon_days: tuple[int, ...] = DEFAULT_HORIZONS_DAYS
    parameter_selection_end: pd.Timestamp | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        default_train_start: object = "1970-01-01T00:00:00Z",
        derived_assessment_end: object | None = None,
        default_horizon_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    ) -> FoldSpec:
        def field(*names: str) -> object:
            for name in names:
                if name in value:
                    return value[name]
            raise ValueError(f"fold is missing one of fields: {names}")

        raw_horizons = value.get("horizon_days", default_horizon_days)
        if not isinstance(raw_horizons, Sequence) or isinstance(raw_horizons, str | bytes):
            raise ValueError("fold horizon_days must be a sequence")
        raw_assessment_end = field(
            "assessment_end_utc",
            "evaluation_end",
            "target_block_end_exclusive",
        )
        if raw_assessment_end == "derived_from_catalog_truth_cutoff":
            if derived_assessment_end is None:
                raise ValueError("derived assessment end requires the frozen truth cutoff")
            raw_assessment_end = derived_assessment_end
        result = cls(
            fold_id=str(field("fold_id", "id")),
            train_start=_utc(
                value.get("train_start_utc", value.get("train_start", default_train_start)),
                label="train_start_utc",
            ),
            train_end=_utc(
                field("train_end_utc", "train_end", "fit_history_end_exclusive"),
                label="train_end_utc",
            ),
            evaluation_start=_utc(
                field("assessment_start_utc", "evaluation_start", "target_block_start"),
                label="assessment_start_utc",
            ),
            evaluation_end=_utc(raw_assessment_end, label="assessment_end_utc"),
            embargo_days=int(value.get("embargo_days", 0)),
            issue_frequency_days=int(value.get("issue_frequency_days", 7)),
            role=str(value.get("role", "development")),
            horizon_days=tuple(int(item) for item in raw_horizons),
            parameter_selection_end=(
                _utc(
                    value["parameter_selection_end_utc"],
                    label="parameter_selection_end_utc",
                )
                if "parameter_selection_end_utc" in value
                else None
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            not self.fold_id
            or not self.role
            or self.embargo_days < 0
            or self.issue_frequency_days <= 0
        ):
            raise ValueError("fold identifiers/frequencies are invalid")
        if (
            not self.horizon_days
            or len(self.horizon_days) != len(set(self.horizon_days))
            or any(value <= 0 for value in self.horizon_days)
        ):
            raise ValueError("fold horizon_days must be unique positive integers")
        if not self.train_start < self.train_end <= self.evaluation_start < self.evaluation_end:
            raise ValueError(
                "fold times must satisfy train_start < train_end <= eval_start < eval_end"
            )
        if self.parameter_selection_end is not None:
            if self.parameter_selection_end > self.train_end:
                raise ValueError("parameter selection end cannot follow training-label end")
            if (
                self.parameter_selection_end + pd.Timedelta(days=self.embargo_days)
                > self.evaluation_start
            ):
                raise ValueError("parameter selection end violates the frozen embargo")


def _issue_axis(fold: FoldSpec) -> list[pd.Timestamp]:
    if fold.issue_frequency_days != 7:
        raise ValueError("catalog issue frequency must remain frozen at seven days")
    local_start = fold.evaluation_start.tz_convert("Asia/Shanghai")
    candidate = local_start.normalize()
    candidate += pd.Timedelta(days=(3 - candidate.weekday()) % 7)
    if candidate < local_start:
        candidate += pd.Timedelta(days=7)
    local_end = fold.evaluation_end.tz_convert("Asia/Shanghai")
    issues = list(pd.date_range(candidate, local_end, inclusive="left", freq="7D"))
    if any(
        issue.weekday() != 3
        or issue.hour != 0
        or issue.minute != 0
        or issue.second != 0
        for issue in issues
    ):
        raise AssertionError("generated catalog issue is not Thursday 00:00 Asia/Shanghai")
    return [issue.tz_convert("UTC") for issue in issues]


def _episode_owners(
    episodes: Sequence[Mapping[str, object]], folds: Sequence[FoldSpec]
) -> dict[str, str | None]:
    owners: dict[str, str | None] = {}
    for episode in episodes:
        anchor = _utc(episode["anchor_time_utc"], label="episode anchor")
        matches = [
            fold.fold_id
            for fold in folds
            if fold.evaluation_start <= anchor < fold.evaluation_end
        ]
        if len(matches) > 1:
            raise ValueError("evaluation folds overlap at an episode anchor")
        owners[str(episode["episode_id"])] = matches[0] if matches else None
    return owners


def summarize_fold_maturity(
    frame: pd.DataFrame,
    *,
    folds: Sequence[FoldSpec | Mapping[str, object]],
    truth_cutoff: object,
    horizons_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    magnitude_bins: Mapping[str, Sequence[float | None]] | None = None,
    train_history_start: object = "1970-01-01T00:00:00Z",
) -> list[dict[str, object]]:
    """Count causal train rows and mature issue/target samples for frozen folds.

    Events are joined to globally constructed episodes before fold ownership is
    assigned.  Therefore a physical episode crossing a fold boundary belongs
    wholly to the fold containing its first event and is never split into two
    independent episode samples.
    """

    catalog = _validate_catalog(frame)
    cutoff = _utc(truth_cutoff, label="truth_cutoff")
    normalized_folds = [
        item
        if isinstance(item, FoldSpec)
        else FoldSpec.from_mapping(
            item,
            default_train_start=train_history_start,
            derived_assessment_end=cutoff,
            default_horizon_days=horizons_days,
        )
        for item in folds
    ]
    for fold in normalized_folds:
        fold.validate()
    if len({fold.fold_id for fold in normalized_folds}) != len(normalized_folds):
        raise ValueError("fold IDs must be unique")
    ordered = sorted(normalized_folds, key=lambda item: item.evaluation_start)
    for left, right in pairwise(ordered):
        if left.evaluation_end > right.evaluation_start:
            raise ValueError("evaluation folds must not overlap")
    horizons = tuple(int(value) for value in horizons_days)
    if not horizons or len(horizons) != len(set(horizons)) or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be unique positive day counts")
    bins = normalize_magnitude_bins(magnitude_bins)
    episode_maps: dict[
        str,
        tuple[dict[str, str | None], dict[str, str], dict[str, tuple[str, int]]],
    ] = {}
    for name in ("m5_6", "m6_plus"):
        if name not in bins:
            continue
        minimum, maximum = bins[name]
        episode_source = filter_catalog(
            catalog,
            available_by=cutoff,
            magnitude_minimum=minimum,
            magnitude_maximum_exclusive=maximum,
            study_area_only=True,
        )
        episodes = build_episodes(episode_source)
        owner_by_episode = _episode_owners(episodes, ordered)
        event_to_episode = {
            str(event_id): str(episode["episode_id"])
            for episode in episodes
            for event_id in episode["member_event_ids"]  # type: ignore[union-attr]
        }
        episode_metadata = {
            str(episode["episode_id"]): (
                str(episode["anchor_event_id"]),
                int(episode["member_count"]),
            )
            for episode in episodes
        }
        episode_maps[name] = (owner_by_episode, event_to_episode, episode_metadata)

    records: list[dict[str, object]] = []
    for fold in normalized_folds:
        train = filter_catalog(
            catalog,
            origin_start=fold.train_start,
            origin_end=fold.train_end,
            available_by=fold.train_end,
            study_area_only=True,
        )
        issues = _issue_axis(fold)
        fold_record: dict[str, object] = {
            "fold_id": fold.fold_id,
            "role": fold.role,
            "train_start_inclusive_utc": _iso_utc(fold.train_start),
            "train_end_exclusive_utc": _iso_utc(fold.train_end),
            "evaluation_start_inclusive_utc": _iso_utc(fold.evaluation_start),
            "evaluation_end_exclusive_utc": _iso_utc(fold.evaluation_end),
            "embargo_days": fold.embargo_days,
            "parameter_selection_end_utc": (
                _iso_utc(fold.parameter_selection_end)
                if fold.parameter_selection_end is not None
                else None
            ),
            "causal_feature_history_rule": "available_at_not_after_each_issue_time",
            "catalog_issue_calendar": {
                "timezone": "Asia/Shanghai",
                "weekday": "THU",
                "local_time": "00:00:00",
                "frequency_days": 7,
                "issue_times_utc": [_iso_utc(issue) for issue in issues],
            },
            "training_event_count_all_magnitudes": len(train),
            "training_counts_by_magnitude_bin": {},
            "horizons": {},
        }
        train_counts = fold_record["training_counts_by_magnitude_bin"]
        assert isinstance(train_counts, dict)
        for name, (minimum, maximum) in bins.items():
            mask = train["magnitude"] >= minimum
            if maximum is not None:
                mask &= train["magnitude"] < maximum
            train_counts[name] = int(mask.sum())

        horizon_records = fold_record["horizons"]
        assert isinstance(horizon_records, dict)
        fold_horizons = tuple(value for value in horizons if value in fold.horizon_days)
        if set(fold.horizon_days) - set(horizons):
            raise ValueError("fold horizon_days must be included in the ledger horizons")
        for horizon in fold_horizons:
            maturity_limit = min(fold.evaluation_end, cutoff)
            mature_issues = [
                issue
                for issue in issues
                if issue + pd.Timedelta(days=horizon) <= maturity_limit
            ]
            primary_issues: list[pd.Timestamp] = []
            for issue in mature_issues:
                if not primary_issues or issue >= primary_issues[-1] + pd.Timedelta(
                    days=horizon + 30
                ):
                    primary_issues.append(issue)
            horizon_record: dict[str, object] = {
                "scheduled_issue_count": len(issues),
                "mature_issue_count": len(mature_issues),
                "immature_issue_count": len(issues) - len(mature_issues),
                "last_mature_issue_utc": (
                    _iso_utc(mature_issues[-1]) if mature_issues else None
                ),
                "maturity_limit_utc": _iso_utc(maturity_limit),
                "operational_weekly": {
                    "statistical_status": "descriptive_operational_nonindependent",
                    "evaluable": bool(mature_issues),
                    "availability_status": (
                        "available" if mature_issues else "unavailable_no_mature_issue"
                    ),
                    "issue_count": len(mature_issues),
                    "magnitude_bins": {},
                },
                "primary_exposure": {
                    "statistical_status": "primary_independent_exposure_axis",
                    "evaluable": bool(primary_issues),
                    "availability_status": (
                        "available" if primary_issues else "unavailable_no_mature_issue"
                    ),
                    "selection_rule": "chronological_greedy_nonoverlap_plus_30d_guard",
                    "issue_count": len(primary_issues),
                    "issue_times_utc": [_iso_utc(issue) for issue in primary_issues],
                    "magnitude_bins": {},
                },
            }
            operational = horizon_record["operational_weekly"]
            primary = horizon_record["primary_exposure"]
            assert isinstance(operational, dict) and isinstance(primary, dict)
            for axis, selected_issues in (
                (operational, mature_issues),
                (primary, primary_issues),
            ):
                magnitude_records = axis["magnitude_bins"]
                assert isinstance(magnitude_records, dict)
                for name, (minimum, maximum) in bins.items():
                    if not selected_issues:
                        magnitude_records[name] = {
                            "evaluable": False,
                            "availability_status": "unavailable_no_mature_issue",
                            "issue_target_pair_count": None,
                            "unique_event_count": None,
                            "episode_sampling_status": (
                                "not_applicable_training_diagnostic_bin"
                                if name not in episode_maps
                                else "unavailable_no_mature_issue"
                            ),
                            "touched_episode_count": None,
                            "anchor_target_count": None,
                            "subsequent_target_event_count": None,
                            "episode_balanced_total_weight": None,
                        }
                        continue
                    unique_event_ids: set[str] = set()
                    issue_target_pair_count = 0
                    for issue in selected_issues:
                        target_end = issue + pd.Timedelta(days=horizon)
                        mask = (
                            catalog["inside_study_area"]
                            & (catalog["origin_time_utc"] > issue)
                            & (catalog["origin_time_utc"] <= target_end)
                            & (catalog["available_at"] <= cutoff)
                            & (catalog["magnitude"] >= minimum)
                        )
                        if maximum is not None:
                            mask &= catalog["magnitude"] < maximum
                        ids = catalog.loc[mask, "event_id"].astype(str).tolist()
                        issue_target_pair_count += len(ids)
                        unique_event_ids.update(ids)
                    if name in episode_maps:
                        owner_by_episode, event_to_episode, episode_metadata = episode_maps[name]
                        touched_episode_ids = {
                            event_to_episode[event_id]
                            for event_id in unique_event_ids
                            if event_id in event_to_episode
                            and owner_by_episode[event_to_episode[event_id]] == fold.fold_id
                        }
                        touched_target_event_ids = {
                            event_id
                            for event_id in unique_event_ids
                            if event_to_episode.get(event_id) in touched_episode_ids
                        }
                        anchor_target_count = sum(
                            episode_metadata[episode_id][0] in unique_event_ids
                            for episode_id in touched_episode_ids
                        )
                        subsequent_target_event_count = sum(
                            event_id != episode_metadata[event_to_episode[event_id]][0]
                            for event_id in touched_target_event_ids
                        )
                        episode_balanced_total_weight = sum(
                            1.0 / episode_metadata[event_to_episode[event_id]][1]
                            for event_id in touched_target_event_ids
                        )
                        episode_values: dict[str, object] = {
                            "episode_sampling_status": "formal_fixed_anchor_episode_population",
                            "touched_episode_count": len(touched_episode_ids),
                            "anchor_target_count": anchor_target_count,
                            "subsequent_target_event_count": subsequent_target_event_count,
                            "episode_balanced_total_weight": episode_balanced_total_weight,
                        }
                    else:
                        episode_values = {
                            "episode_sampling_status": "not_applicable_training_diagnostic_bin",
                            "touched_episode_count": None,
                            "anchor_target_count": None,
                            "subsequent_target_event_count": None,
                            "episode_balanced_total_weight": None,
                        }
                    magnitude_records[name] = {
                        "evaluable": True,
                        "availability_status": "available",
                        "issue_target_pair_count": issue_target_pair_count,
                        "unique_event_count": len(unique_event_ids),
                        **episode_values,
                    }
            horizon_records[str(horizon)] = horizon_record
        records.append(fold_record)
    return records


def build_s0_ledger(
    catalog_path: str | Path,
    *,
    catalog_start: object,
    catalog_cutoff: object,
    folds: Sequence[FoldSpec | Mapping[str, object]],
    horizons_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    magnitude_bins: Mapping[str, Sequence[float | None]] | None = None,
    require_authoritative_identity: bool = True,
) -> dict[str, object]:
    """Build a complete, serializable S0 catalog ledger and stable identity."""

    path = Path(catalog_path)
    identity = (
        verify_authoritative_catalog_identity(path)
        if require_authoritative_identity
        else {"status": "not_verified_non_authoritative_test_input"}
    )
    catalog = load_catalog_frame(path)
    bins = normalize_magnitude_bins(magnitude_bins)
    episode_summaries: dict[str, object] = {
        "m4_plus": {
            "episode_sampling_status": "not_applicable_training_diagnostic_bin",
            "event_count": None,
            "episode_count": None,
        }
    }
    for name in ("m5_6", "m6_plus"):
        if name not in bins:
            continue
        minimum, maximum = bins[name]
        eligible = filter_catalog(
            catalog,
            origin_start=catalog_start,
            origin_end=(
                _utc(catalog_cutoff, label="catalog_cutoff") + pd.Timedelta(nanoseconds=1)
            ),
            available_by=catalog_cutoff,
            magnitude_minimum=minimum,
            magnitude_maximum_exclusive=maximum,
            study_area_only=True,
        )
        episode_summaries[name] = {
            "episode_sampling_status": "formal_fixed_anchor_episode_population",
            **summarize_episode_samples(build_episodes(eligible)),
        }
    ledger: dict[str, object] = {
        "schema_version": 1,
        "ledger_type": "multitask_s0_catalog_sample_ledger",
        "score_blind": True,
        "input": {
            "filename": path.name,
            "file_sha256": _sha256_file(path),
            "required_columns": list(CATALOG_COLUMNS),
            "optional_quality_columns_present": [
                name for name in OPTIONAL_QUALITY_COLUMNS if name in catalog
            ],
            "authoritative_identity": identity,
        },
        "availability_semantics": {
            "stored_available_at_equals_origin_count": int(
                (catalog["available_at"] == catalog["origin_time_utc"]).sum()
            ),
            "stored_available_at_equals_origin_is_optimistic_proxy": True,
            "limitation": (
                "historical publication timestamps are unavailable; equality to origin "
                "does not prove real-time observability"
            ),
            "primary_catalog_latency_hours": 24,
            "catalog_latency_sensitivity_hours": [0, 24, 168],
            "issue_feature_cutoff_rule": "available_at<=T-latency",
        },
        "formal_horizons_days": [int(value) for value in horizons_days],
        "magnitude_bins": {
            name: {"minimum_inclusive": minimum, "maximum_exclusive": maximum}
            for name, (minimum, maximum) in bins.items()
        },
        "episode_definition": {
            "assignment_rule": (
                "online_fixed_anchor_non_transitive_min_"
                "distance_then_time_then_anchor_event_id_utf8"
            ),
            "distance_method": "pyproj.Geod(ellps=WGS84).inv",
            "edge_time_difference_max_days_inclusive": EPISODE_MAX_DAYS,
            "edge_distance_max_km_inclusive": EPISODE_MAX_DISTANCE_KM,
            "fold_assignment": "entire_episode_owned_by_first_event_evaluation_fold",
        },
        "coverage": summarize_coverage(catalog),
        "sample_funnel": summarize_sample_funnel(
            catalog,
            catalog_start=catalog_start,
            catalog_cutoff=catalog_cutoff,
            magnitude_bins=bins,
        ),
        "quality_by_catalog_panel": summarize_quality_by_catalog_panel(
            catalog,
            catalog_start=catalog_start,
            catalog_cutoff=catalog_cutoff,
            magnitude_bins=bins,
        ),
        "episode_summary_by_magnitude_bin": episode_summaries,
        "fold_maturity": summarize_fold_maturity(
            catalog,
            folds=folds,
            truth_cutoff=catalog_cutoff,
            horizons_days=horizons_days,
            magnitude_bins=bins,
        ),
    }
    ledger["content_sha256"] = stable_ledger_sha256(ledger)
    return ledger


__all__ = [
    "AUTHORITATIVE_CATALOG_FILE_SHA256",
    "AUTHORITATIVE_CATALOG_ROW_COUNT",
    "AUTHORITATIVE_CATALOG_SCHEMA",
    "CATALOG_COLUMNS",
    "DEFAULT_HORIZONS_DAYS",
    "DEFAULT_MAGNITUDE_BINS",
    "EPISODE_MAX_DAYS",
    "EPISODE_MAX_DISTANCE_KM",
    "FoldSpec",
    "build_episodes",
    "build_s0_ledger",
    "filter_catalog",
    "load_catalog_frame",
    "normalize_magnitude_bins",
    "stable_ledger_sha256",
    "summarize_coverage",
    "summarize_episode_samples",
    "summarize_fold_maturity",
    "summarize_quality_by_catalog_panel",
    "summarize_sample_funnel",
    "verify_authoritative_catalog_identity",
]
