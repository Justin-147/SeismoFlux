"""D1 time/space placebo reconstruction, refitting orchestration and inference.

The placebo layer deliberately changes anomaly history only.  Earthquake targets,
causal backgrounds, issue calendars and alarm-area rules stay inside the observed
runner and are injected through a small fold-scoring callback.  Every callback is
required to attest that preprocessing, alpha/ridge selection and model fitting were
repeated for the pseudo-history.

Time permutations operate on the complete 205-report axis.  Coverage remains at the
recipient date, the five snapshot fields and their null state move together, the two
200 km radius base series move with the same donor, and the six dynamic predictors are
recomputed causally from the destination pseudo-history.  Space permutations replace
only eligible entity coordinate pairs within authenticated construction strata and
then invoke the accepted Stage-3 200 km spatial and trajectory formulas.
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, cast

_BLAS_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
# Establish the process boundary before this module imports NumPy.  The execution
# entry point also requires a verified runtime limiter for already-loaded BLAS.
for _environment_name in _BLAS_THREAD_ENV:
    os.environ[_environment_name] = "1"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from seismoflux.background.execution import detect_physical_core_count
from seismoflux.d1_replay.features import (
    D1_SOURCE_COLUMNS,
    D1IssueFeatures,
    D1StaticGrid,
    stream_stage3_issue_features,
)
from seismoflux.d1_replay.model import D1ModelFitError
from seismoflux.data.common import canonical_json_bytes, write_json_atomic
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.nulls import spatial_null_reason_codes
from seismoflux.features.anomaly.snapshot import (
    Stage3IssueSnapshot,
    build_issue_snapshots,
    spatial_entity_arrays,
)
from seismoflux.features.anomaly.spatial import compute_stage4_placebo_spatial_features
from seismoflux.features.anomaly.state import states_from_records
from seismoflux.features.anomaly.trajectory import compute_latest_trajectory_features

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
PlaceboKind: TypeAlias = Literal["time", "space"]
PlaceboStatus: TypeAlias = Literal["passed", "evidence_insufficient"]
ProgressCallback: TypeAlias = Callable[[Mapping[str, object]], None]

D1_PLACEBO_REPLICATIONS = 200
D1_PLACEBO_CHECKPOINT_STRIDE = 25
D1_PLACEBO_ROOT_SEED = 147
D1_PLACEBO_MAX_WORKERS = 4
D1_PLACEBO_RESERVED_PHYSICAL_CORES = 2
D1_PLACEBO_FAILURE_FRACTION_MAX = 0.05
D1_PLACEBO_PROMISING_EXCEED_FRACTION = 0.80
D1_PLACEBO_FOLD_IDS = ("fold_1", "fold_2", "fold_3")
D1_PLACEBO_MODEL_IDS = ("B0_C", "B0_C_A_snapshot", "B0_C_A_dynamic")
D1_PLACEBO_CONTRASTS = (
    "B0_C_A_snapshot_minus_B0_C",
    "B0_C_A_dynamic_minus_B0_C",
    "B0_C_A_dynamic_minus_B0_C_A_snapshot",
)


def d1_placebo_schedule() -> Mapping[str, Mapping[str, int]]:
    """Return the frozen 200-time + 200-space schedule for all three folds."""

    return MappingProxyType(
        {
            kind: MappingProxyType(
                {fold_id: D1_PLACEBO_REPLICATIONS for fold_id in D1_PLACEBO_FOLD_IDS}
            )
            for kind in ("time", "space")
        }
    )


_PURPOSE_CODE: Mapping[PlaceboKind, int] = MappingProxyType({"time": 2, "space": 3})
_FOLD_CODE = MappingProxyType({"fold_1": 1, "fold_2": 2, "fold_3": 3})
_PRIMARY_SUPPORT_BY_FOLD = MappingProxyType({"fold_1": 8, "fold_2": 6, "fold_3": 7})
_COVERAGE_SLICE = slice(0, 4)
_SNAPSHOT_SLICE = slice(4, 9)
_DYNAMIC_SLICE = slice(9, 15)
_SNAPSHOT_SOURCE_COLUMNS = D1_SOURCE_COLUMNS[_SNAPSHOT_SLICE]
_TRAJECTORY_BASE_COLUMNS = (
    "radius_200km__listed_count",
    "radius_200km__first_seen_count",
)
_TRAJECTORY_FEATURES = (
    "slope_13w_per_week",
    "acceleration_4v13_per_week2",
    "peak_drop_52w",
)
_SPATIAL_SNAPSHOT_FIELDS = (
    "reliability_weighted_listed_count",
    "first_seen_weighted_count",
    "not_continued_weighted_count",
    "discipline_shannon_normalized",
    "concentration",
)
_MICROSECONDS_PER_SECOND = 1_000_000
_NANOSECONDS_PER_WEEK = 7 * 24 * 60 * 60 * 1_000_000_000


class D1PlaceboScientificFailure(RuntimeError):
    """A completed mapping whose refit cannot produce a valid scientific statistic."""


class D1PlaceboInfrastructureInterruption(RuntimeError):
    """An incomplete/mismatched run that must be resumed, not counted as a null."""


class D1PlaceboFoldScorer(Protocol):
    """Refit and score one fold after replacing its complete anomaly history."""

    def __call__(
        self,
        fold_id: str,
        pseudo_features: Mapping[int, D1IssueFeatures],
    ) -> object: ...


class _ObservedOutcome(Protocol):
    horizon_days: int
    model_id: str
    fold_id: str
    cluster_id: str
    hit_by_area: Sequence[bool]


def _readonly_float(values: object, *, ndim: int) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != ndim or np.isinf(result).any():
        raise ValueError("D1 placebo float array has invalid dimensions or infinity")
    result.setflags(write=False)
    return result


def _readonly_bool(values: object, *, shape: tuple[int, ...]) -> BoolArray:
    result = np.array(values, dtype=np.bool_, copy=True, order="C")
    if result.shape != shape:
        raise ValueError("D1 placebo null mask has the wrong shape")
    result.setflags(write=False)
    return result


def _readonly_int(values: object, *, shape: tuple[int, ...]) -> IntArray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if result.shape != shape:
        raise ValueError("D1 placebo reason-code array has the wrong shape")
    result.setflags(write=False)
    return result


def _utc_epoch_microseconds(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("D1 placebo issue time must be timezone-aware")
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        delta.days * 86_400_000_000 + delta.seconds * _MICROSECONDS_PER_SECOND + delta.microseconds
    )


def _iso_epoch_microseconds(value: object, *, label: str) -> int:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit UTC offset")
    return _utc_epoch_microseconds(parsed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_git_commit(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _emit(progress: ProgressCallback | None, **payload: object) -> None:
    if progress is not None:
        progress(MappingProxyType(dict(payload)))


def _same_static_grid(left: D1StaticGrid, right: D1StaticGrid) -> bool:
    return (
        left.grid_id == right.grid_id
        and left.cell_ids == right.cell_ids
        and np.array_equal(left.rows, right.rows)
        and np.array_equal(left.columns, right.columns)
        and np.array_equal(left.query_x_m, right.query_x_m)
        and np.array_equal(left.query_y_m, right.query_y_m)
        and np.array_equal(left.clipped_area_km2, right.clipped_area_km2)
    )


def d1_placebo_rng(
    kind: PlaceboKind,
    fold_id: str,
    replication_index: int,
) -> np.random.Generator:
    """Return the exact preregistered PCG64 stream for one kind/fold/replication."""

    if kind not in _PURPOSE_CODE:
        raise ValueError("D1 placebo kind must be time or space")
    if fold_id not in _FOLD_CODE:
        raise ValueError("D1 placebo fold must be fold_1, fold_2 or fold_3")
    if (
        isinstance(replication_index, bool)
        or not isinstance(replication_index, int)
        or not 0 <= replication_index < D1_PLACEBO_REPLICATIONS
    ):
        raise ValueError("D1 placebo replication must be an integer in [0, 200)")
    seed = np.random.SeedSequence(
        [
            D1_PLACEBO_ROOT_SEED,
            _PURPOSE_CODE[kind],
            _FOLD_CODE[fold_id],
            replication_index,
        ]
    )
    return np.random.Generator(np.random.PCG64(seed))


@dataclass(frozen=True, slots=True)
class D1PlaceboIssueSource:
    """Observed D1 predictors plus the two raw series needed to rebuild D1/D2."""

    issue_features: D1IssueFeatures
    trajectory_base_values: FloatArray
    trajectory_base_null_mask: BoolArray
    feature_store_file_sha256: str
    snapshot_reason_codes: IntArray | None = None
    trajectory_base_reason_codes: IntArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.issue_features, D1IssueFeatures):
            raise TypeError("issue_features must be D1IssueFeatures")
        feature_store_sha256 = _validated_sha256(
            self.feature_store_file_sha256,
            label="D1 issue-source feature-store SHA-256",
        )
        cells = self.issue_features.grid.cell_count
        bases = _readonly_float(self.trajectory_base_values, ndim=2)
        if bases.shape != (cells, 2):
            raise ValueError("D1 trajectory bases must have shape (cell, 2)")
        base_nulls = _readonly_bool(
            self.trajectory_base_null_mask,
            shape=(cells, 2),
        )
        if not np.isfinite(bases[~base_nulls]).all():
            raise ValueError("non-null D1 trajectory bases must be finite")
        owned_bases = np.array(bases, copy=True)
        owned_bases[base_nulls] = np.nan
        owned_bases.setflags(write=False)
        snapshot_reasons = (
            np.zeros((cells, 5), dtype=np.int64)
            if self.snapshot_reason_codes is None
            else self.snapshot_reason_codes
        )
        base_reasons = (
            np.zeros((cells, 2), dtype=np.int64)
            if self.trajectory_base_reason_codes is None
            else self.trajectory_base_reason_codes
        )
        object.__setattr__(self, "trajectory_base_values", owned_bases)
        object.__setattr__(self, "trajectory_base_null_mask", base_nulls)
        object.__setattr__(self, "feature_store_file_sha256", feature_store_sha256)
        object.__setattr__(
            self,
            "snapshot_reason_codes",
            _readonly_int(snapshot_reasons, shape=(cells, 5)),
        )
        object.__setattr__(
            self,
            "trajectory_base_reason_codes",
            _readonly_int(base_reasons, shape=(cells, 2)),
        )

    @property
    def issue_time_us(self) -> int:
        return _utc_epoch_microseconds(self.issue_features.issue_time_utc)


@dataclass(frozen=True, slots=True)
class D1FoldPlaceboPlan:
    fold_id: str
    fit_cutoff_us: int
    scored_issue_times_us: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.fold_id not in D1_PLACEBO_FOLD_IDS:
            raise ValueError("unknown D1 fold")
        if isinstance(self.fit_cutoff_us, bool) or not isinstance(self.fit_cutoff_us, int):
            raise TypeError("fit_cutoff_us must be epoch microseconds")
        scored = tuple(int(value) for value in self.scored_issue_times_us)
        if not scored or scored != tuple(sorted(set(scored))):
            raise ValueError("scored issue times must be unique and chronological")
        object.__setattr__(self, "scored_issue_times_us", scored)


@dataclass(frozen=True, slots=True)
class D1CoordinateEntity:
    """Minimal target-blind row used to audit one coordinate permutation."""

    state_id: str
    construction_stratum_id: str
    longitude: float
    latitude: float

    def __post_init__(self) -> None:
        if (
            not self.state_id
            or not self.construction_stratum_id
            or self.state_id != self.state_id.strip()
            or self.construction_stratum_id != self.construction_stratum_id.strip()
        ):
            raise ValueError("coordinate entity identities must be non-empty and trimmed")
        if not math.isfinite(self.longitude) or not math.isfinite(self.latitude):
            raise ValueError("coordinate entities require finite longitude/latitude")


@dataclass(frozen=True, slots=True)
class D1CoordinatePermutation:
    recipient_state_ids: tuple[str, ...]
    donor_state_ids: tuple[str, ...]
    permuted_coordinates: tuple[tuple[float, float], ...]
    mapping_sha256: str
    fixed_point_count: int
    moved_coordinate_count: int
    coordinate_multiset_verified: bool


def _parent_zone_id(construction_stratum_id: str) -> str:
    try:
        zone_id, suffix = construction_stratum_id.rsplit(":", maxsplit=1)
    except ValueError:
        return construction_stratum_id
    if suffix not in {"inside", "outside"}:
        return construction_stratum_id
    return zone_id


def permute_d1_coordinates_within_zones(
    entities: Sequence[D1CoordinateEntity],
    all_zone_ids: Sequence[str],
    *,
    rng: np.random.Generator,
) -> tuple[Mapping[str, tuple[float, float]], tuple[D1CoordinatePermutation, ...]]:
    """Permute coordinate pairs in issue→zone→stratum→state-ID order.

    The 39 parent zones are target independent.  The authenticated ``inside`` and
    ``outside`` suffix, when present, is retained as a finer stratum so an outside
    coordinate can never be donated to an inside entity (or vice versa).
    """

    zones = tuple(sorted(set(all_zone_ids), key=lambda value: value.encode("utf-8")))
    if not zones or any(not value or value != value.strip() for value in zones):
        raise ValueError("all_zone_ids must contain non-empty target-independent zones")
    rows = tuple(entities)
    if len({item.state_id for item in rows}) != len(rows):
        raise ValueError("coordinate entity state IDs must be unique within one issue")
    if any(_parent_zone_id(item.construction_stratum_id) not in zones for item in rows):
        raise ValueError("coordinate entity references a zone outside the verified 39-zone set")

    grouped: dict[tuple[str, str], list[D1CoordinateEntity]] = {}
    for item in rows:
        parent = _parent_zone_id(item.construction_stratum_id)
        grouped.setdefault((parent, item.construction_stratum_id), []).append(item)

    replacement: dict[str, tuple[float, float]] = {}
    audits: list[D1CoordinatePermutation] = []
    for zone_id in zones:
        strata = sorted(
            (key for key in grouped if key[0] == zone_id),
            key=lambda key: key[1].encode("utf-8"),
        )
        for key in strata:
            ordered = tuple(sorted(grouped[key], key=lambda item: item.state_id.encode("utf-8")))
            recipient_ids = tuple(item.state_id for item in ordered)
            coordinates = tuple((item.longitude, item.latitude) for item in ordered)
            donor_indices = tuple(int(index) for index in rng.permutation(len(ordered)))
            donor_ids = tuple(recipient_ids[index] for index in donor_indices)
            permuted = tuple(coordinates[index] for index in donor_indices)
            multiset_ok = sorted(coordinates) == sorted(permuted)
            if not multiset_ok:
                raise AssertionError("NumPy permutation changed a coordinate multiset")
            fixed = sum(
                recipient == donor
                for recipient, donor in zip(recipient_ids, donor_ids, strict=True)
            )
            moved = sum(
                original != donor for original, donor in zip(coordinates, permuted, strict=True)
            )
            digest = _canonical_sha256(
                {
                    "direction": "recipient_state_to_donor_coordinate",
                    "donor_state_ids": donor_ids,
                    "recipient_state_ids": recipient_ids,
                    "stratum_id": key[1],
                    "zone_id": zone_id,
                }
            )
            audits.append(
                D1CoordinatePermutation(
                    recipient_state_ids=recipient_ids,
                    donor_state_ids=donor_ids,
                    permuted_coordinates=permuted,
                    mapping_sha256=digest,
                    fixed_point_count=fixed,
                    moved_coordinate_count=moved,
                    coordinate_multiset_verified=multiset_ok,
                )
            )
            replacement.update(zip(recipient_ids, permuted, strict=True))
    if set(replacement) != {item.state_id for item in rows}:
        raise ValueError("coordinate permutation did not cover every eligible entity")
    return MappingProxyType(replacement), tuple(audits)


@dataclass(frozen=True, slots=True)
class D1PseudoHistory:
    kind: PlaceboKind
    fold_id: str
    replication_index: int
    issue_sources: tuple[D1PlaceboIssueSource, ...]
    mapping_sha256: str
    mapping_audit: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.kind not in _PURPOSE_CODE or self.fold_id not in D1_PLACEBO_FOLD_IDS:
            raise ValueError("D1 pseudo-history kind/fold is invalid")
        if len(self.mapping_sha256) != 64:
            raise ValueError("D1 pseudo-history mapping identity must be SHA-256")
        sources = tuple(self.issue_sources)
        times = tuple(item.issue_time_us for item in sources)
        if not sources or times != tuple(sorted(set(times))):
            raise ValueError("D1 pseudo-history must retain the complete chronological axis")
        object.__setattr__(self, "issue_sources", sources)
        object.__setattr__(self, "mapping_audit", MappingProxyType(dict(self.mapping_audit)))

    @property
    def features_by_issue(self) -> Mapping[int, D1IssueFeatures]:
        return MappingProxyType(
            {item.issue_time_us: item.issue_features for item in self.issue_sources}
        )


def _validate_source_axis(
    issue_sources: Sequence[D1PlaceboIssueSource],
    *,
    expected_issue_count: int,
) -> tuple[D1PlaceboIssueSource, ...]:
    sources = tuple(issue_sources)
    if len(sources) != expected_issue_count:
        raise ValueError(f"D1 placebo requires all {expected_issue_count} actual report periods")
    times = tuple(item.issue_time_us for item in sources)
    if times != tuple(sorted(set(times))):
        raise ValueError("D1 placebo issue sources must be unique and chronological")
    file_hashes = {item.feature_store_file_sha256 for item in sources}
    if len(file_hashes) != 1:
        raise ValueError("D1 placebo issue sources do not share one feature-store identity")
    first_grid = sources[0].issue_features.grid
    if any(not _same_static_grid(item.issue_features.grid, first_grid) for item in sources[1:]):
        raise ValueError("D1 placebo issue sources changed the frozen grid")
    return sources


def _latest_trajectory_columns(
    issue_times: NDArray[np.datetime64],
    bases: Sequence[FloatArray],
    base_nulls: Sequence[BoolArray],
    issue_index: int,
) -> tuple[FloatArray, BoolArray]:
    current_ns = int(issue_times[issue_index].astype(np.int64))
    lower_ns = current_ns - 52 * _NANOSECONDS_PER_WEEK
    first = int(np.searchsorted(issue_times.astype(np.int64), lower_ns, side="right"))
    history_values = np.stack(bases[first : issue_index + 1], axis=0)
    history_nulls = np.stack(base_nulls[first : issue_index + 1], axis=0)
    history_values = np.array(history_values, copy=True)
    history_values[history_nulls] = np.nan
    cell_count = history_values.shape[1]
    packed = history_values.reshape(history_values.shape[0], cell_count * 2)
    latest = compute_latest_trajectory_features(
        issue_times[first : issue_index + 1],
        packed,
    )
    values = np.empty((cell_count, 6), dtype=np.float64)
    nulls = np.empty((cell_count, 6), dtype=np.bool_)
    for feature_index, name in enumerate(_TRAJECTORY_FEATURES):
        feature_values = latest.features[name].reshape(cell_count, 2)
        valid = latest.valid_masks[name].reshape(cell_count, 2)
        values[:, feature_index] = feature_values[:, 0]
        values[:, feature_index + 3] = feature_values[:, 1]
        nulls[:, feature_index] = ~valid[:, 0]
        nulls[:, feature_index + 3] = ~valid[:, 1]
    values[nulls] = np.nan
    return values, nulls


def rebuild_d1_dynamic_from_pseudo_history(
    recipient_sources: Sequence[D1PlaceboIssueSource],
    snapshot_values: Sequence[FloatArray],
    snapshot_nulls: Sequence[BoolArray],
    trajectory_base_values: Sequence[FloatArray],
    trajectory_base_nulls: Sequence[BoolArray],
    *,
    snapshot_reason_codes: Sequence[IntArray] | None = None,
    trajectory_base_reason_codes: Sequence[IntArray] | None = None,
) -> tuple[D1PlaceboIssueSource, ...]:
    """Rebuild D1/D2 from destination pseudo-history with Stage-3 formulas."""

    recipients = tuple(recipient_sources)
    count = len(recipients)
    sequences = (
        tuple(snapshot_values),
        tuple(snapshot_nulls),
        tuple(trajectory_base_values),
        tuple(trajectory_base_nulls),
    )
    if any(len(values) != count for values in sequences):
        raise ValueError("pseudo-history arrays must cover every recipient issue")
    snapshot_reasons = (
        tuple(cast(IntArray, item.snapshot_reason_codes) for item in recipients)
        if snapshot_reason_codes is None
        else tuple(snapshot_reason_codes)
    )
    base_reasons = (
        tuple(cast(IntArray, item.trajectory_base_reason_codes) for item in recipients)
        if trajectory_base_reason_codes is None
        else tuple(trajectory_base_reason_codes)
    )
    if len(snapshot_reasons) != count or len(base_reasons) != count:
        raise ValueError("pseudo-history reason codes must cover every recipient issue")
    issue_times = np.asarray(
        [
            np.datetime64(
                item.issue_features.issue_time_utc.astimezone(UTC).replace(tzinfo=None),
                "ns",
            )
            for item in recipients
        ],
        dtype="datetime64[ns]",
    )
    rebuilt: list[D1PlaceboIssueSource] = []
    for index, recipient in enumerate(recipients):
        values = np.array(recipient.issue_features.values, copy=True)
        nulls = np.array(recipient.issue_features.null_mask, copy=True)
        snapshot = np.asarray(sequences[0][index], dtype=np.float64)
        snapshot_missing = np.asarray(sequences[1][index], dtype=np.bool_)
        if snapshot.shape != (recipient.issue_features.grid.cell_count, 5):
            raise ValueError("pseudo snapshot values have the wrong shape")
        if snapshot_missing.shape != snapshot.shape:
            raise ValueError("pseudo snapshot null mask has the wrong shape")
        values[:, _SNAPSHOT_SLICE] = snapshot
        nulls[:, _SNAPSHOT_SLICE] = snapshot_missing
        dynamic_values, dynamic_nulls = _latest_trajectory_columns(
            issue_times,
            sequences[2],
            sequences[3],
            index,
        )
        values[:, _DYNAMIC_SLICE] = dynamic_values
        nulls[:, _DYNAMIC_SLICE] = dynamic_nulls
        values[nulls] = np.nan
        features = D1IssueFeatures(
            issue_time_utc=recipient.issue_features.issue_time_utc,
            issue_report_id=recipient.issue_features.issue_report_id,
            grid=recipient.issue_features.grid,
            source_columns=D1_SOURCE_COLUMNS,
            values=values,
            null_mask=nulls,
        )
        rebuilt.append(
            D1PlaceboIssueSource(
                issue_features=features,
                trajectory_base_values=sequences[2][index],
                trajectory_base_null_mask=sequences[3][index],
                feature_store_file_sha256=recipient.feature_store_file_sha256,
                snapshot_reason_codes=snapshot_reasons[index],
                trajectory_base_reason_codes=base_reasons[index],
            )
        )
        if not np.array_equal(
            features.values[:, _COVERAGE_SLICE],
            recipient.issue_features.values[:, _COVERAGE_SLICE],
            equal_nan=True,
        ) or not np.array_equal(
            features.null_mask[:, _COVERAGE_SLICE],
            recipient.issue_features.null_mask[:, _COVERAGE_SLICE],
        ):
            raise AssertionError("D1 placebo moved recipient coverage")
    return tuple(rebuilt)


def build_d1_time_pseudo_history(
    issue_sources: Sequence[D1PlaceboIssueSource],
    plan: D1FoldPlaceboPlan,
    replication_index: int,
    *,
    expected_issue_count: int = 205,
) -> D1PseudoHistory:
    """Permute complete fit/post-fit report axes separately and rebuild D1/D2."""

    sources = _validate_source_axis(
        issue_sources,
        expected_issue_count=expected_issue_count,
    )
    times = tuple(item.issue_time_us for item in sources)
    if not set(plan.scored_issue_times_us) <= set(times):
        raise ValueError("D1 time placebo full history omits a scored issue")
    fit_indices = tuple(index for index, value in enumerate(times) if value <= plan.fit_cutoff_us)
    post_indices = tuple(index for index, value in enumerate(times) if value > plan.fit_cutoff_us)
    if not fit_indices or not post_indices:
        raise ValueError("D1 time placebo fit and post-fit pools must both be non-empty")
    rng = d1_placebo_rng("time", plan.fold_id, replication_index)
    fit_donors = tuple(fit_indices[int(index)] for index in rng.permutation(len(fit_indices)))
    post_donors = tuple(post_indices[int(index)] for index in rng.permutation(len(post_indices)))
    donor_by_recipient = dict(zip(fit_indices, fit_donors, strict=True))
    donor_by_recipient.update(zip(post_indices, post_donors, strict=True))
    if set(donor_by_recipient) != set(range(len(sources))):
        raise AssertionError("time placebo did not cover the complete report axis")

    snapshots = [
        np.array(
            sources[donor_by_recipient[index]].issue_features.values[:, _SNAPSHOT_SLICE],
            copy=True,
        )
        for index in range(len(sources))
    ]
    snapshot_nulls = [
        np.array(
            sources[donor_by_recipient[index]].issue_features.null_mask[:, _SNAPSHOT_SLICE],
            copy=True,
        )
        for index in range(len(sources))
    ]
    bases = [
        np.array(sources[donor_by_recipient[index]].trajectory_base_values, copy=True)
        for index in range(len(sources))
    ]
    base_nulls = [
        np.array(sources[donor_by_recipient[index]].trajectory_base_null_mask, copy=True)
        for index in range(len(sources))
    ]
    snapshot_reasons = [
        np.array(
            cast(IntArray, sources[donor_by_recipient[index]].snapshot_reason_codes),
            copy=True,
        )
        for index in range(len(sources))
    ]
    base_reasons = [
        np.array(
            cast(IntArray, sources[donor_by_recipient[index]].trajectory_base_reason_codes),
            copy=True,
        )
        for index in range(len(sources))
    ]
    rebuilt = rebuild_d1_dynamic_from_pseudo_history(
        sources,
        snapshots,
        snapshot_nulls,
        bases,
        base_nulls,
        snapshot_reason_codes=snapshot_reasons,
        trajectory_base_reason_codes=base_reasons,
    )
    fit_pairs = tuple((times[index], times[donor_by_recipient[index]]) for index in fit_indices)
    post_pairs = tuple((times[index], times[donor_by_recipient[index]]) for index in post_indices)
    audit: dict[str, object] = {
        "fit_pool_pairs": fit_pairs,
        "post_fit_pool_pairs": post_pairs,
        "fit_fixed_point_count": sum(left == right for left, right in fit_pairs),
        "post_fit_fixed_point_count": sum(left == right for left, right in post_pairs),
        "complete_report_period_count": len(sources),
        "coverage_fixed": True,
        "dynamic_rebuilt_from_destination_history": True,
    }
    mapping_sha256 = _canonical_sha256(
        {
            "fold_id": plan.fold_id,
            "kind": "time",
            "replication_index": replication_index,
            **audit,
        }
    )
    return D1PseudoHistory(
        kind="time",
        fold_id=plan.fold_id,
        replication_index=replication_index,
        issue_sources=rebuilt,
        mapping_sha256=mapping_sha256,
        mapping_audit=audit,
    )


def _permuted_snapshot(
    snapshot: Stage3IssueSnapshot,
    *,
    construction_stratum_by_state_id: Mapping[str, str],
    all_zone_ids: Sequence[str],
    rng: np.random.Generator,
) -> tuple[Stage3IssueSnapshot, tuple[D1CoordinatePermutation, ...]]:
    eligible = tuple(state for state in snapshot.entities if state.spatial_eligible)
    eligible_ids = {state.state_id for state in eligible}
    if not eligible_ids <= set(construction_stratum_by_state_id):
        raise ValueError("verified construction strata omit a spatially eligible entity")
    rows = tuple(
        D1CoordinateEntity(
            state_id=state.state_id,
            construction_stratum_id=construction_stratum_by_state_id[state.state_id],
            longitude=cast(float, state.longitude),
            latitude=cast(float, state.latitude),
        )
        for state in eligible
    )
    replacement_by_state_id, audits = permute_d1_coordinates_within_zones(
        rows,
        all_zone_ids,
        rng=rng,
    )
    permuted_entities = tuple(
        replace(
            state,
            longitude=replacement_by_state_id[state.state_id][0],
            latitude=replacement_by_state_id[state.state_id][1],
        )
        if state.spatial_eligible
        else state
        for state in snapshot.entities
    )
    return replace(snapshot, entities=permuted_entities), audits


def build_d1_space_pseudo_history(
    issue_sources: Sequence[D1PlaceboIssueSource],
    snapshots_by_issue_us: Mapping[int, Stage3IssueSnapshot],
    query_grid: Stage3QueryGrid,
    construction_stratum_by_state_id: Mapping[str, str],
    all_zone_ids: Sequence[str],
    plan: D1FoldPlaceboPlan,
    replication_index: int,
    *,
    expected_issue_count: int = 205,
    expected_zone_count: int = 39,
    query_chunk_size: int = 256,
) -> D1PseudoHistory:
    """Permute entity coordinates, recompute 200 km snapshot/base fields and D1/D2."""

    sources = _validate_source_axis(
        issue_sources,
        expected_issue_count=expected_issue_count,
    )
    zones = tuple(sorted(set(all_zone_ids), key=lambda value: value.encode("utf-8")))
    if len(zones) != expected_zone_count:
        raise ValueError("D1 space placebo did not receive the verified zone count")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    times = tuple(item.issue_time_us for item in sources)
    if set(snapshots_by_issue_us) != set(times):
        raise ValueError("D1 space placebo snapshots must cover the complete report axis")
    rng = d1_placebo_rng("space", plan.fold_id, replication_index)
    snapshots: list[FloatArray] = []
    snapshot_nulls: list[BoolArray] = []
    bases: list[FloatArray] = []
    base_nulls: list[BoolArray] = []
    snapshot_reasons: list[IntArray] = []
    base_reasons: list[IntArray] = []
    group_audits: list[dict[str, object]] = []
    for issue_index, source in enumerate(sources):
        snapshot = snapshots_by_issue_us[source.issue_time_us]
        if snapshot.issue_time_utc.astimezone(UTC) != source.issue_features.issue_time_utc:
            raise ValueError("D1 state snapshot and feature issue times differ")
        if snapshot.summary.issue_report_id != source.issue_features.issue_report_id:
            raise ValueError("D1 state snapshot and feature report identities differ")
        permuted, audits = _permuted_snapshot(
            snapshot,
            construction_stratum_by_state_id=construction_stratum_by_state_id,
            all_zone_ids=zones,
            rng=rng,
        )
        spatial = compute_stage4_placebo_spatial_features(
            query_grid.query_xy_m,
            spatial_entity_arrays(permuted),
            query_chunk_size=query_chunk_size,
        )
        snapshot_values = np.column_stack(
            [spatial.gaussian_features[name][:, 0] for name in _SPATIAL_SNAPSHOT_FIELDS]
        )
        snapshot_missing = ~np.isfinite(snapshot_values)
        reasons = np.column_stack(
            [
                spatial_null_reason_codes(name, spatial.gaussian_features)[:, 0]
                for name in _SPATIAL_SNAPSHOT_FIELDS
            ]
        ).astype(np.int64, copy=False)
        base_values = np.column_stack(
            [spatial.radius_features[name][:, 0] for name in ("listed_count", "first_seen_count")]
        )
        if not np.isfinite(base_values).all() or np.any(base_values < 0.0):
            raise ValueError("recomputed D1 200 km trajectory bases are invalid")
        snapshots.append(snapshot_values)
        snapshot_nulls.append(snapshot_missing)
        bases.append(base_values)
        base_nulls.append(np.zeros(base_values.shape, dtype=np.bool_))
        snapshot_reasons.append(reasons)
        base_reasons.append(np.zeros(base_values.shape, dtype=np.int64))
        group_audits.extend(
            {
                "issue_index": issue_index,
                "mapping_sha256": item.mapping_sha256,
                "fixed_point_count": item.fixed_point_count,
                "moved_coordinate_count": item.moved_coordinate_count,
            }
            for item in audits
        )
    rebuilt = rebuild_d1_dynamic_from_pseudo_history(
        sources,
        snapshots,
        snapshot_nulls,
        bases,
        base_nulls,
        snapshot_reason_codes=snapshot_reasons,
        trajectory_base_reason_codes=base_reasons,
    )
    audit: dict[str, object] = {
        "complete_report_period_count": len(sources),
        "verified_zone_count": len(zones),
        "permutation_group_count": len(group_audits),
        "fixed_point_count": sum(cast(int, item["fixed_point_count"]) for item in group_audits),
        "moved_coordinate_count": sum(
            cast(int, item["moved_coordinate_count"]) for item in group_audits
        ),
        "coordinate_multiset_verified": True,
        "noncoordinate_payload_fixed": True,
        "coverage_fixed": True,
        "snapshot_and_dynamic_rebuilt_at_200km": True,
    }
    mapping_sha256 = _canonical_sha256(
        {
            "fold_id": plan.fold_id,
            "group_mapping_sha256": [item["mapping_sha256"] for item in group_audits],
            "kind": "space",
            "replication_index": replication_index,
        }
    )
    return D1PseudoHistory(
        kind="space",
        fold_id=plan.fold_id,
        replication_index=replication_index,
        issue_sources=rebuilt,
        mapping_sha256=mapping_sha256,
        mapping_audit=audit,
    )


@dataclass(frozen=True, slots=True)
class D1VerifiedSpatialStrata:
    all_zone_ids: tuple[str, ...]
    construction_stratum_by_state_id: Mapping[str, str]
    artifact_sha256: Mapping[str, str]
    public_manifest_content_sha256: str

    def __post_init__(self) -> None:
        zones = tuple(sorted(set(self.all_zone_ids), key=lambda value: value.encode("utf-8")))
        if len(zones) != 39 or any(
            not isinstance(value, str) or not value or value != value.strip() for value in zones
        ):
            raise ValueError("D1 spatial strata must contain exactly 39 nonempty zones")
        strata = dict(self.construction_stratum_by_state_id)
        if any(
            not isinstance(state_id, str)
            or not state_id
            or state_id != state_id.strip()
            or not isinstance(stratum_id, str)
            or not stratum_id
            or stratum_id != stratum_id.strip()
            or _parent_zone_id(stratum_id) not in zones
            for state_id, stratum_id in strata.items()
        ):
            raise ValueError("D1 entity strata contain an invalid state or zone identity")
        if set(self.artifact_sha256) != {
            "cell_mapping",
            "entity_mapping",
            "zone_geometry",
            "connectors",
        }:
            raise ValueError("D1 spatial strata require exactly four artifact hashes")
        artifact_hashes = {
            key: _validated_sha256(value, label=f"D1 spatial artifact {key} SHA-256")
            for key, value in self.artifact_sha256.items()
        }
        manifest_sha256 = _validated_sha256(
            self.public_manifest_content_sha256,
            label="D1 spatial public-manifest content SHA-256",
        )
        object.__setattr__(self, "all_zone_ids", zones)
        object.__setattr__(
            self,
            "construction_stratum_by_state_id",
            MappingProxyType(strata),
        )
        object.__setattr__(self, "artifact_sha256", MappingProxyType(artifact_hashes))
        object.__setattr__(self, "public_manifest_content_sha256", manifest_sha256)


@dataclass(frozen=True, slots=True)
class D1Stage3SnapshotHistory(Mapping[int, Stage3IssueSnapshot]):
    """Complete state-snapshot axis carrying the hash of the bytes actually read."""

    snapshots_by_issue_us: Mapping[int, Stage3IssueSnapshot]
    state_history_file_sha256: str

    def __post_init__(self) -> None:
        snapshots = dict(self.snapshots_by_issue_us)
        if not snapshots:
            raise ValueError("D1 state-snapshot history must not be empty")
        ordered = tuple(sorted(snapshots))
        if tuple(snapshots) != ordered:
            snapshots = {key: snapshots[key] for key in ordered}
        for issue_time_us, snapshot in snapshots.items():
            if _utc_epoch_microseconds(snapshot.issue_time_utc) != issue_time_us:
                raise ValueError("D1 state-snapshot key differs from its issue time")
        object.__setattr__(
            self,
            "snapshots_by_issue_us",
            MappingProxyType(snapshots),
        )
        object.__setattr__(
            self,
            "state_history_file_sha256",
            _validated_sha256(
                self.state_history_file_sha256,
                label="D1 state-history actual file SHA-256",
            ),
        )

    def __getitem__(self, key: int) -> Stage3IssueSnapshot:
        return self.snapshots_by_issue_us[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self.snapshots_by_issue_us)

    def __len__(self) -> int:
        return len(self.snapshots_by_issue_us)


def verify_d1_spatial_strata_files(
    *,
    public_manifest_path: Path,
    cell_mapping_path: Path,
    entity_mapping_path: Path,
    zone_geometry_path: Path,
    connectors_path: Path,
    expected_public_content_sha256: str,
    expected_artifact_sha256: Mapping[str, str],
    snapshots_by_issue_us: Mapping[int, Stage3IssueSnapshot],
) -> D1VerifiedSpatialStrata:
    """Hash-check all four local artifacts and bind entity strata to Stage-3 states."""

    public_payload = json.loads(Path(public_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(public_payload, dict):
        raise TypeError("D1 spatial-strata public manifest must be a JSON mapping")
    stated_content = public_payload.get("content_sha256")
    body = dict(public_payload)
    body.pop("content_sha256", None)
    content_sha = _canonical_sha256(body)
    if stated_content != content_sha or content_sha != expected_public_content_sha256:
        raise ValueError("D1 spatial-strata public manifest content hash changed")
    if public_payload.get("nonempty_stratum_count") != 39:
        raise ValueError("D1 spatial-strata public manifest no longer has 39 zones")

    paths = {
        "cell_mapping": Path(cell_mapping_path),
        "entity_mapping": Path(entity_mapping_path),
        "zone_geometry": Path(zone_geometry_path),
        "connectors": Path(connectors_path),
    }
    observed_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if observed_hashes != dict(expected_artifact_sha256):
        raise ValueError("one or more D1 local spatial-strata artifact hashes changed")

    cell_table = pq.read_table(
        paths["cell_mapping"],
        columns=["construction_zone_id"],
        use_threads=False,
    )
    zone_values = cell_table["construction_zone_id"].combine_chunks().to_pylist()
    if any(not isinstance(value, str) or not value for value in zone_values):
        raise ValueError("D1 cell-zone mapping contains an invalid zone identity")
    zones = tuple(
        sorted(
            set(cast(list[str], zone_values)),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if len(zones) != 39:
        raise ValueError("D1 cell-zone mapping does not contain exactly 39 zones")

    entity_table = pq.read_table(
        paths["entity_mapping"],
        columns=["state_id", "issue_time_utc", "construction_stratum_id"],
        use_threads=False,
    )
    eligible: dict[str, int] = {}
    for issue_time_us, snapshot in snapshots_by_issue_us.items():
        if _utc_epoch_microseconds(snapshot.issue_time_utc) != issue_time_us:
            raise ValueError("D1 snapshot mapping key changed its issue time")
        for state in snapshot.entities:
            if state.spatial_eligible:
                if state.state_id in eligible:
                    raise ValueError("D1 state history duplicated an eligible state ID")
                eligible[state.state_id] = issue_time_us
    state_ids = entity_table["state_id"].combine_chunks().to_pylist()
    issue_times = entity_table["issue_time_utc"].combine_chunks().to_pylist()
    strata = entity_table["construction_stratum_id"].combine_chunks().to_pylist()
    output: dict[str, str] = {}
    for state_id, issue_time, stratum in zip(state_ids, issue_times, strata, strict=True):
        if state_id not in eligible:
            continue
        if (
            not isinstance(state_id, str)
            or not isinstance(issue_time, datetime)
            or not isinstance(stratum, str)
            or not stratum
            or _parent_zone_id(stratum) not in zones
            or _utc_epoch_microseconds(issue_time) != eligible[state_id]
        ):
            raise ValueError("D1 entity-stratum row changed its authenticated identity")
        if state_id in output:
            raise ValueError("D1 entity-stratum mapping duplicated an eligible state")
        output[state_id] = stratum
    if set(output) != set(eligible):
        raise ValueError("D1 entity-stratum mapping does not cover exactly eligible states")
    return D1VerifiedSpatialStrata(
        all_zone_ids=zones,
        construction_stratum_by_state_id=output,
        artifact_sha256=observed_hashes,
        public_manifest_content_sha256=content_sha,
    )


def _arrow_float_with_nulls(table: pa.Table, column: str) -> tuple[FloatArray, BoolArray]:
    array = table[column].combine_chunks()
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
    nulls = np.asarray(array.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
    values = np.array(values, copy=True)
    values[nulls] = np.nan
    return values, nulls


def load_d1_placebo_issue_sources(
    feature_store_path: Path,
    *,
    expected_file_sha256: str,
    expected_grid: D1StaticGrid | None = None,
    expected_issue_count: int = 205,
    expected_cell_count: int = 15_697,
) -> tuple[D1PlaceboIssueSource, ...]:
    """Stream the 15 model fields plus two raw bases for all actual reports."""

    path = Path(feature_store_path)
    expected_sha256 = _validated_sha256(
        expected_file_sha256,
        label="D1 feature-store expected_file_sha256",
    )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("D1 feature store SHA-256 differs from the frozen input")
    parquet = pq.ParquetFile(path)
    schema_names = set(parquet.schema_arrow.names)
    if not set(_TRAJECTORY_BASE_COLUMNS) <= schema_names:
        raise ValueError("D1 feature store omits a trajectory base series")
    feature_stream = stream_stage3_issue_features(
        path,
        expected_issue_count=expected_issue_count,
        expected_cell_count=expected_cell_count,
        expected_grid=expected_grid,
    )
    output: list[D1PlaceboIssueSource] = []
    for row_group_index, features in enumerate(feature_stream):
        selected = ["issue_time_utc", *_TRAJECTORY_BASE_COLUMNS]
        reason_columns = [
            f"{column}__null_reason_code"
            for column in (*_SNAPSHOT_SOURCE_COLUMNS, *_TRAJECTORY_BASE_COLUMNS)
            if f"{column}__null_reason_code" in schema_names
        ]
        table = parquet.read_row_group(
            row_group_index,
            columns=[*selected, *reason_columns],
            use_threads=False,
        )
        issue_times = table["issue_time_utc"].combine_chunks().unique().to_pylist()
        if issue_times != [features.issue_time_utc]:
            raise ValueError("D1 raw trajectory row group changed issue identity")
        base_parts = [_arrow_float_with_nulls(table, column) for column in _TRAJECTORY_BASE_COLUMNS]
        base_values = np.column_stack([item[0] for item in base_parts])
        base_nulls = np.column_stack([item[1] for item in base_parts])
        snapshot_reasons = np.zeros((features.grid.cell_count, 5), dtype=np.int64)
        for index, column in enumerate(_SNAPSHOT_SOURCE_COLUMNS):
            reason_column = f"{column}__null_reason_code"
            if reason_column in table.column_names:
                reason = table[reason_column].combine_chunks()
                if reason.null_count:
                    raise ValueError("D1 snapshot reason code contains Arrow nulls")
                snapshot_reasons[:, index] = np.asarray(
                    reason.to_numpy(zero_copy_only=False),
                    dtype=np.int64,
                )
            elif np.any(features.null_mask[:, index + 4]):
                raise ValueError("nullable D1 snapshot source omitted its reason code")
        base_reasons = np.zeros((features.grid.cell_count, 2), dtype=np.int64)
        for index, column in enumerate(_TRAJECTORY_BASE_COLUMNS):
            reason_column = f"{column}__null_reason_code"
            if reason_column in table.column_names:
                reason = table[reason_column].combine_chunks()
                if reason.null_count:
                    raise ValueError("D1 trajectory-base reason code contains Arrow nulls")
                base_reasons[:, index] = np.asarray(
                    reason.to_numpy(zero_copy_only=False),
                    dtype=np.int64,
                )
            elif np.any(base_nulls[:, index]):
                raise ValueError("nullable D1 trajectory base omitted its reason code")
        output.append(
            D1PlaceboIssueSource(
                issue_features=features,
                trajectory_base_values=base_values,
                trajectory_base_null_mask=base_nulls,
                feature_store_file_sha256=observed_sha256,
                snapshot_reason_codes=snapshot_reasons,
                trajectory_base_reason_codes=base_reasons,
            )
        )
    return _validate_source_axis(output, expected_issue_count=expected_issue_count)


def load_d1_stage3_snapshots(
    state_history_path: Path,
    *,
    expected_file_sha256: str,
    expected_issue_count: int = 205,
) -> D1Stage3SnapshotHistory:
    """Rehydrate the accepted Stage-3 state history without reading any target."""

    path = Path(state_history_path)
    expected_sha256 = _validated_sha256(
        expected_file_sha256,
        label="D1 state-history expected_file_sha256",
    )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("D1 state history SHA-256 differs from the frozen input")
    parquet = pq.ParquetFile(path)
    records: list[Mapping[str, object]] = []
    for row_group_index in range(parquet.num_row_groups):
        records.extend(parquet.read_row_group(row_group_index, use_threads=False).to_pylist())
    snapshots = build_issue_snapshots(
        states_from_records(records),
        expected_issue_count=expected_issue_count,
    )
    return D1Stage3SnapshotHistory(
        snapshots_by_issue_us={
            _utc_epoch_microseconds(item.issue_time_utc): item for item in snapshots
        },
        state_history_file_sha256=observed_sha256,
    )


@dataclass(frozen=True, slots=True)
class D1ObservedPlaceboBaseline:
    observed_statistics: Mapping[str, float]
    support_by_fold: Mapping[str, int]
    identities: Mapping[str, object]

    def __post_init__(self) -> None:
        statistics = {key: float(value) for key, value in self.observed_statistics.items()}
        if set(statistics) != set(D1_PLACEBO_CONTRASTS) or not all(
            math.isfinite(value) for value in statistics.values()
        ):
            raise ValueError("observed D1 placebo baseline omitted a registered contrast")
        support = {key: int(value) for key, value in self.support_by_fold.items()}
        if support != dict(_PRIMARY_SUPPORT_BY_FOLD) or sum(support.values()) != 21:
            raise ValueError("observed D1 placebo baseline changed the frozen 8/6/7 support")
        object.__setattr__(self, "observed_statistics", MappingProxyType(statistics))
        object.__setattr__(self, "support_by_fold", MappingProxyType(support))
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))


def observed_d1_placebo_baseline(observed_result: object) -> D1ObservedPlaceboBaseline:
    """Extract the 30-day/600,000 km² pooled observed statistics."""

    if isinstance(observed_result, D1ObservedPlaceboBaseline):
        return observed_result
    if isinstance(observed_result, Mapping):
        outcomes_value = observed_result.get("outcomes")
        identities_value = observed_result.get("identities", {})
    else:
        outcomes_value = getattr(observed_result, "outcomes", None)
        identities_value = getattr(observed_result, "identities", {})
    if not isinstance(outcomes_value, Sequence) or isinstance(outcomes_value, str | bytes):
        raise TypeError("observed D1 result must expose cluster outcomes")
    if not isinstance(identities_value, Mapping):
        raise TypeError("observed D1 identities must be a mapping")
    hits: dict[tuple[str, str], dict[str, bool]] = {}
    for raw in outcomes_value:
        if isinstance(raw, Mapping):
            horizon = int(raw["horizon_days"])
            model_id = str(raw["model_id"])
            fold_id = str(raw["fold_id"])
            cluster_id = str(raw["cluster_id"])
            hit_by_area = tuple(bool(value) for value in cast(Sequence[object], raw["hit_by_area"]))
        else:
            outcome = cast(_ObservedOutcome, raw)
            horizon = int(outcome.horizon_days)
            model_id = str(outcome.model_id)
            fold_id = str(outcome.fold_id)
            cluster_id = str(outcome.cluster_id)
            hit_by_area = tuple(bool(value) for value in outcome.hit_by_area)
        if horizon != 30 or model_id not in D1_PLACEBO_MODEL_IDS:
            continue
        if fold_id not in D1_PLACEBO_FOLD_IDS or len(hit_by_area) <= 2:
            raise ValueError("observed D1 primary outcome is malformed")
        hits.setdefault((fold_id, cluster_id), {})[model_id] = hit_by_area[2]
    if not hits or any(set(value) != set(D1_PLACEBO_MODEL_IDS) for value in hits.values()):
        raise ValueError("observed D1 result changed primary model/cluster support")
    support = {fold_id: sum(key[0] == fold_id for key in hits) for fold_id in D1_PLACEBO_FOLD_IDS}
    model_hits = {
        model: sum(value[model] for value in hits.values()) for model in D1_PLACEBO_MODEL_IDS
    }
    total = sum(support.values())
    statistics = {
        D1_PLACEBO_CONTRASTS[0]: (model_hits["B0_C_A_snapshot"] - model_hits["B0_C"]) / total,
        D1_PLACEBO_CONTRASTS[1]: (model_hits["B0_C_A_dynamic"] - model_hits["B0_C"]) / total,
        D1_PLACEBO_CONTRASTS[2]: (model_hits["B0_C_A_dynamic"] - model_hits["B0_C_A_snapshot"])
        / total,
    }
    return D1ObservedPlaceboBaseline(statistics, support, identities_value)


@dataclass(frozen=True, slots=True)
class D1FoldPlaceboScore:
    fold_id: str
    support_count: int
    model_hit_counts: Mapping[str, int]
    selected_alpha: float | None = None
    selected_ridge_by_model: Mapping[str, float] = field(default_factory=dict)
    preprocessing_refit: bool = True
    alpha_reselected: bool = True
    ridge_reselected: bool = True
    models_refit: bool = True

    def __post_init__(self) -> None:
        hits = {key: int(value) for key, value in self.model_hit_counts.items()}
        if self.fold_id not in D1_PLACEBO_FOLD_IDS or self.support_count <= 0:
            raise ValueError("D1 placebo fold score has invalid support")
        if set(hits) != set(D1_PLACEBO_MODEL_IDS) or any(
            not 0 <= value <= self.support_count for value in hits.values()
        ):
            raise ValueError("D1 placebo fold score has invalid model hit counts")
        for name in (
            "preprocessing_refit",
            "alpha_reselected",
            "ridge_reselected",
            "models_refit",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"D1 placebo callback must attest {name}=True")
        if self.selected_alpha is not None and not math.isfinite(self.selected_alpha):
            raise ValueError("selected alpha must be finite when reported")
        ridges = {key: float(value) for key, value in self.selected_ridge_by_model.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in ridges.values()):
            raise ValueError("reported D1 placebo ridge values are invalid")
        object.__setattr__(self, "model_hit_counts", MappingProxyType(hits))
        object.__setattr__(self, "selected_ridge_by_model", MappingProxyType(ridges))

    @property
    def hit_gain_by_contrast(self) -> Mapping[str, int]:
        hits = self.model_hit_counts
        return MappingProxyType(
            {
                D1_PLACEBO_CONTRASTS[0]: hits["B0_C_A_snapshot"] - hits["B0_C"],
                D1_PLACEBO_CONTRASTS[1]: hits["B0_C_A_dynamic"] - hits["B0_C"],
                D1_PLACEBO_CONTRASTS[2]: (hits["B0_C_A_dynamic"] - hits["B0_C_A_snapshot"]),
            }
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "support_count": self.support_count,
            "model_hit_counts": dict(self.model_hit_counts),
            "hit_gain_by_contrast": dict(self.hit_gain_by_contrast),
            "selected_alpha": self.selected_alpha,
            "selected_ridge_by_model": dict(self.selected_ridge_by_model),
            "preprocessing_refit": self.preprocessing_refit,
            "alpha_reselected": self.alpha_reselected,
            "ridge_reselected": self.ridge_reselected,
            "models_refit": self.models_refit,
        }


def _normalize_fold_score(raw: object, *, fold_id: str) -> D1FoldPlaceboScore:
    if isinstance(raw, D1FoldPlaceboScore):
        score = raw
    else:
        support = getattr(raw, "support_count", getattr(raw, "cluster_count", None))
        hits = getattr(raw, "model_hit_counts", getattr(raw, "hit_count_by_model", None))
        if (
            not isinstance(support, int)
            or isinstance(support, bool)
            or not isinstance(hits, Mapping)
        ):
            raise TypeError("D1 placebo callback returned an unsupported fold result")
        score = D1FoldPlaceboScore(
            fold_id=str(getattr(raw, "fold_id", "")),
            support_count=support,
            model_hit_counts=cast(Mapping[str, int], hits),
            selected_alpha=cast(float | None, getattr(raw, "selected_alpha", None)),
            selected_ridge_by_model=cast(
                Mapping[str, float],
                getattr(raw, "selected_ridge_by_model", {}),
            ),
            preprocessing_refit=getattr(raw, "preprocessing_refit", False) is True,
            alpha_reselected=getattr(raw, "alpha_reselected", False) is True,
            ridge_reselected=getattr(raw, "ridge_reselected", False) is True,
            models_refit=getattr(raw, "models_refit", False) is True,
        )
    if score.fold_id != fold_id:
        raise ValueError("D1 placebo callback returned another fold")
    return score


@dataclass(frozen=True, slots=True)
class D1PlaceboPreparedReplay:
    issue_sources: tuple[D1PlaceboIssueSource, ...]
    fold_plans: tuple[D1FoldPlaceboPlan, ...]
    snapshots_by_issue_us: Mapping[int, Stage3IssueSnapshot]
    query_grid: Stage3QueryGrid
    construction_stratum_by_state_id: Mapping[str, str]
    all_zone_ids: tuple[str, ...]
    identities: Mapping[str, object]
    score_fold: D1PlaceboFoldScorer = field(repr=False, compare=False)
    expected_issue_count: int = 205
    expected_zone_count: int = 39
    query_chunk_size: int = 256

    def __post_init__(self) -> None:
        sources = _validate_source_axis(
            self.issue_sources,
            expected_issue_count=self.expected_issue_count,
        )
        plans = tuple(self.fold_plans)
        if tuple(item.fold_id for item in plans) != D1_PLACEBO_FOLD_IDS:
            raise ValueError("D1 placebo fold plans must be ordered fold_1, fold_2, fold_3")
        times = {item.issue_time_us for item in sources}
        if any(not set(item.scored_issue_times_us) <= times for item in plans):
            raise ValueError("D1 fold plan references an issue outside the full history")
        if set(self.snapshots_by_issue_us) != times:
            raise ValueError("D1 placebo snapshots do not cover all actual report periods")
        zones = tuple(sorted(set(self.all_zone_ids), key=lambda value: value.encode("utf-8")))
        if len(zones) != self.expected_zone_count:
            raise ValueError("D1 placebo prepared replay has the wrong zone count")
        required_identities = {"contract_sha256", "input_sha256", "git_commit"}
        if not required_identities <= set(self.identities):
            raise ValueError("D1 placebo identities omit contract/input/git binding")
        identities = dict(self.identities)
        _validated_sha256(
            identities["contract_sha256"],
            label="D1 placebo contract SHA-256",
        )
        _validated_sha256(
            identities["input_sha256"],
            label="D1 placebo input SHA-256",
        )
        _validated_git_commit(
            identities["git_commit"],
            label="D1 placebo Git commit",
        )
        if "observed_input_sha256" in identities:
            _validated_sha256(
                identities["observed_input_sha256"],
                label="D1 observed input SHA-256",
            )
        if not callable(self.score_fold):
            raise TypeError("D1 placebo score_fold must be callable")
        object.__setattr__(self, "issue_sources", sources)
        object.__setattr__(self, "fold_plans", plans)
        object.__setattr__(
            self,
            "snapshots_by_issue_us",
            MappingProxyType(dict(self.snapshots_by_issue_us)),
        )
        object.__setattr__(
            self,
            "construction_stratum_by_state_id",
            MappingProxyType(dict(self.construction_stratum_by_state_id)),
        )
        object.__setattr__(self, "all_zone_ids", zones)
        object.__setattr__(self, "identities", MappingProxyType(identities))


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _configured_data_path(
    repository_root: Path,
    configured_data_root: Path,
    raw_path: object,
    *,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} omitted its configured path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must remain repository-relative")
    local = (repository_root / relative).resolve()
    if local.exists():
        return local
    parts = relative.parts
    if not parts or parts[0].casefold() != "data":
        raise FileNotFoundError(local)
    fallback = (configured_data_root / Path(*parts[1:])).resolve()
    try:
        fallback.relative_to(configured_data_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the configured data root") from exc
    return fallback


def _frozen_placebo_bindings(
    prepared: object,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    protocol = getattr(prepared, "protocol", None)
    config = getattr(protocol, "config", None)
    data = _required_mapping(
        _required_mapping(config, label="D1 protocol config").get("data"),
        label="D1 protocol data",
    )
    anomaly = _required_mapping(data.get("anomaly_features"), label="anomaly_features")
    spatial = _required_mapping(data.get("spatial_strata"), label="spatial_strata")
    return data, anomaly, spatial


def bind_d1_placebo_replay(
    observed_prepared_replay: object,
    issue_sources: Sequence[D1PlaceboIssueSource],
    snapshot_history: D1Stage3SnapshotHistory,
    verified_spatial_strata: D1VerifiedSpatialStrata,
    *,
    expected_issue_count: int = 205,
    query_chunk_size: int = 256,
) -> D1PlaceboPreparedReplay:
    """Bind public observed-runner preparation to the complete placebo-only inputs.

    ``prepare_d1_replay`` intentionally reads only model-scored issues.  Placebos add
    the complete 205-report feature/state axis and the separately authenticated local
    spatial strata, while reusing the runner's frozen backgrounds, targets and scorer.
    """

    from seismoflux.d1_replay.runner import (  # Local import avoids a module cycle.
        D1PreparedReplay,
        evaluate_d1_feature_variant_fold,
    )

    if not isinstance(observed_prepared_replay, D1PreparedReplay):
        raise TypeError("bind_d1_placebo_replay requires runner.D1PreparedReplay")
    prepared = observed_prepared_replay
    if not isinstance(snapshot_history, D1Stage3SnapshotHistory):
        raise TypeError("bind_d1_placebo_replay requires hash-carrying snapshot history")
    if not isinstance(verified_spatial_strata, D1VerifiedSpatialStrata):
        raise TypeError("bind_d1_placebo_replay requires verified spatial strata")
    _data, anomaly_binding, spatial_binding = _frozen_placebo_bindings(prepared)
    expected_feature_sha256 = _validated_sha256(
        anomaly_binding.get("feature_store_file_sha256"),
        label="frozen D1 feature-store SHA-256",
    )
    expected_state_sha256 = _validated_sha256(
        anomaly_binding.get("state_history_file_sha256"),
        label="frozen D1 state-history SHA-256",
    )
    source_file_hashes = {item.feature_store_file_sha256 for item in issue_sources}
    if source_file_hashes != {expected_feature_sha256}:
        raise ValueError("D1 issue sources are not the frozen feature-store bytes")
    if snapshot_history.state_history_file_sha256 != expected_state_sha256:
        raise ValueError("D1 snapshots are not the frozen state-history bytes")
    expected_public_manifest_sha256 = _validated_sha256(
        spatial_binding.get("public_manifest_content_sha256"),
        label="frozen D1 spatial public-manifest content SHA-256",
    )
    expected_artifact_hashes_raw = _required_mapping(
        spatial_binding.get("local_artifact_sha256"),
        label="D1 spatial artifact SHA-256 bindings",
    )
    expected_artifact_hashes = {
        key: _validated_sha256(value, label=f"frozen D1 spatial artifact {key} SHA-256")
        for key, value in expected_artifact_hashes_raw.items()
    }
    if (
        verified_spatial_strata.public_manifest_content_sha256 != expected_public_manifest_sha256
        or dict(verified_spatial_strata.artifact_sha256) != expected_artifact_hashes
    ):
        raise ValueError("D1 verified spatial strata differ from the frozen config")
    observed_input_files = _required_mapping(
        prepared.identities.get("input_files"),
        label="D1 observed input-file identities",
    )
    if observed_input_files.get("anomaly_features") != expected_feature_sha256:
        raise ValueError("D1 observed preparation used another feature-store identity")

    state_history_sha256 = snapshot_history.state_history_file_sha256
    snapshots_by_issue_us: Mapping[int, Stage3IssueSnapshot] = snapshot_history
    source_times = {item.issue_time_us for item in issue_sources}
    water_folds_value = prepared.protocol.water_level.get("folds")
    if not isinstance(water_folds_value, list):
        raise ValueError("D1 water-level manifest omitted its frozen folds")
    water_folds: dict[str, Mapping[str, object]] = {}
    for raw_fold in water_folds_value:
        if not isinstance(raw_fold, Mapping):
            raise TypeError("D1 water-level fold must be a mapping")
        fold_id_value = raw_fold.get("fold_id")
        if not isinstance(fold_id_value, str) or fold_id_value in water_folds:
            raise ValueError("D1 water-level fold identity is invalid or duplicated")
        water_folds[fold_id_value] = cast(Mapping[str, object], raw_fold)
    if set(water_folds) != set(D1_PLACEBO_FOLD_IDS):
        raise ValueError("D1 water-level manifest changed its runtime fold identities")

    plans: list[D1FoldPlaceboPlan] = []
    for fold_id in D1_PLACEBO_FOLD_IDS:
        fit_times = tuple(prepared.target_layer.fit_for(fold_id).issue_times_us)
        assessment_times = tuple(
            sorted(
                {
                    value
                    for horizon in (30, 90)
                    for value in prepared.target_layer.assessment_for(
                        fold_id, horizon
                    ).issue_times_us
                }
            )
        )
        if not fit_times or not assessment_times:
            raise ValueError("D1 prepared fold omitted fit or assessment issues")
        water_fold = water_folds[fold_id]
        fit_cutoff_us = _iso_epoch_microseconds(
            water_fold.get("fit_issue_cutoff_local_inclusive"),
            label=f"{fold_id} fit_issue_cutoff_local_inclusive",
        )
        fit_value = water_fold.get("fit")
        if not isinstance(fit_value, Mapping):
            raise TypeError("D1 water-level fit section must be a mapping")
        raw_issue_count = fit_value.get("raw_issue_count")
        latest_raw_issue_us = _iso_epoch_microseconds(
            fit_value.get("latest_raw_issue_local"),
            label=f"{fold_id} latest_raw_issue_local",
        )
        complete_fit_axis = tuple(sorted(value for value in source_times if value <= fit_cutoff_us))
        if (
            isinstance(raw_issue_count, bool)
            or not isinstance(raw_issue_count, int)
            or len(complete_fit_axis) != raw_issue_count
            or not complete_fit_axis
            or complete_fit_axis[-1] != latest_raw_issue_us
            or not set(fit_times) <= set(complete_fit_axis)
        ):
            raise ValueError("D1 complete fit report axis differs from the frozen water level")
        plans.append(
            D1FoldPlaceboPlan(
                fold_id=fold_id,
                fit_cutoff_us=fit_cutoff_us,
                scored_issue_times_us=tuple(sorted({*fit_times, *assessment_times})),
            )
        )

    def score_fold(
        fold_id: str,
        pseudo_features: Mapping[int, D1IssueFeatures],
    ) -> object:
        return evaluate_d1_feature_variant_fold(prepared, fold_id, pseudo_features)

    observed_input_sha256 = _validated_sha256(
        prepared.identities.get("input_sha256"),
        label="D1 observed input SHA-256",
    )
    placebo_only_input_files = {
        "anomaly_feature_store": expected_feature_sha256,
        "anomaly_state_history": state_history_sha256,
        "spatial_public_manifest_content": (verified_spatial_strata.public_manifest_content_sha256),
        **{
            f"spatial_{name}": digest
            for name, digest in sorted(verified_spatial_strata.artifact_sha256.items())
        },
    }
    placebo_input_sha256 = _canonical_sha256(
        {
            "observed_input_sha256": observed_input_sha256,
            "placebo_only_input_files": placebo_only_input_files,
        }
    )
    placebo_input_files = dict(observed_input_files)
    placebo_input_files.update(placebo_only_input_files)
    identities = dict(prepared.identities)
    identities.update(
        {
            "observed_input_sha256": observed_input_sha256,
            "input_sha256": placebo_input_sha256,
            "input_files": placebo_input_files,
            "feature_store_file_sha256": expected_feature_sha256,
            "state_history_file_sha256": state_history_sha256,
            "spatial_public_manifest_content_sha256": (
                verified_spatial_strata.public_manifest_content_sha256
            ),
            "spatial_artifact_sha256": dict(verified_spatial_strata.artifact_sha256),
        }
    )

    return D1PlaceboPreparedReplay(
        issue_sources=tuple(issue_sources),
        fold_plans=tuple(plans),
        snapshots_by_issue_us=snapshots_by_issue_us,
        query_grid=prepared.domain.stage3_grid,
        construction_stratum_by_state_id=(verified_spatial_strata.construction_stratum_by_state_id),
        all_zone_ids=verified_spatial_strata.all_zone_ids,
        identities=identities,
        score_fold=score_fold,
        expected_issue_count=expected_issue_count,
        expected_zone_count=39,
        query_chunk_size=query_chunk_size,
    )


def prepare_d1_placebo_replay(
    observed_prepared_replay: object,
    *,
    query_chunk_size: int = 256,
) -> D1PlaceboPreparedReplay:
    """Load every placebo-only input solely from the observed frozen protocol.

    No path or hash is accepted from a caller.  The actual feature/state file hashes,
    public-manifest content hash and all four local spatial-artifact hashes remain on
    their DTOs and are checked again by :func:`bind_d1_placebo_replay`.
    """

    from seismoflux.d1_replay.runner import D1PreparedReplay

    if not isinstance(observed_prepared_replay, D1PreparedReplay):
        raise TypeError("prepare_d1_placebo_replay requires runner.D1PreparedReplay")
    prepared = observed_prepared_replay
    data, anomaly_binding, spatial_binding = _frozen_placebo_bindings(prepared)
    configured_root_raw = data.get("local_data_root_current_machine")
    if not isinstance(configured_root_raw, str) or not configured_root_raw:
        raise ValueError("D1 protocol omitted local_data_root_current_machine")
    repository_root = Path(prepared.protocol.repository_root).resolve()
    configured_data_root = Path(configured_root_raw).resolve()

    def resolve(raw_path: object, *, label: str) -> Path:
        path = _configured_data_path(
            repository_root,
            configured_data_root,
            raw_path,
            label=label,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    expected_issue_count = anomaly_binding.get("snapshot_count")
    expected_cell_count = anomaly_binding.get("query_cell_count")
    if (
        isinstance(expected_issue_count, bool)
        or not isinstance(expected_issue_count, int)
        or isinstance(expected_cell_count, bool)
        or not isinstance(expected_cell_count, int)
    ):
        raise TypeError("D1 frozen feature dimensions must be integers")
    if expected_issue_count != 205 or expected_cell_count != 15_697:
        raise ValueError("D1 frozen feature dimensions changed from 205 x 15,697")
    stage3_grid = prepared.domain.stage3_grid
    expected_grid = D1StaticGrid(
        grid_id=stage3_grid.grid_id,
        cell_ids=stage3_grid.cell_ids,
        rows=stage3_grid.rows,
        columns=stage3_grid.columns,
        query_x_m=stage3_grid.query_xy_m[:, 0],
        query_y_m=stage3_grid.query_xy_m[:, 1],
        clipped_area_km2=stage3_grid.clipped_area_km2,
    )
    issue_sources = load_d1_placebo_issue_sources(
        resolve(anomaly_binding.get("path"), label="D1 anomaly feature store"),
        expected_file_sha256=_validated_sha256(
            anomaly_binding.get("feature_store_file_sha256"),
            label="frozen D1 feature-store SHA-256",
        ),
        expected_grid=expected_grid,
        expected_issue_count=expected_issue_count,
        expected_cell_count=expected_cell_count,
    )
    snapshot_history = load_d1_stage3_snapshots(
        resolve(
            anomaly_binding.get("state_history_path"),
            label="D1 anomaly state history",
        ),
        expected_file_sha256=_validated_sha256(
            anomaly_binding.get("state_history_file_sha256"),
            label="frozen D1 state-history SHA-256",
        ),
        expected_issue_count=expected_issue_count,
    )
    local_paths = _required_mapping(
        spatial_binding.get("local_coordinate_artifacts_not_committed"),
        label="D1 local spatial-artifact paths",
    )
    expected_artifact_hashes_raw = _required_mapping(
        spatial_binding.get("local_artifact_sha256"),
        label="D1 local spatial-artifact hashes",
    )
    expected_artifact_hashes = {
        key: _validated_sha256(value, label=f"frozen D1 spatial artifact {key} SHA-256")
        for key, value in expected_artifact_hashes_raw.items()
    }
    if set(local_paths) != set(expected_artifact_hashes):
        raise ValueError("D1 spatial artifact paths and hashes have different identities")
    verified_spatial_strata = verify_d1_spatial_strata_files(
        public_manifest_path=resolve(
            spatial_binding.get("public_manifest"),
            label="D1 spatial public manifest",
        ),
        cell_mapping_path=resolve(local_paths.get("cell_mapping"), label="cell mapping"),
        entity_mapping_path=resolve(
            local_paths.get("entity_mapping"),
            label="entity mapping",
        ),
        zone_geometry_path=resolve(
            local_paths.get("zone_geometry"),
            label="zone geometry",
        ),
        connectors_path=resolve(local_paths.get("connectors"), label="zone connectors"),
        expected_public_content_sha256=_validated_sha256(
            spatial_binding.get("public_manifest_content_sha256"),
            label="frozen D1 spatial public-manifest content SHA-256",
        ),
        expected_artifact_sha256=expected_artifact_hashes,
        snapshots_by_issue_us=snapshot_history,
    )
    return bind_d1_placebo_replay(
        prepared,
        issue_sources,
        snapshot_history,
        verified_spatial_strata,
        expected_issue_count=expected_issue_count,
        query_chunk_size=query_chunk_size,
    )


@dataclass(frozen=True, slots=True)
class D1PlaceboFoldReplication:
    replication_index: int
    mapping_sha256: str
    score: D1FoldPlaceboScore | None
    scientific_failure: bool
    failure_type: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.replication_index < D1_PLACEBO_REPLICATIONS:
            raise ValueError("D1 fold placebo replication index is outside [0, 200)")
        _validated_sha256(
            self.mapping_sha256,
            label="D1 fold placebo mapping SHA-256",
        )
        if self.scientific_failure != (self.score is None):
            raise ValueError("D1 fold placebo failure flag disagrees with its score")

    def as_mapping(self) -> dict[str, object]:
        return {
            "replication_index": self.replication_index,
            "mapping_sha256": self.mapping_sha256,
            "scientific_failure": self.scientific_failure,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "score": None if self.score is None else self.score.as_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> D1PlaceboFoldReplication:
        score_value = value.get("score")
        score = None
        if isinstance(score_value, Mapping):
            score = D1FoldPlaceboScore(
                fold_id=str(score_value["fold_id"]),
                support_count=int(cast(int, score_value["support_count"])),
                model_hit_counts=cast(Mapping[str, int], score_value["model_hit_counts"]),
                selected_alpha=cast(float | None, score_value.get("selected_alpha")),
                selected_ridge_by_model=cast(
                    Mapping[str, float], score_value.get("selected_ridge_by_model", {})
                ),
                preprocessing_refit=score_value.get("preprocessing_refit") is True,
                alpha_reselected=score_value.get("alpha_reselected") is True,
                ridge_reselected=score_value.get("ridge_reselected") is True,
                models_refit=score_value.get("models_refit") is True,
            )
        return cls(
            replication_index=int(cast(int, value["replication_index"])),
            mapping_sha256=str(value["mapping_sha256"]),
            score=score,
            scientific_failure=value.get("scientific_failure") is True,
            failure_type=(
                None if value.get("failure_type") is None else str(value["failure_type"])
            ),
            failure_message=(
                None if value.get("failure_message") is None else str(value["failure_message"])
            ),
        )


@dataclass(frozen=True, slots=True)
class D1PlaceboContrastResult:
    contrast: str
    observed_statistic: float
    null_statistics: tuple[float | None, ...]
    null_greater_or_equal_count: int
    scientific_failure_count: int
    scientific_failure_fraction: float
    monte_carlo_p_value: float
    observed_exceeds_fraction: float
    status: PlaceboStatus
    mechanism_promising_for_kind: bool

    def as_mapping(self) -> dict[str, object]:
        return {
            "contrast": self.contrast,
            "observed_statistic": self.observed_statistic,
            "null_statistics": list(self.null_statistics),
            "null_greater_or_equal_count": self.null_greater_or_equal_count,
            "scientific_failure_count": self.scientific_failure_count,
            "scientific_failure_fraction": self.scientific_failure_fraction,
            "monte_carlo_p_value": self.monte_carlo_p_value,
            "denominator": D1_PLACEBO_REPLICATIONS + 1,
            "observed_exceeds_fraction": self.observed_exceeds_fraction,
            "status": self.status,
            "mechanism_promising_for_kind": self.mechanism_promising_for_kind,
        }


def reduce_d1_placebo_contrast(
    contrast: str,
    observed_statistic: float,
    null_statistics: Sequence[float | None],
) -> D1PlaceboContrastResult:
    """Apply the frozen failure-as-null>=observed rule with denominator 201."""

    if contrast not in D1_PLACEBO_CONTRASTS or not math.isfinite(observed_statistic):
        raise ValueError("D1 placebo contrast/observed statistic is invalid")
    nulls = tuple(null_statistics)
    if len(nulls) != D1_PLACEBO_REPLICATIONS:
        raise D1PlaceboInfrastructureInterruption(
            "D1 placebo null sequence is incomplete; resume the same mappings"
        )
    failures = sum(value is None or not math.isfinite(value) for value in nulls)
    finite_nulls = tuple(
        None if value is None or not math.isfinite(value) else float(value) for value in nulls
    )
    greater_or_equal = failures + sum(
        value >= observed_statistic for value in finite_nulls if value is not None
    )
    observed_exceeds = (
        sum(observed_statistic > value for value in finite_nulls if value is not None)
        / D1_PLACEBO_REPLICATIONS
    )
    failure_fraction = failures / D1_PLACEBO_REPLICATIONS
    status: PlaceboStatus = (
        "evidence_insufficient" if failure_fraction > D1_PLACEBO_FAILURE_FRACTION_MAX else "passed"
    )
    return D1PlaceboContrastResult(
        contrast=contrast,
        observed_statistic=float(observed_statistic),
        null_statistics=finite_nulls,
        null_greater_or_equal_count=greater_or_equal,
        scientific_failure_count=failures,
        scientific_failure_fraction=failure_fraction,
        monte_carlo_p_value=(1 + greater_or_equal) / (D1_PLACEBO_REPLICATIONS + 1),
        observed_exceeds_fraction=observed_exceeds,
        status=status,
        mechanism_promising_for_kind=(
            status == "passed" and observed_exceeds > D1_PLACEBO_PROMISING_EXCEED_FRACTION
        ),
    )


@dataclass(frozen=True, slots=True)
class D1PlaceboKindResult:
    kind: PlaceboKind
    contrasts: Mapping[str, D1PlaceboContrastResult]
    fold_scientific_failure_counts: Mapping[str, int]

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "contrasts": {key: value.as_mapping() for key, value in self.contrasts.items()},
            "fold_scientific_failure_counts": dict(self.fold_scientific_failure_counts),
        }


@dataclass(frozen=True, slots=True)
class D1PlaceboResult:
    identities: Mapping[str, object]
    observed_statistics: Mapping[str, float]
    kinds: Mapping[str, D1PlaceboKindResult]
    requested_workers: int
    effective_workers: int
    detected_physical_cores: int
    anomaly_mechanism_promising_by_contrast: Mapping[str, bool]
    status: Literal["completed"] = "completed"

    def __post_init__(self) -> None:
        if (
            isinstance(self.detected_physical_cores, bool)
            or not isinstance(self.detected_physical_cores, int)
            or self.detected_physical_cores <= D1_PLACEBO_RESERVED_PHYSICAL_CORES
            or isinstance(self.requested_workers, bool)
            or not isinstance(self.requested_workers, int)
            or isinstance(self.effective_workers, bool)
            or not isinstance(self.effective_workers, int)
        ):
            raise ValueError("D1 placebo result contains an unsafe resource plan")
        safe_workers = self.detected_physical_cores - D1_PLACEBO_RESERVED_PHYSICAL_CORES
        if self.effective_workers != min(
            self.requested_workers, len(D1_PLACEBO_FOLD_IDS), safe_workers
        ):
            raise ValueError("D1 placebo result contains an unsafe resource plan")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "protocol_version": "d1.0.0",
            "result_kind": "d1_time_and_space_placebos",
            "retrospective_only": True,
            "status": self.status,
            "identities": dict(self.identities),
            "replications_each": D1_PLACEBO_REPLICATIONS,
            "schedule_by_kind_and_fold": {
                kind: dict(folds) for kind, folds in d1_placebo_schedule().items()
            },
            "requested_workers": self.requested_workers,
            "effective_workers": self.effective_workers,
            "blas_threads_per_worker": 1,
            "parallel_unit": "outer_fold",
            "detected_physical_cores": self.detected_physical_cores,
            "reserved_physical_cores": D1_PLACEBO_RESERVED_PHYSICAL_CORES,
            "observed_statistics": dict(self.observed_statistics),
            "kinds": {key: value.as_mapping() for key, value in self.kinds.items()},
            "anomaly_mechanism_promising_by_contrast": dict(
                self.anomaly_mechanism_promising_by_contrast
            ),
            "interpretation": (
                "attribution_only; a failed anomaly attribution does not kill the total model"
            ),
        }


def _checkpoint_path(output_root: Path, kind: PlaceboKind, fold_id: str) -> Path:
    return Path(output_root) / "checkpoints" / f"{kind}_{fold_id}.json"


def _checkpoint_identity(
    identities: Mapping[str, object],
    kind: PlaceboKind,
    fold_id: str,
) -> dict[str, object]:
    if kind not in _PURPOSE_CODE or fold_id not in _PRIMARY_SUPPORT_BY_FOLD:
        raise ValueError("D1 placebo checkpoint kind/fold is invalid")
    return {
        "contract_sha256": _validated_sha256(
            identities.get("contract_sha256"),
            label="D1 checkpoint contract SHA-256",
        ),
        "input_sha256": _validated_sha256(
            identities.get("input_sha256"),
            label="D1 checkpoint input SHA-256",
        ),
        "git_commit": _validated_git_commit(
            identities.get("git_commit"),
            label="D1 checkpoint Git commit",
        ),
        "placebo_kind": kind,
        "fold": fold_id,
        "fold_id": fold_id,
        "expected_support_count": _PRIMARY_SUPPORT_BY_FOLD[fold_id],
    }


def _validate_checkpoint_state(
    *,
    status: object,
    completed: object,
    completed_replication_count: object,
    record_count: int,
) -> Literal["prepared", "running", "invalid_run", "completed"]:
    allowed = {"prepared", "running", "invalid_run", "completed"}
    if not isinstance(status, str) or status not in allowed:
        raise D1PlaceboInfrastructureInterruption("D1 placebo checkpoint status is invalid")
    if not isinstance(completed, bool) or completed != (status == "completed"):
        raise D1PlaceboInfrastructureInterruption(
            "D1 placebo checkpoint completed flag disagrees with status"
        )
    if (
        isinstance(completed_replication_count, bool)
        or not isinstance(completed_replication_count, int)
        or completed_replication_count != record_count
    ):
        raise D1PlaceboInfrastructureInterruption(
            "D1 placebo checkpoint completed count disagrees with records"
        )
    if status == "prepared" and record_count != 0:
        raise D1PlaceboInfrastructureInterruption(
            "prepared D1 placebo checkpoint must contain zero replications"
        )
    if status == "running" and (
        record_count == 0
        or record_count >= D1_PLACEBO_REPLICATIONS
        or record_count % D1_PLACEBO_CHECKPOINT_STRIDE != 0
    ):
        raise D1PlaceboInfrastructureInterruption(
            "running D1 placebo checkpoint is not a partial 25-replication boundary"
        )
    if status == "completed" and record_count != D1_PLACEBO_REPLICATIONS:
        raise D1PlaceboInfrastructureInterruption(
            "completed D1 placebo checkpoint does not contain all 200 replications"
        )
    if status == "invalid_run" and (
        record_count % D1_PLACEBO_CHECKPOINT_STRIDE != 0 and record_count != D1_PLACEBO_REPLICATIONS
    ):
        raise D1PlaceboInfrastructureInterruption(
            "invalid D1 placebo checkpoint is not on a recoverable boundary"
        )
    return cast(
        Literal["prepared", "running", "invalid_run", "completed"],
        status,
    )


def _validate_checkpoint_records(
    records: Sequence[D1PlaceboFoldReplication],
    *,
    fold_id: str,
) -> tuple[D1PlaceboFoldReplication, ...]:
    ordered = tuple(records)
    if tuple(item.replication_index for item in ordered) != tuple(range(len(ordered))):
        raise D1PlaceboInfrastructureInterruption(
            "D1 placebo checkpoint indices are not contiguous"
        )
    expected_support = _PRIMARY_SUPPORT_BY_FOLD[fold_id]
    for item in ordered:
        _validated_sha256(item.mapping_sha256, label="D1 placebo mapping SHA-256")
        if item.score is not None and (
            item.score.fold_id != fold_id or item.score.support_count != expected_support
        ):
            raise D1PlaceboInfrastructureInterruption(
                "D1 placebo checkpoint fold/support differs from frozen 8/6/7"
            )
    return ordered


def write_d1_placebo_checkpoint(
    path: Path,
    *,
    identities: Mapping[str, object],
    kind: PlaceboKind,
    fold_id: str,
    records: Sequence[D1PlaceboFoldReplication],
    status: Literal["prepared", "running", "invalid_run", "completed"],
) -> None:
    _checkpoint_identity(identities, kind, fold_id)
    ordered = _validate_checkpoint_records(records, fold_id=fold_id)
    try:
        _validate_checkpoint_state(
            status=status,
            completed=status == "completed",
            completed_replication_count=len(ordered),
            record_count=len(ordered),
        )
    except D1PlaceboInfrastructureInterruption as exc:
        raise ValueError(str(exc)) from exc
    payload = {
        "schema_version": 1,
        **_checkpoint_identity(identities, kind, fold_id),
        "last_replication": None if not ordered else ordered[-1].replication_index,
        "completed": status == "completed",
        "completed_replication_count": len(ordered),
        "status": status,
        "records": [item.as_mapping() for item in ordered],
    }
    write_json_atomic(Path(path), payload)


def load_d1_placebo_checkpoint(
    path: Path,
    *,
    identities: Mapping[str, object],
    kind: PlaceboKind,
    fold_id: str,
) -> tuple[D1PlaceboFoldReplication, ...]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return ()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise D1PlaceboInfrastructureInterruption("D1 placebo checkpoint is not a mapping")
    if payload.get("schema_version") != 1:
        raise D1PlaceboInfrastructureInterruption("D1 placebo checkpoint schema version changed")
    expected = _checkpoint_identity(identities, kind, fold_id)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise D1PlaceboInfrastructureInterruption(
            "D1 placebo checkpoint contract/input/git/kind/fold identity changed"
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise D1PlaceboInfrastructureInterruption("D1 placebo checkpoint omitted records")
    try:
        records = _validate_checkpoint_records(
            tuple(
                D1PlaceboFoldReplication.from_mapping(
                    _required_mapping(value, label="D1 placebo checkpoint record")
                )
                for value in raw_records
            ),
            fold_id=fold_id,
        )
    except D1PlaceboInfrastructureInterruption:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise D1PlaceboInfrastructureInterruption(
            "D1 placebo checkpoint contains an invalid replication record"
        ) from exc
    if payload.get("last_replication") != (None if not records else records[-1].replication_index):
        raise D1PlaceboInfrastructureInterruption(
            "D1 placebo checkpoint last index is inconsistent"
        )
    _validate_checkpoint_state(
        status=payload.get("status"),
        completed=payload.get("completed"),
        completed_replication_count=payload.get("completed_replication_count"),
        record_count=len(records),
    )
    return records


@dataclass(frozen=True, slots=True)
class _D1PlaceboResourceBoundary:
    physical_cores: int
    effective_workers: int


def _resource_boundary(requested: int) -> _D1PlaceboResourceBoundary:
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or not 1 <= requested <= D1_PLACEBO_MAX_WORKERS
    ):
        raise ValueError(
            f"D1 placebo workers must be an integer from 1 through {D1_PLACEBO_MAX_WORKERS}"
        )
    physical_cores = detect_physical_core_count()
    if (
        isinstance(physical_cores, bool)
        or not isinstance(physical_cores, int)
        or physical_cores <= D1_PLACEBO_RESERVED_PHYSICAL_CORES
    ):
        raise RuntimeError(
            "D1 placebo requires a verified physical-core count above two so two "
            "physical cores can remain free"
        )
    safe_fold_workers = physical_cores - D1_PLACEBO_RESERVED_PHYSICAL_CORES
    return _D1PlaceboResourceBoundary(
        physical_cores=physical_cores,
        effective_workers=min(
            requested,
            len(D1_PLACEBO_FOLD_IDS),
            safe_fold_workers,
        ),
    )


def _bounded_workers(requested: int) -> int:
    """Compatibility helper returning the audited fold-level worker count."""

    return _resource_boundary(requested).effective_workers


def _run_fold_placebos(
    prepared: D1PlaceboPreparedReplay,
    plan: D1FoldPlaceboPlan,
    kind: PlaceboKind,
    output_root: Path,
    baseline: D1ObservedPlaceboBaseline,
    *,
    progress: ProgressCallback | None,
) -> tuple[D1PlaceboFoldReplication, ...]:
    path = _checkpoint_path(output_root, kind, plan.fold_id)
    records = list(
        load_d1_placebo_checkpoint(
            path,
            identities=prepared.identities,
            kind=kind,
            fold_id=plan.fold_id,
        )
    )
    for replication_index in range(len(records), D1_PLACEBO_REPLICATIONS):
        _emit(
            progress,
            phase="placebo",
            kind=kind,
            fold=plan.fold_id,
            replication=replication_index,
        )
        pseudo: D1PseudoHistory | None = None
        if kind == "time":
            pseudo = build_d1_time_pseudo_history(
                prepared.issue_sources,
                plan,
                replication_index,
                expected_issue_count=prepared.expected_issue_count,
            )
        else:
            pseudo = build_d1_space_pseudo_history(
                prepared.issue_sources,
                prepared.snapshots_by_issue_us,
                prepared.query_grid,
                prepared.construction_stratum_by_state_id,
                prepared.all_zone_ids,
                plan,
                replication_index,
                expected_issue_count=prepared.expected_issue_count,
                expected_zone_count=prepared.expected_zone_count,
                query_chunk_size=prepared.query_chunk_size,
            )
        try:
            score = _normalize_fold_score(
                prepared.score_fold(plan.fold_id, pseudo.features_by_issue),
                fold_id=plan.fold_id,
            )
            if score.support_count != baseline.support_by_fold[plan.fold_id]:
                raise D1PlaceboScientificFailure(
                    "pseudo-history changed the observed primary cluster support"
                )
            record = D1PlaceboFoldReplication(
                replication_index=replication_index,
                mapping_sha256=pseudo.mapping_sha256,
                score=score,
                scientific_failure=False,
            )
        except (D1PlaceboScientificFailure, D1ModelFitError, FloatingPointError) as exc:
            # The pseudo mapping/reconstruction completed and is reproducible.  A
            # fold refit/score failure is therefore a scientific failed replicate,
            # conservatively counted as null >= observed.  Reconstruction and
            # checkpoint errors occur outside this block and remain infrastructure
            # interruptions rather than evidence.
            record = D1PlaceboFoldReplication(
                replication_index=replication_index,
                mapping_sha256=pseudo.mapping_sha256,
                score=None,
                scientific_failure=True,
                failure_type=type(exc).__name__,
                failure_message=str(exc)[:500],
            )
        records.append(record)
        if len(records) % D1_PLACEBO_CHECKPOINT_STRIDE == 0:
            write_d1_placebo_checkpoint(
                path,
                identities=prepared.identities,
                kind=kind,
                fold_id=plan.fold_id,
                records=records,
                status=("completed" if len(records) == D1_PLACEBO_REPLICATIONS else "running"),
            )
    return tuple(records)


def _reduce_kind(
    kind: PlaceboKind,
    records_by_fold: Mapping[str, tuple[D1PlaceboFoldReplication, ...]],
    baseline: D1ObservedPlaceboBaseline,
) -> D1PlaceboKindResult:
    nulls: dict[str, list[float | None]] = {contrast: [] for contrast in D1_PLACEBO_CONTRASTS}
    for replication_index in range(D1_PLACEBO_REPLICATIONS):
        records = tuple(records_by_fold[fold][replication_index] for fold in D1_PLACEBO_FOLD_IDS)
        if any(item.scientific_failure for item in records):
            for contrast in D1_PLACEBO_CONTRASTS:
                nulls[contrast].append(None)
            continue
        total_support = sum(cast(D1FoldPlaceboScore, item.score).support_count for item in records)
        for contrast in D1_PLACEBO_CONTRASTS:
            hit_gain = sum(
                cast(D1FoldPlaceboScore, item.score).hit_gain_by_contrast[contrast]
                for item in records
            )
            nulls[contrast].append(hit_gain / total_support)
    contrasts = {
        contrast: reduce_d1_placebo_contrast(
            contrast,
            baseline.observed_statistics[contrast],
            nulls[contrast],
        )
        for contrast in D1_PLACEBO_CONTRASTS
    }
    return D1PlaceboKindResult(
        kind=kind,
        contrasts=MappingProxyType(contrasts),
        fold_scientific_failure_counts=MappingProxyType(
            {
                fold: sum(item.scientific_failure for item in records_by_fold[fold])
                for fold in D1_PLACEBO_FOLD_IDS
            }
        ),
    )


def run_d1_placebos(
    prepared_replay: D1PlaceboPreparedReplay,
    observed_result: object,
    output_root: Path,
    *,
    workers: int = 4,
    progress: ProgressCallback | None = None,
) -> D1PlaceboResult:
    """Run/resume 200 time and 200 space pseudo-histories, each refitted per fold."""

    if not isinstance(prepared_replay, D1PlaceboPreparedReplay):
        raise TypeError("run_d1_placebos requires D1PlaceboPreparedReplay")
    resource_boundary = _resource_boundary(workers)
    effective_workers = resource_boundary.effective_workers
    baseline = observed_d1_placebo_baseline(observed_result)
    for key in ("contract_sha256", "git_commit"):
        observed_identity = baseline.identities.get(key)
        if observed_identity is not None and observed_identity != prepared_replay.identities[key]:
            raise ValueError("observed and placebo D1 identities differ")
    observed_input_identity = baseline.identities.get("input_sha256")
    expected_observed_input_identity = prepared_replay.identities.get(
        "observed_input_sha256",
        prepared_replay.identities["input_sha256"],
    )
    if (
        observed_input_identity is not None
        and observed_input_identity != expected_observed_input_identity
    ):
        raise ValueError("observed and placebo D1 input identities differ")
    destination = Path(output_root).resolve()
    from seismoflux.d1_replay.runner import _single_thread_math_runtime

    progress_lock = Lock()

    def synchronized_progress(payload: Mapping[str, object]) -> None:
        if progress is not None:
            with progress_lock:
                progress(payload)

    thread_safe_progress: ProgressCallback | None = (
        None if progress is None else synchronized_progress
    )
    # This is deliberately fail-closed.  Merely setting environment variables is
    # insufficient when another module imported NumPy first; the runtime limiter
    # must enter successfully and remains held across every fold worker.
    with _single_thread_math_runtime():
        kind_results: dict[str, D1PlaceboKindResult] = {}
        kinds: tuple[PlaceboKind, PlaceboKind] = ("time", "space")
        for kind in kinds:
            if effective_workers == 1:
                fold_records = {
                    plan.fold_id: _run_fold_placebos(
                        prepared_replay,
                        plan,
                        kind,
                        destination,
                        baseline,
                        progress=thread_safe_progress,
                    )
                    for plan in prepared_replay.fold_plans
                }
            else:
                with ThreadPoolExecutor(
                    max_workers=effective_workers,
                    thread_name_prefix=f"d1-{kind}",
                ) as executor:
                    pending = {
                        plan.fold_id: executor.submit(
                            _run_fold_placebos,
                            prepared_replay,
                            plan,
                            kind,
                            destination,
                            baseline,
                            progress=thread_safe_progress,
                        )
                        for plan in prepared_replay.fold_plans
                    }
                    fold_records = {
                        fold_id: pending[fold_id].result() for fold_id in D1_PLACEBO_FOLD_IDS
                    }
            kind_results[kind] = _reduce_kind(kind, fold_records, baseline)
        promising = {
            contrast: all(
                kind_results[kind].contrasts[contrast].mechanism_promising_for_kind
                for kind in ("time", "space")
            )
            for contrast in D1_PLACEBO_CONTRASTS
        }
        result = D1PlaceboResult(
            identities=prepared_replay.identities,
            observed_statistics=baseline.observed_statistics,
            kinds=MappingProxyType(kind_results),
            requested_workers=workers,
            effective_workers=effective_workers,
            detected_physical_cores=resource_boundary.physical_cores,
            anomaly_mechanism_promising_by_contrast=MappingProxyType(promising),
        )
        write_json_atomic(destination / "d1_placebo_result.json", result.as_mapping())
        _emit(progress, phase="placebo", status="completed")
        return result


__all__ = [
    "D1_PLACEBO_CHECKPOINT_STRIDE",
    "D1_PLACEBO_CONTRASTS",
    "D1_PLACEBO_FAILURE_FRACTION_MAX",
    "D1_PLACEBO_MAX_WORKERS",
    "D1_PLACEBO_PROMISING_EXCEED_FRACTION",
    "D1_PLACEBO_REPLICATIONS",
    "D1_PLACEBO_RESERVED_PHYSICAL_CORES",
    "D1CoordinateEntity",
    "D1CoordinatePermutation",
    "D1FoldPlaceboPlan",
    "D1FoldPlaceboScore",
    "D1ObservedPlaceboBaseline",
    "D1PlaceboContrastResult",
    "D1PlaceboFoldReplication",
    "D1PlaceboInfrastructureInterruption",
    "D1PlaceboIssueSource",
    "D1PlaceboKindResult",
    "D1PlaceboPreparedReplay",
    "D1PlaceboResult",
    "D1PlaceboScientificFailure",
    "D1PseudoHistory",
    "D1Stage3SnapshotHistory",
    "D1VerifiedSpatialStrata",
    "bind_d1_placebo_replay",
    "build_d1_space_pseudo_history",
    "build_d1_time_pseudo_history",
    "d1_placebo_rng",
    "d1_placebo_schedule",
    "load_d1_placebo_checkpoint",
    "load_d1_placebo_issue_sources",
    "load_d1_stage3_snapshots",
    "observed_d1_placebo_baseline",
    "permute_d1_coordinates_within_zones",
    "prepare_d1_placebo_replay",
    "rebuild_d1_dynamic_from_pseudo_history",
    "reduce_d1_placebo_contrast",
    "run_d1_placebos",
    "verify_d1_spatial_strata_files",
    "write_d1_placebo_checkpoint",
]
