"""Target-blind assembly and identity preflight for the 153 formal stage-4 issues.

This module is deliberately narrower than the formal scorer.  It can read only
the accepted stage-3 feature/state artifacts, the frozen study area, and the
restricted construction-stratum mapping already covered by a score-blind input
receipt.  It has no catalogue loader, target path parameter, score function, or
locked-test entry point.

The accepted stage-3 and stage-4 25 km grids contain the same cells but use
different role-specific identity digests.  The bridge below first authenticates
every stage-3 grid value and row position, then changes only the in-memory
``grid_id`` column.  It never interpolates, reorders, clips, or reconstructs a
cell from an outcome.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import struct
import time as clock
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta, timezone
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq

from seismoflux.anomaly_increment.compute import Stage4ComputePlan
from seismoflux.anomaly_increment.config import (
    STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH,
    STAGE4_SCORING_CODE_TAG,
    Stage4ProtocolBundle,
    require_stage4_r2_execution_action,
)
from seismoflux.anomaly_increment.contracts import canonical_mapping_sha256
from seismoflux.anomaly_increment.feature_adapter import (
    assert_issue_table_matches_frozen_grid,
    feature_set_contract,
)
from seismoflux.anomaly_increment.formal_assembly import (
    ProspectiveIssuePlan,
    VerifiedStage3Issue,
)
from seismoflux.anomaly_increment.grid_features import (
    SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1,
    Stage4GridFamily,
    Stage4IntegrationGrid,
    assert_selected_columns_logically_exact_r1,
    build_stage4_grid_family,
    selected_table_logical_identity_sha256_r1,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    open_existing_immutable_file,
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
)
from seismoflux.anomaly_increment.placebo import (
    SpaceBijection,
    TimeBijection,
    build_space_bijection,
)
from seismoflux.anomaly_increment.placebo_features import (
    COVERAGE_COLUMNS,
    SNAPSHOT_COLUMNS,
    TRAJECTORY_BASE_COLUMNS,
    TRAJECTORY_COLUMNS,
    SpaceStratumKey,
    rebuild_space_placebo_features,
    rebuild_time_placebo_features,
)
from seismoflux.anomaly_increment.placebo_source import (
    PlaceboScopeAssembler,
    PlaceboSourceUniverse,
)
from seismoflux.anomaly_increment.preregistration import Stage4SeedContext
from seismoflux.anomaly_increment.qualification import ScoreBlindInputEvidence
from seismoflux.anomaly_increment.score_blind_path import (
    require_score_blind_project_path,
)
from seismoflux.anomaly_increment.scoring_pipeline import FeatureLayout
from seismoflux.background.catalog import StudyArea, load_study_area_bytes
from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.data.parquet import schema_sha256
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot, build_issue_snapshots
from seismoflux.features.anomaly.state import states_from_records

FormalInputId = Literal[
    "stage3_registry",
    "stage3_feature_store",
    "stage3_anomaly_state_history",
    "study_area",
]

FORMAL_FIT_ISSUE_COUNT = 50
FORMAL_ASSESSMENT_ISSUE_COUNT = 103
FORMAL_POOL_ISSUE_COUNT = FORMAL_FIT_ISSUE_COUNT + FORMAL_ASSESSMENT_ISSUE_COUNT
EXPECTED_STAGE3_ISSUE_COUNT = 205
FORMAL_PREFLIGHT_RECEIPT_PATH: Final[PurePosixPath] = PurePosixPath(
    STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH
)
AS_IF_SHADOW_STATUS: Final[
    Literal["as_if_shadow_retrospective_target_blind_not_true_prospective"]
] = "as_if_shadow_retrospective_target_blind_not_true_prospective"
GRID_BRIDGE_METHOD: Final[
    Literal[
        "verify_stage3_exact_grid_then_replace_only_grid_id_in_memory_no_reorder_no_interpolation"
    ]
] = "verify_stage3_exact_grid_then_replace_only_grid_id_in_memory_no_reorder_no_interpolation"
IDENTITY_REBUILD_HASH_METHOD: Final[str] = SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1
_SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
_ISSUE_ID = re.compile(r"anomaly-issue-(\d{4}-\d{2}-\d{2})\Z")
_GIT_OID = re.compile(r"[0-9a-f]{40}\Z")

# Exact order consumed by the formal time/space placebo implementation.  The
# Arrow IPC hashes below therefore bind values, validity bitmaps, types, and
# column order without reading the other 1,558 stage-3 columns.
FORMAL_FEATURE_COLUMNS = (
    "issue_index",
    "issue_time_utc",
    "issue_report_id",
    "grid_id",
    "equal_area_crs",
    "cell_size_km",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "clipped_area_km2",
    *COVERAGE_COLUMNS,
    *SNAPSHOT_COLUMNS,
    *TRAJECTORY_BASE_COLUMNS,
    *TRAJECTORY_COLUMNS,
)
GRID_BRIDGE_COLUMNS = (
    "grid_id",
    "equal_area_crs",
    "cell_size_km",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "clipped_area_km2",
)
GRID_BRIDGE_UNCHANGED_COLUMNS = tuple(
    column for column in FORMAL_FEATURE_COLUMNS if column != "grid_id"
)
IDENTITY_REBUILD_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "coverage": tuple(COVERAGE_COLUMNS),
        "snapshot": tuple(SNAPSHOT_COLUMNS),
        "trajectory_base": tuple(TRAJECTORY_BASE_COLUMNS),
        "trajectory": tuple(TRAJECTORY_COLUMNS),
    }
)
_ALLOWED_INPUT_IDS = frozenset(
    {
        "stage3_registry",
        "stage3_feature_store",
        "stage3_anomaly_state_history",
        "study_area",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence")
    output = tuple(value)
    if not output or any(
        not isinstance(item, str) or not item or item != item.strip() for item in output
    ):
        raise ValueError(f"{label} must contain non-empty trimmed strings")
    if len(output) != len(set(output)):
        raise ValueError(f"{label} must contain unique identifiers")
    return cast(tuple[str, ...], output)


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_oid(value: str) -> str:
    if not isinstance(value, str) or _GIT_OID.fullmatch(value) is None:
        raise ValueError("scoring_code_commit must be a lowercase 40-character Git OID")
    return value


def _strict_mapping(
    value: object,
    *,
    label: str,
    keys: frozenset[str],
) -> Mapping[str, object]:
    mapping = _mapping(value, label=label)
    if set(mapping) != keys:
        missing = sorted(keys - set(mapping))
        extra = sorted(set(mapping) - keys)
        raise ValueError(f"{label} schema changed; missing={missing}, extra={extra}")
    return mapping


def _text_field(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty trimmed string")
    return value


def _sha_field(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    return _sha256(mapping.get(key), label=f"{label}.{key}")


def _int_field(
    mapping: Mapping[str, object],
    key: str,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label}.{key} must be an integer >= {minimum}")
    return value


def _optional_int_field(
    mapping: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label}.{key} must be null or a non-negative integer")
    return value


def _false_field(mapping: Mapping[str, object], key: str, *, label: str) -> Literal[False]:
    if mapping.get(key) is not False:
        raise ValueError(f"{label}.{key} must remain false")
    return False


def _utc_text_field(mapping: Mapping[str, object], key: str, *, label: str) -> datetime:
    text = _text_field(mapping, key, label=label)
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}.{key} is not a canonical UTC timestamp") from exc
    parsed = _utc(value, label=f"{label}.{key}")
    if _iso_utc(parsed) != text:
        raise ValueError(f"{label}.{key} is not in canonical UTC form")
    return parsed


def _utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return _utc(value, label="UTC timestamp").isoformat().replace("+00:00", "Z")


def _issue_time(issue_id: str) -> datetime:
    match = _ISSUE_ID.fullmatch(issue_id)
    if match is None:
        raise ValueError(f"invalid frozen anomaly issue ID: {issue_id!r}")
    issue_date = date.fromisoformat(match.group(1))
    return datetime.combine(issue_date, time.min, tzinfo=_SHANGHAI).astimezone(UTC)


def _one_value(table: pa.Table, column: str, *, label: str) -> object:
    if column not in table.column_names:
        raise ValueError(f"{label} omitted {column}")
    values = table[column].combine_chunks().unique().to_pylist()
    if len(values) != 1:
        raise ValueError(f"{label} must contain exactly one {column}")
    return values[0]


def _replace_text_column(table: pa.Table, column: str, value: str) -> pa.Table:
    index = table.schema.get_field_index(column)
    if index < 0:
        raise ValueError(f"cannot replace absent column: {column}")
    field = table.schema.field(index)
    if not pa.types.is_string(field.type):
        raise TypeError(f"{column} must remain a string column")
    return table.set_column(
        index,
        field,
        pa.array([value] * table.num_rows, type=field.type),
    )


def _combined_issue_hash(
    issue_ids: Sequence[str],
    tables: Mapping[str, pa.Table],
    columns: Sequence[str],
) -> str:
    ordered_ids = tuple(issue_ids)
    names = tuple(columns)
    if tuple(tables) != ordered_ids:
        raise ValueError("issue table mapping order differs from the frozen issue order")
    return canonical_mapping_sha256(
        {
            "column_order": list(names),
            "issues": [
                {
                    "issue_id": issue_id,
                    "table_identity_sha256": selected_table_logical_identity_sha256_r1(
                        tables[issue_id], names
                    ),
                }
                for issue_id in ordered_ids
            ],
            "schema_version": 1,
        }
    )


def _logical_selected_table_identity_sha256(
    table: pa.Table,
    columns: Sequence[str],
) -> str:
    """Compatibility wrapper for the frozen R1 logical Arrow identity."""

    return selected_table_logical_identity_sha256_r1(table, columns)


def _combined_logical_identity_hashes(
    issue_ids: Sequence[str],
    columns: Sequence[str],
    table_identity_sha256s: Sequence[str],
) -> str:
    ordered_ids = tuple(issue_ids)
    names = tuple(columns)
    identities = tuple(table_identity_sha256s)
    if len(identities) != len(ordered_ids):
        raise ValueError("logical issue identities differ from the frozen issue count")
    for index, digest in enumerate(identities):
        _sha256(digest, label=f"logical issue identity {index}")
    return canonical_mapping_sha256(
        {
            "column_order": list(names),
            "hash_method": IDENTITY_REBUILD_HASH_METHOD,
            "issues": [
                {
                    "issue_id": issue_id,
                    "table_identity_sha256": digest,
                }
                for issue_id, digest in zip(ordered_ids, identities, strict=True)
            ],
            "schema_version": 1,
        }
    )


def _assert_columns_exact(
    accepted: pa.Table,
    candidate: pa.Table,
    *,
    columns: Sequence[str],
    label: str,
) -> tuple[str, str]:
    names = tuple(columns)
    try:
        accepted_hash = assert_selected_columns_logically_exact_r1(
            accepted,
            candidate,
            columns=names,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} changed R1 logical Arrow identity: {error}") from error
    return accepted_hash, accepted_hash


def resolve_score_blind_input_path(
    protocol: Stage4ProtocolBundle,
    evidence: ScoreBlindInputEvidence,
    *,
    input_id: FormalInputId,
) -> Path:
    """Resolve one allowlisted path; arbitrary and target-path access is impossible."""

    if input_id not in _ALLOWED_INPUT_IDS:
        raise ValueError("formal preflight input is not in the target-blind allowlist")
    if any(name == "earthquake_target" for name, _ in evidence.observed_project_input_hashes):
        raise ValueError("score-blind evidence crossed the target boundary")
    if evidence.protocol_design_sha256 != protocol.design_sha256:
        raise ValueError("score-blind evidence belongs to another protocol design")
    if evidence.random_input_seal_sha256 != protocol.random_input_seal_sha256:
        raise ValueError("score-blind evidence belongs to another random-input seal")
    inputs = _mapping(protocol.protocol.get("inputs"), label="inputs")
    declaration = _mapping(inputs.get(input_id), label=f"inputs.{input_id}")
    relative = declaration.get("path")
    expected = declaration.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"inputs.{input_id}.path must be repository-relative")
    pure = PurePosixPath(relative.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"inputs.{input_id}.path escapes the repository")
    if not isinstance(expected, str):
        raise TypeError(f"inputs.{input_id}.sha256 must be a string")
    observed = dict(evidence.observed_project_input_hashes).get(input_id)
    if observed != expected:
        raise ValueError(f"score-blind evidence does not authenticate inputs.{input_id}")
    root = protocol.repository_root.resolve()
    path = root.joinpath(*pure.parts)
    return require_score_blind_project_path(
        root,
        protocol.protocol,
        path,
        label=f"formal preflight inputs.{input_id}",
    )


@dataclass(frozen=True, slots=True)
class FormalIssueCalendar:
    """The exact frozen formal-validation time-placebo pools."""

    fit_issue_ids: tuple[str, ...]
    assessment_issue_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        fit = _strings(self.fit_issue_ids, label="formal fit issue IDs")
        assessment = _strings(
            self.assessment_issue_ids,
            label="formal assessment issue IDs",
        )
        if len(fit) != FORMAL_FIT_ISSUE_COUNT:
            raise ValueError("formal fit pool must contain exactly 50 issues")
        if len(assessment) != FORMAL_ASSESSMENT_ISSUE_COUNT:
            raise ValueError("formal assessment pool must contain exactly 103 issues")
        if set(fit) & set(assessment):
            raise ValueError("formal fit and assessment issue pools overlap")
        times = tuple(_issue_time(item) for item in (*fit, *assessment))
        if any(left >= right for left, right in pairwise(times)):
            raise ValueError(
                "formal issue pools are missing, reordered, duplicated, or hindsight-filled"
            )
        if (
            fit[0] != "anomaly-issue-2022-07-21"
            or fit[-1] != "anomaly-issue-2023-06-29"
            or assessment[0] != "anomaly-issue-2023-07-06"
            or assessment[-1] != "anomaly-issue-2025-06-26"
        ):
            raise ValueError("formal issue pool boundaries differ from the frozen protocol")
        object.__setattr__(self, "fit_issue_ids", fit)
        object.__setattr__(self, "assessment_issue_ids", assessment)

    @property
    def issue_ids(self) -> tuple[str, ...]:
        return (*self.fit_issue_ids, *self.assessment_issue_ids)

    @property
    def issue_times_utc(self) -> tuple[datetime, ...]:
        return tuple(_issue_time(item) for item in self.issue_ids)

    @classmethod
    def from_protocol(cls, protocol: Stage4ProtocolBundle) -> FormalIssueCalendar:
        formal = _mapping(
            protocol.fold.document.get("formal_validation_fit"),
            label="formal_validation_fit",
        )
        pools = _mapping(
            formal.get("time_permutation_feature_pools"),
            label="formal time_permutation_feature_pools",
        )
        if pools.get("pool_crossing_forbidden") is not True:
            raise ValueError("formal placebo pool crossing prohibition changed")
        if pools.get("pseudo_history_concatenation") != ("permuted_fit_then_permuted_assessment"):
            raise ValueError("formal pseudo-history order changed")
        return cls(
            fit_issue_ids=_strings(pools.get("fit"), label="formal fit pool"),
            assessment_issue_ids=_strings(pools.get("assessment"), label="formal assessment pool"),
        )


@dataclass(frozen=True, slots=True)
class GridIdentityBridge:
    """Authenticated exact-grid role bridge from stage 3 to stage 4."""

    stage3_grid_id: str
    stage4_grid: Stage4IntegrationGrid = field(repr=False, compare=False)
    stage3_grid_identity_sha256: str
    projected_grid_identity_sha256: str
    unchanged_grid_values_sha256: str
    bridge_sha256: str
    method: Literal[
        "verify_stage3_exact_grid_then_replace_only_grid_id_in_memory_no_reorder_no_interpolation"
    ] = GRID_BRIDGE_METHOD

    def __post_init__(self) -> None:
        if not isinstance(self.stage3_grid_id, str) or not self.stage3_grid_id:
            raise ValueError("stage3_grid_id must be non-empty")
        for label, value in (
            ("stage3_grid_identity_sha256", self.stage3_grid_identity_sha256),
            ("projected_grid_identity_sha256", self.projected_grid_identity_sha256),
            ("unchanged_grid_values_sha256", self.unchanged_grid_values_sha256),
            ("bridge_sha256", self.bridge_sha256),
        ):
            _sha256(value, label=label)
        if self.method != GRID_BRIDGE_METHOD:
            raise ValueError("grid identity bridge method changed")
        expected = canonical_mapping_sha256(
            {
                "cell_count": self.stage4_grid.cell_count,
                "method": self.method,
                "projected_grid_identity_sha256": self.projected_grid_identity_sha256,
                "schema_version": 1,
                "stage3_grid_id": self.stage3_grid_id,
                "stage3_grid_identity_sha256": self.stage3_grid_identity_sha256,
                "stage4_grid_id": self.stage4_grid.grid_id,
                "unchanged_grid_values_sha256": self.unchanged_grid_values_sha256,
            }
        )
        if self.bridge_sha256 != expected:
            raise ValueError("grid identity bridge digest changed")

    @property
    def stage3_grid(self) -> Stage4IntegrationGrid:
        return Stage4IntegrationGrid(
            grid_id=self.stage3_grid_id,
            equal_area_crs=self.stage4_grid.equal_area_crs,
            cell_size_km=self.stage4_grid.cell_size_km,
            cell_ids=self.stage4_grid.cell_ids,
            rows=self.stage4_grid.rows,
            columns=self.stage4_grid.columns,
            query_xy_m=self.stage4_grid.query_xy_m,
            clipped_area_km2=self.stage4_grid.clipped_area_km2,
        )

    @classmethod
    def from_accepted_table(
        cls,
        table: pa.Table,
        *,
        stage3_grid_id: str,
        stage4_grid: Stage4IntegrationGrid,
    ) -> GridIdentityBridge:
        _validate_formal_feature_table_schema(table)
        issue_time = _utc(
            _one_value(table, "issue_time_utc", label="accepted stage-3 issue"),
            label="accepted stage-3 issue time",
        )
        accepted_grid = Stage4IntegrationGrid(
            grid_id=stage3_grid_id,
            equal_area_crs=stage4_grid.equal_area_crs,
            cell_size_km=stage4_grid.cell_size_km,
            cell_ids=stage4_grid.cell_ids,
            rows=stage4_grid.rows,
            columns=stage4_grid.columns,
            query_xy_m=stage4_grid.query_xy_m,
            clipped_area_km2=stage4_grid.clipped_area_km2,
        )
        _assert_table_grid(table, issue_time=issue_time, grid=accepted_grid)
        stage3_hash = selected_table_logical_identity_sha256_r1(table, GRID_BRIDGE_COLUMNS)
        unchanged_hash = selected_table_logical_identity_sha256_r1(
            table, tuple(column for column in GRID_BRIDGE_COLUMNS if column != "grid_id")
        )
        projected = _replace_text_column(table, "grid_id", stage4_grid.grid_id)
        _assert_table_grid(projected, issue_time=issue_time, grid=stage4_grid)
        projected_hash = selected_table_logical_identity_sha256_r1(projected, GRID_BRIDGE_COLUMNS)
        bridge_hash = canonical_mapping_sha256(
            {
                "cell_count": stage4_grid.cell_count,
                "method": GRID_BRIDGE_METHOD,
                "projected_grid_identity_sha256": projected_hash,
                "schema_version": 1,
                "stage3_grid_id": stage3_grid_id,
                "stage3_grid_identity_sha256": stage3_hash,
                "stage4_grid_id": stage4_grid.grid_id,
                "unchanged_grid_values_sha256": unchanged_hash,
            }
        )
        return cls(
            stage3_grid_id=stage3_grid_id,
            stage4_grid=stage4_grid,
            stage3_grid_identity_sha256=stage3_hash,
            projected_grid_identity_sha256=projected_hash,
            unchanged_grid_values_sha256=unchanged_hash,
            bridge_sha256=bridge_hash,
        )

    def project(self, table: pa.Table, *, issue_time: datetime) -> pa.Table:
        """Authenticate the accepted grid, then replace only its role digest."""

        _validate_formal_feature_table_schema(table)
        expected_time = _utc(issue_time, label="bridge issue time")
        _assert_table_grid(table, issue_time=expected_time, grid=self.stage3_grid)
        if selected_table_logical_identity_sha256_r1(table, GRID_BRIDGE_COLUMNS) != (
            self.stage3_grid_identity_sha256
        ):
            raise ValueError("accepted stage-3 grid identity changed across issues")
        before = selected_table_logical_identity_sha256_r1(table, GRID_BRIDGE_UNCHANGED_COLUMNS)
        projected = _replace_text_column(table, "grid_id", self.stage4_grid.grid_id)
        _assert_table_grid(projected, issue_time=expected_time, grid=self.stage4_grid)
        after = selected_table_logical_identity_sha256_r1(projected, GRID_BRIDGE_UNCHANGED_COLUMNS)
        if before != after:
            raise ValueError("grid bridge changed a value other than grid_id")
        return projected


@dataclass(frozen=True, slots=True)
class GridIdentityBridgeReceipt:
    """Serializable grid-role identity without reconstructing geometry."""

    stage3_grid_id: str
    stage4_grid_id: str
    cell_count: int
    stage3_grid_identity_sha256: str
    projected_grid_identity_sha256: str
    unchanged_grid_values_sha256: str
    bridge_sha256: str
    method: Literal[
        "verify_stage3_exact_grid_then_replace_only_grid_id_in_memory_no_reorder_no_interpolation"
    ] = GRID_BRIDGE_METHOD

    def __post_init__(self) -> None:
        for value in (self.stage3_grid_id, self.stage4_grid_id):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("grid receipt identifiers must be non-empty")
        if not isinstance(self.cell_count, int) or isinstance(self.cell_count, bool):
            raise TypeError("grid receipt cell_count must be an integer")
        if self.cell_count != 15_697:
            raise ValueError("grid receipt cell count changed")
        for label, value in (
            ("stage3_grid_identity_sha256", self.stage3_grid_identity_sha256),
            ("projected_grid_identity_sha256", self.projected_grid_identity_sha256),
            ("unchanged_grid_values_sha256", self.unchanged_grid_values_sha256),
            ("bridge_sha256", self.bridge_sha256),
        ):
            _sha256(value, label=label)
        if self.method != GRID_BRIDGE_METHOD:
            raise ValueError("grid identity bridge receipt method changed")
        expected = canonical_mapping_sha256(
            {
                "cell_count": self.cell_count,
                "method": self.method,
                "projected_grid_identity_sha256": self.projected_grid_identity_sha256,
                "schema_version": 1,
                "stage3_grid_id": self.stage3_grid_id,
                "stage3_grid_identity_sha256": self.stage3_grid_identity_sha256,
                "stage4_grid_id": self.stage4_grid_id,
                "unchanged_grid_values_sha256": self.unchanged_grid_values_sha256,
            }
        )
        if self.bridge_sha256 != expected:
            raise ValueError("grid identity bridge receipt digest changed")

    @classmethod
    def from_bridge(cls, bridge: GridIdentityBridge) -> GridIdentityBridgeReceipt:
        return cls(
            stage3_grid_id=bridge.stage3_grid_id,
            stage4_grid_id=bridge.stage4_grid.grid_id,
            cell_count=bridge.stage4_grid.cell_count,
            stage3_grid_identity_sha256=bridge.stage3_grid_identity_sha256,
            projected_grid_identity_sha256=bridge.projected_grid_identity_sha256,
            unchanged_grid_values_sha256=bridge.unchanged_grid_values_sha256,
            bridge_sha256=bridge.bridge_sha256,
            method=bridge.method,
        )

    @classmethod
    def from_mapping(cls, value: object) -> GridIdentityBridgeReceipt:
        label = "grid_identity_bridge"
        mapping = _strict_mapping(
            value,
            label=label,
            keys=frozenset(
                {
                    "bridge_sha256",
                    "cell_count",
                    "method",
                    "projected_grid_identity_sha256",
                    "stage3_grid_id",
                    "stage3_grid_identity_sha256",
                    "stage4_grid_id",
                    "unchanged_grid_values_sha256",
                }
            ),
        )
        method = _text_field(mapping, "method", label=label)
        if method != GRID_BRIDGE_METHOD:
            raise ValueError("grid identity bridge receipt method changed")
        return cls(
            stage3_grid_id=_text_field(mapping, "stage3_grid_id", label=label),
            stage4_grid_id=_text_field(mapping, "stage4_grid_id", label=label),
            cell_count=_int_field(mapping, "cell_count", label=label, minimum=1),
            stage3_grid_identity_sha256=_sha_field(
                mapping, "stage3_grid_identity_sha256", label=label
            ),
            projected_grid_identity_sha256=_sha_field(
                mapping, "projected_grid_identity_sha256", label=label
            ),
            unchanged_grid_values_sha256=_sha_field(
                mapping, "unchanged_grid_values_sha256", label=label
            ),
            bridge_sha256=_sha_field(mapping, "bridge_sha256", label=label),
            method=GRID_BRIDGE_METHOD,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "bridge_sha256": self.bridge_sha256,
            "cell_count": self.cell_count,
            "method": self.method,
            "projected_grid_identity_sha256": self.projected_grid_identity_sha256,
            "stage3_grid_id": self.stage3_grid_id,
            "stage3_grid_identity_sha256": self.stage3_grid_identity_sha256,
            "stage4_grid_id": self.stage4_grid_id,
            "unchanged_grid_values_sha256": self.unchanged_grid_values_sha256,
        }


def _assert_table_grid(
    table: pa.Table,
    *,
    issue_time: datetime,
    grid: Stage4IntegrationGrid,
) -> None:
    assert_issue_table_matches_frozen_grid(
        table,
        issue_time_utc=issue_time,
        grid=grid,
    )
    if _one_value(table, "equal_area_crs", label="feature issue table") != (grid.equal_area_crs):
        raise ValueError("feature table equal-area CRS differs from the frozen grid")
    cell_size = _one_value(table, "cell_size_km", label="feature issue table")
    if (
        isinstance(cell_size, bool)
        or not isinstance(cell_size, int | float)
        or float(cell_size) != grid.cell_size_km
    ):
        raise ValueError("feature table cell size differs from the frozen grid")


def _validate_formal_feature_table_schema(table: pa.Table) -> None:
    if not isinstance(table, pa.Table) or table.num_rows <= 0:
        raise TypeError("formal feature issue must be a non-empty Arrow table")
    if tuple(table.column_names) != FORMAL_FEATURE_COLUMNS:
        raise ValueError("formal feature columns are missing, reordered, or drifted")


@dataclass(frozen=True, slots=True)
class FormalIssueIdentityReceipt:
    issue_id: str
    issue_time_utc: datetime
    issue_index: int
    accepted_table_sha256: str
    projected_table_sha256: str
    verified_binding_sha256: str
    state_snapshot_id: str
    lineage_digest: str

    def __post_init__(self) -> None:
        if _issue_time(self.issue_id) != _utc(
            self.issue_time_utc, label="issue identity receipt time"
        ):
            raise ValueError("issue identity receipt date/time changed")
        if (
            not isinstance(self.issue_index, int)
            or isinstance(self.issue_index, bool)
            or self.issue_index < 0
        ):
            raise ValueError("issue identity receipt index must be non-negative")
        for label, value in (
            ("accepted_table_sha256", self.accepted_table_sha256),
            ("projected_table_sha256", self.projected_table_sha256),
            ("verified_binding_sha256", self.verified_binding_sha256),
            ("state_snapshot_id", self.state_snapshot_id),
            ("lineage_digest", self.lineage_digest),
        ):
            _sha256(value, label=label)

    def as_mapping(self) -> dict[str, object]:
        return {
            "accepted_table_sha256": self.accepted_table_sha256,
            "issue_id": self.issue_id,
            "issue_index": self.issue_index,
            "issue_time_utc": _iso_utc(self.issue_time_utc),
            "lineage_digest": self.lineage_digest,
            "projected_table_sha256": self.projected_table_sha256,
            "state_snapshot_id": self.state_snapshot_id,
            "verified_binding_sha256": self.verified_binding_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> FormalIssueIdentityReceipt:
        label = "formal_issue_receipt"
        mapping = _strict_mapping(
            value,
            label=label,
            keys=frozenset(
                {
                    "accepted_table_sha256",
                    "issue_id",
                    "issue_index",
                    "issue_time_utc",
                    "lineage_digest",
                    "projected_table_sha256",
                    "state_snapshot_id",
                    "verified_binding_sha256",
                }
            ),
        )
        return cls(
            issue_id=_text_field(mapping, "issue_id", label=label),
            issue_time_utc=_utc_text_field(mapping, "issue_time_utc", label=label),
            issue_index=_int_field(mapping, "issue_index", label=label),
            accepted_table_sha256=_sha_field(mapping, "accepted_table_sha256", label=label),
            projected_table_sha256=_sha_field(mapping, "projected_table_sha256", label=label),
            verified_binding_sha256=_sha_field(mapping, "verified_binding_sha256", label=label),
            state_snapshot_id=_sha_field(mapping, "state_snapshot_id", label=label),
            lineage_digest=_sha_field(mapping, "lineage_digest", label=label),
        )


@dataclass(frozen=True, slots=True)
class AsIfShadowIdentityReceipt:
    issue_time_utc: datetime
    issue_report_id: str
    issue_index: int
    accepted_table_sha256: str
    projected_table_sha256: str
    verified_binding_sha256: str
    state_snapshot_id: str
    lineage_digest: str
    status: Literal["as_if_shadow_retrospective_target_blind_not_true_prospective"] = (
        AS_IF_SHADOW_STATUS
    )

    def __post_init__(self) -> None:
        _utc(self.issue_time_utc, label="as-if shadow issue time")
        if not self.issue_report_id or self.issue_report_id != self.issue_report_id.strip():
            raise ValueError("as-if shadow report identity is invalid")
        if (
            not isinstance(self.issue_index, int)
            or isinstance(self.issue_index, bool)
            or self.issue_index < FORMAL_POOL_ISSUE_COUNT
        ):
            raise ValueError("as-if shadow must remain physically separate from the formal pool")
        for label, value in (
            ("accepted_table_sha256", self.accepted_table_sha256),
            ("projected_table_sha256", self.projected_table_sha256),
            ("verified_binding_sha256", self.verified_binding_sha256),
            ("state_snapshot_id", self.state_snapshot_id),
            ("lineage_digest", self.lineage_digest),
        ):
            _sha256(value, label=label)
        if self.status != AS_IF_SHADOW_STATUS:
            raise ValueError("as-if shadow status may not claim true prospective operation")

    @property
    def issue_date_local(self) -> date:
        return self.issue_time_utc.astimezone(_SHANGHAI).date()

    def as_mapping(self) -> dict[str, object]:
        return {
            "accepted_table_sha256": self.accepted_table_sha256,
            "issue_date_local": self.issue_date_local.isoformat(),
            "issue_index": self.issue_index,
            "issue_report_id": self.issue_report_id,
            "issue_time_utc": _iso_utc(self.issue_time_utc),
            "lineage_digest": self.lineage_digest,
            "projected_table_sha256": self.projected_table_sha256,
            "state_snapshot_id": self.state_snapshot_id,
            "status": self.status,
            "verified_binding_sha256": self.verified_binding_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsIfShadowIdentityReceipt:
        label = "as_if_shadow"
        mapping = _strict_mapping(
            value,
            label=label,
            keys=frozenset(
                {
                    "accepted_table_sha256",
                    "issue_date_local",
                    "issue_index",
                    "issue_report_id",
                    "issue_time_utc",
                    "lineage_digest",
                    "projected_table_sha256",
                    "state_snapshot_id",
                    "status",
                    "verified_binding_sha256",
                }
            ),
        )
        status = _text_field(mapping, "status", label=label)
        if status != AS_IF_SHADOW_STATUS:
            raise ValueError("as-if shadow status changed")
        receipt = cls(
            issue_time_utc=_utc_text_field(mapping, "issue_time_utc", label=label),
            issue_report_id=_text_field(mapping, "issue_report_id", label=label),
            issue_index=_int_field(
                mapping,
                "issue_index",
                label=label,
                minimum=FORMAL_POOL_ISSUE_COUNT,
            ),
            accepted_table_sha256=_sha_field(mapping, "accepted_table_sha256", label=label),
            projected_table_sha256=_sha_field(mapping, "projected_table_sha256", label=label),
            verified_binding_sha256=_sha_field(mapping, "verified_binding_sha256", label=label),
            state_snapshot_id=_sha_field(mapping, "state_snapshot_id", label=label),
            lineage_digest=_sha_field(mapping, "lineage_digest", label=label),
            status=AS_IF_SHADOW_STATUS,
        )
        if mapping.get("issue_date_local") != receipt.issue_date_local.isoformat():
            raise ValueError("as-if shadow local issue date changed")
        return receipt


@dataclass(frozen=True, slots=True)
class FormalPreflightResources:
    physical_cores: int
    logical_processors: int
    reserve_physical_cores: int
    effective_workers: int
    nested_parallelism: bool
    feature_selected_compressed_bytes: int
    feature_selected_uncompressed_bytes: int
    state_selected_uncompressed_bytes: int
    loading_mode: Literal["row_group_stream_selected_columns"] = "row_group_stream_selected_columns"

    def __post_init__(self) -> None:
        if self.reserve_physical_cores < 2:
            raise ValueError("formal preflight must reserve at least two physical cores")
        if self.effective_workers > max(1, self.physical_cores - self.reserve_physical_cores):
            raise ValueError("formal preflight worker count violates the reserved-core rule")
        if self.effective_workers > 12 or self.nested_parallelism:
            raise ValueError("formal preflight workers exceed the frozen non-nested cap")
        for value in (
            self.feature_selected_compressed_bytes,
            self.feature_selected_uncompressed_bytes,
            self.state_selected_uncompressed_bytes,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("formal preflight byte estimates must be positive integers")
        if self.loading_mode != "row_group_stream_selected_columns":
            raise ValueError("formal preflight loading mode changed")

    def as_mapping(self) -> dict[str, object]:
        return {
            "effective_workers": self.effective_workers,
            "feature_selected_compressed_bytes": self.feature_selected_compressed_bytes,
            "feature_selected_uncompressed_bytes": self.feature_selected_uncompressed_bytes,
            "loading_mode": self.loading_mode,
            "logical_processors": self.logical_processors,
            "nested_parallelism": self.nested_parallelism,
            "physical_cores": self.physical_cores,
            "reserve_physical_cores": self.reserve_physical_cores,
            "state_selected_uncompressed_bytes": self.state_selected_uncompressed_bytes,
        }

    @classmethod
    def from_mapping(cls, value: object) -> FormalPreflightResources:
        label = "resources"
        mapping = _strict_mapping(
            value,
            label=label,
            keys=frozenset(
                {
                    "effective_workers",
                    "feature_selected_compressed_bytes",
                    "feature_selected_uncompressed_bytes",
                    "loading_mode",
                    "logical_processors",
                    "nested_parallelism",
                    "physical_cores",
                    "reserve_physical_cores",
                    "state_selected_uncompressed_bytes",
                }
            ),
        )
        if mapping.get("nested_parallelism") is not False:
            raise ValueError("resources.nested_parallelism must remain false")
        loading_mode = _text_field(mapping, "loading_mode", label=label)
        if loading_mode != "row_group_stream_selected_columns":
            raise ValueError("resources loading mode changed")
        return cls(
            physical_cores=_int_field(mapping, "physical_cores", label=label, minimum=1),
            logical_processors=_int_field(mapping, "logical_processors", label=label, minimum=1),
            reserve_physical_cores=_int_field(
                mapping, "reserve_physical_cores", label=label, minimum=2
            ),
            effective_workers=_int_field(mapping, "effective_workers", label=label, minimum=1),
            nested_parallelism=False,
            feature_selected_compressed_bytes=_int_field(
                mapping,
                "feature_selected_compressed_bytes",
                label=label,
                minimum=1,
            ),
            feature_selected_uncompressed_bytes=_int_field(
                mapping,
                "feature_selected_uncompressed_bytes",
                label=label,
                minimum=1,
            ),
            state_selected_uncompressed_bytes=_int_field(
                mapping,
                "state_selected_uncompressed_bytes",
                label=label,
                minimum=1,
            ),
            loading_mode="row_group_stream_selected_columns",
        )


@dataclass(frozen=True, slots=True)
class SpacePlaceboFeatureIdentityReceipt:
    """Deterministic target-blind replicate-0 feature identity.

    This identity is part of the scientific preflight seal.  It deliberately
    contains no wall-clock or machine-state observation.
    """

    mapping_sha256: str
    output_identity_sha256: str
    coverage_identity_sha256: str
    replication_index: Literal[0]
    group_count: int
    moved_entity_row_count: int
    fixed_point_count: int
    no_effect_group_count: int
    output_arrow_logical_bytes: int
    role: Literal["target_blind_feature_only_benchmark_not_scientific_replication"] = (
        "target_blind_feature_only_benchmark_not_scientific_replication"
    )
    checkpoint_written: Literal[False] = False
    scope_assembled: Literal[False] = False
    score_computed: Literal[False] = False

    def __post_init__(self) -> None:
        for label, digest in (
            ("mapping_sha256", self.mapping_sha256),
            ("output_identity_sha256", self.output_identity_sha256),
            ("coverage_identity_sha256", self.coverage_identity_sha256),
        ):
            _sha256(digest, label=label)
        if self.replication_index != 0:
            raise ValueError("space feature benchmark is frozen to replication index 0")
        for count in (
            self.group_count,
            self.output_arrow_logical_bytes,
        ):
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("space feature identity counts/bytes must be positive")
        for count in (
            self.moved_entity_row_count,
            self.fixed_point_count,
            self.no_effect_group_count,
        ):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("space feature benchmark audit counts must be non-negative")
        if (
            self.role != "target_blind_feature_only_benchmark_not_scientific_replication"
            or self.checkpoint_written
            or self.scope_assembled
            or self.score_computed
        ):
            raise ValueError("space feature benchmark crossed the feature-only boundary")

    def as_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "checkpoint_written": self.checkpoint_written,
            "coverage_identity_sha256": self.coverage_identity_sha256,
            "fixed_point_count": self.fixed_point_count,
            "group_count": self.group_count,
            "mapping_sha256": self.mapping_sha256,
            "moved_entity_row_count": self.moved_entity_row_count,
            "no_effect_group_count": self.no_effect_group_count,
            "output_arrow_logical_bytes": self.output_arrow_logical_bytes,
            "output_identity_sha256": self.output_identity_sha256,
            "replication_index": self.replication_index,
            "role": self.role,
            "schema_version": 1,
            "scope_assembled": self.scope_assembled,
            "score_computed": self.score_computed,
        }
        payload["content_sha256"] = canonical_mapping_sha256(payload)
        return payload

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: object) -> SpacePlaceboFeatureIdentityReceipt:
        label = "space_placebo_feature_identity"
        mapping = _strict_mapping(
            value,
            label=label,
            keys=frozenset(
                {
                    "checkpoint_written",
                    "content_sha256",
                    "coverage_identity_sha256",
                    "fixed_point_count",
                    "group_count",
                    "mapping_sha256",
                    "moved_entity_row_count",
                    "no_effect_group_count",
                    "output_arrow_logical_bytes",
                    "output_identity_sha256",
                    "replication_index",
                    "role",
                    "schema_version",
                    "scope_assembled",
                    "score_computed",
                }
            ),
        )
        if mapping.get("replication_index") != 0 or mapping.get("schema_version") != 1:
            raise ValueError("space feature identity version/index changed")
        for key in ("checkpoint_written", "scope_assembled", "score_computed"):
            _false_field(mapping, key, label=label)
        role = _text_field(mapping, "role", label=label)
        if role != "target_blind_feature_only_benchmark_not_scientific_replication":
            raise ValueError("space feature identity role changed")
        claimed = _sha_field(mapping, "content_sha256", label=label)
        payload = dict(mapping)
        del payload["content_sha256"]
        if canonical_mapping_sha256(payload) != claimed:
            raise ValueError("space feature identity content digest changed")
        receipt = cls(
            mapping_sha256=_sha_field(mapping, "mapping_sha256", label=label),
            output_identity_sha256=_sha_field(mapping, "output_identity_sha256", label=label),
            coverage_identity_sha256=_sha_field(mapping, "coverage_identity_sha256", label=label),
            replication_index=0,
            group_count=_int_field(mapping, "group_count", label=label, minimum=1),
            moved_entity_row_count=_int_field(mapping, "moved_entity_row_count", label=label),
            fixed_point_count=_int_field(mapping, "fixed_point_count", label=label),
            no_effect_group_count=_int_field(mapping, "no_effect_group_count", label=label),
            output_arrow_logical_bytes=_int_field(
                mapping, "output_arrow_logical_bytes", label=label, minimum=1
            ),
        )
        if receipt.content_sha256 != claimed:
            raise ValueError("space feature identity did not canonicalize")
        return receipt


@dataclass(frozen=True, slots=True)
class SpacePlaceboResourceObservationReceipt:
    """Machine-specific replicate-0 telemetry, outside the scientific seal."""

    feature_identity_sha256: str
    output_identity_sha256: str
    elapsed_seconds: float
    process_working_set_before_bytes: int | None
    process_working_set_after_bytes: int | None
    process_peak_working_set_bytes: int | None
    arrow_peak_memory_bytes: int | None
    system_available_memory_bytes: int | None
    conservative_per_replica_bytes: int
    recommended_max_in_flight: int
    role: Literal["target_blind_resource_observation_not_scientific_identity"] = (
        "target_blind_resource_observation_not_scientific_identity"
    )

    def __post_init__(self) -> None:
        _sha256(self.feature_identity_sha256, label="feature_identity_sha256")
        _sha256(self.output_identity_sha256, label="output_identity_sha256")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0.0:
            raise ValueError("space feature resource elapsed time must be positive")
        for count in (
            self.conservative_per_replica_bytes,
            self.recommended_max_in_flight,
        ):
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("space feature resource counts/bytes must be positive")
        for memory_value in (
            self.process_working_set_before_bytes,
            self.process_working_set_after_bytes,
            self.process_peak_working_set_bytes,
            self.arrow_peak_memory_bytes,
            self.system_available_memory_bytes,
        ):
            if memory_value is not None and (
                not isinstance(memory_value, int)
                or isinstance(memory_value, bool)
                or memory_value < 0
            ):
                raise ValueError("space feature resource optional memory values are invalid")
        if self.recommended_max_in_flight > 12:
            raise ValueError("space feature resource exceeded the frozen worker cap")
        if self.role != "target_blind_resource_observation_not_scientific_identity":
            raise ValueError("space feature resource observation role changed")

    def _payload(self) -> dict[str, object]:
        return {
            "arrow_peak_memory_bytes": self.arrow_peak_memory_bytes,
            "conservative_per_replica_bytes": self.conservative_per_replica_bytes,
            "elapsed_seconds_hex": self.elapsed_seconds.hex(),
            "feature_identity_sha256": self.feature_identity_sha256,
            "output_identity_sha256": self.output_identity_sha256,
            "process_peak_working_set_bytes": self.process_peak_working_set_bytes,
            "process_working_set_after_bytes": self.process_working_set_after_bytes,
            "process_working_set_before_bytes": self.process_working_set_before_bytes,
            "recommended_max_in_flight": self.recommended_max_in_flight,
            "role": self.role,
            "schema_version": 1,
            "system_available_memory_bytes": self.system_available_memory_bytes,
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload())

    def as_mapping(self) -> dict[str, object]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> SpacePlaceboResourceObservationReceipt:
        label = "space_placebo_resource_observation"
        mapping = _strict_mapping(
            value,
            label=label,
            keys=frozenset(
                {
                    "arrow_peak_memory_bytes",
                    "conservative_per_replica_bytes",
                    "content_sha256",
                    "elapsed_seconds_hex",
                    "feature_identity_sha256",
                    "output_identity_sha256",
                    "process_peak_working_set_bytes",
                    "process_working_set_after_bytes",
                    "process_working_set_before_bytes",
                    "recommended_max_in_flight",
                    "role",
                    "schema_version",
                    "system_available_memory_bytes",
                }
            ),
        )
        if mapping.get("schema_version") != 1:
            raise ValueError("space resource observation schema version changed")
        role = _text_field(mapping, "role", label=label)
        if role != "target_blind_resource_observation_not_scientific_identity":
            raise ValueError("space resource observation role changed")
        elapsed_hex = _text_field(mapping, "elapsed_seconds_hex", label=label)
        try:
            elapsed = float.fromhex(elapsed_hex)
        except ValueError as exc:
            raise ValueError("space resource elapsed_seconds_hex is invalid") from exc
        claimed = _sha_field(mapping, "content_sha256", label=label)
        payload = dict(mapping)
        del payload["content_sha256"]
        if canonical_mapping_sha256(payload) != claimed:
            raise ValueError("space resource observation content digest changed")
        receipt = cls(
            feature_identity_sha256=_sha_field(mapping, "feature_identity_sha256", label=label),
            output_identity_sha256=_sha_field(mapping, "output_identity_sha256", label=label),
            elapsed_seconds=elapsed,
            process_working_set_before_bytes=_optional_int_field(
                mapping, "process_working_set_before_bytes", label=label
            ),
            process_working_set_after_bytes=_optional_int_field(
                mapping, "process_working_set_after_bytes", label=label
            ),
            process_peak_working_set_bytes=_optional_int_field(
                mapping, "process_peak_working_set_bytes", label=label
            ),
            arrow_peak_memory_bytes=_optional_int_field(
                mapping, "arrow_peak_memory_bytes", label=label
            ),
            system_available_memory_bytes=_optional_int_field(
                mapping, "system_available_memory_bytes", label=label
            ),
            conservative_per_replica_bytes=_int_field(
                mapping, "conservative_per_replica_bytes", label=label, minimum=1
            ),
            recommended_max_in_flight=_int_field(
                mapping, "recommended_max_in_flight", label=label, minimum=1
            ),
        )
        if receipt.content_sha256 != claimed:
            raise ValueError("space resource observation did not canonicalize")
        return receipt


@dataclass(frozen=True, slots=True)
class FormalPreflightReceipt:
    protocol_design_sha256: str
    random_input_seal_sha256: str
    score_blind_input_evidence_sha256: str
    scoring_code_commit: str
    stage3_registry_sha256: str
    stage3_bundle_id: str
    stage3_bundle_identity_sha256: str
    stage3_bundle_manifest_sha256: str
    feature_store_file_sha256: str
    feature_store_content_sha256: str
    feature_store_schema_sha256: str
    state_history_file_sha256: str
    state_history_content_sha256: str
    state_history_schema_sha256: str
    feature_manifest_content_sha256: str
    calendar: FormalIssueCalendar
    bridge: GridIdentityBridgeReceipt
    issue_receipts: tuple[FormalIssueIdentityReceipt, ...]
    accepted_formal_tables_sha256: str
    projected_formal_tables_sha256: str
    state_snapshots_sha256: str
    accepted_identity_group_hashes: tuple[tuple[str, str], ...]
    rebuilt_identity_group_hashes: tuple[tuple[str, str], ...]
    space_placebo_feature_identity: SpacePlaceboFeatureIdentityReceipt
    space_placebo_resource_observation: SpacePlaceboResourceObservationReceipt = field(
        compare=False
    )
    shadow: AsIfShadowIdentityReceipt
    resources: FormalPreflightResources
    target_bytes_read: Literal[False] = False
    target_path_observed: Literal[False] = False
    locked_test_run: Literal[False] = False

    def __post_init__(self) -> None:
        for label, value in (
            ("protocol_design_sha256", self.protocol_design_sha256),
            ("random_input_seal_sha256", self.random_input_seal_sha256),
            ("score_blind_input_evidence_sha256", self.score_blind_input_evidence_sha256),
            ("stage3_registry_sha256", self.stage3_registry_sha256),
            ("stage3_bundle_identity_sha256", self.stage3_bundle_identity_sha256),
            ("stage3_bundle_manifest_sha256", self.stage3_bundle_manifest_sha256),
            ("feature_store_file_sha256", self.feature_store_file_sha256),
            ("feature_store_content_sha256", self.feature_store_content_sha256),
            ("feature_store_schema_sha256", self.feature_store_schema_sha256),
            ("state_history_file_sha256", self.state_history_file_sha256),
            ("state_history_content_sha256", self.state_history_content_sha256),
            ("state_history_schema_sha256", self.state_history_schema_sha256),
            ("feature_manifest_content_sha256", self.feature_manifest_content_sha256),
            ("accepted_formal_tables_sha256", self.accepted_formal_tables_sha256),
            ("projected_formal_tables_sha256", self.projected_formal_tables_sha256),
            ("state_snapshots_sha256", self.state_snapshots_sha256),
        ):
            _sha256(value, label=label)
        _git_oid(self.scoring_code_commit)
        if not self.stage3_bundle_id or self.stage3_bundle_id != self.stage3_bundle_id.strip():
            raise ValueError("stage3_bundle_id must be non-empty")
        receipts = tuple(self.issue_receipts)
        if tuple(item.issue_id for item in receipts) != self.calendar.issue_ids:
            raise ValueError("formal issue receipts are missing or reordered")
        if tuple(item.issue_index for item in receipts) != tuple(range(FORMAL_POOL_ISSUE_COUNT)):
            raise ValueError("formal issue indices changed from the causal 0..152 order")
        accepted_groups = tuple(self.accepted_identity_group_hashes)
        rebuilt_groups = tuple(self.rebuilt_identity_group_hashes)
        expected_names = tuple(IDENTITY_REBUILD_GROUPS)
        if tuple(name for name, _ in accepted_groups) != expected_names:
            raise ValueError("accepted identity rebuild groups changed")
        if tuple(name for name, _ in rebuilt_groups) != expected_names:
            raise ValueError("rebuilt identity rebuild groups changed")
        for label, groups in (
            ("accepted identity group", accepted_groups),
            ("rebuilt identity group", rebuilt_groups),
        ):
            for name, digest in groups:
                if not name:
                    raise ValueError(f"{label} name is empty")
                _sha256(digest, label=f"{label}.{name}")
        if accepted_groups != rebuilt_groups:
            raise ValueError("identity time mapping changed accepted snapshot/trajectory content")
        if self.shadow.issue_time_utc <= self.calendar.issue_times_utc[-1]:
            raise ValueError("as-if shadow input is not physically separate from the formal pool")
        if self.space_placebo_resource_observation.feature_identity_sha256 != (
            self.space_placebo_feature_identity.content_sha256
        ):
            raise ValueError("space resource observation belongs to another feature identity")
        if self.space_placebo_resource_observation.output_identity_sha256 != (
            self.space_placebo_feature_identity.output_identity_sha256
        ):
            raise ValueError("space resource observation output identity changed")
        if self.target_bytes_read or self.target_path_observed or self.locked_test_run:
            raise ValueError("formal preflight crossed a prohibited target boundary")
        object.__setattr__(self, "issue_receipts", receipts)
        object.__setattr__(self, "accepted_identity_group_hashes", accepted_groups)
        object.__setattr__(self, "rebuilt_identity_group_hashes", rebuilt_groups)

    def _payload(self) -> dict[str, object]:
        return {
            "accepted_formal_tables_sha256": self.accepted_formal_tables_sha256,
            "accepted_identity_group_hashes": dict(self.accepted_identity_group_hashes),
            "as_if_shadow": self.shadow.as_mapping(),
            "feature_column_order": list(FORMAL_FEATURE_COLUMNS),
            "feature_manifest_content_sha256": self.feature_manifest_content_sha256,
            "feature_store": {
                "content_sha256": self.feature_store_content_sha256,
                "file_sha256": self.feature_store_file_sha256,
                "schema_sha256": self.feature_store_schema_sha256,
            },
            "formal_assessment_issue_count": len(self.calendar.assessment_issue_ids),
            "formal_fit_issue_count": len(self.calendar.fit_issue_ids),
            "formal_issue_count": len(self.issue_receipts),
            "formal_issue_receipts": [item.as_mapping() for item in self.issue_receipts],
            "grid_identity_bridge": self.bridge.as_mapping(),
            "identity_rebuild_hash_method": IDENTITY_REBUILD_HASH_METHOD,
            "locked_test_run": self.locked_test_run,
            "projected_formal_tables_sha256": self.projected_formal_tables_sha256,
            "protocol_design_sha256": self.protocol_design_sha256,
            "random_input_seal_sha256": self.random_input_seal_sha256,
            "rebuilt_identity_group_hashes": dict(self.rebuilt_identity_group_hashes),
            "role": "stage4_formal_153_issue_target_blind_preflight",
            "schema_version": 1,
            "score_blind_input_evidence_sha256": (self.score_blind_input_evidence_sha256),
            "scoring_code_commit": self.scoring_code_commit,
            "scoring_code_tag": STAGE4_SCORING_CODE_TAG,
            "stage3_bundle": {
                "bundle_id": self.stage3_bundle_id,
                "bundle_identity_sha256": self.stage3_bundle_identity_sha256,
                "bundle_manifest_sha256": self.stage3_bundle_manifest_sha256,
                "registry_sha256": self.stage3_registry_sha256,
            },
            "space_placebo_feature_identity": self.space_placebo_feature_identity.as_mapping(),
            "state_history": {
                "content_sha256": self.state_history_content_sha256,
                "file_sha256": self.state_history_file_sha256,
                "schema_sha256": self.state_history_schema_sha256,
            },
            "state_snapshots_sha256": self.state_snapshots_sha256,
            "target_bytes_read": self.target_bytes_read,
            "target_path_observed": self.target_path_observed,
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload())

    def deterministic_mapping(self) -> dict[str, object]:
        """Return only the machine-independent mapping covered by ``content_sha256``."""

        return self._payload()

    def as_mapping(self) -> dict[str, object]:
        payload = self._payload()
        payload["content_sha256"] = self.content_sha256
        # Machine-dependent observations are frozen in the same JSON envelope,
        # but are intentionally excluded from ``content_sha256`` above.
        payload["resources"] = self.resources.as_mapping()
        payload["space_placebo_resource_observation"] = (
            self.space_placebo_resource_observation.as_mapping()
        )
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> FormalPreflightReceipt:
        """Load a canonical receipt; never trust a self-reported digest alone."""

        label = "formal_preflight_receipt"
        mapping = _strict_mapping(
            value,
            label=label,
            keys=frozenset(
                {
                    "accepted_formal_tables_sha256",
                    "accepted_identity_group_hashes",
                    "as_if_shadow",
                    "content_sha256",
                    "feature_column_order",
                    "feature_manifest_content_sha256",
                    "feature_store",
                    "formal_assessment_issue_count",
                    "formal_fit_issue_count",
                    "formal_issue_count",
                    "formal_issue_receipts",
                    "grid_identity_bridge",
                    "identity_rebuild_hash_method",
                    "locked_test_run",
                    "projected_formal_tables_sha256",
                    "protocol_design_sha256",
                    "random_input_seal_sha256",
                    "rebuilt_identity_group_hashes",
                    "resources",
                    "role",
                    "schema_version",
                    "score_blind_input_evidence_sha256",
                    "scoring_code_commit",
                    "scoring_code_tag",
                    "space_placebo_feature_identity",
                    "space_placebo_resource_observation",
                    "stage3_bundle",
                    "state_history",
                    "state_snapshots_sha256",
                    "target_bytes_read",
                    "target_path_observed",
                }
            ),
        )
        if mapping.get("role") != "stage4_formal_153_issue_target_blind_preflight":
            raise ValueError("formal preflight receipt role changed")
        if mapping.get("schema_version") != 1:
            raise ValueError("formal preflight receipt schema version changed")
        if mapping.get("scoring_code_tag") != STAGE4_SCORING_CODE_TAG:
            raise ValueError("formal preflight scoring-code tag changed")
        if mapping.get("identity_rebuild_hash_method") != IDENTITY_REBUILD_HASH_METHOD:
            raise ValueError("formal preflight identity rebuild hash method changed")
        if mapping.get("feature_column_order") != list(FORMAL_FEATURE_COLUMNS):
            raise ValueError("formal preflight feature column order changed")
        if (
            mapping.get("formal_fit_issue_count") != FORMAL_FIT_ISSUE_COUNT
            or mapping.get("formal_assessment_issue_count") != FORMAL_ASSESSMENT_ISSUE_COUNT
            or mapping.get("formal_issue_count") != FORMAL_POOL_ISSUE_COUNT
        ):
            raise ValueError("formal preflight issue counts changed")
        for key in ("target_bytes_read", "target_path_observed", "locked_test_run"):
            _false_field(mapping, key, label=label)

        claimed = _sha_field(mapping, "content_sha256", label=label)
        deterministic = dict(mapping)
        for key in (
            "content_sha256",
            "resources",
            "space_placebo_resource_observation",
        ):
            del deterministic[key]
        if canonical_mapping_sha256(deterministic) != claimed:
            raise ValueError("formal preflight deterministic content digest changed")

        issue_values = mapping.get("formal_issue_receipts")
        if not isinstance(issue_values, Sequence) or isinstance(issue_values, str | bytes):
            raise TypeError("formal_issue_receipts must be a sequence")
        issue_receipts = tuple(
            FormalIssueIdentityReceipt.from_mapping(item) for item in issue_values
        )
        if len(issue_receipts) != FORMAL_POOL_ISSUE_COUNT:
            raise ValueError("formal preflight must contain exactly 153 issue receipts")
        calendar = FormalIssueCalendar(
            fit_issue_ids=tuple(item.issue_id for item in issue_receipts[:FORMAL_FIT_ISSUE_COUNT]),
            assessment_issue_ids=tuple(
                item.issue_id for item in issue_receipts[FORMAL_FIT_ISSUE_COUNT:]
            ),
        )

        accepted_groups = _strict_mapping(
            mapping.get("accepted_identity_group_hashes"),
            label="accepted_identity_group_hashes",
            keys=frozenset(IDENTITY_REBUILD_GROUPS),
        )
        rebuilt_groups = _strict_mapping(
            mapping.get("rebuilt_identity_group_hashes"),
            label="rebuilt_identity_group_hashes",
            keys=frozenset(IDENTITY_REBUILD_GROUPS),
        )
        accepted_group_pairs = tuple(
            (
                name,
                _sha_field(
                    accepted_groups,
                    name,
                    label="accepted_identity_group_hashes",
                ),
            )
            for name in IDENTITY_REBUILD_GROUPS
        )
        rebuilt_group_pairs = tuple(
            (
                name,
                _sha_field(
                    rebuilt_groups,
                    name,
                    label="rebuilt_identity_group_hashes",
                ),
            )
            for name in IDENTITY_REBUILD_GROUPS
        )
        feature_store = _strict_mapping(
            mapping.get("feature_store"),
            label="feature_store",
            keys=frozenset({"content_sha256", "file_sha256", "schema_sha256"}),
        )
        state_history = _strict_mapping(
            mapping.get("state_history"),
            label="state_history",
            keys=frozenset({"content_sha256", "file_sha256", "schema_sha256"}),
        )
        stage3_bundle = _strict_mapping(
            mapping.get("stage3_bundle"),
            label="stage3_bundle",
            keys=frozenset(
                {
                    "bundle_id",
                    "bundle_identity_sha256",
                    "bundle_manifest_sha256",
                    "registry_sha256",
                }
            ),
        )
        space_identity = SpacePlaceboFeatureIdentityReceipt.from_mapping(
            mapping.get("space_placebo_feature_identity")
        )
        resource_observation = SpacePlaceboResourceObservationReceipt.from_mapping(
            mapping.get("space_placebo_resource_observation")
        )
        receipt = cls(
            protocol_design_sha256=_sha_field(mapping, "protocol_design_sha256", label=label),
            random_input_seal_sha256=_sha_field(mapping, "random_input_seal_sha256", label=label),
            score_blind_input_evidence_sha256=_sha_field(
                mapping, "score_blind_input_evidence_sha256", label=label
            ),
            scoring_code_commit=_git_oid(_text_field(mapping, "scoring_code_commit", label=label)),
            stage3_registry_sha256=_sha_field(
                stage3_bundle, "registry_sha256", label="stage3_bundle"
            ),
            stage3_bundle_id=_text_field(stage3_bundle, "bundle_id", label="stage3_bundle"),
            stage3_bundle_identity_sha256=_sha_field(
                stage3_bundle, "bundle_identity_sha256", label="stage3_bundle"
            ),
            stage3_bundle_manifest_sha256=_sha_field(
                stage3_bundle, "bundle_manifest_sha256", label="stage3_bundle"
            ),
            feature_store_file_sha256=_sha_field(
                feature_store, "file_sha256", label="feature_store"
            ),
            feature_store_content_sha256=_sha_field(
                feature_store, "content_sha256", label="feature_store"
            ),
            feature_store_schema_sha256=_sha_field(
                feature_store, "schema_sha256", label="feature_store"
            ),
            state_history_file_sha256=_sha_field(
                state_history, "file_sha256", label="state_history"
            ),
            state_history_content_sha256=_sha_field(
                state_history, "content_sha256", label="state_history"
            ),
            state_history_schema_sha256=_sha_field(
                state_history, "schema_sha256", label="state_history"
            ),
            feature_manifest_content_sha256=_sha_field(
                mapping, "feature_manifest_content_sha256", label=label
            ),
            calendar=calendar,
            bridge=GridIdentityBridgeReceipt.from_mapping(mapping.get("grid_identity_bridge")),
            issue_receipts=issue_receipts,
            accepted_formal_tables_sha256=_sha_field(
                mapping, "accepted_formal_tables_sha256", label=label
            ),
            projected_formal_tables_sha256=_sha_field(
                mapping, "projected_formal_tables_sha256", label=label
            ),
            state_snapshots_sha256=_sha_field(mapping, "state_snapshots_sha256", label=label),
            accepted_identity_group_hashes=accepted_group_pairs,
            rebuilt_identity_group_hashes=rebuilt_group_pairs,
            space_placebo_feature_identity=space_identity,
            space_placebo_resource_observation=resource_observation,
            shadow=AsIfShadowIdentityReceipt.from_mapping(mapping.get("as_if_shadow")),
            resources=FormalPreflightResources.from_mapping(mapping.get("resources")),
            target_bytes_read=False,
            target_path_observed=False,
            locked_test_run=False,
        )
        if receipt.content_sha256 != claimed or receipt.as_mapping() != dict(mapping):
            raise ValueError("formal preflight receipt failed canonical round-trip")
        return receipt


def _load_formal_preflight_receipt_generic(path: Path) -> FormalPreflightReceipt:
    """Internal receipt parser behind the guarded production loader.

    Unit tests exercise this private parser directly; repository production
    callers must use :func:`load_formal_preflight_receipt`.
    """

    target = Path(os.path.abspath(os.fspath(path)))
    try:
        require_existing_real_directory_tree(
            Path(target.anchor) if target.anchor else Path.cwd(),
            target.parent,
            label="formal preflight receipt directory",
        )
        payload = read_existing_immutable_bytes(
            target,
            label="formal preflight receipt",
        )
        document = json.loads(payload.decode("utf-8"))
    except (UnsafeImmutableFileError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read the formal preflight receipt") from exc
    return FormalPreflightReceipt.from_mapping(document)


def load_formal_preflight_receipt(
    path: Path,
    *,
    protocol: Mapping[str, object],
) -> FormalPreflightReceipt:
    """Strictly load the production receipt after the R2 preflight hard stop."""

    # This call must precede every path conversion, directory probe, open, JSON
    # decode, and digest validation performed by the generic loader.
    require_stage4_r2_execution_action(protocol, action="formal_preflight")
    return _load_formal_preflight_receipt_generic(path)


def load_space_placebo_resource_observation(
    path: Path,
    *,
    protocol: Mapping[str, object],
) -> SpacePlaceboResourceObservationReceipt:
    """Load and authenticate the separately hashed machine resource observation."""

    # Keep this public convenience entry point independently guard-first rather
    # than relying on a downstream public wrapper.
    require_stage4_r2_execution_action(protocol, action="formal_preflight")
    return _load_formal_preflight_receipt_generic(path).space_placebo_resource_observation


@dataclass(frozen=True, slots=True)
class FormalPreflightBundle:
    """In-memory target-blind inputs for formal execution and paired placebos."""

    study_area: StudyArea = field(repr=False, compare=False)
    grid_family: Stage4GridFamily = field(repr=False)
    feature_layout: FeatureLayout
    calendar: FormalIssueCalendar
    issue_tables: Mapping[str, pa.Table] = field(repr=False, compare=False)
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot] = field(repr=False, compare=False)
    verified_issues_by_issue_id: Mapping[str, VerifiedStage3Issue] = field(
        repr=False, compare=False
    )
    construction_stratum_by_state_id: Mapping[str, str] = field(repr=False, compare=False)
    shadow_issue: VerifiedStage3Issue = field(repr=False, compare=False)
    shadow_plans: tuple[ProspectiveIssuePlan, ...]
    receipt: FormalPreflightReceipt

    def __post_init__(self) -> None:
        expected = self.calendar.issue_ids
        tables = dict(self.issue_tables)
        snapshots = dict(self.snapshots_by_issue_id)
        verified = dict(self.verified_issues_by_issue_id)
        if tuple(tables) != expected or tuple(snapshots) != expected or tuple(verified) != expected:
            raise ValueError("formal preflight mappings must follow the exact 153-issue order")
        if tuple(item.issue_index for item in snapshots.values()) != tuple(
            range(FORMAL_POOL_ISSUE_COUNT)
        ):
            raise ValueError("formal snapshots must retain the causal 0..152 order")
        if self.shadow_issue.issue_time_utc <= self.calendar.issue_times_utc[-1]:
            raise ValueError("shadow issue was mixed into the formal placebo pool")
        plans = tuple(self.shadow_plans)
        if not plans or any(
            plan.issue_date_local != self.receipt.shadow.issue_date_local for plan in plans
        ):
            raise ValueError("shadow plans must use only the latest accepted target-blind issue")
        round_trip = FormalPreflightReceipt.from_mapping(self.receipt.as_mapping())
        if round_trip.content_sha256 != self.receipt.content_sha256:
            raise ValueError("formal preflight receipt failed deterministic round-trip")
        object.__setattr__(self, "issue_tables", MappingProxyType(tables))
        object.__setattr__(self, "snapshots_by_issue_id", MappingProxyType(snapshots))
        object.__setattr__(self, "verified_issues_by_issue_id", MappingProxyType(verified))
        object.__setattr__(
            self,
            "construction_stratum_by_state_id",
            MappingProxyType(dict(self.construction_stratum_by_state_id)),
        )
        object.__setattr__(self, "shadow_plans", plans)

    @property
    def verified_issues(self) -> tuple[VerifiedStage3Issue, ...]:
        return (
            *(self.verified_issues_by_issue_id[item] for item in self.calendar.issue_ids),
            self.shadow_issue,
        )

    def build_placebo_universe(
        self,
        *,
        assembly_context_sha256: str,
        scope_assembler: PlaceboScopeAssembler,
        query_chunk_size: int = 256,
    ) -> PlaceboSourceUniverse:
        """Bind target-blind data to the later authorized in-memory assembler.

        Importing lazily avoids making the preflight module a scorer.  The
        ``scope_assembler`` is created only after the one authorized target
        materialization; all data stored in this bundle remain target-blind.
        """

        return PlaceboSourceUniverse(
            fit_issue_ids=self.calendar.fit_issue_ids,
            assessment_issue_ids=self.calendar.assessment_issue_ids,
            issue_tables=self.issue_tables,
            snapshots_by_issue_id=self.snapshots_by_issue_id,
            query_grid=self.grid_family.primary_25km.as_stage3_query_grid(),
            construction_stratum_by_state_id=self.construction_stratum_by_state_id,
            frozen_input_seal_sha256=self.receipt.random_input_seal_sha256,
            source_input_sha256=self.receipt.content_sha256,
            assembly_context_sha256=assembly_context_sha256,
            scope_assembler=scope_assembler,
            query_chunk_size=query_chunk_size,
        )


def _row_group_issue_time(parquet: pq.ParquetFile, index: int) -> datetime:
    schema_index = parquet.schema_arrow.get_field_index("issue_time_utc")
    if schema_index < 0:
        raise ValueError("accepted stage-3 artifact omitted issue_time_utc")
    statistics = parquet.metadata.row_group(index).column(schema_index).statistics
    if statistics is None or not statistics.has_min_max or statistics.min != statistics.max:
        raise ValueError("accepted stage-3 row group must contain exactly one issue time")
    return _utc(statistics.min, label="stage-3 row-group issue time")


def _row_group_map(parquet: pq.ParquetFile, *, label: str) -> dict[datetime, int]:
    times = tuple(
        _row_group_issue_time(parquet, index) for index in range(parquet.metadata.num_row_groups)
    )
    if len(times) != EXPECTED_STAGE3_ISSUE_COUNT or len(set(times)) != len(times):
        raise ValueError(f"{label} must contain 205 unique accepted issue row groups")
    if any(left >= right for left, right in pairwise(times)):
        raise ValueError(f"{label} row groups are missing or reordered")
    return {value: index for index, value in enumerate(times)}


def _selected_storage_bytes(
    parquet: pq.ParquetFile,
    row_groups: Sequence[int],
    columns: Sequence[str] | None,
) -> tuple[int, int]:
    groups = tuple(row_groups)
    if columns is None:
        uncompressed = sum(parquet.metadata.row_group(index).total_byte_size for index in groups)
        return uncompressed, uncompressed
    first = parquet.metadata.row_group(groups[0])
    by_name = {first.column(index).path_in_schema: index for index in range(first.num_columns)}
    missing = sorted(set(columns) - set(by_name))
    if missing:
        raise ValueError(f"accepted stage-3 artifact omitted selected columns: {missing}")
    compressed = sum(
        parquet.metadata.row_group(group).column(by_name[column]).total_compressed_size
        for group in groups
        for column in columns
    )
    uncompressed = sum(
        parquet.metadata.row_group(group).column(by_name[column]).total_uncompressed_size
        for group in groups
        for column in columns
    )
    return compressed, uncompressed


def _load_feature_tables(
    parquet: pq.ParquetFile,
    *,
    calendar: FormalIssueCalendar,
    latest_issue_time: datetime,
    stage3_grid_id: str,
    primary_grid: Stage4IntegrationGrid,
) -> tuple[
    dict[str, pa.Table],
    pa.Table,
    GridIdentityBridge,
    dict[str, str],
    tuple[int, ...],
    int,
]:
    row_group_by_time = _row_group_map(parquet, label="stage-3 feature store")
    required_times = (*calendar.issue_times_utc, latest_issue_time)
    missing = tuple(value for value in required_times if value not in row_group_by_time)
    if missing:
        raise ValueError(f"accepted feature store omitted required issue times: {missing}")
    formal_indices = tuple(row_group_by_time[item] for item in calendar.issue_times_utc)
    if formal_indices != tuple(range(FORMAL_POOL_ISSUE_COUNT)):
        raise ValueError("formal 153 issues differ from the causal first 153 stage-3 row groups")
    if row_group_by_time[latest_issue_time] != EXPECTED_STAGE3_ISSUE_COUNT - 1:
        raise ValueError("as-if shadow input is not the latest accepted stage-3 issue")

    first = parquet.read_row_group(formal_indices[0], columns=list(FORMAL_FEATURE_COLUMNS))
    _validate_formal_feature_table_schema(first)
    bridge = GridIdentityBridge.from_accepted_table(
        first,
        stage3_grid_id=stage3_grid_id,
        stage4_grid=primary_grid,
    )
    accepted_hashes: dict[str, str] = {}
    projected: dict[str, pa.Table] = {}
    for position, (issue_id, issue_time, row_group) in enumerate(
        zip(calendar.issue_ids, calendar.issue_times_utc, formal_indices, strict=True)
    ):
        table = (
            first
            if position == 0
            else parquet.read_row_group(row_group, columns=list(FORMAL_FEATURE_COLUMNS))
        )
        _validate_formal_feature_table_schema(table)
        if (
            _utc(
                _one_value(table, "issue_time_utc", label=f"feature issue {issue_id}"),
                label=f"feature issue {issue_id} time",
            )
            != issue_time
        ):
            raise ValueError("feature issue period is missing, reordered, or hindsight-filled")
        issue_index = _one_value(table, "issue_index", label=f"feature issue {issue_id}")
        if issue_index != position:
            raise ValueError("formal feature issue index differs from causal 0..152 order")
        accepted_hashes[issue_id] = selected_table_logical_identity_sha256_r1(
            table, FORMAL_FEATURE_COLUMNS
        )
        projected[issue_id] = bridge.project(table, issue_time=issue_time)

    shadow = parquet.read_row_group(
        row_group_by_time[latest_issue_time], columns=list(FORMAL_FEATURE_COLUMNS)
    )
    _validate_formal_feature_table_schema(shadow)
    shadow_index = _one_value(shadow, "issue_index", label="as-if shadow feature issue")
    if shadow_index != EXPECTED_STAGE3_ISSUE_COUNT - 1:
        raise ValueError("as-if shadow feature issue index changed")
    return (
        projected,
        shadow,
        bridge,
        accepted_hashes,
        formal_indices,
        row_group_by_time[latest_issue_time],
    )


def _load_snapshots_streaming(
    parquet: pq.ParquetFile,
    *,
    issue_times: Sequence[datetime],
    expected_issue_indices: Sequence[int],
) -> tuple[Stage3IssueSnapshot, ...]:
    row_group_by_time = _row_group_map(parquet, label="stage-3 state history")
    times = tuple(_utc(value, label="requested snapshot time") for value in issue_times)
    indices = tuple(expected_issue_indices)
    if len(times) != len(indices) or not times:
        raise ValueError("snapshot time/index requests must align and be non-empty")
    seen_state_ids: set[str] = set()
    output: list[Stage3IssueSnapshot] = []
    for issue_time, issue_index in zip(times, indices, strict=True):
        row_group = row_group_by_time.get(issue_time)
        if row_group is None:
            raise ValueError("accepted state history omitted a required issue")
        table = parquet.read_row_group(row_group)
        states = states_from_records(table.to_pylist())
        snapshots = build_issue_snapshots(states, expected_issue_count=1)
        snapshot = replace(snapshots[0], issue_index=issue_index)
        if snapshot.issue_time_utc != issue_time:
            raise ValueError("state snapshot issue period changed")
        state_ids = tuple(
            [snapshot.summary.state_id, *(state.state_id for state in snapshot.entities)]
        )
        duplicates = seen_state_ids.intersection(state_ids)
        if duplicates:
            raise ValueError("state IDs were duplicated across accepted issue snapshots")
        seen_state_ids.update(state_ids)
        output.append(snapshot)
    return tuple(output)


def _identity_mapping(issue_ids: tuple[str, ...], *, partition_role: str) -> TimeBijection:
    digest = canonical_mapping_sha256(
        {
            "direction": "recipient_issue_to_same_donor_issue",
            "issue_ids": list(issue_ids),
            "partition_role": partition_role,
            "role": "stage4_preflight_identity_time_mapping",
            "schema_version": 1,
        }
    )
    return TimeBijection(
        recipient_issue_ids=issue_ids,
        donor_issue_ids=issue_ids,
        mapping_sha256=digest,
        fixed_point_count=len(issue_ids),
    )


def _identity_rebuild_hashes(
    tables: Mapping[str, pa.Table],
    calendar: FormalIssueCalendar,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    rebuilt = rebuild_time_placebo_features(
        tables,
        fit_bijection=_identity_mapping(calendar.fit_issue_ids, partition_role="fit"),
        assessment_bijection=_identity_mapping(
            calendar.assessment_issue_ids, partition_role="assessment"
        ),
    )
    if tuple(rebuilt) != calendar.issue_ids:
        raise ValueError("identity trajectory rebuild changed the formal issue order")
    accepted_hashes: list[tuple[str, str]] = []
    rebuilt_hashes: list[tuple[str, str]] = []
    for group, columns in IDENTITY_REBUILD_GROUPS.items():
        accepted_issue_hashes: list[str] = []
        rebuilt_issue_hashes: list[str] = []
        for issue_id in calendar.issue_ids:
            accepted_issue_hash, rebuilt_issue_hash = _assert_columns_exact(
                tables[issue_id],
                rebuilt[issue_id],
                columns=columns,
                label=f"identity rebuild {group} for {issue_id}",
            )
            accepted_issue_hashes.append(accepted_issue_hash)
            rebuilt_issue_hashes.append(rebuilt_issue_hash)
        accepted_hashes.append(
            (
                group,
                _combined_logical_identity_hashes(
                    calendar.issue_ids,
                    columns,
                    accepted_issue_hashes,
                ),
            )
        )
        rebuilt_hashes.append(
            (
                group,
                _combined_logical_identity_hashes(
                    calendar.issue_ids,
                    columns,
                    rebuilt_issue_hashes,
                ),
            )
        )
    return tuple(accepted_hashes), tuple(rebuilt_hashes)


def _feature_layout(protocol: Stage4ProtocolBundle) -> FeatureLayout:
    return FeatureLayout(
        coverage_sources=feature_set_contract(protocol, "coverage_only").source_columns,
        snapshot_sources=feature_set_contract(protocol, "snapshot").source_columns,
        dynamic_sources=feature_set_contract(protocol, "dynamic").source_columns,
    )


@dataclass(frozen=True, slots=True)
class _OpenedStage3FormalInputs:
    feature_declaration: Mapping[str, object]
    state_declaration: Mapping[str, object]
    latest_issue_time: datetime
    issue_tables: dict[str, pa.Table]
    accepted_shadow_table: pa.Table
    bridge: GridIdentityBridge
    accepted_hashes: dict[str, str]
    formal_indices: tuple[int, ...]
    shadow_index: int
    formal_snapshots: tuple[Stage3IssueSnapshot, ...]
    shadow_snapshot: Stage3IssueSnapshot
    feature_selected_compressed_bytes: int
    feature_selected_uncompressed_bytes: int
    state_selected_uncompressed_bytes: int


def _load_opened_stage3_formal_inputs(
    protocol: Stage4ProtocolBundle,
    *,
    feature_path: Path,
    state_path: Path,
    calendar: FormalIssueCalendar,
    bundle_metadata: Mapping[str, object],
    primary_grid: Stage4IntegrationGrid,
) -> _OpenedStage3FormalInputs:
    """Consume both large Parquet inputs through stable no-follow handles."""

    try:
        with (
            open_existing_immutable_file(
                feature_path,
                label="accepted stage-3 feature store",
            ) as feature_handle,
            open_existing_immutable_file(
                state_path,
                label="accepted stage-3 state history",
            ) as state_handle,
        ):
            feature_parquet = pq.ParquetFile(feature_handle)
            state_parquet = pq.ParquetFile(state_handle)
            feature_declaration = _verify_artifact_declaration(
                protocol,
                input_id="stage3_feature_store",
                parquet=feature_parquet,
            )
            state_declaration = _verify_artifact_declaration(
                protocol,
                input_id="stage3_anomaly_state_history",
                parquet=state_parquet,
            )
            feature_times = tuple(
                _row_group_issue_time(feature_parquet, index)
                for index in range(feature_parquet.metadata.num_row_groups)
            )
            latest_issue_time = max(feature_times)
            (
                issue_tables,
                accepted_shadow_table,
                bridge,
                accepted_hashes,
                formal_indices,
                shadow_index,
            ) = _load_feature_tables(
                feature_parquet,
                calendar=calendar,
                latest_issue_time=latest_issue_time,
                stage3_grid_id=cast(str, bundle_metadata["stage3_grid_id"]),
                primary_grid=primary_grid,
            )
            formal_snapshots = _load_snapshots_streaming(
                state_parquet,
                issue_times=calendar.issue_times_utc,
                expected_issue_indices=tuple(range(FORMAL_POOL_ISSUE_COUNT)),
            )
            shadow_snapshot = _load_snapshots_streaming(
                state_parquet,
                issue_times=(latest_issue_time,),
                expected_issue_indices=(shadow_index,),
            )[0]
            selected_compressed, selected_uncompressed = _selected_storage_bytes(
                feature_parquet,
                formal_indices,
                FORMAL_FEATURE_COLUMNS,
            )
            state_indices = tuple(range(FORMAL_POOL_ISSUE_COUNT))
            _, state_uncompressed = _selected_storage_bytes(
                state_parquet,
                state_indices,
                None,
            )
    except UnsafeImmutableFileError as exc:
        raise ValueError("accepted stage-3 Parquet input is an unsafe alias") from exc
    return _OpenedStage3FormalInputs(
        feature_declaration=feature_declaration,
        state_declaration=state_declaration,
        latest_issue_time=latest_issue_time,
        issue_tables=issue_tables,
        accepted_shadow_table=accepted_shadow_table,
        bridge=bridge,
        accepted_hashes=accepted_hashes,
        formal_indices=formal_indices,
        shadow_index=shadow_index,
        formal_snapshots=formal_snapshots,
        shadow_snapshot=shadow_snapshot,
        feature_selected_compressed_bytes=selected_compressed,
        feature_selected_uncompressed_bytes=selected_uncompressed,
        state_selected_uncompressed_bytes=state_uncompressed,
    )


def _stage3_bundle_metadata(
    protocol: Stage4ProtocolBundle,
    feature_store_path: Path,
) -> Mapping[str, object]:
    inputs = _mapping(protocol.protocol.get("inputs"), label="inputs")
    feature = _mapping(inputs.get("stage3_feature_store"), label="stage3_feature_store")
    bundle_manifest = feature_store_path.parent / "manifest.json"
    bundle_manifest = require_score_blind_project_path(
        protocol.repository_root,
        protocol.protocol,
        bundle_manifest,
        label="accepted stage-3 bundle manifest",
    )
    expected_manifest_hash = _sha256(
        feature.get("bundle_manifest_sha256"), label="stage3 bundle manifest SHA-256"
    )
    try:
        manifest_payload = read_existing_immutable_bytes(
            bundle_manifest,
            label="accepted stage-3 bundle manifest",
        )
    except UnsafeImmutableFileError as exc:
        raise ValueError("cannot safely read accepted stage-3 bundle manifest") from exc
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest_hash:
        raise ValueError("accepted stage-3 bundle manifest changed")
    try:
        document = _mapping(
            json.loads(manifest_payload.decode("utf-8")),
            label="accepted stage-3 bundle manifest",
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("accepted stage-3 bundle manifest is invalid JSON") from exc
    if document.get("bundle_id") != feature.get("bundle_id"):
        raise ValueError("stage-3 bundle ID differs from the frozen protocol")
    if document.get("identity_sha256") != feature.get("bundle_identity_sha256"):
        raise ValueError("stage-3 bundle identity differs from the frozen protocol")
    identity = _mapping(document.get("identity"), label="stage-3 bundle identity")
    grid = _mapping(identity.get("grid"), label="stage-3 bundle grid identity")
    if grid.get("cell_count") != 15_697 or float(cast(float, grid.get("cell_size_km"))) != 25.0:
        raise ValueError("accepted stage-3 bundle grid shape changed")
    return MappingProxyType(
        {
            "bundle_id": cast(str, document["bundle_id"]),
            "bundle_identity_sha256": cast(str, document["identity_sha256"]),
            "bundle_manifest_sha256": expected_manifest_hash,
            "stage3_grid_id": cast(str, grid["grid_id"]),
        }
    )


def _verify_artifact_declaration(
    protocol: Stage4ProtocolBundle,
    *,
    input_id: str,
    parquet: pq.ParquetFile,
) -> Mapping[str, object]:
    inputs = _mapping(protocol.protocol.get("inputs"), label="inputs")
    declaration = _mapping(inputs.get(input_id), label=f"inputs.{input_id}")
    expected_schema = _sha256(
        declaration.get("schema_sha256"), label=f"inputs.{input_id}.schema_sha256"
    )
    if schema_sha256(parquet.schema_arrow) != expected_schema:
        raise ValueError(f"inputs.{input_id} schema differs from the frozen declaration")
    if parquet.metadata.num_row_groups != EXPECTED_STAGE3_ISSUE_COUNT:
        raise ValueError(f"inputs.{input_id} row-group count changed")
    if parquet.metadata.num_rows != declaration.get("row_count"):
        raise ValueError(f"inputs.{input_id} row count changed")
    return declaration


def _load_construction_strata(
    protocol: Stage4ProtocolBundle,
    evidence: ScoreBlindInputEvidence,
    *,
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot],
) -> dict[str, str]:
    local = _mapping(
        _mapping(
            protocol.protocol.get("spatial_permutation_topology"),
            label="spatial_permutation_topology",
        ).get("local_restricted_artifacts"),
        label="local_restricted_artifacts",
    )
    relative = local.get("entity_mapping")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("restricted entity mapping path must be repository-relative")
    path = require_score_blind_project_path(
        protocol.repository_root,
        protocol.protocol,
        protocol.repository_root / relative,
        label="restricted entity-stratum mapping",
    )
    expected = dict(evidence.restricted_spatial_artifact_hashes).get("entity_mapping")
    try:
        with open_existing_immutable_file(
            path,
            label="restricted entity-stratum mapping",
        ) as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            if expected is None or digest.hexdigest() != expected:
                raise ValueError("restricted entity-stratum mapping changed")
            handle.seek(0)
            table = pq.read_table(
                handle,
                columns=["state_id", "issue_time_utc", "construction_stratum_id"],
            )
    except UnsafeImmutableFileError as exc:
        raise ValueError("cannot safely read restricted entity-stratum mapping") from exc
    if table.num_rows <= 0:
        raise ValueError("restricted entity-stratum mapping is empty")
    eligible: dict[str, datetime] = {}
    for snapshot in snapshots_by_issue_id.values():
        for state in snapshot.entities:
            if state.spatial_eligible:
                if state.state_id in eligible:
                    raise ValueError("eligible state ID was duplicated across snapshots")
                eligible[state.state_id] = snapshot.issue_time_utc
    state_ids = table["state_id"].combine_chunks().to_pylist()
    times = table["issue_time_utc"].combine_chunks().to_pylist()
    strata = table["construction_stratum_id"].combine_chunks().to_pylist()
    output: dict[str, str] = {}
    for state_id, issue_time, stratum in zip(state_ids, times, strata, strict=True):
        if state_id not in eligible:
            continue
        if (
            not isinstance(state_id, str)
            or not isinstance(stratum, str)
            or not stratum
            or stratum != stratum.strip()
            or _utc(issue_time, label="entity-stratum issue time") != eligible[state_id]
        ):
            raise ValueError("restricted entity-stratum row changed identity")
        if state_id in output:
            raise ValueError("restricted entity-stratum mapping duplicated a state")
        output[state_id] = stratum
    if set(output) != set(eligible):
        raise ValueError("restricted construction strata do not cover exactly eligible states")
    return output


def _shadow_plans(issue_date_local: date) -> tuple[ProspectiveIssuePlan, ...]:
    magnitude_bins: tuple[Literal["M5_6", "M6_plus"], ...] = ("M5_6", "M6_plus")
    return tuple(
        ProspectiveIssuePlan(
            issue_date_local=issue_date_local,
            horizon_days=horizon,
            magnitude_bin=magnitude,
        )
        for horizon in (7, 30, 90, 180, 365)
        for magnitude in magnitude_bins
    )


@dataclass(frozen=True, slots=True)
class _HostMemoryObservation:
    process_working_set_bytes: int | None
    process_peak_working_set_bytes: int | None
    system_available_memory_bytes: int | None


def _host_memory_observation() -> _HostMemoryObservation:
    """Best-effort process/system memory observation with no optional dependency."""

    if os.name != "nt":
        return _HostMemoryObservation(None, None, None)
    try:
        api = cast(Any, ctypes).windll
        kernel32 = api.kernel32
        psapi = api.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        pointer_bytes = ctypes.sizeof(ctypes.c_void_p)
        counter_size = 88 if pointer_bytes == 8 else 44
        counters = ctypes.create_string_buffer(counter_size)
        struct.pack_into("<I", counters, 0, counter_size)
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counter_size,
        ):
            working_set = None
            peak_working_set = None
        else:
            size_format = "<Q" if pointer_bytes == 8 else "<I"
            size_width = 8 if pointer_bytes == 8 else 4
            peak_working_set = int(struct.unpack_from(size_format, counters.raw, 8)[0])
            working_set = int(struct.unpack_from(size_format, counters.raw, 8 + size_width)[0])

        memory_status = ctypes.create_string_buffer(64)
        struct.pack_into("<I", memory_status, 0, 64)
        available = (
            int(struct.unpack_from("<Q", memory_status.raw, 16)[0])
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
            else None
        )
        return _HostMemoryObservation(working_set, peak_working_set, available)
    except (AttributeError, OSError, TypeError, ValueError, struct.error):
        return _HostMemoryObservation(None, None, None)


def _space_bijections_for_replication_zero(
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot],
    *,
    calendar: FormalIssueCalendar,
    construction_stratum_by_state_id: Mapping[str, str],
    frozen_input_seal_sha256: str,
) -> tuple[dict[SpaceStratumKey, SpaceBijection], str, int, int, int]:
    fit = set(calendar.fit_issue_ids)
    assessment = set(calendar.assessment_issue_ids)
    eligible_state_ids: set[str] = set()
    grouped: dict[SpaceStratumKey, list[tuple[str, tuple[float, float]]]] = {}
    for issue_id, snapshot in sorted(
        snapshots_by_issue_id.items(),
        key=lambda item: item[1].issue_index,
    ):
        for state in snapshot.entities:
            if not state.spatial_eligible:
                continue
            if (
                state.state_id in eligible_state_ids
                or state.longitude is None
                or state.latitude is None
                or not math.isfinite(state.longitude)
                or not math.isfinite(state.latitude)
            ):
                raise ValueError("eligible space benchmark states must be unique and finite")
            stratum = construction_stratum_by_state_id.get(state.state_id)
            if not isinstance(stratum, str) or not stratum or stratum != stratum.strip():
                raise ValueError("eligible space benchmark state omitted its frozen stratum")
            eligible_state_ids.add(state.state_id)
            grouped.setdefault((issue_id, stratum), []).append(
                (state.state_id, (float(state.longitude), float(state.latitude)))
            )
    if set(construction_stratum_by_state_id) != eligible_state_ids:
        raise ValueError("space benchmark strata do not exactly cover eligible states")
    if not grouped:
        raise ValueError("space benchmark has no eligible issue/stratum groups")

    mappings: dict[SpaceStratumKey, SpaceBijection] = {}
    audits: list[dict[str, object]] = []
    for (issue_id, stratum), rows in sorted(grouped.items()):
        role: Literal["fit", "assessment"]
        if issue_id in fit:
            role = "fit"
        elif issue_id in assessment:
            role = "assessment"
        else:
            raise ValueError("space benchmark group escaped the frozen formal pools")
        ordered = tuple(sorted(rows, key=lambda item: item[0]))
        mapping = build_space_bijection(
            tuple(item[0] for item in ordered),
            tuple(item[1] for item in ordered),
            context=Stage4SeedContext(
                purpose="space_permutation",
                evaluation_id="formal-validation",
                partition_role=role,
                replicate_index=0,
                frozen_input_seal_sha256=frozen_input_seal_sha256,
                issue_id=issue_id,
                construction_stratum_id=stratum,
            ),
        )
        mappings[(issue_id, stratum)] = mapping
        audits.append(
            {
                "construction_stratum_id": stratum,
                "fixed_point_count": mapping.fixed_point_count,
                "issue_id": issue_id,
                "mapping_sha256": mapping.mapping_sha256,
                "moved_entity_row_count": mapping.moved_entity_row_count,
                "no_effect": mapping.no_effect,
                "partition_role": role,
            }
        )
    aggregate = canonical_mapping_sha256(
        {
            "evaluation_id": "formal-validation",
            "frozen_input_seal_sha256": frozen_input_seal_sha256,
            "group_audits": audits,
            "kind": "space_feature_only_preflight_benchmark",
            "replication_index": 0,
            "schema_version": 1,
        }
    )
    return (
        mappings,
        aggregate,
        sum(mapping.moved_entity_row_count for mapping in mappings.values()),
        sum(mapping.fixed_point_count for mapping in mappings.values()),
        sum(mapping.no_effect for mapping in mappings.values()),
    )


def _space_placebo_feature_benchmark(
    issue_tables: Mapping[str, pa.Table],
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot],
    *,
    calendar: FormalIssueCalendar,
    grid: Stage4IntegrationGrid,
    construction_stratum_by_state_id: Mapping[str, str],
    frozen_input_seal_sha256: str,
    resources: FormalPreflightResources,
    query_chunk_size: int = 256,
) -> tuple[SpacePlaceboFeatureIdentityReceipt, SpacePlaceboResourceObservationReceipt]:
    """Run one target-blind feature-only space permutation for sizing/identity."""

    if tuple(issue_tables) != calendar.issue_ids:
        raise ValueError("space benchmark issue order differs from the frozen formal calendar")
    mappings, mapping_sha256, moved, fixed, no_effect = _space_bijections_for_replication_zero(
        snapshots_by_issue_id,
        calendar=calendar,
        construction_stratum_by_state_id=construction_stratum_by_state_id,
        frozen_input_seal_sha256=frozen_input_seal_sha256,
    )
    before = _host_memory_observation()
    started = clock.perf_counter()
    rebuilt = rebuild_space_placebo_features(
        issue_tables,
        snapshots_by_issue_id,
        grid.as_stage3_query_grid(),
        verified_construction_stratum_by_state_id=construction_stratum_by_state_id,
        bijections_by_issue_stratum=mappings,
        query_chunk_size=query_chunk_size,
    )
    elapsed = clock.perf_counter() - started
    after = _host_memory_observation()
    if tuple(rebuilt) != calendar.issue_ids:
        raise ValueError("space benchmark output changed the formal issue order")
    for issue_id in calendar.issue_ids:
        if tuple(rebuilt[issue_id].column_names) != FORMAL_FEATURE_COLUMNS:
            raise ValueError("space benchmark output changed the 79-column schema/order")
        _assert_columns_exact(
            issue_tables[issue_id],
            rebuilt[issue_id],
            columns=COVERAGE_COLUMNS,
            label=f"space benchmark coverage for {issue_id}",
        )
    output_identity = _combined_issue_hash(
        calendar.issue_ids,
        rebuilt,
        FORMAL_FEATURE_COLUMNS,
    )
    coverage_identity = _combined_issue_hash(
        calendar.issue_ids,
        rebuilt,
        COVERAGE_COLUMNS,
    )
    output_bytes = sum(table.nbytes for table in rebuilt.values())
    identity = SpacePlaceboFeatureIdentityReceipt(
        mapping_sha256=mapping_sha256,
        output_identity_sha256=output_identity,
        coverage_identity_sha256=coverage_identity,
        replication_index=0,
        group_count=len(mappings),
        moved_entity_row_count=moved,
        fixed_point_count=fixed,
        no_effect_group_count=no_effect,
        output_arrow_logical_bytes=output_bytes,
    )
    conservative_bytes = (
        resources.feature_selected_uncompressed_bytes
        + resources.state_selected_uncompressed_bytes
        + output_bytes
    )
    available = before.system_available_memory_bytes
    memory_cap = (
        max(1, int((available * 0.5) // conservative_bytes)) if available is not None else 1
    )
    recommended = min(resources.effective_workers, 12, memory_cap)
    observation = SpacePlaceboResourceObservationReceipt(
        feature_identity_sha256=identity.content_sha256,
        output_identity_sha256=identity.output_identity_sha256,
        elapsed_seconds=elapsed,
        process_working_set_before_bytes=before.process_working_set_bytes,
        process_working_set_after_bytes=after.process_working_set_bytes,
        process_peak_working_set_bytes=after.process_peak_working_set_bytes,
        arrow_peak_memory_bytes=int(pa.default_memory_pool().max_memory()),
        system_available_memory_bytes=available,
        conservative_per_replica_bytes=conservative_bytes,
        recommended_max_in_flight=recommended,
    )
    return identity, observation


def load_formal_preflight(
    protocol: Stage4ProtocolBundle,
    score_blind_inputs: ScoreBlindInputEvidence,
    compute_plan: Stage4ComputePlan,
    *,
    scoring_code_commit: str,
) -> FormalPreflightBundle:
    """Load, authenticate, and assemble every target-blind formal input.

    The 153 formal issues are fixed by the formal-validation placebo pools.  The
    latest accepted stage-3 issue is loaded separately as an ``as-if shadow``
    input and is never inserted into either formal pool.
    """

    require_stage4_r2_execution_action(protocol.protocol, action="formal_preflight")
    commit = _git_oid(scoring_code_commit)
    if compute_plan.backend != "cpu_float64":
        raise ValueError("formal preflight must use the frozen CPU float64 backend")
    workers = compute_plan.workers
    if workers.reserve_physical_cores < 2 or workers.nested_parallelism:
        raise ValueError("formal preflight compute plan violates reserved-core governance")

    calendar = FormalIssueCalendar.from_protocol(protocol)
    feature_layout = _feature_layout(protocol)
    feature_path = resolve_score_blind_input_path(
        protocol,
        score_blind_inputs,
        input_id="stage3_feature_store",
    )
    state_path = resolve_score_blind_input_path(
        protocol,
        score_blind_inputs,
        input_id="stage3_anomaly_state_history",
    )
    study_path = resolve_score_blind_input_path(
        protocol,
        score_blind_inputs,
        input_id="study_area",
    )
    # Resolving the registry binds the accepted public manifest without parsing
    # any catalogue or score-bearing artifact.
    resolve_score_blind_input_path(
        protocol,
        score_blind_inputs,
        input_id="stage3_registry",
    )

    try:
        study_payload = read_existing_immutable_bytes(
            study_path,
            label="frozen stage-4 study area",
        )
    except UnsafeImmutableFileError as exc:
        raise ValueError("frozen stage-4 study area is an unsafe alias") from exc
    study_area = load_study_area_bytes(study_payload, EQUAL_AREA_CRS)
    grid_family = build_stage4_grid_family(study_area.geographic)
    primary_grid = grid_family.primary_25km
    bundle_metadata = _stage3_bundle_metadata(protocol, feature_path)
    opened = _load_opened_stage3_formal_inputs(
        protocol,
        feature_path=feature_path,
        state_path=state_path,
        calendar=calendar,
        bundle_metadata=bundle_metadata,
        primary_grid=primary_grid,
    )
    feature_declaration = opened.feature_declaration
    state_declaration = opened.state_declaration
    latest_issue_time = opened.latest_issue_time
    issue_tables = opened.issue_tables
    accepted_shadow_table = opened.accepted_shadow_table
    bridge = opened.bridge
    accepted_hashes = opened.accepted_hashes
    shadow_index = opened.shadow_index
    projected_shadow_table = bridge.project(
        accepted_shadow_table,
        issue_time=latest_issue_time,
    )
    formal_snapshots = opened.formal_snapshots
    shadow_snapshot = opened.shadow_snapshot
    snapshots_by_issue_id = dict(zip(calendar.issue_ids, formal_snapshots, strict=True))
    verified: dict[str, VerifiedStage3Issue] = {}
    issue_receipts: list[FormalIssueIdentityReceipt] = []
    sources = tuple(feature_layout.dynamic_sources)
    for issue_id, snapshot in snapshots_by_issue_id.items():
        table = issue_tables[issue_id]
        bound = VerifiedStage3Issue.bind(
            table,
            snapshot,
            primary_grid=primary_grid,
            source_columns=sources,
        )
        verified[issue_id] = bound
        issue_receipts.append(
            FormalIssueIdentityReceipt(
                issue_id=issue_id,
                issue_time_utc=snapshot.issue_time_utc,
                issue_index=snapshot.issue_index,
                accepted_table_sha256=accepted_hashes[issue_id],
                projected_table_sha256=selected_table_logical_identity_sha256_r1(
                    table, FORMAL_FEATURE_COLUMNS
                ),
                verified_binding_sha256=bound.binding_sha256,
                state_snapshot_id=snapshot.state_snapshot_id,
                lineage_digest=snapshot.lineage_digest,
            )
        )
    bound_shadow = VerifiedStage3Issue.bind(
        projected_shadow_table,
        shadow_snapshot,
        primary_grid=primary_grid,
        source_columns=sources,
    )

    accepted_group_hashes, rebuilt_group_hashes = _identity_rebuild_hashes(
        issue_tables,
        calendar,
    )
    construction_strata = _load_construction_strata(
        protocol,
        score_blind_inputs,
        snapshots_by_issue_id=snapshots_by_issue_id,
    )

    resources = FormalPreflightResources(
        physical_cores=workers.physical_cores,
        logical_processors=workers.logical_processors,
        reserve_physical_cores=workers.reserve_physical_cores,
        effective_workers=workers.effective_workers,
        nested_parallelism=workers.nested_parallelism,
        feature_selected_compressed_bytes=(opened.feature_selected_compressed_bytes),
        feature_selected_uncompressed_bytes=(opened.feature_selected_uncompressed_bytes),
        state_selected_uncompressed_bytes=(opened.state_selected_uncompressed_bytes),
    )
    space_identity, space_resource_observation = _space_placebo_feature_benchmark(
        issue_tables,
        snapshots_by_issue_id,
        calendar=calendar,
        grid=primary_grid,
        construction_stratum_by_state_id=construction_strata,
        frozen_input_seal_sha256=protocol.random_input_seal_sha256,
        resources=resources,
    )

    inputs = _mapping(protocol.protocol.get("inputs"), label="inputs")
    stage3_registry = _mapping(inputs.get("stage3_registry"), label="stage3_registry")
    shadow_receipt = AsIfShadowIdentityReceipt(
        issue_time_utc=latest_issue_time,
        issue_report_id=bound_shadow.issue_report_id,
        issue_index=shadow_index,
        accepted_table_sha256=selected_table_logical_identity_sha256_r1(
            accepted_shadow_table, FORMAL_FEATURE_COLUMNS
        ),
        projected_table_sha256=selected_table_logical_identity_sha256_r1(
            projected_shadow_table, FORMAL_FEATURE_COLUMNS
        ),
        verified_binding_sha256=bound_shadow.binding_sha256,
        state_snapshot_id=bound_shadow.state_snapshot_id,
        lineage_digest=bound_shadow.lineage_digest,
    )
    receipt = FormalPreflightReceipt(
        protocol_design_sha256=protocol.design_sha256,
        random_input_seal_sha256=protocol.random_input_seal_sha256,
        score_blind_input_evidence_sha256=score_blind_inputs.content_sha256,
        scoring_code_commit=commit,
        stage3_registry_sha256=cast(str, stage3_registry["sha256"]),
        stage3_bundle_id=cast(str, bundle_metadata["bundle_id"]),
        stage3_bundle_identity_sha256=cast(str, bundle_metadata["bundle_identity_sha256"]),
        stage3_bundle_manifest_sha256=cast(str, bundle_metadata["bundle_manifest_sha256"]),
        feature_store_file_sha256=cast(str, feature_declaration["sha256"]),
        feature_store_content_sha256=cast(str, feature_declaration["content_sha256"]),
        feature_store_schema_sha256=cast(str, feature_declaration["schema_sha256"]),
        state_history_file_sha256=cast(str, state_declaration["sha256"]),
        state_history_content_sha256=cast(str, state_declaration["content_sha256"]),
        state_history_schema_sha256=cast(str, state_declaration["schema_sha256"]),
        feature_manifest_content_sha256=protocol.feature_set.content_sha256,
        calendar=calendar,
        bridge=GridIdentityBridgeReceipt.from_bridge(bridge),
        issue_receipts=tuple(issue_receipts),
        accepted_formal_tables_sha256=canonical_mapping_sha256(
            {
                "column_order": list(FORMAL_FEATURE_COLUMNS),
                "issues": [
                    {"issue_id": item, "sha256": accepted_hashes[item]}
                    for item in calendar.issue_ids
                ],
            }
        ),
        projected_formal_tables_sha256=_combined_issue_hash(
            calendar.issue_ids,
            issue_tables,
            FORMAL_FEATURE_COLUMNS,
        ),
        state_snapshots_sha256=canonical_mapping_sha256(
            {
                "issues": [
                    {
                        "issue_id": issue_id,
                        "issue_time_utc": _iso_utc(snapshot.issue_time_utc),
                        "lineage_digest": snapshot.lineage_digest,
                        "state_snapshot_id": snapshot.state_snapshot_id,
                    }
                    for issue_id, snapshot in snapshots_by_issue_id.items()
                ],
                "schema_version": 1,
            }
        ),
        accepted_identity_group_hashes=accepted_group_hashes,
        rebuilt_identity_group_hashes=rebuilt_group_hashes,
        space_placebo_feature_identity=space_identity,
        space_placebo_resource_observation=space_resource_observation,
        shadow=shadow_receipt,
        resources=resources,
    )
    return FormalPreflightBundle(
        study_area=study_area,
        grid_family=grid_family,
        feature_layout=feature_layout,
        calendar=calendar,
        issue_tables=issue_tables,
        snapshots_by_issue_id=snapshots_by_issue_id,
        verified_issues_by_issue_id=verified,
        construction_stratum_by_state_id=construction_strata,
        shadow_issue=bound_shadow,
        shadow_plans=_shadow_plans(shadow_receipt.issue_date_local),
        receipt=receipt,
    )


__all__ = [
    "AS_IF_SHADOW_STATUS",
    "FORMAL_ASSESSMENT_ISSUE_COUNT",
    "FORMAL_FEATURE_COLUMNS",
    "FORMAL_FIT_ISSUE_COUNT",
    "FORMAL_POOL_ISSUE_COUNT",
    "FORMAL_PREFLIGHT_RECEIPT_PATH",
    "AsIfShadowIdentityReceipt",
    "FormalIssueCalendar",
    "FormalIssueIdentityReceipt",
    "FormalPreflightBundle",
    "FormalPreflightReceipt",
    "FormalPreflightResources",
    "GridIdentityBridge",
    "GridIdentityBridgeReceipt",
    "SpacePlaceboFeatureIdentityReceipt",
    "SpacePlaceboResourceObservationReceipt",
    "load_formal_preflight",
    "load_formal_preflight_receipt",
    "load_space_placebo_resource_observation",
    "resolve_score_blind_input_path",
]
