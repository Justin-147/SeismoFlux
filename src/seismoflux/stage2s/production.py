"""Fail-closed formal Stage 2S execution.

The public entry point in :mod:`seismoflux.stage2s.runner` delegates here only
after the immutable code tag exists.  This module keeps the one-shot boundary
small and explicit:

* remote release identities and tracked worktree bytes are verified first;
* synthetic acceptance and the transitive import audit precede real inputs;
* target-independent spatial inputs are opened before the attempt is claimed;
* the attempt and target read are each claimed by one canonical ``O_EXCL`` file;
* the earthquake catalogue path is physically opened exactly once;
* all fit, prediction, and assessment roles share the parsed in-memory object.

Tests may inject :class:`FormalExecutionServices`; the default services are the
only path used by the public formal runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely import covers, points

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.local_support import (
    LocalSupportCellLocator,
    LocalSupportSnapshot,
    build_local_support_manifest,
    build_local_support_snapshot,
)
from seismoflux.background.local_support_manifest import (
    BackgroundLocalSupportManifest,
    validate_background_local_support_study_area,
)
from seismoflux.background.scientific import scientific_mapping
from seismoflux.stage2s.calendar import (
    M4_MINIMUM,
    M5_6_MAXIMUM_EXCLUSIVE,
    M5_6_MINIMUM,
    AssessmentTargetMembership,
    FitTargetMembership,
    FoldCalendar,
    Stage2SFoldCalendar,
    assessment_target_memberships,
    parse_frozen_fold_manifest_bytes,
)
from seismoflux.stage2s.catalog import (
    Stage2SEarthquakeCatalog,
    parse_frozen_catalog_bytes,
)
from seismoflux.stage2s.contracts import (
    AlarmMask,
    AlphaFit,
    FitEventOrder,
    NormalizedSpatialDensity,
    SharedRate,
    SignedLogDerivative,
    SpatialQuadratureFamily,
    Stage2SModels,
)
from seismoflux.stage2s.evaluation import (
    CONTRASTS,
    HORIZONS,
    METRICS,
    BootstrapFamilies,
    CellScore,
    EventBlock,
    GateAssessment,
    LatencyMetrics,
    MetricKey,
    RegionContribution,
    RegionRobustness,
    SequenceDiagnostic,
    SequenceEvent,
    Stage2SEvidenceInsufficient,
    bootstrap_families,
    build_sequence_closure_evidence,
    compute_region_robustness,
    compute_sequence_diagnostic,
    descriptive_sp_minus_s0_point_estimates,
    evaluate_stage2s_gate,
    score_fold_horizon,
)
from seismoflux.stage2s.execution_environment import (
    require_prepared_formal_execution_environment,
)
from seismoflux.stage2s.governance import (
    audit_stage2s_import_closure,
    verify_stage2s_import_closure_release,
)
from seismoflux.stage2s.inputs import (
    NonTargetPreflight,
    run_non_target_spatial_preflight,
    to_spatial_quadrature_family,
    validated_non_target_preflight_receipt_bindings,
)
from seismoflux.stage2s.protocol import (
    CODE_TAG,
    PROTOCOL_COMMIT,
    PROTOCOL_TAG,
    Stage2SProtocolBundle,
    verify_local_code_tag,
    verify_local_protocol_tag,
)
from seismoflux.stage2s.records import Stage2SWholeRunRecord
from seismoflux.stage2s.rendering import (
    ALL_ARTIFACT_NAMES,
    Stage2SMapFrame,
    Stage2SRenderedBundle,
    Stage2SRenderPayload,
    build_rank_map_frame,
    render_stage2s_bundle,
)
from seismoflux.stage2s.seals import (
    RAW_CATALOG_FIELDS,
    RoleSession,
    SealedRecord,
    write_o_excl_record,
)
from seismoflux.stage2s.spatial import (
    build_normalized_kde,
    build_recent_component,
    build_stage2s_models,
    estimate_shared_m5_6_rate,
    event_cell_index_25km,
    fit_alpha_log_density,
    select_alarm_prefix,
)
from seismoflux.stage2s.synthetic import run_synthetic_acceptance

_FOLDS = (1, 2, 3)
_DELAYS = (0, 1, 7)
_MODEL_ORDER = ("S0", "S1", "SP")
_ONE_MEGABYTE = 1_048_576
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_EXPECTED_REMOTE_REPOSITORY = "github.com/Justin-147/SeismoFlux"
_RESULT_RELATIVE_PATH = Path(
    "data/interim/stage2s/causal_seismicity_screen/stage2s_whole_run_record.json"
)
_TERMINAL_RELATIVE_PATH = Path(
    "data/interim/stage2s/causal_seismicity_screen/terminal_failure_record.json"
)
_ARTIFACT_ROOT_RELATIVE_PATH = Path("outputs/stage2s/causal_seismicity_screen")
_NON_TARGET_PREFLIGHT_REQUIRED_BINDINGS = (
    "protocol_commit_tag_and_config_fold_input_hashes",
    "code_commit_and_tag",
    "study_area_file_projected_geometry_area_and_identity_algorithm",
    "loader_query_and_operational_grid_builder_source_hashes",
    "fold4_support_identity_and_manifest_hash",
    "cell_mapping_file_hash_schema_grid_id_15697_cells_and_39_zones",
    "aligned_12_5_25_50km_grid_ids_areas_parent_relations_and_representative_points",
    "query_grid_and_cell_zone_mapping_returned_together",
)
_BOOTSTRAP_COLUMN_ORDER: tuple[MetricKey, ...] = (
    ("S1_minus_S0", "IG"),
    ("S1_minus_S0", "recall"),
    ("S1_minus_SP", "IG"),
    ("S1_minus_SP", "recall"),
)
ProgressCallback = Callable[[str], None]


class Stage2SFormalError(RuntimeError):
    """Raised when the immutable formal path cannot continue safely."""


@dataclass(frozen=True, slots=True)
class FormalPreflightContext:
    """Target-independent evidence carried across the one-shot boundary."""

    spatial: NonTargetPreflight
    support_manifest: BackgroundLocalSupportManifest
    calendar: Stage2SFoldCalendar
    receipt_bindings: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FormalScienceInputs:
    """All identities already consumed before formal science starts."""

    protocol: Stage2SProtocolBundle
    code_commit: str
    preflight: FormalPreflightContext
    preflight_receipt: SealedRecord
    attempt_ledger: SealedRecord
    target_read_receipt: SealedRecord
    progress: ProgressCallback | None = None


@dataclass(frozen=True, slots=True)
class FormalExecutionServices:
    """Injectable one-shot adapters used to prove ordering without real data."""

    verify_release: Callable[[Stage2SProtocolBundle], str]
    run_code_acceptance: Callable[[Stage2SProtocolBundle], str]
    audit_imports: Callable[[Path], str]
    build_preflight: Callable[
        [Stage2SProtocolBundle, str, str, str],
        FormalPreflightContext,
    ]
    read_catalog_once: Callable[[Path], bytes]
    parse_catalog: Callable[[bytes], Stage2SEarthquakeCatalog]
    run_science: Callable[
        [FormalScienceInputs, Stage2SEarthquakeCatalog],
        Stage2SWholeRunRecord,
    ]
    audit_imports_release: Callable[[Path, str], Mapping[str, object]] | None = None
    execution_environment_bindings: Mapping[str, object] = field(
        default_factory=dict,
    )


@dataclass(frozen=True, slots=True)
class _PredictionSpec:
    fold_index: int
    issue_date: str
    issue_time_utc: datetime
    horizons_days: tuple[int, ...]
    alpha_by_delay: Mapping[int, tuple[float, float]]
    receipt_by_delay: Mapping[int, Mapping[str, object]]
    issue_seal: SealedRecord
    source_view: _CatalogRoleView


@dataclass(frozen=True, slots=True)
class _FoldFit:
    fold: FoldCalendar
    alpha_fit_by_delay: Mapping[int, tuple[AlphaFit, AlphaFit]]
    shared_rate: SharedRate
    fit_receipt: SealedRecord


@dataclass(frozen=True, slots=True)
class _CatalogRoleView:
    """One coordinate-bearing catalogue slice opened for one legal role cutoff."""

    catalog: Stage2SEarthquakeCatalog
    role_id: str
    cutoff_utc: datetime | None
    indices: NDArray[np.int64]
    origin_time_us: NDArray[np.int64]
    available_at_us: NDArray[np.int64]
    longitude: NDArray[np.float64]
    latitude: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    inside_study_area: NDArray[np.bool_]
    x_km: NDArray[np.float64]
    y_km: NDArray[np.float64]

    def _positions(
        self,
        indices: Sequence[int] | NDArray[np.int64],
    ) -> NDArray[np.int64]:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1:
            raise Stage2SFormalError("role indices must be one-dimensional")
        if requested.size == 0:
            return np.empty(0, dtype=np.int64)
        positions = np.searchsorted(self.indices, requested)
        if np.any(positions >= self.indices.size) or not np.array_equal(
            self.indices[positions], requested
        ):
            raise Stage2SFormalError(
                f"{self.role_id} attempted to access rows outside its legal cutoff"
            )
        return np.asarray(positions, dtype=np.int64)

    def xy(
        self,
        indices: Sequence[int] | NDArray[np.int64],
    ) -> NDArray[np.float64]:
        positions = self._positions(indices)
        if positions.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack((self.x_km[positions], self.y_km[positions]))

    def raw_rows(self) -> tuple[dict[str, object], ...]:
        rows = tuple(
            {
                "event_id": self.catalog.event_ids[int(index)],
                "origin_time_utc": _datetime_from_us(int(origin_us)),
                "available_at": _datetime_from_us(int(available_us)),
                "longitude": float(longitude),
                "latitude": float(latitude),
                "magnitude": float(magnitude),
                "inside_study_area": bool(inside),
            }
            for (
                index,
                origin_us,
                available_us,
                longitude,
                latitude,
                magnitude,
                inside,
            ) in zip(
                self.indices,
                self.origin_time_us,
                self.available_at_us,
                self.longitude,
                self.latitude,
                self.magnitude,
                self.inside_study_area,
                strict=True,
            )
        )
        if any(set(row) != set(RAW_CATALOG_FIELDS) for row in rows):
            raise Stage2SFormalError("formal role rows differ from the raw-field allowlist")
        return rows


class _CatalogRoleAccess:
    """Fail-closed coordinate gateway for pre-master roles and assessment."""

    __slots__ = (
        "_assessment_opened",
        "_catalog",
        "_geometry",
        "_transformer",
    )

    def __init__(
        self,
        catalog: Stage2SEarthquakeCatalog,
        preflight: NonTargetPreflight,
    ) -> None:
        self._catalog = catalog
        self._geometry = preflight.adapter.grid_family.study_area_equal_area
        self._transformer = Transformer.from_crs(
            CRS.from_epsg(4326),
            CRS.from_user_input(preflight.adapter.query_grid.equal_area_crs),
            always_xy=True,
        )
        self._assessment_opened = False

    @property
    def catalog(self) -> Stage2SEarthquakeCatalog:
        return self._catalog

    def _open_indices(
        self,
        indices: NDArray[np.int64],
        *,
        role_id: str,
        cutoff_utc: datetime | None,
    ) -> _CatalogRoleView:
        selected = np.asarray(indices, dtype=np.int64)
        if (
            selected.ndim != 1
            or np.any(selected < 0)
            or np.any(selected >= self._catalog.row_count)
            or np.any(np.diff(selected) <= 0)
        ):
            raise Stage2SFormalError("role view indices must be unique ascending catalogue rows")
        origin = np.asarray(self._catalog.origin_time_us[selected], dtype=np.int64)
        available = np.asarray(self._catalog.available_at_us[selected], dtype=np.int64)
        if cutoff_utc is not None:
            cutoff_us = int((cutoff_utc.astimezone(UTC) - _EPOCH).total_seconds() * 1_000_000)
            if np.any(origin > cutoff_us) or np.any(available > cutoff_us):
                raise Stage2SFormalError(
                    f"{role_id} attempted to expose a row after its legal cutoff"
                )
        longitude = np.asarray(self._catalog.longitude[selected], dtype=np.float64)
        latitude = np.asarray(self._catalog.latitude[selected], dtype=np.float64)
        magnitude = np.asarray(self._catalog.magnitude[selected], dtype=np.float64)
        inside = np.asarray(self._catalog.inside_study_area[selected], dtype=np.bool_)
        x_m_raw, y_m_raw = self._transformer.transform(longitude, latitude)
        x_m = np.asarray(x_m_raw, dtype=np.float64)
        y_m = np.asarray(y_m_raw, dtype=np.float64)
        if (
            x_m.shape != selected.shape
            or y_m.shape != selected.shape
            or not np.isfinite(x_m).all()
            or not np.isfinite(y_m).all()
        ):
            raise Stage2SFormalError("catalogue role projection produced invalid coordinates")
        observed_inside = np.asarray(covers(self._geometry, points(x_m, y_m)), dtype=np.bool_)
        if not np.array_equal(observed_inside, inside):
            raise Stage2SFormalError(f"{role_id} study-area flags differ from the frozen geometry")
        x_km = np.asarray(x_m / 1_000.0, dtype=np.float64)
        y_km = np.asarray(y_m / 1_000.0, dtype=np.float64)
        for value in (
            selected,
            origin,
            available,
            longitude,
            latitude,
            magnitude,
            inside,
            x_km,
            y_km,
        ):
            value.setflags(write=False)
        return _CatalogRoleView(
            catalog=self._catalog,
            role_id=role_id,
            cutoff_utc=cutoff_utc,
            indices=selected,
            origin_time_us=origin,
            available_at_us=available,
            longitude=longitude,
            latitude=latitude,
            magnitude=magnitude,
            inside_study_area=inside,
            x_km=x_km,
            y_km=y_km,
        )

    def open_before(
        self,
        *,
        role_id: str,
        cutoff_utc: datetime,
    ) -> _CatalogRoleView:
        if self._assessment_opened:
            raise Stage2SFormalError("pre-master role views cannot open after assessment")
        if not isinstance(cutoff_utc, datetime) or cutoff_utc.tzinfo is None:
            raise Stage2SFormalError("role cutoff must be timezone-aware")
        cutoff = cutoff_utc.astimezone(UTC)
        cutoff_us = int((cutoff - _EPOCH).total_seconds() * 1_000_000)
        stop = int(np.searchsorted(self._catalog.origin_time_us, cutoff_us, side="right"))
        prefix = np.arange(stop, dtype=np.int64)
        available = np.asarray(self._catalog.available_at_us[:stop], dtype=np.int64)
        indices = np.asarray(prefix[available <= cutoff_us], dtype=np.int64)
        return self._open_indices(indices, role_id=role_id, cutoff_utc=cutoff)

    def open_assessment(
        self,
        *,
        session: RoleSession,
        master_seal: SealedRecord,
        target_indices: NDArray[np.int64],
    ) -> _CatalogRoleView:
        if self._assessment_opened:
            raise Stage2SFormalError("assessment coordinates can open only once")
        if session.master_record is not master_seal:
            raise Stage2SFormalError(
                "assessment coordinates require this role session's master seal"
            )
        self._assessment_opened = True
        indices = np.asarray(
            tuple(sorted(set(int(value) for value in target_indices))), dtype=np.int64
        )
        return self._open_indices(
            indices,
            role_id="assessment_after_master",
            cutoff_utc=None,
        )


@dataclass(frozen=True, slots=True)
class _ScoredDelay:
    scores: Mapping[tuple[int, int], CellScore]
    mass_by_cell_model: Mapping[tuple[int, int, str], NDArray[np.float64]]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Stage2SFormalError(f"{label} must be a mapping")
    return cast(dict[str, Any], value)


def _relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Stage2SFormalError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Stage2SFormalError(f"{label} must stay inside the repository")
    return path


def _stream_bytes_once(path: Path) -> bytes:
    """Open one path once and stream it into one immutable bytes object."""

    chunks: list[bytes] = []
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_ONE_MEGABYTE)
                if not chunk:
                    break
                chunks.append(chunk)
    except OSError as exc:
        raise Stage2SFormalError("the claimed catalogue path could not be opened once") from exc
    payload = b"".join(chunks)
    if not payload:
        raise Stage2SFormalError("the claimed catalogue byte stream is empty")
    return payload


def _read_public_bytes(path: Path, *, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise Stage2SFormalError(f"cannot read {label}") from exc
    if not payload:
        raise Stage2SFormalError(f"{label} is empty")
    return payload


def _git(
    repository_root: Path,
    *arguments: str,
    check_stdout: bool = True,
) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise Stage2SFormalError(f"Git verification failed: {' '.join(arguments)}") from exc
    output = completed.stdout.strip()
    if check_stdout and not output:
        raise Stage2SFormalError(f"Git verification returned no output: {' '.join(arguments)}")
    return output


def _canonical_remote(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = "github.com/" + normalized.removeprefix("git@github.com:")
    for prefix in ("https://", "http://", "ssh://git@"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
    return normalized.rstrip("/")


def _remote_tag_commit(repository_root: Path, tag: str) -> str:
    output = _git(
        repository_root,
        "ls-remote",
        "--tags",
        "origin",
        f"refs/tags/{tag}",
        f"refs/tags/{tag}^{{}}",
    )
    refs: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise Stage2SFormalError("remote tag response is malformed")
        refs[fields[1]] = fields[0]
    peeled = refs.get(f"refs/tags/{tag}^{{}}")
    direct = refs.get(f"refs/tags/{tag}")
    commit = peeled or direct
    if commit is None or len(commit) != 40:
        raise Stage2SFormalError(f"remote tag is missing: {tag}")
    return commit


def verify_formal_release(protocol: Stage2SProtocolBundle) -> str:
    """Verify local/remote tags and reject modified tracked execution bytes."""

    root = protocol.repository_root
    verify_local_protocol_tag(root)
    code_commit = verify_local_code_tag(root)
    remote = _canonical_remote(_git(root, "remote", "get-url", "origin"))
    if remote.casefold() != _EXPECTED_REMOTE_REPOSITORY.casefold():
        raise Stage2SFormalError("origin is not the frozen public SeismoFlux repository")
    if _remote_tag_commit(root, PROTOCOL_TAG) != PROTOCOL_COMMIT:
        raise Stage2SFormalError("remote protocol tag does not peel to the frozen commit")
    if _remote_tag_commit(root, CODE_TAG) != code_commit:
        raise Stage2SFormalError("remote code tag does not peel to the local code commit")
    _git(root, "diff", "--quiet", check_stdout=False)
    _git(root, "diff", "--cached", "--quiet", check_stdout=False)
    return code_commit


def _synthetic_acceptance(protocol: Stage2SProtocolBundle) -> str:
    with tempfile.TemporaryDirectory(prefix="seismoflux-stage2s-acceptance-") as directory:
        result = run_synthetic_acceptance(
            protocol=protocol,
            scratch_root=Path(directory).resolve(),
        )
    return result.acceptance_sha256


def _audit_imports(repository_root: Path) -> str:
    report = audit_stage2s_import_closure(repository_root)
    return _sha256(
        canonical_json_bytes(
            {
                "root_modules": list(report.root_modules),
                "visited_modules": list(report.visited_modules),
                "visited_paths": list(report.visited_paths),
            }
        )
    )


def _audit_imports_release(
    repository_root: Path,
    code_commit: str,
) -> Mapping[str, object]:
    """Bind the audited files to clean Git bytes at the verified code commit."""

    report = audit_stage2s_import_closure(repository_root)
    release = verify_stage2s_import_closure_release(
        repository_root,
        report=report,
        code_commit=code_commit,
    )
    return MappingProxyType(
        {
            "root_modules": list(report.root_modules),
            "visited_modules": list(report.visited_modules),
            **release.receipt_bindings(),
        }
    )


def _parse_support_manifest(payload: bytes) -> BackgroundLocalSupportManifest:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2SFormalError("background support manifest is not UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise Stage2SFormalError("background support manifest root must be a mapping")
    try:
        return BackgroundLocalSupportManifest.model_validate(raw)
    except ValueError as exc:
        raise Stage2SFormalError("background support manifest failed strict validation") from exc


def _fold4_entry(
    manifest: BackgroundLocalSupportManifest,
) -> object:
    entries = tuple(item for item in manifest.snapshots if item.snapshot_id == "fold_4")
    if len(entries) != 1:
        raise Stage2SFormalError("support manifest must contain exactly one fold_4")
    return entries[0]


def _verify_target_free_fold4(
    protocol: Stage2SProtocolBundle,
    *,
    spatial: NonTargetPreflight,
    manifest: BackgroundLocalSupportManifest,
) -> None:
    sources = _config_mapping(protocol.config["source_contracts"], label="source_contracts")
    support_contract = _config_mapping(
        sources["background_local_support_manifest"],
        label="background_local_support_manifest",
    )
    if manifest.manifest_id != support_contract["manifest_id"]:
        raise Stage2SFormalError("background support manifest_id changed")
    validate_background_local_support_study_area(
        manifest,
        spatial.adapter.grid_family.study_area_equal_area,
    )
    background = _config_mapping(
        protocol.config["long_term_background"], label="long_term_background"
    )
    entry = cast(Any, _fold4_entry(manifest))
    support = entry.support
    exact_pairs = (
        (support.support_id, background["support_id"], "support_id"),
        (support.fit_end_utc, background["support_fit_end_utc_exact_manifest"], "fit_end_utc"),
        (support.common_mc, background["support_common_mc"], "common_mc"),
        (
            support.retained_area_fraction,
            background["support_retained_area_fraction"],
            "retained_area_fraction",
        ),
        (support.retained_area_m2, background["support_retained_area_m2"], "retained_area_m2"),
        (support.total_area_m2, background["support_total_area_m2"], "total_area_m2"),
        (
            support.historical_event_count,
            background["support_historical_event_count"],
            "historical_event_count",
        ),
        (
            support.historical_event_sha256,
            background["support_historical_event_sha256"],
            "historical_event_sha256",
        ),
        (
            support.retained_selected_event_count,
            background["support_retained_selected_event_count"],
            "retained_selected_event_count",
        ),
        (
            support.base_cell_size_km,
            background["support_base_cell_size_km"],
            "base_cell_size_km",
        ),
        (
            support.parent_cell_size_km,
            background["support_parent_cell_size_km"],
            "parent_cell_size_km",
        ),
    )
    mismatches = [label for observed, expected, label in exact_pairs if observed != expected]
    if mismatches:
        raise Stage2SFormalError("target-free fold4 identities changed: " + ", ".join(mismatches))
    if support.retained_area_fraction != 1.0:
        raise Stage2SFormalError("Stage2S requires the frozen fold4 retained fraction of one")


def _verify_source_hashes(protocol: Stage2SProtocolBundle) -> dict[str, str]:
    sources = _config_mapping(protocol.config["source_contracts"], label="source_contracts")
    study = _config_mapping(sources["study_area"], label="study_area")
    contracts = (
        (
            "projected_geometry_identity_source",
            study["projected_geometry_identity_algorithm"]["source_path"],
            study["projected_geometry_identity_algorithm"]["source_sha256_at_protocol"],
        ),
        (
            "study_area_loader_source",
            study["loader_source_path"],
            study["loader_source_sha256_at_protocol"],
        ),
        (
            "query_grid_builder_source",
            study["builder_source_path"],
            study["builder_source_sha256_at_protocol"],
        ),
        (
            "grid_primitives_source",
            study["grid_primitives_source_path"],
            study["grid_primitives_source_sha256_at_protocol"],
        ),
    )
    observed: dict[str, str] = {}
    for label, raw_path, raw_digest in contracts:
        path = protocol.repository_root / _relative_path(raw_path, label=f"{label}.path")
        digest = _sha256(_read_public_bytes(path, label=label))
        if digest != raw_digest:
            raise Stage2SFormalError(f"{label} differs from the frozen protocol source")
        observed[label] = digest
    return observed


def _validate_formal_preflight_receipt(
    protocol: Stage2SProtocolBundle,
    bindings: Mapping[str, object],
) -> None:
    control = _config_mapping(protocol.config["execution_control"], label="execution_control")
    receipt_contract = _config_mapping(
        control["non_target_preflight_receipt"],
        label="non_target_preflight_receipt",
    )
    if tuple(receipt_contract.get("required_bindings", ())) != (
        _NON_TARGET_PREFLIGHT_REQUIRED_BINDINGS
    ):
        raise Stage2SFormalError("frozen non-target preflight binding requirements changed")
    identity = protocol.identity_mapping()
    if any(bindings.get(key) != value for key, value in identity.items()):
        raise Stage2SFormalError("preflight protocol identity bindings are incomplete")
    if bindings.get("code_tag") != CODE_TAG or not isinstance(
        bindings.get("code_commit"),
        str,
    ):
        raise Stage2SFormalError("preflight code identity bindings are incomplete")
    source_hashes = bindings.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or tuple(source_hashes) != (
        "projected_geometry_identity_source",
        "study_area_loader_source",
        "query_grid_builder_source",
        "grid_primitives_source",
    ):
        raise Stage2SFormalError("preflight source hash bindings are incomplete")
    aligned = bindings.get("aligned_grid_identity")
    if not isinstance(aligned, Mapping):
        raise Stage2SFormalError("preflight aligned grid identity is missing")
    layers = aligned.get("layers")
    relations = aligned.get("parent_relations")
    if (
        aligned.get("layer_order") != ["50", "25", "12.5"]
        or aligned.get("parent_relation_order") != ["25_to_50", "12.5_to_25"]
        or aligned.get("target_or_score_input_count") != 0
        or not isinstance(layers, Mapping)
        or tuple(layers) != ("50", "25", "12.5")
        or not isinstance(relations, Mapping)
        or tuple(relations) != ("25_to_50", "12.5_to_25")
        or not isinstance(aligned.get("identity_sha256"), str)
    ):
        raise Stage2SFormalError("preflight aligned grid identity fields are incomplete")
    if any(not isinstance(layers[layer], Mapping) for layer in ("50", "25", "12.5")):
        raise Stage2SFormalError("preflight aligned grid layer fields are incomplete")
    layer_25 = cast(Mapping[str, object], layers["25"])
    sources = _config_mapping(protocol.config["source_contracts"], label="source_contracts")
    study_contract = _config_mapping(sources["study_area"], label="study_area")
    mapping_contract = _config_mapping(
        sources["cell_zone_mapping"],
        label="cell_zone_mapping",
    )
    zone_ids = bindings.get("construction_zone_ids")
    if (
        bindings.get("projected_geometry_identity_algorithm")
        != study_contract["projected_geometry_identity_algorithm"]
        or bindings.get("operational_quadrature_representative_point")
        != study_contract["operational_quadrature_representative_point"]
        or bindings.get("query_grid_id") != layer_25.get("grid_id")
        or bindings.get("query_grid_cell_count") != mapping_contract["row_count"]
        or not isinstance(zone_ids, list)
        or len(zone_ids) != mapping_contract["required_nonempty_zone_count"]
        or bindings.get("query_grid_and_cell_zone_mapping_returned_together") is not True
        or bindings.get("aligned_grid_parent_relations_verified") is not True
        or bindings.get("representative_point_identities_verified") is not True
    ):
        raise Stage2SFormalError("preflight grid/mapping relationship bindings are incomplete")
    security = _config_mapping(
        receipt_contract["security_assertions"],
        label="non-target preflight security assertions",
    )
    if any(bindings.get(key) is not expected for key, expected in security.items()):
        raise Stage2SFormalError("preflight security assertions differ from the frozen protocol")


def build_formal_preflight(
    protocol: Stage2SProtocolBundle,
    code_commit: str,
    synthetic_acceptance_sha256: str,
    import_audit_sha256: str,
) -> FormalPreflightContext:
    """Run the real non-target preflight without opening the earthquake catalogue."""

    spatial = run_non_target_spatial_preflight(protocol)
    sources = _config_mapping(protocol.config["source_contracts"], label="source_contracts")
    support_contract = _config_mapping(
        sources["background_local_support_manifest"],
        label="background_local_support_manifest",
    )
    support_path = protocol.repository_root / _relative_path(
        support_contract["path"],
        label="background support manifest path",
    )
    support_bytes = _read_public_bytes(support_path, label="background support manifest")
    if _sha256(support_bytes) != support_contract["sha256"]:
        raise Stage2SFormalError("background support manifest file hash changed")
    support_manifest = _parse_support_manifest(support_bytes)
    _verify_target_free_fold4(protocol, spatial=spatial, manifest=support_manifest)
    calendar_contract = _config_mapping(
        sources["rolling_fold_manifest"], label="rolling_fold_manifest"
    )
    calendar_path = protocol.repository_root / _relative_path(
        calendar_contract["path"],
        label="rolling fold manifest path",
    )
    calendar = parse_frozen_fold_manifest_bytes(
        _read_public_bytes(calendar_path, label="Stage2S fold manifest")
    )
    source_hashes = _verify_source_hashes(protocol)
    spatial_bindings = validated_non_target_preflight_receipt_bindings(spatial)
    study_contract = _config_mapping(sources["study_area"], label="study_area")
    receipt_bindings = {
        **protocol.identity_mapping(),
        "code_commit": code_commit,
        "code_tag": CODE_TAG,
        "synthetic_acceptance_sha256": synthetic_acceptance_sha256,
        "stage2s_import_closure_sha256": import_audit_sha256,
        "source_sha256": source_hashes,
        "background_local_support_manifest_sha256": support_contract["sha256"],
        "background_local_support_manifest_id": support_manifest.manifest_id,
        "fold4_support_id": cast(Any, _fold4_entry(support_manifest)).support.support_id,
        "projected_geometry_identity_algorithm": study_contract[
            "projected_geometry_identity_algorithm"
        ],
        "operational_quadrature_representative_point": study_contract[
            "operational_quadrature_representative_point"
        ],
        "non_target_preflight_required_bindings": list(_NON_TARGET_PREFLIGHT_REQUIRED_BINDINGS),
        "query_grid_and_cell_zone_mapping_returned_together": True,
        "aligned_grid_parent_relations_verified": True,
        "representative_point_identities_verified": True,
        **spatial_bindings,
    }
    _validate_formal_preflight_receipt(protocol, receipt_bindings)
    return FormalPreflightContext(
        spatial=spatial,
        support_manifest=support_manifest,
        calendar=calendar,
        receipt_bindings=MappingProxyType(receipt_bindings),
    )


def _load_sealed_record(path: Path, *, expected_type: str) -> SealedRecord:
    payload_bytes = _read_public_bytes(path, label=expected_type)
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2SFormalError(f"existing {expected_type} is not canonical JSON") from exc
    if not isinstance(payload, dict) or payload.get("record_type") != expected_type:
        raise Stage2SFormalError(f"existing {expected_type} has the wrong identity")
    content_sha256 = payload.get("content_sha256")
    if not isinstance(content_sha256, str):
        raise Stage2SFormalError(f"existing {expected_type} lacks content_sha256")
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    if _sha256(canonical_json_bytes(unsigned)) != content_sha256:
        raise Stage2SFormalError(f"existing {expected_type} content hash is invalid")
    canonical = canonical_json_bytes(payload) + b"\n"
    if canonical != payload_bytes:
        raise Stage2SFormalError(f"existing {expected_type} is not canonical UTF-8 JSON+LF")
    return SealedRecord(
        path=path,
        record_type=expected_type,
        content_sha256=content_sha256,
        file_sha256=_sha256(payload_bytes),
        payload=MappingProxyType(payload),
    )


def _write_bytes_o_excl(path: Path, payload: bytes) -> str:
    if not path.is_absolute() or not payload:
        raise ValueError("immutable result path/payload is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise Stage2SFormalError(f"immutable Stage2S result already exists: {path}") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise Stage2SFormalError("short write while creating immutable result")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise Stage2SFormalError(
            "immutable result write failed and is intentionally not rolled back"
        ) from exc
    finally:
        os.close(descriptor)
    return _sha256(payload)


@dataclass(frozen=True, slots=True)
class _FormalPathSpec:
    label: str
    relative_path: Path
    expected_record_type: str | None

    def __post_init__(self) -> None:
        normalized = _relative_path(
            self.relative_path.as_posix(),
            label=f"{self.label} transaction path",
        )
        object.__setattr__(self, "relative_path", normalized)


@dataclass(frozen=True, slots=True)
class _FormalAbsenceEvidence:
    relative_paths: tuple[str, ...]
    path_count: int
    absence_sha256: str

    def receipt_bindings(self) -> dict[str, object]:
        return {
            "relative_paths": list(self.relative_paths),
            "path_count": self.path_count,
            "all_absent": True,
            "absence_sha256": self.absence_sha256,
        }


def _fold_issue_dates(protocol: Stage2SProtocolBundle) -> Mapping[int, tuple[str, ...]]:
    folds = protocol.fold_manifest.get("folds")
    if not isinstance(folds, list):
        raise Stage2SFormalError("fold manifest must expose the three frozen folds")
    observed: dict[int, tuple[str, ...]] = {}
    for raw_fold in folds:
        fold = _config_mapping(raw_fold, label="fold manifest entry")
        fold_index = fold.get("fold_index")
        if not isinstance(fold_index, int) or isinstance(fold_index, bool):
            raise Stage2SFormalError("fold manifest index must be an integer")
        by_horizon = _config_mapping(
            fold.get("assessment_issue_dates_local_by_horizon"),
            label=f"fold {fold_index} assessment issue dates",
        )
        if set(by_horizon) != {"7", "30", "90"}:
            raise Stage2SFormalError(
                f"fold {fold_index} must expose the frozen 7/30/90-day schedules"
            )
        dates: set[str] = set()
        for horizon in ("7", "30", "90"):
            values = by_horizon[horizon]
            if not isinstance(values, list) or not values:
                raise Stage2SFormalError(
                    f"fold {fold_index} horizon {horizon} has no issue schedule"
                )
            for value in values:
                if (
                    not isinstance(value, str)
                    or re.fullmatch(
                        r"\d{4}-\d{2}-\d{2}",
                        value,
                    )
                    is None
                ):
                    raise Stage2SFormalError("fold issue date is not canonical YYYY-MM-DD")
                dates.add(value)
        observed[fold_index] = tuple(sorted(dates))
    if tuple(sorted(observed)) != _FOLDS:
        raise Stage2SFormalError("fold manifest must contain folds 1, 2, and 3")
    return MappingProxyType(observed)


def _formal_path_specs(
    protocol: Stage2SProtocolBundle,
) -> tuple[_FormalPathSpec, ...]:
    control = _config_mapping(protocol.config["execution_control"], label="execution_control")
    preflight = _config_mapping(
        control["non_target_preflight_receipt"],
        label="non_target_preflight_receipt",
    )
    attempt = _config_mapping(control["attempt_ledger"], label="attempt_ledger")
    target = _config_mapping(control["target_read_ledger"], label="target_read_ledger")
    specs = [
        _FormalPathSpec(
            label="non_target_preflight",
            relative_path=_relative_path(
                preflight["path"],
                label="non-target preflight receipt path",
            ),
            expected_record_type="stage2s_non_target_preflight_receipt",
        ),
        _FormalPathSpec(
            label="attempt",
            relative_path=_relative_path(
                attempt["path"],
                label="attempt ledger path",
            ),
            expected_record_type="stage2s_formal_attempt_ledger",
        ),
        _FormalPathSpec(
            label="target_read",
            relative_path=_relative_path(
                target["path"],
                label="target read receipt path",
            ),
            expected_record_type="stage2s_target_read_receipt",
        ),
        _FormalPathSpec(
            label="result",
            relative_path=_RESULT_RELATIVE_PATH,
            expected_record_type=None,
        ),
        _FormalPathSpec(
            label="terminal",
            relative_path=_TERMINAL_RELATIVE_PATH,
            expected_record_type="stage2s_terminal_failure_record",
        ),
    ]
    seal_root = Path("data/interim/stage2s/causal_seismicity_screen")
    for fold_index, issue_dates in _fold_issue_dates(protocol).items():
        fold_root = seal_root / f"fold_{fold_index}"
        specs.append(
            _FormalPathSpec(
                label=f"fold_{fold_index}_fit",
                relative_path=fold_root / "fit_receipt.json",
                expected_record_type="stage2s_fold_fit_receipt",
            )
        )
        specs.extend(
            _FormalPathSpec(
                label=f"fold_{fold_index}_issue_{issue_date}",
                relative_path=fold_root / f"issue_{issue_date}" / "prediction_seal.json",
                expected_record_type="stage2s_issue_prediction_seal",
            )
            for issue_date in issue_dates
        )
        specs.append(
            _FormalPathSpec(
                label=f"fold_{fold_index}_prediction",
                relative_path=fold_root / "fold_prediction_seal.json",
                expected_record_type="stage2s_fold_prediction_seal",
            )
        )
    specs.append(
        _FormalPathSpec(
            label="master_prediction",
            relative_path=seal_root / "prediction_seal.json",
            expected_record_type="stage2s_master_prediction_seal",
        )
    )
    specs.extend(
        _FormalPathSpec(
            label=f"artifact_{name}",
            relative_path=_ARTIFACT_ROOT_RELATIVE_PATH / name,
            expected_record_type=None,
        )
        for name in ALL_ARTIFACT_NAMES
    )
    ordered = tuple(
        sorted(
            specs,
            key=lambda item: item.relative_path.as_posix().encode("utf-8"),
        )
    )
    paths = tuple(spec.relative_path.as_posix() for spec in ordered)
    labels = tuple(spec.label for spec in ordered)
    if len(set(paths)) != len(paths) or len(set(labels)) != len(labels):
        raise Stage2SFormalError("formal transaction paths or labels are duplicated")
    return ordered


def _path_is_present(repository_root: Path, spec: _FormalPathSpec) -> bool:
    return os.path.lexists(repository_root / spec.relative_path)


def _present_path_specs(
    repository_root: Path,
    specs: Sequence[_FormalPathSpec],
) -> tuple[_FormalPathSpec, ...]:
    return tuple(spec for spec in specs if _path_is_present(repository_root, spec))


def _absence_evidence(specs: Sequence[_FormalPathSpec]) -> _FormalAbsenceEvidence:
    relative_paths = tuple(spec.relative_path.as_posix() for spec in specs)
    preimage = {
        "relative_paths": list(relative_paths),
        "path_count": len(relative_paths),
        "all_absent": True,
    }
    return _FormalAbsenceEvidence(
        relative_paths=relative_paths,
        path_count=len(relative_paths),
        absence_sha256=_sha256(canonical_json_bytes(preimage)),
    )


def _path_spec(
    specs: Sequence[_FormalPathSpec],
    label: str,
) -> _FormalPathSpec:
    matches = tuple(spec for spec in specs if spec.label == label)
    if len(matches) != 1:
        raise Stage2SFormalError(f"formal transaction path is missing: {label}")
    return matches[0]


def _execution_paths(
    protocol: Stage2SProtocolBundle,
) -> tuple[Path, Path, Path, Path]:
    control = _config_mapping(protocol.config["execution_control"], label="execution_control")
    preflight = _config_mapping(
        control["non_target_preflight_receipt"],
        label="non_target_preflight_receipt",
    )
    attempt = _config_mapping(control["attempt_ledger"], label="attempt_ledger")
    target = _config_mapping(control["target_read_ledger"], label="target_read_ledger")
    return (
        protocol.repository_root
        / _relative_path(preflight["path"], label="non-target preflight receipt path"),
        protocol.repository_root / _relative_path(attempt["path"], label="attempt ledger path"),
        protocol.repository_root / _relative_path(target["path"], label="target read receipt path"),
        protocol.repository_root / _RESULT_RELATIVE_PATH,
    )


def _catalog_contract(protocol: Stage2SProtocolBundle) -> dict[str, Any]:
    sources = _config_mapping(protocol.config["source_contracts"], label="source_contracts")
    return _config_mapping(sources["earthquake_catalog"], label="earthquake_catalog")


def _attempt_bindings(
    protocol: Stage2SProtocolBundle,
    *,
    code_commit: str,
    preflight_receipt: SealedRecord,
) -> dict[str, object]:
    catalog = _catalog_contract(protocol)
    return {
        **protocol.identity_mapping(),
        "code_commit": code_commit,
        "code_tag": CODE_TAG,
        "expected_catalog_file_sha256": catalog["file_sha256"],
        "expected_catalog_content_sha256": catalog["content_sha256"],
        "expected_catalog_schema_sha256": catalog["schema_sha256"],
        "expected_catalog_row_count": catalog["row_count"],
        "non_target_preflight_receipt_sha256": preflight_receipt.file_sha256,
    }


def _target_read_bindings(
    protocol: Stage2SProtocolBundle,
    *,
    code_commit: str,
    attempt: SealedRecord,
) -> dict[str, object]:
    catalog = _catalog_contract(protocol)
    return {
        **protocol.identity_mapping(),
        "code_commit": code_commit,
        "code_tag": CODE_TAG,
        "attempt_ledger_sha256": attempt.file_sha256,
        "prior_state": {
            "file_absent": True,
            "logical_open_count": 0,
            "physical_open_count": 0,
        },
        "claimed_state": {
            "logical_open_count": 1,
            "physical_open_count_authorized": 1,
        },
        "expected_catalog_file_sha256": catalog["file_sha256"],
        "expected_catalog_content_sha256": catalog["content_sha256"],
        "expected_catalog_schema_sha256": catalog["schema_sha256"],
    }


def _release_import_bindings(
    services: FormalExecutionServices,
    *,
    repository_root: Path,
    code_commit: str,
) -> tuple[str, Mapping[str, object]]:
    if services.audit_imports_release is None:
        legacy_sha256 = services.audit_imports(repository_root)
        if re.fullmatch(r"[0-9a-f]{64}", legacy_sha256) is None:
            raise Stage2SFormalError("injected import audit did not return a SHA-256")
        return (
            legacy_sha256,
            MappingProxyType(
                {
                    "legacy_injected_audit_sha256": legacy_sha256,
                    "release_evidence_injected": True,
                }
            ),
        )
    bindings = services.audit_imports_release(repository_root, code_commit)
    if not isinstance(bindings, Mapping):
        raise Stage2SFormalError("release import audit did not return receipt bindings")
    evidence_sha256 = bindings.get("evidence_sha256")
    visited_path_sha256 = bindings.get("visited_path_sha256")
    if (
        not isinstance(evidence_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None
        or not isinstance(visited_path_sha256, Mapping)
        or not visited_path_sha256
        or bindings.get("git_tracked") is not True
        or bindings.get("path_scoped_status_clean") is not True
        or bindings.get("working_tree_equals_head_and_code_commit") is not True
    ):
        raise Stage2SFormalError("release import audit evidence is incomplete")
    return evidence_sha256, MappingProxyType(dict(bindings))


def _existing_result_state(path: Path) -> tuple[str, str | None]:
    try:
        payload_bytes = _read_public_bytes(path, label="existing Stage2S result")
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (Stage2SFormalError, UnicodeError, json.JSONDecodeError):
        return "invalid", None
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("record_type") != "stage2s_whole_run_record"
        or canonical_payload + b"\n" != payload_bytes
    ):
        return "invalid", None
    observed_sha256 = payload.get("run_record_sha256")
    unsigned = dict(payload)
    unsigned.pop("run_record_sha256", None)
    if (
        not isinstance(observed_sha256, str)
        or _sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        != observed_sha256
    ):
        return "invalid", None
    gate = payload.get("gate_evidence")
    gate_status = gate.get("status") if isinstance(gate, dict) else None
    return "valid", gate_status if isinstance(gate_status, str) else None


def _existing_terminal_state(path: Path) -> tuple[str, str | None]:
    try:
        record = _load_sealed_record(
            path,
            expected_type="stage2s_terminal_failure_record",
        )
    except (OSError, Stage2SFormalError):
        return "invalid", None
    bindings = record.payload.get("bindings")
    status = bindings.get("terminal_status") if isinstance(bindings, dict) else None
    return "valid", status if isinstance(status, str) else None


def _seal_status_by_label(
    protocol: Stage2SProtocolBundle,
    specs: Sequence[_FormalPathSpec],
) -> dict[str, object]:
    observed: dict[str, object] = {}
    for spec in specs:
        if spec.expected_record_type is None or spec.label == "terminal":
            continue
        path = protocol.repository_root / spec.relative_path
        if not _path_is_present(protocol.repository_root, spec):
            observed[spec.label] = {
                "relative_path": spec.relative_path.as_posix(),
                "expected_record_type": spec.expected_record_type,
                "status": "missing",
                "file_sha256": None,
            }
            continue
        try:
            record = _load_sealed_record(
                path,
                expected_type=spec.expected_record_type,
            )
        except (OSError, Stage2SFormalError):
            observed[spec.label] = {
                "relative_path": spec.relative_path.as_posix(),
                "expected_record_type": spec.expected_record_type,
                "status": "invalid",
                "file_sha256": None,
            }
        else:
            observed[spec.label] = {
                "relative_path": spec.relative_path.as_posix(),
                "expected_record_type": spec.expected_record_type,
                "status": "valid",
                "file_sha256": record.file_sha256,
            }
    return observed


def _artifact_status_by_name(
    protocol: Stage2SProtocolBundle,
    specs: Sequence[_FormalPathSpec],
) -> dict[str, object]:
    return {
        spec.relative_path.name: {
            "relative_path": spec.relative_path.as_posix(),
            "status": (
                "present_unverified"
                if _path_is_present(protocol.repository_root, spec)
                else "missing"
            ),
        }
        for spec in specs
        if spec.label.startswith("artifact_")
    }


def _terminal_classification(
    error: BaseException,
) -> tuple[str, str, str]:
    if isinstance(error, Stage2SEvidenceInsufficient) or any(
        marker in type(error).__name__ for marker in ("EvidenceInsufficient", "ScientificInability")
    ):
        return (
            "evidence_insufficient",
            "evidence_insufficient",
            "formal evidence was insufficient under the frozen protocol",
        )
    if isinstance(error, KeyboardInterrupt):
        return (
            "interrupted",
            "keyboard_interrupt",
            "formal execution was interrupted before completion",
        )
    if isinstance(error, SystemExit):
        return (
            "interrupted",
            "system_exit",
            "formal execution requested process exit before completion",
        )
    return (
        "invalid",
        "exception",
        "formal execution failed closed; original exception text was suppressed",
    )


def _write_terminal_record(
    *,
    protocol: Stage2SProtocolBundle,
    code_commit: str,
    phase: str,
    error: BaseException,
    specs: Sequence[_FormalPathSpec],
) -> SealedRecord:
    status, category, message = _terminal_classification(error)
    terminal_spec = _path_spec(specs, "terminal")
    return write_o_excl_record(
        protocol.repository_root / terminal_spec.relative_path,
        record_type="stage2s_terminal_failure_record",
        bindings={
            **protocol.identity_mapping(),
            "code_commit": code_commit,
            "code_tag": CODE_TAG,
            "terminal_status": status,
            "terminal_phase": phase,
            "exception_category": category,
            "exception_message": message,
            "seal_status_by_label": _seal_status_by_label(protocol, specs),
            "artifact_status_by_name": _artifact_status_by_name(protocol, specs),
            "attempt_consumed": True,
            "retry_or_rollback_authorized": False,
            "catalog_reopen_authorized": False,
        },
    )


def _finalize_existing_partial_state(
    *,
    protocol: Stage2SProtocolBundle,
    code_commit: str,
    specs: Sequence[_FormalPathSpec],
) -> None:
    _write_terminal_record(
        protocol=protocol,
        code_commit=code_commit,
        phase="startup_crash_inspection",
        error=KeyboardInterrupt(),
        specs=specs,
    )


def execute_formal_once(
    protocol: Stage2SProtocolBundle,
    *,
    services: FormalExecutionServices,
    progress: ProgressCallback | None = None,
) -> Stage2SWholeRunRecord:
    """Execute the transaction boundary with injectable target-free adapters."""

    current_phase = "release_verification"

    def emit(phase: str) -> None:
        nonlocal current_phase
        current_phase = phase
        _notify(progress, phase)

    code_commit = services.verify_release(protocol)
    emit("remote_tags_verified")
    specs = _formal_path_specs(protocol)
    result_spec = _path_spec(specs, "result")
    terminal_spec = _path_spec(specs, "terminal")
    terminal_path = protocol.repository_root / terminal_spec.relative_path
    result_path = protocol.repository_root / result_spec.relative_path
    if _path_is_present(protocol.repository_root, terminal_spec):
        terminal_state, terminal_status = _existing_terminal_state(terminal_path)
        raise Stage2SFormalError(
            "an immutable Stage2S terminal record already exists "
            f"(record={terminal_state}, status={terminal_status or 'unavailable'}); "
            "no execution state was changed"
        )
    if _path_is_present(protocol.repository_root, result_spec):
        result_state, gate_status = _existing_result_state(result_path)
        if result_state == "valid":
            raise Stage2SFormalError(
                "an immutable completed Stage2S result already exists "
                f"(gate={gate_status or 'unavailable'}); no execution state was changed"
            )
        _finalize_existing_partial_state(
            protocol=protocol,
            code_commit=code_commit,
            specs=specs,
        )
        raise Stage2SFormalError(
            "an incomplete Stage2S result was finalized without reopening the catalog"
        )
    present = _present_path_specs(protocol.repository_root, specs)
    if present:
        _finalize_existing_partial_state(
            protocol=protocol,
            code_commit=code_commit,
            specs=specs,
        )
        raise Stage2SFormalError(
            "an existing partial Stage2S transaction was finalized without reopening the catalog"
        )
    absence = _absence_evidence(specs)
    synthetic_sha256 = services.run_code_acceptance(protocol)
    import_audit_sha256, import_release = _release_import_bindings(
        services,
        repository_root=protocol.repository_root,
        code_commit=code_commit,
    )
    preflight = services.build_preflight(
        protocol,
        code_commit,
        synthetic_sha256,
        import_audit_sha256,
    )
    emit("non_target_preflight_passed")
    preflight_path, attempt_path, target_path, result_path = _execution_paths(protocol)
    preflight_receipt: SealedRecord | None = None
    attempt: SealedRecord | None = None
    target_read: SealedRecord | None = None
    try:
        raced = _present_path_specs(protocol.repository_root, specs)
        if raced:
            raise Stage2SFormalError(
                "a formal transaction output appeared after initial absence verification"
            )
        preflight_bindings = {
            **dict(preflight.receipt_bindings),
            "formal_execution_environment": dict(services.execution_environment_bindings),
            "stage2s_import_closure_release": dict(import_release),
            "initial_formal_output_absence": absence.receipt_bindings(),
        }
        preflight_receipt = write_o_excl_record(
            preflight_path,
            record_type="stage2s_non_target_preflight_receipt",
            bindings=preflight_bindings,
        )
        unexpected_before_attempt = tuple(
            spec
            for spec in _present_path_specs(protocol.repository_root, specs)
            if spec.label != "non_target_preflight"
        )
        if unexpected_before_attempt:
            raise Stage2SFormalError(
                "a formal transaction output appeared before the attempt claim"
            )
        attempt = write_o_excl_record(
            attempt_path,
            record_type="stage2s_formal_attempt_ledger",
            bindings=_attempt_bindings(
                protocol,
                code_commit=code_commit,
                preflight_receipt=preflight_receipt,
            ),
        )
        emit("attempt_claimed")
        target_read = write_o_excl_record(
            target_path,
            record_type="stage2s_target_read_receipt",
            bindings=_target_read_bindings(
                protocol,
                code_commit=code_commit,
                attempt=attempt,
            ),
        )
        emit("target_read_claimed")
        catalog_contract = _catalog_contract(protocol)
        catalog_path = protocol.repository_root / _relative_path(
            catalog_contract["path"],
            label="earthquake catalogue path",
        )
        catalog_bytes = services.read_catalog_once(catalog_path)
        if _sha256(catalog_bytes) != catalog_contract["file_sha256"]:
            raise Stage2SFormalError(
                "catalogue bytes differ after the one-shot read; the attempt remains consumed"
            )
        catalog = services.parse_catalog(catalog_bytes)
        emit("catalog_parsed")
        inputs = FormalScienceInputs(
            protocol=protocol,
            code_commit=code_commit,
            preflight=preflight,
            preflight_receipt=preflight_receipt,
            attempt_ledger=attempt,
            target_read_receipt=target_read,
            progress=emit,
        )
        result = services.run_science(inputs, catalog)
        if result.mode != "formal_development":
            raise Stage2SFormalError("formal science returned a non-formal record")
        _write_bytes_o_excl(result_path, result.to_canonical_bytes())
        emit("result_written")
        return result
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        _write_terminal_record(
            protocol=protocol,
            code_commit=code_commit,
            phase=current_phase,
            error=exc,
            specs=specs,
        )
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        raise Stage2SFormalError(
            "the one-shot Stage2S attempt stopped and an immutable terminal record was written"
        ) from exc


def default_execution_services() -> FormalExecutionServices:
    """Return the closed production service set used by the public runner."""

    environment = require_prepared_formal_execution_environment()
    return FormalExecutionServices(
        verify_release=verify_formal_release,
        run_code_acceptance=_synthetic_acceptance,
        audit_imports=_audit_imports,
        build_preflight=build_formal_preflight,
        read_catalog_once=_stream_bytes_once,
        parse_catalog=parse_frozen_catalog_bytes,
        run_science=run_formal_science,
        audit_imports_release=_audit_imports_release,
        execution_environment_bindings=MappingProxyType(environment.receipt_bindings()),
    )


def _datetime_from_us(value: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(value))


def _notify(progress: ProgressCallback | None, phase: str) -> None:
    if progress is not None:
        try:
            progress(phase)
        except Exception:
            return


def _completeness_events(
    view: _CatalogRoleView,
) -> tuple[CompletenessEvent, ...]:
    return tuple(
        CompletenessEvent(
            event_id=view.catalog.event_ids[int(index)],
            origin_time_utc=_datetime_from_us(int(origin_us)),
            available_at=_datetime_from_us(int(available_us)),
            magnitude=float(magnitude),
            inside_study_area=bool(inside),
            x_m=float(x_km) * 1_000.0,
            y_m=float(y_km) * 1_000.0,
        )
        for index, origin_us, available_us, magnitude, inside, x_km, y_km in zip(
            view.indices,
            view.origin_time_us,
            view.available_at_us,
            view.magnitude,
            view.inside_study_area,
            view.x_km,
            view.y_km,
            strict=True,
        )
    )


def _expected_fold4_mapping(
    manifest: BackgroundLocalSupportManifest,
) -> dict[str, object]:
    entry = cast(Any, _fold4_entry(manifest))
    support = entry.support.model_dump(mode="python")
    decisions = cast(list[dict[str, object]], support.pop("cells"))
    if len(decisions) != len(manifest.fixed_cells):
        raise Stage2SFormalError("fold4 fixed and decision cell counts differ")
    support["cells"] = [
        {
            **fixed.model_dump(mode="python"),
            **decision,
        }
        for fixed, decision in zip(manifest.fixed_cells, decisions, strict=True)
    ]
    return cast(dict[str, object], scientific_mapping(support))


def _rebuild_fold4(
    inputs: FormalScienceInputs,
    support_view: _CatalogRoleView,
) -> LocalSupportSnapshot:
    background = _config_mapping(
        inputs.protocol.config["long_term_background"],
        label="long_term_background",
    )
    fit_end = datetime.fromisoformat(
        cast(str, background["fit_end_utc"]).replace("Z", "+00:00")
    ).astimezone(UTC)
    if support_view.cutoff_utc != fit_end:
        raise Stage2SFormalError("fold4 support view is not bound to fit_end_utc")
    snapshot = build_local_support_snapshot(
        _completeness_events(support_view),
        fit_end_utc=fit_end,
        study_area_equal_area=inputs.preflight.spatial.adapter.grid_family.study_area_equal_area,
    )
    rebuilt = scientific_mapping(build_local_support_manifest(snapshot))
    expected = _expected_fold4_mapping(inputs.preflight.support_manifest)
    if rebuilt != expected:
        raise Stage2SFormalError(
            "fold4 support did not reproduce every frozen manifest field; attempt consumed"
        )
    if snapshot.support_id != background["support_id"] or snapshot.retained_area_fraction != 1.0:
        raise Stage2SFormalError("rebuilt fold4 support identity or retained fraction changed")
    return snapshot


def _model(models: Stage2SModels, model_id: str) -> NormalizedSpatialDensity:
    if model_id == "S0":
        return models.s0
    if model_id == "S1":
        return models.s1
    if model_id == "SP":
        return models.sp
    raise KeyError(model_id)


def _array_sha256(values: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _id_receipt(
    catalog: Stage2SEarthquakeCatalog,
    indices: Sequence[int] | NDArray[np.int64],
) -> dict[str, object]:
    ordered = tuple(int(value) for value in indices)
    event_ids = tuple(catalog.event_ids[index] for index in ordered)
    origins = tuple(int(catalog.origin_time_us[index]) for index in ordered)
    available = tuple(int(catalog.available_at_us[index]) for index in ordered)
    return {
        "event_count": len(event_ids),
        "ordered_event_ids_sha256": _sha256(canonical_json_bytes(list(event_ids))),
        "first_event_id": None if not event_ids else event_ids[0],
        "last_event_id": None if not event_ids else event_ids[-1],
        "maximum_origin_time_utc": (
            None if not origins else _datetime_from_us(max(origins)).isoformat()
        ),
        "maximum_available_at": (
            None if not available else _datetime_from_us(max(available)).isoformat()
        ),
    }


def _model_receipt(model: NormalizedSpatialDensity) -> dict[str, object]:
    return {
        "model_id": model.model_id,
        "alpha": model.alpha,
        "source_event_count": model.source_event_count,
        "mass_12_5km_sha256": _array_sha256(model.mass_12_5km),
        "mass_25km_sha256": _array_sha256(model.mass_25km),
        "direct_mass_25km_sha256": _array_sha256(model.direct_mass_25km),
        "direct_mass_50km_sha256": _array_sha256(model.direct_mass_50km),
        "normalization_mass": model.normalization_mass,
        "convergence": {
            "primary_relative_count_difference": (
                model.convergence.primary_relative_count_difference
            ),
            "primary_density_l1": model.convergence.primary_density_l1,
            "diagnostic_relative_count_difference": (
                model.convergence.diagnostic_relative_count_difference
            ),
            "diagnostic_density_l1": model.convergence.diagnostic_density_l1,
            "passed": model.convergence.passed,
        },
    }


def _alarm_receipt(alarm: AlarmMask) -> dict[str, object]:
    return {
        "model_id": alarm.model_id,
        "selected_cell_count": len(alarm.selected_cell_ids),
        "selected_cell_ids_sha256": _sha256(canonical_json_bytes(list(alarm.selected_cell_ids))),
        "actual_area_km2": alarm.actual_area_km2,
        "budget_km2": alarm.budget_km2,
        "grid_id": alarm.grid_id,
        "ranking_sha256": alarm.ranking_sha256,
    }


def _alpha_receipt(value: AlphaFit) -> dict[str, object]:
    def derivative_record(
        derivative: SignedLogDerivative,
    ) -> dict[str, object]:
        return {
            "sign": derivative.sign,
            "log_abs_mean": derivative.log_abs_mean,
            "finite_float_value": derivative.finite_float_value,
            "comparison": derivative.comparison,
        }

    return {
        "alpha": value.alpha,
        "solver_case": value.solver_case,
        "iterations": value.iterations,
        "derivative_at_zero": derivative_record(value.derivative_at_zero),
        "derivative_at_one": derivative_record(value.derivative_at_one),
        "fit_event_count": value.fit_event_count,
    }


def _xy(
    coordinates: _CatalogRoleView,
    indices: Sequence[int] | NDArray[np.int64],
) -> NDArray[np.float64]:
    return coordinates.xy(indices)


def _support_mask(
    snapshot: LocalSupportSnapshot,
    coordinates: _CatalogRoleView,
    indices: Sequence[int] | NDArray[np.int64],
) -> NDArray[np.bool_]:
    locator = LocalSupportCellLocator(snapshot.cells)
    values = np.asarray(indices, dtype=np.int64)
    xy = _xy(coordinates, values)
    supported = np.asarray(
        [
            (
                (
                    cell := locator.resolve(
                        x_m=float(x_km) * 1_000.0,
                        y_m=float(y_km) * 1_000.0,
                    )
                )
                is not None
                and cell.status != "unsupported"
            )
            for x_km, y_km in xy
        ],
        dtype=np.bool_,
    )
    supported.setflags(write=False)
    return supported


def _build_s0(
    inputs: FormalScienceInputs,
    support_view: _CatalogRoleView,
    snapshot: LocalSupportSnapshot,
    family: SpatialQuadratureFamily,
) -> NormalizedSpatialDensity:
    background = _config_mapping(
        inputs.protocol.config["long_term_background"],
        label="long_term_background",
    )
    cutoff = datetime.fromisoformat(
        cast(str, background["fit_end_utc"]).replace("Z", "+00:00")
    ).astimezone(UTC)
    if support_view.cutoff_utc != cutoff:
        raise Stage2SFormalError("S0 support view is not bound to fit_end_utc")
    cutoff_us = int((cutoff - _EPOCH).total_seconds() * 1_000_000)
    selected_local = np.flatnonzero(
        support_view.inside_study_area
        & (support_view.origin_time_us <= cutoff_us)
        & (support_view.available_at_us <= cutoff_us)
        & (support_view.magnitude >= snapshot.common_mc)
    ).astype(np.int64)
    selected = np.asarray(support_view.indices[selected_local], dtype=np.int64)
    retained = _support_mask(snapshot, support_view, selected)
    selected = selected[retained]
    if selected.size != snapshot.retained_selected_event_count:
        raise Stage2SFormalError("S0 training event count differs from rebuilt fold4 support")
    return build_normalized_kde(
        _xy(support_view, selected),
        family,
        model_id="S0",
    )


def _causal_source_indices(
    view: _CatalogRoleView,
    *,
    issue_time_utc: datetime,
    component_id: str,
    delay_days: int,
) -> NDArray[np.int64]:
    if delay_days not in _DELAYS:
        raise ValueError("delay_days must be 0, 1, or 7")
    issue_us = int((issue_time_utc.astimezone(UTC) - _EPOCH).total_seconds() * 1_000_000)
    day_us = 86_400_000_000
    if component_id == "R":
        lower_exclusive = issue_us - 30 * day_us
        upper_inclusive = issue_us
        availability_inclusive = issue_us - delay_days * day_us
    elif component_id == "RP":
        lower_exclusive = issue_us - 60 * day_us
        upper_inclusive = issue_us - 30 * day_us
        availability_inclusive = upper_inclusive - delay_days * day_us
    else:
        raise ValueError("component_id must be R or RP")
    local = np.flatnonzero(
        view.inside_study_area
        & (view.magnitude >= M4_MINIMUM)
        & (view.origin_time_us > lower_exclusive)
        & (view.origin_time_us <= upper_inclusive)
        & (view.available_at_us <= availability_inclusive)
    ).astype(np.int64)
    indices = np.asarray(view.indices[local], dtype=np.int64)
    indices.setflags(write=False)
    return indices


def _fit_target_membership(
    view: _CatalogRoleView,
    fold: FoldCalendar,
) -> FitTargetMembership:
    eligible = (
        view.inside_study_area
        & (view.magnitude >= M5_6_MINIMUM)
        & (view.magnitude < M5_6_MAXIMUM_EXCLUSIVE)
    )
    assigned_issue = np.full(view.indices.size, np.iinfo(np.int64).min, dtype=np.int64)
    assigned = np.zeros(view.indices.size, dtype=np.bool_)
    for exposure in fold.fit_exposures:
        membership = (
            eligible
            & (view.origin_time_us > exposure.issue_time_us)
            & (view.origin_time_us <= exposure.target_end_inclusive_us)
        )
        if np.any(assigned & membership):
            raise Stage2SFormalError("fit event matched more than one h007 issue")
        assigned[membership] = True
        assigned_issue[membership] = exposure.issue_time_us
    local = np.asarray(np.flatnonzero(assigned), dtype=np.int64)
    indices = np.asarray(view.indices[local], dtype=np.int64)
    issue_times = np.asarray(assigned_issue[local], dtype=np.int64)
    indices.setflags(write=False)
    issue_times.setflags(write=False)
    return FitTargetMembership(
        catalog=view.catalog,
        fold_index=fold.fold_index,
        event_indices=indices,
        assigned_issue_time_us=issue_times,
        exposure_days=float(len(fold.fit_exposures) * 7),
    )


def _recent_component(
    *,
    view: _CatalogRoleView,
    snapshot: LocalSupportSnapshot,
    family: SpatialQuadratureFamily,
    s0: NormalizedSpatialDensity,
    issue_time_utc: datetime,
    component_id: str,
    delay_days: int,
) -> tuple[NormalizedSpatialDensity, NDArray[np.int64]]:
    indices = _causal_source_indices(
        view,
        issue_time_utc=issue_time_utc,
        component_id=component_id,
        delay_days=delay_days,
    )
    supported = _support_mask(snapshot, view, indices)
    indices = np.asarray(indices[supported], dtype=np.int64)
    indices.setflags(write=False)
    component = build_recent_component(
        _xy(view, indices),
        family,
        component_id=cast(Any, component_id),
        empty_fallback_s0=s0,
    )
    return component, indices


def _fit_fold(
    *,
    access: _CatalogRoleAccess,
    fit_view: _CatalogRoleView,
    snapshot: LocalSupportSnapshot,
    family: SpatialQuadratureFamily,
    s0: NormalizedSpatialDensity,
    fold: FoldCalendar,
) -> tuple[
    Mapping[int, tuple[AlphaFit, AlphaFit]],
    SharedRate,
    Mapping[str, object],
]:
    catalog = access.catalog
    membership = _fit_target_membership(fit_view, fold)
    supported = _support_mask(snapshot, fit_view, membership.event_indices)
    target_indices = np.asarray(membership.event_indices[supported], dtype=np.int64)
    assigned_issue_us = np.asarray(
        membership.assigned_issue_time_us[supported],
        dtype=np.int64,
    )
    _require_supported_fit_targets(int(target_indices.size))
    target_xy = _xy(fit_view, target_indices)
    q0 = s0.log_density(target_xy[:, 0], target_xy[:, 1])
    order = FitEventOrder(
        origin_time_ns=np.asarray(
            catalog.origin_time_us[target_indices] * np.int64(1_000),
            dtype=np.int64,
        ),
        event_ids=tuple(catalog.event_ids[int(index)] for index in target_indices),
    )
    fits: dict[int, tuple[AlphaFit, AlphaFit]] = {}
    source_receipts: dict[str, object] = {}
    for delay_days in _DELAYS:
        q_r = np.empty(target_indices.size, dtype=np.float64)
        q_p = np.empty(target_indices.size, dtype=np.float64)
        delay_receipts: dict[str, object] = {}
        for issue_us in sorted(set(int(value) for value in assigned_issue_us)):
            issue_time = _datetime_from_us(issue_us)
            source_view = access.open_before(
                role_id=(
                    f"fold_{fold.fold_index}_fit_source_{issue_time.isoformat()}_delay_{delay_days}"
                ),
                cutoff_utc=issue_time,
            )
            positions = np.flatnonzero(assigned_issue_us == issue_us)
            recent, recent_indices = _recent_component(
                view=source_view,
                snapshot=snapshot,
                family=family,
                s0=s0,
                issue_time_utc=issue_time,
                component_id="R",
                delay_days=delay_days,
            )
            preceding, preceding_indices = _recent_component(
                view=source_view,
                snapshot=snapshot,
                family=family,
                s0=s0,
                issue_time_utc=issue_time,
                component_id="RP",
                delay_days=delay_days,
            )
            subset = target_xy[positions]
            q_r[positions] = recent.log_density(subset[:, 0], subset[:, 1])
            q_p[positions] = preceding.log_density(subset[:, 0], subset[:, 1])
            delay_receipts[issue_time.isoformat()] = {
                "R": _id_receipt(catalog, recent_indices),
                "RP": _id_receipt(catalog, preceding_indices),
            }
        alpha_r = fit_alpha_log_density(q0, q_r, order)
        alpha_p = fit_alpha_log_density(q0, q_p, order)
        fits[delay_days] = (alpha_r, alpha_p)
        source_receipts[str(delay_days)] = delay_receipts
    target_ids = tuple(catalog.event_ids[int(index)] for index in target_indices)
    shared_rate = estimate_shared_m5_6_rate(
        target_ids,
        total_exposure_days=membership.exposure_days,
    )
    fit_evidence = {
        "fit_scope_id": fold.fit_scope_id,
        "fit_target_end_inclusive_utc": fold.fit_target_end_inclusive_utc.isoformat(),
        "fit_target_event_ids": list(target_ids),
        "fit_target_event_ids_sha256": _sha256(canonical_json_bytes(list(target_ids))),
        "fit_target_event_count": len(target_ids),
        "fit_exposure_days": membership.exposure_days,
        "shared_rate": {
            "rate_per_day": shared_rate.rate_per_day,
            "event_count": shared_rate.event_count,
            "exposure_days": shared_rate.exposure_days,
            "assigned_event_ids_sha256": _sha256(
                canonical_json_bytes(list(shared_rate.assigned_event_ids))
            ),
        },
        "alpha_fit_by_delay": {
            str(delay): {
                "R": _alpha_receipt(fits[delay][0]),
                "RP": _alpha_receipt(fits[delay][1]),
            }
            for delay in _DELAYS
        },
        "source_map_receipts_by_delay_and_issue": source_receipts,
    }
    return MappingProxyType(fits), shared_rate, fit_evidence


def _require_supported_fit_targets(count: int) -> None:
    if count <= 0:
        raise Stage2SEvidenceInsufficient("a fold has no supported M5-6 fit targets")


def _prediction_variant(
    *,
    view: _CatalogRoleView,
    snapshot: LocalSupportSnapshot,
    family: SpatialQuadratureFamily,
    s0: NormalizedSpatialDensity,
    issue_time_utc: datetime,
    delay_days: int,
    alpha_r: float,
    alpha_p: float,
) -> tuple[Stage2SModels, Mapping[str, AlarmMask], Mapping[str, object]]:
    recent, recent_indices = _recent_component(
        view=view,
        snapshot=snapshot,
        family=family,
        s0=s0,
        issue_time_utc=issue_time_utc,
        component_id="R",
        delay_days=delay_days,
    )
    preceding, preceding_indices = _recent_component(
        view=view,
        snapshot=snapshot,
        family=family,
        s0=s0,
        issue_time_utc=issue_time_utc,
        component_id="RP",
        delay_days=delay_days,
    )
    models = build_stage2s_models(
        s0,
        recent,
        preceding,
        alpha_r=alpha_r,
        alpha_p=alpha_p,
    )
    alarms = MappingProxyType(
        {model_id: select_alarm_prefix(_model(models, model_id)) for model_id in _MODEL_ORDER}
    )
    receipt = {
        "delay_days": delay_days,
        "causal_sources": {
            "R": _id_receipt(view.catalog, recent_indices),
            "RP": _id_receipt(view.catalog, preceding_indices),
        },
        "models": {model_id: _model_receipt(_model(models, model_id)) for model_id in _MODEL_ORDER},
        "alarms": {model_id: _alarm_receipt(alarms[model_id]) for model_id in _MODEL_ORDER},
    }
    return models, alarms, MappingProxyType(receipt)


def _indices_for_role_rows(
    catalog: Stage2SEarthquakeCatalog,
    event_ids: Sequence[str],
) -> NDArray[np.int64]:
    by_id = {event_id: index for index, event_id in enumerate(catalog.event_ids)}
    try:
        indices = np.asarray([by_id[event_id] for event_id in event_ids], dtype=np.int64)
    except KeyError as exc:
        raise Stage2SFormalError("role session exposed an unknown event ID") from exc
    indices.setflags(write=False)
    return indices


def _seal_all_predictions(
    *,
    inputs: FormalScienceInputs,
    access: _CatalogRoleAccess,
    snapshot: LocalSupportSnapshot,
    family: SpatialQuadratureFamily,
    s0: NormalizedSpatialDensity,
) -> tuple[
    Mapping[int, _FoldFit],
    Mapping[tuple[int, int], _PredictionSpec],
    SealedRecord,
    _CatalogRoleView,
    tuple[AssessmentTargetMembership, ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    Mapping[str, object],
]:
    catalog = access.catalog
    calendar = inputs.preflight.calendar
    issue_dates = {
        fold.fold_index: tuple(
            issue.issue_date_local.isoformat() for issue in fold.assessment_issues
        )
        for fold in calendar.folds
    }
    seal_root = (
        inputs.protocol.repository_root / "data/interim/stage2s/causal_seismicity_screen"
    ).resolve()
    session = RoleSession(seal_root=seal_root, issue_dates_by_fold=issue_dates)
    fold_fits: dict[int, _FoldFit] = {}
    predictions: dict[tuple[int, int], _PredictionSpec] = {}
    fold_summaries: list[dict[str, object]] = []
    issue_summaries: list[dict[str, object]] = []
    fold_seals: list[str] = []
    issue_seals: list[str] = []
    fit_seals: list[str] = []
    grid_ids = [grid.grid_id for grid in family.grids]
    zone_ids = list(inputs.preflight.spatial.zone_ids)
    zone_mapping_sha256 = _sha256(
        canonical_json_bytes(
            {
                cell_id: inputs.preflight.spatial.adapter.construction_zone_id_by_cell_id[cell_id]
                for cell_id in inputs.preflight.spatial.adapter.query_grid.cell_ids
            }
        )
    )
    for fold in calendar.folds:
        fit_view = access.open_before(
            role_id=f"fold_{fold.fold_index}_fit",
            cutoff_utc=fold.fit_target_end_inclusive_utc,
        )
        fit_rows = session.open_fit_view(
            fold_index=fold.fold_index,
            rows=fit_view.raw_rows(),
            fit_cutoff_utc=fold.fit_target_end_inclusive_utc,
        )
        alpha_fit_by_delay, shared_rate, fit_evidence = _fit_fold(
            access=access,
            fit_view=fit_view,
            snapshot=snapshot,
            family=family,
            s0=s0,
            fold=fold,
        )
        fit_indices = _indices_for_role_rows(
            catalog,
            tuple(row.event_id for row in fit_rows),
        )
        fit_receipt = session.seal_fit(
            fold_index=fold.fold_index,
            bindings={
                **fit_evidence,
                "calendar_sha256": calendar.manifest_sha256,
                "fit_raw_view": _id_receipt(catalog, fit_indices),
                "support_id": snapshot.support_id,
                "grid_ids_50_25_12_5km": grid_ids,
                "zone_ids": zone_ids,
                "cell_zone_mapping_sha256": zone_mapping_sha256,
                "code_commit": inputs.code_commit,
            },
        )
        fit_seals.append(fit_receipt.file_sha256)
        fold_issue_seals: list[str] = []
        for issue in fold.assessment_issues:
            issue_date = issue.issue_date_local.isoformat()
            issue_view = access.open_before(
                role_id=f"fold_{fold.fold_index}_issue_{issue_date}_causal_source",
                cutoff_utc=issue.issue_time_utc,
            )
            source_rows = session.open_causal_source_view(
                fold_index=fold.fold_index,
                issue_date=issue_date,
                issue_time_utc=issue.issue_time_utc,
                rows=issue_view.raw_rows(),
            )
            source_indices = _indices_for_role_rows(
                catalog,
                tuple(row.event_id for row in source_rows),
            )
            receipt_by_delay: dict[int, Mapping[str, object]] = {}
            alarm_areas: dict[str, float] = {}
            for delay_days in _DELAYS:
                alpha_r = alpha_fit_by_delay[delay_days][0].alpha
                alpha_p = alpha_fit_by_delay[delay_days][1].alpha
                _, alarms, receipt = _prediction_variant(
                    view=issue_view,
                    snapshot=snapshot,
                    family=family,
                    s0=s0,
                    issue_time_utc=issue.issue_time_utc,
                    delay_days=delay_days,
                    alpha_r=alpha_r,
                    alpha_p=alpha_p,
                )
                receipt_by_delay[delay_days] = receipt
                for model_id in _MODEL_ORDER:
                    alarm_areas[f"delay{delay_days}:{model_id}"] = alarms[model_id].actual_area_km2
            issue_binding = {
                "issue_time_utc": issue.issue_time_utc.isoformat(),
                "horizons_days": list(issue.horizons_days),
                "causal_raw_view": _id_receipt(catalog, source_indices),
                "variants_by_delay": {
                    str(delay): dict(receipt_by_delay[delay]) for delay in _DELAYS
                },
                "alpha_by_delay": {
                    str(delay): {
                        "alpha_R": alpha_fit_by_delay[delay][0].alpha,
                        "alpha_P": alpha_fit_by_delay[delay][1].alpha,
                    }
                    for delay in _DELAYS
                },
                "shared_rate_per_day": shared_rate.rate_per_day,
                "support_id": snapshot.support_id,
                "grid_ids_50_25_12_5km": grid_ids,
                "cell_zone_mapping_sha256": zone_mapping_sha256,
            }
            issue_seal = session.seal_issue(
                fold_index=fold.fold_index,
                issue_date=issue_date,
                bindings=issue_binding,
            )
            fold_issue_seals.append(issue_seal.file_sha256)
            issue_seals.append(issue_seal.file_sha256)
            prediction = _PredictionSpec(
                fold_index=fold.fold_index,
                issue_date=issue_date,
                issue_time_utc=issue.issue_time_utc,
                horizons_days=issue.horizons_days,
                alpha_by_delay=MappingProxyType(
                    {
                        delay: (
                            alpha_fit_by_delay[delay][0].alpha,
                            alpha_fit_by_delay[delay][1].alpha,
                        )
                        for delay in _DELAYS
                    }
                ),
                receipt_by_delay=MappingProxyType(receipt_by_delay),
                issue_seal=issue_seal,
                source_view=issue_view,
            )
            predictions[(fold.fold_index, issue.issue_time_utc.timestamp().__int__())] = prediction
            issue_summaries.append(
                {
                    "fold_index": fold.fold_index,
                    "issue_date": issue_date,
                    "issue_time_utc": issue.issue_time_utc.isoformat(),
                    "horizons_days": list(issue.horizons_days),
                    "issue_prediction_seal_sha256": issue_seal.file_sha256,
                    "actual_alarm_area_km2": alarm_areas,
                }
            )
        fold_seal = session.seal_fold(
            fold_index=fold.fold_index,
            bindings={
                "fit_scope_id": fold.fit_scope_id,
                "fit_receipt_sha256": fit_receipt.file_sha256,
                "issue_prediction_seal_sha256": fold_issue_seals,
                "shared_rate_per_day": shared_rate.rate_per_day,
                "alpha_by_delay": {
                    str(delay): {
                        "alpha_R": alpha_fit_by_delay[delay][0].alpha,
                        "alpha_P": alpha_fit_by_delay[delay][1].alpha,
                    }
                    for delay in _DELAYS
                },
            },
        )
        fold_seals.append(fold_seal.file_sha256)
        _notify(inputs.progress, f"fold_{fold.fold_index}_sealed")
        fold_fits[fold.fold_index] = _FoldFit(
            fold=fold,
            alpha_fit_by_delay=alpha_fit_by_delay,
            shared_rate=shared_rate,
            fit_receipt=fit_receipt,
        )
        fold_summaries.append(
            {
                "fold_index": fold.fold_index,
                "fit_receipt_sha256": fit_receipt.file_sha256,
                "fold_prediction_seal_sha256": fold_seal.file_sha256,
                "fit_event_count": shared_rate.event_count,
                "fit_exposure_days": shared_rate.exposure_days,
                "shared_rate_per_day": shared_rate.rate_per_day,
                "alpha_R_by_delay": {
                    str(delay): alpha_fit_by_delay[delay][0].alpha for delay in _DELAYS
                },
                "alpha_P_by_delay": {
                    str(delay): alpha_fit_by_delay[delay][1].alpha for delay in _DELAYS
                },
            }
        )
    master = session.seal_master(
        bindings={
            **inputs.protocol.identity_mapping(),
            "code_commit": inputs.code_commit,
            "code_tag": CODE_TAG,
            "attempt_ledger_sha256": inputs.attempt_ledger.file_sha256,
            "target_read_receipt_sha256": inputs.target_read_receipt.file_sha256,
            "non_target_preflight_receipt_sha256": inputs.preflight_receipt.file_sha256,
            "catalog_file_sha256": catalog.identity.file_sha256,
            "catalog_content_sha256": catalog.identity.content_sha256,
            "catalog_schema_sha256": catalog.identity.schema_sha256,
            "catalog_row_count": catalog.identity.row_count,
            "support_id": snapshot.support_id,
            "grid_ids_50_25_12_5km": grid_ids,
            "cell_zone_mapping_sha256": zone_mapping_sha256,
        }
    )
    _notify(inputs.progress, "master_prediction_sealed")
    memberships = assessment_target_memberships(
        catalog,
        calendar,
        master_seal=master,
    )
    if len(memberships) != 9:
        raise Stage2SFormalError("master-sealed assessment must contain all nine cells")
    target_indices = np.asarray(
        sorted({int(index) for membership in memberships for index in membership.event_indices}),
        dtype=np.int64,
    )
    assessment_view = access.open_assessment(
        session=session,
        master_seal=master,
        target_indices=target_indices,
    )
    assessment_rows = session.open_assessment_view(rows=assessment_view.raw_rows())
    if len(assessment_rows) != target_indices.size:
        raise Stage2SFormalError("master-sealed assessment target view is incomplete")
    seal_chain: Mapping[str, object] = {
        "fold_fit_receipt_sha256": fit_seals,
        "issue_prediction_seal_sha256": issue_seals,
        "fold_prediction_seal_sha256": fold_seals,
        "master_prediction_seal_sha256": master.file_sha256,
    }
    return (
        MappingProxyType(fold_fits),
        MappingProxyType(predictions),
        master,
        assessment_view,
        memberships,
        tuple(fold_summaries),
        tuple(issue_summaries),
        seal_chain,
    )


def _issue_key(fold_index: int, issue_time_us: int) -> tuple[int, int]:
    return (fold_index, int(_datetime_from_us(issue_time_us).timestamp()))


def _grid_lookups(
    family: SpatialQuadratureFamily,
) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    grid = family.at(25.0)
    by_position = {
        (int(row), int(column)): index
        for index, (row, column) in enumerate(zip(grid.rows, grid.columns, strict=True))
    }
    by_id = {cell_id: index for index, cell_id in enumerate(grid.cell_ids)}
    return by_position, by_id


def _score_delay(
    *,
    delay_days: int,
    catalog: Stage2SEarthquakeCatalog,
    assessment_view: _CatalogRoleView,
    snapshot: LocalSupportSnapshot,
    family: SpatialQuadratureFamily,
    s0: NormalizedSpatialDensity,
    fold_fits: Mapping[int, _FoldFit],
    predictions: Mapping[tuple[int, int], _PredictionSpec],
    memberships: Sequence[AssessmentTargetMembership],
) -> tuple[_ScoredDelay, tuple[Stage2SMapFrame, ...]]:
    if delay_days not in _DELAYS:
        raise ValueError("delay_days must be 0, 1, or 7")
    membership_by_cell = {(item.fold_index, item.horizon_days): item for item in memberships}
    logs: dict[tuple[int, int, str], NDArray[np.float64]] = {}
    hits: dict[tuple[int, int, str], NDArray[np.bool_]] = {}
    filled: dict[tuple[int, int], NDArray[np.bool_]] = {}
    masses: dict[tuple[int, int, str], NDArray[np.float64]] = {}
    issue_row_by_cell: dict[tuple[int, int], dict[int, int]] = {}
    support_by_cell: dict[tuple[int, int], NDArray[np.bool_]] = {}
    grid = family.at(25.0)
    for cell_key, membership in membership_by_cell.items():
        length = membership.event_indices.size
        filled[cell_key] = np.zeros(length, dtype=np.bool_)
        support_by_cell[cell_key] = _support_mask(
            snapshot,
            assessment_view,
            membership.event_indices,
        )
        fold = fold_fits[cell_key[0]].fold
        exposures = tuple(
            item for item in fold.assessment_exposures if item.horizon_days == cell_key[1]
        )
        issue_row_by_cell[cell_key] = {
            item.issue_time_us: index for index, item in enumerate(exposures)
        }
        for model_id in _MODEL_ORDER:
            logs[(*cell_key, model_id)] = np.full(length, np.nan, dtype=np.float64)
            hits[(*cell_key, model_id)] = np.zeros(length, dtype=np.bool_)
            masses[(*cell_key, model_id)] = np.full(
                (len(exposures), grid.cell_count),
                np.nan,
                dtype=np.float64,
            )
    grid_position, _ = _grid_lookups(family)
    map_frames: list[Stage2SMapFrame] = []
    for fold_index in _FOLDS:
        fold_predictions = tuple(
            prediction for prediction in predictions.values() if prediction.fold_index == fold_index
        )
        fold_predictions = tuple(sorted(fold_predictions, key=lambda item: item.issue_time_utc))
        for prediction in fold_predictions:
            alpha_r, alpha_p = prediction.alpha_by_delay[delay_days]
            models, alarms, observed_receipt = _prediction_variant(
                view=prediction.source_view,
                snapshot=snapshot,
                family=family,
                s0=s0,
                issue_time_utc=prediction.issue_time_utc,
                delay_days=delay_days,
                alpha_r=alpha_r,
                alpha_p=alpha_p,
            )
            if dict(observed_receipt) != dict(prediction.receipt_by_delay[delay_days]):
                raise Stage2SFormalError(
                    "master-sealed prediction rematerialization differs from its issue seal"
                )
            if delay_days == 0:
                for horizon in prediction.horizons_days:
                    for model_id in _MODEL_ORDER:
                        alarm = alarms[model_id]
                        mask = np.zeros(grid.cell_count, dtype=np.bool_)
                        mask[alarm.selected_indices] = True
                        map_frames.append(
                            build_rank_map_frame(
                                issue_time_utc=prediction.issue_time_utc.isoformat(),
                                data_cutoff_utc=prediction.issue_time_utc.isoformat(),
                                fold_index=prediction.fold_index,
                                horizon_days=horizon,
                                model_id=cast(Any, model_id),
                                projected_x_m=grid.query_xy_km[:, 0] * 1_000.0,
                                projected_y_m=grid.query_xy_km[:, 1] * 1_000.0,
                                relative_mass=_model(models, model_id).mass_25km,
                                clipped_area_km2=grid.clipped_area_km2,
                                alarm=mask,
                                actual_alarm_area_km2=alarm.actual_area_km2,
                            )
                        )
            issue_us = int((prediction.issue_time_utc - _EPOCH).total_seconds() * 1_000_000)
            for horizon in prediction.horizons_days:
                cell_key = (fold_index, horizon)
                membership = membership_by_cell[cell_key]
                positions = np.flatnonzero(membership.assigned_issue_time_us == issue_us)
                if positions.size:
                    target_indices = membership.event_indices[positions]
                    target_xy = _xy(assessment_view, target_indices)
                    for model_id in _MODEL_ORDER:
                        model = _model(models, model_id)
                        logs[(*cell_key, model_id)][positions] = model.log_density(
                            target_xy[:, 0],
                            target_xy[:, 1],
                        )
                        selected = set(int(value) for value in alarms[model_id].selected_indices)
                        target_hits: list[bool] = []
                        for position, (x_km, y_km) in zip(
                            positions,
                            target_xy,
                            strict=True,
                        ):
                            if not support_by_cell[cell_key][int(position)]:
                                target_hits.append(False)
                                continue
                            row_column = event_cell_index_25km(
                                float(x_km) * 1_000.0,
                                float(y_km) * 1_000.0,
                            )
                            grid_index = grid_position.get(row_column)
                            target_hits.append(grid_index is not None and grid_index in selected)
                        hits[(*cell_key, model_id)][positions] = np.asarray(
                            target_hits,
                            dtype=np.bool_,
                        )
                    filled[cell_key][positions] = True
                mass_row = issue_row_by_cell[cell_key][issue_us]
                for model_id in _MODEL_ORDER:
                    masses[(*cell_key, model_id)][mass_row, :] = _model(
                        models,
                        model_id,
                    ).mass_25km
    scores: dict[tuple[int, int], CellScore] = {}
    for fold_index in _FOLDS:
        for horizon in HORIZONS:
            cell_key = (fold_index, horizon)
            membership = membership_by_cell[cell_key]
            if not filled[cell_key].all():
                raise Stage2SFormalError("not every assessment target was bound to a sealed issue")
            for model_id in _MODEL_ORDER:
                if not np.isfinite(logs[(*cell_key, model_id)]).all():
                    raise Stage2SFormalError("assessment log densities are incomplete")
                if not np.isfinite(masses[(*cell_key, model_id)]).all():
                    raise Stage2SFormalError("assessment operational masses are incomplete")
            scores[cell_key] = score_fold_horizon(
                fold_index=fold_index,
                horizon_days=horizon,
                event_ids=membership.event_ids,
                supported_ig=support_by_cell[cell_key],
                log_density_by_model={
                    model_id: logs[(*cell_key, model_id)] for model_id in _MODEL_ORDER
                },
                hit_by_model={model_id: hits[(*cell_key, model_id)] for model_id in _MODEL_ORDER},
                operational_mass_by_model={
                    model_id: masses[(*cell_key, model_id)] for model_id in _MODEL_ORDER
                },
                shared_rate_per_day=fold_fits[fold_index].shared_rate.rate_per_day,
            )
    return (
        _ScoredDelay(
            scores=MappingProxyType(scores),
            mass_by_cell_model=MappingProxyType(masses),
        ),
        tuple(map_frames),
    )


def _primary_metrics(
    scores: Mapping[tuple[int, int], CellScore],
) -> dict[MetricKey, float]:
    return {
        (contrast, metric): math.fsum(
            (score.information_gain[contrast] if metric == "IG" else score.recall_gain[contrast])
            for score in scores.values()
        )
        / 9.0
        for contrast in CONTRASTS
        for metric in METRICS
    }


def _ig_by_event(score: CellScore, contrast: str) -> dict[str, float]:
    values: dict[str, float] = {}
    position = 0
    for event_id, supported in zip(score.event_ids, score.supported_ig, strict=True):
        if bool(supported):
            values[event_id] = float(score.ig_event_log_ratios[cast(Any, contrast)][position])
            position += 1
    if position != score.ig_event_log_ratios[cast(Any, contrast)].size:
        raise Stage2SFormalError("supported IG event terms do not close to event IDs")
    return values


def _event_blocks(
    scores: Mapping[tuple[int, int], CellScore],
    catalog: Stage2SEarthquakeCatalog,
) -> tuple[EventBlock, ...]:
    catalog_index = {event_id: index for index, event_id in enumerate(catalog.event_ids)}
    blocks: list[EventBlock] = []
    seen: set[str] = set()
    for fold_index in _FOLDS:
        event_horizons: dict[str, list[int]] = {}
        support_by_event: dict[str, bool] = {}
        ig_by_event_horizon: dict[tuple[str, str, int], float] = {}
        recall_by_event_horizon: dict[tuple[str, str, int], float] = {}
        for horizon in HORIZONS:
            score = scores[(fold_index, horizon)]
            ig_maps = {contrast: _ig_by_event(score, contrast) for contrast in CONTRASTS}
            for position, (event_id, supported) in enumerate(
                zip(score.event_ids, score.supported_ig, strict=True)
            ):
                event_horizons.setdefault(event_id, []).append(horizon)
                prior = support_by_event.setdefault(event_id, bool(supported))
                if prior is not bool(supported):
                    raise Stage2SFormalError(
                        "one assessment event has inconsistent support membership"
                    )
                for contrast in CONTRASTS:
                    if bool(supported):
                        ig_by_event_horizon[(event_id, contrast, horizon)] = ig_maps[contrast][
                            event_id
                        ]
                    recall_by_event_horizon[(event_id, contrast, horizon)] = float(
                        score.recall_hit_differences[contrast][position]
                    )
        for event_id in sorted(
            event_horizons,
            key=lambda value: (
                int(catalog.origin_time_us[catalog_index[value]]),
                value.encode("utf-8"),
            ),
        ):
            if event_id in seen:
                raise Stage2SFormalError(
                    "one physical assessment event belongs to multiple disjoint folds"
                )
            seen.add(event_id)
            horizons = tuple(sorted(event_horizons[event_id]))
            supported = support_by_event[event_id]
            blocks.append(
                EventBlock(
                    event_id=event_id,
                    origin_time_utc=_datetime_from_us(
                        int(catalog.origin_time_us[catalog_index[event_id]])
                    ),
                    fold_index=fold_index,
                    horizons=horizons,
                    supported_ig=supported,
                    ig_by_contrast_horizon={
                        (contrast, horizon): ig_by_event_horizon[(event_id, contrast, horizon)]
                        for contrast in CONTRASTS
                        for horizon in horizons
                        if supported
                    },
                    recall_by_contrast_horizon={
                        (contrast, horizon): recall_by_event_horizon[(event_id, contrast, horizon)]
                        for contrast in CONTRASTS
                        for horizon in horizons
                    },
                )
            )
    return tuple(blocks)


def _target_zone_by_event(
    *,
    scores: Mapping[tuple[int, int], CellScore],
    catalog: Stage2SEarthquakeCatalog,
    assessment_view: _CatalogRoleView,
    family: SpatialQuadratureFamily,
    zone_by_cell: Mapping[str, str],
) -> dict[str, str]:
    grid = family.at(25.0)
    by_position = {
        (int(row), int(column)): cell_id
        for cell_id, row, column in zip(
            grid.cell_ids,
            grid.rows,
            grid.columns,
            strict=True,
        )
    }
    catalog_index = {event_id: index for index, event_id in enumerate(catalog.event_ids)}
    result: dict[str, str] = {}
    for score in scores.values():
        for event_id in score.event_ids:
            index = catalog_index[event_id]
            xy = assessment_view.xy(np.asarray([index], dtype=np.int64))
            cell_position = event_cell_index_25km(
                float(xy[0, 0]) * 1_000.0,
                float(xy[0, 1]) * 1_000.0,
            )
            cell_id = by_position.get(cell_position)
            if cell_id is None or cell_id not in zone_by_cell:
                raise Stage2SFormalError(
                    "an inside-study assessment target has no frozen construction zone"
                )
            prior = result.setdefault(event_id, zone_by_cell[cell_id])
            if prior != zone_by_cell[cell_id]:
                raise Stage2SFormalError("one event has inconsistent construction zones")
    return result


def _regional_evidence(
    *,
    scored: _ScoredDelay,
    catalog: Stage2SEarthquakeCatalog,
    assessment_view: _CatalogRoleView,
    family: SpatialQuadratureFamily,
    zone_by_cell: Mapping[str, str],
    zone_ids: Sequence[str],
    fold_fits: Mapping[int, _FoldFit],
    primary_metrics: Mapping[MetricKey, float],
) -> tuple[tuple[RegionContribution, ...], RegionRobustness]:
    grid = family.at(25.0)
    target_zone = _target_zone_by_event(
        scores=scored.scores,
        catalog=catalog,
        assessment_view=assessment_view,
        family=family,
        zone_by_cell=zone_by_cell,
    )
    indices_by_zone = {
        zone_id: np.asarray(
            [
                index
                for index, cell_id in enumerate(grid.cell_ids)
                if zone_by_cell[cell_id] == zone_id
            ],
            dtype=np.int64,
        )
        for zone_id in zone_ids
    }
    contributions = {
        zone_id: {(contrast, metric): 0.0 for contrast in CONTRASTS for metric in METRICS}
        for zone_id in zone_ids
    }
    ig_events: dict[str, set[str]] = {zone_id: set() for zone_id in zone_ids}
    recall_events: dict[str, set[str]] = {zone_id: set() for zone_id in zone_ids}
    for fold_index in _FOLDS:
        rate = fold_fits[fold_index].shared_rate.rate_per_day
        for horizon in HORIZONS:
            score = scored.scores[(fold_index, horizon)]
            supported_count = int(np.count_nonzero(score.supported_ig))
            recall_count = len(score.event_ids)
            ig_maps = {contrast: _ig_by_event(score, contrast) for contrast in CONTRASTS}
            for position, (event_id, supported) in enumerate(
                zip(score.event_ids, score.supported_ig, strict=True)
            ):
                zone_id = target_zone[event_id]
                recall_events[zone_id].add(event_id)
                if bool(supported):
                    ig_events[zone_id].add(event_id)
                for contrast in CONTRASTS:
                    if bool(supported):
                        contributions[zone_id][(contrast, "IG")] += (
                            ig_maps[contrast][event_id] / supported_count / 9.0
                        )
                    contributions[zone_id][(contrast, "recall")] += (
                        float(score.recall_hit_differences[contrast][position]) / recall_count / 9.0
                    )
            for zone_id in zone_ids:
                indices = indices_by_zone[zone_id]
                for contrast, comparator in (
                    ("S1_minus_S0", "S0"),
                    ("S1_minus_SP", "SP"),
                ):
                    candidate = scored.mass_by_cell_model[(fold_index, horizon, "S1")][:, indices]
                    reference = scored.mass_by_cell_model[(fold_index, horizon, comparator)][
                        :, indices
                    ]
                    compensator = math.fsum(
                        rate
                        * horizon
                        * (
                            math.fsum(float(value) for value in candidate[row])
                            - math.fsum(float(value) for value in reference[row])
                        )
                        for row in range(candidate.shape[0])
                    )
                    contributions[zone_id][(contrast, "IG")] -= compensator / supported_count / 9.0
    regions = tuple(
        RegionContribution(
            zone_id=zone_id,
            ig_event_count=len(ig_events[zone_id]),
            recall_event_count=len(recall_events[zone_id]),
            contributions=contributions[zone_id],
        )
        for zone_id in zone_ids
    )
    return regions, compute_region_robustness(
        regions,
        primary_metrics=primary_metrics,
        required_zone_count=len(tuple(zone_ids)),
    )


def _sequence_evidence(
    *,
    scores: Mapping[tuple[int, int], CellScore],
    catalog: Stage2SEarthquakeCatalog,
    primary_metrics: Mapping[MetricKey, float],
) -> SequenceDiagnostic:
    catalog_index = {event_id: index for index, event_id in enumerate(catalog.event_ids)}
    contributions_by_event: dict[str, dict[MetricKey, float]] = {}
    model_hits_by_event: dict[str, dict[str, float]] = {}
    for fold_index in _FOLDS:
        for horizon in HORIZONS:
            score = scores[(fold_index, horizon)]
            supported_count = int(np.count_nonzero(score.supported_ig))
            recall_count = len(score.event_ids)
            ig_maps = {contrast: _ig_by_event(score, contrast) for contrast in CONTRASTS}
            for position, (event_id, supported) in enumerate(
                zip(score.event_ids, score.supported_ig, strict=True)
            ):
                values = contributions_by_event.setdefault(
                    event_id,
                    {(contrast, metric): 0.0 for contrast in CONTRASTS for metric in METRICS},
                )
                model_hits = model_hits_by_event.setdefault(
                    event_id,
                    {model_id: 0.0 for model_id in _MODEL_ORDER},
                )
                for contrast in CONTRASTS:
                    if bool(supported):
                        values[(contrast, "IG")] += (
                            ig_maps[contrast][event_id] / supported_count / 9.0
                        )
                    values[(contrast, "recall")] += (
                        float(score.recall_hit_differences[contrast][position]) / recall_count / 9.0
                    )
                for model_id in _MODEL_ORDER:
                    model_hits[model_id] += (
                        float(score.hit_by_model[cast(Any, model_id)][position])
                        / recall_count
                        / 9.0
                    )
    events = tuple(
        SequenceEvent(
            event_id=event_id,
            origin_time_utc=_datetime_from_us(int(catalog.origin_time_us[catalog_index[event_id]])),
            longitude=float(catalog.longitude[catalog_index[event_id]]),
            latitude=float(catalog.latitude[catalog_index[event_id]]),
            contributions=values,
            model_hit_contributions=cast(Any, model_hits_by_event[event_id]),
        )
        for event_id, values in sorted(
            contributions_by_event.items(),
            key=lambda item: (
                int(catalog.origin_time_us[catalog_index[item[0]]]),
                item[0].encode("utf-8"),
            ),
        )
    )
    return compute_sequence_diagnostic(
        events,
        primary_metrics=primary_metrics,
        closure=build_sequence_closure_evidence(scores),
    )


def _cell_record(score: CellScore) -> dict[str, object]:
    return {
        "fold_index": score.fold_index,
        "horizon_days": score.horizon_days,
        "issue_count": score.issue_count,
        "event_ids": list(score.event_ids),
        "supported_ig": [bool(value) for value in score.supported_ig],
        "hit_by_model": {
            model_id: [bool(value) for value in score.hit_by_model[cast(Any, model_id)]]
            for model_id in _MODEL_ORDER
        },
        "ig_event_log_ratios": {
            contrast: [float(value) for value in score.ig_event_log_ratios[contrast]]
            for contrast in CONTRASTS
        },
        "recall_hit_differences": {
            contrast: [float(value) for value in score.recall_hit_differences[contrast]]
            for contrast in CONTRASTS
        },
        "compensator_differences": dict(score.compensator_differences),
        "information_gain": dict(score.information_gain),
        "recall_gain": dict(score.recall_gain),
    }


def _bootstrap_records(
    bootstrap: BootstrapFamilies,
) -> tuple[dict[str, object], tuple[tuple[float, float, float, float], ...]]:
    summary = {
        "entropy_uint128": bootstrap.entropy_uint128,
        "replications": 2_000,
        "column_order": [f"{contrast}:{metric}" for contrast, metric in _BOOTSTRAP_COLUMN_ORDER],
        "intervals": {
            f"{contrast}:{metric}": {
                "point": bootstrap.intervals[(contrast, metric)].point,
                "lower": bootstrap.intervals[(contrast, metric)].lower,
                "upper": bootstrap.intervals[(contrast, metric)].upper,
            }
            for contrast, metric in _BOOTSTRAP_COLUMN_ORDER
        },
    }
    rows = tuple(
        cast(
            tuple[float, float, float, float],
            tuple(
                bootstrap.intervals[key].replicates[replicate_index]
                for key in _BOOTSTRAP_COLUMN_ORDER
            ),
        )
        for replicate_index in range(2_000)
    )
    return summary, rows


def _regional_record(
    regions: Sequence[RegionContribution],
    robustness: RegionRobustness,
) -> dict[str, object]:
    return {
        "regions": [
            {
                "zone_id": region.zone_id,
                "ig_event_count": region.ig_event_count,
                "recall_event_count": region.recall_event_count,
                "contributions": {
                    f"{contrast}:{metric}": region.contributions[(contrast, metric)]
                    for contrast in CONTRASTS
                    for metric in METRICS
                },
            }
            for region in regions
        ],
        "results": {
            f"{contrast}:{metric}": {
                "event_bearing_zone_count": result.event_bearing_zone_count,
                "positive_event_bearing_zone_count": (result.positive_event_bearing_zone_count),
                "strongest_positive_zone_id": result.strongest_positive_zone_id,
                "strongest_positive_contribution": (result.strongest_positive_contribution),
                "leave_strongest_out_residual": (result.leave_strongest_out_residual),
                "passed": result.passed,
            }
            for (contrast, metric), result in robustness.results.items()
        },
        "failures": list(robustness.failures),
        "passed": robustness.passed,
    }


def _sequence_record(sequence: SequenceDiagnostic) -> dict[str, object]:
    largest_count = next(
        component
        for component in sequence.components
        if component.component_id == sequence.largest_count_component_id
    )

    def component_record(component: object) -> dict[str, object]:
        typed = cast(Any, component)
        return {
            "component_id": typed.component_id,
            "event_ids": list(typed.event_ids),
            "event_count": len(typed.event_ids),
            "event_fraction": typed.event_fraction,
            "origin_time_span_days": typed.origin_time_span_days,
            "max_pairwise_geodesic_distance_km": (typed.max_pairwise_geodesic_distance_km),
            "contributions": {
                f"{contrast}:{metric}": typed.contributions[(contrast, metric)]
                for contrast in CONTRASTS
                for metric in METRICS
            },
            "model_hits": {
                model_id: {
                    "raw": typed.model_hit_contributions[model_id],
                    "fraction": typed.model_hit_fractions[model_id],
                }
                for model_id in _MODEL_ORDER
            },
            "information_gain": {
                contrast: {
                    "raw": typed.contributions[(contrast, "IG")],
                    "fraction": typed.ig_fractions[contrast],
                }
                for contrast in CONTRASTS
            },
        }

    return {
        "component_count": len(sequence.components),
        "event_resampling_unit_count": sum(
            len(component.event_ids) for component in sequence.components
        ),
        "global_residual": {
            f"{contrast}:{metric}": sequence.global_residual[(contrast, metric)]
            for contrast in CONTRASTS
            for metric in METRICS
        },
        "primary_model_recall": dict(sequence.primary_model_recall),
        "components": [component_record(component) for component in sequence.components],
        "largest_count_component_id": sequence.largest_count_component_id,
        "largest_count_component": {
            **component_record(largest_count),
            "leave_out": {
                f"{contrast}:{metric}": sequence.leave_largest_count_out[(contrast, metric)]
                for contrast in CONTRASTS
                for metric in METRICS
            },
        },
        "largest_gain_component_id": {
            f"{contrast}:{metric}": component_id
            for (contrast, metric), component_id in (sequence.largest_gain_component_id.items())
        },
        "largest_gain_component": {
            f"{contrast}:{metric}": {
                "component_id": sequence.largest_gain_component_id[(contrast, metric)],
                "raw_contribution": next(
                    component.contributions[(contrast, metric)]
                    for component in sequence.components
                    if component.component_id
                    == sequence.largest_gain_component_id[(contrast, metric)]
                ),
                "leave_out": sequence.leave_largest_gain_out[(contrast, metric)],
            }
            for contrast in CONTRASTS
            for metric in METRICS
        },
        "leave_out_residual": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in sequence.leave_largest_gain_out.items()
        },
        "leave_largest_count_out": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in sequence.leave_largest_count_out.items()
        },
        "leave_largest_gain_out": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in sequence.leave_largest_gain_out.items()
        },
        "claim_limited": sequence.claim_limited,
        "interpretation_limit": sequence.interpretation_limit,
    }


def _gate_record(assessment: GateAssessment) -> dict[str, object]:
    passed = assessment.decision.status == "passed_development_signal"
    return {
        "status": assessment.decision.status,
        "reasons": list(assessment.decision.reasons),
        "supported_event_union_count": assessment.supported_event_union_count,
        "recall_event_union_count": assessment.recall_event_union_count,
        "fold_macros": {
            f"{contrast}:{metric}:fold{fold_index}": value
            for (contrast, metric, fold_index), value in (assessment.fold_macros.items())
        },
        "horizon_macros": {
            f"{contrast}:{metric}:h{horizon}": value
            for (contrast, metric, horizon), value in (assessment.horizon_macros.items())
        },
        "overall_macros": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in assessment.overall_macros.items()
        },
        "claim_limited": assessment.claim_limited,
        "interpretation_limit": assessment.interpretation_limit,
        "interpretation_scope": (
            "sequence_associated_continuation_only"
            if assessment.claim_limited
            else (
                "broad_regional_gain_not_sequence_limited"
                if passed
                else "no_sequence_interpretation_limit"
            )
        ),
        "science_value_category": ("direct_improvement" if passed else "no_material_progress"),
        "direct_prediction_improvement": (
            "reused_development_signal_only_requires_new_prospective_validation"
            if passed
            else "none"
        ),
        "evidence_scope": (
            "historical_reused_stage2r_development_assessment_not_independent_validation"
        ),
        "route_decision": (
            "freeze_for_separate_prospective_protocol"
            if passed
            else "stop_stage2s_without_result_based_model_or_window_retry"
        ),
    }


def _new_formal_record(
    *,
    inputs: FormalScienceInputs,
    catalog: Stage2SEarthquakeCatalog,
    snapshot: LocalSupportSnapshot,
    s0: NormalizedSpatialDensity,
    fold_summaries: Sequence[Mapping[str, object]],
    issue_summaries: Sequence[Mapping[str, object]],
    seal_chain: Mapping[str, object],
    scores: Mapping[tuple[int, int], CellScore],
    bootstrap_summary: Mapping[str, object],
    bootstrap_rows: Sequence[Sequence[float]],
    regions: Sequence[RegionContribution],
    regional: RegionRobustness,
    sequence: SequenceDiagnostic,
    latency: Sequence[LatencyMetrics],
    gate: GateAssessment,
    artifact_sha256_by_name: Mapping[str, object],
) -> Stage2SWholeRunRecord:
    return Stage2SWholeRunRecord(
        mode="formal_development",
        identity={
            **inputs.protocol.identity_mapping(),
            "code_commit": inputs.code_commit,
            "code_tag": CODE_TAG,
            "execution_role": ("one_shot_historical_reused_development_assessment"),
            "not_independent_validation": True,
            "not_current_forecast": True,
            "not_absolute_earthquake_probability": True,
        },
        input_receipts={
            "non_target_preflight_receipt_sha256": (inputs.preflight_receipt.file_sha256),
            "attempt_ledger_sha256": inputs.attempt_ledger.file_sha256,
            "target_read_receipt_sha256": (inputs.target_read_receipt.file_sha256),
            "catalog_file_sha256": catalog.identity.file_sha256,
            "catalog_content_sha256": catalog.identity.content_sha256,
            "catalog_schema_sha256": catalog.identity.schema_sha256,
            "catalog_row_count": catalog.identity.row_count,
            "catalog_physical_open_count": 1,
            "catalog_parquet_path_parse_count": 0,
            "catalog_buffer_reader_parse_count": 1,
            "fold4_support_id": snapshot.support_id,
            "fold4_support_manifest_full_record_sha256": _sha256(
                canonical_json_bytes(scientific_mapping(build_local_support_manifest(snapshot)))
            ),
            "s0_model_receipt": _model_receipt(s0),
        },
        fold_fit_summaries=fold_summaries,
        issue_prediction_summaries=issue_summaries,
        seal_chain=seal_chain,
        cell_scores=[
            _cell_record(scores[(fold_index, horizon)])
            for fold_index in _FOLDS
            for horizon in HORIZONS
        ],
        bootstrap_summary=bootstrap_summary,
        bootstrap_rows=bootstrap_rows,
        regional_evidence=_regional_record(regions, regional),
        sequence_evidence=_sequence_record(sequence),
        descriptive_point_estimates={
            "SP_minus_S0": {
                "information_gain": descriptive_sp_minus_s0_point_estimates(gate.overall_macros)[
                    "IG"
                ],
                "recall_gain": descriptive_sp_minus_s0_point_estimates(gate.overall_macros)[
                    "recall"
                ],
                "derivation": "S1_minus_S0_minus_S1_minus_SP",
                "inferential_status": "descriptive_point_estimate_only",
                "included_in_bootstrap_ci": False,
                "included_in_gate": False,
            }
        },
        latency_evidence=[
            {
                "delay_days": item.delay_days,
                "metrics": {
                    f"{contrast}:{metric}": item.values[(contrast, metric)]
                    for contrast in CONTRASTS
                    for metric in METRICS
                },
            }
            for item in latency
        ],
        gate_evidence=_gate_record(gate),
        artifact_sha256_by_name=artifact_sha256_by_name,
    )


def _render_payload(
    *,
    inputs: FormalScienceInputs,
    record: Stage2SWholeRunRecord,
    map_frames: Sequence[Stage2SMapFrame],
) -> Stage2SRenderPayload:
    background = _config_mapping(
        inputs.protocol.config["long_term_background"],
        label="long_term_background",
    )
    return Stage2SRenderPayload(
        record=record,
        s0_training_cutoff_utc=cast(str, background["fit_end_utc"]),
        recent_origin_window="T-30d < origin_time_utc <= T",
        preceding_origin_window="T-60d < origin_time_utc <= T-30d",
        available_at_cutoff="available_at <= T (primary 0d feed delay)",
        map_frames=tuple(map_frames),
    )


def _render_formal_artifacts(
    *,
    inputs: FormalScienceInputs,
    record: Stage2SWholeRunRecord,
    map_frames: Sequence[Stage2SMapFrame],
) -> Mapping[str, str]:
    bundle = render_stage2s_bundle(
        _render_payload(
            inputs=inputs,
            record=record,
            map_frames=map_frames,
        )
    )
    return bundle.artifact_sha256_by_name


def _verify_and_write_formal_artifacts(
    *,
    inputs: FormalScienceInputs,
    record: Stage2SWholeRunRecord,
    map_frames: Sequence[Stage2SMapFrame],
    expected_hashes: Mapping[str, str],
) -> None:
    bundle: Stage2SRenderedBundle = render_stage2s_bundle(
        _render_payload(
            inputs=inputs,
            record=record,
            map_frames=map_frames,
        )
    )
    if dict(bundle.artifact_sha256_by_name) != dict(expected_hashes):
        raise Stage2SFormalError(
            "final record changed deterministic render bytes after artifact binding"
        )
    output_dir = (
        inputs.protocol.repository_root / "outputs/stage2s/causal_seismicity_screen"
    ).resolve()
    for artifact in bundle.artifacts:
        _write_bytes_o_excl(output_dir / artifact.name, artifact.payload)


def run_formal_science(
    inputs: FormalScienceInputs,
    catalog: Stage2SEarthquakeCatalog,
) -> Stage2SWholeRunRecord:
    """Run the sealed three-fold assessment from the single in-memory catalogue."""

    access = _CatalogRoleAccess(catalog, inputs.preflight.spatial)
    background = _config_mapping(
        inputs.protocol.config["long_term_background"],
        label="long_term_background",
    )
    fit_end = datetime.fromisoformat(
        cast(str, background["fit_end_utc"]).replace("Z", "+00:00")
    ).astimezone(UTC)
    support_view = access.open_before(
        role_id="fold4_support_and_s0",
        cutoff_utc=fit_end,
    )
    snapshot = _rebuild_fold4(inputs, support_view)
    _notify(inputs.progress, "fold4_support_rebuilt")
    family = to_spatial_quadrature_family(inputs.preflight.spatial.adapter)
    s0 = _build_s0(inputs, support_view, snapshot, family)
    (
        fold_fits,
        predictions,
        _master,
        assessment_view,
        memberships,
        fold_summaries,
        issue_summaries,
        seal_chain,
    ) = _seal_all_predictions(
        inputs=inputs,
        access=access,
        snapshot=snapshot,
        family=family,
        s0=s0,
    )
    primary, map_frames = _score_delay(
        delay_days=0,
        catalog=catalog,
        assessment_view=assessment_view,
        snapshot=snapshot,
        family=family,
        s0=s0,
        fold_fits=fold_fits,
        predictions=predictions,
        memberships=memberships,
    )
    delayed: dict[int, _ScoredDelay] = {}
    for delay_days in (1, 7):
        delayed[delay_days], _ = _score_delay(
            delay_days=delay_days,
            catalog=catalog,
            assessment_view=assessment_view,
            snapshot=snapshot,
            family=family,
            s0=s0,
            fold_fits=fold_fits,
            predictions=predictions,
            memberships=memberships,
        )
    _notify(inputs.progress, "assessment_scored")
    blocks = _event_blocks(primary.scores, catalog)
    compensators = {
        (contrast, fold_index, horizon): primary.scores[
            (fold_index, horizon)
        ].compensator_differences[contrast]
        for contrast in CONTRASTS
        for fold_index in _FOLDS
        for horizon in HORIZONS
    }
    bootstrap = bootstrap_families(blocks, compensators=compensators)
    primary_metrics = _primary_metrics(primary.scores)
    zone_mapping = inputs.preflight.spatial.adapter.construction_zone_id_by_cell_id
    regions, regional = _regional_evidence(
        scored=primary,
        catalog=catalog,
        assessment_view=assessment_view,
        family=family,
        zone_by_cell=zone_mapping,
        zone_ids=inputs.preflight.spatial.zone_ids,
        fold_fits=fold_fits,
        primary_metrics=primary_metrics,
    )
    sequence = _sequence_evidence(
        scores=primary.scores,
        catalog=catalog,
        primary_metrics=primary_metrics,
    )
    latency = tuple(
        LatencyMetrics(
            delay_days=delay_days,
            values=_primary_metrics(delayed[delay_days].scores),
        )
        for delay_days in (1, 7)
    )
    gate = evaluate_stage2s_gate(
        primary.scores,
        bootstrap=bootstrap,
        regional=regional,
        latency=latency,
        sequence=sequence,
    )
    bootstrap_summary, bootstrap_rows = _bootstrap_records(bootstrap)
    provisional = _new_formal_record(
        inputs=inputs,
        catalog=catalog,
        snapshot=snapshot,
        s0=s0,
        fold_summaries=fold_summaries,
        issue_summaries=issue_summaries,
        seal_chain=seal_chain,
        scores=primary.scores,
        bootstrap_summary=bootstrap_summary,
        bootstrap_rows=bootstrap_rows,
        regions=regions,
        regional=regional,
        sequence=sequence,
        latency=latency,
        gate=gate,
        artifact_sha256_by_name={},
    )
    if not map_frames:
        raise Stage2SFormalError("formal result rendering requires primary sealed maps")
    artifact_hashes = _render_formal_artifacts(
        inputs=inputs,
        record=provisional,
        map_frames=map_frames,
    )
    final = _new_formal_record(
        inputs=inputs,
        catalog=catalog,
        snapshot=snapshot,
        s0=s0,
        fold_summaries=fold_summaries,
        issue_summaries=issue_summaries,
        seal_chain=seal_chain,
        scores=primary.scores,
        bootstrap_summary=bootstrap_summary,
        bootstrap_rows=bootstrap_rows,
        regions=regions,
        regional=regional,
        sequence=sequence,
        latency=latency,
        gate=gate,
        artifact_sha256_by_name=artifact_hashes,
    )
    _verify_and_write_formal_artifacts(
        inputs=inputs,
        record=final,
        map_frames=map_frames,
        expected_hashes=artifact_hashes,
    )
    _notify(inputs.progress, "artifacts_rendered")
    return final
