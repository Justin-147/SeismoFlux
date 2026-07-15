"""Sealed dry-run plan and immutable local execution for stage-3 anomaly features.

This module is deliberately target blind.  Its scientific loader accepts only the two
anomaly datasets frozen by :mod:`seismoflux.features.anomaly.input`; earthquake catalogs,
target labels, completeness masks, scores, and locked-test artifacts have no parameter or
filesystem-discovery path into this runner.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import shape

from seismoflux.background.execution import (
    ExecutionSealError,
    GitCommandRunner,
    RepositoryIdentity,
    detect_physical_core_count,
    require_repository_identity,
    subprocess_git_runner,
)
from seismoflux.background.runner import process_peak_working_set_bytes
from seismoflux.config import (
    SeismoFluxConfig,
    load_config,
    normalize_relative_path,
    project_root_for,
    sha256_file,
)
from seismoflux.features.anomaly.audit import AuditResult, run_lineage_replay_audit
from seismoflux.features.anomaly.config import (
    ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256,
    AnomalyHistoryConfig,
    load_anomaly_history_config,
)
from seismoflux.features.anomaly.contracts import ANOMALY_STATE_HISTORY_CONTRACT
from seismoflux.features.anomaly.dictionary import (
    DEFAULT_FEATURE_DICTIONARY,
    FEATURE_STORE_SORT_KEYS,
    FeatureDictionary,
    build_feature_store_schema,
)
from seismoflux.features.anomaly.engine import (
    Stage3FeatureEngine,
    Stage3IssueFeatureAudit,
)
from seismoflux.features.anomaly.grid import Stage3QueryGrid, build_stage3_query_grid
from seismoflux.features.anomaly.input import (
    Stage3InputSpec,
    load_stage3_inputs,
    stage3_input_spec_from_config,
)
from seismoflux.features.anomaly.public_deliverables import (
    PUBLIC_AUDIT_SVG_PATH,
    PUBLIC_DICTIONARY_PATH,
    PUBLIC_REGISTRY_PATH,
    PUBLIC_REPORT_PATH,
    PublishedStage3PublicDeliverables,
    build_stage3_public_deliverables,
    publish_stage3_public_deliverables,
)
from seismoflux.features.anomaly.publication import (
    STAGE3_MANIFEST_FILENAME,
    build_stage3_manifest,
    load_and_verify_stage3_bundle,
    stage3_bundle_id,
    stage3_bundle_workspace,
    stage3_identity_sha256,
    write_stage3_manifest,
)
from seismoflux.features.anomaly.snapshot import build_issue_snapshots
from seismoflux.features.anomaly.state import (
    AnomalyState,
    build_anomaly_state_history,
    state_records,
)
from seismoflux.features.anomaly.storage import (
    Stage3DatasetArtifact,
    verify_stage3_parquet_artifact,
    write_parquet_row_groups_atomic,
)
from seismoflux.features.anomaly.synthetic_audit import (
    SyntheticPrefixAuditResult,
    run_synthetic_prefix_audit,
)

ANOMALY_HISTORY_CONFIG_REFERENCE = "configs/anomaly_history.yaml"
ANOMALY_FEATURE_PROTOCOL_REFERENCE = "docs/anomaly_feature_protocol.md"
_STATE_FILENAME = "anomaly_state_history.parquet"
_FEATURE_FILENAME = "anomaly_feature_store.parquet"
_QUERY_CHUNK_SIZE = 256

ProgressCallback = Callable[[str], None]
MemoryProbe = Callable[[], int | None]
PhysicalCoreProbe = Callable[[], int | None]


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty trimmed string")
    return value


def _safe_project_file(project_root: Path, reference: str, *, label: str) -> Path:
    normalized = normalize_relative_path(reference)
    root = project_root.resolve()
    path = (root / normalized).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or unsafe {label}: {normalized}")
    return path


def _safe_project_directory(project_root: Path, reference: str) -> Path:
    normalized = normalize_relative_path(reference)
    root = project_root.resolve()
    path = (root / normalized).resolve()
    if not path.is_relative_to(root):
        raise ValueError("stage-3 output root escapes the project")
    return path


def _require_json_subset(
    actual: object,
    expected: object,
    *,
    location: str,
) -> None:
    if isinstance(expected, Mapping):
        observed = _mapping(actual, label=location)
        for key, expected_value in expected.items():
            if key not in observed:
                raise ValueError(f"missing prerequisite registry assertion: {location}.{key}")
            _require_json_subset(
                observed[key],
                expected_value,
                location=f"{location}.{key}",
            )
        return
    if actual != expected:
        raise ValueError(
            f"prerequisite registry assertion differs at {location}: "
            f"expected={expected!r}, observed={actual!r}"
        )


@dataclass(frozen=True, slots=True)
class _FrozenFile:
    reference: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference", normalize_relative_path(self.reference))
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("stage-3 frozen-file identity must be lowercase SHA-256")


def _load_authorization_registry(
    project_root: Path,
    config: AnomalyHistoryConfig,
) -> _FrozenFile:
    authorization = _mapping(config.authorization, label="authorization")
    if authorization.get("prerequisite_gate") != "G1-LS":
        raise ValueError("stage 3 requires the passed G1-LS prerequisite gate")
    if authorization.get("role") != "governance_only_not_a_scientific_feature_input":
        raise ValueError("the G1-LS registry must remain governance-only")
    reference = _text(
        authorization,
        "prerequisite_registry",
        label="authorization",
    )
    expected_sha256 = _text(
        authorization,
        "prerequisite_registry_sha256",
        label="authorization",
    )
    path = _safe_project_file(project_root, reference, label="G1-LS prerequisite registry")
    if sha256_file(path) != expected_sha256:
        raise ValueError("G1-LS prerequisite registry hash mismatch")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("G1-LS prerequisite registry must be valid UTF-8 JSON") from exc
    expected_assertions = _mapping(
        authorization.get("required_registry_assertions"),
        label="authorization.required_registry_assertions",
    )
    _require_json_subset(document, expected_assertions, location="prerequisite_registry")
    return _FrozenFile(reference, expected_sha256)


def _validate_project_alignment(
    project: SeismoFluxConfig,
    config: AnomalyHistoryConfig,
    spec: Stage3InputSpec,
) -> None:
    if project.project.random_seed != 147:
        raise ValueError("stage-3 audit seed requires project random_seed 147")
    if project.study_area.polygon != spec.study_area.path:
        raise ValueError("base and stage-3 protocols disagree on the study-area path")
    query_grid = _mapping(config.query_grid, label="query_grid")
    if project.study_area.equal_area_crs != query_grid.get("equal_area_crs"):
        raise ValueError("base and stage-3 protocols disagree on the equal-area CRS")
    if project.integration.base_cell_km != spec.query_grid_cell_km:
        raise ValueError("base and stage-3 protocols disagree on the 25 km query grid")
    if project.parallel.reserve_physical_cores < 2:
        raise ValueError("stage-3 execution must reserve at least two physical cores")
    if project.parallel.nested_parallelism is not False:
        raise ValueError("stage-3 execution forbids nested parallelism")


def _resource_plan(
    project: SeismoFluxConfig,
    *,
    physical_core_probe: PhysicalCoreProbe,
) -> Stage3ResourcePlan:
    detected = physical_core_probe()
    if detected is not None and (
        isinstance(detected, bool) or not isinstance(detected, int) or detected <= 0
    ):
        raise ValueError("physical-core probe must return a positive integer or None")
    reserve = project.parallel.reserve_physical_cores
    if detected is None:
        raise ValueError(
            "stage-3 physical-core detection failed; cannot verify the reserved-core budget"
        )
    if detected <= reserve:
        raise ValueError("stage-3 detected physical cores must exceed the reserved-core budget")
    effective = min(project.parallel.max_workers, detected - reserve)
    return Stage3ResourcePlan(
        detected_physical_cores=detected,
        reserve_physical_cores=reserve,
        configured_max_workers=project.parallel.max_workers,
        effective_workers=effective,
        spatial_workers=2 if effective >= 2 else 1,
        nested_parallelism=False,
    )


@dataclass(frozen=True, slots=True)
class Stage3ResourcePlan:
    """Bounded deterministic parallel plan for the two independent spatial aggregates."""

    detected_physical_cores: int | None
    reserve_physical_cores: int
    configured_max_workers: int
    effective_workers: int
    spatial_workers: int
    nested_parallelism: bool

    def __post_init__(self) -> None:
        if self.reserve_physical_cores < 2:
            raise ValueError("stage-3 resource plan must reserve at least two physical cores")
        if self.detected_physical_cores is None:
            raise ValueError("stage-3 resource plan requires a verified physical-core count")
        if self.detected_physical_cores <= self.reserve_physical_cores:
            raise ValueError("stage-3 resource plan does not leave a worker after core reserve")
        if self.configured_max_workers <= 0 or self.effective_workers <= 0:
            raise ValueError("stage-3 worker counts must be positive")
        if self.effective_workers > (self.detected_physical_cores - self.reserve_physical_cores):
            raise ValueError("stage-3 effective workers violate the reserved-core budget")
        if self.spatial_workers not in (1, 2):
            raise ValueError("stage-3 spatial worker count must be one or two")
        if self.spatial_workers > self.effective_workers:
            raise ValueError("spatial workers exceed the effective worker budget")
        if self.nested_parallelism:
            raise ValueError("stage-3 nested parallelism is forbidden")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Stage3RunPlan:
    """Validated score-free plan; constructing it never loads scientific Parquet rows."""

    project_root: Path
    main_config_path: Path
    anomaly_config_path: Path
    protocol_document_path: Path
    project: SeismoFluxConfig
    config: AnomalyHistoryConfig
    input_spec: Stage3InputSpec
    resources: Stage3ResourcePlan
    authorization_registry: _FrozenFile
    frozen_files: tuple[_FrozenFile, ...]
    planned_inputs: tuple[str, ...]
    planned_outputs: tuple[str, ...]

    def to_manifest_details(self) -> dict[str, object]:
        return {
            "protocol_version": self.config.protocol_version,
            "protocol_semantic_sha256": ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256,
            "execution_mode": self.config.execution_mode,
            "expected_actual_snapshot_count": self.config.expected_report_period_count,
            "scientific_datasets": list(self.input_spec_dataset_names),
            "query_grid": {
                "role": "target_independent_local_feature_queries",
                "cell_size_km": self.input_spec.query_grid_cell_km,
                "adaptive_refinement": False,
            },
            "temporal_windows_weeks": list(self.config.temporal_windows_weeks),
            "closed_ball_radii_km": list(self.config.spatial_radii_km),
            "gaussian_bandwidths_km": list(self.config.gaussian_bandwidths_km),
            "query_chunk_size": _QUERY_CHUNK_SIZE,
            "resources": self.resources.as_dict(),
            "forbidden_inputs": [
                "earthquake_catalogs",
                "earthquake_targets_or_labels",
                "completeness_Mc_or_G1_support_masks",
                "human_forecasts_or_results",
                "locked_test",
            ],
            "locked_test": {
                "action": "do_not_run",
                "run": False,
                "target_count": None,
                "target_ids": [],
                "score_ids": [],
                "artifact_ids": [],
                "result": None,
            },
            "publication": {
                "local_content_addressed_bundle": True,
                "overwrite_existing_bundle": False,
                "public_fixed_deliverables": "published_only_after_local_audit",
            },
        }

    @property
    def input_spec_dataset_names(self) -> tuple[str, str]:
        return (
            self.input_spec.observation.dataset_name,
            self.input_spec.report_period.dataset_name,
        )

    @property
    def local_output_root_reference(self) -> str:
        outputs = _mapping(self.config.outputs, label="outputs")
        return normalize_relative_path(
            _text(outputs, "local_content_addressed_root", label="outputs")
        )

    @property
    def public_output_references(self) -> tuple[str, ...]:
        outputs = _mapping(self.config.outputs, label="outputs")
        public = _mapping(outputs.get("public"), label="outputs.public")
        return tuple(
            normalize_relative_path(_text(public, key, label="outputs.public"))
            for key in ("registry", "feature_dictionary", "report", "audit_svg")
        )


def _output_references(config: AnomalyHistoryConfig) -> tuple[str, ...]:
    outputs = _mapping(config.outputs, label="outputs")
    root = _text(outputs, "local_content_addressed_root", label="outputs")
    public = _mapping(outputs.get("public"), label="outputs.public")
    return (
        f"{normalize_relative_path(root)}/{{bundle_id}}/{_STATE_FILENAME}",
        f"{normalize_relative_path(root)}/{{bundle_id}}/{_FEATURE_FILENAME}",
        normalize_relative_path(_text(public, "registry", label="outputs.public")),
        normalize_relative_path(_text(public, "feature_dictionary", label="outputs.public")),
        normalize_relative_path(_text(public, "report", label="outputs.public")),
        normalize_relative_path(_text(public, "audit_svg", label="outputs.public")),
    )


def _frozen_files(
    *,
    project_root: Path,
    config: AnomalyHistoryConfig,
    spec: Stage3InputSpec,
    authorization: _FrozenFile,
    anomaly_config_path: Path,
    protocol_document_path: Path,
) -> tuple[_FrozenFile, ...]:
    scientific = _mapping(config.scientific_inputs, label="scientific_inputs")
    environment_reference = _text(
        scientific,
        "environment_lock",
        label="scientific_inputs",
    )
    environment_sha256 = _text(
        scientific,
        "environment_lock_sha256",
        label="scientific_inputs",
    )
    root = project_root.resolve()
    return (
        _FrozenFile(
            anomaly_config_path.resolve().relative_to(root).as_posix(),
            sha256_file(anomaly_config_path),
        ),
        _FrozenFile(
            protocol_document_path.resolve().relative_to(root).as_posix(),
            sha256_file(protocol_document_path),
        ),
        authorization,
        _FrozenFile(environment_reference, environment_sha256),
        _FrozenFile(spec.data_catalog_path, spec.data_catalog_sha256),
        _FrozenFile(spec.observation.path, spec.observation.file_sha256),
        _FrozenFile(spec.report_period.path, spec.report_period.file_sha256),
        _FrozenFile(spec.study_area.path, spec.study_area.sha256),
    )


def build_stage3_plan(
    config_path: str | Path = Path("configs/base.yaml"),
    *,
    physical_core_probe: PhysicalCoreProbe = detect_physical_core_count,
) -> Stage3RunPlan:
    """Validate governance and emit a plan without loading rows or creating outputs."""

    main_path = Path(config_path).resolve(strict=True)
    project_root = project_root_for(main_path).resolve()
    project = load_config(main_path)
    anomaly_path = _safe_project_file(
        project_root,
        ANOMALY_HISTORY_CONFIG_REFERENCE,
        label="stage-3 anomaly-history configuration",
    )
    protocol_path = _safe_project_file(
        project_root,
        ANOMALY_FEATURE_PROTOCOL_REFERENCE,
        label="stage-3 anomaly-feature protocol",
    )
    config = load_anomaly_history_config(anomaly_path)
    spec = stage3_input_spec_from_config(config)
    _validate_project_alignment(project, config, spec)
    authorization = _load_authorization_registry(project_root, config)
    frozen_files = _frozen_files(
        project_root=project_root,
        config=config,
        spec=spec,
        authorization=authorization,
        anomaly_config_path=anomaly_path,
        protocol_document_path=protocol_path,
    )
    planned_inputs = tuple(item.reference for item in frozen_files)
    if len(planned_inputs) != len(set(planned_inputs)):
        raise ValueError("stage-3 frozen input references must be unique")
    return Stage3RunPlan(
        project_root=project_root,
        main_config_path=main_path,
        anomaly_config_path=anomaly_path,
        protocol_document_path=protocol_path,
        project=project,
        config=config,
        input_spec=spec,
        resources=_resource_plan(project, physical_core_probe=physical_core_probe),
        authorization_registry=authorization,
        frozen_files=frozen_files,
        planned_inputs=planned_inputs,
        planned_outputs=_output_references(config),
    )


def _verify_frozen_files(plan: Stage3RunPlan) -> None:
    for item in plan.frozen_files:
        path = _safe_project_file(plan.project_root, item.reference, label=item.reference)
        observed = sha256_file(path)
        if observed != item.sha256:
            raise ExecutionSealError(
                "stage-3 frozen input changed or differs from its protocol identity: "
                f"{item.reference}"
            )


def _identity_payload(
    plan: Stage3RunPlan,
    repository: RepositoryIdentity,
    *,
    dictionary: FeatureDictionary,
    query_grid: Stage3QueryGrid,
) -> dict[str, object]:
    scientific = plan.config.scientific_inputs
    config_sha256 = next(
        item.sha256
        for item in plan.frozen_files
        if item.reference == ANOMALY_HISTORY_CONFIG_REFERENCE
    )
    protocol_sha256 = next(
        item.sha256
        for item in plan.frozen_files
        if item.reference == ANOMALY_FEATURE_PROTOCOL_REFERENCE
    )
    return {
        "schema_version": 1,
        "protocol_version": plan.config.protocol_version,
        "execution_mode": plan.config.execution_mode,
        "protocol_freeze_tag": plan.config.freeze_tag,
        "code_commit": repository.code_commit,
        "feature_dictionary_sha256": dictionary.sha256,
        "grid": {
            "grid_id": query_grid.grid_id,
            "cell_count": query_grid.cell_count,
            "cell_size_km": query_grid.cell_size_km,
        },
        "input_hashes": {
            "protocol_bytes_sha256": protocol_sha256,
            "anomaly_history_config_bytes_sha256": config_sha256,
            "environment_lock_sha256": scientific["environment_lock_sha256"],
            "data_catalog_sha256": plan.input_spec.data_catalog_sha256,
            "anomaly_observation_file_sha256": plan.input_spec.observation.file_sha256,
            "anomaly_observation_content_sha256": (plan.input_spec.observation.content_sha256),
            "anomaly_observation_schema_sha256": plan.input_spec.observation.schema_sha256,
            "anomaly_report_period_file_sha256": plan.input_spec.report_period.file_sha256,
            "anomaly_report_period_content_sha256": (plan.input_spec.report_period.content_sha256),
            "anomaly_report_period_schema_sha256": (plan.input_spec.report_period.schema_sha256),
            "study_area_sha256": plan.input_spec.study_area.sha256,
        },
    }


def _state_row_groups(states: Sequence[AnomalyState]) -> Iterator[pa.Table]:
    for _, grouped in groupby(states, key=lambda state: state.issue_time_utc):
        records = state_records(tuple(grouped))
        yield pa.Table.from_pylist(list(records), schema=ANOMALY_STATE_HISTORY_CONTRACT.schema)


def _bundle_payload_name(
    config: AnomalyHistoryConfig,
    *,
    dataset_name: str,
    bundle_id: str,
) -> str:
    outputs = _mapping(config.outputs, label="outputs")
    layout = _mapping(outputs.get("local_bundle_layout"), label="outputs.local_bundle_layout")
    template = _text(layout, dataset_name, label="outputs.local_bundle_layout")
    rendered = normalize_relative_path(template.format(bundle_id=bundle_id))
    path = Path(rendered)
    if path.parent.as_posix() != bundle_id or path.name not in {
        _STATE_FILENAME,
        _FEATURE_FILENAME,
    }:
        raise ValueError("stage-3 local bundle layout differs from the frozen two-file layout")
    return path.name


def _final_artifact_path(
    artifact: Stage3DatasetArtifact,
    *,
    project_root: Path,
    destination: Path,
    filename: str,
) -> Stage3DatasetArtifact:
    final_path = (destination / filename).resolve()
    reference = final_path.relative_to(project_root.resolve()).as_posix()
    return replace(artifact, path=reference)


def _generation_audit(
    *,
    config: AnomalyHistoryConfig,
    states: Sequence[AnomalyState],
    issue_audits: Sequence[Stage3IssueFeatureAudit],
    query_cell_count: int,
    state_artifact: Stage3DatasetArtifact,
    feature_artifact: Stage3DatasetArtifact,
    dictionary: FeatureDictionary,
    replay: AuditResult,
    synthetic_prefix: SyntheticPrefixAuditResult,
) -> dict[str, object]:
    issue_count = len(issue_audits)
    if issue_count != 205 or state_artifact.row_group_count != 205:
        raise ValueError("stage-3 formal output must contain all 205 actual snapshots")
    if state_artifact.row_count != len(states):
        raise ValueError("stage-3 state-history artifact row count differs from reconstruction")
    if feature_artifact.row_group_count != issue_count:
        raise ValueError("stage-3 feature store must contain one row group per actual issue")
    if feature_artifact.row_count != issue_count * query_cell_count:
        raise ValueError("stage-3 feature-store row count differs from issue-by-cell exposure")
    if any(item.row_count != query_cell_count for item in issue_audits):
        raise ValueError("stage-3 feature issue row count differs from the fixed query grid")
    future_sources = sum(state.max_source_available_at > state.issue_time_utc for state in states)
    if future_sources:
        raise ValueError("stage-3 state history contains a future source reference")
    summaries = tuple(state for state in states if state.state_row_kind == "report_period_summary")
    entities = tuple(state for state in states if state.state_row_kind == "entity_state")
    if len(summaries) != issue_count:
        raise ValueError("stage-3 state history must contain one summary per actual issue")
    summary_dates = {state.issue_report_date for state in summaries}
    missing = _mapping(config.missing_periods, label="missing_periods")
    known_gaps_value = missing.get("known_gaps")
    if not isinstance(known_gaps_value, list | tuple):
        raise ValueError("stage-3 known missing periods must be a sequence")
    known_missing_dates: set[date] = set()
    for index, raw_gap in enumerate(known_gaps_value):
        gap = _mapping(raw_gap, label=f"missing_periods.known_gaps[{index}]")
        expected_date = _text(gap, "expected_report_date", label="known gap")
        try:
            known_missing_dates.add(date.fromisoformat(expected_date))
        except ValueError as exc:
            raise ValueError("known missing report date must use ISO YYYY-MM-DD") from exc
    if len(known_missing_dates) != 2:
        raise ValueError("stage-3 protocol must identify exactly two known missing periods")
    zero_imputed = len(summary_dates & known_missing_dates)
    if zero_imputed:
        raise ValueError("a known missing period was incorrectly materialized as a snapshot")
    complete_entities = tuple(state for state in entities if state.identity_complete)
    temporary_entities = tuple(state for state in entities if not state.identity_complete)
    if sum(item.entity_state_count for item in issue_audits) != len(entities):
        raise ValueError("feature-generation entity counts differ from state-history rows")
    reliability_counts = {
        grade: sum(state.reliability_grade == grade for state in entities)
        for grade in ("high", "cautious", "excluded")
    }
    return {
        "actual_snapshot_count": issue_count,
        "first_report_date": min(summary_dates).isoformat(),
        "last_report_date": max(summary_dates).isoformat(),
        "report_summary_row_count": len(summaries),
        "state_row_count": state_artifact.row_count,
        "complete_entity_state_row_count": len(complete_entities),
        "temporary_entity_state_row_count": len(temporary_entities),
        "reliability_entity_state_counts": reliability_counts,
        "query_cell_count": query_cell_count,
        "feature_row_count": feature_artifact.row_count,
        "state_row_group_count": state_artifact.row_group_count,
        "feature_row_group_count": feature_artifact.row_group_count,
        "entity_state_count": sum(item.entity_state_count for item in issue_audits),
        "spatial_entity_count": sum(item.spatial_entity_count for item in issue_audits),
        "missing_coordinate_count": sum(item.missing_coordinate_count for item in issue_audits),
        "nullable_value_count": sum(item.nullable_value_count for item in issue_audits),
        "null_value_count": sum(item.null_value_count for item in issue_audits),
        "feature_dictionary_sha256": dictionary.sha256,
        "future_source_reference_count": future_sources,
        "known_missing_period_count": len(known_missing_dates),
        "missing_period_zero_imputation_count": zero_imputed,
        "source_duration_feature_use_count": 0,
        "forbidden_source_field_use_count": 0,
        "target_or_earthquake_label_read_count": 0,
        "lineage_replay": asdict(replay),
        "synthetic_prefix": asdict(synthetic_prefix),
        "locked_test": {
            "run": False,
            "target_count": None,
            "target_ids": [],
            "score_ids": [],
            "artifact_ids": [],
            "result": None,
        },
    }


def _dataset_entry(
    manifest: Mapping[str, object],
    name: str,
) -> dict[str, Any]:
    datasets = _mapping(manifest.get("datasets"), label="stage-3 manifest datasets")
    if set(datasets) != {"anomaly_feature_store", "anomaly_state_history"}:
        raise ValueError("stage-3 manifest must contain exactly the two frozen datasets")
    return dict(_mapping(datasets.get(name), label=f"stage-3 dataset {name}"))


def _artifact_from_entry(name: str, entry: Mapping[str, object]) -> Stage3DatasetArtifact:
    sort_keys_value = entry.get("sort_keys")
    if not isinstance(sort_keys_value, list | tuple) or not all(
        isinstance(item, str) for item in sort_keys_value
    ):
        raise ValueError(f"invalid stage-3 sort keys for {name}")

    def integer(key: str) -> int:
        value = entry.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"invalid stage-3 dataset integer: {name}.{key}")
        return value

    return Stage3DatasetArtifact(
        name=name,
        path=_text(entry, "path", label=name),
        row_count=integer("row_count"),
        row_group_count=integer("row_group_count"),
        file_size_bytes=integer("file_size_bytes"),
        file_sha256=_text(entry, "file_sha256", label=name),
        content_sha256=_text(entry, "content_sha256", label=name),
        schema_sha256=_text(entry, "schema_sha256", label=name),
        sort_keys=tuple(sort_keys_value),
    )


def _require_bundle_audit_passed(manifest: Mapping[str, object]) -> dict[str, object]:
    audit = dict(_mapping(manifest.get("audit"), label="stage-3 manifest audit"))
    replay = _mapping(audit.get("lineage_replay"), label="stage-3 lineage replay audit")
    if replay.get("passed") is not True or replay.get("selected_issue_count") != 12:
        raise ValueError("stage-3 bundle does not contain a passed 12-issue replay audit")
    if audit.get("actual_snapshot_count") != 205:
        raise ValueError("stage-3 bundle audit does not cover all 205 actual snapshots")
    if audit.get("future_source_reference_count") != 0:
        raise ValueError("stage-3 bundle audit contains future source references")
    synthetic = _mapping(audit.get("synthetic_prefix"), label="synthetic-prefix audit")
    if synthetic != {
        "passed": True,
        "seed_count": 32,
        "invariant_count": 7,
        "check_count": 224,
        "failure_count": 0,
    }:
        raise ValueError("stage-3 bundle does not contain the passed 32-seed synthetic audit")
    return audit


@dataclass(frozen=True, slots=True)
class Stage3RunTelemetry:
    """Runtime-only measurements excluded from the scientific bundle identity."""

    elapsed_seconds: float
    cpu_seconds: float
    process_peak_working_set_bytes: int | None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.elapsed_seconds)
            or not math.isfinite(self.cpu_seconds)
            or self.elapsed_seconds < 0.0
            or self.cpu_seconds < 0.0
        ):
            raise ValueError("stage-3 runtime measurements must be finite and non-negative")
        if (
            self.process_peak_working_set_bytes is not None
            and self.process_peak_working_set_bytes <= 0
        ):
            raise ValueError("stage-3 peak working set must be positive when available")


@dataclass(frozen=True, slots=True)
class Stage3RunResult:
    """A verified immutable local bundle and its four fixed public projections."""

    repository: RepositoryIdentity
    identity: dict[str, object]
    bundle_id: str
    bundle_path: str
    manifest_sha256: str
    manifest: dict[str, object]
    state_artifact: Stage3DatasetArtifact
    feature_artifact: Stage3DatasetArtifact
    audit: dict[str, object]
    resources: Stage3ResourcePlan
    telemetry: Stage3RunTelemetry
    reused_existing_bundle: bool
    public_publication: PublishedStage3PublicDeliverables | None

    def to_manifest_details(self) -> dict[str, object]:
        if self.public_publication is None:
            raise RuntimeError("stage-3 public fixed delivery has not been confirmed")

        def artifact_summary(artifact: Stage3DatasetArtifact) -> dict[str, object]:
            return {
                "path": artifact.path,
                "row_count": artifact.row_count,
                "row_group_count": artifact.row_group_count,
                "file_size_bytes": artifact.file_size_bytes,
                "file_sha256": artifact.file_sha256,
                "content_sha256": artifact.content_sha256,
                "schema_sha256": artifact.schema_sha256,
            }

        def public_summary(
            path: str,
            publication: object,
        ) -> dict[str, object]:
            sha256 = getattr(publication, "sha256", None)
            created = getattr(publication, "created", None)
            if not isinstance(sha256, str) or not isinstance(created, bool):
                raise RuntimeError("invalid stage-3 fixed-file publication result")
            return {"path": path, "sha256": sha256, "created": created}

        public = self.public_publication

        return {
            "code_commit": self.repository.code_commit,
            "protocol_freeze_tag": self.repository.freeze_tag,
            "identity_sha256": stage3_identity_sha256(self.identity),
            "bundle_id": self.bundle_id,
            "bundle_path": self.bundle_path,
            "manifest_sha256": self.manifest_sha256,
            "local_bundle_confirmed": True,
            "public_delivery_confirmed": True,
            "reused_existing_bundle": self.reused_existing_bundle,
            "datasets": {
                "anomaly_state_history": artifact_summary(self.state_artifact),
                "anomaly_feature_store": artifact_summary(self.feature_artifact),
            },
            "audit": self.audit,
            "resources": self.resources.as_dict(),
            "telemetry": asdict(self.telemetry),
            "public_files": {
                "registry": public_summary(PUBLIC_REGISTRY_PATH, public.registry),
                "report": public_summary(PUBLIC_REPORT_PATH, public.report),
                "audit_svg": public_summary(PUBLIC_AUDIT_SVG_PATH, public.audit_svg),
                "feature_dictionary": public_summary(
                    PUBLIC_DICTIONARY_PATH,
                    public.feature_dictionary,
                ),
            },
            "locked_test": {
                "run": False,
                "target_count": None,
                "target_ids": [],
                "score_ids": [],
                "artifact_ids": [],
                "result": None,
            },
        }


def _verified_result_from_bundle(
    *,
    plan: Stage3RunPlan,
    repository: RepositoryIdentity,
    identity: dict[str, object],
    bundle_id: str,
    destination: Path,
    dictionary: FeatureDictionary,
    telemetry: Stage3RunTelemetry,
    reused: bool,
) -> Stage3RunResult:
    manifest = load_and_verify_stage3_bundle(destination, expected_bundle_id=bundle_id)
    if manifest.get("identity") != identity:
        raise ValueError("stage-3 bundle ID prefix collision or identity mismatch")
    state_entry = _dataset_entry(manifest, "anomaly_state_history")
    feature_entry = _dataset_entry(manifest, "anomaly_feature_store")
    expected_state_path = (
        (destination / _STATE_FILENAME).resolve().relative_to(plan.project_root).as_posix()
    )
    expected_feature_path = (
        (destination / _FEATURE_FILENAME).resolve().relative_to(plan.project_root).as_posix()
    )
    if state_entry.get("path") != expected_state_path:
        raise ValueError("stage-3 state-history manifest path is not the final bundle path")
    if feature_entry.get("path") != expected_feature_path:
        raise ValueError("stage-3 feature-store manifest path is not the final bundle path")
    state_errors = verify_stage3_parquet_artifact(
        project_root=plan.project_root,
        entry=state_entry,
        schema=ANOMALY_STATE_HISTORY_CONTRACT.schema,
    )
    feature_errors = verify_stage3_parquet_artifact(
        project_root=plan.project_root,
        entry=feature_entry,
        schema=build_feature_store_schema(dictionary),
    )
    if state_errors or feature_errors:
        errors = state_errors + feature_errors
        raise ValueError(f"stage-3 bundle dataset verification failed: {errors}")
    audit = _require_bundle_audit_passed(manifest)
    return Stage3RunResult(
        repository=repository,
        identity=identity,
        bundle_id=bundle_id,
        bundle_path=destination.resolve().relative_to(plan.project_root).as_posix(),
        manifest_sha256=sha256_file(destination / STAGE3_MANIFEST_FILENAME),
        manifest=manifest,
        state_artifact=_artifact_from_entry("anomaly_state_history", state_entry),
        feature_artifact=_artifact_from_entry("anomaly_feature_store", feature_entry),
        audit=audit,
        resources=plan.resources,
        telemetry=telemetry,
        reused_existing_bundle=reused,
        public_publication=None,
    )


def _audit_results_from_manifest(
    manifest: Mapping[str, object],
) -> tuple[AuditResult, SyntheticPrefixAuditResult]:
    generation = _mapping(manifest.get("audit"), label="stage-3 manifest audit")
    lineage_mapping = dict(_mapping(generation.get("lineage_replay"), label="lineage replay audit"))
    synthetic_mapping = dict(
        _mapping(generation.get("synthetic_prefix"), label="synthetic-prefix audit")
    )
    try:
        lineage = AuditResult(**cast(dict[str, Any], lineage_mapping))
        synthetic = SyntheticPrefixAuditResult(**cast(dict[str, Any], synthetic_mapping))
    except TypeError as exc:
        raise ValueError("stage-3 sealed audit fields differ from their frozen contracts") from exc
    if asdict(lineage) != lineage_mapping or asdict(synthetic) != synthetic_mapping:
        raise ValueError("stage-3 sealed audit values changed during reconstruction")
    return lineage, synthetic


def run_anomaly_history_stage3(
    config_path: str | Path = Path("configs/base.yaml"),
    *,
    progress: ProgressCallback | None = None,
    git_runner: GitCommandRunner = subprocess_git_runner,
    physical_core_probe: PhysicalCoreProbe = detect_physical_core_count,
    memory_probe: MemoryProbe = process_peak_working_set_bytes,
    dictionary: FeatureDictionary | None = None,
) -> Stage3RunResult:
    """Build, replay-audit, and atomically publish the immutable local stage-3 bundle."""

    plan = build_stage3_plan(config_path, physical_core_probe=physical_core_probe)
    resolved_dictionary = DEFAULT_FEATURE_DICTIONARY if dictionary is None else dictionary
    _notify(progress, "repository_identity:start")
    repository = require_repository_identity(
        plan.project_root,
        freeze_tag=plan.config.freeze_tag,
        runner=git_runner,
    )
    _verify_frozen_files(plan)
    _notify(progress, f"repository_identity:ready:{repository.code_commit}")

    _notify(progress, "synthetic_prefix_audit:start")
    synthetic_prefix = run_synthetic_prefix_audit(plan.config)
    if not synthetic_prefix.passed or synthetic_prefix.failure_count:
        raise ValueError("stage-3 synthetic-prefix audit did not pass")
    _notify(progress, "synthetic_prefix_audit:done:checks=224:failures=0")

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    _notify(progress, "inputs:strict_anomaly_boundary:start")
    loaded = load_stage3_inputs(plan.input_spec, plan.project_root)
    _notify(
        progress,
        "inputs:strict_anomaly_boundary:done:"
        f"observations={loaded.observation_table.num_rows}:"
        f"reports={loaded.report_period_table.num_rows}",
    )
    _notify(progress, "state_history:start")
    observation_records = cast(Sequence[Mapping[str, object]], loaded.observation_table.to_pylist())
    report_records = cast(Sequence[Mapping[str, object]], loaded.report_period_table.to_pylist())
    states = build_anomaly_state_history(
        observation_records,
        report_records,
        expected_report_period_count=plan.config.expected_report_period_count,
    )
    snapshots = build_issue_snapshots(
        states,
        expected_issue_count=plan.config.expected_report_period_count,
    )
    _notify(progress, f"state_history:done:rows={len(states)}:snapshots={len(snapshots)}")

    _notify(progress, "query_grid:start")
    geometry_document = _mapping(
        loaded.study_area_document.get("geometry"),
        label="study area geometry",
    )
    query_grid = build_stage3_query_grid(shape(dict(geometry_document)))
    _notify(progress, f"query_grid:done:cells={query_grid.cell_count}")

    identity = _identity_payload(
        plan,
        repository,
        dictionary=resolved_dictionary,
        query_grid=query_grid,
    )
    bundle_id = stage3_bundle_id(identity)
    outputs = _mapping(plan.config.outputs, label="outputs")
    output_root_reference = _text(
        outputs,
        "local_content_addressed_root",
        label="outputs",
    )
    output_root = _safe_project_directory(plan.project_root, output_root_reference)
    state_filename = _bundle_payload_name(
        plan.config,
        dataset_name="anomaly_state_history",
        bundle_id=bundle_id,
    )
    feature_filename = _bundle_payload_name(
        plan.config,
        dataset_name="anomaly_feature_store",
        bundle_id=bundle_id,
    )
    destination = output_root / bundle_id
    reused = False

    _notify(progress, f"bundle:start:{bundle_id}")
    with stage3_bundle_workspace(output_root, bundle_id=bundle_id) as workspace:
        if not workspace.created:
            reused = True
            _notify(progress, f"bundle:verified_existing:{bundle_id}")
        else:
            _notify(progress, "state_history:write:start")
            raw_state_artifact = write_parquet_row_groups_atomic(
                name="anomaly_state_history",
                row_groups=_state_row_groups(states),
                schema=ANOMALY_STATE_HISTORY_CONTRACT.schema,
                output_path=workspace.path / state_filename,
                project_root=plan.project_root,
                sort_keys=ANOMALY_STATE_HISTORY_CONTRACT.sort_keys,
            )
            state_artifact = _final_artifact_path(
                raw_state_artifact,
                project_root=plan.project_root,
                destination=workspace.destination,
                filename=state_filename,
            )
            _notify(progress, f"state_history:write:done:rows={state_artifact.row_count}")

            engine = Stage3FeatureEngine(
                snapshots,
                query_grid,
                dictionary=resolved_dictionary,
                query_chunk_size=_QUERY_CHUNK_SIZE,
                spatial_workers=plan.resources.spatial_workers,
            )
            issue_audits: list[Stage3IssueFeatureAudit] = []

            def feature_row_groups() -> Iterator[pa.Table]:
                while engine.next_issue_index < len(snapshots):
                    issue = engine.build_next_issue()
                    issue_audits.append(issue.audit)
                    completed = engine.next_issue_index
                    if completed == 1 or completed % 10 == 0 or completed == len(snapshots):
                        _notify(
                            progress,
                            f"feature_store:progress:{completed}/{len(snapshots)}",
                        )
                    yield issue.table

            _notify(progress, "feature_store:write:start")
            raw_feature_artifact = write_parquet_row_groups_atomic(
                name="anomaly_feature_store",
                row_groups=feature_row_groups(),
                schema=engine.schema,
                output_path=workspace.path / feature_filename,
                project_root=plan.project_root,
                sort_keys=FEATURE_STORE_SORT_KEYS,
            )
            feature_artifact = _final_artifact_path(
                raw_feature_artifact,
                project_root=plan.project_root,
                destination=workspace.destination,
                filename=feature_filename,
            )
            _notify(progress, f"feature_store:write:done:rows={feature_artifact.row_count}")

            _notify(progress, "lineage_replay_audit:start")
            replay = run_lineage_replay_audit(
                config=plan.config,
                anomaly_state_history_path=workspace.path / state_filename,
                anomaly_feature_store_path=workspace.path / feature_filename,
                query_grid=query_grid,
                observation_table=loaded.observation_table,
                report_period_table=loaded.report_period_table,
                dictionary=resolved_dictionary,
                query_chunk_size=_QUERY_CHUNK_SIZE,
            )
            if not replay.passed:
                raise ValueError("stage-3 lineage replay audit did not pass")
            audit = _generation_audit(
                config=plan.config,
                states=states,
                issue_audits=issue_audits,
                query_cell_count=query_grid.cell_count,
                state_artifact=state_artifact,
                feature_artifact=feature_artifact,
                dictionary=resolved_dictionary,
                replay=replay,
                synthetic_prefix=synthetic_prefix,
            )
            _notify(progress, "lineage_replay_audit:done:failures=0")

            _notify(progress, "repository_identity:pre_publication_recheck:start")
            rechecked_repository = require_repository_identity(
                plan.project_root,
                freeze_tag=plan.config.freeze_tag,
                runner=git_runner,
            )
            if rechecked_repository != repository:
                raise ExecutionSealError("stage-3 repository identity changed during execution")
            _verify_frozen_files(plan)
            _notify(progress, "repository_identity:pre_publication_recheck:done")

            state_schema = pq.read_schema(workspace.path / state_filename)
            feature_schema = pq.read_schema(workspace.path / feature_filename)
            manifest = build_stage3_manifest(
                bundle_id=bundle_id,
                identity=identity,
                directory=workspace.path,
                payload_paths=[state_filename, feature_filename],
                datasets={
                    "anomaly_state_history": state_artifact.as_manifest_entry(state_schema),
                    "anomaly_feature_store": feature_artifact.as_manifest_entry(feature_schema),
                },
                audit=audit,
            )
            write_stage3_manifest(workspace.path, manifest)

    final_repository = require_repository_identity(
        plan.project_root,
        freeze_tag=plan.config.freeze_tag,
        runner=git_runner,
    )
    if final_repository != repository:
        raise ExecutionSealError("stage-3 repository identity changed before final verification")
    _verify_frozen_files(plan)
    preliminary_telemetry = Stage3RunTelemetry(
        elapsed_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        process_peak_working_set_bytes=None,
    )
    result = _verified_result_from_bundle(
        plan=plan,
        repository=repository,
        identity=identity,
        bundle_id=bundle_id,
        destination=destination,
        dictionary=resolved_dictionary,
        telemetry=preliminary_telemetry,
        reused=reused,
    )
    _notify(progress, f"bundle:done:{bundle_id}")

    lineage_audit, sealed_synthetic_audit = _audit_results_from_manifest(result.manifest)
    raw_input_hashes = _mapping(identity.get("input_hashes"), label="identity.input_hashes")
    input_hashes = cast(dict[str, str], dict(raw_input_hashes))
    _notify(progress, "public_fixed_delivery:start")
    public_deliverables = build_stage3_public_deliverables(
        manifest=result.manifest,
        dictionary=resolved_dictionary,
        audit=lineage_audit,
        synthetic=sealed_synthetic_audit,
        code_commit=repository.code_commit,
        input_hashes=input_hashes,
    )
    public_publication = publish_stage3_public_deliverables(
        plan.project_root,
        public_deliverables,
    )
    _notify(progress, "public_fixed_delivery:done:files=4")
    telemetry = Stage3RunTelemetry(
        elapsed_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        process_peak_working_set_bytes=memory_probe(),
    )
    return replace(
        result,
        telemetry=telemetry,
        public_publication=public_publication,
    )


__all__ = [
    "ANOMALY_FEATURE_PROTOCOL_REFERENCE",
    "ANOMALY_HISTORY_CONFIG_REFERENCE",
    "MemoryProbe",
    "PhysicalCoreProbe",
    "ProgressCallback",
    "Stage3ResourcePlan",
    "Stage3RunPlan",
    "Stage3RunResult",
    "Stage3RunTelemetry",
    "build_stage3_plan",
    "run_anomaly_history_stage3",
]
