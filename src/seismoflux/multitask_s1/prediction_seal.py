"""Minimal immutable prediction seals for the four S1 development folds.

This module deliberately does not score predictions.  It only writes canonical,
exclusive prediction bundles and verifies that the complete four-fold chain still
matches before granting a scoring entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, cast
from zipfile import BadZipFile

import numpy as np
import yaml
from numpy.lib.npyio import NpzFile

from seismoflux.d1_replay.spatial import build_d1_spatial_domain_from_bytes
from seismoflux.multitask_s0 import verify_authoritative_catalog_identity
from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
    load_development_contract,
)
from seismoflux.multitask_s1.development_predict import (
    LOCATION_MODEL_IDS,
    MAGNITUDE_MODEL_IDS,
    NB2_REASON_CODES,
    NB2_STATUS_CODES,
    PREDICTION_ARRAY_SCHEMA_VERSION,
    TIME_BANDS,
    frozen_fold_prediction_npz_schema,
    validate_frozen_fold_prediction_npz_arrays,
)
from seismoflux.multitask_s1.location import (
    FROZEN_KDE_BANDWIDTHS_KM,
    FROZEN_R30_ALPHA_CANDIDATES,
    FROZEN_REGIONAL_TAU_YEARS,
    EarlierInnerBoundary,
    select_kde_bandwidth_one_se,
    select_recent_alpha,
    select_regional_tau,
)
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_RELATIVE_PATH,
    EXPECTED_25KM_CELL_COUNT,
    EXPECTED_25KM_GRID_ID,
    EXPECTED_STUDY_AREA_SHA256,
    EXPECTED_TOTAL_AREA_KM2,
    STUDY_AREA_RELATIVE_PATH,
)

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_OID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_PROHIBITED_ROLE_TOKENS = frozenset({"holdout", "audit", "locked"})
_RUN_CONTRACT_RELATIVE_PATH: Final = Path("configs/multitask_s1_development_run.yaml")
_PARENT_CONTRACT_RELATIVE_PATH: Final = Path("configs/multitask_s1_development_contract.yaml")
_ISSUE_LEDGER_RELATIVE_PATH: Final = Path(
    "outputs/multitask_s0/s0_score_blind_20260901/issue_maturity_ledger.csv"
)
_ALWAYS_PROHIBITED_MANIFEST_CONCEPTS = (
    "target",
    "truth",
    "observed",
    "hit",
    "recall",
    "label",
    "outcome",
    "informationgain",
    "infogain",
    "信息增益",
)
_FOLD_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "role",
        "fold_id",
        "input_identities",
        "prediction_manifest",
        "prediction_artifacts",
    }
)
_MASTER_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "role",
        "input_identities",
        "ordered_fold_predictions",
        "prediction_phase_artifacts",
    }
)
_FOLD_REFERENCE_KEYS = frozenset({"fold_id", "relative_path", "sha256", "size_bytes"})
_PREDICTION_ARTIFACT_REFERENCE_KEYS = frozenset({"relative_path", "sha256", "size_bytes", "schema"})
_ARRAY_SCHEMA_KEYS = frozenset({"shape", "dtype"})
_PHASE_REFERENCE_KEYS = frozenset({"relative_path", "sha256", "size_bytes"})
_RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_contract_id",
        "stage",
        "role",
        "git_commit_oid",
        "input_identities",
        "fold_ids",
        "maximum_fold_workers",
        "numerical_threads_per_worker",
        "outer_targets_constructed",
        "model_scores_read",
        "locked_test_run",
    }
)
_PARAMETER_SELECTION_KEYS = frozenset(
    {"schema_version", "record_type", "run_contract_id", "role", "folds"}
)
_RUN_CONTRACT_ID: Final = "multitask-s1-c0-all-m4-screen-v1"
_EXPECTED_PRIMARY_COUNT: Final = 99
_EXPECTED_WEEKLY_COUNT: Final[Mapping[str, int]] = {
    "C_DEV_2000_2004": 261,
    "C_DEV_2005_2009": 261,
    "C_DEV_2010_2014": 260,
    "C_DEV_2015_2019": 261,
}


class PredictionSealError(ValueError):
    """Raised when a development prediction seal fails closed."""


class PredictionSealExistsError(FileExistsError):
    """Raised when an immutable prediction artifact already exists."""


@dataclass(frozen=True)
class PredictionInputIdentities:
    """Recomputed identities of every input that defines one prediction run.

    ``run_contract_sha256`` is the SHA-256 of the latest S1-C development-run YAML,
    not merely its S1-B parent contract.  The run YAML is responsible for binding
    that parent and the complete score-blind prediction protocol.
    """

    run_contract_sha256: str
    parent_contract_sha256: str
    catalog_sha256: str
    study_sha256: str
    grid_sha256: str
    issue_ledger_sha256: str
    code_sha256: str
    git_commit_oid: str

    def __post_init__(self) -> None:
        for field_name, value in self.as_mapping().items():
            if field_name == "git_commit_oid":
                if _GIT_OID_PATTERN.fullmatch(value) is None:
                    raise PredictionSealError("git_commit_oid must be a lowercase Git OID")
                continue
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise PredictionSealError(f"{field_name} must be a lowercase SHA-256")

    def as_mapping(self) -> dict[str, str]:
        """Return the canonical input-identity mapping."""

        return {
            "run_contract_sha256": self.run_contract_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "catalog_sha256": self.catalog_sha256,
            "study_sha256": self.study_sha256,
            "grid_sha256": self.grid_sha256,
            "issue_ledger_sha256": self.issue_ledger_sha256,
            "code_sha256": self.code_sha256,
            "git_commit_oid": self.git_commit_oid,
        }


@dataclass(frozen=True)
class SealedArtifact:
    """Filesystem identity of one canonical O_EXCL JSON artifact."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PredictionArtifactInput:
    """One pre-existing prediction payload and its declared array schema.

    ``path`` may be absolute or relative to ``output_root``.  The seal stores only
    its canonical relative path.  NPZ payloads require a non-empty schema mapping
    every array name to exactly ``{"shape": [...], "dtype": "..."}``.
    """

    path: str | Path
    schema: Mapping[str, object] | None = None


@dataclass(frozen=True)
class DevelopmentScoringAuthorization:
    """Evidence that all four development predictions are sealed and unchanged."""

    seal: SealedArtifact
    input_identities: PredictionInputIdentities
    ordered_fold_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _SelectionSnapshot:
    horizon_days: int
    regional_tau_years: float
    kde_bandwidth_km: float
    recent_alpha: float
    time_qualifications: tuple[dict[str, JsonValue], ...]


def _tokens(value: str) -> tuple[str, ...]:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return tuple(
        token for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", camel_split.lower()) if token
    )


def _collapsed(value: str) -> str:
    return "".join(_tokens(value))


def _normalise_json(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PredictionSealError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise PredictionSealError(f"{path} contains a non-string JSON key")
            result[raw_key] = _normalise_json(raw_value, path=f"{path}.{raw_key}")
        return result
    if isinstance(value, list | tuple):
        return [_normalise_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise PredictionSealError(f"{path} is not a JSON value")


def canonical_json_bytes(value: object) -> bytes:
    """Return one deterministic UTF-8 JSON representation without a trailing newline."""

    normalised = _normalise_json(value, path="$")
    return json.dumps(
        normalised,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_explicit_root(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, str | Path):
        raise PredictionSealError(f"{label} must be supplied explicitly")
    try:
        path = Path(value).resolve(strict=True)
    except OSError as exc:
        raise PredictionSealError(f"{label} cannot be resolved") from exc
    if not path.is_dir():
        raise PredictionSealError(f"{label} must be an existing directory")
    return path


def _scoped_identity_file(root: Path, relative_path: Path, *, label: str) -> Path:
    candidate = root / relative_path
    if candidate.is_symlink():
        raise PredictionSealError(f"{label} cannot be a symlink")
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PredictionSealError(f"{label} is missing or escaped its explicit root") from exc
    if not path.is_file():
        raise PredictionSealError(f"{label} must be a regular file")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PredictionSealError(f"identity file cannot be read: {path}") from exc
    return digest.hexdigest()


def _required_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PredictionSealError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _git_commit_identity(project_root: Path) -> tuple[str, str]:
    def run_git(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                check=False,
                capture_output=True,
            )
        except OSError as exc:
            raise PredictionSealError("Git cannot be executed for code identity") from exc
        if completed.returncode != 0:
            raise PredictionSealError("Git code identity command failed")
        return completed.stdout

    top_level_raw = run_git("rev-parse", "--show-toplevel").decode("utf-8").strip()
    try:
        top_level = Path(top_level_raw).resolve(strict=True)
    except OSError as exc:
        raise PredictionSealError("Git top-level path cannot be resolved") from exc
    if top_level != project_root:
        raise PredictionSealError("project_root must be the exact Git worktree root")
    if run_git("status", "--porcelain=v1", "--untracked-files=all").strip():
        raise PredictionSealError("Git worktree must be completely clean before sealing")
    commit_oid = run_git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if _GIT_OID_PATTERN.fullmatch(commit_oid) is None:
        raise PredictionSealError("Git returned a non-canonical commit OID")
    commit_payload = run_git("cat-file", "commit", commit_oid)
    preimage = b"commit " + str(len(commit_payload)).encode("ascii") + b"\0" + commit_payload
    return commit_oid, _sha256_bytes(preimage)


def recompute_prediction_input_identities(
    *, project_root: str | Path, data_root: str | Path
) -> PredictionInputIdentities:
    """Recompute the frozen S1-C input and Git identities from explicit roots."""

    project = _resolve_explicit_root(project_root, label="project_root")
    data = _resolve_explicit_root(data_root, label="data_root")
    run_path = _scoped_identity_file(
        project, _RUN_CONTRACT_RELATIVE_PATH, label="S1-C run contract"
    )
    run_bytes = run_path.read_bytes()
    try:
        run_document = yaml.safe_load(run_bytes)
    except yaml.YAMLError as exc:
        raise PredictionSealError("S1-C run contract is not valid YAML") from exc
    run = _required_mapping(run_document, label="S1-C run contract")
    if run.get("run_contract_id") != "multitask-s1-c0-all-m4-screen-v1":
        raise PredictionSealError("S1-C run_contract_id changed")

    parent = _required_mapping(run.get("parent_contract"), label="parent_contract")
    if parent.get("path") != _PARENT_CONTRACT_RELATIVE_PATH.as_posix():
        raise PredictionSealError("parent contract path changed")
    parent_path = _scoped_identity_file(
        project, _PARENT_CONTRACT_RELATIVE_PATH, label="S1-B parent contract"
    )
    parent_sha256 = _sha256_file(parent_path)
    if parent.get("sha256") != parent_sha256:
        raise PredictionSealError("S1-B parent contract SHA-256 changed")
    try:
        load_development_contract(parent_path, project_root=project)
    except (OSError, ValueError) as exc:
        raise PredictionSealError("S1-B parent contract verification failed") from exc

    sources = _required_mapping(run.get("input_identities"), label="input_identities")
    catalog_entry = _required_mapping(
        sources.get("authoritative_catalog"), label="authoritative_catalog"
    )
    if (
        catalog_entry.get("root") != "data_root"
        or catalog_entry.get("path") != CATALOG_RELATIVE_PATH.as_posix()
    ):
        raise PredictionSealError("authoritative catalog path binding changed")
    catalog_path = _scoped_identity_file(data, CATALOG_RELATIVE_PATH, label="authoritative catalog")
    try:
        catalog_identity = verify_authoritative_catalog_identity(catalog_path)
    except (OSError, ValueError) as exc:
        raise PredictionSealError("authoritative catalog verification failed") from exc
    catalog_sha256 = str(catalog_identity.get("file_sha256"))
    if catalog_entry.get("sha256") != catalog_sha256:
        raise PredictionSealError("run contract catalog SHA-256 changed")

    study_entry = _required_mapping(sources.get("study_area"), label="study_area")
    if (
        study_entry.get("root") != "data_root"
        or study_entry.get("path") != STUDY_AREA_RELATIVE_PATH.as_posix()
    ):
        raise PredictionSealError("study-area path binding changed")
    study_path = _scoped_identity_file(data, STUDY_AREA_RELATIVE_PATH, label="study area")
    study_bytes = study_path.read_bytes()
    study_sha256 = _sha256_bytes(study_bytes)
    if study_sha256 != EXPECTED_STUDY_AREA_SHA256 or study_entry.get("sha256") != study_sha256:
        raise PredictionSealError("study-area SHA-256 changed")

    grid_entry = _required_mapping(sources.get("operational_grid"), label="operational_grid")
    try:
        domain = build_d1_spatial_domain_from_bytes(study_bytes)
    except (OSError, ValueError) as exc:
        raise PredictionSealError("operational grid rebuild failed") from exc
    grid = domain.operational_grid
    total_area = math.fsum(float(value) for value in grid.clipped_area_km2)
    if (
        grid_entry.get("cell_size_km") != 25.0
        or grid_entry.get("grid_id") != EXPECTED_25KM_GRID_ID
        or grid_entry.get("cell_count") != EXPECTED_25KM_CELL_COUNT
        or not math.isclose(
            float(grid_entry.get("exact_area_km2", float("nan"))),
            EXPECTED_TOTAL_AREA_KM2,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or grid.grid_id != EXPECTED_25KM_GRID_ID
        or grid.cell_count != EXPECTED_25KM_CELL_COUNT
        or not math.isclose(
            total_area,
            EXPECTED_TOTAL_AREA_KM2,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise PredictionSealError("operational 25 km grid identity changed")

    issue_entry = _required_mapping(sources.get("issue_ledger"), label="issue_ledger")
    if (
        issue_entry.get("root") != "project_root"
        or issue_entry.get("path") != _ISSUE_LEDGER_RELATIVE_PATH.as_posix()
    ):
        raise PredictionSealError("issue-ledger path binding changed")
    issue_path = _scoped_identity_file(
        project, _ISSUE_LEDGER_RELATIVE_PATH, label="issue maturity ledger"
    )
    issue_sha256 = _sha256_file(issue_path)
    if issue_entry.get("sha256") != issue_sha256:
        raise PredictionSealError("issue-ledger SHA-256 changed")

    commit_oid, code_sha256 = _git_commit_identity(project)
    return PredictionInputIdentities(
        run_contract_sha256=_sha256_bytes(run_bytes),
        parent_contract_sha256=parent_sha256,
        catalog_sha256=catalog_sha256,
        study_sha256=study_sha256,
        grid_sha256=grid.grid_id,
        issue_ledger_sha256=issue_sha256,
        code_sha256=code_sha256,
        git_commit_oid=commit_oid,
    )


def _audit_manifest_keys(value: JsonValue, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_collapsed = _collapsed(key)
            path_tokens = tuple(token for part in (*path, key) for token in _tokens(part))
            if "outer" in key_collapsed:
                raise PredictionSealError(
                    f"prediction manifest contains an outer-fold field: {'.'.join((*path, key))}"
                )
            inner_selection = "inner" in path_tokens and "selection" in path_tokens
            if (
                any(concept in key_collapsed for concept in _ALWAYS_PROHIBITED_MANIFEST_CONCEPTS)
                and not inner_selection
            ):
                raise PredictionSealError(
                    f"prediction manifest contains target-derived field: {'.'.join((*path, key))}"
                )
            if "score" in key_collapsed and not inner_selection:
                raise PredictionSealError(
                    "prediction manifest score fields are allowed only under explicit "
                    f"inner parameter selection: {'.'.join((*path, key))}"
                )
            _audit_manifest_keys(child, path=(*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _audit_manifest_keys(child, path=(*path, f"[{index}]"))
    elif isinstance(value, str):
        value_collapsed = _collapsed(value)
        path_tokens = tuple(token for part in path for token in _tokens(part))
        if "outer" in value_collapsed:
            raise PredictionSealError(
                f"prediction manifest contains an outer-fold value: {'.'.join(path)}"
            )
        inner_selection = "inner" in path_tokens and "selection" in path_tokens
        if (
            any(concept in value_collapsed for concept in _ALWAYS_PROHIBITED_MANIFEST_CONCEPTS)
            and not inner_selection
        ):
            raise PredictionSealError(
                f"prediction manifest contains target-derived value: {'.'.join(path)}"
            )
        if "score" in value_collapsed and not inner_selection:
            raise PredictionSealError(
                "prediction manifest score values are allowed only under explicit "
                f"inner parameter selection: {'.'.join(path)}"
            )


def _audit_no_prohibited_roles(value: JsonValue, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _PROHIBITED_ROLE_TOKENS.intersection(_tokens(key)):
                raise PredictionSealError(f"{path} contains a prohibited role name: {key}")
            _audit_no_prohibited_roles(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _audit_no_prohibited_roles(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and _PROHIBITED_ROLE_TOKENS.intersection(_tokens(value)):
        raise PredictionSealError(f"{path} contains a prohibited role value")


def _validate_output_root(root: Path) -> None:
    if _PROHIBITED_ROLE_TOKENS.intersection(_tokens(root.name)):
        raise PredictionSealError("prediction output root uses a prohibited role name")


def _require_configured_prediction_root(project_root: str | Path, output_root: str | Path) -> Path:
    project = _resolve_explicit_root(project_root, label="project_root")
    run_path = project / _RUN_CONTRACT_RELATIVE_PATH
    try:
        raw = yaml.safe_load(run_path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise PredictionSealError("development-run contract cannot be read") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("outputs"), Mapping):
        raise PredictionSealError("development-run outputs are missing")
    outputs = cast(Mapping[str, object], raw["outputs"])
    expected_values = {
        "root": "outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2",
        "prediction_root": ("outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2/prediction_phase"),
        "score_root": "outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2/score_phase",
        "phase_roots_must_be_siblings_and_nonoverlapping": True,
    }
    if any(outputs.get(key) != value for key, value in expected_values.items()):
        raise PredictionSealError("development-run output roots differ from the frozen protocol")
    relative = Path(cast(str, outputs["prediction_root"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PredictionSealError("configured prediction_root must be project-relative")
    expected = (project / relative).resolve()
    actual = Path(output_root).resolve()
    if actual != expected:
        raise PredictionSealError("output_root is not the configured prediction_phase")
    _validate_output_root(actual)
    return actual


def _allowed_output_directories() -> frozenset[str]:
    return frozenset({"folds"} | {f"folds/{fold_id}" for fold_id in DEVELOPMENT_FOLD_IDS})


def _allowed_output_files() -> frozenset[str]:
    return frozenset(
        {
            "run_manifest.json",
            "parameter_selection.json",
            "four_fold_prediction_seal.json",
        }
        | {f"folds/{fold_id}/predictions.npz" for fold_id in DEVELOPMENT_FOLD_IDS}
        | {f"folds/{fold_id}/prediction_bundle.json" for fold_id in DEVELOPMENT_FOLD_IDS}
    )


def _validate_output_tree(root: Path) -> None:
    if not root.exists():
        raise PredictionSealError("prediction output root does not exist")
    if root.is_symlink():
        raise PredictionSealError("prediction output root cannot be a symlink")
    allowed_directories = _allowed_output_directories()
    allowed_files = _allowed_output_files()
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise PredictionSealError(f"symlinks are prohibited in output_root: {relative}")
        if entry.is_dir():
            if relative not in allowed_directories:
                raise PredictionSealError(f"unexpected output directory: {relative}")
        elif entry.is_file():
            if relative not in allowed_files:
                raise PredictionSealError(f"unexpected output file: {relative}")
        else:
            raise PredictionSealError(f"non-regular output entry is prohibited: {relative}")


def frozen_fold_prediction_manifest(fold_id: str) -> dict[str, JsonValue]:
    """Return the only fold manifest accepted by the S1-C0 prediction seal."""

    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise PredictionSealError("only the four frozen development folds have a manifest")
    return {
        "schema_version": 1,
        "run_contract_id": _RUN_CONTRACT_ID,
        "role": "development_prediction_only",
        "fold_id": fold_id,
        "prediction_path": f"folds/{fold_id}/predictions.npz",
        "prediction_array_schema_version": PREDICTION_ARRAY_SCHEMA_VERSION,
        "primary_row_count": _EXPECTED_PRIMARY_COUNT,
        "weekly_row_count": _EXPECTED_WEEKLY_COUNT[fold_id],
        "cell_count": EXPECTED_25KM_CELL_COUNT,
        "model_axes": {
            "location": list(LOCATION_MODEL_IDS),
            "time": list(TIME_BANDS),
            "magnitude": list(MAGNITUDE_MODEL_IDS),
        },
    }


def _normalise_prediction_schema(
    value: object, *, artifact_path: Path, fold_id: str
) -> dict[str, JsonValue]:
    if artifact_path.suffix.lower() != ".npz":
        raise PredictionSealError("the sole prediction payload must be an NPZ file")
    if value is None:
        raise PredictionSealError("NPZ prediction artifacts require the frozen array schema")
    normalised = _normalise_json(value, path="$.prediction_artifact.schema")
    if not isinstance(normalised, dict) or not normalised:
        raise PredictionSealError("prediction artifact schema must be a non-empty JSON object")
    trusted = _normalise_json(
        frozen_fold_prediction_npz_schema(fold_id, cell_count=EXPECTED_25KM_CELL_COUNT),
        path="$.frozen_prediction_schema",
    )
    if not isinstance(trusted, dict) or normalised != trusted:
        raise PredictionSealError("caller schema differs from the frozen fold prediction schema")
    for array_name, raw_specification in trusted.items():
        if not isinstance(raw_specification, dict) or set(raw_specification) != _ARRAY_SCHEMA_KEYS:
            raise PredictionSealError(f"frozen prediction schema is invalid: {array_name}")
        raw_shape = raw_specification.get("shape")
        raw_dtype = raw_specification.get("dtype")
        if not isinstance(raw_shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in raw_shape
        ):
            raise PredictionSealError(
                f"frozen prediction array has a zero/invalid dimension: {array_name}"
            )
        if not isinstance(raw_dtype, str) or not raw_dtype.strip():
            raise PredictionSealError(f"frozen prediction dtype is invalid: {array_name}")
        try:
            dtype = np.dtype(raw_dtype)
        except TypeError as exc:
            raise PredictionSealError(f"frozen prediction dtype is invalid: {array_name}") from exc
        if dtype.hasobject:
            raise PredictionSealError(f"object/pickle arrays are prohibited: {array_name}")
    return trusted


def _verify_actual_npz_schema(
    path: Path, declared: Mapping[str, JsonValue], *, fold_id: str
) -> None:
    try:
        loaded = np.load(path, allow_pickle=False)
        if not isinstance(loaded, NpzFile):
            raise PredictionSealError(f"prediction payload is not an NPZ archive: {path}")
        with loaded as archive:
            names = list(archive.files)
            if len(names) != len(set(names)):
                raise PredictionSealError(f"prediction NPZ contains duplicate arrays: {path}")
            if set(names) != set(declared):
                raise PredictionSealError(
                    f"prediction NPZ arrays do not match the declared schema: {path}"
                )
            arrays: dict[str, object] = {}
            for array_name in names:
                array = np.asarray(archive[array_name])
                specification = declared[array_name]
                if not isinstance(specification, dict):  # already normalised, kept fail-closed
                    raise PredictionSealError(
                        f"prediction NPZ schema is invalid for array: {array_name}"
                    )
                if array.dtype.hasobject:
                    raise PredictionSealError(
                        f"object/pickle prediction arrays are prohibited: {array_name}"
                    )
                if specification.get("shape") != list(array.shape) or specification.get(
                    "dtype"
                ) != str(array.dtype):
                    raise PredictionSealError(
                        "prediction NPZ shape or dtype does not match the declared schema: "
                        f"{array_name}"
                    )
                arrays[array_name] = array
            validate_frozen_fold_prediction_npz_arrays(
                fold_id,
                arrays,
                cell_count=EXPECTED_25KM_CELL_COUNT,
            )
    except PredictionSealError:
        raise
    except (BadZipFile, EOFError, OSError, ValueError) as exc:
        raise PredictionSealError(
            f"prediction payload is not a safe, readable NPZ archive: {path}"
        ) from exc


def _canonical_prediction_artifact_path(
    root: Path, raw_path: str | Path, *, fold_id: str
) -> tuple[Path, str]:
    if not root.exists():
        raise PredictionSealError("prediction output root must exist before sealing payloads")
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        raise PredictionSealError(f"prediction output root cannot be resolved: {root}") from exc
    expected_relative = Path("folds") / fold_id / "predictions.npz"
    expected = root / expected_relative
    supplied = Path(raw_path)
    candidate = supplied if supplied.is_absolute() else root / supplied
    for component in (root, root / "folds", root / "folds" / fold_id, expected):
        if component.is_symlink():
            raise PredictionSealError("prediction payload path cannot contain a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PredictionSealError(f"required prediction payload is missing: {candidate}") from exc
    except OSError as exc:
        raise PredictionSealError(f"prediction payload cannot be resolved: {candidate}") from exc
    try:
        expected_resolved = expected.resolve(strict=True)
        relative = resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise PredictionSealError("prediction payload escapes output_root or wrong fold") from exc
    if resolved != expected_resolved or relative.as_posix() != expected_relative.as_posix():
        raise PredictionSealError(f"fold payload must be exactly {expected_relative.as_posix()}")
    if not resolved.is_file():
        raise PredictionSealError(
            f"prediction payload is not a regular file: {relative.as_posix()}"
        )
    return resolved, relative.as_posix()


def _read_prediction_payload_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            size = 0
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
        path_after = path.stat()
    except FileNotFoundError as exc:
        raise PredictionSealError(f"required prediction payload is missing: {path}") from exc
    except OSError as exc:
        raise PredictionSealError(f"prediction payload cannot be read: {path}") from exc
    if (
        size <= 0
        or before.st_size != size
        or after.st_size != size
        or path_after.st_size != size
        or before.st_mtime_ns != after.st_mtime_ns
        or after.st_mtime_ns != path_after.st_mtime_ns
    ):
        raise PredictionSealError(f"prediction payload is empty or changed while read: {path}")
    return digest.hexdigest(), size


def _build_prediction_artifact_reference(
    root: Path, artifact: PredictionArtifactInput, *, fold_id: str
) -> dict[str, JsonValue]:
    resolved, relative_path = _canonical_prediction_artifact_path(
        root, artifact.path, fold_id=fold_id
    )
    schema = _normalise_prediction_schema(artifact.schema, artifact_path=resolved, fold_id=fold_id)
    _verify_actual_npz_schema(resolved, schema, fold_id=fold_id)
    sha256, size_bytes = _read_prediction_payload_identity(resolved)
    return {
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "schema": schema,
    }


def _verify_prediction_artifact_reference(
    root: Path, raw_reference: JsonValue, *, fold_id: str
) -> dict[str, JsonValue]:
    if (
        not isinstance(raw_reference, dict)
        or set(raw_reference) != _PREDICTION_ARTIFACT_REFERENCE_KEYS
    ):
        raise PredictionSealError("prediction payload reference schema changed")
    raw_relative_path = raw_reference.get("relative_path")
    if not isinstance(raw_relative_path, str) or not raw_relative_path:
        raise PredictionSealError("prediction payload relative path is invalid")
    schema_value = raw_reference.get("schema")
    schema_mapping: Mapping[str, object] | None
    if schema_value is None:
        schema_mapping = None
    elif isinstance(schema_value, dict):
        schema_mapping = schema_value
    else:
        raise PredictionSealError("prediction payload schema is invalid")
    actual = _build_prediction_artifact_reference(
        root,
        PredictionArtifactInput(path=raw_relative_path, schema=schema_mapping),
        fold_id=fold_id,
    )
    if raw_reference != actual:
        raise PredictionSealError(
            f"prediction payload hash, size, path, or schema mismatch: {raw_relative_path}"
        )
    return actual


def _write_exclusive_json(path: Path, payload: object) -> SealedArtifact:
    data = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | cast(int, getattr(os, "O_BINARY", 0))
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise PredictionSealExistsError(
            f"immutable prediction artifact already exists: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial exclusive claim remains deliberately visible and blocks silent replacement.
        raise
    return SealedArtifact(path=path, sha256=_sha256_bytes(data), size_bytes=len(data))


def _read_canonical_mapping(path: Path) -> tuple[dict[str, JsonValue], SealedArtifact]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise PredictionSealError(f"required prediction artifact is missing: {path}") from exc
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionSealError(f"prediction artifact is not valid UTF-8 JSON: {path}") from exc
    normalised = _normalise_json(parsed, path="$")
    if not isinstance(normalised, dict):
        raise PredictionSealError(f"prediction artifact must be a JSON object: {path}")
    if canonical_json_bytes(normalised) != data:
        raise PredictionSealError(f"prediction artifact is not canonical JSON: {path}")
    return normalised, SealedArtifact(path, _sha256_bytes(data), len(data))


def _fold_path(root: Path, fold_id: str) -> Path:
    return root / "folds" / fold_id / "prediction_bundle.json"


def _master_path(root: Path) -> Path:
    return root / "four_fold_prediction_seal.json"


def _prediction_phase_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "run_manifest.json",
        root / "parameter_selection.json",
    )


def _exact_mapping(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PredictionSealError(f"{label} schema changed")
    return cast(Mapping[str, object], value)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PredictionSealError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PredictionSealError(f"{label} must be finite numeric")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PredictionSealError(f"{label} must be a nonnegative integer")
    return value


def _candidate_scores(
    value: object, axis: tuple[float, ...], label: str
) -> dict[float, tuple[float, float, float]]:
    if not isinstance(value, list) or len(value) != len(axis):
        raise PredictionSealError(f"{label} candidate axis changed")
    result: dict[float, tuple[float, float, float]] = {}
    for expected, raw in zip(axis, value, strict=True):
        row = _exact_mapping(raw, {"parameter_value", "inner_block_mean_log_density"}, label)
        parameter = _finite_float(row["parameter_value"], f"{label}.parameter_value")
        scores = row["inner_block_mean_log_density"]
        if parameter != expected or not isinstance(scores, list) or len(scores) != 3:
            raise PredictionSealError(f"{label} candidate axis or block count changed")
        result[parameter] = cast(
            tuple[float, float, float],
            tuple(_finite_float(item, f"{label}.score") for item in scores),
        )
    return result


def _mean_scores(
    values: Mapping[float, tuple[float, float, float]],
) -> dict[float, float]:
    return {key: math.fsum(scores) / 3.0 for key, scores in values.items()}


def _parse_time_qualification(value: object, label: str) -> dict[str, JsonValue]:
    keys = {
        "status",
        "reason",
        "historical_block_count",
        "sample_mean_count",
        "sample_variance_count",
        "dispersion_k",
        "observed_information_k",
        "standard_error_k",
    }
    raw = _exact_mapping(value, keys, label)
    status = raw["status"]
    reason = raw["reason"]
    if status not in NB2_STATUS_CODES or reason not in NB2_REASON_CODES:
        raise PredictionSealError(f"{label} status or reason is invalid")
    expected_status = (0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 1)[NB2_REASON_CODES.index(reason)]
    if NB2_STATUS_CODES.index(status) != expected_status:
        raise PredictionSealError(f"{label} status and reason disagree")
    block_count = _nonnegative_int(raw["historical_block_count"], label)
    mean = _finite_float(raw["sample_mean_count"], label)
    variance_raw = raw["sample_variance_count"]
    variance = None if variance_raw is None else _finite_float(variance_raw, label)
    fitted: list[float | None] = []
    for key in ("dispersion_k", "observed_information_k", "standard_error_k"):
        item = raw[key]
        fitted.append(None if item is None else _finite_float(item, f"{label}.{key}"))
    if block_count < 0 or mean < 0.0 or (variance is not None and variance < 0.0):
        raise PredictionSealError(f"{label} contains invalid count evidence")
    if expected_status == 2:
        if any(item is None or item <= 0.0 for item in fitted):
            raise PredictionSealError(f"{label} evaluable fit is incomplete")
    elif any(item is not None for item in fitted):
        raise PredictionSealError(f"{label} non-evaluable fit leaked fitted parameters")
    return cast(dict[str, JsonValue], dict(raw))


def _parse_parameter_selection(
    selection: Mapping[str, JsonValue],
) -> dict[str, tuple[_SelectionSnapshot, ...]]:
    folds = cast(dict[str, JsonValue], selection["folds"])
    parsed: dict[str, tuple[_SelectionSnapshot, ...]] = {}
    for fold_id in DEVELOPMENT_FOLD_IDS:
        entries = folds[fold_id]
        if not isinstance(entries, list) or len(entries) != len(HORIZONS_DAYS):
            raise PredictionSealError(f"parameter selection horizons changed for {fold_id}")
        snapshots: list[_SelectionSnapshot] = []
        for horizon, raw_entry in zip(HORIZONS_DAYS, entries, strict=True):
            entry = _exact_mapping(raw_entry, {"horizon_days", "inner_evidence"}, "selection entry")
            if entry["horizon_days"] != horizon:
                raise PredictionSealError("parameter selection horizon order changed")
            evidence = _exact_mapping(
                entry["inner_evidence"], {"location", "time"}, "inner evidence"
            )
            location = _exact_mapping(
                evidence["location"],
                {
                    "latest_inner_target_end_us",
                    "evaluation_start_boundary_us",
                    "inner_block_event_counts",
                    "selected_regional_tau_years",
                    "selected_kde_bandwidth_km",
                    "selected_recent_alpha",
                    "regional_candidates",
                    "kde_candidates",
                    "recent_candidates",
                },
                "location selection",
            )
            latest = _nonnegative_int(location["latest_inner_target_end_us"], "latest inner end")
            boundary_us = _nonnegative_int(
                location["evaluation_start_boundary_us"], "outer boundary"
            )
            boundary = EarlierInnerBoundary(latest, boundary_us)
            counts = location["inner_block_event_counts"]
            if not isinstance(counts, list) or len(counts) != 3:
                raise PredictionSealError("inner block event counts changed")
            inner_count = sum(_nonnegative_int(item, "inner block count") for item in counts)
            regional = _candidate_scores(
                location["regional_candidates"], FROZEN_REGIONAL_TAU_YEARS, "regional"
            )
            kde = _candidate_scores(location["kde_candidates"], FROZEN_KDE_BANDWIDTHS_KM, "kde")
            recent = _candidate_scores(
                location["recent_candidates"], FROZEN_R30_ALPHA_CANDIDATES, "recent"
            )
            tau = select_regional_tau(_mean_scores(regional), boundary=boundary)
            bandwidth = select_kde_bandwidth_one_se(kde, boundary=boundary).selected_bandwidth_km
            alpha = select_recent_alpha(
                _mean_scores(recent), inner_target_count=inner_count, boundary=boundary
            )
            declared = (
                _finite_float(location["selected_regional_tau_years"], "selected tau"),
                _finite_float(location["selected_kde_bandwidth_km"], "selected bandwidth"),
                _finite_float(location["selected_recent_alpha"], "selected alpha"),
            )
            if declared != (tau, bandwidth, alpha):
                raise PredictionSealError("declared selection differs from frozen selection rule")
            time = evidence["time"]
            if not isinstance(time, list) or len(time) != len(TIME_BANDS):
                raise PredictionSealError("time qualification bands changed")
            qualifications: list[dict[str, JsonValue]] = []
            for band, raw_time in zip(TIME_BANDS, time, strict=True):
                time_entry = _exact_mapping(raw_time, {"band", "qualification"}, "time selection")
                if time_entry["band"] != band:
                    raise PredictionSealError("time qualification band order changed")
                qualifications.append(
                    _parse_time_qualification(time_entry["qualification"], "time qualification")
                )
            snapshots.append(
                _SelectionSnapshot(horizon, tau, bandwidth, alpha, tuple(qualifications))
            )
        parsed[fold_id] = tuple(snapshots)
    return parsed


def _cross_validate_selection_npz(
    root: Path, selections: Mapping[str, tuple[_SelectionSnapshot, ...]], fold_ids: Sequence[str]
) -> None:
    status_fields = ("t1_status_code", "t1_reason_code", "t1_historical_block_count")
    optional_fields = (
        ("sample_variance_count", "t1_sample_variance_count", "t1_sample_variance_applicable"),
        ("dispersion_k", "t1_dispersion_k", "t1_dispersion_k_applicable"),
        (
            "observed_information_k",
            "t1_observed_information_k",
            "t1_observed_information_k_applicable",
        ),
        ("standard_error_k", "t1_standard_error_k", "t1_standard_error_k_applicable"),
    )
    for fold_id in fold_ids:
        try:
            with np.load(
                root / "folds" / fold_id / "predictions.npz", allow_pickle=False
            ) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
        except (OSError, ValueError, BadZipFile) as exc:
            raise PredictionSealError("prediction NPZ cannot be cross-validated") from exc
        horizons = arrays["primary_horizon_days"]
        for snapshot in selections[fold_id]:
            rows = np.flatnonzero(horizons == snapshot.horizon_days)
            if rows.size == 0:
                raise PredictionSealError("prediction horizon has no primary rows")
            checks = (
                arrays["location_regional_tau_years"][rows],
                arrays["location_bandwidth_km"][rows, LOCATION_MODEL_IDS.index("L2_KDE_CAUSAL")],
                arrays["location_bandwidth_km"][rows, LOCATION_MODEL_IDS.index("L3_B0_R30_CAUSAL")],
                arrays["location_alpha"][rows, LOCATION_MODEL_IDS.index("L3_B0_R30_CAUSAL")],
            )
            expected = (
                snapshot.regional_tau_years,
                snapshot.kde_bandwidth_km,
                snapshot.kde_bandwidth_km,
                snapshot.recent_alpha,
            )
            if any(
                not np.all(values == value) for values, value in zip(checks, expected, strict=True)
            ):
                raise PredictionSealError("parameter selection differs from prediction NPZ")
            for band_index, qualification in enumerate(snapshot.time_qualifications):
                direct = (
                    NB2_STATUS_CODES.index(cast(str, qualification["status"])),
                    NB2_REASON_CODES.index(cast(str, qualification["reason"])),
                    cast(int, qualification["historical_block_count"]),
                )
                for field, value in zip(status_fields, direct, strict=True):
                    if not np.all(arrays[field][rows, band_index] == value):
                        raise PredictionSealError("time qualification differs from prediction NPZ")
                if not np.all(
                    arrays["t1_sample_mean_count"][rows, band_index]
                    == qualification["sample_mean_count"]
                ):
                    raise PredictionSealError("time mean differs from prediction NPZ")
                for json_key, value_field, mask_field in optional_fields:
                    optional_value = qualification[json_key]
                    expected_mask = 0 if optional_value is None else 1
                    expected_value = 0.0 if optional_value is None else optional_value
                    if not np.all(
                        arrays[mask_field][rows, band_index] == expected_mask
                    ) or not np.all(arrays[value_field][rows, band_index] == expected_value):
                        raise PredictionSealError(
                            "time qualification optional field differs from prediction NPZ"
                        )


def _validate_prediction_phase_artifacts(
    root: Path,
    input_identities: PredictionInputIdentities,
    *,
    fold_ids: Sequence[str] = DEVELOPMENT_FOLD_IDS,
) -> list[JsonValue]:
    run_path, selection_path = _prediction_phase_paths(root)
    run, run_artifact = _read_canonical_mapping(run_path)
    if set(run) != _RUN_MANIFEST_KEYS:
        raise PredictionSealError("prediction run manifest schema changed")
    workers = run.get("maximum_fold_workers")
    if (
        run.get("schema_version") != 1
        or run.get("record_type") != "s1_c0_prediction_run_manifest"
        or run.get("run_contract_id") != _RUN_CONTRACT_ID
        or run.get("stage") != "S1-C0"
        or run.get("role") != "development_prediction_only"
        or run.get("git_commit_oid") != input_identities.git_commit_oid
        or run.get("input_identities") != input_identities.as_mapping()
        or run.get("fold_ids") != list(DEVELOPMENT_FOLD_IDS)
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 3
        or run.get("numerical_threads_per_worker") != 1
        or run.get("outer_targets_constructed") is not False
        or run.get("model_scores_read") is not False
        or run.get("locked_test_run") is not False
    ):
        raise PredictionSealError("prediction run manifest identity or safety flags changed")

    selection, selection_artifact = _read_canonical_mapping(selection_path)
    if set(selection) != _PARAMETER_SELECTION_KEYS:
        raise PredictionSealError("parameter-selection manifest schema changed")
    folds = selection.get("folds")
    if (
        selection.get("schema_version") != 1
        or selection.get("record_type") != "s1_c0_inner_parameter_selection"
        or selection.get("run_contract_id") != _RUN_CONTRACT_ID
        or selection.get("role") != "strictly_earlier_inner_selection_only"
        or not isinstance(folds, dict)
        or set(folds) != set(DEVELOPMENT_FOLD_IDS)
    ):
        raise PredictionSealError("parameter-selection manifest identity changed")
    _audit_manifest_keys(selection, path=("inner", "selection"))
    _audit_no_prohibited_roles(selection, path="$.inner.selection")
    selections = _parse_parameter_selection(selection)
    _cross_validate_selection_npz(root, selections, fold_ids)

    return [
        {
            "relative_path": run_artifact.path.relative_to(root).as_posix(),
            "sha256": run_artifact.sha256,
            "size_bytes": run_artifact.size_bytes,
        },
        {
            "relative_path": selection_artifact.path.relative_to(root).as_posix(),
            "sha256": selection_artifact.sha256,
            "size_bytes": selection_artifact.size_bytes,
        },
    ]


def _validate_global_payload_uniqueness(root: Path) -> None:
    paths = [root / "folds" / fold_id / "predictions.npz" for fold_id in DEVELOPMENT_FOLD_IDS]
    hashes: list[str] = []
    for fold_id, _path in zip(DEVELOPMENT_FOLD_IDS, paths, strict=True):
        record, _ = _read_canonical_mapping(_fold_path(root, fold_id))
        references = record.get("prediction_artifacts")
        if not isinstance(references, list) or len(references) != 1:
            raise PredictionSealError(f"fold must bind exactly one payload: {fold_id}")
        reference = references[0]
        if not isinstance(reference, dict) or not isinstance(reference.get("sha256"), str):
            raise PredictionSealError(f"fold payload reference is invalid: {fold_id}")
        hashes.append(cast(str, reference["sha256"]))
    if len(set(hashes)) != len(hashes):
        raise PredictionSealError("four fold payload contents must be globally unique")
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            try:
                if os.path.samefile(first, second):
                    raise PredictionSealError(
                        "four fold payloads cannot share one physical file identity"
                    )
            except OSError as exc:
                raise PredictionSealError("fold payload file identity cannot be verified") from exc


def seal_fold_prediction(
    output_root: str | Path,
    fold_id: str,
    *,
    prediction_manifest: Mapping[str, object],
    prediction_artifacts: Sequence[PredictionArtifactInput],
    project_root: str | Path,
    data_root: str | Path,
) -> SealedArtifact:
    """Bind real payloads into one immutable, target-blind fold prediction seal."""

    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise PredictionSealError("only the four frozen development folds may be sealed")
    root = _require_configured_prediction_root(project_root, output_root)
    _validate_output_tree(root)
    input_identities = recompute_prediction_input_identities(
        project_root=project_root, data_root=data_root
    )
    manifest = _normalise_json(prediction_manifest, path="$.prediction_manifest")
    if manifest != frozen_fold_prediction_manifest(fold_id):
        raise PredictionSealError("prediction_manifest differs from the frozen fold manifest")
    if len(prediction_artifacts) != 1 or not isinstance(
        prediction_artifacts[0], PredictionArtifactInput
    ):
        raise PredictionSealError("each fold requires exactly one PredictionArtifactInput")
    artifact_references: list[JsonValue] = [
        _build_prediction_artifact_reference(root, prediction_artifacts[0], fold_id=fold_id)
    ]
    _validate_prediction_phase_artifacts(root, input_identities, fold_ids=(fold_id,))
    payload: dict[str, JsonValue] = {
        "schema_version": 3,
        "record_type": "s1_development_fold_prediction",
        "role": "development",
        "fold_id": fold_id,
        "input_identities": cast(dict[str, JsonValue], input_identities.as_mapping()),
        "prediction_manifest": manifest,
        "prediction_artifacts": artifact_references,
    }
    sealed = _write_exclusive_json(_fold_path(root, fold_id), payload)
    # Close the seal only after a second read of every payload.  A concurrent
    # mutation leaves a visible but unusable fold seal and therefore fails closed.
    _validate_fold_prediction(root, fold_id, input_identities)
    return sealed


def _validate_fold_prediction(
    root: Path, fold_id: str, input_identities: PredictionInputIdentities
) -> SealedArtifact:
    path = _fold_path(root, fold_id)
    record, artifact = _read_canonical_mapping(path)
    if set(record) != _FOLD_RECORD_KEYS:
        raise PredictionSealError(f"fold prediction schema changed for {fold_id}")
    if (
        record.get("schema_version") != 3
        or record.get("record_type") != "s1_development_fold_prediction"
        or record.get("role") != "development"
        or record.get("fold_id") != fold_id
    ):
        raise PredictionSealError(f"fold prediction identity changed for {fold_id}")
    if record.get("input_identities") != input_identities.as_mapping():
        raise PredictionSealError(f"fold prediction input identities changed for {fold_id}")
    manifest = record.get("prediction_manifest")
    if manifest != frozen_fold_prediction_manifest(fold_id):
        raise PredictionSealError(f"fold prediction manifest changed for {fold_id}")
    raw_artifacts = record.get("prediction_artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 1:
        raise PredictionSealError(f"fold prediction must bind exactly one payload: {fold_id}")
    _verify_prediction_artifact_reference(root, raw_artifacts[0], fold_id=fold_id)
    return artifact


def verify_fold_prediction(
    output_root: str | Path,
    fold_id: str,
    *,
    project_root: str | Path,
    data_root: str | Path,
) -> SealedArtifact:
    """Read-only verification of one completed fold for interruption-safe resume."""

    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise PredictionSealError("only the four frozen development folds may be verified")
    root = _require_configured_prediction_root(project_root, output_root)
    _validate_output_tree(root)
    identities = recompute_prediction_input_identities(
        project_root=project_root, data_root=data_root
    )
    artifact = _validate_fold_prediction(root, fold_id, identities)
    _validate_prediction_phase_artifacts(root, identities, fold_ids=(fold_id,))
    return artifact


def seal_four_fold_predictions(
    output_root: str | Path, *, project_root: str | Path, data_root: str | Path
) -> DevelopmentScoringAuthorization:
    """Seal all four fold references and return authorization only after re-verification."""

    root = _require_configured_prediction_root(project_root, output_root)
    _validate_output_tree(root)
    input_identities = recompute_prediction_input_identities(
        project_root=project_root, data_root=data_root
    )
    fold_artifacts = [
        _validate_fold_prediction(root, fold_id, input_identities)
        for fold_id in DEVELOPMENT_FOLD_IDS
    ]
    _validate_global_payload_uniqueness(root)
    phase_references = _validate_prediction_phase_artifacts(root, input_identities)
    references: list[JsonValue] = []
    for fold_id, artifact in zip(DEVELOPMENT_FOLD_IDS, fold_artifacts, strict=True):
        references.append(
            {
                "fold_id": fold_id,
                "relative_path": artifact.path.relative_to(root).as_posix(),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
        )
    payload: dict[str, JsonValue] = {
        "schema_version": 2,
        "record_type": "s1_development_four_fold_prediction_seal",
        "role": "development",
        "input_identities": cast(dict[str, JsonValue], input_identities.as_mapping()),
        "ordered_fold_predictions": references,
        "prediction_phase_artifacts": phase_references,
    }
    master = _write_exclusive_json(_master_path(root), payload)
    return authorize_development_scoring(
        root,
        expected_seal_sha256=master.sha256,
        project_root=project_root,
        data_root=data_root,
    )


def authorize_development_scoring(
    output_root: str | Path,
    *,
    expected_seal_sha256: str,
    project_root: str | Path,
    data_root: str | Path,
) -> DevelopmentScoringAuthorization:
    """Verify the complete immutable chain before exposing a scoring authorization."""

    if _SHA256_PATTERN.fullmatch(expected_seal_sha256) is None:
        raise PredictionSealError("expected_seal_sha256 must be a lowercase SHA-256")
    root = _require_configured_prediction_root(project_root, output_root)
    _validate_output_tree(root)
    input_identities = recompute_prediction_input_identities(
        project_root=project_root, data_root=data_root
    )
    master_record, master = _read_canonical_mapping(_master_path(root))
    if master.sha256 != expected_seal_sha256:
        raise PredictionSealError("four-fold prediction seal hash mismatch")
    if set(master_record) != _MASTER_RECORD_KEYS:
        raise PredictionSealError("four-fold prediction seal schema changed")
    if (
        master_record.get("schema_version") != 2
        or master_record.get("record_type") != "s1_development_four_fold_prediction_seal"
        or master_record.get("role") != "development"
        or master_record.get("input_identities") != input_identities.as_mapping()
    ):
        raise PredictionSealError("four-fold prediction seal identity changed")
    raw_references = master_record.get("ordered_fold_predictions")
    if not isinstance(raw_references, list) or len(raw_references) != len(DEVELOPMENT_FOLD_IDS):
        raise PredictionSealError("four-fold prediction seal is incomplete")
    ordered_hashes: list[tuple[str, str]] = []
    for fold_id, raw_reference in zip(DEVELOPMENT_FOLD_IDS, raw_references, strict=True):
        if not isinstance(raw_reference, dict) or set(raw_reference) != _FOLD_REFERENCE_KEYS:
            raise PredictionSealError(f"fold reference schema changed for {fold_id}")
        expected_path = _fold_path(root, fold_id)
        expected_relative = expected_path.relative_to(root).as_posix()
        if (
            raw_reference.get("fold_id") != fold_id
            or raw_reference.get("relative_path") != expected_relative
        ):
            raise PredictionSealError(f"fold reference identity changed for {fold_id}")
        artifact = _validate_fold_prediction(root, fold_id, input_identities)
        if (
            raw_reference.get("sha256") != artifact.sha256
            or raw_reference.get("size_bytes") != artifact.size_bytes
        ):
            raise PredictionSealError(f"fold prediction hash or size mismatch for {fold_id}")
        ordered_hashes.append((fold_id, artifact.sha256))
    _validate_global_payload_uniqueness(root)
    actual_phase_references = _validate_prediction_phase_artifacts(root, input_identities)
    raw_phase_references = master_record.get("prediction_phase_artifacts")
    if (
        not isinstance(raw_phase_references, list)
        or len(raw_phase_references) != 2
        or raw_phase_references != actual_phase_references
        or any(
            not isinstance(reference, dict) or set(reference) != _PHASE_REFERENCE_KEYS
            for reference in raw_phase_references
        )
    ):
        raise PredictionSealError("prediction-phase artifact identities changed")
    return DevelopmentScoringAuthorization(
        seal=master,
        input_identities=input_identities,
        ordered_fold_sha256=tuple(ordered_hashes),
    )


__all__ = [
    "DevelopmentScoringAuthorization",
    "PredictionArtifactInput",
    "PredictionInputIdentities",
    "PredictionSealError",
    "PredictionSealExistsError",
    "SealedArtifact",
    "authorize_development_scoring",
    "canonical_json_bytes",
    "frozen_fold_prediction_manifest",
    "recompute_prediction_input_identities",
    "seal_fold_prediction",
    "seal_four_fold_predictions",
    "verify_fold_prediction",
]
