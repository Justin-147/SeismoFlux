"""Small loader for the score-blind D1 scientific contract.

The loader deliberately validates only the contract boundary. It neither reads
target rows nor computes a model score. Large scientific inputs are verified by
the runner once, immediately before the first real replay checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from seismoflux.data.common import canonical_json_bytes

CONFIG_RELATIVE_PATH = Path("configs/d1_retrospective_development.yaml")
WATER_LEVEL_RELATIVE_PATH = Path("data/manifests/d1_fold_water_level_manifest.json")
EXPECTED_PROTOCOL_VERSION = "d1.0.0"
EXPECTED_MODEL_IDS = (
    "B0",
    "B0_R30",
    "B0_C",
    "B0_C_A_snapshot",
    "B0_C_A_dynamic",
    "B0_R30_C_A_dynamic",
)
EXPECTED_FEATURE_GROUP_IDS = frozenset({"C1", "C2", "S1", "S2", "S3", "S4", "S5", "D1", "D2"})
CONFIG_TO_RUNTIME_FOLD_ID = {
    "F1": "fold_1",
    "F2": "fold_2",
    "F3": "fold_3",
}
RUNTIME_FOLD_SEED_CODE = {"fold_1": 1, "fold_2": 2, "fold_3": 3}


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a lowercase SHA-256 without loading a large input into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _resolve_inside(root: Path, relative: str, *, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    return path


@dataclass(frozen=True, slots=True)
class D1Protocol:
    """Validated D1 config and target-blind water-level manifest."""

    repository_root: Path
    config_path: Path
    water_level_path: Path
    config_sha256: str
    water_level_content_sha256: str
    config: Mapping[str, Any]
    water_level: Mapping[str, Any]

    def resolve_repository_path(self, relative: str, *, label: str = "path") -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{label} escapes the repository")
        try:
            return _resolve_inside(self.repository_root, relative, label=label)
        except ValueError:
            # Large immutable data are shared into this nested Git worktree as
            # links to the frozen machine-local data root.  The config bytes
            # are hash-bound, and every caller must still verify the file SHA.
            data = _mapping(self.config.get("data"), label="D1 data")
            configured_root_raw = data.get("local_data_root_current_machine")
            if (
                not isinstance(configured_root_raw, str)
                or not configured_root_raw
                or not relative_path.parts
                or relative_path.parts[0].lower() != "data"
            ):
                raise
            configured_root = Path(configured_root_raw).resolve()
            shared_path = (configured_root / Path(*relative_path.parts[1:])).resolve()
            try:
                shared_path.relative_to(configured_root)
            except ValueError as exc:
                raise ValueError(f"{label} escapes the configured D1 data root") from exc
            return shared_path


def load_d1_protocol(repository_root: Path) -> D1Protocol:
    """Load the frozen D1-0 pair and fail closed on any identity drift."""

    root = repository_root.resolve()
    config_path = _resolve_inside(root, CONFIG_RELATIVE_PATH.as_posix(), label="D1 config")
    water_path = _resolve_inside(root, WATER_LEVEL_RELATIVE_PATH.as_posix(), label="water level")
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), label="D1 config")
    water = _mapping(json.loads(water_path.read_text(encoding="utf-8")), label="water level")

    config_sha = sha256_file(config_path)
    binding = _mapping(water.get("d1_contract_binding"), label="water.d1_contract_binding")
    if binding.get("path") != CONFIG_RELATIVE_PATH.as_posix():
        raise ValueError("water-level manifest points to another D1 config")
    if binding.get("file_sha256") != config_sha:
        raise ValueError("D1 config bytes differ from the frozen water-level binding")

    content_sha = water.get("content_sha256")
    if not isinstance(content_sha, str):
        raise ValueError("water-level manifest omitted content_sha256")
    content_preimage = dict(water)
    content_preimage.pop("content_sha256", None)
    observed_content_sha = hashlib.sha256(canonical_json_bytes(content_preimage)).hexdigest()
    if observed_content_sha != content_sha:
        raise ValueError("water-level canonical content SHA-256 changed")

    if config.get("protocol_version") != EXPECTED_PROTOCOL_VERSION or config.get("stage") != "D1":
        raise ValueError("D1 protocol identity changed")
    authorization = _mapping(config.get("authorization"), label="authorization")
    if authorization.get("locked_test_run") is not False:
        raise ValueError("locked test must remain unopened in D1 development")
    if authorization.get("real_prospective_issue_authorized") is not False:
        raise ValueError("D1 development may not authorize a real prospective issue")
    if water.get("model_effect_fields_read") != []:
        raise ValueError("D1-0 water level must remain score blind")

    models = config.get("models")
    if not isinstance(models, list):
        raise ValueError("D1 models must be a list")
    model_ids = tuple(_mapping(item, label="model").get("id") for item in models)
    if model_ids != EXPECTED_MODEL_IDS:
        raise ValueError("D1 six-model order changed")
    groups = _mapping(config.get("feature_groups"), label="feature_groups")
    if frozenset(groups) != EXPECTED_FEATURE_GROUP_IDS:
        raise ValueError("D1 nine scientific groups changed")

    time_contract = _mapping(config.get("time"), label="time")
    config_folds = time_contract.get("folds")
    water_folds = water.get("folds")
    if not isinstance(config_folds, list) or not isinstance(water_folds, list):
        raise ValueError("D1 fold definitions must be lists")
    if len(config_folds) != 3 or len(water_folds) != 3:
        raise ValueError("D1 requires exactly three outer folds")
    for config_fold_raw, water_fold_raw in zip(config_folds, water_folds, strict=True):
        config_fold = _mapping(config_fold_raw, label="time.fold")
        water_fold = _mapping(water_fold_raw, label="water.fold")
        config_id = config_fold.get("id")
        runtime_id = water_fold.get("fold_id")
        if not isinstance(config_id, str) or CONFIG_TO_RUNTIME_FOLD_ID.get(config_id) != runtime_id:
            raise ValueError("D1 config/runtime fold identity mapping changed")
        interval = _mapping(
            water_fold.get("assessment_interval_local"),
            label=f"water.{runtime_id}.assessment_interval_local",
        )
        if interval.get("start_inclusive") != config_fold.get(
            "assessment_start_local"
        ) or interval.get("end_exclusive") != config_fold.get("assessment_end_exclusive_local"):
            raise ValueError("D1 config/runtime fold assessment intervals differ")

    return D1Protocol(
        repository_root=root,
        config_path=config_path,
        water_level_path=water_path,
        config_sha256=config_sha,
        water_level_content_sha256=content_sha,
        config=config,
        water_level=water,
    )


__all__ = [
    "CONFIG_RELATIVE_PATH",
    "CONFIG_TO_RUNTIME_FOLD_ID",
    "EXPECTED_FEATURE_GROUP_IDS",
    "EXPECTED_MODEL_IDS",
    "RUNTIME_FOLD_SEED_CODE",
    "WATER_LEVEL_RELATIVE_PATH",
    "D1Protocol",
    "load_d1_protocol",
    "sha256_file",
]
