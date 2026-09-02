"""Frozen S2A geometry increment, with earlier-only catalogue selection and reuse."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml

from seismoflux.multitask_s1 import c2b_predict as catalog_predict
from seismoflux.multitask_s1.c2b_predict import (
    COMPONENT_IDS,
    HORIZONS,
    InnerSample,
    _atomic_npz,
    _candidate_logmass,
    _checked,
    _emit,
    _epoch_us,
    _expected_horizon_axis,
    _inner_samples,
    _read_json,
    _record,
    _run_lock,
    _sha,
    _utc,
    _validate_log_mass,
    _write_json,
    legal_ridge_training,
    select_kernel_parameters,
)
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.runner_inputs import S1RunnerInputs
from seismoflux.multitask_s2.fault_geometry import (
    FaultSurfaces,
    blend_log_masses,
    load_fault_surfaces,
)

PROTOCOL_PATH = Path("configs/multitask_s2_a_fault_geometry.yaml")
PROTOCOL_SHA256 = "d6e19dca67063030e8eafdfd766f13f2310a5e52790cc4b2b7fe8707cf58b5c9"
MODEL_IDS = (
    "S2A_SIMPLE_FAULT_ONLY",
    "S2A_SIMPLE_CATALOG_MIX",
    "S2A_SIMPLE_COARSE_MIX",
    "S2A_TRACE_FAULT_ONLY",
    "S2A_TRACE_CATALOG_MIX",
    "S2A_TRACE_COARSE_MIX",
)
_NUMERICAL_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def load_protocol(project: Path) -> dict[str, Any]:
    path = project / PROTOCOL_PATH
    if _sha(path) != PROTOCOL_SHA256:
        raise ValueError("S2A frozen protocol changed; do not reuse or overwrite this run")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        tuple(protocol["models"]) != MODEL_IDS
        or tuple(protocol["calendar"]["outer_folds"]) != DEVELOPMENT_FOLD_IDS
        or tuple(protocol["calendar"]["horizons_days"]) != HORIZONS
        or _sha(project / protocol["inputs"]["catalog_protocol"])
        != protocol["inputs"]["catalog_protocol_sha256"]
    ):
        raise ValueError("S2A catalogue, finite models, or development axes changed")
    return protocol


def _identity(protocol: dict[str, Any]) -> dict[str, Any]:
    directory = Path(__file__).parent
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "scientific_role": protocol["scientific_role"],
        "grid_id": protocol["inputs"]["grid_id"],
        "model_ids": list(MODEL_IDS),
        "catalog_prediction_manifest_sha256": protocol["inputs"][
            "catalog_prediction_manifest_sha256"
        ],
        "catalog_score_manifest_sha256": protocol["inputs"]["catalog_score_manifest_sha256"],
        "catalog_main_model": protocol["inputs"]["catalog_main_model"],
        "geometry_source_hashes": {
            key: spec["sha256"] for key, spec in protocol["inputs"]["geometry_sources"].items()
        },
        "implementation_hashes": {
            name: _sha(directory / name) for name in ("fault_predict.py", "fault_geometry.py")
        },
    }


class ReadOnlyComponentCache:
    """Authenticate the existing 413 kernels; never create, repair, or recalculate one."""

    def __init__(
        self,
        root: Path,
        identity: Mapping[str, Any],
        area: np.ndarray,
        *,
        expected_files: int = 413,
    ) -> None:
        self.root = root / "component_cache"
        self.area = np.asarray(area, dtype=np.float64)
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if not self.root.is_dir():
            raise FileNotFoundError("old C2B component cache is missing; no new kernel is allowed")
        files = []
        for path in self.root.glob("issue_*.npz"):
            match = re.fullmatch(r"issue_(-?\d+)\.npz", path.name)
            if match is not None:
                files.append((int(match.group(1)), path))
        if len(files) != expected_files:
            raise ValueError("old C2B completed component inventory differs; do not repair it")
        self.records = {}
        for issue_us, path in sorted(files):
            self._read(path, issue_us)
            self.records[issue_us] = _record(root, path, issue_time_us=issue_us)
        self.audit = {
            "source_run_read_only": True,
            "cache_identity": self.identity,
            "completed_component_count": len(files),
            "components": list(self.records.values()),
        }

    def _read(self, path: Path, issue_us: int) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"identity", "issue_time_us", "log_masses", "history_counts"}:
                raise ValueError("old C2B component fields changed")
            if (
                str(saved["identity"].item()) != self.identity
                or saved["issue_time_us"].dtype != np.dtype("int64")
                or int(saved["issue_time_us"].item()) != issue_us
            ):
                raise ValueError("old C2B cache identity or issue differs")
            logs = saved["log_masses"].copy()
            counts = saved["history_counts"].copy()
        _validate_log_mass(logs, (len(COMPONENT_IDS), self.area.size))
        if counts.shape != (6,) or counts.dtype != np.dtype("int64") or np.any(counts < 0):
            raise ValueError("old C2B history count shape or values differ")
        logs.setflags(write=False)
        counts.setflags(write=False)
        return {**dict(zip(COMPONENT_IDS, logs, strict=True)), "history_counts": counts}

    def get(self, issue: datetime) -> dict[str, np.ndarray]:
        issue_us = _epoch_us(issue)
        if issue_us not in self.records:
            raise FileNotFoundError("required old C2B kernel is missing; do not calculate it")
        path = _checked(self.root.parent, self.records[issue_us])
        return self._read(path, issue_us)


@dataclass(frozen=True)
class GeometryValidationSample:
    block_id: str
    issue: datetime
    end: datetime
    catalog_log_mass: np.ndarray
    target_counts: np.ndarray


def _sample_audit(samples: Sequence[InnerSample], cutoff: datetime) -> dict[str, Any]:
    counts = [int(sample.counts(cutoff).sum()) for sample in samples]
    return {
        "issue_count": len(samples),
        "target_count": sum(counts),
        "empty_issue_count": sum(count == 0 for count in counts),
        "issues": [
            {
                "block_id": sample.block_id,
                "issue_time_utc": sample.issue.isoformat(),
                "label_end_utc": sample.end.isoformat(),
                "visible_target_count": count,
            }
            for sample, count in zip(samples, counts, strict=True)
        ],
    }


def build_inner_catalog_validation(
    samples: Sequence[InnerSample],
    fold: Mapping[str, Any],
    area: np.ndarray,
    catalog_protocol: dict[str, Any],
) -> tuple[list[GeometryValidationSample], list[dict[str, Any]]]:
    """I2 sees only I1; I3 sees I1+I2, with separate end and availability cutoffs."""
    blocks = {block["id"]: block for block in fold["inner_blocks"]}
    outer_cutoff = _utc(fold["outer_start"]) - timedelta(days=30)
    validation, branches = [], []
    for train_blocks, validate_block in ((["I1"], "I2"), (["I1", "I2"], "I3")):
        start = _utc(blocks[validate_block]["start"])
        cutoff = start - timedelta(days=30)
        train = legal_ridge_training(samples, train_blocks, start)
        selection = select_kernel_parameters(train, cutoff, area, catalog_protocol)["multiscale"]
        if validate_block == "I2" and (
            selection["selected"] != "K75"
            or selection["status"] != "insufficient_nonempty_blocks_fixed_K75"
        ):
            raise ValueError("I2 must retain the exact earlier-only C2B K75 fallback")
        block_samples = [sample for sample in samples if sample.block_id == validate_block]
        if any(
            sample.end > outer_cutoff
            or sample.issue < start
            or sample.end > _utc(blocks[validate_block]["end"])
            for sample in block_samples
        ):
            raise ValueError("S2A validation labels leave the fixed block or outer embargo")
        for sample in block_samples:
            validation.append(
                GeometryValidationSample(
                    sample.block_id,
                    sample.issue,
                    sample.end,
                    _candidate_logmass(
                        sample.components, ("multiscale", selection["selected"]), catalog_protocol
                    ),
                    sample.counts(outer_cutoff),
                )
            )
        branches.append(
            {
                "train_blocks": train_blocks,
                "validate_block": validate_block,
                "training_label_cutoff_utc": cutoff.isoformat(),
                "training": _sample_audit(train, cutoff),
                "catalog_multiscale_selection": selection,
                "I2_explicit_K75_fallback": validate_block == "I2",
                "validation_label_cutoff_utc": outer_cutoff.isoformat(),
                "validation": _sample_audit(block_samples, outer_cutoff),
            }
        )
    return validation, branches


def geometry_candidates(family: str, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    scales = protocol["geometry_math"]["scale_tie_order_km"]
    if family == "fault_only":
        return [{"alpha": 1.0, "scale_km": float(scale)} for scale in scales]
    if family != "catalog_mixture":
        raise ValueError("unknown frozen S2A model family")
    return [{"alpha": 0.0, "scale_km": None}] + [
        {"alpha": float(alpha), "scale_km": float(scale)}
        for alpha in protocol["geometry_math"]["alpha_candidates"]
        if alpha > 0.0
        for scale in scales
    ]


def _geometry_prediction(
    catalog_log_mass: np.ndarray,
    fields: Mapping[float, np.ndarray],
    candidate: Mapping[str, Any],
) -> np.ndarray:
    if candidate["alpha"] == 0.0:
        return catalog_log_mass.copy()
    return blend_log_masses(
        catalog_log_mass, fields[float(candidate["scale_km"])], float(candidate["alpha"])
    )


def select_geometry_parameters(
    samples: Sequence[GeometryValidationSample],
    surfaces: FaultSurfaces,
    area: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """One fixed finite search; equal means of nonempty I2/I3 event-average densities."""
    log_area = np.log(area)
    by_block = {
        block: [sample for sample in samples if sample.block_id == block] for block in ("I2", "I3")
    }
    results = {}
    for model_id, specification in protocol["models"].items():
        fields = getattr(surfaces, specification["representation"])[specification["source"]]
        evidence = []
        for candidate in geometry_candidates(specification["family"], protocol):
            block_scores = []
            for block_samples in by_block.values():
                total, count = 0.0, 0.0
                for sample in block_samples:
                    n = float(sample.target_counts.sum())
                    if n:
                        prediction = _geometry_prediction(
                            sample.catalog_log_mass, fields, candidate
                        )
                        total += float(np.dot(sample.target_counts, prediction - log_area))
                        count += n
                block_scores.append(total / count if count else None)
            nonempty = [value for value in block_scores if value is not None]
            evidence.append(
                {
                    "candidate": candidate,
                    "block_scores": block_scores,
                    "mean": float(np.mean(nonempty)) if nonempty else None,
                }
            )
        eligible = [row for row in evidence if row["mean"] is not None]
        if eligible:
            best = max(row["mean"] for row in eligible)
            selected = next(
                row["candidate"]
                for row in eligible
                if best - row["mean"] <= protocol["selection"]["tie_tolerance"]
            )
            status = "selected_from_nonempty_earlier_validation_blocks"
        else:
            selected = evidence[0]["candidate"]
            status = (
                "no_earlier_validation_labels_fixed_scale_75"
                if specification["family"] == "fault_only"
                else "no_earlier_validation_labels_exact_catalog"
            )
        results[model_id] = {
            "selected": selected,
            "status": status,
            "validation_blocks": ["I2", "I3"],
            "validation_issue_counts": [len(block) for block in by_block.values()],
            "validation_target_counts": [
                int(sum(sample.target_counts.sum() for sample in block))
                for block in by_block.values()
            ],
            "candidates": evidence,
        }
    return results


def _prediction_models(
    catalog_log_mass: np.ndarray,
    surfaces: FaultSurfaces,
    selections: Mapping[str, Any],
    protocol: dict[str, Any],
) -> np.ndarray:
    return np.stack(
        [
            _geometry_prediction(
                catalog_log_mass,
                getattr(surfaces, specification["representation"])[specification["source"]],
                selections[model_id]["selected"],
            )
            for model_id, specification in protocol["models"].items()
        ]
    )


def _verify_horizon(
    project: Path,
    root: Path,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    fold_id: str,
) -> dict[str, np.ndarray]:
    metadata = _read_json(_checked(root, record))
    horizon = int(metadata["horizon_days"])
    if (
        metadata["identity"] != identity
        or metadata["fold_id"] != fold_id
        or horizon not in HORIZONS
        or horizon != int(record["horizon_days"])
    ):
        raise ValueError("S2A horizon identity or finite horizon axis changed")
    with np.load(_checked(root, metadata["predictions"]), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    expected = _expected_horizon_axis(project, fold_id, horizon)
    if set(arrays) != {"fold_id", "issue_times_us", "horizons_days", "model_ids", "log_cell_mass"}:
        raise ValueError("unexpected S2A prediction fields")
    if (
        str(arrays["fold_id"].item()) != fold_id
        or arrays["issue_times_us"].dtype != np.dtype("int64")
        or arrays["horizons_days"].dtype != np.dtype("int64")
        or arrays["issue_times_us"].tolist() != expected
        or arrays["horizons_days"].tolist() != [horizon] * len(expected)
        or arrays["model_ids"].tolist() != list(MODEL_IDS)
    ):
        raise ValueError("S2A saved issue, horizon, or model axis differs")
    _validate_log_mass(
        arrays["log_cell_mass"],
        (len(expected), len(MODEL_IDS), int(load_protocol(project)["inputs"]["grid_cells"])),
    )
    return arrays


def load_fold_arrays(output_root: Path, fold_record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """C2B-compatible layout; verify the complete S2A prediction manifest before reading."""
    return catalog_predict.load_fold_arrays(output_root, fold_record)


def _verify_fold(
    project: Path,
    root: Path,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    metadata = _read_json(_checked(root, record))
    if metadata["identity"] != identity or metadata["fold_id"] != record["fold_id"]:
        raise ValueError("S2A fold identity changed")
    if tuple(item["horizon_days"] for item in metadata["horizons"]) != HORIZONS:
        raise ValueError("S2A fold must contain exactly the five horizons")
    for horizon in metadata["horizons"]:
        _verify_horizon(project, root, horizon, identity, record["fold_id"])


def verify_prediction_manifest(project_root: Path, output_root: Path) -> dict[str, Any]:
    project, root = project_root.resolve(), output_root.resolve()
    identity = _identity(load_protocol(project))
    manifest = _read_json(root / "prediction_manifest.json")
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("S2A final prediction identity changed")
    if tuple(item["fold_id"] for item in manifest["folds"]) != DEVELOPMENT_FOLD_IDS:
        raise ValueError("all four S2A folds must be saved before any outer scoring")
    if manifest["issue_horizon_pairs"] != 396 or manifest["outer_targets_read"] is not False:
        raise ValueError("S2A final pair count or prediction-only boundary changed")
    for name in ("catalog_cache_inventory", "geometry_audit"):
        _checked(root, manifest[name])
    for fold in manifest["folds"]:
        _verify_fold(project, root, fold, identity)
    return manifest


def _save_prediction_payload(path: Path, arrays: Mapping[str, Any]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[name], value) for name, value in arrays.items()
            ):
                raise ValueError("unsealed S2A prediction differs; preserve the original payload")
    else:
        _atomic_npz(path, arrays)


def _save_or_verify_json(path: Path, record: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != record:
            raise ValueError("S2A saved input or identity changed; do not overwrite it")
    else:
        _write_json(path, record)


def _run_fold(
    inputs: S1RunnerInputs,
    protocol: dict[str, Any],
    catalog_protocol: dict[str, Any],
    cache: ReadOnlyComponentCache,
    surfaces: FaultSurfaces,
    root: Path,
    identity: dict[str, Any],
    fold: dict[str, Any],
    catalog_root: Path,
    catalog_fold_record: Mapping[str, Any],
) -> dict[str, Any]:
    fold_id = fold["id"]
    fold_root = root / "folds" / fold_id
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_path = fold_root / "fold_manifest.json"
    if fold_path.exists():
        record = _record(root, fold_path, fold_id=fold_id, issue_count=99)
        _verify_fold(inputs.project_root, root, record, identity)
        _emit("s2a_fold_reused", fold_id=fold_id)
        return record
    old_arrays = catalog_predict.load_fold_arrays(catalog_root, catalog_fold_record)
    old_ids = old_arrays["model_ids"].tolist()
    main_id = protocol["inputs"]["catalog_main_model"]
    if old_ids.count(main_id) != 1 or str(old_arrays["fold_id"].item()) != fold_id:
        raise ValueError("saved C2B reference model or fold differs")
    model_index = old_ids.index(main_id)
    records = []
    for horizon in HORIZONS:
        horizon_root = fold_root / f"horizon_{horizon:03d}"
        path = horizon_root / "horizon_manifest.json"
        if path.exists():
            record = _record(root, path, horizon_days=horizon)
            _verify_horizon(inputs.project_root, root, record, identity, fold_id)
            records.append(record)
            _emit("s2a_horizon_reused", fold_id=fold_id, horizon_days=horizon)
            continue
        samples = _inner_samples(inputs, fold, horizon, cache)
        validation, branches = build_inner_catalog_validation(
            samples, fold, cache.area, catalog_protocol
        )
        selections = select_geometry_parameters(validation, surfaces, cache.area, protocol)
        selected = np.flatnonzero(old_arrays["horizons_days"] == horizon)
        issues = old_arrays["issue_times_us"][selected]
        if issues.tolist() != _expected_horizon_axis(inputs.project_root, fold_id, horizon):
            raise ValueError("saved C2B reference issue axis differs from S2A frozen calendar")
        logs = np.stack(
            [
                _prediction_models(
                    old_arrays["log_cell_mass"][index, model_index], surfaces, selections, protocol
                )
                for index in selected
            ]
        )
        _validate_log_mass(logs, (len(issues), len(MODEL_IDS), cache.area.size))
        prediction_path = horizon_root / "predictions.npz"
        _save_prediction_payload(
            prediction_path,
            {
                "fold_id": np.asarray(fold_id),
                "issue_times_us": issues.copy(),
                "horizons_days": np.full(len(issues), horizon, dtype=np.int64),
                "model_ids": np.asarray(MODEL_IDS),
                "log_cell_mass": logs,
            },
        )
        _write_json(
            path,
            {
                "identity": identity,
                "fold_id": fold_id,
                "horizon_days": horizon,
                "issue_count": len(issues),
                "completed_utc": datetime.now(UTC).isoformat(),
                "predictions": _record(root, prediction_path),
                "inner_catalog_branches": branches,
                "geometry_selection": selections,
                "catalog_reference": {
                    "fold_record": dict(catalog_fold_record),
                    "model_id": main_id,
                    "outer_surface_reused_exactly_before_mixing": True,
                    "outer_parameters_backfilled_into_inner": False,
                },
                "outer_targets_read": False,
            },
        )
        records.append(_record(root, path, horizon_days=horizon))
        _emit(
            "s2a_horizon_complete",
            fold_id=fold_id,
            horizon_days=horizon,
            issue_count=len(issues),
            no_outer_scores=True,
        )
    _write_json(
        fold_path,
        {
            "identity": identity,
            "fold_id": fold_id,
            "horizons": records,
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )
    record = _record(root, fold_path, fold_id=fold_id, issue_count=99)
    _verify_fold(inputs.project_root, root, record, identity)
    _emit("s2a_fold_complete", fold_id=fold_id, issue_count=99)
    return record


def run_prediction_phase(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path | None = None,
    workers: int = 2,
) -> Path:
    if isinstance(workers, bool) or workers not in (1, 2, 3):
        raise ValueError("S2A permits at most three fold workers")
    if any(os.environ.get(name) != "1" for name in _NUMERICAL_ENV):
        raise ValueError("set numerical-library thread limits to one before importing the runner")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    project, data = project_root.resolve(), data_root.resolve()
    protocol = load_protocol(project)
    root = (output_root or project / protocol["outputs"]["root"]).resolve()
    allowed = (project / "outputs" / "multitask_s2").resolve()
    if (
        not root.is_relative_to(allowed)
        or root == allowed
        or not root.is_relative_to(project / "outputs")
        or root.is_relative_to((project / "outputs" / "multitask_s1").resolve())
    ):
        raise ValueError(
            "S2A outputs must be a run within outputs/multitask_s2; old S1 is read-only"
        )
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        manifest_path = root / "prediction_manifest.json"
        if manifest_path.is_file():
            verify_prediction_manifest(project, root)
            return manifest_path
        identity = _identity(protocol)
        _save_or_verify_json(root / "run_identity.json", identity)
        catalog_root = (project / protocol["inputs"]["catalog_run"]).resolve()
        if not catalog_root.is_relative_to(project / "outputs" / "multitask_s1"):
            raise ValueError("old catalogue run must remain in its frozen S1 location")
        for filename, expected in (
            ("prediction_manifest.json", protocol["inputs"]["catalog_prediction_manifest_sha256"]),
            (
                "score_phase/score_manifest.json",
                protocol["inputs"]["catalog_score_manifest_sha256"],
            ),
        ):
            if _sha(catalog_root / filename) != expected:
                raise ValueError("frozen old C2B manifest differs; preserve S1 and stop")
        catalog_protocol = catalog_predict.load_protocol(project)
        catalog_manifest = catalog_predict.verify_prediction_manifest(project, catalog_root)
        old_identity = catalog_predict._identity(catalog_protocol)
        if _read_json(catalog_root / "run_identity.json") != old_identity:
            raise ValueError("old C2B run identity differs from verified saved predictions")
        inputs, _ = catalog_predict.load_inputs(project, data, catalog_protocol)
        cache = ReadOnlyComponentCache(catalog_root, old_identity, inputs.location_grid.area_km2)
        cache_path = root / "catalog_cache_inventory.json"
        _save_or_verify_json(cache_path, cache.audit)
        surfaces = load_fault_surfaces(data, protocol, inputs.spatial_domain)
        geometry_path = root / "geometry_audit.json"
        _save_or_verify_json(geometry_path, surfaces.audit)
        old_folds = {record["fold_id"]: record for record in catalog_manifest["folds"]}
        _emit(
            "s2a_prediction_started",
            fold_workers=workers,
            total_issue_horizon_pairs=396,
            numerical_threads=1,
            static_geometry_role=protocol["scientific_role"],
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_fold,
                    inputs,
                    protocol,
                    catalog_protocol,
                    cache,
                    surfaces,
                    root,
                    identity,
                    fold,
                    catalog_root,
                    old_folds[fold["id"]],
                )
                for fold in inputs.contract["outer_folds"]
            ]
            records = [future.result() for future in futures]
        _write_json(
            manifest_path,
            {
                **identity,
                "folds": records,
                "catalog_cache_inventory": _record(root, cache_path),
                "geometry_audit": _record(root, geometry_path),
                "completed_utc": datetime.now(UTC).isoformat(),
                "issue_horizon_pairs": 396,
                "outer_targets_read": False,
                "resource_profile": {"fold_workers": workers, "numerical_threads": 1},
            },
        )
        verify_prediction_manifest(project, root)
        return manifest_path
