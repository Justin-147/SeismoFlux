"""Frozen S2-C strain fields combined with the unchanged causal catalogue backbone."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml

from seismoflux.multitask_s1 import c2b_predict as catalog_predict
from seismoflux.multitask_s1.c2b_predict import (
    HORIZONS,
    _atomic_npz,
    _checked,
    _emit,
    _expected_horizon_axis,
    _inner_samples,
    _read_json,
    _record,
    _run_lock,
    _sha,
    _write_json,
)
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.runner_inputs import S1RunnerInputs
from seismoflux.multitask_s2.fault_predict import (
    GeometryValidationSample,
    ReadOnlyComponentCache,
    _save_or_verify_json,
    _save_prediction_payload,
    build_inner_catalog_validation,
)
from seismoflux.multitask_s2.strain import (
    StrainSurfaces,
    blend_log_masses,
    load_strain_surfaces,
    validate_log_mass,
)

PROTOCOL_PATH = Path("configs/multitask_s2_c_strain.yaml")
PROTOCOL_SHA256 = "e9e0800279b9db64722c35763b0dae4d228d90173ddceb660e614ffbcc78876e"
LAYER_IDS = ("UNIT", "STRAIN")
MODEL_IDS = tuple(
    f"S2C_{layer}_{suffix}" for layer in LAYER_IDS for suffix in ("ONLY", "CATALOG_MIX")
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
        raise ValueError("S2C frozen protocol changed; preserve the existing run")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        tuple(protocol["models"]) != MODEL_IDS
        or tuple(protocol["calendar"]["outer_folds"]) != DEVELOPMENT_FOLD_IDS
        or tuple(protocol["calendar"]["horizons_days"]) != HORIZONS
        or _sha(project / protocol["parent_protocol"]) != protocol["parent_protocol_sha256"]
        or _sha(project / protocol["inputs"]["catalog_protocol"])
        != protocol["inputs"]["catalog_protocol_sha256"]
    ):
        raise ValueError("S2C parent protocol, catalogue or finite axes changed")
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
        "strain_source_sha256": protocol["inputs"]["strain_source"]["sha256"],
        "implementation_hashes": {
            name: _sha(directory / name)
            for name in ("strain_predict.py", "strain.py", "fault_predict.py")
        },
    }


def _output_root(project: Path, protocol: dict[str, Any], output_root: Path | None) -> Path:
    root = (output_root or project / protocol["outputs"]["root"]).resolve()
    base = (project / "outputs" / "multitask_s2").resolve()
    branch = root.relative_to(base).parts if root.is_relative_to(base) else ()
    if not branch or not branch[0].casefold().startswith("s2c_"):
        raise ValueError("S2C outputs must be a new s2c_ run; old S1, S2A and S2B are read-only")
    return root


def _validate_prediction_log_mass(values: np.ndarray, shape: tuple[int, ...]) -> None:
    if values.dtype != np.dtype("float64") or values.shape != shape:
        raise ValueError("S2C log-mass array shape or float64 dtype changed")
    for row in values.reshape((-1, shape[-1])):
        validate_log_mass(row, expected_cells=shape[-1])


def strain_candidates(family: str, protocol: dict[str, Any]) -> list[dict[str, float]]:
    if family == "static_only":
        return [{"alpha": 1.0}]
    if family != "catalog_mixture":
        raise ValueError("unknown frozen S2C model family")
    return [{"alpha": float(alpha)} for alpha in protocol["selection"]["alpha_candidates"]]


def _json_score(value: float | None) -> tuple[float | None, str]:
    if value is None:
        return None, "no_events"
    if np.isneginf(value):
        return None, "negative_infinity_from_zero_mass"
    if not np.isfinite(value):
        raise ValueError("S2C log-density score must be finite, negative infinity or no events")
    return float(value), "finite"


def _block_log_density(
    samples: Sequence[GeometryValidationSample],
    static: np.ndarray,
    alpha: float,
    log_area: np.ndarray,
) -> float | None:
    total, count = 0.0, 0.0
    for sample in samples:
        positive = sample.target_counts > 0
        n = float(sample.target_counts[positive].sum())
        if not n:
            continue
        prediction = blend_log_masses(sample.catalog_log_mass, static, alpha)
        # A true zero with an observed event is -inf; a count-zero cell contributes nothing.
        total += float(np.dot(sample.target_counts[positive], (prediction - log_area)[positive]))
        count += n
    return total / count if count else None


def select_strain_parameters(
    samples: Sequence[GeometryValidationSample],
    surfaces: StrainSurfaces,
    area: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Registered alpha search using equal nonempty I2/I3 block scores; retain true zeros."""
    area = np.asarray(area, dtype=np.float64)
    if area.ndim != 1 or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("S2C cell areas must be a positive finite vector")
    log_area = np.log(area)
    for sample in samples:
        if sample.block_id not in ("I2", "I3"):
            raise ValueError("S2C parameter selection may only use I2 and I3")
        counts = sample.target_counts
        if counts.shape != area.shape or not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("S2C validation counts must be finite, nonnegative and grid aligned")
        validate_log_mass(sample.catalog_log_mass, expected_cells=area.size)
    by_block = [
        [sample for sample in samples if sample.block_id == block] for block in ("I2", "I3")
    ]
    results = {}
    for model_id, specification in protocol["models"].items():
        static = validate_log_mass(
            surfaces.layers[specification["layer"]], expected_cells=area.size
        )
        candidates = strain_candidates(specification["family"], protocol)
        evidence, means = [], []
        for candidate in candidates:
            block_scores = [
                _block_log_density(block, static, candidate["alpha"], log_area)
                for block in by_block
            ]
            nonempty = [value for value in block_scores if value is not None]
            mean = float(np.mean(nonempty)) if nonempty else None
            means.append(mean)
            encoded_blocks = [_json_score(value) for value in block_scores]
            encoded_mean, mean_status = _json_score(mean)
            evidence.append(
                {
                    "candidate": candidate,
                    "block_scores": [value for value, _ in encoded_blocks],
                    "block_score_statuses": [status for _, status in encoded_blocks],
                    "mean": encoded_mean,
                    "mean_status": mean_status,
                }
            )
        eligible = [index for index, mean in enumerate(means) if mean is not None]
        if specification["family"] == "static_only":
            selected_index, status = 0, "fixed_static_field_no_fitted_parameters"
        elif not eligible:
            selected_index, status = 0, "no_earlier_validation_labels_exact_catalog"
        else:
            best = max(means[index] for index in eligible)
            if np.isneginf(best):
                selected_index = eligible[0]
                status = "all_candidates_negative_infinity_first_registered_alpha"
            else:
                selected_index = next(
                    index
                    for index in eligible
                    if np.isfinite(means[index])
                    and best - means[index] <= protocol["selection"]["tie_tolerance"]
                )
                status = "selected_from_nonempty_earlier_validation_blocks"
        results[model_id] = {
            "selected": candidates[selected_index],
            "status": status,
            "validation_blocks": ["I2", "I3"],
            "validation_issue_counts": [len(block) for block in by_block],
            "validation_target_counts": [
                int(sum(sample.target_counts.sum() for sample in block)) for block in by_block
            ],
            "candidates": evidence,
        }
    return results


def _prediction_models(
    catalog_log_mass: np.ndarray,
    surfaces: StrainSurfaces,
    selections: Mapping[str, Any],
    protocol: dict[str, Any],
) -> np.ndarray:
    return np.stack(
        [
            blend_log_masses(
                catalog_log_mass,
                surfaces.layers[specification["layer"]],
                float(selections[model_id]["selected"]["alpha"]),
            )
            for model_id, specification in protocol["models"].items()
        ]
    )


def _read_surface_payload(
    path: Path, protocol: dict[str, Any], identity: Mapping[str, Any]
) -> StrainSurfaces:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"identity_json", "audit_json", "layer_ids", "log_cell_mass"}:
            raise ValueError("S2C saved surface fields changed")
        if json.loads(str(saved["identity_json"].item())) != identity:
            raise ValueError("S2C saved surface identity changed")
        if saved["layer_ids"].tolist() != list(LAYER_IDS):
            raise ValueError("S2C saved surface layer axis changed")
        logs = saved["log_cell_mass"].copy()
        audit = json.loads(str(saved["audit_json"].item()))
    _validate_prediction_log_mass(logs, (len(LAYER_IDS), protocol["inputs"]["grid_cells"]))
    logs.setflags(write=False)
    return StrainSurfaces(
        layers={layer: logs[index] for index, layer in enumerate(LAYER_IDS)}, audit=audit
    )


def _verify_surface_checkpoint(
    root: Path, protocol: dict[str, Any], identity: Mapping[str, Any]
) -> StrainSurfaces:
    record = _read_json(root / "strain_surfaces" / "surface_manifest.json")
    if record["identity"] != identity:
        raise ValueError("S2C surface checkpoint identity changed")
    surfaces = _read_surface_payload(_checked(root, record["surfaces"]), protocol, identity)
    if _read_json(_checked(root, record["audit"])) != {
        "identity": identity,
        "audit": surfaces.audit,
    }:
        raise ValueError("S2C saved surface diagnostic differs from its complete payload")
    return surfaces


def _load_or_build_surfaces(
    root: Path,
    data: Path,
    protocol: dict[str, Any],
    domain: Any,
    identity: dict[str, Any],
) -> tuple[StrainSurfaces, Path]:
    """One immutable static-field checkpoint; no scales and no repeated remapping on resume."""
    directory = root / "strain_surfaces"
    manifest_path = directory / "surface_manifest.json"
    if manifest_path.exists():
        return _verify_surface_checkpoint(root, protocol, identity), manifest_path
    payload = directory / "surfaces.npz"
    if payload.exists():
        surfaces = _read_surface_payload(payload, protocol, identity)
    else:
        _emit("s2c_static_surfaces_started", layer_count=len(LAYER_IDS), no_smoothing_scales=True)
        surfaces = load_strain_surfaces(data_root=data, domain=domain, protocol=protocol)
        logs = np.stack([surfaces.layers[layer] for layer in LAYER_IDS])
        _validate_prediction_log_mass(logs, (len(LAYER_IDS), protocol["inputs"]["grid_cells"]))
        _atomic_npz(
            payload,
            {
                "identity_json": np.asarray(json.dumps(identity, sort_keys=True)),
                "audit_json": np.asarray(
                    json.dumps(surfaces.audit, sort_keys=True, allow_nan=False)
                ),
                "layer_ids": np.asarray(LAYER_IDS),
                "log_cell_mass": logs,
            },
        )
        _emit("s2c_static_surfaces_saved", layer_count=len(LAYER_IDS))
    audit_path = directory / "audit.json"
    _save_or_verify_json(audit_path, {"identity": identity, "audit": surfaces.audit})
    _write_json(
        manifest_path,
        {
            "identity": identity,
            "surfaces": _record(root, payload),
            "audit": _record(root, audit_path),
        },
    )
    return _verify_surface_checkpoint(root, protocol, identity), manifest_path


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
        or metadata["outer_targets_read"] is not False
    ):
        raise ValueError(
            "S2C horizon identity, finite horizon axis or prediction-only boundary changed"
        )
    with np.load(_checked(root, metadata["predictions"]), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    expected = _expected_horizon_axis(project, fold_id, horizon)
    if set(arrays) != {"fold_id", "issue_times_us", "horizons_days", "model_ids", "log_cell_mass"}:
        raise ValueError("unexpected S2C prediction fields")
    if (
        str(arrays["fold_id"].item()) != fold_id
        or arrays["issue_times_us"].dtype != np.dtype("int64")
        or arrays["horizons_days"].dtype != np.dtype("int64")
        or arrays["issue_times_us"].tolist() != expected
        or arrays["horizons_days"].tolist() != [horizon] * len(expected)
        or arrays["model_ids"].tolist() != list(MODEL_IDS)
    ):
        raise ValueError("S2C saved issue, horizon, or model axis differs")
    _validate_prediction_log_mass(
        arrays["log_cell_mass"],
        (len(expected), len(MODEL_IDS), int(load_protocol(project)["inputs"]["grid_cells"])),
    )
    return arrays


def load_fold_arrays(output_root: Path, fold_record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """C2B-compatible array layout; caller verifies the complete S2C manifest first."""
    return catalog_predict.load_fold_arrays(output_root, fold_record)


def _verify_fold(
    project: Path, root: Path, record: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    metadata = _read_json(_checked(root, record))
    if metadata["identity"] != identity or metadata["fold_id"] != record["fold_id"]:
        raise ValueError("S2C fold identity changed")
    if tuple(item["horizon_days"] for item in metadata["horizons"]) != HORIZONS:
        raise ValueError("S2C fold must contain exactly the five horizons")
    for horizon in metadata["horizons"]:
        _verify_horizon(project, root, horizon, identity, record["fold_id"])


def verify_prediction_manifest(project_root: Path, output_root: Path) -> dict[str, Any]:
    project, root = project_root.resolve(), output_root.resolve()
    protocol = load_protocol(project)
    identity = _identity(protocol)
    manifest = _read_json(root / "prediction_manifest.json")
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("S2C final prediction identity changed")
    if tuple(item["fold_id"] for item in manifest["folds"]) != DEVELOPMENT_FOLD_IDS:
        raise ValueError("all four S2C folds must be saved before any outer scoring")
    if manifest["issue_horizon_pairs"] != 396 or manifest["outer_targets_read"] is not False:
        raise ValueError("S2C final pair count or prediction-only boundary changed")
    for name in ("catalog_cache_inventory", "strain_surfaces"):
        _checked(root, manifest[name])
    _verify_surface_checkpoint(root, protocol, identity)
    for fold in manifest["folds"]:
        _verify_fold(project, root, fold, identity)
    return manifest


def _run_fold(
    inputs: S1RunnerInputs,
    protocol: dict[str, Any],
    catalog_protocol: dict[str, Any],
    cache: ReadOnlyComponentCache,
    surfaces: StrainSurfaces,
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
        _emit("s2c_fold_reused", fold_id=fold_id)
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
            _emit("s2c_horizon_reused", fold_id=fold_id, horizon_days=horizon)
            continue
        samples = _inner_samples(inputs, fold, horizon, cache)
        validation, branches = build_inner_catalog_validation(
            samples, fold, cache.area, catalog_protocol
        )
        selections = select_strain_parameters(validation, surfaces, cache.area, protocol)
        selected = np.flatnonzero(old_arrays["horizons_days"] == horizon)
        issues = old_arrays["issue_times_us"][selected]
        if issues.tolist() != _expected_horizon_axis(inputs.project_root, fold_id, horizon):
            raise ValueError("saved C2B reference issue axis differs from S2C frozen calendar")
        logs = np.stack(
            [
                _prediction_models(
                    old_arrays["log_cell_mass"][index, model_index], surfaces, selections, protocol
                )
                for index in selected
            ]
        )
        _validate_prediction_log_mass(logs, (len(issues), len(MODEL_IDS), cache.area.size))
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
                "strain_selection": selections,
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
            "s2c_horizon_complete",
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
    _emit("s2c_fold_complete", fold_id=fold_id, issue_count=99)
    return record


def run_prediction_phase(
    *, project_root: Path, data_root: Path, output_root: Path | None = None, workers: int = 2
) -> Path:
    if isinstance(workers, bool) or workers not in (1, 2, 3):
        raise ValueError("S2C permits at most three fold workers")
    if any(os.environ.get(name) != "1" for name in _NUMERICAL_ENV):
        raise ValueError("set numerical-library thread limits to one before importing the runner")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    project, data = project_root.resolve(), data_root.resolve()
    protocol = load_protocol(project)
    root = _output_root(project, protocol, output_root)
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
        surfaces, surface_path = _load_or_build_surfaces(
            root, data, protocol, inputs.spatial_domain, identity
        )
        old_folds = {record["fold_id"]: record for record in catalog_manifest["folds"]}
        _emit(
            "s2c_prediction_started",
            fold_workers=workers,
            total_issue_horizon_pairs=396,
            numerical_threads=1,
            static_strain_role=protocol["scientific_role"],
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
                "strain_surfaces": _record(root, surface_path),
                "completed_utc": datetime.now(UTC).isoformat(),
                "issue_horizon_pairs": 396,
                "outer_targets_read": False,
                "resource_profile": {"fold_workers": workers, "numerical_threads": 1},
            },
        )
        verify_prediction_manifest(project, root)
        return manifest_path
