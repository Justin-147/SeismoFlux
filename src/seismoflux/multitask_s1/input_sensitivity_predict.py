"""Location-only C2A input ablation; no outcome table is opened here.

The two fixed C1 masks change training centres, never the national prediction
grid. C0 parameters and unchanged-input predictions are reused verbatim.
Completed folds are resumable; incomplete attempt files are left untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.background.local_support import (
    LocalSupportBasePartition,
    build_local_support_base_partition,
)
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.development_predict import _l3_from_cached_surfaces
from seismoflux.multitask_s1.local_completeness import (
    LocalCompletenessEvent,
    locate_completeness_events,
)
from seismoflux.multitask_s1.location import (
    CausalRecent30History,
    CausalSpatialHistory,
    FrozenSpatialGrid,
    l0_uniform_relative_mass,
    l1_regional_constant_relative_mass,
    l2_gaussian_kde_relative_mass,
)
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_HISTORY_START_UTC,
    CausalMagnitudeHistory,
    S1RunnerInputs,
    causal_catalog_histories,
    load_s1_runner_inputs,
)

PROTOCOL_PATH = Path("configs/multitask_s1_c2a_input_sensitivity.yaml")
PROTOCOL_SHA256 = "4aa9070c190f0e599870bde93899bcb2153f34e5069bf923a2499bce8e9c64bd"
MODEL_IDS = tuple(
    f"{treatment}_{model}"
    for treatment in ("A", "B")
    for model in ("L1_REGIONAL_CONSTANT", "L2_KDE_CAUSAL", "L3_B0_R30_CAUSAL")
)
TREATMENTS = ("A_KEEP_INDETERMINATE", "B_EXCLUDE_INDETERMINATE")
_C0_INDICES = (1, 2, 4)
_DAY_US = 86_400_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NUMERICAL_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class InputSensitivityError(ValueError):
    """The frozen comparison or its input identity cannot be preserved."""


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise InputSensitivityError(f"expected a JSON object: {path}")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _scoped(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InputSensitivityError("artifact path escaped its explicit root")
    return path


def _checked(root: Path, record: Mapping[str, Any], key: str = "path") -> Path:
    path = _scoped(root, record[key])
    if not path.is_file() or _sha(path) != record["sha256"]:
        raise InputSensitivityError(f"frozen artifact SHA-256 mismatch: {path}")
    return path


def _epoch_us(value: datetime) -> int:
    delta = value.astimezone(UTC) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _time(value: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(value))


def _event_digest(event_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(event_ids, separators=(",", ":")).encode()).hexdigest()


def _fixed_parameters(protocol: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    parameters = protocol["models"]["per_fold_fixed_parameters"]
    expected = {
        fold: {"regional_tau_years": tau, "kde_bandwidth_km": 75.0, "recent_alpha": 0.25}
        for fold, tau in zip(DEVELOPMENT_FOLD_IDS, (5.0, 1.0, 1.0, 5.0), strict=True)
    }
    if parameters != expected:
        raise InputSensitivityError("C2A may not reselect or change C0's 30-day parameters")
    return expected


def _load_protocol(project: Path) -> dict[str, Any]:
    path = _checked(project, {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256})
    protocol: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = _fixed_parameters(protocol)
    for name, record in protocol["parent_artifacts"].items():
        if name != "C0_raw_scores_score_phase_only":
            _checked(project, record)
    selection_path = _checked(project, protocol["parent_artifacts"]["C0_parameter_selection"])
    selections = _json(selection_path)["folds"]
    for fold in DEVELOPMENT_FOLD_IDS:
        selected = [row for row in selections[fold] if row["horizon_days"] == 30]
        if len(selected) != 1:
            raise InputSensitivityError("C0 must have one 30-day parameter selection per fold")
        location = selected[0]["inner_evidence"]["location"]
        observed = {
            "regional_tau_years": location["selected_regional_tau_years"],
            "kde_bandwidth_km": location["selected_kde_bandwidth_km"],
            "recent_alpha": location["selected_recent_alpha"],
        }
        if observed != parameters[fold]:
            raise InputSensitivityError(
                "fixed parameters differ from the authenticated C0 selection"
            )
    return protocol


def _identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "run_id": protocol["run_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_sha256": _sha(Path(__file__)),
        "grid_id": protocol["development_design"]["grid_id"],
        "catalog_sha256": protocol["catalog"]["sha256"],
        "model_ids": list(MODEL_IDS),
        "fixed_parameters": _fixed_parameters(protocol),
        "parent_artifacts": {
            name: value
            for name, value in protocol["parent_artifacts"].items()
            if name != "C0_raw_scores_score_phase_only"
        },
    }


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise InputSensitivityError(f"C1 boolean must be explicit True/False, got {value!r}")


def _load_mask(
    csv_path: Path,
    partition: LocalSupportBasePartition,
    fold: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["snapshot_id"] == f"{fold}__OUTER"]
    cells = {cell.cell_id: cell for cell in partition.cells}
    if len(rows) != len(cells) or {row["cell_id"] for row in rows} != set(cells):
        raise InputSensitivityError("C1 outer mask must contain the entire fixed base partition")
    year = int(fold.split("_")[2])
    anchor = datetime(year, 1, 1, tzinfo=timezone(timedelta(hours=8))).astimezone(UTC)
    cutoff = anchor - timedelta(hours=24)
    statuses: dict[str, str] = {}
    area = {status: 0.0 for status in ("supported", "indeterminate", "unsupported")}
    for row in rows:
        status = row["status"]
        cell = cells[row["cell_id"]]
        if (
            row["fold_id"] != fold
            or row["anchor_role"] != "outer_fold_start"
            or datetime.fromisoformat(row["anchor_utc"]) != anchor
            or datetime.fromisoformat(row["cutoff_utc"]) != cutoff
            or (int(row["row"]), int(row["column"])) != (cell.row, cell.column)
            or float(row["clipped_area_m2"]) != cell.clipped_area_m2
            or status not in area
        ):
            raise InputSensitivityError("C1 mask geometry, date, or status changed")
        expected = (status != "unsupported", status == "supported", status == "supported")
        observed = tuple(
            _parse_bool(row[key])
            for key in (
                "main_common_mc4_training_allowed",
                "exclude_indeterminate_training_allowed",
                "supported_area_contributor",
            )
        )
        if observed != expected:
            raise InputSensitivityError("C1 status and explicitly parsed eligibility disagree")
        statuses[cell.cell_id] = status
        area[status] += cell.clipped_area_m2
    return statuses, {
        "snapshot_id": f"{fold}__OUTER",
        "mask_cutoff_utc": cutoff.isoformat(),
        "supported_area_fraction": area["supported"] / partition.total_area_m2,
        "indeterminate_area_fraction": area["indeterminate"] / partition.total_area_m2,
        "unsupported_area_fraction": area["unsupported"] / partition.total_area_m2,
        "national_area_stop_gate_enabled": False,
        "prediction_domain_unchanged": True,
    }


def _locate_training_events(
    inputs: S1RunnerInputs,
    partition: LocalSupportBasePartition,
    last_cutoff_us: int,
) -> dict[str, str]:
    catalog = inputs.catalog
    selected = np.flatnonzero(
        catalog.inside_study_area
        & (catalog.magnitude >= 4.0)
        & (catalog.origin_time_us >= _epoch_us(CATALOG_HISTORY_START_UTC))
        & (catalog.origin_time_us <= last_cutoff_us)
        & (catalog.available_at_us <= last_cutoff_us)
    )
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_user_input(EQUAL_AREA_CRS),
        always_xy=True,
    )
    x_m, y_m = transformer.transform(catalog.longitude[selected], catalog.latitude[selected])
    events = (
        LocalCompletenessEvent(
            event_id=catalog.event_ids[int(index)],
            origin_time_utc=_time(int(catalog.origin_time_us[index])),
            available_at_utc=_time(int(catalog.available_at_us[index])),
            magnitude=float(catalog.magnitude[index]),
            x_m=float(x),
            y_m=float(y),
        )
        for index, x, y in zip(selected, x_m, y_m, strict=True)
    )
    return {
        item.event.event_id: item.cell_id for item in locate_completeness_events(events, partition)
    }


def _load_c0_fold(
    project: Path,
    protocol: Mapping[str, Any],
    fold: str,
    issues: NDArray[np.int64],
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    seal_path = _checked(project, protocol["parent_artifacts"]["C0_four_fold_prediction_seal"])
    seal = _json(seal_path)
    if (
        seal["input_identities"]["grid_sha256"] != protocol["development_design"]["grid_id"]
        or seal["input_identities"]["catalog_sha256"] != protocol["catalog"]["sha256"]
    ):
        raise InputSensitivityError("C0 grid or catalog identity differs from C2A")
    descriptors = {item["fold_id"]: item for item in seal["ordered_fold_predictions"]}
    if tuple(descriptors) != DEVELOPMENT_FOLD_IDS:
        raise InputSensitivityError("C0 seal does not contain the four ordered development folds")
    bundle_path = _checked(seal_path.parent, descriptors[fold], "relative_path")
    bundle = _json(bundle_path)
    expected_models = [
        "L0_UNIFORM",
        "L1_REGIONAL_CONSTANT",
        "L2_KDE_CAUSAL",
        "L2_KDE75_LEGACY",
        "L3_B0_R30_CAUSAL",
    ]
    if (
        bundle["fold_id"] != fold
        or bundle["prediction_manifest"]["model_axes"]["location"] != expected_models
        or bundle["input_identities"] != seal["input_identities"]
    ):
        raise InputSensitivityError("C0 fold, model axis, or input identity mismatch")
    artifacts = bundle["prediction_artifacts"]
    if len(artifacts) != 1:
        raise InputSensitivityError("C0 must have exactly one numeric prediction artifact")
    path = _checked(seal_path.parent, artifacts[0], "relative_path")
    with np.load(path, allow_pickle=False) as archive:
        selected = archive["primary_horizon_days"] == 30
        observed_issues = archive["primary_issue_time_us"][selected]
        parameters = _fixed_parameters(protocol)[fold]
        if (
            not np.array_equal(observed_issues, issues)
            or not np.array_equal(archive["location_model_index"], np.arange(5))
            or not np.all(
                archive["location_regional_tau_years"][selected] == parameters["regional_tau_years"]
            )
            or not np.all(
                archive["location_bandwidth_km"][selected][:, (2, 4)]
                == parameters["kde_bandwidth_km"]
            )
            or not np.all(archive["location_alpha"][selected, 4] == parameters["recent_alpha"])
        ):
            raise InputSensitivityError(
                "C0 primary issue, model, or fixed parameter alignment changed"
            )
        result = {
            "mass": archive["location_relative_mass"][selected][:, _C0_INDICES, :],
            "source_count": archive["location_source_event_count"][selected][:, _C0_INDICES],
            "recent_fallback": archive["location_recent_fallback"][selected, 4],
        }
    if result["mass"].shape != (len(issues), 3, protocol["development_design"]["grid_cell_count"]):
        raise InputSensitivityError("C0 location mass has an unexpected grid shape")
    return result, {"path": path.relative_to(project).as_posix(), "sha256": artifacts[0]["sha256"]}


def _predict_treatment(
    history: CausalMagnitudeHistory,
    keep: NDArray[np.bool_],
    grid: FrozenSpatialGrid,
    parameters: Mapping[str, float],
    c0_mass: NDArray[np.float64],
    cache: dict[tuple[str, ...], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    if keep.dtype != np.bool_ or keep.shape != (history.event_count,):
        raise InputSensitivityError("training eligibility must align with the visible event IDs")
    ids = tuple(
        event_id for event_id, allowed in zip(history.event_ids, keep, strict=True) if allowed
    )
    issue_us = _epoch_us(history.issue_time_utc)
    cutoff_us = _epoch_us(history.data_cutoff_utc)
    if (
        cutoff_us != issue_us - _DAY_US
        or np.any(history.origin_time_us > cutoff_us)
        or np.any(history.available_at_us > cutoff_us)
    ):
        raise InputSensitivityError("training history violates the T-minus-24-hour boundary")
    recent_keep = keep & (history.origin_time_us > issue_us - 30 * _DAY_US)
    recent_count = int(np.count_nonzero(recent_keep))
    diagnostic = {
        "training_event_count": len(ids),
        "recent_event_count": recent_count,
        "recent_fallback": recent_count == 0,
        "empty_all_history_fallback": not ids,
        "training_event_ids_sha256": _event_digest(ids),
        "input_identical_to_C0": ids == history.event_ids,
    }
    if ids == history.event_ids and ids:
        mass = np.array(c0_mass, dtype=np.float64, copy=True)
        source = "C0_exact_same_issue_grid_parameters_and_event_ids"
    elif ids in cache:
        mass = cache[ids]
        source = "same_issue_other_treatment_identical_event_ids"
    elif not ids:
        mass = np.repeat(l0_uniform_relative_mass(grid).cell_relative_mass[None, :], 3, axis=0)
        source = "explicit_empty_history_uniform_reference"
    else:
        spatial = CausalSpatialHistory(history.spatial.x_km[keep], history.spatial.y_km[keep])
        recent = CausalRecent30History(
            x_km=history.spatial.x_km[recent_keep],
            y_km=history.spatial.y_km[recent_keep],
            origin_time_us=history.origin_time_us[recent_keep],
            available_at_us=history.available_at_us[recent_keep],
            issue_time_us=issue_us,
            data_cutoff_us=cutoff_us,
        )
        exposure_days = (
            history.data_cutoff_utc - CATALOG_HISTORY_START_UTC
        ).total_seconds() / 86_400.0
        exposure_years = exposure_days / 365.2425
        regional = l1_regional_constant_relative_mass(
            spatial,
            grid,
            exposure_years=exposure_years,
            tau_years=parameters["regional_tau_years"],
        )
        long = l2_gaussian_kde_relative_mass(
            spatial,
            grid,
            bandwidth_km=parameters["kde_bandwidth_km"],
        )
        recent_surface = (
            l2_gaussian_kde_relative_mass(
                recent.as_spatial_history(),
                grid,
                bandwidth_km=parameters["kde_bandwidth_km"],
                model_id="R30_COMPONENT",
            )
            if recent_count
            else None
        )
        mixed = _l3_from_cached_surfaces(
            long,
            recent_surface,
            recent_event_count=recent_count,
            alpha=parameters["recent_alpha"],
        )
        mass = np.stack([surface.cell_relative_mass for surface in (regional, long, mixed)])
        source = "C0_primitives_with_filtered_training_centres"
    cache[ids] = mass
    diagnostic["prediction_source"] = source
    return mass, diagnostic


def _validate_arrays(arrays: Mapping[str, NDArray[Any]], issue_count: int, cell_count: int) -> None:
    schema = {
        "issue_time_us": ((issue_count,), "int64"),
        "location_relative_mass": ((issue_count, 6, cell_count), "float64"),
        "training_event_count": ((issue_count, 2), "int32"),
        "recent_event_count": ((issue_count, 2), "int32"),
        "recent_fallback": ((issue_count, 2), "uint8"),
    }
    if set(arrays) != set(schema):
        raise InputSensitivityError("C2A NPZ fields differ from the location-only schema")
    for name, (shape, dtype) in schema.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise InputSensitivityError(f"C2A array shape or dtype changed: {name}")
    mass = arrays["location_relative_mass"]
    counts, recent = arrays["training_event_count"], arrays["recent_event_count"]
    if (
        not np.isfinite(mass).all()
        or np.any(mass < 0.0)
        or not np.allclose(mass.sum(axis=2), 1.0, atol=1e-12, rtol=0.0)
        or np.any(np.diff(arrays["issue_time_us"]) <= 0)
        or np.any(counts < 0)
        or np.any(recent < 0)
        or np.any(recent > counts)
        or not np.array_equal(arrays["recent_fallback"], (recent == 0).astype(np.uint8))
    ):
        raise InputSensitivityError("invalid mass, issue axis, training count, or fallback")
    for treatment in range(2):
        rows = recent[:, treatment] == 0
        if not np.array_equal(mass[rows, treatment * 3 + 1], mass[rows, treatment * 3 + 2]):
            raise InputSensitivityError("empty recent history must make L3 exactly equal to L2")


def _verify_fold(
    output: Path,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    cell_count: int,
) -> None:
    if record["fold_id"] not in DEVELOPMENT_FOLD_IDS or record["issue_count"] != 29:
        raise InputSensitivityError("completed fold identity or exposure count changed")
    npz = _checked(output, {"path": record["npz_path"], "sha256": record["npz_sha256"]})
    diag_path = _checked(
        output,
        {
            "path": record["diagnostics_path"],
            "sha256": record["diagnostics_sha256"],
        },
    )
    diagnostic = _json(diag_path)
    if diagnostic["identity"] != identity or diagnostic["fold_id"] != record["fold_id"]:
        raise InputSensitivityError("completed fold belongs to a different frozen run")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _validate_arrays(arrays, record["issue_count"], cell_count)
    if not np.array_equal(
        arrays["issue_time_us"], [row["issue_time_us"] for row in diagnostic["issues"]]
    ):
        raise InputSensitivityError("completed fold issue diagnostics are misaligned")


def _run_fold(
    inputs: S1RunnerInputs,
    protocol: Mapping[str, Any],
    identity: Mapping[str, Any],
    output: Path,
    partition: LocalSupportBasePartition,
    event_cells: Mapping[str, str],
    fold: str,
) -> dict[str, Any]:
    directory = output / "folds" / fold
    complete = directory / "complete.json"
    if complete.is_file():
        record = _json(complete)
        _verify_fold(output, record, identity, inputs.location_grid.cell_count)
        print(f"{fold}: reused verified completed fold", flush=True)
        return record
    directory.mkdir(parents=True, exist_ok=True)
    issues = tuple(
        row.issue_time_utc
        for row in inputs.outer_issues
        if row.fold_id == fold and row.horizon_days == 30 and row.primary_exposure_selected
    )
    if len(issues) != 29 or len(set(issues)) != 29:
        raise InputSensitivityError(
            "C2A requires the same 29 primary 30-day exposures in each fold"
        )
    issue_us = np.asarray([_epoch_us(issue) for issue in issues], dtype=np.int64)
    c0, c0_identity = _load_c0_fold(inputs.project_root, protocol, fold, issue_us)
    mask_path = _checked(inputs.project_root, protocol["parent_artifacts"]["C1_support_cells"])
    statuses, coverage = _load_mask(mask_path, partition, fold)
    arrays: dict[str, NDArray[Any]] = {
        "issue_time_us": issue_us,
        "location_relative_mass": np.empty(
            (29, 6, inputs.location_grid.cell_count), dtype=np.float64
        ),
        "training_event_count": np.zeros((29, 2), dtype=np.int32),
        "recent_event_count": np.zeros((29, 2), dtype=np.int32),
        "recent_fallback": np.zeros((29, 2), dtype=np.uint8),
    }
    diagnostics = []
    for index, issue in enumerate(issues):
        history = causal_catalog_histories(inputs.catalog, issue)["m4_plus"]
        if not np.all(c0["source_count"][index] == history.event_count):
            raise InputSensitivityError(
                "reconstructed C0 visible input differs from saved source count"
            )
        recent_count = int(
            np.count_nonzero(history.origin_time_us > issue_us[index] - 30 * _DAY_US)
        )
        if bool(c0["recent_fallback"][index]) != (recent_count == 0):
            raise InputSensitivityError(
                "reconstructed C0 recent window differs from saved fallback"
            )
        event_statuses = [statuses[event_cells[event_id]] for event_id in history.event_ids]
        row: dict[str, Any] = {
            "issue_time_us": int(issue_us[index]),
            "issue_time_utc": issue.isoformat(),
            "data_cutoff_utc": history.data_cutoff_utc.isoformat(),
            "C0_training_event_count": history.event_count,
            "C0_training_event_ids_sha256": _event_digest(history.event_ids),
            "treatments": {},
        }
        cache: dict[tuple[str, ...], NDArray[np.float64]] = {}
        for treatment, name in enumerate(TREATMENTS):
            keep = np.asarray(
                [
                    status != "unsupported" if treatment == 0 else status == "supported"
                    for status in event_statuses
                ],
                dtype=np.bool_,
            )
            mass, detail = _predict_treatment(
                history,
                keep,
                inputs.location_grid,
                _fixed_parameters(protocol)[fold],
                c0["mass"][index],
                cache,
            )
            arrays["location_relative_mass"][index, treatment * 3 : treatment * 3 + 3] = mass
            for key in ("training_event_count", "recent_event_count", "recent_fallback"):
                arrays[key][index, treatment] = detail[key]
            row["treatments"][name] = detail
        diagnostics.append(row)
        if (index + 1) % 5 == 0 or index == len(issues) - 1:
            print(f"{fold}: {index + 1}/29 issues predicted; no outcomes read", flush=True)
    _validate_arrays(arrays, 29, inputs.location_grid.cell_count)
    attempt = uuid.uuid4().hex[:12]
    npz = directory / f"predictions_{attempt}.npz"
    diag = directory / f"input_diagnostics_{attempt}.json"
    with npz.open("xb") as stream:
        np.savez(
            stream,
            issue_time_us=arrays["issue_time_us"],
            location_relative_mass=arrays["location_relative_mass"],
            training_event_count=arrays["training_event_count"],
            recent_event_count=arrays["recent_event_count"],
            recent_fallback=arrays["recent_fallback"],
        )
    _write_json(
        diag,
        {
            "identity": identity,
            "fold_id": fold,
            "coverage": coverage,
            "C0_prediction_source": c0_identity,
            "issues": diagnostics,
        },
    )
    record = {
        "fold_id": fold,
        "issue_count": 29,
        "npz_path": npz.relative_to(output).as_posix(),
        "npz_sha256": _sha(npz),
        "diagnostics_path": diag.relative_to(output).as_posix(),
        "diagnostics_sha256": _sha(diag),
    }
    _verify_fold(output, record, identity, inputs.location_grid.cell_count)
    _write_json(complete, record)
    print(f"{fold}: complete checkpoint saved and verified", flush=True)
    return record


def verify_prediction_manifest(project_root: Path, output_root: Path) -> dict[str, Any]:
    """Verify all four saved location curves without loading any outcomes."""
    project, output = Path(project_root).resolve(), Path(output_root).resolve()
    protocol = _load_protocol(project)
    identity = _identity(protocol)
    manifest = _json(output / "prediction_manifest.json")
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise InputSensitivityError("prediction manifest frozen identity changed")
    records = manifest["folds"]
    if tuple(record["fold_id"] for record in records) != DEVELOPMENT_FOLD_IDS:
        raise InputSensitivityError("scoring requires all four ordered C2A development folds")
    for record in records:
        _verify_fold(output, record, identity, protocol["development_design"]["grid_cell_count"])
    return manifest


def run_prediction_phase(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path | None = None,
    workers: int = 2,
) -> Path:
    """Save six fixed-parameter location surfaces per issue, then a four-fold manifest."""
    if type(workers) is not int or not 1 <= workers <= 3:
        raise InputSensitivityError("C2A supports only one to three fold threads")
    for name in _NUMERICAL_ENV:
        os.environ[name] = "1"
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    project = Path(project_root).resolve()
    protocol = _load_protocol(project)
    output = (
        Path(output_root).resolve()
        if output_root is not None
        else _scoped(project, protocol["execution"]["output_root"])
    )
    if not output.is_relative_to(project / "outputs"):
        raise InputSensitivityError(
            "C2A outputs must remain under this worktree's outputs directory"
        )
    identity = _identity(protocol)
    manifest_path = output / "prediction_manifest.json"
    if manifest_path.is_file():
        verify_prediction_manifest(project, output)
        return manifest_path
    identity_path = output / "prediction_identity.json"
    output.mkdir(parents=True, exist_ok=True)
    if identity_path.is_file():
        if _json(identity_path) != identity:
            raise InputSensitivityError(
                "existing prediction directory has a different frozen identity"
            )
    else:
        if (output / "folds").exists():
            raise InputSensitivityError("refusing to reuse an unowned old prediction directory")
        _write_json(identity_path, identity)
    inputs = load_s1_runner_inputs(project_root=project, data_root=data_root)
    partition = build_local_support_base_partition(inputs.spatial_domain.study_area_equal_area)
    if not math.isclose(
        partition.total_area_m2 / 1e6,
        inputs.location_grid.total_area_km2,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise InputSensitivityError("training partition changed the national domain")
    issues = [
        row.issue_time_utc
        for row in inputs.outer_issues
        if row.horizon_days == 30 and row.primary_exposure_selected
    ]
    event_cells = _locate_training_events(inputs, partition, _epoch_us(max(issues)) - _DAY_US)
    print(
        f"C2A prediction started: {workers} fold threads, 116 issues, six location curves",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="C2A-fold") as executor:
        futures = {
            fold: executor.submit(
                _run_fold, inputs, protocol, identity, output, partition, event_cells, fold
            )
            for fold in DEVELOPMENT_FOLD_IDS
        }
        records = [futures[fold].result() for fold in DEVELOPMENT_FOLD_IDS]
    _write_json(
        manifest_path,
        {
            **identity,
            "folds": records,
            "issue_count_total": 116,
            "created_utc": datetime.now(UTC).isoformat(),
            "outcomes_read": False,
        },
    )
    verify_prediction_manifest(project, output)
    return manifest_path
