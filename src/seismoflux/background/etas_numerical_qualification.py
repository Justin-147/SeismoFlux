# ruff: noqa: RUF001
"""One bounded, target-blind ETAS numerical qualification."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import platform
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import yaml

from seismoflux.background.adapters import (
    build_etas_model_spec,
    build_etas_parameter_bounds,
    build_optimizer_options,
    build_stability_thresholds,
    point_area_quadrature_from_grid,
)
from seismoflux.background.catalog import (
    EarthquakeCatalog,
    load_earthquake_catalog,
    load_study_area,
    utc_timestamp_to_day,
)
from seismoflux.background.completeness import CATALOG_ANCHOR_UTC
from seismoflux.background.config import (
    BackgroundConfig,
    load_background_protocol,
)
from seismoflux.background.etas_fit import (
    ETASFitResult,
    ETASLikelihoodProblem,
    ETASModelSpec,
    ETASParameterBounds,
    ETASStartResult,
    OptimizerOptions,
    StabilityThresholds,
    audit_stability,
    etas_objective,
    fit_etas,
    optimizer_start,
    three_point_gradient,
)
from seismoflux.background.grid import EqualAreaGridFamily
from seismoflux.background.local_support_manifest import (
    load_background_local_support_manifest,
)
from seismoflux.background.local_support_runtime import build_local_support_runtime
from seismoflux.background.pipeline_etas import _grid_gate_evidence, _KDEBackgroundDensity
from seismoflux.background.poisson import SpatialQuadrature, fit_spatial_poisson_family
from seismoflux.background.workflow import (
    build_local_support_etas_parent_roles,
    catalog_completeness_events,
    catalog_etas_events,
    historical_training_mask,
)
from seismoflux.config import load_config, resolve_project_path

SNAPSHOT_ORDER = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
Progress = Callable[[str], None]
_SENSITIVITY = frozenset({"fold_1", "fold_3"})
_FROZEN_PROTOCOL_SHA256 = "dc602e6f3e543d124e7e3d4b363ac45bdfa1ba7d3773f13bf816035cf10b51c6"
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "event_id",
        "event_ids",
        "physical_event_id",
        "longitude",
        "latitude",
        "x_km",
        "y_km",
        "absolute_local_path",
        "anomaly_feature",
        "target_label",
        "score",
    }
)


class Fitter(Protocol):
    def __call__(
        self,
        problem: ETASLikelihoodProblem,
        spec: ETASModelSpec,
        *,
        root_seed: int,
        protocol_version: str,
        model_id: str,
        bounds: ETASParameterBounds,
        options: OptimizerOptions,
        thresholds: StabilityThresholds,
    ) -> ETASFitResult: ...


@dataclass(frozen=True, slots=True)
class QualificationProtocol:
    path: Path
    root: Path
    raw: Mapping[str, Any]
    sha256: str

    def output(self, name: str) -> Path:
        value = _mapping(self.raw["outputs"], "outputs").get(name)
        return _relative(self.root, value, f"outputs.{name}")

    @property
    def attempt_root(self) -> Path:
        value = _mapping(self.raw["attempt"], "attempt").get("root")
        return _relative(self.root, value, "attempt.root")


@dataclass(frozen=True, slots=True)
class PreparedSnapshot:
    """Private in-memory fit input; only its hashes and diagnostics are public."""

    snapshot_id: str
    fit_end_utc: str
    scientific_fit_input_sha256: str
    membership_sha256: str
    maximum_origin_time: str
    maximum_available_at: str
    fit_event_count: int
    parent_event_count: int
    kde_training_event_count: int
    support_id: str
    compensator_domain_id: str
    model_id: str
    starts: tuple[tuple[float, ...], ...]
    problem: ETASLikelihoodProblem
    sensitivity_problem: ETASLikelihoodProblem | None
    spec: ETASModelSpec
    bounds: ETASParameterBounds
    options: OptimizerOptions
    thresholds: StabilityThresholds
    grid_family: EqualAreaGridFamily | None = None
    seed_protocol_version: str = "0.2.1"


@dataclass(frozen=True, slots=True)
class PreparedQualification:
    protocol: QualificationProtocol
    snapshots: tuple[PreparedSnapshot, ...]
    catalog_sha256: str
    study_area_sha256: str
    support_manifest_sha256: str
    start_manifest_sha256: str


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be a sequence")
    return value


def _relative(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes project root") from error
    return path


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(
    path: str | Path = "configs/background_etas_numerical_qualification.yaml",
) -> QualificationProtocol:
    path = Path(path).resolve(strict=True)
    payload_bytes = path.read_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if payload_sha256 != _FROZEN_PROTOCOL_SHA256:
        raise ValueError("qualification protocol bytes differ from the Q0 tag")
    try:
        raw = _mapping(yaml.safe_load(payload_bytes.decode()), "protocol")
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError("qualification protocol is not valid UTF-8 YAML") from error
    blind = _mapping(raw.get("target_blindness"), "target_blindness")
    snapshots = _mapping(raw.get("snapshots"), "snapshots")
    optimizer = _mapping(raw.get("optimizer"), "optimizer")
    attempt = _mapping(raw.get("attempt"), "attempt")
    resources = _mapping(raw.get("resources"), "resources")
    forbidden_reads = (
        "anomaly_feature_read",
        "stage4_target_read",
        "assessment_event_read",
        "prior_score_read",
        "locked_test_read_or_run",
    )
    if (
        raw.get("protocol_version") != "0.2.2"
        or raw.get("stage") != "2-ETAS-Q"
        or any(blind.get(name) is not False for name in forbidden_reads)
        or tuple(_sequence(snapshots.get("order"), "snapshots.order")) != SNAPSHOT_ORDER
        or snapshots.get("unsupported_parent_sensitivity_optimizer_call_count") != 0
        or optimizer.get("method") != "L-BFGS-B"
        or optimizer.get("starts_per_snapshot") != 5
        or tuple(_sequence(optimizer.get("start_indices"), "start_indices")) != (0, 1, 2, 3, 4)
        or optimizer.get("retry_with_new_seed") is not False
        or attempt.get("attempt_id") != "etas_qualification_q1"
        or attempt.get("exactly_one_scientific_attempt") is not True
        or resources.get("max_workers") != 1
        or resources.get("blas_threads") != 1
    ):
        raise ValueError("qualification protocol differs from frozen Q1")
    entries = tuple(
        _mapping(item, "snapshot entry")
        for item in _sequence(snapshots.get("entries"), "snapshots.entries")
    )
    if tuple(item.get("snapshot_id") for item in entries) != SNAPSHOT_ORDER:
        raise ValueError("snapshot entries differ from frozen order")
    return QualificationProtocol(
        path=path,
        root=path.parent.parent.resolve(),
        raw=raw,
        sha256=payload_sha256,
    )


def _load_target_blind_background(project_path: Path, background_path: Path) -> BackgroundConfig:
    """Load frozen configuration models without opening referenced data inputs."""

    project = load_config(project_path)
    referenced_background = resolve_project_path(
        project_path, project.config_files.background
    ).resolve()
    if referenced_background != background_path.resolve():
        raise ValueError("project configuration points to another background protocol")
    background = load_background_protocol(background_path)
    shared = (
        (background.randomness.root_seed, project.project.random_seed, "root seed"),
        (
            background.integration.equal_area_crs,
            project.study_area.equal_area_crs,
            "equal-area CRS",
        ),
        (
            background.inputs.include_external_trigger_buffer_km,
            project.study_area.include_external_trigger_buffer_km,
            "external trigger buffer",
        ),
        (background.time.horizons_days, project.forecast.horizons_days, "forecast horizons"),
        (
            background.integration.grid_cells_km,
            project.integration.convergence_cells_km,
            "integration grids",
        ),
    )
    for actual, expected, name in shared:
        if actual != expected:
            raise ValueError(f"project and background {name} differ")
    return background


def _validate_runtime_numeric_contract(
    protocol: QualificationProtocol, background: BackgroundConfig
) -> None:
    """Bind executable numerical objects to the exact target-blind Q0 protocol."""

    model = _mapping(protocol.raw["model"], "model")
    spatial = _mapping(model["spatial_kernel"], "model.spatial_kernel")
    temporal = _mapping(model["temporal_kernel"], "model.temporal_kernel")
    actual_model = (
        background.etas.magnitude_model.upper_magnitude,
        background.etas.spatial_kernel.d_km2,
        background.etas.spatial_kernel.q,
        background.etas.spatial_kernel.gamma,
        background.etas.spatial_kernel.support_radius_km,
        background.etas.temporal_kernel.history_parent_cutoff_days,
        background.etas.temporal_kernel.form,
        background.etas.branching_ratio.maximum,
        background.integration.grid_cells_km,
    )
    frozen_model = (
        model["maximum_magnitude"],
        spatial["d_km2"],
        spatial["q"],
        spatial["gamma"],
        spatial["cutoff_km"],
        temporal["history_parent_cutoff_days"],
        temporal["form"],
        model["branching_ratio_maximum"],
        tuple(_sequence(model["quadrature_grid_km"], "model.quadrature_grid_km")),
    )
    if actual_model != frozen_model or model["background_kde_bandwidth_km"] != 75.0:
        raise ValueError("runtime ETAS model differs from qualification protocol")

    bounds = build_etas_parameter_bounds(background)
    frozen_bounds = _mapping(model["parameter_bounds"], "model.parameter_bounds")
    for name in ("background_rate_per_day", "productivity_k", "alpha", "c_days", "p"):
        if tuple(getattr(bounds, name)) != tuple(
            _sequence(frozen_bounds[name], f"model.parameter_bounds.{name}")
        ):
            raise ValueError(f"runtime ETAS bound differs for {name}")

    optimizer = _mapping(protocol.raw["optimizer"], "optimizer")
    options = _mapping(optimizer["options"], "optimizer.options")
    runtime_options = build_optimizer_options(background)
    if (
        optimizer["method"] != "L-BFGS-B"
        or background.etas.optimizer != "scipy_lbfgsb"
        or optimizer["root_seed"] != background.randomness.root_seed
        or optimizer["bit_generator"] != background.randomness.bit_generator
        or tuple(_sequence(optimizer["start_indices"], "optimizer.start_indices"))
        != background.etas.multi_start_indices
        or any(getattr(runtime_options, name) != options[name] for name in asdict(runtime_options))
    ):
        raise ValueError("runtime optimizer differs from qualification protocol")

    qualification = _mapping(protocol.raw["qualification"], "qualification")
    thresholds = build_stability_thresholds(background)
    threshold_pairs = (
        (thresholds.minimum_converged_starts, qualification["minimum_converged_starts"]),
        (
            thresholds.gradient_infinity_norm_maximum,
            qualification["gradient_infinity_norm_maximum"],
        ),
        (
            thresholds.best_three_relative_objective_range_maximum,
            qualification["best_three_relative_objective_range_maximum"],
        ),
        (
            thresholds.transformed_parameter_maximum_range,
            qualification["best_three_transformed_parameter_range_maximum"],
        ),
        (
            thresholds.hessian_minimum_eigenvalue,
            qualification["hessian_minimum_eigenvalue"],
        ),
        (
            thresholds.hessian_condition_number_maximum,
            qualification["hessian_condition_number_maximum"],
        ),
        (thresholds.hessian_relative_step, qualification["hessian_relative_step"]),
        (
            background.integration.relative_expected_count_tolerance,
            qualification["grid_25_to_12_5_expected_count_relative_difference_maximum"],
        ),
        (
            background.integration.density_l1_tolerance,
            qualification["grid_25_to_12_5_density_l1_maximum"],
        ),
    )
    if any(actual != expected for actual, expected in threshold_pairs):
        raise ValueError("runtime qualification thresholds differ from Q0 protocol")


def causal_fit_masks(
    catalog: EarthquakeCatalog,
    *,
    supported: object,
    parents: object,
    mc: float,
    fit_start_day: float,
    fit_end_day: float,
    history_start_day: float,
    parent_cutoff_days: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    supported_mask = np.asarray(supported, dtype=np.bool_)
    parent_mask = np.asarray(parents, dtype=np.bool_)
    if supported_mask.shape != (len(catalog),) or parent_mask.shape != (len(catalog),):
        raise ValueError("fit masks must align with catalog")
    history_start = max(history_start_day, fit_start_day - parent_cutoff_days)
    targets = (
        supported_mask
        & (catalog.magnitude >= mc)
        & (catalog.origin_day > fit_start_day)
        & (catalog.origin_day <= fit_end_day)
        & (catalog.available_day <= fit_end_day)
    )
    history = (
        parent_mask
        & (catalog.origin_day >= history_start)
        & (catalog.origin_day <= fit_end_day)
        & (catalog.available_day <= fit_end_day)
    )
    if np.any(targets & ~history):
        raise ValueError("every fit target must also be in causal parent history")
    return np.asarray(targets), np.asarray(history)


def classify_snapshot(
    stability: bool, branching: bool, three_grid: bool
) -> tuple[str, tuple[str, ...]]:
    failures = tuple(
        name
        for name, passed in (
            ("numerical_stability_failed", stability),
            ("branching_ratio_failed_or_unavailable", branching),
            ("three_grid_failed_or_unavailable", three_grid),
        )
        if not passed
    )
    return ("evaluable", ()) if not failures else ("not_evaluable", failures)


def atomic_create(path: Path, payload: bytes) -> None:
    """Create a complete immutable file; abandoned temporary files are ignored."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != payload:
                raise ValueError(f"immutable artifact differs: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_create(path, canonical_bytes(value))


def require_public_safe(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"forbidden public field: {key}")
            require_public_safe(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            require_public_safe(item)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if "data/processed/" in normalized or normalized.startswith(("c:/", "d:/")):
            raise ValueError("public payload contains local path")


def _verified_path(protocol: QualificationProtocol, key: str, source_root: Path) -> Path:
    frozen = _mapping(protocol.raw["frozen_identity"], "frozen_identity")
    relative = protocol.raw.get("blueprint") if key == "blueprint" else frozen.get(f"{key}_path")
    expected = frozen.get(f"{key}_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"frozen {key} identity is incomplete")
    root = (
        source_root if relative.replace("\\", "/").startswith("data/processed/") else protocol.root
    )
    path = _relative(root, relative, f"frozen_identity.{key}_path")
    if file_sha256(path) != expected:
        raise ValueError(f"frozen {key} hash mismatch")
    return path


def _timestamp(day: float) -> str:
    return datetime.fromtimestamp(day * 86_400.0, tz=UTC).isoformat().replace("+00:00", "Z")


def _events_payload(events: Sequence[Any]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            event.event_id,
            float(event.time_days).hex(),
            float(event.available_time_days).hex(),
            float(event.x_km).hex(),
            float(event.y_km).hex(),
            float(event.magnitude).hex(),
            event.inside_study_area,
            event.inside_parent_domain,
        )
        for event in events
    )


def _frozen_starts(
    manifest: Mapping[str, Any], snapshot_id: str, bounds: ETASParameterBounds
) -> tuple[str, tuple[tuple[float, ...], ...]]:
    entry = next(
        _mapping(item, "start snapshot")
        for item in _sequence(manifest["snapshots"], "start snapshots")
        if isinstance(item, Mapping) and item.get("snapshot_id") == snapshot_id
    )
    if manifest.get("seed_protocol_version") != "0.2.1" or manifest.get("root_seed") != 147:
        raise ValueError("frozen starts must use seed protocol 0.2.1 and seed 147")
    model_id = cast(str, entry["model_id"])
    starts: list[tuple[float, ...]] = []
    for index, item in enumerate(_sequence(entry["starts"], "starts")):
        row = _mapping(item, "start")
        expected_hex = tuple(_sequence(row["transformed_hex"], "transformed_hex"))
        generated = optimizer_start(
            bounds.transformed(),
            root_seed=147,
            protocol_version="0.2.1",
            model_id=model_id,
            start_index=index,
        )
        generated_hex = tuple(float(value).hex() for value in generated)
        if row.get("start_index") != index or generated_hex != expected_hex:
            raise ValueError(f"{snapshot_id} start {index} differs from frozen manifest")
        starts.append(tuple(float.fromhex(cast(str, value)) for value in expected_hex))
    if len(starts) != 5:
        raise ValueError("each snapshot requires exactly five starts")
    return model_id, tuple(starts)


def _prepare_one(
    protocol: QualificationProtocol,
    entry: Mapping[str, Any],
    background: BackgroundConfig,
    catalog: EarthquakeCatalog,
    runtime: Any,
    start_manifest: Mapping[str, Any],
) -> PreparedSnapshot:
    snapshot_id, fit_end_utc = cast(str, entry["snapshot_id"]), cast(str, entry["fit_end_utc"])
    fit_end = utc_timestamp_to_day(fit_end_utc)
    fit_start_utc = (
        background.etas.final_fit_start_utc
        if snapshot_id == "final_validation"
        else background.etas.historical_fold_fit_start_utc
    )
    fit_start = utc_timestamp_to_day(fit_start_utc)
    history_start = utc_timestamp_to_day(background.etas.history_start_utc)
    local = runtime.snapshot(snapshot_id)
    if (
        local.support.support_id != entry["support_id"]
        or local.compensator_domain_id != entry["compensator_domain_id"]
        or local.support.retained_area_fraction != entry["retained_area_fraction"]
    ):
        raise ValueError(f"{snapshot_id} support differs from Q1")
    supported = local.supported_mask
    unsupported = np.asarray(catalog.inside_study_area & ~supported)
    roles = build_local_support_etas_parent_roles(
        catalog,
        supported_domain_mask=supported,
        unsupported_domain_mask=unsupported,
        common_mc=local.support.common_mc,
        prevalidated_unsupported_parent_mask=np.asarray(
            local.etas_primary_parent_role_mask & unsupported
        ),
    )
    spec = build_etas_model_spec(
        background,
        selected_mc=local.support.common_mc,
        aki_b_value=local.support.retained_selected_aki_b_value,
    )
    targets_mask, parents_mask = causal_fit_masks(
        catalog,
        supported=supported,
        parents=roles.parent_mask,
        mc=local.support.common_mc,
        fit_start_day=fit_start,
        fit_end_day=fit_end,
        history_start_day=history_start,
        parent_cutoff_days=spec.history_parent_cutoff_days,
    )
    target_domain = np.asarray(supported)
    targets = catalog_etas_events(
        catalog,
        targets_mask,
        inside_target_domain_mask=target_domain,
        inside_parent_domain_mask=roles.parent_mask,
    )
    parents = catalog_etas_events(
        catalog,
        parents_mask,
        inside_target_domain_mask=target_domain,
        inside_parent_domain_mask=roles.parent_mask,
    )
    sensitivity_parents = None
    if snapshot_id in _SENSITIVITY:
        excluded_roles = roles.excluding_unsupported_parents()
        _, excluded_mask = causal_fit_masks(
            catalog,
            supported=supported,
            parents=excluded_roles.parent_mask,
            mc=local.support.common_mc,
            fit_start_day=fit_start,
            fit_end_day=fit_end,
            history_start_day=history_start,
            parent_cutoff_days=spec.history_parent_cutoff_days,
        )
        sensitivity_parents = catalog_etas_events(
            catalog,
            excluded_mask,
            inside_target_domain_mask=target_domain,
            inside_parent_domain_mask=excluded_roles.parent_mask,
        )
    training = historical_training_mask(
        catalog, minimum_magnitude=local.support.common_mc, fit_end_day=fit_end
    )
    training &= supported
    indices = np.flatnonzero(training)
    duration = fit_end - CATALOG_ANCHOR_UTC.timestamp() / 86_400.0
    quadrature = SpatialQuadrature.from_grid(local.grid_family.at(12.5))
    kde = fit_spatial_poisson_family(
        catalog.x_km[indices],
        catalog.y_km[indices],
        training_duration_days=duration,
        normalization_quadrature=quadrature,
        bandwidths_km=(75.0,),
    )[75.0]
    integrator = point_area_quadrature_from_grid(local.grid_family.at(12.5))
    problem = ETASLikelihoodProblem(
        fit_start, fit_end, targets, parents, _KDEBackgroundDensity(kde), integrator
    )
    sensitivity_problem = (
        ETASLikelihoodProblem(
            fit_start,
            fit_end,
            targets,
            sensitivity_parents,
            _KDEBackgroundDensity(kde),
            integrator,
        )
        if sensitivity_parents is not None
        else None
    )
    counts = (len(targets), len(parents), len(indices))
    expected = (
        entry["fit_event_count"],
        entry["parent_event_count"],
        entry["immigrant_kde_training_event_count"],
    )
    if counts != expected:
        raise ValueError(f"{snapshot_id} counts differ: observed={counts}, expected={expected}")
    bounds = build_etas_parameter_bounds(background)
    model_id, starts = _frozen_starts(start_manifest, snapshot_id, bounds)
    membership = {
        "targets": tuple(event.event_id for event in targets),
        "parents": tuple(event.event_id for event in parents),
        "kde": tuple(str(catalog.event_id[index]) for index in indices),
    }
    private = {
        "snapshot_id": snapshot_id,
        "fit_start_utc": fit_start_utc,
        "fit_end_utc": fit_end_utc,
        "support_id": local.support.support_id,
        "domain_id": local.compensator_domain_id,
        "targets": _events_payload(targets),
        "parents": _events_payload(parents),
        "sensitivity_parents": (
            _events_payload(sensitivity_parents) if sensitivity_parents is not None else None
        ),
        "kde_training": tuple(
            (
                str(catalog.event_id[index]),
                float(catalog.x_km[index]).hex(),
                float(catalog.y_km[index]).hex(),
            )
            for index in indices
        ),
        "kde_normalization_mass": float(kde.normalization_mass).hex(),
        "grids": tuple(
            (
                float(grid.spec.cell_size_km).hex(),
                tuple(
                    (
                        cell.id,
                        float(cell.representative_point.x).hex(),
                        float(cell.representative_point.y).hex(),
                        float(cell.clipped_area_m2).hex(),
                    )
                    for cell in grid.cells
                ),
            )
            for grid in local.grid_family.grids
        ),
        "spec": asdict(spec),
        "bounds": asdict(bounds),
        "starts": tuple(tuple(map(float.hex, row)) for row in starts),
    }
    selected = np.flatnonzero(targets_mask | parents_mask | training)
    maximum_origin = max(float(catalog.origin_day[index]) for index in selected)
    maximum_available = max(float(catalog.available_day[index]) for index in selected)
    if max(maximum_origin, maximum_available) > fit_end:
        raise ValueError(f"{snapshot_id} contains post-cutoff information")
    return PreparedSnapshot(
        snapshot_id=snapshot_id,
        fit_end_utc=fit_end_utc,
        scientific_fit_input_sha256=canonical_sha256(private),
        membership_sha256=canonical_sha256(membership),
        maximum_origin_time=_timestamp(maximum_origin),
        maximum_available_at=_timestamp(maximum_available),
        fit_event_count=counts[0],
        parent_event_count=counts[1],
        kde_training_event_count=counts[2],
        support_id=local.support.support_id,
        compensator_domain_id=local.compensator_domain_id,
        model_id=model_id,
        starts=starts,
        problem=problem,
        sensitivity_problem=sensitivity_problem,
        spec=spec,
        bounds=bounds,
        options=build_optimizer_options(background),
        thresholds=build_stability_thresholds(background),
        grid_family=local.grid_family,
    )


def prepare_real_inputs(
    protocol_path: str | Path = "configs/background_etas_numerical_qualification.yaml",
    *,
    source_root: str | Path | None = None,
    progress: Progress | None = None,
) -> PreparedQualification:
    """Thin real-input builder; the caller controls when processed inputs may open."""

    protocol = load_protocol(protocol_path)
    source = protocol.root if source_root is None else Path(source_root).resolve(strict=True)
    paths = {
        key: _verified_path(protocol, key, source)
        for key in (
            "blueprint",
            "parent_protocol",
            "parent_background",
            "project_config",
            "uv_lock",
            "catalog",
            "study_area",
            "support_manifest",
            "start_manifest",
            "production_fixture",
            "independent_fixture",
        )
    }
    background = _load_target_blind_background(paths["project_config"], paths["parent_background"])
    if str(background.protocol_version) != "0.2.1":
        raise ValueError("Q1 must reuse background protocol 0.2.1")
    _validate_runtime_numeric_contract(protocol, background)
    entries = tuple(
        _mapping(item, "snapshot entry")
        for item in _sequence(
            _mapping(protocol.raw["snapshots"], "snapshots")["entries"], "entries"
        )
    )
    study = load_study_area(paths["study_area"], background.integration.equal_area_crs)
    catalog = load_earthquake_catalog(
        paths["catalog"],
        study_area=study,
        external_buffer_km=background.inputs.include_external_trigger_buffer_km,
        maximum_event_time_utc=entries[-1]["fit_end_utc"],
    )
    runtime = build_local_support_runtime(
        load_background_local_support_manifest(paths["support_manifest"]),
        catalog_completeness_events(catalog),
        study_area_equal_area=study.projected,
    )
    start_manifest = _mapping(
        json.loads(paths["start_manifest"].read_text(encoding="utf-8")), "start manifest"
    )
    frozen = _mapping(protocol.raw["frozen_identity"], "frozen_identity")
    if start_manifest.get("vector_payload_sha256") != frozen["start_vector_payload_sha256"]:
        raise ValueError("start vector payload hash mismatch")
    snapshots = []
    for entry in entries:
        if progress:
            progress(f"prepare:{entry['snapshot_id']}:start")
        snapshots.append(
            _prepare_one(protocol, entry, background, catalog, runtime, start_manifest)
        )
        if progress:
            progress(f"prepare:{entry['snapshot_id']}:done")
    return PreparedQualification(
        protocol=protocol,
        snapshots=tuple(snapshots),
        catalog_sha256=cast(str, frozen["catalog_sha256"]),
        study_area_sha256=cast(str, frozen["study_area_sha256"]),
        support_manifest_sha256=cast(str, frozen["support_manifest_sha256"]),
        start_manifest_sha256=cast(str, frozen["start_manifest_sha256"]),
    )


def build_input_manifest(
    prepared: PreparedQualification, code_commit: str, code_tag: str
) -> Mapping[str, object]:
    rows = tuple(
        {
            "snapshot_id": item.snapshot_id,
            "fit_end_utc": item.fit_end_utc,
            "scientific_fit_input_sha256": item.scientific_fit_input_sha256,
            "snapshot_membership_sha256": item.membership_sha256,
            "maximum_origin_time": item.maximum_origin_time,
            "maximum_available_at": item.maximum_available_at,
            "fit_event_count": item.fit_event_count,
            "parent_event_count": item.parent_event_count,
            "immigrant_kde_training_event_count": item.kde_training_event_count,
            "support_id": item.support_id,
            "compensator_domain_id": item.compensator_domain_id,
        }
        for item in prepared.snapshots
    )
    return {
        "schema_version": 1,
        "attempt_id": "etas_qualification_q1",
        "protocol_config_sha256": prepared.protocol.sha256,
        "code_commit": code_commit,
        "code_tag": code_tag,
        "catalog_sha256": prepared.catalog_sha256,
        "study_area_sha256": prepared.study_area_sha256,
        "support_manifest_sha256": prepared.support_manifest_sha256,
        "start_manifest_sha256": prepared.start_manifest_sha256,
        "snapshot_membership_sha256": canonical_sha256(
            tuple(row["snapshot_membership_sha256"] for row in rows)
        ),
        "maximum_origin_time": max(cast(str, row["maximum_origin_time"]) for row in rows),
        "maximum_available_at": max(cast(str, row["maximum_available_at"]) for row in rows),
        "snapshots": rows,
        "target_blind": True,
    }


def _environment_seal(prepared: PreparedQualification) -> Mapping[str, object]:
    frozen = _mapping(prepared.protocol.raw["frozen_identity"], "frozen_identity")
    return {
        "schema_version": 1,
        "attempt_id": "etas_qualification_q1",
        "protocol_config_sha256": prepared.protocol.sha256,
        "environment_lock_sha256": frozen["uv_lock_sha256"],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": version("scipy"),
    }


def _environment_seal_path(protocol: QualificationProtocol) -> Path:
    return protocol.attempt_root / "environment_seal.json"


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _physical_hex(snapshot: PreparedSnapshot, terminal: Sequence[float]) -> tuple[str, ...] | None:
    try:
        value = snapshot.bounds.from_transformed(terminal)
    except ValueError:
        return None
    return tuple(
        map(
            float.hex,
            (
                value.background_rate_per_day,
                value.productivity_k,
                value.alpha,
                value.c_days,
                value.p,
            ),
        )
    )


def _recompute(
    snapshot: PreparedSnapshot,
    rows: Sequence[ETASStartResult],
    input_sha256: str,
) -> Mapping[str, object]:
    if len(rows) != 5 or tuple(row.start_index for row in rows) != (0, 1, 2, 3, 4):
        raise ValueError("snapshot requires exactly start rows 0..4")
    if tuple(tuple(row.initial_transformed) for row in rows) != snapshot.starts:
        raise ValueError("fit used non-frozen start vectors")
    objective = etas_objective(snapshot.problem, snapshot.spec, snapshot.bounds)
    bounds = snapshot.bounds.transformed()
    recalculated = []
    for row in rows:
        terminal = np.asarray(row.final_transformed)
        value = float(objective(terminal))
        try:
            gradient = three_point_gradient(
                objective,
                terminal,
                bounds,
                relative_step=snapshot.options.gradient_relative_step,
            )
            norm = float(np.linalg.norm(gradient, ord=np.inf))
        except ValueError:
            norm = math.inf
        recalculated.append(
            ETASStartResult(
                row.start_index,
                tuple(row.initial_transformed),
                tuple(row.final_transformed),
                value,
                row.scipy_converged,
                norm,
                row.iterations,
                row.function_evaluations,
                row.message,
            )
        )
    rows = tuple(recalculated)
    stability = audit_stability(objective, rows, bounds, thresholds=snapshot.thresholds)
    eligible = [row for row in rows if row.scipy_converged and math.isfinite(row.objective)]
    best = min(eligible, key=lambda row: (row.objective, row.start_index)) if eligible else None
    parameters, branching, grid = None, None, None
    branching_passed = grid_passed = False
    if best is not None:
        try:
            parameters = snapshot.bounds.from_transformed(best.final_transformed)
            branching = snapshot.spec.branching_ratio(parameters)
            branching_passed = branching < snapshot.spec.branching_ratio_maximum
        except ValueError:
            parameters = None
    if parameters is not None and stability.stable and branching_passed:
        if best is None:
            raise AssertionError("decoded parameters require a selected optimizer row")
        if snapshot.grid_family is None:
            raise ValueError("stable snapshot omitted grid family")
        evidence = _grid_gate_evidence(
            protocol_sha256=input_sha256,
            snapshot_id=snapshot.snapshot_id,
            parameter_snapshot_id=canonical_sha256(tuple(best.final_transformed)),
            problem=snapshot.problem,
            parameters=parameters,
            spec=snapshot.spec,
            grid_family=snapshot.grid_family,
        )
        grid_passed = evidence.passed
        grid = {
            "passed": evidence.passed,
            "resolutions": tuple(asdict(item) for item in evidence.resolutions),
            "comparisons": tuple(asdict(item) for item in evidence.convergence.comparisons),
        }
    sensitivity: Mapping[str, object]
    if snapshot.sensitivity_problem is None:
        sensitivity = {"status": "not_required", "objective_difference": None}
    elif best is None or parameters is None:
        sensitivity = {
            "status": "not_computable_primary_parameters_absent",
            "objective_difference": None,
        }
    else:
        secondary = etas_objective(snapshot.sensitivity_problem, snapshot.spec, snapshot.bounds)(
            np.asarray(best.final_transformed)
        )
        sensitivity = {
            "status": "computed",
            "objective_difference": float(secondary) - float(best.objective),
        }
    status, failures = classify_snapshot(stability.stable, branching_passed, grid_passed)
    public_rows = tuple(
        {
            "snapshot_id": snapshot.snapshot_id,
            "scientific_fit_input_sha256": snapshot.scientific_fit_input_sha256,
            "start_index": row.start_index,
            "initial_transformed_hex": tuple(map(float.hex, row.initial_transformed)),
            "terminal_transformed_hex": tuple(map(float.hex, row.final_transformed)),
            "terminal_physical_parameters_hex": _physical_hex(snapshot, row.final_transformed),
            "numerical_status": ("converged" if row.scipy_converged else "optimizer_not_converged"),
            "objective": _finite(row.objective),
            "gradient_infinity_norm": _finite(row.gradient_infinity_norm),
            "iterations": row.iterations,
            "function_evaluations": row.function_evaluations,
            "gate_name": "optimizer_start",
            "gate_status": (
                "passed"
                if row.scipy_converged
                and row.gradient_infinity_norm <= snapshot.thresholds.gradient_infinity_norm_maximum
                else "failed"
            ),
            "failure_code": (
                "scipy_not_converged"
                if not row.scipy_converged
                else (
                    "gradient_too_large_or_nonfinite"
                    if row.gradient_infinity_norm
                    > snapshot.thresholds.gradient_infinity_norm_maximum
                    else None
                )
            ),
        }
        for row in rows
    )
    result: Mapping[str, object] = {
        "schema_version": 1,
        "attempt_id": "etas_qualification_q1",
        "input_manifest_sha256": input_sha256,
        "snapshot_id": snapshot.snapshot_id,
        "scientific_fit_input_sha256": snapshot.scientific_fit_input_sha256,
        "start_rows": public_rows,
        "stability": {
            "stable": stability.stable,
            "converged_start_count": stability.converged_start_count,
            "best_three_relative_objective_range": (stability.best_three_relative_objective_range),
            "best_three_transformed_parameter_range": (
                stability.best_three_transformed_parameter_range
            ),
            "hessian": asdict(stability.hessian),
            "failure_reasons": stability.failure_reasons,
        },
        "branching_ratio": branching,
        "branching_ratio_passed": branching_passed,
        "three_grid": grid,
        "unsupported_parent_objective_sensitivity": sensitivity,
        "parameter_snapshot": (
            {
                "terminal_transformed_hex": tuple(map(float.hex, best.final_transformed)),
                "physical_parameters_hex": _physical_hex(snapshot, best.final_transformed),
            }
            if status == "evaluable" and best is not None
            else None
        ),
        "qualification_status": status,
        "failure_codes": failures,
    }
    require_public_safe(result)
    return result


def _rows(payload: Mapping[str, Any]) -> tuple[ETASStartResult, ...]:
    return tuple(
        ETASStartResult(
            int(row["start_index"]),
            tuple(float.fromhex(value) for value in row["initial_transformed_hex"]),
            tuple(float.fromhex(value) for value in row["terminal_transformed_hex"]),
            math.inf if row["objective"] is None else float(row["objective"]),
            row["numerical_status"] == "converged",
            (
                math.inf
                if row["gradient_infinity_norm"] is None
                else float(row["gradient_infinity_norm"])
            ),
            int(row["iterations"]),
            int(row["function_evaluations"]),
            "public_terminal",
        )
        for row in _sequence(payload["start_rows"], "start_rows")
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _snapshot_path(protocol: QualificationProtocol, snapshot_id: str) -> Path:
    return protocol.attempt_root / "snapshots" / f"{snapshot_id}.json"


def _snapshot_receipt_path(protocol: QualificationProtocol, snapshot_id: str) -> Path:
    return protocol.attempt_root / "snapshot_receipts" / f"{snapshot_id}.sha256"


def run_prepared(
    prepared: PreparedQualification,
    *,
    code_commit: str,
    code_tag: str,
    fitter: Fitter = fit_etas,
    progress: Progress | None = None,
) -> Mapping[str, object]:
    if len(code_commit) != 40 or code_tag != prepared.protocol.raw["publication"]["code_tag"]:
        raise ValueError("run must use the frozen code commit and tag")
    input_manifest = build_input_manifest(prepared, code_commit, code_tag)
    input_sha = canonical_sha256(input_manifest)
    atomic_json(prepared.protocol.output("input_manifest"), input_manifest)
    environment = _environment_seal(prepared)
    atomic_json(_environment_seal_path(prepared.protocol), environment)
    results, hashes = [], {}
    for snapshot in prepared.snapshots:
        path = _snapshot_path(prepared.protocol, snapshot.snapshot_id)
        receipt_path = _snapshot_receipt_path(prepared.protocol, snapshot.snapshot_id)
        if not path.exists() and receipt_path.exists():
            raise ValueError(f"{snapshot.snapshot_id} completed snapshot was deleted")
        created = False
        created_result: Mapping[str, object] | None = None
        if not path.exists():
            if progress:
                progress(f"fit:{snapshot.snapshot_id}:start")
            fit = fitter(
                snapshot.problem,
                snapshot.spec,
                root_seed=147,
                protocol_version=snapshot.seed_protocol_version,
                model_id=snapshot.model_id,
                bounds=snapshot.bounds,
                options=snapshot.options,
                thresholds=snapshot.thresholds,
            )
            created_result = _recompute(snapshot, fit.start_results, input_sha)
            expected_stability = _mapping(created_result["stability"], "stability")
            observed_stability = {
                "stable": fit.stability.stable,
                "converged_start_count": fit.stability.converged_start_count,
                "best_three_relative_objective_range": (
                    fit.stability.best_three_relative_objective_range
                ),
                "best_three_transformed_parameter_range": (
                    fit.stability.best_three_transformed_parameter_range
                ),
                "hessian": asdict(fit.stability.hessian),
                "failure_reasons": fit.stability.failure_reasons,
            }
            if canonical_bytes(expected_stability) != canonical_bytes(observed_stability):
                raise ValueError("fit_etas stability differs from independent recalculation")
            atomic_json(path, created_result)
            created = True
            if progress:
                progress(f"fit:{snapshot.snapshot_id}:done")
        result = _read_json(path)
        if created:
            if created_result is None:
                raise AssertionError("new snapshot result was not retained in memory")
            if canonical_bytes(result) != canonical_bytes(created_result):
                raise AssertionError("new snapshot did not reopen canonically")
        else:
            expected = _recompute(snapshot, _rows(result), input_sha)
            if canonical_bytes(result) != canonical_bytes(expected):
                raise ValueError(f"{snapshot.snapshot_id} result is incompatible or tampered")
        results.append(result)
        digest = file_sha256(path)
        atomic_create(receipt_path, f"{digest}\n".encode())
        hashes[snapshot.snapshot_id] = digest
    status = (
        "evaluable"
        if all(result["qualification_status"] == "evaluable" for result in results)
        else "not_evaluable"
    )
    manifest: Mapping[str, object] = {
        "schema_version": 1,
        "attempt_id": "etas_qualification_q1",
        "input_manifest_sha256": input_sha,
        "protocol_config_sha256": prepared.protocol.sha256,
        "code_commit": code_commit,
        "code_tag": code_tag,
        "python_version": environment["python_version"],
        "numpy_version": environment["numpy_version"],
        "scipy_version": environment["scipy_version"],
        "snapshot_results_sha256": hashes,
        "qualification_status": status,
        "snapshot_statuses": {
            result["snapshot_id"]: result["qualification_status"] for result in results
        },
        "completed_snapshot_count": 5,
        "completed_start_row_count": 25,
        "assessment_event_read": False,
        "anomaly_feature_read": False,
        "locked_test_read_or_run": False,
    }
    atomic_json(prepared.protocol.output("result_manifest"), manifest)
    publish_views(prepared.protocol, manifest, results)
    return manifest


def verify_prepared(
    prepared: PreparedQualification,
    *,
    code_commit: str,
    code_tag: str,
    persist: bool = True,
) -> Mapping[str, object]:
    expected_environment = _environment_seal(prepared)
    if canonical_bytes(_read_json(_environment_seal_path(prepared.protocol))) != canonical_bytes(
        expected_environment
    ):
        raise ValueError("attempt environment differs from the frozen environment seal")
    input_manifest = _read_json(prepared.protocol.output("input_manifest"))
    expected_input_manifest = build_input_manifest(prepared, code_commit, code_tag)
    if canonical_bytes(input_manifest) != canonical_bytes(expected_input_manifest):
        raise ValueError("input manifest differs from independently rebuilt inputs")
    input_sha = canonical_sha256(expected_input_manifest)
    result = _read_json(prepared.protocol.output("result_manifest"))
    if result["input_manifest_sha256"] != input_sha:
        raise ValueError("result is bound to another input manifest")
    recorded_hashes = _mapping(result["snapshot_results_sha256"], "snapshot hashes")
    statuses, hashes, payloads = {}, {}, []
    for snapshot in prepared.snapshots:
        path = _snapshot_path(prepared.protocol, snapshot.snapshot_id)
        payload, digest = _read_json(path), file_sha256(path)
        if recorded_hashes.get(snapshot.snapshot_id) != digest:
            raise ValueError(f"{snapshot.snapshot_id} hash mismatch")
        expected = _recompute(snapshot, _rows(payload), input_sha)
        if canonical_bytes(payload) != canonical_bytes(expected):
            raise ValueError(f"{snapshot.snapshot_id} independent recalculation failed")
        statuses[snapshot.snapshot_id] = payload["qualification_status"]
        hashes[snapshot.snapshot_id] = digest
        payloads.append(payload)
    status = (
        "evaluable" if all(value == "evaluable" for value in statuses.values()) else "not_evaluable"
    )
    expected_result: Mapping[str, object] = {
        "schema_version": 1,
        "attempt_id": "etas_qualification_q1",
        "input_manifest_sha256": input_sha,
        "protocol_config_sha256": prepared.protocol.sha256,
        "code_commit": code_commit,
        "code_tag": code_tag,
        "python_version": expected_environment["python_version"],
        "numpy_version": expected_environment["numpy_version"],
        "scipy_version": expected_environment["scipy_version"],
        "snapshot_results_sha256": hashes,
        "qualification_status": status,
        "snapshot_statuses": statuses,
        "completed_snapshot_count": 5,
        "completed_start_row_count": 25,
        "assessment_event_read": False,
        "anomaly_feature_read": False,
        "locked_test_read_or_run": False,
    }
    if canonical_bytes(result) != canonical_bytes(expected_result):
        raise ValueError("result manifest differs from independent recalculation")
    expected_views = (
        (prepared.protocol.output("report"), build_markdown(result, payloads).encode()),
        (prepared.protocol.output("static_figure"), build_svg(result, payloads).encode()),
        (
            prepared.protocol.output("interactive_report"),
            build_html(result, payloads).encode(),
        ),
    )
    for path, expected_bytes in expected_views:
        if path.read_bytes() != expected_bytes:
            raise ValueError(f"public view differs from independent recalculation: {path.name}")
    verification: Mapping[str, object] = {
        "schema_version": 1,
        "attempt_id": "etas_qualification_q1",
        "input_manifest_sha256": input_sha,
        "result_manifest_sha256": file_sha256(prepared.protocol.output("result_manifest")),
        "snapshot_results_sha256": hashes,
        "snapshot_statuses": statuses,
        "qualification_status": status,
        "verification_status": "passed",
        "recomputed": (
            "physical_parameters",
            "objective",
            "gradient",
            "best_three",
            "hessian",
            "branching_ratio",
            "three_grid_metrics",
            "unsupported_parent_objective_sensitivity",
            "snapshot_status",
            "qualification_status",
            "public_views",
        ),
    }
    if persist:
        atomic_json(prepared.protocol.output("verification_manifest"), verification)
    return verification


def _summary(results: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, object], ...]:
    rows = []
    for result in results:
        stability = _mapping(result["stability"], "stability")
        hessian = _mapping(stability["hessian"], "hessian")
        grid = result.get("three_grid")
        rows.append(
            {
                "snapshot_id": result["snapshot_id"],
                "status": result["qualification_status"],
                "converged": stability["converged_start_count"],
                "hessian": hessian["success"],
                "branching": result["branching_ratio_passed"],
                "grid": _mapping(grid, "grid")["passed"] if isinstance(grid, Mapping) else False,
            }
        )
    return tuple(rows)


def build_markdown(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> str:
    def yes(value: object) -> str:
        return "通过" if value else "未通过"

    lines = [
        "# ETAS 数值资格检验",
        "",
        f"总状态：**{manifest['qualification_status']}**。这只是拟合稳定性检验，不是预测命中率。",
        "",
        "| 快照 | 收敛起点 | Hessian | 分支比 | 三网格 | 结论 |",
        "|---|---:|---|---|---|---|",
    ]
    for row in _summary(results):
        lines.append(
            f"| {row['snapshot_id']} | {row['converged']}/5 | {yes(row['hessian'])} | "
            f"{yes(row['branching'])} | {yes(row['grid'])} | {row['status']} |"
        )
    lines += [
        "",
        "五个起点检查是否找到同一稳定解；Hessian 检查解是否清楚；分支比防止模型失稳；"
        "三网格检查结论是否依赖网格粗细。局部高 Mc 只影响对应固定单元。",
        "",
        "本报告不含事件编号、坐标、异常特征、评分或锁定测试信息。",
    ]
    return "\n".join(lines) + "\n"


def build_svg(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="340">',
        (
            "<style>text{font-family:Arial,'Microsoft YaHei';fill:#172033}"
            ".p{fill:#2d9d78}.f{fill:#d95d5d}</style>"
        ),
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        (
            '<text x="25" y="34" font-size="21">ETAS 五快照资格：'
            f"{html.escape(str(manifest['qualification_status']))}</text>"
        ),
    ]
    for index, row in enumerate(_summary(results)):
        y = 72 + index * 48
        parts.append(f'<text x="25" y="{y + 22}">{row["snapshot_id"]}</text>')
        values = (
            cast(int, row["converged"]) >= 4,
            row["hessian"],
            row["branching"],
            row["grid"],
        )
        for column, passed in enumerate(values):
            x = 220 + column * 145
            parts.append(
                f'<rect x="{x}" y="{y}" width="115" height="32" rx="6" '
                f'class="{"p" if passed else "f"}"/>'
                f'<text x="{x + 57}" y="{y + 21}" text-anchor="middle" style="fill:white">'
                f"{'通过' if passed else '未通过'}</text>"
            )
    parts.append(
        '<text x="25" y="326" font-size="13">仅显示数值门控；不含事件编号和坐标。</text></svg>'
    )
    return "".join(parts)


def build_html(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> str:
    data = json.dumps(
        {"overall": manifest["qualification_status"], "rows": _summary(results)},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return """<!doctype html><meta charset="utf-8"><title>ETAS资格检验</title>
<style>body{font-family:Arial,"Microsoft YaHei";max-width:850px;margin:30px auto;background:#f5f7fb}
.card{background:white;padding:20px;border-radius:10px}.g{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.p,.f{color:white;padding:12px;border-radius:7px}.p{background:#2d9d78}.f{background:#d95d5d}</style>
<div class="card"><h1>ETAS 数值资格检验</h1>
<p>拟合稳定性，不是预测命中率；完全离线且无事件坐标。</p>
<select id="s"></select><h2 id="t"></h2><div id="g" class="g"></div></div>
<script id="d" type="application/json">DATA</script><script>
const d=JSON.parse(document.getElementById("d").textContent),s=document.getElementById("s");
d.rows.forEach(r=>s.add(new Option(r.snapshot_id,r.snapshot_id)));
function draw(){const r=d.rows.find(x=>x.snapshot_id===s.value);t.textContent="结论："+r.status;
g.replaceChildren(...[["多起点",r.converged>=4],["Hessian",r.hessian],["分支比",r.branching],["三网格",r.grid]]
.map(x=>Object.assign(document.createElement("div"),{className:x[1]?"p":"f",textContent:x[0]+"："+(x[1]?"通过":"未通过")})))}
s.onchange=draw;draw()</script>""".replace("DATA", data)


def publish_views(
    protocol: QualificationProtocol,
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> None:
    atomic_create(protocol.output("report"), build_markdown(manifest, results).encode())
    atomic_create(protocol.output("static_figure"), build_svg(manifest, results).encode())
    atomic_create(protocol.output("interactive_report"), build_html(manifest, results).encode())


__all__ = [
    "PreparedQualification",
    "PreparedSnapshot",
    "QualificationProtocol",
    "atomic_create",
    "build_html",
    "build_markdown",
    "build_svg",
    "causal_fit_masks",
    "classify_snapshot",
    "load_protocol",
    "prepare_real_inputs",
    "run_prepared",
    "verify_prepared",
]
