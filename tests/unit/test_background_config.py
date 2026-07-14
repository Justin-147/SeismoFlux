from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import seismoflux.background.config as background_config_module
from seismoflux.background.config import (
    BackgroundConfig,
    load_background_config,
    load_project_background_config,
)
from seismoflux.config import (
    SeismoFluxConfig,
    load_config,
    load_yaml_mapping,
    resolve_project_path,
    sha256_file,
)

BACKGROUND_CONFIG = Path("configs/background.yaml")
BASE_CONFIG = Path("configs/base.yaml")

PathPart = str | int
MappingPath = tuple[PathPart, ...]


def _mapping_paths(value: object, path: MappingPath = ()) -> Iterator[MappingPath]:
    if isinstance(value, dict):
        yield path
        for key, child in value.items():
            yield from _mapping_paths(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _mapping_paths(child, (*path, index))


def _at_path(value: object, path: MappingPath) -> dict[str, Any]:
    current = value
    for part in path:
        if isinstance(part, int):
            assert isinstance(current, list)
            current = current[part]
        else:
            assert isinstance(current, dict)
            current = current[part]
    assert isinstance(current, dict)
    return current


def test_project_background_configuration_loads_with_frozen_invariants() -> None:
    config = load_project_background_config(BASE_CONFIG)

    assert config.background_scores_seen_before_freeze is False
    assert config.randomness.root_seed == 147
    assert config.inputs.include_external_trigger_buffer_km == 300
    assert config.etas.spatial_kernel.support_radius_km == 300
    assert config.time.horizons_days == (7, 30, 90, 180, 365)
    assert config.integration.grid_cells_km == (50, 25, 12.5)
    assert config.evaluation.g1_primary_endpoint.horizon_aggregation == "none"


def test_every_background_mapping_level_rejects_unknown_fields() -> None:
    pristine = load_yaml_mapping(BACKGROUND_CONFIG)
    paths = list(_mapping_paths(pristine))
    assert len(paths) >= 40

    for path in paths:
        raw = deepcopy(pristine)
        _at_path(raw, path)["unexpected_protocol_field"] = "forbidden"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BackgroundConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("time", "horizons_days", [7, 30], "background horizons"),
        ("integration", "grid_cells_km", [25, 12.5], "background grids"),
        (
            "etas.spatial_kernel",
            "support_radius_km",
            301,
            "ETAS spatial cutoff",
        ),
        (
            "randomness.future_simulation",
            "reuse_same_catalog_for_horizons_days",
            [7, 30],
            "simulation horizons",
        ),
    ],
)
def test_background_configuration_rejects_cross_field_drift(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    raw = deepcopy(load_yaml_mapping(BACKGROUND_CONFIG))
    node: object = raw
    for component in section.split("."):
        assert isinstance(node, dict)
        node = node[component]
    assert isinstance(node, dict)
    node[field] = value

    with pytest.raises(ValidationError, match=message):
        BackgroundConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("inputs", "environment_lock"),
        ("inputs", "data_catalog"),
        ("numerical_regression", "production_fixture"),
        ("outputs", "registry"),
    ],
)
def test_background_paths_must_be_project_relative(section: str, field: str) -> None:
    raw = deepcopy(load_yaml_mapping(BACKGROUND_CONFIG))
    node = raw[section]
    assert isinstance(node, dict)
    node[field] = "D:/outside/protocol-file.json"

    with pytest.raises(ValidationError, match="project-relative"):
        BackgroundConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("background_scores_seen_before_freeze",), True),
        (("status",), "background_scores_already_seen"),
        (
            ("evaluation", "g1_primary_endpoint", "horizon_aggregation"),
            "mean_across_horizons",
        ),
        (
            ("evaluation", "g1_primary_endpoint", "score_formula"),
            "event_terms_without_compensator",
        ),
    ],
)
def test_freeze_and_g1_endpoint_cannot_be_relaxed(path: tuple[str, ...], value: object) -> None:
    raw = deepcopy(load_yaml_mapping(BACKGROUND_CONFIG))
    node: object = raw
    for component in path[:-1]:
        assert isinstance(node, dict)
        node = node[component]
    assert isinstance(node, dict)
    node[path[-1]] = value

    with pytest.raises(ValidationError):
        BackgroundConfig.model_validate(raw)


def test_arbitrary_protocol_value_drift_fails_the_canonical_fingerprint() -> None:
    raw = deepcopy(load_yaml_mapping(BACKGROUND_CONFIG))
    time = raw["time"]
    assert isinstance(time, dict)
    time["representative_issue_date_local"] = "2025-06-19"

    with pytest.raises(ValidationError, match="frozen canonical fingerprint"):
        BackgroundConfig.model_validate(raw)


def test_output_write_protocol_drift_fails_the_canonical_fingerprint() -> None:
    raw = deepcopy(load_yaml_mapping(BACKGROUND_CONFIG))
    outputs = raw["outputs"]
    assert isinstance(outputs, dict)
    write_protocol = outputs["write_protocol"]
    assert isinstance(write_protocol, dict)
    write_protocol["publish"] = "overwrite_existing_directory"

    with pytest.raises(ValidationError, match="frozen canonical fingerprint"):
        BackgroundConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("parameter_selection_folds", "target_events_must_not_enter_fit"), False),
        (("time", "final_parameter_fit_end_utc"), "2025-07-01T00:00:00Z"),
        (("completeness", "target_event_domain"), "inside_and_buffer"),
        (("completeness", "estimate_above_maximum_candidate_action"), "continue"),
        (("completeness", "no_eligible_temporal_or_spatial_strata_action"), "continue"),
        (("etas", "likelihood", "target_events"), "inside_and_buffer_events"),
        (("etas", "likelihood", "parent_events"), "all_global_events"),
        (("spatial_poisson", "spatial_density_integral_over_study_area"), 0.99),
        (("spatial_poisson", "normalization_grid_km"), 25.0),
        (
            ("spatial_poisson", "standard_error_formula"),
            "best_bandwidth_own_fold_standard_error",
        ),
        (("spatial_poisson", "exact_tie_rule"), "smallest_bandwidth_km"),
        (("etas", "background_component", "density_integral_over_study_area"), 0.99),
        (("etas", "likelihood", "background_normalization_grid_km"), 25.0),
        (("etas", "likelihood", "compensator_grid_km"), 25.0),
        (("etas", "temporal_kernel", "history_parent_cutoff_days"), 999999.0),
        (("etas", "magnitude_model", "upper_magnitude"), 10.0),
        (("etas", "branching_ratio", "equality_tolerance"), 1.0e-10),
        (("etas", "branching_ratio", "maximum"), 0.99),
        (("etas", "optimizer_options", "ftol"), 1.0e-9),
        (("etas", "optimizer_options", "gtol"), 1.0e-4),
        (("etas", "optimizer_options", "gradient_relative_step"), 1.0e-4),
        (
            ("randomness", "seed_derivation", "namespace_contexts", "optimizer_start", "model_ids"),
            ["etas/final_validation"],
        ),
        (("evaluation", "g1_pass_rule", "minimum_positive_development_folds"), 1),
        (("evaluation", "g1_pass_rule", "conjunction_unit"), "combine_across_models"),
        (("evaluation", "model_selection", "eligible_model_pool"), "all_models"),
        (("evaluation", "model_selection", "failed_model_role"), "retain_for_threshold"),
        (("evaluation", "model_selection", "simplicity_order"), ["etas", "spatial_poisson"]),
    ],
)
def test_scientific_guardrails_reject_known_value_drift(
    path: tuple[str, ...], value: object
) -> None:
    raw = deepcopy(load_yaml_mapping(BACKGROUND_CONFIG))
    node: object = raw
    for component in path[:-1]:
        assert isinstance(node, dict)
        node = node[component]
    assert isinstance(node, dict)
    node[path[-1]] = value

    with pytest.raises(ValidationError):
        BackgroundConfig.model_validate(raw)


def test_fold_fit_end_cannot_reach_assessment_start() -> None:
    raw = deepcopy(load_yaml_mapping(BACKGROUND_CONFIG))
    folds = raw["parameter_selection_folds"]["folds"]
    assert isinstance(folds, list)
    assert isinstance(folds[0], dict)
    folds[0]["fit_end_utc"] = folds[0]["assessment_start_utc"]

    with pytest.raises(ValidationError, match="365-day purge"):
        BackgroundConfig.model_validate(raw)


def test_background_loader_rejects_content_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def wrong_hash(_path: Path) -> str:
        return "0" * 64

    monkeypatch.setattr(background_config_module, "sha256_file", wrong_hash)

    with pytest.raises(ValueError, match="environment lock SHA-256"):
        load_background_config(BACKGROUND_CONFIG)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("uv.lock", "environment lock SHA-256"),
        ("data_catalog.json", "data catalog SHA-256"),
        ("china_mainland.geojson", "study-area SHA-256"),
        ("earthquake_event.parquet", "local earthquake dataset SHA-256"),
        ("background_fold_manifest.json", "issue manifest SHA-256"),
        ("etas_micro_reference.json", "production fixture SHA-256"),
        ("jss_japan_reference.json", "oracle metadata SHA-256"),
    ],
)
def test_loader_rejects_selective_input_hash_drift(
    monkeypatch: pytest.MonkeyPatch, filename: str, message: str
) -> None:
    original_hash = sha256_file

    def selective_hash(path: Path) -> str:
        return "0" * 64 if path.name == filename else original_hash(path)

    monkeypatch.setattr(background_config_module, "sha256_file", selective_hash)

    with pytest.raises(ValueError, match=message):
        load_background_config(BACKGROUND_CONFIG)


@pytest.mark.parametrize(
    ("missing_value", "message"),
    [
        ("data/processed/china_mainland.geojson", "study-area file is missing"),
        (
            "data/processed/stage1/debc98054172a4a1/earthquake_event.parquet",
            "local earthquake dataset file is missing",
        ),
    ],
)
def test_loader_rejects_missing_frozen_data_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_value: str,
    message: str,
) -> None:
    original_resolver = resolve_project_path

    def selective_resolver(config_path: Path, value: str) -> Path:
        if value == missing_value:
            return tmp_path / Path(value).name
        return original_resolver(config_path, value)

    monkeypatch.setattr(background_config_module, "resolve_project_path", selective_resolver)

    with pytest.raises(ValueError, match=message):
        load_background_config(BACKGROUND_CONFIG)


def test_loader_never_resolves_or_hashes_physical_anomaly_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resolver = resolve_project_path
    original_hash = sha256_file

    def guarded_resolver(config_path: Path, value: str) -> Path:
        if value.endswith("anomaly_report_period.parquet"):
            raise AssertionError("stage 2 resolved the forbidden anomaly source")
        return original_resolver(config_path, value)

    def guarded_hash(path: Path) -> str:
        if path.name == "anomaly_report_period.parquet":
            raise AssertionError("stage 2 opened the forbidden anomaly source")
        return original_hash(path)

    monkeypatch.setattr(background_config_module, "resolve_project_path", guarded_resolver)
    monkeypatch.setattr(background_config_module, "sha256_file", guarded_hash)

    load_background_config(BACKGROUND_CONFIG)


def test_project_loader_rejects_base_seed_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    project = load_config(BASE_CONFIG)
    changed_project = project.project.model_copy(update={"random_seed": 148})
    changed_config = project.model_copy(update={"project": changed_project})
    assert isinstance(changed_config, SeismoFluxConfig)

    def load_changed_project(_path: str | Path) -> SeismoFluxConfig:
        return changed_config

    monkeypatch.setattr(background_config_module, "load_config", load_changed_project)

    with pytest.raises(ValueError, match="background root seed"):
        load_project_background_config(BASE_CONFIG)
