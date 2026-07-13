from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from seismoflux.config import SeismoFluxConfig, load_config, load_yaml_mapping

BASE_CONFIG = Path("configs/base.yaml")


def test_base_configuration_matches_preregistered_contract() -> None:
    config = load_config(BASE_CONFIG)

    assert config.project.random_seed == 147
    assert config.time_semantics.forecast_interval == "(T,T+h]"
    assert config.forecast.horizons_days == (7, 30, 90, 180, 365)
    assert config.integration.convergence_cells_km == (50, 25, 12.5)
    assert config.regions.operational_max_components == 10
    assert config.regions.operational_max_union_area_km2 == 960000
    assert config.parallel.reserve_physical_cores == 2
    assert config.parallel.nested_parallelism is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("forecast", "horizons_days", [7, 30]),
        ("integration", "convergence_cells_km", [50, 25]),
        ("regions", "operational_max_components", 12),
        ("parallel", "reserve_physical_cores", 1),
        ("parallel", "nested_parallelism", True),
    ],
)
def test_configuration_rejects_contract_drift(section: str, field: str, value: object) -> None:
    raw = deepcopy(load_yaml_mapping(BASE_CONFIG))
    nested = raw[section]
    assert isinstance(nested, dict)
    nested[field] = value

    with pytest.raises(ValidationError):
        SeismoFluxConfig.model_validate(raw)


@pytest.mark.parametrize(
    "invalid_path",
    ["D:/outside/data_sources.yaml", "C:drive-relative.yaml", r"\rooted.yaml", "/rooted.yaml"],
)
def test_project_paths_must_be_relative(invalid_path: str) -> None:
    raw = deepcopy(load_yaml_mapping(BASE_CONFIG))
    config_files = raw["config_files"]
    assert isinstance(config_files, dict)
    config_files["data_sources"] = invalid_path

    with pytest.raises(ValidationError, match="project-relative"):
        SeismoFluxConfig.model_validate(raw)


def test_magnitude_bins_must_be_unique() -> None:
    raw = deepcopy(load_yaml_mapping(BASE_CONFIG))
    forecast = raw["forecast"]
    assert isinstance(forecast, dict)
    magnitude_bins = forecast["magnitude_bins"]
    assert isinstance(magnitude_bins, list)
    magnitude_bins.append(deepcopy(magnitude_bins[0]))

    with pytest.raises(ValidationError, match="magnitude bins"):
        SeismoFluxConfig.model_validate(raw)
