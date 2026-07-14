"""Validated project configuration for SeismoFlux."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects undocumented configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def normalize_relative_path(value: str) -> str:
    """Normalize a portable relative path and reject rooted or drive-relative forms."""

    posix_path = PurePosixPath(value.replace("\\", "/"))
    windows_path = PureWindowsPath(value)
    if (
        not value
        or posix_path.is_absolute()
        or bool(windows_path.anchor)
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in posix_path.parts
    ):
        raise ValueError("project configuration paths must be project-relative")
    return posix_path.as_posix()


class ProjectConfig(StrictModel):
    name: Literal["SeismoFlux"]
    timezone: Literal["Asia/Shanghai"]
    random_seed: int = Field(ge=0)


class ConfigFiles(StrictModel):
    data_sources: str
    ingestion: str
    research_protocol: str
    background: str
    operating_points: str

    _data_sources_relative = field_validator("data_sources")(normalize_relative_path)
    _ingestion_relative = field_validator("ingestion")(normalize_relative_path)
    _research_protocol_relative = field_validator("research_protocol")(normalize_relative_path)
    _background_relative = field_validator("background")(normalize_relative_path)
    _operating_points_relative = field_validator("operating_points")(normalize_relative_path)


class StudyAreaConfig(StrictModel):
    polygon: str
    include_external_trigger_buffer_km: float = Field(ge=0)
    equal_area_crs: str = Field(min_length=1)

    _polygon_relative = field_validator("polygon")(normalize_relative_path)


class TimeSemanticsConfig(StrictModel):
    issue_timezone: Literal["Asia/Shanghai"]
    issue_time_local: Literal["00:00:00"]
    report_date_fallback_available_at: Literal["next_day_00:00:00_Asia/Shanghai"]
    availability_rule: Literal["available_at_lte_issue_time"]
    forecast_interval: Literal["(T,T+h]"]


class MagnitudeBin(StrictModel):
    id: Literal["M5_6", "M6_plus"]
    minimum: float = Field(alias="min")
    max_exclusive: float | None


class ForecastConfig(StrictModel):
    horizons_days: tuple[int, ...]
    magnitude_bins: tuple[MagnitudeBin, ...]

    @model_validator(mode="after")
    def validate_forecast_contract(self) -> Self:
        if self.horizons_days != (7, 30, 90, 180, 365):
            raise ValueError("forecast horizons must be exactly 7, 30, 90, 180, and 365 days")
        bins = {item.id: item for item in self.magnitude_bins}
        if len(self.magnitude_bins) != 2 or set(bins) != {"M5_6", "M6_plus"}:
            raise ValueError("forecast magnitude bins must be M5_6 and M6_plus")
        if bins["M5_6"].minimum != 5.0 or bins["M5_6"].max_exclusive != 6.0:
            raise ValueError("M5_6 must represent [5.0, 6.0)")
        if bins["M6_plus"].minimum != 6.0 or bins["M6_plus"].max_exclusive is not None:
            raise ValueError("M6_plus must represent [6.0, +infinity)")
        return self


class IntegrationConfig(StrictModel):
    base_cell_km: float = Field(gt=0)
    refine_cell_km: float = Field(gt=0)
    convergence_cells_km: tuple[float, ...]

    @model_validator(mode="after")
    def validate_grid_contract(self) -> Self:
        if self.base_cell_km != 25 or self.refine_cell_km != 12.5:
            raise ValueError("base and refined integration cells must be 25 km and 12.5 km")
        if self.convergence_cells_km != (50, 25, 12.5):
            raise ValueError("grid convergence audit must use 50, 25, and 12.5 km cells")
        return self


class RegionConfig(StrictModel):
    component_counts: tuple[int, ...]
    union_area_budgets_km2: tuple[int, ...]
    operational_max_components: int = Field(gt=0)
    operational_max_union_area_km2: int = Field(gt=0)
    strict_coverage_buffer_km: float = Field(ge=0)
    tolerant_coverage_buffer_km: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_region_contract(self) -> Self:
        if self.component_counts != (5, 8, 10, 12):
            raise ValueError("region component candidates must be 5, 8, 10, and 12")
        if self.union_area_budgets_km2 != (300000, 450000, 600000, 750000, 960000):
            raise ValueError("union-area candidates do not match the preregistered Pareto frontier")
        if self.operational_max_components != 10:
            raise ValueError("the operational output may contain at most 10 regions")
        if self.operational_max_union_area_km2 != 960000:
            raise ValueError("the operational union area may not exceed 960000 km2")
        if self.strict_coverage_buffer_km != 0 or self.tolerant_coverage_buffer_km != 70:
            raise ValueError("strict and tolerant coverage buffers must be 0 km and 70 km")
        return self


class ParallelConfig(StrictModel):
    reserve_physical_cores: int = Field(ge=2)
    max_workers: int = Field(gt=0)
    nested_parallelism: Literal[False]


class SeismoFluxConfig(StrictModel):
    config_version: Literal[1]
    project: ProjectConfig
    config_files: ConfigFiles
    time_semantics: TimeSemanticsConfig
    study_area: StudyAreaConfig
    forecast: ForecastConfig
    integration: IntegrationConfig
    regions: RegionConfig
    parallel: ParallelConfig


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a UTF-8 YAML document and require a mapping at its root."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read configuration: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return cast(dict[str, Any], raw)


def load_config(path: str | Path) -> SeismoFluxConfig:
    """Load and validate the main project configuration."""

    config_path = Path(path)
    return SeismoFluxConfig.model_validate(load_yaml_mapping(config_path))


def project_root_for(config_path: str | Path) -> Path:
    """Derive the project root from a main configuration path."""

    path = Path(config_path)
    parent = path.parent
    return parent.parent if parent.name == "configs" else parent


def resolve_project_path(config_path: str | Path, reference: str) -> Path:
    """Resolve a validated project-relative reference from the project root."""

    normalized = normalize_relative_path(reference)
    root = project_root_for(config_path).resolve()
    candidate = (root / Path(normalized)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("resolved project path escapes the project root")
    return candidate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file using bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
