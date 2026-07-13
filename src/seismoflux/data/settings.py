"""Strict stage-1 ingestion configuration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from seismoflux.config import StrictModel, load_yaml_mapping, normalize_relative_path


class IngestionOutputs(StrictModel):
    processed_root: str
    contracts_root: str
    data_catalog: str
    quality_json: str
    quality_markdown: str
    study_area_geojson: str

    @field_validator(
        "processed_root",
        "contracts_root",
        "data_catalog",
        "quality_json",
        "quality_markdown",
        "study_area_geojson",
    )
    @classmethod
    def validate_output_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class SourceRoles(StrictModel):
    anomaly: str
    earthquake_m3_plus: str
    earthquake_m5_plus: str
    fault_coordinates: str
    fault_attributes: str
    long_term_hazard: str
    basemap_and_fault_trace: str

    @model_validator(mode="after")
    def unique_source_roles(self) -> Self:
        values = list(self.model_dump().values())
        if len(values) != len(set(values)):
            raise ValueError("ingestion source roles must be unique")
        return self


class TimeAssumptions(StrictModel):
    anomaly_report_fallback: Literal["max_file_and_row_report_date_next_day_00_fixed_UTC_plus_08"]
    anomaly_occurrence_time_precision: Literal[
        "source_date_only_as_00_00_fixed_UTC_plus_08_flagged"
    ]
    earthquake_origin_timezone: Literal["fixed_UTC_plus_08"]
    earthquake_publication_time: Literal["origin_time_optimistic_assumption_flagged"]
    geology_snapshot_available_at: datetime

    @field_validator("geology_snapshot_available_at")
    @classmethod
    def validate_geology_snapshot_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("geology snapshot availability must include a timezone")
        return value


class DedupThreshold(StrictModel):
    max_time_delta_seconds: float = Field(gt=0)
    max_distance_km: float = Field(gt=0)
    max_magnitude_delta: float = Field(ge=0)
    require_unique_degree_both_sides: bool | None = None


class EarthquakeDeduplication(StrictModel):
    auto_merge: DedupThreshold
    manual_review: DedupThreshold
    canonical_source_priority: tuple[str, ...]
    same_source_rule: Literal["exact_semantic_duplicates_only"]

    @model_validator(mode="after")
    def validate_nested_thresholds(self) -> Self:
        if self.auto_merge.require_unique_degree_both_sides is not True:
            raise ValueError("automatic earthquake matching must require unique degree")
        if self.manual_review.require_unique_degree_both_sides is not None:
            raise ValueError("manual-review matching must not imply automatic uniqueness")
        if (
            self.manual_review.max_time_delta_seconds < self.auto_merge.max_time_delta_seconds
            or self.manual_review.max_distance_km < self.auto_merge.max_distance_km
            or self.manual_review.max_magnitude_delta < self.auto_merge.max_magnitude_delta
        ):
            raise ValueError("manual-review thresholds must contain auto-merge thresholds")
        return self


class StudyAreaSettings(StrictModel):
    source_relative_path: str
    selection_rule: Literal["largest_valid_closed_ring_by_equal_area_then_lowest_segment_number"]
    expected_segment_number_1based: int = Field(gt=0)
    boundary_predicate: Literal["covers"]
    island_policy: Literal["continuous_mainland_excludes_hainan_taiwan_and_other_islands"]
    simplify_geometry: Literal[False]
    expected_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_coordinate_count: int = Field(gt=3)
    expected_geodesic_area_km2: float = Field(gt=0)
    area_tolerance_km2: float = Field(gt=0)

    _source_path_relative = field_validator("source_relative_path")(normalize_relative_path)


class ParquetSettings(StrictModel):
    compression: Literal["zstd"]
    compression_level: Literal[9]
    version: Literal["2.6"]
    data_page_version: Literal["1.0"]
    use_dictionary: Literal[False]
    timestamp_unit: Literal["us"]


class IngestionSettings(StrictModel):
    schema_version: Literal[1]
    contract_version: Literal["0.1.0"]
    source_inventory: str
    outputs: IngestionOutputs
    source_roles: SourceRoles
    time_assumptions: TimeAssumptions
    earthquake_deduplication: EarthquakeDeduplication
    study_area: StudyAreaSettings
    parquet: ParquetSettings

    _inventory_relative = field_validator("source_inventory")(normalize_relative_path)

    @model_validator(mode="after")
    def validate_canonical_source_priority(self) -> Self:
        expected = (
            self.source_roles.earthquake_m5_plus,
            self.source_roles.earthquake_m3_plus,
        )
        if self.earthquake_deduplication.canonical_source_priority != expected:
            raise ValueError(
                "canonical source priority must match the parser's frozen M5-then-M3 order"
            )
        return self


def load_ingestion_settings(path: str | Path) -> IngestionSettings:
    return IngestionSettings.model_validate(load_yaml_mapping(Path(path)))
