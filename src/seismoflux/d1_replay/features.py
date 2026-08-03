"""Causal, training-only feature preparation for the D1 retrospective replay.

This module deliberately has no earthquake-catalogue reader.  It streams the
accepted stage-3 anomaly feature store one issue (one parquet row group) at a
time and turns the preregistered source fields into the small D1 group design.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

D1_SOURCE_COLUMNS = (
    "gaussian_200km__current_to_trailing_station_reporting_coverage_proxy",
    "gaussian_200km__current_to_trailing_measurement_reporting_coverage_proxy",
    "gaussian_200km__distinct_reporting_station_count_reporting_coverage_proxy",
    "gaussian_200km__distinct_reporting_measurement_count_reporting_coverage_proxy",
    "gaussian_200km__reliability_weighted_listed_count",
    "gaussian_200km__first_seen_weighted_count",
    "gaussian_200km__not_continued_weighted_count",
    "gaussian_200km__discipline_shannon_normalized",
    "gaussian_200km__concentration",
    "radius_200km__listed_count__slope_13w_per_week",
    "radius_200km__listed_count__acceleration_4v13_per_week2",
    "radius_200km__listed_count__peak_drop_52w",
    "radius_200km__first_seen_count__slope_13w_per_week",
    "radius_200km__first_seen_count__acceleration_4v13_per_week2",
    "radius_200km__first_seen_count__peak_drop_52w",
)

_GRID_COLUMNS = (
    "issue_time_utc",
    "issue_report_id",
    "grid_id",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "clipped_area_km2",
)
_ALLOWED_TRANSFORMS = {"identity", "nonnegative", "signed_trajectory"}
_ALLOWED_CONTROL_FORMULAS = {
    "float(any_original_source_is_null)",
    "original_null_source_count_div_3",
}


def _readonly(array: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence")
    output = tuple(value)
    if not all(isinstance(item, str) and item for item in output):
        raise TypeError(f"{label} must contain non-empty strings")
    return cast(tuple[str, ...], output)


def _utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class D1StaticGrid:
    """Static identity established by the first accepted stage-3 row group."""

    grid_id: str
    cell_ids: tuple[str, ...]
    rows: IntArray
    columns: IntArray
    query_x_m: FloatArray
    query_y_m: FloatArray
    clipped_area_km2: FloatArray

    def __post_init__(self) -> None:
        count = len(self.cell_ids)
        if not self.grid_id or count == 0 or len(set(self.cell_ids)) != count:
            raise ValueError("D1 grid identity must be non-empty and unique")
        arrays = (
            ("rows", self.rows, (count,)),
            ("columns", self.columns, (count,)),
            ("query_x_m", self.query_x_m, (count,)),
            ("query_y_m", self.query_y_m, (count,)),
            ("clipped_area_km2", self.clipped_area_km2, (count,)),
        )
        for name, array, shape in arrays:
            if np.asarray(array).shape != shape:
                raise ValueError(f"D1 grid {name} has the wrong shape")
        if not np.isfinite(self.query_x_m).all() or not np.isfinite(self.query_y_m).all():
            raise ValueError("D1 grid coordinates must be finite")
        if not np.isfinite(self.clipped_area_km2).all() or np.any(self.clipped_area_km2 <= 0.0):
            raise ValueError("D1 clipped cell areas must be finite and positive")
        object.__setattr__(self, "rows", _readonly(np.asarray(self.rows, dtype=np.int64)))
        object.__setattr__(
            self,
            "columns",
            _readonly(np.asarray(self.columns, dtype=np.int64)),
        )
        object.__setattr__(
            self,
            "query_x_m",
            _readonly(np.asarray(self.query_x_m, dtype=np.float64)),
        )
        object.__setattr__(
            self,
            "query_y_m",
            _readonly(np.asarray(self.query_y_m, dtype=np.float64)),
        )
        object.__setattr__(
            self,
            "clipped_area_km2",
            _readonly(np.asarray(self.clipped_area_km2, dtype=np.float64)),
        )

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)


@dataclass(frozen=True, slots=True)
class D1IssueFeatures:
    """One issue's 15 raw source fields and their original Arrow null mask."""

    issue_time_utc: datetime
    issue_report_id: str
    grid: D1StaticGrid
    source_columns: tuple[str, ...]
    values: FloatArray
    null_mask: BoolArray

    def __post_init__(self) -> None:
        issue_time = _utc(self.issue_time_utc, label="D1 issue time")
        columns = tuple(self.source_columns)
        values = np.asarray(self.values, dtype=np.float64)
        nulls = np.asarray(self.null_mask, dtype=np.bool_)
        expected_shape = (self.grid.cell_count, len(columns))
        if not self.issue_report_id or columns != D1_SOURCE_COLUMNS:
            raise ValueError("D1 issue must carry the exact preregistered 15 source columns")
        if values.shape != expected_shape or nulls.shape != expected_shape:
            raise ValueError("D1 source values/null mask do not align with the grid")
        if not np.isfinite(values[~nulls]).all():
            raise ValueError("non-null D1 source values must be finite")
        owned_values = np.array(values, dtype=np.float64, copy=True, order="C")
        owned_values[nulls] = np.nan
        object.__setattr__(self, "issue_time_utc", issue_time)
        object.__setattr__(self, "source_columns", columns)
        object.__setattr__(self, "values", _readonly(owned_values))
        object.__setattr__(self, "null_mask", _readonly(nulls))


def _single_value(table: pa.Table, name: str, *, label: str) -> object:
    column = table[name].combine_chunks()
    if column.null_count or len(column) == 0:
        raise ValueError(f"{label} must be non-null and non-empty")
    value = column[0].as_py()
    equal = pc.all(pc.equal(column, pa.scalar(value, type=column.type))).as_py()
    if equal is not True:
        raise ValueError(f"one row group contains multiple {label} values")
    return value


def _numpy_column(table: pa.Table, name: str, dtype: np.dtype[np.generic]) -> NDArray[np.generic]:
    column = table[name].combine_chunks()
    if column.null_count:
        raise ValueError(f"static D1 grid column {name} contains nulls")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def _grid_from_table(table: pa.Table) -> D1StaticGrid:
    grid_id = _single_value(table, "grid_id", label="grid_id")
    cell_ids = tuple(table["cell_id"].combine_chunks().to_pylist())
    if not isinstance(grid_id, str) or not all(isinstance(item, str) for item in cell_ids):
        raise TypeError("D1 grid string identities changed type")
    return D1StaticGrid(
        grid_id=grid_id,
        cell_ids=cast(tuple[str, ...], cell_ids),
        rows=cast(IntArray, _numpy_column(table, "cell_row", np.dtype(np.int64))),
        columns=cast(IntArray, _numpy_column(table, "cell_column", np.dtype(np.int64))),
        query_x_m=cast(FloatArray, _numpy_column(table, "query_x_m", np.dtype(np.float64))),
        query_y_m=cast(FloatArray, _numpy_column(table, "query_y_m", np.dtype(np.float64))),
        clipped_area_km2=cast(
            FloatArray,
            _numpy_column(table, "clipped_area_km2", np.dtype(np.float64)),
        ),
    )


def _assert_same_grid(observed: D1StaticGrid, expected: D1StaticGrid) -> None:
    if observed.grid_id != expected.grid_id or observed.cell_ids != expected.cell_ids:
        raise ValueError("stage-3 row group changed D1 grid string identity or cell order")
    for name in ("rows", "columns", "query_x_m", "query_y_m", "clipped_area_km2"):
        if not np.array_equal(getattr(observed, name), getattr(expected, name)):
            raise ValueError(f"stage-3 row group changed static D1 grid column {name}")


def stream_stage3_issue_features(
    path: Path,
    *,
    expected_issue_count: int = 205,
    expected_cell_count: int = 15_697,
    expected_grid: D1StaticGrid | None = None,
) -> Iterator[D1IssueFeatures]:
    """Stream exactly one accepted issue row group while reading only 15 source fields.

    The identity columns are read solely to establish and verify the static grid.
    No quality, score, earthquake, or unused feature column is selected.
    """

    parquet = pq.ParquetFile(Path(path))
    if parquet.metadata.num_row_groups != expected_issue_count:
        raise ValueError("stage-3 feature store does not contain the expected issue count")
    selected = (*_GRID_COLUMNS, *D1_SOURCE_COLUMNS)
    missing = sorted(set(selected) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"stage-3 feature store is missing D1 columns: {missing}")
    frozen_grid = expected_grid
    previous_issue_time: datetime | None = None
    for row_group_index in range(parquet.metadata.num_row_groups):
        if parquet.metadata.row_group(row_group_index).num_rows != expected_cell_count:
            raise ValueError("stage-3 issue row group has the wrong cell count")
        table = parquet.read_row_group(row_group_index, columns=list(selected))
        issue_time = _utc(
            _single_value(table, "issue_time_utc", label="issue_time_utc"),
            label="issue_time_utc",
        )
        if previous_issue_time is not None and issue_time <= previous_issue_time:
            raise ValueError("stage-3 issue row groups must be strictly chronological")
        previous_issue_time = issue_time
        report_id = _single_value(table, "issue_report_id", label="issue_report_id")
        if not isinstance(report_id, str) or not report_id:
            raise TypeError("D1 issue_report_id must be a non-empty string")
        observed_grid = _grid_from_table(table)
        if frozen_grid is None:
            frozen_grid = observed_grid
        else:
            _assert_same_grid(observed_grid, frozen_grid)
        values = np.empty((expected_cell_count, len(D1_SOURCE_COLUMNS)), dtype=np.float64)
        null_mask = np.empty(values.shape, dtype=np.bool_)
        for column_index, name in enumerate(D1_SOURCE_COLUMNS):
            column = table[name].combine_chunks().cast(pa.float64())
            missing_values = np.asarray(
                column.is_null().to_numpy(zero_copy_only=False),
                dtype=np.bool_,
            )
            raw = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
            if not np.isfinite(raw[~missing_values]).all():
                raise ValueError(f"stage-3 D1 source {name} contains non-finite valid values")
            values[:, column_index] = raw
            null_mask[:, column_index] = missing_values
        yield D1IssueFeatures(
            issue_time_utc=issue_time,
            issue_report_id=report_id,
            grid=frozen_grid,
            source_columns=D1_SOURCE_COLUMNS,
            values=values,
            null_mask=null_mask,
        )


@dataclass(frozen=True, slots=True)
class D1SourceSpec:
    column: str
    transform: str
    sign: float


@dataclass(frozen=True, slots=True)
class D1GroupSpec:
    group_id: str
    sources: tuple[D1SourceSpec, ...]


@dataclass(frozen=True, slots=True)
class D1ControlSpec:
    control_id: str
    parent_groups: tuple[str, ...]
    sources: tuple[str, ...]
    formula: str


@dataclass(frozen=True, slots=True)
class D1FeatureContract:
    groups: tuple[D1GroupSpec, ...]
    controls: tuple[D1ControlSpec, ...]

    @classmethod
    def from_mapping(cls, document: Mapping[str, object]) -> D1FeatureContract:
        groups_node = _mapping(document.get("feature_groups"), label="feature_groups")
        groups: list[D1GroupSpec] = []
        seen_sources: list[str] = []
        for group_id, raw_group in groups_node.items():
            group = _mapping(raw_group, label=f"feature_groups.{group_id}")
            raw_sources = group.get("sources")
            if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, str | bytes):
                raise TypeError(f"feature_groups.{group_id}.sources must be a sequence")
            sources: list[D1SourceSpec] = []
            for raw_source in raw_sources:
                source = _mapping(raw_source, label=f"feature_groups.{group_id}.sources")
                column = source.get("column")
                transform = source.get("transform")
                sign = source.get("sign")
                if (
                    not isinstance(column, str)
                    or transform not in _ALLOWED_TRANSFORMS
                    or not isinstance(sign, int | float)
                    or isinstance(sign, bool)
                    or float(sign) not in {-1.0, 1.0}
                ):
                    raise ValueError(f"feature group {group_id} has an invalid source contract")
                sources.append(D1SourceSpec(column, cast(str, transform), float(sign)))
                seen_sources.append(column)
            if not sources:
                raise ValueError(f"feature group {group_id} has no sources")
            groups.append(D1GroupSpec(group_id, tuple(sources)))
        if tuple(seen_sources) != D1_SOURCE_COLUMNS or len(set(seen_sources)) != 15:
            raise ValueError("D1 feature groups must define the exact ordered 15-source contract")

        controls_node = _mapping(document.get("design_controls"), label="design_controls")
        controls: list[D1ControlSpec] = []
        for control_id, raw_control in controls_node.items():
            if control_id in {
                "count_as_scientific_groups",
                "included_automatically_when_any_parent_group_is_used",
                "coefficient_penalty_factor",
            }:
                continue
            control_mapping = _mapping(raw_control, label=f"design_controls.{control_id}")
            formula = control_mapping.get("formula")
            if formula not in _ALLOWED_CONTROL_FORMULAS:
                raise ValueError(f"design control {control_id} has an invalid formula")
            controls.append(
                D1ControlSpec(
                    control_id=control_id,
                    parent_groups=_string_sequence(
                        control_mapping.get("parent_groups"),
                        label=f"design_controls.{control_id}.parent_groups",
                    ),
                    sources=_string_sequence(
                        control_mapping.get("sources"),
                        label=f"design_controls.{control_id}.sources",
                    ),
                    formula=cast(str, formula),
                )
            )
        if tuple(item.control_id for item in controls) != ("MC1", "MS45", "MD1", "MD2"):
            raise ValueError("D1 design controls or their stable order changed")
        known_groups = {item.group_id for item in groups}
        for control in controls:
            if not set(control.parent_groups) <= known_groups:
                raise ValueError(f"control {control.control_id} names an unknown parent group")
            if not set(control.sources) <= set(D1_SOURCE_COLUMNS):
                raise ValueError(f"control {control.control_id} names an unknown source")
        return cls(groups=tuple(groups), controls=tuple(controls))


def load_d1_feature_contract(config_path: Path) -> D1FeatureContract:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return D1FeatureContract.from_mapping(_mapping(document, label="D1 configuration"))


@dataclass(frozen=True, slots=True)
class D1SourceFit:
    transform: str
    clip_low: float
    clip_high: float
    center: float
    scale: float


@dataclass(frozen=True, slots=True)
class D1GroupFit:
    center: float
    scale: float
    active: bool


@dataclass(frozen=True, slots=True)
class D1DesignMatrix:
    issue_time_utc: datetime
    column_names: tuple[str, ...]
    values: FloatArray
    active_coefficients: BoolArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        active = np.asarray(self.active_coefficients, dtype=np.bool_)
        if values.ndim != 2 or values.shape[1] != len(self.column_names):
            raise ValueError("D1 design matrix columns do not align")
        if active.shape != (values.shape[1],) or not np.isfinite(values).all():
            raise ValueError("D1 design values/active mask are invalid")
        object.__setattr__(self, "issue_time_utc", _utc(self.issue_time_utc, label="issue time"))
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "active_coefficients", _readonly(active))


def _transform_valid(values: FloatArray, transform: str) -> FloatArray:
    if not np.isfinite(values).all():
        raise ValueError("valid source values must be finite before transformation")
    if transform == "identity":
        return np.asarray(values, dtype=np.float64)
    if transform == "nonnegative":
        if np.any(values < 0.0):
            raise ValueError("nonnegative D1 source contains a negative valid value")
        return np.log1p(values, dtype=np.float64)
    if transform == "signed_trajectory":
        return np.arcsinh(values, dtype=np.float64)
    raise ValueError(f"unknown D1 source transform: {transform}")


def _scale(values: FloatArray, center: float) -> tuple[float, bool]:
    mad_scale = 1.4826 * float(np.median(np.abs(values - center)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    standard_deviation = float(np.std(values, ddof=0))
    for candidate in (mad_scale, iqr_scale, standard_deviation):
        if math.isfinite(candidate) and candidate > 0.0:
            return candidate, False
    return 1.0, True


def _fit_source(values: FloatArray, nulls: BoolArray, transform: str) -> D1SourceFit:
    valid = _transform_valid(values[~nulls], transform)
    if valid.size == 0:
        raise ValueError("D1 source is entirely null in the outer training period")
    low, high = np.quantile(valid, [0.005, 0.995])
    clipped = np.clip(valid, low, high)
    center = float(np.median(clipped))
    scale, _ = _scale(np.asarray(clipped, dtype=np.float64), center)
    return D1SourceFit(transform, float(low), float(high), center, scale)


def _apply_source(values: FloatArray, nulls: BoolArray, fit: D1SourceFit) -> FloatArray:
    output = np.empty(values.shape, dtype=np.float64)
    output[nulls] = fit.center
    output[~nulls] = np.clip(
        _transform_valid(values[~nulls], fit.transform),
        fit.clip_low,
        fit.clip_high,
    )
    output -= fit.center
    output /= fit.scale
    if not np.isfinite(output).all():
        raise FloatingPointError("D1 source preprocessing produced non-finite values")
    return output


@dataclass(frozen=True, slots=True)
class D1GroupPreprocessor:
    """Frozen training-only statistics for a selected nested D1 feature model."""

    contract: D1FeatureContract
    selected_group_ids: tuple[str, ...]
    selected_control_ids: tuple[str, ...]
    source_fits: Mapping[str, D1SourceFit]
    group_fits: Mapping[str, D1GroupFit]
    output_column_names: tuple[str, ...]
    active_coefficients: BoolArray
    fitted_issue_times_utc: tuple[datetime, ...]

    def __post_init__(self) -> None:
        active = np.asarray(self.active_coefficients, dtype=np.bool_)
        if active.shape != (len(self.output_column_names),):
            raise ValueError("D1 preprocessor active mask does not align with output columns")
        object.__setattr__(self, "source_fits", MappingProxyType(dict(self.source_fits)))
        object.__setattr__(self, "group_fits", MappingProxyType(dict(self.group_fits)))
        object.__setattr__(self, "active_coefficients", _readonly(active))

    @classmethod
    def fit(
        cls,
        contract: D1FeatureContract,
        selected_groups: Sequence[str],
        training_issues: Sequence[D1IssueFeatures],
    ) -> D1GroupPreprocessor:
        requested = tuple(selected_groups)
        if len(set(requested)) != len(requested):
            raise ValueError("selected D1 groups must be unique")
        known = {item.group_id for item in contract.groups}
        if not set(requested) <= known:
            raise ValueError("selected D1 groups contain an unknown identifier")
        selected = tuple(item.group_id for item in contract.groups if item.group_id in requested)
        issues = tuple(training_issues)
        if not issues:
            raise ValueError("D1 preprocessor requires at least one outer-training issue")
        times = tuple(item.issue_time_utc for item in issues)
        if len(set(times)) != len(times) or tuple(sorted(times)) != times:
            raise ValueError("D1 preprocessor training issues must be unique and chronological")
        if any(item.source_columns != D1_SOURCE_COLUMNS for item in issues):
            raise ValueError("D1 training issue source contract changed")
        all_values = np.concatenate([item.values for item in issues], axis=0)
        all_nulls = np.concatenate([item.null_mask for item in issues], axis=0)
        index_by_source = {name: index for index, name in enumerate(D1_SOURCE_COLUMNS)}
        selected_specs = tuple(item for item in contract.groups if item.group_id in selected)
        needed_sources = {source.column for group in selected_specs for source in group.sources}
        selected_controls = tuple(
            item for item in contract.controls if set(item.parent_groups) & set(selected)
        )
        needed_sources.update(source for item in selected_controls for source in item.sources)
        source_fits: dict[str, D1SourceFit] = {}
        source_spec_by_name = {
            source.column: source for group in contract.groups for source in group.sources
        }
        for source_name in D1_SOURCE_COLUMNS:
            if source_name not in needed_sources:
                continue
            position = index_by_source[source_name]
            source_fits[source_name] = _fit_source(
                all_values[:, position],
                all_nulls[:, position],
                source_spec_by_name[source_name].transform,
            )
        group_fits: dict[str, D1GroupFit] = {}
        active_groups: list[bool] = []
        for group in selected_specs:
            components = []
            for source in group.sources:
                position = index_by_source[source.column]
                standardized = _apply_source(
                    all_values[:, position],
                    all_nulls[:, position],
                    source_fits[source.column],
                )
                components.append(source.sign * standardized)
            score = np.asarray(
                np.mean(np.column_stack(components), axis=1, dtype=np.float64),
                dtype=np.float64,
            )
            center = float(np.median(score))
            scale, constant = _scale(score, center)
            group_fits[group.group_id] = D1GroupFit(center, scale, not constant)
            active_groups.append(not constant)
        output_columns = (*selected, *(item.control_id for item in selected_controls))
        active = np.asarray(
            (*active_groups, *(True for _ in selected_controls)),
            dtype=np.bool_,
        )
        return cls(
            contract=contract,
            selected_group_ids=selected,
            selected_control_ids=tuple(item.control_id for item in selected_controls),
            source_fits=source_fits,
            group_fits=group_fits,
            output_column_names=output_columns,
            active_coefficients=active,
            fitted_issue_times_utc=times,
        )

    def _control_values(self, issue: D1IssueFeatures, control: D1ControlSpec) -> FloatArray:
        index_by_source = {name: index for index, name in enumerate(D1_SOURCE_COLUMNS)}
        positions = [index_by_source[name] for name in control.sources]
        nulls = issue.null_mask[:, positions]
        if control.formula == "float(any_original_source_is_null)":
            return np.asarray(np.any(nulls, axis=1), dtype=np.float64)
        if control.formula == "original_null_source_count_div_3":
            if len(positions) != 3:
                raise ValueError(f"D1 fraction control {control.control_id} must have 3 sources")
            return np.asarray(np.mean(nulls, axis=1, dtype=np.float64), dtype=np.float64)
        raise ValueError(f"unknown D1 control formula: {control.formula}")

    def transform(self, issue: D1IssueFeatures) -> D1DesignMatrix:
        index_by_source = {name: index for index, name in enumerate(D1_SOURCE_COLUMNS)}
        group_by_id = {item.group_id: item for item in self.contract.groups}
        columns: list[FloatArray] = []
        for group_id in self.selected_group_ids:
            group = group_by_id[group_id]
            components = []
            for source in group.sources:
                position = index_by_source[source.column]
                components.append(
                    source.sign
                    * _apply_source(
                        issue.values[:, position],
                        issue.null_mask[:, position],
                        self.source_fits[source.column],
                    )
                )
            raw_score = np.asarray(
                np.mean(np.column_stack(components), axis=1, dtype=np.float64),
                dtype=np.float64,
            )
            group_fit = self.group_fits[group_id]
            if group_fit.active:
                columns.append((raw_score - group_fit.center) / group_fit.scale)
            else:
                columns.append(np.zeros(issue.grid.cell_count, dtype=np.float64))
        control_by_id = {item.control_id: item for item in self.contract.controls}
        for control_id in self.selected_control_ids:
            columns.append(self._control_values(issue, control_by_id[control_id]))
        values = (
            np.column_stack(columns)
            if columns
            else np.empty((issue.grid.cell_count, 0), dtype=np.float64)
        )
        return D1DesignMatrix(
            issue_time_utc=issue.issue_time_utc,
            column_names=self.output_column_names,
            values=np.asarray(values, dtype=np.float64),
            active_coefficients=self.active_coefficients,
        )


__all__ = [
    "D1_SOURCE_COLUMNS",
    "D1ControlSpec",
    "D1DesignMatrix",
    "D1FeatureContract",
    "D1GroupFit",
    "D1GroupPreprocessor",
    "D1GroupSpec",
    "D1IssueFeatures",
    "D1SourceFit",
    "D1SourceSpec",
    "D1StaticGrid",
    "load_d1_feature_contract",
    "stream_stage3_issue_features",
]
