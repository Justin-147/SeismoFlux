"""Frozen, fit-scope-only preprocessing for stage-4 anomaly increments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from seismoflux.anomaly_increment.contracts import (
    STAGE4_MATHEMATICS_PROTOCOL_VERSION,
    DesignMatrix,
    FeatureColumnContract,
    FloatArray,
    ScaleBranch,
    TransformName,
    canonical_mapping_sha256,
)

LOWER_CLIP_QUANTILE = 0.005
UPPER_CLIP_QUANTILE = 0.995
MAD_SCALE_FACTOR = 1.4826
IQR_SCALE_DIVISOR = 1.349


class PreprocessingDomainError(ValueError):
    """A finite source value violates its preregistered transform domain."""


def _transform_finite(
    values: FloatArray,
    transform: TransformName,
    *,
    source_column: str,
) -> FloatArray:
    if transform == "identity_finite":
        result = np.array(values, dtype=np.float64, copy=True)
    elif transform == "identity_binary":
        if np.any((values != 0.0) & (values != 1.0)):
            raise PreprocessingDomainError(
                f"{source_column} contains a finite value outside the frozen binary domain"
            )
        result = np.array(values, dtype=np.float64, copy=True)
    elif transform == "log1p_nonnegative":
        if np.any(values < 0.0):
            raise PreprocessingDomainError(
                f"{source_column} contains a negative value for log1p_nonnegative"
            )
        result = np.log1p(values).astype(np.float64, copy=False)
    elif transform == "asinh_signed":
        result = np.arcsinh(values).astype(np.float64, copy=False)
    else:  # pragma: no cover - guarded by FeatureColumnContract
        raise ValueError(f"unsupported transform: {transform}")
    if not np.isfinite(result).all():
        raise PreprocessingDomainError(f"{source_column} transform produced a non-finite value")
    return result


def _raw_column(name: str, value: object, *, allow_empty: bool) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PreprocessingDomainError(f"{name} cannot be represented as float64") from error
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    return result


def _validated_columns(
    contracts: tuple[FeatureColumnContract, ...],
    columns: Mapping[str, object],
    *,
    allow_empty: bool,
) -> tuple[dict[str, FloatArray], int]:
    expected = tuple(item.source_column for item in contracts)
    expected_set = set(expected)
    supplied_set = set(columns)
    if supplied_set != expected_set:
        missing = sorted(expected_set - supplied_set)
        unexpected = sorted(supplied_set - expected_set)
        raise ValueError(
            "preprocessing columns must exactly match the frozen feature contract; "
            f"missing={missing}, unexpected={unexpected}"
        )
    output: dict[str, FloatArray] = {}
    row_count: int | None = None
    for contract in contracts:
        values = _raw_column(
            contract.source_column,
            columns[contract.source_column],
            allow_empty=allow_empty,
        )
        if row_count is None:
            row_count = int(values.size)
        elif values.size != row_count:
            raise ValueError("all preprocessing columns must have the same row count")
        finite = np.isfinite(values)
        _transform_finite(
            values[finite],
            contract.transform,
            source_column=contract.source_column,
        )
        output[contract.source_column] = values
    if row_count is None:
        raise ValueError("at least one feature contract is required")
    return output, row_count


@dataclass(frozen=True, slots=True)
class FeatureFitStatistics:
    """All fit-only statistics needed to replay one feature transform."""

    source_column: str
    logical_feature: str
    transform: TransformName
    finite_training_count: int
    missing_training_count: int
    training_minimum: float | None
    training_maximum: float | None
    lower_clip: float | None
    upper_clip: float | None
    training_median: float | None
    scale: float | None
    scale_branch: ScaleBranch
    value_coefficient_active: bool
    missing_coefficient_active: bool
    fixed_zero_reason: str | None

    def __post_init__(self) -> None:
        if self.finite_training_count < 0 or self.missing_training_count < 0:
            raise ValueError("feature fit counts must be non-negative")
        if self.finite_training_count + self.missing_training_count <= 0:
            raise ValueError("feature statistics require at least one training row")
        numeric = (
            self.training_minimum,
            self.training_maximum,
            self.lower_clip,
            self.upper_clip,
            self.training_median,
            self.scale,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("serialized preprocessing statistics must be finite")
        if self.finite_training_count == 0:
            if any(value is not None for value in numeric):
                raise ValueError("an all-null fit column must not serialize invented statistics")
            if self.scale_branch != "fixed_zero_no_finite_values":
                raise ValueError("an all-null fit column must use the frozen no-finite branch")
            if self.value_coefficient_active:
                raise ValueError("an all-null value coefficient must be fixed zero")
        else:
            if self.training_minimum is None or self.training_maximum is None:
                raise ValueError("finite training values require min and max")
            if self.training_median is None:
                raise ValueError("finite training values require a median")
            if self.value_coefficient_active:
                if self.scale is None or self.scale <= 0.0:
                    raise ValueError("an active value coefficient requires a positive scale")
            elif self.training_minimum != self.training_maximum:
                raise ValueError("a finite value coefficient is fixed only when training min=max")

    def as_mapping(self) -> dict[str, object]:
        return {
            "finite_training_count": self.finite_training_count,
            "fixed_zero_reason": self.fixed_zero_reason,
            "logical_feature": self.logical_feature,
            "lower_clip": self.lower_clip,
            "missing_coefficient_active": self.missing_coefficient_active,
            "missing_training_count": self.missing_training_count,
            "scale": self.scale,
            "scale_branch": self.scale_branch,
            "source_column": self.source_column,
            "training_maximum": self.training_maximum,
            "training_median": self.training_median,
            "training_minimum": self.training_minimum,
            "transform": self.transform,
            "upper_clip": self.upper_clip,
            "value_coefficient_active": self.value_coefficient_active,
        }


def _fit_feature_statistics(
    contract: FeatureColumnContract,
    raw: FloatArray,
) -> FeatureFitStatistics:
    finite_bitmap = np.isfinite(raw)
    finite_raw = raw[finite_bitmap]
    missing_count = int(raw.size - finite_raw.size)
    missing_active = bool(finite_raw.size > 0 and missing_count > 0)
    if finite_raw.size == 0:
        return FeatureFitStatistics(
            source_column=contract.source_column,
            logical_feature=contract.logical_feature,
            transform=contract.transform,
            finite_training_count=0,
            missing_training_count=missing_count,
            training_minimum=None,
            training_maximum=None,
            lower_clip=None,
            upper_clip=None,
            training_median=None,
            scale=None,
            scale_branch="fixed_zero_no_finite_values",
            value_coefficient_active=False,
            missing_coefficient_active=False,
            fixed_zero_reason="no_finite_values_in_fit_scope",
        )

    transformed = _transform_finite(
        finite_raw,
        contract.transform,
        source_column=contract.source_column,
    )
    training_minimum = float(np.min(transformed))
    training_maximum = float(np.max(transformed))
    training_median = float(np.median(transformed))
    is_constant = training_minimum == training_maximum
    if contract.is_binary:
        return FeatureFitStatistics(
            source_column=contract.source_column,
            logical_feature=contract.logical_feature,
            transform=contract.transform,
            finite_training_count=int(finite_raw.size),
            missing_training_count=missing_count,
            training_minimum=training_minimum,
            training_maximum=training_maximum,
            lower_clip=None,
            upper_clip=None,
            training_median=training_median,
            scale=None if is_constant else 1.0,
            scale_branch="fixed_zero_training_constant" if is_constant else "none_binary",
            value_coefficient_active=not is_constant,
            missing_coefficient_active=missing_active,
            fixed_zero_reason="training_min_equals_max" if is_constant else None,
        )

    lower, upper = np.quantile(
        transformed,
        [LOWER_CLIP_QUANTILE, UPPER_CLIP_QUANTILE],
        method="linear",
    )
    lower_clip = float(lower)
    upper_clip = float(upper)
    if is_constant:
        return FeatureFitStatistics(
            source_column=contract.source_column,
            logical_feature=contract.logical_feature,
            transform=contract.transform,
            finite_training_count=int(finite_raw.size),
            missing_training_count=missing_count,
            training_minimum=training_minimum,
            training_maximum=training_maximum,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
            training_median=training_median,
            scale=None,
            scale_branch="fixed_zero_training_constant",
            value_coefficient_active=False,
            missing_coefficient_active=missing_active,
            fixed_zero_reason="training_min_equals_max",
        )

    clipped = np.clip(transformed, lower_clip, upper_clip)
    mad_scale = MAD_SCALE_FACTOR * float(np.median(np.abs(clipped - training_median)))
    if math.isfinite(mad_scale) and mad_scale > 0.0:
        scale = mad_scale
        branch: ScaleBranch = "1.4826_times_training_MAD"
    else:
        q25, q75 = np.quantile(clipped, [0.25, 0.75], method="linear")
        iqr_scale = float(q75 - q25) / IQR_SCALE_DIVISOR
        if math.isfinite(iqr_scale) and iqr_scale > 0.0:
            scale = iqr_scale
            branch = "training_IQR_divided_by_1.349"
        else:
            population_scale = float(np.std(clipped, ddof=0))
            if not math.isfinite(population_scale) or population_scale <= 0.0:
                raise ArithmeticError(
                    "non-constant finite training values produced no positive frozen scale"
                )
            scale = population_scale
            branch = "training_population_standard_deviation"
    return FeatureFitStatistics(
        source_column=contract.source_column,
        logical_feature=contract.logical_feature,
        transform=contract.transform,
        finite_training_count=int(finite_raw.size),
        missing_training_count=missing_count,
        training_minimum=training_minimum,
        training_maximum=training_maximum,
        lower_clip=lower_clip,
        upper_clip=upper_clip,
        training_median=training_median,
        scale=scale,
        scale_branch=branch,
        value_coefficient_active=True,
        missing_coefficient_active=missing_active,
        fixed_zero_reason=None,
    )


@dataclass(frozen=True, slots=True)
class FrozenPreprocessor:
    """A deterministic preprocessor that cannot learn from assessment rows."""

    contracts: tuple[FeatureColumnContract, ...]
    statistics: tuple[FeatureFitStatistics, ...]
    fit_row_count: int
    protocol_version: str = STAGE4_MATHEMATICS_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        contracts = tuple(self.contracts)
        statistics = tuple(self.statistics)
        if not contracts or len(contracts) != len(statistics):
            raise ValueError("preprocessor contracts and statistics must be non-empty and aligned")
        if self.fit_row_count <= 0:
            raise ValueError("fit_row_count must be positive")
        if self.protocol_version != STAGE4_MATHEMATICS_PROTOCOL_VERSION:
            raise ValueError("preprocessor protocol version must be 0.4.0")
        source_columns = tuple(item.source_column for item in contracts)
        output_columns = tuple(
            name
            for item in contracts
            for name in (item.value_output_column, item.missing_output_column)
        )
        if len(set(source_columns)) != len(source_columns):
            raise ValueError("preprocessor source columns must be unique")
        if len(set(output_columns)) != len(output_columns):
            raise ValueError("preprocessor output columns must be unique")
        for contract, stats in zip(contracts, statistics, strict=True):
            if (
                contract.source_column != stats.source_column
                or contract.logical_feature != stats.logical_feature
                or contract.transform != stats.transform
            ):
                raise ValueError("preprocessor statistics do not match their feature contracts")
            if stats.finite_training_count + stats.missing_training_count != self.fit_row_count:
                raise ValueError("every feature statistic must account for all fit rows")
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "statistics", statistics)

    @property
    def design_column_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for item in self.contracts
            for name in (item.value_output_column, item.missing_output_column)
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "clipping": {
                "lower_quantile": LOWER_CLIP_QUANTILE,
                "method": "numpy_linear",
                "upper_quantile": UPPER_CLIP_QUANTILE,
            },
            "contracts": [item.as_mapping() for item in self.contracts],
            "fit_row_count": self.fit_row_count,
            "null_is_not_zero": True,
            "operation_order": [
                "validate_source_domain_and_keep_only_finite_for_statistics",
                "apply_frozen_transform",
                "compute_training_quantiles_with_numpy_linear",
                "clip_continuous_transformed_values",
                "impute_transformed_missing_with_training_median",
                "choose_training_scale_by_MAD_then_IQR_then_population_SD",
                "center_and_scale_continuous_values",
                "append_unclipped_unstandardized_binary_missing_indicators",
            ],
            "protocol_version": self.protocol_version,
            "statistics": [item.as_mapping() for item in self.statistics],
            "validation_statistics_may_change_preprocessor": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_mapping_sha256(self.as_mapping())

    def transform(self, columns: Mapping[str, object]) -> DesignMatrix:
        """Replay the frozen transform without changing any fit statistic."""

        raw_columns, row_count = _validated_columns(
            self.contracts,
            columns,
            allow_empty=True,
        )
        output = np.empty((row_count, len(self.design_column_names)), dtype=np.float64)
        penalties = np.empty(len(self.design_column_names), dtype=np.float64)
        active = np.empty(len(self.design_column_names), dtype=np.bool_)
        for feature_index, (contract, stats) in enumerate(
            zip(self.contracts, self.statistics, strict=True)
        ):
            raw = raw_columns[contract.source_column]
            finite = np.isfinite(raw)
            transformed = _transform_finite(
                raw[finite],
                contract.transform,
                source_column=contract.source_column,
            )
            value_column_index = feature_index * 2
            missing_column_index = value_column_index + 1
            value_output = np.zeros(row_count, dtype=np.float64)
            if stats.value_coefficient_active:
                median = stats.training_median
                scale = stats.scale
                if median is None or scale is None:  # pragma: no cover - guarded by contract
                    raise AssertionError("active feature omitted frozen median or scale")
                if contract.is_binary:
                    value_output.fill(median)
                    value_output[finite] = transformed
                else:
                    lower = stats.lower_clip
                    upper = stats.upper_clip
                    if lower is None or upper is None:  # pragma: no cover
                        raise AssertionError("continuous feature omitted clipping bounds")
                    value_output.fill(median)
                    value_output[finite] = np.clip(transformed, lower, upper)
                    value_output -= median
                    value_output /= scale
            output[:, value_column_index] = value_output
            output[:, missing_column_index] = (~finite).astype(np.float64)
            penalties[value_column_index : missing_column_index + 1] = contract.penalty_factor
            active[value_column_index] = stats.value_coefficient_active
            active[missing_column_index] = stats.missing_coefficient_active
        return DesignMatrix(
            values=output,
            column_names=self.design_column_names,
            penalty_factors=penalties,
            active_coefficients=active,
        )


def fit_frozen_preprocessor(
    contracts: Sequence[FeatureColumnContract],
    columns: Mapping[str, object],
) -> FrozenPreprocessor:
    """Fit the preregistered transforms on one shared 7-day fit scope."""

    frozen_contracts = tuple(contracts)
    if not frozen_contracts:
        raise ValueError("at least one feature contract is required")
    raw_columns, row_count = _validated_columns(
        frozen_contracts,
        columns,
        allow_empty=False,
    )
    statistics = tuple(
        _fit_feature_statistics(contract, raw_columns[contract.source_column])
        for contract in frozen_contracts
    )
    return FrozenPreprocessor(
        contracts=frozen_contracts,
        statistics=statistics,
        fit_row_count=row_count,
    )


__all__ = [
    "IQR_SCALE_DIVISOR",
    "LOWER_CLIP_QUANTILE",
    "MAD_SCALE_FACTOR",
    "UPPER_CLIP_QUANTILE",
    "FeatureFitStatistics",
    "FrozenPreprocessor",
    "PreprocessingDomainError",
    "fit_frozen_preprocessor",
]
