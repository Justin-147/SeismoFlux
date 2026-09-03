"""In-memory S3 fitting; no source loading, outer scoring, or calendar selection.

Inputs already contain the 20 transformed features from ``features.py``. The
caller supplies only calendar-authorized, mature labels. Every inner fit learns
its own imputation and scaling; later validation features never enter that fit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp  # type: ignore[import-untyped]

from seismoflux.multitask_s1.c2b_models import (
    C2BFitError,
    C2BRidgeFit,
    C2BTrainingIssue,
    fit_spatial_ridge,
)
from seismoflux.multitask_s3.features import DESIGN_FEATURE_IDS, DESIGN_INDICES
from seismoflux.multitask_s3.models import (
    OffsetPoissonFit,
    TrainingAreaImputer,
    fit_offset_poisson_intercept,
    fit_offset_poisson_ridge,
    fit_training_area_imputer,
)

FloatArray = NDArray[np.float64]
Design = Literal["COV", "SNAP", "DYN"]
RIDGE_CANDIDATES = (0.1, 1.0, 10.0)
DEFAULT_RIDGE = 10.0
TIE_TOLERANCE = 1.0e-10


def _readonly(values: FloatArray) -> FloatArray:
    # Share the already read-only feature matrix across horizons without copying
    # hundreds of MB per fold; copy mutable callers before retaining their data.
    array = np.asarray(values, dtype=np.float64)
    if array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, dtype=np.float64, copy=True, order="C")
    array.setflags(write=False)
    return array


def _area(values: FloatArray) -> FloatArray:
    area = np.asarray(values, dtype=np.float64)
    if area.ndim != 1 or not area.size or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("actual areas must be a nonempty finite positive vector")
    return _readonly(area)


def _weights(area: FloatArray) -> FloatArray:
    scaled = area / np.max(area)
    return scaled / np.sum(scaled)


def _features(values: FloatArray, cells: int) -> FloatArray:
    features = np.asarray(values, dtype=np.float64)
    if features.shape != (cells, 20) or np.isinf(features).any():
        raise ValueError("features must have shape (cells, 20), finite or NaN")
    return features


def _log_mass(values: FloatArray, cells: int) -> FloatArray:
    base = np.asarray(values, dtype=np.float64)
    if base.shape != (cells,) or not np.isfinite(base).all():
        raise ValueError("background log mass must be a finite aligned vector")
    return np.asarray(base - logsumexp(base), dtype=np.float64)


def _ridge(value: float) -> float:
    result = float(value)
    if result not in RIDGE_CANDIDATES:
        raise ValueError("S3 ridge must be one of the frozen 0.1, 1, 10 candidates")
    return result


@dataclass(frozen=True, slots=True)
class S3TrainingSample:
    """One complete window; Ms4+ spatial counts and Ms5+ nationwide count."""

    issue_time_utc: datetime
    features: FloatArray
    background_log_mass: FloatArray
    offset_ms5plus: float
    spatial_event_counts: FloatArray
    count_ms5plus: int

    def __post_init__(self) -> None:
        issue = self.issue_time_utc
        if not isinstance(issue, datetime) or issue.tzinfo is None or issue.utcoffset() is None:
            raise ValueError("issue time must be a timezone-aware datetime")
        base = np.asarray(self.background_log_mass, dtype=np.float64)
        if base.ndim != 1 or not base.size or not np.isfinite(base).all():
            raise ValueError("background log mass must be a nonempty finite vector")
        features = _features(self.features, base.size)
        counts = np.asarray(self.spatial_event_counts, dtype=np.float64)
        if (
            counts.shape != base.shape
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
            or not np.array_equal(counts, np.floor(counts))
            or not math.isfinite(float(np.sum(counts)))
        ):
            raise ValueError("spatial counts must be aligned finite nonnegative integers")
        count = self.count_ms5plus
        if isinstance(count, bool | np.bool_) or not isinstance(count, int | np.integer):
            raise ValueError("count_ms5plus must be a nonnegative integer")
        if count < 0:
            raise ValueError("count_ms5plus must be a nonnegative integer")
        offset = float(self.offset_ms5plus)
        if not math.isfinite(offset) or offset <= 0:
            raise ValueError("offset_ms5plus must be finite and positive")
        object.__setattr__(self, "issue_time_utc", issue.astimezone(UTC))
        object.__setattr__(self, "features", _readonly(features))
        object.__setattr__(self, "background_log_mass", _readonly(base))
        object.__setattr__(self, "spatial_event_counts", _readonly(counts))
        object.__setattr__(self, "offset_ms5plus", offset)
        object.__setattr__(self, "count_ms5plus", int(count))

    @property
    def spatial_event_count(self) -> int:
        return int(np.sum(self.spatial_event_counts))


def _samples(values: Sequence[S3TrainingSample], cells: int) -> tuple[S3TrainingSample, ...]:
    samples = tuple(values)
    if len({sample.issue_time_utc for sample in samples}) != len(samples):
        raise ValueError("duplicate issue in one training or validation sample set")
    if any(sample.features.shape != (cells, 20) for sample in samples):
        raise ValueError("every sample must align with the independent actual-area grid")
    return tuple(sorted(samples, key=lambda sample: sample.issue_time_utc))


@dataclass(frozen=True, slots=True)
class S3InnerBlock:
    block_id: str
    training_samples: tuple[S3TrainingSample, ...]
    validation_samples: tuple[S3TrainingSample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("inner block_id must be nonempty")
        training, validation = tuple(self.training_samples), tuple(self.validation_samples)
        if (
            training
            and validation
            and max(item.issue_time_utc for item in training)
            >= min(item.issue_time_utc for item in validation)
        ):
            raise ValueError("inner training issues must precede every validation issue")
        object.__setattr__(self, "training_samples", training)
        object.__setattr__(self, "validation_samples", validation)


@dataclass(frozen=True, slots=True)
class S3Performance:
    """Unpenalized losses; zeros remain count observations, not spatial events."""

    issue_count: int
    spatial_event_count: int
    count_event_count: int
    spatial_nll_sum: float | None
    count_nll_sum: float | None
    catalog_spatial_nll_sum: float | None
    baseline_count_nll_sum: float | None
    calibrated_count_nll_sum: float | None

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "issue_count": self.issue_count,
            "spatial_event_count": self.spatial_event_count,
            "count_event_count": self.count_event_count,
        }
        for key in (
            "spatial_nll",
            "count_nll",
            "catalog_spatial_nll",
            "baseline_count_nll",
            "calibrated_count_nll",
        ):
            value = getattr(self, f"{key}_sum")
            denominator = self.spatial_event_count if "spatial" in key else self.issue_count
            values[f"{key}_sum"] = value
            values[f"{key}_mean"] = (
                value / denominator if value is not None and denominator else None
            )
        return values


@dataclass(frozen=True, slots=True)
class S3InnerScore:
    block_id: str
    ridge_lambda: float
    spatial_eligible: bool
    count_eligible: bool
    spatial_status: str
    count_status: str
    training: S3Performance
    validation: S3Performance
    imputation_fill_values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "ridge_lambda": self.ridge_lambda,
            "spatial_eligible": self.spatial_eligible,
            "count_eligible": self.count_eligible,
            "spatial_status": self.spatial_status,
            "count_status": self.count_status,
            "training": self.training.to_dict(),
            "validation": self.validation.to_dict(),
            "imputation_fill_values": list(self.imputation_fill_values),
        }


@dataclass(frozen=True, slots=True)
class S3Selection:
    spatial_ridge_lambda: float
    count_ridge_lambda: float
    spatial_reason: str
    count_reason: str
    inner_scores: tuple[S3InnerScore, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "spatial_ridge_lambda": self.spatial_ridge_lambda,
            "count_ridge_lambda": self.count_ridge_lambda,
            "spatial_reason": self.spatial_reason,
            "count_reason": self.count_reason,
            "candidates": list(RIDGE_CANDIDATES),
            "required_evaluable_inner_blocks": 2,
            "absolute_tie_tolerance": TIE_TOLERANCE,
            "inner_scores": [score.to_dict() for score in self.inner_scores],
        }


@dataclass(frozen=True, slots=True)
class S3ModelFit:
    design: Design
    areas_km2: FloatArray
    imputer: TrainingAreaImputer
    spatial: C2BRidgeFit | None
    spatial_status: str
    spatial_message: str
    spatial_ridge_lambda: float
    count: OffsetPoissonFit
    count_calibration: OffsetPoissonFit
    training_performance: S3Performance | None = None
    selection: S3Selection | None = None

    def _prepared(self, features: FloatArray) -> FloatArray:
        values = _features(features, self.areas_km2.size)
        return self.imputer.transform(values[:, DESIGN_INDICES[self.design]])

    def predict_log_mass(self, features: FloatArray, background_log_mass: FloatArray) -> FloatArray:
        """Relative cell mass, not an absolute earthquake occurrence probability."""
        values = self._prepared(features)
        base = _log_mass(background_log_mass, self.areas_km2.size)
        if self.spatial is None:
            return _readonly(base)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return self.spatial.predict_log_mass(base, values)

    def predict_log_mean(self, features: FloatArray, offset_ms5plus: float) -> float:
        """Use finite log means directly in Poisson scores, without exponentiating."""
        values = self._prepared(features)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            national = (_weights(self.areas_km2) @ values)[None, :]
        return float(self.count.predict_log_mean(national, np.array([offset_ms5plus]))[0])

    def predict_calibrated_log_mean(self, offset_ms5plus: float) -> float:
        return float(
            self.count_calibration.predict_log_mean(np.empty((1, 0)), np.array([offset_ms5plus]))[0]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "design": self.design,
            "feature_ids": list(DESIGN_FEATURE_IDS[self.design]),
            "cell_count": self.areas_km2.size,
            "total_area_km2": float(np.sum(self.areas_km2)),
            "imputer": {
                "fill_values": self.imputer.fill_values.tolist(),
                "active_columns": self.imputer.active_columns.tolist(),
                "training_issue_count": self.imputer.training_issue_count,
            },
            "spatial": None if self.spatial is None else self.spatial.to_dict(),
            "spatial_status": self.spatial_status,
            "spatial_message": self.spatial_message,
            "spatial_ridge_lambda": self.spatial_ridge_lambda,
            "count": self.count.to_dict(),
            "count_calibration": self.count_calibration.to_dict(),
            "training_performance": (
                None if self.training_performance is None else self.training_performance.to_dict()
            ),
            "selection": None if self.selection is None else self.selection.to_dict(),
        }


def _poisson_nll(log_mean: float, count: int) -> float:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        value = float(np.exp(log_mean) - count * log_mean + gammaln(count + 1.0))
    if not math.isfinite(value):
        raise FloatingPointError("Poisson validation loss is not finite")
    return value


def _performance(fit: S3ModelFit, samples: tuple[S3TrainingSample, ...]) -> S3Performance:
    spatial_events = sum(sample.spatial_event_count for sample in samples)
    spatial, catalog, counts, baseline, calibration = [], [], [], [], []
    failed_spatial = failed_count = failed_baseline = failed_calibration = False
    for sample in samples:
        if sample.spatial_event_count:
            catalog.append(
                -float(
                    sample.spatial_event_counts
                    @ _log_mass(sample.background_log_mass, fit.areas_km2.size)
                )
            )
            try:
                value = -float(
                    sample.spatial_event_counts
                    @ fit.predict_log_mass(sample.features, sample.background_log_mass)
                )
                if not math.isfinite(value):
                    raise FloatingPointError("spatial validation loss is not finite")
                spatial.append(value)
            except (FloatingPointError, OverflowError):
                failed_spatial = True
        try:
            counts.append(
                _poisson_nll(
                    fit.predict_log_mean(sample.features, sample.offset_ms5plus),
                    sample.count_ms5plus,
                )
            )
        except (FloatingPointError, OverflowError):
            failed_count = True
        try:
            baseline.append(_poisson_nll(math.log(sample.offset_ms5plus), sample.count_ms5plus))
        except (FloatingPointError, OverflowError):
            failed_baseline = True
        try:
            calibration.append(
                _poisson_nll(
                    fit.predict_calibrated_log_mean(sample.offset_ms5plus), sample.count_ms5plus
                )
            )
        except (FloatingPointError, OverflowError):
            failed_calibration = True
    return S3Performance(
        len(samples),
        spatial_events,
        sum(sample.count_ms5plus for sample in samples),
        None if failed_spatial or not spatial_events else math.fsum(spatial),
        None if failed_count or not samples else math.fsum(counts),
        math.fsum(catalog) if spatial_events else None,
        None if failed_baseline or not samples else math.fsum(baseline),
        None if failed_calibration or not samples else math.fsum(calibration),
    )


def fit_model(
    training_samples: Sequence[S3TrainingSample],
    *,
    design: Design,
    areas_km2: FloatArray,
    ridge_lambda: float,
    count_ridge_lambda: float | None = None,
) -> S3ModelFit:
    """Fit both tasks and the intercept-only reference using only these samples."""
    if design not in DESIGN_INDICES:
        raise ValueError("design must be COV, SNAP, or DYN")
    spatial_ridge = _ridge(ridge_lambda)
    count_ridge = _ridge(ridge_lambda if count_ridge_lambda is None else count_ridge_lambda)
    area = _area(areas_km2)
    samples = _samples(training_samples, area.size)
    indices = DESIGN_INDICES[design]
    transformed = (
        np.stack([sample.features[:, indices] for sample in samples])
        if samples
        else np.empty((0, area.size, len(indices)))
    )
    imputer = fit_training_area_imputer(transformed, area)
    prepared = imputer.transform(transformed)
    # Area aggregation retains zero-event issues, unlike the spatial likelihood.
    national = np.einsum("c,icf->if", _weights(area), prepared)
    offsets = np.array([sample.offset_ms5plus for sample in samples], dtype=np.float64)
    counts = np.array([sample.count_ms5plus for sample in samples], dtype=np.float64)
    count = fit_offset_poisson_ridge(national, offsets, counts, ridge_lambda=count_ridge)
    calibration = fit_offset_poisson_intercept(offsets, counts)
    spatial = None
    if not samples:
        spatial_status, message = "baseline_no_training", "No legal training issues."
    elif not any(sample.spatial_event_count for sample in samples):
        spatial_status, message = "baseline_no_training_events", "No Ms4+ training events."
    else:
        issues = tuple(
            C2BTrainingIssue(
                sample.issue_time_utc.isoformat(),
                sample.background_log_mass,
                prepared[index],
                sample.spatial_event_counts,
            )
            for index, sample in enumerate(samples)
        )
        try:
            spatial = fit_spatial_ridge(issues, area, ridge_lambda=spatial_ridge)
            spatial_status, message = spatial.optimizer_status, spatial.optimizer_message
        except C2BFitError as exc:
            spatial_status, message = "baseline_fit_not_evaluable", str(exc)
    fitted = S3ModelFit(
        design, area, imputer, spatial, spatial_status, message, spatial_ridge, count, calibration
    )
    return replace(fitted, training_performance=_performance(fitted, samples))


def _eligible(block: S3InnerBlock, *, spatial: bool) -> bool:
    if not block.training_samples or not block.validation_samples:
        return False
    if spatial:
        return any(sample.spatial_event_count for sample in block.training_samples) and any(
            sample.spatial_event_count for sample in block.validation_samples
        )
    return any(sample.count_ms5plus for sample in block.training_samples)


def _select(scores: tuple[S3InnerScore, ...], *, spatial: bool) -> tuple[float, str]:
    eligible_ids = {
        score.block_id
        for score in scores
        if (score.spatial_eligible if spatial else score.count_eligible)
    }
    if len(eligible_ids) < 2:
        return DEFAULT_RIDGE, "fixed_10_fewer_than_two_evaluable_inner_blocks"
    candidate_values: list[tuple[float, float]] = []
    for ridge in RIDGE_CANDIDATES:
        rows = [
            score
            for score in scores
            if score.ridge_lambda == ridge and score.block_id in eligible_ids
        ]
        values: list[float] = []
        denominator = 0
        failed = len(rows) != len(eligible_ids)
        for row in rows:
            status = row.spatial_status if spatial else row.count_status
            value = row.validation.spatial_nll_sum if spatial else row.validation.count_nll_sum
            if status.startswith("baseline_") or value is None:
                failed = True
                break
            values.append(value)
            denominator += (
                row.validation.spatial_event_count if spatial else row.validation.issue_count
            )
        if not failed and denominator:
            candidate_values.append((ridge, math.fsum(values) / denominator))
    if not candidate_values:
        return DEFAULT_RIDGE, "fixed_10_no_candidate_evaluable_on_all_registered_blocks"
    best = min(value for _, value in candidate_values)
    chosen = max(ridge for ridge, value in candidate_values if abs(value - best) <= TIE_TOLERANCE)
    return chosen, "pooled_inner_event_nll" if spatial else "pooled_inner_issue_poisson_nll"


def select_and_fit(
    training_samples: Sequence[S3TrainingSample],
    *,
    inner_blocks: Sequence[S3InnerBlock],
    design: Design,
    areas_km2: FloatArray,
) -> S3ModelFit:
    """Select each task independently, then refit on the supplied outer training.

    Failed candidate fits are recorded and cannot benefit from dropping a hard
    validation block. A fully unevaluable selection uses the frozen lambda 10;
    this is numerical availability handling, not a scientific improvement gate.
    No outer validation samples or scores are accepted by this interface.
    """
    area = _area(areas_km2)
    samples = _samples(training_samples, area.size)
    blocks = tuple(inner_blocks)
    if len({block.block_id for block in blocks}) != len(blocks):
        raise ValueError("duplicate inner block_id")
    blocks = tuple(
        S3InnerBlock(
            block.block_id,
            _samples(block.training_samples, area.size),
            _samples(block.validation_samples, area.size),
        )
        for block in blocks
    )
    needs_selection = any(
        sum(_eligible(block, spatial=spatial) for block in blocks) >= 2 for spatial in (True, False)
    )
    candidates = RIDGE_CANDIDATES if needs_selection else (DEFAULT_RIDGE,)
    scores: list[S3InnerScore] = []
    for block in blocks:
        for ridge in candidates:
            inner_fit = fit_model(
                block.training_samples, design=design, areas_km2=area, ridge_lambda=ridge
            )
            assert inner_fit.training_performance is not None
            scores.append(
                S3InnerScore(
                    block.block_id,
                    ridge,
                    _eligible(block, spatial=True),
                    _eligible(block, spatial=False),
                    inner_fit.spatial_status,
                    inner_fit.count.status,
                    inner_fit.training_performance,
                    _performance(inner_fit, block.validation_samples),
                    tuple(float(value) for value in inner_fit.imputer.fill_values),
                )
            )
    spatial_ridge, spatial_reason = _select(tuple(scores), spatial=True)
    count_ridge, count_reason = _select(tuple(scores), spatial=False)
    fitted = fit_model(
        samples,
        design=design,
        areas_km2=area,
        ridge_lambda=spatial_ridge,
        count_ridge_lambda=count_ridge,
    )
    selection = S3Selection(spatial_ridge, count_ridge, spatial_reason, count_reason, tuple(scores))
    return replace(fitted, selection=selection)


__all__ = [
    "DEFAULT_RIDGE",
    "RIDGE_CANDIDATES",
    "S3InnerBlock",
    "S3InnerScore",
    "S3ModelFit",
    "S3Performance",
    "S3Selection",
    "S3TrainingSample",
    "fit_model",
    "select_and_fit",
]
