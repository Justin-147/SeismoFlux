# ruff: noqa: E501, RUF001
"""Auditable static and offline-interactive deliverables for the D1 replay.

The renderer consumes only a *completed* observed retrospective replay.  It
does not fit a model, select an alarm from targets, or run a prospective issue.
Targets are read only after the alarm ranks have been frozen and are used as a
retrospective overlay.  Alarm exposure is summarized over every assessment
issue, including issues without an M5--6 target.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import re
import sys
import tempfile
from array import array
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, cast
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pyarrow.parquet as pq

from seismoflux.d1_replay.evaluation import (
    D1_AREA_BUDGETS_KM2,
    D1_ASSESSMENT_ISSUES_PER_FOLD,
    D1_FOLD_IDS,
    D1_HORIZONS_DAYS,
    D1_MODEL_ORDER,
    D1_PRIMARY_AREA_KM2,
    D1_PRIMARY_HORIZON_DAYS,
    D1AlarmExposureMetric,
    D1BootstrapEffect,
    D1ClusterModelOutcome,
    D1IssueAlarmOutcome,
    D1Metric,
    D1RawEffectDecision,
    classify_primary_raw_effect,
    minimum_area_reaching_recall,
    paired_cluster_bootstrap,
    summarize_alarm_exposure,
    summarize_metrics,
    validate_complete_alarm_outcomes,
)
from seismoflux.data.common import canonical_json_bytes, write_json_atomic

RETROSPECTIVE_LABEL: Final = "真实历史开发回放，不是真正前瞻预测"
STRENGTH_LABEL: Final = "相对条件强度/顺位，不是绝对发震概率"
PLACEBO_PENDING_LABEL: Final = (
    "时间置乱、空间置乱、区域贡献和去单震群诊断尚未完成；当前只能预分类原始预测效果。"
)
ROBUSTNESS_PENDING_LABEL: Final = "区域贡献和去单震群稳健性诊断尚未完成。"
FULL_MODEL: Final = "B0_R30_C_A_dynamic"
_SCHEMA_VERSION: Final = 1
_PROTOCOL_VERSION: Final = "d1.0.0"
_EXPECTED_CLUSTER_COUNTS: Final = {30: 21, 90: 22}
_EXPECTED_CLUSTER_COUNTS_BY_FOLD: Final = {
    30: {"fold_1": 8, "fold_2": 6, "fold_3": 7},
    90: {"fold_1": 8, "fold_2": 6, "fold_3": 8},
}
_EXPECTED_ISSUE_COUNTS: Final = {30: 24, 90: 9}
_EXPECTED_ISSUE_COUNTS_BY_FOLD: Final = {
    30: {"fold_1": 8, "fold_2": 8, "fold_3": 8},
    90: {"fold_1": 3, "fold_2": 3, "fold_3": 3},
}
_PLACEBO_REPLICATIONS: Final = 200
_PLACEBO_DENOMINATOR: Final = 201
_ROBUSTNESS_CLUSTER_COUNT: Final = 21
_ROBUSTNESS_ZONE_COUNT: Final = 39
_ROBUSTNESS_SPATIAL_ARTIFACTS: Final = (
    "cell_mapping",
    "entity_mapping",
    "zone_geometry",
    "connectors",
)
_ROBUSTNESS_TARGET_MAPPING_ROLE: Final = (
    "posthoc_observed_cluster_to_target_independent_zone_only_never_alarm_generation"
)
_COMPONENT_SPECS: Final = (
    (
        "B0_C_A_snapshot_minus_B0_C",
        "单期异常增量",
        "B0_C_A_snapshot",
        "B0_C",
    ),
    (
        "B0_C_A_dynamic_minus_B0_C",
        "全部异常增量",
        "B0_C_A_dynamic",
        "B0_C",
    ),
    (
        "B0_C_A_dynamic_minus_B0_C_A_snapshot",
        "动态演化增量",
        "B0_C_A_dynamic",
        "B0_C_A_snapshot",
    ),
)
_COMPONENT_CONTRASTS: Final = tuple(item[0] for item in _COMPONENT_SPECS)
INTERMEDIATE_MODEL_ORDER: Final = (
    "B0_R30",
    "B0_C",
    "B0_C_A_snapshot",
    "B0_C_A_dynamic",
)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.:+-]{1,256}")
_AREA_COLUMN_SUFFIXES: Final = ("300000", "450000", "600000", "750000", "960000")
_MODEL_COLORS: Final = {
    "B0": "#222222",
    "B0_R30": "#0072B2",
    "B0_C": "#009E73",
    "B0_C_A_snapshot": "#E69F00",
    "B0_C_A_dynamic": "#CC79A7",
    "B0_R30_C_A_dynamic": "#D55E00",
}
_MODEL_ZH: Final = {
    "B0": "长期地震背景",
    "B0_R30": "背景+近30天地震",
    "B0_C": "背景+报告覆盖",
    "B0_C_A_snapshot": "背景+覆盖+单期异常",
    "B0_C_A_dynamic": "背景+覆盖+单期及动态异常",
    "B0_R30_C_A_dynamic": "近震+覆盖+全部异常组合",
}
_MODEL_BASE: Final = {
    "B0": "B0",
    "B0_R30": "R30",
    "B0_C": "B0",
    "B0_C_A_snapshot": "B0",
    "B0_C_A_dynamic": "B0",
    "B0_R30_C_A_dynamic": "R30",
}
_MODEL_FEATURE_GROUPS: Final = {
    "B0": (),
    "B0_R30": (),
    "B0_C": ("C1", "C2"),
    "B0_C_A_snapshot": ("C1", "C2", "S1", "S2", "S3", "S4", "S5"),
    "B0_C_A_dynamic": (
        "C1",
        "C2",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "D1",
        "D2",
    ),
    "B0_R30_C_A_dynamic": (
        "C1",
        "C2",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "D1",
        "D2",
    ),
}


class D1RenderingError(ValueError):
    """Raised when a purported observed replay cannot be rendered safely."""


@dataclass(frozen=True, slots=True)
class D1RenderedDeliverables:
    """Paths and identities of one complete D1 rendering bundle."""

    effects_svg_path: Path
    maps_svg_path: Path
    science_report_path: Path
    explorer_html_path: Path
    science_summary_path: Path
    manifest_path: Path
    manifest_identity_sha256: str
    manifest_file_sha256: str
    best_intermediate_model: str


@dataclass(frozen=True, slots=True)
class _ComponentEvidence:
    contrast: str
    label: str
    candidate_model: str
    reference_model: str
    observed_recall_difference: float
    time_p_value: float | None
    space_p_value: float | None
    time_status: str
    space_status: str
    attribution_status: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "contrast": self.contrast,
            "label": self.label,
            "candidate_model": self.candidate_model,
            "reference_model": self.reference_model,
            "horizon_days": D1_PRIMARY_HORIZON_DAYS,
            "area_budget_km2": D1_PRIMARY_AREA_KM2,
            "observed_recall_difference": self.observed_recall_difference,
            "time_placebo_p_value": self.time_p_value,
            "space_placebo_p_value": self.space_p_value,
            "time_placebo_status": self.time_status,
            "space_placebo_status": self.space_status,
            "attribution_status": self.attribution_status,
        }


@dataclass(frozen=True, slots=True)
class _SupplementalEvidence:
    status: str
    path: Path | None
    file_sha256: str | None
    payload: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class _OutcomeOverlay:
    cluster_id: str
    frame_key: tuple[str, str, str, int, str]
    representative_cell_index: int | None
    outside_support: bool
    hit_by_area: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _IssueRow:
    fold_id: str
    issue_id: str
    issue_time_utc: str
    horizon_days: int
    model_id: str
    alarm_prefix_counts: tuple[int, ...]
    actual_area_km2: tuple[float, ...]

    @property
    def frame_key(self) -> tuple[str, str, str, int, str]:
        return (
            self.fold_id,
            self.issue_id,
            self.issue_time_utc,
            self.horizon_days,
            self.model_id,
        )


@dataclass(frozen=True, slots=True)
class _GridCell:
    cell_index: int
    cell_id: str
    row: int
    column: int
    x_m: float
    y_m: float
    area_km2: float


@dataclass(frozen=True, slots=True)
class _TargetMarker:
    cluster_id: str
    cell_position: int | None
    outside_support: bool
    hit_by_area: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _EncodedFrame:
    fold_id: str
    issue_id: str
    issue_time_utc: str
    horizon_days: int
    model_id: str
    order_u16_b64: str
    strength_u16_b64: str
    strength_log_min: float
    strength_log_max: float
    alarm_prefix_counts: tuple[int, ...]
    actual_area_km2: tuple[float, ...]
    targets: tuple[_TargetMarker, ...]

    @property
    def frame_key(self) -> tuple[str, str, str, int, str]:
        return (
            self.fold_id,
            self.issue_id,
            self.issue_time_utc,
            self.horizon_days,
            self.model_id,
        )


@dataclass(frozen=True, slots=True)
class _FrameBuildResult:
    grid: tuple[_GridCell, ...]
    frames: tuple[_EncodedFrame, ...]
    study_area_km2: float


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise D1RenderingError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise D1RenderingError(f"{label} must be a list")
    return cast(Sequence[object], value)


def _safe_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise D1RenderingError(f"{label} must be a safe non-empty identifier")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise D1RenderingError(f"{label} must be a non-empty string")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise D1RenderingError(f"{label} must be an integer >= {minimum}")
    return value


def _signed_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise D1RenderingError(f"{label} must be an integer")
    return value


def _finite_float(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise D1RenderingError(f"{label} must be numeric")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise D1RenderingError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise D1RenderingError(f"{label} is outside its finite range")
    return result


def _canonical_timestamp(value: object, *, label: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = _string(value, label=label)
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise D1RenderingError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise D1RenderingError(f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(value)
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _read_observed_result(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise D1RenderingError("observed_result_path is not readable canonical JSON") from exc
    payload = _mapping(raw, label="observed result")
    if payload.get("result_kind") != "observed_replay":
        raise D1RenderingError("rendering accepts observed_replay only")
    if payload.get("status") != "completed":
        raise D1RenderingError("rendering requires a completed observed replay")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise D1RenderingError("observed replay schema_version must equal 1")
    if payload.get("protocol_version") != _PROTOCOL_VERSION:
        raise D1RenderingError("observed replay protocol_version must equal d1.0.0")
    if payload.get("retrospective_only") is not True:
        raise D1RenderingError("observed replay must be explicitly retrospective-only")
    if payload.get("relative_strength_not_absolute_probability") is not True:
        raise D1RenderingError("observed replay omitted relative-strength semantics")
    _observed_identities(payload)
    return payload


def _sha256_identity(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise D1RenderingError(f"{label} must be a lowercase SHA-256")
    return text


def _git_identity(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if re.fullmatch(r"[0-9a-f]{40,64}", text) is None:
        raise D1RenderingError(f"{label} must be a lowercase git object identity")
    return text


def _observed_identities(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    identities = _mapping(payload.get("identities"), label="observed identities")
    _sha256_identity(identities.get("contract_sha256"), label="observed contract_sha256")
    _sha256_identity(
        identities.get("manifest_content_sha256"),
        label="observed manifest_content_sha256",
    )
    _sha256_identity(identities.get("input_sha256"), label="observed input_sha256")
    _git_identity(identities.get("git_commit"), label="observed git_commit")
    return identities


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise D1RenderingError(f"{label} is not readable JSON") from exc
    return _mapping(raw, label=label)


def _validate_supplemental_identity(
    supplemental: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    label: str,
    placebo_input_binding: bool,
) -> None:
    if supplemental.get("schema_version") != _SCHEMA_VERSION:
        raise D1RenderingError(f"{label} schema_version must equal 1")
    if supplemental.get("protocol_version") != _PROTOCOL_VERSION:
        raise D1RenderingError(f"{label} protocol_version must equal d1.0.0")
    observed_ids = _observed_identities(observed)
    identities = _mapping(supplemental.get("identities"), label=f"{label} identities")
    _sha256_identity(identities.get("input_sha256"), label=f"{label} input_sha256")
    for key in ("contract_sha256", "manifest_content_sha256", "git_commit"):
        if identities.get(key) != observed_ids.get(key):
            raise D1RenderingError(f"{label} {key} differs from observed replay")
    expected_input_key = "observed_input_sha256" if placebo_input_binding else "input_sha256"
    _sha256_identity(
        identities.get(expected_input_key),
        label=f"{label} {expected_input_key}",
    )
    if identities.get(expected_input_key) != observed_ids.get("input_sha256"):
        raise D1RenderingError(f"{label} input identity differs from observed replay")


def _expected_component_differences(
    metrics: Sequence[D1Metric],
) -> dict[str, float]:
    lookup = _metric_lookup(metrics)
    result: dict[str, float] = {}
    for contrast, _label, candidate, reference in _COMPONENT_SPECS:
        candidate_metric = lookup[(candidate, D1_PRIMARY_HORIZON_DAYS, None, D1_PRIMARY_AREA_KM2)]
        reference_metric = lookup[(reference, D1_PRIMARY_HORIZON_DAYS, None, D1_PRIMARY_AREA_KM2)]
        if candidate_metric.recall is None or reference_metric.recall is None:
            raise D1RenderingError("component contribution requires pooled primary recall")
        result[contrast] = candidate_metric.recall - reference_metric.recall
    return result


def _validate_placebo_result(
    path: Path | None,
    *,
    observed: Mapping[str, Any],
    metrics: Sequence[D1Metric],
) -> _SupplementalEvidence:
    if path is None:
        return _SupplementalEvidence("pending", None, None, None)
    if not path.is_file():
        raise D1RenderingError("placebo_result_path does not exist")
    payload = _read_json_mapping(path, label="placebo result")
    if payload.get("result_kind") != "d1_time_and_space_placebos":
        raise D1RenderingError("placebo result_kind is not d1_time_and_space_placebos")
    if payload.get("status") != "completed" or payload.get("retrospective_only") is not True:
        raise D1RenderingError("placebo result must be completed and retrospective-only")
    if payload.get("replications_each") != _PLACEBO_REPLICATIONS:
        raise D1RenderingError("placebo result must contain 200 time and 200 space replications")
    schedule = _mapping(
        payload.get("schedule_by_kind_and_fold"),
        label="placebo schedule_by_kind_and_fold",
    )
    if set(schedule) != {"time", "space"}:
        raise D1RenderingError("placebo schedule must contain time and space")
    for kind in ("time", "space"):
        by_fold = _mapping(schedule[kind], label=f"{kind} placebo schedule")
        if set(by_fold) != set(D1_FOLD_IDS) or any(
            value != _PLACEBO_REPLICATIONS for value in by_fold.values()
        ):
            raise D1RenderingError("each placebo kind must register 200 replications in every fold")
    _validate_supplemental_identity(
        payload,
        observed,
        label="placebo result",
        placebo_input_binding=True,
    )
    expected_observed = _expected_component_differences(metrics)
    observed_statistics = _mapping(
        payload.get("observed_statistics"), label="placebo observed_statistics"
    )
    if set(observed_statistics) != set(_COMPONENT_CONTRASTS):
        raise D1RenderingError("placebo observed_statistics changed the three contrasts")
    for contrast, expected in expected_observed.items():
        actual = _finite_float(observed_statistics[contrast], label=f"placebo observed {contrast}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise D1RenderingError("placebo observed statistic differs from observed replay")

    kinds = _mapping(payload.get("kinds"), label="placebo kinds")
    if set(kinds) != {"time", "space"}:
        raise D1RenderingError("placebo result must contain time and space kinds")
    promising_by_kind: dict[str, dict[str, bool]] = {}
    for kind in ("time", "space"):
        kind_payload = _mapping(kinds[kind], label=f"{kind} placebo")
        if kind_payload.get("kind") != kind:
            raise D1RenderingError(f"{kind} placebo kind identity changed")
        fold_failures = _mapping(
            kind_payload.get("fold_scientific_failure_counts"),
            label=f"{kind} fold failures",
        )
        if set(fold_failures) != set(D1_FOLD_IDS):
            raise D1RenderingError(f"{kind} placebo fold failure axis changed")
        for fold, value in fold_failures.items():
            count = _exact_int(value, label=f"{kind} {fold} failure count")
            if count > _PLACEBO_REPLICATIONS:
                raise D1RenderingError("placebo fold failure count exceeds 200")
        contrasts = _mapping(kind_payload.get("contrasts"), label=f"{kind} placebo contrasts")
        if set(contrasts) != set(_COMPONENT_CONTRASTS):
            raise D1RenderingError("placebo result changed the three preregistered contrasts")
        promising_by_kind[kind] = {}
        for contrast in _COMPONENT_CONTRASTS:
            row = _mapping(contrasts[contrast], label=f"{kind} {contrast}")
            if row.get("contrast") != contrast:
                raise D1RenderingError("placebo contrast identity changed")
            observed_statistic = _finite_float(
                row.get("observed_statistic"), label="placebo observed_statistic"
            )
            if not math.isclose(
                observed_statistic,
                expected_observed[contrast],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise D1RenderingError("placebo contrast observed statistic changed")
            null_values = _sequence(row.get("null_statistics"), label="placebo null_statistics")
            if len(null_values) != _PLACEBO_REPLICATIONS:
                raise D1RenderingError("each placebo null distribution must contain 200 values")
            normalized_nulls: list[float | None] = []
            for index, value in enumerate(null_values):
                normalized_nulls.append(
                    None if value is None else _finite_float(value, label=f"placebo null[{index}]")
                )
            failures = sum(value is None for value in normalized_nulls)
            greater_or_equal = failures + sum(
                value >= observed_statistic for value in normalized_nulls if value is not None
            )
            observed_exceeds = (
                sum(observed_statistic > value for value in normalized_nulls if value is not None)
                / _PLACEBO_REPLICATIONS
            )
            failure_fraction = failures / _PLACEBO_REPLICATIONS
            if row.get("denominator") != _PLACEBO_DENOMINATOR:
                raise D1RenderingError("placebo p-value denominator must equal 201")
            if (
                _exact_int(
                    row.get("scientific_failure_count"),
                    label="placebo scientific_failure_count",
                )
                != failures
            ):
                raise D1RenderingError("placebo scientific failure count is inconsistent")
            if (
                _exact_int(
                    row.get("null_greater_or_equal_count"),
                    label="placebo null_greater_or_equal_count",
                )
                != greater_or_equal
            ):
                raise D1RenderingError("placebo greater-or-equal count is inconsistent")
            expected_status = "evidence_insufficient" if failure_fraction > 0.05 else "passed"
            if row.get("status") != expected_status:
                raise D1RenderingError("placebo status is inconsistent with failure fraction")
            expected_promising = expected_status == "passed" and observed_exceeds > 0.80
            if row.get("mechanism_promising_for_kind") is not expected_promising:
                raise D1RenderingError("placebo mechanism status is inconsistent")
            for key, expected_value in (
                ("scientific_failure_fraction", failure_fraction),
                ("monte_carlo_p_value", (1 + greater_or_equal) / _PLACEBO_DENOMINATOR),
                ("observed_exceeds_fraction", observed_exceeds),
            ):
                actual = _finite_float(row.get(key), label=f"placebo {key}")
                if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1.0e-12):
                    raise D1RenderingError(f"placebo {key} is inconsistent")
            promising_by_kind[kind][contrast] = expected_promising
    overall = _mapping(
        payload.get("anomaly_mechanism_promising_by_contrast"),
        label="placebo attribution decisions",
    )
    if set(overall) != set(_COMPONENT_CONTRASTS):
        raise D1RenderingError("placebo attribution decision axis changed")
    for contrast in _COMPONENT_CONTRASTS:
        expected = promising_by_kind["time"][contrast] and promising_by_kind["space"][contrast]
        if overall[contrast] is not expected:
            raise D1RenderingError("placebo combined attribution decision is inconsistent")
    return _SupplementalEvidence("completed", path, _sha256_file(path), payload)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise D1RenderingError(f"{label} changed its canonical field axis")


def _expect_float(
    value: object,
    expected: float,
    *,
    label: str,
    tolerance: float = 1.0e-12,
) -> None:
    actual = _finite_float(value, label=label)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise D1RenderingError(f"{label} is inconsistent with the observed replay")


def _validate_sign_counts(value: object, values: Sequence[int], *, label: str) -> None:
    counts = _mapping(value, label=label)
    _require_exact_keys(
        counts,
        {"positive_count", "zero_count", "negative_count"},
        label=label,
    )
    expected = {
        "positive_count": sum(item > 0 for item in values),
        "zero_count": sum(item == 0 for item in values),
        "negative_count": sum(item < 0 for item in values),
    }
    for key, count in expected.items():
        if _exact_int(counts.get(key), label=f"{label}.{key}") != count:
            raise D1RenderingError(f"{label} is inconsistent with its canonical rows")


def _validate_spatial_strata_identity(value: object) -> None:
    identity = _mapping(value, label="robustness spatial_strata_identity")
    _require_exact_keys(
        identity,
        {
            "public_manifest_content_sha256",
            "artifact_sha256",
            "operational_grid_id",
            "operational_cell_count",
            "nonempty_zone_count",
            "geometry_zone_count",
            "zero_cell_geometry_zone_count",
        },
        label="robustness spatial_strata_identity",
    )
    _sha256_identity(
        identity.get("public_manifest_content_sha256"),
        label="robustness public spatial manifest identity",
    )
    artifacts = _mapping(
        identity.get("artifact_sha256"),
        label="robustness spatial artifact identities",
    )
    _require_exact_keys(
        artifacts,
        set(_ROBUSTNESS_SPATIAL_ARTIFACTS),
        label="robustness spatial artifact identities",
    )
    for name in _ROBUSTNESS_SPATIAL_ARTIFACTS:
        _sha256_identity(artifacts.get(name), label=f"robustness spatial {name} identity")
    _string(identity.get("operational_grid_id"), label="robustness operational_grid_id")
    expected_counts = {
        "operational_cell_count": 15_697,
        "nonempty_zone_count": _ROBUSTNESS_ZONE_COUNT,
        "geometry_zone_count": 65,
        "zero_cell_geometry_zone_count": 26,
    }
    for key, expected in expected_counts.items():
        if _exact_int(identity.get(key), label=f"robustness {key}") != expected:
            raise D1RenderingError(f"robustness {key} changed its frozen identity")


def _validate_robustness_result(
    path: Path | None,
    *,
    observed: Mapping[str, Any],
    outcomes: Sequence[D1ClusterModelOutcome],
    overlays: Sequence[_OutcomeOverlay],
    expected_support: Mapping[int, Sequence[tuple[str, str, str]]],
) -> _SupplementalEvidence:
    if path is None:
        return _SupplementalEvidence("pending", None, None, None)
    if not path.is_file():
        raise D1RenderingError("robustness_result_path does not exist")
    payload = _read_json_mapping(path, label="robustness result")
    if payload.get("result_kind") != "d1_regional_and_leave_one_cluster_robustness":
        raise D1RenderingError("robustness result_kind is not canonical D1 robustness")
    if payload.get("status") != "completed" or payload.get("retrospective_only") is not True:
        raise D1RenderingError("robustness result must be completed and retrospective-only")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_version",
            "result_kind",
            "status",
            "retrospective_only",
            "identities",
            "regional_diagnostic_completed",
            "leave_one_cluster_out_completed",
            "model_refit_performed",
            "locked_test_read",
            "target_mapping_role",
            "primary_endpoint",
            "spatial_strata_identity",
            "zone_axis",
            "contrasts",
        },
        label="robustness result",
    )
    _validate_supplemental_identity(
        payload,
        observed,
        label="robustness result",
        placebo_input_binding=False,
    )
    if dict(_mapping(payload.get("identities"), label="robustness identities")) != dict(
        _observed_identities(observed)
    ):
        raise D1RenderingError("robustness identities do not exactly match observed replay")
    if payload.get("regional_diagnostic_completed") is not True:
        raise D1RenderingError("robustness result omitted regional diagnostic")
    if payload.get("leave_one_cluster_out_completed") is not True:
        raise D1RenderingError("robustness result omitted leave-one-cluster-out diagnostic")
    if payload.get("model_refit_performed") is not False:
        raise D1RenderingError("robustness diagnostic must not refit models")
    if payload.get("locked_test_read") is not False:
        raise D1RenderingError("robustness diagnostic must not read the locked test")
    if payload.get("target_mapping_role") != _ROBUSTNESS_TARGET_MAPPING_ROLE:
        raise D1RenderingError("robustness target mapping scientific role changed")

    endpoint = _mapping(payload.get("primary_endpoint"), label="robustness primary_endpoint")
    _require_exact_keys(
        endpoint,
        {
            "horizon_days",
            "alarm_area_km2",
            "cluster_count",
            "fold_cluster_counts",
            "construction_zone_count",
        },
        label="robustness primary_endpoint",
    )
    for key, expected in (
        ("horizon_days", D1_PRIMARY_HORIZON_DAYS),
        ("alarm_area_km2", int(D1_PRIMARY_AREA_KM2)),
        ("cluster_count", _ROBUSTNESS_CLUSTER_COUNT),
        ("construction_zone_count", _ROBUSTNESS_ZONE_COUNT),
    ):
        if _exact_int(endpoint.get(key), label=f"robustness primary {key}") != expected:
            raise D1RenderingError(f"robustness primary {key} changed")
    fold_counts = _mapping(
        endpoint.get("fold_cluster_counts"),
        label="robustness fold_cluster_counts",
    )
    if dict(fold_counts) != _EXPECTED_CLUSTER_COUNTS_BY_FOLD[D1_PRIMARY_HORIZON_DAYS]:
        raise D1RenderingError("robustness primary fold allocation changed")
    _validate_spatial_strata_identity(payload.get("spatial_strata_identity"))

    zone_axis_raw = _sequence(payload.get("zone_axis"), label="robustness zone_axis")
    zone_axis = tuple(
        _sha256_identity(value, label=f"robustness zone_axis[{index}]")
        for index, value in enumerate(zone_axis_raw)
    )
    if (
        len(zone_axis) != _ROBUSTNESS_ZONE_COUNT
        or len(set(zone_axis)) != _ROBUSTNESS_ZONE_COUNT
        or zone_axis != tuple(sorted(zone_axis))
    ):
        raise D1RenderingError("robustness zone_axis must be 39 sorted unique zone identities")

    support = tuple(expected_support[D1_PRIMARY_HORIZON_DAYS])
    if len(support) != _ROBUSTNESS_CLUSTER_COUNT:
        raise D1RenderingError("robustness requires the frozen 21-cluster support")
    cluster_order = tuple(item[0] for item in support)
    fold_issue_by_cluster = {cluster: (fold, issue) for cluster, fold, issue in support}
    outcome_lookup = {
        (item.cluster_id, item.model_id): item
        for item in outcomes
        if item.horizon_days == D1_PRIMARY_HORIZON_DAYS
    }
    expected_outcome_keys = {
        (cluster, model) for cluster in cluster_order for model in D1_MODEL_ORDER
    }
    if set(outcome_lookup) != expected_outcome_keys:
        raise D1RenderingError("robustness observed basis is not 21 clusters by six models")
    overlay_lookup = {
        (item.cluster_id, item.frame_key[4]): item
        for item in overlays
        if item.frame_key[3] == D1_PRIMARY_HORIZON_DAYS
    }
    if set(overlay_lookup) != expected_outcome_keys:
        raise D1RenderingError("robustness target mapping basis is incomplete")
    outside_by_cluster = {
        cluster: overlay_lookup[(cluster, D1_MODEL_ORDER[0])].outside_support
        for cluster in cluster_order
    }
    expected_inside_clusters = {
        cluster for cluster in cluster_order if not outside_by_cluster[cluster]
    }

    contrast_rows = _sequence(payload.get("contrasts"), label="robustness contrasts")
    if len(contrast_rows) != len(_COMPONENT_SPECS):
        raise D1RenderingError("robustness must contain the three preregistered contrasts")
    canonical_zone_membership: dict[str, str] | None = None
    contrast_keys = {
        "contrast_id",
        "candidate_model_id",
        "reference_model_id",
        "observed_candidate_hit_count",
        "observed_reference_hit_count",
        "observed_hit_gain_sum",
        "observed_recall_gain",
        "regional",
        "leave_one_cluster_out",
    }
    for contrast_index, (raw, spec) in enumerate(zip(contrast_rows, _COMPONENT_SPECS, strict=True)):
        row = _mapping(raw, label=f"robustness contrasts[{contrast_index}]")
        _require_exact_keys(row, contrast_keys, label=f"robustness contrast {contrast_index}")
        contrast_id, _label, candidate_model, reference_model = spec
        if (
            row.get("contrast_id") != contrast_id
            or row.get("candidate_model_id") != candidate_model
            or row.get("reference_model_id") != reference_model
        ):
            raise D1RenderingError("robustness contrast identity or order changed")
        candidate_hits = {
            cluster: int(outcome_lookup[(cluster, candidate_model)].hit_by_area[2])
            for cluster in cluster_order
        }
        reference_hits = {
            cluster: int(outcome_lookup[(cluster, reference_model)].hit_by_area[2])
            for cluster in cluster_order
        }
        gains = {
            cluster: candidate_hits[cluster] - reference_hits[cluster] for cluster in cluster_order
        }
        candidate_total = sum(candidate_hits.values())
        reference_total = sum(reference_hits.values())
        total_gain = candidate_total - reference_total
        for key, expected in (
            ("observed_candidate_hit_count", candidate_total),
            ("observed_reference_hit_count", reference_total),
            ("observed_hit_gain_sum", total_gain),
        ):
            if _signed_int(row.get(key), label=f"robustness {contrast_id}.{key}") != expected:
                raise D1RenderingError(f"robustness {contrast_id}.{key} is inconsistent")
        _expect_float(
            row.get("observed_recall_gain"),
            total_gain / _ROBUSTNESS_CLUSTER_COUNT,
            label=f"robustness {contrast_id}.observed_recall_gain",
        )

        regional = _mapping(row.get("regional"), label=f"robustness {contrast_id}.regional")
        _require_exact_keys(
            regional,
            {
                "zone_count",
                "target_bearing_zone_count",
                "outside_support_cluster_count",
                "target_bearing_zone_gain_sign_counts",
                "positive_zone_gain_sum",
                "largest_positive_zone",
                "direction_survives_largest_positive_zone_removal",
                "single_zone_direction_dominant",
                "additive_recall_gain_closure",
                "zone_rows",
            },
            label=f"robustness {contrast_id}.regional",
        )
        if _exact_int(regional.get("zone_count"), label="robustness zone_count") != len(zone_axis):
            raise D1RenderingError("robustness regional zone_count changed")
        zone_rows = _sequence(
            regional.get("zone_rows"),
            label=f"robustness {contrast_id}.zone_rows",
        )
        if len(zone_rows) != len(zone_axis):
            raise D1RenderingError("robustness regional table must contain all 39 zones")
        zone_membership: dict[str, str] = {}
        zone_gain_values: list[int] = []
        positive_rows: list[tuple[str, int, int]] = []
        target_bearing_count = 0
        positive_gain_sum = 0
        additive_closure = 0.0
        zone_row_keys = {
            "zone_id",
            "target_cluster_count",
            "candidate_hit_count",
            "reference_hit_count",
            "hit_gain_sum",
            "additive_recall_gain_on_full_21_cluster_denominator",
            "within_zone_recall_gain",
            "cluster_gain_sign_counts",
            "cluster_ids",
        }
        for zone_index, (raw_zone, zone_id) in enumerate(zip(zone_rows, zone_axis, strict=True)):
            zone = _mapping(raw_zone, label=f"robustness zone_rows[{zone_index}]")
            _require_exact_keys(zone, zone_row_keys, label=f"robustness zone_rows[{zone_index}]")
            if zone.get("zone_id") != zone_id:
                raise D1RenderingError("robustness regional zone order changed")
            cluster_values = tuple(
                _safe_id(value, label="robustness regional cluster_id")
                for value in _sequence(zone.get("cluster_ids"), label="robustness cluster_ids")
            )
            if len(cluster_values) != len(set(cluster_values)) or any(
                cluster not in fold_issue_by_cluster for cluster in cluster_values
            ):
                raise D1RenderingError("robustness regional cluster membership is invalid")
            for cluster in cluster_values:
                if cluster in zone_membership:
                    raise D1RenderingError("one target cluster appears in multiple zones")
                zone_membership[cluster] = zone_id
            zone_candidate = sum(candidate_hits[cluster] for cluster in cluster_values)
            zone_reference = sum(reference_hits[cluster] for cluster in cluster_values)
            zone_gain = zone_candidate - zone_reference
            for key, expected in (
                ("target_cluster_count", len(cluster_values)),
                ("candidate_hit_count", zone_candidate),
                ("reference_hit_count", zone_reference),
                ("hit_gain_sum", zone_gain),
            ):
                if _signed_int(zone.get(key), label=f"robustness zone {key}") != expected:
                    raise D1RenderingError("robustness regional row is numerically inconsistent")
            _expect_float(
                zone.get("additive_recall_gain_on_full_21_cluster_denominator"),
                zone_gain / _ROBUSTNESS_CLUSTER_COUNT,
                label="robustness additive regional contribution",
                tolerance=1.0e-15,
            )
            within = zone.get("within_zone_recall_gain")
            if not cluster_values:
                if within is not None:
                    raise D1RenderingError("empty robustness zone has a within-zone recall")
            else:
                _expect_float(
                    within,
                    zone_gain / len(cluster_values),
                    label="robustness within-zone recall gain",
                )
                target_bearing_count += 1
                zone_gain_values.append(zone_gain)
                if zone_gain > 0:
                    positive_rows.append((zone_id, len(cluster_values), zone_gain))
                    positive_gain_sum += zone_gain
            _validate_sign_counts(
                zone.get("cluster_gain_sign_counts"),
                [gains[cluster] for cluster in cluster_values],
                label="robustness zone cluster gain signs",
            )
            additive_closure += zone_gain / _ROBUSTNESS_CLUSTER_COUNT
        if set(zone_membership) != expected_inside_clusters:
            raise D1RenderingError(
                "robustness regional mapping does not match observed inside/outside support"
            )
        if canonical_zone_membership is None:
            canonical_zone_membership = dict(zone_membership)
        elif zone_membership != canonical_zone_membership:
            raise D1RenderingError("robustness contrasts use different regional memberships")
        for key, expected in (
            ("target_bearing_zone_count", target_bearing_count),
            (
                "outside_support_cluster_count",
                _ROBUSTNESS_CLUSTER_COUNT - len(expected_inside_clusters),
            ),
            ("positive_zone_gain_sum", positive_gain_sum),
        ):
            if _exact_int(regional.get(key), label=f"robustness regional {key}") != expected:
                raise D1RenderingError(f"robustness regional {key} is inconsistent")
        _validate_sign_counts(
            regional.get("target_bearing_zone_gain_sign_counts"),
            zone_gain_values,
            label="robustness target-bearing zone gain signs",
        )
        largest = (
            None if not positive_rows else min(positive_rows, key=lambda item: (-item[2], item[0]))
        )
        largest_payload = regional.get("largest_positive_zone")
        if largest is None:
            if largest_payload is not None:
                raise D1RenderingError("robustness reports a nonexistent positive zone")
            survives: bool | None = None
            single_zone_dominant = False
        else:
            largest_row = _mapping(
                largest_payload,
                label="robustness largest_positive_zone",
            )
            _require_exact_keys(
                largest_row,
                {
                    "zone_id",
                    "target_cluster_count",
                    "hit_gain_sum",
                    "fraction_of_all_positive_zone_gain",
                    "remaining_cluster_count",
                    "remaining_hit_gain_sum",
                    "remaining_recall_gain",
                },
                label="robustness largest_positive_zone",
            )
            zone_id, removed_count, removed_gain = largest
            remaining_count = _ROBUSTNESS_CLUSTER_COUNT - removed_count
            remaining_gain = total_gain - removed_gain
            if largest_row.get("zone_id") != zone_id:
                raise D1RenderingError("robustness largest positive zone identity is inconsistent")
            for key, expected in (
                ("target_cluster_count", removed_count),
                ("hit_gain_sum", removed_gain),
                ("remaining_cluster_count", remaining_count),
                ("remaining_hit_gain_sum", remaining_gain),
            ):
                if _signed_int(largest_row.get(key), label=f"robustness largest {key}") != expected:
                    raise D1RenderingError("robustness largest positive zone is inconsistent")
            _expect_float(
                largest_row.get("fraction_of_all_positive_zone_gain"),
                removed_gain / positive_gain_sum,
                label="robustness largest positive zone fraction",
            )
            _expect_float(
                largest_row.get("remaining_recall_gain"),
                remaining_gain / remaining_count,
                label="robustness largest-zone-removed recall gain",
            )
            survives = remaining_count > 0 and remaining_gain > 0
            single_zone_dominant = total_gain > 0 and not survives
        if regional.get("direction_survives_largest_positive_zone_removal") is not survives:
            raise D1RenderingError("robustness largest-zone direction flag is inconsistent")
        if regional.get("single_zone_direction_dominant") is not single_zone_dominant:
            raise D1RenderingError("robustness single-zone dominance flag is inconsistent")
        _expect_float(
            regional.get("additive_recall_gain_closure"),
            additive_closure,
            label="robustness regional additive closure",
            tolerance=1.0e-15,
        )
        if not math.isclose(
            additive_closure,
            total_gain / _ROBUSTNESS_CLUSTER_COUNT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise D1RenderingError("robustness regional contributions do not close")

        loco = _mapping(
            row.get("leave_one_cluster_out"),
            label=f"robustness {contrast_id}.leave_one_cluster_out",
        )
        _require_exact_keys(
            loco,
            {
                "replication_count",
                "remaining_cluster_count_per_replication",
                "recall_gain_minimum",
                "recall_gain_maximum",
                "recall_gain_sign_counts",
                "direction_survives_every_cluster_removal",
                "single_cluster_direction_dominant",
                "rows",
            },
            label=f"robustness {contrast_id}.leave_one_cluster_out",
        )
        if (
            _exact_int(loco.get("replication_count"), label="robustness LOCO replications")
            != _ROBUSTNESS_CLUSTER_COUNT
            or _exact_int(
                loco.get("remaining_cluster_count_per_replication"),
                label="robustness LOCO remaining cluster count",
            )
            != _ROBUSTNESS_CLUSTER_COUNT - 1
        ):
            raise D1RenderingError("robustness LOCO replication axis changed")
        loco_rows = _sequence(loco.get("rows"), label="robustness LOCO rows")
        if len(loco_rows) != _ROBUSTNESS_CLUSTER_COUNT:
            raise D1RenderingError("robustness LOCO must contain 21 rows")
        remaining_gains: list[int] = []
        loco_row_keys = {
            "omitted_cluster_id",
            "fold_id",
            "issue_id",
            "posthoc_zone_id",
            "omitted_cluster_hit_gain",
            "remaining_cluster_count",
            "remaining_candidate_hit_count",
            "remaining_reference_hit_count",
            "remaining_hit_gain_sum",
            "remaining_recall_gain",
        }
        for loco_index, (raw_loco, cluster) in enumerate(
            zip(loco_rows, cluster_order, strict=True)
        ):
            loco_row = _mapping(raw_loco, label=f"robustness LOCO rows[{loco_index}]")
            _require_exact_keys(
                loco_row,
                loco_row_keys,
                label=f"robustness LOCO rows[{loco_index}]",
            )
            fold, issue = fold_issue_by_cluster[cluster]
            expected_zone = cast(Mapping[str, str], canonical_zone_membership).get(cluster)
            if (
                loco_row.get("omitted_cluster_id") != cluster
                or loco_row.get("fold_id") != fold
                or loco_row.get("issue_id") != issue
                or loco_row.get("posthoc_zone_id") != expected_zone
            ):
                raise D1RenderingError("robustness LOCO row identity or zone is inconsistent")
            omitted_gain = gains[cluster]
            remaining_candidate = candidate_total - candidate_hits[cluster]
            remaining_reference = reference_total - reference_hits[cluster]
            remaining_gain = remaining_candidate - remaining_reference
            remaining_gains.append(remaining_gain)
            for key, expected in (
                ("omitted_cluster_hit_gain", omitted_gain),
                ("remaining_cluster_count", _ROBUSTNESS_CLUSTER_COUNT - 1),
                ("remaining_candidate_hit_count", remaining_candidate),
                ("remaining_reference_hit_count", remaining_reference),
                ("remaining_hit_gain_sum", remaining_gain),
            ):
                if _signed_int(loco_row.get(key), label=f"robustness LOCO {key}") != expected:
                    raise D1RenderingError("robustness LOCO row is numerically inconsistent")
            _expect_float(
                loco_row.get("remaining_recall_gain"),
                remaining_gain / (_ROBUSTNESS_CLUSTER_COUNT - 1),
                label="robustness LOCO remaining recall gain",
            )
        recall_gains = [value / (_ROBUSTNESS_CLUSTER_COUNT - 1) for value in remaining_gains]
        _expect_float(
            loco.get("recall_gain_minimum"),
            min(recall_gains),
            label="robustness LOCO recall minimum",
        )
        _expect_float(
            loco.get("recall_gain_maximum"),
            max(recall_gains),
            label="robustness LOCO recall maximum",
        )
        _validate_sign_counts(
            loco.get("recall_gain_sign_counts"),
            remaining_gains,
            label="robustness LOCO recall gain signs",
        )
        survives_all = all(value > 0 for value in remaining_gains)
        single_cluster_dominant = total_gain > 0 and min(remaining_gains) <= 0
        if loco.get("direction_survives_every_cluster_removal") is not survives_all:
            raise D1RenderingError("robustness LOCO direction flag is inconsistent")
        if loco.get("single_cluster_direction_dominant") is not single_cluster_dominant:
            raise D1RenderingError("robustness single-cluster dominance flag is inconsistent")
    return _SupplementalEvidence("completed", path, _sha256_file(path), payload)


def _validate_cell_binding(
    payload: Mapping[str, Any],
    *,
    observed_path: Path,
    scores_path: Path,
    expected_frame_count: int,
) -> None:
    binding = _mapping(payload.get("cell_scores"), label="cell_scores binding")
    relative = _string(binding.get("path"), label="cell_scores path")
    bound_path = Path(relative)
    if not bound_path.is_absolute():
        bound_path = observed_path.parent / bound_path
    if bound_path.resolve() != scores_path.resolve():
        raise D1RenderingError("observed result is bound to a different cell score parquet")
    expected_sha = _string(binding.get("file_sha256"), label="cell_scores file_sha256")
    if len(expected_sha) != 64 or expected_sha != _sha256_file(scores_path):
        raise D1RenderingError("cell score parquet hash differs from observed result")
    frame_count = _exact_int(binding.get("frame_count"), label="cell_scores frame_count")
    if frame_count != expected_frame_count:
        raise D1RenderingError("cell score binding has the wrong frame count")
    row_count = _exact_int(binding.get("row_count"), label="cell_scores row_count")
    try:
        actual_rows = pq.ParquetFile(scores_path).metadata.num_rows
    except Exception as exc:
        raise D1RenderingError("cell score parquet metadata cannot be read") from exc
    if row_count != actual_rows:
        raise D1RenderingError("cell score binding has the wrong row count")


def _parse_bool_vector(value: object, *, label: str) -> tuple[bool, ...]:
    values = _sequence(value, label=label)
    if len(values) != len(D1_AREA_BUDGETS_KM2) or any(type(item) is not bool for item in values):
        raise D1RenderingError(f"{label} must contain five booleans")
    result = tuple(cast(bool, item) for item in values)
    if any(left and not right for left, right in pairwise(result)):
        raise D1RenderingError(f"{label} must be monotone")
    return result


def _parse_float_vector(value: object, *, label: str) -> tuple[float, ...]:
    values = _sequence(value, label=label)
    if len(values) != len(D1_AREA_BUDGETS_KM2):
        raise D1RenderingError(f"{label} must contain five values")
    result = tuple(
        _finite_float(item, label=f"{label}[{index}]", minimum=0.0)
        for index, item in enumerate(values)
    )
    previous = -math.inf
    for actual, budget in zip(result, D1_AREA_BUDGETS_KM2, strict=True):
        if actual < previous or actual > budget + 1.0e-6:
            raise D1RenderingError(f"{label} must be monotone and not exceed budgets")
        previous = actual
    return result


def _parse_count_vector(value: object, *, label: str) -> tuple[int, ...]:
    values = _sequence(value, label=label)
    if len(values) != len(D1_AREA_BUDGETS_KM2):
        raise D1RenderingError(f"{label} must contain five counts")
    result = tuple(_exact_int(item, label=f"{label}[{index}]") for index, item in enumerate(values))
    if any(left > right for left, right in pairwise(result)):
        raise D1RenderingError(f"{label} must be monotone")
    return result


def _parse_outcomes(
    payload: Mapping[str, Any],
) -> tuple[tuple[D1ClusterModelOutcome, ...], tuple[_OutcomeOverlay, ...]]:
    raw_rows = _sequence(payload.get("outcomes"), label="outcomes")
    if not raw_rows:
        raise D1RenderingError("completed observed replay contains no real cluster outcomes")
    outcomes: list[D1ClusterModelOutcome] = []
    overlays: list[_OutcomeOverlay] = []
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, label=f"outcomes[{index}]")
        fold = _safe_id(row.get("fold_id"), label="outcome fold_id")
        issue = _safe_id(row.get("issue_id"), label="outcome issue_id")
        issue_time = _canonical_timestamp(row.get("issue_time_utc"), label="outcome issue_time_utc")
        horizon = _exact_int(row.get("horizon_days"), label="outcome horizon_days")
        model = _safe_id(row.get("model_id"), label="outcome model_id")
        cluster = _safe_id(row.get("cluster_id"), label="outcome cluster_id")
        outside = row.get("outside_support")
        if type(outside) is not bool:
            raise D1RenderingError("outcome outside_support must be boolean")
        hits = _parse_bool_vector(row.get("hit_by_area"), label="outcome hit_by_area")
        log_density_raw = row.get("log_density")
        log_density = (
            None
            if log_density_raw is None
            else _finite_float(log_density_raw, label="outcome log_density")
        )
        outcome = D1ClusterModelOutcome(
            cluster_id=cluster,
            fold_id=fold,
            issue_id=issue,
            horizon_days=horizon,
            model_id=model,
            log_density=log_density,
            outside_support=outside,
            hit_by_area=hits,
        )
        cell_raw = row.get("representative_cell_index")
        cell_index = (
            None if cell_raw is None else _exact_int(cell_raw, label="representative_cell_index")
        )
        if outcome.outside_support and cell_index is not None:
            raise D1RenderingError("outside-support outcome must not claim a grid cell")
        if not outcome.outside_support and cell_index is None:
            raise D1RenderingError("inside-support outcome is missing representative grid cell")
        outcomes.append(outcome)
        overlays.append(
            _OutcomeOverlay(
                cluster_id=cluster,
                frame_key=(fold, issue, issue_time, horizon, model),
                representative_cell_index=cell_index,
                outside_support=outcome.outside_support,
                hit_by_area=hits,
            )
        )
    _validate_cross_model_target_mapping(overlays)
    return tuple(outcomes), tuple(overlays)


def _validate_cross_model_target_mapping(overlays: Sequence[_OutcomeOverlay]) -> None:
    """Require one target mapping shared by all six independently ranked models."""

    grouped: dict[
        tuple[str, str, str, str, int],
        dict[str, tuple[int | None, bool]],
    ] = defaultdict(dict)
    for overlay in overlays:
        fold, issue, issue_time, horizon, model = overlay.frame_key
        target_key = (overlay.cluster_id, fold, issue, issue_time, horizon)
        if model in grouped[target_key]:
            raise D1RenderingError("one target has a duplicate model outcome")
        grouped[target_key][model] = (
            overlay.representative_cell_index,
            overlay.outside_support,
        )
    for models in grouped.values():
        if set(models) != set(D1_MODEL_ORDER):
            raise D1RenderingError("one target is not evaluated by all six models")
        if len(set(models.values())) != 1:
            raise D1RenderingError(
                "models disagree on representative_cell_index or outside_support"
            )


def _parse_expected_support(
    payload: Mapping[str, Any],
) -> dict[int, tuple[tuple[str, str, str], ...]]:
    raw = _mapping(
        payload.get("expected_support_by_horizon"),
        label="expected_support_by_horizon",
    )
    if set(raw) != {"30", "90"}:
        raise D1RenderingError("expected support must contain exactly 30 and 90")
    result: dict[int, tuple[tuple[str, str, str], ...]] = {}
    for horizon in D1_HORIZONS_DAYS:
        rows = _sequence(raw[str(horizon)], label=f"expected support {horizon}")
        values: list[tuple[str, str, str]] = []
        for index, item in enumerate(rows):
            row = _mapping(item, label=f"expected support {horizon}[{index}]")
            values.append(
                (
                    _safe_id(row.get("cluster_id"), label="expected cluster_id"),
                    _safe_id(row.get("fold_id"), label="expected fold_id"),
                    _safe_id(row.get("issue_id"), label="expected issue_id"),
                )
            )
        if not values or len(values) != len(set(values)):
            raise D1RenderingError("expected cluster support must be non-empty and unique")
        counts_by_fold = {fold: sum(item[1] == fold for item in values) for fold in D1_FOLD_IDS}
        if (
            len(values) != _EXPECTED_CLUSTER_COUNTS[horizon]
            or counts_by_fold != _EXPECTED_CLUSTER_COUNTS_BY_FOLD[horizon]
        ):
            raise D1RenderingError(
                f"expected {horizon}d cluster support must match the frozen "
                f"{_EXPECTED_CLUSTER_COUNTS[horizon]}-cluster axis"
            )
        result[horizon] = tuple(values)
    return result


def _parse_expected_issues(
    payload: Mapping[str, Any],
) -> dict[int, tuple[tuple[str, str], ...]]:
    raw = _mapping(
        payload.get("expected_issues_by_horizon"),
        label="expected_issues_by_horizon",
    )
    if set(raw) != {"30", "90"}:
        raise D1RenderingError("expected issues must contain exactly 30 and 90")
    result: dict[int, tuple[tuple[str, str], ...]] = {}
    for horizon in D1_HORIZONS_DAYS:
        rows = _sequence(raw[str(horizon)], label=f"expected issues {horizon}")
        values: list[tuple[str, str]] = []
        for index, item in enumerate(rows):
            row = _mapping(item, label=f"expected issues {horizon}[{index}]")
            values.append(
                (
                    _safe_id(row.get("fold_id"), label="expected fold_id"),
                    _safe_id(row.get("issue_id"), label="expected issue_id"),
                )
            )
        if len(values) != len(set(values)):
            raise D1RenderingError("expected issue support must be unique")
        counts_by_fold = {fold: sum(item[0] == fold for item in values) for fold in D1_FOLD_IDS}
        if (
            len(values) != _EXPECTED_ISSUE_COUNTS[horizon]
            or counts_by_fold != _EXPECTED_ISSUE_COUNTS_BY_FOLD[horizon]
        ):
            raise D1RenderingError(
                f"expected {horizon}d issues must match the frozen "
                f"{_EXPECTED_ISSUE_COUNTS[horizon]}-issue axis"
            )
        result[horizon] = tuple(values)
    return result


def _parse_issue_rows(
    payload: Mapping[str, Any],
    *,
    expected_issues: Mapping[int, Sequence[tuple[str, str]]],
) -> tuple[tuple[_IssueRow, ...], tuple[D1IssueAlarmOutcome, ...]]:
    raw_rows = _sequence(payload.get("issue_alarm_outcomes"), label="issue_alarm_outcomes")
    if not raw_rows:
        raise D1RenderingError("completed replay lacks independent issue alarm outcomes")
    issue_rows: list[_IssueRow] = []
    alarm_outcomes: list[D1IssueAlarmOutcome] = []
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, label=f"issue_alarm_outcomes[{index}]")
        fold = _safe_id(row.get("fold_id"), label="issue alarm fold_id")
        issue = _safe_id(row.get("issue_id"), label="issue alarm issue_id")
        issue_time = _canonical_timestamp(
            row.get("issue_time_utc"), label="issue alarm issue_time_utc"
        )
        horizon = _exact_int(row.get("horizon_days"), label="issue alarm horizon_days")
        model = _safe_id(row.get("model_id"), label="issue alarm model_id")
        counts = _parse_count_vector(
            row.get("alarm_prefix_counts"), label="issue alarm prefix counts"
        )
        areas = _parse_float_vector(row.get("actual_area_km2"), label="issue alarm actual areas")
        issue_rows.append(
            _IssueRow(
                fold_id=fold,
                issue_id=issue,
                issue_time_utc=issue_time,
                horizon_days=horizon,
                model_id=model,
                alarm_prefix_counts=counts,
                actual_area_km2=areas,
            )
        )
        alarm_outcomes.append(
            D1IssueAlarmOutcome(
                fold_id=fold,
                issue_id=issue,
                horizon_days=horizon,
                model_id=model,
                actual_area_km2=areas,
            )
        )
    derived_expected = _expected_issue_support(issue_rows)
    if any(
        set(derived_expected[horizon]) != set(expected_issues[horizon])
        for horizon in D1_HORIZONS_DAYS
    ):
        raise D1RenderingError("issue alarm table differs from frozen expected issue support")
    validate_complete_alarm_outcomes(
        alarm_outcomes,
        expected_issues_by_horizon=expected_issues,
    )
    if len(issue_rows) != sum(
        len(D1_FOLD_IDS) * D1_ASSESSMENT_ISSUES_PER_FOLD[horizon] * len(D1_MODEL_ORDER)
        for horizon in D1_HORIZONS_DAYS
    ):
        raise D1RenderingError("issue alarm table is not the complete frozen 198-frame axis")
    return tuple(issue_rows), tuple(alarm_outcomes)


def _expected_issue_support(
    rows: Sequence[_IssueRow],
) -> dict[int, tuple[tuple[str, str], ...]]:
    by_horizon: dict[int, set[tuple[str, str]]] = defaultdict(set)
    times: dict[tuple[str, str], str] = {}
    for row in rows:
        by_horizon[row.horizon_days].add((row.fold_id, row.issue_id))
        key = (row.fold_id, row.issue_id)
        previous = times.setdefault(key, row.issue_time_utc)
        if previous != row.issue_time_utc:
            raise D1RenderingError("one issue identity has inconsistent timestamps")
    return {horizon: tuple(sorted(by_horizon[horizon])) for horizon in D1_HORIZONS_DAYS}


def _validate_training_summary(payload: Mapping[str, Any]) -> None:
    folds = _sequence(payload.get("folds"), label="folds")
    if len(folds) != len(D1_FOLD_IDS):
        raise D1RenderingError("completed replay must report all three training folds")
    found_folds: set[str] = set()
    for index, raw in enumerate(folds):
        fold = _mapping(raw, label=f"folds[{index}]")
        fold_id = _safe_id(fold.get("fold_id"), label="training fold_id")
        found_folds.add(fold_id)
        models = _sequence(fold.get("models"), label="fold models")
        model_rows = tuple(_mapping(item, label="fold model") for item in models)
        model_ids = {_safe_id(item.get("model_id"), label="model_id") for item in model_rows}
        if len(model_rows) != len(D1_MODEL_ORDER) or model_ids != set(D1_MODEL_ORDER):
            raise D1RenderingError("each fold must report all six trained/scored models")
        alphas = {
            _finite_float(
                item.get("selected_alpha"),
                label="selected_alpha",
                minimum=0.0,
            )
            for item in model_rows
        }
        if len(alphas) != 1:
            raise D1RenderingError("one fold must reuse one background-selected alpha")
        if not alphas <= {0.0, 0.25, 0.5, 0.75}:
            raise D1RenderingError("selected alpha is outside the frozen candidates")
        for model in model_rows:
            model_id = _safe_id(model.get("model_id"), label="model_id")
            if _safe_id(model.get("fold_id"), label="model fold_id") != fold_id:
                raise D1RenderingError("fold model identity differs from its parent fold")
            if model.get("base") != _MODEL_BASE[model_id]:
                raise D1RenderingError("fold model base differs from the frozen registry")
            groups = tuple(
                _safe_id(item, label="feature group")
                for item in _sequence(model.get("feature_groups"), label="feature_groups")
            )
            if groups != _MODEL_FEATURE_GROUPS[model_id]:
                raise D1RenderingError("fold model features differ from the frozen registry")
            ridge = model.get("selected_ridge")
            if groups:
                ridge_value = _finite_float(
                    ridge,
                    label="selected_ridge",
                    minimum=0.0,
                )
                if ridge_value not in {0.1, 1.0, 10.0}:
                    raise D1RenderingError("selected ridge is outside the frozen candidates")
            elif ridge is not None:
                raise D1RenderingError("background-only model may not carry a ridge penalty")
            diagnostic = _mapping(
                model.get("training_diagnostic"),
                label="training_diagnostic",
            )
            _exact_int(
                diagnostic.get("fit_issue_count"),
                label="fit_issue_count",
                minimum=1,
            )
            _exact_int(
                diagnostic.get("fit_catalog_m4plus_event_count"),
                label="fit_catalog_m4plus_event_count",
                minimum=1,
            )
            _exact_int(
                diagnostic.get("iteration_count"),
                label="iteration_count",
            )
            objective = diagnostic.get("objective")
            if groups and objective is None:
                raise D1RenderingError("trained feature model is missing its objective")
            if objective is not None:
                _finite_float(objective, label="training objective")
    if found_folds != set(D1_FOLD_IDS):
        raise D1RenderingError("training folds do not match the frozen fold identities")


def select_best_intermediate_model(metrics: Sequence[D1Metric]) -> str:
    """Apply the registered pooled-recall, equal-recall-area, fixed-order rule."""

    primary: dict[str, D1Metric] = {}
    for model in INTERMEDIATE_MODEL_ORDER:
        matches = [
            item
            for item in metrics
            if item.model_id == model
            and item.horizon_days == D1_PRIMARY_HORIZON_DAYS
            and item.fold_id is None
            and item.area_budget_km2 == D1_PRIMARY_AREA_KM2
        ]
        if len(matches) != 1 or matches[0].recall is None:
            raise D1RenderingError("primary metrics are incomplete for intermediate selection")
        primary[model] = matches[0]
    best_recall = max(cast(float, item.recall) for item in primary.values())

    def selection_key(model: str) -> tuple[float, int]:
        efficiency = minimum_area_reaching_recall(
            metrics,
            model_id=model,
            horizon_days=D1_PRIMARY_HORIZON_DAYS,
            target_recall=best_recall,
        )
        return (
            math.inf if efficiency is None else efficiency,
            INTERMEDIATE_MODEL_ORDER.index(model),
        )

    tied = [
        model
        for model in INTERMEDIATE_MODEL_ORDER
        if cast(float, primary[model].recall) == best_recall
    ]
    return min(tied, key=selection_key)


def _u16_b64(values: Sequence[int]) -> str:
    if any(value < 0 or value > 65_535 for value in values):
        raise D1RenderingError("offline payload exceeds uint16 range")
    packed = array("H", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _quantize_strength(values: Sequence[float]) -> tuple[str, float, float]:
    logged = np.log1p(np.asarray(values, dtype=np.float64))
    if logged.ndim != 1 or logged.size == 0 or not bool(np.isfinite(logged).all()):
        raise D1RenderingError("relative-strength frame is empty or nonfinite")
    low = float(logged.min())
    high = float(logged.max())
    if high == low:
        quantized = np.zeros(logged.size, dtype=np.uint16)
    else:
        quantized = np.rint((logged - low) * (65_535.0 / (high - low))).astype(np.uint16)
    values_u16 = [int(value) for value in quantized]
    return _u16_b64(values_u16), low, high


def _frame_columns() -> tuple[str, ...]:
    return (
        "fold_id",
        "issue_id",
        "issue_time_utc",
        "horizon_days",
        "model_id",
        "cell_index",
        "cell_id",
        "row",
        "column",
        "query_x_m",
        "query_y_m",
        "clipped_area_km2",
        "relative_cell_mass",
        "relative_strength_per_km2",
        "rank",
        *(f"alarm_{suffix}" for suffix in _AREA_COLUMN_SUFFIXES),
    )


def _build_frames(
    cell_scores_path: Path,
    *,
    issue_rows: Sequence[_IssueRow],
    overlays: Sequence[_OutcomeOverlay],
) -> _FrameBuildResult:
    try:
        parquet = pq.ParquetFile(cell_scores_path)
    except Exception as exc:
        raise D1RenderingError("cell score parquet cannot be opened") from exc
    required = set(_frame_columns())
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise D1RenderingError(f"cell score parquet missing columns: {sorted(missing)}")

    issue_by_key = {row.frame_key: row for row in issue_rows}
    if len(issue_by_key) != len(issue_rows):
        raise D1RenderingError("issue alarm frames are duplicated")
    overlay_by_key: dict[tuple[str, str, str, int, str], list[_OutcomeOverlay]] = defaultdict(list)
    for overlay in overlays:
        overlay_by_key[overlay.frame_key].append(overlay)
    if any(key not in issue_by_key for key in overlay_by_key):
        raise D1RenderingError("cluster outcome refers to a frame outside the issue table")

    grid: tuple[_GridCell, ...] | None = None
    grid_lookup: dict[int, int] = {}
    frames: list[_EncodedFrame] = []
    seen_keys: set[tuple[str, str, str, int, str]] = set()
    current_key: tuple[str, str, str, int, str] | None = None
    current_cells: list[_GridCell] = []
    current_indices: list[int] = []
    current_strengths: list[float] = []
    current_mass_sum = 0.0
    current_alarm_counts = [0] * len(D1_AREA_BUDGETS_KM2)
    current_alarm_areas = [0.0] * len(D1_AREA_BUDGETS_KM2)

    def finish_frame() -> None:
        nonlocal grid, grid_lookup
        if current_key is None:
            return
        if current_key in seen_keys:
            raise D1RenderingError("cell parquet frame is not contiguous")
        seen_keys.add(current_key)
        expected = issue_by_key.get(current_key)
        if expected is None:
            raise D1RenderingError("cell parquet contains a frame outside the issue table")
        if not current_cells or len(current_cells) > 65_535:
            raise D1RenderingError("cell frame count is empty or exceeds compact payload limit")
        if not math.isclose(current_mass_sum, 1.0, rel_tol=1.0e-8, abs_tol=1.0e-10):
            raise D1RenderingError("relative cell mass does not sum to one")
        for index in range(len(current_cells) - 1):
            left_strength = current_strengths[index]
            right_strength = current_strengths[index + 1]
            if left_strength < right_strength:
                raise D1RenderingError("cell rank is not descending by mass per area")
            if left_strength == right_strength:
                left = current_cells[index]
                right = current_cells[index + 1]
                if (left.row, left.column, left.cell_id) > (
                    right.row,
                    right.column,
                    right.cell_id,
                ):
                    raise D1RenderingError("cell rank violates the frozen spatial tie rule")
        if tuple(current_alarm_counts) != expected.alarm_prefix_counts:
            raise D1RenderingError("parquet alarm prefix counts differ from issue outcomes")
        for actual, registered in zip(current_alarm_areas, expected.actual_area_km2, strict=True):
            if not math.isclose(actual, registered, rel_tol=0.0, abs_tol=1.0e-5):
                raise D1RenderingError("parquet alarm areas differ from issue outcomes")

        if grid is None:
            ordered_grid = tuple(sorted(current_cells, key=lambda cell: cell.cell_index))
            if len({cell.cell_index for cell in ordered_grid}) != len(ordered_grid):
                raise D1RenderingError("first cell frame contains duplicate cell indices")
            grid = ordered_grid
            grid_lookup = {cell.cell_index: index for index, cell in enumerate(grid)}
        else:
            if len(current_cells) != len(grid):
                raise D1RenderingError("cell count changed between frames")
            for cell in current_cells:
                position = grid_lookup.get(cell.cell_index)
                if position is None or cell != grid[position]:
                    raise D1RenderingError("frozen grid identity changed between frames")
        positions = [grid_lookup[index] for index in current_indices]
        if len(set(positions)) != len(grid_lookup):
            raise D1RenderingError("cell frame does not contain the full frozen grid")
        rank_by_cell_index = {
            cell_index: rank for rank, cell_index in enumerate(current_indices, start=1)
        }
        strength_b64, strength_low, strength_high = _quantize_strength(current_strengths)
        target_markers: list[_TargetMarker] = []
        for overlay in sorted(
            overlay_by_key.get(current_key, ()), key=lambda item: item.cluster_id
        ):
            position = (
                None
                if overlay.representative_cell_index is None
                else grid_lookup.get(overlay.representative_cell_index)
            )
            if not overlay.outside_support and position is None:
                raise D1RenderingError("target overlay references a cell outside frozen grid")
            if overlay.outside_support:
                recomputed_hits = (False,) * len(D1_AREA_BUDGETS_KM2)
            else:
                representative = cast(int, overlay.representative_cell_index)
                representative_rank = rank_by_cell_index.get(representative)
                if representative_rank is None:
                    raise D1RenderingError("target representative cell is absent from its frame")
                recomputed_hits = tuple(
                    representative_rank <= prefix for prefix in current_alarm_counts
                )
            if recomputed_hits != overlay.hit_by_area:
                raise D1RenderingError(
                    "outcome hit_by_area differs from the ranked parquet alarm prefixes"
                )
            target_markers.append(
                _TargetMarker(
                    cluster_id=overlay.cluster_id,
                    cell_position=position,
                    outside_support=overlay.outside_support,
                    hit_by_area=overlay.hit_by_area,
                )
            )
        frames.append(
            _EncodedFrame(
                fold_id=expected.fold_id,
                issue_id=expected.issue_id,
                issue_time_utc=expected.issue_time_utc,
                horizon_days=expected.horizon_days,
                model_id=expected.model_id,
                order_u16_b64=_u16_b64(positions),
                strength_u16_b64=strength_b64,
                strength_log_min=strength_low,
                strength_log_max=strength_high,
                alarm_prefix_counts=expected.alarm_prefix_counts,
                actual_area_km2=expected.actual_area_km2,
                targets=tuple(target_markers),
            )
        )

    try:
        for batch in parquet.iter_batches(batch_size=65_536, columns=list(_frame_columns())):
            columns = cast(dict[str, list[Any]], batch.to_pydict())
            for offset in range(batch.num_rows):
                key = (
                    _safe_id(columns["fold_id"][offset], label="parquet fold_id"),
                    _safe_id(columns["issue_id"][offset], label="parquet issue_id"),
                    _canonical_timestamp(
                        columns["issue_time_utc"][offset], label="parquet issue_time_utc"
                    ),
                    _exact_int(columns["horizon_days"][offset], label="parquet horizon_days"),
                    _safe_id(columns["model_id"][offset], label="parquet model_id"),
                )
                if current_key is not None and key != current_key:
                    finish_frame()
                    current_cells.clear()
                    current_indices.clear()
                    current_strengths.clear()
                    current_mass_sum = 0.0
                    current_alarm_counts[:] = [0] * len(D1_AREA_BUDGETS_KM2)
                    current_alarm_areas[:] = [0.0] * len(D1_AREA_BUDGETS_KM2)
                current_key = key
                rank = _exact_int(columns["rank"][offset], label="rank", minimum=1)
                if rank != len(current_cells) + 1:
                    raise D1RenderingError("each cell frame rank must be contiguous from one")
                cell = _GridCell(
                    cell_index=_exact_int(columns["cell_index"][offset], label="cell_index"),
                    cell_id=_safe_id(columns["cell_id"][offset], label="cell_id"),
                    row=_signed_int(columns["row"][offset], label="row"),
                    column=_signed_int(columns["column"][offset], label="column"),
                    x_m=_finite_float(columns["query_x_m"][offset], label="query_x_m"),
                    y_m=_finite_float(columns["query_y_m"][offset], label="query_y_m"),
                    area_km2=_finite_float(
                        columns["clipped_area_km2"][offset],
                        label="clipped_area_km2",
                        minimum=1.0e-12,
                    ),
                )
                mass = _finite_float(
                    columns["relative_cell_mass"][offset],
                    label="relative_cell_mass",
                    minimum=0.0,
                )
                strength = _finite_float(
                    columns["relative_strength_per_km2"][offset],
                    label="relative_strength_per_km2",
                    minimum=0.0,
                )
                expected_strength = mass / cell.area_km2
                if not math.isclose(
                    strength,
                    expected_strength,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                ):
                    raise D1RenderingError("relative strength is not cell mass divided by area")
                current_cells.append(cell)
                current_indices.append(cell.cell_index)
                current_strengths.append(strength)
                current_mass_sum += mass
                for area_index, suffix in enumerate(_AREA_COLUMN_SUFFIXES):
                    selected = columns[f"alarm_{suffix}"][offset]
                    if type(selected) is not bool:
                        raise D1RenderingError("alarm columns must be boolean")
                    registered = issue_by_key.get(key)
                    if registered is None:
                        raise D1RenderingError("cell parquet frame is absent from issue outcomes")
                    if selected != (rank <= registered.alarm_prefix_counts[area_index]):
                        raise D1RenderingError("alarm mask is not the registered complete prefix")
                    if selected:
                        current_alarm_counts[area_index] += 1
                        current_alarm_areas[area_index] += cell.area_km2
        finish_frame()
    except D1RenderingError:
        raise
    except Exception as exc:
        raise D1RenderingError("cell score parquet failed deterministic validation") from exc

    if grid is None or set(seen_keys) != set(issue_by_key):
        raise D1RenderingError("cell score parquet does not cover all 198 issue/model frames")
    study_area = math.fsum(cell.area_km2 for cell in grid)
    if not math.isfinite(study_area) or study_area <= 0.0:
        raise D1RenderingError("frozen grid study area is invalid")
    return _FrameBuildResult(grid=grid, frames=tuple(frames), study_area_km2=study_area)


def _metric_lookup(
    metrics: Sequence[D1Metric],
) -> dict[tuple[str, int, str | None, float], D1Metric]:
    return {
        (item.model_id, item.horizon_days, item.fold_id, item.area_budget_km2): item
        for item in metrics
    }


def _exposure_lookup(
    metrics: Sequence[D1AlarmExposureMetric],
) -> dict[tuple[str, int, str | None, float], D1AlarmExposureMetric]:
    return {
        (item.model_id, item.horizon_days, item.fold_id, item.area_budget_km2): item
        for item in metrics
    }


def _build_component_evidence(
    metrics: Sequence[D1Metric],
    placebo: _SupplementalEvidence,
) -> tuple[_ComponentEvidence, ...]:
    differences = _expected_component_differences(metrics)
    kinds: Mapping[str, Any] | None = None
    if placebo.payload is not None:
        kinds = _mapping(placebo.payload.get("kinds"), label="placebo kinds")
    rows: list[_ComponentEvidence] = []
    for contrast, label, candidate, reference in _COMPONENT_SPECS:
        time_p: float | None = None
        space_p: float | None = None
        time_status = "pending"
        space_status = "pending"
        if kinds is not None:
            time_row = _mapping(
                _mapping(kinds["time"], label="time placebo")["contrasts"][contrast],
                label=f"time {contrast}",
            )
            space_row = _mapping(
                _mapping(kinds["space"], label="space placebo")["contrasts"][contrast],
                label=f"space {contrast}",
            )
            time_p = _finite_float(
                time_row.get("monte_carlo_p_value"), label="time placebo p-value"
            )
            space_p = _finite_float(
                space_row.get("monte_carlo_p_value"), label="space placebo p-value"
            )
            time_status = _string(time_row.get("status"), label="time placebo status")
            space_status = _string(space_row.get("status"), label="space placebo status")
        if kinds is None:
            attribution = "pending"
        elif time_status == "evidence_insufficient" or space_status == "evidence_insufficient":
            attribution = "evidence_insufficient"
        elif differences[contrast] <= 0.0:
            attribution = "no_positive_observed_gain"
        else:
            top_level = _mapping(
                cast(Mapping[str, Any], placebo.payload).get(
                    "anomaly_mechanism_promising_by_contrast"
                ),
                label="placebo attribution decisions",
            )
            attribution = (
                "supported_against_time_and_space_placebos"
                if top_level[contrast] is True
                else "not_supported_against_both_placebos"
            )
        rows.append(
            _ComponentEvidence(
                contrast=contrast,
                label=label,
                candidate_model=candidate,
                reference_model=reference,
                observed_recall_difference=differences[contrast],
                time_p_value=time_p,
                space_p_value=space_p,
                time_status=time_status,
                space_status=space_status,
                attribution_status=attribution,
            )
        )
    return tuple(rows)


def _robustness_numeric_summary(
    evidence: _SupplementalEvidence,
) -> dict[str, object] | None:
    """Return the compact, already-validated numerical robustness evidence."""

    if evidence.status != "completed" or evidence.payload is None:
        return None
    payload = evidence.payload
    endpoint = _mapping(payload.get("primary_endpoint"), label="robustness primary_endpoint")
    spatial = _mapping(
        payload.get("spatial_strata_identity"),
        label="robustness spatial_strata_identity",
    )
    contrast_summaries: list[dict[str, object]] = []
    for raw in _sequence(payload.get("contrasts"), label="robustness contrasts"):
        row = _mapping(raw, label="robustness contrast")
        regional = _mapping(row.get("regional"), label="robustness regional")
        loco = _mapping(
            row.get("leave_one_cluster_out"),
            label="robustness leave_one_cluster_out",
        )
        largest = regional.get("largest_positive_zone")
        contrast_summaries.append(
            {
                "contrast_id": row["contrast_id"],
                "candidate_model_id": row["candidate_model_id"],
                "reference_model_id": row["reference_model_id"],
                "observed_candidate_hit_count": row["observed_candidate_hit_count"],
                "observed_reference_hit_count": row["observed_reference_hit_count"],
                "observed_hit_gain_sum": row["observed_hit_gain_sum"],
                "observed_recall_gain": row["observed_recall_gain"],
                "regional": {
                    "zone_count": regional["zone_count"],
                    "target_bearing_zone_count": regional["target_bearing_zone_count"],
                    "outside_support_cluster_count": regional["outside_support_cluster_count"],
                    "target_bearing_zone_gain_sign_counts": dict(
                        _mapping(
                            regional["target_bearing_zone_gain_sign_counts"],
                            label="robustness regional gain signs",
                        )
                    ),
                    "positive_zone_gain_sum": regional["positive_zone_gain_sum"],
                    "largest_positive_zone": (
                        None
                        if largest is None
                        else dict(
                            _mapping(
                                largest,
                                label="robustness largest_positive_zone",
                            )
                        )
                    ),
                    "direction_survives_largest_positive_zone_removal": regional[
                        "direction_survives_largest_positive_zone_removal"
                    ],
                    "single_zone_direction_dominant": regional["single_zone_direction_dominant"],
                    "additive_recall_gain_closure": regional["additive_recall_gain_closure"],
                },
                "leave_one_cluster_out": {
                    "replication_count": loco["replication_count"],
                    "remaining_cluster_count_per_replication": loco[
                        "remaining_cluster_count_per_replication"
                    ],
                    "recall_gain_minimum": loco["recall_gain_minimum"],
                    "recall_gain_maximum": loco["recall_gain_maximum"],
                    "recall_gain_sign_counts": dict(
                        _mapping(
                            loco["recall_gain_sign_counts"],
                            label="robustness LOCO gain signs",
                        )
                    ),
                    "direction_survives_every_cluster_removal": loco[
                        "direction_survives_every_cluster_removal"
                    ],
                    "single_cluster_direction_dominant": loco["single_cluster_direction_dominant"],
                },
            }
        )
    return {
        "primary_endpoint": dict(endpoint),
        "spatial_identity": {
            "public_manifest_content_sha256": spatial["public_manifest_content_sha256"],
            "operational_grid_id": spatial["operational_grid_id"],
            "operational_cell_count": spatial["operational_cell_count"],
            "nonempty_zone_count": spatial["nonempty_zone_count"],
        },
        "contrast_count": len(contrast_summaries),
        "contrasts": contrast_summaries,
    }


def _component_status_zh(value: str) -> str:
    return {
        "pending": "待完成",
        "evidence_insufficient": "置乱失败过多，证据不足",
        "no_positive_observed_gain": "观测增量不为正",
        "supported_against_time_and_space_placebos": "同时超过时间与空间置乱",
        "not_supported_against_both_placebos": "未同时超过时间与空间置乱",
    }[value]


def _format_p_value(value: float | None) -> str:
    return "待完成" if value is None else f"{value:.4f}"


def _svg_polyline(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _render_effects_svg(
    metrics: Sequence[D1Metric],
    exposure: Sequence[D1AlarmExposureMetric],
    bootstrap: Sequence[D1BootstrapEffect],
    components: Sequence[_ComponentEvidence],
    *,
    placebo_complete: bool,
) -> str:
    width, height = 1280, 1165
    lookup = _metric_lookup(metrics)
    exposure_lookup = _exposure_lookup(exposure)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">D1真实历史开发回放效果</title>',
        f'<desc id="desc">{xml_escape(RETROSPECTIVE_LABEL)}；六模型召回、Molchan与Bootstrap效果。</desc>',
        "<style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif;fill:#18212b}"
        ".axis{stroke:#4d5966;stroke-width:1}.grid{stroke:#d8dee5;stroke-width:1}"
        ".warn{fill:#9c2c22;font-weight:700}.small{font-size:12px}.label{font-size:14px}"
        ".title{font-size:25px;font-weight:700}.panel{fill:#fff;stroke:#bac4cf}</style>",
        '<rect width="100%" height="100%" fill="#f5f7fa"/>',
        '<text x="40" y="42" class="title">D1 模型历史时间外推效果</text>',
        f'<text x="40" y="70" class="warn">{xml_escape(RETROSPECTIVE_LABEL)}</text>',
        f'<text x="40" y="92" class="label">{xml_escape(STRENGTH_LABEL)}</text>',
    ]
    panel_origins = ((45, 135), (655, 135), (45, 455), (655, 455))
    panel_titles = (
        "30天：独立震群召回—报警面积",
        "90天：独立震群召回—报警面积",
        "30天：Molchan（漏报率—平均报警面积占比）",
        "90天：Molchan（漏报率—平均报警面积占比）",
    )
    chart_w, chart_h = 555, 260
    plot_left, plot_top, plot_w, plot_h = 62, 42, 460, 172
    for panel_index, ((origin_x, origin_y), title) in enumerate(
        zip(panel_origins, panel_titles, strict=True)
    ):
        horizon = 30 if panel_index % 2 == 0 else 90
        molchan = panel_index >= 2
        parts.extend(
            [
                f'<rect class="panel" x="{origin_x}" y="{origin_y}" width="{chart_w}" height="{chart_h}" rx="8"/>',
                f'<text x="{origin_x + 18}" y="{origin_y + 27}" class="label" font-weight="700">{xml_escape(title)}</text>',
            ]
        )
        x0 = origin_x + plot_left
        y0 = origin_y + plot_top
        for tick in range(6):
            y = y0 + plot_h - tick * plot_h / 5
            parts.append(
                f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}"/>'
            )
            parts.append(
                f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{tick / 5:.1f}</text>'
            )
        parts.extend(
            [
                f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}"/>',
                f'<line class="axis" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>',
            ]
        )
        if molchan:
            max_fraction = max(
                exposure_lookup[(model, horizon, None, area)].mean_alarm_fraction
                for model in D1_MODEL_ORDER
                for area in D1_AREA_BUDGETS_KM2
            )
            max_x = max(0.01, max_fraction * 1.05)
            for tick in range(6):
                x = x0 + tick * plot_w / 5
                parts.append(
                    f'<text x="{x:.2f}" y="{y0 + plot_h + 18}" text-anchor="middle" class="small">{tick * max_x / 5:.2f}</text>'
                )
            x_label = "完整起报日平均报警面积 / 研究区面积"
            y_label = "漏报率"
        else:
            min_area = D1_AREA_BUDGETS_KM2[0]
            max_area = D1_AREA_BUDGETS_KM2[-1]
            max_x = max_area
            for area in D1_AREA_BUDGETS_KM2:
                x = x0 + (area - min_area) / (max_area - min_area) * plot_w
                parts.append(
                    f'<text x="{x:.2f}" y="{y0 + plot_h + 18}" text-anchor="middle" class="small">{area / 1000:.0f}</text>'
                )
            x_label = "报警面积上限（千 km²）"
            y_label = "召回率"
        parts.append(
            f'<text x="{x0 + plot_w / 2:.2f}" y="{origin_y + 250}" text-anchor="middle" class="small">{xml_escape(x_label)}</text>'
        )
        parts.append(
            f'<text transform="translate({origin_x + 16},{y0 + plot_h / 2}) rotate(-90)" text-anchor="middle" class="small">{xml_escape(y_label)}</text>'
        )
        for model in D1_MODEL_ORDER:
            points: list[tuple[float, float]] = []
            for area in D1_AREA_BUDGETS_KM2:
                metric = lookup[(model, horizon, None, area)]
                if metric.recall is None:
                    raise D1RenderingError("pooled recall is missing")
                if molchan:
                    x_value = exposure_lookup[(model, horizon, None, area)].mean_alarm_fraction
                    y_value = 1.0 - metric.recall
                    x = x0 + x_value / max_x * plot_w
                else:
                    x = x0 + (area - min_area) / (max_x - min_area) * plot_w
                    y_value = metric.recall
                y = y0 + (1.0 - y_value) * plot_h
                points.append((x, y))
            color = _MODEL_COLORS[model]
            parts.append(
                f'<polyline points="{_svg_polyline(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            )
            parts.extend(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x, y in points
            )

    legend_y = 755
    for index, model in enumerate(D1_MODEL_ORDER):
        x = 50 + (index % 3) * 400
        y = legend_y + (index // 3) * 25
        parts.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 25}" y2="{y}" stroke="{_MODEL_COLORS[model]}" stroke-width="4"/>',
                f'<text x="{x + 32}" y="{y + 4}" class="small">{xml_escape(model)}：{xml_escape(_MODEL_ZH[model])}</text>',
            ]
        )

    effect_lookup = {
        item.model_id: item
        for item in bootstrap
        if item.horizon_days == D1_PRIMARY_HORIZON_DAYS
        and item.area_budget_km2 == D1_PRIMARY_AREA_KM2
    }
    parts.append('<rect class="panel" x="45" y="820" width="1190" height="120" rx="8"/>')
    parts.append(
        '<text x="65" y="847" class="label" font-weight="700">30天、60万 km²：相对 B0 的原始召回效果（2000次配对震群Bootstrap）</text>'
    )
    parts.append('<line x1="65" y1="900" x2="1195" y2="900" stroke="#5f6873"/>')
    for index, model in enumerate(D1_MODEL_ORDER):
        x = 90 + index * 205
        if model == "B0":
            gain = lower = upper = 0.0
        else:
            effect = effect_lookup.get(model)
            if effect is None or effect.replication_count != 2000:
                raise D1RenderingError("registered 2000-replication bootstrap is incomplete")
            gain, lower, upper = (
                effect.observed_recall_gain,
                effect.lower_95,
                effect.upper_95,
            )
        scale = 120.0
        y_gain = 900 - gain * scale
        y_low = 900 - lower * scale
        y_high = 900 - upper * scale
        parts.extend(
            [
                f'<line x1="{x}" y1="{y_low:.2f}" x2="{x}" y2="{y_high:.2f}" stroke="{_MODEL_COLORS[model]}" stroke-width="3"/>',
                f'<circle cx="{x}" cy="{y_gain:.2f}" r="5" fill="{_MODEL_COLORS[model]}"/>',
                f'<text x="{x}" y="928" text-anchor="middle" class="small">{xml_escape(model)}</text>',
            ]
        )
        fold_gains = []
        for fold in D1_FOLD_IDS:
            model_metric = lookup[(model, 30, fold, 600_000.0)]
            baseline_metric = lookup[("B0", 30, fold, 600_000.0)]
            fold_gains.append(model_metric.hit_count - baseline_metric.hit_count)
        fold_text = "/".join(f"{gain:+d}" for gain in fold_gains)
        parts.append(
            f'<text x="{x}" y="947" text-anchor="middle" class="small">折增益 {fold_text}</text>'
        )
    parts.append('<rect class="panel" x="45" y="965" width="1190" height="150" rx="8"/>')
    parts.append(
        '<text x="65" y="990" class="label" font-weight="700">异常组件贡献（30天、60万 km²；观测召回差与置乱归因）</text>'
    )
    headers = ("组件", "模型差", "观测差", "时间置乱 p", "空间置乱 p", "归因状态")
    header_x = (65, 240, 570, 690, 820, 960)
    for x, title in zip(header_x, headers, strict=True):
        parts.append(
            f'<text x="{x}" y="1013" class="small" font-weight="700">{xml_escape(title)}</text>'
        )
    for index, row in enumerate(components):
        y = 1038 + index * 23
        values = (
            row.label,
            f"{row.candidate_model} − {row.reference_model}",
            f"{row.observed_recall_difference:+.3f}",
            _format_p_value(row.time_p_value),
            _format_p_value(row.space_p_value),
            _component_status_zh(row.attribution_status),
        )
        for x, value in zip(header_x, values, strict=True):
            parts.append(f'<text x="{x}" y="{y}" class="small">{xml_escape(value)}</text>')
    pending = "时间/空间置乱归因已完成" if placebo_complete else PLACEBO_PENDING_LABEL
    parts.append(f'<text x="45" y="1145" class="warn">{xml_escape(pending)}</text>')
    parts.append("</svg>\n")
    return "".join(parts)


def _decode_u16(value: str) -> tuple[int, ...]:
    payload = base64.b64decode(value, validate=True)
    if len(payload) % 2:
        raise D1RenderingError("internal uint16 payload is truncated")
    values = array("H")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(int(item) for item in values)


def _selected_static_issues(rows: Sequence[_IssueRow]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for fold in D1_FOLD_IDS:
        issues = sorted(
            {
                (row.issue_time_utc, row.issue_id)
                for row in rows
                if row.fold_id == fold and row.horizon_days == 30
            }
        )
        expected = D1_ASSESSMENT_ISSUES_PER_FOLD[30]
        if len(issues) != expected:
            raise D1RenderingError("static issue selection lacks the frozen 30d issue axis")
        issue_time, issue_id = issues[(len(issues) - 1) // 2]
        result[fold] = (issue_id, issue_time)
    return result


def _selected_inside_support_cases(
    frames: Sequence[_EncodedFrame],
) -> dict[str, tuple[_EncodedFrame, _TargetMarker] | None]:
    area_index = D1_AREA_BUDGETS_KM2.index(D1_PRIMARY_AREA_KM2)
    result: dict[str, tuple[_EncodedFrame, _TargetMarker] | None] = {
        "命中": None,
        "漏报": None,
    }
    candidates = sorted(
        (
            frame
            for frame in frames
            if frame.model_id == FULL_MODEL and frame.horizon_days == D1_PRIMARY_HORIZON_DAYS
        ),
        key=lambda frame: (
            frame.issue_time_utc,
            frame.fold_id,
            frame.issue_id,
        ),
    )
    for frame in candidates:
        for target in sorted(frame.targets, key=lambda item: item.cluster_id):
            if target.outside_support or target.cell_position is None:
                continue
            label = "命中" if target.hit_by_area[area_index] else "漏报"
            if result[label] is None:
                result[label] = (frame, target)
    return result


def _render_maps_svg(
    grid: Sequence[_GridCell],
    frames: Sequence[_EncodedFrame],
    *,
    issue_rows: Sequence[_IssueRow],
    best_intermediate: str,
) -> str:
    selected_issues = _selected_static_issues(issue_rows)
    frame_lookup = {frame.frame_key: frame for frame in frames}
    models = ("B0", best_intermediate, FULL_MODEL)
    width, height = 1280, 1405
    panel_w, panel_h = 390, 275
    map_x_pad, map_y_pad = 24, 55
    map_w, map_h = 340, 185
    min_x = min(cell.x_m for cell in grid)
    max_x = max(cell.x_m for cell in grid)
    min_y = min(cell.y_m for cell in grid)
    max_y = max(cell.y_m for cell in grid)
    if max_x <= min_x or max_y <= min_y:
        raise D1RenderingError("grid map extent is degenerate")
    colors = (
        "#313695",
        "#4575b4",
        "#74add1",
        "#abd9e9",
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#a50026",
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="map-title map-desc">',
        '<title id="map-title">D1三折同面积历史回放空间图</title>',
        f'<desc id="map-desc">{xml_escape(RETROSPECTIVE_LABEL)}；目标只作事后叠加，不参与报警区生成。</desc>',
        "<style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif;fill:#17202a}"
        ".panel{fill:#fff;stroke:#aeb9c4}.title{font-size:25px;font-weight:700}"
        ".warn{fill:#9c2c22;font-weight:700}.small{font-size:12px}.label{font-size:14px}</style>",
        '<rect width="100%" height="100%" fill="#f4f7fa"/>',
        '<text x="40" y="40" class="title">D1 30天、60万 km² 三折空间回放</text>',
        f'<text x="40" y="68" class="warn">{xml_escape(RETROSPECTIVE_LABEL)}</text>',
        f'<text x="40" y="91" class="label">{xml_escape(STRENGTH_LABEL)}；网格代表点示意，报警面积以数值为准；震群目标仅在评分后叠加。</text>',
    ]
    area_index = D1_AREA_BUDGETS_KM2.index(D1_PRIMARY_AREA_KM2)
    for fold_index, fold in enumerate(D1_FOLD_IDS):
        issue_id, issue_time = selected_issues[fold]
        for model_index, model in enumerate(models):
            key = (fold, issue_id, issue_time, 30, model)
            frame = frame_lookup.get(key)
            if frame is None:
                raise D1RenderingError("static map frame is missing")
            origin_x = 35 + model_index * 415
            origin_y = 115 + fold_index * 292
            parts.append(
                f'<rect class="panel" x="{origin_x}" y="{origin_y}" width="{panel_w}" height="{panel_h}" rx="7"/>'
            )
            title = f"{fold} · {model} · {issue_time[:10]}"
            parts.append(
                f'<text x="{origin_x + 15}" y="{origin_y + 27}" class="label" font-weight="700">{xml_escape(title)}</text>'
            )
            order = _decode_u16(frame.order_u16_b64)
            quantized_rank_order = _decode_u16(frame.strength_u16_b64)
            if len(order) != len(grid) or len(quantized_rank_order) != len(grid):
                raise D1RenderingError("static frame compact arrays are inconsistent")
            q_by_position = [0] * len(grid)
            for position, quantized in zip(order, quantized_rank_order, strict=True):
                q_by_position[position] = quantized

            def transform(
                cell: _GridCell,
                panel_x: int = origin_x,
                panel_y: int = origin_y,
            ) -> tuple[float, float]:
                x = panel_x + map_x_pad + (cell.x_m - min_x) / (max_x - min_x) * map_w
                y = panel_y + map_y_pad + (max_y - cell.y_m) / (max_y - min_y) * map_h
                return x, y

            for color_index, color in enumerate(colors):
                lower = color_index * 65_536 // len(colors)
                upper = (color_index + 1) * 65_536 // len(colors)
                commands: list[str] = []
                for position, cell in enumerate(grid):
                    if lower <= q_by_position[position] < upper:
                        x, y = transform(cell)
                        commands.append(f"M{x:.1f},{y:.1f}h1.8v1.8h-1.8z")
                parts.append(f'<path d="{"".join(commands)}" fill="{color}" opacity="0.88"/>')
            alarm_positions = order[: frame.alarm_prefix_counts[area_index]]
            alarm_commands: list[str] = []
            for position in alarm_positions:
                x, y = transform(grid[position])
                alarm_commands.append(f"M{x - 0.6:.1f},{y - 0.6:.1f}h3v3h-3z")
            parts.append(
                f'<path d="{"".join(alarm_commands)}" fill="none" stroke="#ffd400" stroke-width="0.55"/>'
            )
            hit_count = 0
            miss_count = 0
            outside_count = 0
            for target in frame.targets:
                if target.outside_support:
                    outside_count += 1
                    continue
                if target.cell_position is None:
                    raise D1RenderingError("inside-support target lacks map position")
                x, y = transform(grid[target.cell_position])
                hit = target.hit_by_area[area_index]
                hit_count += int(hit)
                miss_count += int(not hit)
                color = "#16a34a" if hit else "#dc2626"
                parts.append(
                    f'<circle cx="{x + 0.9:.1f}" cy="{y + 0.9:.1f}" r="4.5" fill="none" stroke="{color}" stroke-width="2.2"/>'
                )
            area = frame.actual_area_km2[area_index]
            summary = f"实际报警 {area:,.0f} km²；成熟震群 命中{hit_count}/漏报{miss_count}" + (
                f"/支持域外{outside_count}" if outside_count else ""
            )
            parts.append(
                f'<text x="{origin_x + 15}" y="{origin_y + 260}" class="small">{xml_escape(summary)}</text>'
            )
    parts.extend(
        [
            '<circle cx="55" cy="1002" r="5" fill="none" stroke="#16a34a" stroke-width="2"/>',
            '<text x="66" y="1006" class="small">命中成熟震群</text>',
            '<circle cx="190" cy="1002" r="5" fill="none" stroke="#dc2626" stroke-width="2"/>',
            '<text x="201" y="1006" class="small">漏报成熟震群</text>',
            '<rect x="330" y="996" width="9" height="9" fill="none" stroke="#ffd400"/>',
            '<text x="345" y="1006" class="small">报警格（完整前缀）</text>',
            f'<text x="550" y="1006" class="warn">最佳中间模型：{xml_escape(best_intermediate)}（冻结规则确定）</text>',
            '<text x="40" y="1042" class="label" font-weight="700">完整组合的确定性实际案例（仅从支持域内目标选择）</text>',
        ]
    )
    cases = _selected_inside_support_cases(frames)
    for case_index, label in enumerate(("命中", "漏报")):
        origin_x = 35 + case_index * 620
        origin_y = 1060
        case = cases[label]
        parts.append(
            f'<rect class="panel" x="{origin_x}" y="{origin_y}" width="590" height="295" rx="7"/>'
        )
        if case is None:
            parts.append(
                f'<text x="{origin_x + 20}" y="{origin_y + 45}" class="label">支持域内没有可展示的{label}案例</text>'
            )
            continue
        frame, target = case
        if target.outside_support or target.cell_position is None:
            raise D1RenderingError("static case selection may not use outside-support targets")
        title = (
            f"最早支持域内{label} · {frame.issue_time_utc[:10]} · "
            f"{frame.fold_id} · 震群 {target.cluster_id[:18]}"
        )
        parts.append(
            f'<text x="{origin_x + 15}" y="{origin_y + 27}" class="label" font-weight="700">{xml_escape(title)}</text>'
        )
        order = _decode_u16(frame.order_u16_b64)
        quantized_rank_order = _decode_u16(frame.strength_u16_b64)
        q_by_position = [0] * len(grid)
        for position, quantized in zip(order, quantized_rank_order, strict=True):
            q_by_position[position] = quantized

        def case_transform(
            cell: _GridCell,
            panel_x: int = origin_x,
            panel_y: int = origin_y,
        ) -> tuple[float, float]:
            x = panel_x + 25 + (cell.x_m - min_x) / (max_x - min_x) * 535
            y = panel_y + 55 + (max_y - cell.y_m) / (max_y - min_y) * 185
            return x, y

        for color_index, color in enumerate(colors):
            lower = color_index * 65_536 // len(colors)
            upper = (color_index + 1) * 65_536 // len(colors)
            commands = []
            for position, cell in enumerate(grid):
                if lower <= q_by_position[position] < upper:
                    x, y = case_transform(cell)
                    commands.append(f"M{x:.1f},{y:.1f}h2v2h-2z")
            parts.append(f'<path d="{"".join(commands)}" fill="{color}" opacity="0.88"/>')
        alarm_commands = []
        for position in order[: frame.alarm_prefix_counts[area_index]]:
            x, y = case_transform(grid[position])
            alarm_commands.append(f"M{x - 0.7:.1f},{y - 0.7:.1f}h3.4v3.4h-3.4z")
        parts.append(
            f'<path d="{"".join(alarm_commands)}" fill="none" stroke="#ffd400" stroke-width="0.65"/>'
        )
        x, y = case_transform(grid[target.cell_position])
        marker_color = "#16a34a" if label == "命中" else "#dc2626"
        parts.append(
            f'<circle cx="{x + 1:.1f}" cy="{y + 1:.1f}" r="7" fill="none" stroke="{marker_color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{origin_x + 15}" y="{origin_y + 275}" class="small">30天 · 60万 km²上限 · 实际报警 {frame.actual_area_km2[area_index]:,.0f} km²；目标仅事后叠加</text>'
        )
    parts.append(
        '<text x="40" y="1388" class="warn">案例选择按时间、折、起报日、震群ID确定性排序；支持域外目标绝不参与命中/漏报案例选择。</text>'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def _frame_payload(frame: _EncodedFrame) -> dict[str, object]:
    return {
        "fold": frame.fold_id,
        "issue": frame.issue_id,
        "time": frame.issue_time_utc,
        "horizon": frame.horizon_days,
        "model": frame.model_id,
        "order": frame.order_u16_b64,
        "strength": frame.strength_u16_b64,
        "strengthLogMin": frame.strength_log_min,
        "strengthLogMax": frame.strength_log_max,
        "prefix": list(frame.alarm_prefix_counts),
        "actualArea": list(frame.actual_area_km2),
        "targets": [
            {
                "cluster": target.cluster_id[:12],
                "cell": target.cell_position,
                "outside": target.outside_support,
                "hits": list(target.hit_by_area),
            }
            for target in frame.targets
        ],
    }


def _render_explorer_html(
    grid: Sequence[_GridCell],
    frames: Sequence[_EncodedFrame],
    metrics: Sequence[D1Metric],
    exposure: Sequence[D1AlarmExposureMetric],
    decisions: Sequence[D1RawEffectDecision],
    components: Sequence[_ComponentEvidence],
    *,
    best_intermediate: str,
    placebo_complete: bool,
    robustness: _SupplementalEvidence,
) -> str:
    robustness_summary = _robustness_numeric_summary(robustness)
    robustness_complete = robustness_summary is not None
    payload = {
        "label": RETROSPECTIVE_LABEL,
        "strengthLabel": STRENGTH_LABEL,
        "placeboStatus": ("时间/空间置乱归因已完成" if placebo_complete else PLACEBO_PENDING_LABEL),
        "robustnessStatus": (
            "区域贡献与去单震群诊断已完成" if robustness_complete else ROBUSTNESS_PENDING_LABEL
        ),
        "models": list(D1_MODEL_ORDER),
        "areas": [int(value) for value in D1_AREA_BUDGETS_KM2],
        "bestIntermediate": best_intermediate,
        "grid": [
            [round(cell.x_m, 3), round(cell.y_m, 3), round(cell.area_km2, 6)] for cell in grid
        ],
        "frames": [_frame_payload(frame) for frame in frames],
        "metrics": [item.as_mapping() for item in metrics if item.fold_id is None],
        "exposure": [item.as_mapping() for item in exposure if item.fold_id is None],
        "rawDecisions": [item.as_mapping() for item in decisions],
        "components": [item.as_mapping() for item in components],
        "robustness": robustness_summary,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("</", "<\\/")
    warning = html.escape(RETROSPECTIVE_LABEL)
    strength = html.escape(STRENGTH_LABEL)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
<title>D1 离线历史回放</title>
<style>
:root{{--ink:#17202a;--muted:#52606d;--paper:#f4f7fa;--card:#fff;--alarm:#ffd400;--hit:#16a34a;--miss:#dc2626}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif}}
header{{padding:18px 24px;background:#14213d;color:#fff}}header h1{{margin:0 0 8px;font-size:24px}}.warning{{font-weight:700;color:#ffd6d2}}
main{{padding:18px;display:grid;grid-template-columns:minmax(620px,1.4fr) minmax(430px,1fr);gap:16px}}.card{{background:var(--card);border:1px solid #c6d0da;border-radius:9px;padding:14px}}
.controls{{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px;margin-bottom:12px}}label{{font-size:13px;color:var(--muted)}}select{{width:100%;margin-top:4px;padding:7px}}
canvas{{width:100%;height:620px;border:1px solid #aeb9c4;background:#fff}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{border-bottom:1px solid #dde3e9;padding:6px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{font-size:13px;line-height:1.6;color:var(--muted)}}
@media(max-width:1100px){{main{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><h1>D1 真实历史开发回放浏览器</h1><div class="warning">{warning}</div><div>{strength}</div></header>
<main>
<section class="card">
  <div class="controls">
    <label>起报日<select id="issueControl"></select></label>
    <label>模型<select id="modelControl"></select></label>
    <label>窗口<select id="horizonControl"><option value="30">30天</option><option value="90">90天</option></select></label>
    <label>报警面积<select id="areaControl"></select></label>
  </div>
  <canvas id="mapCanvas" width="1000" height="620" aria-label="相对条件强度、报警区和成熟震群回放图"></canvas>
  <p id="frameSummary" class="note"></p>
</section>
<aside class="card">
  <h2>全部六模型指标</h2><p id="scienceStatus" class="note"></p>
  <table><thead><tr><th>模型</th><th>命中/震群</th><th>召回</th><th>平均实际面积</th><th>漏报率</th></tr></thead><tbody id="metricRows"></tbody></table>
  <h2>异常组件贡献（30天、60万 km²）</h2>
  <table><thead><tr><th>组件</th><th>观测召回差</th><th>时间 p</th><th>空间 p</th><th>归因</th></tr></thead><tbody id="componentRows"></tbody></table>
  <p id="robustnessStatus" class="note"></p>
  <h2>区域与逐震群稳健性（30天、60万 km²）</h2>
  <table><thead><tr><th>异常增量对比</th><th>命中增量</th><th>有目标区域 正/零/负</th><th>去最大正贡献区后仍为正</th><th>逐震群剔除范围</th><th>剔除后 正/零/负</th></tr></thead><tbody id="robustnessRows"></tbody></table>
  <h2>怎么看</h2><p class="note">颜色只表示同一期、同一模型内的相对条件强度。黄色边框是按目标无关顺位形成的完整网格报警前缀；绿色圆圈为事后命中，红色圆圈为事后漏报。震中只在报警区冻结后叠加。</p>
  <p class="note">最佳中间模型：<span id="bestIntermediate"></span>。预分类并非最终科学结论；必须结合时间/空间置乱、区域贡献与去单震群诊断。</p>
</aside>
</main>
<script id="d1Payload" type="application/json">{encoded}</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('d1Payload').textContent);
const issue=document.getElementById('issueControl'),model=document.getElementById('modelControl'),horizon=document.getElementById('horizonControl'),area=document.getElementById('areaControl');
const canvas=document.getElementById('mapCanvas'),ctx=canvas.getContext('2d');
const palette=['#313695','#4575b4','#74add1','#abd9e9','#fee090','#fdae61','#f46d43','#a50026'];
function option(select,value,label){{const node=document.createElement('option');node.value=String(value);node.textContent=label;select.appendChild(node)}}
DATA.models.forEach(value=>option(model,value,value));DATA.areas.forEach(value=>option(area,value,`${{Math.round(value/1000)}}千 km²`));
function decodeU16(encoded){{const raw=atob(encoded),result=new Uint16Array(raw.length/2);for(let i=0;i<result.length;i++)result[i]=raw.charCodeAt(2*i)|(raw.charCodeAt(2*i+1)<<8);return result}}
function eligibleFrames(){{return DATA.frames.filter(frame=>frame.model===model.value&&frame.horizon===Number(horizon.value))}}
function refreshIssues(){{const previous=issue.value;issue.textContent='';eligibleFrames().sort((a,b)=>a.time.localeCompare(b.time)).forEach(frame=>option(issue,`${{frame.fold}}|${{frame.issue}}|${{frame.time}}`,`${{frame.time.slice(0,10)}} · ${{frame.fold}}`));if([...issue.options].some(node=>node.value===previous))issue.value=previous;draw()}}
function chosenFrame(){{const parts=issue.value.split('|');return DATA.frames.find(frame=>frame.fold===parts[0]&&frame.issue===parts[1]&&frame.time===parts[2]&&frame.model===model.value&&frame.horizon===Number(horizon.value))}}
function metric(modelId){{return DATA.metrics.find(row=>row.model_id===modelId&&row.horizon_days===Number(horizon.value)&&row.area_budget_km2===Number(area.value))}}
function exposure(modelId){{return DATA.exposure.find(row=>row.model_id===modelId&&row.horizon_days===Number(horizon.value)&&row.area_budget_km2===Number(area.value))}}
function draw(){{const frame=chosenFrame();if(!frame)return;const xs=DATA.grid.map(row=>row[0]),ys=DATA.grid.map(row=>row[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);ctx.clearRect(0,0,canvas.width,canvas.height);const order=decodeU16(frame.order),q=decodeU16(frame.strength),areaIndex=DATA.areas.indexOf(Number(area.value)),prefix=frame.prefix[areaIndex];for(let rank=order.length-1;rank>=0;rank--){{const pos=order[rank],cell=DATA.grid[pos],x=35+(cell[0]-minX)/(maxX-minX)*930,y=30+(maxY-cell[1])/(maxY-minY)*535,bin=Math.min(7,Math.floor(q[rank]/8192));ctx.fillStyle=palette[bin];ctx.fillRect(x,y,2.2,2.2)}}ctx.strokeStyle='#ffd400';ctx.lineWidth=1;for(let rank=0;rank<prefix;rank++){{const cell=DATA.grid[order[rank]],x=35+(cell[0]-minX)/(maxX-minX)*930,y=30+(maxY-cell[1])/(maxY-minY)*535;ctx.strokeRect(x-1,y-1,4,4)}}let hits=0,misses=0,outside=0;frame.targets.forEach(target=>{{if(target.outside||target.cell===null){{outside++;return}}const cell=DATA.grid[target.cell],x=36+(cell[0]-minX)/(maxX-minX)*930,y=31+(maxY-cell[1])/(maxY-minY)*535,hit=target.hits[areaIndex];hits+=hit?1:0;misses+=hit?0:1;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.strokeStyle=hit?'#16a34a':'#dc2626';ctx.lineWidth=3;ctx.stroke()}});document.getElementById('frameSummary').textContent=`${{DATA.label}}；${{frame.time.slice(0,10)}}，${{frame.model}}，${{frame.horizon}}天，实际报警 ${{Math.round(frame.actualArea[areaIndex]).toLocaleString()}} km²；成熟震群命中${{hits}}、漏报${{misses}}、支持域外${{outside}}。`;renderMetrics()}}
function componentStatus(value){{return ({{pending:'待完成',evidence_insufficient:'证据不足',no_positive_observed_gain:'观测增量不为正',supported_against_time_and_space_placebos:'同时超过时间与空间置乱',not_supported_against_both_placebos:'未同时超过两类置乱'}})[value]||value}}
function renderComponents(){{const body=document.getElementById('componentRows');body.textContent='';DATA.components.forEach(item=>{{const row=document.createElement('tr'),values=[item.label,item.observed_recall_difference.toFixed(3),item.time_placebo_p_value===null?'待完成':item.time_placebo_p_value.toFixed(4),item.space_placebo_p_value===null?'待完成':item.space_placebo_p_value.toFixed(4),componentStatus(item.attribution_status)];values.forEach(value=>{{const cell=document.createElement('td');cell.textContent=String(value);row.appendChild(cell)}});body.appendChild(row)}});document.getElementById('robustnessStatus').textContent=DATA.robustnessStatus}}
function signText(value){{return `${{value.positive_count}}/${{value.zero_count}}/${{value.negative_count}}`}}
function renderRobustness(){{const body=document.getElementById('robustnessRows');body.textContent='';if(DATA.robustness===null){{const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=6;cell.textContent='完整数值诊断尚未完成，不能形成最终归因。';row.appendChild(cell);body.appendChild(row);return}}DATA.robustness.contrasts.forEach(item=>{{const regional=item.regional,loco=item.leave_one_cluster_out,row=document.createElement('tr'),values=[item.contrast_id,`${{item.observed_hit_gain_sum>=0?'+':''}}${{item.observed_hit_gain_sum}} (${{(100*item.observed_recall_gain).toFixed(1)}}%)`,signText(regional.target_bearing_zone_gain_sign_counts),regional.direction_survives_largest_positive_zone_removal===null?'无正贡献区':(regional.direction_survives_largest_positive_zone_removal?'是':'否'),`${{(100*loco.recall_gain_minimum).toFixed(1)}}% 至 ${{(100*loco.recall_gain_maximum).toFixed(1)}}%`,signText(loco.recall_gain_sign_counts)];values.forEach(value=>{{const cell=document.createElement('td');cell.textContent=String(value);row.appendChild(cell)}});body.appendChild(row)}})}}
function renderMetrics(){{const body=document.getElementById('metricRows');body.textContent='';DATA.models.forEach(modelId=>{{const m=metric(modelId),e=exposure(modelId),row=document.createElement('tr');[modelId,`${{m.hit_count}}/${{m.cluster_count}}`,m.recall===null?'证据不足':m.recall.toFixed(3),Math.round(e.mean_actual_area_km2).toLocaleString(),m.recall===null?'证据不足':(1-m.recall).toFixed(3)].forEach(value=>{{const cell=document.createElement('td');cell.textContent=String(value);row.appendChild(cell)}});body.appendChild(row)}});document.getElementById('scienceStatus').textContent=DATA.placeboStatus;renderComponents();renderRobustness()}}
model.addEventListener('change',refreshIssues);horizon.addEventListener('change',refreshIssues);area.addEventListener('change',draw);issue.addEventListener('change',draw);
document.getElementById('bestIntermediate').textContent=DATA.bestIntermediate;refreshIssues();
</script>
</body></html>
"""


def _training_lines(payload: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for raw_fold in _sequence(payload.get("folds"), label="folds"):
        fold = _mapping(raw_fold, label="fold")
        fold_id = _safe_id(fold.get("fold_id"), label="fold_id")
        ridge_parts: list[str] = []
        diagnostic_parts: list[str] = []
        fit_issue_counts: set[int] = set()
        fit_catalog_event_counts: set[int] = set()
        model_rows = tuple(
            _mapping(raw_model, label="fold model")
            for raw_model in _sequence(fold.get("models"), label="fold models")
        )
        alpha_values = {
            _finite_float(model.get("selected_alpha"), label="selected_alpha")
            for model in model_rows
        }
        if len(alpha_values) != 1:
            raise D1RenderingError("fold alpha summary is inconsistent")
        alpha = next(iter(alpha_values))
        for model in model_rows:
            model_id = _safe_id(model.get("model_id"), label="model_id")
            ridge = model.get("selected_ridge")
            if ridge is not None:
                ridge_parts.append(f"{model_id}={_finite_float(ridge, label='selected_ridge'):g}")
            diagnostic = _mapping(
                model.get("training_diagnostic"),
                label="training_diagnostic",
            )
            fit_issue_counts.add(
                _exact_int(
                    diagnostic.get("fit_issue_count"),
                    label="fit_issue_count",
                    minimum=1,
                )
            )
            fit_catalog_event_counts.add(
                _exact_int(
                    diagnostic.get("fit_catalog_m4plus_event_count"),
                    label="fit_catalog_m4plus_event_count",
                    minimum=1,
                )
            )
            objective = diagnostic.get("objective")
            if objective is not None:
                diagnostic_parts.append(
                    f"{model_id}目标值={_finite_float(objective, label='objective'):.4g}"
                    f"/{_exact_int(diagnostic.get('iteration_count'), label='iterations')}次迭代"
                )
        ridge_text = "、".join(ridge_parts) if ridge_parts else "背景模型不使用岭惩罚"
        if len(fit_issue_counts) != 1 or len(fit_catalog_event_counts) != 1:
            raise D1RenderingError("one fold must use one common final-fit issue/catalog axis")
        issue_text = str(next(iter(fit_issue_counts)))
        event_text = "/".join(str(value) for value in sorted(fit_catalog_event_counts))
        diagnostic_text = "、".join(diagnostic_parts)
        lines.append(
            f"- {fold_id}：最终拟合使用 {issue_text} 个历史起报日、目录 M4+ 事件数 {event_text}；"
            f"近期地震混合权重 α={alpha:g}；"
            f"岭惩罚 {ridge_text}；特征模型优化 {diagnostic_text}。"
        )
    return lines


def _robustness_report_section(evidence: _SupplementalEvidence) -> str:
    summary = _robustness_numeric_summary(evidence)
    if summary is None:
        return (
            "## 区域与逐震群稳健性\n\n"
            "完整数值诊断尚未完成，因此不能把异常增量解释成稳定的科学贡献，"
            "最终归因状态仍为未就绪。"
        )
    rows: list[str] = []
    for raw in cast(list[dict[str, Any]], summary["contrasts"]):
        regional = cast(dict[str, Any], raw["regional"])
        loco = cast(dict[str, Any], raw["leave_one_cluster_out"])
        zone_signs = cast(dict[str, int], regional["target_bearing_zone_gain_sign_counts"])
        loco_signs = cast(dict[str, int], loco["recall_gain_sign_counts"])
        largest = cast(dict[str, Any] | None, regional["largest_positive_zone"])
        largest_text = (
            "无正贡献区"
            if largest is None
            else f"{100 * float(largest['fraction_of_all_positive_zone_gain']):.1f}%"
        )
        survives_zone = regional["direction_survives_largest_positive_zone_removal"]
        rows.append(
            "| "
            + " | ".join(
                (
                    str(raw["contrast_id"]),
                    f"{int(raw['observed_hit_gain_sum']):+d} / {100 * float(raw['observed_recall_gain']):+.1f}%",
                    f"{zone_signs['positive_count']}/{zone_signs['zero_count']}/{zone_signs['negative_count']}",
                    largest_text,
                    "不适用" if survives_zone is None else ("是" if survives_zone else "否"),
                    f"{100 * float(loco['recall_gain_minimum']):+.1f}% 至 {100 * float(loco['recall_gain_maximum']):+.1f}%",
                    f"{loco_signs['positive_count']}/{loco_signs['zero_count']}/{loco_signs['negative_count']}",
                    "是" if loco["direction_survives_every_cluster_removal"] else "否",
                )
            )
            + " |"
        )
    spatial = cast(dict[str, Any], summary["spatial_identity"])
    return f"""## 区域与逐震群稳健性（30天、60万 km²）

这里检查模型的增益是否只来自一个热点地区或某一个震群。区域是目标无关构建的39个固定区；逐震群诊断每次去掉21个独立震群中的一个，并重新计算20个剩余震群的召回差，不重新训练模型。

| 异常增量对比 | 命中/召回增量 | 有目标区域 正/零/负 | 最大正贡献区占全部正贡献 | 去该区后方向仍为正 | 逐震群剔除召回差范围 | 剔除后 正/零/负 | 每次剔除后均为正 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
{chr(10).join(rows)}

空间身份：固定网格 `{spatial["operational_grid_id"]}`，{spatial["operational_cell_count"]} 个格点、{spatial["nonempty_zone_count"]} 个非空目标无关区域；空间公开清单内容身份 `{spatial["public_manifest_content_sha256"]}`。上表是描述性稳健性诊断，不会把相对强度改称为绝对概率。"""


def _render_science_report(
    payload: Mapping[str, Any],
    metrics: Sequence[D1Metric],
    exposure: Sequence[D1AlarmExposureMetric],
    decisions: Sequence[D1RawEffectDecision],
    components: Sequence[_ComponentEvidence],
    *,
    best_intermediate: str,
    placebo_complete: bool,
    robustness: _SupplementalEvidence,
) -> str:
    robustness_complete = _robustness_numeric_summary(robustness) is not None
    robustness_section = _robustness_report_section(robustness)
    lookup = _metric_lookup(metrics)
    exposure_lookup = _exposure_lookup(exposure)
    primary_rows: list[str] = []
    for model in D1_MODEL_ORDER:
        metric = lookup[(model, 30, None, 600_000.0)]
        alarm = exposure_lookup[(model, 30, None, 600_000.0)]
        recall = "证据不足" if metric.recall is None else f"{metric.recall:.1%}"
        primary_rows.append(
            f"| {model} | {metric.hit_count}/{metric.cluster_count} | {recall} | "
            f"{alarm.mean_actual_area_km2:,.0f} | {metric.outside_support_count} |"
        )
    decision_lookup = {item.model_id: item for item in decisions}
    full_decision = decision_lookup[FULL_MODEL]
    level_zh = {
        "strong": "强",
        "promising": "有希望",
        "weak": "弱",
        "none_or_harmful": "无实质提升或可能有害",
    }[full_decision.raw_effect_level]
    if placebo_complete and robustness_complete:
        final_status = "时间/空间置乱和稳健性诊断均已完成，可以结合这些诊断形成最终科学解释。"
    elif placebo_complete:
        final_status = f"时间/空间置乱已完成；{ROBUSTNESS_PENDING_LABEL}"
    else:
        final_status = PLACEBO_PENDING_LABEL
    component_rows = "\n".join(
        "| "
        + " | ".join(
            (
                item.label,
                f"{item.candidate_model} − {item.reference_model}",
                f"{item.observed_recall_difference:+.3f}",
                _format_p_value(item.time_p_value),
                _format_p_value(item.space_p_value),
                _component_status_zh(item.attribution_status),
            )
        )
        + " |"
        for item in components
    )
    training = "\n".join(_training_lines(payload))
    return f"""# SeismoFlux D1 真实历史开发回放科学报告

> **{RETROSPECTIVE_LABEL}**<br>
> **{STRENGTH_LABEL}**

## 一句话结论

本轮把六种方法放到过去从未参与各折训练的时间段上，统一限制报警面积，再看能否多覆盖独立的 M5–6 规则震群。完整组合 `{FULL_MODEL}` 的**原始预测效果预分类**为“{level_zh}”：相对长期背景 B0 多命中 {full_decision.pooled_hit_gain} 个震群，召回变化 {full_decision.pooled_recall_gain:+.1%}。这还不是最终科学结论。{final_status}

## 用了什么数据

- 40,898 条冻结地震目录：历史 M4+ 地震用于学习空间背景和近30天地震活动；未来窗口内 M5–6 地震只用于事后评分。
- 205 个真实异常报告期及其完整历史状态：从中提取报告覆盖、单期异常和随时间变化的异常。
- 15,697 个固定约25 km查询格和冻结中国大陆研究区；所有模型使用同一网格和同一面积规则。
- 39 个由目标无关信息形成的空间区只用于后续空间置乱和区域诊断，不进入模型训练。

## 考虑了哪些特征

- `C1/C2`：某地报告是否相对完整、报告来源是否足够广。
- `S1–S5`：当前异常负荷、首次出现负荷、未延续负荷、学科多样性和空间集中度。
- `D1/D2`：异常数量和首次出现数量的13周趋势、加速度与52周峰值回落。
- 近30天 M4+ 地震活动作为一个独立短期背景分量。

没有使用人工预测地点/震级/时间、真实震中来生成候选区、未来报告、锁定测试、旧模型效果分数、断层属性或长期危险性快照。

## 用了什么模型

六个模型从简单到完整依次为：B0长期背景；B0_R30加入近30天地震；B0_C加入报告覆盖；B0_C_A_snapshot再加入单期异常；B0_C_A_dynamic再加入动态异常；B0_R30_C_A_dynamic把近期地震与全部异常组合。训练目标是未来30天 M4+ 地震在固定格上的相对空间分布，岭惩罚防止小样本下系数过大。

## 训练情况

每一折只用该折评估期之前已经可获得的数据选择近期地震混合权重和岭惩罚，再在该折完整历史训练期重拟合。选择结果：

{training}

训练拟合好坏不能代替预测检验；真正关键的是下面的时间外推结果。

## 时间外推效果（30天、60万 km²）

| 模型 | 命中/独立震群 | 召回 | 全部起报日平均实际报警面积 km² | 支持域外震群 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(primary_rows)}

最佳中间模型按“30天60万 km²召回最高 → 达到相同召回所需面积更小 → 冻结模型顺序”确定为 `{best_intermediate}`。Bootstrap以独立规则震群为抽样单位，配对运行2000次；没有把网格或重复地震当成独立样本。30天和90天的全部五档面积曲线见 `d1_effects.svg`。

## 三类异常组件的贡献

| 组件 | 模型差 | 30天60万 km²观测召回差 | 时间置乱 p | 空间置乱 p | 归因状态 |
| --- | --- | ---: | ---: | ---: | --- |
{component_rows}

这里把“单期异常增量”“全部异常增量”“动态演化在单期异常之上的额外增量”分开显示。置乱结果缺失时只报告观测差，不把它解释为异常机制贡献。区域贡献与去单震群诊断状态：{"已完成" if robustness_complete else ROBUSTNESS_PENDING_LABEL}

{robustness_section}

## 图件怎么读

- `d1_effects.svg`：六模型在30天/90天、五档面积下的召回曲线、Molchan曲线和30天60万 km²配对Bootstrap效果。
- `d1_maps.svg`：每折中位30天起报日比较B0、最佳中间模型和完整组合，并给出完整组合中按固定顺序挑出的支持域内真实命中/漏报案例；黄色是目标无关报警前缀，绿/红圆圈是事后命中/漏报。
- `d1_explorer.html`：完全离线切换起报日、模型、30/90天和面积，始终显示全部六模型指标。

## 当前科学价值与下一步

这一步直接回答“模型在未参与训练的后来时间段上，固定报警面积后是否比长期背景多覆盖独立震群”，因此属于对最终预测目标的直接检验。当前 strong/promising/weak 等词只表示**原始预测效果预分类**。必须完成时间置乱、空间置乱、区域贡献和去单震群诊断，确认增益不是报告日历、空间结构或单一震群造成，之后才能给出最终机制归因与是否进入真正前瞻预测的决定。锁定测试和真正前瞻预测均未运行。
"""


def _supplemental_summary(
    evidence: _SupplementalEvidence,
    *,
    project_root: Path,
) -> dict[str, object]:
    return {
        "status": evidence.status,
        "path": (None if evidence.path is None else _relative_path(evidence.path, project_root)),
        "sha256": evidence.file_sha256,
    }


def _actual_case_summary(
    frames: Sequence[_EncodedFrame],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for label, case in _selected_inside_support_cases(frames).items():
        if case is None:
            result[label] = None
            continue
        frame, target = case
        result[label] = {
            "fold_id": frame.fold_id,
            "issue_id": frame.issue_id,
            "issue_time_utc": frame.issue_time_utc,
            "horizon_days": frame.horizon_days,
            "model_id": frame.model_id,
            "area_budget_km2": D1_PRIMARY_AREA_KM2,
            "cluster_id": target.cluster_id,
            "outside_support": False,
            "hit": label == "命中",
        }
    return result


def _build_science_summary(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    observed_path: Path,
    scores_path: Path,
    metrics: Sequence[D1Metric],
    exposure: Sequence[D1AlarmExposureMetric],
    bootstrap: Sequence[D1BootstrapEffect],
    decisions: Sequence[D1RawEffectDecision],
    components: Sequence[_ComponentEvidence],
    frames: Sequence[_EncodedFrame],
    best_intermediate: str,
    placebo: _SupplementalEvidence,
    robustness: _SupplementalEvidence,
) -> dict[str, object]:
    robustness_summary = _robustness_numeric_summary(robustness)
    primary_metrics = [
        item.as_mapping()
        for item in metrics
        if item.fold_id is None
        and item.horizon_days == D1_PRIMARY_HORIZON_DAYS
        and item.area_budget_km2 == D1_PRIMARY_AREA_KM2
    ]
    if len(primary_metrics) != len(D1_MODEL_ORDER):
        raise D1RenderingError("science summary lacks the six-model primary axis")
    return {
        "schema_version": _SCHEMA_VERSION,
        "protocol_version": _PROTOCOL_VERSION,
        "result_kind": "d1_science_summary",
        "status": "completed",
        "retrospective_only": True,
        "locked_test_run": False,
        "prospective_forecast_run": False,
        "relative_strength_not_absolute_probability": True,
        "retrospective_label": RETROSPECTIVE_LABEL,
        "relative_strength_semantics": STRENGTH_LABEL,
        "identities": dict(_observed_identities(payload)),
        "frozen_axis": {
            "models": list(D1_MODEL_ORDER),
            "folds": list(D1_FOLD_IDS),
            "horizons_days": list(D1_HORIZONS_DAYS),
            "area_budgets_km2": list(D1_AREA_BUDGETS_KM2),
            "cluster_counts_by_horizon": {
                str(key): value for key, value in _EXPECTED_CLUSTER_COUNTS.items()
            },
            "cluster_counts_by_horizon_and_fold": {
                str(key): dict(value) for key, value in _EXPECTED_CLUSTER_COUNTS_BY_FOLD.items()
            },
            "issue_counts_by_horizon": {
                str(key): value for key, value in _EXPECTED_ISSUE_COUNTS.items()
            },
            "issue_counts_by_horizon_and_fold": {
                str(key): dict(value) for key, value in _EXPECTED_ISSUE_COUNTS_BY_FOLD.items()
            },
            "issue_model_frame_count": 198,
        },
        "primary_endpoint": {
            "horizon_days": D1_PRIMARY_HORIZON_DAYS,
            "area_budget_km2": D1_PRIMARY_AREA_KM2,
            "best_intermediate_model": best_intermediate,
            "metrics": primary_metrics,
            "raw_effect_decisions": [item.as_mapping() for item in decisions],
        },
        "all_metrics": [item.as_mapping() for item in metrics],
        "all_alarm_exposure": [item.as_mapping() for item in exposure],
        "paired_cluster_bootstrap": [item.as_mapping() for item in bootstrap],
        "component_contributions": [item.as_mapping() for item in components],
        "deterministic_inside_support_cases": _actual_case_summary(frames),
        "supplemental_evidence": {
            "time_and_space_placebos": _supplemental_summary(placebo, project_root=project_root),
            "regional_and_leave_one_cluster_robustness": _supplemental_summary(
                robustness, project_root=project_root
            )
            | {"diagnostic_summary": robustness_summary},
            "final_attribution_ready": (
                placebo.status == "completed" and robustness_summary is not None
            ),
        },
        "scientific_value_review": {
            "classification": "direct_test",
            "reason": (
                "At fixed alarm-area budgets, the replay directly tests whether "
                "independent physical M5-6 cluster recall improves out of time."
            ),
        },
        "inputs": {
            "observed_result": {
                "path": _relative_path(observed_path, project_root),
                "sha256": _sha256_file(observed_path),
            },
            "cell_scores": {
                "path": _relative_path(scores_path, project_root),
                "sha256": _sha256_file(scores_path),
            },
        },
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def render_d1_deliverables(
    project_root: Path | str,
    observed_result_path: Path | str,
    cell_scores_path: Path | str,
    output_root: Path | str,
    *,
    placebo_result_path: Path | str | None = None,
    robustness_result_path: Path | str | None = None,
) -> D1RenderedDeliverables:
    """Render all required D1 observed-replay deliverables and their manifest.

    The function fails closed before writing if the result is incomplete, the
    six-model support differs, the 198 issue/model alarm frames are incomplete,
    or the Parquet alarm masks disagree with their registered issue outcomes.
    """

    project = Path(project_root).resolve()
    observed_path = Path(observed_result_path).resolve()
    scores_path = Path(cell_scores_path).resolve()
    output = Path(output_root).resolve()
    placebo_path = None if placebo_result_path is None else Path(placebo_result_path).resolve()
    robustness_path = (
        None if robustness_result_path is None else Path(robustness_result_path).resolve()
    )
    if not project.is_dir() or not observed_path.is_file() or not scores_path.is_file():
        raise D1RenderingError("project root and observed replay inputs must exist")
    payload = _read_observed_result(observed_path)
    _validate_training_summary(payload)
    expected_support = _parse_expected_support(payload)
    expected_issues = _parse_expected_issues(payload)
    outcomes, overlays = _parse_outcomes(payload)
    issue_rows, alarm_outcomes = _parse_issue_rows(
        payload,
        expected_issues=expected_issues,
    )
    _validate_cell_binding(
        payload,
        observed_path=observed_path,
        scores_path=scores_path,
        expected_frame_count=len(issue_rows),
    )
    metrics = summarize_metrics(
        outcomes,
        expected_support_by_horizon=expected_support,
    )
    bootstrap = paired_cluster_bootstrap(
        outcomes,
        replications=2000,
        expected_support_by_horizon=expected_support,
    )
    decisions = tuple(
        classify_primary_raw_effect(metrics, bootstrap, model_id=model)
        for model in D1_MODEL_ORDER[1:]
    )
    placebo = _validate_placebo_result(
        placebo_path,
        observed=payload,
        metrics=metrics,
    )
    robustness = _validate_robustness_result(
        robustness_path,
        observed=payload,
        outcomes=outcomes,
        overlays=overlays,
        expected_support=expected_support,
    )
    components = _build_component_evidence(metrics, placebo)
    best_intermediate = select_best_intermediate_model(metrics)
    frame_result = _build_frames(
        scores_path,
        issue_rows=issue_rows,
        overlays=overlays,
    )
    registered_study_area = _finite_float(
        payload.get("study_area_km2"),
        label="study_area_km2",
        minimum=1.0e-12,
    )
    if not math.isclose(
        registered_study_area,
        frame_result.study_area_km2,
        rel_tol=1.0e-12,
        abs_tol=1.0e-8,
    ):
        raise D1RenderingError("registered study area differs from the frozen parquet grid")
    exposure = summarize_alarm_exposure(
        alarm_outcomes,
        expected_issues_by_horizon=expected_issues,
        study_area_km2=frame_result.study_area_km2,
    )
    placebo_complete = placebo.status == "completed"
    robustness_complete = robustness.status == "completed"

    effects_svg = _render_effects_svg(
        metrics,
        exposure,
        bootstrap,
        components,
        placebo_complete=placebo_complete,
    )
    maps_svg = _render_maps_svg(
        frame_result.grid,
        frame_result.frames,
        issue_rows=issue_rows,
        best_intermediate=best_intermediate,
    )
    report = _render_science_report(
        payload,
        metrics,
        exposure,
        decisions,
        components,
        best_intermediate=best_intermediate,
        placebo_complete=placebo_complete,
        robustness=robustness,
    )
    explorer = _render_explorer_html(
        frame_result.grid,
        frame_result.frames,
        metrics,
        exposure,
        decisions,
        components,
        best_intermediate=best_intermediate,
        placebo_complete=placebo_complete,
        robustness=robustness,
    )
    science_summary = _build_science_summary(
        payload,
        project_root=project,
        observed_path=observed_path,
        scores_path=scores_path,
        metrics=metrics,
        exposure=exposure,
        bootstrap=bootstrap,
        decisions=decisions,
        components=components,
        frames=frame_result.frames,
        best_intermediate=best_intermediate,
        placebo=placebo,
        robustness=robustness,
    )

    output.mkdir(parents=True, exist_ok=True)
    effects_path = output / "d1_effects.svg"
    maps_path = output / "d1_maps.svg"
    report_path = output / "d1_science_report.md"
    explorer_path = output / "d1_explorer.html"
    summary_path = output / "d1_science_summary.json"
    manifest_path = output / "d1_rendered_deliverables_manifest.json"
    _write_text_atomic(effects_path, effects_svg)
    _write_text_atomic(maps_path, maps_svg)
    _write_text_atomic(report_path, report)
    _write_text_atomic(explorer_path, explorer)
    write_json_atomic(summary_path, science_summary)

    outputs = []
    for path, media_type in (
        (effects_path, "image/svg+xml"),
        (maps_path, "image/svg+xml"),
        (report_path, "text/markdown; charset=utf-8"),
        (explorer_path, "text/html; charset=utf-8"),
        (summary_path, "application/json"),
    ):
        outputs.append(
            {
                "path": _relative_path(path, project),
                "media_type": media_type,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest_preimage: dict[str, object] = {
        "schema_version": 1,
        "result_kind": "d1_observed_replay_rendered_deliverables",
        "retrospective_label": RETROSPECTIVE_LABEL,
        "relative_strength_semantics": STRENGTH_LABEL,
        "best_intermediate_model": best_intermediate,
        "placebo_attribution_complete": placebo_complete,
        "robustness_diagnostics_complete": robustness_complete,
        "inputs": {
            "observed_result": {
                "path": _relative_path(observed_path, project),
                "sha256": _sha256_file(observed_path),
            },
            "cell_scores": {
                "path": _relative_path(scores_path, project),
                "sha256": _sha256_file(scores_path),
            },
            "placebo_result": _supplemental_summary(
                placebo,
                project_root=project,
            ),
            "robustness_result": _supplemental_summary(
                robustness,
                project_root=project,
            ),
        },
        "outputs": outputs,
    }
    manifest_identity = hashlib.sha256(canonical_json_bytes(manifest_preimage)).hexdigest()
    manifest = dict(manifest_preimage)
    manifest["manifest_identity_sha256"] = manifest_identity
    write_json_atomic(manifest_path, manifest)
    return D1RenderedDeliverables(
        effects_svg_path=effects_path,
        maps_svg_path=maps_path,
        science_report_path=report_path,
        explorer_html_path=explorer_path,
        science_summary_path=summary_path,
        manifest_path=manifest_path,
        manifest_identity_sha256=manifest_identity,
        manifest_file_sha256=_sha256_file(manifest_path),
        best_intermediate_model=best_intermediate,
    )


__all__ = [
    "INTERMEDIATE_MODEL_ORDER",
    "RETROSPECTIVE_LABEL",
    "STRENGTH_LABEL",
    "D1RenderedDeliverables",
    "D1RenderingError",
    "render_d1_deliverables",
    "select_best_intermediate_model",
]
