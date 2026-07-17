"""Coordinate-free static and interactive publication templates for stage 4.

The retrospective and prospective payloads are intentionally different Python
types.  Prospective records have no target, hit, recall, score, or information-
gain fields, which makes target overlays impossible to serialize in true
forecast mode by construction.
"""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import dataclasses
import html
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

from seismoflux.anomaly_increment.evaluation import AdoptionChoice, GateStatus, MagnitudeBin
from seismoflux.anomaly_increment.preregistration import (
    PRIMARY_MACRO_HORIZONS_DAYS,
    STAGE4_HORIZONS_DAYS,
    STAGE4_PROTOCOL_VERSION,
)
from seismoflux.anomaly_increment.publication_diagnostics import (
    AlarmBudgetRecallCurve,
    CoefficientEffectCurve,
    DataMethodFlowDiagnostics,
    DistanceLeadDecayDiagnostics,
    FoldMacroValue,
    InformationGainInterval,
    PermutationDistribution,
    PublicationDiagnostics,
    RegionHorizonMetric,
    SameRecallAreaReduction,
)

ResultStatus: TypeAlias = Literal["passed", "failed", "evidence_insufficient"]
HorizonEvidenceStatus: TypeAlias = Literal[
    "confirmatory",
    "evidence_insufficient_no_random_split",
]
ModelVariant: TypeAlias = Literal["background_no_increment", "coverage_only", "snapshot", "dynamic"]
ForecastStatus: TypeAlias = Literal[
    "genuine_prospective",
    "retrospective_generated_target_blind_shadow",
]

_RESTRICTED_GEOMETRY_TOKENS = (
    "longitude",
    "latitude",
    "geojson",
    "linestring",
    "polygon_wkb",
    "query_x_m",
    "query_y_m",
    "cell_id",
    "per_cell_mapping",
    "construction_zone_geometry",
)
_PROSPECTIVE_FORBIDDEN_KEY_PARTS = (
    "target",
    "hit",
    "recall",
    "information_gain",
    "p_value",
    "score",
    "event_count",
    "observed_event",
)


def _finite_optional(value: float | None, *, label: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{label} must be finite when present")


@dataclass(frozen=True, slots=True)
class HorizonProtocolDisplay:
    horizon_days: int
    evidence_status: HorizonEvidenceStatus

    def __post_init__(self) -> None:
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("horizon_days is outside the frozen five horizons")
        expected: HorizonEvidenceStatus = (
            "confirmatory"
            if self.horizon_days in PRIMARY_MACRO_HORIZONS_DAYS
            else "evidence_insufficient_no_random_split"
        )
        if self.evidence_status != expected:
            raise ValueError("horizon evidence label differs from the frozen protocol")


@dataclass(frozen=True, slots=True)
class ProtocolDisplay:
    protocol_version: str
    model_version: str
    background_variant: str
    etas_status: Literal["not_evaluable"]
    etas_reason: Literal["failed_numerical_stability_no_qualified_parameters"]
    horizons: tuple[HorizonProtocolDisplay, ...]
    interpretation: Literal["conditional_relative_intensity_and_rank"] = (
        "conditional_relative_intensity_and_rank"
    )

    def __post_init__(self) -> None:
        if self.protocol_version != STAGE4_PROTOCOL_VERSION:
            raise ValueError("protocol_version differs from the frozen stage-4 protocol")
        if not self.model_version or self.model_version != self.model_version.strip():
            raise ValueError("model_version must be a non-empty trimmed string")
        if (
            not self.background_variant
            or self.background_variant != self.background_variant.strip()
        ):
            raise ValueError("background_variant must be a non-empty trimmed string")
        if tuple(item.horizon_days for item in self.horizons) != STAGE4_HORIZONS_DAYS:
            raise ValueError("protocol display must retain all five frozen horizons in order")


def default_protocol_display(*, model_version: str) -> ProtocolDisplay:
    return ProtocolDisplay(
        protocol_version=STAGE4_PROTOCOL_VERSION,
        model_version=model_version,
        background_variant="spatial_poisson/gaussian_kde_bw75km",
        etas_status="not_evaluable",
        etas_reason="failed_numerical_stability_no_qualified_parameters",
        horizons=tuple(
            HorizonProtocolDisplay(
                horizon,
                "confirmatory"
                if horizon in PRIMARY_MACRO_HORIZONS_DAYS
                else "evidence_insufficient_no_random_split",
            )
            for horizon in STAGE4_HORIZONS_DAYS
        ),
    )


@dataclass(frozen=True, slots=True)
class RetrospectiveHorizonResult:
    horizon_days: int
    magnitude_bin: MagnitudeBin
    result_status: ResultStatus
    independent_event_count: int | None
    information_gain_nats_per_event: float | None
    information_gain_lower_95: float | None
    information_gain_upper_95: float | None
    strict_recall_at_600000_km2: float | None

    def __post_init__(self) -> None:
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("retrospective horizon is outside the frozen five horizons")
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("magnitude_bin is outside the frozen bins")
        if self.independent_event_count is not None and (
            isinstance(self.independent_event_count, bool) or self.independent_event_count < 0
        ):
            raise ValueError("independent_event_count must be non-negative")
        _finite_optional(
            self.information_gain_nats_per_event,
            label="information_gain_nats_per_event",
        )
        _finite_optional(self.information_gain_lower_95, label="information_gain_lower_95")
        _finite_optional(self.information_gain_upper_95, label="information_gain_upper_95")
        _finite_optional(self.strict_recall_at_600000_km2, label="strict_recall_at_600000_km2")
        if self.strict_recall_at_600000_km2 is not None and not (
            0.0 <= self.strict_recall_at_600000_km2 <= 1.0
        ):
            raise ValueError("strict recall must be in [0, 1]")
        if self.horizon_days in {180, 365} and self.result_status != "evidence_insufficient":
            raise ValueError("180/365 day results must remain evidence insufficient")


@dataclass(frozen=True, slots=True)
class RetrospectiveTargetCoverage:
    """Aggregate target coverage, deliberately unavailable to prospective payloads."""

    horizon_days: int
    all_study_area_event_count: int
    strict_hit_count_at_600000_km2: int

    def __post_init__(self) -> None:
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("coverage horizon is outside the frozen five horizons")
        if self.all_study_area_event_count < 0 or self.strict_hit_count_at_600000_km2 < 0:
            raise ValueError("coverage counts must be non-negative")
        if self.strict_hit_count_at_600000_km2 > self.all_study_area_event_count:
            raise ValueError("strict hit count may not exceed study-area event count")


@dataclass(frozen=True, slots=True)
class RetrospectivePayload:
    protocol: ProtocolDisplay
    outcome_status: ResultStatus
    g2_status: GateStatus
    g3_status: GateStatus
    adoption: AdoptionChoice
    issue_dates: tuple[str, ...]
    horizon_results: tuple[RetrospectiveHorizonResult, ...]
    target_coverage: tuple[RetrospectiveTargetCoverage, ...]
    time_permutation_p_value: float | None
    space_permutation_p_value: float | None

    def __post_init__(self) -> None:
        if not self.issue_dates or any(
            not value or value != value.strip() for value in self.issue_dates
        ):
            raise ValueError("retrospective issue_dates must be non-empty trimmed strings")
        if len(set(self.issue_dates)) != len(self.issue_dates):
            raise ValueError("retrospective issue_dates must be unique")
        if not self.horizon_results:
            raise ValueError("retrospective horizon_results must not be empty")
        _finite_optional(self.time_permutation_p_value, label="time_permutation_p_value")
        _finite_optional(self.space_permutation_p_value, label="space_permutation_p_value")
        for value in (self.time_permutation_p_value, self.space_permutation_p_value):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("permutation p-values must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ProspectiveRelativeRank:
    issue_date: str
    magnitude_bin: MagnitudeBin
    horizon_days: int
    model_variant: ModelVariant
    rank_band: str
    relative_strength_index: float
    rank_percentile: float

    def __post_init__(self) -> None:
        if not self.issue_date or self.issue_date != self.issue_date.strip():
            raise ValueError("prospective issue_date must be a non-empty trimmed string")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("prospective horizon is outside the frozen five horizons")
        if not self.rank_band or self.rank_band != self.rank_band.strip():
            raise ValueError("rank_band must be a non-empty trimmed string")
        if not math.isfinite(self.relative_strength_index) or self.relative_strength_index < 0.0:
            raise ValueError("relative_strength_index must be finite and non-negative")
        if not math.isfinite(self.rank_percentile) or not 0.0 <= self.rank_percentile <= 100.0:
            raise ValueError("rank_percentile must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class ProspectivePayload:
    """Target-blind forecast payload whose temporal status is always explicit."""

    protocol: ProtocolDisplay
    forecast_status: ForecastStatus
    issue_dates: tuple[str, ...]
    relative_ranks: tuple[ProspectiveRelativeRank, ...]

    def __post_init__(self) -> None:
        if self.forecast_status not in {
            "genuine_prospective",
            "retrospective_generated_target_blind_shadow",
        }:
            raise ValueError("forecast_status is outside the explicit forecast semantics")
        if not self.issue_dates or any(
            not value or value != value.strip() for value in self.issue_dates
        ):
            raise ValueError("prospective issue_dates must be non-empty trimmed strings")
        if len(set(self.issue_dates)) != len(self.issue_dates):
            raise ValueError("prospective issue_dates must be unique")
        if any(item.issue_date not in self.issue_dates for item in self.relative_ranks):
            raise ValueError("every relative-rank row must use a declared issue date")
        assert_prospective_payload_is_target_blind(self)


def _iter_mapping_keys(value: object) -> tuple[str, ...]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        keys = [field.name for field in dataclasses.fields(value)]
        nested: list[str] = []
        for field in dataclasses.fields(value):
            nested.extend(_iter_mapping_keys(getattr(value, field.name)))
        return tuple(keys + nested)
    if isinstance(value, dict):
        nested = list(str(key) for key in value)
        for item in value.values():
            nested.extend(_iter_mapping_keys(item))
        return tuple(nested)
    if isinstance(value, tuple | list):
        return tuple(key for item in value for key in _iter_mapping_keys(item))
    return ()


def assert_prospective_payload_is_target_blind(payload: ProspectivePayload) -> None:
    for key in _iter_mapping_keys(payload):
        folded = key.casefold()
        if any(token in folded for token in _PROSPECTIVE_FORBIDDEN_KEY_PARTS):
            raise ValueError(f"prospective payload contains forbidden evaluation field: {key}")


def _as_json(value: object) -> str:
    serializable = (
        dataclasses.asdict(value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type)
        else value
    )
    serialized = json.dumps(
        serializable,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _assert_coordinate_free(serialized: str) -> None:
    folded = serialized.casefold()
    leaked = [token for token in _RESTRICTED_GEOMETRY_TOKENS if token in folded]
    if leaked:
        raise ValueError(f"restricted geometry field escaped into deliverable: {leaked[0]}")


def _assert_relative_intensity_terminology(serialized: str) -> None:
    folded = serialized.casefold()
    if "probability" in folded or "概率" in serialized:
        raise ValueError(
            "deliverables must use conditional relative intensity and rank terminology"
        )


_STATIC_PANEL_NAMES = (
    "data_and_model_flow",
    "coefficient_and_effect_curves",
    "distance_and_lead_decay",
    "foldwise_dynamic_vs_snapshot",
    "observed_vs_time_and_space_permutation",
    "information_gain_with_bootstrap_intervals",
    "region_by_horizon_increment_heatmap",
    "molchan_curve",
    "fixed_area_recall_curve",
)


def _scaled(
    value: float,
    lower: float,
    upper: float,
    output_lower: float,
    output_upper: float,
) -> float:
    if upper == lower:
        return 0.5 * (output_lower + output_upper)
    fraction = (value - lower) / (upper - lower)
    return output_lower + fraction * (output_upper - output_lower)


def _svg_path(points: tuple[tuple[float, float], ...]) -> str:
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}" for index, (x, y) in enumerate(points)
    )


def _series_class(variant: str) -> str:
    return {
        "background_no_increment": "series-background",
        "coverage_only": "series-coverage",
        "snapshot": "series-snapshot",
        "dynamic": "series-dynamic",
    }[variant]


def _diagnostic_panel(
    name: str,
    title: str,
    *,
    x: int,
    y: int,
    point_count: int,
    body: str,
    attributes: str = "",
) -> str:
    if name not in _STATIC_PANEL_NAMES or point_count <= 0:
        raise ValueError("diagnostic panels require a frozen name and real numerical points")
    return f"""<g data-panel="{name}" data-point-count="{point_count}" {attributes} transform="translate({x} {y})">
<rect class="panel" width="360" height="278" rx="8" />
<text class="panel-index" x="18" y="30">{_STATIC_PANEL_NAMES.index(name) + 1}</text>
<text class="subheading" x="48" y="30">{html.escape(title)}</text>
{body}
</g>"""


def _data_flow_body(values: DataMethodFlowDiagnostics) -> str:
    return f"""<text class="muted" x="20" y="56">起报期</text><text class="metric-small" x="20" y="78">{values.training_issue_count}</text>
<path class="flow" d="M82 72 H112"/><text class="muted" x="116" y="56">格-期行</text><text class="metric-small" x="116" y="78">{values.training_cell_count:,}</text>
<path class="flow" d="M190 72 H220"/><text class="muted" x="224" y="56">冻结特征</text><text class="metric-small" x="224" y="78">{values.fitted_feature_count}</text>
<rect class="node" x="20" y="104" width="98" height="44" rx="5"/><text x="69" y="123" text-anchor="middle">冻结背景</text><text class="value" x="69" y="140" text-anchor="middle">4 变体同域</text>
<path class="flow" d="M118 126 H138"/><rect class="node" x="138" y="104" width="98" height="44" rx="5"/><text x="187" y="123" text-anchor="middle">异常增量</text><text class="value" x="187" y="140" text-anchor="middle">{values.model_variant_count - 1} 个拟合层</text>
<path class="flow" d="M236 126 H256"/><rect class="node" x="256" y="104" width="84" height="44" rx="5"/><text x="298" y="123" text-anchor="middle">评分</text><text class="value" x="298" y="140" text-anchor="middle">独立事件</text>
<text class="muted" x="20" y="184">全区独立事件</text><text class="metric-small" x="142" y="184">{values.independent_event_count}</text>
<text class="muted" x="20" y="214">研究区面积</text><text class="metric-small" x="142" y="214">{values.study_area_km2:,.0f} km²</text>
<text class="value" x="20" y="246">背景质量 × exp(衰减 × 冻结效应) → 总补偿强度</text>"""


def _coefficient_effect_body(curves: tuple[CoefficientEffectCurve, ...]) -> str:
    all_x = tuple(value for curve in curves for value in curve.input_values)
    all_y = (0.0, *(value for curve in curves for value in curve.effect_values))
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    marks = [
        '<line class="axis" x1="38" y1="220" x2="338" y2="220"/>',
        '<line class="axis" x1="38" y1="70" x2="38" y2="220"/>',
    ]
    for index, curve in enumerate(curves):
        points = tuple(
            (
                _scaled(x_value, x_min, x_max, 44.0, 334.0),
                _scaled(y_value, y_min, y_max, 214.0, 76.0),
            )
            for x_value, y_value in zip(
                curve.input_values,
                curve.effect_values,
                strict=True,
            )
        )
        css_class = _series_class(curve.variant)
        marks.append(
            f'<path class="{css_class}" data-series="{curve.variant}" '
            f'data-coefficient="{html.escape(curve.coefficient_name)}" '
            f'data-point-count="{len(points)}" d="{_svg_path(points)}"/>'
        )
        marks.extend(
            f'<circle class="mark-{curve.variant}" cx="{x:.2f}" cy="{y:.2f}" r="3" '
            f'data-input="{input_value:.8g}" data-effect="{effect_value:.8g}"/>'
            for (x, y), input_value, effect_value in zip(
                points,
                curve.input_values,
                curve.effect_values,
                strict=True,
            )
        )
        if index < 3:
            marks.append(
                f'<text class="legend-{curve.variant}" x="{20 + index * 112}" y="55">'
                f"{html.escape(curve.variant)} β={curve.coefficient_estimate:.3f}</text>"
            )
    marks.append(f'<text class="muted" x="38" y="246">输入 {x_min:.2f} → {x_max:.2f}</text>')
    return "\n".join(marks)


def _distance_lead_body(values: DistanceLeadDecayDiagnostics) -> str:
    distance_points = tuple(
        (
            _scaled(value, values.distance_km[0], values.distance_km[-1], 30.0, 166.0),
            _scaled(weight, 0.0, 1.0, 214.0, 76.0),
        )
        for value, weight in zip(
            values.distance_km,
            values.spatial_relative_weight,
            strict=True,
        )
    )
    lead_points = tuple(
        (
            _scaled(value, values.lead_days[0], values.lead_days[-1], 194.0, 334.0),
            _scaled(weight, 0.0, 1.0, 214.0, 76.0),
        )
        for value, weight in zip(
            values.lead_days,
            values.temporal_relative_weight,
            strict=True,
        )
    )
    circles = [
        f'<circle class="mark-distance" cx="{x:.2f}" cy="{y:.2f}" r="3" '
        f'data-distance-km="{value:.8g}" data-weight="{weight:.8g}"/>'
        for (x, y), value, weight in zip(
            distance_points,
            values.distance_km,
            values.spatial_relative_weight,
            strict=True,
        )
    ]
    circles.extend(
        f'<circle class="mark-lead" cx="{x:.2f}" cy="{y:.2f}" r="3" '
        f'data-lead-days="{value:.8g}" data-weight="{weight:.8g}"/>'
        for (x, y), value, weight in zip(
            lead_points,
            values.lead_days,
            values.temporal_relative_weight,
            strict=True,
        )
    )
    return "\n".join(
        [
            '<text class="legend-distance" x="30" y="55">空间权重</text>',
            '<text class="legend-lead" x="194" y="55">提前期权重</text>',
            '<line class="axis" x1="30" y1="220" x2="166" y2="220"/>',
            '<line class="axis" x1="194" y1="220" x2="334" y2="220"/>',
            f'<path class="series-distance" data-point-count="{len(distance_points)}" d="{_svg_path(distance_points)}"/>',
            f'<path class="series-lead" data-point-count="{len(lead_points)}" d="{_svg_path(lead_points)}"/>',
            *circles,
            f'<text class="muted" x="30" y="246">0–{values.distance_km[-1]:g} km</text>',
            f'<text class="muted" x="194" y="246">0–{values.lead_days[-1]:g} 天</text>',
        ]
    )


def _fold_body(values: tuple[FoldMacroValue, ...]) -> str:
    numeric = (
        0.0,
        *(
            value
            for item in values
            for value in (
                item.dynamic_macro_information_gain,
                item.snapshot_macro_information_gain,
            )
            if value is not None
        ),
    )
    lower, upper = min(numeric), max(numeric)
    marks = [
        '<line class="axis" x1="38" y1="220" x2="338" y2="220"/>',
        f'<line class="zero" x1="38" y1="{_scaled(0.0, lower, upper, 214.0, 76.0):.2f}" x2="338" y2="{_scaled(0.0, lower, upper, 214.0, 76.0):.2f}"/>',
        '<text class="legend-dynamic" x="38" y="55">dynamic ●</text>',
        '<text class="legend-snapshot" x="178" y="55">snapshot ■</text>',
    ]
    for index, item in enumerate(values):
        x = 82 + index * 104
        counts_text = "/".join(str(value) for value in item.primary_horizon_event_counts)
        if (
            item.dynamic_macro_information_gain is None
            or item.snapshot_macro_information_gain is None
        ):
            marks.extend(
                (
                    f'<path class="insufficient" d="M {x - 5} 132 L {x + 5} 142 M {x + 5} 132 L {x - 5} 142" data-evidence="{item.evidence_status}"/>',
                    f'<text class="value" x="{x}" y="242" text-anchor="middle">fold {item.fold_index}</text>',
                    f'<text class="muted" x="{x}" y="258" text-anchor="middle">n={counts_text} · 证据不足</text>',
                )
            )
            continue
        dynamic_y = _scaled(item.dynamic_macro_information_gain, lower, upper, 214.0, 76.0)
        snapshot_y = _scaled(item.snapshot_macro_information_gain, lower, upper, 214.0, 76.0)
        marks.extend(
            (
                f'<line class="pair" x1="{x - 9}" y1="{snapshot_y:.2f}" x2="{x + 9}" y2="{dynamic_y:.2f}"/>',
                f'<rect class="mark-snapshot" x="{x - 13}" y="{snapshot_y - 4:.2f}" width="8" height="8" data-value="{item.snapshot_macro_information_gain:.8g}"/>',
                f'<circle class="mark-dynamic" cx="{x + 9}" cy="{dynamic_y:.2f}" r="4" data-value="{item.dynamic_macro_information_gain:.8g}"/>',
                f'<text class="value" x="{x}" y="242" text-anchor="middle">fold {item.fold_index}</text>',
                f'<text class="muted" x="{x}" y="258" text-anchor="middle">{item.snapshot_macro_information_gain:.3f}→{item.dynamic_macro_information_gain:.3f} · n={counts_text}</text>',
            )
        )
    return "\n".join(marks)


def _histogram(values: tuple[float, ...], lower: float, upper: float, bins: int) -> tuple[int, ...]:
    counts = [0] * bins
    span = upper - lower
    for value in values:
        if value == math.inf:
            counts[-1] += 1
        elif span == 0.0:
            counts[bins // 2] += 1
        else:
            index = min(bins - 1, max(0, int((value - lower) / span * bins)))
            counts[index] += 1
    return tuple(counts)


def _permutation_body(values: tuple[PermutationDistribution, ...]) -> str:
    finite = tuple(
        value
        for distribution in values
        for value in distribution.null_statistics
        if math.isfinite(value)
    )
    observed = tuple(
        item.observed_statistic for item in values if item.observed_statistic is not None
    )
    numeric = (*finite, *observed)
    lower, upper = (min(numeric), max(numeric)) if numeric else (0.0, 1.0)
    marks: list[str] = []
    band_height = 47.0
    for series_index, distribution in enumerate(values):
        top = 58.0 + series_index * 66.0
        baseline = top + band_height
        if distribution.observed_statistic is None:
            marks.extend(
                (
                    f'<path class="insufficient" d="M 178 {top + 14:.2f} L 190 {top + 26:.2f} M 190 {top + 14:.2f} L 178 {top + 26:.2f}" data-evidence="{distribution.evidence_status}"/>',
                    f'<text class="muted" x="20" y="{top - 8:.2f}">{distribution.variant}-{distribution.kind} · 证据不足 · {distribution.evidence_status}</text>',
                )
            )
            continue
        counts = _histogram(distribution.null_statistics, lower, upper, 14)
        maximum = max(counts)
        for index, count in enumerate(counts):
            height = 0.0 if maximum == 0 else band_height * count / maximum
            marks.append(
                f'<rect class="hist-{distribution.variant}-{distribution.kind}" x="{36 + index * 21}" '
                f'y="{baseline - height:.2f}" width="18" height="{height:.2f}" data-count="{count}"/>'
            )
        observed_x = _scaled(distribution.observed_statistic, lower, upper, 36.0, 330.0)
        marks.extend(
            (
                f'<line class="observed" x1="{observed_x:.2f}" y1="{top - 4:.2f}" x2="{observed_x:.2f}" y2="{baseline + 2:.2f}" data-observed="{distribution.observed_statistic:.8g}"/>',
                f'<text class="muted" x="20" y="{top - 8:.2f}">{distribution.variant}-{distribution.kind} · n={len(distribution.null_statistics)} · p={distribution.p_value:.4f} · +∞={distribution.scientific_failure_count}</text>',
            )
        )
    return "\n".join(marks)


def _information_gain_body(values: tuple[InformationGainInterval, ...]) -> str:
    evaluated = tuple(item for item in values if item.point_estimate is not None)
    bounds = (
        0.0,
        *(item.lower_95 for item in evaluated if item.lower_95 is not None),
        *(item.upper_95 for item in evaluated if item.upper_95 is not None),
    )
    lower, upper = min(bounds), max(bounds)
    zero_y = _scaled(0.0, lower, upper, 205.0, 70.0)
    marks = [f'<line class="zero" x1="34" y1="{zero_y:.2f}" x2="338" y2="{zero_y:.2f}"/>']
    for index, item in enumerate(values):
        x = 48 + index * 30
        label = f"{'5' if item.magnitude_bin == 'M5_6' else '6+'}/{item.horizon_days}"
        if item.point_estimate is None or item.lower_95 is None or item.upper_95 is None:
            marks.extend(
                (
                    f'<path class="insufficient" d="M {x - 4} 138 L {x + 4} 146 M {x + 4} 138 L {x - 4} 146" data-evidence="{item.evidence_status}"/>',
                    f'<text class="micro" x="{x}" y="164" text-anchor="middle">n={item.independent_event_count}</text>',
                )
            )
        else:
            point_y = _scaled(item.point_estimate, lower, upper, 205.0, 70.0)
            low_y = _scaled(item.lower_95, lower, upper, 205.0, 70.0)
            high_y = _scaled(item.upper_95, lower, upper, 205.0, 70.0)
            exploratory = item.evidence_status.startswith("exploratory_low_sample")
            mark = (
                f'<rect class="mark-exploratory" x="{x - 3:.2f}" y="{point_y - 3:.2f}" width="6" height="6" transform="rotate(45 {x} {point_y:.2f})" data-point="{item.point_estimate:.8g}" data-evidence="{item.evidence_status}"/>'
                if exploratory
                else f'<circle class="mark-primary" cx="{x}" cy="{point_y:.2f}" r="4" data-point="{item.point_estimate:.8g}" data-evidence="{item.evidence_status}"/>'
            )
            marks.extend(
                (
                    f'<line class="interval" x1="{x}" y1="{high_y:.2f}" x2="{x}" y2="{low_y:.2f}" data-low="{item.lower_95:.8g}" data-high="{item.upper_95:.8g}" data-evidence="{item.evidence_status}"/>',
                    mark,
                    f'<text class="micro" x="{x}" y="{point_y - 7:.2f}" text-anchor="middle">{item.point_estimate:.2f}</text>',
                    f'<text class="micro" x="{x}" y="220" text-anchor="middle">n={item.independent_event_count}{"探" if exploratory else ""}</text>',
                )
            )
        marks.append(f'<text class="micro" x="{x}" y="238" text-anchor="middle">{label}</text>')
    marks.append(
        '<text class="muted" x="20" y="258">nats/独立事件；×=证据不足；◇/探=M6+探索性 n&lt;10（长窗同时无随机切分）</text>'
    )
    return "\n".join(marks)


def _region_heatmap_body(
    region_ids: tuple[str, ...],
    metrics: tuple[RegionHorizonMetric, ...],
) -> str:
    by_key = {(item.region_id, item.horizon_days): item for item in metrics}
    finite = tuple(
        item.information_gain_nats_per_event
        for item in metrics
        if item.information_gain_nats_per_event is not None
    )
    scale = max((abs(value) for value in finite), default=1.0)
    block_count = 3
    rows_per_block = math.ceil(len(region_ids) / block_count)
    row_height = min(12.0, 174.0 / rows_per_block)
    marks = ['<text class="muted" x="20" y="52">IG 主色；标题含支持事件/全区事件与严格召回</text>']
    for region_index, region_id in enumerate(region_ids):
        block = region_index // rows_per_block
        row = region_index % rows_per_block
        origin_x = 18.0 + block * 114.0
        y = 64.0 + row * row_height
        alias = f"R{region_index + 1:02d}"
        marks.append(
            f'<text class="micro" x="{origin_x:.2f}" y="{y + row_height * 0.75:.2f}">{alias}</text>'
        )
        for horizon_index, horizon in enumerate(STAGE4_HORIZONS_DAYS):
            metric = by_key[(region_id, horizon)]
            value = metric.information_gain_nats_per_event
            css_class = (
                "heat-missing"
                if value is None
                else ("heat-positive" if value >= 0.0 else "heat-negative")
            )
            opacity = 1.0 if value is None else 0.2 + 0.8 * abs(value) / scale
            recall_text = "None" if metric.strict_recall is None else f"{metric.strict_recall:.4f}"
            value_text = "None" if value is None else f"{value:.6f}"
            x = origin_x + 25.0 + horizon_index * 16.0
            marks.append(
                f'<rect class="{css_class}" x="{x:.2f}" y="{y:.2f}" width="14" height="{max(2.0, row_height - 1.0):.2f}" '
                f'style="fill-opacity:{opacity:.4f}" data-region-alias="{alias}" data-region-id="{html.escape(region_id)}" '
                f'data-horizon="{horizon}" data-information-gain="{value_text}" data-supported-event-count="{metric.supported_event_count}" '
                f'data-all-study-area-event-count="{metric.all_study_area_event_count}" data-strict-recall="{recall_text}" '
                f'data-information-gain-evidence="{metric.information_gain_evidence_status}" '
                f'data-strict-recall-evidence="{metric.strict_recall_evidence_status}"><title>{alias}={html.escape(region_id)} · {horizon}天 · IG={value_text} · n支持={metric.supported_event_count} · n全区={metric.all_study_area_event_count} · recall={recall_text} · IG证据={metric.information_gain_evidence_status} · 召回证据={metric.strict_recall_evidence_status}</title></rect>'
            )
    marks.append(
        '<text class="micro" x="20" y="258">列：7 · 30 · 90 · 180 · 365 天；别名按输入顺序确定</text>'
    )
    return "\n".join(marks)


def _budget_curve_body(
    curves: tuple[AlarmBudgetRecallCurve, ...],
    *,
    study_area_km2: float,
    molchan: bool,
    y_top: float,
    y_bottom: float,
) -> str:
    selected_areas = tuple(
        point.selected_alarm_area_km2 for curve in curves for point in curve.points
    )
    x_values = (
        tuple(value / study_area_km2 for value in selected_areas) if molchan else selected_areas
    )
    x_min, x_max = min(x_values), max(x_values)
    marks = [
        f'<line class="axis" x1="42" y1="{y_bottom}" x2="338" y2="{y_bottom}"/>',
        f'<line class="axis" x1="42" y1="{y_top}" x2="42" y2="{y_bottom}"/>',
    ]
    for curve_index, curve in enumerate(curves):
        if all(point.strict_recall is None for point in curve.points):
            evidence_y = 0.5 * (y_top + y_bottom)
            for point in curve.points:
                x = _scaled(
                    point.selected_alarm_area_km2 / study_area_km2
                    if molchan
                    else point.selected_alarm_area_km2,
                    x_min,
                    x_max,
                    46.0,
                    334.0,
                )
                marks.append(
                    f'<path class="insufficient" d="M {x - 3:.2f} {evidence_y - 3:.2f} L {x + 3:.2f} {evidence_y + 3:.2f} M {x + 3:.2f} {evidence_y - 3:.2f} L {x - 3:.2f} {evidence_y + 3:.2f}" '
                    f'data-variant="{curve.variant}" data-horizon="{curve.horizon_days}" data-budget-km2="{point.budget_km2:.8g}" '
                    f'data-selected-area-km2="{point.selected_alarm_area_km2:.8g}" data-all-study-area-event-count="{point.all_study_area_event_count}" data-evidence="{point.evidence_status}"/>'
                )
            marks.append(
                f'<text class="micro" x="{20 + (curve_index % 3) * 112}" y="{52 + (curve_index // 3) * 13}">{curve.variant[:3]}-{curve.horizon_days}d · 证据不足</text>'
            )
            continue
        if any(point.strict_recall is None for point in curve.points):
            raise AssertionError("alarm-budget curve mixed evaluated and missing recall")
        strict_recalls = tuple(
            point.strict_recall for point in curve.points if point.strict_recall is not None
        )
        numeric_points: list[tuple[float, float]] = []
        for point in curve.points:
            if point.strict_recall is None:
                raise AssertionError("evaluated alarm-budget point lost strict recall")
            numeric_points.append(
                (
                    _scaled(
                        point.selected_alarm_area_km2 / study_area_km2
                        if molchan
                        else point.selected_alarm_area_km2,
                        x_min,
                        x_max,
                        46.0,
                        334.0,
                    ),
                    _scaled(
                        1.0 - point.strict_recall if molchan else point.strict_recall,
                        0.0,
                        1.0,
                        y_bottom,
                        y_top,
                    ),
                )
            )
        points = tuple(numeric_points)
        css_class = _series_class(curve.variant)
        marks.append(
            f'<path class="{css_class}" data-variant="{curve.variant}" data-horizon="{curve.horizon_days}" '
            f'data-point-count="{len(points)}" d="{_svg_path(points)}"/>'
        )
        marks.extend(
            f'<circle class="mark-{curve.variant}" cx="{x:.2f}" cy="{y:.2f}" r="2.5" '
            f'data-budget-km2="{point.budget_km2:.8g}" data-selected-area-km2="{point.selected_alarm_area_km2:.8g}" '
            f'data-strict-recall="{strict_recall:.8g}" data-all-study-area-event-count="{point.all_study_area_event_count}" data-evidence="{point.evidence_status}"/>'
            for (x, y), point, strict_recall in zip(
                points,
                curve.points,
                strict_recalls,
                strict=True,
            )
        )
        if curve_index < 6:
            marks.append(
                f'<text class="micro legend-{curve.variant}" x="{20 + (curve_index % 3) * 112}" y="{52 + (curve_index // 3) * 13}">{curve.variant[:3]}-{curve.horizon_days}d</text>'
            )
    return "\n".join(marks)


def _same_recall_marks(values: tuple[SameRecallAreaReduction, ...]) -> str:
    marks: list[str] = []
    for index, item in enumerate(values):
        y = 192 + index * 22
        if (
            item.area_reduction_fraction is None
            or item.area_reduction_lower_95 is None
            or item.area_reduction_upper_95 is None
        ):
            target = "—" if item.target_recall is None else f"{item.target_recall:.2f}"
            marks.append(
                f'<text class="micro" x="20" y="{y}" data-evidence="{item.evidence_status}">{item.horizon_days}天 · recall={target} · 证据不足 · {item.evidence_status}</text>'
            )
            continue
        if item.comparator_area_km2 is None or item.candidate_area_km2 is None:
            raise AssertionError("evaluated same-recall diagnostics lost their areas")
        marks.append(
            f'<text class="micro" x="20" y="{y}" data-evidence="{item.evidence_status}">{item.horizon_days}天 · recall={item.target_recall:.2f} · '
            f"{item.comparator_area_km2 / 1000:.0f}k→{item.candidate_area_km2 / 1000:.0f}k km² · "
            f"缩减={item.area_reduction_fraction:.1%} [{item.area_reduction_lower_95:.1%},{item.area_reduction_upper_95:.1%}]</text>"
        )
    return "\n".join(marks)


def build_static_results_svg(
    payload: RetrospectivePayload,
    diagnostics: PublicationDiagnostics,
) -> str:
    """Render all nine frozen panels exclusively from complete numerical diagnostics."""

    if not isinstance(diagnostics, PublicationDiagnostics):
        raise TypeError("complete PublicationDiagnostics are required for formal publication")
    curve_points = sum(len(item.points) for item in diagnostics.alarm_budget_curves)
    panels = (
        _diagnostic_panel(
            "data_and_model_flow",
            "数据与方法流程",
            x=36,
            y=100,
            point_count=5,
            body=_data_flow_body(diagnostics.data_flow),
        ),
        _diagnostic_panel(
            "coefficient_and_effect_curves",
            "冻结系数与效应曲线",
            x=420,
            y=100,
            point_count=sum(len(item.input_values) for item in diagnostics.coefficient_effects),
            body=_coefficient_effect_body(diagnostics.coefficient_effects),
            attributes=f'data-curve-count="{len(diagnostics.coefficient_effects)}"',
        ),
        _diagnostic_panel(
            "distance_and_lead_decay",
            "距离与提前期衰减",
            x=804,
            y=100,
            point_count=len(diagnostics.distance_lead_decay.distance_km)
            + len(diagnostics.distance_lead_decay.lead_days),
            body=_distance_lead_body(diagnostics.distance_lead_decay),
        ),
        _diagnostic_panel(
            "foldwise_dynamic_vs_snapshot",
            "三开发折 dynamic 对 snapshot",
            x=36,
            y=400,
            point_count=sum(
                2 if item.evidence_status == "evaluated" else 1
                for item in diagnostics.fold_macro_values
            ),
            body=_fold_body(diagnostics.fold_macro_values),
            attributes='data-fold-count="3"',
        ),
        _diagnostic_panel(
            "observed_vs_time_and_space_permutation",
            "观测量与真实置乱分布",
            x=420,
            y=400,
            point_count=sum(
                len(item.null_statistics) + 1 for item in diagnostics.permutation_distributions
            ),
            body=_permutation_body(diagnostics.permutation_distributions),
            attributes=" ".join(
                f'data-{item.variant}-{item.kind}-null-count="{len(item.null_statistics)}"'
                for item in diagnostics.permutation_distributions
            ),
        ),
        _diagnostic_panel(
            "information_gain_with_bootstrap_intervals",
            "信息增益与 Bootstrap 区间",
            x=804,
            y=400,
            point_count=len(diagnostics.information_gain_intervals),
            body=_information_gain_body(diagnostics.information_gain_intervals),
            attributes='data-magnitude-bin-count="2" data-horizon-count="5"',
        ),
        _diagnostic_panel(
            "region_by_horizon_increment_heatmap",
            "构造子区 × 窗口诊断",
            x=36,
            y=700,
            point_count=len(diagnostics.region_horizon_metrics),
            body=_region_heatmap_body(
                diagnostics.region_ids,
                diagnostics.region_horizon_metrics,
            ),
            attributes=f'data-region-count="{len(diagnostics.region_ids)}" data-horizon-count="5"',
        ),
        _diagnostic_panel(
            "molchan_curve",
            "Molchan 漏报—面积分数曲线",
            x=420,
            y=700,
            point_count=curve_points,
            body=_budget_curve_body(
                diagnostics.alarm_budget_curves,
                study_area_km2=diagnostics.data_flow.study_area_km2,
                molchan=True,
                y_top=78.0,
                y_bottom=225.0,
            )
            + '<text class="muted" x="180" y="252">实际选中面积 / 研究区面积 →</text>',
            attributes=f'data-curve-count="{len(diagnostics.alarm_budget_curves)}"',
        ),
        _diagnostic_panel(
            "fixed_area_recall_curve",
            "多预算召回与同召回面积缩减",
            x=804,
            y=700,
            point_count=curve_points + len(diagnostics.same_recall_area_reductions),
            body=_budget_curve_body(
                diagnostics.alarm_budget_curves,
                study_area_km2=diagnostics.data_flow.study_area_km2,
                molchan=False,
                y_top=76.0,
                y_bottom=166.0,
            )
            + _same_recall_marks(diagnostics.same_recall_area_reductions),
            attributes=f'data-budget-count="5" data-same-recall-count="{len(diagnostics.same_recall_area_reductions)}"',
        ),
    )
    if (
        tuple(
            re.search(r'data-panel="([^"]+)"', panel).group(1)  # type: ignore[union-attr]
            for panel in panels
        )
        != _STATIC_PANEL_NAMES
    ):
        raise AssertionError("static nine-panel order changed")

    status_text = html.escape(payload.outcome_status)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1040" viewBox="0 0 1200 1040" role="img" aria-labelledby="title description" data-template="stage4-static-nine-panel-v2" data-diagnostics-sha256="{diagnostics.content_sha256}">
<title id="title">SeismoFlux 阶段4九联诊断</title>
<desc id="description">九个面板由完整数值诊断绘制：流程、冻结效应、衰减、三折宏观值、置乱分布、信息增益、全部构造子区、多面积Molchan和同召回面积缩减。</desc>
<style>
svg {{ --bg:#fff; --fg:#172033; --muted:#5f6b7a; --line:#cbd2dc; --panel:#f4f6f9; --node:#fff; --background:#747f8d; --coverage:#3e8d78; --snapshot:#7f4aa8; --dynamic:#2469a6; --negative:#b34a45; --missing:#aab1bc; background:var(--bg); color:var(--fg); font-family:system-ui,sans-serif; }}
@media (prefers-color-scheme:dark) {{ svg {{ --bg:#111723; --fg:#edf1f7; --muted:#b3bdca; --line:#445064; --panel:#1b2433; --node:#111723; --background:#aeb7c4; --coverage:#72c5ad; --snapshot:#c59bea; --dynamic:#75b8ee; --negative:#ee8a84; --missing:#596679; }} }}
text {{ fill:var(--fg); font-weight:400; font-size:13px; }} .heading {{ font-size:24px; font-weight:500; }} .subheading {{ font-size:15px; font-weight:500; }} .muted {{ fill:var(--muted); font-size:11px; }} .value {{ font-size:11px; font-weight:500; }} .micro {{ fill:var(--muted); font-size:8px; }} .metric-small {{ font-size:18px; font-weight:500; }}
.panel {{ fill:var(--panel); stroke:var(--line); stroke-width:1; }} .panel-index {{ fill:var(--muted); font-size:12px; }} .node {{ fill:var(--node); stroke:var(--line); }} .flow,.pair {{ fill:none; stroke:var(--line); stroke-width:1.5; }} .axis,.zero {{ stroke:var(--line); stroke-width:1; }} .zero {{ stroke-dasharray:3 3; }} .interval {{ stroke:var(--fg); stroke-width:2; }} .insufficient {{ stroke:var(--muted); stroke-width:1.5; }}
.series-background,.series-coverage,.series-snapshot,.series-dynamic,.series-distance,.series-lead {{ fill:none; stroke-width:1.8; }} .series-background {{ stroke:var(--background); }} .series-coverage {{ stroke:var(--coverage); }} .series-snapshot {{ stroke:var(--snapshot); }} .series-dynamic,.series-distance {{ stroke:var(--dynamic); }} .series-lead {{ stroke:var(--snapshot); stroke-dasharray:4 3; }}
.mark-background {{ fill:var(--background); }} .mark-coverage {{ fill:var(--coverage); }} .mark-snapshot,.mark-exploratory {{ fill:var(--snapshot); }} .mark-dynamic,.mark-distance,.mark-primary {{ fill:var(--dynamic); }} .mark-lead {{ fill:var(--snapshot); }} .legend-background {{ fill:var(--background); font-size:8px; }} .legend-coverage {{ fill:var(--coverage); font-size:8px; }} .legend-snapshot {{ fill:var(--snapshot); font-size:8px; }} .legend-dynamic,.legend-distance {{ fill:var(--dynamic); font-size:8px; }} .legend-lead {{ fill:var(--snapshot); font-size:8px; }}
.hist-dynamic-time {{ fill:var(--dynamic); opacity:.65; }} .hist-dynamic-space {{ fill:var(--coverage); opacity:.65; }} .hist-snapshot-time,.hist-snapshot-space {{ fill:var(--snapshot); opacity:.65; }} .observed {{ stroke:var(--negative); stroke-width:2; }} .heat-positive {{ fill:var(--dynamic); }} .heat-negative {{ fill:var(--negative); }} .heat-missing {{ fill:var(--missing); }}
</style>
<rect width="1200" height="1040" fill="var(--bg)"/>
<text class="heading" x="36" y="44">阶段4：真实数值驱动的九联方法诊断</text>
<text class="muted" x="36" y="72">条件相对强度与顺位 · 总体 {status_text} · 模型 {html.escape(payload.protocol.model_version)} · 诊断 {diagnostics.content_sha256[:12]}</text>
{"".join(panels)}
<text class="muted" x="36" y="996">ETAS 对照：{html.escape(payload.protocol.etas_status)} · {html.escape(payload.protocol.etas_reason)}</text>
<text class="muted" x="36" y="1018">所有点均来自传入诊断；静态公共件仅含聚合数值和稳定区域标识，不含受限几何或单格映射。</text>
</svg>"""
    _assert_coordinate_free(svg)
    _assert_relative_intensity_terminology(svg)
    return svg


def build_interactive_results_html(
    retrospective: RetrospectivePayload,
    prospective: ProspectivePayload,
    diagnostics: PublicationDiagnostics,
) -> str:
    """Build a self-contained, target-safe retrospective/prospective explorer."""

    if retrospective.protocol != prospective.protocol:
        raise ValueError("retrospective and prospective payloads must share one protocol identity")
    if not isinstance(diagnostics, PublicationDiagnostics):
        raise TypeError("complete PublicationDiagnostics are required for formal publication")
    assert_prospective_payload_is_target_blind(prospective)
    static_svg = build_static_results_svg(retrospective, diagnostics)
    retro_json = _as_json(
        {
            "diagnostics": diagnostics.as_mapping(),
            "summary": dataclasses.asdict(retrospective),
        }
    )
    prospective_json = _as_json(prospective)
    _assert_coordinate_free(retro_json)
    _assert_coordinate_free(prospective_json)
    forecast_label = (
        "真正前瞻预测"
        if prospective.forecast_status == "genuine_prospective"
        else "回溯生成的目标盲影子预测"
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>SeismoFlux 阶段4方法效果</title>
<style>
:root {{ color-scheme:light dark; --bg:#f6f7fa; --fg:#172033; --muted:#5c6877; --card:#fff; --line:#ccd3dd; --accent:#2469a6; --positive:#267a55; --negative:#b34a45; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#111723; --fg:#edf1f7; --muted:#b3bdca; --card:#1b2433; --line:#445064; --accent:#75b8ee; --positive:#64c795; --negative:#ee8a84; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:auto; padding:24px; }} h1 {{ margin:0 0 6px; font-size:clamp(1.35rem,3vw,2rem); font-weight:500; }} h2 {{ font-size:1.05rem; font-weight:500; margin:0 0 14px; }}
.muted {{ color:var(--muted); }} .controls {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:22px 0; }} label {{ display:grid; gap:5px; color:var(--muted); font-size:.88rem; }} select {{ width:100%; padding:9px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(235px,1fr)); gap:14px; }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; }} .value {{ font-size:1.4rem; font-weight:500; margin-top:6px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-weight:500; }} .table-wrap {{ overflow-x:auto; }}
.bar-track {{ height:18px; background:color-mix(in srgb,var(--line) 45%,transparent); border-radius:4px; overflow:hidden; }} .bar {{ height:100%; background:var(--accent); min-width:2px; }}
.status-passed {{ color:var(--positive); }} .status-failed {{ color:var(--negative); }} .status-evidence_insufficient {{ color:var(--muted); }} [hidden] {{ display:none !important; }}
.protocol {{ margin-top:14px; }} .horizon-line {{ display:flex; flex-wrap:wrap; gap:10px 18px; color:var(--muted); font-size:.88rem; }}
.diagnostic-svg {{ margin-top:14px; }} .diagnostic-svg svg {{ display:block; width:100%; height:auto; }}
@media (max-width:540px) {{ main {{ padding:16px; }} th,td {{ padding:8px 5px; font-size:.88rem; }} }}
</style>
</head>
<body>
<main id="stage4-explorer">
<h1>异常增量方法效果</h1>
<div class="muted">条件相对强度与顺位；回溯检验与目标盲预测载荷物理分离；当前状态：{forecast_label}</div>
<div class="controls" aria-label="展示筛选">
<label>模式<select id="mode"><option value="prospective" selected>{forecast_label}</option><option value="retrospective">回溯检验</option></select></label>
<label>起报日<select id="issueDate"></select></label>
<label>震级层<select id="magnitude"><option value="M5_6">M5–6</option><option value="M6_plus">M6+</option></select></label>
<label>窗口<select id="horizon"><option value="7">7天</option><option value="30">30天</option><option value="90">90天</option><option value="180">180天</option><option value="365">365天</option></select></label>
<label>模型<select id="variant"><option value="background_no_increment">冻结背景</option><option value="snapshot">快照增量</option><option value="dynamic">动态增量</option></select></label>
<label>版本<select id="version"><option value="{html.escape(prospective.protocol.model_version)}" selected>{html.escape(prospective.protocol.model_version)}</option></select></label>
</div>
<section id="retrospectivePanel" hidden aria-labelledby="retroTitle">
<h2 id="retroTitle">回溯：独立物理事件与对照检验</h2>
<div class="grid" id="retroSummary"></div>
<div class="card table-wrap" style="margin-top:14px"><table><thead><tr><th>窗口</th><th>事件数</th><th>信息增益</th><th>95%区间</th><th>600,000 km²严格召回</th><th>状态</th></tr></thead><tbody id="retroRows"></tbody></table></div>
<section class="card" id="retrospective-target-coverage" style="margin-top:14px" aria-labelledby="coverageTitle"><h2 id="coverageTitle">回溯目标覆盖（仅回溯模式）</h2><div id="coverageRows" class="horizon-line"></div></section>
<div id="retrospectiveDiagnostics" class="diagnostic-svg" aria-label="回溯九联数值诊断">{static_svg}</div>
</section>
<section id="prospectivePanel" aria-labelledby="prosTitle">
<h2 id="prosTitle">{forecast_label}：条件相对强度与顺位</h2>
<div class="card"><div id="prospectiveDetail" class="muted"></div><div id="rankRows" style="margin-top:14px"></div></div>
</section>
<section class="card protocol" aria-labelledby="protocolTitle"><h2 id="protocolTitle">固定方法边界</h2><div id="protocolRows" class="horizon-line"></div><p class="muted">ETAS：not_evaluable（failed_numerical_stability_no_qualified_parameters）</p></section>
</main>
<script type="application/json" id="prospective-payload">{prospective_json}</script>
<script type="application/json" id="retrospective-payload">{retro_json}</script>
<script>
(() => {{
  "use strict";
  const pros = JSON.parse(document.getElementById("prospective-payload").textContent);
  let retroCache = null;
  function retrospectivePayload() {{
    if (retroCache === null) retroCache = JSON.parse(document.getElementById("retrospective-payload").textContent);
    return retroCache;
  }}
  const controls = {{ mode:document.getElementById("mode"), issue:document.getElementById("issueDate"), magnitude:document.getElementById("magnitude"), horizon:document.getElementById("horizon"), variant:document.getElementById("variant"), version:document.getElementById("version") }};
  const fmt = value => value === null || value === undefined ? "—" : Number(value).toFixed(3);
  function setIssues(values) {{
    const chosen = controls.issue.value;
    controls.issue.replaceChildren(...values.map(value => {{ const option=document.createElement("option"); option.value=value; option.textContent=value; return option; }}));
    if (values.includes(chosen)) controls.issue.value=chosen;
  }}
  function summaryCard(label,value,status) {{ const card=document.createElement("div"); card.className="card"; const small=document.createElement("div"); small.className="muted"; small.textContent=label; const main=document.createElement("div"); main.className="value status-"+status; main.textContent=value; card.append(small,main); return card; }}
  function renderRetro() {{
    const retrospective=retrospectivePayload();
    const retro=retrospective.summary;
    setIssues(retro.issue_dates);
    const summary=document.getElementById("retroSummary"); summary.replaceChildren(summaryCard("总体",retro.outcome_status,retro.outcome_status),summaryCard("G2",retro.g2_status,retro.g2_status),summaryCard("G3",retro.g3_status,retro.g3_status));
    const rows=retro.horizon_results.filter(row => row.magnitude_bin===controls.magnitude.value && Number(row.horizon_days)===Number(controls.horizon.value));
    const body=document.getElementById("retroRows"); body.replaceChildren(...rows.map(row => {{ const tr=document.createElement("tr"); [row.horizon_days+"天",row.independent_event_count,fmt(row.information_gain_nats_per_event),fmt(row.information_gain_lower_95)+" ～ "+fmt(row.information_gain_upper_95),fmt(row.strict_recall_at_600000_km2),row.result_status].forEach(value => {{ const td=document.createElement("td"); td.textContent=value ?? "—"; tr.append(td); }}); return tr; }}));
    const coverage=document.getElementById("coverageRows"); const items=retro.target_coverage.filter(row => Number(row.horizon_days)===Number(controls.horizon.value)); coverage.replaceChildren(...items.map(row => {{ const span=document.createElement("span"); span.textContent=row.horizon_days+"天："+row.strict_hit_count_at_600000_km2+" / "+row.all_study_area_event_count; return span; }}));
  }}
  function renderPros() {{
    setIssues(pros.issue_dates);
    const rows=pros.relative_ranks.filter(row => row.issue_date===controls.issue.value && row.magnitude_bin===controls.magnitude.value && Number(row.horizon_days)===Number(controls.horizon.value) && row.model_variant===controls.variant.value);
    document.getElementById("prospectiveDetail").textContent=controls.issue.value+" · "+controls.horizon.value+"天 · "+controls.magnitude.value+" · "+controls.variant.value+" · "+controls.version.value;
    const holder=document.getElementById("rankRows"); holder.replaceChildren(...rows.map(row => {{ const wrap=document.createElement("div"); const label=document.createElement("div"); label.className="horizon-line"; label.textContent=row.rank_band+" · 顺位 "+Number(row.rank_percentile).toFixed(1)+" · 相对强度 "+Number(row.relative_strength_index).toFixed(3); const track=document.createElement("div"); track.className="bar-track"; const bar=document.createElement("div"); bar.className="bar"; bar.style.width=Math.max(0,Math.min(100,row.rank_percentile))+"%"; track.append(bar); wrap.append(label,track); return wrap; }}));
  }}
  function renderProtocol() {{ const holder=document.getElementById("protocolRows"); holder.replaceChildren(...pros.protocol.horizons.map(row => {{ const span=document.createElement("span"); span.textContent=row.horizon_days+"天："+row.evidence_status; return span; }})); }}
  function render() {{ const isRetro=controls.mode.value==="retrospective"; document.getElementById("retrospectivePanel").hidden=!isRetro; document.getElementById("prospectivePanel").hidden=isRetro; if (isRetro) renderRetro(); else renderPros(); renderProtocol(); }}
  Object.values(controls).forEach(control => control.addEventListener("change",render));
  render();
}})();
</script>
</body>
</html>"""
    _assert_coordinate_free(document)
    _assert_relative_intensity_terminology(document)
    return document


__all__ = [
    "ForecastStatus",
    "HorizonEvidenceStatus",
    "HorizonProtocolDisplay",
    "ModelVariant",
    "ProspectivePayload",
    "ProspectiveRelativeRank",
    "ProtocolDisplay",
    "ResultStatus",
    "RetrospectiveHorizonResult",
    "RetrospectivePayload",
    "RetrospectiveTargetCoverage",
    "assert_prospective_payload_is_target_blind",
    "build_interactive_results_html",
    "build_static_results_svg",
    "default_protocol_display",
]
