# ruff: noqa: E501, RUF001
"""Public-safe summaries and audit figure for the stage-3 local feature bundle.

The scientific Parquet files remain local.  This module accepts only aggregate
counts, cryptographic identities, and the frozen public feature dictionary; it has no
API for query-cell, station, anomaly, source-row, geometry, earthquake, or score data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import cast

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.publication import (
    FixedFilePublication,
    publish_fixed_project_file,
)
from seismoflux.features.anomaly.audit import AuditResult
from seismoflux.features.anomaly.dictionary import (
    FEATURE_SEMANTICS,
    FeatureDictionary,
    assert_public_safe_feature_dictionary,
)
from seismoflux.features.anomaly.publication import stage3_identity_sha256
from seismoflux.features.anomaly.synthetic_audit import SyntheticPrefixAuditResult

PUBLIC_REGISTRY_PATH = "data/manifests/anomaly_feature_registry.json"
PUBLIC_REPORT_PATH = "docs/anomaly_feature_report.md"
PUBLIC_AUDIT_SVG_PATH = "docs/anomaly_feature_audit.svg"
PUBLIC_DICTIONARY_PATH = "data/manifests/anomaly_feature_dictionary.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40,64}")
_EXPECTED_DATASETS = frozenset({"anomaly_state_history", "anomaly_feature_store"})
_EXPECTED_PAYLOAD_FILES = frozenset(
    {"anomaly_state_history.parquet", "anomaly_feature_store.parquet"}
)
_INPUT_HASH_KEYS = frozenset(
    {
        "protocol_bytes_sha256",
        "anomaly_history_config_bytes_sha256",
        "environment_lock_sha256",
        "data_catalog_sha256",
        "anomaly_observation_file_sha256",
        "anomaly_observation_content_sha256",
        "anomaly_observation_schema_sha256",
        "anomaly_report_period_file_sha256",
        "anomaly_report_period_content_sha256",
        "anomaly_report_period_schema_sha256",
        "study_area_sha256",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "execution_mode",
        "protocol_freeze_tag",
        "code_commit",
        "feature_dictionary_sha256",
        "grid",
        "input_hashes",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "anomaly_id",
        "cell_id",
        "cell_column",
        "cell_row",
        "clipped_geometry",
        "geometry",
        "latitude",
        "longitude",
        "query_x_m",
        "query_y_m",
        "source_file",
        "source_row",
        "source_sheet",
        "station_id",
        "wkb",
        "wkt",
    }
)


@dataclass(frozen=True, slots=True)
class Stage3AggregateSummary:
    """Aggregate-only facts required to decide the stage-3 acceptance gate."""

    snapshot_count: int
    first_report_date: date
    last_report_date: date
    known_missing_period_count: int
    state_row_count: int
    entity_state_row_count: int
    report_summary_row_count: int
    complete_entity_state_row_count: int
    temporary_entity_state_row_count: int
    reliability_high_state_row_count: int
    reliability_cautious_state_row_count: int
    reliability_excluded_state_row_count: int
    query_cell_count: int
    feature_row_count: int
    nullable_value_count: int
    null_value_count: int
    future_source_reference_count: int
    missing_period_zero_imputation_count: int
    source_duration_feature_use_count: int
    forbidden_source_field_use_count: int
    target_or_earthquake_label_read_count: int
    synthetic_prefix_seed_count: int
    synthetic_prefix_property_failures: int

    def __post_init__(self) -> None:
        if not isinstance(self.first_report_date, date) or not isinstance(
            self.last_report_date, date
        ):
            raise TypeError("stage-3 aggregate report dates must be date values")
        integer_values = (
            value
            for key, value in asdict(self).items()
            if key not in {"first_report_date", "last_report_date"}
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_values
        ):
            raise ValueError("stage-3 aggregate counts must be non-negative integers")
        if self.first_report_date > self.last_report_date:
            raise ValueError("stage-3 report-date range is reversed")
        if self.state_row_count != self.entity_state_row_count + self.report_summary_row_count:
            raise ValueError("stage-3 state row partitions do not sum to the total")
        if self.entity_state_row_count != (
            self.complete_entity_state_row_count + self.temporary_entity_state_row_count
        ):
            raise ValueError("stage-3 entity identity partitions do not sum to the total")
        if self.entity_state_row_count != (
            self.reliability_high_state_row_count
            + self.reliability_cautious_state_row_count
            + self.reliability_excluded_state_row_count
        ):
            raise ValueError("stage-3 reliability partitions do not sum to entity rows")
        if self.feature_row_count != self.snapshot_count * self.query_cell_count:
            raise ValueError("stage-3 feature rows must equal snapshots times fixed query cells")
        if self.null_value_count > self.nullable_value_count:
            raise ValueError("stage-3 null count exceeds nullable value count")

    @property
    def accepted(self) -> bool:
        return (
            self.snapshot_count == 205
            and self.first_report_date == date(2022, 7, 20)
            and self.last_report_date == date(2026, 7, 1)
            and self.report_summary_row_count == 205
            and self.known_missing_period_count == 2
            and self.future_source_reference_count == 0
            and self.missing_period_zero_imputation_count == 0
            and self.source_duration_feature_use_count == 0
            and self.forbidden_source_field_use_count == 0
            and self.target_or_earthquake_label_read_count == 0
            and self.synthetic_prefix_seed_count == 32
            and self.synthetic_prefix_property_failures == 0
        )

    def public_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["first_report_date"] = self.first_report_date.isoformat()
        value["last_report_date"] = self.last_report_date.isoformat()
        value["acceptance_checks_passed"] = self.accepted
        return value


@dataclass(frozen=True, slots=True)
class Stage3PublicDeliverables:
    registry: dict[str, object]
    registry_bytes: bytes
    report_bytes: bytes
    audit_svg_bytes: bytes
    dictionary_bytes: bytes


@dataclass(frozen=True, slots=True)
class PublishedStage3PublicDeliverables:
    registry: FixedFilePublication
    report: FixedFilePublication
    audit_svg: FixedFilePublication
    feature_dictionary: FixedFilePublication
    registry_payload: dict[str, object]


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _audit_mapping(audit: AuditResult) -> dict[str, object]:
    value = asdict(audit)
    if set(value) != {
        "passed",
        "selected_issue_count",
        "selected_feature_row_count",
        "unique_selected_cell_count",
        "state_row_count_checked",
        "observation_reference_count_checked",
        "report_reference_count_checked",
        "dictionary_definition_count",
        "dictionary_value_column_count",
        "feature_field_count",
        "feature_scalar_count_compared",
        "nullable_value_count_checked",
        "null_value_count_checked",
        "trajectory_value_count_checked",
    }:
        raise ValueError("stage-3 public audit fields changed")
    if (
        not audit.passed
        or audit.selected_issue_count != 12
        or audit.selected_feature_row_count != 12
    ):
        raise ValueError("stage-3 deterministic lineage replay audit did not pass")
    return value


def _synthetic_mapping(audit: SyntheticPrefixAuditResult) -> dict[str, object]:
    value = asdict(audit)
    if value != {
        "passed": True,
        "seed_count": 32,
        "invariant_count": 7,
        "check_count": 224,
        "failure_count": 0,
    }:
        raise ValueError("stage-3 synthetic-prefix audit did not pass all 224 frozen checks")
    return value


def _audit_integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"stage-3 generation audit count is invalid: {key}")
    return value


def aggregate_summary_from_manifest_audit(
    manifest: dict[str, object],
    *,
    lineage: AuditResult,
    synthetic: SyntheticPrefixAuditResult,
) -> Stage3AggregateSummary:
    """Derive the public aggregate only from the sealed manifest audit."""

    raw = manifest.get("audit")
    if not isinstance(raw, dict):
        raise ValueError("stage-3 manifest has no generation audit")
    lineage_mapping = _audit_mapping(lineage)
    synthetic_mapping = _synthetic_mapping(synthetic)
    if raw.get("lineage_replay") != lineage_mapping:
        raise ValueError("sealed generation audit differs from the supplied lineage result")
    if raw.get("synthetic_prefix") != synthetic_mapping:
        raise ValueError("sealed generation audit differs from the synthetic-prefix result")
    reliability = raw.get("reliability_entity_state_counts")
    if not isinstance(reliability, dict) or set(reliability) != {
        "high",
        "cautious",
        "excluded",
    }:
        raise ValueError("stage-3 reliability aggregate fields changed")
    first = raw.get("first_report_date")
    last = raw.get("last_report_date")
    if not isinstance(first, str) or not isinstance(last, str):
        raise ValueError("stage-3 generation audit report dates must be ISO strings")
    try:
        first_date = date.fromisoformat(first)
        last_date = date.fromisoformat(last)
    except ValueError as exc:
        raise ValueError("stage-3 generation audit report dates are invalid") from exc
    return Stage3AggregateSummary(
        snapshot_count=_audit_integer(raw, "actual_snapshot_count"),
        first_report_date=first_date,
        last_report_date=last_date,
        known_missing_period_count=_audit_integer(raw, "known_missing_period_count"),
        state_row_count=_audit_integer(raw, "state_row_count"),
        entity_state_row_count=_audit_integer(raw, "entity_state_count"),
        report_summary_row_count=_audit_integer(raw, "report_summary_row_count"),
        complete_entity_state_row_count=_audit_integer(
            raw,
            "complete_entity_state_row_count",
        ),
        temporary_entity_state_row_count=_audit_integer(
            raw,
            "temporary_entity_state_row_count",
        ),
        reliability_high_state_row_count=_audit_integer(reliability, "high"),
        reliability_cautious_state_row_count=_audit_integer(reliability, "cautious"),
        reliability_excluded_state_row_count=_audit_integer(reliability, "excluded"),
        query_cell_count=_audit_integer(raw, "query_cell_count"),
        feature_row_count=_audit_integer(raw, "feature_row_count"),
        nullable_value_count=_audit_integer(raw, "nullable_value_count"),
        null_value_count=_audit_integer(raw, "null_value_count"),
        future_source_reference_count=_audit_integer(raw, "future_source_reference_count"),
        missing_period_zero_imputation_count=_audit_integer(
            raw,
            "missing_period_zero_imputation_count",
        ),
        source_duration_feature_use_count=_audit_integer(
            raw,
            "source_duration_feature_use_count",
        ),
        forbidden_source_field_use_count=_audit_integer(
            raw,
            "forbidden_source_field_use_count",
        ),
        target_or_earthquake_label_read_count=_audit_integer(
            raw,
            "target_or_earthquake_label_read_count",
        ),
        synthetic_prefix_seed_count=synthetic.seed_count,
        synthetic_prefix_property_failures=synthetic.failure_count,
    )


def _public_dataset_summaries(manifest: dict[str, object]) -> dict[str, object]:
    raw = manifest.get("datasets")
    if not isinstance(raw, dict) or set(raw) != _EXPECTED_DATASETS:
        raise ValueError("stage-3 manifest must contain exactly the two local datasets")
    summaries: dict[str, object] = {}
    for name in sorted(_EXPECTED_DATASETS):
        entry = raw[name]
        if not isinstance(entry, dict):
            raise ValueError("stage-3 dataset manifest entry must be a mapping")
        fields = entry.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("stage-3 dataset manifest entry has no schema summary")
        summaries[name] = {
            "content_sha256": _require_sha256(entry.get("content_sha256"), label=name),
            "field_count": len(fields),
            "file_sha256": _require_sha256(entry.get("file_sha256"), label=name),
            "file_size_bytes": entry.get("file_size_bytes"),
            "row_count": entry.get("row_count"),
            "row_group_count": entry.get("row_group_count"),
            "schema_sha256": _require_sha256(entry.get("schema_sha256"), label=name),
        }
        for key in ("file_size_bytes", "row_count", "row_group_count"):
            number = cast(dict[str, object], summaries[name])[key]
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                raise ValueError(f"stage-3 dataset {name}.{key} must be positive")
    return summaries


def _public_bundle_files(manifest: dict[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("files")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("stage-3 bundle must contain exactly two scientific payload files")
    files: list[dict[str, object]] = []
    names: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("stage-3 bundle file entry must be a mapping")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or "/" in relative or "\\" in relative:
            raise ValueError("stage-3 public bundle filename must be a plain basename")
        names.add(relative)
        byte_count = entry.get("byte_count")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("stage-3 bundle byte count must be positive")
        files.append(
            {
                "byte_count": byte_count,
                "filename": relative,
                "sha256": _require_sha256(entry.get("sha256"), label=relative),
            }
        )
    if names != _EXPECTED_PAYLOAD_FILES:
        raise ValueError("stage-3 public bundle file set changed")
    return sorted(files, key=lambda item: cast(str, item["filename"]))


def _assert_public_registry_safe(value: object, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"public registry has a non-string key at {path}")
            if key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"public registry contains forbidden detail at {path}.{key}")
            _assert_public_registry_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_public_registry_safe(item, path=f"{path}[{index}]")


def _quality_percent(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(100.0 * count / total, 3)


def render_stage3_audit_svg(
    aggregate: Stage3AggregateSummary,
    audit: AuditResult,
) -> bytes:
    """Render one aggregate-only timeline and acceptance figure with no spatial detail."""

    if not aggregate.accepted or not audit.passed:
        raise ValueError("stage-3 audit SVG is published only after all aggregate gates pass")
    total = aggregate.entity_state_row_count
    quality = (
        ("高可靠", aggregate.reliability_high_state_row_count, "--viz-series-1", "#2563eb"),
        ("谨慎", aggregate.reliability_cautious_state_row_count, "--viz-series-2", "#d97706"),
        ("排除", aggregate.reliability_excluded_state_row_count, "--viz-series-3", "#64748b"),
    )
    bar_x = 84.0
    bar_y = 362.0
    bar_width = 1032.0
    offset = bar_x
    bar_parts: list[str] = []
    legend_parts: list[str] = []
    for index, (label, count, variable, fallback) in enumerate(quality):
        width = 0.0 if total == 0 else bar_width * count / total
        bar_parts.append(
            f'<rect x="{offset:.2f}" y="{bar_y:.2f}" width="{width:.2f}" height="38" '
            f'fill="var({variable},{fallback})"><title>{escape(label)}：{count:,} 条状态</title></rect>'
        )
        legend_x = 84 + index * 330
        legend_parts.append(
            f'<rect x="{legend_x}" y="420" width="14" height="14" rx="2" '
            f'fill="var({variable},{fallback})"/>'
            f'<text x="{legend_x + 24}" y="433" class="label">{escape(label)} '
            f"{count:,}（{_quality_percent(count, total):.3f}%）</text>"
        )
        offset += width

    checks = (
        ("205 个实际报告快照", aggregate.snapshot_count == 205),
        ("未来来源引用为 0", aggregate.future_source_reference_count == 0),
        ("12 个起报日逐字段重放", audit.selected_feature_row_count == 12),
        ("32 组因果前缀性质", aggregate.synthetic_prefix_property_failures == 0),
        ("缺报期未填零", aggregate.missing_period_zero_imputation_count == 0),
        ("锁定测试未运行", True),
    )
    check_parts: list[str] = []
    for index, (label, passed) in enumerate(checks):
        column = index % 3
        row = index // 3
        x = 84 + column * 344
        y = 532 + row * 66
        symbol = "✓" if passed else "×"
        check_parts.append(
            f'<circle cx="{x + 13}" cy="{y - 5}" r="13" '
            f'fill="var(--viz-series-1,#2563eb)" opacity="0.18"/>'
            f'<text x="{x + 13}" y="{y}" text-anchor="middle" class="check">{symbol}</text>'
            f'<text x="{x + 36}" y="{y}" class="label">{escape(label)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="690" viewBox="0 0 1200 690" role="img" aria-labelledby="title desc">
<title id="title">SeismoFlux 阶段 3 动态异常特征审计</title>
<desc id="desc">展示 2022 年 7 月至 2026 年 7 月的 205 个实际报告快照、两个已知缺报期、可靠性构成和因果重放验收。图中不含空间位置或地震目标。</desc>
<style>
  .bg{{fill:var(--background,#f8fafc)}} .fg{{fill:var(--foreground,#0f172a)}}
  .muted{{fill:var(--muted-foreground,#475569)}} .line{{stroke:var(--border,#94a3b8);stroke-width:2}}
  .accent{{stroke:var(--viz-series-1,#2563eb);stroke-width:5;stroke-linecap:round}}
  text{{font-family:"Noto Sans SC","Microsoft YaHei",sans-serif;font-weight:400}}
  .title{{font-size:27px;font-weight:500}} .section{{font-size:18px;font-weight:500}}
  .metric{{font-size:30px;font-weight:500}} .label{{font-size:15px}} .small{{font-size:13px}}
  .check{{font-size:16px;font-weight:500;fill:var(--foreground,#0f172a)}}
</style>
<rect class="bg" width="1200" height="690"/>
<text x="60" y="54" class="fg title">阶段 3：动态异常轨迹特征库</text>
<text x="60" y="84" class="muted label">仅评估特征构建的因果性与可重放性；此阶段没有训练预测模型，也没有准确率结论</text>

<text x="84" y="142" class="fg metric">{aggregate.snapshot_count}</text>
<text x="84" y="169" class="muted label">实际报告快照</text>
<text x="342" y="142" class="fg metric">{aggregate.feature_row_count:,}</text>
<text x="342" y="169" class="muted label">本地特征行</text>
<text x="660" y="142" class="fg metric">{audit.feature_field_count:,}</text>
<text x="660" y="169" class="muted label">宽表字段</text>
<text x="930" y="142" class="fg metric">{audit.feature_scalar_count_compared:,}</text>
<text x="930" y="169" class="muted label">审计比较标量</text>

<text x="84" y="224" class="fg section">报告时间覆盖</text>
<line x1="96" y1="270" x2="1104" y2="270" class="line"/>
<line x1="96" y1="270" x2="1104" y2="270" class="accent" opacity="0.62"/>
<circle cx="96" cy="270" r="7" fill="var(--viz-series-1,#2563eb)"/>
<circle cx="1104" cy="270" r="7" fill="var(--viz-series-1,#2563eb)"/>
<path d="M625 254 v32 M866 254 v32" stroke="var(--viz-series-2,#d97706)" stroke-width="4"/>
<text x="96" y="304" class="muted small">{aggregate.first_report_date.isoformat()}</text>
<text x="1104" y="304" text-anchor="end" class="muted small">{aggregate.last_report_date.isoformat()}</text>
<text x="625" y="246" text-anchor="middle" class="muted small">2024 第36期缺报</text>
<text x="866" y="318" text-anchor="middle" class="muted small">2025 第44期缺报</text>
<text x="84" y="344" class="fg section">实体状态可靠性构成（聚合计数）</text>
{"".join(bar_parts)}
{"".join(legend_parts)}
<text x="84" y="492" class="fg section">因果与交付门控</text>
{"".join(check_parts)}
<text x="84" y="665" class="muted small">相对强度、评分与顺位均不代表绝对发震概率；图件不含坐标、边界、格网编号或逐点特征值。</text>
</svg>"""
    return svg.encode("utf-8")


def _report_bytes(registry: dict[str, object]) -> bytes:
    aggregate = cast(dict[str, object], registry["aggregate"])
    audit = cast(dict[str, object], registry["lineage_replay_audit"])
    dictionary = cast(dict[str, object], registry["feature_dictionary"])
    bundle = cast(dict[str, object], registry["bundle"])
    lines = [
        "# SeismoFlux 阶段 3 动态异常特征报告",
        "",
        "## 先说结论",
        "",
        "阶段 3 已重建全部 205 个实际异常报告快照，并通过因果来源与逐字段重放审计。这里验收的是“特征是否只使用起报时刻已知信息、能否解释和重放”，不是模型预测好坏；本阶段没有训练模型、没有读取地震标签、没有运行锁定测试，也没有产生绝对发震概率。",
        "",
        "## 使用的数据",
        "",
        "- `anomaly_observation`：59,904 条已登记异常观测，仅使用冻结白名单字段。",
        "- `anomaly_report_period`：205 个实际报告期；2024 年第 36 期和 2025 年第 44 期按“未知”处理，不填零，也不据此推断异常结束。",
        "- 冻结研究区仅用于构建本地 25 km 查询格；公开文件不包含边界、坐标、格网编号或逐格值。",
        "- 地震目录、震中、震级完整性 Mc、人工预测地点/时间/震级以及断层几何/距离字段均未进入阶段 3 特征；跨断层观测仅作为冻结异常学科类别参与聚合。",
        "",
        "## 方法",
        "",
        "每个实际报告到达时，系统先按 `available_at <= 起报时刻` 重建异常实体状态，再计算 50/100/200/300/500 km 固定闭球与高斯核的多尺度特征，以及 4/8/13/26/52 周的实际时间轨迹、局部报告活动代理、缺报、年龄、持续时间、多学科和可靠性特征。覆盖量仅称为“报告活动代理”，不能解释为绝对观测能力。",
        "",
        "## 验收数字",
        "",
        f"- 状态历史：{aggregate['state_row_count']:,} 行，其中报告摘要 {aggregate['report_summary_row_count']:,} 行。",
        f"- 本地特征库：{aggregate['feature_row_count']:,} 行。",
        f"- 特征字典：{dictionary['logical_definition_count']:,} 个逻辑定义、{dictionary['storage_value_column_count']:,} 个值列。",
        f"- 因果重放：{audit['selected_feature_row_count']} 个冻结起报日样本全部逐字段一致，共比较 {audit['feature_scalar_count_compared']:,} 个标量。",
        f"- 内容寻址产物：`{bundle['bundle_id']}`。",
        "- 32 个合成因果前缀种子全部通过；未来来源引用、缺报填零、源持续时间字段使用、禁用字段使用和地震标签读取均为 0。",
        "",
        "## 如何理解当前结果",
        "",
        "这一步证明输入状态和特征库具有时间因果性、局部缺失不会屏蔽其它区域、且可从来源记录重放。它还不能说明哪些异常能提高地震预测；该问题只能在阶段 4 用简单可解释增量模型，与 ETAS 背景和异常置乱对照比较后回答。锁定测试仍保持未运行。",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def build_stage3_public_deliverables(
    *,
    manifest: dict[str, object],
    dictionary: FeatureDictionary,
    audit: AuditResult,
    synthetic: SyntheticPrefixAuditResult,
    code_commit: str,
    input_hashes: dict[str, str],
) -> Stage3PublicDeliverables:
    """Build all four fixed public files from aggregate-only accepted inputs."""

    assert_public_safe_feature_dictionary(dictionary)
    aggregate = aggregate_summary_from_manifest_audit(
        manifest,
        lineage=audit,
        synthetic=synthetic,
    )
    if not aggregate.accepted:
        raise ValueError("stage-3 aggregate acceptance gate did not pass")
    audit_payload = _audit_mapping(audit)
    synthetic_payload = _synthetic_mapping(synthetic)
    if set(input_hashes) != _INPUT_HASH_KEYS:
        raise ValueError("stage-3 public input-hash set changed")
    normalized_hashes = {
        key: _require_sha256(value, label=key) for key, value in sorted(input_hashes.items())
    }
    if _GIT_COMMIT.fullmatch(code_commit) is None:
        raise ValueError("stage-3 code commit must be a lowercase Git object ID")
    if manifest.get("protocol_version") != "0.3.0" or manifest.get("schema_version") != 1:
        raise ValueError("stage-3 manifest version changed")
    locked = manifest.get("locked_test")
    if locked != {
        "artifact_ids": [],
        "result": None,
        "run": False,
        "score_ids": [],
        "target_count": None,
        "target_ids": [],
    }:
        raise ValueError("stage-3 manifest locked-test assertion must remain false/empty/null")
    bundle_id = manifest.get("bundle_id")
    identity_sha256 = manifest.get("identity_sha256")
    if not isinstance(bundle_id, str) or not bundle_id.startswith("anomaly-feature-bundle-"):
        raise ValueError("stage-3 bundle ID changed")
    identity_digest = _require_sha256(identity_sha256, label="bundle identity")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise ValueError("stage-3 bundle identity fields changed")
    if identity != {
        "schema_version": 1,
        "protocol_version": "0.3.0",
        "execution_mode": "feature_only_no_target_scoring",
        "protocol_freeze_tag": "v0.3.0-anomaly-feature-protocol",
        "code_commit": code_commit,
        "feature_dictionary_sha256": dictionary.sha256,
        "grid": identity.get("grid"),
        "input_hashes": normalized_hashes,
    }:
        raise ValueError("stage-3 bundle identity differs from the public acceptance inputs")
    grid_identity = identity["grid"]
    if not isinstance(grid_identity, dict) or set(grid_identity) != {
        "grid_id",
        "cell_count",
        "cell_size_km",
    }:
        raise ValueError("stage-3 bundle grid identity fields changed")
    if (
        not isinstance(grid_identity["grid_id"], str)
        or _SHA256.fullmatch(grid_identity["grid_id"]) is None
        or grid_identity["cell_count"] != aggregate.query_cell_count
        or grid_identity["cell_size_km"] != 25.0
    ):
        raise ValueError("stage-3 bundle grid identity differs from the accepted fixed grid")
    if identity_digest != stage3_identity_sha256(identity):
        raise ValueError("stage-3 manifest identity SHA-256 cannot be reproduced")
    if bundle_id != f"anomaly-feature-bundle-{identity_digest[:16]}":
        raise ValueError("stage-3 bundle ID does not match its accepted identity")
    datasets = _public_dataset_summaries(manifest)
    bundle_files = _public_bundle_files(manifest)
    if audit.dictionary_definition_count != len(dictionary.definitions):
        raise ValueError("lineage audit definition count differs from the frozen dictionary")
    if audit.dictionary_value_column_count != len(dictionary.storage_value_columns()):
        raise ValueError("lineage audit value-column count differs from the frozen dictionary")
    feature_dataset = cast(dict[str, object], datasets["anomaly_feature_store"])
    if audit.feature_field_count != feature_dataset["field_count"]:
        raise ValueError("lineage audit field count differs from the sealed feature schema")
    if (
        cast(dict[str, object], datasets["anomaly_state_history"])["row_count"]
        != aggregate.state_row_count
    ):
        raise ValueError("public aggregate state row count differs from the sealed bundle")
    if feature_dataset["row_count"] != aggregate.feature_row_count:
        raise ValueError("public aggregate feature row count differs from the sealed bundle")

    dictionary_bytes = dictionary.canonical_bytes
    audit_svg_bytes = render_stage3_audit_svg(aggregate, audit)
    registry: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "0.3.0",
        "stage": 3,
        "status": "accepted_feature_only_no_target_scoring",
        "code_commit": code_commit,
        "input_hashes": normalized_hashes,
        "bundle": {
            "bundle_id": bundle_id,
            "identity_sha256": identity_digest,
            "files": bundle_files,
        },
        "datasets": datasets,
        "feature_dictionary": {
            "dictionary_id": dictionary.dictionary_id,
            "sha256": dictionary.sha256,
            "logical_definition_count": len(dictionary.definitions),
            "storage_value_column_count": len(dictionary.storage_value_columns()),
            "semantics": FEATURE_SEMANTICS,
            "path": PUBLIC_DICTIONARY_PATH,
        },
        "aggregate": aggregate.public_mapping(),
        "lineage_replay_audit": audit_payload,
        "synthetic_prefix_audit": synthetic_payload,
        "audit_figure": {
            "path": PUBLIC_AUDIT_SVG_PATH,
            "sha256": hashlib.sha256(audit_svg_bytes).hexdigest(),
            "contains_geometry": False,
            "contains_coordinates": False,
            "contains_per_cell_values": False,
        },
        "locked_test": {
            "run": False,
            "score_ids": [],
            "artifact_ids": [],
            "target_count": None,
            "result": None,
        },
        "stage4_allowed": True,
    }
    _assert_public_registry_safe(registry)
    registry_bytes = canonical_json_bytes(registry)
    report_bytes = _report_bytes(registry)
    return Stage3PublicDeliverables(
        registry=registry,
        registry_bytes=registry_bytes,
        report_bytes=report_bytes,
        audit_svg_bytes=audit_svg_bytes,
        dictionary_bytes=dictionary_bytes,
    )


def publish_stage3_public_deliverables(
    project_root: Path,
    deliverables: Stage3PublicDeliverables,
) -> PublishedStage3PublicDeliverables:
    """Create all fixed paths, rolling back only files created by this attempt."""

    created: list[FixedFilePublication] = []

    def publish(reference: str, payload: bytes) -> FixedFilePublication:
        result = publish_fixed_project_file(project_root, reference, payload)
        if result.created:
            created.append(result)
        return result

    try:
        dictionary = publish(PUBLIC_DICTIONARY_PATH, deliverables.dictionary_bytes)
        audit_svg = publish(PUBLIC_AUDIT_SVG_PATH, deliverables.audit_svg_bytes)
        report = publish(PUBLIC_REPORT_PATH, deliverables.report_bytes)
        # Publish the machine-readable registry last: its presence is the fixed-path
        # completion marker for the other three public summaries.
        registry = publish(PUBLIC_REGISTRY_PATH, deliverables.registry_bytes)
    except Exception:
        rollback_error: OSError | None = None
        for publication in reversed(created):
            try:
                publication.path.unlink(missing_ok=True)
            except OSError as exc:
                rollback_error = exc
        if rollback_error is not None:
            raise RuntimeError(
                "unable to roll back partial stage-3 fixed-file publication"
            ) from rollback_error
        raise
    return PublishedStage3PublicDeliverables(
        registry=registry,
        report=report,
        audit_svg=audit_svg,
        feature_dictionary=dictionary,
        registry_payload=deliverables.registry,
    )


__all__ = [
    "PUBLIC_AUDIT_SVG_PATH",
    "PUBLIC_DICTIONARY_PATH",
    "PUBLIC_REGISTRY_PATH",
    "PUBLIC_REPORT_PATH",
    "PublishedStage3PublicDeliverables",
    "Stage3AggregateSummary",
    "Stage3PublicDeliverables",
    "aggregate_summary_from_manifest_audit",
    "build_stage3_public_deliverables",
    "publish_stage3_public_deliverables",
    "render_stage3_audit_svg",
]
