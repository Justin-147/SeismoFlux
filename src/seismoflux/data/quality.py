"""Machine-readable and human-readable stage-1 quality reporting."""

# ruff: noqa: RUF001 - Chinese punctuation is intentional in the published report.

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from seismoflux.data.common import write_json_atomic


def _display(value: object) -> str:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, date | datetime):
        return value.isoformat()
    return str(value)


def count_leakage_violations(
    observations: Sequence[Mapping[str, Any]],
    geology_records: Sequence[Mapping[str, Any]],
    earthquake_records: Sequence[Mapping[str, Any]] = (),
    basemap_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[int, dict[str, int]]:
    """Check only rules that can be proven at the standardized-data boundary."""

    anomaly_availability = 0
    anomaly_backfill = 0
    anomaly_occurrence_precision = 0
    for record in observations:
        report_date = record["report_date"]
        raw_report_date = record["raw_row_report_date"]
        available_at = record["available_at"]
        if not isinstance(report_date, date) or not isinstance(raw_report_date, date):
            anomaly_availability += 1
            continue
        if not isinstance(available_at, datetime) or available_at.tzinfo is None:
            anomaly_availability += 1
            continue
        expected_local_date = max(report_date, raw_report_date)
        local_available = available_at.astimezone(UTC)
        expected_utc_date = expected_local_date
        expected_hour = 16
        if not (
            local_available.date() == expected_utc_date
            and local_available.hour == expected_hour
            and local_available.minute == 0
            and local_available.second == 0
            and local_available.microsecond == 0
        ):
            anomaly_availability += 1
        end_time = record.get("reported_end_time")
        flags = set(record.get("reliability_flags", ()))
        occurrence_times = [record.get("start_time")]
        if end_time is not None:
            occurrence_times.append(end_time)
        occurrence_times_valid = all(
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() == timedelta(hours=8)
            and value.hour == 0
            and value.minute == 0
            and value.second == 0
            and value.microsecond == 0
            for value in occurrence_times
        )
        if (
            not occurrence_times_valid
            or "anomaly_time_date_only_assumed_fixed_utc_plus_08_midnight" not in flags
        ):
            anomaly_occurrence_precision += 1
        if end_time is None and not bool(record["right_censored"]):
            anomaly_backfill += 1
        if end_time is not None:
            future_end = "future_reported_end_time" in flags
            if bool(record["right_censored"]) != future_end:
                anomaly_backfill += 1
        if "manual_prediction_fields_excluded" not in flags:
            anomaly_backfill += 1

    earthquake_availability = 0
    for record in earthquake_records:
        origin_time = record.get("origin_time_utc")
        available_at = record.get("available_at")
        flags = set(record.get("quality_flags", record.get("normalization_flags", ())))
        if (
            not isinstance(origin_time, datetime)
            or origin_time.tzinfo is None
            or origin_time.utcoffset() != timedelta(0)
            or not isinstance(available_at, datetime)
            or available_at.tzinfo is None
            or available_at.utcoffset() != timedelta(0)
            or available_at != origin_time
            or "publication_time_assumed_origin_time" not in flags
        ):
            earthquake_availability += 1

    geology_early_use = sum(
        bool(record.get("historical_model_eligible")) for record in geology_records
    )
    basemap_model_use = sum(
        bool(record.get("model_feature_eligible")) for record in basemap_records
    )
    checks = {
        "anomaly_available_at_rule": anomaly_availability,
        "anomaly_end_or_manual_field_backfill": anomaly_backfill,
        "anomaly_occurrence_time_precision": anomaly_occurrence_precision,
        "earthquake_available_at_assumption": earthquake_availability,
        "historical_geology_early_use": geology_early_use,
        "basemap_model_feature_use": basemap_model_use,
    }
    return sum(checks.values()), checks


def build_quality_report(
    *,
    source_inventory_sha256: str,
    anomaly_quality: Mapping[str, Any],
    earthquake_quality: Mapping[str, Any],
    geology_quality: Mapping[str, Any],
    leakage_checks: Mapping[str, int],
    leakage_violations: int,
) -> dict[str, Any]:
    warnings = [
        "所有原始和标准化数据授权仍为 unknown_no_redistribution，标准化数据不得提交 Git。",
        "两个地震目录缺少权威时区、坐标系、震级类型和逐事件发布时间元数据。",
        "地震目录暂按固定 UTC+08 与发震时刻可用的乐观假设标准化，并逐条标记。",
        "异常起止时间源字段只有日期，暂按固定 UTC+08 当日 00:00 表示并逐条标记。",
        "当前断层属性和长期危险性快照不得用于其 available_at 之前的历史回测。",
        "研究区只含连续大陆最大闭合环，明确排除海南、台湾及其他离岛。",
        "宽阈值地震重复和断层真实迹线映射均保持未审计状态，不自动接受。",
    ]
    return {
        "schema_version": 1,
        "contract_version": "0.1.0",
        "status": "pass_with_documented_warnings" if leakage_violations == 0 else "fail",
        "source_inventory_sha256": source_inventory_sha256,
        "leakage": {
            "violations": leakage_violations,
            "checks": dict(sorted(leakage_checks.items())),
        },
        "domains": {
            "anomaly": dict(sorted(anomaly_quality.items())),
            "earthquake": dict(sorted(earthquake_quality.items())),
            "geology_and_basemap": dict(sorted(geology_quality.items())),
        },
        "warnings": warnings,
        "blocking_errors": [] if leakage_violations == 0 else ["future_information_leakage"],
    }


def render_quality_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 阶段 1 数据质量报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- 数据契约：`{report['contract_version']}`",
        f"- 原始清单 SHA-256：`{report['source_inventory_sha256']}`",
        f"- 未来信息泄漏违规：**{report['leakage']['violations']}**",
        "",
        "## 防泄漏检查",
        "",
        "| 检查 | 违规数 |",
        "| --- | ---: |",
    ]
    for name, count in sorted(report["leakage"]["checks"].items()):
        lines.append(f"| `{name}` | {count} |")

    for domain, values in report["domains"].items():
        lines.extend(["", f"## {domain}", "", "| 项目 | 结果 |", "| --- | --- |"])
        for name, value in sorted(values.items()):
            escaped_value = _display(value).replace("|", "\\|")
            lines.append(f"| `{name}` | {escaped_value} |")

    lines.extend(["", "## 已知限制", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(
        [
            "",
            "本报告只公开聚合质量指标。原始数据和标准化逐行数据均保留在 Git 之外。",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_atomic(path: Path, payload: str) -> None:
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
            handle.write(payload)
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def write_quality_outputs(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    write_json_atomic(json_path, report)
    write_text_atomic(markdown_path, render_quality_markdown(report))
