"""Crash-safe publication of one immutable stage-4 formal-result bundle.

The content-addressed bundle is finalized before any fixed public path changes.
Fixed report, model-card, SVG, local HTML, and optional local extension paths are
then created transactionally and may only be replayed with identical bytes.  No
historical formal result is overwritten.  The public registry is created last
and is the sole fixed-path completion marker.
"""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import hashlib
import html
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypeAlias, cast

from seismoflux.anomaly_increment.convergence import CompensatorConvergenceAudit
from seismoflux.anomaly_increment.immutable_file import (
    ImmutableFileSnapshot,
    UnsafeImmutableFileError,
    read_existing_immutable_file,
    require_existing_real_directory,
    unlink_existing_immutable_file,
)
from seismoflux.anomaly_increment.preregistration import with_content_sha256
from seismoflux.anomaly_increment.runner import Stage4PublicationPlan
from seismoflux.anomaly_increment.scoring_pipeline import PipelineResult
from seismoflux.data.common import canonical_json_bytes

FORMAL_PUBLICATION_SCHEMA_VERSION: Final[int] = 1
FORMAL_BUNDLE_ROOT: Final[PurePosixPath] = PurePosixPath("models/registry/anomaly_increment_r1")

PublicationStatus: TypeAlias = Literal["succeeded", "failed"]
CreateFile: TypeAlias = Callable[[Path, Path], bool]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FAILURE_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}")
_FORBIDDEN_TERMS = ("probability", "概率")
_PUBLIC_RESTRICTED_TOKENS = (
    "coordinates",
    "epicenter",
    "event_id",
    "eventid",
    "geometry_geojson",
    "geometrycollection",
    "geojson",
    "inside_study_area",
    "linestring",
    "longitude",
    "latitude",
    "multilinestring",
    "multipolygon",
    "origin_time_utc",
    "point (",
    "polygon (",
    "query_x_m",
    "query_y_m",
    "polygon_wkb",
    "per_cell_mapping",
    "construction_zone_geometry",
    "target_count",
    "target_id",
    "target_ids",
    "target_locations",
    "wgs84",
    "震中",
)
_ABSOLUTE_LOCAL_PATH = re.compile(
    r"(?ix)(?:"
    r"(?<![a-z0-9])[a-z]:[\\/]"
    r"|\\\\[^\\/\s]+[\\/]"
    r"|file://"
    r"|(?:^|[\"'`=\s(])/(?!/)(?:[a-z0-9._-]+/)+[a-z0-9._-]+"
    r")"
)
_MOJIBAKE_FRAGMENTS = (
    "闃舵",
    "姒傜巼",
    "锛",
    "璇婃柇",
    "妯″瀷",
)
_DANGEROUS_LOCAL_HTML_TOKENS = (
    "<script src=",
    "fetch(",
    "xmlhttprequest",
    "websocket",
    "https://",
)


class FormalPublicationError(RuntimeError):
    """Raised when an immutable bundle or fixed-path transaction is invalid."""


def _sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _failure_code(value: str) -> str:
    if not isinstance(value, str) or _FAILURE_CODE.fullmatch(value) is None:
        raise ValueError("failure_code must be a normalized identifier")
    return value


def _relative_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a normalized project-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} must be a normalized project-relative path")
    return path


def _project_path(root: Path, relative: str, *, label: str) -> Path:
    pure = _relative_path(relative, label=label)
    resolved_root = root.resolve()
    # Keep the final directory entry lexical.  Resolving it here would follow a
    # pre-positioned symlink before the no-follow immutable-file gate can reject it.
    result = resolved_root.joinpath(*pure.parts)
    if not result.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes the project root")
    return result


def _ensure_real_directory_tree(root: Path, directory: Path, *, label: str) -> None:
    resolved_root = root.resolve()
    if not directory.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes the project root")
    try:
        require_existing_real_directory(resolved_root, label="publication project root")
        current = resolved_root
        for part in directory.relative_to(resolved_root).parts:
            current /= part
            try:
                os.mkdir(current)
            except FileExistsError:
                require_existing_real_directory(current, label=label)
    except UnsafeImmutableFileError as exc:
        raise FormalPublicationError(f"unsafe publication directory: {label}") from exc


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decoded_validated_text(name: str, payload: bytes) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormalPublicationError(f"publication payload is not UTF-8: {name}") from exc
    folded = text.casefold()
    if any(term in folded if term.isascii() else term in text for term in _FORBIDDEN_TERMS):
        raise FormalPublicationError(
            f"publication payload uses forbidden absolute-probability wording: {name}"
        )
    corrupted = tuple(fragment for fragment in _MOJIBAKE_FRAGMENTS if fragment in text)
    if corrupted:
        raise FormalPublicationError(
            f"publication payload contains probable UTF-8 mojibake {corrupted[0]}: {name}"
        )
    return text


def _validate_public_payload(name: str, payload: bytes) -> None:
    text = _decoded_validated_text(name, payload)
    folded = text.casefold()
    leaked = tuple(token for token in _PUBLIC_RESTRICTED_TOKENS if token in folded)
    if leaked:
        raise FormalPublicationError(
            f"publication payload exposes restricted geometry token {leaked[0]}: {name}"
        )
    if _ABSOLUTE_LOCAL_PATH.search(text) is not None:
        raise FormalPublicationError(f"publication payload exposes an absolute local path: {name}")


def _validate_local_payload(name: str, payload: bytes) -> None:
    """Allow local coordinates while rejecting remote/external script capabilities."""

    text = _decoded_validated_text(name, payload)
    folded = text.casefold()
    dangerous = tuple(token for token in _DANGEROUS_LOCAL_HTML_TOKENS if token in folded)
    if dangerous:
        raise FormalPublicationError(
            f"local publication payload contains dangerous capability {dangerous[0]}: {name}"
        )


@dataclass(frozen=True, slots=True)
class AdditionalLocalArtifact:
    """Optional target-safe local output hook, including a later spatial dashboard."""

    relative_path: str
    bundle_filename: str
    payload: bytes

    def __post_init__(self) -> None:
        path = _relative_path(self.relative_path, label="additional local artifact path")
        if path.parts[:2] != ("outputs", "visualizations"):
            raise ValueError("additional local artifacts must stay under outputs/visualizations")
        filename = PurePosixPath(self.bundle_filename)
        if (
            len(filename.parts) != 1
            or filename.name != self.bundle_filename
            or filename.name in {".", ".."}
        ):
            raise ValueError("additional local bundle filename must be one safe filename")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("additional local artifact payload must be non-empty bytes")
        _validate_local_payload(self.bundle_filename, self.payload)


@dataclass(frozen=True, slots=True)
class FormalPublicationReceipt:
    status: PublicationStatus
    bundle_id: str
    bundle_directory: Path
    registry_sha256: str
    manifest_sha256: str
    fixed_path_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("bundle_id", self.bundle_id),
            ("registry_sha256", self.registry_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            _sha256(value, label=label)
        names = tuple(name for name, _ in self.fixed_path_sha256)
        if names != tuple(sorted(set(names))):
            raise ValueError("fixed publication receipt paths must be sorted and unique")
        for name, value in self.fixed_path_sha256:
            _relative_path(name, label="fixed publication receipt path")
            _sha256(value, label=f"fixed publication receipt {name}")


def _convergence_publication_summary(
    audit: CompensatorConvergenceAudit,
    extensions: Sequence[AdditionalLocalArtifact],
) -> dict[str, object]:
    if not isinstance(audit, CompensatorConvergenceAudit) or not audit.passed:
        raise ValueError("successful publication requires a passed convergence audit")
    artifacts = tuple(
        item for item in extensions if item.bundle_filename == "convergence-audit.json"
    )
    if len(artifacts) != 1:
        raise ValueError("successful publication requires one local convergence audit artifact")
    artifact = artifacts[0]
    passed_count = sum(item.passed for item in audit.results)
    total_count = len(audit.results)
    if passed_count != 40 or total_count != 40:
        raise ValueError("formal convergence publication requires all 40 checks")
    return {
        "audit_content_sha256": audit.content_sha256,
        "coarse_50km_vs_25km": "diagnostic_only_not_a_gate",
        "local_artifact_path": artifact.relative_path,
        "local_artifact_sha256": _digest(artifact.payload),
        "near_zero_absolute_tolerance": audit.near_zero_absolute_tolerance,
        "passed_check_count": passed_count,
        "relative_tolerance": audit.relative_tolerance,
        "spatial_gate": "25km_vs_12.5km",
        "status": audit.status,
        "temporal_gate": "1day_vs_0.5day",
        "total_check_count": total_count,
    }


def _format_optional(value: float | None) -> str:
    return "证据不足" if value is None else f"{value:.8g}"


def _next_step(result: PipelineResult) -> str:
    if result.dynamic_g2.status == "passed" and result.dynamic_g3.status == "passed":
        return "G2/G3 均通过；可按蓝图进入下一阶段，但不得用本次正式验证继续调参。"
    if "evidence_insufficient" in {
        result.dynamic_g2.status,
        result.dynamic_g3.status,
    }:
        return "G2/G3 至少一个门控证据不足；保留背景模型并停止复杂化，等待真正新增独立证据。"
    return "G2/G3 至少一个门控失败；发布可信负结果，保留背景模型并停止复杂异常模型。"


def _success_report(
    result: PipelineResult,
    publication: Stage4PublicationPlan,
    extensions: Sequence[AdditionalLocalArtifact],
    convergence: Mapping[str, object],
) -> str:
    rows = []
    for item in result.retrospective.horizon_results:
        interval = (
            "证据不足"
            if item.information_gain_lower_95 is None or item.information_gain_upper_95 is None
            else (f"[{item.information_gain_lower_95:.8g}, {item.information_gain_upper_95:.8g}]")
        )
        rows.append(
            "| "
            + " | ".join(
                (
                    item.magnitude_bin,
                    str(item.horizon_days),
                    item.result_status,
                    "证据不足"
                    if item.independent_event_count is None
                    else str(item.independent_event_count),
                    _format_optional(item.information_gain_nats_per_event),
                    interval,
                    _format_optional(item.strict_recall_at_600000_km2),
                )
            )
            + " |"
        )
    convergence_lines = (
        "## Compensator convergence gate",
        "",
        f"- Status: `{convergence['status']}`",
        f"- Gating checks passed: `{convergence['passed_check_count']}/{convergence['total_check_count']}`",
        "- Relative tolerance: `0.005` (0.5%).",
        "- Near-zero absolute tolerance: `1e-10`.",
        "- Spatial gate: 25 km versus 12.5 km.",
        "- Temporal gate: 1 day versus 0.5 day.",
        "- The 50 km versus 25 km comparison is a coarse diagnostic only, not a gate.",
        f"- Local audit artifact: `{convergence['local_artifact_path']}`",
        f"- Local audit SHA-256: `{convergence['local_artifact_sha256']}`",
        "",
    )
    return (
        "\n".join(
            (
                "# SeismoFlux 阶段 4 异常增量正式报告",
                "",
                f"- 执行结论：`{result.retrospective.outcome_status}`",
                f"- G2：`{result.dynamic_g2.status}`",
                f"- G3：`{result.dynamic_g3.status}`",
                f"- 采用决策：`{result.adoption.choice}`（{result.adoption.reason}）",
                f"- 预测载荷状态：`{result.prospective.forecast_status}`",
                f"- 结果指纹：`{result.result_fingerprint_sha256}`",
                f"- 诊断指纹：`{result.publication_diagnostics.content_sha256}`",
                "",
                "所有强度均为条件相对强度或顺位，不是绝对发震频率断言。",
                "",
                "## 回溯窗口",
                "",
                "| 震级档 | 天数 | 证据状态 | 独立事件数 | 每事件信息增益 | 95% 区间 | 严格召回 |",
                "|---|---:|---|---:|---:|---|---:|",
                *rows,
                "",
                "## 置乱对照",
                "",
                f"- 动态时间置乱 p 值：{_format_optional(result.retrospective.time_permutation_p_value)}",
                f"- 动态空间置乱 p 值：{_format_optional(result.retrospective.space_permutation_p_value)}",
                "",
                "ETAS 数值拟合当前不可用；本阶段按冻结蓝图使用空间背景基线，不把 ETAS 缺失伪装为通过。",
                "M6+ 与 180/365 天窗口始终按证据不足解释；数值仅用于透明展示。",
                "局部 Mc 偏高只降低对应区域的背景支持与证据水位，不向其他区域外溢。",
                "当前 shadow 载荷是在冻结后回溯生成的目标盲影子预测，不是真正前瞻预测。",
                "",
                "## 图件与下一步",
                "",
                f"![阶段4方法效果]({PurePosixPath(publication.public_static_svg).name})",
                "",
                f"- 本地交互图：`{publication.local_interactive_html}`",
                *(f"- 本地扩展图：`{item.relative_path}`" for item in extensions),
                *convergence_lines,
                f"- 下一步：{_next_step(result)}",
            )
        )
        + "\n"
    )


def _success_model_card(
    result: PipelineResult,
    convergence: Mapping[str, object],
) -> str:
    variants = ", ".join(item.variant for item in result.fitted_scopes[-1].variants)
    design_names = ", ".join(result.fitted_scopes[-1].preprocessor.design_column_names)
    return (
        "\n".join(
            (
                "# SeismoFlux 异常增量模型卡",
                "",
                f"- 模型版本：`{result.retrospective.protocol.model_version}`",
                f"- 协议版本：`{result.retrospective.protocol.protocol_version}`",
                f"- 背景模型：`{result.retrospective.protocol.background_variant}`",
                f"- 候选变体：{variants}",
                f"- 冻结设计列：{design_names}",
                f"- G2/G3：`{result.dynamic_g2.status}` / `{result.dynamic_g3.status}`",
                f"- 最终采用：`{result.adoption.choice}`",
                f"- 预测载荷状态：`{result.prospective.forecast_status}`",
                "",
                "## 用途",
                "",
                "在冻结背景之上评估异常特征的增量信息，并以受控面积下的条件相对强度和顺位辅助回溯。",
                "",
                "## 限制",
                "",
                "- ETAS 数值拟合未形成合格参数，本阶段保留已冻结的空间背景基线。",
                "- M6+ 样本及长窗口证据不足，不进入通过门控的结论。",
                "- 当前预测载荷是回溯生成的目标盲影子预测，不等同于真正前瞻运行。",
                "- 局部 Mc 偏高只影响对应区域的背景支持和证据水位，不改变其他区域。",
                "- 不得使用该模型生成绝对发震频率陈述。",
                "",
                "## 门控后的动作",
                "",
                _next_step(result),
                "",
                "## Compensator convergence gate",
                "",
                f"- Status: `{convergence['status']}`",
                f"- Passed checks: `{convergence['passed_check_count']}/{convergence['total_check_count']}`",
                "- Relative tolerance: `0.005` (0.5%); near-zero absolute tolerance: `1e-10`.",
                "- The 50 km versus 25 km comparison is diagnostic only and is not a gate.",
                f"- Local audit SHA-256: `{convergence['local_artifact_sha256']}`",
            )
        )
        + "\n"
    )


def _failure_svg(*, model_version: str, failure_code: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="420" viewBox="0 0 1000 420" role="img" aria-labelledby="title desc" data-scientific-values="absent">
<title id="title">SeismoFlux 阶段 4 正式执行失败</title>
<desc id="desc">执行失败；没有生成科学数值、曲线或替代结果。</desc>
<rect width="1000" height="420" fill="#f6f7fa"/><text x="48" y="84" font-size="30" fill="#172033">阶段 4 正式执行失败</text>
<text x="48" y="142" font-size="18" fill="#5c6877">未生成科学数值、假曲线或替代结论</text>
<text x="48" y="204" font-size="16" fill="#172033">失败代码：{html.escape(failure_code)}</text>
<text x="48" y="244" font-size="16" fill="#172033">模型版本：{html.escape(model_version)}</text>
<text x="48" y="326" font-size="15" fill="#5c6877">条件相对强度与顺位结果不可用；请查阅审计台账。</text>
</svg>"""


def _failure_html(*, model_version: str, failure_code: str) -> str:
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SeismoFlux 阶段 4 执行失败</title><style>body{{font-family:system-ui,sans-serif;margin:0;background:#f6f7fa;color:#172033}}main{{max-width:760px;margin:10vh auto;padding:28px;background:#fff;border:1px solid #ccd3dd;border-radius:10px}}.muted{{color:#5c6877}}</style></head><body><main data-scientific-values="absent"><h1>阶段 4 正式执行失败</h1><p>没有生成科学数值、曲线或替代结论。</p><p>失败代码：<code>{html.escape(failure_code)}</code></p><p>模型版本：<code>{html.escape(model_version)}</code></p><p class="muted">条件相对强度与顺位结果不可用；硬崩溃时审计台账会保留 started 状态且禁止伪恢复。</p></main></body></html>"""


def _failure_report(*, model_version: str, failure_code: str) -> str:
    return (
        "\n".join(
            (
                "# SeismoFlux 阶段 4 异常增量正式报告",
                "",
                "正式执行失败，未生成科学数值、曲线或替代结论。",
                "",
                f"- 模型版本：`{model_version}`",
                f"- 失败代码：`{failure_code}`",
                "- 科学数值：不可用",
                "- 条件相对强度与顺位：不可用",
            )
        )
        + "\n"
    )


def _failure_model_card(*, model_version: str, failure_code: str) -> str:
    return (
        "\n".join(
            (
                "# SeismoFlux 异常增量模型卡",
                "",
                f"- 模型版本：`{model_version}`",
                "- 发布状态：正式执行失败",
                f"- 失败代码：`{failure_code}`",
                "- 科学数值：不可用",
                "",
                "本次失败不产生候选模型、条件相对强度曲线、顺位或替代结论。",
            )
        )
        + "\n"
    )


def _artifact_hashes(payloads: Mapping[str, bytes]) -> dict[str, str]:
    return {name: _digest(payload) for name, payload in sorted(payloads.items())}


def _registry_payload(
    *,
    status: PublicationStatus,
    bundle_id: str,
    execution_binding_id: str,
    authorization_id: str,
    model_version: str,
    artifact_hashes: Mapping[str, str],
    publication: Stage4PublicationPlan,
    result: PipelineResult | None,
    failure_code: str | None,
    additional_local_artifact_sha256: Mapping[str, str],
    convergence: Mapping[str, object] | None,
) -> bytes:
    payload: dict[str, object] = {
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "additional_local_artifact_sha256": dict(sorted(additional_local_artifact_sha256.items())),
        "authorization_id": authorization_id,
        "bundle_id": bundle_id,
        "execution_binding_id": execution_binding_id,
        "failure_code": failure_code,
        "fixed_paths": {
            "interactive_local": publication.local_interactive_html,
            "model_card": publication.public_model_card,
            "registry": publication.public_registry,
            "report": publication.public_report,
            "static_svg": publication.public_static_svg,
        },
        "interpretation": "conditional_relative_intensity_and_rank",
        "model_version": model_version,
        "schema_version": FORMAL_PUBLICATION_SCHEMA_VERSION,
        "scientific_values_available": status == "succeeded",
        "status": status,
        "convergence": None if convergence is None else dict(convergence),
    }
    if result is not None:
        payload.update(
            {
                "adoption": result.adoption.choice,
                "diagnostics_sha256": result.publication_diagnostics.content_sha256,
                "forecast_status": result.prospective.forecast_status,
                "g2_status": result.dynamic_g2.status,
                "g3_status": result.dynamic_g3.status,
                "result_fingerprint_sha256": result.result_fingerprint_sha256,
            }
        )
    else:
        payload.update(
            {
                "adoption": None,
                "diagnostics_sha256": None,
                "forecast_status": None,
                "g2_status": None,
                "g3_status": None,
                "result_fingerprint_sha256": None,
            }
        )
    return canonical_json_bytes(with_content_sha256(payload)) + b"\n"


def _safe_unlink_staged(path: Path) -> None:
    """Unlink a private staging entry itself without following its target."""

    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _verified_immutable_snapshot(
    path: Path,
    payload: bytes,
    *,
    label: str,
) -> ImmutableFileSnapshot:
    try:
        observed, snapshot = read_existing_immutable_file(path, label=label)
    except UnsafeImmutableFileError as exc:
        raise FormalPublicationError(f"unsafe immutable publication file: {label}") from exc
    if observed != payload:
        raise FormalPublicationError(
            f"immutable publication file contains different bytes: {label}"
        )
    return snapshot


def _optional_immutable_snapshot(
    path: Path,
    payload: bytes,
    *,
    label: str,
) -> ImmutableFileSnapshot | None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FormalPublicationError(f"cannot inspect immutable publication file: {label}") from exc
    return _verified_immutable_snapshot(path, payload, label=label)


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _verify_bundle(
    directory: Path,
    expected: Mapping[str, str],
    *,
    allow_incomplete: bool = False,
) -> tuple[str, ...]:
    try:
        before = require_existing_real_directory(directory, label="immutable bundle")
        with os.scandir(directory) as entries:
            observed_names = tuple(sorted(entry.name for entry in entries))
        after_listing = require_existing_real_directory(directory, label="immutable bundle")
    except (OSError, UnsafeImmutableFileError) as exc:
        raise FormalPublicationError("cannot safely inspect immutable bundle") from exc
    if _directory_identity(before) != _directory_identity(after_listing):
        raise FormalPublicationError("immutable bundle directory changed while listing")
    expected_names = tuple(sorted(expected))
    observed_set = set(observed_names)
    expected_set = set(expected_names)
    if observed_names != expected_names:
        incomplete_without_marker = (
            allow_incomplete
            and "bundle_manifest.json" not in observed_set
            and observed_set.issubset(expected_set)
        )
        if not incomplete_without_marker:
            raise FormalPublicationError("immutable bundle file set changed")
    for name in observed_names:
        artifact = directory / name
        try:
            observed, _ = read_existing_immutable_file(
                artifact,
                label=f"immutable bundle artifact {name}",
            )
        except UnsafeImmutableFileError as exc:
            raise FormalPublicationError(f"unsafe immutable bundle artifact: {name}") from exc
        if _digest(observed) != expected[name]:
            raise FormalPublicationError(f"immutable bundle artifact changed: {name}")
    try:
        after_read = require_existing_real_directory(directory, label="immutable bundle")
    except UnsafeImmutableFileError as exc:
        raise FormalPublicationError("immutable bundle directory changed while reading") from exc
    if _directory_identity(before) != _directory_identity(after_read):
        raise FormalPublicationError("immutable bundle directory changed while reading")
    return observed_names


def _finalize_bundle(
    root: Path,
    *,
    bundle_root_relative: str,
    bundle_id: str,
    payloads: Mapping[str, bytes],
) -> tuple[Path, str]:
    bundle_root = _project_path(
        root,
        bundle_root_relative,
        label="formal bundle root",
    )
    _ensure_real_directory_tree(root, bundle_root, label="formal bundle root")
    final = bundle_root / bundle_id
    expected = _artifact_hashes(payloads)
    try:
        os.mkdir(final)
    except FileExistsError:
        try:
            require_existing_real_directory(final, label="immutable bundle address")
        except UnsafeImmutableFileError as exc:
            raise FormalPublicationError(
                "immutable bundle address is not a real directory"
            ) from exc
    except OSError as exc:
        raise FormalPublicationError("cannot create immutable bundle address") from exc
    observed_names = _verify_bundle(final, expected, allow_incomplete=True)
    observed = set(observed_names)
    ordered_names = (
        *sorted(name for name in payloads if name != "bundle_manifest.json"),
        "bundle_manifest.json",
    )
    for name in ordered_names:
        if name in observed:
            continue
        require_existing_real_directory(final, label="immutable bundle address")
        destination = final / name
        temporary = _stage_fixed_payload(destination, payloads[name])
        try:
            _atomic_create_only(temporary, destination)
        finally:
            _safe_unlink_staged(temporary)
        _verified_immutable_snapshot(
            destination,
            payloads[name],
            label=f"immutable bundle artifact {name}",
        )
        observed.add(name)
    _verify_bundle(final, expected)
    return final, expected["bundle_manifest.json"]


def _stage_fixed_payload(destination: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        _safe_unlink_staged(temporary)
        raise
    return temporary


def _atomic_create_only(source: Path, destination: Path) -> bool:
    """Atomically link a fully written sibling file without replacing history.

    Return ``True`` only when this call created the destination.  The caller
    retains and removes ``source`` so the destination can be verified only
    after its temporary sibling link is gone and ``st_nlink == 1``.
    """

    try:
        source_payload, _ = read_existing_immutable_file(
            source,
            label="staged immutable publication payload",
        )
    except UnsafeImmutableFileError as exc:
        raise FormalPublicationError("unsafe staged publication payload") from exc
    try:
        os.link(source, destination)
    except FileExistsError:
        try:
            destination_payload, _ = read_existing_immutable_file(
                destination,
                label="existing fixed publication path",
            )
        except UnsafeImmutableFileError as exc:
            raise FormalPublicationError(
                f"immutable fixed publication path is unsafe or contains different bytes: {destination}"
            ) from exc
        if destination_payload == source_payload:
            return False
        raise FormalPublicationError(
            f"immutable fixed publication path already contains different bytes: {destination}"
        ) from None
    return True


def _publish_fixed_paths(
    root: Path,
    fixed_payloads: Sequence[tuple[str, bytes]],
    *,
    create_file: CreateFile,
) -> tuple[tuple[str, str], ...]:
    entries = tuple(
        (
            relative,
            payload,
            _project_path(root, relative, label="fixed publication path"),
        )
        for relative, payload in fixed_payloads
    )
    destinations = tuple(destination for _, _, destination in entries)
    if len(destinations) != len(set(destinations)):
        raise ValueError("fixed publication destinations must be unique")
    for destination in destinations:
        _ensure_real_directory_tree(
            root,
            destination.parent,
            label="fixed publication parent",
        )
    # Check every historical path before creating anything.  Identical bytes are
    # an idempotent replay; different bytes are an immutable-history conflict.
    historical: set[Path] = set()
    for _, payload, destination in entries:
        if (
            _optional_immutable_snapshot(
                destination,
                payload,
                label="fixed publication path",
            )
            is not None
        ):
            historical.add(destination)

    created: list[tuple[Path, ImmutableFileSnapshot]] = []
    staged: list[Path] = []
    try:
        for _, payload, destination in entries:
            if destination in historical:
                if (
                    _optional_immutable_snapshot(
                        destination,
                        payload,
                        label="historical fixed publication path",
                    )
                    is None
                ):
                    raise FormalPublicationError("historical fixed publication path disappeared")
                continue
            temporary = _stage_fixed_payload(destination, payload)
            staged.append(temporary)
            try:
                was_created = create_file(temporary, destination)
            finally:
                _safe_unlink_staged(temporary)
                staged.remove(temporary)
            snapshot = _verified_immutable_snapshot(
                destination,
                payload,
                label="new fixed publication path",
            )
            if was_created:
                created.append((destination, snapshot))
    except Exception as exc:
        for temporary in staged:
            with suppress(OSError):
                _safe_unlink_staged(temporary)
        rollback_errors: list[Exception] = []
        for destination, snapshot in reversed(created):
            try:
                unlink_existing_immutable_file(
                    destination,
                    expected=snapshot,
                    label="created fixed publication path",
                )
            except (OSError, UnsafeImmutableFileError) as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise FormalPublicationError(
                "fixed publication failed and rollback was incomplete"
            ) from exc
        raise FormalPublicationError("fixed publication transaction rolled back") from exc
    return tuple(sorted((relative, _digest(payload)) for relative, payload in fixed_payloads))


def _publish(
    project_root: Path,
    publication: Stage4PublicationPlan,
    *,
    execution_binding_id: str,
    authorization_id: str,
    model_version: str,
    status: PublicationStatus,
    result: PipelineResult | None,
    failure_code: str | None,
    convergence_audit: CompensatorConvergenceAudit | None,
    additional_local_artifacts: Sequence[AdditionalLocalArtifact],
    create_file: CreateFile,
) -> FormalPublicationReceipt:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory")
    _sha256(execution_binding_id, label="execution_binding_id")
    _sha256(authorization_id, label="authorization_id")
    if not model_version or model_version != model_version.strip():
        raise ValueError("model_version must be a non-empty trimmed string")
    extensions = tuple(additional_local_artifacts)
    if status == "failed" and extensions:
        raise ValueError("failed publication cannot fabricate extension outputs")
    if len({item.relative_path for item in extensions}) != len(extensions):
        raise ValueError("additional local artifact paths must be unique")
    if len({item.bundle_filename for item in extensions}) != len(extensions):
        raise ValueError("additional local bundle filenames must be unique")
    convergence: dict[str, object] | None = None
    if status == "succeeded":
        if result is None or failure_code is not None:
            raise ValueError("successful publication requires one result and no failure code")
        if result.retrospective.protocol.model_version != model_version:
            raise ValueError("publication model version differs from the pipeline result")
        convergence = _convergence_publication_summary(
            cast(CompensatorConvergenceAudit, convergence_audit),
            extensions,
        )
        base_payloads = {
            "anomaly_increment_report.md": _success_report(
                result,
                publication,
                extensions,
                convergence,
            ).encode("utf-8"),
            "anomaly_increment_results.svg": result.static_svg.encode("utf-8"),
            "anomaly_increment_model_card.md": _success_model_card(
                result,
                convergence,
            ).encode("utf-8"),
            "anomaly_increment_dashboard.html": result.interactive_html.encode("utf-8"),
        }
    else:
        if convergence_audit is not None:
            raise ValueError("failed publication cannot contain convergence science values")
        code = _failure_code(cast(str, failure_code))
        if result is not None:
            raise ValueError("failed publication cannot contain a pipeline result")
        base_payloads = {
            "anomaly_increment_report.md": _failure_report(
                model_version=model_version, failure_code=code
            ).encode("utf-8"),
            "anomaly_increment_results.svg": _failure_svg(
                model_version=model_version, failure_code=code
            ).encode("utf-8"),
            "anomaly_increment_model_card.md": _failure_model_card(
                model_version=model_version, failure_code=code
            ).encode("utf-8"),
            "anomaly_increment_dashboard.html": _failure_html(
                model_version=model_version, failure_code=code
            ).encode("utf-8"),
        }
    payloads = dict(base_payloads)
    for item in extensions:
        bundle_name = f"local-{item.bundle_filename}"
        if bundle_name in payloads:
            raise ValueError("additional local bundle filename collides")
        payloads[bundle_name] = item.payload
    for name, payload in payloads.items():
        if name.startswith("local-") or name == "anomaly_increment_dashboard.html":
            _validate_local_payload(name, payload)
        else:
            _validate_public_payload(name, payload)
    semantic_identity = {
        "artifact_sha256": _artifact_hashes(payloads),
        "authorization_id": authorization_id,
        "execution_binding_id": execution_binding_id,
        "failure_code": failure_code,
        "model_version": model_version,
        "result_fingerprint_sha256": (None if result is None else result.result_fingerprint_sha256),
        "convergence_audit_sha256": (
            None if convergence_audit is None else convergence_audit.content_sha256
        ),
        "schema_version": FORMAL_PUBLICATION_SCHEMA_VERSION,
        "status": status,
    }
    bundle_id = _digest(canonical_json_bytes(semantic_identity))
    registry = _registry_payload(
        status=status,
        bundle_id=bundle_id,
        execution_binding_id=execution_binding_id,
        authorization_id=authorization_id,
        model_version=model_version,
        artifact_hashes=_artifact_hashes(payloads),
        publication=publication,
        result=result,
        failure_code=failure_code,
        additional_local_artifact_sha256={
            item.relative_path: _digest(item.payload) for item in extensions
        },
        convergence=convergence,
    )
    _validate_public_payload("anomaly_increment_model_registry.json", registry)
    payloads["anomaly_increment_model_registry.json"] = registry
    manifest = (
        canonical_json_bytes(
            with_content_sha256(
                {
                    "artifact_sha256": _artifact_hashes(payloads),
                    "bundle_id": bundle_id,
                    "schema_version": FORMAL_PUBLICATION_SCHEMA_VERSION,
                    "status": status,
                }
            )
        )
        + b"\n"
    )
    payloads["bundle_manifest.json"] = manifest
    bundle_directory, manifest_sha256 = _finalize_bundle(
        root,
        bundle_root_relative=publication.bundle_root,
        bundle_id=bundle_id,
        payloads=payloads,
    )
    fixed = [
        (publication.public_report, base_payloads["anomaly_increment_report.md"]),
        (publication.public_static_svg, base_payloads["anomaly_increment_results.svg"]),
        (publication.public_model_card, base_payloads["anomaly_increment_model_card.md"]),
        (
            publication.local_interactive_html,
            base_payloads["anomaly_increment_dashboard.html"],
        ),
        *((item.relative_path, item.payload) for item in extensions),
        # The registry is deliberately last: it is the fixed-path completion marker.
        (publication.public_registry, registry),
    ]
    fixed_receipt = _publish_fixed_paths(root, fixed, create_file=create_file)
    return FormalPublicationReceipt(
        status=status,
        bundle_id=bundle_id,
        bundle_directory=bundle_directory,
        registry_sha256=_digest(registry),
        manifest_sha256=manifest_sha256,
        fixed_path_sha256=fixed_receipt,
    )


def publish_successful_formal_result(
    project_root: Path,
    publication: Stage4PublicationPlan,
    *,
    execution_binding_id: str,
    authorization_id: str,
    result: PipelineResult,
    convergence_audit: CompensatorConvergenceAudit,
    additional_local_artifacts: Sequence[AdditionalLocalArtifact] = (),
    create_file: CreateFile = _atomic_create_only,
) -> FormalPublicationReceipt:
    """Publish a real pipeline result; the registry is the last fixed path replaced."""

    if not isinstance(result, PipelineResult):
        raise TypeError("successful formal publication requires PipelineResult")
    return _publish(
        project_root,
        publication,
        execution_binding_id=execution_binding_id,
        authorization_id=authorization_id,
        model_version=result.retrospective.protocol.model_version,
        status="succeeded",
        result=result,
        failure_code=None,
        convergence_audit=convergence_audit,
        additional_local_artifacts=additional_local_artifacts,
        create_file=create_file,
    )


def publish_failed_formal_result(
    project_root: Path,
    publication: Stage4PublicationPlan,
    *,
    execution_binding_id: str,
    authorization_id: str,
    model_version: str,
    failure_code: str,
    create_file: CreateFile = _atomic_create_only,
) -> FormalPublicationReceipt:
    """Publish explicit value-free failure pages without fake points or curves."""

    return _publish(
        project_root,
        publication,
        execution_binding_id=execution_binding_id,
        authorization_id=authorization_id,
        model_version=model_version,
        status="failed",
        result=None,
        failure_code=_failure_code(failure_code),
        convergence_audit=None,
        additional_local_artifacts=(),
        create_file=create_file,
    )


__all__ = [
    "FORMAL_BUNDLE_ROOT",
    "FORMAL_PUBLICATION_SCHEMA_VERSION",
    "AdditionalLocalArtifact",
    "FormalPublicationError",
    "FormalPublicationReceipt",
    "PublicationStatus",
    "publish_failed_formal_result",
    "publish_successful_formal_result",
]
