# ruff: noqa: RUF001
"""Render the deterministic, target-independent Stage 2S causal timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE_PATH = Path("configs/causal_seismicity_screen.yaml")
FOLD_MANIFEST_RELATIVE_PATH = Path("data/manifests/causal_seismicity_screen_fold_manifest.json")
OUTPUT_RELATIVE_PATH = Path("docs/stage2s_data_method_causal_timeline.svg")
EXPECTED_CONFIG_SHA256 = "a85df78348c0f033444db4c9e3edc81b70ef436da3b108139feab39cd49d8c42"
EXPECTED_FOLD_MANIFEST_SHA256 = "c3e2444e8892addd03d4c57526c007e2a861137dac50d5abe2e53bac004456e6"


class TimelineContractError(ValueError):
    """Raised when the frozen target-independent inputs do not match the protocol."""


@dataclass(frozen=True, slots=True)
class FoldTimeline:
    fold_index: int
    fit_issue_count: int
    fit_end_utc: str
    assessment_start_local: str
    assessment_end_local: str


@dataclass(frozen=True, slots=True)
class TimelineSpec:
    protocol_version: str
    experiment_id: str
    protocol_tag: str
    code_tag: str
    config_sha256: str
    fold_manifest_sha256: str
    s0_fit_end_utc: str
    bandwidth_km: float
    recent_interval: str
    recent_available_at: str
    preceding_interval: str
    preceding_available_at: str
    s1_formula: str
    sp_formula: str
    alpha_r_id: str
    alpha_p_id: str
    shared_rate_id: str
    horizons_days: tuple[int, int, int]
    alarm_budget_km2: float
    folds: tuple[FoldTimeline, FoldTimeline, FoldTimeline]
    target_reads_at_freeze: int


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TimelineContractError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TimelineContractError(f"{name} must be a sequence")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TimelineContractError(f"{name} must be non-empty text")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TimelineContractError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TimelineContractError(f"{name} must be numeric")
    return float(value)


def _require_equal(value: object, expected: object, name: str) -> None:
    if value != expected:
        raise TimelineContractError(f"{name} must equal {expected!r}, got {value!r}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_yaml_bytes(payload: bytes) -> Mapping[str, Any]:
    try:
        decoded = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise TimelineContractError(f"cannot decode frozen YAML: {exc}") from exc
    return _mapping(decoded, "config")


def _load_json_bytes(payload: bytes) -> Mapping[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TimelineContractError(f"cannot decode frozen fold manifest: {exc}") from exc
    return _mapping(decoded, "fold_manifest")


def parse_timeline_spec(config_bytes: bytes, fold_manifest_bytes: bytes) -> TimelineSpec:
    """Parse only the two frozen, target-independent Stage 2S inputs."""

    config_sha256 = _sha256(config_bytes)
    fold_sha256 = _sha256(fold_manifest_bytes)
    _require_equal(config_sha256, EXPECTED_CONFIG_SHA256, "config_sha256")
    _require_equal(
        fold_sha256,
        EXPECTED_FOLD_MANIFEST_SHA256,
        "fold_manifest_sha256",
    )

    config = _load_yaml_bytes(config_bytes)
    manifest = _load_json_bytes(fold_manifest_bytes)
    protocol_version = _text(config.get("protocol_version"), "protocol_version")
    experiment_id = _text(config.get("experiment_id"), "experiment_id")
    _require_equal(protocol_version, "0.2.3", "protocol_version")
    _require_equal(
        experiment_id,
        "stage2s-causal-seismicity-development-v1",
        "experiment_id",
    )
    _require_equal(manifest.get("protocol_version"), protocol_version, "manifest.protocol_version")
    _require_equal(manifest.get("experiment_id"), experiment_id, "manifest.experiment_id")
    _require_equal(
        manifest.get("status"),
        "target_blind_calendar_only_no_execution_or_scoring_authority",
        "manifest.status",
    )

    governance = _mapping(config.get("governance"), "governance")
    protocol_tag = _text(governance.get("protocol_tag"), "governance.protocol_tag")
    code_tag = _text(governance.get("expected_code_tag"), "governance.expected_code_tag")
    _require_equal(
        protocol_tag,
        "v0.2.3-causal-seismicity-screen-protocol",
        "governance.protocol_tag",
    )
    _require_equal(
        code_tag,
        "v0.2.3-causal-seismicity-screen-code",
        "governance.expected_code_tag",
    )
    _require_equal(
        governance.get("development_target_read_count_at_preregistration"),
        0,
        "governance.development_target_read_count_at_preregistration",
    )
    _require_equal(
        governance.get("independent_validation_target_read_count_at_preregistration"),
        0,
        "governance.independent_validation_target_read_count_at_preregistration",
    )
    _require_equal(
        governance.get("locked_test_read_count_at_preregistration"),
        0,
        "governance.locked_test_read_count_at_preregistration",
    )

    allowed_models = _mapping(config.get("allowed_models"), "allowed_models")
    _require_equal(
        tuple(_sequence(allowed_models.get("exact_order"), "allowed_models.exact_order")),
        ("S0", "S1", "SP"),
        "allowed_models.exact_order",
    )
    long_term = _mapping(config.get("long_term_background"), "long_term_background")
    recent = _mapping(config.get("recent_seismicity"), "recent_seismicity")
    recent_window = _mapping(recent.get("most_recent_window"), "recent.most_recent_window")
    preceding_window = _mapping(
        recent.get("preceding_window_control"),
        "recent.preceding_window_control",
    )
    mixtures = _mapping(config.get("mixtures"), "mixtures")
    s1 = _mapping(mixtures.get("S1"), "mixtures.S1")
    sp = _mapping(mixtures.get("SP"), "mixtures.SP")
    shared_rate = _mapping(
        config.get("shared_rate_and_compensator"),
        "shared_rate_and_compensator",
    )
    calendar = _mapping(config.get("calendar_and_targets"), "calendar_and_targets")
    horizons_raw = _sequence(
        calendar.get("assessment_horizons_days"),
        "calendar.assessment_horizons_days",
    )
    horizons = tuple(
        _integer(value, f"calendar.assessment_horizons_days[{index}]")
        for index, value in enumerate(horizons_raw)
    )
    _require_equal(horizons, (7, 30, 90), "calendar.assessment_horizons_days")
    alarm = _mapping(config.get("alarm_area"), "alarm_area")

    source_contracts = _mapping(config.get("source_contracts"), "source_contracts")
    rolling_manifest = _mapping(
        source_contracts.get("rolling_fold_manifest"),
        "source_contracts.rolling_fold_manifest",
    )
    _require_equal(
        rolling_manifest.get("sha256"),
        fold_sha256,
        "source_contracts.rolling_fold_manifest.sha256",
    )

    issue_semantics = _mapping(manifest.get("issue_semantics"), "manifest.issue_semantics")
    _require_equal(
        tuple(
            _integer(value, f"manifest.issue_semantics.assessment_horizons_days[{index}]")
            for index, value in enumerate(
                _sequence(
                    issue_semantics.get("assessment_horizons_days"),
                    "manifest.issue_semantics.assessment_horizons_days",
                )
            )
        ),
        horizons,
        "manifest.issue_semantics.assessment_horizons_days",
    )
    security = _mapping(manifest.get("security"), "manifest.security")
    _require_equal(
        security.get("contains_target_ids_coordinates_scores_hits_or_model_results"),
        False,
        "manifest.security.contains_target_ids_coordinates_scores_hits_or_model_results",
    )
    _require_equal(
        security.get("development_target_read_authorized"),
        False,
        "manifest.security.development_target_read_authorized",
    )
    _require_equal(
        security.get("independent_validation_or_locked_test_authorized"),
        False,
        "manifest.security.independent_validation_or_locked_test_authorized",
    )

    fold_values = _sequence(manifest.get("folds"), "manifest.folds")
    parsed_folds: list[FoldTimeline] = []
    for expected_index, raw_fold in enumerate(fold_values, start=1):
        fold = _mapping(raw_fold, f"manifest.folds[{expected_index - 1}]")
        fold_index = _integer(fold.get("fold_index"), f"fold_{expected_index}.fold_index")
        _require_equal(fold_index, expected_index, f"fold_{expected_index}.fold_index")
        fit_issue_dates = _sequence(
            fold.get("fit_issue_dates_local_h007"),
            f"fold_{fold_index}.fit_issue_dates_local_h007",
        )
        if not fit_issue_dates:
            raise TimelineContractError(f"fold_{fold_index} must have fit issue dates")
        assessment = _mapping(
            fold.get("assessment_band"),
            f"fold_{fold_index}.assessment_band",
        )
        parsed_folds.append(
            FoldTimeline(
                fold_index=fold_index,
                fit_issue_count=len(fit_issue_dates),
                fit_end_utc=_text(
                    fold.get("fit_target_end_inclusive_utc"),
                    f"fold_{fold_index}.fit_target_end_inclusive_utc",
                ),
                assessment_start_local=_text(
                    assessment.get("start_exclusive_local"),
                    f"fold_{fold_index}.assessment_start_local",
                ),
                assessment_end_local=_text(
                    assessment.get("end_inclusive_local"),
                    f"fold_{fold_index}.assessment_end_local",
                ),
            )
        )
    _require_equal(len(parsed_folds), 3, "manifest.fold_count")

    execution = _mapping(config.get("execution_control"), "execution_control")
    single_open = _mapping(execution.get("catalog_single_open"), "execution.catalog_single_open")
    seal_chain = _mapping(
        single_open.get("prediction_seal_chain"),
        "execution.catalog_single_open.prediction_seal_chain",
    )
    _require_equal(
        tuple(_sequence(seal_chain.get("fold_order"), "seal_chain.fold_order")),
        (1, 2, 3),
        "seal_chain.fold_order",
    )
    _require_equal(
        seal_chain.get("master_required_before_assessment_target_view_and_scoring"),
        True,
        "seal_chain.master_required_before_assessment_target_view_and_scoring",
    )

    return TimelineSpec(
        protocol_version=protocol_version,
        experiment_id=experiment_id,
        protocol_tag=protocol_tag,
        code_tag=code_tag,
        config_sha256=config_sha256,
        fold_manifest_sha256=fold_sha256,
        s0_fit_end_utc=_text(long_term.get("fit_end_utc"), "long_term.fit_end_utc"),
        bandwidth_km=_number(long_term.get("bandwidth_km"), "long_term.bandwidth_km"),
        recent_interval=_text(recent_window.get("origin_interval"), "recent.origin_interval"),
        recent_available_at="available_at ≤ T",
        preceding_interval=_text(
            preceding_window.get("origin_interval"),
            "preceding.origin_interval",
        ),
        preceding_available_at="available_at ≤ T−30d",
        s1_formula=_text(s1.get("formula"), "mixtures.S1.formula"),
        sp_formula=_text(sp.get("formula"), "mixtures.SP.formula"),
        alpha_r_id=_text(s1.get("alpha_id"), "mixtures.S1.alpha_id"),
        alpha_p_id=_text(sp.get("alpha_id"), "mixtures.SP.alpha_id"),
        shared_rate_id=_text(shared_rate.get("rate_id"), "shared_rate.rate_id"),
        horizons_days=(horizons[0], horizons[1], horizons[2]),
        alarm_budget_km2=_number(alarm.get("primary_budget_km2"), "alarm.primary_budget_km2"),
        folds=(parsed_folds[0], parsed_folds[1], parsed_folds[2]),
        target_reads_at_freeze=0,
    )


def load_repository_timeline_spec(repository_root: Path) -> TimelineSpec:
    """Load exactly the frozen YAML and fold manifest from a repository root."""

    config_bytes = (repository_root / CONFIG_RELATIVE_PATH).read_bytes()
    fold_manifest_bytes = (repository_root / FOLD_MANIFEST_RELATIVE_PATH).read_bytes()
    return parse_timeline_spec(config_bytes, fold_manifest_bytes)


def _svg_text(
    x: int,
    y: int,
    value: str,
    *,
    css_class: str = "body",
    anchor: str | None = None,
) -> str:
    anchor_attribute = f' text-anchor="{escape(anchor, quote=True)}"' if anchor else ""
    return (
        f'<text x="{x}" y="{y}" class="{escape(css_class, quote=True)}"'
        f"{anchor_attribute}>{escape(value)}</text>"
    )


def _svg_multiline(
    x: int,
    y: int,
    lines: Sequence[str],
    *,
    css_class: str = "body",
    line_height: int = 25,
) -> list[str]:
    return [
        _svg_text(x, y + index * line_height, line, css_class=css_class)
        for index, line in enumerate(lines)
    ]


def _svg_box(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    css_class: str,
    radius: int = 16,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" class="{escape(css_class, quote=True)}"/>'
    )


def _svg_arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        'class="arrow" marker-end="url(#arrowhead)"/>'
    )


def _wrapped_tag(value: str) -> tuple[str, str]:
    split_at = value.rfind("screen-")
    if split_at <= 0:
        raise TimelineContractError(f"cannot wrap frozen tag {value!r}")
    return value[:split_at], value[split_at:]


def render_timeline_svg(spec: TimelineSpec) -> bytes:
    """Return stable SVG bytes without target rows, scores, or runtime state."""

    horizons = "/".join(str(value) for value in spec.horizons_days)
    area = f"{spec.alarm_budget_km2:,.0f}"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1120" '
        'role="img" aria-labelledby="title description">',
        '<title id="title">Stage 2S 数据—方法—因果时间线</title>',
        '<desc id="description">目标读取前冻结的数据、模型、折序与封印边界；不含目标成绩。</desc>',
        "<defs>",
        '<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" '
        'orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#486581"/></marker>',
        "</defs>",
        "<style>",
        "text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#102a43}",
        ".title{font-size:34px;font-weight:700}.subtitle{font-size:17px;fill:#486581}",
        ".section{font-size:20px;font-weight:700}.cardTitle{font-size:19px;font-weight:700}",
        ".body{font-size:15px}.small{font-size:13px;fill:#486581}"
        ".badgeText{font-size:14px;font-weight:700;fill:#ffffff}",
        ".panel{fill:#ffffff;stroke:#bcccdc;stroke-width:1.5}"
        ".blue{fill:#eaf2ff;stroke:#2f80ed;stroke-width:2}"
        ".green{fill:#e8f7f1;stroke:#2d9d78;stroke-width:2}"
        ".orange{fill:#fff3e7;stroke:#f2994a;stroke-width:2}"
        ".purple{fill:#f1edff;stroke:#7b61ff;stroke-width:2}"
        ".barrier{fill:#fff0f0;stroke:#c0392b;stroke-width:2.5}"
        ".badgeBlue{fill:#2f80ed}.badgeGreen{fill:#2d9d78}.badgeGray{fill:#627d98}",
        ".arrow{stroke:#486581;stroke-width:2;fill:none}",
        ".divider{stroke:#d9e2ec;stroke-width:1}",
        "</style>",
        '<rect width="1600" height="1120" fill="#f7f9fc"/>',
        _svg_text(60, 60, "Stage 2S 数据—方法—因果时间线", css_class="title"),
        _svg_text(
            60,
            92,
            "代码冻结前的目标无关交付｜复用开发期历史回溯｜不含目标成绩",
            css_class="subtitle",
        ),
        _svg_box(1125, 35, 190, 38, css_class="badgeBlue", radius=19),
        _svg_text(
            1220,
            60,
            f"协议 {spec.protocol_version}",
            css_class="badgeText",
            anchor="middle",
        ),
        _svg_box(1330, 35, 210, 38, css_class="badgeGreen", radius=19),
        _svg_text(
            1435,
            60,
            f"目标读取 {spec.target_reads_at_freeze}",
            css_class="badgeText",
            anchor="middle",
        ),
        _svg_text(60, 140, "治理顺序：只有远端代码标签后才允许非目标预检", css_class="section"),
    ]

    governance_boxes = (
        (60, 165, 250, "协议标签", _wrapped_tag(spec.protocol_tag)),
        (355, 165, 250, "代码标签", _wrapped_tag(spec.code_tag)),
        (650, 165, 250, "非目标预检", ("study-area / grid", "cell-zone mapping")),
        (945, 165, 250, "唯一 attempt", ("O_EXCL: 不存在", "→ 注册一次")),
        (1240, 165, 300, "唯一 target read", ("O_EXCL 后才可读", "同一目录字节流")),
    )
    for x, y, width, heading, details in governance_boxes:
        lines.append(_svg_box(x, y, width, 92, css_class="panel"))
        lines.append(_svg_text(x + 18, y + 31, heading, css_class="cardTitle"))
        lines.extend(
            _svg_multiline(
                x + 18,
                y + 56,
                details,
                css_class="small",
                line_height=18,
            )
        )
    for governance_left, governance_right in pairwise(governance_boxes):
        lines.append(
            _svg_arrow(
                governance_left[0] + governance_left[2],
                211,
                governance_right[0] - 10,
                211,
            )
        )

    lines.extend(
        [
            _svg_text(60, 310, "冻结模型与因果窗口", css_class="section"),
            _svg_box(60, 335, 305, 150, css_class="blue"),
            _svg_text(82, 370, "S0｜长期背景", css_class="cardTitle"),
        ]
    )
    lines.extend(
        _svg_multiline(
            82,
            402,
            (
                "fold_4；训练截止",
                spec.s0_fit_end_utc,
                f"{spec.bandwidth_km:g} km 等面积 Gaussian KDE",
                "授权后从同一目录 bytes 精确重物化",
            ),
            css_class="body",
            line_height=23,
        )
    )
    lines.extend(
        [
            _svg_box(390, 335, 305, 150, css_class="green"),
            _svg_text(412, 370, "R｜最近窗口", css_class="cardTitle"),
        ]
    )
    lines.extend(
        _svg_multiline(
            412,
            402,
            (
                f"M4+；{spec.recent_interval}",
                spec.recent_available_at,
                f"{spec.bandwidth_km:g} km；等权；无时间衰减",
            ),
            css_class="body",
        )
    )
    lines.extend(
        [
            _svg_box(720, 335, 305, 150, css_class="orange"),
            _svg_text(742, 370, "RP｜紧邻过去对照", css_class="cardTitle"),
        ]
    )
    lines.extend(
        _svg_multiline(
            742,
            402,
            (
                f"M4+；{spec.preceding_interval}",
                spec.preceding_available_at,
                f"{spec.bandwidth_km:g} km；与 R 同结构",
            ),
            css_class="body",
        )
    )
    lines.extend(
        [
            _svg_box(1050, 335, 490, 150, css_class="purple"),
            _svg_text(1072, 370, "三模型共享折内 M5–6 日率", css_class="cardTitle"),
        ]
    )
    lines.extend(
        _svg_multiline(
            1072,
            402,
            (
                f"S1 = {spec.s1_formula}",
                f"SP = {spec.sp_formula}",
                f"{spec.alpha_r_id} / {spec.alpha_p_id} / {spec.shared_rate_id}：本图无数值",
            ),
            css_class="body",
        )
    )

    lines.append(
        _svg_text(
            60,
            535,
            f"目标盲日历：fold 1 → 2 → 3；每折权重共享 {horizons} 天；完整前缀 ≤ {area} km²",
            css_class="section",
        )
    )
    fold_x_positions = (60, 565, 1070)
    for x, fold in zip(fold_x_positions, spec.folds, strict=True):
        lines.append(_svg_box(x, 560, 470, 170, css_class="panel"))
        lines.append(_svg_text(x + 22, 595, f"Fold {fold.fold_index}", css_class="cardTitle"))
        lines.extend(
            _svg_multiline(
                x + 22,
                627,
                (
                    f"非重叠 h007 fit 起报：{fold.fit_issue_count} 个",
                    f"fit 标签截止：{fold.fit_end_utc}",
                    "assessment band（目标盲日期）：",
                    f"({fold.assessment_start_local}, {fold.assessment_end_local}]",
                ),
                css_class="body",
                line_height=25,
            )
        )
    lines.extend(
        [
            _svg_text(60, 780, "不可改写的因果封印链", css_class="section"),
            _svg_box(60, 805, 270, 86, css_class="blue"),
            _svg_text(195, 839, "fold fit receipt", css_class="cardTitle", anchor="middle"),
            _svg_text(
                195,
                866,
                "先冻结 alpha / rate / fit view",
                css_class="small",
                anchor="middle",
            ),
            _svg_box(385, 805, 270, 86, css_class="green"),
            _svg_text(520, 839, "issue prediction seal", css_class="cardTitle", anchor="middle"),
            _svg_text(520, 866, "按 T 递增；三窗口一起写", css_class="small", anchor="middle"),
            _svg_box(710, 805, 270, 86, css_class="orange"),
            _svg_text(845, 839, "fold prediction seal", css_class="cardTitle", anchor="middle"),
            _svg_text(845, 866, "前一折完成后才开放后一折", css_class="small", anchor="middle"),
            _svg_box(1035, 805, 270, 86, css_class="purple"),
            _svg_text(1170, 839, "master prediction seal", css_class="cardTitle", anchor="middle"),
            _svg_text(1170, 866, "绑定 fold 1 / 2 / 3", css_class="small", anchor="middle"),
            _svg_box(1360, 805, 180, 86, css_class="barrier"),
            _svg_text(1450, 839, "才开放", css_class="cardTitle", anchor="middle"),
            _svg_text(1450, 866, "assessment / scoring", css_class="small", anchor="middle"),
        ]
    )
    seal_boxes = ((60, 270), (385, 270), (710, 270), (1035, 270), (1360, 180))
    for seal_left, seal_right in pairwise(seal_boxes):
        lines.append(_svg_arrow(seal_left[0] + seal_left[1], 848, seal_right[0] - 10, 848))

    lines.extend(
        [
            _svg_box(60, 930, 1480, 105, css_class="barrier"),
            _svg_text(85, 965, "解释边界", css_class="cardTitle"),
        ]
    )
    lines.extend(
        _svg_multiline(
            85,
            994,
            (
                "这是复用开发期的历史方法图，不是当前预测、独立验证、锁定测试或 G8 前瞻证据。",
                "只表达相对强度与固定面积排序；不得解释为绝对发震概率，也不判断未来总数或具体日期。",
            ),
            css_class="body",
            line_height=25,
        )
    )
    lines.extend(
        [
            '<line x1="60" y1="1062" x2="1540" y2="1062" class="divider"/>',
            _svg_text(
                60,
                1087,
                f"config SHA-256  {spec.config_sha256}",
                css_class="small",
            ),
            _svg_text(
                820,
                1087,
                f"fold manifest SHA-256  {spec.fold_manifest_sha256}",
                css_class="small",
            ),
            _svg_text(
                60,
                1107,
                f"experiment_id  {spec.experiment_id}",
                css_class="small",
            ),
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_repository_timeline(repository_root: Path) -> bytes:
    return render_timeline_svg(load_repository_timeline_spec(repository_root))


def write_or_check_timeline(
    *,
    repository_root: Path,
    output_path: Path,
    check: bool,
) -> None:
    payload = render_repository_timeline(repository_root)
    if check:
        try:
            existing = output_path.read_bytes()
        except FileNotFoundError as exc:
            raise TimelineContractError(f"timeline output does not exist: {output_path}") from exc
        if existing != payload:
            raise TimelineContractError(
                f"timeline output is stale: {output_path}; regenerate without --check"
            )
        return
    output_path.write_bytes(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the deterministic target-independent Stage 2S timeline."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed SVG differs from deterministic regeneration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / OUTPUT_RELATIVE_PATH,
        help="SVG output path; frozen input paths are not configurable",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root; only the two fixed, hash-locked relative inputs are read",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        write_or_check_timeline(
            repository_root=args.repository_root,
            output_path=args.output,
            check=bool(args.check),
        )
    except (OSError, TimelineContractError) as exc:
        print(f"stage2s timeline error: {exc}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "wrote"
    print(f"{action} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
