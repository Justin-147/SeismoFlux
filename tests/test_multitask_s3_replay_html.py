"""Synthetic-data checks for the read-only, offline S3 replay document."""

# Expected interface copy is Chinese and intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import re

import pytest

from seismoflux.multitask_s3.replay_html import render_replay_html


def payload() -> dict:
    return {
        "version": "synthetic-v1",
        "local_only": True,
        "notes": ["合成测试，不是真实地震成绩。"],
        "budgets": [300000, 600000],
        "contrasts": [
            {
                "id": "CAT_DYN_minus_CAT_COV",
                "candidate": "CAT_DYN",
                "reference": "CAT_COV",
                "label": "动态异常对覆盖模型",
            }
        ],
        "variants": {"CAT_COV": "覆盖模型", "CAT_DYN": "动态异常"},
        "geometry": {
            "bounds": [0, 0, 1000, 1000],
            "cells": [[[[[0, 0], [1000, 0], [1000, 1000], [0, 1000], [0, 0]]]]],
        },
        "events": {},
        "frames": [
            {
                "id": "empty-frame",
                "fold_id": "A_DEV_2023_2024",
                "horizon_days": 30,
                "issue_time_utc": "2023-08-03T00:00:00Z",
                "target_end_utc": "2023-09-02T00:00:00Z",
                "primary_nonoverlap": True,
                "models": {},
                "bands": {
                    "Ms5_6": {
                        "event_ids": [],
                        "anchor_mask": [],
                        "outcomes": {},
                        "counts": {},
                    }
                },
            }
        ],
        "provenance": {"source": "合成输入"},
        "unevaluable": [
            {"fold_id": "A_DEV_2023_2024", "horizon_days": 365, "reason": "样本未成熟"}
        ],
    }


def embedded_payload(html: str) -> dict:
    match = re.search(
        r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S
    )
    assert match
    return json.loads(match.group(1))


def test_complete_document_retains_empty_frames_and_unevaluable_tasks() -> None:
    data = payload()
    html = render_replay_html(data)
    assert html.startswith("<!doctype html>")
    assert html.endswith("</html>\n")
    assert embedded_payload(html) == data
    assert "NA／不可评价" in html
    assert "空窗口仍保留在起报列表中" in html
    assert "默认第一合法起报，不挑选命中案例" in html
    assert "较晚历史开发；非独立测试、非真实前瞻" in html


def test_script_significant_text_is_safely_escaped_and_round_trips() -> None:
    data = payload()
    attack = '</script><script>alert("x")</script>&\u2028\u2029'
    data["notes"].append(attack)
    data["variants"]["CAT_DYN"] = attack
    html = render_replay_html(data)
    assert attack not in html
    assert "\\u003c/script\\u003e" in html
    assert "\\u2028\\u2029" in html
    assert embedded_payload(html) == data
    assert html.count("</script>") == 2
    assert "innerHTML" not in html
    assert "document.write" not in html


def test_all_controls_and_literal_dom_references_exist() -> None:
    html = render_replay_html(payload())
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    required = {
        "fold-select", "horizon-select", "band-select", "axis-select",
        "contrast-select", "budget-select", "mode-select", "event-view-select",
        "frame-select", "frame-prev", "frame-next", "reference-map", "candidate-map",
        "events-body", "counts-body", "event-detail", "unevaluable-list",
        "focus-event", "national-view", "viewport-note",
    }
    assert required <= set(ids)
    literal_refs = set(re.findall(r'\$\("([^"]+)"\)', html))
    literal_refs |= set(re.findall(r'document\.getElementById\("([^"]+)"\)', html))
    assert literal_refs <= set(ids)
    # Dynamically addressed paired map headings are declared for both roles.
    for role in ("reference", "candidate"):
        for suffix in ("title", "area", "map"):
            assert f"{role}-{suffix}" in ids


def test_no_network_dependencies_or_mutating_ui() -> None:
    html = render_replay_html(payload())
    assert not re.search(r'<(?:script|img|link|iframe)\b[^>]*(?:src|href)\s*=', html, re.I)
    for token in ("fetch(", "XMLHttpRequest", "WebSocket", "https://", "http://"):
        assert token not in html
    assert "connect-src 'none'" in html
    assert "不联网、不训练、不修改参数、不发布预测" in html
    assert "主起报与全部报告有重叠差别，回放窗口不是独立地震样本" in html
    assert "报警网格并非最终≤10区域产品" in html
    assert "70km仅辅助判定，图上实际报警面积不扩张" in html
    assert "查看所选震例附近" in html
    assert "报警面积和保存的命中判定均未改变" in html


def test_explicit_local_only_and_finite_json_required() -> None:
    for value in (False, None, 1, "true"):
        data = payload()
        data["local_only"] = value
        with pytest.raises(ValueError, match="local_only"):
            render_replay_html(data)
    data = payload()
    data["provenance"]["invalid"] = float("nan")
    with pytest.raises(ValueError):
        render_replay_html(data)
