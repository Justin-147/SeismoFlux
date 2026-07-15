from __future__ import annotations

import hashlib
import json
from pathlib import Path

_FROZEN_V020_FILES = {
    Path("configs/background.yaml"): (
        "d845f7aea4b8cc50f264560a074e93c91efc5cf026a1425f04f1c160d595e24b"
    ),
    Path("data/manifests/background_model_registry.json"): (
        "34ff1f342912e1bd4d6183b4199833c959596c41109f7e613cead57792481be3"
    ),
    Path("docs/background_baseline_report.md"): (
        "358a82299c802c2e43966f0ab7b3d31815c191dc81db5fdbc97c2ff5ad6e1f6c"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(item) for item in value))
    return set()


def test_public_local_support_manifest_contains_no_derived_geometry_bytes() -> None:
    path = Path("data/manifests/background_local_support_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    forbidden_geometry_fields = {
        "clipped_geometry_wkb_hex",
        "coordinates",
        "geometry",
        "wkb",
        "wkb_hex",
    }

    assert _all_mapping_keys(payload).isdisjoint(forbidden_geometry_fields)


def test_v020_protocol_registry_and_report_remain_byte_exact() -> None:
    assert {path: _sha256(path) for path in _FROZEN_V020_FILES} == _FROZEN_V020_FILES


def test_local_support_amendment_preserves_the_original_negative_result() -> None:
    blueprint = Path("SEISMOFLUX_IMPLEMENTATION_HANDOFF.md").read_text(encoding="utf-8")
    protocol = Path("docs/background_local_support_protocol.md").read_text(encoding="utf-8")

    assert "阶段2R" in blueprint and "局部支持域修订" in blueprint
    assert "v0.2.0` 全域协议及其可信负结果保持不可变" in blueprint
    assert "门控名称" in protocol and "`G1-LS`" in protocol
    assert "没有生成任何背景模型分数" in protocol
    assert "锁定测试未运行" in protocol
    assert "至少95%" in protocol
    assert "按未覆盖计算" in protocol
