# ruff: noqa: RUF001
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "render_stage2s_data_method_causal_timeline.py"
CONFIG_RELATIVE_PATH = Path("configs/causal_seismicity_screen.yaml")
FOLD_MANIFEST_RELATIVE_PATH = Path("data/manifests/causal_seismicity_screen_fold_manifest.json")
OUTPUT_PATH = ROOT / "docs" / "stage2s_data_method_causal_timeline.svg"


def _run_renderer(
    *,
    repository_root: Path,
    output_path: Path,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--repository-root",
        str(repository_root),
        "--output",
        str(output_path),
    ]
    if check:
        command.append("--check")
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_timeline_is_deterministic_and_matches_committed_svg(tmp_path: Path) -> None:
    first_path = tmp_path / "first.svg"
    second_path = tmp_path / "second.svg"
    first_run = _run_renderer(repository_root=ROOT, output_path=first_path)
    second_run = _run_renderer(repository_root=ROOT, output_path=second_path)

    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes() == OUTPUT_PATH.read_bytes()


def test_timeline_can_be_rebuilt_with_only_two_target_blind_inputs(
    tmp_path: Path,
) -> None:
    for relative_path in (CONFIG_RELATIVE_PATH, FOLD_MANIFEST_RELATIVE_PATH):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative_path).read_bytes())

    output = tmp_path / "timeline.svg"
    completed = _run_renderer(repository_root=tmp_path, output_path=output)

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == OUTPUT_PATH.read_bytes()


def test_timeline_is_well_formed_and_contains_all_causal_boundaries() -> None:
    payload = OUTPUT_PATH.read_bytes()
    document = ElementTree.fromstring(payload)
    text = " ".join(fragment.strip() for fragment in document.itertext() if fragment.strip())
    tag_names = {element.tag.rsplit("}", maxsplit=1)[-1] for element in document.iter()}

    assert document.tag.endswith("svg")
    assert not {"image", "script", "foreignObject", "use"} & tag_names
    assert not any(
        attribute.endswith("href") for element in document.iter() for attribute in element.attrib
    )
    for required in (
        "S0｜长期背景",
        "R｜最近窗口",
        "RP｜紧邻过去对照",
        "(T-30d,T]",
        "(T-60d,T-30d]",
        "available_at ≤ T",
        "Fold 1",
        "Fold 2",
        "Fold 3",
        "fold fit receipt",
        "issue prediction seal",
        "fold prediction seal",
        "master prediction seal",
        "assessment / scoring",
        "600,000 km²",
        "目标读取 0",
        "复用开发期",
        "不是当前预测",
        "不得解释为绝对发震概率",
    ):
        assert required in text


def test_timeline_embeds_no_target_result_geometry_or_machine_path() -> None:
    payload = OUTPUT_PATH.read_text(encoding="utf-8")
    forbidden_fragments = (
        "data/processed",
        "data\\processed",
        "D:\\",
        "query_x_m",
        "query_y_m",
        "longitude",
        "latitude",
        "event_id",
        "construction_zone_id",
        "<image",
        "xlink:href",
    )

    assert all(fragment not in payload for fragment in forbidden_fragments)
    assert "本图无数值" in payload
    assert "不含目标成绩" in payload


def test_check_mode_detects_stale_output_without_rewriting(tmp_path: Path) -> None:
    output = tmp_path / "timeline.svg"
    generated = _run_renderer(repository_root=ROOT, output_path=output)
    assert generated.returncode == 0, generated.stderr
    checked = _run_renderer(repository_root=ROOT, output_path=output, check=True)
    assert checked.returncode == 0, checked.stderr
    output.write_text("stale", encoding="utf-8")

    stale = _run_renderer(repository_root=ROOT, output_path=output, check=True)
    assert stale.returncode == 1
    assert "timeline output is stale" in stale.stderr
    assert output.read_text(encoding="utf-8") == "stale"


def test_cli_check_matches_committed_svg() -> None:
    completed = _run_renderer(
        repository_root=ROOT,
        output_path=OUTPUT_PATH,
        check=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.startswith("verified ")
