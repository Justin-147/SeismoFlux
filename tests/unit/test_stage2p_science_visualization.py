"""Static and offline-interactive checks for the Stage 2P synthetic figures."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest

from seismoflux.stage2p.synthetic_experiment import run_all_synthetic_scenarios
from seismoflux.stage2p.visualization import (
    RenderedArtifact,
    render_artifacts,
    write_artifacts,
)


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.start_tags: list[str] = []
        self.end_tags: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.start_tags.append(tag)
        for name, value in attrs:
            if name == "id" and value is not None:
                self.ids.add(value)

    def handle_endtag(self, tag: str) -> None:
        self.end_tags.append(tag)


@pytest.fixture(scope="module")
def rendered() -> tuple[RenderedArtifact, ...]:
    results = run_all_synthetic_scenarios(bootstrap_replicates=100)
    return render_artifacts(results)


def test_static_figures_are_parseable_deterministic_and_plainly_labeled(
    rendered: tuple[RenderedArtifact, ...],
) -> None:
    results = run_all_synthetic_scenarios(bootstrap_replicates=100)
    repeated = render_artifacts(results)

    assert [artifact.name for artifact in rendered] == [
        "recent_activity_predictive.svg",
        "no_recent_signal.svg",
        "recent_activity_misleading.svg",
        "scenario_comparison.svg",
        "stage2p_science_mvp_explorer.html",
        "metrics.json",
    ]
    assert [artifact.content for artifact in rendered] == [
        artifact.content for artifact in repeated
    ]
    for artifact in rendered:
        assert len(artifact.content) < 2_000_000
        if not artifact.name.endswith(".svg"):
            continue
        root = ET.fromstring(artifact.content)
        assert root.tag.endswith("svg")
        text = artifact.content.decode("utf-8")
        assert "纯合成演练" in text
        assert "真实预测证据" in text
        assert "合成已知答案检查" in text
        assert "正式门" not in text
        assert "预登记门" not in text
        assert "门控=" not in text


def test_interactive_page_is_self_contained_and_has_working_control_bindings(
    rendered: tuple[RenderedArtifact, ...],
) -> None:
    html = next(
        artifact.content.decode("utf-8") for artifact in rendered if artifact.name.endswith(".html")
    )
    parser = _DocumentParser()
    parser.feed(html)

    assert parser.start_tags.count("html") == 1
    assert parser.end_tags.count("html") == 1
    assert {"scenario", "model", "horizon", "map", "metric-list"} <= parser.ids
    assert all(
        forbidden not in html
        for forbidden in (
            "http://",
            "https://",
            "fetch(",
            "XMLHttpRequest",
            "WebSocket",
            "EventSource",
            "<script src=",
            "<link href=",
        )
    )
    assert all(
        label in html
        for label in (
            "近期活动有效",
            "近期活动无增量",
            "近期活动误导",
            "P0",
            "P1",
            "PP",
            "7 天",
            "30 天",
            "90 天",
            "600000",
            "纯合成演练",
            "不是绝对发震概率",
            "合成已知答案检查",
        )
    )
    assert "正式门" not in html
    assert "预登记门" not in html
    assert "门控=" not in html
    assert 'addEventListener("change",draw)' in html
    assert "draw();" in html


def test_metrics_are_public_safe_and_writer_is_create_once(
    rendered: tuple[RenderedArtifact, ...],
    tmp_path: Path,
) -> None:
    metrics = json.loads(
        next(artifact.content for artifact in rendered if artifact.name == "metrics.json")
    )
    assert metrics["generated_from_real_data"] is False
    assert metrics["real_prediction_evidence"] is False
    assert len(metrics["scenarios"]) == 3
    assert all(item["synthetic_known_answer_status"] == "passed" for item in metrics["scenarios"])
    encoded = json.dumps(metrics, ensure_ascii=False)
    assert '"gate"' not in encoded
    assert "counterfactual" not in encoded
    assert "expected_behavior_passed" not in encoded
    assert "x_km" not in encoded
    assert "y_km" not in encoded
    assert "-target-" not in encoded

    output = tmp_path / "stage2p-rendered"
    hashes = write_artifacts(rendered, output)
    assert set(hashes) == {artifact.name for artifact in rendered}
    assert hashes == write_artifacts(rendered, output, check=True)
    with pytest.raises(FileExistsError):
        write_artifacts(rendered, output)

    (output / "stale-result.svg").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="contents differ"):
        write_artifacts(rendered, output, check=True)


def test_artifact_bundle_rejects_unsafe_or_duplicate_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe basename"):
        RenderedArtifact("../escape.svg", b"<svg/>", "image/svg+xml")
    with pytest.raises(ValueError, match="safe basename"):
        RenderedArtifact("nested/result.svg", b"<svg/>", "image/svg+xml")

    duplicate = RenderedArtifact("same.svg", b"<svg/>", "image/svg+xml")
    with pytest.raises(ValueError, match="unique"):
        write_artifacts((duplicate, duplicate), tmp_path / "duplicate")
