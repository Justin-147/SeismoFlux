from __future__ import annotations

from pathlib import Path

import yaml


def test_source_code_does_not_import_or_hardcode_legacy_project() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("src").rglob("*.py"))

    assert "LocationPred" not in source_text
    assert "D:/AIPred" not in source_text
    assert "D:\\AIPred" not in source_text


def test_protocol_preregisters_all_hypotheses_and_gates() -> None:
    machine = yaml.safe_load(Path("configs/research_protocol.yaml").read_text(encoding="utf-8"))
    document = Path("docs/research_protocol.md").read_text(encoding="utf-8")

    assert machine["hypotheses"] == [f"H{index}" for index in range(6)]
    assert machine["gates"] == [f"G{index}" for index in range(9)]
    assert machine["locked_test"]["formal_runs_allowed"] == 1
    assert machine["calendar_split"]["locked_test_issue_dates"]["start"] is None
    assert machine["calendar_split"]["prospective_shadow_issue_dates"]["start"] is None
    assert (
        machine["calendar_split"]["prospective_shadow_issue_dates"]["historical_backfill_forbidden"]
        is True
    )
    for identifier in [*machine["hypotheses"], *machine["gates"]]:
        assert identifier in document


def test_agents_file_contains_all_blueprint_rules() -> None:
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert all(f"{index}." in agents for index in range(1, 16))
    assert "测试、验收、提交和推送" in agents
    assert "至少保留 2 个物理核心" in agents


def test_operating_point_candidates_match_base_configuration() -> None:
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    operating = yaml.safe_load(Path("configs/operating_points.yaml").read_text(encoding="utf-8"))

    assert operating["component_count_candidates"] == base["regions"]["component_counts"]
    assert (
        operating["union_area_budget_candidates_km2"] == base["regions"]["union_area_budgets_km2"]
    )
    assert operating["operational_limits"]["max_components"] == 10
    assert operating["operational_limits"]["max_union_area_km2"] == 960000
    assert operating["diagnostic_only_component_counts"] == [12]
    assert operating["fallback_when_no_candidate_passes"]["union_area_budget_km2"] == 300000
