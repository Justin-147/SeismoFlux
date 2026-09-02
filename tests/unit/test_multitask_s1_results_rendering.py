from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_renderer() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/render_multitask_s1_results.py"
    spec = importlib.util.spec_from_file_location("multitask_s1_result_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer()


def _episode(
    episode_id: str,
    *,
    hit: bool,
    l2_hit: bool,
    member_count: int,
    l3_minus_l2: float,
) -> object:
    return renderer.EpisodeRecord(
        episode_id=episode_id,
        event_id=f"event-{episode_id}",
        longitude=100.0,
        latitude=30.0,
        cell_index=1,
        hit=hit,
        fold_id="C_DEV_2000_2004",
        issue_time_utc="2001-01-03T16:00:00Z",
        global_episode_member_count=member_count,
        l3_log_density=-10.0 + l3_minus_l2,
        l2_log_density=-10.0,
        l0_log_density=-11.0,
        l2_hit=l2_hit,
        l3_minus_l2_log_density=l3_minus_l2,
        l3_minus_l0_log_density=1.0 + l3_minus_l2,
    )


def test_case_selection_uses_incremental_hit_median_and_largest_miss() -> None:
    episodes = [
        _episode("hit-a", hit=True, l2_hit=False, member_count=1, l3_minus_l2=0.5),
        _episode("hit-b", hit=True, l2_hit=False, member_count=2, l3_minus_l2=1.5),
        _episode("hit-c", hit=True, l2_hit=False, member_count=3, l3_minus_l2=2.5),
        _episode("shared-hit", hit=True, l2_hit=True, member_count=20, l3_minus_l2=9.0),
        _episode("miss-small", hit=False, l2_hit=False, member_count=4, l3_minus_l2=-1.0),
        _episode("miss-large-z", hit=False, l2_hit=False, member_count=12, l3_minus_l2=-2.0),
        _episode("miss-large-a", hit=False, l2_hit=False, member_count=12, l3_minus_l2=-3.0),
    ]

    selected = renderer.select_representative_cases(episodes)

    assert selected[0].role == "representative_incremental_hit"
    assert selected[0].episode.episode_id == "hit-b"
    assert selected[1].role == "largest_member_episode_miss"
    assert selected[1].episode.episode_id == "miss-large-a"
    assert "L3-minus-L2" in selected[0].rule
    assert "largest frozen all-catalog episode member" in selected[1].rule


def test_offline_json_escapes_script_terminator_and_area_audit_is_exact() -> None:
    encoded = renderer._json_for_script({"value": "</script>"})

    assert "</script>" not in encoded
    assert "<\\/script>" in encoded
    assert pytest.approx(0.06372412) == renderer.UNIFORM_RANDOM_AREA_EXPECTATION
    assert (
        pytest.approx(9.367446, abs=1.0e-5)
        == renderer.UNIFORM_RANDOM_AREA_EXPECTATION * renderer.EXPECTED_EPISODE_COUNT
    )


def test_wilson_interval_contains_observed_recall() -> None:
    low, high = renderer._wilson_interval(54 / 147, 147)

    assert low < 54 / 147 < high
    assert 0.0 < low < high < 1.0


def test_science_diagnostic_requires_cluster_robust_magnitude_contract() -> None:
    diagnostic = {
        "record_type": "s1_c0_post_score_science_diagnostic",
        "holdout_opened": False,
        "audit_opened": False,
        "locked_test_run": False,
        "magnitude_cluster_dependence_diagnostic": {
            "combined_M5_plus_episode_count": 314,
            "point_mean_effect_nats_per_event": 0.011324700512151587,
            "wording_gate": "cluster_robust_small_development_signal",
            "cluster_bootstrap": {
                "replicates": 20_000,
                "unit": "combined_M5_plus_fixed_anchor_episode",
                "strictly_positive": True,
                "lower": 0.006183811540067173,
                "upper": 0.01633976230512831,
            },
            "remove_largest_combined_episode": {
                "full_catalog_member_count": 19,
                "remaining_mean_effect_nats_per_event": 0.011209540044119509,
            },
        },
    }

    renderer._validate_science_diagnostic(diagnostic)
    assert renderer.STATIC_NAMES[-1] == "05_representative_hit_and_major_miss.png"

    diagnostic["magnitude_cluster_dependence_diagnostic"]["cluster_bootstrap"]["replicates"] = 2_000
    with pytest.raises(renderer.RenderingError, match="contract changed"):
        renderer._validate_science_diagnostic(diagnostic)
