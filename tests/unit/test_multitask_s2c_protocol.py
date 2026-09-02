"""Small, data-free checks for the finite S2-C scientific protocol."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _protocol():
    return yaml.safe_load((ROOT / "configs/multitask_s2_c_strain.yaml").read_text("utf-8"))


def test_original_axes_and_finite_comparisons():
    p = _protocol()
    assert p["calendar"]["horizons_days"] == [7, 30, 90, 180, 365]
    assert p["calendar"]["outer_total_issue_horizon_pairs"] == 396
    assert len(p["calendar"]["outer_folds"]) == 4
    assert len(p["models"]) == 4
    assert len(p["planned_pairs"]) == 6
    assert p["evaluation"]["expected_main_anchor_count"] == 147
    assert p["evaluation"]["post_release_expected_main_anchor_count"] == 32
    assert p["calendar"]["post_release_development_folds"] == ["C_DEV_2015_2019"]


def test_official_grid_and_scalar_without_new_search():
    p = _protocol()
    s = p["spatial_math"]
    assert s["longitude_width_degrees"] == 0.25
    assert s["latitude_height_degrees"] == 0.2
    assert s["strain_scalar"] == "sqrt(exx**2+eyy**2+2*exy**2)"
    assert s["smoothing_scales"] == []
    assert s["artificial_probability_floor"] is None
    assert p["selection"]["alpha_candidates"] == [0, 0.25, 0.5, 0.75, 1]


def test_zero_is_not_missing_and_no_old_gates():
    p = _protocol()
    assert p["zero_mass"]["positive_count_dot_only"]
    assert p["zero_mass"]["negative_infinity_minus_negative_infinity"] == "undefined_not_zero"
    assert p["selection"]["target_in_zero_mass_cell"] == "negative_infinity_block_score_not_missing"
    assert p["inputs"]["national_support_percentage_gate"] is None
    assert not p["inputs"]["local_Mc_masks_applied"]
    assert p["inputs"]["local_two_catalogs_effective_magnitude_type"] == "Ms"
    assert p["inputs"]["numeric_magnitude_conversion"] == "none"


def test_inputs_and_safety_remain_separate():
    p = _protocol()
    assert p["inputs"]["strain_source"]["expected_rows"] == 145086
    assert len(p["inputs"]["strain_source"]["sha256"]) == 64
    assert p["inputs"]["strain_source"]["used_columns"] == ["lat", "long", "exx", "eyy", "exy"]
    for key in [
        "holdout_2020_2022_opened",
        "audit_2023_plus_opened",
        "locked_test_run",
        "future_catalog_downloads",
        "frozen_P1_modified",
        "hazard_scores_or_dynamic_anomalies_used",
    ]:
        assert p["run_boundary"][key] is False
    assert p["resources"]["maximum_fold_workers"] == 3
    assert p["resources"]["minimum_reserved_physical_cores"] >= 2
