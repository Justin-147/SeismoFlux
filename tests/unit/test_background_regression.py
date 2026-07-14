from __future__ import annotations

import json
from pathlib import Path

import pytest

from seismoflux.background.regression import (
    run_analytic_simulation_regression,
    run_production_fixture_regression,
)

MICRO_FIXTURE = Path("tests/fixtures/background/etas_micro_reference.json")


def test_analytic_simulation_regression_passes_all_frozen_distributional_gates() -> None:
    result = run_analytic_simulation_regression()

    assert result.passed is True
    assert result.replicate_count == 8192
    assert result.event_cap_hits == 0
    assert result.nonfinite_values == 0
    assert result.pooled_direct_offspring_count >= 3500
    assert result.root_total_descendants_mean == pytest.approx(0.8953578796104242, abs=0.08)
    assert result.root_zero_descendant_probability == pytest.approx(
        0.580621402297363,
        abs=0.03,
    )


def test_analytic_regression_rejects_post_hoc_replication_or_cap_changes() -> None:
    with pytest.raises(ValueError, match="8192"):
        run_analytic_simulation_regression(replicate_count=100)
    with pytest.raises(ValueError, match="100000"):
        run_analytic_simulation_regression(maximum_events_per_replicate=1000)


def test_production_fixture_runtime_regression_passes_every_frozen_value() -> None:
    result = run_production_fixture_regression(MICRO_FIXTURE)

    assert result.fixture_id == "etas_inverse_power_cut300_v1"
    assert len(result.checks) == 23
    assert result.passed is True


def test_production_fixture_runtime_regression_exposes_tampered_expectation(
    tmp_path: Path,
) -> None:
    raw = json.loads(MICRO_FIXTURE.read_text(encoding="utf-8"))
    raw["expected"]["lambda_probe_t2_r5_per_day_km2"] *= 1.1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")

    result = run_production_fixture_regression(changed)
    assert result.passed is False
    assert dict(result.checks)["lambda_probe_t2_r5_per_day_km2"] is False
