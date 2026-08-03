from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from seismoflux.d1_replay.robustness import (
    D1_ROBUSTNESS_CONTRASTS,
    D1RobustnessError,
    build_d1_robustness_result,
)

_MODELS = (
    "B0",
    "B0_R30",
    "B0_C",
    "B0_C_A_snapshot",
    "B0_C_A_dynamic",
    "B0_R30_C_A_dynamic",
)
_CONTRACT_SHA = "a" * 64
_MANIFEST_SHA = "b" * 64


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _axes() -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int]]:
    zones = tuple(sorted(_identity(f"zone-{index}") for index in range(39)))
    clusters = tuple(sorted(_identity(f"cluster-{index}") for index in range(21)))
    cell_by_cluster = {cluster: index for index, cluster in enumerate(clusters)}
    return zones, clusters, cell_by_cluster


def _observed(
    *,
    b0_c_hits: set[int] | None = None,
    snapshot_hits: set[int] | None = None,
    dynamic_hits: set[int] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    zones, clusters, cell_by_cluster = _axes()
    base = set() if b0_c_hits is None else set(b0_c_hits)
    snapshot = set() if snapshot_hits is None else set(snapshot_hits)
    dynamic = set() if dynamic_hits is None else set(dynamic_hits)
    hit_indices = {
        "B0": set(),
        "B0_R30": set(),
        "B0_C": base,
        "B0_C_A_snapshot": snapshot,
        "B0_C_A_dynamic": dynamic,
        "B0_R30_C_A_dynamic": dynamic,
    }
    folds = ("fold_1",) * 8 + ("fold_2",) * 6 + ("fold_3",) * 7
    support = [
        {
            "cluster_id": cluster,
            "fold_id": fold,
            "issue_id": f"issue-{index:02d}",
        }
        for index, (cluster, fold) in enumerate(zip(clusters, folds, strict=True))
    ]
    outcomes: list[dict[str, object]] = []
    for index, (cluster, fold) in enumerate(zip(clusters, folds, strict=True)):
        for model in _MODELS:
            hit = index in hit_indices[model]
            outcomes.append(
                {
                    "cluster_id": cluster,
                    "fold_id": fold,
                    "issue_id": f"issue-{index:02d}",
                    "issue_time_utc": "2024-01-01T00:00:00Z",
                    "horizon_days": 30,
                    "model_id": model,
                    "representative_cell_index": cell_by_cluster[cluster],
                    "outside_support": False,
                    "log_density": -8.0,
                    "hit_by_area": [hit, hit, hit, hit, hit],
                }
            )
    return (
        {
            "schema_version": 1,
            "protocol_version": "d1.0.0",
            "result_kind": "observed_replay",
            "status": "completed",
            "retrospective_only": True,
            "relative_strength_not_absolute_probability": True,
            "identities": {
                "contract_sha256": _CONTRACT_SHA,
                "manifest_content_sha256": _MANIFEST_SHA,
                "input_sha256": "c" * 64,
                "git_commit": "d" * 40,
            },
            "expected_support_by_horizon": {"30": support},
            "outcomes": outcomes,
        },
        zones,
        clusters,
    )


def _build(observed: dict[str, Any], zones: tuple[str, ...]) -> dict[str, object]:
    return build_d1_robustness_result(
        observed,
        expected_contract_sha256=_CONTRACT_SHA,
        expected_manifest_content_sha256=_MANIFEST_SHA,
        zone_ids=zones,
        zone_id_by_cell_index=zones,
        spatial_strata_identity={
            "public_manifest_content_sha256": "e" * 64,
            "artifact_sha256": {
                "cell_mapping": "f" * 64,
                "entity_mapping": "1" * 64,
                "zone_geometry": "2" * 64,
                "connectors": "3" * 64,
            },
            "operational_cell_count": 39,
            "nonempty_zone_count": 39,
            "geometry_zone_count": 65,
        },
    )


def _contrast(result: dict[str, object], contrast_id: str) -> dict[str, Any]:
    contrasts = result["contrasts"]
    assert isinstance(contrasts, list)
    return next(item for item in contrasts if item["contrast_id"] == contrast_id)


def test_regional_diagnostic_detects_single_zone_and_cross_zone_gain() -> None:
    observed, zones, _ = _observed(snapshot_hits={0}, dynamic_hits={0, 1})
    result = _build(observed, zones)

    assert result["result_kind"] == "d1_regional_and_leave_one_cluster_robustness"
    assert result["regional_diagnostic_completed"] is True
    assert result["leave_one_cluster_out_completed"] is True
    assert result["model_refit_performed"] is False
    assert result["locked_test_read"] is False
    assert result["identities"] == observed["identities"]
    contrasts = result["contrasts"]
    assert isinstance(contrasts, list)
    assert [item["contrast_id"] for item in contrasts] == [
        item[0] for item in D1_ROBUSTNESS_CONTRASTS
    ]

    snapshot = _contrast(result, "B0_C_A_snapshot_minus_B0_C")
    snapshot_regional = snapshot["regional"]
    assert snapshot["observed_hit_gain_sum"] == 1
    assert snapshot_regional["target_bearing_zone_gain_sign_counts"] == {
        "positive_count": 1,
        "zero_count": 20,
        "negative_count": 0,
    }
    assert snapshot_regional["single_zone_direction_dominant"] is True
    assert snapshot_regional["direction_survives_largest_positive_zone_removal"] is False
    assert len(snapshot_regional["zone_rows"]) == 39
    assert snapshot_regional["additive_recall_gain_closure"] == pytest.approx(1 / 21)

    dynamic = _contrast(result, "B0_C_A_dynamic_minus_B0_C")
    dynamic_regional = dynamic["regional"]
    assert dynamic["observed_hit_gain_sum"] == 2
    assert dynamic_regional["target_bearing_zone_gain_sign_counts"]["positive_count"] == 2
    assert dynamic_regional["single_zone_direction_dominant"] is False
    assert dynamic_regional["direction_survives_largest_positive_zone_removal"] is True


def test_leave_one_cluster_out_reports_positive_zero_and_negative_directions() -> None:
    observed, zones, clusters = _observed(b0_c_hits={1}, snapshot_hits={0}, dynamic_hits={0})
    result = _build(observed, zones)
    snapshot = _contrast(result, "B0_C_A_snapshot_minus_B0_C")
    loco = snapshot["leave_one_cluster_out"]

    assert snapshot["observed_hit_gain_sum"] == 0
    assert loco["replication_count"] == 21
    assert loco["recall_gain_minimum"] == pytest.approx(-1 / 20)
    assert loco["recall_gain_maximum"] == pytest.approx(1 / 20)
    assert loco["recall_gain_sign_counts"] == {
        "positive_count": 1,
        "zero_count": 19,
        "negative_count": 1,
    }
    assert loco["direction_survives_every_cluster_removal"] is False
    assert {row["omitted_cluster_id"] for row in loco["rows"]} == set(clusters)


@pytest.mark.parametrize(
    ("identity_field", "replacement", "message"),
    (
        ("contract_sha256", "9" * 64, "contract identity"),
        ("manifest_content_sha256", "8" * 64, "manifest identity"),
        ("git_commit", "not-a-commit", "git_commit"),
    ),
)
def test_observed_identity_tampering_is_rejected(
    identity_field: str,
    replacement: str,
    message: str,
) -> None:
    observed, zones, _ = _observed(snapshot_hits={0}, dynamic_hits={0, 1})
    tampered = copy.deepcopy(observed)
    tampered["identities"][identity_field] = replacement

    with pytest.raises(D1RobustnessError, match=message):
        _build(tampered, zones)
