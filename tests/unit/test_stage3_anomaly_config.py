from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from seismoflux.config import load_yaml_mapping
from seismoflux.data.contracts import CONTRACTS as STAGE1_CONTRACTS
from seismoflux.features.anomaly.config import (
    ANOMALY_HISTORY_CONTRACT_VERSION,
    ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256,
    GAUSSIAN_BANDWIDTHS_KM,
    SPATIAL_RADII_KM,
    TEMPORAL_WINDOWS_WEEKS,
    AnomalyHistoryConfig,
    load_anomaly_history_config,
)
from seismoflux.features.anomaly.contracts import (
    ANOMALY_STATE_HISTORY_CONTRACT,
    STAGE3_CONTRACTS,
    contract_document,
)


def _mapping() -> dict[str, Any]:
    return load_yaml_mapping(Path("configs/anomaly_history.yaml"))


def test_complete_stage3_machine_protocol_is_strictly_validated() -> None:
    config = load_anomaly_history_config("configs/anomaly_history.yaml")

    assert config.contract_version == ANOMALY_HISTORY_CONTRACT_VERSION == "0.3.0"
    assert len(ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256) == 64
    assert config.expected_report_period_count == 205
    assert config.temporal_windows_weeks == TEMPORAL_WINDOWS_WEEKS
    assert config.spatial_radii_km == SPATIAL_RADII_KM
    assert config.gaussian_bandwidths_km == GAUSSIAN_BANDWIDTHS_KM
    assert config.query_grid_cell_km == 25.0
    assert config.gaussian_truncate_at_sigma == 3.0
    assert config.locked_test_run is False
    assert config.scientific_inputs["earthquake_catalog_forbidden"] is True
    assert config.state_reconstruction["history_scope"] == "all_205_actual_report_periods"
    assert config.feature_families["fault_features"] == {
        "status": "deferred_to_stage_5",
        "distance_to_fault": "forbidden_in_stage_3",
        "along_fault_projection": "forbidden_in_stage_3",
        "fault_segment_aggregation": "forbidden_in_stage_3",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["scientific_inputs"].__setitem__(
            "exact_dataset_names", ["anomaly_report_period", "anomaly_observation"]
        ),
        lambda raw: raw["time_semantics"].__setitem__("actual_report_period_count", 204),
        lambda raw: raw["time_semantics"]["windows"].__setitem__("lookback_weeks", [4, 8, 13, 26]),
        lambda raw: raw["spatial_features"]["fixed_closed_balls"].__setitem__(
            "radii_km", [50.0, 100.0, 200.0, 500.0]
        ),
        lambda raw: raw["spatial_features"]["gaussian_kernels"].__setitem__(
            "truncate_at_sigma", 4.0
        ),
        lambda raw: raw["feature_families"]["fault_features"].__setitem__("status", "enabled"),
        lambda raw: raw["reliability"]["grades"]["cautious"].__setitem__("weight", 0.75),
        lambda raw: raw["locked_test"].__setitem__("run", True),
        lambda raw: raw.__setitem__("undocumented", True),
    ],
)
def test_any_nested_or_top_level_protocol_drift_is_rejected(mutate: Any) -> None:
    raw = deepcopy(_mapping())
    mutate(raw)

    with pytest.raises(ValidationError, match="complete frozen semantic contract"):
        AnomalyHistoryConfig.model_validate(raw)


def test_model_dump_round_trips_through_the_same_complete_validator() -> None:
    config = load_anomaly_history_config("configs/anomaly_history.yaml")

    round_tripped = AnomalyHistoryConfig.model_validate(config.model_dump(mode="python"))

    assert round_tripped == config


def test_stage3_contract_is_independent_of_the_frozen_stage1_registry() -> None:
    assert "anomaly_state_history" not in STAGE1_CONTRACTS
    assert set(STAGE3_CONTRACTS) == {"anomaly_state_history"}

    document = contract_document(ANOMALY_STATE_HISTORY_CONTRACT)

    assert document["contract_version"] == "0.3.0"
    assert document["dataset"] == "anomaly_state_history"
    assert document["sort_keys"] == [
        "issue_time_utc",
        "state_row_kind",
        "anomaly_id",
        "entity_scope",
        "state_id",
    ]
    assert len(ANOMALY_STATE_HISTORY_CONTRACT.schema.names) == len(
        set(ANOMALY_STATE_HISTORY_CONTRACT.schema.names)
    )
