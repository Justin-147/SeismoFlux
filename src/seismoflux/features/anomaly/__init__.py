"""Causal dynamic-anomaly state and feature contracts."""

from seismoflux.features.anomaly.config import (
    ANOMALY_HISTORY_CONTRACT_VERSION,
    AnomalyHistoryConfig,
    load_anomaly_history_config,
)
from seismoflux.features.anomaly.contracts import ANOMALY_STATE_HISTORY_CONTRACT
from seismoflux.features.anomaly.input import (
    LoadedStage3Inputs,
    RegisteredParquetInput,
    RegisteredStudyAreaInput,
    Stage3InputSpec,
    load_stage3_inputs,
    require_stage3_dataset_boundary,
    select_allowlisted_columns,
    stage3_input_spec_from_config,
    validate_stage3_input_tables,
)
from seismoflux.features.anomaly.state import (
    AnomalyState,
    build_anomaly_state_history,
    state_records,
)

__all__ = [
    "ANOMALY_HISTORY_CONTRACT_VERSION",
    "ANOMALY_STATE_HISTORY_CONTRACT",
    "AnomalyHistoryConfig",
    "AnomalyState",
    "LoadedStage3Inputs",
    "RegisteredParquetInput",
    "RegisteredStudyAreaInput",
    "Stage3InputSpec",
    "build_anomaly_state_history",
    "load_anomaly_history_config",
    "load_stage3_inputs",
    "require_stage3_dataset_boundary",
    "select_allowlisted_columns",
    "stage3_input_spec_from_config",
    "state_records",
    "validate_stage3_input_tables",
]
