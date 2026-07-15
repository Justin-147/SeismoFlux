"""Pure in-memory compensator convergence gate for frozen stage-4 models.

The audit accepts one frozen preprocessor, one frozen set of variant
coefficients, one frozen magnitude-rate head, and target-blind fields recomputed
on the three preregistered grids.  It never fits parameters, reads files, loads
catalogues, or evaluates an event log-intensity term.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa

from seismoflux.anomaly_increment.background_adapter import Stage4BackgroundFit
from seismoflux.anomaly_increment.contracts import (
    FloatArray,
    FrozenTargetRateHead,
    canonical_mapping_sha256,
    readonly_float_matrix,
    readonly_float_vector,
)
from seismoflux.anomaly_increment.feature_adapter import (
    FEATURE_IDENTITY_COLUMNS,
    assert_issue_table_matches_frozen_grid,
    concatenate_source_columns,
)
from seismoflux.anomaly_increment.grid_features import (
    SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1,
    Stage4GridFamily,
    assert_selected_columns_logically_exact_r1,
    selected_table_logical_identity_sha256_r1,
)
from seismoflux.anomaly_increment.integration import (
    INTEGRATION_NEAR_ZERO_ABSOLUTE_TOLERANCE,
    INTEGRATION_RELATIVE_TOLERANCE,
    PRIMARY_TIME_STEP_DAYS,
    REFERENCE_TIME_STEP_DAYS,
    SPATIAL_RELATIVE_DENOMINATOR_FLOOR,
    ConvergenceCheck,
    compare_integrals,
    integrate_conditional_intensity,
)
from seismoflux.anomaly_increment.preprocessing import FrozenPreprocessor
from seismoflux.anomaly_increment.preregistration import STAGE4_HORIZONS_DAYS
from seismoflux.features.anomaly.engine import Stage3FeatureEngine
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot

MagnitudeBin: TypeAlias = Literal["M5_6", "M6_plus"]
ModelVariant: TypeAlias = Literal[
    "background_no_increment",
    "coverage_only",
    "snapshot",
    "dynamic",
]
GateStatus: TypeAlias = Literal["passed", "failed"]
PublicationAction: TypeAlias = Literal[
    "allow_spatial_publication_subject_to_remaining_gates",
    "fail_model_and_forbid_spatial_publication",
]

GRID_SIZES_KM = (50.0, 25.0, 12.5)
FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT = 153
MODEL_VARIANTS: tuple[ModelVariant, ...] = (
    "background_no_increment",
    "coverage_only",
    "snapshot",
    "dynamic",
)
MAGNITUDE_BINS: tuple[MagnitudeBin, ...] = ("M5_6", "M6_plus")
CONVERGENCE_FAILURE_ACTION: PublicationAction = "fail_model_and_forbid_spatial_publication"
CONVERGENCE_PASS_ACTION: PublicationAction = "allow_spatial_publication_subject_to_remaining_gates"
EVENT_LOG_TERM_POLICY = "excluded_from_compensator_convergence_and_unchanged"

_FORBIDDEN_TARGET_NAME_PARTS = (
    "target",
    "epicenter",
    "hypocenter",
    "true_longitude",
    "true_latitude",
    "event_longitude",
    "event_latitude",
    "震中",
)


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _sha256_digest(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _assert_target_blind_name(value: str, *, label: str) -> None:
    folded = value.casefold()
    if any(token in folded for token in _FORBIDDEN_TARGET_NAME_PARTS):
        raise ValueError(f"{label} contains a forbidden target/epicenter field name: {value}")


def _float_matrix_hex(values: FloatArray) -> list[list[str]]:
    return [[float(value).hex() for value in row] for row in values]


def _float_vector_hex(values: FloatArray) -> list[str]:
    return [float(value).hex() for value in values]


def _float64_vector_sha256(values: FloatArray) -> str:
    canonical = np.asarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenVariantCoefficients:
    """Already-fitted coefficients for one frozen model variant."""

    variant: ModelVariant
    design_column_indices: tuple[int, ...]
    beta: FloatArray

    def __post_init__(self) -> None:
        if self.variant not in MODEL_VARIANTS:
            raise ValueError("model variant is outside the frozen stage-4 variants")
        indices = tuple(self.design_column_indices)
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices
        ):
            raise ValueError("design column indices must be non-negative integers")
        if len(indices) != len(set(indices)):
            raise ValueError("design column indices must be unique")
        beta = readonly_float_vector(
            "frozen convergence beta",
            self.beta,
            allow_empty=True,
        )
        if beta.size != len(indices):
            raise ValueError("frozen beta must align with design column indices")
        if self.variant == "background_no_increment":
            if indices or beta.size:
                raise ValueError("background-only convergence must contain no increment beta")
        elif not indices:
            raise ValueError("increment variants must contain fitted coefficients")
        object.__setattr__(self, "design_column_indices", indices)
        object.__setattr__(self, "beta", beta)

    def as_mapping(self) -> dict[str, object]:
        return {
            "beta_float64_hex": _float_vector_hex(self.beta),
            "design_column_indices": list(self.design_column_indices),
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class FrozenConvergenceModel:
    """Single fitted model identity shared unchanged across every audit grid."""

    preprocessor: FrozenPreprocessor
    rate_head: FrozenTargetRateHead
    variants: tuple[FrozenVariantCoefficients, ...]

    def __post_init__(self) -> None:
        variants = tuple(self.variants)
        if tuple(item.variant for item in variants) != MODEL_VARIANTS:
            raise ValueError("convergence model must retain all four variants in frozen order")
        column_count = len(self.preprocessor.design_column_names)
        if any(
            index >= column_count for variant in variants for index in variant.design_column_indices
        ):
            raise ValueError("variant coefficient index is outside the frozen preprocessor")
        for contract in self.preprocessor.contracts:
            _assert_target_blind_name(
                contract.source_column,
                label="frozen preprocessor source column",
            )
            _assert_target_blind_name(
                contract.logical_feature,
                label="frozen preprocessor logical feature",
            )
        object.__setattr__(self, "variants", variants)

    def variant(self, name: ModelVariant) -> FrozenVariantCoefficients:
        return next(item for item in self.variants if item.variant == name)

    def _payload_mapping(self) -> dict[str, object]:
        return {
            "preprocessor_sha256": self.preprocessor.sha256,
            "rate_head_sha256": self.rate_head.sha256,
            "variants": [item.as_mapping() for item in self.variants],
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload_mapping())

    @property
    def sha256(self) -> str:
        return self.content_sha256

    def as_mapping(self) -> dict[str, object]:
        return {**self._payload_mapping(), "content_sha256": self.content_sha256}


def _frozen_feature_columns(
    columns: Mapping[str, object],
    *,
    expected_row_count: int,
) -> Mapping[str, FloatArray]:
    if not isinstance(columns, Mapping) or not columns:
        raise ValueError("recomputed feature columns must be a non-empty mapping")
    output: dict[str, FloatArray] = {}
    for raw_name, raw_values in columns.items():
        name = _identifier(raw_name, label="recomputed feature column")
        _assert_target_blind_name(name, label="recomputed feature column")
        values = np.array(raw_values, dtype=np.float64, copy=True, order="C")
        if values.ndim != 1 or values.size != expected_row_count:
            raise ValueError("every recomputed feature column must align with grid cells")
        # Canonicalize NaN payloads before hashing/freezing.  Missingness remains
        # scientifically unchanged and the accepted 25 km Arrow validity bitmap
        # is authenticated separately below.
        values[np.isnan(values)] = np.nan
        values.setflags(write=False)
        output[name] = values
    return MappingProxyType(output)


@dataclass(frozen=True, slots=True)
class TargetBlindGridFeatures:
    """Frozen raw anomaly fields for one grid, constructed before target access."""

    grid_id: str
    cell_size_km: float
    cell_ids: tuple[str, ...] = field(repr=False, compare=False)
    feature_columns: Mapping[str, FloatArray] = field(repr=False, compare=False)
    feature_table_identity_sha256: str

    def __post_init__(self) -> None:
        grid_id = _sha256_digest(self.grid_id, label="target-blind convergence grid_id")
        cell_size = float(self.cell_size_km)
        if cell_size not in GRID_SIZES_KM:
            raise ValueError("target-blind convergence grid must be 50, 25, or 12.5 km")
        cells = tuple(self.cell_ids)
        if (
            not cells
            or len(cells) != len(set(cells))
            or any(not isinstance(item, str) or not item for item in cells)
        ):
            raise ValueError("target-blind convergence cell IDs must be non-empty and unique")
        columns = _frozen_feature_columns(
            self.feature_columns,
            expected_row_count=len(cells),
        )
        _sha256_digest(
            self.feature_table_identity_sha256,
            label="target-blind feature table identity",
        )
        object.__setattr__(self, "grid_id", grid_id)
        object.__setattr__(self, "cell_size_km", cell_size)
        object.__setattr__(self, "cell_ids", cells)
        object.__setattr__(self, "feature_columns", columns)

    def _payload_mapping(self) -> dict[str, object]:
        return {
            "cell_count": len(self.cell_ids),
            "cell_ids_sha256": canonical_mapping_sha256(
                {"cell_ids": list(self.cell_ids), "schema_version": 1}
            ),
            "cell_size_km_hex": self.cell_size_km.hex(),
            "feature_column_float64_sha256": {
                name: _float64_vector_sha256(values)
                for name, values in self.feature_columns.items()
            },
            "feature_table_identity_sha256": self.feature_table_identity_sha256,
            "grid_id": self.grid_id,
            "source_columns": list(self.feature_columns),
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload_mapping())

    def as_mapping(self) -> dict[str, object]:
        return {**self._payload_mapping(), "content_sha256": self.content_sha256}


@dataclass(frozen=True, slots=True)
class PrimaryGridReproductionReceipt:
    """R1 logical-bit accepted-versus-recomputed proof for one causal 25 km issue."""

    issue_id: str
    issue_index: int
    issue_report_id: str
    accepted_table_sha256: str
    recomputed_table_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.issue_id, label="primary reproduction issue_id")
        _identifier(self.issue_report_id, label="primary reproduction issue_report_id")
        if (
            not isinstance(self.issue_index, int)
            or isinstance(self.issue_index, bool)
            or self.issue_index < 0
        ):
            raise ValueError("primary reproduction issue index must be non-negative")
        accepted = _sha256_digest(
            self.accepted_table_sha256,
            label="accepted primary reproduction table",
        )
        recomputed = _sha256_digest(
            self.recomputed_table_sha256,
            label="recomputed primary reproduction table",
        )
        if accepted != recomputed:
            raise ValueError("25 km issue reconstruction is not R1-logically exact")

    def as_mapping(self) -> dict[str, object]:
        return {
            "accepted_table_sha256": self.accepted_table_sha256,
            "issue_id": self.issue_id,
            "issue_index": self.issue_index,
            "issue_report_id": self.issue_report_id,
            "recomputed_table_sha256": self.recomputed_table_sha256,
        }


@dataclass(frozen=True, slots=True)
class PrimaryGridLogicalReplayAuditR1:
    """Target-blind 153-issue proof that 1/2 workers preserve every R1 identity."""

    grid_id: str
    source_columns: tuple[str, ...]
    source_input_sha256: str
    query_chunk_size: int
    worker_counts: tuple[int, int]
    receipts_by_worker: tuple[
        tuple[PrimaryGridReproductionReceipt, ...],
        tuple[PrimaryGridReproductionReceipt, ...],
    ] = field(repr=False)
    target_bytes_read: Literal[False] = False
    target_path_observed: Literal[False] = False
    role: Literal["stage4_r1_primary_grid_logical_identity_worker_replay"] = (
        "stage4_r1_primary_grid_logical_identity_worker_replay"
    )

    def __post_init__(self) -> None:
        _identifier(self.grid_id, label="R1 replay grid_id")
        sources = tuple(self.source_columns)
        if not sources or len(sources) != len(set(sources)):
            raise ValueError("R1 replay source columns must be non-empty and unique")
        for source in sources:
            _assert_target_blind_name(source, label="R1 replay source column")
        _sha256_digest(self.source_input_sha256, label="R1 replay source input")
        if (
            not isinstance(self.query_chunk_size, int)
            or isinstance(self.query_chunk_size, bool)
            or self.query_chunk_size <= 0
        ):
            raise ValueError("R1 replay query chunk size must be positive")
        if self.worker_counts != (1, 2):
            raise ValueError("R1 replay must use the frozen 1/2-worker matrix")
        replays = tuple(tuple(items) for items in self.receipts_by_worker)
        if len(replays) != 2:
            raise ValueError("R1 replay must contain exactly two worker results")
        expected_indices = tuple(range(FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT))
        for receipts in replays:
            if (
                len(receipts) != FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT
                or tuple(item.issue_index for item in receipts) != expected_indices
                or len({item.issue_id for item in receipts}) != len(receipts)
            ):
                raise ValueError("R1 replay must cover all 153 issues exactly once in order")
        if replays[0] != replays[1]:
            raise ValueError("R1 logical identities differ between 1 and 2 workers")
        if self.target_bytes_read or self.target_path_observed:
            raise ValueError("R1 logical replay crossed the target boundary")
        if self.role != "stage4_r1_primary_grid_logical_identity_worker_replay":
            raise ValueError("R1 logical replay role changed")
        object.__setattr__(self, "source_columns", sources)
        object.__setattr__(self, "receipts_by_worker", replays)

    @property
    def reproduction_identity_sha256(self) -> str:
        return canonical_mapping_sha256(
            {
                "grid_id": self.grid_id,
                "identity_method": SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1,
                "issues": [item.as_mapping() for item in self.receipts_by_worker[0]],
                "query_chunk_size": self.query_chunk_size,
                "source_columns": list(self.source_columns),
                "source_input_sha256": self.source_input_sha256,
            }
        )

    def _payload_mapping(self) -> dict[str, object]:
        return {
            "grid_id": self.grid_id,
            "identity_method": SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1,
            "issue_count": FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT,
            "query_chunk_size": self.query_chunk_size,
            "reproduction_identity_sha256": self.reproduction_identity_sha256,
            "role": self.role,
            "source_columns": list(self.source_columns),
            "source_input_sha256": self.source_input_sha256,
            "target_bytes_read": self.target_bytes_read,
            "target_path_observed": self.target_path_observed,
            "worker_replays": [
                {
                    "receipts": [item.as_mapping() for item in receipts],
                    "reproduction_identity_sha256": self.reproduction_identity_sha256,
                    "spatial_workers": workers,
                }
                for workers, receipts in zip(
                    self.worker_counts,
                    self.receipts_by_worker,
                    strict=True,
                )
            ],
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload_mapping())

    def as_mapping(self) -> dict[str, object]:
        return {**self._payload_mapping(), "content_sha256": self.content_sha256}


@dataclass(frozen=True, slots=True)
class FrozenTargetBlindConvergenceInputs:
    """Pretarget-frozen multigrid input and exact 25 km reproduction proof."""

    issue_id: str
    issue_report_id: str
    selected_issue_index: int
    selected_state_snapshot_id: str
    selected_lineage_digest: str
    source_columns: tuple[str, ...]
    source_input_sha256: str
    grids: tuple[
        TargetBlindGridFeatures,
        TargetBlindGridFeatures,
        TargetBlindGridFeatures,
    ] = field(repr=False, compare=False)
    primary_reproduction_receipts: tuple[PrimaryGridReproductionReceipt, ...]
    query_chunk_size: int
    spatial_workers: int
    target_bytes_read: Literal[False] = False
    target_path_observed: Literal[False] = False
    role: Literal["stage4_last_formal_issue_pretarget_multigrid_convergence"] = (
        "stage4_last_formal_issue_pretarget_multigrid_convergence"
    )

    def __post_init__(self) -> None:
        for label, value in (
            ("issue_id", self.issue_id),
            ("issue_report_id", self.issue_report_id),
        ):
            _identifier(value, label=label)
        if (
            not isinstance(self.selected_issue_index, int)
            or isinstance(self.selected_issue_index, bool)
            or self.selected_issue_index < 0
        ):
            raise ValueError("selected convergence issue index must be non-negative")
        _sha256_digest(
            self.selected_state_snapshot_id,
            label="selected convergence state snapshot",
        )
        _sha256_digest(
            self.selected_lineage_digest,
            label="selected convergence lineage digest",
        )
        sources = tuple(self.source_columns)
        if not sources or len(sources) != len(set(sources)):
            raise ValueError("convergence source columns must be non-empty and unique")
        for source in sources:
            _assert_target_blind_name(source, label="convergence source column")
        _sha256_digest(self.source_input_sha256, label="convergence source input")
        grids = tuple(self.grids)
        if tuple(item.cell_size_km for item in grids) != GRID_SIZES_KM:
            raise ValueError("target-blind convergence grids must be ordered 50, 25, 12.5 km")
        if tuple(tuple(item.feature_columns) for item in grids) != (sources,) * 3:
            raise ValueError("each convergence grid must retain the frozen source order")
        reproductions = tuple(self.primary_reproduction_receipts)
        if not reproductions or tuple(item.issue_index for item in reproductions) != tuple(
            range(len(reproductions))
        ):
            raise ValueError("25 km convergence proof must cover its causal issues in order")
        if len({item.issue_id for item in reproductions}) != len(reproductions):
            raise ValueError("25 km convergence proof contains duplicate issue IDs")
        final_reproduction = reproductions[-1]
        if (
            final_reproduction.issue_id != self.issue_id
            or final_reproduction.issue_index != self.selected_issue_index
            or final_reproduction.issue_report_id != self.issue_report_id
            or grids[1].feature_table_identity_sha256 != final_reproduction.recomputed_table_sha256
        ):
            raise ValueError("final 25 km reproduction differs from the frozen audit issue")
        if (
            not isinstance(self.query_chunk_size, int)
            or isinstance(self.query_chunk_size, bool)
            or self.query_chunk_size <= 0
        ):
            raise ValueError("convergence query chunk size must be positive")
        if (
            not isinstance(self.spatial_workers, int)
            or isinstance(self.spatial_workers, bool)
            or self.spatial_workers <= 0
        ):
            raise ValueError("convergence spatial worker count must be positive")
        if self.target_bytes_read or self.target_path_observed:
            raise ValueError("convergence input reconstruction crossed the target boundary")
        if self.role != "stage4_last_formal_issue_pretarget_multigrid_convergence":
            raise ValueError("target-blind convergence input role changed")
        object.__setattr__(self, "source_columns", sources)
        object.__setattr__(self, "grids", grids)
        object.__setattr__(self, "primary_reproduction_receipts", reproductions)

    @property
    def primary_reproduction_sha256(self) -> str:
        return canonical_mapping_sha256(
            {
                "issue_count": len(self.primary_reproduction_receipts),
                "issues": [item.as_mapping() for item in self.primary_reproduction_receipts],
                "schema_version": 1,
            }
        )

    def _payload_mapping(self) -> dict[str, object]:
        return {
            "grids": [item.as_mapping() for item in self.grids],
            "issue_id": self.issue_id,
            "issue_report_id": self.issue_report_id,
            "query_chunk_size": self.query_chunk_size,
            "primary_reproduction": {
                "issue_count": len(self.primary_reproduction_receipts),
                "sha256": self.primary_reproduction_sha256,
            },
            "role": self.role,
            "schema_version": 1,
            "selected_issue_index": self.selected_issue_index,
            "selected_lineage_digest": self.selected_lineage_digest,
            "selected_state_snapshot_id": self.selected_state_snapshot_id,
            "source_columns": list(self.source_columns),
            "source_input_sha256": self.source_input_sha256,
            "spatial_workers": self.spatial_workers,
            "target_bytes_read": self.target_bytes_read,
            "target_path_observed": self.target_path_observed,
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload_mapping())

    def as_mapping(self) -> dict[str, object]:
        return {**self._payload_mapping(), "content_sha256": self.content_sha256}


@dataclass(frozen=True, slots=True)
class RecomputedGridInputs:
    """Target-blind background masses and raw features for one frozen grid."""

    grid_id: str
    cell_size_km: float
    background_spatial_mass_by_cell_and_bin: FloatArray
    feature_columns: Mapping[str, object]

    def __post_init__(self) -> None:
        grid_id = _identifier(self.grid_id, label="grid_id")
        cell_size = float(self.cell_size_km)
        if cell_size not in GRID_SIZES_KM:
            raise ValueError("convergence cell_size_km must be exactly 50, 25, or 12.5")
        background = readonly_float_matrix(
            "background_spatial_mass_by_cell_and_bin",
            self.background_spatial_mass_by_cell_and_bin,
            allow_empty_rows=False,
            allow_empty_columns=False,
        )
        if background.shape[1] != len(MAGNITUDE_BINS):
            raise ValueError("background mass must contain M5_6 and M6_plus columns")
        if np.any(background < 0.0) or np.any(np.sum(background, axis=0) <= 0.0):
            raise ValueError("each magnitude-bin background mass must have positive total")
        columns = _frozen_feature_columns(
            self.feature_columns,
            expected_row_count=int(background.shape[0]),
        )
        object.__setattr__(self, "grid_id", grid_id)
        object.__setattr__(self, "cell_size_km", cell_size)
        object.__setattr__(self, "background_spatial_mass_by_cell_and_bin", background)
        object.__setattr__(self, "feature_columns", columns)


@dataclass(frozen=True, slots=True)
class CompensatorConvergenceResult:
    """One variant/bin/horizon total-compensator convergence result."""

    variant: ModelVariant
    magnitude_bin: MagnitudeBin
    horizon_days: int
    input_bundle_sha256: str
    total_compensator_50km_step_1d: float
    total_compensator_25km_step_1d: float
    total_compensator_12_5km_step_1d: float
    total_compensator_25km_step_0_5d: float
    spatial_25_vs_12_5: ConvergenceCheck
    temporal_1d_vs_0_5d: ConvergenceCheck
    coarse_50_vs_25: ConvergenceCheck

    def __post_init__(self) -> None:
        if self.variant not in MODEL_VARIANTS or self.magnitude_bin not in MAGNITUDE_BINS:
            raise ValueError("convergence result identity is outside the frozen model")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("convergence result horizon is outside the frozen windows")
        if len(self.input_bundle_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.input_bundle_sha256
        ):
            raise ValueError("convergence result must bind a lowercase SHA-256 input identity")
        values = (
            self.total_compensator_50km_step_1d,
            self.total_compensator_25km_step_1d,
            self.total_compensator_12_5km_step_1d,
            self.total_compensator_25km_step_0_5d,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("total compensator intensities must be finite and non-negative")
        if (
            self.spatial_25_vs_12_5.candidate != self.total_compensator_25km_step_1d
            or self.spatial_25_vs_12_5.reference != self.total_compensator_12_5km_step_1d
        ):
            raise ValueError("spatial comparison is not bound to 25 km versus 12.5 km")
        if (
            self.temporal_1d_vs_0_5d.candidate != self.total_compensator_25km_step_1d
            or self.temporal_1d_vs_0_5d.reference != self.total_compensator_25km_step_0_5d
        ):
            raise ValueError("temporal comparison is not bound to 1 day versus 0.5 day")
        if (
            self.coarse_50_vs_25.candidate != self.total_compensator_50km_step_1d
            or self.coarse_50_vs_25.reference != self.total_compensator_25km_step_1d
        ):
            raise ValueError("coarse diagnostic is not bound to 50 km versus 25 km")

    @property
    def passed(self) -> bool:
        return self.spatial_25_vs_12_5.passed and self.temporal_1d_vs_0_5d.passed

    @property
    def status(self) -> GateStatus:
        return "passed" if self.passed else "failed"

    def _payload_mapping(self) -> dict[str, object]:
        coarse = {
            **self.coarse_50_vs_25.as_mapping(),
            "gate_applied": False,
            "role": "required_reported_coarse_trend_diagnostic_not_gate_reference",
        }
        return {
            "coarse_50_vs_25": coarse,
            "horizon_days": self.horizon_days,
            "input_bundle_sha256": self.input_bundle_sha256,
            "magnitude_bin": self.magnitude_bin,
            "spatial_25_vs_12_5": self.spatial_25_vs_12_5.as_mapping(),
            "status": self.status,
            "temporal_1d_vs_0_5d": self.temporal_1d_vs_0_5d.as_mapping(),
            "total_compensator": {
                "grid_12_5km_step_1d": self.total_compensator_12_5km_step_1d,
                "grid_25km_step_0_5d": self.total_compensator_25km_step_0_5d,
                "grid_25km_step_1d": self.total_compensator_25km_step_1d,
                "grid_50km_step_1d": self.total_compensator_50km_step_1d,
            },
            "variant": self.variant,
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload_mapping())

    @property
    def sha256(self) -> str:
        return self.content_sha256

    def as_mapping(self) -> dict[str, object]:
        return {**self._payload_mapping(), "content_sha256": self.content_sha256}


@dataclass(frozen=True, slots=True)
class CompensatorConvergenceAudit:
    """Complete frozen convergence decision and publication action."""

    model_sha256: str
    input_bundle_sha256: str
    results: tuple[CompensatorConvergenceResult, ...]
    publication_action: PublicationAction
    event_log_term_policy: str = EVENT_LOG_TERM_POLICY
    event_log_term_included: bool = False
    event_log_term_change: float = 0.0
    primary_time_step_days: float = PRIMARY_TIME_STEP_DAYS
    reference_time_step_days: float = REFERENCE_TIME_STEP_DAYS
    relative_tolerance: float = INTEGRATION_RELATIVE_TOLERANCE
    near_zero_absolute_tolerance: float = INTEGRATION_NEAR_ZERO_ABSOLUTE_TOLERANCE

    def __post_init__(self) -> None:
        for label, value in (
            ("model_sha256", self.model_sha256),
            ("input_bundle_sha256", self.input_bundle_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        results = tuple(self.results)
        expected_keys = tuple(
            (variant, magnitude_bin, horizon)
            for variant in MODEL_VARIANTS
            for magnitude_bin in MAGNITUDE_BINS
            for horizon in STAGE4_HORIZONS_DAYS
        )
        actual_keys = tuple(
            (item.variant, item.magnitude_bin, item.horizon_days) for item in results
        )
        if actual_keys != expected_keys:
            raise ValueError("convergence audit must contain every variant/bin/window in order")
        if any(item.input_bundle_sha256 != self.input_bundle_sha256 for item in results):
            raise ValueError("convergence result crossed its frozen input bundle")
        expected_action = (
            CONVERGENCE_PASS_ACTION
            if all(item.passed for item in results)
            else CONVERGENCE_FAILURE_ACTION
        )
        if self.publication_action != expected_action:
            raise ValueError("convergence publication action disagrees with gate results")
        if (
            self.event_log_term_policy != EVENT_LOG_TERM_POLICY
            or self.event_log_term_included
            or self.event_log_term_change != 0.0
        ):
            raise ValueError("event log term must remain excluded and unchanged")
        if self.primary_time_step_days != PRIMARY_TIME_STEP_DAYS:
            raise ValueError("primary convergence time step must remain 1 day")
        if self.reference_time_step_days != REFERENCE_TIME_STEP_DAYS:
            raise ValueError("reference convergence time step must remain 0.5 day")
        if self.relative_tolerance != INTEGRATION_RELATIVE_TOLERANCE:
            raise ValueError("relative convergence tolerance must remain 0.5 percent")
        if self.near_zero_absolute_tolerance != INTEGRATION_NEAR_ZERO_ABSOLUTE_TOLERANCE:
            raise ValueError("near-zero convergence tolerance must remain 1e-10")
        object.__setattr__(self, "results", results)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)

    @property
    def status(self) -> GateStatus:
        return "passed" if self.passed else "failed"

    def _payload_mapping(self) -> dict[str, object]:
        return {
            "event_log_term": {
                "change": self.event_log_term_change,
                "included": self.event_log_term_included,
                "policy": self.event_log_term_policy,
            },
            "input_bundle_sha256": self.input_bundle_sha256,
            "model_sha256": self.model_sha256,
            "near_zero_absolute_tolerance": self.near_zero_absolute_tolerance,
            "numerical_backend": "cpu_numpy_float64_authoritative",
            "primary_time_step_days": self.primary_time_step_days,
            "publication_action": self.publication_action,
            "reference_time_step_days": self.reference_time_step_days,
            "relative_tolerance": self.relative_tolerance,
            "results": [item.as_mapping() for item in self.results],
            "status": self.status,
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self._payload_mapping())

    @property
    def sha256(self) -> str:
        return self.content_sha256

    def as_mapping(self) -> dict[str, object]:
        return {**self._payload_mapping(), "content_sha256": self.content_sha256}


def audit_primary_grid_logical_replay_r1(
    *,
    issue_ids: Sequence[str],
    snapshots: Sequence[Stage3IssueSnapshot],
    grid_family: Stage4GridFamily,
    accepted_primary_issue_tables: Mapping[str, pa.Table],
    source_columns: Sequence[str],
    source_input_sha256: str,
    query_chunk_size: int = 256,
    worker_counts: tuple[int, int] = (1, 2),
) -> PrimaryGridLogicalReplayAuditR1:
    """Replay all formal 25 km issues with the frozen 1/2-worker R1 matrix.

    This audit is deliberately narrower than compensator convergence: it proves
    the final versioned Arrow identity across worker counts before the scoring
    tag, without rebuilding the unrelated 50 km and 12.5 km integration grids.
    """

    frozen_issue_ids = tuple(_identifier(item, label="R1 replay issue_id") for item in issue_ids)
    history = tuple(snapshots)
    if (
        len(history) != FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT
        or len(frozen_issue_ids) != FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT
        or len(set(frozen_issue_ids)) != len(frozen_issue_ids)
        or tuple(item.issue_index for item in history)
        != tuple(range(FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT))
    ):
        raise ValueError("R1 replay must retain all 153 causal issues in order")
    if not isinstance(grid_family, Stage4GridFamily):
        raise TypeError("R1 replay grid family must be Stage4GridFamily")
    accepted_tables = dict(accepted_primary_issue_tables)
    if tuple(accepted_tables) != frozen_issue_ids or any(
        not isinstance(item, pa.Table) for item in accepted_tables.values()
    ):
        raise TypeError("R1 accepted tables must cover all 153 issues in order")
    sources = tuple(source_columns)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("R1 replay source columns must be non-empty and unique")
    for source in sources:
        _assert_target_blind_name(source, label="R1 replay source column")
    _sha256_digest(source_input_sha256, label="R1 replay source input")
    if (
        not isinstance(query_chunk_size, int)
        or isinstance(query_chunk_size, bool)
        or query_chunk_size <= 0
    ):
        raise ValueError("R1 replay query chunk size must be positive")
    if worker_counts != (1, 2):
        raise ValueError("R1 replay must use exactly 1 and 2 workers")

    primary = grid_family.primary_25km
    selected_columns = (*FEATURE_IDENTITY_COLUMNS, *sources)
    worker_receipts: list[tuple[PrimaryGridReproductionReceipt, ...]] = []
    for spatial_workers in worker_counts:
        engine = Stage3FeatureEngine(
            history,
            primary.as_stage3_query_grid(),
            query_chunk_size=query_chunk_size,
            spatial_workers=spatial_workers,
        )
        receipts: list[PrimaryGridReproductionReceipt] = []
        for issue_id, snapshot in zip(frozen_issue_ids, history, strict=True):
            recomputed = engine.build_next_issue().table
            accepted = accepted_tables[issue_id]
            assert_issue_table_matches_frozen_grid(
                accepted,
                issue_time_utc=snapshot.issue_time_utc,
                grid=primary,
            )
            report_ids = accepted["issue_report_id"].combine_chunks().unique().to_pylist()
            if report_ids != [snapshot.summary.issue_report_id]:
                raise ValueError("accepted R1 replay issue report identity changed")
            accepted_sha256 = selected_table_logical_identity_sha256_r1(
                accepted,
                selected_columns,
            )
            recomputed_sha256 = assert_selected_columns_logically_exact_r1(
                accepted,
                recomputed,
                columns=selected_columns,
            )
            receipts.append(
                PrimaryGridReproductionReceipt(
                    issue_id=issue_id,
                    issue_index=snapshot.issue_index,
                    issue_report_id=snapshot.summary.issue_report_id,
                    accepted_table_sha256=accepted_sha256,
                    recomputed_table_sha256=recomputed_sha256,
                )
            )
        worker_receipts.append(tuple(receipts))
    return PrimaryGridLogicalReplayAuditR1(
        grid_id=primary.grid_id,
        source_columns=sources,
        source_input_sha256=source_input_sha256,
        query_chunk_size=query_chunk_size,
        worker_counts=worker_counts,
        receipts_by_worker=(worker_receipts[0], worker_receipts[1]),
    )


def build_target_blind_convergence_inputs(
    *,
    issue_ids: Sequence[str],
    snapshots: Sequence[Stage3IssueSnapshot],
    grid_family: Stage4GridFamily,
    accepted_primary_issue_tables: Mapping[str, pa.Table],
    source_columns: Sequence[str],
    source_input_sha256: str,
    query_chunk_size: int = 256,
    spatial_workers: int = 1,
) -> FrozenTargetBlindConvergenceInputs:
    """Rebuild and freeze the last formal issue before any target capability exists.

    All 153 causal snapshots are replayed independently on each target-independent
    grid.  Every accepted 25 km source/identity Arrow projection must reproduce
    exactly before the final-issue input bundle can be constructed.
    """

    frozen_issue_ids = tuple(_identifier(item, label="convergence issue_id") for item in issue_ids)
    history = tuple(snapshots)
    if (
        len(history) != FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT
        or len(frozen_issue_ids) != FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT
        or len(set(frozen_issue_ids)) != len(frozen_issue_ids)
        or tuple(item.issue_index for item in history)
        != tuple(range(FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT))
    ):
        raise ValueError("convergence inputs must retain all 153 causal issues in order")
    if not isinstance(grid_family, Stage4GridFamily):
        raise TypeError("convergence grid family must be Stage4GridFamily")
    accepted_tables = dict(accepted_primary_issue_tables)
    if tuple(accepted_tables) != frozen_issue_ids or any(
        not isinstance(item, pa.Table) for item in accepted_tables.values()
    ):
        raise TypeError("accepted convergence tables must cover all 153 issues in order")
    sources = tuple(source_columns)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("convergence source columns must be non-empty and unique")
    _sha256_digest(source_input_sha256, label="convergence source input")
    if (
        not isinstance(query_chunk_size, int)
        or isinstance(query_chunk_size, bool)
        or query_chunk_size <= 0
    ):
        raise ValueError("convergence query chunk size must be positive")
    if (
        not isinstance(spatial_workers, int)
        or isinstance(spatial_workers, bool)
        or spatial_workers <= 0
    ):
        raise ValueError("convergence spatial worker count must be positive")

    selected = history[-1]
    primary = grid_family.primary_25km
    selected_columns = (*FEATURE_IDENTITY_COLUMNS, *sources)

    frozen_grids: list[TargetBlindGridFeatures] = []
    reproductions: list[PrimaryGridReproductionReceipt] = []
    for grid in grid_family.grids():
        engine = Stage3FeatureEngine(
            history,
            grid.as_stage3_query_grid(),
            query_chunk_size=query_chunk_size,
            spatial_workers=spatial_workers,
        )
        latest: pa.Table | None = None
        for issue_id, snapshot in zip(frozen_issue_ids, history, strict=True):
            latest = engine.build_next_issue().table
            if grid.cell_size_km == 25.0:
                accepted = accepted_tables[issue_id]
                assert_issue_table_matches_frozen_grid(
                    accepted,
                    issue_time_utc=snapshot.issue_time_utc,
                    grid=primary,
                )
                report_ids = accepted["issue_report_id"].combine_chunks().unique().to_pylist()
                if report_ids != [snapshot.summary.issue_report_id]:
                    raise ValueError("accepted convergence issue report identity changed")
                accepted_sha256 = selected_table_logical_identity_sha256_r1(
                    accepted,
                    selected_columns,
                )
                recomputed_sha256 = assert_selected_columns_logically_exact_r1(
                    accepted,
                    latest,
                    columns=selected_columns,
                )
                reproductions.append(
                    PrimaryGridReproductionReceipt(
                        issue_id=issue_id,
                        issue_index=snapshot.issue_index,
                        issue_report_id=snapshot.summary.issue_report_id,
                        accepted_table_sha256=accepted_sha256,
                        recomputed_table_sha256=recomputed_sha256,
                    )
                )
        if latest is None:  # pragma: no cover - history is non-empty by construction
            raise AssertionError("convergence reconstruction produced no issue table")
        table_sha256 = selected_table_logical_identity_sha256_r1(latest, selected_columns)
        frozen_grids.append(
            TargetBlindGridFeatures(
                grid_id=grid.grid_id,
                cell_size_km=grid.cell_size_km,
                cell_ids=grid.cell_ids,
                feature_columns=concatenate_source_columns(
                    (latest,),
                    source_columns=sources,
                ),
                feature_table_identity_sha256=table_sha256,
            )
        )
    if len(reproductions) != FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT:
        raise AssertionError("convergence grid family omitted the primary grid")
    return FrozenTargetBlindConvergenceInputs(
        issue_id=frozen_issue_ids[-1],
        issue_report_id=selected.summary.issue_report_id,
        selected_issue_index=selected.issue_index,
        selected_state_snapshot_id=selected.state_snapshot_id,
        selected_lineage_digest=selected.lineage_digest,
        source_columns=sources,
        source_input_sha256=source_input_sha256,
        grids=tuple(frozen_grids),  # type: ignore[arg-type]
        primary_reproduction_receipts=tuple(reproductions),
        query_chunk_size=query_chunk_size,
        spatial_workers=spatial_workers,
    )


def bind_target_blind_convergence_background(
    *,
    inputs: FrozenTargetBlindConvergenceInputs,
    formal_background: Stage4BackgroundFit,
) -> tuple[RecomputedGridInputs, RecomputedGridInputs, RecomputedGridInputs]:
    """Bind the post-target causal background without changing frozen features."""

    if not isinstance(inputs, FrozenTargetBlindConvergenceInputs):
        raise TypeError("convergence inputs must be pretarget-frozen")
    if not isinstance(formal_background, Stage4BackgroundFit):
        raise TypeError("convergence background must be Stage4BackgroundFit")
    if formal_background.snapshot.evaluation_id != "formal-validation":
        raise ValueError("convergence requires the formal-validation background")
    output: list[RecomputedGridInputs] = []
    for features in inputs.grids:
        background = formal_background.grid(features.cell_size_km)
        if background.grid_id != features.grid_id or background.cell_ids != features.cell_ids:
            raise ValueError("formal background differs from frozen target-blind grid identity")
        mass = background.spatial_cell_mass
        output.append(
            RecomputedGridInputs(
                grid_id=features.grid_id,
                cell_size_km=features.cell_size_km,
                background_spatial_mass_by_cell_and_bin=np.column_stack((mass, mass)),
                feature_columns=features.feature_columns,
            )
        )
    return tuple(output)  # type: ignore[return-value]


def audit_frozen_compensator_convergence(
    *,
    model: FrozenConvergenceModel,
    inputs: FrozenTargetBlindConvergenceInputs,
    formal_background: Stage4BackgroundFit,
) -> CompensatorConvergenceAudit:
    """Apply fitted coefficients to already-frozen inputs; never rebuild fields."""

    if tuple(item.source_column for item in model.preprocessor.contracts) != (
        inputs.source_columns
    ):
        raise ValueError("fitted formal preprocessor differs from frozen convergence sources")
    return audit_compensator_convergence(
        model=model,
        grids=bind_target_blind_convergence_background(
            inputs=inputs,
            formal_background=formal_background,
        ),
    )


def _input_bundle_sha256(
    *,
    model: FrozenConvergenceModel,
    grids: tuple[RecomputedGridInputs, ...],
    designs: Mapping[float, FloatArray],
) -> str:
    return canonical_mapping_sha256(
        {
            "grids": [
                {
                    "background_spatial_mass_float64_hex": _float_matrix_hex(
                        grid.background_spatial_mass_by_cell_and_bin
                    ),
                    "cell_size_km_hex": grid.cell_size_km.hex(),
                    "design_values_float64_hex": _float_matrix_hex(designs[grid.cell_size_km]),
                    "grid_id": grid.grid_id,
                }
                for grid in grids
            ],
            "model_sha256": model.content_sha256,
        }
    )


def _linear_predictor(
    *,
    design_values: FloatArray,
    variant: FrozenVariantCoefficients,
) -> FloatArray:
    if variant.variant == "background_no_increment":
        return readonly_float_vector(
            "background-only linear predictor",
            np.zeros(design_values.shape[0], dtype=np.float64),
        )
    selected = design_values[:, variant.design_column_indices]
    predictor = np.asarray(selected @ variant.beta, dtype=np.float64)
    return readonly_float_vector("frozen issue linear predictor", predictor)


def audit_compensator_convergence(
    *,
    model: FrozenConvergenceModel,
    grids: tuple[RecomputedGridInputs, ...],
    horizons_days: tuple[int, ...] = STAGE4_HORIZONS_DAYS,
    primary_time_step_days: float = PRIMARY_TIME_STEP_DAYS,
    reference_time_step_days: float = REFERENCE_TIME_STEP_DAYS,
) -> CompensatorConvergenceAudit:
    """Audit all variant/bin/window compensators without event or target terms."""

    if tuple(horizons_days) != STAGE4_HORIZONS_DAYS:
        raise ValueError("convergence audit must retain all five frozen horizons in order")
    if primary_time_step_days != PRIMARY_TIME_STEP_DAYS:
        raise ValueError("primary convergence time step must be exactly 1 day")
    if reference_time_step_days != REFERENCE_TIME_STEP_DAYS:
        raise ValueError("reference convergence time step must be exactly 0.5 day")
    ordered_grids = tuple(grids)
    if tuple(item.cell_size_km for item in ordered_grids) != GRID_SIZES_KM:
        raise ValueError("convergence grids must be ordered exactly 50, 25, and 12.5 km")
    if len({item.grid_id for item in ordered_grids}) != len(ordered_grids):
        raise ValueError("convergence grid IDs must be unique")

    design_matrices = {
        grid.cell_size_km: model.preprocessor.transform(grid.feature_columns)
        for grid in ordered_grids
    }
    design_values = {cell_size: matrix.values for cell_size, matrix in design_matrices.items()}
    bundle_sha256 = _input_bundle_sha256(
        model=model,
        grids=ordered_grids,
        designs=design_values,
    )
    by_size = {grid.cell_size_km: grid for grid in ordered_grids}
    results: list[CompensatorConvergenceResult] = []
    for variant_name in MODEL_VARIANTS:
        variant = model.variant(variant_name)
        predictors = {
            cell_size: _linear_predictor(
                design_values=values,
                variant=variant,
            )
            for cell_size, values in design_values.items()
        }
        for bin_index, magnitude_bin in enumerate(MAGNITUDE_BINS):
            rate = model.rate_head.by_id(magnitude_bin).rate_multiplier
            for horizon in STAGE4_HORIZONS_DAYS:
                primary: dict[float, float] = {}
                for cell_size in GRID_SIZES_KM:
                    primary[cell_size] = integrate_conditional_intensity(
                        background_spatial_mass=by_size[
                            cell_size
                        ].background_spatial_mass_by_cell_and_bin[:, bin_index],
                        issue_linear_predictor=predictors[cell_size],
                        rate_multiplier=rate,
                        horizon_days=float(horizon),
                        maximum_step_days=primary_time_step_days,
                    )
                reference_25 = integrate_conditional_intensity(
                    background_spatial_mass=by_size[25.0].background_spatial_mass_by_cell_and_bin[
                        :, bin_index
                    ],
                    issue_linear_predictor=predictors[25.0],
                    rate_multiplier=rate,
                    horizon_days=float(horizon),
                    maximum_step_days=reference_time_step_days,
                )
                results.append(
                    CompensatorConvergenceResult(
                        variant=variant_name,
                        magnitude_bin=magnitude_bin,
                        horizon_days=horizon,
                        input_bundle_sha256=bundle_sha256,
                        total_compensator_50km_step_1d=primary[50.0],
                        total_compensator_25km_step_1d=primary[25.0],
                        total_compensator_12_5km_step_1d=primary[12.5],
                        total_compensator_25km_step_0_5d=reference_25,
                        spatial_25_vs_12_5=compare_integrals(
                            primary[25.0],
                            primary[12.5],
                        ),
                        temporal_1d_vs_0_5d=compare_integrals(
                            primary[25.0],
                            reference_25,
                        ),
                        coarse_50_vs_25=compare_integrals(
                            primary[50.0],
                            primary[25.0],
                        ),
                    )
                )
    result_tuple = tuple(results)
    action = (
        CONVERGENCE_PASS_ACTION
        if all(item.passed for item in result_tuple)
        else CONVERGENCE_FAILURE_ACTION
    )
    return CompensatorConvergenceAudit(
        model_sha256=model.content_sha256,
        input_bundle_sha256=bundle_sha256,
        results=result_tuple,
        publication_action=action,
        primary_time_step_days=primary_time_step_days,
        reference_time_step_days=reference_time_step_days,
    )


def coarse_relative_difference(candidate_50km: float, reference_25km: float) -> float:
    """Return the non-gating 50 km trend difference with the frozen denominator."""

    candidate = float(candidate_50km)
    reference = float(reference_25km)
    if any(not math.isfinite(value) or value < 0.0 for value in (candidate, reference)):
        raise ValueError("coarse diagnostic integrals must be finite and non-negative")
    return abs(candidate - reference) / max(
        abs(reference),
        SPATIAL_RELATIVE_DENOMINATOR_FLOOR,
    )


__all__ = [
    "CONVERGENCE_FAILURE_ACTION",
    "CONVERGENCE_PASS_ACTION",
    "EVENT_LOG_TERM_POLICY",
    "FORMAL_CONVERGENCE_HISTORY_ISSUE_COUNT",
    "GRID_SIZES_KM",
    "MAGNITUDE_BINS",
    "MODEL_VARIANTS",
    "CompensatorConvergenceAudit",
    "CompensatorConvergenceResult",
    "FrozenConvergenceModel",
    "FrozenTargetBlindConvergenceInputs",
    "FrozenVariantCoefficients",
    "GateStatus",
    "MagnitudeBin",
    "ModelVariant",
    "PrimaryGridLogicalReplayAuditR1",
    "PrimaryGridReproductionReceipt",
    "PublicationAction",
    "RecomputedGridInputs",
    "TargetBlindGridFeatures",
    "audit_compensator_convergence",
    "audit_frozen_compensator_convergence",
    "audit_primary_grid_logical_replay_r1",
    "bind_target_blind_convergence_background",
    "build_target_blind_convergence_inputs",
    "coarse_relative_difference",
]
