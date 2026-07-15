"""In-memory reconstruction of the frozen stage-4 spatial-Poisson background.

This adapter accepts only an already-authorized :class:`Stage4TargetCatalog`, a
target-independent grid family, and the already-loaded protocol mapping.  It has
no path or byte-loading API.  Training remains causal at the frozen background
snapshot cutoff and uses the accepted common Mc and 75 km KDE bandwidth.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.anomaly_increment.grid_features import (
    Stage4GridFamily,
    Stage4IntegrationGrid,
)
from seismoflux.background.poisson import (
    SpatialPoissonModel,
    SpatialQuadrature,
    fit_spatial_poisson_family,
)
from seismoflux.data.common import canonical_json_bytes

if TYPE_CHECKING:
    from seismoflux.anomaly_increment.targets import Stage4TargetCatalog

EvaluationId: TypeAlias = Literal[
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
]
SnapshotRole: TypeAlias = Literal["development", "validation"]
FloatArray: TypeAlias = NDArray[np.float64]

FROZEN_BACKGROUND_VARIANT_ID = "spatial_poisson/gaussian_kde_bw75km"
FROZEN_KDE_BANDWIDTH_KM = 75.0
FROZEN_COMMON_MC = 4.0
FROZEN_KDE_CHUNK_SIZE = 256
FROZEN_STUDY_AREA_SHA256 = "5e5dcf012e080882161c95bf592a1ee39a0f0fdad7114bcff58d645aeb30bb02"
FROZEN_COMPENSATOR_DOMAIN_ID = "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55"
CATALOG_ANCHOR_UTC = datetime(1970, 1, 1, tzinfo=UTC)

_FROZEN_SNAPSHOT_FIELDS: dict[SnapshotRole, dict[str, object]] = {
    "development": {
        "snapshot_id": "fold_4",
        "parameter_snapshot_id": (
            "83a0c60d4b62ba6a6e849ac2d5f430001d054b7aec3af40f76193180a18bf4c5"
        ),
        "fit_end_utc": "2019-12-31T16:00:00Z",
        "support_id": "local-support-788851371baf0e3b",
        "compensator_domain_id": FROZEN_COMPENSATOR_DOMAIN_ID,
        "common_mc": FROZEN_COMMON_MC,
        "supported_area_fraction": 1.0,
    },
    "validation": {
        "snapshot_id": "final_validation",
        "parameter_snapshot_id": (
            "252f14cad07205b10c1a605fdd21613044bc4072c98bcaa74cf357b7d766ed02"
        ),
        "fit_end_utc": "2023-06-30T16:00:00Z",
        "support_id": "local-support-f6816ab6c6581306",
        "compensator_domain_id": FROZEN_COMPENSATOR_DOMAIN_ID,
        "common_mc": FROZEN_COMMON_MC,
        "supported_area_fraction": 1.0,
    },
}


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _readonly_float(values: object, *, label: str) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite one-dimensional float64 vector")
    result.setflags(write=False)
    return result


def _parse_frozen_utc(value: object, *, label: str) -> datetime:
    text = _string(value, label=label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone aware")
    return parsed.astimezone(UTC)


def _grid_identity_payload(grid: Stage4IntegrationGrid) -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "stage4_target_independent_integration_grid",
        "equal_area_crs": grid.equal_area_crs,
        "cell_size_km_hex": float(grid.cell_size_km).hex(),
        "cells": [
            {
                "cell_id": cell_id,
                "row": int(row),
                "column": int(column),
                "query_x_m_hex": float(x_m).hex(),
                "query_y_m_hex": float(y_m).hex(),
                "clipped_area_km2_hex": float(area).hex(),
            }
            for cell_id, row, column, (x_m, y_m), area in zip(
                grid.cell_ids,
                grid.rows,
                grid.columns,
                grid.query_xy_m,
                grid.clipped_area_km2,
                strict=True,
            )
        ],
    }


def _computed_grid_id(grid: Stage4IntegrationGrid) -> str:
    return hashlib.sha256(canonical_json_bytes(_grid_identity_payload(grid))).hexdigest()


@dataclass(frozen=True, slots=True)
class BackgroundDomainBinding:
    """Upstream-verified study/support identity and exact grid identities."""

    study_area_sha256: str
    compensator_domain_id: str
    supported_area_km2: float
    grid_ids: tuple[str, str, str]

    def __post_init__(self) -> None:
        _sha256(self.study_area_sha256, label="study_area_sha256")
        _sha256(self.compensator_domain_id, label="compensator_domain_id")
        if not math.isfinite(self.supported_area_km2) or self.supported_area_km2 <= 0.0:
            raise ValueError("supported_area_km2 must be finite and positive")
        if len(set(self.grid_ids)) != 3:
            raise ValueError("the 50/25/12.5 km grid identities must be distinct")
        for index, grid_id in enumerate(self.grid_ids):
            _sha256(grid_id, label=f"grid_ids[{index}]")

    @classmethod
    def from_verified_grid_family(
        cls,
        grid_family: Stage4GridFamily,
        *,
        study_area_sha256: str,
        compensator_domain_id: str,
    ) -> BackgroundDomainBinding:
        """Capture grid IDs after upstream study-area hash verification."""

        return cls(
            study_area_sha256=study_area_sha256,
            compensator_domain_id=compensator_domain_id,
            supported_area_km2=grid_family.reference_12_5km.total_area_km2,
            grid_ids=(
                grid_family.coarse_50km.grid_id,
                grid_family.primary_25km.grid_id,
                grid_family.reference_12_5km.grid_id,
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenBackgroundSnapshot:
    evaluation_id: EvaluationId
    role: SnapshotRole
    snapshot_id: str
    parameter_snapshot_id: str
    fit_end_utc: datetime
    support_id: str
    compensator_domain_id: str
    common_mc: float
    bandwidth_km: float
    supported_area_fraction: float
    study_area_sha256: str

    def __post_init__(self) -> None:
        if self.evaluation_id not in {
            "development-fold-1",
            "development-fold-2",
            "development-fold-3",
            "formal-validation",
        }:
            raise ValueError("evaluation_id is outside the four formal stage-4 scopes")
        expected_role: SnapshotRole = (
            "validation" if self.evaluation_id == "formal-validation" else "development"
        )
        if self.role != expected_role:
            raise ValueError("background snapshot role differs from its evaluation scope")
        offset = self.fit_end_utc.utcoffset()
        if self.fit_end_utc.tzinfo is None or offset is None:
            raise ValueError("fit_end_utc must be timezone aware")
        if offset.total_seconds() != 0.0:
            raise ValueError("fit_end_utc must be UTC")
        if self.fit_end_utc <= CATALOG_ANCHOR_UTC:
            raise ValueError("background fit cutoff must follow the catalog anchor")
        _sha256(self.parameter_snapshot_id, label="parameter_snapshot_id")
        _sha256(self.compensator_domain_id, label="compensator_domain_id")
        _sha256(self.study_area_sha256, label="study_area_sha256")
        if self.common_mc != FROZEN_COMMON_MC:
            raise ValueError("stage-4 background common Mc changed")
        if self.bandwidth_km != FROZEN_KDE_BANDWIDTH_KM:
            raise ValueError("stage-4 background bandwidth changed")
        if self.supported_area_fraction != 1.0:
            raise ValueError("stage-4 background must retain the full supported study area")


def resolve_frozen_background_snapshot(
    protocol: Mapping[str, object],
    *,
    evaluation_id: EvaluationId,
) -> FrozenBackgroundSnapshot:
    """Cross-check the loaded protocol against every accepted background constant."""

    if evaluation_id not in {
        "development-fold-1",
        "development-fold-2",
        "development-fold-3",
        "formal-validation",
    }:
        raise ValueError("evaluation_id is outside the four formal stage-4 scopes")
    role: SnapshotRole = "validation" if evaluation_id == "formal-validation" else "development"
    background = _mapping(protocol.get("background"), label="background")
    if background.get("background_variant_id") != FROZEN_BACKGROUND_VARIANT_ID:
        raise ValueError("frozen background variant identity changed")
    if background.get("family") != "spatial_poisson":
        raise ValueError("frozen background family changed")
    if background.get("bandwidth_km") != FROZEN_KDE_BANDWIDTH_KM:
        raise ValueError("frozen background KDE bandwidth changed")
    if background.get("model_reselection_forbidden") is not True:
        raise ValueError("stage-4 background model reselection must remain forbidden")
    declaration = _mapping(background.get(role), label=f"background.{role}")
    frozen = _FROZEN_SNAPSHOT_FIELDS[role]
    for key, expected in frozen.items():
        if declaration.get(key) != expected:
            raise ValueError(f"background.{role}.{key} differs from the frozen snapshot")

    inputs = _mapping(protocol.get("inputs"), label="inputs")
    study_area = _mapping(inputs.get("study_area"), label="inputs.study_area")
    study_area_sha256 = _string(study_area.get("sha256"), label="inputs.study_area.sha256")
    if study_area_sha256 != FROZEN_STUDY_AREA_SHA256:
        raise ValueError("stage-4 study-area identity changed")
    return FrozenBackgroundSnapshot(
        evaluation_id=evaluation_id,
        role=role,
        snapshot_id=cast(str, declaration["snapshot_id"]),
        parameter_snapshot_id=cast(str, declaration["parameter_snapshot_id"]),
        fit_end_utc=_parse_frozen_utc(
            declaration["fit_end_utc"], label=f"background.{role}.fit_end_utc"
        ),
        support_id=cast(str, declaration["support_id"]),
        compensator_domain_id=cast(str, declaration["compensator_domain_id"]),
        common_mc=float(cast(float, declaration["common_mc"])),
        bandwidth_km=float(cast(float, background["bandwidth_km"])),
        supported_area_fraction=float(cast(float, declaration["supported_area_fraction"])),
        study_area_sha256=study_area_sha256,
    )


@dataclass(frozen=True, slots=True)
class BackgroundGridDensity:
    grid_id: str
    cell_size_km: float
    cell_ids: tuple[str, ...]
    spatial_density_per_km2: FloatArray
    spatial_cell_mass: FloatArray
    expected_cell_count_per_day: FloatArray

    def __post_init__(self) -> None:
        _sha256(self.grid_id, label="grid_id")
        count = len(self.cell_ids)
        density = _readonly_float(self.spatial_density_per_km2, label="spatial_density_per_km2")
        masses = _readonly_float(self.spatial_cell_mass, label="spatial_cell_mass")
        expected = _readonly_float(
            self.expected_cell_count_per_day, label="expected_cell_count_per_day"
        )
        if count == 0 or any(values.shape != (count,) for values in (density, masses, expected)):
            raise ValueError("background grid columns must have one nonzero common length")
        if len(set(self.cell_ids)) != count:
            raise ValueError("background grid cell IDs must be unique")
        if np.any(density <= 0.0) or np.any(masses <= 0.0) or np.any(expected <= 0.0):
            raise ValueError("background density, mass, and expected count must be positive")
        object.__setattr__(self, "spatial_density_per_km2", density)
        object.__setattr__(self, "spatial_cell_mass", masses)
        object.__setattr__(self, "expected_cell_count_per_day", expected)

    @property
    def spatial_mass_sum(self) -> float:
        return float(np.sum(self.spatial_cell_mass, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class CausalBackgroundAudit:
    catalog_event_count: int
    training_event_count: int
    excluded_before_anchor_count: int
    excluded_after_origin_cutoff_count: int
    excluded_after_availability_cutoff_count: int
    excluded_below_mc_count: int
    excluded_outside_support_count: int
    latest_training_origin_utc: datetime
    latest_training_available_at_utc: datetime
    post_cutoff_training_event_count: Literal[0]

    def __post_init__(self) -> None:
        counts = (
            self.catalog_event_count,
            self.training_event_count,
            self.excluded_before_anchor_count,
            self.excluded_after_origin_cutoff_count,
            self.excluded_after_availability_cutoff_count,
            self.excluded_below_mc_count,
            self.excluded_outside_support_count,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("causal background audit counts must be non-negative integers")
        if self.training_event_count <= 0 or self.training_event_count > self.catalog_event_count:
            raise ValueError("causal background audit training count is invalid")
        if self.post_cutoff_training_event_count != 0:
            raise ValueError("post-cutoff events may never enter background training")


@dataclass(frozen=True, slots=True)
class Stage4BackgroundFit:
    snapshot: FrozenBackgroundSnapshot
    domain: BackgroundDomainBinding
    training_event_ids: tuple[str, ...]
    training_duration_days: float
    rate_per_day: float
    normalization_mass: float
    grids: tuple[BackgroundGridDensity, BackgroundGridDensity, BackgroundGridDensity]
    causal_audit: CausalBackgroundAudit
    scientific_identity_sha256: str
    model: SpatialPoissonModel

    def __post_init__(self) -> None:
        if len(self.training_event_ids) != self.causal_audit.training_event_count:
            raise ValueError("training event IDs disagree with the causal audit")
        if len(set(self.training_event_ids)) != len(self.training_event_ids):
            raise ValueError("background training physical-event IDs must be unique")
        if not math.isfinite(self.training_duration_days) or self.training_duration_days <= 0.0:
            raise ValueError("training_duration_days must be finite and positive")
        if not math.isfinite(self.rate_per_day) or self.rate_per_day <= 0.0:
            raise ValueError("rate_per_day must be finite and positive")
        if not math.isfinite(self.normalization_mass) or self.normalization_mass <= 0.0:
            raise ValueError("normalization_mass must be finite and positive")
        if tuple(item.grid_id for item in self.grids) != self.domain.grid_ids:
            raise ValueError("background grid outputs differ from the domain binding")
        if tuple(item.cell_size_km for item in self.grids) != (50.0, 25.0, 12.5):
            raise ValueError("background grid output order changed")
        if not math.isclose(self.grids[-1].spatial_mass_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("12.5 km reference background density does not integrate to one")
        _sha256(self.scientific_identity_sha256, label="scientific_identity_sha256")
        if self.model.bandwidth_km != FROZEN_KDE_BANDWIDTH_KM:
            raise ValueError("retained background model has another bandwidth")
        if self.model.rate_per_day != self.rate_per_day:
            raise ValueError("retained background model has another event rate")

    def grid(self, cell_size_km: float) -> BackgroundGridDensity:
        requested = float(cell_size_km)
        for item in self.grids:
            if item.cell_size_km == requested:
                return item
        raise KeyError(f"stage-4 background has no {requested:g} km grid")


def _verify_domain_binding(
    grid_family: Stage4GridFamily,
    domain: BackgroundDomainBinding,
    snapshot: FrozenBackgroundSnapshot,
) -> None:
    if domain.study_area_sha256 != snapshot.study_area_sha256:
        raise ValueError("background domain uses another study-area identity")
    if domain.compensator_domain_id != snapshot.compensator_domain_id:
        raise ValueError("background domain uses another compensator/support identity")
    grids = grid_family.grids()
    observed_ids = tuple(grid.grid_id for grid in grids)
    if observed_ids != domain.grid_ids:
        raise ValueError("background grids differ from the upstream frozen grid identities")
    for grid in grids:
        if grid.grid_id != _computed_grid_id(grid):
            raise ValueError("background grid ID does not match its target-independent content")
        if not math.isclose(
            grid.total_area_km2,
            domain.supported_area_km2,
            rel_tol=1.0e-12,
            abs_tol=1.0e-6,
        ):
            raise ValueError("background grid area differs from the frozen supported area")


def _quadrature(grid: Stage4IntegrationGrid) -> SpatialQuadrature:
    return SpatialQuadrature(
        cell_ids=grid.cell_ids,
        x_km=np.asarray(grid.query_xy_m[:, 0] / 1_000.0, dtype=np.float64),
        y_km=np.asarray(grid.query_xy_m[:, 1] / 1_000.0, dtype=np.float64),
        area_km2=grid.clipped_area_km2,
    )


def _grid_density(
    grid: Stage4IntegrationGrid,
    quadrature: SpatialQuadrature,
    model: SpatialPoissonModel,
) -> BackgroundGridDensity:
    masses = model.cell_masses(quadrature)
    density = np.asarray(masses / quadrature.area_km2, dtype=np.float64)
    expected = np.asarray(model.rate_per_day * masses, dtype=np.float64)
    return BackgroundGridDensity(
        grid_id=grid.grid_id,
        cell_size_km=grid.cell_size_km,
        cell_ids=grid.cell_ids,
        spatial_density_per_km2=density,
        spatial_cell_mass=masses,
        expected_cell_count_per_day=expected,
    )


def _scientific_identity(
    *,
    snapshot: FrozenBackgroundSnapshot,
    domain: BackgroundDomainBinding,
    catalog: Stage4TargetCatalog,
    training_indices: NDArray[np.int64],
    training_duration_days: float,
    model: SpatialPoissonModel,
    grids: tuple[BackgroundGridDensity, BackgroundGridDensity, BackgroundGridDensity],
) -> str:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "background_variant_id": FROZEN_BACKGROUND_VARIANT_ID,
        "snapshot_id": snapshot.snapshot_id,
        "parameter_snapshot_id": snapshot.parameter_snapshot_id,
        "fit_end_utc": snapshot.fit_end_utc.isoformat().replace("+00:00", "Z"),
        "support_id": snapshot.support_id,
        "compensator_domain_id": snapshot.compensator_domain_id,
        "study_area_sha256": snapshot.study_area_sha256,
        "common_mc_hex": snapshot.common_mc.hex(),
        "bandwidth_km_hex": snapshot.bandwidth_km.hex(),
        "supported_area_km2_hex": domain.supported_area_km2.hex(),
        "grid_ids": domain.grid_ids,
        "training_duration_days_hex": training_duration_days.hex(),
        "rate_per_day_hex": model.rate_per_day.hex(),
        "normalization_mass_hex": model.normalization_mass.hex(),
        "training_events": [
            {
                "event_id": str(catalog.event_id[index]),
                "origin_time_utc": catalog.origin_time_utc[index]
                .isoformat()
                .replace("+00:00", "Z"),
                "available_at_utc": catalog.available_at_utc[index]
                .isoformat()
                .replace("+00:00", "Z"),
                "x_m_hex": float(catalog.x_m[index]).hex(),
                "y_m_hex": float(catalog.y_m[index]).hex(),
                "magnitude_hex": float(catalog.magnitude[index]).hex(),
            }
            for index in training_indices
        ],
        "grid_cell_mass_hex": {
            item.grid_id: tuple(float(value).hex() for value in item.spatial_cell_mass)
            for item in grids
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def rebuild_stage4_background(
    catalog: Stage4TargetCatalog,
    grid_family: Stage4GridFamily,
    *,
    protocol: Mapping[str, object],
    evaluation_id: EvaluationId,
    domain: BackgroundDomainBinding,
) -> Stage4BackgroundFit:
    """Rebuild one causal 75 km background from an authorized in-memory catalog."""

    from seismoflux.anomaly_increment.targets import Stage4TargetCatalog as TargetCatalogType

    if not isinstance(catalog, TargetCatalogType):
        raise TypeError("catalog must be an authorized in-memory Stage4TargetCatalog")
    if not isinstance(grid_family, Stage4GridFamily):
        raise TypeError("grid_family must be a target-independent Stage4GridFamily")
    if not isinstance(domain, BackgroundDomainBinding):
        raise TypeError("domain must be an upstream-verified BackgroundDomainBinding")
    snapshot = resolve_frozen_background_snapshot(protocol, evaluation_id=evaluation_id)
    _verify_domain_binding(grid_family, domain, snapshot)
    expected_order = sorted(
        range(len(catalog)),
        key=lambda index: (catalog.origin_time_utc[index], str(catalog.event_id[index])),
    )
    if expected_order != list(range(len(catalog))):
        raise ValueError("authorized catalog order must be origin time then physical event ID")

    origin_after_anchor = np.fromiter(
        (value >= CATALOG_ANCHOR_UTC for value in catalog.origin_time_utc),
        dtype=np.bool_,
        count=len(catalog),
    )
    origin_by_cutoff = np.fromiter(
        (value <= snapshot.fit_end_utc for value in catalog.origin_time_utc),
        dtype=np.bool_,
        count=len(catalog),
    )
    available_by_cutoff = np.fromiter(
        (value <= snapshot.fit_end_utc for value in catalog.available_at_utc),
        dtype=np.bool_,
        count=len(catalog),
    )
    magnitude_eligible = np.asarray(catalog.magnitude >= snapshot.common_mc, dtype=np.bool_)
    inside_support = np.asarray(catalog.inside_study_area, dtype=np.bool_)
    training_mask = (
        origin_after_anchor
        & origin_by_cutoff
        & available_by_cutoff
        & magnitude_eligible
        & inside_support
    )
    training_indices = np.asarray(np.flatnonzero(training_mask), dtype=np.int64)
    if training_indices.size == 0:
        raise ValueError("frozen background snapshot has zero eligible training events")
    training_duration_days = (snapshot.fit_end_utc - CATALOG_ANCHOR_UTC).total_seconds() / 86_400.0
    quadratures = tuple(_quadrature(grid) for grid in grid_family.grids())
    family = fit_spatial_poisson_family(
        catalog.x_m[training_indices] / 1_000.0,
        catalog.y_m[training_indices] / 1_000.0,
        training_duration_days=training_duration_days,
        normalization_quadrature=quadratures[-1],
        bandwidths_km=(FROZEN_KDE_BANDWIDTH_KM,),
        chunk_size=FROZEN_KDE_CHUNK_SIZE,
    )
    model = family[FROZEN_KDE_BANDWIDTH_KM]
    grid_outputs = cast(
        tuple[BackgroundGridDensity, BackgroundGridDensity, BackgroundGridDensity],
        tuple(
            _grid_density(grid, quadrature, model)
            for grid, quadrature in zip(grid_family.grids(), quadratures, strict=True)
        ),
    )
    training_origins = tuple(catalog.origin_time_utc[index] for index in training_indices)
    training_available = tuple(catalog.available_at_utc[index] for index in training_indices)
    post_cutoff_count = sum(
        origin > snapshot.fit_end_utc or available > snapshot.fit_end_utc
        for origin, available in zip(training_origins, training_available, strict=True)
    )
    audit = CausalBackgroundAudit(
        catalog_event_count=len(catalog),
        training_event_count=int(training_indices.size),
        excluded_before_anchor_count=int(np.count_nonzero(~origin_after_anchor)),
        excluded_after_origin_cutoff_count=int(np.count_nonzero(~origin_by_cutoff)),
        excluded_after_availability_cutoff_count=int(np.count_nonzero(~available_by_cutoff)),
        excluded_below_mc_count=int(np.count_nonzero(~magnitude_eligible)),
        excluded_outside_support_count=int(np.count_nonzero(~inside_support)),
        latest_training_origin_utc=max(training_origins),
        latest_training_available_at_utc=max(training_available),
        post_cutoff_training_event_count=cast(Literal[0], post_cutoff_count),
    )
    identity = _scientific_identity(
        snapshot=snapshot,
        domain=domain,
        catalog=catalog,
        training_indices=training_indices,
        training_duration_days=training_duration_days,
        model=model,
        grids=grid_outputs,
    )
    return Stage4BackgroundFit(
        snapshot=snapshot,
        domain=domain,
        training_event_ids=tuple(str(catalog.event_id[index]) for index in training_indices),
        training_duration_days=training_duration_days,
        rate_per_day=model.rate_per_day,
        normalization_mass=model.normalization_mass,
        grids=grid_outputs,
        causal_audit=audit,
        scientific_identity_sha256=identity,
        model=model,
    )


__all__ = [
    "CATALOG_ANCHOR_UTC",
    "FROZEN_BACKGROUND_VARIANT_ID",
    "FROZEN_COMMON_MC",
    "FROZEN_COMPENSATOR_DOMAIN_ID",
    "FROZEN_KDE_BANDWIDTH_KM",
    "FROZEN_KDE_CHUNK_SIZE",
    "FROZEN_STUDY_AREA_SHA256",
    "BackgroundDomainBinding",
    "BackgroundGridDensity",
    "CausalBackgroundAudit",
    "EvaluationId",
    "FrozenBackgroundSnapshot",
    "SnapshotRole",
    "Stage4BackgroundFit",
    "rebuild_stage4_background",
    "resolve_frozen_background_snapshot",
]
