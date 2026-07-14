"""Future validation ensembles driven by the final frozen G1-LS support domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from seismoflux.background.adapters import build_etas_model_spec
from seismoflux.background.catalog import EarthquakeCatalog, StudyArea
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.future import (
    ValidationFutureEnsembles,
    simulate_all_validation_issue_ensembles,
)
from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.background.issues import FrozenIssueCalendar
from seismoflux.background.local_support_runtime import LocalSupportRuntime
from seismoflux.background.pipeline_local_support import LocalSupportPrimaryPipelineResult
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    require_background_scoring_authorized,
)
from seismoflux.background.workflow import (
    build_local_support_etas_parent_roles,
    catalog_etas_events,
)


@dataclass(frozen=True, slots=True)
class LocalSupportFutureOutcome:
    status: Literal["succeeded", "not_run"]
    authorization_id: str
    support_id: str
    compensator_domain_id: str
    parameter_snapshot_id: str | None
    ensembles: ValidationFutureEnsembles | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if self.status == "succeeded":
            if (
                self.ensembles is None
                or self.failure_reason is not None
                or self.parameter_snapshot_id is None
            ):
                raise ValueError("successful local future outcome is incomplete")
        elif self.status == "not_run":
            if self.ensembles is not None or not self.failure_reason:
                raise ValueError("not-run local future outcome requires one reason")
        else:
            raise ValueError("unknown local future outcome status")


def _support_study_area(projected: BaseGeometry) -> StudyArea:
    inverse = Transformer.from_crs(
        CRS.from_user_input(EQUAL_AREA_CRS),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    geographic = transform(inverse.transform, projected)
    return StudyArea(
        geographic=geographic,
        projected=projected,
        equal_area_crs=CRS.from_user_input(EQUAL_AREA_CRS).to_string(),
        area_km2=float(projected.area) / 1_000_000.0,
    )


def run_local_support_future_ensembles(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    runtime: LocalSupportRuntime,
    primary: LocalSupportPrimaryPipelineResult,
    authorized_execution: AuthorizedExecution,
    *,
    detected_physical_cores: int | None,
    max_workers: int,
    reserve_physical_cores: int,
    progress: object | None = None,
) -> LocalSupportFutureOutcome:
    """Run all validation ensembles using the stage's one frozen core-count probe."""

    require_background_scoring_authorized(config, authorized_execution)
    if detected_physical_cores is not None and (
        not isinstance(detected_physical_cores, int)
        or isinstance(detected_physical_cores, bool)
        or detected_physical_cores <= 0
    ):
        raise ValueError("detected_physical_cores must be a positive integer or None")
    if primary.authorization_id != authorized_execution.authorization_id:
        raise ValueError("local future input uses another scoring authorization")
    if tuple(str(value) for value in catalog.event_id) != runtime.event_ids:
        raise ValueError("local future catalog differs from the support runtime")
    final_runtime = runtime.snapshot("final_validation")
    final_attempt = primary.etas.primary.attempt("final_validation")
    support = final_runtime.support
    if (
        not final_attempt.succeeded
        or final_attempt.fit_result is None
        or not final_attempt.fit_result.stability.stable
        or final_attempt.fit_result.best_parameters is None
        or final_attempt.parameter_snapshot_id is None
        or final_attempt.grid_gate_evidence is None
        or not final_attempt.grid_gate_evidence.passed
    ):
        reason = (
            "; ".join(final_attempt.failure_reasons)
            if final_attempt.failure_reasons
            else "final ETAS parameters are unavailable"
        )
        return LocalSupportFutureOutcome(
            status="not_run",
            authorization_id=authorized_execution.authorization_id,
            support_id=support.support_id,
            compensator_domain_id=final_runtime.compensator_domain_id,
            parameter_snapshot_id=final_attempt.parameter_snapshot_id,
            ensembles=None,
            failure_reason=reason,
        )

    parameters = final_attempt.fit_result.best_parameters
    spec = build_etas_model_spec(
        config,
        selected_mc=support.common_mc,
        aki_b_value=support.retained_selected_aki_b_value,
    )
    spec.validate_parameters(parameters)
    supported = final_runtime.supported_mask
    unsupported = np.asarray(catalog.inside_study_area & ~supported, dtype=np.bool_)
    eligible_unsupported = np.asarray(
        final_runtime.etas_primary_parent_role_mask & unsupported,
        dtype=np.bool_,
    )
    roles = build_local_support_etas_parent_roles(
        catalog,
        supported_domain_mask=supported,
        unsupported_domain_mask=unsupported,
        common_mc=support.common_mc,
        prevalidated_unsupported_parent_mask=eligible_unsupported,
    )
    issue_dates = calendar.validation.actual_issue_dates_local
    issue_days = calendar.validation.actual_issue_days
    histories = {}
    for issue_date, issue_day in zip(issue_dates, issue_days, strict=True):
        history_mask = np.asarray(
            roles.parent_mask
            & (catalog.origin_day > issue_day - spec.history_parent_cutoff_days)
            & (catalog.origin_day <= issue_day)
            & (catalog.available_day <= issue_day),
            dtype=np.bool_,
        )
        histories[issue_date] = catalog_etas_events(
            catalog,
            history_mask,
            time_origin_day=issue_day,
            inside_target_domain_mask=supported,
            inside_parent_domain_mask=roles.parent_mask,
        )
    spatial_model = primary.poisson.selected_kde_model("final_validation")
    study_area = _support_study_area(support.retained_geometry)
    ensembles = simulate_all_validation_issue_ensembles(
        parameters,
        spec,
        histories,
        spatial_model,
        study_area,
        final_runtime.grid_family,
        calendar,
        max_workers=max_workers,
        reserve_physical_cores=reserve_physical_cores,
        physical_core_probe=lambda: detected_physical_cores,
        progress=progress,  # type: ignore[arg-type]
        protocol_version="0.2.1",
    )
    return LocalSupportFutureOutcome(
        status="succeeded",
        authorization_id=authorized_execution.authorization_id,
        support_id=support.support_id,
        compensator_domain_id=final_runtime.compensator_domain_id,
        parameter_snapshot_id=final_attempt.parameter_snapshot_id,
        ensembles=ensembles,
        failure_reason=None,
    )


__all__ = [
    "LocalSupportFutureOutcome",
    "run_local_support_future_ensembles",
]
