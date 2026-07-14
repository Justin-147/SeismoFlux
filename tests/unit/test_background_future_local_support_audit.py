from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from seismoflux.background import future_local_support
from seismoflux.background.config import BackgroundConfig, load_background_protocol
from seismoflux.background.future import FutureWorkerResources
from seismoflux.background.local_support_deliverables import _future_summary

CONFIG = Path("configs/background_local_support.yaml")


@pytest.fixture(scope="module")
def config() -> BackgroundConfig:
    return load_background_protocol(CONFIG)


def _result(*, replicate_count: int = 128) -> Any:
    issues = tuple(
        SimpleNamespace(
            issue_date_local=issue_date,
            issue_id=f"validation/{issue_date}",
            replicate_count=replicate_count,
            horizons=(
                SimpleNamespace(
                    horizon_days=7,
                    mean_count=1.25,
                    quantiles=(),
                ),
            ),
        )
        for issue_date in ("2024-07-25", "2024-08-22")
    )
    future = SimpleNamespace(
        status="succeeded",
        ensembles=SimpleNamespace(
            issues=issues,
            resources=FutureWorkerResources(
                detected_physical_cores=8,
                reserve_physical_cores=2,
                configured_max_workers=12,
                effective_workers=6,
            ),
        ),
        failure_reason=None,
        support_id="local-support-1111111111111111",
        compensator_domain_id="a" * 64,
        parameter_snapshot_id="b" * 64,
    )
    return cast(Any, SimpleNamespace(future=future))


def test_completed_future_summary_exposes_frozen_seed_and_event_cap_audit(
    config: BackgroundConfig,
) -> None:
    summary = _future_summary(config, _result())
    seed = cast(dict[str, object], summary["seed_derivation"])
    context = config.randomness.seed_derivation.namespace_contexts.future_simulation

    assert summary["status"] == "completed"
    assert summary["root_seed"] == config.randomness.root_seed == 147
    assert summary["replicate_count"] == 128
    assert summary["maximum_events_per_replicate"] == (
        config.randomness.future_simulation.maximum_events_per_replicate
    )
    assert summary["event_cap_hit_is_failure"] is True
    assert summary["event_cap_exceeded"] is False
    assert seed["root_seed"] == config.randomness.root_seed
    assert seed["protocol_version"] == config.protocol_version
    assert seed["namespace"] == "future_simulation"
    assert seed["model_id"] == context.model_id
    assert seed["issue_id_rule"] == context.issue_id_rule
    assert seed["replicate_index_role"] == context.replicate_index_role
    assert seed["replicate_index_first"] == context.replicate_index_first
    assert seed["replicate_index_last_inclusive"] == context.replicate_index_last_inclusive
    assert seed["ordered_fields"] == config.randomness.seed_derivation.ordered_fields
    assert len(cast(str, seed["seed_context_identity"])) == 64
    issues = cast(list[dict[str, object]], summary["issues"])
    assert [item["issue_id"] for item in issues] == [
        "validation/2024-07-25",
        "validation/2024-08-22",
    ]


def test_completed_future_summary_rejects_replicate_count_drift(
    config: BackgroundConfig,
) -> None:
    with pytest.raises(ValueError, match="replicate count"):
        _future_summary(config, _result(replicate_count=127))


def test_local_future_reuses_explicit_stage_core_count_without_hardware_reprobe(
    monkeypatch: pytest.MonkeyPatch,
    config: BackgroundConfig,
) -> None:
    authorized = cast(Any, SimpleNamespace(authorization_id="authorization"))
    catalog = cast(
        Any,
        SimpleNamespace(
            event_id=np.asarray(["event-1"], dtype=np.str_),
            origin_day=np.asarray([1.0]),
            available_day=np.asarray([1.0]),
            inside_study_area=np.asarray([True], dtype=np.bool_),
        ),
    )
    support = SimpleNamespace(
        support_id="local-support-1111111111111111",
        common_mc=4.0,
        retained_selected_aki_b_value=1.0,
        retained_geometry=object(),
    )
    final_runtime = SimpleNamespace(
        support=support,
        compensator_domain_id="a" * 64,
        supported_mask=np.asarray([True], dtype=np.bool_),
        etas_primary_parent_role_mask=np.asarray([True], dtype=np.bool_),
        grid_family=object(),
    )
    runtime = cast(
        Any,
        SimpleNamespace(
            event_ids=("event-1",),
            snapshot=lambda _: final_runtime,
        ),
    )
    parameters = object()
    fit_result = SimpleNamespace(
        stability=SimpleNamespace(stable=True),
        best_parameters=parameters,
    )
    attempt = SimpleNamespace(
        succeeded=True,
        fit_result=fit_result,
        parameter_snapshot_id="b" * 64,
        grid_gate_evidence=SimpleNamespace(passed=True),
        failure_reasons=(),
    )
    primary = cast(
        Any,
        SimpleNamespace(
            authorization_id="authorization",
            etas=SimpleNamespace(primary=SimpleNamespace(attempt=lambda _: attempt)),
            poisson=SimpleNamespace(selected_kde_model=lambda _: object()),
        ),
    )
    calendar = cast(
        Any,
        SimpleNamespace(
            validation=SimpleNamespace(
                actual_issue_dates_local=("2024-07-25",),
                actual_issue_days=(2.0,),
            )
        ),
    )
    spec = SimpleNamespace(
        history_parent_cutoff_days=365.0,
        validate_parameters=lambda _: None,
    )
    captured: dict[str, object] = {}

    def simulate(*_: object, **kwargs: object) -> Any:
        probe = cast(Any, kwargs["physical_core_probe"])
        captured["detected_physical_cores"] = probe()
        return SimpleNamespace(
            resources=FutureWorkerResources(
                detected_physical_cores=10,
                reserve_physical_cores=2,
                configured_max_workers=12,
                effective_workers=8,
            )
        )

    monkeypatch.setattr(
        future_local_support,
        "require_background_scoring_authorized",
        lambda *_: None,
    )
    monkeypatch.setattr(
        future_local_support,
        "build_etas_model_spec",
        lambda *_args, **_kwargs: spec,
    )
    monkeypatch.setattr(
        future_local_support,
        "build_local_support_etas_parent_roles",
        lambda *_args, **_kwargs: SimpleNamespace(parent_mask=np.asarray([True], dtype=np.bool_)),
    )
    monkeypatch.setattr(future_local_support, "catalog_etas_events", lambda *_args, **_: ())
    monkeypatch.setattr(future_local_support, "_support_study_area", lambda _: object())
    monkeypatch.setattr(
        future_local_support,
        "simulate_all_validation_issue_ensembles",
        simulate,
    )

    outcome = future_local_support.run_local_support_future_ensembles(
        config,
        catalog,
        calendar,
        runtime,
        primary,
        authorized,
        detected_physical_cores=10,
        max_workers=12,
        reserve_physical_cores=2,
    )

    assert outcome.status == "succeeded"
    assert captured == {"detected_physical_cores": 10}
