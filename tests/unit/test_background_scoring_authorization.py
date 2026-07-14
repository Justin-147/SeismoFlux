from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from seismoflux.background.config import load_background_protocol
from seismoflux.background.deliverables import (
    build_background_deliverables,
    publish_background_deliverables,
)
from seismoflux.background.horizons import run_issue_horizon_backtests
from seismoflux.background.pipeline import run_background_pipeline
from seismoflux.background.pipeline_etas import run_etas_pipeline
from seismoflux.background.pipeline_poisson import run_poisson_kde_pipeline
from seismoflux.background.publication import (
    build_background_registry,
    publish_backtest_bundle,
    publish_experiment_bundle,
    publish_model_bundle,
    publish_registry_and_report,
    publish_registry_and_report_sealed,
)
from seismoflux.background.runner import run_background_stage2
from seismoflux.background.scoring_authorization import (
    BackgroundScoringNotAuthorizedError,
    require_background_scoring_authorized,
)

LOCAL_SUPPORT_CONFIG = Path("configs/background_local_support.yaml")
LOCAL_SUPPORT_PROJECT_CONFIG = Path("configs/base_local_support.yaml")
ORIGINAL_CONFIG = Path("configs/background.yaml")


def test_original_v020_scoring_remains_authorized() -> None:
    require_background_scoring_authorized(load_background_protocol(ORIGINAL_CONFIG))


def test_score_free_v021_is_rejected_by_every_public_scoring_pipeline() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)

    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_background_pipeline(
            config,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            production_fixture_path=Path("unused.json"),
        )
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_poisson_kde_pipeline(config, unavailable, unavailable, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_etas_pipeline(config, unavailable, unavailable, unavailable, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_issue_horizon_backtests(
            config,
            unavailable,
            unavailable,
            unavailable,
            selected_mc=4.0,
            uniform_model=unavailable,
            spatial_model=unavailable,
            etas_parameters=unavailable,
            etas_spec=unavailable,
            etas_parameter_snapshot_id="0" * 64,
            etas_model_variant_id="unused",
        )


def test_production_runner_rejects_v021_before_sealing_or_loading_rows() -> None:
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_background_stage2(LOCAL_SUPPORT_PROJECT_CONFIG)


def test_score_free_v021_is_rejected_before_score_bearing_deliverable_inputs() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)
    expected = pytest.raises(
        BackgroundScoringNotAuthorizedError,
        match="score-free stage-2R-0",
    )

    with expected:
        build_background_deliverables(config, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publish_background_deliverables(
            Path("unused"),
            config,
            unavailable,
            unavailable,
        )


@pytest.mark.parametrize(
    "publisher",
    [publish_model_bundle, publish_backtest_bundle, publish_experiment_bundle],
)
def test_score_free_v021_is_rejected_before_score_bearing_bundle_publication(
    publisher: Any,
) -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)

    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publisher(
            Path("unused"),
            config,
            unavailable,
            unavailable,
            uv_lock_sha256="unused",
        )


def test_score_free_v021_is_rejected_before_registry_inputs_or_filesystem_access() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)

    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        build_background_registry(
            config,
            unavailable,
            unavailable,
            unavailable,
            scientific_summary=unavailable,
            g1=unavailable,
            selection=unavailable,
            stage3_allowed=False,
            uv_lock_sha256="unused",
        )
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publish_registry_and_report(Path("unused"), config, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publish_registry_and_report_sealed(
            Path("unused"),
            config,
            unavailable,
            unavailable,
        )
