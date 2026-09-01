from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import seismoflux.multitask_s1.development_score as development_score
import seismoflux.multitask_s1.prediction_seal as prediction_seal
from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
)
from seismoflux.multitask_s1.development_predict import TIME_BANDS
from seismoflux.multitask_s1.development_score import (
    CountBand,
    CountDistribution,
    CountForecast,
    DevelopmentScoreError,
    JointModelSpec,
    LocationForecast,
    MagnitudeForecast,
    PrimaryDevelopmentTargets,
    ScoringContext,
    _assign_standalone_magnitude_targets,
    _build_primary_exposure_targets,
    _score_development,
    _score_joint,
    _score_locations,
    _score_magnitudes,
    _score_time,
    score_authorized_development_from_context,
)
from seismoflux.multitask_s1.prediction_seal import (
    DevelopmentScoringAuthorization,
    PredictionArtifactInput,
    PredictionInputIdentities,
    PredictionSealError,
    seal_fold_prediction,
    seal_four_fold_predictions,
)
from seismoflux.multitask_s1.runner_inputs import (
    CatalogEventTable,
    OuterIssueRow,
    catalog_event_table_from_frame,
)
from seismoflux.multitask_s1.time_magnitude import (
    NB2DispersionQualification,
    fit_m0_gr_global,
    fit_m3_gr_long_m5,
)
from seismoflux.stage2s.contracts import SpatialGrid

_SHANGHAI = timezone(timedelta(hours=8))
_FOLD_BOUNDS = {
    "C_DEV_2000_2004": (
        datetime(1999, 12, 31, 16, tzinfo=UTC),
        datetime(2004, 12, 31, 16, tzinfo=UTC),
    ),
    "C_DEV_2005_2009": (
        datetime(2004, 12, 31, 16, tzinfo=UTC),
        datetime(2009, 12, 31, 16, tzinfo=UTC),
    ),
    "C_DEV_2010_2014": (
        datetime(2009, 12, 31, 16, tzinfo=UTC),
        datetime(2014, 12, 31, 16, tzinfo=UTC),
    ),
    "C_DEV_2015_2019": (
        datetime(2014, 12, 31, 16, tzinfo=UTC),
        datetime(2019, 12, 31, 16, tzinfo=UTC),
    ),
}


def _identities() -> PredictionInputIdentities:
    return PredictionInputIdentities(
        run_contract_sha256="1" * 64,
        parent_contract_sha256="2" * 64,
        catalog_sha256="3" * 64,
        study_sha256="4" * 64,
        grid_sha256="5" * 64,
        issue_ledger_sha256="6" * 64,
        code_sha256="7" * 64,
        git_commit_oid="8" * 40,
    )


def _candidate_rows(axis: tuple[float, ...], best: float | None = None) -> list[dict[str, object]]:
    return [
        {
            "parameter_value": value,
            "inner_block_mean_log_density": [1.0 if best == value else 0.0] * 3,
        }
        for value in axis
    ]


def _selection_entries() -> list[dict[str, object]]:
    qualification = {
        "status": "poisson_limit",
        "reason": "sample_variance_not_greater_than_sample_mean",
        "historical_block_count": 3,
        "sample_mean_count": 1.0,
        "sample_variance_count": 1.0,
        "dispersion_k": None,
        "observed_information_k": None,
        "standard_error_k": None,
    }
    return [
        {
            "horizon_days": horizon,
            "inner_evidence": {
                "location": {
                    "latest_inner_target_end_us": 0,
                    "evaluation_start_boundary_us": 31 * 86_400_000_000,
                    "inner_block_event_counts": [4, 3, 3],
                    "selected_regional_tau_years": 10.0,
                    "selected_kde_bandwidth_km": 75.0,
                    "selected_recent_alpha": 0.25,
                    "regional_candidates": _candidate_rows((1.0, 5.0, 10.0)),
                    "kde_candidates": _candidate_rows((75.0, 100.0, 150.0, 200.0, 300.0), 75.0),
                    "recent_candidates": _candidate_rows((0.0, 0.25, 0.5, 0.75), 0.25),
                },
                "time": [{"band": band, "qualification": qualification} for band in TIME_BANDS],
            },
        }
        for horizon in HORIZONS_DAYS
    ]


def _sealed_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ScoringContext:
    identities = _identities()
    project_root = tmp_path / "synthetic_project"
    data_root = tmp_path / "synthetic_data"
    project_root.mkdir()
    data_root.mkdir()
    run_contract = project_root / "configs" / "multitask_s1_development_run.yaml"
    run_contract.parent.mkdir(parents=True)
    run_contract.write_text(
        """outputs:
  root: outputs/multitask_s1/s1c0_all_m4_screen_v1
  prediction_root: outputs/multitask_s1/s1c0_all_m4_screen_v1/prediction_phase
  score_root: outputs/multitask_s1/s1c0_all_m4_screen_v1/score_phase
  phase_roots_must_be_siblings_and_nonoverlapping: true
""",
        encoding="utf-8",
    )
    output_root = (
        project_root / "outputs" / "multitask_s1" / "s1c0_all_m4_screen_v1" / "prediction_phase"
    )
    monkeypatch.setattr(
        prediction_seal,
        "recompute_prediction_input_identities",
        lambda *, project_root, data_root: identities,
    )
    horizon_count = len(HORIZONS_DAYS)
    band_count = len(TIME_BANDS)
    schema: dict[str, object] = {
        "relative_mass": {"shape": [4], "dtype": "float64"},
        "primary_horizon_days": {"shape": [horizon_count], "dtype": "int16"},
        "location_regional_tau_years": {
            "shape": [horizon_count],
            "dtype": "float64",
        },
        "location_bandwidth_km": {
            "shape": [horizon_count, 5],
            "dtype": "float64",
        },
        "location_alpha": {"shape": [horizon_count, 5], "dtype": "float64"},
        "t1_status_code": {"shape": [horizon_count, band_count], "dtype": "int8"},
        "t1_reason_code": {"shape": [horizon_count, band_count], "dtype": "int8"},
        "t1_historical_block_count": {
            "shape": [horizon_count, band_count],
            "dtype": "int16",
        },
        "t1_sample_mean_count": {
            "shape": [horizon_count, band_count],
            "dtype": "float64",
        },
        "t1_sample_variance_count": {
            "shape": [horizon_count, band_count],
            "dtype": "float64",
        },
        "t1_sample_variance_applicable": {
            "shape": [horizon_count, band_count],
            "dtype": "int8",
        },
        "t1_dispersion_k": {
            "shape": [horizon_count, band_count],
            "dtype": "float64",
        },
        "t1_dispersion_k_applicable": {
            "shape": [horizon_count, band_count],
            "dtype": "int8",
        },
        "t1_observed_information_k": {
            "shape": [horizon_count, band_count],
            "dtype": "float64",
        },
        "t1_observed_information_k_applicable": {
            "shape": [horizon_count, band_count],
            "dtype": "int8",
        },
        "t1_standard_error_k": {
            "shape": [horizon_count, band_count],
            "dtype": "float64",
        },
        "t1_standard_error_k_applicable": {
            "shape": [horizon_count, band_count],
            "dtype": "int8",
        },
    }

    def tiny_schema(fold_id: str, *, cell_count: int) -> dict[str, object]:
        assert fold_id in DEVELOPMENT_FOLD_IDS
        assert cell_count > 0
        return schema

    def validate_tiny_arrays(fold_id: str, arrays: object, *, cell_count: int) -> None:
        assert fold_id in DEVELOPMENT_FOLD_IDS
        assert cell_count > 0
        assert isinstance(arrays, dict)

    monkeypatch.setattr(prediction_seal, "frozen_fold_prediction_npz_schema", tiny_schema)
    monkeypatch.setattr(
        prediction_seal,
        "validate_frozen_fold_prediction_npz_arrays",
        validate_tiny_arrays,
    )
    output_root.mkdir(parents=True)
    run_manifest = {
        "schema_version": 1,
        "record_type": "s1_c0_prediction_run_manifest",
        "run_contract_id": "multitask-s1-c0-all-m4-screen-v1",
        "stage": "S1-C0",
        "role": "development_prediction_only",
        "git_commit_oid": identities.git_commit_oid,
        "input_identities": identities.as_mapping(),
        "fold_ids": list(DEVELOPMENT_FOLD_IDS),
        "maximum_fold_workers": 1,
        "numerical_threads_per_worker": 1,
        "outer_targets_constructed": False,
        "model_scores_read": False,
        "locked_test_run": False,
    }
    parameter_selection = {
        "schema_version": 1,
        "record_type": "s1_c0_inner_parameter_selection",
        "run_contract_id": "multitask-s1-c0-all-m4-screen-v1",
        "role": "strictly_earlier_inner_selection_only",
        "folds": {fold_id: _selection_entries() for fold_id in DEVELOPMENT_FOLD_IDS},
    }
    (output_root / "run_manifest.json").write_bytes(
        prediction_seal.canonical_json_bytes(run_manifest)
    )
    (output_root / "parameter_selection.json").write_bytes(
        prediction_seal.canonical_json_bytes(parameter_selection)
    )
    for fold_index, fold_id in enumerate(DEVELOPMENT_FOLD_IDS):
        path = output_root / "folds" / fold_id / "predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            relative_mass=np.asarray([0.4, 0.3, 0.2, 0.1]) + fold_index,
            primary_horizon_days=np.asarray(HORIZONS_DAYS, dtype=np.int16),
            location_regional_tau_years=np.full(len(HORIZONS_DAYS), 10.0),
            location_bandwidth_km=np.column_stack(
                (
                    np.zeros(len(HORIZONS_DAYS)),
                    np.zeros(len(HORIZONS_DAYS)),
                    np.full(len(HORIZONS_DAYS), 75.0),
                    np.full(len(HORIZONS_DAYS), 75.0),
                    np.full(len(HORIZONS_DAYS), 75.0),
                )
            ),
            location_alpha=np.column_stack(
                (
                    np.zeros(len(HORIZONS_DAYS)),
                    np.zeros(len(HORIZONS_DAYS)),
                    np.zeros(len(HORIZONS_DAYS)),
                    np.zeros(len(HORIZONS_DAYS)),
                    np.full(len(HORIZONS_DAYS), 0.25),
                )
            ),
            t1_status_code=np.ones((len(HORIZONS_DAYS), len(TIME_BANDS)), dtype=np.int8),
            t1_reason_code=np.full((len(HORIZONS_DAYS), len(TIME_BANDS)), 3, dtype=np.int8),
            t1_historical_block_count=np.full(
                (len(HORIZONS_DAYS), len(TIME_BANDS)), 3, dtype=np.int16
            ),
            t1_sample_mean_count=np.ones((len(HORIZONS_DAYS), len(TIME_BANDS))),
            t1_sample_variance_count=np.ones((len(HORIZONS_DAYS), len(TIME_BANDS))),
            t1_sample_variance_applicable=np.ones(
                (len(HORIZONS_DAYS), len(TIME_BANDS)), dtype=np.int8
            ),
            t1_dispersion_k=np.zeros((len(HORIZONS_DAYS), len(TIME_BANDS))),
            t1_dispersion_k_applicable=np.zeros(
                (len(HORIZONS_DAYS), len(TIME_BANDS)), dtype=np.int8
            ),
            t1_observed_information_k=np.zeros((len(HORIZONS_DAYS), len(TIME_BANDS))),
            t1_observed_information_k_applicable=np.zeros(
                (len(HORIZONS_DAYS), len(TIME_BANDS)), dtype=np.int8
            ),
            t1_standard_error_k=np.zeros((len(HORIZONS_DAYS), len(TIME_BANDS))),
            t1_standard_error_k_applicable=np.zeros(
                (len(HORIZONS_DAYS), len(TIME_BANDS)), dtype=np.int8
            ),
        )
        seal_fold_prediction(
            output_root,
            fold_id,
            prediction_manifest=prediction_seal.frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[PredictionArtifactInput(path=path, schema=schema)],
            project_root=project_root,
            data_root=data_root,
        )
    authorization = seal_four_fold_predictions(
        output_root,
        project_root=project_root,
        data_root=data_root,
    )
    return ScoringContext(
        output_root=output_root,
        expected_seal_sha256=authorization.seal.sha256,
        project_root=project_root,
        data_root=data_root,
    )


def _weekly(fold_id: str) -> tuple[datetime, ...]:
    start, end = _FOLD_BOUNDS[fold_id]
    candidate = start.astimezone(_SHANGHAI)
    candidate += timedelta(days=(3 - candidate.weekday()) % 7)
    result: list[datetime] = []
    while candidate.astimezone(UTC) < end:
        result.append(candidate.astimezone(UTC))
        candidate += timedelta(days=7)
    return tuple(result)


def _primary_rows() -> tuple[OuterIssueRow, ...]:
    rows: list[OuterIssueRow] = []
    for fold_id in DEVELOPMENT_FOLD_IDS:
        _, end = _FOLD_BOUNDS[fold_id]
        weekly = _weekly(fold_id)
        for horizon in HORIZONS_DAYS:
            mature = tuple(issue for issue in weekly if issue + timedelta(days=horizon) <= end)
            selected: list[datetime] = []
            for issue in mature:
                if not selected or issue >= selected[-1] + timedelta(days=horizon + 30):
                    selected.append(issue)
            rows.extend(
                OuterIssueRow(
                    fold_id=fold_id,
                    issue_time_utc=issue,
                    horizon_days=horizon,
                    target_end_utc=issue + timedelta(days=horizon),
                    maturity_status="mature",
                    primary_exposure_selected=True,
                )
                for issue in selected
            )
    return tuple(rows)


def _catalog() -> CatalogEventTable:
    rows = [
        ("a", "2000-01-10T00:00:00Z", 100.0, 30.0, 5.2),
        ("m6", "2000-01-11T00:00:00Z", 103.0, 30.0, 6.2),
        # Exactly a scheduled Thursday: latest strictly earlier means the prior Thursday.
        ("m4_exact_issue", "2000-01-12T16:00:00Z", 101.0, 30.0, 4.5),
        ("b", "2000-01-30T00:00:00Z", 100.0, 30.0, 5.4),
        # Chained to b but 40 days after a: fixed-anchor semantics make a new episode.
        ("c", "2000-02-19T00:00:00Z", 100.0, 30.0, 5.6),
        ("cross_anchor", "2004-12-20T00:00:00Z", 102.0, 30.0, 5.3),
        ("cross_member", "2005-01-05T00:00:00Z", 102.0, 30.0, 5.4),
        ("fold2", "2005-01-10T00:00:00Z", 105.0, 30.0, 5.7),
        ("fold3", "2010-01-10T00:00:00Z", 106.0, 30.0, 5.8),
        ("fold4", "2015-01-10T00:00:00Z", 107.0, 30.0, 6.1),
        ("outside", "2015-01-11T00:00:00Z", 108.0, 30.0, 7.0),
    ]
    frame = pd.DataFrame(
        {
            "event_id": [row[0] for row in rows],
            "origin_time_utc": [row[1] for row in rows],
            "available_at": [row[1] for row in rows],
            "longitude": [row[2] for row in rows],
            "latitude": [row[3] for row in rows],
            "magnitude": [row[4] for row in rows],
            "inside_study_area": [True] * (len(rows) - 1) + [False],
        }
    )
    return catalog_event_table_from_frame(frame)


def _grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="synthetic-s1-score-grid",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.asarray([0, 0, 0, 0], dtype=np.int64),
        columns=np.asarray([0, 1, 2, 3], dtype=np.int64),
        query_xy_km=np.asarray(
            [[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0]],
            dtype=np.float64,
        ),
        clipped_area_km2=np.asarray([200_000.0, 250_000.0, 300_000.0, 300_000.0], dtype=np.float64),
    )


class _Locator:
    def locate_lonlat(self, longitude: float, latitude: float) -> int | None:
        del latitude
        return int(longitude) % 4


class _Predictions:
    def __init__(self, *, shift_magnitude_issue: bool = False) -> None:
        self.shift_magnitude_issue = shift_magnitude_issue
        self.m0 = fit_m0_gr_global([4.0, 4.2, 4.6, 5.0, 5.4, 5.8, 6.2, 6.8])
        self.m3 = fit_m3_gr_long_m5([5.0, 5.2, 5.4, 5.8, 6.2, 6.8])
        self.nb = NB2DispersionQualification(
            status="evaluable",
            reason="synthetic_frozen_dispersion",
            historical_block_count=6,
            sample_mean_count=1.0,
            sample_variance_count=2.0,
            dispersion_k=2.0,
            observed_information_k=3.0,
            standard_error_k=0.5,
        )

    def location_forecast(
        self, *, fold_id: str, issue_time_utc: datetime, horizon_days: int, model_id: str
    ) -> LocationForecast:
        return LocationForecast(
            fold_id,
            issue_time_utc,
            horizon_days,
            model_id,
            np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
        )

    def count_forecast(
        self,
        *,
        fold_id: str,
        issue_time_utc: datetime,
        horizon_days: int,
        model_id: str,
        magnitude_band: CountBand,
    ) -> CountForecast:
        distribution: CountDistribution = "nb2" if model_id.startswith("T1") else "poisson"
        return CountForecast(
            fold_id=fold_id,
            issue_time_utc=issue_time_utc,
            horizon_days=horizon_days,
            model_id=model_id,
            magnitude_band=magnitude_band,
            expected_count=0.5,
            distribution=distribution,
            nb2_qualification=self.nb if distribution == "nb2" else None,
        )

    def magnitude_forecast(
        self, *, fold_id: str, issue_time_utc: datetime, model_id: str
    ) -> MagnitudeForecast:
        returned_issue = (
            issue_time_utc + timedelta(days=7) if self.shift_magnitude_issue else issue_time_utc
        )
        model = self.m3 if model_id == "M3_GR_LONG_M5" else self.m0
        return MagnitudeForecast(fold_id, returned_issue, model_id, model)


def _targets(context: ScoringContext) -> PrimaryDevelopmentTargets:
    return _build_primary_exposure_targets(
        context,
        catalog=_catalog(),
        primary_issue_rows=_primary_rows(),
        grid=_grid(),
        locator=_Locator(),
    )


def test_authorization_and_primary_scope_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    catalog = _catalog()
    with pytest.raises(TypeError, match="ScoringContext"):
        _build_primary_exposure_targets(
            cast(Any, object()),
            catalog=catalog,
            primary_issue_rows=_primary_rows(),
            grid=_grid(),
            locator=_Locator(),
        )
    with pytest.raises(PredictionSealError, match="not the configured prediction_phase"):
        _build_primary_exposure_targets(
            replace(context, output_root=tmp_path / "missing_predictions"),
            catalog=catalog,
            primary_issue_rows=_primary_rows(),
            grid=_grid(),
            locator=_Locator(),
        )
    nonprimary = list(_primary_rows())
    nonprimary[0] = replace(nonprimary[0], primary_exposure_selected=False)
    with pytest.raises(DevelopmentScoreError, match="only mature primary"):
        _build_primary_exposure_targets(
            context,
            catalog=catalog,
            primary_issue_rows=nonprimary,
            grid=_grid(),
            locator=_Locator(),
        )
    forbidden = list(_primary_rows())
    forbidden[0] = replace(forbidden[0], fold_id="C_HOLDOUT_2020_2022")
    with pytest.raises(DevelopmentScoreError, match="only the four"):
        _build_primary_exposure_targets(
            context,
            catalog=catalog,
            primary_issue_rows=forbidden,
            grid=_grid(),
            locator=_Locator(),
        )


def test_missing_or_tampered_master_seal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    master_path = context.output_root / "four_fold_prediction_seal.json"
    master_path.unlink()
    with pytest.raises(PredictionSealError, match="missing"):
        score_authorized_development_from_context(context)


def test_tampered_master_seal_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    master_path = context.output_root / "four_fold_prediction_seal.json"
    master_path.write_bytes(master_path.read_bytes() + b" ")
    with pytest.raises(PredictionSealError, match="canonical"):
        score_authorized_development_from_context(context)


def test_official_entry_rejects_every_caller_supplied_scientific_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    official = cast(Any, score_authorized_development_from_context)
    injected = {
        "catalog": _catalog(),
        "grid": _grid(),
        "primary_issue_rows": _primary_rows(),
        "targets": object(),
        "predictions": _Predictions(),
    }
    for argument_name, value in injected.items():
        with pytest.raises(TypeError, match="unexpected keyword"):
            official(context, **{argument_name: value})
    assert not {
        "build_primary_exposure_targets",
        "assign_standalone_magnitude_targets",
        "score_authorized_locations",
        "score_authorized_time",
        "score_authorized_magnitudes",
        "score_authorized_joint",
        "score_authorized_development",
    }.intersection(development_score.__all__)
    assert tuple(name for name in development_score.__all__ if name.startswith("score_")) == (
        "score_authorized_development_from_context",
    )
    with pytest.raises(PredictionSealError, match="hash mismatch"):
        score_authorized_development_from_context(replace(context, expected_seal_sha256="f" * 64))


def test_full_catalog_fixed_anchor_episodes_and_nonoverlap_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets = _targets(_sealed_context(tmp_path, monkeypatch))
    first_90 = min(
        (
            item
            for item in targets.exposures
            if item.fold_id == "C_DEV_2000_2004"
            and item.horizon_days == 90
            and item.magnitude_bin == "M5_6"
        ),
        key=lambda item: item.issue_time_utc,
    )
    events = {item.event_id: item for item in first_90.events}
    assert (events["a"].global_episode_member_count, events["a"].is_episode_anchor) == (
        2,
        True,
    )
    assert (events["b"].global_episode_member_count, events["b"].is_episode_anchor) == (
        2,
        False,
    )
    assert (events["c"].global_episode_member_count, events["c"].is_episode_anchor) == (
        1,
        True,
    )
    assert all(
        event.event_id != "cross_member"
        for exposure in targets.exposures
        for event in exposure.events
    )
    for magnitude_bin in ("M5_6", "M6_plus"):
        for horizon in HORIZONS_DAYS:
            event_ids = [
                event.event_id
                for exposure in targets.exposures
                if exposure.magnitude_bin == magnitude_bin and exposure.horizon_days == horizon
                for event in exposure.events
            ]
            assert len(event_ids) == len(set(event_ids))
    assert any(not exposure.events for exposure in targets.exposures)


def test_standalone_magnitude_is_unique_strictly_earlier_and_common_m5_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    result = _assign_standalone_magnitude_targets(
        context,
        catalog=_catalog(),
        scheduled_issue_times_by_fold={
            fold_id: _weekly(fold_id) for fold_id in DEVELOPMENT_FOLD_IDS
        },
    )
    exact = next(item for item in result.m0_m4_events if item.event_id == "m4_exact_issue")
    event_time = datetime(2000, 1, 12, 16, tzinfo=UTC)
    assert exact.forecast_issue_time_utc == event_time - timedelta(days=7)
    assert exact.forecast_issue_time_utc < exact.origin_time_utc
    assert len({item.event_id for item in result.m0_m4_events}) == len(result.m0_m4_events)
    assert tuple(item for item in result.m0_m4_events if item.magnitude >= 5.0) == (
        result.common_m5_events
    )


def test_raw_scores_keep_area_horizon_zero_windows_and_main_anchor_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    targets = _targets(context)
    predictions = _Predictions()
    location = _score_locations(
        context,
        targets=targets,
        grid=_grid(),
        predictions=predictions,
        model_ids=("L0_UNIFORM",),
    )
    time = _score_time(
        context,
        targets=targets,
        grid=_grid(),
        predictions=predictions,
        model_ids=("T0_POISSON_EXPANDING",),
    )
    main = [item for item in location if item.is_main_scientific_anchor]
    assert main
    assert all(
        item.horizon_days == 30
        and item.magnitude_bin == "M5_6"
        and item.area_budget_km2 == 600_000.0
        and item.basis == "anchor"
        and item.hit_tolerance_km == 0.0
        and item.scientific_anchor_id is not None
        for item in main
    )
    assert {item.area_budget_km2 for item in location if item.metric == "strict_recall"} == {
        300_000.0,
        450_000.0,
        600_000.0,
        750_000.0,
        960_000.0,
    }
    assert {item.horizon_days for item in time} == set(HORIZONS_DAYS)
    zero_rows = [item for item in time if item.observed_count == 0]
    assert zero_rows and all(item.count_log_score is not None for item in zero_rows)
    nonempty_density = next(
        item for item in location if item.metric == "spatial_log_density" and item.event_count > 0
    )
    assert len(nonempty_density.event_ids) == nonempty_density.event_count
    assert len(nonempty_density.event_log_densities_per_km2) == nonempty_density.event_count
    assert len(nonempty_density.event_longitudes) == nonempty_density.event_count
    paired_recall = next(
        item
        for item in location
        if item.metric == "strict_recall"
        and item.fold_id == nonempty_density.fold_id
        and item.issue_time_utc == nonempty_density.issue_time_utc
        and item.horizon_days == nonempty_density.horizon_days
        and item.magnitude_bin == nonempty_density.magnitude_bin
        and item.model_id == nonempty_density.model_id
    )
    assert paired_recall.event_ids == nonempty_density.event_ids
    assert paired_recall.event_log_densities_per_km2 == (
        nonempty_density.event_log_densities_per_km2
    )
    assert paired_recall.hit_flags is not None
    empty_density = next(
        item for item in location if item.metric == "spatial_log_density" and item.event_count == 0
    )
    assert empty_density.event_ids == ()
    assert empty_density.event_longitudes == ()
    assert empty_density.event_log_densities_per_km2 == ()


def test_every_private_scoring_helper_reauthorizes_the_disk_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    real_authorize = prediction_seal.authorize_development_scoring
    calls: list[tuple[Path, str]] = []

    def audited_authorize(
        output_root: str | Path,
        *,
        expected_seal_sha256: str,
        project_root: str | Path,
        data_root: str | Path,
    ) -> DevelopmentScoringAuthorization:
        calls.append((Path(output_root), expected_seal_sha256))
        return real_authorize(
            output_root,
            expected_seal_sha256=expected_seal_sha256,
            project_root=project_root,
            data_root=data_root,
        )

    monkeypatch.setattr(
        development_score,
        "authorize_development_scoring",
        audited_authorize,
    )
    before = len(calls)
    primary_targets = _targets(context)
    assert len(calls) > before

    before = len(calls)
    magnitude_targets = _assign_standalone_magnitude_targets(
        context,
        catalog=_catalog(),
        scheduled_issue_times_by_fold={
            fold_id: _weekly(fold_id) for fold_id in DEVELOPMENT_FOLD_IDS
        },
    )
    assert len(calls) > before

    predictions = _Predictions()
    j0 = JointModelSpec("J0_U_P_GR", "T0_POISSON_EXPANDING", "L0_UNIFORM", "poisson")
    before = len(calls)
    _score_locations(
        context,
        targets=primary_targets,
        grid=_grid(),
        predictions=predictions,
        model_ids=("L0_UNIFORM",),
    )
    assert len(calls) > before

    before = len(calls)
    _score_time(
        context,
        targets=primary_targets,
        grid=_grid(),
        predictions=predictions,
        model_ids=("T0_POISSON_EXPANDING",),
    )
    assert len(calls) > before

    before = len(calls)
    _score_magnitudes(
        context,
        targets=magnitude_targets,
        predictions=predictions,
    )
    assert len(calls) > before

    before = len(calls)
    _score_joint(
        context,
        targets=primary_targets,
        grid=_grid(),
        predictions=predictions,
        joint_models=(j0,),
    )
    assert len(calls) > before

    before = len(calls)
    _score_development(
        context,
        primary_targets=primary_targets,
        magnitude_targets=magnitude_targets,
        grid=_grid(),
        predictions=predictions,
        location_model_ids=("L0_UNIFORM",),
        time_model_ids=("T0_POISSON_EXPANDING",),
        joint_models=(j0,),
    )
    assert len(calls) > before


def test_magnitude_common_support_and_joint_same_issue_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    targets = _targets(context)
    magnitude_targets = _assign_standalone_magnitude_targets(
        context,
        catalog=_catalog(),
        scheduled_issue_times_by_fold={
            fold_id: _weekly(fold_id) for fold_id in DEVELOPMENT_FOLD_IDS
        },
    )
    rows = _score_magnitudes(context, targets=magnitude_targets, predictions=_Predictions())
    assert {item.model_id for item in rows} == {"M0_GR_GLOBAL", "M3_GR_LONG_M5"}
    m5_populations = {
        item.event_ids
        for item in rows
        if item.conditional_support.startswith("M>=5 unique physical events")
    }
    assert m5_populations

    j0 = JointModelSpec("J0_U_P_GR", "T0_POISSON_EXPANDING", "L0_UNIFORM", "poisson")
    joint = _score_joint(
        context,
        targets=targets,
        grid=_grid(),
        predictions=_Predictions(),
        joint_models=(j0,),
    )
    assert joint
    assert {item.horizon_days for item in joint} == set(HORIZONS_DAYS)
    assert any(item.event_count == 0 and item.joint_log_score is not None for item in joint)
    with pytest.raises(DevelopmentScoreError, match="differs from the requested issue"):
        _score_joint(
            context,
            targets=targets,
            grid=_grid(),
            predictions=_Predictions(shift_magnitude_issue=True),
            joint_models=(j0,),
        )


def test_official_entry_rechecks_actual_prediction_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _sealed_context(tmp_path, monkeypatch)
    payload = context.output_root / "folds" / DEVELOPMENT_FOLD_IDS[0] / "predictions.npz"
    np.savez(payload, relative_mass=np.asarray([0.1, 0.2, 0.3, 0.4]))
    with pytest.raises(PredictionSealError, match="do not match the declared schema"):
        score_authorized_development_from_context(context)
