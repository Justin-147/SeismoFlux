from __future__ import annotations

import copy
import csv
import inspect
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from pyproj import Transformer
from shapely.geometry import box

from seismoflux.background.local_support import build_local_support_base_partition
from seismoflux.multitask_s1 import input_sensitivity_predict as predictor
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.local_completeness import (
    LocalCompletenessEvent,
    locate_completeness_events,
)
from seismoflux.multitask_s1.location import (
    CausalRecent30History,
    CausalSpatialHistory,
    FrozenSpatialGrid,
    LocationSurface,
    l1_regional_constant_relative_mass,
    l2_gaussian_kde_relative_mass,
    l3_b0_r30_relative_mass,
)
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_HISTORY_START_UTC,
    CatalogEventTable,
    CausalMagnitudeHistory,
    catalog_event_table_from_frame,
    causal_catalog_histories,
)


def _parameters() -> dict[str, float]:
    return {"regional_tau_years": 5.0, "kde_bandwidth_km": 75.0, "recent_alpha": 0.25}


def _grid() -> FrozenSpatialGrid:
    return FrozenSpatialGrid(
        np.asarray([10.0, 110.0, 510.0]),
        np.asarray([10.0, 10.0, 10.0]),
        np.asarray([100.0, 200.0, 100.0]),
    )


def _history(days: tuple[int, ...] = (100, 30, 20, 1)) -> CausalMagnitudeHistory:
    issue = datetime(2000, 1, 6, tzinfo=UTC)
    count = len(days)
    origin = np.asarray(
        [predictor._epoch_us(issue - timedelta(days=day)) for day in days], dtype=np.int64
    )
    return CausalMagnitudeHistory(
        magnitude_bin="m4_plus",
        issue_time_utc=issue,
        data_cutoff_utc=issue - timedelta(days=1),
        event_ids=tuple(f"event-{index}" for index in range(count)),
        origin_time_us=origin,
        available_at_us=origin.copy(),
        magnitude=np.full(count, 4.0),
        spatial=CausalSpatialHistory(
            np.arange(count, dtype=np.float64) * 10 + 10, np.full(count, 10.0)
        ),
    )


def _reference() -> NDArray[np.float64]:
    return np.asarray([[0.20, 0.40, 0.40], [0.25, 0.25, 0.50], [0.125, 0.25, 0.625]])


def test_bool_false_is_false_not_truthy_csv_string() -> None:
    assert predictor._parse_bool("False") is False
    assert predictor._parse_bool("True") is True
    for invalid in ("false", "0", "", "unknown"):
        with pytest.raises(predictor.InputSensitivityError, match="explicit"):
            predictor._parse_bool(invalid)


def test_parameters_are_fixed_and_not_reselected() -> None:
    table = {
        fold: {**_parameters(), "regional_tau_years": tau}
        for fold, tau in zip(DEVELOPMENT_FOLD_IDS, (5.0, 1.0, 1.0, 5.0), strict=True)
    }
    protocol = {"models": {"per_fold_fixed_parameters": table}}
    assert predictor._fixed_parameters(protocol) == table
    changed = copy.deepcopy(protocol)
    changed["models"]["per_fold_fixed_parameters"][DEVELOPMENT_FOLD_IDS[0]]["recent_alpha"] = 0.5
    with pytest.raises(predictor.InputSensitivityError, match="may not reselect"):
        predictor._fixed_parameters(changed)


def test_outer_mask_preserves_unknown_and_no_national_gate(tmp_path: Path) -> None:
    partition = build_local_support_base_partition(box(0, 0, 1_500_000, 500_000))
    fold = DEVELOPMENT_FOLD_IDS[0]
    records = []
    for cell, status in zip(
        partition.cells, ("supported", "indeterminate", "unsupported"), strict=True
    ):
        records.append(
            {
                "snapshot_id": f"{fold}__OUTER",
                "fold_id": fold,
                "anchor_role": "outer_fold_start",
                "anchor_utc": "1999-12-31T16:00:00Z",
                "cutoff_utc": "1999-12-30T16:00:00Z",
                "cell_id": cell.cell_id,
                "row": cell.row,
                "column": cell.column,
                "clipped_area_m2": cell.clipped_area_m2,
                "status": status,
                "main_common_mc4_training_allowed": status != "unsupported",
                "exclude_indeterminate_training_allowed": status == "supported",
                "supported_area_contributor": status == "supported",
            }
        )
    path = tmp_path / "support.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        writer.writerow({**records[0], "snapshot_id": f"{fold}__I1", "status": "unsupported"})
    statuses, diagnostic = predictor._load_mask(path, partition, fold)
    assert list(statuses.values()) == ["supported", "indeterminate", "unsupported"]
    assert diagnostic["supported_area_fraction"] == pytest.approx(1 / 3)
    assert diagnostic["indeterminate_area_fraction"] == pytest.approx(1 / 3)
    assert diagnostic["unsupported_area_fraction"] == pytest.approx(1 / 3)
    assert diagnostic["national_area_stop_gate_enabled"] is False


def test_official_partition_preserves_high_side_and_national_boundary_fallback() -> None:
    issue = datetime(1999, 1, 1, tzinfo=UTC)
    partition = build_local_support_base_partition(box(0, 0, 1_000_000, 500_000))
    events = [
        LocalCompletenessEvent(str(x), issue, issue, 4.0, x, 250_000.0)
        for x in (500_000.0, 1_000_000.0)
    ]
    located = locate_completeness_events(events, partition)
    assert [event.column for event in located] == [1, 1]
    assert len({event.cell_id for event in located}) == 1
    assert "locate_completeness_events(events, partition)" in inspect.getsource(predictor)


def test_causal_windows_keep_cutoff_but_exclude_future_availability_and_30d_lower() -> None:
    issue = datetime(2000, 1, 6, tzinfo=UTC)
    rows = []
    for identifier, day, delay in (
        ("old", 100, 0),
        ("lower", 30, 0),
        ("recent", 20, 0),
        ("cutoff", 1, 0),
        ("late_report", 20, 20),
        ("future", 0, 0),
    ):
        origin = issue - timedelta(days=day)
        rows.append(
            {
                "event_id": identifier,
                "origin_time_utc": origin,
                "available_at": origin + timedelta(days=delay),
                "longitude": 105.0,
                "latitude": 35.0,
                "magnitude": 4.0,
                "inside_study_area": True,
            }
        )
    catalog = catalog_event_table_from_frame(pd.DataFrame(rows))
    history = causal_catalog_histories(catalog, issue)["m4_plus"]
    assert history.event_ids == ("old", "lower", "recent", "cutoff")
    _, diagnostic = predictor._predict_treatment(
        history,
        np.ones(4, dtype=np.bool_),
        _grid(),
        _parameters(),
        _reference(),
        {},
    )
    assert diagnostic["training_event_count"] == 4
    assert diagnostic["recent_event_count"] == 2


def test_unchanged_event_ids_reuse_exact_C0_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("identical input must reuse C0, not recompute")

    monkeypatch.setattr(predictor, "l1_regional_constant_relative_mass", forbidden)
    monkeypatch.setattr(predictor, "l2_gaussian_kde_relative_mass", forbidden)
    mass, diagnostic = predictor._predict_treatment(
        _history(),
        np.ones(4, dtype=np.bool_),
        _grid(),
        _parameters(),
        _reference(),
        {},
    )
    assert np.array_equal(mass, _reference())
    assert diagnostic["input_identical_to_C0"]
    assert diagnostic["prediction_source"].startswith("C0_exact_same")


def test_changed_history_uses_original_formulas_and_original_1970_exposure() -> None:
    history, grid = _history(), _grid()
    keep = np.asarray([True, False, True, True])
    cache: dict[tuple[str, ...], NDArray[np.float64]] = {}
    mass, diagnostic = predictor._predict_treatment(
        history,
        keep,
        grid,
        _parameters(),
        _reference(),
        cache,
    )
    spatial = CausalSpatialHistory(history.spatial.x_km[keep], history.spatial.y_km[keep])
    recent_keep = keep & (
        history.origin_time_us
        > predictor._epoch_us(history.issue_time_utc) - 30 * predictor._DAY_US
    )
    recent = CausalRecent30History(
        history.spatial.x_km[recent_keep],
        history.spatial.y_km[recent_keep],
        history.origin_time_us[recent_keep],
        history.available_at_us[recent_keep],
        predictor._epoch_us(history.issue_time_utc),
        predictor._epoch_us(history.data_cutoff_utc),
    )
    exposure = (
        (history.data_cutoff_utc - CATALOG_HISTORY_START_UTC).total_seconds() / 86400.0 / 365.2425
    )
    expected = np.stack(
        [
            l1_regional_constant_relative_mass(
                spatial, grid, exposure_years=exposure, tau_years=5.0
            ).cell_relative_mass,
            l2_gaussian_kde_relative_mass(spatial, grid, bandwidth_km=75.0).cell_relative_mass,
            l3_b0_r30_relative_mass(
                spatial, recent, grid, bandwidth_km=75.0, alpha=0.25
            ).cell_relative_mass,
        ]
    )
    assert np.array_equal(mass, expected)
    assert diagnostic["training_event_count"] == 3
    assert diagnostic["recent_event_count"] == 2
    assert not diagnostic["input_identical_to_C0"]
    again, reused = predictor._predict_treatment(
        history,
        keep,
        grid,
        _parameters(),
        _reference(),
        cache,
    )
    assert again is mass
    assert reused["prediction_source"] == "same_issue_other_treatment_identical_event_ids"


def test_long_and_recent_KDE_are_computed_once_each(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = l2_gaussian_kde_relative_mass

    def counted(*args: Any, **kwargs: Any) -> LocationSurface:
        calls.append(kwargs.get("model_id", "L2"))
        return original(*args, **kwargs)

    monkeypatch.setattr(predictor, "l2_gaussian_kde_relative_mass", counted)
    predictor._predict_treatment(
        _history(),
        np.asarray([True, False, True, True]),
        _grid(),
        _parameters(),
        _reference(),
        {},
    )
    assert calls == ["L2", "R30_COMPONENT"]


def test_empty_recent_falls_back_to_same_long_and_empty_all_to_uniform() -> None:
    history, grid = _history(), _grid()
    mass, diagnostic = predictor._predict_treatment(
        history,
        np.asarray([True, True, False, False]),
        grid,
        _parameters(),
        _reference(),
        {},
    )
    assert np.array_equal(mass[1], mass[2])
    assert diagnostic["recent_fallback"]
    empty, diagnostic = predictor._predict_treatment(
        history,
        np.zeros(4, dtype=np.bool_),
        grid,
        _parameters(),
        _reference(),
        {},
    )
    assert np.array_equal(
        empty, np.repeat((grid.area_km2 / grid.total_area_km2)[None, :], 3, axis=0)
    )
    assert diagnostic["empty_all_history_fallback"]
    assert diagnostic["training_event_count"] == 0


def test_late_training_availability_is_rejected_even_for_reuse() -> None:
    history = _history()
    history.available_at_us[-1] = predictor._epoch_us(history.issue_time_utc)
    with pytest.raises(predictor.InputSensitivityError, match="T-minus-24-hour"):
        predictor._predict_treatment(
            history,
            np.ones(4, dtype=np.bool_),
            _grid(),
            _parameters(),
            _reference(),
            {},
        )


def test_locating_only_uses_visible_training_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    partition = build_local_support_base_partition(box(0, 0, 500_000, 500_000))
    times = np.asarray([0, 1, 2], dtype=np.int64)
    catalog = CatalogEventTable(
        ("past", "not_yet_available", "future"),
        times,
        np.asarray([0, 10, 2], dtype=np.int64),
        np.full(3, 105.0),
        np.full(3, 35.0),
        np.full(3, 4.0),
        np.ones(3, dtype=np.bool_),
    )

    class FakeTransformer:
        def transform(
            self, longitude: Any, latitude: Any
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return np.full(len(longitude), 500_000.0), np.full(len(latitude), 250_000.0)

    monkeypatch.setattr(Transformer, "from_crs", lambda *args, **kwargs: FakeTransformer())
    result = predictor._locate_training_events(
        cast(Any, SimpleNamespace(catalog=catalog)), partition, 1
    )
    assert result == {"past": partition.cells[0].cell_id}


def test_completed_fold_verification_detects_mutation_and_mismatched_identity(
    tmp_path: Path,
) -> None:
    fold = DEVELOPMENT_FOLD_IDS[0]
    npz = tmp_path / "predictions.npz"
    diag = tmp_path / "diagnostic.json"
    identity = {"protocol_id": "synthetic"}
    issues = np.arange(29, dtype=np.int64)
    with npz.open("xb") as stream:
        np.savez(
            stream,
            issue_time_us=issues,
            location_relative_mass=np.full((29, 6, 2), 0.5),
            training_event_count=np.ones((29, 2), dtype=np.int32),
            recent_event_count=np.zeros((29, 2), dtype=np.int32),
            recent_fallback=np.ones((29, 2), dtype=np.uint8),
        )
    predictor._write_json(
        diag,
        {
            "identity": identity,
            "fold_id": fold,
            "issues": [{"issue_time_us": int(issue)} for issue in issues],
        },
    )
    record = {
        "fold_id": fold,
        "issue_count": 29,
        "npz_path": npz.name,
        "npz_sha256": predictor._sha(npz),
        "diagnostics_path": diag.name,
        "diagnostics_sha256": predictor._sha(diag),
    }
    predictor._verify_fold(tmp_path, record, identity, 2)
    with pytest.raises(predictor.InputSensitivityError, match="different frozen run"):
        predictor._verify_fold(tmp_path, record, {"protocol_id": "changed"}, 2)
    with pytest.raises(FileExistsError):
        predictor._write_json(diag, {"overwriting": True})
    with npz.open("ab") as stream:
        stream.write(b"mutated")
    with pytest.raises(predictor.InputSensitivityError, match="SHA-256 mismatch"):
        predictor._verify_fold(tmp_path, record, identity, 2)


def test_prediction_module_never_imports_score_or_opens_target_tables() -> None:
    source = inspect.getsource(predictor)
    assert "import development_score" not in source
    assert "read_parquet" not in source
    assert "select_fold_horizon_parameters" not in source
    assert "C_HOLDOUT" not in source


def test_fold_completion_resumes_without_recomputation_and_keeps_partial_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold = DEVELOPMENT_FOLD_IDS[0]
    reference = _reference()
    history = _history()
    issues = [history.issue_time_utc + timedelta(days=63 * index) for index in range(29)]
    outer = tuple(
        SimpleNamespace(
            fold_id=fold, issue_time_utc=issue, horizon_days=30, primary_exposure_selected=True
        )
        for issue in issues
    )
    inputs = cast(
        Any,
        SimpleNamespace(
            project_root=tmp_path, outer_issues=outer, location_grid=_grid(), catalog=None
        ),
    )
    table = {
        name: {**_parameters(), "regional_tau_years": tau}
        for name, tau in zip(DEVELOPMENT_FOLD_IDS, (5.0, 1.0, 1.0, 5.0), strict=True)
    }
    protocol = {
        "models": {"per_fold_fixed_parameters": table},
        "parent_artifacts": {"C1_support_cells": {"path": "not-opened"}},
    }
    monkeypatch.setattr(
        predictor, "_checked", lambda root, record, key="path": tmp_path / record[key]
    )
    monkeypatch.setattr(predictor, "_load_mask", lambda *args: ({"cell": "supported"}, {}))
    monkeypatch.setattr(
        predictor,
        "_load_c0_fold",
        lambda *args: (
            {
                "mass": np.repeat(reference[None], 29, axis=0),
                "source_count": np.full((29, 3), 4, dtype=np.int32),
                "recent_fallback": np.zeros(29, dtype=np.uint8),
            },
            {"path": "synthetic-C0"},
        ),
    )

    def causal(catalog: Any, issue: datetime) -> dict[str, CausalMagnitudeHistory]:
        delta_us = predictor._epoch_us(issue) - predictor._epoch_us(history.issue_time_utc)
        return {
            "m4_plus": replace(
                history,
                issue_time_utc=issue,
                data_cutoff_utc=issue - timedelta(days=1),
                origin_time_us=history.origin_time_us + delta_us,
                available_at_us=history.available_at_us + delta_us,
            )
        }

    monkeypatch.setattr(predictor, "causal_catalog_histories", causal)
    directory = tmp_path / "folds" / fold
    directory.mkdir(parents=True)
    partial = directory / "predictions_previous_incomplete.npz"
    partial.write_bytes(b"incomplete attempt must survive")
    identity = {"protocol_id": "synthetic-checkpoint"}
    partition = build_local_support_base_partition(box(0, 0, 500_000, 500_000))
    event_cells = dict.fromkeys(history.event_ids, "cell")
    record = predictor._run_fold(inputs, protocol, identity, tmp_path, partition, event_cells, fold)
    assert partial.read_bytes() == b"incomplete attempt must survive"
    with np.load(tmp_path / record["npz_path"], allow_pickle=False) as archive:
        assert archive["location_relative_mass"].shape == (29, 6, 3)
        assert np.array_equal(archive["location_relative_mass"][0, :3], reference)
        assert np.array_equal(archive["location_relative_mass"][0, 3:], reference)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("complete fold must not recompute")

    monkeypatch.setattr(predictor, "causal_catalog_histories", forbidden)
    assert (
        predictor._run_fold(inputs, protocol, identity, tmp_path, partition, event_cells, fold)
        == record
    )


def test_worker_limit_rejects_oversubscription_before_file_access(tmp_path: Path) -> None:
    for workers in (0, 4, True):
        with pytest.raises(predictor.InputSensitivityError, match="one to three"):
            predictor.run_prediction_phase(
                project_root=tmp_path, data_root=tmp_path, workers=workers
            )
