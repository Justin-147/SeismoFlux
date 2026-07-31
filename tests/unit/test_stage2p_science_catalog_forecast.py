"""Target-free synthetic tests for the Stage 2P catalog and forecast core."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from seismoflux.stage2p.catalog import (
    CausalCatalogWindows,
    SyntheticEvent,
    select_causal_windows,
)
from seismoflux.stage2p.forecast import (
    ALARM_BUDGET_KM2,
    FROZEN_BANDWIDTH_KM,
    FROZEN_MIXTURE_WEIGHT,
    alarm_area_spread_km2,
    build_forecast_from_windows,
    build_science_forecast,
)
from seismoflux.stage2s.contracts import SpatialGrid, SpatialQuadratureFamily
from seismoflux.stage2s.spatial import build_normalized_kde

T = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
Q = T - timedelta(minutes=15)
START = Q - timedelta(days=365)


def _event(
    event_id: str,
    origin_time: datetime,
    *,
    magnitude: float = 4.5,
    x_km: float = 100.0,
    y_km: float = 100.0,
    first_seen: datetime | None = None,
) -> SyntheticEvent:
    return SyntheticEvent(
        id=event_id,
        origin_time=origin_time,
        first_seen=origin_time if first_seen is None else first_seen,
        x_km=x_km,
        y_km=y_km,
        magnitude=magnitude,
    )


def _square_grid(cell_size_km: float, *, extent_km: float) -> SpatialGrid:
    cells_per_side = int(extent_km / cell_size_km)
    rows: list[int] = []
    columns: list[int] = []
    xy: list[tuple[float, float]] = []
    areas: list[float] = []
    cell_ids: list[str] = []
    for row in range(cells_per_side):
        for column in range(cells_per_side):
            rows.append(row)
            columns.append(column)
            xy.append(
                (
                    column * cell_size_km + cell_size_km / 2.0,
                    row * cell_size_km + cell_size_km / 2.0,
                )
            )
            areas.append(cell_size_km**2)
            cell_ids.append(f"g{cell_size_km:g}-r{row:03d}-c{column:03d}")
    return SpatialGrid(
        grid_id=f"synthetic-{cell_size_km:g}-{extent_km:g}",
        cell_size_km=cell_size_km,
        cell_ids=tuple(cell_ids),
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        query_xy_km=np.asarray(xy, dtype=np.float64),
        clipped_area_km2=np.asarray(areas, dtype=np.float64),
    )


def _family(*, extent_km: float = 800.0) -> SpatialQuadratureFamily:
    return SpatialQuadratureFamily(
        grids=(
            _square_grid(50.0, extent_km=extent_km),
            _square_grid(25.0, extent_km=extent_km),
            _square_grid(12.5, extent_km=extent_km),
        )
    )


def test_causal_windows_use_exact_open_closed_boundaries_and_m4_recent_filter() -> None:
    events = (
        _event("historical-low", Q - timedelta(days=90), magnitude=3.1),
        _event("rp-left-open", Q - timedelta(days=60), magnitude=5.0),
        _event("rp-middle", Q - timedelta(days=45), magnitude=4.0),
        _event("rp-right-closed", Q - timedelta(days=30), magnitude=4.2),
        _event("r-left-open", Q - timedelta(days=30) + timedelta(microseconds=1)),
        _event("r-low", Q - timedelta(days=10), magnitude=3.9),
        _event("r-right-closed", Q, magnitude=5.1),
    )

    windows = select_causal_windows(
        reversed(events),
        issue_time=T,
        query_cutoff=Q,
        training_start=START,
    )

    assert tuple(event.id for event in windows.p0_events) == tuple(event.id for event in events)
    assert tuple(event.id for event in windows.r30_events) == (
        "r-left-open",
        "r-right-closed",
    )
    assert tuple(event.id for event in windows.rp30_events) == (
        "rp-middle",
        "rp-right-closed",
    )


@pytest.mark.parametrize(
    "event, message",
    [
        (
            _event("future-origin", Q + timedelta(microseconds=1)),
            "originating after Q",
        ),
        (
            _event(
                "future-visible",
                Q - timedelta(days=1),
                first_seen=T,
            ),
            "first seen at or after T",
        ),
    ],
)
def test_future_origin_or_visibility_is_rejected_not_silently_filtered(
    event: SyntheticEvent,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_causal_windows(
            (event,),
            issue_time=T,
            query_cutoff=Q,
            training_start=START,
        )


def test_catalog_rejects_duplicate_ids_invalid_times_and_nonfinite_values() -> None:
    duplicate = _event("same", Q - timedelta(days=1))
    with pytest.raises(ValueError, match="unique"):
        select_causal_windows(
            (duplicate, duplicate),
            issue_time=T,
            query_cutoff=Q,
            training_start=START,
        )
    with pytest.raises(ValueError, match="15 minutes"):
        select_causal_windows(
            (duplicate,),
            issue_time=T,
            query_cutoff=Q - timedelta(seconds=1),
            training_start=START,
        )
    with pytest.raises(ValueError, match="cannot precede"):
        _event(
            "impossible-visibility",
            Q - timedelta(days=1),
            first_seen=Q - timedelta(days=2),
        )
    with pytest.raises(ValueError, match="finite"):
        _event("bad-coordinate", Q - timedelta(days=1), x_km=math.nan)


def test_direct_window_construction_rechecks_p0_causality_order_and_uniqueness() -> None:
    old = _event("old", Q - timedelta(days=90))
    recent = _event("recent", Q - timedelta(days=1))

    with pytest.raises(ValueError, match="training_start through Q"):
        CausalCatalogWindows(T, Q, START, (_event("future", Q + timedelta(seconds=1)),), (), ())
    with pytest.raises(ValueError, match="first seen before T"):
        CausalCatalogWindows(
            T,
            Q,
            START,
            (_event("late", Q - timedelta(days=1), first_seen=T),),
            (),
            (),
        )
    with pytest.raises(ValueError, match="unique"):
        CausalCatalogWindows(T, Q, START, (old, old), (), ())
    with pytest.raises(ValueError, match="fixed"):
        CausalCatalogWindows(T, Q, START, (recent, old), (recent,), ())


def test_direct_window_construction_requires_exact_complete_recent_subsets() -> None:
    old = _event("old", Q - timedelta(days=90))
    preceding = _event("preceding", Q - timedelta(days=45), magnitude=4.0)
    boundary = _event("boundary", Q - timedelta(days=30), magnitude=4.0)
    recent = _event("recent", Q - timedelta(days=1), magnitude=4.0)
    low = _event("low", Q - timedelta(hours=1), magnitude=3.9)
    p0 = (old, preceding, boundary, recent, low)

    valid = CausalCatalogWindows(
        T,
        Q,
        START,
        p0,
        (recent,),
        (preceding, boundary),
    )
    assert valid.r30_events == (recent,)
    assert valid.rp30_events == (preceding, boundary)

    with pytest.raises(ValueError, match="complete ordered R30"):
        CausalCatalogWindows(T, Q, START, p0, (), (preceding, boundary))
    with pytest.raises(ValueError, match="complete ordered R30"):
        CausalCatalogWindows(T, Q, START, p0, (recent, low), (preceding, boundary))
    with pytest.raises(ValueError, match="complete ordered R30"):
        CausalCatalogWindows(T, Q, START, p0, (preceding, recent), (preceding, boundary))
    with pytest.raises(ValueError, match="complete ordered RP30"):
        CausalCatalogWindows(T, Q, START, p0, (recent,), (boundary,))
    with pytest.raises(ValueError, match="complete ordered RP30"):
        CausalCatalogWindows(T, Q, START, p0, (recent,), (boundary, preceding))


def test_fixed_forecasts_share_grid_area_rule_and_exact_half_mixtures() -> None:
    events = (
        _event("old-west", Q - timedelta(days=120), x_km=100.0, y_km=200.0),
        _event("old-east", Q - timedelta(days=100), x_km=700.0, y_km=600.0),
        _event("preceding", Q - timedelta(days=45), x_km=650.0, y_km=150.0),
        _event("recent", Q - timedelta(days=10), x_km=150.0, y_km=650.0),
    )
    family = _family()

    bundle = build_science_forecast(
        events,
        issue_time=T,
        query_cutoff=Q,
        training_start=START,
        grid_family=family,
    )

    assert tuple(model.model_id for model in bundle.models) == ("P0", "P1", "PP")
    assert all(model.spatial_density.grid_family is family for model in bundle.models)
    assert all(model.spatial_density.bandwidth_km == FROZEN_BANDWIDTH_KM for model in bundle.models)
    assert bundle.p1.component_event_ids == ("recent",)
    assert bundle.pp.component_event_ids == ("preceding",)
    recent_density = build_normalized_kde(
        np.asarray([[150.0, 650.0]], dtype=np.float64),
        family,
        model_id="R",
    )
    preceding_density = build_normalized_kde(
        np.asarray([[650.0, 150.0]], dtype=np.float64),
        family,
        model_id="RP",
    )
    assert np.allclose(
        bundle.p1.spatial_density.mass_25km,
        (1.0 - FROZEN_MIXTURE_WEIGHT) * bundle.p0.spatial_density.mass_25km
        + FROZEN_MIXTURE_WEIGHT * recent_density.mass_25km,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.allclose(
        bundle.pp.spatial_density.mass_25km,
        (1.0 - FROZEN_MIXTURE_WEIGHT) * bundle.p0.spatial_density.mass_25km
        + FROZEN_MIXTURE_WEIGHT * preceding_density.mass_25km,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert all(model.alarm.actual_area_km2 <= ALARM_BUDGET_KM2 for model in bundle.models)
    assert all(model.alarm.actual_area_km2 == ALARM_BUDGET_KM2 for model in bundle.models)
    assert all(len(model.alarm.selected_cell_ids) == 960 for model in bundle.models)
    assert alarm_area_spread_km2(bundle) <= 625.0


def test_empty_recent_windows_fall_back_to_the_exact_p0_density_and_alarm() -> None:
    windows = select_causal_windows(
        (_event("old", Q - timedelta(days=120)),),
        issue_time=T,
        query_cutoff=Q,
        training_start=START,
    )
    bundle = build_forecast_from_windows(windows, _family())

    assert bundle.p1.spatial_density is bundle.p0.spatial_density
    assert bundle.pp.spatial_density is bundle.p0.spatial_density
    assert bundle.p1.alarm == bundle.p0.alarm
    assert bundle.pp.alarm == bundle.p0.alarm
    assert bundle.p1.component_event_ids == ()
    assert bundle.pp.component_event_ids == ()


def test_p0_must_not_be_empty() -> None:
    windows = select_causal_windows(
        (),
        issue_time=T,
        query_cutoff=Q,
        training_start=START,
    )
    with pytest.raises(ValueError, match="P0 requires"):
        build_forecast_from_windows(windows, _family(extent_km=50.0))


def test_forecast_entry_revalidates_a_tampered_window_object() -> None:
    windows = select_causal_windows(
        (
            _event("old", Q - timedelta(days=90)),
            _event("recent", Q - timedelta(days=1)),
        ),
        issue_time=T,
        query_cutoff=Q,
        training_start=START,
    )
    object.__setattr__(windows, "r30_events", ())

    with pytest.raises(ValueError, match="complete ordered R30"):
        build_forecast_from_windows(windows, _family(extent_km=50.0))


def test_catalog_and_forecast_modules_have_no_real_io_or_forbidden_stage_reuse() -> None:
    root = Path(__file__).resolve().parents[2]
    source = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/seismoflux/stage2p/catalog.py",
            "src/seismoflux/stage2p/forecast.py",
        )
    )
    for forbidden in (
        "data/processed",
        "requests",
        "urllib",
        "pyarrow",
        "build_stage2s_models",
        "fit_alpha",
        "stage2s.runner",
        "stage2s.evaluation",
        "stage2s.records",
        "stage2s.seals",
        "seismoflux.anomaly_increment",
    ):
        assert forbidden not in source
