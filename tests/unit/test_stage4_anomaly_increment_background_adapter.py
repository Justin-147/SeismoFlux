from __future__ import annotations

import builtins
import copy
from datetime import UTC, datetime
from typing import NoReturn

import numpy as np
import pytest
from shapely.geometry import box

from seismoflux.anomaly_increment.background_adapter import (
    FROZEN_COMPENSATOR_DOMAIN_ID,
    FROZEN_STUDY_AREA_SHA256,
    BackgroundDomainBinding,
    rebuild_stage4_background,
    resolve_frozen_background_snapshot,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4GridFamily,
    Stage4IntegrationGrid,
    build_stage4_grid_family,
)
from seismoflux.anomaly_increment.targets import Stage4TargetCatalog


def _protocol() -> dict[str, object]:
    return {
        "inputs": {
            "study_area": {
                "sha256": FROZEN_STUDY_AREA_SHA256,
            }
        },
        "background": {
            "background_variant_id": "spatial_poisson/gaussian_kde_bw75km",
            "family": "spatial_poisson",
            "bandwidth_km": 75.0,
            "model_reselection_forbidden": True,
            "development": {
                "snapshot_id": "fold_4",
                "parameter_snapshot_id": (
                    "83a0c60d4b62ba6a6e849ac2d5f430001d054b7aec3af40f76193180a18bf4c5"
                ),
                "fit_end_utc": "2019-12-31T16:00:00Z",
                "support_id": "local-support-788851371baf0e3b",
                "compensator_domain_id": FROZEN_COMPENSATOR_DOMAIN_ID,
                "common_mc": 4.0,
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
                "common_mc": 4.0,
                "supported_area_fraction": 1.0,
            },
        },
    }


@pytest.fixture(scope="module")
def grid_family() -> Stage4GridFamily:
    return build_stage4_grid_family(box(104.95, 29.95, 105.25, 30.25))


def _catalog(
    grid_family: Stage4GridFamily,
    *,
    future_xy_offset_m: float = 0.0,
) -> Stage4TargetCatalog:
    rows = (
        ("pre-anchor", datetime(1969, 12, 31, tzinfo=UTC), 4.5, True),
        ("at-anchor", datetime(1970, 1, 1, tzinfo=UTC), 4.0, True),
        ("below-mc", datetime(2011, 1, 1, tzinfo=UTC), 3.999, True),
        ("known-2018", datetime(2018, 1, 1, tzinfo=UTC), 4.2, True),
        ("late-report", datetime(2018, 6, 1, tzinfo=UTC), 4.3, True),
        ("outside", datetime(2018, 7, 1, tzinfo=UTC), 4.4, False),
        ("at-dev-cutoff", datetime(2019, 12, 31, 16, tzinfo=UTC), 4.0, True),
        ("after-dev", datetime(2020, 1, 2, tzinfo=UTC), 4.5, True),
        ("after-validation", datetime(2024, 1, 1, tzinfo=UTC), 6.0, True),
    )
    available = tuple(
        datetime(2020, 1, 1, tzinfo=UTC) if event_id == "late-report" else origin
        for event_id, origin, _, _ in rows
    )
    centers = grid_family.primary_25km.query_xy_m
    xy = np.asarray([centers[index % len(centers)] for index in range(len(rows))], dtype=np.float64)
    xy[-1] += future_xy_offset_m
    return Stage4TargetCatalog(
        event_id=np.asarray([item[0] for item in rows], dtype=np.str_),
        origin_time_utc=tuple(item[1] for item in rows),
        available_at_utc=available,
        longitude=np.full(len(rows), 105.0, dtype=np.float64),
        latitude=np.full(len(rows), 30.0, dtype=np.float64),
        x_m=xy[:, 0],
        y_m=xy[:, 1],
        magnitude=np.asarray([item[2] for item in rows], dtype=np.float64),
        inside_study_area=np.asarray([item[3] for item in rows], dtype=np.bool_),
        source_content_sha256="a" * 64,
        source_schema_sha256="b" * 64,
    )


def _domain(grid_family: Stage4GridFamily) -> BackgroundDomainBinding:
    return BackgroundDomainBinding.from_verified_grid_family(
        grid_family,
        study_area_sha256=FROZEN_STUDY_AREA_SHA256,
        compensator_domain_id=FROZEN_COMPENSATOR_DOMAIN_ID,
    )


def test_snapshot_resolution_locks_cutoff_mc_bandwidth_and_support_identity() -> None:
    protocol = _protocol()
    for evaluation_id in (
        "development-fold-1",
        "development-fold-2",
        "development-fold-3",
    ):
        snapshot = resolve_frozen_background_snapshot(protocol, evaluation_id=evaluation_id)
        assert snapshot.snapshot_id == "fold_4"
        assert snapshot.fit_end_utc == datetime(2019, 12, 31, 16, tzinfo=UTC)
        assert snapshot.common_mc == 4.0
        assert snapshot.bandwidth_km == 75.0
        assert snapshot.support_id == "local-support-788851371baf0e3b"
        assert snapshot.compensator_domain_id == FROZEN_COMPENSATOR_DOMAIN_ID
        assert snapshot.study_area_sha256 == FROZEN_STUDY_AREA_SHA256

    validation = resolve_frozen_background_snapshot(protocol, evaluation_id="formal-validation")
    assert validation.snapshot_id == "final_validation"
    assert validation.fit_end_utc == datetime(2023, 6, 30, 16, tzinfo=UTC)
    assert validation.support_id == "local-support-f6816ab6c6581306"


def test_synthetic_rebuild_is_normalized_causal_and_uses_exact_mc(
    grid_family: Stage4GridFamily,
) -> None:
    result = rebuild_stage4_background(
        _catalog(grid_family),
        grid_family,
        protocol=_protocol(),
        evaluation_id="development-fold-1",
        domain=_domain(grid_family),
    )
    assert result.training_event_ids == ("at-anchor", "known-2018", "at-dev-cutoff")
    assert result.causal_audit.training_event_count == 3
    assert result.causal_audit.excluded_below_mc_count == 1
    assert result.causal_audit.excluded_after_origin_cutoff_count == 2
    assert result.causal_audit.excluded_after_availability_cutoff_count == 3
    assert result.causal_audit.post_cutoff_training_event_count == 0
    assert result.causal_audit.latest_training_origin_utc == result.snapshot.fit_end_utc
    assert result.causal_audit.latest_training_available_at_utc <= result.snapshot.fit_end_utc
    assert result.model.bandwidth_km == 75.0
    assert result.rate_per_day == pytest.approx(3 / result.training_duration_days)
    assert result.grid(12.5).spatial_mass_sum == pytest.approx(1.0, abs=1e-12)
    assert np.sum(result.grid(12.5).expected_cell_count_per_day) == pytest.approx(
        result.rate_per_day,
        abs=1e-15,
    )
    assert tuple(item.grid_id for item in result.grids) == _domain(grid_family).grid_ids
    assert all(not item.spatial_cell_mass.flags.writeable for item in result.grids)


def test_validation_snapshot_includes_only_events_known_by_its_later_cutoff(
    grid_family: Stage4GridFamily,
) -> None:
    result = rebuild_stage4_background(
        _catalog(grid_family),
        grid_family,
        protocol=_protocol(),
        evaluation_id="formal-validation",
        domain=_domain(grid_family),
    )
    assert result.training_event_ids == (
        "at-anchor",
        "known-2018",
        "late-report",
        "at-dev-cutoff",
        "after-dev",
    )
    assert "after-validation" not in result.training_event_ids
    assert result.causal_audit.post_cutoff_training_event_count == 0


def test_future_values_cannot_change_development_background_and_rebuild_is_deterministic(
    grid_family: Stage4GridFamily,
) -> None:
    protocol = _protocol()
    domain = _domain(grid_family)
    first = rebuild_stage4_background(
        _catalog(grid_family),
        grid_family,
        protocol=protocol,
        evaluation_id="development-fold-2",
        domain=domain,
    )
    same = rebuild_stage4_background(
        _catalog(grid_family),
        grid_family,
        protocol=protocol,
        evaluation_id="development-fold-2",
        domain=domain,
    )
    future_changed = rebuild_stage4_background(
        _catalog(grid_family, future_xy_offset_m=9_000_000.0),
        grid_family,
        protocol=protocol,
        evaluation_id="development-fold-2",
        domain=domain,
    )
    assert first.scientific_identity_sha256 == same.scientific_identity_sha256
    assert first.scientific_identity_sha256 == future_changed.scientific_identity_sha256
    assert first.training_event_ids == future_changed.training_event_ids
    for left, right in zip(first.grids, future_changed.grids, strict=True):
        np.testing.assert_array_equal(left.spatial_density_per_km2, right.spatial_density_per_km2)
        np.testing.assert_array_equal(left.spatial_cell_mass, right.spatial_cell_mass)

    fold_three = rebuild_stage4_background(
        _catalog(grid_family),
        grid_family,
        protocol=_protocol(),
        evaluation_id="development-fold-3",
        domain=_domain(grid_family),
    )
    assert fold_three.scientific_identity_sha256 == first.scientific_identity_sha256


def test_protocol_mutation_and_scope_expansion_fail_closed(
    grid_family: Stage4GridFamily,
) -> None:
    protocol = _protocol()
    changed_mc = copy.deepcopy(protocol)
    changed_mc["background"]["development"]["common_mc"] = 3.5  # type: ignore[index]
    with pytest.raises(ValueError, match="common_mc"):
        rebuild_stage4_background(
            _catalog(grid_family),
            grid_family,
            protocol=changed_mc,
            evaluation_id="development-fold-1",
            domain=_domain(grid_family),
        )

    changed_cutoff = copy.deepcopy(protocol)
    changed_cutoff["background"]["validation"]["fit_end_utc"] = (  # type: ignore[index]
        "2024-06-30T16:00:00Z"
    )
    with pytest.raises(ValueError, match="fit_end_utc"):
        rebuild_stage4_background(
            _catalog(grid_family),
            grid_family,
            protocol=changed_cutoff,
            evaluation_id="formal-validation",
            domain=_domain(grid_family),
        )

    changed_support = copy.deepcopy(protocol)
    changed_support["background"]["development"]["support_id"] = (  # type: ignore[index]
        "local-support-0000000000000000"
    )
    with pytest.raises(ValueError, match="support_id"):
        rebuild_stage4_background(
            _catalog(grid_family),
            grid_family,
            protocol=changed_support,
            evaluation_id="development-fold-1",
            domain=_domain(grid_family),
        )

    with pytest.raises(ValueError, match="four formal"):
        resolve_frozen_background_snapshot(
            protocol,
            evaluation_id="locked-test",  # type: ignore[arg-type]
        )


def test_study_support_and_grid_identities_are_verified_before_fit(
    grid_family: Stage4GridFamily,
) -> None:
    wrong_study = BackgroundDomainBinding(
        study_area_sha256="f" * 64,
        compensator_domain_id=FROZEN_COMPENSATOR_DOMAIN_ID,
        supported_area_km2=grid_family.reference_12_5km.total_area_km2,
        grid_ids=(
            grid_family.coarse_50km.grid_id,
            grid_family.primary_25km.grid_id,
            grid_family.reference_12_5km.grid_id,
        ),
    )
    with pytest.raises(ValueError, match="study-area"):
        rebuild_stage4_background(
            _catalog(grid_family),
            grid_family,
            protocol=_protocol(),
            evaluation_id="development-fold-1",
            domain=wrong_study,
        )

    original = grid_family.primary_25km
    tampered_primary = Stage4IntegrationGrid(
        grid_id=original.grid_id,
        equal_area_crs=original.equal_area_crs,
        cell_size_km=original.cell_size_km,
        cell_ids=original.cell_ids,
        rows=original.rows,
        columns=original.columns,
        query_xy_m=original.query_xy_m + np.asarray([1.0, 0.0]),
        clipped_area_km2=original.clipped_area_km2,
    )
    tampered_family = Stage4GridFamily(
        grid_family.coarse_50km,
        tampered_primary,
        grid_family.reference_12_5km,
    )
    with pytest.raises(ValueError, match="does not match"):
        rebuild_stage4_background(
            _catalog(grid_family),
            tampered_family,
            protocol=_protocol(),
            evaluation_id="development-fold-1",
            domain=_domain(grid_family),
        )


def test_adapter_has_no_file_read_path(
    grid_family: Stage4GridFamily,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_file_read(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"unexpected file access: {args!r} {kwargs!r}")

    monkeypatch.setattr(builtins, "open", forbidden_file_read)
    result = rebuild_stage4_background(
        _catalog(grid_family),
        grid_family,
        protocol=_protocol(),
        evaluation_id="development-fold-1",
        domain=_domain(grid_family),
    )
    assert result.causal_audit.post_cutoff_training_event_count == 0
