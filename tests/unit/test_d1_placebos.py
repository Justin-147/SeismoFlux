from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import seismoflux.d1_replay.placebos as placebos
from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.features import D1_SOURCE_COLUMNS, D1IssueFeatures, D1StaticGrid
from seismoflux.d1_replay.placebos import (
    D1_PLACEBO_CONTRASTS,
    D1CoordinateEntity,
    D1FoldPlaceboPlan,
    D1FoldPlaceboScore,
    D1ObservedPlaceboBaseline,
    D1PlaceboFoldReplication,
    D1PlaceboInfrastructureInterruption,
    D1PlaceboIssueSource,
    D1PlaceboPreparedReplay,
    D1Stage3SnapshotHistory,
    D1VerifiedSpatialStrata,
    build_d1_space_pseudo_history,
    build_d1_time_pseudo_history,
    d1_placebo_rng,
    d1_placebo_schedule,
    load_d1_placebo_checkpoint,
    permute_d1_coordinates_within_zones,
    prepare_d1_placebo_replay,
    reduce_d1_placebo_contrast,
    run_d1_placebos,
    write_d1_placebo_checkpoint,
)
from seismoflux.d1_replay.runner import D1PreparedReplay
from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.state import AnomalyState
from seismoflux.features.anomaly.trajectory import compute_latest_trajectory_features


def _grid() -> D1StaticGrid:
    return D1StaticGrid(
        grid_id="synthetic-d1-grid",
        cell_ids=("cell-0", "cell-1"),
        rows=np.asarray([0, 0], dtype=np.int64),
        columns=np.asarray([0, 1], dtype=np.int64),
        query_x_m=np.asarray([0.0, 25_000.0], dtype=np.float64),
        query_y_m=np.asarray([0.0, 0.0], dtype=np.float64),
        clipped_area_km2=np.asarray([625.0, 625.0], dtype=np.float64),
    )


def _source(index: int, grid: D1StaticGrid) -> D1PlaceboIssueSource:
    issue_time = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(weeks=index)
    values = np.zeros((grid.cell_count, len(D1_SOURCE_COLUMNS)), dtype=np.float64)
    nulls = np.zeros(values.shape, dtype=np.bool_)
    values[:, :4] = np.asarray([[100.0 + index, 200.0 + index, 300.0 + index, 400.0 + index]] * 2)
    values[:, 4:9] = np.asarray(
        [
            [10.0 + index, 20.0 + index, 30.0 + index, 0.2 + index, 0.3 + index],
            [11.0 + index, 21.0 + index, 31.0 + index, 0.4 + index, 0.5 + index],
        ]
    )
    values[:, 9:15] = 999.0
    if index == 2:
        nulls[0, 1] = True
        values[0, 1] = np.nan
    if index == 3:
        nulls[1, 7] = True
        values[1, 7] = np.nan
    bases = np.asarray(
        [
            [1.0 + index, 2.0 + 2.0 * index],
            [3.0 + index, 4.0 + 2.0 * index],
        ],
        dtype=np.float64,
    )
    return D1PlaceboIssueSource(
        issue_features=D1IssueFeatures(
            issue_time_utc=issue_time,
            issue_report_id=f"report-{index}",
            grid=grid,
            source_columns=D1_SOURCE_COLUMNS,
            values=values,
            null_mask=nulls,
        ),
        trajectory_base_values=bases,
        trajectory_base_null_mask=np.zeros(bases.shape, dtype=np.bool_),
        feature_store_file_sha256="f" * 64,
        snapshot_reason_codes=np.full((grid.cell_count, 5), index, dtype=np.int64),
        trajectory_base_reason_codes=np.full((grid.cell_count, 2), 100 + index, dtype=np.int64),
    )


def _six_sources() -> tuple[D1PlaceboIssueSource, ...]:
    grid = _grid()
    return tuple(_source(index, grid) for index in range(6))


def _microseconds(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000)


def _snapshot_history(
    sources: tuple[D1PlaceboIssueSource, ...],
    *,
    file_sha256: str = "e" * 64,
) -> D1Stage3SnapshotHistory:
    snapshots = {
        source.issue_time_us: Stage3IssueSnapshot(
            issue_index=index,
            issue_time_utc=source.issue_features.issue_time_utc,
            summary=cast(
                AnomalyState,
                SimpleNamespace(issue_report_id=source.issue_features.issue_report_id),
            ),
            entities=(),
            state_snapshot_id=f"{index:064x}",
            lineage_digest=f"{index + 1:064x}",
        )
        for index, source in enumerate(sources)
    }
    return D1Stage3SnapshotHistory(snapshots, file_sha256)


def _stage3_grid(grid: D1StaticGrid) -> Stage3QueryGrid:
    return Stage3QueryGrid(
        grid_id=grid.grid_id,
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=25.0,
        cell_ids=grid.cell_ids,
        rows=grid.rows,
        columns=grid.columns,
        query_xy_m=np.column_stack([grid.query_x_m, grid.query_y_m]),
        clipped_area_km2=grid.clipped_area_km2,
    )


def _small_prepared_placebo() -> D1PlaceboPreparedReplay:
    sources = _six_sources()
    plans = tuple(
        D1FoldPlaceboPlan(
            fold_id=fold_id,
            fit_cutoff_us=sources[2].issue_time_us,
            scored_issue_times_us=tuple(item.issue_time_us for item in sources[2:]),
        )
        for fold_id in ("fold_1", "fold_2", "fold_3")
    )
    history = _snapshot_history(sources)
    return D1PlaceboPreparedReplay(
        issue_sources=sources,
        fold_plans=plans,
        snapshots_by_issue_us=history,
        query_grid=_stage3_grid(_grid()),
        construction_stratum_by_state_id={},
        all_zone_ids=("zone-00",),
        identities=MappingProxyType(
            {
                "contract_sha256": "a" * 64,
                "observed_input_sha256": "d" * 64,
                "input_sha256": "b" * 64,
                "git_commit": "c" * 40,
            }
        ),
        score_fold=lambda _fold, _features: None,
        expected_issue_count=len(sources),
        expected_zone_count=1,
    )


def _touch_repository_file(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode("utf-8"))
    return path


def _synthetic_observed_preparation(
    repository_root: Path,
    sources: tuple[D1PlaceboIssueSource, ...],
) -> D1PreparedReplay:
    feature_sha256 = "f" * 64
    state_sha256 = "e" * 64
    public_sha256 = "d" * 64
    artifact_sha256 = {
        "cell_mapping": "1" * 64,
        "entity_mapping": "2" * 64,
        "zone_geometry": "3" * 64,
        "connectors": "4" * 64,
    }
    anomaly_path = "data/feature.parquet"
    state_path = "data/state.parquet"
    public_path = "data/spatial.json"
    local_paths = {
        "cell_mapping": "data/cell.parquet",
        "entity_mapping": "data/entity.parquet",
        "zone_geometry": "data/zones.parquet",
        "connectors": "data/connectors.json",
    }
    for relative in (anomaly_path, state_path, public_path, *local_paths.values()):
        _touch_repository_file(repository_root, relative)

    cutoffs = {"fold_1": 50, "fold_2": 100, "fold_3": 150}
    water_folds: list[dict[str, object]] = []
    fit_times: dict[str, tuple[int, ...]] = {}
    assessment_times: dict[tuple[str, int], tuple[int, ...]] = {}
    for fold_id, cutoff_index in cutoffs.items():
        cutoff = sources[cutoff_index].issue_features.issue_time_utc
        fit_times[fold_id] = (
            sources[cutoff_index - 20].issue_time_us,
            sources[cutoff_index - 10].issue_time_us,
        )
        assessment_times[(fold_id, 30)] = (sources[cutoff_index + 1].issue_time_us,)
        assessment_times[(fold_id, 90)] = (sources[cutoff_index + 2].issue_time_us,)
        water_folds.append(
            {
                "fold_id": fold_id,
                "fit_issue_cutoff_local_inclusive": cutoff.isoformat(),
                "fit": {
                    "raw_issue_count": cutoff_index + 1,
                    "latest_raw_issue_local": cutoff.isoformat(),
                },
            }
        )

    config = MappingProxyType(
        {
            "data": {
                "local_data_root_current_machine": str(repository_root / "data"),
                "anomaly_features": {
                    "path": anomaly_path,
                    "feature_store_file_sha256": feature_sha256,
                    "state_history_path": state_path,
                    "state_history_file_sha256": state_sha256,
                    "snapshot_count": 205,
                    "query_cell_count": 15_697,
                },
                "spatial_strata": {
                    "public_manifest": public_path,
                    "public_manifest_content_sha256": public_sha256,
                    "local_coordinate_artifacts_not_committed": local_paths,
                    "local_artifact_sha256": artifact_sha256,
                },
            }
        }
    )
    protocol = SimpleNamespace(
        repository_root=repository_root,
        config=config,
        water_level={"folds": water_folds},
    )
    target_layer = SimpleNamespace(
        fit_for=lambda fold_id: SimpleNamespace(issue_times_us=fit_times[fold_id]),
        assessment_for=lambda fold_id, horizon: SimpleNamespace(
            issue_times_us=assessment_times[(fold_id, horizon)]
        ),
    )
    return D1PreparedReplay(
        protocol=cast(Any, protocol),
        identities={
            "contract_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "git_commit": "c" * 40,
            "input_files": {"anomaly_features": feature_sha256},
        },
        catalog=cast(Any, None),
        target_layer=cast(Any, target_layer),
        domain=cast(Any, SimpleNamespace(stage3_grid=_stage3_grid(_grid()))),
        feature_contract=cast(Any, None),
        features_by_issue={},
        backgrounds={},
        counts_by_fold={},
    )


def test_rng_is_exact_seedsequence_pcg64_stream() -> None:
    observed = d1_placebo_rng("time", "fold_2", 17).integers(0, 2**31, size=12)
    expected = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([147, 2, 2, 17]))
    ).integers(0, 2**31, size=12)
    assert np.array_equal(observed, expected)
    assert not np.array_equal(
        observed,
        d1_placebo_rng("space", "fold_2", 17).integers(0, 2**31, size=12),
    )


def test_prepare_binds_every_frozen_placebo_input_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    sources = tuple(_source(index, _grid()) for index in range(205))
    observed = _synthetic_observed_preparation(repository_root, sources)
    snapshot_history = _snapshot_history(sources, file_sha256="e" * 64)
    artifact_hashes = {
        "cell_mapping": "1" * 64,
        "entity_mapping": "2" * 64,
        "zone_geometry": "3" * 64,
        "connectors": "4" * 64,
    }
    verified = D1VerifiedSpatialStrata(
        all_zone_ids=tuple(f"zone-{index:02d}" for index in range(39)),
        construction_stratum_by_state_id={},
        artifact_sha256=artifact_hashes,
        public_manifest_content_sha256="d" * 64,
    )
    calls: dict[str, object] = {}

    def fake_load_sources(path: Path, **kwargs: object) -> object:
        calls["feature_path"] = path
        calls["feature_kwargs"] = kwargs
        return sources

    def fake_load_snapshots(path: Path, **kwargs: object) -> object:
        calls["state_path"] = path
        calls["state_kwargs"] = kwargs
        return snapshot_history

    def fake_verify(**kwargs: object) -> object:
        calls["spatial_kwargs"] = kwargs
        return verified

    monkeypatch.setattr(placebos, "load_d1_placebo_issue_sources", fake_load_sources)
    monkeypatch.setattr(placebos, "load_d1_stage3_snapshots", fake_load_snapshots)
    monkeypatch.setattr(placebos, "verify_d1_spatial_strata_files", fake_verify)

    prepared = prepare_d1_placebo_replay(observed)
    expected_placebo_only = {
        "anomaly_feature_store": "f" * 64,
        "anomaly_state_history": "e" * 64,
        "spatial_public_manifest_content": "d" * 64,
        **{f"spatial_{name}": digest for name, digest in sorted(artifact_hashes.items())},
    }
    expected_input_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "observed_input_sha256": "b" * 64,
                "placebo_only_input_files": expected_placebo_only,
            }
        )
    ).hexdigest()
    assert prepared.identities["observed_input_sha256"] == "b" * 64
    assert prepared.identities["input_sha256"] == expected_input_sha256
    assert prepared.identities["feature_store_file_sha256"] == "f" * 64
    assert prepared.identities["state_history_file_sha256"] == "e" * 64
    assert dict(cast(dict[str, str], prepared.identities["input_files"])) == {
        "anomaly_features": "f" * 64,
        **expected_placebo_only,
    }
    assert cast(dict[str, object], calls["feature_kwargs"])["expected_file_sha256"] == "f" * 64
    assert cast(dict[str, object], calls["state_kwargs"])["expected_file_sha256"] == "e" * 64
    assert (
        cast(dict[str, object], calls["spatial_kwargs"])["expected_artifact_sha256"]
        == artifact_hashes
    )
    assert tuple(item.fold_id for item in prepared.fold_plans) == (
        "fold_1",
        "fold_2",
        "fold_3",
    )


def test_configured_placebo_path_cannot_escape_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        placebos._configured_data_path(
            tmp_path / "repository",
            tmp_path / "data",
            "../outside.parquet",
            label="synthetic",
        )


def test_time_placebo_keeps_two_pools_and_recipient_coverage() -> None:
    sources = _six_sources()
    cutoff = sources[3].issue_time_us
    plan = D1FoldPlaceboPlan(
        fold_id="fold_1",
        fit_cutoff_us=cutoff,
        scored_issue_times_us=(
            sources[2].issue_time_us,
            sources[4].issue_time_us,
            sources[5].issue_time_us,
        ),
    )
    pseudo = build_d1_time_pseudo_history(
        sources,
        plan,
        4,
        expected_issue_count=len(sources),
    )
    assert tuple(pseudo.features_by_issue) == tuple(item.issue_time_us for item in sources)
    fit_pairs = cast(tuple[tuple[int, int], ...], pseudo.mapping_audit["fit_pool_pairs"])
    post_pairs = cast(
        tuple[tuple[int, int], ...],
        pseudo.mapping_audit["post_fit_pool_pairs"],
    )
    assert all(recipient <= cutoff and donor <= cutoff for recipient, donor in fit_pairs)
    assert all(recipient > cutoff and donor > cutoff for recipient, donor in post_pairs)
    assert {item[0] for item in (*fit_pairs, *post_pairs)} == {
        item.issue_time_us for item in sources
    }
    assert sources[3].issue_time_us in {recipient for recipient, _donor in fit_pairs}
    for recipient, rebuilt in zip(sources, pseudo.issue_sources, strict=True):
        assert np.array_equal(
            rebuilt.issue_features.values[:, :4],
            recipient.issue_features.values[:, :4],
            equal_nan=True,
        )
        assert np.array_equal(
            rebuilt.issue_features.null_mask[:, :4],
            recipient.issue_features.null_mask[:, :4],
        )


def test_time_placebo_fit_pool_includes_unscored_2023_03_30_manifest_issue() -> None:
    dates = (
        datetime(2023, 3, 2, tzinfo=UTC),
        datetime(2023, 3, 23, tzinfo=UTC),
        datetime(2023, 3, 30, tzinfo=UTC),
        datetime(2023, 4, 6, tzinfo=UTC),
    )
    sources = tuple(
        replace(
            _source(index, _grid()),
            issue_features=replace(
                _source(index, _grid()).issue_features,
                issue_time_utc=issue_time,
                issue_report_id=f"manifest-report-{index}",
            ),
        )
        for index, issue_time in enumerate(dates)
    )
    plan = D1FoldPlaceboPlan(
        fold_id="fold_1",
        fit_cutoff_us=_microseconds(datetime(2023, 4, 2, tzinfo=UTC)),
        scored_issue_times_us=(
            sources[1].issue_time_us,
            sources[3].issue_time_us,
        ),
    )
    pseudo = build_d1_time_pseudo_history(
        sources,
        plan,
        0,
        expected_issue_count=len(sources),
    )
    fit_recipients = {
        recipient
        for recipient, _donor in cast(
            tuple[tuple[int, int], ...],
            pseudo.mapping_audit["fit_pool_pairs"],
        )
    }
    assert _microseconds(datetime(2023, 3, 30, tzinfo=UTC)) in fit_recipients


def test_time_placebo_moves_joint_raw_fields_and_rebuilds_dynamic() -> None:
    sources = _six_sources()
    plan = D1FoldPlaceboPlan(
        fold_id="fold_3",
        fit_cutoff_us=sources[2].issue_time_us,
        scored_issue_times_us=tuple(item.issue_time_us for item in sources[2:]),
    )
    pseudo = build_d1_time_pseudo_history(
        sources,
        plan,
        9,
        expected_issue_count=len(sources),
    )
    pairs = cast(tuple[tuple[int, int], ...], pseudo.mapping_audit["fit_pool_pairs"]) + cast(
        tuple[tuple[int, int], ...],
        pseudo.mapping_audit["post_fit_pool_pairs"],
    )
    donor_time = dict(pairs)
    observed_by_time = {item.issue_time_us: item for item in sources}
    for rebuilt in pseudo.issue_sources:
        donor = observed_by_time[donor_time[rebuilt.issue_time_us]]
        assert np.array_equal(
            rebuilt.issue_features.values[:, 4:9],
            donor.issue_features.values[:, 4:9],
            equal_nan=True,
        )
        assert np.array_equal(
            rebuilt.issue_features.null_mask[:, 4:9],
            donor.issue_features.null_mask[:, 4:9],
        )
        assert np.array_equal(rebuilt.trajectory_base_values, donor.trajectory_base_values)
        assert np.array_equal(
            cast(np.ndarray[tuple[int, int], np.dtype[np.int64]], rebuilt.snapshot_reason_codes),
            cast(np.ndarray[tuple[int, int], np.dtype[np.int64]], donor.snapshot_reason_codes),
        )
        assert np.array_equal(
            cast(
                np.ndarray[tuple[int, int], np.dtype[np.int64]],
                rebuilt.trajectory_base_reason_codes,
            ),
            cast(
                np.ndarray[tuple[int, int], np.dtype[np.int64]],
                donor.trajectory_base_reason_codes,
            ),
        )

    issue_times = np.asarray(
        [
            np.datetime64(
                item.issue_features.issue_time_utc.astimezone(UTC).replace(tzinfo=None),
                "ns",
            )
            for item in pseudo.issue_sources
        ],
        dtype="datetime64[ns]",
    )
    packed = np.stack(
        [item.trajectory_base_values for item in pseudo.issue_sources],
        axis=0,
    ).reshape(len(pseudo.issue_sources), -1)
    latest = compute_latest_trajectory_features(issue_times, packed)
    final = pseudo.issue_sources[-1].issue_features
    for feature_index, name in enumerate(
        ("slope_13w_per_week", "acceleration_4v13_per_week2", "peak_drop_52w")
    ):
        expected = latest.features[name].reshape(final.grid.cell_count, 2)
        assert np.allclose(
            final.values[:, 9 + feature_index],
            expected[:, 0],
            equal_nan=True,
        )
        assert np.allclose(
            final.values[:, 12 + feature_index],
            expected[:, 1],
            equal_nan=True,
        )
    assert not np.all(final.values[:, 9:15] == 999.0)


def test_space_coordinate_bijection_preserves_each_zone_multiset() -> None:
    entities = (
        D1CoordinateEntity("a", "zone-a:inside", 100.0, 30.0),
        D1CoordinateEntity("b", "zone-a:inside", 101.0, 31.0),
        D1CoordinateEntity("c", "zone-a:outside", 102.0, 32.0),
        D1CoordinateEntity("d", "zone-b:inside", 110.0, 40.0),
        D1CoordinateEntity("e", "zone-b:inside", 111.0, 41.0),
    )
    replacement, audits = permute_d1_coordinates_within_zones(
        entities,
        ("zone-b", "zone-a"),
        rng=d1_placebo_rng("space", "fold_1", 3),
    )
    assert all(item.coordinate_multiset_verified for item in audits)
    assert set(replacement) == {item.state_id for item in entities}
    for stratum in {item.construction_stratum_id for item in entities}:
        original = sorted(
            (item.longitude, item.latitude)
            for item in entities
            if item.construction_stratum_id == stratum
        )
        permuted = sorted(
            replacement[item.state_id]
            for item in entities
            if item.construction_stratum_id == stratum
        )
        assert original == permuted


def test_space_placebo_recomputes_every_issue_snapshot_base_and_dynamic() -> None:
    sources = _six_sources()
    history = _snapshot_history(sources)
    plan = D1FoldPlaceboPlan(
        fold_id="fold_2",
        fit_cutoff_us=sources[2].issue_time_us,
        scored_issue_times_us=tuple(item.issue_time_us for item in sources[2:]),
    )
    pseudo = build_d1_space_pseudo_history(
        sources,
        history,
        _stage3_grid(_grid()),
        {},
        ("target-independent-zone",),
        plan,
        0,
        expected_issue_count=len(sources),
        expected_zone_count=1,
    )
    assert pseudo.mapping_audit["complete_report_period_count"] == len(sources)
    assert pseudo.mapping_audit["snapshot_and_dynamic_rebuilt_at_200km"] is True
    assert len(pseudo.issue_sources) == len(sources)
    for observed in pseudo.issue_sources:
        assert np.array_equal(
            observed.trajectory_base_values,
            np.zeros((observed.issue_features.grid.cell_count, 2)),
        )
        assert not np.array_equal(
            observed.issue_features.values[:, 4:9],
            sources[0].issue_features.values[:, 4:9],
            equal_nan=True,
        )
    assert not np.all(pseudo.issue_sources[-1].issue_features.values[:, 9:15] == 999.0)


def test_failed_replications_count_as_null_greater_or_equal() -> None:
    nulls: list[float | None] = [0.1] * 190 + [None] * 10
    result = reduce_d1_placebo_contrast(
        D1_PLACEBO_CONTRASTS[0],
        0.5,
        nulls,
    )
    assert result.null_greater_or_equal_count == 10
    assert result.monte_carlo_p_value == pytest.approx(11 / 201)
    assert result.observed_exceeds_fraction == pytest.approx(0.95)
    assert result.scientific_failure_fraction == pytest.approx(0.05)
    assert result.status == "passed"
    insufficient = reduce_d1_placebo_contrast(
        D1_PLACEBO_CONTRASTS[0],
        0.5,
        [0.1] * 189 + [None] * 11,
    )
    assert insufficient.status == "evidence_insufficient"


def _replication(
    index: int,
    *,
    fold_id: str = "fold_1",
    support_count: int = 8,
) -> D1PlaceboFoldReplication:
    return D1PlaceboFoldReplication(
        replication_index=index,
        mapping_sha256=f"{index:064x}",
        score=D1FoldPlaceboScore(
            fold_id=fold_id,
            support_count=support_count,
            model_hit_counts={
                "B0_C": 2,
                "B0_C_A_snapshot": 3,
                "B0_C_A_dynamic": 4,
            },
            selected_alpha=0.25,
            selected_ridge_by_model={
                "B0_C": 1.0,
                "B0_C_A_snapshot": 1.0,
                "B0_C_A_dynamic": 10.0,
            },
        ),
        scientific_failure=False,
    )


def test_checkpoint_is_identity_bound_and_only_written_at_25_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoints" / "time_fold_1.json"
    identities = {
        "contract_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }
    with pytest.raises(ValueError, match="25"):
        write_d1_placebo_checkpoint(
            path,
            identities=identities,
            kind="time",
            fold_id="fold_1",
            records=tuple(_replication(index) for index in range(24)),
            status="running",
        )
    records = tuple(_replication(index) for index in range(25))
    write_d1_placebo_checkpoint(
        path,
        identities=identities,
        kind="time",
        fold_id="fold_1",
        records=records,
        status="running",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["last_replication"] == 24
    assert payload["completed_replication_count"] == 25
    assert (
        load_d1_placebo_checkpoint(
            path,
            identities=identities,
            kind="time",
            fold_id="fold_1",
        )
        == records
    )
    with pytest.raises(D1PlaceboInfrastructureInterruption, match="identity"):
        load_d1_placebo_checkpoint(
            path,
            identities={**identities, "input_sha256": "d" * 64},
            kind="time",
            fold_id="fold_1",
        )


def test_schedule_and_completed_checkpoints_are_exactly_200_with_8_6_7_support(
    tmp_path: Path,
) -> None:
    schedule = d1_placebo_schedule()
    assert set(schedule) == {"time", "space"}
    assert all(
        dict(schedule[kind]) == {"fold_1": 200, "fold_2": 200, "fold_3": 200}
        for kind in ("time", "space")
    )
    identities = {
        "contract_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }
    support_by_fold = {"fold_1": 8, "fold_2": 6, "fold_3": 7}
    for fold_id, support_count in support_by_fold.items():
        path = tmp_path / f"time_{fold_id}.json"
        partial = tuple(
            _replication(index, fold_id=fold_id, support_count=support_count) for index in range(25)
        )
        with pytest.raises(ValueError, match="all 200"):
            write_d1_placebo_checkpoint(
                path,
                identities=identities,
                kind="time",
                fold_id=fold_id,
                records=partial,
                status="completed",
            )
        complete = tuple(
            _replication(index, fold_id=fold_id, support_count=support_count)
            for index in range(200)
        )
        write_d1_placebo_checkpoint(
            path,
            identities=identities,
            kind="time",
            fold_id=fold_id,
            records=complete,
            status="completed",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
        assert payload["completed"] is True
        assert payload["last_replication"] == 199
        assert payload["completed_replication_count"] == 200
        assert payload["expected_support_count"] == support_count
        assert (
            load_d1_placebo_checkpoint(
                path,
                identities=identities,
                kind="time",
                fold_id=fold_id,
            )
            == complete
        )


def test_checkpoint_rejects_wrong_fold_support_and_tampered_status(
    tmp_path: Path,
) -> None:
    identities = {
        "contract_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }
    path = tmp_path / "time_fold_2.json"
    wrong_support = tuple(
        _replication(index, fold_id="fold_2", support_count=8) for index in range(25)
    )
    with pytest.raises(D1PlaceboInfrastructureInterruption, match="8/6/7"):
        write_d1_placebo_checkpoint(
            path,
            identities=identities,
            kind="time",
            fold_id="fold_2",
            records=wrong_support,
            status="running",
        )

    records = tuple(_replication(index, fold_id="fold_2", support_count=6) for index in range(25))
    write_d1_placebo_checkpoint(
        path,
        identities=identities,
        kind="time",
        fold_id="fold_2",
        records=records,
        status="running",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(D1PlaceboInfrastructureInterruption, match="completed flag"):
        load_d1_placebo_checkpoint(
            path,
            identities=identities,
            kind="time",
            fold_id="fold_2",
        )
    payload["completed"] = False
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(D1PlaceboInfrastructureInterruption, match="schema version"):
        load_d1_placebo_checkpoint(
            path,
            identities=identities,
            kind="time",
            fold_id="fold_2",
        )


def test_worker_boundary_reserves_two_physical_cores_and_never_exceeds_three_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(placebos, "detect_physical_core_count", lambda: 48)
    assert placebos._bounded_workers(4) == 3
    monkeypatch.setattr(placebos, "detect_physical_core_count", lambda: 4)
    assert placebos._bounded_workers(4) == 2
    monkeypatch.setattr(placebos, "detect_physical_core_count", lambda: 3)
    assert placebos._bounded_workers(4) == 1
    monkeypatch.setattr(placebos, "detect_physical_core_count", lambda: 2)
    with pytest.raises(RuntimeError, match="two physical cores"):
        placebos._bounded_workers(1)


def test_blas_runtime_limit_is_fail_closed_before_any_replication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seismoflux.d1_replay.runner as runner

    class BrokenRuntime:
        def __enter__(self) -> None:
            raise RuntimeError("synthetic BLAS limiter failure")

        def __exit__(self, *_args: object) -> None:
            return None

    def broken_runtime() -> BrokenRuntime:
        return BrokenRuntime()

    monkeypatch.setattr(placebos, "detect_physical_core_count", lambda: 48)
    monkeypatch.setattr(runner, "_single_thread_math_runtime", broken_runtime)
    baseline = D1ObservedPlaceboBaseline(
        observed_statistics={contrast: 0.0 for contrast in D1_PLACEBO_CONTRASTS},
        support_by_fold={"fold_1": 8, "fold_2": 6, "fold_3": 7},
        identities={
            "contract_sha256": "a" * 64,
            "input_sha256": "d" * 64,
            "git_commit": "c" * 40,
        },
    )
    with pytest.raises(RuntimeError, match="BLAS limiter failure"):
        run_d1_placebos(
            _small_prepared_placebo(),
            baseline,
            tmp_path,
            workers=4,
        )
    assert not (tmp_path / "checkpoints").exists()
    assert all(os.environ[name] == "1" for name in placebos._BLAS_THREAD_ENV)


def test_rng_validation_rejects_noncanonical_context() -> None:
    with pytest.raises(ValueError):
        d1_placebo_rng("time", "fold_4", 0)
    with pytest.raises(ValueError):
        d1_placebo_rng("time", "fold_1", 200)
    assert _microseconds(datetime(1970, 1, 1, tzinfo=UTC)) == 0
