from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from seismoflux.anomaly_increment import preregistration


def _assert_two_atomic_parquet_replacements(
    path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    write_first: Callable[[], None],
    write_second: Callable[[], None],
) -> None:
    stale_evidence = path.parent / f".{path.name}.prior-run.tmp"
    stale_evidence.write_bytes(b"preserve prior failed-run evidence")
    temporary_pattern = f".{path.name}.*.tmp"
    temporary_files_before = set(path.parent.glob(temporary_pattern))

    actual_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def tracked_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        replacements.append((Path(source), Path(destination)))
        actual_replace(source, destination)

    monkeypatch.setattr(os, "replace", tracked_replace)

    write_first()
    first_payload = path.read_bytes()
    write_second()
    second_payload = path.read_bytes()

    assert first_payload != second_payload
    assert len(replacements) == 2
    assert all(destination == path for _, destination in replacements)
    assert all(source.name.startswith(f".{path.name}.") for source, _ in replacements)
    assert all(source.suffix == ".tmp" for source, _ in replacements)
    assert all(not source.exists() for source, _ in replacements)
    assert set(path.parent.glob(temporary_pattern)) == temporary_files_before
    assert stale_evidence.read_bytes() == b"preserve prior failed-run evidence"


def test_cell_mapping_writer_replaces_same_path_twice_without_temp_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / preregistration.LOCAL_MAPPING_FILENAME
    grid = preregistration._Stage3GridRows(
        grid_id="grid-v1",
        cell_ids=("cell-0001",),
        rows=np.asarray([0], dtype=np.int64),
        columns=np.asarray([0], dtype=np.int64),
        query_x_m=np.asarray([100.0], dtype=np.float64),
        query_y_m=np.asarray([200.0], dtype=np.float64),
    )
    first = preregistration._CellAssignments(("zone-a",), 0, 0)
    second = preregistration._CellAssignments(("zone-b",), 0, 0)

    _assert_two_atomic_parquet_replacements(
        path,
        monkeypatch=monkeypatch,
        write_first=lambda: preregistration._write_mapping_parquet(
            path,
            grid=grid,
            assignments=first,
        ),
        write_second=lambda: preregistration._write_mapping_parquet(
            path,
            grid=grid,
            assignments=second,
        ),
    )


def test_entity_mapping_writer_replaces_same_path_twice_without_temp_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / preregistration.LOCAL_ENTITY_MAPPING_FILENAME

    def assignments(anomaly_id: str) -> preregistration._EntityAssignments:
        return preregistration._EntityAssignments(
            state_ids=("state-0001",),
            anomaly_ids=(anomaly_id,),
            issue_times_utc=(datetime(2023, 1, 1, tzinfo=UTC),),
            construction_stratum_ids=("zone-a:inside",),
            coordinate_pair_sha256=("0" * 64,),
            outside_study_area=(False,),
            total_entity_state_count=1,
            spatially_ineligible_count=0,
            boundary_tie_count=0,
            precision_snap_count=0,
            nearest_distance_tie_count=0,
        )

    first = assignments("anomaly-a")
    second = assignments("anomaly-b")

    _assert_two_atomic_parquet_replacements(
        path,
        monkeypatch=monkeypatch,
        write_first=lambda: preregistration._write_entity_mapping_parquet(path, first),
        write_second=lambda: preregistration._write_entity_mapping_parquet(path, second),
    )


def test_zone_geometry_writer_replaces_same_path_twice_without_temp_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / preregistration.LOCAL_ZONES_FILENAME

    _assert_two_atomic_parquet_replacements(
        path,
        monkeypatch=monkeypatch,
        write_first=lambda: preregistration._write_zones_parquet(
            path,
            (box(0.0, 0.0, 1.0, 1.0),),
        ),
        write_second=lambda: preregistration._write_zones_parquet(
            path,
            (box(0.0, 0.0, 2.0, 2.0),),
        ),
    )
