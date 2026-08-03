from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

from seismoflux.d1_replay.protocol import load_d1_protocol
from seismoflux.d1_replay.targets import (
    MICROSECONDS_PER_DAY,
    assign_target_events,
    build_global_clusters,
    build_score_blind_target_layer,
)
from seismoflux.stage2s.catalog import (
    CatalogIdentity,
    Stage2SEarthquakeCatalog,
    parse_frozen_catalog_bytes,
)


def _catalog(
    rows: list[tuple[str, int, int, float, float, float, bool]],
) -> Stage2SEarthquakeCatalog:
    """Build the smallest valid shared catalog for target-only tests."""

    ordered = sorted(rows, key=lambda row: (row[1], row[0].encode("utf-8")))
    count = len(ordered)
    return Stage2SEarthquakeCatalog(
        identity=CatalogIdentity(
            row_count=count,
            file_sha256="0" * 64,
            content_sha256="1" * 64,
            schema_sha256="2" * 64,
        ),
        event_ids=tuple(row[0] for row in ordered),
        origin_time_us=np.asarray([row[1] for row in ordered], dtype=np.int64),
        available_at_us=np.asarray([row[2] for row in ordered], dtype=np.int64),
        longitude=np.asarray([row[3] for row in ordered], dtype=np.float64),
        latitude=np.asarray([row[4] for row in ordered], dtype=np.float64),
        magnitude=np.asarray([row[5] for row in ordered], dtype=np.float64),
        inside_study_area=np.asarray([row[6] for row in ordered], dtype=np.bool_),
        _table=pa.table({"row": pa.array(range(count), type=pa.int64())}),
    )


def test_target_window_is_open_left_closed_right_and_freeze_causal() -> None:
    issue = 1_700_000_000_000_000
    horizon = 30 * MICROSECONDS_PER_DAY
    freeze = issue + horizon + MICROSECONDS_PER_DAY
    catalog = _catalog(
        [
            ("at_left", issue, issue, 100.0, 30.0, 5.0, True),
            ("after_left", issue + 1, issue + 1, 100.0, 30.0, 4.0, True),
            ("below_m4", issue + 2, issue + 2, 100.0, 30.0, 3.9, True),
            ("outside", issue + 3, issue + 3, 100.0, 30.0, 5.0, False),
            ("late", issue + 4, freeze + 1, 100.0, 30.0, 5.0, True),
            ("at_right", issue + horizon, issue + horizon, 100.0, 30.0, 5.9, True),
            (
                "after_right",
                issue + horizon + 1,
                issue + horizon + 1,
                100.0,
                30.0,
                5.0,
                True,
            ),
        ]
    )

    targets = assign_target_events(
        catalog,
        role="fit",
        fold_id="fold_1",
        horizon_days=30,
        issue_times_us=(issue,),
        catalog_freeze_us=freeze,
        magnitude_minimum_inclusive=4.0,
        magnitude_maximum_exclusive=None,
    )

    assert set(targets.event_ids(catalog)) == {"after_left", "at_right"}
    assert targets.assigned_issue_times_us == (issue, issue)
    assert targets.late_available_target_count == 1


def test_global_clusters_use_transitive_wgs84_connected_components() -> None:
    issue = 1_700_000_000_000_000
    origin = issue + MICROSECONDS_PER_DAY
    catalog = _catalog(
        [
            ("a", origin, origin, 100.0, 30.0, 5.2, True),
            ("b", origin, origin, 100.6, 30.0, 5.2, True),
            ("c", origin, origin, 101.2, 30.0, 5.2, True),
            ("far", origin, origin, 110.0, 30.0, 5.2, True),
        ]
    )
    targets = assign_target_events(
        catalog,
        role="assessment",
        fold_id="fold_1",
        horizon_days=30,
        issue_times_us=(issue,),
        catalog_freeze_us=origin,
        magnitude_minimum_inclusive=5.0,
        magnitude_maximum_exclusive=6.0,
    )

    clusters = build_global_clusters(catalog, (targets,))

    members = [
        {catalog.event_ids[index] for index in cluster.member_event_indices} for cluster in clusters
    ]
    members.sort(key=len, reverse=True)
    assert members == [{"a", "b", "c"}, {"far"}]


def test_cluster_representative_is_earliest_eligible_event_per_horizon() -> None:
    issue_90 = 1_700_000_000_000_000
    issue_30 = issue_90 + 20 * MICROSECONDS_PER_DAY
    early = issue_90 + 10 * MICROSECONDS_PER_DAY
    later = issue_90 + 25 * MICROSECONDS_PER_DAY
    catalog = _catalog(
        [
            ("early_90_only", early, early, 100.0, 30.0, 5.1, True),
            ("later_both", later, later, 100.0, 30.0, 5.2, True),
        ]
    )
    targets_90 = assign_target_events(
        catalog,
        role="assessment",
        fold_id="fold_1",
        horizon_days=90,
        issue_times_us=(issue_90,),
        catalog_freeze_us=later,
        magnitude_minimum_inclusive=5.0,
        magnitude_maximum_exclusive=6.0,
    )
    targets_30 = assign_target_events(
        catalog,
        role="assessment",
        fold_id="fold_2",
        horizon_days=30,
        issue_times_us=(issue_30,),
        catalog_freeze_us=later,
        magnitude_minimum_inclusive=5.0,
        magnitude_maximum_exclusive=6.0,
    )

    (cluster,) = build_global_clusters(catalog, (targets_30, targets_90))
    representative_30 = cluster.representative(30)
    representative_90 = cluster.representative(90)

    assert representative_30 is not None
    assert catalog.event_ids[representative_30.event_index] == "later_both"
    assert representative_30.fold_id == "fold_2"
    assert representative_30.assigned_issue_time_us == issue_30
    assert representative_90 is not None
    assert catalog.event_ids[representative_90.event_index] == "early_90_only"
    assert representative_90.fold_id == "fold_1"
    assert representative_90.assigned_issue_time_us == issue_90


def test_frozen_manifest_reconstructs_every_count_and_canonical_identity() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protocol = load_d1_protocol(repository_root)
    catalog_binding = protocol.water_level["input_bindings"]["earthquake_event"]
    catalog_path = protocol.resolve_repository_path(
        catalog_binding["path"], label="frozen earthquake catalog"
    )
    catalog = parse_frozen_catalog_bytes(catalog_path.read_bytes())

    layer = build_score_blind_target_layer(protocol, catalog)

    assert [targets.event_count for targets in layer.fit_targets] == [60, 182, 280]
    assert [targets.event_count for targets in layer.assessment_targets] == [
        16,
        18,
        6,
        6,
        7,
        8,
    ]
    assert len(layer.clusters) == 23
    assert Counter(len(cluster.member_event_indices) for cluster in layer.clusters) == {
        1: 21,
        2: 1,
        10: 1,
    }
    assert sum(cluster.representative(30) is not None for cluster in layer.clusters) == 21
    assert sum(cluster.representative(90) is not None for cluster in layer.clusters) == 22
    assert protocol.water_level["model_effect_fields_read"] == []
