from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.anomaly_increment.config import (
    ExpectedTargetIdentity,
    FrozenManifest,
    Stage4ProtocolBundle,
    load_stage4_protocol_bundle,
)
from seismoflux.anomaly_increment.feature_adapter import (
    assert_issue_table_matches_frozen_grid,
    concatenate_source_columns,
    feature_set_contract,
    load_verified_issue_feature_tables,
)
from seismoflux.anomaly_increment.grid_features import Stage4IntegrationGrid
from seismoflux.anomaly_increment.qualification import ScoreBlindInputEvidence
from seismoflux.anomaly_increment.restricted_access import (
    RESTRICTED_ARTIFACT_IDS,
    RestrictedLocalArtifactAccessControlEvidence,
    RestrictedPathAccessObservation,
    canonical_permission_descriptor,
)


def _grid() -> Stage4IntegrationGrid:
    return Stage4IntegrationGrid(
        grid_id="synthetic-grid",
        equal_area_crs=(
            "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
            "+datum=WGS84 +units=m +no_defs +type=crs"
        ),
        cell_size_km=25.0,
        cell_ids=("c0", "c1"),
        rows=np.asarray([0, 0], dtype=np.int64),
        columns=np.asarray([0, 1], dtype=np.int64),
        query_xy_m=np.asarray([[0.0, 0.0], [25_000.0, 0.0]], dtype=np.float64),
        clipped_area_km2=np.asarray([625.0, 500.0], dtype=np.float64),
    )


def _table(issue_time: datetime) -> pa.Table:
    return pa.table(
        {
            "issue_time_utc": pa.array(
                [issue_time, issue_time],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "issue_report_id": ["issue", "issue"],
            "grid_id": ["synthetic-grid", "synthetic-grid"],
            "cell_id": ["c0", "c1"],
            "cell_row": pa.array([0, 0], type=pa.int32()),
            "cell_column": pa.array([0, 1], type=pa.int32()),
            "query_x_m": [0.0, 25_000.0],
            "query_y_m": [0.0, 0.0],
            "clipped_area_km2": [625.0, 500.0],
            "signal": pa.array([1.5, None], type=pa.float64()),
        }
    )


def _bundle(root: Path) -> Stage4ProtocolBundle:
    digest = "a" * 64
    empty_manifest = FrozenManifest(
        manifest_id="dummy",
        relative_path="dummy.json",
        expected_file_sha256=digest,
        document=MappingProxyType({"content_sha256": digest}),
    )
    target = ExpectedTargetIdentity(
        relative_path="unobserved/target.parquet",
        expected_file_sha256="b" * 64,
        expected_content_sha256="c" * 64,
        expected_schema_sha256="d" * 64,
        contract_relative_path="unobserved/target.json",
        expected_contract_sha256="e" * 64,
        physical_event_id_column="event_id",
    )
    return Stage4ProtocolBundle(
        repository_root=root,
        protocol_path=root / "protocol.yaml",
        protocol=MappingProxyType(
            {
                "inputs": {
                    "stage3_feature_store": {
                        "path": "synthetic.parquet",
                        "sha256": digest,
                    }
                }
            }
        ),
        fold=empty_manifest,
        feature_set=empty_manifest,
        randomness=empty_manifest,
        spatial_strata=empty_manifest,
        expected_target=target,
        validation_receipt=MappingProxyType({}),
    )


def _evidence() -> ScoreBlindInputEvidence:
    digest = "a" * 64
    directory = "restricted"
    artifacts = tuple(
        (artifact_id, f"{directory}/{artifact_id}.bin", digest)
        for artifact_id in RESTRICTED_ARTIFACT_IDS
    )
    observations = [
        RestrictedPathAccessObservation(
            relative_path=directory,
            path_kind="directory",
            verified_sha256=None,
            owner_principal="uid:1000",
            link_count=1,
            reparse_point=False,
            permission_descriptor_json=canonical_permission_descriptor(
                {
                    "acl_model": "classic_mode_bits_without_acl_xattrs",
                    "acl_related_xattrs": [],
                    "filesystem_locality": "local",
                    "filesystem_magic_hex": "0x0000ef53",
                    "filesystem_type": "ext2_ext3_ext4",
                    "mode_octal": "0700",
                    "owner_uid": 1000,
                    "platform_family": "linux",
                    "queried_by_handle": True,
                }
            ),
        ),
        *(
            RestrictedPathAccessObservation(
                relative_path=path,
                path_kind="regular_file",
                verified_sha256=file_sha256,
                owner_principal="uid:1000",
                link_count=1,
                reparse_point=False,
                permission_descriptor_json=canonical_permission_descriptor(
                    {
                        "acl_model": "classic_mode_bits_without_acl_xattrs",
                        "acl_related_xattrs": [],
                        "filesystem_locality": "local",
                        "filesystem_magic_hex": "0x0000ef53",
                        "filesystem_type": "ext2_ext3_ext4",
                        "mode_octal": "0600",
                        "owner_uid": 1000,
                        "platform_family": "linux",
                        "queried_by_handle": True,
                    }
                ),
            )
            for _, path, file_sha256 in artifacts
        ),
    ]
    access = RestrictedLocalArtifactAccessControlEvidence(
        protocol_design_sha256=digest,
        access_contract_sha256=digest,
        platform="posix",
        current_principal="uid:1000",
        directory_relative_path=directory,
        file_artifacts=artifacts,
        observations=tuple(sorted(observations, key=lambda item: item.relative_path)),
    )
    return ScoreBlindInputEvidence(
        protocol_design_sha256=digest,
        random_input_seal_sha256=digest,
        protocol_validation_sha256=digest,
        observed_project_input_hashes=(("stage3_feature_store", digest),),
        generated_manifest_hashes=(),
        restricted_spatial_artifact_hashes=tuple(
            (artifact_id, file_sha256) for artifact_id, _, file_sha256 in artifacts
        ),
        restricted_access_control=access,
    )


def test_real_feature_manifest_builds_exact_nested_executable_contracts() -> None:
    bundle = load_stage4_protocol_bundle(Path(__file__).resolve().parents[2])
    coverage = feature_set_contract(bundle, "coverage_only")
    snapshot = feature_set_contract(bundle, "snapshot")
    dynamic = feature_set_contract(bundle, "dynamic")

    assert len(coverage.contracts) == 9
    assert len(snapshot.contracts) == 17
    assert len(dynamic.contracts) == 27
    assert set(coverage.source_columns) < set(snapshot.source_columns) < set(dynamic.source_columns)
    assert all(item.penalty_factor == 1.0 for item in dynamic.contracts)


def test_issue_table_grid_identity_and_null_bitmap_are_preserved() -> None:
    issue_time = datetime(2025, 1, 1, tzinfo=UTC)
    table = _table(issue_time)
    assert_issue_table_matches_frozen_grid(table, issue_time_utc=issue_time, grid=_grid())
    columns = concatenate_source_columns((table,), source_columns=("signal",))
    np.testing.assert_array_equal(columns["signal"][:1], np.asarray([1.5]))
    assert np.isnan(columns["signal"][1])

    tampered = table.set_column(
        table.schema.get_field_index("cell_column"),
        "cell_column",
        pa.array([1, 0], type=pa.int32()),
    )
    with pytest.raises(ValueError, match="cell_column"):
        assert_issue_table_matches_frozen_grid(
            tampered,
            issue_time_utc=issue_time,
            grid=_grid(),
        )


def test_verified_reader_selects_only_the_requested_score_blind_row_group(
    tmp_path: Path,
) -> None:
    issue_time = datetime(2025, 1, 1, tzinfo=UTC)
    pq.write_table(_table(issue_time), tmp_path / "synthetic.parquet", row_group_size=2)
    output = load_verified_issue_feature_tables(
        _bundle(tmp_path),
        _evidence(),
        issue_times_utc=(issue_time,),
        columns=("signal",),
        primary_grid=_grid(),
    )
    assert tuple(output) == (issue_time,)
    assert output[issue_time].column_names[-1] == "signal"

    wrong = replace(
        _evidence(),
        observed_project_input_hashes=(("stage3_feature_store", "f" * 64),),
    )
    with pytest.raises(ValueError, match="does not verify"):
        load_verified_issue_feature_tables(
            _bundle(tmp_path),
            wrong,
            issue_times_utc=(issue_time,),
            columns=("signal",),
            primary_grid=_grid(),
        )
