"""Reusable lightweight typed stage-4 formal-preflight receipt fixture."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from seismoflux.anomaly_increment.config import load_stage4_protocol_bundle
from seismoflux.anomaly_increment.contracts import canonical_mapping_sha256
from seismoflux.anomaly_increment.formal_preflight import (
    AsIfShadowIdentityReceipt,
    FormalIssueCalendar,
    FormalIssueIdentityReceipt,
    FormalPreflightReceipt,
    FormalPreflightResources,
    GridIdentityBridgeReceipt,
    SpacePlaceboFeatureIdentityReceipt,
    SpacePlaceboResourceObservationReceipt,
)

ROOT = Path(__file__).resolve().parents[2]


def make_formal_preflight_receipt() -> FormalPreflightReceipt:
    """Return a complete 153-issue receipt without reading any target artifact."""

    protocol = load_stage4_protocol_bundle(ROOT)
    calendar = FormalIssueCalendar.from_protocol(protocol)
    issues = tuple(
        FormalIssueIdentityReceipt(
            issue_id=issue_id,
            issue_time_utc=issue_time,
            issue_index=index,
            accepted_table_sha256="1" * 64,
            projected_table_sha256="2" * 64,
            verified_binding_sha256="3" * 64,
            state_snapshot_id="4" * 64,
            lineage_digest="5" * 64,
        )
        for index, (issue_id, issue_time) in enumerate(
            zip(calendar.issue_ids, calendar.issue_times_utc, strict=True)
        )
    )
    bridge_payload = {
        "cell_count": 15_697,
        "method": (
            "verify_stage3_exact_grid_then_replace_only_grid_id_in_memory_no_reorder_"
            "no_interpolation"
        ),
        "projected_grid_identity_sha256": "7" * 64,
        "schema_version": 1,
        "stage3_grid_id": "stage3-grid",
        "stage3_grid_identity_sha256": "6" * 64,
        "stage4_grid_id": "stage4-grid",
        "unchanged_grid_values_sha256": "8" * 64,
    }
    bridge = GridIdentityBridgeReceipt(
        stage3_grid_id="stage3-grid",
        stage4_grid_id="stage4-grid",
        cell_count=15_697,
        stage3_grid_identity_sha256="6" * 64,
        projected_grid_identity_sha256="7" * 64,
        unchanged_grid_values_sha256="8" * 64,
        bridge_sha256=canonical_mapping_sha256(bridge_payload),
    )
    space_identity = SpacePlaceboFeatureIdentityReceipt(
        mapping_sha256="9" * 64,
        output_identity_sha256="a" * 64,
        coverage_identity_sha256="b" * 64,
        replication_index=0,
        group_count=3,
        moved_entity_row_count=2,
        fixed_point_count=1,
        no_effect_group_count=1,
        output_arrow_logical_bytes=123,
    )
    resource = SpacePlaceboResourceObservationReceipt(
        feature_identity_sha256=space_identity.content_sha256,
        output_identity_sha256=space_identity.output_identity_sha256,
        elapsed_seconds=1.0,
        process_working_set_before_bytes=100,
        process_working_set_after_bytes=200,
        process_peak_working_set_bytes=300,
        arrow_peak_memory_bytes=400,
        system_available_memory_bytes=10_000,
        conservative_per_replica_bytes=1_000,
        recommended_max_in_flight=2,
    )
    groups = tuple(
        (name, "c" * 64) for name in ("coverage", "snapshot", "trajectory_base", "trajectory")
    )
    return FormalPreflightReceipt(
        protocol_design_sha256="d" * 64,
        random_input_seal_sha256="e" * 64,
        score_blind_input_evidence_sha256="f" * 64,
        scoring_code_commit="a" * 40,
        stage3_registry_sha256="1" * 64,
        stage3_bundle_id="bundle-id",
        stage3_bundle_identity_sha256="2" * 64,
        stage3_bundle_manifest_sha256="3" * 64,
        feature_store_file_sha256="4" * 64,
        feature_store_content_sha256="5" * 64,
        feature_store_schema_sha256="6" * 64,
        state_history_file_sha256="7" * 64,
        state_history_content_sha256="8" * 64,
        state_history_schema_sha256="9" * 64,
        feature_manifest_content_sha256="a" * 64,
        calendar=calendar,
        bridge=bridge,
        issue_receipts=issues,
        accepted_formal_tables_sha256="b" * 64,
        projected_formal_tables_sha256="c" * 64,
        state_snapshots_sha256="d" * 64,
        accepted_identity_group_hashes=groups,
        rebuilt_identity_group_hashes=groups,
        space_placebo_feature_identity=space_identity,
        space_placebo_resource_observation=resource,
        shadow=AsIfShadowIdentityReceipt(
            issue_time_utc=datetime(2026, 7, 1, 16, tzinfo=UTC),
            issue_report_id="shadow-report",
            issue_index=204,
            accepted_table_sha256="e" * 64,
            projected_table_sha256="f" * 64,
            verified_binding_sha256="1" * 64,
            state_snapshot_id="2" * 64,
            lineage_digest="3" * 64,
        ),
        resources=FormalPreflightResources(
            physical_cores=8,
            logical_processors=16,
            reserve_physical_cores=2,
            effective_workers=6,
            nested_parallelism=False,
            feature_selected_compressed_bytes=100,
            feature_selected_uncompressed_bytes=200,
            state_selected_uncompressed_bytes=300,
        ),
    )


__all__ = ["make_formal_preflight_receipt"]
