"""Generate or verify the deterministic P1-0B synthetic science artifacts.

This command is deliberately unable to accept a catalogue path or a network
endpoint.  It exercises only the in-memory, known-answer B0 versus B0_R30
rehearsal and therefore creates no evidence about real earthquake prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from seismoflux.data.common import canonical_json_bytes  # noqa: E402
from seismoflux.p1_b0_r30.core import (  # noqa: E402
    PRIMARY_HORIZON_DAYS,
    ScoreSummary,
    build_pending_sequential_reviews,
    ordered_cluster_registry_sha256,
)
from seismoflux.p1_b0_r30.records import (  # noqa: E402
    build_record,
    validate_record_chain,
)
from seismoflux.p1_b0_r30.rendering import (  # noqa: E402
    build_offline_synthetic_explorer_html,
    build_offline_synthetic_forecast_html,
    render_synthetic_forecast_svg,
    render_synthetic_scenarios_svg,
)
from seismoflux.p1_b0_r30.synthetic import (  # noqa: E402
    SYNTHETIC_ISSUE_TIME_UTC,
    SYNTHETIC_QUERY_CUTOFF_UTC,
    SyntheticScenarioResult,
    build_all_synthetic_scenarios,
    build_synthetic_scenario,
    make_synthetic_model_events,
)

RESULT_NAME = "p1_0b_synthetic_result.json"
FORECAST_SVG_NAME = "p1_0b_synthetic_forecast.svg"
FORECAST_HTML_NAME = "p1_0b_synthetic_forecast.html"
SVG_NAME = "p1_0b_synthetic_comparison.svg"
HTML_NAME = "p1_0b_synthetic_explorer.html"
MANIFEST_NAME = "artifact_manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the pure-synthetic P1 B0 versus B0_R30 known-answer "
            "science check. No real catalogue, network, or locked test is read."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new artifact directory, or an existing directory when --check is used",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and byte-check an existing artifact directory without rewriting it",
    )
    parser.add_argument(
        "--refresh-candidate",
        action="store_true",
        help="refresh only the known files of an unaccepted candidate directory",
    )
    return parser


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _summary(scenario: SyntheticScenarioResult) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "expected_direction": scenario.expected_direction,
        "observed_direction": scenario.observed_direction,
        "cluster_count": scenario.score.cluster_count,
        "B0_hit_clusters": scenario.score.B0_hit_clusters,
        "B0_R30_hit_clusters": scenario.score.B0_R30_hit_clusters,
        "B0_recall": scenario.score.B0_recall,
        "B0_R30_recall": scenario.score.B0_R30_recall,
        "recall_gain_percentage_points": scenario.score.recall_gain_percentage_points,
        "B0_actual_alarm_area_km2": scenario.forecast.B0_alarm.actual_area_km2,
        "B0_R30_actual_alarm_area_km2": scenario.forecast.B0_R30_alarm.actual_area_km2,
        "actual_area_difference_km2": scenario.forecast.actual_area_difference_km2,
        "known_answer_passed": scenario.expected_direction == scenario.observed_direction,
    }


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _load_frozen_contract() -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    protocol_path = REPOSITORY_ROOT / "configs" / "p1_b0_r30_prospective.yaml"
    model_path = REPOSITORY_ROOT / "data" / "manifests" / "p1_model_manifest.json"
    source_path = REPOSITORY_ROOT / "data" / "manifests" / "p1_source_boundary_manifest.json"
    schema_path = REPOSITORY_ROOT / "data" / "contracts" / "p1_prospective_records_v1.json"
    protocol_value = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(protocol_value, dict) or not isinstance(schema_value, dict):
        raise ValueError("frozen P1 protocol and record schema must be mappings")
    protocol = protocol_value.get("protocol")
    models = protocol_value.get("models")
    if not isinstance(protocol, dict) or not isinstance(models, dict):
        raise ValueError("frozen P1 protocol is incomplete")
    challenger = models.get("B0_R30")
    if not isinstance(challenger, dict):
        raise ValueError("frozen P1 challenger definition is missing")
    if (
        protocol.get("real_issue_authorized") is not False
        or protocol.get("real_catalog_read_authorized") is not False
        or protocol.get("real_network_fetch_authorized") is not False
        or protocol.get("locked_test_authorized") is not False
        or protocol.get("next_authorized_action") != "P1-0B_synthetic_dual_model_acceptance_only"
        or challenger.get("alpha") != 0.25
    ):
        raise ValueError("frozen protocol does not authorize this synthetic-only rehearsal")
    hashes = {
        "protocol_yaml_sha256": _sha256(protocol_path.read_bytes()),
        "model_manifest_sha256": _sha256(model_path.read_bytes()),
        "source_boundary_manifest_sha256": _sha256(source_path.read_bytes()),
        "record_schema_sha256": _sha256(schema_path.read_bytes()),
    }
    return protocol_value, schema_value, hashes


def _synthetic_commit(label: str, payload: bytes) -> str:
    return hashlib.sha256(label.encode("utf-8") + b"\0" + payload).hexdigest()[:40]


def _build_schema_rehearsal_chain(
    scenario: SyntheticScenarioResult,
    *,
    forecast_svg: bytes,
    forecast_html: bytes,
    schema: Mapping[str, object],
    contract_hashes: Mapping[str, str],
) -> list[dict[str, object]]:
    """Build a schema-valid in-memory chain that never grants real authority."""

    issue = SYNTHETIC_ISSUE_TIME_UTC
    issue_id = f"p1-{issue.strftime('%Y%m%dT%H%M%SZ')}"
    if scenario.forecast.issue_id != issue_id:
        raise ValueError("synthetic forecast issue identity differs from the schema rehearsal")
    if any(cluster.issue_id != issue_id for cluster in scenario.target_clusters):
        raise ValueError("synthetic target cluster issue identity differs from the forecast")
    if any(score.issue_id != issue_id for score in scenario.score.scores):
        raise ValueError("synthetic score issue identity differs from the forecast")
    protocol_commit = "0f43f15bc983a37157f1b129976c7ec0ea47fc7d"
    code_payload = b"".join(
        (REPOSITORY_ROOT / relative).read_bytes()
        for relative in (
            "src/seismoflux/p1_b0_r30/core.py",
            "src/seismoflux/p1_b0_r30/records.py",
            "src/seismoflux/p1_b0_r30/synthetic.py",
            "src/seismoflux/p1_b0_r30/rendering.py",
            "scripts/run_p1_b0_r30_synthetic.py",
        )
    )
    code_commit = _synthetic_commit("synthetic-code-rehearsal", code_payload)
    authorization_commit = _synthetic_commit(
        "synthetic-authorization-rehearsal", canonical_json_bytes(contract_hashes)
    )
    protocol_record = build_record(
        "ProtocolDefinition",
        recorded_at_utc="2026-08-29T23:59:00Z",
        previous_record=None,
        fields={
            "protocol_id": "p1-b0-r30-prospective-v1",
            "protocol_tag": "v0.2.7-p1-b0-r30-protocol",
            "code_tag": "v0.2.7-p1-b0-r30-code",
            "valid_from_utc": _utc_text(issue),
            "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
            "source_boundary_manifest_sha256": contract_hashes["source_boundary_manifest_sha256"],
            "model_manifest_sha256": contract_hashes["model_manifest_sha256"],
            "protocol_commit": protocol_commit,
            "real_issue_authorized": False,
        },
    )
    authorization = build_record(
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-08-30T00:01:00Z",
        previous_record=protocol_record,
        fields={
            "protocol_definition_sha256": protocol_record["content_sha256"],
            "authorization_commit": authorization_commit,
            "code_commit": code_commit,
            "remote_verified_at_utc": "2026-08-30T00:00:00Z",
            "authorized_from_scheduled_issue_utc": _utc_text(issue),
            "real_issue_authorized": True,
        },
    )
    forecast = scenario.forecast
    source_snapshot_sha = _sha256(
        canonical_json_bytes(
            [
                event.as_mapping()
                for event in sorted(
                    make_synthetic_model_events(), key=lambda item: item.event_id.encode("utf-8")
                )
            ]
        )
    )
    forecast_record = build_record(
        "ForecastIssueRecord",
        recorded_at_utc=_utc_text(issue - timedelta(seconds=30)),
        previous_record=authorization,
        fields={
            "issue_id": issue_id,
            "status": "on_time",
            "scheduled_issue_time_utc": _utc_text(issue),
            "query_cutoff_utc": _utc_text(issue - timedelta(minutes=15)),
            "forecast_created_at_utc": _utc_text(issue - timedelta(minutes=10)),
            "publication_completed_at_utc": _utc_text(issue - timedelta(minutes=1)),
            "protocol_definition_sha256": protocol_record["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "model_manifest_sha256": contract_hashes["model_manifest_sha256"],
            "source_boundary_manifest_sha256": contract_hashes["source_boundary_manifest_sha256"],
            "source_snapshot_sha256": source_snapshot_sha,
            "code_commit": code_commit,
            "forecasts": [
                {
                    "model_id": model_id,
                    "relative_intensity_grid_sha256": surface.sha256,
                    "alarm_mask_sha256": alarm.mask_sha256,
                    "alarm_ranking_sha256": alarm.ranking_sha256,
                    "actual_alarm_area_km2": alarm.actual_area_km2,
                }
                for model_id, surface, alarm in (
                    ("B0", forecast.B0, forecast.B0_alarm),
                    ("B0_R30", forecast.B0_R30, forecast.B0_R30_alarm),
                )
            ],
            "static_svg_sha256": _sha256(forecast_svg),
            "offline_interactive_html_sha256": _sha256(forecast_html),
            "B0_reference_area_km2": forecast.B0_reference_area_km2,
            "B0_R30_next_complete_cell_area_km2": (
                forecast.B0_R30_alarm.next_complete_cell_area_km2
            ),
            "actual_area_difference_km2": forecast.actual_area_difference_km2,
            "area_fairness_status": "passed",
            "original_artifacts_immutable": True,
        },
    )
    truth_recorded_at = issue + timedelta(days=61)
    missed_records: list[dict[str, object]] = []
    previous_issue_record: dict[str, object] = forecast_record
    next_issue = issue + timedelta(days=7)
    while next_issue < truth_recorded_at:
        missed = build_record(
            "MissedIssueRecord",
            recorded_at_utc=_utc_text(next_issue + timedelta(minutes=5)),
            previous_record=previous_issue_record,
            fields={
                "issue_id": f"p1-{next_issue.strftime('%Y%m%dT%H%M%SZ')}",
                "status": "missed_issue",
                "scheduled_issue_time_utc": _utc_text(next_issue),
                "authorization_state": "authorized",
                "authorization_record_sha256": authorization["content_sha256"],
                "reason": "source_snapshot_unavailable_before_T",
                "prediction_generated": False,
                "backfill_forbidden": True,
                "valid_from_remains_fixed": True,
            },
        )
        missed_records.append(missed)
        previous_issue_record = missed
        next_issue += timedelta(days=7)
    source_truth_payload = [
        event.as_mapping()
        for event in sorted(
            scenario.target_events,
            key=lambda event: (event.origin_time_utc, event.event_id.encode("utf-8")),
        )
    ]
    cluster_payload = [
        {
            "cluster_id": cluster.cluster_id,
            "member_event_ids": list(cluster.member_event_ids),
            "representative_event": cluster.representative.as_mapping(),
        }
        for cluster in scenario.target_clusters
    ]
    score_payload = [score.as_mapping() for score in scenario.score.scores]
    score_registry_sha = ordered_cluster_registry_sha256(scenario.score.scores)
    truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=_utc_text(truth_recorded_at),
        previous_record=previous_issue_record,
        fields={
            "issue_id": issue_id,
            "horizon_days": 30,
            "protocol_definition_sha256": protocol_record["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "source_snapshot_sha256": _sha256(canonical_json_bytes(source_truth_payload)),
            "status": "mature_truth",
            "mature_after_utc": _utc_text(issue + timedelta(days=60)),
            "truth_fetched_at_utc": _utc_text(issue + timedelta(days=61)),
            "target_event_count": sum(
                len(cluster.member_event_ids) for cluster in scenario.target_clusters
            ),
            "independent_cluster_count": scenario.score.cluster_count,
            "cluster_assignment_sha256": _sha256(canonical_json_bytes(cluster_payload)),
            "exposure_cluster_registry_sha256": score_registry_sha,
            "magnitude_minimum": 5.0,
            "magnitude_maximum_exclusive": 6.0,
        },
    )
    review_fields = scenario.reviews[0].as_mapping()
    review = build_record(
        "SequentialReviewRecord",
        recorded_at_utc=_utc_text(issue + timedelta(days=62)),
        previous_record=truth,
        fields={
            "protocol_definition_sha256": protocol_record["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **review_fields,
        },
    )
    chain = [
        protocol_record,
        authorization,
        forecast_record,
        *missed_records,
        truth,
        review,
    ]
    validate_record_chain(
        chain,
        schema,
        score_registries_by_sha256={score_registry_sha: score_payload},
    )
    return chain


def build_artifacts() -> dict[str, bytes]:
    """Return every deterministic public artifact as bytes."""

    _, record_schema, contract_hashes = _load_frozen_contract()
    scenarios = build_all_synthetic_scenarios()
    mappings = [scenario.as_mapping() for scenario in scenarios]
    forecast_svg = render_synthetic_forecast_svg(mappings[0])
    forecast_html = build_offline_synthetic_forecast_html(mappings[0]).encode("utf-8")
    replay_svg = render_synthetic_scenarios_svg(mappings)
    replay_html = build_offline_synthetic_explorer_html(mappings).encode("utf-8")
    rehearsal_chain = _build_schema_rehearsal_chain(
        scenarios[0],
        forecast_svg=forecast_svg,
        forecast_html=forecast_html,
        schema=record_schema,
        contract_hashes=contract_hashes,
    )
    empty_recent = build_synthetic_scenario("zero", empty_recent=True)
    observed_zero_review = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=PRIMARY_HORIZON_DAYS, scores=()),
        elapsed_months=36.0,
    )[0]
    integrity_pause_review = observed_zero_review.as_mapping()
    integrity_pause_review["decision"] = "pause_scientific_integrity_failure"

    result = {
        "schema_version": 1,
        "stage_id": "P1-0B",
        "protocol_id": "p1-b0-r30-prospective-v1",
        "status": "synthetic_known_answer_accepted_candidate",
        "synthetic_only": True,
        "real_issue_authorized": False,
        "real_catalog_read_authorized": False,
        "real_network_fetch_authorized": False,
        "locked_test_authorized": False,
        "issue_time_utc": SYNTHETIC_ISSUE_TIME_UTC.isoformat().replace("+00:00", "Z"),
        "query_cutoff_utc": SYNTHETIC_QUERY_CUTOFF_UTC.isoformat().replace("+00:00", "Z"),
        "query_cutoff_rule": "Q_equals_T_minus_15_minutes",
        "frozen_contract_hashes": contract_hashes,
        "models": {
            "baseline": "B0",
            "challenger": "B0_R30",
            "challenger_formula": "0.75*B0+0.25*R30",
            "bandwidth_km": 75,
            "relative_intensity_not_absolute_probability": True,
        },
        "alarm": {
            "maximum_area_km2": 600000,
            "complete_cells_only": True,
            "challenger_may_not_exceed_B0_reference_area": True,
        },
        "scenario_summaries": [_summary(scenario) for scenario in scenarios],
        "scenarios": mappings,
        "empty_recent_fallback": {
            **_summary(empty_recent),
            "B0_surface_sha256": empty_recent.forecast.B0.sha256,
            "B0_R30_surface_sha256": empty_recent.forecast.B0_R30.sha256,
            "surfaces_exactly_equal": (
                empty_recent.forecast.B0.relative_intensity.tolist()
                == empty_recent.forecast.B0_R30.relative_intensity.tolist()
            ),
            "alarm_masks_exactly_equal": (
                empty_recent.forecast.B0_alarm.selected_cell_ids
                == empty_recent.forecast.B0_R30_alarm.selected_cell_ids
            ),
        },
        "zero_cluster_36_month_boundaries": {
            "observed_zero_with_due_mature_truth": {
                "scientific_condition": (
                    "at_least_one_due_guard_selected_30_day_exposure; all due truth "
                    "snapshots are mature_truth; total independent clusters equals zero"
                ),
                "review": observed_zero_review.as_mapping(),
            },
            "due_selected_truth_unavailable": {
                "scientific_condition": (
                    "at least one due guard-selected 30-day truth snapshot is unavailable"
                ),
                "review": integrity_pause_review,
            },
            "no_due_selected_exposure": {
                "scientific_condition": (
                    "no due guard-selected 30-day exposure exists at the terminal"
                ),
                "review": integrity_pause_review,
            },
        },
        "synthetic_schema_record_chain": {
            "scope": "schema_rehearsal_only_never_real_issue_authority",
            "record_count": len(rehearsal_chain),
            "record_types": [record["record_type"] for record in rehearsal_chain],
            "chain_head_sha256": rehearsal_chain[0]["content_sha256"],
            "chain_tail_sha256": rehearsal_chain[-1]["content_sha256"],
            "records": rehearsal_chain,
        },
        "scientific_value_review": {
            "category": "necessary_enabler",
            "direct_prediction_evidence_created": False,
            "meaning": (
                "Known-answer directions and frozen fairness logic are correct; "
                "real prospective performance remains unknown."
            ),
        },
    }
    payloads = {
        RESULT_NAME: _json_bytes(result),
        FORECAST_SVG_NAME: forecast_svg,
        FORECAST_HTML_NAME: forecast_html,
        SVG_NAME: replay_svg,
        HTML_NAME: replay_html,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "stage_id": "P1-0B",
        "synthetic_only": True,
        "canonical_result_sha256": _sha256(canonical_json_bytes(result)),
        "artifacts": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in sorted(payloads.items())
        },
    }
    payloads[MANIFEST_NAME] = _json_bytes(manifest)
    return payloads


def _write_or_check(
    payloads: Mapping[str, bytes],
    output_dir: Path,
    *,
    check: bool,
    refresh_candidate: bool,
) -> None:
    if check:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"artifact directory does not exist: {output_dir}")
        unexpected = sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file() and path.name not in payloads
        )
        if unexpected:
            raise ValueError(f"unexpected files in artifact directory: {unexpected}")
        for name, expected in payloads.items():
            path = output_dir / name
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError(f"artifact byte check failed: {name}")
        return
    if refresh_candidate:
        output_dir.mkdir(parents=True, exist_ok=True)
        unexpected = sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file() and path.name not in payloads
        )
        if unexpected:
            raise ValueError(f"refusing to refresh directory with unexpected files: {unexpected}")
        for name, payload in payloads.items():
            (output_dir / name).write_bytes(payload)
        return
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output_dir}")
    output_dir.mkdir(parents=True)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check and args.refresh_candidate:
            raise ValueError("--check and --refresh-candidate are mutually exclusive")
        payloads = build_artifacts()
        _write_or_check(
            payloads,
            args.output_dir,
            check=bool(args.check),
            refresh_candidate=bool(args.refresh_candidate),
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"P1-0B synthetic science check failed: {error}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "refreshed" if args.refresh_candidate else "wrote"
    print(f"{action} {len(payloads)} P1-0B pure-synthetic artifacts in {args.output_dir.resolve()}")
    for name, payload in sorted(payloads.items()):
        print(f"{_sha256(payload)}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
