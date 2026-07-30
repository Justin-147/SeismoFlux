from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

import pytest

from seismoflux.stage2s.records import (
    RecordMode,
    Stage2SRecordError,
    Stage2SSyntheticAcceptanceRecord,
    Stage2SWholeRunRecord,
    _validate_gate,
)


def _arguments() -> dict[str, object]:
    overall = {
        "S1_minus_S0:IG": 0.30,
        "S1_minus_S0:recall": 1.0,
        "S1_minus_SP:IG": 0.10,
        "S1_minus_SP:recall": 1.0,
    }
    residual = {
        "S1_minus_S0:IG": 0.05,
        "S1_minus_S0:recall": 0.0,
        "S1_minus_SP:IG": 0.02,
        "S1_minus_SP:recall": 0.0,
    }
    contributions = {
        "S1_minus_S0:IG": 0.25,
        "S1_minus_S0:recall": 1.0,
        "S1_minus_SP:IG": 0.08,
        "S1_minus_SP:recall": 1.0,
    }
    leave_out = dict(residual)
    component = {
        "component_id": "event-a",
        "event_ids": ["event-a"],
        "event_count": 1,
        "event_fraction": 1.0,
        "origin_time_span_days": 0.0,
        "max_pairwise_geodesic_distance_km": 0.0,
        "contributions": contributions,
        "model_hits": {
            "S0": {"raw": 0.0, "fraction": None},
            "S1": {"raw": 1.0, "fraction": 1.0},
            "SP": {"raw": 0.0, "fraction": None},
        },
        "information_gain": {
            "S1_minus_S0": {"raw": 0.25, "fraction": 0.25 / 0.30},
            "S1_minus_SP": {"raw": 0.08, "fraction": 0.8},
        },
    }
    return {
        "mode": "synthetic_acceptance",
        "identity": {"experiment_id": "synthetic", "code_sha256": "a" * 64},
        "input_receipts": {"synthetic_fixture_sha256": "b" * 64},
        "fold_fit_summaries": [
            {"fold_index": fold_index, "alpha_R": 1.0} for fold_index in (1, 2, 3)
        ],
        "issue_prediction_summaries": [
            {"fold_index": fold_index, "issue": f"issue-{fold_index}"} for fold_index in (1, 2, 3)
        ],
        "seal_chain": {"master_prediction_seal_sha256": "c" * 64},
        "cell_scores": [
            {
                "fold_index": fold_index,
                "horizon_days": horizon_days,
                "event_ids": ["event-a"],
                "hit_by_model": {
                    "S0": [False],
                    "S1": [True],
                    "SP": [False],
                },
                "IG": 0.1,
                "recall": 0.2,
            }
            for fold_index in (1, 2, 3)
            for horizon_days in (7, 30, 90)
        ],
        "bootstrap_summary": {
            "entropy": 147,
            "rows": 2000,
            "column_order": list(overall),
            "intervals": {
                key: {"point": value, "lower": value, "upper": value}
                for key, value in overall.items()
            },
        },
        "bootstrap_rows": [[0.1, 0.2, 0.3, 0.4] for _ in range(2000)],
        "regional_evidence": {"zones": 39, "passed": True},
        "sequence_evidence": {
            "component_count": 1,
            "event_resampling_unit_count": 1,
            "global_residual": residual,
            "primary_model_recall": {"S0": 0.0, "S1": 1.0, "SP": 0.0},
            "components": [component],
            "largest_count_component_id": "event-a",
            "largest_count_component": {
                **component,
                "leave_out": leave_out,
            },
            "largest_gain_component_id": {key: "event-a" for key in overall},
            "largest_gain_component": {
                key: {
                    "component_id": "event-a",
                    "raw_contribution": contributions[key],
                    "leave_out": leave_out[key],
                }
                for key in overall
            },
            "leave_out_residual": leave_out,
            "leave_largest_count_out": leave_out,
            "leave_largest_gain_out": leave_out,
            "claim_limited": True,
            "interpretation_limit": (
                "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
            ),
        },
        "descriptive_point_estimates": {
            "SP_minus_S0": {
                "information_gain": 0.20,
                "recall_gain": 0.0,
                "derivation": "S1_minus_S0_minus_S1_minus_SP",
                "inferential_status": "descriptive_point_estimate_only",
                "included_in_bootstrap_ci": False,
                "included_in_gate": False,
            }
        },
        "latency_evidence": [
            {"delay_days": 1, "IG": 0.1, "recall": 0.2},
            {"delay_days": 7, "IG": 0.1, "recall": 0.2},
        ],
        "gate_evidence": {
            "status": "passed_development_signal",
            "reasons": [],
            "overall_macros": overall,
            "claim_limited": True,
            "interpretation_limit": (
                "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
            ),
            "interpretation_scope": "sequence_associated_continuation_only",
        },
        "artifact_sha256_by_name": {"timeline.svg": "d" * 64},
    }


def _record(arguments: dict[str, object]) -> Stage2SWholeRunRecord:
    return Stage2SWholeRunRecord(
        mode=cast(RecordMode, arguments["mode"]),
        identity=cast(Mapping[str, object], arguments["identity"]),
        input_receipts=cast(Mapping[str, object], arguments["input_receipts"]),
        fold_fit_summaries=cast(
            Sequence[Mapping[str, object]],
            arguments["fold_fit_summaries"],
        ),
        issue_prediction_summaries=cast(
            Sequence[Mapping[str, object]],
            arguments["issue_prediction_summaries"],
        ),
        seal_chain=cast(Mapping[str, object], arguments["seal_chain"]),
        cell_scores=cast(
            Sequence[Mapping[str, object]],
            arguments["cell_scores"],
        ),
        bootstrap_summary=cast(
            Mapping[str, object],
            arguments["bootstrap_summary"],
        ),
        bootstrap_rows=cast(
            Sequence[Sequence[float]],
            arguments["bootstrap_rows"],
        ),
        regional_evidence=cast(
            Mapping[str, object],
            arguments["regional_evidence"],
        ),
        sequence_evidence=cast(
            Mapping[str, object],
            arguments["sequence_evidence"],
        ),
        descriptive_point_estimates=cast(
            Mapping[str, object],
            arguments["descriptive_point_estimates"],
        ),
        latency_evidence=cast(
            Sequence[Mapping[str, object]],
            arguments["latency_evidence"],
        ),
        gate_evidence=cast(Mapping[str, object], arguments["gate_evidence"]),
        artifact_sha256_by_name=cast(
            Mapping[str, object],
            arguments["artifact_sha256_by_name"],
        ),
    )


def test_whole_run_record_is_deterministic_and_immutable() -> None:
    arguments = _arguments()
    first = _record(arguments)
    second = _record(deepcopy(arguments))
    original_hash = first.run_record_sha256

    assert first.run_record_sha256 == second.run_record_sha256
    assert first.to_canonical_bytes().endswith(b"\n")
    cast_cells = arguments["cell_scores"]
    assert isinstance(cast_cells, list)
    cast_cells[0]["IG"] = 99.0
    assert first.run_record_sha256 == original_hash
    assert first.cell_scores[0]["IG"] == 0.1


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("cell_scores", lambda value: value[0].update({"IG": 0.11})),
        ("bootstrap_rows", lambda value: value[0].__setitem__(0, 0.11)),
        ("regional_evidence", lambda value: value.update({"passed": False})),
        (
            "sequence_evidence",
            lambda value: value["components"][0].update({"max_pairwise_geodesic_distance_km": 1.0}),
        ),
        (
            "descriptive_point_estimates",
            lambda value: value["SP_minus_S0"].update({"information_gain": 0.20000000001}),
        ),
        ("latency_evidence", lambda value: value[0].update({"IG": 0.11})),
        (
            "gate_evidence",
            lambda value: value.update({"status": "failed", "reasons": ["gate"]}),
        ),
    ],
)
def test_every_observed_evidence_family_changes_whole_record_hash(
    field: str,
    mutator: object,
) -> None:
    baseline_arguments = _arguments()
    changed_arguments = deepcopy(baseline_arguments)
    mutation = mutator
    assert callable(mutation)
    mutation(changed_arguments[field])

    assert (
        _record(changed_arguments).run_record_sha256
        != _record(baseline_arguments).run_record_sha256
    )


def test_whole_record_requires_all_bootstrap_rows() -> None:
    arguments = _arguments()
    rows = arguments["bootstrap_rows"]
    assert isinstance(rows, list)
    rows.pop()

    with pytest.raises(Stage2SRecordError, match="2,000"):
        _record(arguments)


@pytest.mark.parametrize(
    ("status", "claim_limited", "scope"),
    [
        ("passed_development_signal", False, "broad_regional_gain_not_sequence_limited"),
        ("failed", False, "no_sequence_interpretation_limit"),
        ("evidence_insufficient", False, "no_sequence_interpretation_limit"),
        ("invalid", False, "no_sequence_interpretation_limit"),
        ("failed", True, "sequence_associated_continuation_only"),
    ],
)
def test_gate_scope_requires_passed_status_for_broad_regional_wording(
    status: str,
    claim_limited: bool,
    scope: str,
) -> None:
    arguments = _arguments()
    gate = cast(dict[str, object], arguments["gate_evidence"])
    gate["status"] = status
    gate["claim_limited"] = claim_limited
    gate["interpretation_limit"] = (
        "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
        if claim_limited
        else "no_sequence_interpretation_limit"
    )
    gate["interpretation_scope"] = scope

    _validate_gate(gate)


def test_failed_nonlimited_gate_rejects_positive_broad_regional_wording() -> None:
    arguments = _arguments()
    gate = cast(dict[str, object], arguments["gate_evidence"])
    gate.update(
        {
            "status": "failed",
            "claim_limited": False,
            "interpretation_limit": "no_sequence_interpretation_limit",
            "interpretation_scope": "broad_regional_gain_not_sequence_limited",
        }
    )

    with pytest.raises(Stage2SRecordError, match="status and claim_limited"):
        _validate_gate(gate)


def test_synthetic_acceptance_binds_the_whole_record_without_real_reads() -> None:
    record = _record(_arguments())
    acceptance = Stage2SSyntheticAcceptanceRecord(whole_run=record)

    assert acceptance.real_input_path_open_count == 0
    assert acceptance.real_target_byte_read_count == 0
    assert acceptance.as_mapping()["whole_run_record_sha256"] == record.run_record_sha256
