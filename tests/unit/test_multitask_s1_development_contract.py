from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from seismoflux.multitask_s1.development_contract import (
    DevelopmentContractError,
    load_development_contract,
    validate_development_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "multitask_s1_development_contract.yaml"


def _mutable_contract() -> dict[str, Any]:
    contract, _ = load_development_contract(CONTRACT_PATH)
    return cast(dict[str, Any], copy.deepcopy(contract))


def test_frozen_contract_loads_and_verifies_score_blind_source_identities() -> None:
    contract, summary = load_development_contract(CONTRACT_PATH, project_root=PROJECT_ROOT)
    assert contract["score_blind"] is True
    assert contract["model_scores_read"] is False
    assert summary.fold_count == 4
    assert summary.inner_block_count == 12
    assert summary.waterlevel_row_count == 12
    assert summary.earliest_inner_start == datetime(1985, 1, 1, tzinfo=timezone(timedelta(hours=8)))
    assert summary.latest_inner_end == datetime(2014, 12, 2, tzinfo=timezone(timedelta(hours=8)))
    assert summary.m6_plus_7d_zero_anchor_block_count == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("main_catalog_delay_hours", 0),
        ("parameter_label_embargo_days", 29),
        ("inner_target_rule", "issue<inner_end"),
        ("fit_label_rule", "fit_issue<inner_start"),
    ],
)
def test_causal_boundaries_fail_closed(key: str, value: object) -> None:
    contract = _mutable_contract()
    causal = cast(dict[str, Any], contract["causal_boundaries"])
    causal[key] = value
    with pytest.raises(DevelopmentContractError, match="causal boundary changed"):
        validate_development_contract(contract)


def test_holdout_audit_and_locked_test_cannot_be_enabled() -> None:
    for key in ("holdout_enabled", "audit_enabled", "locked_test_enabled"):
        contract = _mutable_contract()
        scope = cast(dict[str, Any], contract["execution_scope"])
        scope[key] = True
        with pytest.raises(DevelopmentContractError, match=key):
            validate_development_contract(contract)


def test_inner_calendar_and_posthoc_merge_cannot_change() -> None:
    contract = _mutable_contract()
    folds = cast(list[dict[str, Any]], contract["outer_folds"])
    folds[0]["inner_blocks"][2]["end"] = "1999-12-03T00:00:00+08:00"
    with pytest.raises(DevelopmentContractError, match="inner calendar changed"):
        validate_development_contract(contract)

    contract = _mutable_contract()
    fallbacks = cast(dict[str, Any], contract["selection_fallbacks"])
    fallbacks["posthoc_block_merge"] = True
    with pytest.raises(DevelopmentContractError, match="fallback semantics"):
        validate_development_contract(contract)


def test_entire_fold_calendar_cannot_be_shifted_together() -> None:
    contract = _mutable_contract()
    folds = cast(list[dict[str, Any]], contract["outer_folds"])
    fold = folds[0]
    fold["outer_start"] = "2001-01-01T00:00:00+08:00"
    fold["outer_end"] = "2006-01-01T00:00:00+08:00"
    shifted_bounds = (
        ("1986-01-01T00:00:00+08:00", "1991-01-01T00:00:00+08:00"),
        ("1991-01-01T00:00:00+08:00", "1996-01-01T00:00:00+08:00"),
        ("1996-01-01T00:00:00+08:00", "2000-12-02T00:00:00+08:00"),
    )
    for block, (start, end) in zip(fold["inner_blocks"], shifted_bounds, strict=True):
        block["start"] = start
        block["end"] = end
    with pytest.raises(DevelopmentContractError, match="outer calendar changed"):
        validate_development_contract(contract)


@pytest.mark.parametrize("mutation", ["missing", "extra", "path", "hash"])
def test_source_identity_set_and_values_are_exactly_frozen(mutation: str) -> None:
    contract = _mutable_contract()
    sources = cast(dict[str, Any], contract["source_identities"])
    if mutation == "missing":
        sources.pop("catalog_sample_ledger")
    elif mutation == "extra":
        sources["unregistered"] = {
            "path": "outputs/unregistered.json",
            "sha256": "0" * 64,
        }
    elif mutation == "path":
        sources["multitask_s0_config"]["path"] = "../multitask_s0.yaml"
    else:
        sources["multitask_s0_config"]["sha256"] = "0" * 64
    with pytest.raises(DevelopmentContractError, match="source identity"):
        validate_development_contract(contract)


def test_sparse_fallback_text_and_waterlevel_values_cannot_change() -> None:
    contract = _mutable_contract()
    fallbacks = cast(dict[str, Any], contract["selection_fallbacks"])
    fallbacks["zero_anchor_block"] = "merge_with_neighbor"
    with pytest.raises(DevelopmentContractError, match="fallback semantics"):
        validate_development_contract(contract)

    contract = _mutable_contract()
    rows = cast(list[dict[str, Any]], contract["inner_block_waterlevels"])
    cast(list[int], rows[0]["m5_6_events"])[0] += 1
    with pytest.raises(DevelopmentContractError, match="water-level values"):
        validate_development_contract(contract)


def test_sparse_m6_7d_zero_block_is_retained_and_cannot_tune_alone() -> None:
    contract = _mutable_contract()
    rows = cast(list[dict[str, Any]], contract["inner_block_waterlevels"])
    zero_rows = [row for row in rows if cast(list[int], row["m6_plus_anchors"])[0] == 0]
    assert [(row["fold"], row["block"]) for row in zero_rows] == [("C_DEV_2015_2019", "I3")]
    fallbacks = cast(Mapping[str, Any], contract["selection_fallbacks"])
    assert fallbacks["m6_plus_7d_independent_hyperparameter_selection"] is False

    cast(list[int], zero_rows[0]["m6_plus_anchors"])[0] = 1
    cast(list[int], zero_rows[0]["m6_plus_events"])[0] = 1
    with pytest.raises(DevelopmentContractError, match="water-level values"):
        validate_development_contract(contract)

    contract = _mutable_contract()
    rows = cast(list[dict[str, Any]], contract["inner_block_waterlevels"])
    first = rows[0]
    last = rows[-1]
    cast(list[int], first["m6_plus_events"])[0] = 0
    cast(list[int], first["m6_plus_anchors"])[0] = 0
    cast(list[int], last["m6_plus_events"])[0] = 1
    cast(list[int], last["m6_plus_anchors"])[0] = 1
    with pytest.raises(DevelopmentContractError, match="water-level values"):
        validate_development_contract(contract)
