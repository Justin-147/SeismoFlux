from __future__ import annotations

from dataclasses import asdict, fields
from typing import cast

import pytest

from seismoflux.features.anomaly.config import load_anomaly_history_config
from seismoflux.features.anomaly.synthetic_audit import (
    SyntheticPrefixAuditResult,
    run_synthetic_prefix_audit,
)


def test_all_32_synthetic_prefix_seeds_pass_all_seven_frozen_invariants() -> None:
    config = load_anomaly_history_config("configs/anomaly_history.yaml")

    first = run_synthetic_prefix_audit(config)
    second = run_synthetic_prefix_audit(config)

    assert first == second
    assert first == SyntheticPrefixAuditResult(
        passed=True,
        seed_count=32,
        invariant_count=7,
        check_count=224,
        failure_count=0,
    )


def test_result_contains_only_public_safe_aggregate_counts() -> None:
    result_fields = tuple(field.name for field in fields(SyntheticPrefixAuditResult))

    assert result_fields == (
        "passed",
        "seed_count",
        "invariant_count",
        "check_count",
        "failure_count",
    )
    assert set(
        asdict(
            SyntheticPrefixAuditResult(
                passed=True,
                seed_count=32,
                invariant_count=7,
                check_count=224,
                failure_count=0,
            )
        )
    ) == set(result_fields)


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    (
        ("generator", "different-generator"),
        ("seed_start_inclusive", 1),
        ("seed_stop_exclusive", 31),
        ("seed_count", 31),
        (
            "invariants",
            [
                "full_input_equals_available_prefix_scientific_payload",
                "future_mutations_do_not_change_prior_snapshot",
            ],
        ),
    ),
)
def test_synthetic_prefix_plan_rejects_any_frozen_field_drift(
    field_name: str,
    unsafe_value: object,
) -> None:
    config = load_anomaly_history_config("configs/anomaly_history.yaml").model_copy(deep=True)
    property_config = cast(
        dict[str, object],
        config.audit["synthetic_prefix_property"],
    )
    property_config[field_name] = unsafe_value

    with pytest.raises(ValueError, match="synthetic-prefix"):
        run_synthetic_prefix_audit(config)


def test_synthetic_prefix_plan_rejects_undocumented_fields() -> None:
    config = load_anomaly_history_config("configs/anomaly_history.yaml").model_copy(deep=True)
    property_config = cast(
        dict[str, object],
        config.audit["synthetic_prefix_property"],
    )
    property_config["extra"] = True

    with pytest.raises(ValueError, match="fields differ"):
        run_synthetic_prefix_audit(config)


def test_aggregate_result_rejects_internally_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="check count"):
        SyntheticPrefixAuditResult(
            passed=True,
            seed_count=32,
            invariant_count=7,
            check_count=223,
            failure_count=0,
        )

    with pytest.raises(ValueError, match="passed flag"):
        SyntheticPrefixAuditResult(
            passed=True,
            seed_count=32,
            invariant_count=7,
            check_count=224,
            failure_count=1,
        )
