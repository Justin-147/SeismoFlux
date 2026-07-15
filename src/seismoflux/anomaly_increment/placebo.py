"""Target-blind permutation helpers for the frozen stage-4 protocol.

The functions in this module only construct deterministic mappings and reduce
already-computed null statistics.  They never read targets, score models, or
touch the locked test partition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from seismoflux.anomaly_increment.preregistration import Stage4SeedContext

CoordinatePair: TypeAlias = tuple[float, float]
PermutationStatus: TypeAlias = Literal["passed", "evidence_insufficient"]
PERMUTATION_REPLICATIONS = 1_000
MAXIMUM_SCIENTIFIC_FAILURE_FRACTION = 0.01


class InfrastructureInterruption(RuntimeError):
    """A resumable interruption that must not be counted as a scientific failure."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_trimmed(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in items):
        raise ValueError(f"{label} must contain non-empty trimmed strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{label} must be unique")
    return items


@dataclass(frozen=True, slots=True)
class TimeBijection:
    """A complete recipient-issue to donor-issue bijection."""

    recipient_issue_ids: tuple[str, ...]
    donor_issue_ids: tuple[str, ...]
    mapping_sha256: str
    fixed_point_count: int

    def __post_init__(self) -> None:
        recipients = _unique_trimmed(self.recipient_issue_ids, label="recipient_issue_ids")
        donors = _unique_trimmed(self.donor_issue_ids, label="donor_issue_ids")
        if len(recipients) != len(donors) or set(recipients) != set(donors):
            raise ValueError("time mapping must be a complete bijection over one issue pool")
        if self.fixed_point_count != sum(
            recipient == donor for recipient, donor in zip(recipients, donors, strict=True)
        ):
            raise ValueError("fixed_point_count disagrees with the time mapping")
        if len(self.mapping_sha256) != 64:
            raise ValueError("mapping_sha256 must be a SHA-256")

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.recipient_issue_ids, self.donor_issue_ids, strict=True))


def build_time_bijection(
    recipient_issue_ids: Sequence[str],
    *,
    context: Stage4SeedContext,
) -> TimeBijection:
    """Permute one frozen fit or assessment issue pool with its typed RNG context."""

    if context.purpose != "time_permutation":
        raise ValueError("time bijection requires a time_permutation context")
    recipients = _unique_trimmed(recipient_issue_ids, label="recipient_issue_ids")
    order = context.generator().permutation(len(recipients))
    donors = tuple(recipients[int(index)] for index in order)
    pairs = tuple(zip(recipients, donors, strict=True))
    digest = _canonical_sha256(
        {
            "context": context.as_mapping(),
            "direction": "recipient_issue_to_donor_issue",
            "pairs": pairs,
        }
    )
    return TimeBijection(
        recipient_issue_ids=recipients,
        donor_issue_ids=donors,
        mapping_sha256=digest,
        fixed_point_count=sum(recipient == donor for recipient, donor in pairs),
    )


@dataclass(frozen=True, slots=True)
class SpaceBijection:
    """An entity to donor-coordinate bijection within one issue and stratum."""

    entity_state_ids: tuple[str, ...]
    donor_state_ids: tuple[str, ...]
    permuted_coordinate_pairs: tuple[CoordinatePair, ...]
    mapping_sha256: str
    fixed_point_count: int
    moved_entity_row_count: int
    no_effect: bool
    coordinate_multiset_verified: bool

    def __post_init__(self) -> None:
        states = _unique_trimmed(self.entity_state_ids, label="entity_state_ids")
        donors = _unique_trimmed(self.donor_state_ids, label="donor_state_ids")
        if len(states) != len(donors) or set(states) != set(donors):
            raise ValueError("space mapping must be a complete entity-coordinate bijection")
        if len(self.permuted_coordinate_pairs) != len(states):
            raise ValueError("one permuted coordinate pair is required per entity state")
        if self.fixed_point_count != sum(
            state_id == donor_id for state_id, donor_id in zip(states, donors, strict=True)
        ):
            raise ValueError("fixed_point_count disagrees with the space mapping")
        if not 0 <= self.moved_entity_row_count <= len(states):
            raise ValueError("moved_entity_row_count is outside the entity range")
        if self.no_effect != (self.moved_entity_row_count == 0):
            raise ValueError("no_effect disagrees with moved_entity_row_count")
        if not self.coordinate_multiset_verified:
            raise ValueError("space permutation must preserve the coordinate-pair multiset")


def _validated_coordinates(values: Sequence[CoordinatePair]) -> tuple[CoordinatePair, ...]:
    coordinates = tuple(values)
    if not coordinates:
        raise ValueError("coordinate_pairs must not be empty")
    for pair in coordinates:
        if len(pair) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in pair
        ):
            raise ValueError("coordinate_pairs must contain finite numeric pairs")
    return tuple((float(x), float(y)) for x, y in coordinates)


def build_space_bijection(
    entity_state_ids: Sequence[str],
    coordinate_pairs: Sequence[CoordinatePair],
    *,
    context: Stage4SeedContext,
) -> SpaceBijection:
    """Permute coordinate pairs in stable state-ID order inside one frozen stratum.

    Singleton and identical-coordinate strata are explicit identity mappings.  Raw
    coordinates are never included in the public mapping digest.
    """

    if context.purpose != "space_permutation":
        raise ValueError("space bijection requires a space_permutation context")
    states_input = _unique_trimmed(entity_state_ids, label="entity_state_ids")
    coordinates_input = _validated_coordinates(coordinate_pairs)
    if len(states_input) != len(coordinates_input):
        raise ValueError("entity_state_ids and coordinate_pairs must have equal length")

    stable_rows = sorted(
        zip(states_input, coordinates_input, strict=True), key=lambda item: item[0]
    )
    states = tuple(item[0] for item in stable_rows)
    coordinates = tuple(item[1] for item in stable_rows)
    if len(states) == 1 or len(set(coordinates)) == 1:
        donor_indices = tuple(range(len(states)))
    else:
        donor_indices = tuple(int(index) for index in context.generator().permutation(len(states)))
    donors = tuple(states[index] for index in donor_indices)
    permuted = tuple(coordinates[index] for index in donor_indices)
    moved = sum(
        original != replacement for original, replacement in zip(coordinates, permuted, strict=True)
    )
    original_hashes = tuple(_canonical_sha256(pair) for pair in coordinates)
    permuted_hashes = tuple(_canonical_sha256(pair) for pair in permuted)
    digest = _canonical_sha256(
        {
            "context": context.as_mapping(),
            "coordinate_pair_hashes": permuted_hashes,
            "direction": "recipient_entity_to_donor_coordinate_pair",
            "donor_state_ids": donors,
            "entity_state_ids": states,
        }
    )
    return SpaceBijection(
        entity_state_ids=states,
        donor_state_ids=donors,
        permuted_coordinate_pairs=permuted,
        mapping_sha256=digest,
        fixed_point_count=sum(
            state_id == donor_id for state_id, donor_id in zip(states, donors, strict=True)
        ),
        moved_entity_row_count=moved,
        no_effect=moved == 0,
        coordinate_multiset_verified=sorted(original_hashes) == sorted(permuted_hashes),
    )


@dataclass(frozen=True, slots=True)
class PermutationReplication:
    """One completed scientific permutation outcome."""

    replication_index: int
    statistic: float | None
    converged: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.replication_index, bool) or self.replication_index < 0:
            raise ValueError("replication_index must be non-negative")


@dataclass(frozen=True, slots=True)
class PermutationTestResult:
    observed_statistic: float
    null_statistics: tuple[float, ...]
    null_greater_or_equal_count: int
    scientific_failure_count: int
    scientific_failure_fraction: float
    monte_carlo_p_value: float
    denominator: int
    status: PermutationStatus


def reduce_permutation_test(
    observed_statistic: float,
    replications: Sequence[PermutationReplication],
) -> PermutationTestResult:
    """Apply the frozen +infinity failure rule and fixed Monte Carlo denominator.

    Missing or duplicated indices are infrastructure interruptions: callers must
    resume the same mapping instead of silently dropping or replacing it.
    """

    if not math.isfinite(observed_statistic):
        raise ValueError("observed_statistic must be finite")
    ordered = sorted(replications, key=lambda item: item.replication_index)
    expected_indices = tuple(range(PERMUTATION_REPLICATIONS))
    actual_indices = tuple(item.replication_index for item in ordered)
    if actual_indices != expected_indices:
        raise InfrastructureInterruption(
            "permutation indices are incomplete or duplicated; resume the frozen mappings"
        )

    null_statistics: list[float] = []
    failure_count = 0
    for replication in ordered:
        statistic = replication.statistic
        if not replication.converged or statistic is None or not math.isfinite(statistic):
            statistic = math.inf
            failure_count += 1
        null_statistics.append(statistic)
    greater_or_equal = sum(value >= observed_statistic for value in null_statistics)
    denominator = PERMUTATION_REPLICATIONS + 1
    failure_fraction = failure_count / PERMUTATION_REPLICATIONS
    status: PermutationStatus = (
        "evidence_insufficient"
        if failure_fraction > MAXIMUM_SCIENTIFIC_FAILURE_FRACTION
        else "passed"
    )
    return PermutationTestResult(
        observed_statistic=float(observed_statistic),
        null_statistics=tuple(null_statistics),
        null_greater_or_equal_count=greater_or_equal,
        scientific_failure_count=failure_count,
        scientific_failure_fraction=failure_fraction,
        monte_carlo_p_value=(1 + greater_or_equal) / denominator,
        denominator=denominator,
        status=status,
    )


__all__ = [
    "MAXIMUM_SCIENTIFIC_FAILURE_FRACTION",
    "PERMUTATION_REPLICATIONS",
    "CoordinatePair",
    "InfrastructureInterruption",
    "PermutationReplication",
    "PermutationStatus",
    "PermutationTestResult",
    "SpaceBijection",
    "TimeBijection",
    "build_space_bijection",
    "build_time_bijection",
    "reduce_permutation_test",
]
