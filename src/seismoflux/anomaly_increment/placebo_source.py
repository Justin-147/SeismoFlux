"""Deterministic lazy sources for the frozen stage-4 time and space placebos.

The source universe is target-blind.  A caller supplies a pure in-memory scope
assembler only after the formal evaluation context exists; this module itself has
no path, catalogue, target, score, or locked-test loading capability.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, cast

import pyarrow as pa

from seismoflux.anomaly_increment.contracts import canonical_mapping_sha256
from seismoflux.anomaly_increment.placebo import (
    SpaceBijection,
    TimeBijection,
    build_space_bijection,
    build_time_bijection,
)
from seismoflux.anomaly_increment.placebo_features import (
    SpaceStratumKey,
    rebuild_space_placebo_features,
    rebuild_time_placebo_features,
)
from seismoflux.anomaly_increment.preregistration import Stage4SeedContext
from seismoflux.anomaly_increment.scoring_pipeline import (
    EvaluationScope,
    PlaceboReplicateInput,
    PlaceboRequest,
    PlaceboSource,
)
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot

EvaluationId = Literal["formal-validation"]
PartitionRole = Literal["fit", "assessment"]


class PlaceboScopeAssembler(Protocol):
    """Convert rebuilt recipient tables into the already-frozen formal scope."""

    def __call__(self, issue_tables: Mapping[str, pa.Table]) -> EvaluationScope: ...


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identifiers(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    output = tuple(values)
    if (
        not output
        or len(output) != len(set(output))
        or any(not value or value != value.strip() for value in output)
    ):
        raise ValueError(f"{label} must contain unique non-empty trimmed identifiers")
    return output


@dataclass(frozen=True, slots=True)
class _SpaceGroup:
    issue_id: str
    construction_stratum_id: str
    partition_role: PartitionRole
    entity_state_ids: tuple[str, ...]
    coordinate_pairs: tuple[tuple[float, float], ...]

    def build(self, *, replication_index: int, frozen_input_seal_sha256: str) -> SpaceBijection:
        return build_space_bijection(
            self.entity_state_ids,
            self.coordinate_pairs,
            context=Stage4SeedContext(
                purpose="space_permutation",
                evaluation_id="formal-validation",
                partition_role=self.partition_role,
                replicate_index=replication_index,
                frozen_input_seal_sha256=frozen_input_seal_sha256,
                issue_id=self.issue_id,
                construction_stratum_id=self.construction_stratum_id,
            ),
        )


def _space_groups(
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot],
    *,
    fit_issue_ids: tuple[str, ...],
    assessment_issue_ids: tuple[str, ...],
    construction_stratum_by_state_id: Mapping[str, str],
) -> tuple[_SpaceGroup, ...]:
    fit = set(fit_issue_ids)
    assessment = set(assessment_issue_ids)
    eligible_state_ids: set[str] = set()
    grouped: dict[SpaceStratumKey, list[tuple[str, tuple[float, float]]]] = {}
    for issue_id, snapshot in sorted(
        snapshots_by_issue_id.items(),
        key=lambda item: item[1].issue_index,
    ):
        for state in snapshot.entities:
            if not state.spatial_eligible:
                continue
            if (
                state.state_id in eligible_state_ids
                or state.longitude is None
                or state.latitude is None
                or not math.isfinite(state.longitude)
                or not math.isfinite(state.latitude)
            ):
                raise ValueError("eligible placebo entity states must be unique and finite")
            eligible_state_ids.add(state.state_id)
            stratum = construction_stratum_by_state_id.get(state.state_id)
            if not isinstance(stratum, str) or not stratum or stratum != stratum.strip():
                raise ValueError(
                    "every eligible entity state needs one frozen construction stratum"
                )
            grouped.setdefault((issue_id, stratum), []).append(
                (state.state_id, (float(state.longitude), float(state.latitude)))
            )
    if set(construction_stratum_by_state_id) != eligible_state_ids:
        raise ValueError("construction strata must cover exactly the eligible entity states")

    groups: list[_SpaceGroup] = []
    for (issue_id, stratum), rows in sorted(grouped.items()):
        ordered = tuple(sorted(rows, key=lambda item: item[0]))
        role: PartitionRole
        if issue_id in fit:
            role = "fit"
        elif issue_id in assessment:
            role = "assessment"
        else:  # pragma: no cover - universe validation makes this unreachable
            raise AssertionError("space group escaped the frozen issue pools")
        groups.append(
            _SpaceGroup(
                issue_id=issue_id,
                construction_stratum_id=stratum,
                partition_role=role,
                entity_state_ids=tuple(item[0] for item in ordered),
                coordinate_pairs=tuple(item[1] for item in ordered),
            )
        )
    if not groups:
        raise ValueError("space placebo requires at least one eligible issue/stratum group")
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class PlaceboSourceUniverse:
    """Immutable score-blind feature universe shared by all paired mappings."""

    fit_issue_ids: tuple[str, ...]
    assessment_issue_ids: tuple[str, ...]
    issue_tables: Mapping[str, pa.Table]
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot]
    query_grid: Stage3QueryGrid
    construction_stratum_by_state_id: Mapping[str, str]
    frozen_input_seal_sha256: str
    source_input_sha256: str
    assembly_context_sha256: str
    scope_assembler: PlaceboScopeAssembler = field(repr=False, compare=False)
    evaluation_id: EvaluationId = "formal-validation"
    query_chunk_size: int = 256
    _groups: tuple[_SpaceGroup, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.evaluation_id != "formal-validation":
            raise ValueError("stage-4 G2 placebos are restricted to formal-validation")
        fit = _identifiers(self.fit_issue_ids, label="fit_issue_ids")
        assessment = _identifiers(self.assessment_issue_ids, label="assessment_issue_ids")
        if set(fit) & set(assessment):
            raise ValueError("placebo fit and assessment issue pools must be disjoint")
        tables = dict(self.issue_tables)
        snapshots = dict(self.snapshots_by_issue_id)
        expected = set(fit) | set(assessment)
        if set(tables) != expected or set(snapshots) != expected:
            raise ValueError("placebo tables and snapshots must exactly cover both issue pools")
        if any(not isinstance(table, pa.Table) or table.num_rows <= 0 for table in tables.values()):
            raise TypeError("placebo issue tables must be non-empty Arrow tables")
        ordered_snapshots = tuple(sorted(snapshots.items(), key=lambda item: item[1].issue_index))
        if tuple(item[1].issue_index for item in ordered_snapshots) != tuple(
            range(len(ordered_snapshots))
        ):
            raise ValueError("placebo snapshots must retain contiguous causal issue indices")
        if tuple(item[0] for item in ordered_snapshots) != (*fit, *assessment):
            raise ValueError("placebo issue pools must follow the causal snapshot order")
        _sha256(self.frozen_input_seal_sha256, label="frozen_input_seal_sha256")
        _sha256(self.source_input_sha256, label="source_input_sha256")
        _sha256(self.assembly_context_sha256, label="assembly_context_sha256")
        if not callable(self.scope_assembler):
            raise TypeError("scope_assembler must be callable")
        if (
            not isinstance(self.query_chunk_size, int)
            or isinstance(self.query_chunk_size, bool)
            or self.query_chunk_size <= 0
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        strata = dict(self.construction_stratum_by_state_id)
        groups = _space_groups(
            snapshots,
            fit_issue_ids=fit,
            assessment_issue_ids=assessment,
            construction_stratum_by_state_id=strata,
        )
        object.__setattr__(self, "fit_issue_ids", fit)
        object.__setattr__(self, "assessment_issue_ids", assessment)
        object.__setattr__(self, "issue_tables", MappingProxyType(tables))
        object.__setattr__(self, "snapshots_by_issue_id", MappingProxyType(snapshots))
        object.__setattr__(
            self,
            "construction_stratum_by_state_id",
            MappingProxyType(strata),
        )
        object.__setattr__(self, "_groups", groups)

    @property
    def source_binding_sha256(self) -> str:
        return canonical_mapping_sha256(
            {
                "assembly_context_sha256": self.assembly_context_sha256,
                "assessment_issue_ids": list(self.assessment_issue_ids),
                "evaluation_id": self.evaluation_id,
                "fit_issue_ids": list(self.fit_issue_ids),
                "frozen_input_seal_sha256": self.frozen_input_seal_sha256,
                "query_chunk_size": self.query_chunk_size,
                "source_input_sha256": self.source_input_sha256,
                "space_group_count": len(self._groups),
            }
        )

    def _assemble(
        self,
        rebuilt: Mapping[str, pa.Table],
        *,
        replication_index: int,
        mapping_sha256: str,
    ) -> PlaceboReplicateInput:
        scope = self.scope_assembler(rebuilt)
        if not isinstance(scope, EvaluationScope) or scope.fit.evaluation_id != self.evaluation_id:
            raise TypeError("placebo scope assembler returned another evaluation scope")
        return PlaceboReplicateInput(
            replication_index=replication_index,
            mapping_sha256=mapping_sha256,
            fit_scope=scope.fit,
            exposures=scope.exposures,
        )

    def _time_mapping(
        self,
        replication_index: int,
    ) -> tuple[TimeBijection, TimeBijection, str]:
        fit_mapping = build_time_bijection(
            self.fit_issue_ids,
            context=Stage4SeedContext(
                purpose="time_permutation",
                evaluation_id=self.evaluation_id,
                partition_role="fit",
                replicate_index=replication_index,
                frozen_input_seal_sha256=self.frozen_input_seal_sha256,
            ),
        )
        assessment_mapping = build_time_bijection(
            self.assessment_issue_ids,
            context=Stage4SeedContext(
                purpose="time_permutation",
                evaluation_id=self.evaluation_id,
                partition_role="assessment",
                replicate_index=replication_index,
                frozen_input_seal_sha256=self.frozen_input_seal_sha256,
            ),
        )
        mapping_sha256 = canonical_mapping_sha256(
            {
                "assessment_mapping_sha256": assessment_mapping.mapping_sha256,
                "evaluation_id": self.evaluation_id,
                "fit_mapping_sha256": fit_mapping.mapping_sha256,
                "kind": "time",
                "replication_index": replication_index,
                "source_binding_sha256": self.source_binding_sha256,
            }
        )
        return fit_mapping, assessment_mapping, mapping_sha256

    def _time_mapping_sha256(self, replication_index: int) -> str:
        return self._time_mapping(replication_index)[2]

    def _time_replicate(self, replication_index: int) -> PlaceboReplicateInput:
        fit_mapping, assessment_mapping, mapping_sha256 = self._time_mapping(replication_index)
        rebuilt = rebuild_time_placebo_features(
            self.issue_tables,
            fit_bijection=fit_mapping,
            assessment_bijection=assessment_mapping,
        )
        return self._assemble(
            rebuilt,
            replication_index=replication_index,
            mapping_sha256=mapping_sha256,
        )

    def _space_mapping(
        self,
        replication_index: int,
    ) -> tuple[dict[SpaceStratumKey, SpaceBijection], str]:
        mappings: dict[SpaceStratumKey, SpaceBijection] = {}
        audits: list[dict[str, object]] = []
        for group in self._groups:
            mapping = group.build(
                replication_index=replication_index,
                frozen_input_seal_sha256=self.frozen_input_seal_sha256,
            )
            key = (group.issue_id, group.construction_stratum_id)
            mappings[key] = mapping
            audits.append(
                {
                    "construction_stratum_id": group.construction_stratum_id,
                    "fixed_point_count": mapping.fixed_point_count,
                    "issue_id": group.issue_id,
                    "mapping_sha256": mapping.mapping_sha256,
                    "moved_entity_row_count": mapping.moved_entity_row_count,
                    "no_effect": mapping.no_effect,
                    "partition_role": group.partition_role,
                }
            )
        mapping_sha256 = canonical_mapping_sha256(
            {
                "evaluation_id": self.evaluation_id,
                "group_audits": audits,
                "kind": "space",
                "replication_index": replication_index,
                "source_binding_sha256": self.source_binding_sha256,
            }
        )
        return mappings, mapping_sha256

    def _space_mapping_sha256(self, replication_index: int) -> str:
        return self._space_mapping(replication_index)[1]

    def _space_replicate(self, replication_index: int) -> PlaceboReplicateInput:
        mappings, mapping_sha256 = self._space_mapping(replication_index)
        rebuilt = rebuild_space_placebo_features(
            self.issue_tables,
            self.snapshots_by_issue_id,
            self.query_grid,
            verified_construction_stratum_by_state_id=(self.construction_stratum_by_state_id),
            bijections_by_issue_stratum=mappings,
            query_chunk_size=self.query_chunk_size,
        )
        return self._assemble(
            rebuilt,
            replication_index=replication_index,
            mapping_sha256=mapping_sha256,
        )


def build_placebo_source(
    universe: PlaceboSourceUniverse,
    request: PlaceboRequest,
) -> PlaceboSource:
    """Build one lazy paired mapping source for a validated pipeline request."""

    if not isinstance(universe, PlaceboSourceUniverse):
        raise TypeError("universe must be PlaceboSourceUniverse")
    if not isinstance(request, PlaceboRequest):
        raise TypeError("request must be PlaceboRequest")
    if request.evaluation_id != universe.evaluation_id:
        raise ValueError("placebo request crossed the frozen formal evaluation")
    replicate_factory: Callable[[int], PlaceboReplicateInput]
    mapping_sha256_factory: Callable[[int], str]
    if request.kind == "time":
        replicate_factory = universe._time_replicate
        mapping_sha256_factory = universe._time_mapping_sha256
    elif request.kind == "space":
        replicate_factory = universe._space_replicate
        mapping_sha256_factory = universe._space_mapping_sha256
    else:  # pragma: no cover - PlaceboRequest validates the literal
        raise AssertionError("unknown placebo kind")
    source_id = canonical_mapping_sha256(
        {
            "evaluation_id": universe.evaluation_id,
            "kind": request.kind,
            "source_binding_sha256": universe.source_binding_sha256,
        }
    )
    return PlaceboSource(
        source_id_sha256=source_id,
        frozen_rate_head_sha256=request.frozen_rate_head_sha256,
        mapping_sha256_factory=mapping_sha256_factory,
        replicate_factory=cast(Callable[[int], PlaceboReplicateInput], replicate_factory),
    )


__all__ = [
    "PlaceboScopeAssembler",
    "PlaceboSourceUniverse",
    "build_placebo_source",
]
