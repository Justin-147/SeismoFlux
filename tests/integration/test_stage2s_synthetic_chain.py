from __future__ import annotations

import builtins
import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

import pytest

from seismoflux.stage2s.records import Stage2SSyntheticAcceptanceRecord
from seismoflux.stage2s.runner import run_stage2s_synthetic_acceptance
from seismoflux.stage2s.seals import Stage2SSealExists

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_PathValue: TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
_OpenPathValue: TypeAlias = int | _PathValue


class _ProcessedPathGuard:
    """Reject access only inside this repository's real data/processed tree."""

    def __init__(self, repository_root: Path) -> None:
        forbidden_root = self._normalize(repository_root / "data" / "processed")
        assert forbidden_root is not None
        self._forbidden_root = forbidden_root
        self.attempts: list[tuple[str, str]] = []

    @staticmethod
    def _normalize(value: object) -> str | None:
        if isinstance(value, int):
            return None
        if not isinstance(value, str | bytes | os.PathLike):
            return None
        try:
            raw_path = os.fspath(cast(_PathValue, value))
        except TypeError:
            return None
        if isinstance(raw_path, bytes):
            raw_path = os.fsdecode(raw_path)
        return os.path.normcase(os.path.abspath(os.path.normpath(raw_path)))

    def check(self, value: object, *, operation: str) -> None:
        candidate = self._normalize(value)
        if candidate is None:
            return
        try:
            inside_forbidden = (
                os.path.commonpath((candidate, self._forbidden_root)) == self._forbidden_root
            )
        except ValueError:
            inside_forbidden = False
        if inside_forbidden:
            self.attempts.append((operation, candidate))
            raise AssertionError(
                f"synthetic acceptance attempted {operation} under real data/processed: {candidate}"
            )


def _install_processed_path_guard(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository_root: Path,
) -> _ProcessedPathGuard:
    guard = _ProcessedPathGuard(repository_root)

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_os_stat = os.stat
    original_os_lstat = os.lstat
    original_os_listdir = os.listdir
    original_os_scandir = os.scandir
    original_path_open = Path.open
    original_path_stat = Path.stat

    def guarded_builtin_open(file: _OpenPathValue, *args: Any, **kwargs: Any) -> Any:
        guard.check(file, operation="builtins.open")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: _OpenPathValue, *args: Any, **kwargs: Any) -> Any:
        guard.check(file, operation="io.open")
        return original_io_open(file, *args, **kwargs)

    def guarded_os_open(path: _PathValue, *args: Any, **kwargs: Any) -> int:
        guard.check(path, operation="os.open")
        return original_os_open(path, *args, **kwargs)

    def guarded_os_stat(path: _OpenPathValue, *args: Any, **kwargs: Any) -> os.stat_result:
        guard.check(path, operation="os.stat")
        return original_os_stat(path, *args, **kwargs)

    def guarded_os_lstat(path: _PathValue, *args: Any, **kwargs: Any) -> os.stat_result:
        guard.check(path, operation="os.lstat")
        return original_os_lstat(path, *args, **kwargs)

    def guarded_os_listdir(
        path: int | str | os.PathLike[str] | None = ".",
    ) -> list[str]:
        guard.check(path, operation="os.listdir")
        return original_os_listdir(path)

    def guarded_os_scandir(path: object = ".") -> Any:
        guard.check(path, operation="os.scandir")
        return original_os_scandir(path)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        guard.check(path, operation="Path.open")
        return original_path_open(path, *args, **kwargs)

    def guarded_path_stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        guard.check(path, operation="Path.stat")
        return original_path_stat(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(os, "stat", guarded_os_stat)
    monkeypatch.setattr(os, "lstat", guarded_os_lstat)
    monkeypatch.setattr(os, "listdir", guarded_os_listdir)
    monkeypatch.setattr(os, "scandir", guarded_os_scandir)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "stat", guarded_path_stat)
    return guard


def _assert_sha256(value: object) -> str:
    assert isinstance(value, str)
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")
    return value


def _assert_sha256_sequence(
    value: object,
    *,
    expected_length: int,
) -> tuple[str, ...]:
    assert isinstance(value, tuple)
    assert len(value) == expected_length
    hashes = tuple(_assert_sha256(item) for item in value)
    assert len(set(hashes)) == expected_length
    return hashes


def _scratch_file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_complete_acceptance_record(
    record: Stage2SSyntheticAcceptanceRecord,
) -> None:
    assert record.real_input_path_open_count == 0
    assert record.real_target_byte_read_count == 0
    _assert_sha256(record.acceptance_sha256)

    whole_run = record.whole_run
    assert whole_run.mode == "synthetic_acceptance"
    _assert_sha256(whole_run.run_record_sha256)

    expected_cells = tuple(
        (fold_index, horizon) for fold_index in (1, 2, 3) for horizon in (7, 30, 90)
    )
    assert (
        tuple((cell.get("fold_index"), cell.get("horizon_days")) for cell in whole_run.cell_scores)
        == expected_cells
    )

    assert len(whole_run.bootstrap_rows) == 2000
    assert all(len(row) == 4 for row in whole_run.bootstrap_rows)

    regions = whole_run.regional_evidence.get("regions")
    assert isinstance(regions, tuple)
    assert len(regions) == 39
    zone_ids: list[str] = []
    for region in regions:
        assert isinstance(region, Mapping)
        zone_id = region.get("zone_id")
        assert isinstance(zone_id, str)
        zone_ids.append(zone_id)
    assert len(set(zone_ids)) == 39

    assert tuple(item.get("delay_days") for item in whole_run.latency_evidence) == (1, 7)
    assert tuple(item.get("fold_index") for item in whole_run.fold_fit_summaries) == (1, 2, 3)
    assert tuple(item.get("fold_index") for item in whole_run.issue_prediction_summaries) == (
        1,
        2,
        3,
    )

    seal_chain = whole_run.seal_chain
    _assert_sha256_sequence(
        seal_chain.get("fold_fit_receipt_sha256"),
        expected_length=3,
    )
    _assert_sha256_sequence(
        seal_chain.get("issue_prediction_seal_sha256"),
        expected_length=3,
    )
    _assert_sha256_sequence(
        seal_chain.get("fold_prediction_seal_sha256"),
        expected_length=3,
    )
    _assert_sha256(seal_chain.get("master_prediction_seal_sha256"))

    gate = whole_run.gate_evidence
    assert gate.get("science_value_category") == "necessary_enabler"
    assert gate.get("direct_prediction_improvement") == "none"
    assert (
        gate.get("evidence_scope") == "synthetic_engineering_acceptance_not_prediction_improvement"
    )
    expected_scope = (
        "sequence_associated_continuation_only"
        if gate.get("claim_limited") is True
        else (
            "broad_regional_gain_not_sequence_limited"
            if gate.get("status") == "passed_development_signal"
            else "no_sequence_interpretation_limit"
        )
    )
    assert gate.get("interpretation_scope") == expected_scope


def test_synthetic_chain_is_deterministic_target_free_and_o_excl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_a = tmp_path / "synthetic-a"
    scratch_b = tmp_path / "synthetic-b"
    guard = _install_processed_path_guard(
        monkeypatch,
        repository_root=REPOSITORY_ROOT,
    )

    first = run_stage2s_synthetic_acceptance(
        repository_root=REPOSITORY_ROOT,
        scratch_root=scratch_a,
    )
    second = run_stage2s_synthetic_acceptance(
        repository_root=REPOSITORY_ROOT,
        scratch_root=scratch_b,
    )

    _assert_complete_acceptance_record(first)
    _assert_complete_acceptance_record(second)
    assert first.acceptance_sha256 == second.acceptance_sha256
    assert first.whole_run.run_record_sha256 == second.whole_run.run_record_sha256
    assert first.as_mapping() == second.as_mapping()
    assert first.whole_run.to_canonical_bytes() == second.whole_run.to_canonical_bytes()
    assert guard.attempts == []

    before_retry = _scratch_file_snapshot(scratch_a)
    assert before_retry
    with pytest.raises(Stage2SSealExists):
        run_stage2s_synthetic_acceptance(
            repository_root=REPOSITORY_ROOT,
            scratch_root=scratch_a,
        )
    assert _scratch_file_snapshot(scratch_a) == before_retry
    assert guard.attempts == []
