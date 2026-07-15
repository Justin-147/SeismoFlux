from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TypeAlias, cast

import pytest

import seismoflux.anomaly_increment.immutable_file as immutable_file_module
import seismoflux.anomaly_increment.placebo_runtime as placebo_runtime_module
from seismoflux.anomaly_increment.compute import Stage4WorkerPlan
from seismoflux.anomaly_increment.placebo_runtime import (
    CHECKPOINT_EVERY_REPLICATIONS,
    HEARTBEAT_SECONDS,
    PlaceboCheckpointError,
    PlaceboRuntime,
    load_placebo_checkpoint,
)
from seismoflux.anomaly_increment.scoring_pipeline import (
    CompletedPlaceboReplication,
    PlaceboReplicateInput,
    PlaceboRequest,
    PlaceboSource,
)

_PathArgument: TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
_WINDOWS_REPARSE_POINT = 0x0400


def _install_lstat_fallback(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    reparse: bool,
    link_count: int | None = None,
) -> None:
    """Simulate an unsafe final entry when Windows cannot create real links."""

    original_lstat = os.lstat
    expected = os.path.normcase(os.path.abspath(os.fspath(path)))

    def simulated_lstat(
        candidate: _PathArgument,
        *,
        dir_fd: int | None = None,
    ) -> os.stat_result:
        observed = (
            original_lstat(candidate)
            if dir_fd is None
            else original_lstat(candidate, dir_fd=dir_fd)
        )
        actual = os.path.normcase(os.path.abspath(os.fsdecode(candidate)))
        if actual != expected:
            return observed
        attributes = int(getattr(observed, "st_file_attributes", 0))
        attributes = (
            attributes | _WINDOWS_REPARSE_POINT if reparse else attributes & ~_WINDOWS_REPARSE_POINT
        )
        return cast(
            os.stat_result,
            SimpleNamespace(
                st_mode=observed.st_mode,
                st_file_attributes=attributes,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_nlink=observed.st_nlink if link_count is None else link_count,
                st_size=observed.st_size,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns,
            ),
        )

    monkeypatch.setattr(os, "lstat", simulated_lstat)


def _worker_plan(workers: int = 2) -> Stage4WorkerPlan:
    return Stage4WorkerPlan(
        physical_cores=workers + 2,
        logical_processors=workers + 2,
        reserve_physical_cores=2,
        configured_max_workers=workers,
        effective_workers=workers,
        blas_threads_per_worker=1,
        nested_parallelism=False,
    )


def _request(observed: float = 0.25) -> PlaceboRequest:
    return PlaceboRequest(
        kind="time",
        evaluation_id="formal-validation",
        model_variant="dynamic",
        observed_statistic=observed,
        frozen_rate_head_sha256="a" * 64,
    )


def _source_factory(request: PlaceboRequest) -> PlaceboSource:
    def forbidden_build(_: int) -> PlaceboReplicateInput:
        raise AssertionError("runtime checkpoint callbacks must not eagerly build a replication")

    return PlaceboSource(
        source_id_sha256=hashlib.sha256(
            f"runtime-source:{request.kind}:{request.model_variant}".encode()
        ).hexdigest(),
        frozen_rate_head_sha256=request.frozen_rate_head_sha256,
        replicate_factory=forbidden_build,
        mapping_sha256_factory=lambda index: hashlib.sha256(
            f"mapping:{index}".encode()
        ).hexdigest(),
    )


def _result(index: int, *, converged: bool = True) -> CompletedPlaceboReplication:
    return CompletedPlaceboReplication(
        replication_index=index,
        mapping_sha256=hashlib.sha256(f"mapping:{index}".encode()).hexdigest(),
        statistic=float(index) if converged else None,
        converged=converged,
        scientific_failure_code=None if converged else "optimizer_nonconvergence",
    )


def _runtime(
    tmp_path: Path,
    *,
    binding: str = "b" * 64,
    max_in_flight: int = 2,
) -> PlaceboRuntime:
    return PlaceboRuntime(
        checkpoint_directory=tmp_path,
        execution_binding_id=binding,
        source_factory=_source_factory,
        worker_plan=_worker_plan(),
        max_in_flight=max_in_flight,
        backend="cpu_float64",
        gpu_device="synthetic-visible-gpu",
        memory_probe=lambda: 123_456,
        gpu_memory_probe=lambda: 654_321,
    )


def test_checkpoint_every_25_interruption_flush_and_exact_prefix_resume(
    tmp_path: Path,
) -> None:
    request = _request()
    runtime = _runtime(tmp_path)
    execution = runtime(request)
    path = runtime.checkpoint_path_for(request)
    assert execution.on_result is not None
    assert execution.on_interruption is not None

    initial = load_placebo_checkpoint(path)
    assert initial.status == "running" and initial.results == ()
    assert os.stat(path).st_nlink == 1
    for index in range(CHECKPOINT_EVERY_REPLICATIONS - 1):
        execution.on_result(request, _result(index))
    assert load_placebo_checkpoint(path).results == ()
    execution.on_result(request, _result(CHECKPOINT_EVERY_REPLICATIONS - 1))
    first_checkpoint = load_placebo_checkpoint(path)
    assert len(first_checkpoint.results) == CHECKPOINT_EVERY_REPLICATIONS
    assert os.stat(path).st_nlink == 1

    for index in range(CHECKPOINT_EVERY_REPLICATIONS, 30):
        execution.on_result(request, _result(index, converged=index != 27))
    execution.on_interruption(request, OSError("synthetic infrastructure stop"))
    interrupted = load_placebo_checkpoint(path)
    assert interrupted.status == "interrupted"
    assert interrupted.infrastructure_failure_code == "o_s_error"
    assert len(interrupted.results) == 30
    assert interrupted.results[27].scientific_failure_code == "optimizer_nonconvergence"
    assert interrupted.process_peak_working_set_bytes == 123_456
    assert interrupted.gpu_device == "synthetic-visible-gpu"
    assert interrupted.gpu_memory_used_bytes == 654_321
    assert interrupted.worker_plan.effective_workers == 2
    assert interrupted.max_in_flight == 2
    assert set(dict(interrupted.blas_environment).values()) == {"1"}
    assert interrupted.elapsed_seconds >= 0.0 and interrupted.cpu_seconds >= 0.0
    assert os.stat(path).st_nlink == 1

    resumed_execution = _runtime(tmp_path)(request)
    try:
        assert resumed_execution.completed_results == interrupted.results
        assert tuple(
            item.replication_index for item in resumed_execution.completed_results
        ) == tuple(range(30))
    finally:
        assert resumed_execution.on_interruption is not None
        resumed_execution.on_interruption(request, RuntimeError("test cleanup"))
    assert not list(tmp_path.glob(".*.tmp"))
    assert all(value == "1" for value in _worker_plan().blas_environment().values())
    assert all(os.environ.get(name) == "1" for name in _worker_plan().blas_environment())


@pytest.mark.parametrize("link_kind", ("hardlink", "symlink"))
def test_checkpoint_read_rejects_links_before_opening_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    target = tmp_path / "restricted-target.json"
    target.write_bytes(b'{"restricted":true}\n')
    checkpoint_path = tmp_path / "linked-checkpoint.json"
    try:
        if link_kind == "hardlink":
            os.link(target, checkpoint_path)
        else:
            os.symlink(target, checkpoint_path)
    except OSError:
        checkpoint_path.write_bytes(b"simulated-unsafe-link-entry")
        _install_lstat_fallback(
            monkeypatch,
            checkpoint_path,
            reparse=link_kind == "symlink",
            link_count=2 if link_kind == "hardlink" else None,
        )

    def forbidden_open(_path: Path) -> int:
        raise AssertionError("linked checkpoint target was opened")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    with pytest.raises(PlaceboCheckpointError, match="cannot read placebo checkpoint"):
        load_placebo_checkpoint(checkpoint_path)
    assert target.read_bytes() == b'{"restricted":true}\n'


@pytest.mark.parametrize("link_kind", ("hardlink", "symlink"))
def test_checkpoint_update_rejects_preexisting_unsafe_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    request = _request()
    safe_directory = tmp_path / "safe"
    runtime = _runtime(safe_directory)
    execution = runtime(request)
    assert execution.on_interruption is not None
    execution.on_interruption(request, RuntimeError("freeze checkpoint fixture"))
    checkpoint = load_placebo_checkpoint(runtime.checkpoint_path_for(request))

    target = tmp_path / "restricted-target.json"
    target.write_bytes(b'{"restricted":true}\n')
    checkpoint_path = tmp_path / "unsafe" / "checkpoint.json"
    checkpoint_path.parent.mkdir()
    try:
        if link_kind == "hardlink":
            os.link(target, checkpoint_path)
        else:
            os.symlink(target, checkpoint_path)
    except OSError:
        checkpoint_path.write_bytes(b"simulated-unsafe-link-entry")
        _install_lstat_fallback(
            monkeypatch,
            checkpoint_path,
            reparse=link_kind == "symlink",
            link_count=2 if link_kind == "hardlink" else None,
        )

    def forbidden_open(_path: Path) -> int:
        raise AssertionError("preexisting unsafe checkpoint link was opened")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    with pytest.raises(PlaceboCheckpointError, match="unsafe existing placebo checkpoint"):
        placebo_runtime_module._write_checkpoint_atomic(checkpoint_path, checkpoint)
    assert target.read_bytes() == b'{"restricted":true}\n'


def test_checkpoint_update_rejects_linked_parent_directory_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    safe_directory = tmp_path / "safe"
    runtime = _runtime(safe_directory)
    execution = runtime(request)
    assert execution.on_interruption is not None
    execution.on_interruption(request, RuntimeError("freeze checkpoint fixture"))
    checkpoint = load_placebo_checkpoint(runtime.checkpoint_path_for(request))

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError:
        linked_parent.mkdir()
        _install_lstat_fallback(
            monkeypatch,
            linked_parent,
            reparse=True,
        )

    with pytest.raises(PlaceboCheckpointError, match="unsafe existing placebo checkpoint"):
        placebo_runtime_module._write_checkpoint_atomic(
            linked_parent / "checkpoint.json",
            checkpoint,
        )
    assert not (real_parent / "checkpoint.json").exists()
    assert not (linked_parent / "checkpoint.json").exists()


def test_checkpoint_update_rejects_windows_reparse_parent_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    safe_directory = tmp_path / "safe"
    runtime = _runtime(safe_directory)
    execution = runtime(request)
    assert execution.on_interruption is not None
    execution.on_interruption(request, RuntimeError("freeze checkpoint fixture"))
    checkpoint = load_placebo_checkpoint(runtime.checkpoint_path_for(request))

    parent = tmp_path / "simulated-junction"
    parent.mkdir()
    parent_stat = os.lstat(parent)
    original = immutable_file_module._is_reparse_point

    def simulated_reparse(value: os.stat_result) -> bool:
        return (value.st_dev, value.st_ino) == (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ) or original(value)

    monkeypatch.setattr(
        immutable_file_module,
        "_is_reparse_point",
        simulated_reparse,
    )
    with pytest.raises(PlaceboCheckpointError, match="unsafe existing placebo checkpoint"):
        placebo_runtime_module._write_checkpoint_atomic(
            parent / "checkpoint.json",
            checkpoint,
        )
    assert not (parent / "checkpoint.json").exists()


def test_checkpoint_atomic_replace_verifies_installed_inode_after_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    safe_directory = tmp_path / "safe"
    runtime = _runtime(safe_directory)
    execution = runtime(request)
    assert execution.on_interruption is not None
    execution.on_interruption(request, RuntimeError("freeze checkpoint fixture"))
    safe_path = runtime.checkpoint_path_for(request)
    checkpoint = load_placebo_checkpoint(safe_path)

    raced_path = tmp_path / "raced" / "checkpoint.json"
    raced_path.parent.mkdir()
    attacker = tmp_path / "same-bytes-different-inode.json"
    attacker.write_bytes(safe_path.read_bytes())
    original_replace = os.replace

    def replace_then_swap(
        source: _PathArgument,
        destination: _PathArgument,
    ) -> None:
        original_replace(source, destination)
        original_replace(attacker, destination)

    monkeypatch.setattr(os, "replace", replace_then_swap)
    with pytest.raises(
        PlaceboCheckpointError,
        match="installed placebo checkpoint failed identity verification",
    ):
        placebo_runtime_module._write_checkpoint_atomic(raced_path, checkpoint)
    assert raced_path.read_bytes() == safe_path.read_bytes()
    assert os.stat(raced_path).st_nlink == 1


def test_checkpoint_tamper_request_binding_and_live_resume_fail_closed(
    tmp_path: Path,
) -> None:
    request = _request()
    runtime = _runtime(tmp_path)
    execution = runtime(request)
    path = runtime.checkpoint_path_for(request)
    assert execution.on_result is not None
    assert execution.on_interruption is not None

    with pytest.raises(PlaceboCheckpointError, match="live heartbeat"):
        _runtime(tmp_path)(request)
    execution.on_result(request, _result(0))
    execution.on_interruption(request, OSError("persist one result"))

    with pytest.raises(PlaceboCheckpointError, match="another placebo request"):
        _runtime(tmp_path)(_request(observed=0.5))
    with pytest.raises(PlaceboCheckpointError, match="another execution binding"):
        _runtime(tmp_path, binding="c" * 64)(request)
    with pytest.raises(PlaceboCheckpointError, match="another sealed max_in_flight"):
        _runtime(tmp_path, max_in_flight=1)(request)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["results"][0]["mapping_sha256"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PlaceboCheckpointError, match="hash or schema"):
        load_placebo_checkpoint(path)


def test_runtime_records_frozen_checkpoint_and_heartbeat_policy(tmp_path: Path) -> None:
    request = _request()
    runtime = _runtime(tmp_path)
    execution = runtime(request)
    try:
        checkpoint = load_placebo_checkpoint(runtime.checkpoint_path_for(request))
        mapping = checkpoint.as_mapping()
        assert mapping["checkpoint_every_replications"] == 25
        assert mapping["heartbeat_seconds"] == 30
        assert CHECKPOINT_EVERY_REPLICATIONS == 25
        assert HEARTBEAT_SECONDS == 30
        assert mapping["request_sha256"] == request.content_sha256
        assert mapping["execution_binding_id"] == "b" * 64
        assert mapping["source_id_sha256"] == _source_factory(request).source_id_sha256
        assert mapping["backend"] == "cpu_float64"
        assert mapping["max_in_flight"] == 2
    finally:
        assert execution.on_interruption is not None
        execution.on_interruption(request, RuntimeError("test cleanup"))
