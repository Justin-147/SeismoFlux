from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from seismoflux.p1_b0_r30.production import issue_schedule
from seismoflux.p1_b0_r30.runtime import AuthorizedIssueContext

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_p1_b0_r30_prospective.py"


def _load_runner() -> ModuleType:
    name = "seismoflux_p1_runner_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the P1 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _context() -> AuthorizedIssueContext:
    schedule = issue_schedule(datetime(2026, 9, 9, 16, tzinfo=UTC))
    return AuthorizedIssueContext(
        schedule=schedule,
        protocol_definition={"content_sha256": "a" * 64},
        authorization_record={
            "content_sha256": "b" * 64,
            "authorization_commit": "c" * 40,
        },
        code_commit="d" * 40,
    )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_runner_does_not_accept_operator_supplied_production_timestamps() -> None:
    runner = _load_runner()
    parser = runner._build_parser()  # type: ignore[attr-defined]
    option_strings = {option for action in parser._actions for option in action.option_strings}
    for subparser_action in parser._actions:
        choices = getattr(subparser_action, "choices", None)
        if isinstance(choices, dict):
            for subparser in choices.values():
                option_strings.update(
                    option for action in subparser._actions for option in action.option_strings
                )
    assert "--remote-verified-at-utc" not in option_strings
    assert "--publication-completed-at-utc" not in option_strings
    assert "--recorded-at-utc" not in option_strings


def test_operational_identity_rejects_dirty_critical_paths_before_remote_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "_verify_runtime_environment", lambda: None)
    monkeypatch.setattr(runner, "_verify_frozen_tags", lambda: None)
    monkeypatch.setattr(runner, "_rev_parse", lambda _: "d" * 40)
    monkeypatch.setattr(runner, "_is_ancestor", lambda *_: True)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""),
    )

    def fake_git(arguments: tuple[str, ...], *, text: bool = True) -> str | bytes:
        assert "ls-remote" not in arguments
        assert text is True
        return " M src/seismoflux/background/catalog.py"

    monkeypatch.setattr(runner, "_run_git", fake_git)
    with pytest.raises(ValueError, match="local changes"):
        runner._verify_operational_identity("d" * 40)


def test_remote_publication_requires_exact_canonical_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repository = tmp_path / "repository"
    issue_parent = repository / "outputs/prospective/p1_b0_r30"
    context = _context()
    issue_directory = issue_parent / context.schedule.issue_id
    issue_directory.mkdir(parents=True)
    remote_payloads: dict[str, bytes] = {}
    for index, filename in enumerate(cast(tuple[str, ...], runner.REMOTE_ISSUE_FILENAMES)):
        payload = f"artifact-{index}\r\n".encode()
        (issue_directory / filename).write_bytes(payload)
        relative = (issue_directory / filename).relative_to(repository).as_posix()
        remote_payloads[relative] = payload

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runner, "ISSUE_PARENT", issue_parent)
    monkeypatch.setattr(
        runner,
        "_verify_remote_ledger_anchor",
        lambda **_: ("e" * 40, datetime.now(UTC)),
    )

    def fake_git(arguments: tuple[str, ...], *, text: bool = True) -> str | bytes:
        assert arguments[:3] == ("ls-tree", "-r", "--name-only")
        assert text is True
        return "\n".join(remote_payloads)

    monkeypatch.setattr(runner, "_run_git", fake_git)
    monkeypatch.setattr(
        runner,
        "_remote_file_bytes",
        lambda _head, path: remote_payloads[path.as_posix()],
    )

    head, _ = runner._verify_remote_issue_publication(
        context=context,
        code_commit="d" * 40,
        issue_directory=issue_directory,
    )
    assert head == "e" * 40

    with pytest.raises(ValueError, match="canonical public issue directory"):
        runner._verify_remote_issue_publication(
            context=context,
            code_commit="d" * 40,
            issue_directory=tmp_path / "self-consistent-but-private",
        )

    remote_payloads.pop(next(iter(remote_payloads)))
    with pytest.raises(ValueError, match="exact canonical issue package"):
        runner._verify_remote_issue_publication(
            context=context,
            code_commit="d" * 40,
            issue_directory=issue_directory,
        )


def test_candidate_commit_is_exactly_one_ledger_change_on_verified_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet", "--initial-branch", "codex/stage2-etas-science-first")
    _git(repository, "config", "user.name", "SeismoFlux Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    ledger = repository / cast(Path, runner.LEDGER_RELATIVE_PATH)
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b'{"record_type":"RealIssueAuthorizationRecord"}\n')
    issue_directory = repository / "outputs/prospective/p1_b0_r30/p1-20260909T160000Z"
    issue_directory.mkdir(parents=True)
    for filename in cast(tuple[str, ...], runner.REMOTE_ISSUE_FILENAMES):
        (issue_directory / filename).write_bytes(filename.encode())
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "-m", "artifact parent")
    parent = _git(repository, "rev-parse", "HEAD")
    candidate_ledger = ledger.read_bytes() + b'{"record_type":"ForecastIssueRecord"}\n'
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)

    candidate = runner._build_ledger_only_commit(
        parent_commit=parent,
        candidate_ledger_bytes=candidate_ledger,
        issue_id="p1-20260909T160000Z",
        recorded_at_utc=datetime(2026, 9, 9, 15, 50, tzinfo=UTC),
    )

    assert _git(repository, "rev-parse", f"{candidate}^") == parent
    assert (
        _git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", parent, candidate)
        == "outputs/prospective/p1_b0_r30_records_v1.jsonl"
    )
    assert (
        subprocess.run(
            ["git", "show", f"{candidate}:outputs/prospective/p1_b0_r30_records_v1.jsonl"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        == candidate_ledger
    )
    assert _git(repository, "rev-parse", "HEAD") == parent
    assert ledger.read_bytes() != candidate_ledger


def test_failed_forecast_push_leaves_official_local_ledger_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repository = tmp_path / "repository"
    issue_directory = repository / "outputs/prospective/p1_b0_r30/p1-20260909T160000Z"
    issue_directory.mkdir(parents=True)
    ledger = repository / cast(Path, runner.LEDGER_RELATIVE_PATH)
    ledger.write_bytes(b"authorized-ledger\n")
    before = ledger.read_bytes()
    artifacts: dict[str, bytes] = {}
    for filename in cast(tuple[str, ...], runner.REMOTE_ISSUE_FILENAMES):
        payload = filename.encode()
        (issue_directory / filename).write_bytes(payload)
        artifacts[(issue_directory / filename).relative_to(repository).as_posix()] = payload
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        runner,
        "_run_git",
        lambda *_, **__: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git push")),
    )
    remote_heads = iter(("a" * 40, "a" * 40))
    monkeypatch.setattr(runner, "_remote_branch_head", lambda: next(remote_heads))

    def remote_bytes(head: str, path: Path) -> bytes:
        if path == cast(Path, runner.LEDGER_RELATIVE_PATH):
            assert head == "a" * 40
            return before
        return artifacts[path.as_posix()]

    monkeypatch.setattr(runner, "_remote_file_bytes", remote_bytes)

    with pytest.raises(subprocess.CalledProcessError):
        runner._push_forecast_commit_before_t(
            parent_commit="a" * 40,
            candidate_commit="b" * 40,
            previous_ledger_bytes=before,
            candidate_ledger_bytes=before + b"forecast\n",
            issue_directory=issue_directory,
            scheduled_issue_time_utc=datetime(2099, 1, 1, tzinfo=UTC),
        )
    assert ledger.read_bytes() == before


def test_concurrent_local_ledger_change_is_rejected_before_push(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repository = tmp_path / "repository"
    ledger = repository / cast(Path, runner.LEDGER_RELATIVE_PATH)
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"concurrently-changed\n")
    issue_directory = repository / "outputs/prospective/p1_b0_r30/p1-20260909T160000Z"
    issue_directory.mkdir(parents=True)
    push_called = False

    def fail_if_git_runs(*args: object, **kwargs: object) -> str:
        nonlocal push_called
        push_called = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runner, "_run_git", fail_if_git_runs)
    with pytest.raises(ValueError, match="local P1 ledger changed"):
        runner._push_forecast_commit_before_t(
            parent_commit="a" * 40,
            candidate_commit="b" * 40,
            previous_ledger_bytes=b"frozen-before-build\n",
            candidate_ledger_bytes=b"frozen-before-build\nforecast\n",
            issue_directory=issue_directory,
            scheduled_issue_time_utc=datetime(2099, 1, 1, tzinfo=UTC),
        )
    assert push_called is False


def test_remote_forecast_readback_finishing_at_t_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    repository = tmp_path / "repository"
    issue_directory = repository / "outputs/prospective/p1_b0_r30/p1-20260909T160000Z"
    issue_directory.mkdir(parents=True)
    remote_payloads: dict[str, bytes] = {}
    for filename in cast(tuple[str, ...], runner.REMOTE_ISSUE_FILENAMES):
        payload = filename.encode()
        (issue_directory / filename).write_bytes(payload)
        remote_payloads[(issue_directory / filename).relative_to(repository).as_posix()] = payload
    previous_ledger_bytes = b"authorized-ledger\n"
    ledger_path = repository / cast(Path, runner.LEDGER_RELATIVE_PATH)
    ledger_path.write_bytes(previous_ledger_bytes)
    ledger_bytes = previous_ledger_bytes + b"forecast\n"
    t = datetime(2026, 9, 9, 16, tzinfo=UTC)

    class SequenceDateTime:
        values = iter((t.replace(hour=15, minute=50), t.replace(hour=15, minute=51), t))

        @classmethod
        def now(cls, timezone: object) -> datetime:
            assert timezone is UTC
            return next(cls.values)

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runner, "datetime", SequenceDateTime)
    monkeypatch.setattr(runner, "_run_git", lambda *_, **__: "")
    remote_heads = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(runner, "_remote_branch_head", lambda: next(remote_heads))

    def remote_bytes(head: str, path: Path) -> bytes:
        if path == cast(Path, runner.LEDGER_RELATIVE_PATH):
            return previous_ledger_bytes if head == "a" * 40 else ledger_bytes
        return remote_payloads[path.as_posix()]

    monkeypatch.setattr(runner, "_remote_file_bytes", remote_bytes)

    with pytest.raises(ValueError, match="not observed on the public branch before T"):
        runner._push_forecast_commit_before_t(
            parent_commit="a" * 40,
            candidate_commit="b" * 40,
            previous_ledger_bytes=previous_ledger_bytes,
            candidate_ledger_bytes=ledger_bytes,
            issue_directory=issue_directory,
            scheduled_issue_time_utc=t,
        )


def test_seal_recovers_before_normal_next_issue_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    recovered_record = {"content_sha256": "f" * 64}
    monkeypatch.setattr(runner, "_verify_operational_identity", lambda _: None)
    monkeypatch.setattr(
        runner,
        "_recover_remotely_closed_forecast_if_present",
        lambda **_: (recovered_record, "e" * 40),
    )
    monkeypatch.setattr(
        runner,
        "load_authorized_issue_context",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("normal loader ran before recovery")),
    )
    outputs: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "_write_output", lambda value: outputs.append(dict(value)))

    result = runner._command_seal(
        SimpleNamespace(
            scheduled_issue_time_utc="2026-09-09T16:00:00Z",
            code_commit="d" * 40,
        )
    )
    assert result == 0
    assert outputs[0]["status"] == "forecast_record_recovered_from_public_branch"


def test_missed_command_never_appends_when_remote_recovery_is_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    appended = False

    def unresolved(**_: object) -> None:
        raise ValueError("post-T remote/local Forecast divergence requires manual review")

    def append_forbidden(*args: object, **kwargs: object) -> None:
        nonlocal appended
        appended = True
        raise AssertionError((args, kwargs))

    class AfterTDateTime(datetime):
        @classmethod
        def now(cls, timezone: object) -> datetime:
            assert timezone is UTC
            return cls(2026, 9, 9, 16, 1, tzinfo=UTC)

    monkeypatch.setattr(runner, "datetime", AfterTDateTime)
    monkeypatch.setattr(runner, "_recover_remotely_closed_forecast_if_present", unresolved)
    monkeypatch.setattr(runner, "append_new_p1_record", append_forbidden)
    with pytest.raises(ValueError, match="requires manual review"):
        runner._command_append_missed(
            SimpleNamespace(
                scheduled_issue_time_utc="2026-09-09T16:00:00Z",
                reason="forecast_not_frozen_before_T",
            )
        )
    assert appended is False
