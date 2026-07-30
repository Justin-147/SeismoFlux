from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from seismoflux.stage2s.governance import (
    Stage2SImportIsolationError,
    audit_stage2s_import_closure,
    forbidden_stage4_paths,
    verify_stage2s_import_closure_release,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_python(repository_root: Path, relative_path: str, source: str) -> Path:
    path = repository_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="\n")
    return path


def _root_module(repository_root: Path, source: str) -> Path:
    return _write_python(
        repository_root,
        "src/seismoflux/stage2s/root.py",
        "from __future__ import annotations\n\n" + source,
    )


def _forbidden_draft(repository_root: Path) -> Path:
    return _write_python(
        repository_root,
        "src/seismoflux/anomaly_increment/kde_dev_fit.py",
        "raise AssertionError('the forbidden draft must never be opened or imported')\n",
    )


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return completed.stdout.strip()


def _initialize_git_repository(repository_root: Path, *tracked_paths: Path) -> str:
    _git(repository_root, "init", "--quiet")
    _git(repository_root, "config", "user.name", "Stage2S Test")
    _git(repository_root, "config", "user.email", "stage2s-test@example.invalid")
    _git(repository_root, "config", "core.autocrlf", "false")
    _git(
        repository_root,
        "add",
        "--",
        *(path.relative_to(repository_root).as_posix() for path in tracked_paths),
    )
    _git(repository_root, "commit", "--quiet", "-m", "initial")
    return _git(repository_root, "rev-parse", "HEAD")


def test_direct_import_canary_fails_without_opening_forbidden_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root_module(
        tmp_path,
        "import seismoflux.anomaly_increment.kde_dev_fit\n",
    )
    forbidden = _forbidden_draft(tmp_path)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == forbidden:
            raise AssertionError("forbidden draft content was opened")
        return original_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    with pytest.raises(Stage2SImportIsolationError, match="kde_dev production module"):
        audit_stage2s_import_closure(tmp_path, root_paths=[root])


def test_from_import_alias_canary_is_rejected(tmp_path: Path) -> None:
    root = _root_module(
        tmp_path,
        "from seismoflux.anomaly_increment import kde_dev_fit as legacy_fit\n",
    )
    _forbidden_draft(tmp_path)

    with pytest.raises(Stage2SImportIsolationError, match="kde_dev production module"):
        audit_stage2s_import_closure(tmp_path, root_paths=[root])


def test_relative_import_canary_is_rejected(tmp_path: Path) -> None:
    root = _root_module(
        tmp_path,
        "from ..anomaly_increment import kde_dev_fit\n",
    )
    _forbidden_draft(tmp_path)

    with pytest.raises(Stage2SImportIsolationError, match="kde_dev production module"):
        audit_stage2s_import_closure(tmp_path, root_paths=[root])


def test_transitive_import_canary_is_rejected(tmp_path: Path) -> None:
    root = _root_module(tmp_path, "import seismoflux.safe_bridge\n")
    _write_python(
        tmp_path,
        "src/seismoflux/safe_bridge.py",
        "from seismoflux.anomaly_increment import kde_dev_fit\n",
    )
    _forbidden_draft(tmp_path)

    with pytest.raises(Stage2SImportIsolationError, match="kde_dev production module"):
        audit_stage2s_import_closure(tmp_path, root_paths=[root])


def test_nonliteral_dynamic_import_canary_fails_closed(tmp_path: Path) -> None:
    root = _root_module(
        tmp_path,
        "\n".join(
            (
                "import importlib",
                "",
                "MODULE_NAME = 'json'",
                "importlib.import_module(MODULE_NAME)",
                "",
            )
        ),
    )

    with pytest.raises(Stage2SImportIsolationError, match="non-literal dynamic import"):
        audit_stage2s_import_closure(tmp_path, root_paths=[root])


def test_literal_dynamic_forbidden_import_canary_is_rejected(tmp_path: Path) -> None:
    root = _root_module(
        tmp_path,
        "\n".join(
            (
                "from importlib import import_module as load_module",
                "",
                "load_module('seismoflux.anomaly_increment.kde_dev_fit')",
                "",
            )
        ),
    )
    _forbidden_draft(tmp_path)

    with pytest.raises(Stage2SImportIsolationError, match="kde_dev production module"):
        audit_stage2s_import_closure(tmp_path, root_paths=[root])


def test_file_based_dynamic_loader_is_unconditionally_rejected(tmp_path: Path) -> None:
    root = _root_module(
        tmp_path,
        "\n".join(
            (
                "from importlib.util import spec_from_file_location",
                "",
                "spec_from_file_location('safe_name', 'safe_path.py')",
                "",
            )
        ),
    )

    with pytest.raises(Stage2SImportIsolationError, match="file-based dynamic loading"):
        audit_stage2s_import_closure(tmp_path, root_paths=[root])


def test_filesystem_discovery_includes_untracked_safe_transitive_module(
    tmp_path: Path,
) -> None:
    root = _root_module(tmp_path, "import seismoflux.untracked_helper\n")
    _write_python(
        tmp_path,
        "src/seismoflux/untracked_helper.py",
        "VALUE = 147\n",
    )

    report = audit_stage2s_import_closure(tmp_path, root_paths=[root])

    assert report.root_modules == ("seismoflux.stage2s.root",)
    assert "seismoflux.untracked_helper" in report.visited_modules
    assert "src/seismoflux/untracked_helper.py" in report.visited_paths
    assert (
        dict(report.visited_path_sha256)["src/seismoflux/untracked_helper.py"]
        == hashlib.sha256(b"VALUE = 147\n").hexdigest()
    )


def test_release_guard_rejects_untracked_imported_helper(tmp_path: Path) -> None:
    root = _root_module(tmp_path, "import seismoflux.untracked_helper\n")
    helper = _write_python(
        tmp_path,
        "src/seismoflux/untracked_helper.py",
        "VALUE = 147\n",
    )
    code_commit = _initialize_git_repository(tmp_path, root)
    report = audit_stage2s_import_closure(tmp_path, root_paths=[root])

    assert helper.exists()
    with pytest.raises(Stage2SImportIsolationError, match="ls-files"):
        verify_stage2s_import_closure_release(
            tmp_path,
            report=report,
            code_commit=code_commit,
        )


def test_release_guard_rejects_tracked_worktree_change(tmp_path: Path) -> None:
    root = _root_module(tmp_path, "import seismoflux.helper\n")
    helper = _write_python(tmp_path, "src/seismoflux/helper.py", "VALUE = 1\n")
    code_commit = _initialize_git_repository(tmp_path, root, helper)
    helper.write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    report = audit_stage2s_import_closure(tmp_path, root_paths=[root])

    with pytest.raises(Stage2SImportIsolationError, match="modified"):
        verify_stage2s_import_closure_release(
            tmp_path,
            report=report,
            code_commit=code_commit,
        )


def test_release_guard_rejects_staged_import_change(tmp_path: Path) -> None:
    root = _root_module(tmp_path, "import seismoflux.helper\n")
    helper = _write_python(tmp_path, "src/seismoflux/helper.py", "VALUE = 1\n")
    code_commit = _initialize_git_repository(tmp_path, root, helper)
    helper.write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    _git(tmp_path, "add", "--", "src/seismoflux/helper.py")
    report = audit_stage2s_import_closure(tmp_path, root_paths=[root])

    with pytest.raises(Stage2SImportIsolationError, match="staged"):
        verify_stage2s_import_closure_release(
            tmp_path,
            report=report,
            code_commit=code_commit,
        )


def test_release_guard_binds_clean_closure_bytes_to_head_and_code_commit(
    tmp_path: Path,
) -> None:
    root = _root_module(tmp_path, "import seismoflux.helper\n")
    helper = _write_python(tmp_path, "src/seismoflux/helper.py", "VALUE = 1\n")
    code_commit = _initialize_git_repository(tmp_path, root, helper)
    report = audit_stage2s_import_closure(tmp_path, root_paths=[root])

    evidence = verify_stage2s_import_closure_release(
        tmp_path,
        report=report,
        code_commit=code_commit,
    )

    assert evidence.head_commit == code_commit
    assert evidence.code_commit == code_commit
    bindings = evidence.receipt_bindings()
    assert bindings["path_scoped_status_clean"] is True
    assert bindings["working_tree_equals_head_and_code_commit"] is True
    assert bindings["visited_path_sha256"] == dict(report.visited_path_sha256)


def test_commit_path_guard_rejects_both_forbidden_draft_families() -> None:
    assert forbidden_stage4_paths(
        [
            "src/seismoflux/stage2s/governance.py",
            "src/seismoflux/anomaly_increment/kde_dev.py",
            "src/seismoflux/anomaly_increment/kde_dev_fit.py",
            "tests/unit/test_stage4_kde_dev.py",
            "tests/unit/test_stage4_kde_dev_synthetic_chain.py",
        ]
    ) == (
        "src/seismoflux/anomaly_increment/kde_dev.py",
        "src/seismoflux/anomaly_increment/kde_dev_fit.py",
        "tests/unit/test_stage4_kde_dev.py",
        "tests/unit/test_stage4_kde_dev_synthetic_chain.py",
    )


def test_current_stage2s_first_party_transitive_closure_is_isolated() -> None:
    report = audit_stage2s_import_closure(ROOT)

    assert "seismoflux.stage2s.governance" in report.visited_modules
    assert report.root_modules
    assert not any("kde_dev" in module for module in report.visited_modules)
