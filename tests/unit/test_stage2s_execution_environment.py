from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from seismoflux.stage2s.execution_environment import (
    THREAD_LIMIT_ENVIRONMENT_VARIABLES,
    Stage2SExecutionEnvironmentError,
    prepare_formal_execution_environment,
)

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/run_stage2s_once.py"


def test_prepare_overrides_external_thread_counts_and_reserves_two_cores() -> None:
    environment = {name: "64" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES}

    evidence = prepare_formal_execution_environment(
        environ=environment,
        loaded_module_names=(),
        physical_core_probe=lambda: 4,
    )

    assert environment == {name: "1" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES}
    assert evidence.worker_count == 1
    assert evidence.physical_core_count == 4
    assert evidence.reserved_physical_core_count == 2
    assert evidence.receipt_bindings()["available_physical_cores_after_reservation"] == 2
    assert evidence.receipt_bindings()["thread_environment"] == environment


@pytest.mark.parametrize("observed", [None, 0, 1, 2, True])
def test_prepare_fails_closed_without_three_verified_physical_cores(
    observed: int | None,
) -> None:
    environment: dict[str, str] = {}

    with pytest.raises(Stage2SExecutionEnvironmentError, match="three verified"):
        prepare_formal_execution_environment(
            environ=environment,
            loaded_module_names=(),
            physical_core_probe=lambda: observed,
        )

    assert environment == {name: "1" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES}


def test_prepare_rejects_numeric_modules_loaded_before_configuration() -> None:
    environment = {name: "32" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES}

    with pytest.raises(Stage2SExecutionEnvironmentError, match="after numeric imports"):
        prepare_formal_execution_environment(
            environ=environment,
            loaded_module_names=("numpy", "pyarrow.parquet"),
            physical_core_probe=lambda: 8,
        )

    assert environment == {name: "32" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES}


def test_formal_launcher_overrides_high_thread_environment_before_imports() -> None:
    environment = os.environ.copy()
    environment.update({name: "99" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES})
    environment["PYTHONPATH"] = str(ROOT / "src")
    probe_and_launch = "\n".join(
        (
            "import runpy",
            "import sys",
            "from seismoflux.stage2s import execution_environment as environment",
            "environment.detect_physical_core_count = lambda: 4",
            (
                "sys.argv = ['run_stage2s_once.py', '--repository-root', "
                f"{str(ROOT)!r}, '--environment-check-only']"
            ),
            f"runpy.run_path({str(LAUNCHER)!r}, run_name='__main__')",
        )
    )

    completed = subprocess.run(
        (sys.executable, "-c", probe_and_launch),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )

    payload = json.loads(completed.stdout)
    assert payload["workers"] == 1
    assert payload["physical_core_count"] == 4
    assert payload["reserved_physical_cores_minimum"] == 2
    assert payload["thread_environment"] == {
        name: "1" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    }
    assert payload["preimport_module_canary"] == {
        "matplotlib": False,
        "numpy": False,
        "pandas": False,
        "pyarrow": False,
        "scipy": False,
        "seismoflux.stage2s.production": False,
    }


def test_runner_fails_before_importing_production_without_launcher_evidence() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    code = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from seismoflux.stage2s.runner import run_stage2s_once",
            "assert 'seismoflux.stage2s.production' not in sys.modules",
            "try:",
            f"    run_stage2s_once(repository_root=Path({str(ROOT)!r}), progress=None)",
            "except RuntimeError:",
            "    pass",
            "else:",
            "    raise AssertionError('formal runner did not fail closed')",
            "assert 'seismoflux.stage2s.production' not in sys.modules",
        )
    )

    subprocess.run(
        (sys.executable, "-c", code),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
