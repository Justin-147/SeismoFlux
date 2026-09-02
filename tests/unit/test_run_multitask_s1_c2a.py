"""The C2A CLI never advances from prediction into scoring on its own."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_cli() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_multitask_s1_c2a.py"
    specification = importlib.util.spec_from_file_location("c2a_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_worker_limit_is_explicit() -> None:
    parser = _load_cli().build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--phase", "predict", "--project-root", ".", "--data-root", ".", "--workers", "4"]
        )


@pytest.mark.parametrize("phase", ["predict", "score"])
def test_one_phase_only(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    calls: list[str] = []

    def predict(**kwargs: Any) -> Path:
        assert kwargs["workers"] == 2
        calls.append("predict")
        return Path("prediction_manifest.json")

    def score(**kwargs: Any) -> Path:
        assert "workers" not in kwargs
        calls.append("score")
        return Path("summary.json")

    prediction_module = ModuleType("seismoflux.multitask_s1.input_sensitivity_predict")
    prediction_module.run_prediction_phase = predict  # type: ignore[attr-defined]
    score_module = ModuleType("seismoflux.multitask_s1.input_sensitivity_score")
    score_module.run_score_phase = score  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, prediction_module.__name__, prediction_module)
    monkeypatch.setitem(sys.modules, score_module.__name__, score_module)
    assert _load_cli().main(["--phase", phase, "--project-root", ".", "--data-root", "."]) == 0
    assert calls == [phase]
