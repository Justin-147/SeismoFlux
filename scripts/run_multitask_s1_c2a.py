#!/usr/bin/env python
"""Run one phase of the finite, location-only C2A input sensitivity."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

# Keep the user's desktop responsive before importing numerical libraries.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C2A fixed-parameter input sensitivity; one explicit phase per invocation."
    )
    parser.add_argument("--phase", choices=("predict", "score"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    print(
        json.dumps(
            {
                "phase": arguments.phase,
                "status": "started",
                "time_utc": datetime.now(UTC).isoformat(),
                "fold_workers": arguments.workers if arguments.phase == "predict" else 1,
                "numerical_threads": 1,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if arguments.phase == "predict":
        from seismoflux.multitask_s1.input_sensitivity_predict import run_prediction_phase

        result = run_prediction_phase(
            project_root=arguments.project_root,
            data_root=arguments.data_root,
            output_root=arguments.output_root,
            workers=arguments.workers,
        )
        status = "predictions_complete_no_scoring_in_this_command"
    else:
        from seismoflux.multitask_s1.input_sensitivity_score import run_score_phase

        result = run_score_phase(
            project_root=arguments.project_root,
            data_root=arguments.data_root,
            output_root=arguments.output_root,
        )
        status = "development_input_sensitivity_scored_not_independent_confirmation"
    print(
        json.dumps(
            {
                "phase": arguments.phase,
                "status": status,
                "artifact": str(result),
                "time_utc": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
