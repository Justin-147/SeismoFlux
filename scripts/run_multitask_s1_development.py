#!/usr/bin/env python
"""Run exactly one frozen S1-C0 phase; never auto-advance across the seal gate."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

# These must be set before importing NumPy/SciPy through the project package.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

from seismoflux.multitask_s1.development_runtime import (  # noqa: E402
    PredictionPhaseResult,
    ScorePhaseResult,
    run_phase,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one S1-C0 phase. 'predict' cannot read development targets; "
            "'score' requires the explicit four-fold master-seal SHA-256."
        )
    )
    parser.add_argument("--phase", choices=("predict", "score"), required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True, help="overall S1-C0 output root")
    parser.add_argument("--expected-seal-sha256")
    parser.add_argument("--maximum-fold-workers", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.phase == "score" and arguments.expected_seal_sha256 is None:
        parser.error("--phase score requires --expected-seal-sha256")
    if arguments.phase == "predict" and arguments.expected_seal_sha256 is not None:
        parser.error("--phase predict must not receive --expected-seal-sha256")
    result = run_phase(
        phase=arguments.phase,
        project_root=arguments.project_root,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        expected_seal_sha256=arguments.expected_seal_sha256,
        maximum_fold_workers=arguments.maximum_fold_workers,
    )
    if isinstance(result, PredictionPhaseResult):
        payload: dict[str, object] = {
            "phase": "predict",
            "status": "four_fold_predictions_sealed_no_score_action_in_this_command",
            "prediction_root": str(result.prediction_root),
            "master_seal_sha256": result.master_seal_sha256,
            "fold_ids": list(result.fold_ids),
        }
    elif isinstance(result, ScorePhaseResult):
        payload = {
            "phase": "score",
            "status": "development_directional_summary_written_S1_C1_still_mandatory",
            "score_root": str(result.score_root),
            "raw_scores_path": str(result.raw_scores_path),
            "summary_path": str(result.summary_path),
            "raw_score_row_counts": dict(result.raw_score_row_counts),
        }
    else:
        raise AssertionError("runtime returned an unknown phase result")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
