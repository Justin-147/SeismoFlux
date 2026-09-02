#!/usr/bin/env python
"""One explicit C2B prediction or scoring phase, with numerical threads limited."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("predict", "score"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if args.phase == "predict":
        from seismoflux.multitask_s1.c2b_predict import run_prediction_phase

        artifact = run_prediction_phase(
            project_root=args.project_root,
            data_root=args.data_root,
            output_root=args.output_root,
            workers=args.workers,
        )
    else:
        from seismoflux.multitask_s1.c2b_score import run_score_phase

        artifact = run_score_phase(
            project_root=args.project_root, data_root=args.data_root, output_root=args.output_root
        )
    print(artifact, flush=True)


if __name__ == "__main__":
    main()
