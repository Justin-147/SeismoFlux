#!/usr/bin/env python
"""Run the frozen S2-C prediction, verification or scoring phase with bounded threads."""

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
    parser.add_argument("--phase", choices=("predict", "verify", "score"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()

    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if args.phase == "predict":
        from seismoflux.multitask_s2.strain_predict import run_prediction_phase

        artifact = run_prediction_phase(
            project_root=args.project_root,
            data_root=args.data_root,
            output_root=args.output_root,
            workers=args.workers,
        )
    elif args.phase == "verify":
        from seismoflux.multitask_s2.strain_predict import load_protocol, verify_prediction_manifest

        output = (
            args.output_root
            or args.project_root / load_protocol(args.project_root)["outputs"]["root"]
        )
        verify_prediction_manifest(args.project_root, output)
        artifact = output / "prediction_manifest.json"
    else:
        from seismoflux.multitask_s2.strain_score import run_score_phase

        artifact = run_score_phase(
            project_root=args.project_root,
            data_root=args.data_root,
            output_root=args.output_root,
        )
    print(artifact, flush=True)


if __name__ == "__main__":
    main()
