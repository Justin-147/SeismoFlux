"""Sole executable production entry for the stage-4 formal score."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from seismoflux.anomaly_increment.formal_production import (
    authorize_stage4_formal_readiness,
    load_stage4_formal_readiness,
    verify_stage4_formal_readiness,
)
from seismoflux.anomaly_increment.formal_run import run_formal_stage4

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or explicitly execute the sole stage-4 formal chain."
    )
    parser.add_argument("mode", choices=("check", "run"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    readiness = load_stage4_formal_readiness(ROOT)
    if args.mode == "check":
        proof = verify_stage4_formal_readiness(readiness)
        print(
            json.dumps(
                {
                    **readiness.as_mapping(),
                    **proof.as_mapping(),
                    "mode": "check",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    inputs = authorize_stage4_formal_readiness(readiness)
    outcome = run_formal_stage4(inputs)
    print(
        json.dumps(
            {
                "failure_code": outcome.failure_code,
                "mode": "run",
                "publication_bundle_id": (
                    None if outcome.publication is None else outcome.publication.bundle_id
                ),
                "same_process_resume_count": outcome.same_process_resume_count,
                "session_seal_sha256": outcome.session_seal_sha256,
                "status": outcome.status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if outcome.status == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
