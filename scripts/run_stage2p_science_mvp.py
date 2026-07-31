"""Run the Stage 2P known-answer synthetic experiment and write its figures."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from seismoflux.stage2p.synthetic_experiment import (  # noqa: E402
    run_all_synthetic_scenarios,
)
from seismoflux.stage2p.visualization import (  # noqa: E402
    render_artifacts,
    write_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the three purely synthetic P0/P1/PP known-answer scenarios. "
            "This command never reads a real earthquake catalog or network resource."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for deterministic SVG, HTML, and metrics outputs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and verify an existing output directory without rewriting it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        results = run_all_synthetic_scenarios(bootstrap_replicates=2_000)
        artifacts = render_artifacts(results)
        hashes = write_artifacts(artifacts, args.output_dir, check=bool(args.check))
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"stage2p synthetic science MVP failed: {error}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "wrote"
    print(
        f"{action} {len(artifacts)} purely synthetic Stage2P artifacts "
        f"in {args.output_dir.resolve()}"
    )
    for result in results:
        p0 = result.evaluation.comparisons["P1_minus_P0"]
        pp = result.evaluation.comparisons["P1_minus_PP"]
        print(
            f"{result.scenario.scenario_id}: "
            f"synthetic_known_answer={result.synthetic_known_answer_status}; "
            f"P1-P0 recall={p0.macro_recall_gain_percentage_points:+.1f}pp "
            f"IG={p0.macro_information_gain_nats_per_event:+.3f}; "
            f"P1-PP recall={pp.macro_recall_gain_percentage_points:+.1f}pp "
            f"IG={pp.macro_information_gain_nats_per_event:+.3f}"
        )
    for name, digest in hashes.items():
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
