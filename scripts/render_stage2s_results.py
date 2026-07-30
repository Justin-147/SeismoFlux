"""Render the frozen Stage 2S result bundle from one immutable JSON payload."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CANONICAL_FORMAL_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs/stage2s/causal_seismicity_screen"
).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from seismoflux.stage2s.rendering import (  # noqa: E402
    Stage2SRenderingError,
    parse_stage2s_render_payload,
    render_stage2s_bundle,
    verify_stage2s_bundle_against_record,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the three preregistered static SVGs, two offline HTML explorers, "
            "and one PNG companion from an already-frozen Stage 2S render payload."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="UTF-8 JSON containing a whole-run record, provenance, and derived map rasters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory that will contain the deterministic result files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing files instead of writing them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    try:
        if not args.check and output_dir == CANONICAL_FORMAL_OUTPUT_DIR:
            raise Stage2SRenderingError(
                "canonical formal Stage2S output is immutable; use --check only"
            )
        payload = parse_stage2s_render_payload(input_path.read_bytes())
        bundle = render_stage2s_bundle(payload)
        verify_stage2s_bundle_against_record(payload, bundle)
        bundle.write_to(output_dir, check=bool(args.check))
    except (OSError, Stage2SRenderingError, TypeError) as exc:
        print(f"stage2s result render error: {exc}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "wrote"
    print(f"{action} {len(bundle.artifacts)} deterministic Stage2S result files in {output_dir}")
    for name, digest in bundle.artifact_sha256_by_name.items():
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
