"""Launch the one-shot Stage 2S run with resource controls set before imports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the registered Stage 2S historical development attempt after "
            "fail-closed pre-import CPU/thread checks."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SeismoFlux repository root; defaults to the parent of this script directory",
    )
    parser.add_argument(
        "--environment-check-only",
        action="store_true",
        help="verify and print pre-import resource evidence without running an attempt",
    )
    return parser


def _add_source_root(repository_root: Path) -> None:
    source_root = repository_root / "src"
    if not source_root.is_dir():
        raise ValueError(f"repository source directory is missing: {source_root}")
    value = str(source_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve(strict=True)
    try:
        _add_source_root(repository_root)
    except (OSError, ValueError) as exc:
        print(f"Stage2S formal environment error: {exc}", file=sys.stderr)
        return 2

    from seismoflux.stage2s.execution_environment import (
        Stage2SExecutionEnvironmentError,
        prepare_formal_execution_environment,
    )

    try:
        evidence = prepare_formal_execution_environment()
    except Stage2SExecutionEnvironmentError as exc:
        print(f"Stage2S formal environment error: {exc}", file=sys.stderr)
        return 2

    if args.environment_check_only:
        payload = evidence.receipt_bindings()
        payload["preimport_module_canary"] = {
            name: name in sys.modules
            for name in (
                "numpy",
                "pyarrow",
                "scipy",
                "pandas",
                "matplotlib",
                "seismoflux.stage2s.production",
            )
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0

    from seismoflux.stage2s.runner import run_stage2s_once

    try:
        result = run_stage2s_once(
            repository_root=repository_root,
            progress=print,
        )
    except Exception as exc:
        print(f"Stage2S formal execution failed: {exc}", file=sys.stderr)
        return 1
    print(f"Stage2S formal execution completed with mode={result.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
