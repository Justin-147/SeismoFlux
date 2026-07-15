"""Audit the final R1 Arrow identity on all 153 target-blind 25 km issues."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from seismoflux.anomaly_increment.compute import build_compute_plan
from seismoflux.anomaly_increment.config import (
    STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH,
    load_stage4_protocol_bundle,
)
from seismoflux.anomaly_increment.convergence import (
    audit_primary_grid_logical_replay_r1,
)
from seismoflux.anomaly_increment.formal_preflight import load_formal_preflight
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
)
from seismoflux.anomaly_increment.preregistration import with_content_sha256
from seismoflux.anomaly_increment.qualification import observe_score_blind_inputs
from seismoflux.background.execution import (
    CommandResult,
    detect_physical_core_count,
    subprocess_git_runner,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.joinpath(*STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH.parts)
_GIT_OID = re.compile(r"[0-9a-f]{40}\Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay all 153 accepted 25 km issues with 1/2 workers using the final "
            "R1 logical Arrow identity."
        )
    )
    parser.add_argument("mode", choices=("dry-run", "generate", "check"))
    parser.add_argument(
        "--scoring-code-commit",
        required=True,
        help="Exact 40-character lowercase commit containing the audited scoring code.",
    )
    return parser


def _create_or_verify(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with ExitStack() as cleanup:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            staged = Path(handle.name)
            cleanup.callback(staged.unlink, missing_ok=True)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            try:
                existing = read_existing_immutable_bytes(
                    path,
                    label="existing R1 logical replay audit",
                )
            except UnsafeImmutableFileError as exc:
                raise ValueError("existing R1 replay audit is unsafe") from exc
            if existing != payload:
                raise ValueError("existing R1 replay audit contains different bytes") from None


def _git_output(command: tuple[str, ...]) -> str:
    result = subprocess_git_runner(command, ROOT)
    if not isinstance(result, CommandResult) or result.returncode != 0:
        raise ValueError(f"Git proof failed: {' '.join(command[1:])}")
    return result.stdout.strip()


def _require_exact_clean_head(commit: str) -> None:
    if _git_output(("git", "rev-parse", "HEAD")) != commit:
        raise ValueError("R1 replay commit must equal the current repository HEAD")
    if _git_output(("git", "status", "--porcelain=v1", "--untracked-files=all")):
        raise ValueError("R1 replay requires a completely clean tracked/untracked worktree")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commit = str(args.scoring_code_commit)
    if _GIT_OID.fullmatch(commit) is None:
        raise ValueError("scoring-code commit must be a 40-character lowercase Git OID")
    protocol = load_stage4_protocol_bundle(ROOT)
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "issue_count": 153,
                    "locked_test_run": False,
                    "output": OUTPUT.relative_to(ROOT).as_posix(),
                    "scoring_code_commit": commit,
                    "target_bytes_read": False,
                    "worker_counts": [1, 2],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    _require_exact_clean_head(commit)
    compute = build_compute_plan(
        protocol,
        detected_physical_cores=detect_physical_core_count(),
        detected_logical_processors=os.cpu_count(),
    )
    score_blind_inputs = observe_score_blind_inputs(ROOT, protocol.protocol)
    bundle = load_formal_preflight(
        protocol,
        score_blind_inputs,
        compute,
        scoring_code_commit=commit,
    )
    audit = audit_primary_grid_logical_replay_r1(
        issue_ids=bundle.calendar.issue_ids,
        snapshots=tuple(
            bundle.snapshots_by_issue_id[issue_id] for issue_id in bundle.calendar.issue_ids
        ),
        grid_family=bundle.grid_family,
        accepted_primary_issue_tables=bundle.issue_tables,
        source_columns=bundle.feature_layout.dynamic_sources,
        source_input_sha256=score_blind_inputs.content_sha256,
        worker_counts=(1, 2),
    )
    document = with_content_sha256(
        {
            "audit": audit.as_mapping(),
            "locked_test_run": False,
            "protocol_design_sha256": protocol.design_sha256,
            "random_input_seal_sha256": protocol.random_input_seal_sha256,
            "schema_version": 1,
            "scoring_code_commit": commit,
            "worktree_clean_before_and_after": True,
            "target_bytes_read": False,
            "target_path_observed": False,
        }
    )
    _require_exact_clean_head(commit)
    if args.mode == "generate":
        _create_or_verify(OUTPUT, document)
    else:
        try:
            observed = read_existing_immutable_bytes(
                OUTPUT,
                label="stored R1 logical replay audit",
            )
        except UnsafeImmutableFileError as exc:
            raise ValueError("stored R1 logical replay audit is unsafe") from exc
        expected = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if observed != expected:
            raise ValueError("stored R1 logical replay audit differs from fresh replay")
    _require_exact_clean_head(commit)
    print(
        json.dumps(
            {
                "audit_content_sha256": audit.content_sha256,
                "formal_issue_count": len(bundle.calendar.issue_ids),
                "locked_test_run": False,
                "mode": args.mode,
                "output": OUTPUT.relative_to(ROOT).as_posix(),
                "reproduction_identity_sha256": audit.reproduction_identity_sha256,
                "target_bytes_read": False,
                "worker_counts": list(audit.worker_counts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
