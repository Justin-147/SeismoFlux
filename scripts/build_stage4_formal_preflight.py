"""Build or verify the local target-blind 153-issue stage-4 preflight receipt."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from seismoflux.anomaly_increment.compute import build_compute_plan
from seismoflux.anomaly_increment.config import (
    load_stage4_protocol_bundle,
    require_stage4_r2_execution_action,
)
from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_PREFLIGHT_RECEIPT_PATH,
    FormalIssueCalendar,
    load_formal_preflight,
    load_formal_preflight_receipt,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
)
from seismoflux.anomaly_increment.qualification import observe_score_blind_inputs
from seismoflux.background.execution import detect_physical_core_count

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/check the score-blind stage-4 formal-input receipt."
    )
    parser.add_argument("mode", choices=("dry-run", "generate", "check"))
    parser.add_argument(
        "--scoring-code-commit",
        required=True,
        help="Frozen 40-character lowercase scoring-code Git OID.",
    )
    return parser


def _write_atomic(
    path: Path,
    document: dict[str, object],
    *,
    protocol: Mapping[str, object],
) -> None:
    """Create the canonical receipt once; permit only byte-identical replay."""

    # This writer is intentionally guarded even though its leading underscore
    # marks it private: direct imports must not bypass the canonical R2 stop.
    # Keep this before ``path.parent``, serialization, hashing, mkdir, tempfile,
    # link, or open operations.
    require_stage4_r2_execution_action(protocol, action="formal_preflight")
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
    temporary: str | None = None
    created = False
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        staged = Path(temporary)
        try:
            os.link(staged, path)
            created = True
        except FileExistsError:
            try:
                existing = read_existing_immutable_bytes(
                    path,
                    label="existing formal preflight receipt",
                )
            except UnsafeImmutableFileError:
                existing = None
            if existing == payload:
                return
            raise ValueError(
                "immutable formal preflight receipt already contains different bytes "
                "or is not a safe single-link regular file"
            ) from None
        finally:
            staged.unlink(missing_ok=True)
            temporary = None
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise
    if not created:
        raise AssertionError("formal preflight publication reached no terminal state")
    try:
        if (
            read_existing_immutable_bytes(
                path,
                label="new formal preflight receipt",
            )
            != payload
        ):
            raise UnsafeImmutableFileError("new formal preflight receipt bytes changed")
    except UnsafeImmutableFileError as exc:
        raise ValueError(
            "new formal preflight receipt is not a safe single-link regular file"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_stage4_protocol_bundle(ROOT)
    if args.mode != "dry-run":
        require_stage4_r2_execution_action(protocol.protocol, action="formal_preflight")
    calendar = FormalIssueCalendar.from_protocol(protocol)
    physical = detect_physical_core_count()
    compute = build_compute_plan(
        protocol,
        detected_physical_cores=physical,
        detected_logical_processors=os.cpu_count(),
    )
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "assessment_issue_count": len(calendar.assessment_issue_ids),
                    "backend": compute.backend,
                    "effective_workers": compute.workers.effective_workers,
                    "fit_issue_count": len(calendar.fit_issue_ids),
                    "formal_issue_count": len(calendar.issue_ids),
                    "locked_test_run": False,
                    "output": OUTPUT.relative_to(ROOT).as_posix(),
                    "reserve_physical_cores": compute.workers.reserve_physical_cores,
                    "target_bytes_read": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    evidence = observe_score_blind_inputs(ROOT, protocol.protocol)
    bundle = load_formal_preflight(
        protocol,
        evidence,
        compute,
        scoring_code_commit=args.scoring_code_commit,
    )
    document = bundle.receipt.as_mapping()
    if args.mode == "generate":
        _write_atomic(OUTPUT, document, protocol=protocol.protocol)
    else:
        if not OUTPUT.is_file():
            raise FileNotFoundError("formal preflight receipt has not been generated")
        observed = load_formal_preflight_receipt(OUTPUT, protocol=protocol.protocol)
        if observed.content_sha256 != bundle.receipt.content_sha256:
            raise ValueError("deterministic preflight identity differs from fresh inputs")
        if observed.deterministic_mapping() != bundle.receipt.deterministic_mapping():
            raise ValueError("deterministic preflight mapping differs from fresh inputs")
        if observed.space_placebo_resource_observation.feature_identity_sha256 != (
            bundle.receipt.space_placebo_feature_identity.content_sha256
        ):
            raise ValueError("stored resource observation belongs to another feature identity")
        if observed.space_placebo_resource_observation.output_identity_sha256 != (
            bundle.receipt.space_placebo_feature_identity.output_identity_sha256
        ):
            raise ValueError("stored resource observation output identity changed")
        if (
            bundle.receipt.space_placebo_resource_observation.recommended_max_in_flight
            < observed.space_placebo_resource_observation.recommended_max_in_flight
        ):
            raise ValueError(
                "fresh target-blind memory observation cannot safely support the stored "
                "space-placebo concurrency"
            )
    print(
        json.dumps(
            {
                "as_if_shadow_status": bundle.receipt.shadow.status,
                "content_sha256": bundle.receipt.content_sha256,
                "formal_issue_count": len(bundle.calendar.issue_ids),
                "locked_test_run": False,
                "mode": args.mode,
                "recommended_max_in_flight": (
                    bundle.receipt.space_placebo_resource_observation.recommended_max_in_flight
                ),
                "resource_observation_sha256": (
                    bundle.receipt.space_placebo_resource_observation.content_sha256
                ),
                "target_bytes_read": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
