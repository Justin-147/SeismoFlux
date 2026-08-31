"""Operate the frozen P1 B0 versus B0_R30 prospective experiment.

Administrative commands create the public genesis/authorization chain.  The
``prepare`` command is the only network-capable path and is doubly gated by the
public authorization record and the legal ``[Q,T)`` window.  Prepared artifacts
must be committed, pushed, and remotely read back before ``seal`` appends the
ForecastIssueRecord; ``seal`` first replays exact public ComCat bytes through the
frozen local inputs and model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

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

from seismoflux.data.common import canonical_json_bytes  # noqa: E402
from seismoflux.p1_b0_r30.ledger import (  # noqa: E402
    append_new_p1_record,
    append_p1_record,
    build_next_p1_record,
    p1_ledger_path,
    read_p1_ledger,
)
from seismoflux.p1_b0_r30.operations import (  # noqa: E402
    ARTIFACT_MANIFEST_FILENAME,
    COUNT_BODY_FILENAME,
    COUNT_HEADERS_FILENAME,
    FORECAST_GRID_FILENAME,
    OFFLINE_HTML_FILENAME,
    PREPARED_RECEIPT_FILENAME,
    QUERY_BODY_FILENAME,
    QUERY_HEADERS_FILENAME,
    SOURCE_RECEIPT_FILENAME,
    STATIC_SVG_FILENAME,
    build_forecast_issue_record_fields,
    prepare_production_issue_artifacts,
    verify_prepared_issue_against_frozen_inputs,
)
from seismoflux.p1_b0_r30.production import (  # noqa: E402
    P1IssueSchedule,
    acquire_issue_input,
    issue_schedule,
)
from seismoflux.p1_b0_r30.prospective import (  # noqa: E402
    P1_MODEL_MANIFEST_SHA256,
    P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
    build_production_forecast,
    validate_washout,
)
from seismoflux.p1_b0_r30.runtime import (  # noqa: E402
    AuthorizedIssueContext,
    _build_guarded_time_gated_comcat_transport,
    load_authorized_issue_context,
)

PROTOCOL_COMMIT: Final = "0f43f15bc983a37157f1b129976c7ec0ea47fc7d"
SCIENTIFIC_CODE_COMMIT: Final = "c71c97790adcf33f6c8121e367317857dc8dff31"
PROTOCOL_TAG: Final = "v0.2.7-p1-b0-r30-protocol"
SCIENTIFIC_CODE_TAG: Final = "v0.2.7-p1-b0-r30-code"
OPERATIONS_TAG: Final = "v0.2.7-p1-b0-r30-ops"
REMOTE_NAME: Final = "origin"
REMOTE_BRANCH_NAME: Final = "codex/stage2-etas-science-first"
EXPECTED_ORIGIN_URL: Final = "https://github.com/Justin-147/SeismoFlux.git"
AUTHORIZATION_EVIDENCE_RELATIVE_PATH: Final = Path("docs/p1_real_issue_authorization_2026-08-31.md")
AUTHORIZED_QUOTE: Final = (
    "我授权依据冻结的 P1 v0.2.7 协议创建 RealIssueAuthorizationRecord，"  # noqa: RUF001
    "并从下一个合法规则时刻开始真实前瞻预测；"  # noqa: RUF001
    "不得补发、不得修改冻结模型或利用未来地震信息。"
)

SCHEMA_PATH: Final = REPOSITORY_ROOT / "data/contracts/p1_prospective_records_v1.json"
LOCAL_CATALOG_PATH: Final = (
    REPOSITORY_ROOT / "data/processed/stage1/debc98054172a4a1/earthquake_event.parquet"
)
STUDY_AREA_PATH: Final = REPOSITORY_ROOT / "data/processed/china_mainland.geojson"
SUPPORT_MANIFEST_PATH: Final = (
    REPOSITORY_ROOT / "data/manifests/background_local_support_manifest.json"
)
SOURCE_MANIFEST_PATH: Final = REPOSITORY_ROOT / "data/manifests/p1_source_boundary_manifest.json"
MODEL_MANIFEST_PATH: Final = REPOSITORY_ROOT / "data/manifests/p1_model_manifest.json"
ISSUE_PARENT: Final = REPOSITORY_ROOT / "outputs/prospective/p1_b0_r30"
LEDGER_RELATIVE_PATH: Final = Path("outputs/prospective/p1_b0_r30_records_v1.jsonl")
EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS: Final = {
    "jsonschema": "4.20.0",
    "numpy": "2.4.6",
    "pyarrow": "25.0.0",
    "pyproj": "3.7.2",
    "scipy": "1.17.1",
    "shapely": "2.1.2",
}
REMOTE_COMMAND_TIMEOUT_SECONDS: Final = 30.0
FINAL_RECORD_START_MARGIN_SECONDS: Final = 120.0
FINAL_RECORD_READBACK_RESERVE_SECONDS: Final = 45.0

REMOTE_ISSUE_FILENAMES: Final = (
    COUNT_BODY_FILENAME,
    COUNT_HEADERS_FILENAME,
    QUERY_BODY_FILENAME,
    QUERY_HEADERS_FILENAME,
    SOURCE_RECEIPT_FILENAME,
    FORECAST_GRID_FILENAME,
    STATIC_SVG_FILENAME,
    OFFLINE_HTML_FILENAME,
    PREPARED_RECEIPT_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
)

EXPECTED_LOCAL_CATALOG_SHA256: Final = (
    "2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347"
)
EXPECTED_STUDY_AREA_SHA256: Final = (
    "5e5dcf012e080882161c95bf592a1ee39a0f0fdad7114bcff58d645aeb30bb02"
)
EXPECTED_SUPPORT_MANIFEST_SHA256: Final = (
    "632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d"
)

OPERATIONAL_IDENTITY_PATHS: Final = (
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "uv.lock",
    "configs/p1_b0_r30_prospective.yaml",
    "data/contracts/p1_prospective_records_v1.json",
    "data/manifests/p1_source_boundary_manifest.json",
    "data/manifests/p1_model_manifest.json",
    "data/manifests/background_local_support_manifest.json",
    "scripts/run_p1_b0_r30_prospective.py",
    "src/seismoflux/__init__.py",
    "src/seismoflux/config.py",
    "src/seismoflux/background",
    "src/seismoflux/d1_replay",
    "src/seismoflux/data",
    "src/seismoflux/features",
    "src/seismoflux/stage2s",
    "src/seismoflux/p1_b0_r30",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str, *, label: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"{label} must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _git_sha(value: str, *, label: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase 40-character Git SHA")
    return value


def _schema() -> Mapping[str, object]:
    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("P1 schema must be a JSON object")
    return cast(Mapping[str, object], value)


def _write_output(value: Mapping[str, object]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b"\n")


def _run_git(
    arguments: Sequence[str],
    *,
    text: bool = True,
    timeout_seconds: float | None = None,
) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=text,
        timeout=timeout_seconds,
    )
    if text:
        return cast(str, completed.stdout).strip()
    return cast(bytes, completed.stdout)


def _run_git_with_environment(
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    input_bytes: bytes | None = None,
    timeout_seconds: float | None = None,
) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        input=input_bytes,
        env=dict(environment),
        timeout=timeout_seconds,
    )
    return completed.stdout


def _rev_parse(revision: str) -> str:
    output = _run_git(("rev-parse", revision))
    if not isinstance(output, str):
        raise TypeError("git rev-parse returned non-text output")
    return output


def _verify_frozen_tags() -> None:
    if _rev_parse(f"{PROTOCOL_TAG}^{{}}") != PROTOCOL_COMMIT:
        raise ValueError("frozen protocol tag does not resolve to the accepted protocol commit")
    if _rev_parse(f"{SCIENTIFIC_CODE_TAG}^{{}}") != SCIENTIFIC_CODE_COMMIT:
        raise ValueError("frozen scientific code tag does not resolve to P1-0C")


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError("git could not verify commit ancestry")
    return completed.returncode == 0


def _verify_runtime_environment() -> None:
    expected_python = (REPOSITORY_ROOT / ".venv/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ValueError("real P1 execution requires the repository's frozen .venv Python")
    for package, expected_version in EXPECTED_SCIENTIFIC_PACKAGE_VERSIONS.items():
        if importlib.metadata.version(package) != expected_version:
            raise ValueError(f"real P1 execution requires the frozen {package} version")
    source_root = SOURCE_ROOT.resolve()
    for function in (
        canonical_json_bytes,
        prepare_production_issue_artifacts,
        build_production_forecast,
        load_authorized_issue_context,
    ):
        source_path = Path(inspect.getfile(function)).resolve()
        if source_root not in source_path.parents:
            raise ValueError("an imported P1 implementation escaped the frozen repository source")


def _verify_operational_identity(code_commit: str) -> None:
    commit = _git_sha(code_commit, label="code_commit")
    _verify_runtime_environment()
    _verify_frozen_tags()
    if _rev_parse(f"{OPERATIONS_TAG}^{{}}") != commit:
        raise ValueError("operations tag does not resolve to the authorized code commit")
    if not _is_ancestor(commit, "HEAD"):
        raise ValueError("authorized operations commit is not an ancestor of HEAD")
    completed = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", *OPERATIONAL_IDENTITY_PATHS],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode == 1:
        raise ValueError("frozen operational/scientific files differ from the authorized commit")
    if completed.returncode != 0:
        raise RuntimeError("git could not verify the frozen operational file identity")
    status = _run_git(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *OPERATIONAL_IDENTITY_PATHS,
        )
    )
    if not isinstance(status, str):
        raise TypeError("git status returned non-text output")
    if status:
        raise ValueError("frozen operational/scientific paths contain local changes")


def _verify_origin_url() -> None:
    for direction in ((), ("--push",)):
        observed = _run_git(("remote", "get-url", *direction, REMOTE_NAME))
        if observed != EXPECTED_ORIGIN_URL:
            raise ValueError("origin does not point to the authorized public SeismoFlux repository")


def _parse_ls_remote_sha(output: str | bytes, *, label: str) -> str:
    if not isinstance(output, str):
        raise TypeError(f"{label} lookup returned non-text output")
    lines = tuple(line for line in output.splitlines() if line.strip())
    if len(lines) != 1:
        raise ValueError(f"{label} must resolve to exactly one remote Git object")
    fields = lines[0].split()
    if len(fields) != 2:
        raise ValueError(f"{label} remote response is malformed")
    return _git_sha(fields[0], label=label)


def _remote_branch_head() -> str:
    _verify_origin_url()
    return _parse_ls_remote_sha(
        _run_git(
            ("ls-remote", REMOTE_NAME, f"refs/heads/{REMOTE_BRANCH_NAME}"),
            timeout_seconds=REMOTE_COMMAND_TIMEOUT_SECONDS,
        ),
        label="remote branch head",
    )


def _verify_remote_operations_tag(code_commit: str) -> None:
    observed = _parse_ls_remote_sha(
        _run_git(
            ("ls-remote", REMOTE_NAME, f"refs/tags/{OPERATIONS_TAG}^{{}}"),
            timeout_seconds=REMOTE_COMMAND_TIMEOUT_SECONDS,
        ),
        label="remote operations tag",
    )
    if observed != code_commit:
        raise ValueError("public operations tag does not resolve to the authorized code commit")


def _remote_file_bytes(remote_head: str, relative_path: Path) -> bytes:
    payload = _run_git(
        ("show", f"{remote_head}:{relative_path.as_posix()}"),
        text=False,
    )
    if not isinstance(payload, bytes):
        raise TypeError("git show returned non-byte output")
    return payload


def _verify_remote_ledger_anchor(
    *,
    context: AuthorizedIssueContext,
    code_commit: str,
) -> tuple[str, datetime]:
    authorization_commit = _git_sha(
        cast(str, context.authorization_record.get("authorization_commit")),
        label="authorization_commit",
    )
    remote_head = _remote_branch_head()
    _verify_remote_operations_tag(code_commit)
    for ancestor, label in (
        (code_commit, "authorized operations commit"),
        (authorization_commit, "authorization evidence commit"),
    ):
        if not _is_ancestor(ancestor, remote_head):
            raise ValueError(f"{label} is not present on the actual public branch")
    local_ledger = p1_ledger_path(REPOSITORY_ROOT).read_bytes()
    if _remote_file_bytes(remote_head, LEDGER_RELATIVE_PATH) != local_ledger:
        raise ValueError("actual public branch ledger differs byte-for-byte from the local chain")
    verified_at = datetime.now(UTC)
    if verified_at >= context.schedule.scheduled_issue_time_utc:
        raise ValueError("public ledger verification did not finish before T")
    return remote_head, verified_at


def _verify_remote_issue_publication(
    *,
    context: AuthorizedIssueContext,
    code_commit: str,
    issue_directory: Path,
) -> tuple[str, datetime]:
    issue_id = context.schedule.issue_id
    expected_directory = (ISSUE_PARENT / issue_id).resolve()
    if issue_directory.resolve() != expected_directory:
        raise ValueError("only the canonical public issue directory can be sealed")
    remote_head, _ = _verify_remote_ledger_anchor(context=context, code_commit=code_commit)
    relative_directory = expected_directory.relative_to(REPOSITORY_ROOT.resolve())
    names_output = _run_git(
        (
            "ls-tree",
            "-r",
            "--name-only",
            remote_head,
            "--",
            relative_directory.as_posix(),
        )
    )
    if not isinstance(names_output, str):
        raise TypeError("remote issue tree lookup returned non-text output")
    expected_paths = tuple(
        (relative_directory / filename).as_posix() for filename in REMOTE_ISSUE_FILENAMES
    )
    observed_paths = tuple(line for line in names_output.splitlines() if line)
    if set(observed_paths) != set(expected_paths) or len(observed_paths) != len(expected_paths):
        raise ValueError("actual public branch does not contain the exact canonical issue package")
    for filename, relative_path in zip(REMOTE_ISSUE_FILENAMES, expected_paths, strict=True):
        local_payload = (expected_directory / filename).read_bytes()
        if _remote_file_bytes(remote_head, Path(relative_path)) != local_payload:
            raise ValueError(f"public issue artifact differs byte-for-byte: {filename}")
    verified_at = datetime.now(UTC)
    if verified_at >= context.schedule.scheduled_issue_time_utc:
        raise ValueError("public issue readback did not finish before T")
    return remote_head, verified_at


def _require_clean_tracked_state(remote_head: str) -> None:
    if _rev_parse("HEAD") != remote_head:
        raise ValueError("local HEAD must equal the remotely verified artifact commit")
    branch = _run_git(("symbolic-ref", "--short", "HEAD"))
    if branch != REMOTE_BRANCH_NAME:
        raise ValueError("real P1 finalization must run on the fixed public branch")
    for arguments, label in (
        (("diff", "--quiet", "HEAD", "--"), "tracked worktree"),
        (("diff", "--cached", "--quiet", "HEAD", "--"), "Git index"),
    ):
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        if completed.returncode == 1:
            raise ValueError(f"{label} must be clean before forecast finalization")
        if completed.returncode != 0:
            raise RuntimeError(f"git could not verify the {label}")


def _read_validated_candidate_ledger_bytes(
    candidate_ledger_bytes: bytes,
) -> tuple[dict[str, object], ...]:
    prospective_parent = REPOSITORY_ROOT / "outputs/prospective"
    prospective_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".p1_ledger_validation_",
        dir=prospective_parent,
    ) as temporary_directory:
        validation_root = Path(temporary_directory)
        candidate_path = validation_root / LEDGER_RELATIVE_PATH
        candidate_path.parent.mkdir(parents=True)
        candidate_path.write_bytes(candidate_ledger_bytes)
        records = read_p1_ledger(
            validation_root,
            schema=_schema(),
            require_exists=True,
        )
    return records


def _validate_candidate_ledger_bytes(
    candidate_ledger_bytes: bytes,
    *,
    expected_record: Mapping[str, object],
) -> None:
    records = _read_validated_candidate_ledger_bytes(candidate_ledger_bytes)
    if not records or records[-1] != dict(expected_record):
        raise ValueError("candidate ledger does not end in the unique next Forecast record")


def _build_ledger_only_commit(
    *,
    parent_commit: str,
    candidate_ledger_bytes: bytes,
    issue_id: str,
    recorded_at_utc: datetime,
) -> str:
    prospective_parent = REPOSITORY_ROOT / "outputs/prospective"
    prospective_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".p1_forecast_commit_",
        dir=prospective_parent,
    ) as temporary_directory:
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(Path(temporary_directory) / "index")
        environment["GIT_AUTHOR_DATE"] = _utc_text(recorded_at_utc)
        environment["GIT_COMMITTER_DATE"] = _utc_text(recorded_at_utc)
        _run_git_with_environment(("read-tree", parent_commit), environment=environment)
        blob_output = _run_git_with_environment(
            ("hash-object", "-w", "--stdin"),
            environment=environment,
            input_bytes=candidate_ledger_bytes,
        )
        blob_sha = _git_sha(blob_output.decode("ascii").strip(), label="candidate ledger blob")
        _run_git_with_environment(
            (
                "update-index",
                "--add",
                "--cacheinfo",
                f"100644,{blob_sha},{LEDGER_RELATIVE_PATH.as_posix()}",
            ),
            environment=environment,
        )
        tree_output = _run_git_with_environment(("write-tree",), environment=environment)
        tree_sha = _git_sha(tree_output.decode("ascii").strip(), label="candidate tree")
        commit_output = _run_git_with_environment(
            (
                "commit-tree",
                tree_sha,
                "-p",
                parent_commit,
                "-m",
                f"science(p1): close {issue_id} forecast record",
            ),
            environment=environment,
        )
    candidate_commit = _git_sha(
        commit_output.decode("ascii").strip(),
        label="candidate forecast commit",
    )
    parent = _rev_parse(f"{candidate_commit}^")
    if parent != parent_commit or not _is_ancestor(parent_commit, candidate_commit):
        raise ValueError("candidate forecast commit is not a child of the verified artifact commit")
    changed = _run_git(
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            parent_commit,
            candidate_commit,
        )
    )
    if changed != LEDGER_RELATIVE_PATH.as_posix():
        raise ValueError("candidate forecast commit must change only the fixed P1 ledger")
    if _remote_file_bytes(candidate_commit, LEDGER_RELATIVE_PATH) != candidate_ledger_bytes:
        raise ValueError("candidate commit ledger differs from the pure next-record bytes")
    return candidate_commit


def _push_forecast_commit_before_t(
    *,
    parent_commit: str,
    candidate_commit: str,
    previous_ledger_bytes: bytes,
    candidate_ledger_bytes: bytes,
    issue_directory: Path,
    scheduled_issue_time_utc: datetime,
) -> datetime:
    start_deadline = scheduled_issue_time_utc - timedelta(seconds=FINAL_RECORD_START_MARGIN_SECONDS)
    if datetime.now(UTC) >= start_deadline:
        raise ValueError(
            "insufficient safety margin remains to publish the forecast record before T"
        )
    if p1_ledger_path(REPOSITORY_ROOT).read_bytes() != previous_ledger_bytes:
        raise ValueError("local P1 ledger changed before the forecast record push")
    if _remote_branch_head() != parent_commit:
        raise ValueError("public branch moved after the issue artifact readback")
    if _remote_file_bytes(parent_commit, LEDGER_RELATIVE_PATH) != previous_ledger_bytes:
        raise ValueError("public parent ledger changed before the forecast record push")
    relative_directory = issue_directory.resolve().relative_to(REPOSITORY_ROOT.resolve())
    for filename in REMOTE_ISSUE_FILENAMES:
        relative_path = relative_directory / filename
        if (
            _remote_file_bytes(candidate_commit, relative_path)
            != (issue_directory / filename).read_bytes()
        ):
            raise ValueError(f"candidate forecast commit changed an issue artifact: {filename}")
    push_deadline = scheduled_issue_time_utc - timedelta(
        seconds=FINAL_RECORD_READBACK_RESERVE_SECONDS
    )
    remaining = (push_deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise ValueError("remote pre-push checks consumed the forecast finalization margin")
    ref = f"refs/heads/{REMOTE_BRANCH_NAME}"
    push_arguments = (
        "push",
        "--porcelain",
        f"--force-with-lease={ref}:{parent_commit}",
        REMOTE_NAME,
        f"{candidate_commit}:{ref}",
    )
    push_error: subprocess.CalledProcessError | subprocess.TimeoutExpired | None = None
    try:
        _run_git(push_arguments, timeout_seconds=remaining)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        push_error = exc
    observed_remote_head = _remote_branch_head()
    remotely_observed_at = datetime.now(UTC)
    if observed_remote_head != candidate_commit:
        if push_error is not None:
            raise push_error
        raise ValueError(
            "actual public branch did not atomically accept the forecast record commit"
        )
    if remotely_observed_at >= scheduled_issue_time_utc:
        raise ValueError("forecast record was not observed on the public branch before T")
    if _remote_file_bytes(candidate_commit, LEDGER_RELATIVE_PATH) != candidate_ledger_bytes:
        raise ValueError("public forecast ledger differs from the candidate bytes")
    for filename in REMOTE_ISSUE_FILENAMES:
        relative_path = relative_directory / filename
        if (
            _remote_file_bytes(candidate_commit, relative_path)
            != (issue_directory / filename).read_bytes()
        ):
            raise ValueError(f"public forecast commit changed a frozen issue artifact: {filename}")
    return remotely_observed_at


def _install_remotely_closed_forecast_locally(
    *,
    parent_commit: str,
    candidate_commit: str,
    previous_ledger_bytes: bytes,
    candidate_ledger_bytes: bytes,
    record: Mapping[str, object],
) -> None:
    ledger_path = p1_ledger_path(REPOSITORY_ROOT)
    if ledger_path.read_bytes() != previous_ledger_bytes:
        raise ValueError("local P1 ledger changed before the remotely closed record was installed")
    append_p1_record(REPOSITORY_ROOT, record, schema=_schema())
    if ledger_path.read_bytes() != candidate_ledger_bytes:
        raise ValueError("local installed forecast ledger differs from the public candidate")
    _run_git(
        (
            "update-ref",
            f"refs/heads/{REMOTE_BRANCH_NAME}",
            candidate_commit,
            parent_commit,
        )
    )
    _run_git(("read-tree", candidate_commit))
    _run_git(
        (
            "update-ref",
            f"refs/remotes/{REMOTE_NAME}/{REMOTE_BRANCH_NAME}",
            candidate_commit,
        )
    )
    if _rev_parse("HEAD") != candidate_commit:
        raise ValueError("local branch did not fast-forward to the public forecast commit")
    if p1_ledger_path(REPOSITORY_ROOT).read_bytes() != candidate_ledger_bytes:
        raise ValueError("local ledger changed after public forecast installation")


def _read_frozen_inputs() -> tuple[bytes, bytes, bytes]:
    catalog = LOCAL_CATALOG_PATH.read_bytes()
    study_area = STUDY_AREA_PATH.read_bytes()
    support = SUPPORT_MANIFEST_PATH.read_bytes()
    checks = (
        (catalog, EXPECTED_LOCAL_CATALOG_SHA256, "local catalogue"),
        (study_area, EXPECTED_STUDY_AREA_SHA256, "study area"),
        (support, EXPECTED_SUPPORT_MANIFEST_SHA256, "support manifest"),
        (
            SOURCE_MANIFEST_PATH.read_bytes(),
            P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
            "source boundary manifest",
        ),
        (MODEL_MANIFEST_PATH.read_bytes(), P1_MODEL_MANIFEST_SHA256, "model manifest"),
    )
    for payload, expected, label in checks:
        if _sha256(payload) != expected:
            raise ValueError(f"{label} differs from the frozen P1 identity")
    return catalog, study_area, support


def _recovery_context_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    schedule: P1IssueSchedule,
    expected_code_commit: str | None,
) -> AuthorizedIssueContext:
    if not records:
        raise ValueError("recovery chain is empty")
    protocol = records[0]
    authorizations = tuple(
        record for record in records if record.get("record_type") == "RealIssueAuthorizationRecord"
    )
    if len(authorizations) != 1 or authorizations[0].get("real_issue_authorized") is not True:
        raise ValueError("recovery requires the unique active public authorization")
    authorization = authorizations[0]
    authorized_from = _parse_utc(
        cast(str, authorization["authorized_from_scheduled_issue_utc"]),
        label="authorized_from_scheduled_issue_utc",
    )
    if schedule.scheduled_issue_time_utc < authorized_from:
        raise ValueError("recovery issue predates the public authorization")
    code_commit = _git_sha(cast(str, authorization["code_commit"]), label="code_commit")
    if expected_code_commit is not None and code_commit != expected_code_commit:
        raise ValueError("recovery code differs from the requested frozen operations commit")
    return AuthorizedIssueContext(
        schedule=schedule,
        protocol_definition=dict(protocol),
        authorization_record=dict(authorization),
        code_commit=code_commit,
    )


def _parse_git_iso(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid Git timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _require_recovery_only_ledger_changes() -> None:
    for arguments in (
        ("diff", "--name-only", "--"),
        ("diff", "--cached", "--name-only", "--"),
    ):
        dirty = _run_git(arguments)
        if not isinstance(dirty, str):
            raise TypeError("Git recovery dirty-path lookup returned non-text output")
        dirty_paths = {line for line in dirty.splitlines() if line}
        if not dirty_paths.issubset({LEDGER_RELATIVE_PATH.as_posix()}):
            raise ValueError("recovery found unrelated tracked changes")


def _recover_remotely_closed_forecast_if_present(
    *,
    schedule: P1IssueSchedule,
    expected_code_commit: str | None,
) -> tuple[dict[str, object], str] | None:
    local_head = _rev_parse("HEAD")
    remote_head = _remote_branch_head()
    local_ledger_bytes = p1_ledger_path(REPOSITORY_ROOT).read_bytes()
    remote_ledger_bytes = _remote_file_bytes(remote_head, LEDGER_RELATIVE_PATH)
    remote_records = _read_validated_candidate_ledger_bytes(remote_ledger_bytes)
    remote_last = remote_records[-1]
    is_requested_forecast = (
        remote_last.get("record_type") == "ForecastIssueRecord"
        and remote_last.get("issue_id") == schedule.issue_id
        and remote_last.get("scheduled_issue_time_utc")
        == _utc_text(schedule.scheduled_issue_time_utc)
    )
    if remote_head == local_head and remote_ledger_bytes == local_ledger_bytes:
        if not is_requested_forecast:
            return None
        context = _recovery_context_from_records(
            remote_records,
            schedule=schedule,
            expected_code_commit=expected_code_commit,
        )
        _verify_operational_identity(context.code_commit)
        if remote_last.get("code_commit") != context.code_commit:
            raise ValueError("installed Forecast code differs from the public authorization")
        _require_recovery_only_ledger_changes()
        _run_git(("read-tree", remote_head))
        _run_git(
            (
                "update-ref",
                f"refs/remotes/{REMOTE_NAME}/{REMOTE_BRANCH_NAME}",
                remote_head,
            )
        )
        return dict(remote_last), remote_head
    if not is_requested_forecast:
        raise ValueError("public/local divergence is not the requested Forecast record")

    parent_commit = _rev_parse(f"{remote_head}^")
    if local_head not in {parent_commit, remote_head} or not _is_ancestor(
        parent_commit, remote_head
    ):
        raise ValueError("public branch divergence is not a single recoverable forecast child")
    changed = _run_git(
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            parent_commit,
            remote_head,
        )
    )
    if changed != LEDGER_RELATIVE_PATH.as_posix():
        raise ValueError("remote recovery commit changed more than the fixed P1 ledger")
    previous_ledger_bytes = _remote_file_bytes(parent_commit, LEDGER_RELATIVE_PATH)
    candidate_ledger_bytes = remote_ledger_bytes
    if not candidate_ledger_bytes.startswith(previous_ledger_bytes):
        raise ValueError("remote recovery ledger does not extend the exact local chain")
    if local_ledger_bytes not in {previous_ledger_bytes, candidate_ledger_bytes}:
        raise ValueError("local recovery ledger is neither the public parent nor candidate")
    parent_records = _read_validated_candidate_ledger_bytes(previous_ledger_bytes)
    candidate_records = _read_validated_candidate_ledger_bytes(candidate_ledger_bytes)
    if (
        len(candidate_records) != len(parent_records) + 1
        or candidate_records[:-1] != parent_records
    ):
        raise ValueError("remote recovery ledger is not the unique next chain record")
    record = candidate_records[-1]
    context = _recovery_context_from_records(
        candidate_records,
        schedule=schedule,
        expected_code_commit=expected_code_commit,
    )
    _verify_operational_identity(context.code_commit)
    if (
        record.get("record_type") != "ForecastIssueRecord"
        or record.get("status") != "on_time"
        or record.get("issue_id") != schedule.issue_id
        or record.get("scheduled_issue_time_utc") != _utc_text(schedule.scheduled_issue_time_utc)
        or record.get("query_cutoff_utc") != _utc_text(schedule.query_cutoff_utc)
        or record.get("code_commit") != context.code_commit
        or record.get("protocol_definition_sha256") != context.protocol_definition_sha256
        or record.get("authorization_record_sha256") != context.authorization_record_sha256
    ):
        raise ValueError("remote recovery record differs from the authorized frozen issue")
    recorded_at = _parse_utc(cast(str, record["recorded_at_utc"]), label="recorded_at_utc")
    git_times = _run_git(("show", "-s", "--format=%aI%n%cI", remote_head))
    if not isinstance(git_times, str):
        raise TypeError("Git recovery timestamp lookup returned non-text output")
    time_lines = git_times.splitlines()
    if len(time_lines) != 2 or any(
        _parse_git_iso(value, label="forecast commit time") != recorded_at for value in time_lines
    ):
        raise ValueError("remote forecast commit time differs from its actual record time")
    if recorded_at >= schedule.scheduled_issue_time_utc:
        raise ValueError("remote recovery Forecast record was not created before T")
    if datetime.now(UTC) >= schedule.scheduled_issue_time_utc:
        raise ValueError(
            "post-T remote/local Forecast divergence requires manual integrity review; "
            "a missed record is forbidden"
        )

    issue_directory = ISSUE_PARENT / schedule.issue_id
    relative_directory = issue_directory.resolve().relative_to(REPOSITORY_ROOT.resolve())
    for filename in REMOTE_ISSUE_FILENAMES:
        relative_path = relative_directory / filename
        if (
            _remote_file_bytes(remote_head, relative_path)
            != (issue_directory / filename).read_bytes()
        ):
            raise ValueError(f"remote recovery issue artifact differs: {filename}")
    local_catalog, study_area, support = _read_frozen_inputs()
    verified = verify_prepared_issue_against_frozen_inputs(
        issue_directory,
        local_catalog_bytes=local_catalog,
        study_area_bytes=study_area,
        support_manifest_bytes=support,
    )
    expected_fields = build_forecast_issue_record_fields(
        verified,
        protocol_definition_sha256=context.protocol_definition_sha256,
        authorization_record_sha256=context.authorization_record_sha256,
        publication_completed_at_utc=_parse_utc(
            cast(str, record["publication_completed_at_utc"]),
            label="publication_completed_at_utc",
        ),
        recorded_at_utc=recorded_at,
    )
    if any(record.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("remote recovery Forecast record differs from frozen artifact replay")
    _require_recovery_only_ledger_changes()
    if local_ledger_bytes == previous_ledger_bytes:
        append_p1_record(REPOSITORY_ROOT, record, schema=_schema())
    if p1_ledger_path(REPOSITORY_ROOT).read_bytes() != candidate_ledger_bytes:
        raise ValueError("recovered local ledger differs from the public Forecast chain")
    if local_head == parent_commit:
        _run_git(
            (
                "update-ref",
                f"refs/heads/{REMOTE_BRANCH_NAME}",
                remote_head,
                parent_commit,
            )
        )
    _run_git(("read-tree", remote_head))
    _run_git(
        (
            "update-ref",
            f"refs/remotes/{REMOTE_NAME}/{REMOTE_BRANCH_NAME}",
            remote_head,
        )
    )
    if _rev_parse("HEAD") != remote_head:
        raise ValueError("local recovery did not converge to the public Forecast commit")
    return dict(record), remote_head


def _scheduled_time(value: str) -> datetime:
    return issue_schedule(_parse_utc(value, label="scheduled issue time")).scheduled_issue_time_utc


def _command_status(_: argparse.Namespace) -> int:
    ledger = p1_ledger_path(REPOSITORY_ROOT)
    if not ledger.exists():
        _write_output(
            {
                "ledger_exists": False,
                "record_count": 0,
                "real_issue_authorized": False,
                "next_scheduled_issue_utc": "2026-09-09T16:00:00Z",
                "prediction_effect_evidence": "none_prospective_yet",
            }
        )
        return 0
    records = read_p1_ledger(REPOSITORY_ROOT, schema=_schema(), require_exists=True)
    scheduled_records = tuple(
        record
        for record in records
        if record.get("record_type") in {"ForecastIssueRecord", "MissedIssueRecord"}
    )
    if scheduled_records:
        last = _parse_utc(
            cast(str, scheduled_records[-1]["scheduled_issue_time_utc"]),
            label="last scheduled issue",
        )
        next_scheduled = last + timedelta(days=7)
    else:
        next_scheduled = _parse_utc(cast(str, records[0]["valid_from_utc"]), label="valid_from_utc")
    authorization = next(
        (
            record
            for record in records
            if record.get("record_type") == "RealIssueAuthorizationRecord"
        ),
        None,
    )
    _write_output(
        {
            "ledger_exists": True,
            "record_count": len(records),
            "chain_head_type": records[-1]["record_type"],
            "chain_head_sha256": records[-1]["content_sha256"],
            "real_issue_authorized": authorization is not None,
            "authorization_record_sha256": (
                None if authorization is None else authorization["content_sha256"]
            ),
            "next_scheduled_issue_utc": _utc_text(next_scheduled),
            "forecast_issue_count": sum(
                record.get("record_type") == "ForecastIssueRecord" for record in records
            ),
            "prediction_effect_evidence": "await_future_30_and_90_day_truth",
        }
    )
    return 0


def _command_append_protocol(_: argparse.Namespace) -> int:
    _verify_frozen_tags()
    recorded = datetime.now(UTC)
    if recorded >= _parse_utc("2026-09-09T16:00:00Z", label="valid_from_utc"):
        raise ValueError("the protocol genesis cannot be backdated after its first valid issue")
    record = append_new_p1_record(
        REPOSITORY_ROOT,
        "ProtocolDefinition",
        recorded_at_utc=_utc_text(recorded),
        fields={
            "protocol_id": "p1-b0-r30-prospective-v1",
            "protocol_tag": PROTOCOL_TAG,
            "code_tag": SCIENTIFIC_CODE_TAG,
            "valid_from_utc": "2026-09-09T16:00:00Z",
            "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
            "source_boundary_manifest_sha256": P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
            "model_manifest_sha256": P1_MODEL_MANIFEST_SHA256,
            "protocol_commit": PROTOCOL_COMMIT,
            "real_issue_authorized": False,
        },
        schema=_schema(),
    )
    _write_output(
        {
            "status": "protocol_definition_appended",
            "content_sha256": record["content_sha256"],
            "recorded_at_utc": record["recorded_at_utc"],
        }
    )
    return 0


def _verify_authorization_commit(
    authorization_commit: str,
    *,
    code_commit: str,
) -> tuple[str, datetime]:
    commit = _git_sha(authorization_commit, label="authorization_commit")
    remote_head = _remote_branch_head()
    _verify_remote_operations_tag(code_commit)
    for ancestor, label in (
        (commit, "authorization evidence commit"),
        (code_commit, "authorized operations commit"),
    ):
        if not _is_ancestor(ancestor, remote_head):
            raise ValueError(f"{label} is not present on the actual public branch")
    ledger_at_commit = _remote_file_bytes(commit, LEDGER_RELATIVE_PATH)
    if (
        ledger_at_commit != p1_ledger_path(REPOSITORY_ROOT).read_bytes()
        or _remote_file_bytes(remote_head, LEDGER_RELATIVE_PATH) != ledger_at_commit
    ):
        raise ValueError("actual public authorization commit does not contain the genesis chain")
    evidence_at_commit = _remote_file_bytes(commit, AUTHORIZATION_EVIDENCE_RELATIVE_PATH)
    if (
        AUTHORIZED_QUOTE.encode("utf-8") not in evidence_at_commit
        or _remote_file_bytes(remote_head, AUTHORIZATION_EVIDENCE_RELATIVE_PATH)
        != evidence_at_commit
    ):
        raise ValueError("actual public authorization evidence omits the user's verbatim authority")
    return remote_head, datetime.now(UTC)


def _command_append_authorization(arguments: argparse.Namespace) -> int:
    code_commit = _git_sha(arguments.code_commit, label="code_commit")
    authorization_commit = _git_sha(arguments.authorization_commit, label="authorization_commit")
    _verify_operational_identity(code_commit)
    remote_head, remote_verified = _verify_authorization_commit(
        authorization_commit,
        code_commit=code_commit,
    )
    authorized_from = _scheduled_time(arguments.authorized_from_scheduled_issue_utc)
    recorded = datetime.now(UTC)
    if not remote_verified <= recorded < authorized_from:
        raise ValueError("authorization must be remotely verified and recorded before its first T")
    records = read_p1_ledger(REPOSITORY_ROOT, schema=_schema(), require_exists=True)
    protocol = records[0]
    record = append_new_p1_record(
        REPOSITORY_ROOT,
        "RealIssueAuthorizationRecord",
        recorded_at_utc=_utc_text(recorded),
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_commit": authorization_commit,
            "code_commit": code_commit,
            "remote_verified_at_utc": _utc_text(remote_verified),
            "authorized_from_scheduled_issue_utc": _utc_text(authorized_from),
            "real_issue_authorized": True,
        },
        schema=_schema(),
    )
    _write_output(
        {
            "status": "real_issue_authorization_appended",
            "content_sha256": record["content_sha256"],
            "authorized_from_scheduled_issue_utc": record["authorized_from_scheduled_issue_utc"],
            "code_commit": code_commit,
            "public_authorization_commit": remote_head,
            "public_authorization_verified_at_utc": _utc_text(remote_verified),
        }
    )
    return 0


def _command_prepare(arguments: argparse.Namespace) -> int:
    scheduled = _scheduled_time(arguments.scheduled_issue_time_utc)
    schedule = issue_schedule(scheduled)
    code_commit = _git_sha(arguments.code_commit, label="code_commit")
    _verify_operational_identity(code_commit)
    local_catalog, study_area, support = _read_frozen_inputs()
    validate_washout(schedule)
    context = load_authorized_issue_context(
        REPOSITORY_ROOT,
        schedule=schedule,
        code_commit=code_commit,
    )
    ISSUE_PARENT.mkdir(parents=True, exist_ok=True)
    if (ISSUE_PARENT / schedule.issue_id).exists():
        raise ValueError("the write-once issue directory already exists")
    remote_checks: list[tuple[str, datetime]] = []

    def pre_request_gate() -> None:
        _verify_operational_identity(code_commit)
        refreshed_context = load_authorized_issue_context(
            REPOSITORY_ROOT,
            schedule=schedule,
            code_commit=code_commit,
        )
        remote_checks.append(
            _verify_remote_ledger_anchor(
                context=refreshed_context,
                code_commit=code_commit,
            )
        )

    transport = _build_guarded_time_gated_comcat_transport(
        schedule=schedule,
        pre_request_gate=pre_request_gate,
        timeout_seconds=arguments.timeout_seconds,
    )
    acquisition = acquire_issue_input(schedule=schedule, transport=transport)
    if acquisition.status != "available":
        _write_output(
            {
                "status": acquisition.status,
                "issue_id": schedule.issue_id,
                "count_snapshot": acquisition.count_snapshot.as_mapping(),
                "prediction_generated": False,
                "must_record_missed_after_T": True,
            }
        )
        return 3
    bundle = build_production_forecast(
        schedule=schedule,
        acquisition=acquisition,
        local_catalog_bytes=local_catalog,
        study_area_bytes=study_area,
        support_manifest_bytes=support,
    )
    created = datetime.now(UTC)
    prepared = prepare_production_issue_artifacts(
        bundle,
        issue_parent=ISSUE_PARENT,
        code_commit=context.code_commit,
        forecast_created_at_utc=created,
    )
    if len(remote_checks) != 2:
        raise ValueError("count and query did not each pass the actual public ledger gate")
    remote_head, remote_verified = remote_checks[-1]
    _write_output(
        {
            "status": "prepared_before_T_not_yet_a_ledger_forecast",
            "issue_id": schedule.issue_id,
            "issue_directory": prepared.issue_directory.relative_to(REPOSITORY_ROOT).as_posix(),
            "forecast_created_at_utc": _utc_text(created),
            "prepared_receipt_sha256": prepared.prepared_receipt_sha256,
            "artifact_manifest_sha256": prepared.artifact_manifest_sha256,
            "B0_source_count": bundle.b0_source_count,
            "R30_source_count": bundle.recent_source_count,
            "B0_actual_area_km2": bundle.b0_alarm.actual_area_km2,
            "B0_R30_actual_area_km2": bundle.challenger_alarm.actual_area_km2,
            "public_ledger_commit": remote_head,
            "public_ledger_verified_at_utc": _utc_text(remote_verified),
            "future_outcomes_read": False,
        }
    )
    return 0


def _command_verify(arguments: argparse.Namespace) -> int:
    schedule = issue_schedule(_scheduled_time(arguments.scheduled_issue_time_utc))
    local_catalog, study_area, support = _read_frozen_inputs()
    verified = verify_prepared_issue_against_frozen_inputs(
        ISSUE_PARENT / schedule.issue_id,
        local_catalog_bytes=local_catalog,
        study_area_bytes=study_area,
        support_manifest_bytes=support,
    )
    _write_output(
        {
            "status": "raw_to_forecast_replay_verified",
            "issue_id": verified.issue_id,
            "source_snapshot_sha256": verified.source_snapshot_sha256,
            "artifact_manifest_sha256": verified.artifact_manifest_sha256,
            "code_commit": verified.code_commit,
        }
    )
    return 0


def _command_seal(arguments: argparse.Namespace) -> int:
    scheduled = _scheduled_time(arguments.scheduled_issue_time_utc)
    schedule = issue_schedule(scheduled)
    code_commit = _git_sha(arguments.code_commit, label="code_commit")
    _verify_operational_identity(code_commit)
    recovered = _recover_remotely_closed_forecast_if_present(
        schedule=schedule,
        expected_code_commit=code_commit,
    )
    if recovered is not None:
        recovered_record, recovered_commit = recovered
        _write_output(
            {
                "status": "forecast_record_recovered_from_public_branch",
                "issue_id": schedule.issue_id,
                "content_sha256": recovered_record["content_sha256"],
                "public_forecast_record_commit": recovered_commit,
                "missed_record_appended": False,
            }
        )
        return 0
    context = load_authorized_issue_context(
        REPOSITORY_ROOT,
        schedule=schedule,
        code_commit=code_commit,
    )
    local_catalog, study_area, support = _read_frozen_inputs()
    verified = verify_prepared_issue_against_frozen_inputs(
        ISSUE_PARENT / schedule.issue_id,
        local_catalog_bytes=local_catalog,
        study_area_bytes=study_area,
        support_manifest_bytes=support,
    )
    if verified.issue_id != schedule.issue_id or verified.code_commit != context.code_commit:
        raise ValueError("prepared issue differs from the authorized schedule or code")
    remote_head, publication_completed = _verify_remote_issue_publication(
        context=context,
        code_commit=code_commit,
        issue_directory=ISSUE_PARENT / schedule.issue_id,
    )
    _require_clean_tracked_state(remote_head)
    previous_ledger_bytes = p1_ledger_path(REPOSITORY_ROOT).read_bytes()
    if _remote_file_bytes(remote_head, LEDGER_RELATIVE_PATH) != previous_ledger_bytes:
        raise ValueError("local ledger differs from the verified artifact commit parent")
    now = datetime.now(UTC).replace(microsecond=0)
    fields = build_forecast_issue_record_fields(
        verified,
        protocol_definition_sha256=context.protocol_definition_sha256,
        authorization_record_sha256=context.authorization_record_sha256,
        publication_completed_at_utc=publication_completed,
        recorded_at_utc=now,
    )
    record = build_next_p1_record(
        REPOSITORY_ROOT,
        "ForecastIssueRecord",
        recorded_at_utc=_utc_text(now),
        fields=fields,
        schema=_schema(),
    )
    if p1_ledger_path(REPOSITORY_ROOT).read_bytes() != previous_ledger_bytes:
        raise ValueError("local P1 ledger changed while the next Forecast record was built")
    candidate_ledger_bytes = previous_ledger_bytes + canonical_json_bytes(record) + b"\n"
    _validate_candidate_ledger_bytes(candidate_ledger_bytes, expected_record=record)
    candidate_commit = _build_ledger_only_commit(
        parent_commit=remote_head,
        candidate_ledger_bytes=candidate_ledger_bytes,
        issue_id=schedule.issue_id,
        recorded_at_utc=now,
    )
    remotely_closed_at = _push_forecast_commit_before_t(
        parent_commit=remote_head,
        candidate_commit=candidate_commit,
        previous_ledger_bytes=previous_ledger_bytes,
        candidate_ledger_bytes=candidate_ledger_bytes,
        issue_directory=ISSUE_PARENT / schedule.issue_id,
        scheduled_issue_time_utc=schedule.scheduled_issue_time_utc,
    )
    _install_remotely_closed_forecast_locally(
        parent_commit=remote_head,
        candidate_commit=candidate_commit,
        previous_ledger_bytes=previous_ledger_bytes,
        candidate_ledger_bytes=candidate_ledger_bytes,
        record=record,
    )
    _write_output(
        {
            "status": "forecast_record_remotely_closed_on_time",
            "issue_id": schedule.issue_id,
            "content_sha256": record["content_sha256"],
            "recorded_at_utc": record["recorded_at_utc"],
            "scheduled_issue_time_utc": record["scheduled_issue_time_utc"],
            "public_issue_artifact_commit": remote_head,
            "public_issue_verified_at_utc": _utc_text(publication_completed),
            "public_forecast_record_commit": candidate_commit,
            "public_forecast_record_closed_at_utc": _utc_text(remotely_closed_at),
        }
    )
    return 0


def _command_append_missed(arguments: argparse.Namespace) -> int:
    schedule = issue_schedule(_scheduled_time(arguments.scheduled_issue_time_utc))
    recorded = datetime.now(UTC)
    if recorded < schedule.scheduled_issue_time_utc:
        raise ValueError("a missed issue can be recorded only at or after T")
    recovered = _recover_remotely_closed_forecast_if_present(
        schedule=schedule,
        expected_code_commit=None,
    )
    if recovered is not None:
        recovered_record, recovered_commit = recovered
        _write_output(
            {
                "status": "forecast_record_recovered_no_missed_record_allowed",
                "issue_id": schedule.issue_id,
                "content_sha256": recovered_record["content_sha256"],
                "public_forecast_record_commit": recovered_commit,
                "missed_record_appended": False,
            }
        )
        return 0
    records = read_p1_ledger(REPOSITORY_ROOT, schema=_schema(), require_exists=True)
    authorization = next(
        (
            record
            for record in records
            if record.get("record_type") == "RealIssueAuthorizationRecord"
        ),
        None,
    )
    authorized = False
    authorization_sha: object = None
    if authorization is not None:
        authorized_from = _parse_utc(
            cast(str, authorization["authorized_from_scheduled_issue_utc"]),
            label="authorized_from_scheduled_issue_utc",
        )
        authorized = schedule.scheduled_issue_time_utc >= authorized_from
        if authorized:
            authorization_sha = authorization["content_sha256"]
            code_commit = _git_sha(cast(str, authorization["code_commit"]), label="code_commit")
            _verify_operational_identity(code_commit)
    if _remote_branch_head() != _rev_parse("HEAD"):
        raise ValueError("public branch differs from the local chain; missed record is forbidden")
    record = append_new_p1_record(
        REPOSITORY_ROOT,
        "MissedIssueRecord",
        recorded_at_utc=_utc_text(recorded),
        fields={
            "issue_id": schedule.issue_id,
            "status": "missed_issue",
            "scheduled_issue_time_utc": _utc_text(schedule.scheduled_issue_time_utc),
            "authorization_state": "authorized" if authorized else "not_authorized",
            "authorization_record_sha256": authorization_sha,
            "reason": arguments.reason,
            "prediction_generated": False,
            "backfill_forbidden": True,
            "valid_from_remains_fixed": True,
        },
        schema=_schema(),
    )
    _write_output(
        {
            "status": "missed_issue_appended_no_backfill_allowed",
            "issue_id": schedule.issue_id,
            "content_sha256": record["content_sha256"],
            "reason": record["reason"],
        }
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="validate and report the public P1 chain")
    status.set_defaults(handler=_command_status)

    protocol = subparsers.add_parser(
        "append-protocol", help="append the fixed ProtocolDefinition genesis"
    )
    protocol.set_defaults(handler=_command_append_protocol)

    authorization = subparsers.add_parser(
        "append-authorization", help="append a remotely evidenced real-issue authorization"
    )
    authorization.add_argument("--authorization-commit", required=True)
    authorization.add_argument("--code-commit", required=True)
    authorization.add_argument("--authorized-from-scheduled-issue-utc", required=True)
    authorization.set_defaults(handler=_command_append_authorization)

    prepare = subparsers.add_parser(
        "prepare", help="within [Q,T), fetch ComCat and prepare one write-once issue"
    )
    prepare.add_argument("--scheduled-issue-time-utc", required=True)
    prepare.add_argument("--code-commit", required=True)
    prepare.add_argument("--timeout-seconds", type=float, default=45.0)
    prepare.set_defaults(handler=_command_prepare)

    verify = subparsers.add_parser(
        "verify", help="replay one prepared issue from exact raw bytes without network"
    )
    verify.add_argument("--scheduled-issue-time-utc", required=True)
    verify.set_defaults(handler=_command_verify)

    seal = subparsers.add_parser(
        "seal", help="after artifact remote readback, append the ForecastIssueRecord"
    )
    seal.add_argument("--scheduled-issue-time-utc", required=True)
    seal.add_argument("--code-commit", required=True)
    seal.set_defaults(handler=_command_seal)

    missed = subparsers.add_parser(
        "append-missed", help="after T, append the next immutable missed issue fact"
    )
    missed.add_argument("--scheduled-issue-time-utc", required=True)
    missed.add_argument(
        "--reason",
        required=True,
        choices=(
            "protocol_not_remotely_closed_before_T",
            "code_not_remotely_closed_before_T",
            "real_issue_not_authorized_before_T",
            "source_snapshot_unavailable_before_T",
            "forecast_not_frozen_before_T",
        ),
    )
    missed.set_defaults(handler=_command_append_missed)
    return parser


def main() -> int:
    parser = _build_parser()
    arguments = parser.parse_args()
    handler = cast(object, arguments.handler)
    if not callable(handler):
        raise TypeError("selected command handler is not callable")
    try:
        return int(handler(arguments))
    except Exception as exc:
        sys.stderr.buffer.write(
            canonical_json_bytes(
                {
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            + b"\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
