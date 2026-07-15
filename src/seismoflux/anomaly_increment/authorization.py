"""Repository-bound scoring seal and target-access authorization for stage 4.

This module proves code, tags, the live public remote, score-blind inputs, synthetic
qualification, and empty local ledgers without inspecting the earthquake target.  It
can grant a capability for the sole target entrance, but it never opens that entrance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, Protocol, cast

from seismoflux.anomaly_increment.attempt_ledger import (
    Stage4AuditLedger,
    Stage4LedgerError,
    read_stage4_ledger,
)
from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_PREFLIGHT_RECEIPT_PATH,
    FormalPreflightReceipt,
    load_formal_preflight_receipt,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
)
from seismoflux.anomaly_increment.preregistration import (
    protocol_design_sha256,
    verify_content_sha256,
    with_content_sha256,
)
from seismoflux.anomaly_increment.qualification import (
    STAGE4_PROTOCOL_VERSION,
    GpuQualificationStatus,
    ScoreBlindInputEvidence,
    Stage4QualificationEvidence,
    expected_target_identity_from_protocol,
    observe_score_blind_inputs,
    validate_stage4_qualification_against_formal_preflight,
    validate_stage4_qualification_against_protocol,
)
from seismoflux.anomaly_increment.score_blind_path import (
    require_score_blind_project_path,
)
from seismoflux.background.execution import (
    CommandResult,
    GitCommandRunner,
    subprocess_git_runner,
)
from seismoflux.data.common import canonical_json_bytes

STAGE4_SCORING_SEAL_SCHEMA_VERSION: Final[int] = 2
STAGE4_PUBLIC_REPOSITORY: Final[str] = "github.com/Justin-147/SeismoFlux"
STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH: Final[str] = (
    "data/manifests/anomaly_increment_attempt_ledger.json"
)
STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH: Final[str] = (
    "data/manifests/anomaly_increment_target_read_ledger.json"
)
STAGE4_FROZEN_PROTOCOL_PATHS: Final[tuple[str, ...]] = (
    "configs/anomaly_increment.yaml",
    "data/manifests/anomaly_increment_feature_set.json",
    "data/manifests/anomaly_increment_fold_manifest.json",
    "data/manifests/anomaly_increment_randomness.json",
    "data/manifests/anomaly_increment_spatial_strata.json",
    "docs/anomaly_increment_protocol.md",
)

_GIT_OID_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_AUTHORIZATION_SENTINEL = object()
_EXPECTED_TARGET_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "content_sha256",
    "contract_path",
    "contract_sha256",
    "path",
    "physical_event_id_column",
    "schema_sha256",
    "sha256",
)


class Stage4ScoringNotAuthorizedError(RuntimeError):
    """Raised before target access when any frozen identity cannot be proven."""


@dataclass(frozen=True, slots=True)
class PublicRepositoryEvidence:
    """Unauthenticated GitHub API proof that the authorized repository is public."""

    full_name: str
    visibility: str
    private: bool
    html_url: str

    def __post_init__(self) -> None:
        if self.full_name.casefold() != "justin-147/seismoflux":
            raise ValueError("public evidence names another repository")
        if self.private or self.visibility != "public":
            raise ValueError("GitHub repository is not publicly visible")
        if self.html_url.casefold().rstrip("/") != ("https://github.com/justin-147/seismoflux"):
            raise ValueError("public evidence has another repository URL")

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "access": "unauthenticated_github_rest_api",
                "full_name": self.full_name,
                "html_url": self.html_url,
                "private": self.private,
                "user_publication_authorization_recorded": True,
                "visibility": self.visibility,
            }
        )

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PublicRepositoryEvidence:
        expected = {
            "access",
            "content_sha256",
            "full_name",
            "html_url",
            "private",
            "user_publication_authorization_recorded",
            "visibility",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise Stage4ScoringNotAuthorizedError("public repository evidence is invalid")
        if (
            value.get("access") != "unauthenticated_github_rest_api"
            or value.get("user_publication_authorization_recorded") is not True
            or not isinstance(value.get("private"), bool)
        ):
            raise Stage4ScoringNotAuthorizedError("public repository evidence changed")
        try:
            return cls(
                full_name=cast(str, value["full_name"]),
                visibility=cast(str, value["visibility"]),
                private=cast(bool, value["private"]),
                html_url=cast(str, value["html_url"]),
            )
        except (TypeError, ValueError) as exc:
            raise Stage4ScoringNotAuthorizedError("repository is not proven public") from exc


class PublicRepositoryProbe(Protocol):
    def observe(self) -> PublicRepositoryEvidence: ...


@dataclass(frozen=True, slots=True)
class GitHubUnauthenticatedPublicProbe:
    """Read the public REST endpoint without an Authorization header."""

    timeout_seconds: float = 15.0

    def observe(self) -> PublicRepositoryEvidence:
        request = urllib.request.Request(
            "https://api.github.com/repos/Justin-147/SeismoFlux",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "SeismoFlux-stage4-public-proof",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="GET",
        )
        if request.has_header("Authorization"):
            raise Stage4ScoringNotAuthorizedError("public probe must not use credentials")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise Stage4ScoringNotAuthorizedError(
                "unable to prove public repository visibility without credentials"
            ) from exc
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Stage4ScoringNotAuthorizedError(
                "unauthenticated GitHub visibility response is malformed"
            ) from exc
        document = _mapping(raw, label="unauthenticated GitHub repository response")
        try:
            return PublicRepositoryEvidence(
                full_name=cast(str, document["full_name"]),
                visibility=cast(str, document["visibility"]),
                private=cast(bool, document["private"]),
                html_url=cast(str, document["html_url"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Stage4ScoringNotAuthorizedError(
                "unauthenticated GitHub response does not prove a public repository"
            ) from exc


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage4ScoringNotAuthorizedError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Stage4ScoringNotAuthorizedError(f"{label} must be a lowercase SHA-256 string")
    return value


def _oid(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_OID_PATTERN.fullmatch(value) is None:
        raise Stage4ScoringNotAuthorizedError(f"{label} must be a lowercase Git object ID")
    return value


def _relative_posix(value: str, *, label: str) -> str:
    if not value or "\\" in value:
        raise Stage4ScoringNotAuthorizedError(f"{label} is not a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise Stage4ScoringNotAuthorizedError(f"{label} is not a normalized relative path")
    return value


def _canonical_remote_repository(value: str) -> str:
    patterns = (
        re.compile(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?"),
        re.compile(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?"),
        re.compile(r"ssh://git@github\.com/([^/]+)/([^/]+?)(?:\.git)?"),
    )
    match = None
    for pattern in patterns:
        match = pattern.fullmatch(value.strip())
        if match is not None:
            break
    if match is None:
        raise Stage4ScoringNotAuthorizedError(
            "stage-4 scoring remote is not an authorized public GitHub repository"
        )
    owner, repository = match.groups()
    if (owner.casefold(), repository.casefold()) != ("justin-147", "seismoflux"):
        raise Stage4ScoringNotAuthorizedError("stage-4 scoring remote is not Justin-147/SeismoFlux")
    return STAGE4_PUBLIC_REPOSITORY


def _validate_expected_target_identity(values: tuple[tuple[str, str], ...]) -> None:
    names = tuple(name for name, _ in values)
    if names != _EXPECTED_TARGET_IDENTITY_FIELDS:
        raise ValueError("expected target identity fields changed")
    identity = dict(values)
    _relative_posix(identity["path"], label="expected target path")
    _relative_posix(identity["contract_path"], label="expected target contract path")
    for field_name in ("content_sha256", "contract_sha256", "schema_sha256", "sha256"):
        _sha256(identity[field_name], label=f"expected target {field_name}")
    if identity["physical_event_id_column"] != "event_id":
        raise ValueError("expected target physical-event identity changed")


@dataclass(frozen=True, slots=True)
class Stage4RepositoryEvidence:
    """Exact local and live-remote identity of the scoring-code freeze."""

    code_commit: str
    branch: str
    upstream: str
    remote: str
    remote_repository: str
    remote_branch_ref: str
    remote_branch_commit: str
    public_repository: PublicRepositoryEvidence
    protocol_tag: str
    protocol_tag_object: str
    protocol_tag_commit: str
    scoring_code_tag: str
    scoring_code_tag_object: str
    scoring_code_tag_commit: str
    frozen_blob_ids: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("code_commit", self.code_commit),
            ("remote_branch_commit", self.remote_branch_commit),
            ("protocol_tag_object", self.protocol_tag_object),
            ("protocol_tag_commit", self.protocol_tag_commit),
            ("scoring_code_tag_object", self.scoring_code_tag_object),
            ("scoring_code_tag_commit", self.scoring_code_tag_commit),
        ):
            _oid(value, label=label)
        for label, value in (
            ("branch", self.branch),
            ("upstream", self.upstream),
            ("remote", self.remote),
            ("remote_branch_ref", self.remote_branch_ref),
            ("protocol_tag", self.protocol_tag),
            ("scoring_code_tag", self.scoring_code_tag),
        ):
            if not value or value != value.strip():
                raise ValueError(f"{label} must be a non-empty trimmed string")
        if self.remote_repository != STAGE4_PUBLIC_REPOSITORY:
            raise ValueError("repository evidence uses another public repository")
        if not isinstance(self.public_repository, PublicRepositoryEvidence):
            raise TypeError("repository evidence requires unauthenticated public proof")
        if self.scoring_code_tag_commit != self.code_commit:
            raise ValueError("scoring-code tag must point exactly at the scoring commit")
        if self.remote_branch_commit != self.code_commit:
            raise ValueError("live remote branch must point exactly at the scoring commit")
        paths = tuple(path for path, _ in self.frozen_blob_ids)
        if paths != STAGE4_FROZEN_PROTOCOL_PATHS:
            raise ValueError("frozen protocol Git blob set changed")
        for path, blob_id in self.frozen_blob_ids:
            _relative_posix(path, label="frozen protocol path")
            _oid(blob_id, label=f"frozen protocol blob {path}")

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "branch": self.branch,
                "code_commit": self.code_commit,
                "frozen_blob_ids": dict(self.frozen_blob_ids),
                "protocol_tag": self.protocol_tag,
                "protocol_tag_commit": self.protocol_tag_commit,
                "protocol_tag_object": self.protocol_tag_object,
                "public_repository": self.public_repository.as_mapping(),
                "remote": self.remote,
                "remote_branch_commit": self.remote_branch_commit,
                "remote_branch_ref": self.remote_branch_ref,
                "remote_repository": self.remote_repository,
                "scoring_code_tag": self.scoring_code_tag,
                "scoring_code_tag_commit": self.scoring_code_tag_commit,
                "scoring_code_tag_object": self.scoring_code_tag_object,
                "upstream": self.upstream,
            }
        )

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Stage4RepositoryEvidence:
        expected = {
            "branch",
            "code_commit",
            "content_sha256",
            "frozen_blob_ids",
            "protocol_tag",
            "protocol_tag_commit",
            "protocol_tag_object",
            "public_repository",
            "remote",
            "remote_branch_commit",
            "remote_branch_ref",
            "remote_repository",
            "scoring_code_tag",
            "scoring_code_tag_commit",
            "scoring_code_tag_object",
            "upstream",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise Stage4ScoringNotAuthorizedError("repository evidence hash or schema is invalid")
        blobs = _mapping(value.get("frozen_blob_ids"), label="frozen_blob_ids")
        if set(blobs) != set(STAGE4_FROZEN_PROTOCOL_PATHS):
            raise Stage4ScoringNotAuthorizedError("repository frozen blob set changed")
        try:
            return cls(
                code_commit=cast(str, value["code_commit"]),
                branch=cast(str, value["branch"]),
                upstream=cast(str, value["upstream"]),
                remote=cast(str, value["remote"]),
                remote_repository=cast(str, value["remote_repository"]),
                remote_branch_ref=cast(str, value["remote_branch_ref"]),
                remote_branch_commit=cast(str, value["remote_branch_commit"]),
                public_repository=PublicRepositoryEvidence.from_mapping(
                    _mapping(value["public_repository"], label="public_repository")
                ),
                protocol_tag=cast(str, value["protocol_tag"]),
                protocol_tag_object=cast(str, value["protocol_tag_object"]),
                protocol_tag_commit=cast(str, value["protocol_tag_commit"]),
                scoring_code_tag=cast(str, value["scoring_code_tag"]),
                scoring_code_tag_object=cast(str, value["scoring_code_tag_object"]),
                scoring_code_tag_commit=cast(str, value["scoring_code_tag_commit"]),
                frozen_blob_ids=tuple(
                    (path, cast(str, blobs[path])) for path in STAGE4_FROZEN_PROTOCOL_PATHS
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Stage4ScoringNotAuthorizedError(
                "repository evidence invariants are invalid"
            ) from exc


class Stage4RepositoryAdapter(Protocol):
    """Adapter boundary for local Git and live-remote identity verification."""

    def observe(
        self,
        project_root: Path,
        *,
        protocol_tag: str,
        scoring_code_tag: str,
        allowed_untracked_paths: Sequence[str] = (),
    ) -> Stage4RepositoryEvidence: ...


@dataclass(frozen=True, slots=True)
class GitStage4RepositoryAdapter:
    """Git-backed implementation of :class:`Stage4RepositoryAdapter`."""

    runner: GitCommandRunner = subprocess_git_runner
    public_probe: PublicRepositoryProbe = field(default_factory=GitHubUnauthenticatedPublicProbe)

    def _git(self, root: Path, *arguments: str, description: str) -> str:
        try:
            result = self.runner(("git", *arguments), root)
        except Exception as exc:
            raise Stage4ScoringNotAuthorizedError(f"unable to {description}") from exc
        if not isinstance(result, CommandResult):
            raise TypeError("Git runner must return CommandResult")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            suffix = f": {detail}" if detail else ""
            raise Stage4ScoringNotAuthorizedError(f"unable to {description}{suffix}")
        return result.stdout.strip()

    @staticmethod
    def _require_clean_status(payload: str, allowed_untracked_paths: Sequence[str]) -> None:
        allowed = {
            _relative_posix(item, label="allowed untracked path")
            for item in allowed_untracked_paths
        }
        unexpected: list[str] = []
        for line in payload.splitlines():
            if not line:
                continue
            if line.startswith("?? ") and line[3:] in allowed:
                continue
            unexpected.append(line)
        if unexpected:
            raise Stage4ScoringNotAuthorizedError(
                "stage-4 scoring repository must be clean except for the local scoring seal"
            )

    def observe(
        self,
        project_root: Path,
        *,
        protocol_tag: str,
        scoring_code_tag: str,
        allowed_untracked_paths: Sequence[str] = (),
    ) -> Stage4RepositoryEvidence:
        root = Path(project_root).resolve()
        if (
            self._git(root, "rev-parse", "--is-inside-work-tree", description="verify Git worktree")
            != "true"
        ):
            raise Stage4ScoringNotAuthorizedError("project root is not a Git worktree")
        code_commit = _oid(
            self._git(root, "rev-parse", "--verify", "HEAD^{commit}", description="resolve HEAD"),
            label="HEAD",
        )
        status = self._git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            description="inspect worktree state",
        )
        self._require_clean_status(status, allowed_untracked_paths)
        branch = self._git(
            root,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            description="resolve current branch",
        )
        upstream = self._git(
            root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            description="resolve branch upstream",
        )
        if "/" not in upstream:
            raise Stage4ScoringNotAuthorizedError("stage-4 branch upstream is malformed")
        remote, upstream_branch = upstream.split("/", maxsplit=1)
        upstream_commit = _oid(
            self._git(
                root,
                "rev-parse",
                "--verify",
                "@{upstream}^{commit}",
                description="resolve upstream commit",
            ),
            label="upstream commit",
        )
        if upstream_commit != code_commit:
            raise Stage4ScoringNotAuthorizedError(
                "stage-4 scoring commit has not been pushed to its upstream branch"
            )

        protocol_tag_object = _oid(
            self._git(
                root,
                "rev-parse",
                "--verify",
                f"refs/tags/{protocol_tag}^{{tag}}",
                description="resolve annotated stage-4 protocol tag",
            ),
            label="protocol tag object",
        )
        protocol_tag_commit = _oid(
            self._git(
                root,
                "rev-parse",
                "--verify",
                f"refs/tags/{protocol_tag}^{{commit}}",
                description="resolve stage-4 protocol commit",
            ),
            label="protocol tag commit",
        )
        self._git(
            root,
            "merge-base",
            "--is-ancestor",
            protocol_tag_commit,
            code_commit,
            description="verify protocol tag is an ancestor of scoring code",
        )
        scoring_tag_object = _oid(
            self._git(
                root,
                "rev-parse",
                "--verify",
                f"refs/tags/{scoring_code_tag}^{{tag}}",
                description="resolve annotated stage-4 scoring-code tag",
            ),
            label="scoring-code tag object",
        )
        scoring_tag_commit = _oid(
            self._git(
                root,
                "rev-parse",
                "--verify",
                f"refs/tags/{scoring_code_tag}^{{commit}}",
                description="resolve stage-4 scoring-code commit",
            ),
            label="scoring-code tag commit",
        )
        if scoring_tag_commit != code_commit:
            raise Stage4ScoringNotAuthorizedError(
                "stage-4 scoring-code tag must point exactly at HEAD"
            )

        frozen_blobs: list[tuple[str, str]] = []
        for path in STAGE4_FROZEN_PROTOCOL_PATHS:
            protocol_blob = _oid(
                self._git(
                    root,
                    "rev-parse",
                    "--verify",
                    f"{protocol_tag}:{path}",
                    description=f"resolve protocol blob {path}",
                ),
                label=f"protocol blob {path}",
            )
            head_blob = _oid(
                self._git(
                    root,
                    "rev-parse",
                    "--verify",
                    f"HEAD:{path}",
                    description=f"resolve current blob {path}",
                ),
                label=f"current blob {path}",
            )
            if protocol_blob != head_blob:
                raise Stage4ScoringNotAuthorizedError(
                    f"score-blind protocol bytes changed after the protocol tag: {path}"
                )
            frozen_blobs.append((path, protocol_blob))

        remote_repository = _canonical_remote_repository(
            self._git(
                root,
                "remote",
                "get-url",
                remote,
                description="resolve stage-4 scoring remote URL",
            )
        )
        public_repository = self.public_probe.observe()
        branch_ref = f"refs/heads/{upstream_branch}"
        protocol_ref = f"refs/tags/{protocol_tag}"
        scoring_ref = f"refs/tags/{scoring_code_tag}"
        remote_payload = self._git(
            root,
            "ls-remote",
            "--refs",
            remote,
            branch_ref,
            protocol_ref,
            scoring_ref,
            description="verify live remote branch and tag identities",
        )
        remote_refs: dict[str, str] = {}
        for line in remote_payload.splitlines():
            fields = line.split()
            if len(fields) != 2 or fields[1] in remote_refs:
                raise Stage4ScoringNotAuthorizedError("remote reference response is malformed")
            remote_refs[fields[1]] = _oid(fields[0], label="remote reference object")
        expected_refs = {
            branch_ref: code_commit,
            protocol_ref: protocol_tag_object,
            scoring_ref: scoring_tag_object,
        }
        if remote_refs != expected_refs:
            raise Stage4ScoringNotAuthorizedError(
                "live public branch or stage-4 tag identity differs from local execution"
            )

        final_head = _oid(
            self._git(root, "rev-parse", "--verify", "HEAD^{commit}", description="recheck HEAD"),
            label="rechecked HEAD",
        )
        final_status = self._git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            description="recheck worktree state",
        )
        self._require_clean_status(final_status, allowed_untracked_paths)
        if final_head != code_commit:
            raise Stage4ScoringNotAuthorizedError("HEAD changed during stage-4 authorization")
        return Stage4RepositoryEvidence(
            code_commit=code_commit,
            branch=branch,
            upstream=upstream,
            remote=remote,
            remote_repository=remote_repository,
            remote_branch_ref=branch_ref,
            remote_branch_commit=remote_refs[branch_ref],
            public_repository=public_repository,
            protocol_tag=protocol_tag,
            protocol_tag_object=protocol_tag_object,
            protocol_tag_commit=protocol_tag_commit,
            scoring_code_tag=scoring_code_tag,
            scoring_code_tag_object=scoring_tag_object,
            scoring_code_tag_commit=scoring_tag_commit,
            frozen_blob_ids=tuple(frozen_blobs),
        )


def stage4_execution_binding_id(
    repository: Stage4RepositoryEvidence,
    score_blind_inputs: ScoreBlindInputEvidence,
    qualification: Stage4QualificationEvidence,
) -> str:
    """Return the non-circular binding shared by both empty local ledgers."""

    if qualification.scoring_code_commit != repository.code_commit:
        raise Stage4ScoringNotAuthorizedError(
            "qualification evidence belongs to another scoring-code commit"
        )
    if qualification.protocol_design_sha256 != score_blind_inputs.protocol_design_sha256:
        raise Stage4ScoringNotAuthorizedError(
            "qualification evidence belongs to another stage-4 protocol"
        )
    if qualification.score_blind_input_evidence_sha256 != score_blind_inputs.content_sha256:
        raise Stage4ScoringNotAuthorizedError(
            "qualification evidence belongs to different score-blind inputs"
        )
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "qualification_evidence_sha256": qualification.content_sha256,
                "repository_evidence_sha256": repository.content_sha256,
                "score_blind_input_evidence_sha256": score_blind_inputs.content_sha256,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage4ScoringSeal:
    """Immutable proof that all pre-target scoring-code gates passed."""

    repository: Stage4RepositoryEvidence
    score_blind_inputs: ScoreBlindInputEvidence
    qualification: Stage4QualificationEvidence
    execution_binding_id: str
    expected_target_identity: tuple[tuple[str, str], ...]
    initial_attempt_ledger_sha256: str
    initial_target_read_ledger_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.execution_binding_id, label="execution_binding_id")
        _sha256(self.initial_attempt_ledger_sha256, label="initial_attempt_ledger_sha256")
        _sha256(
            self.initial_target_read_ledger_sha256,
            label="initial_target_read_ledger_sha256",
        )
        expected_binding = stage4_execution_binding_id(
            self.repository,
            self.score_blind_inputs,
            self.qualification,
        )
        if self.execution_binding_id != expected_binding:
            raise ValueError("scoring seal execution binding is inconsistent")
        _validate_expected_target_identity(self.expected_target_identity)

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "execution_binding_id": self.execution_binding_id,
                "expected_target_identity": dict(self.expected_target_identity),
                "formal_backend": self.qualification.formal_backend,
                "initial_ledgers": {
                    "formal_attempt": {
                        "content_sha256": self.initial_attempt_ledger_sha256,
                        "operation_count": 0,
                        "relative_path": STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
                    },
                    "target_read": {
                        "content_sha256": self.initial_target_read_ledger_sha256,
                        "operation_count": 0,
                        "relative_path": STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
                    },
                },
                "locked_test": {
                    "contacted": False,
                    "formal_execution_forbidden": True,
                    "run": False,
                },
                "protocol_version": STAGE4_PROTOCOL_VERSION,
                "qualification": self.qualification.as_mapping(),
                "repository": self.repository.as_mapping(),
                "schema_version": STAGE4_SCORING_SEAL_SCHEMA_VERSION,
                "score_blind_inputs": self.score_blind_inputs.as_mapping(),
                "stage": 4,
                "status": "scoring_code_frozen_target_unread",
                "target_observation": {
                    "expected_identity_copied_from_protocol": True,
                    "path_observed": False,
                    "read_count": 0,
                    "stat_called": False,
                },
            }
        )

    @property
    def seal_id(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Stage4ScoringSeal:
        expected = {
            "content_sha256",
            "execution_binding_id",
            "expected_target_identity",
            "formal_backend",
            "initial_ledgers",
            "locked_test",
            "protocol_version",
            "qualification",
            "repository",
            "schema_version",
            "score_blind_inputs",
            "stage",
            "status",
            "target_observation",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise Stage4ScoringNotAuthorizedError("stage-4 scoring seal hash or schema is invalid")
        if (
            value.get("schema_version") != STAGE4_SCORING_SEAL_SCHEMA_VERSION
            or value.get("protocol_version") != STAGE4_PROTOCOL_VERSION
            or value.get("stage") != 4
            or value.get("status") != "scoring_code_frozen_target_unread"
        ):
            raise Stage4ScoringNotAuthorizedError("stage-4 scoring seal version/status changed")
        locked = _mapping(value.get("locked_test"), label="scoring seal locked_test")
        target_observation = _mapping(
            value.get("target_observation"),
            label="scoring seal target_observation",
        )
        if dict(locked) != {
            "contacted": False,
            "formal_execution_forbidden": True,
            "run": False,
        } or dict(target_observation) != {
            "expected_identity_copied_from_protocol": True,
            "path_observed": False,
            "read_count": 0,
            "stat_called": False,
        }:
            raise Stage4ScoringNotAuthorizedError(
                "scoring seal crossed a forbidden target boundary"
            )
        repository = Stage4RepositoryEvidence.from_mapping(
            _mapping(value.get("repository"), label="repository")
        )
        score_blind_inputs = ScoreBlindInputEvidence.from_mapping(
            _mapping(value.get("score_blind_inputs"), label="score_blind_inputs")
        )
        qualification = Stage4QualificationEvidence.from_mapping(
            _mapping(value.get("qualification"), label="qualification")
        )
        if value.get("formal_backend") != qualification.formal_backend:
            raise Stage4ScoringNotAuthorizedError("scoring seal formal backend changed")
        ledgers = _mapping(value.get("initial_ledgers"), label="initial_ledgers")
        if set(ledgers) != {"formal_attempt", "target_read"}:
            raise Stage4ScoringNotAuthorizedError("scoring seal ledger set changed")
        attempt = _mapping(ledgers["formal_attempt"], label="formal attempt ledger")
        target = _mapping(ledgers["target_read"], label="target read ledger")
        if (
            set(attempt) != {"content_sha256", "operation_count", "relative_path"}
            or set(target) != {"content_sha256", "operation_count", "relative_path"}
            or attempt.get("operation_count") != 0
            or target.get("operation_count") != 0
            or attempt.get("relative_path") != STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH
            or target.get("relative_path") != STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH
        ):
            raise Stage4ScoringNotAuthorizedError("scoring seal ledgers were not empty")
        target_identity = _mapping(
            value.get("expected_target_identity"),
            label="expected_target_identity",
        )
        if set(target_identity) != set(_EXPECTED_TARGET_IDENTITY_FIELDS) or any(
            not isinstance(item, str) for item in target_identity.values()
        ):
            raise Stage4ScoringNotAuthorizedError("scoring seal target identity changed")
        try:
            return cls(
                repository=repository,
                score_blind_inputs=score_blind_inputs,
                qualification=qualification,
                execution_binding_id=cast(str, value["execution_binding_id"]),
                expected_target_identity=tuple(
                    sorted((name, cast(str, item)) for name, item in target_identity.items())
                ),
                initial_attempt_ledger_sha256=cast(str, attempt["content_sha256"]),
                initial_target_read_ledger_sha256=cast(str, target["content_sha256"]),
            )
        except (TypeError, ValueError) as exc:
            raise Stage4ScoringNotAuthorizedError("stage-4 scoring seal invariants failed") from exc


@dataclass(frozen=True, slots=True)
class Stage4TargetReadinessEvidence:
    """Read-only proof of every pre-target gate, without granting a capability."""

    scoring_seal: Stage4ScoringSeal = field(repr=False)
    preflight_receipt: FormalPreflightReceipt = field(repr=False)
    repository: Stage4RepositoryEvidence = field(repr=False)
    attempt_ledger: Stage4AuditLedger = field(repr=False)
    target_read_ledger: Stage4AuditLedger = field(repr=False)
    attempt_ledger_path: Path
    target_read_ledger_path: Path

    def __post_init__(self) -> None:
        seal = self.scoring_seal
        if self.repository != seal.repository:
            raise ValueError("target-readiness repository differs from the scoring seal")
        if self.preflight_receipt.content_sha256 != (
            seal.qualification.formal_preflight_receipt_sha256
        ):
            raise ValueError("target-readiness preflight differs from the scoring seal")
        if self.preflight_receipt.space_placebo_resource_observation.content_sha256 != (
            seal.qualification.space_placebo_resource_observation_sha256
        ):
            raise ValueError("target-readiness resource observation differs from the seal")
        if (
            self.attempt_ledger.kind != "formal_attempt"
            or self.target_read_ledger.kind != "target_read"
            or self.attempt_ledger.execution_binding_id != seal.execution_binding_id
            or self.target_read_ledger.execution_binding_id != seal.execution_binding_id
            or self.attempt_ledger.operation_count != 0
            or self.target_read_ledger.operation_count != 0
        ):
            raise ValueError("target-readiness requires both canonical empty ledgers")
        if (
            self.attempt_ledger.content_sha256 != seal.initial_attempt_ledger_sha256
            or self.target_read_ledger.content_sha256 != seal.initial_target_read_ledger_sha256
        ):
            raise ValueError("target-readiness ledgers differ from the scoring seal")
        object.__setattr__(
            self,
            "attempt_ledger_path",
            Path(os.path.abspath(os.fspath(self.attempt_ledger_path))),
        )
        object.__setattr__(
            self,
            "target_read_ledger_path",
            Path(os.path.abspath(os.fspath(self.target_read_ledger_path))),
        )

    @property
    def readiness_id(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "attempt_ledger_sha256": self.attempt_ledger.content_sha256,
                    "formal_preflight_receipt_sha256": (self.preflight_receipt.content_sha256),
                    "repository_evidence_sha256": self.repository.content_sha256,
                    "scoring_seal_id": self.scoring_seal.seal_id,
                    "space_placebo_resource_observation_sha256": (
                        self.preflight_receipt.space_placebo_resource_observation.content_sha256
                    ),
                    "target_read_ledger_sha256": self.target_read_ledger.content_sha256,
                }
            )
        ).hexdigest()

    def as_mapping(self) -> dict[str, object]:
        return {
            "authorization_granted": False,
            "formal_attempt_count": 0,
            "readiness_id": self.readiness_id,
            "repository_and_tags_verified": True,
            "target_path_observed": False,
            "target_read_count": 0,
        }


def _freeze_tags(protocol: Mapping[str, object]) -> tuple[str, str]:
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    scoring = _mapping(freeze.get("scoring_code_freeze"), label="scoring_code_freeze")
    protocol_tag = freeze.get("pre_score_tag")
    scoring_tag = scoring.get("expected_tag")
    if not isinstance(protocol_tag, str) or not isinstance(scoring_tag, str):
        raise Stage4ScoringNotAuthorizedError("stage-4 freeze tags are missing")
    if freeze.get("protocol_tag_authorizes_only_score_free_implementation") is not True:
        raise Stage4ScoringNotAuthorizedError("protocol-tag score-free boundary changed")
    return protocol_tag, scoring_tag


def build_stage4_scoring_seal(
    protocol: Mapping[str, object],
    *,
    repository: Stage4RepositoryEvidence,
    score_blind_inputs: ScoreBlindInputEvidence,
    qualification: Stage4QualificationEvidence,
    formal_preflight_receipt: FormalPreflightReceipt,
    attempt_ledger: Stage4AuditLedger,
    target_read_ledger: Stage4AuditLedger,
) -> Stage4ScoringSeal:
    """Build a target-unread scoring seal from fresh score-blind evidence."""

    protocol_tag, scoring_tag = _freeze_tags(protocol)
    if repository.protocol_tag != protocol_tag or repository.scoring_code_tag != scoring_tag:
        raise Stage4ScoringNotAuthorizedError("repository tags differ from the frozen protocol")
    design_sha256 = protocol_design_sha256(protocol)
    if score_blind_inputs.protocol_design_sha256 != design_sha256:
        raise Stage4ScoringNotAuthorizedError("score-blind inputs use another protocol design")
    if qualification.protocol_design_sha256 != design_sha256:
        raise Stage4ScoringNotAuthorizedError("qualification uses another protocol design")
    try:
        validate_stage4_qualification_against_protocol(protocol, qualification)
        validate_stage4_qualification_against_formal_preflight(
            qualification,
            formal_preflight_receipt,
        )
    except Exception as exc:
        raise Stage4ScoringNotAuthorizedError(
            "structured stage-4 qualification evidence does not match the protocol"
        ) from exc
    binding = stage4_execution_binding_id(repository, score_blind_inputs, qualification)
    if (
        attempt_ledger.kind != "formal_attempt"
        or target_read_ledger.kind != "target_read"
        or attempt_ledger.execution_binding_id != binding
        or target_read_ledger.execution_binding_id != binding
    ):
        raise Stage4ScoringNotAuthorizedError("local ledgers use another execution binding")
    if attempt_ledger.operation_count != 0 or target_read_ledger.operation_count != 0:
        raise Stage4ScoringNotAuthorizedError(
            "stage-4 scoring seal requires zero attempts and zero target reads"
        )
    return Stage4ScoringSeal(
        repository=repository,
        score_blind_inputs=score_blind_inputs,
        qualification=qualification,
        execution_binding_id=binding,
        expected_target_identity=tuple(
            sorted(expected_target_identity_from_protocol(protocol).items())
        ),
        initial_attempt_ledger_sha256=attempt_ledger.content_sha256,
        initial_target_read_ledger_sha256=target_read_ledger.content_sha256,
    )


def write_stage4_scoring_seal_atomic(path: Path, seal: Stage4ScoringSeal) -> str:
    """Create the frozen scoring seal once; only identical replay is permitted."""

    if not isinstance(seal, Stage4ScoringSeal):
        raise TypeError("seal must be Stage4ScoringSeal")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json_bytes(seal.as_mapping()) + b"\n"
    serialized_sha256 = hashlib.sha256(serialized).hexdigest()
    temporary_name: str | None = None
    created = False
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        staged = Path(temporary_name)
        try:
            # A hard link is an atomic create-only publication on the same
            # filesystem.  Unlike os.replace, it can never rewrite a frozen
            # seal after another process or an earlier invocation created it.
            os.link(staged, target)
            created = True
        except FileExistsError:
            try:
                existing = read_existing_immutable_bytes(
                    target,
                    label="existing stage-4 scoring seal",
                )
            except UnsafeImmutableFileError:
                existing = None
            if existing == serialized:
                return serialized_sha256
            raise Stage4ScoringNotAuthorizedError(
                "immutable stage-4 scoring seal already contains different bytes "
                "or is not a safe single-link regular file"
            ) from None
        finally:
            staged.unlink(missing_ok=True)
            temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    if not created:
        raise AssertionError("stage-4 scoring seal publication reached no terminal state")
    try:
        if (
            read_existing_immutable_bytes(
                target,
                label="new stage-4 scoring seal",
            )
            != serialized
        ):
            raise UnsafeImmutableFileError("new stage-4 scoring seal bytes changed")
    except UnsafeImmutableFileError as exc:
        raise Stage4ScoringNotAuthorizedError(
            "new stage-4 scoring seal is not a safe single-link regular file"
        ) from exc
    return serialized_sha256


def load_stage4_scoring_seal(path: Path) -> Stage4ScoringSeal:
    target = Path(os.path.abspath(os.fspath(path)))
    try:
        require_existing_real_directory_tree(
            Path(target.anchor) if target.anchor else Path.cwd(),
            target.parent,
            label="stage-4 scoring seal directory",
        )
        payload = read_existing_immutable_bytes(
            target,
            label="stage-4 scoring seal",
        )
        value = json.loads(payload.decode("utf-8"))
    except (UnsafeImmutableFileError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4ScoringNotAuthorizedError("cannot read local stage-4 scoring seal") from exc
    return Stage4ScoringSeal.from_mapping(_mapping(value, label="scoring seal"))


@dataclass(frozen=True, slots=True)
class Stage4TargetAuthorization:
    """Non-transferable capability accepted by the sole target entrance."""

    seal_id: str
    execution_binding_id: str
    repository_evidence_sha256: str
    expected_target_identity: tuple[tuple[str, str], ...]
    formal_backend: str
    formal_preflight_receipt_sha256: str
    space_placebo_resource_observation_sha256: str
    space_placebo_recommended_max_in_flight: int
    gpu_requested: bool
    gpu_status: GpuQualificationStatus
    attempt_ledger_path: Path
    target_read_ledger_path: Path
    _sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._sentinel is not _AUTHORIZATION_SENTINEL:
            raise ValueError("stage-4 target authorization cannot be constructed directly")
        for label, value in (
            ("seal_id", self.seal_id),
            ("execution_binding_id", self.execution_binding_id),
            ("repository_evidence_sha256", self.repository_evidence_sha256),
            ("formal_preflight_receipt_sha256", self.formal_preflight_receipt_sha256),
            (
                "space_placebo_resource_observation_sha256",
                self.space_placebo_resource_observation_sha256,
            ),
        ):
            _sha256(value, label=label)
        if self.formal_backend not in {"cpu_float64", "gpu_float64"}:
            raise ValueError("stage-4 formal backend is not frozen")
        if not isinstance(self.gpu_requested, bool) or self.gpu_status not in {
            "not_requested",
            "blocked_no_frozen_backend",
            "not_equivalent",
            "equivalent_and_frozen",
        }:
            raise ValueError("stage-4 GPU qualification state is invalid")
        if (self.gpu_status == "equivalent_and_frozen") != (self.formal_backend == "gpu_float64"):
            raise ValueError("stage-4 GPU state and formal backend differ")
        if (
            not isinstance(self.space_placebo_recommended_max_in_flight, int)
            or isinstance(self.space_placebo_recommended_max_in_flight, bool)
            or not 1 <= self.space_placebo_recommended_max_in_flight <= 12
        ):
            raise ValueError("stage-4 space placebo recommendation is invalid")

    @property
    def authorization_id(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "execution_binding_id": self.execution_binding_id,
                    "expected_target_identity": dict(self.expected_target_identity),
                    "formal_backend": self.formal_backend,
                    "formal_preflight_receipt_sha256": (self.formal_preflight_receipt_sha256),
                    "gpu_requested": self.gpu_requested,
                    "gpu_status": self.gpu_status,
                    "formal_attempt_ledger": STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
                    "repository_evidence_sha256": self.repository_evidence_sha256,
                    "seal_id": self.seal_id,
                    "space_placebo_recommended_max_in_flight": (
                        self.space_placebo_recommended_max_in_flight
                    ),
                    "space_placebo_resource_observation_sha256": (
                        self.space_placebo_resource_observation_sha256
                    ),
                    "target_read_ledger": STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
                }
            )
        ).hexdigest()

    def expected_target_mapping(self) -> dict[str, str]:
        return dict(self.expected_target_identity)


def require_stage4_target_authorization(
    value: Stage4TargetAuthorization,
    *,
    project_root: Path | None = None,
) -> Stage4TargetAuthorization:
    """Reject forged, malformed, or root-transferred target capabilities."""

    if (
        not isinstance(value, Stage4TargetAuthorization)
        or value._sentinel is not _AUTHORIZATION_SENTINEL
    ):
        raise Stage4ScoringNotAuthorizedError("a valid stage-4 target capability is required")
    _sha256(value.authorization_id, label="authorization_id")
    if project_root is not None:
        root = Path(project_root).resolve()
        _require_canonical_ledger_path(
            root,
            value.attempt_ledger_path,
            kind="formal_attempt",
        )
        _require_canonical_ledger_path(
            root,
            value.target_read_ledger_path,
            kind="target_read",
        )
    return value


def _relative_to_root(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    lexical = Path(os.path.abspath(os.fspath(path)))
    if not lexical.is_relative_to(resolved_root):
        raise Stage4ScoringNotAuthorizedError("local scoring seal escapes the project root")
    return lexical.relative_to(resolved_root).as_posix()


def _require_canonical_ledger_path(root: Path, path: Path, *, kind: str) -> Path:
    relative = {
        "formal_attempt": STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
        "target_read": STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
    }[kind]
    canonical = Path(os.path.abspath(os.fspath(root.joinpath(*PurePosixPath(relative).parts))))
    supplied = Path(os.path.abspath(os.fspath(path)))
    if os.path.normcase(os.fspath(supplied)) != os.path.normcase(os.fspath(canonical)):
        raise Stage4ScoringNotAuthorizedError(
            f"stage-4 {kind} ledger must use its sole repository-root path"
        )
    return canonical


def verify_stage4_target_readiness(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    scoring_seal_path: Path,
    attempt_ledger_path: Path,
    target_read_ledger_path: Path,
    repository_adapter: Stage4RepositoryAdapter | None = None,
) -> Stage4TargetReadinessEvidence:
    """Reprove seal, tags, public remote, inputs, and empty ledgers without authorization."""

    root = Path(project_root).resolve()
    seal_path = require_score_blind_project_path(
        root,
        protocol,
        scoring_seal_path,
        label="stage-4 scoring seal",
    )
    safe_attempt_ledger = require_score_blind_project_path(
        root,
        protocol,
        attempt_ledger_path,
        label="stage-4 formal-attempt ledger",
    )
    safe_target_ledger = require_score_blind_project_path(
        root,
        protocol,
        target_read_ledger_path,
        label="stage-4 target-read audit ledger",
    )
    canonical_attempt_ledger = _require_canonical_ledger_path(
        root,
        safe_attempt_ledger,
        kind="formal_attempt",
    )
    canonical_target_ledger = _require_canonical_ledger_path(
        root,
        safe_target_ledger,
        kind="target_read",
    )
    loaded = load_stage4_scoring_seal(seal_path)
    preflight_receipt_path = require_score_blind_project_path(
        root,
        protocol,
        root.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts),
        label="formal preflight receipt",
    )
    try:
        preflight_receipt = load_formal_preflight_receipt(preflight_receipt_path)
        validate_stage4_qualification_against_formal_preflight(
            loaded.qualification,
            preflight_receipt,
        )
    except Exception as exc:
        raise Stage4ScoringNotAuthorizedError(
            "formal preflight receipt is missing, altered, or belongs to another qualification"
        ) from exc
    protocol_tag, scoring_tag = _freeze_tags(protocol)
    score_blind_inputs = observe_score_blind_inputs(root, protocol)
    adapter = repository_adapter or GitStage4RepositoryAdapter()
    repository = adapter.observe(
        root,
        protocol_tag=protocol_tag,
        scoring_code_tag=scoring_tag,
        allowed_untracked_paths=(_relative_to_root(root, seal_path),),
    )
    if repository != loaded.repository:
        raise Stage4ScoringNotAuthorizedError("repository identity changed after seal generation")
    if score_blind_inputs != loaded.score_blind_inputs:
        raise Stage4ScoringNotAuthorizedError("score-blind inputs changed after seal generation")
    try:
        attempt_ledger = read_stage4_ledger(
            canonical_attempt_ledger,
            expected_kind="formal_attempt",
            expected_binding_id=loaded.execution_binding_id,
        )
        target_ledger = read_stage4_ledger(
            canonical_target_ledger,
            expected_kind="target_read",
            expected_binding_id=loaded.execution_binding_id,
        )
    except Stage4LedgerError as exc:
        raise Stage4ScoringNotAuthorizedError("stage-4 local ledger verification failed") from exc
    rebuilt = build_stage4_scoring_seal(
        protocol,
        repository=repository,
        score_blind_inputs=score_blind_inputs,
        qualification=loaded.qualification,
        formal_preflight_receipt=preflight_receipt,
        attempt_ledger=attempt_ledger,
        target_read_ledger=target_ledger,
    )
    if rebuilt.as_mapping() != loaded.as_mapping():
        raise Stage4ScoringNotAuthorizedError("local stage-4 scoring seal is stale or altered")
    try:
        return Stage4TargetReadinessEvidence(
            scoring_seal=loaded,
            preflight_receipt=preflight_receipt,
            repository=repository,
            attempt_ledger=attempt_ledger,
            target_read_ledger=target_ledger,
            attempt_ledger_path=canonical_attempt_ledger,
            target_read_ledger_path=canonical_target_ledger,
        )
    except ValueError as exc:
        raise Stage4ScoringNotAuthorizedError(
            "stage-4 target-readiness proof is inconsistent"
        ) from exc


def authorize_stage4_target_access(
    project_root: Path,
    protocol: Mapping[str, object],
    *,
    scoring_seal_path: Path,
    attempt_ledger_path: Path,
    target_read_ledger_path: Path,
    repository_adapter: Stage4RepositoryAdapter | None = None,
) -> Stage4TargetAuthorization:
    """Reprove the complete seal and grant one target capability without touching target."""

    evidence = verify_stage4_target_readiness(
        project_root,
        protocol,
        scoring_seal_path=scoring_seal_path,
        attempt_ledger_path=attempt_ledger_path,
        target_read_ledger_path=target_read_ledger_path,
        repository_adapter=repository_adapter,
    )
    loaded = evidence.scoring_seal
    preflight_receipt = evidence.preflight_receipt
    repository = evidence.repository
    return Stage4TargetAuthorization(
        seal_id=loaded.seal_id,
        execution_binding_id=loaded.execution_binding_id,
        repository_evidence_sha256=repository.content_sha256,
        expected_target_identity=loaded.expected_target_identity,
        formal_backend=loaded.qualification.formal_backend,
        formal_preflight_receipt_sha256=preflight_receipt.content_sha256,
        space_placebo_resource_observation_sha256=(
            preflight_receipt.space_placebo_resource_observation.content_sha256
        ),
        space_placebo_recommended_max_in_flight=(
            preflight_receipt.space_placebo_resource_observation.recommended_max_in_flight
        ),
        gpu_requested=loaded.qualification.gpu_requested,
        gpu_status=loaded.qualification.gpu_status,
        attempt_ledger_path=evidence.attempt_ledger_path,
        target_read_ledger_path=evidence.target_read_ledger_path,
        _sentinel=_AUTHORIZATION_SENTINEL,
    )


__all__ = [
    "STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH",
    "STAGE4_FROZEN_PROTOCOL_PATHS",
    "STAGE4_PUBLIC_REPOSITORY",
    "STAGE4_SCORING_SEAL_SCHEMA_VERSION",
    "STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH",
    "GitHubUnauthenticatedPublicProbe",
    "GitStage4RepositoryAdapter",
    "PublicRepositoryEvidence",
    "PublicRepositoryProbe",
    "Stage4RepositoryAdapter",
    "Stage4RepositoryEvidence",
    "Stage4ScoringNotAuthorizedError",
    "Stage4ScoringSeal",
    "Stage4TargetAuthorization",
    "Stage4TargetReadinessEvidence",
    "authorize_stage4_target_access",
    "build_stage4_scoring_seal",
    "load_stage4_scoring_seal",
    "require_stage4_target_authorization",
    "stage4_execution_binding_id",
    "verify_stage4_target_readiness",
    "write_stage4_scoring_seal_atomic",
]
