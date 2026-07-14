"""Repository-bound authorization for background-model fitting and scoring.

Protocol 0.2.0 remains directly authorized for backward compatibility.  Protocol
0.2.1 can only be scored by an execution that proves the immutable preregistration,
the separately tagged scoring implementation, the pushed remote identity, and every
sealed input before any catalog row is opened.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.execution import (
    CommandResult,
    ExecutionSeal,
    GitCommandRunner,
    require_execution_seal_unchanged,
    subprocess_git_runner,
)
from seismoflux.config import sha256_file

_GIT_OID_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_LOCAL_SUPPORT_PROTOCOL_VERSION = "0.2.1"
_LOCAL_SUPPORT_FREEZE_TAG = "v0.2.1-background-local-support-protocol"
_LOCAL_SUPPORT_FREEZE_TAG_OBJECT = "06136e22bb8c6e2606a9debd5e00d53b500f758d"
_LOCAL_SUPPORT_FREEZE_COMMIT = "966fb4e84c36aba373d90b81fe9e1350ffe349b6"
_LOCAL_SUPPORT_SCORING_CODE_TAG = "v0.2.1-background-local-support-scoring-code-r1"
_LOCAL_SUPPORT_REMOTE_REPOSITORY = "github.com/Justin-147/SeismoFlux"
_LOCAL_SUPPORT_PROTOCOL_SHA256 = "c7d6488bd97f0017867573c8b99230d79091412322652af25badfe732606e76a"
_LOCAL_SUPPORT_FROZEN_FILES = (
    (
        "configs/background_local_support.yaml",
        "d12bf40de8f5814e3e33b988f106ed8621538487",
        "85c7c05f353e25d742e1d4d694012205bba8fa3a1b6f4e40722bbd17ad92fd1c",
    ),
    (
        "data/manifests/background_local_support_fold_manifest.json",
        "0ff9be3c5b4b330569fcf549616745910c734ecf",
        "d7ae5266c9143ed0a67a9954da52039b2753f108698fb05477466a6d5b934e38",
    ),
    (
        "data/manifests/background_local_support_manifest.json",
        "1e93b6a0e76825bea0482bc25540c3782202b2aa",
        "632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d",
    ),
)


class BackgroundScoringNotAuthorizedError(RuntimeError):
    """Raised when a scoring entrypoint lacks the exact repository capability."""


def require_background_scoring_protocol_eligible(config: BackgroundConfig) -> None:
    """Validate protocol identity without opening data or granting score access."""

    protocol_version = str(config.protocol_version)
    if protocol_version == "0.2.0":
        return
    if protocol_version != _LOCAL_SUPPORT_PROTOCOL_VERSION:
        raise BackgroundScoringNotAuthorizedError(
            f"background scoring is not authorized for protocol version {protocol_version!r}"
        )
    protocol_sha256 = hashlib.sha256(
        canonical_json_bytes(config.model_dump(mode="python"))
    ).hexdigest()
    if (
        config.freeze_tag != _LOCAL_SUPPORT_FREEZE_TAG
        or protocol_sha256 != _LOCAL_SUPPORT_PROTOCOL_SHA256
    ):
        raise BackgroundScoringNotAuthorizedError(
            "local-support protocol differs from the frozen stage-2R-0 preregistration"
        )


def _oid(value: str, *, name: str) -> str:
    candidate = value.strip()
    if _GIT_OID_PATTERN.fullmatch(candidate) is None:
        raise BackgroundScoringNotAuthorizedError(f"{name} must be a lowercase Git object ID")
    return candidate


def _git(
    runner: GitCommandRunner,
    root: Path,
    *arguments: str,
    description: str,
) -> str:
    try:
        result = runner(("git", *arguments), root)
    except Exception as exc:
        raise BackgroundScoringNotAuthorizedError(f"unable to {description}") from exc
    if not isinstance(result, CommandResult):
        raise TypeError("Git command runners must return CommandResult")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise BackgroundScoringNotAuthorizedError(f"unable to {description}{suffix}")
    return result.stdout.strip()


def _remote_refs(payload: str) -> dict[str, str]:
    refs: dict[str, str] = {}
    for line in payload.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise BackgroundScoringNotAuthorizedError("remote reference response is malformed")
        object_id = _oid(fields[0], name="remote reference object")
        reference = fields[1]
        if reference in refs:
            raise BackgroundScoringNotAuthorizedError(
                "remote reference response contains duplicates"
            )
        refs[reference] = object_id
    return refs


def _canonical_remote_repository(value: str) -> str:
    """Accept only the user-authorized public GitHub repository identity."""

    candidate = value.strip()
    patterns = (
        re.compile(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?"),
        re.compile(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?"),
        re.compile(r"ssh://git@github\.com/([^/]+)/([^/]+?)(?:\.git)?"),
    )
    match = None
    for pattern in patterns:
        match = pattern.fullmatch(candidate)
        if match is not None:
            break
    if match is None:
        raise BackgroundScoringNotAuthorizedError(
            "scoring remote URL is not the authorized public GitHub repository"
        )
    owner, repository = match.groups()
    if (owner.casefold(), repository.casefold()) != ("justin-147", "seismoflux"):
        raise BackgroundScoringNotAuthorizedError(
            "scoring remote URL is not the authorized Justin-147/SeismoFlux repository"
        )
    return _LOCAL_SUPPORT_REMOTE_REPOSITORY


@dataclass(frozen=True, slots=True)
class BackgroundScoringAuthorization:
    """Content-addressed proof that the 0.2.1 scoring boundary may be crossed."""

    execution_seal_id: str
    protocol_sha256: str
    freeze_tag: str
    freeze_tag_object: str
    freeze_tag_commit: str
    scoring_code_tag: str
    scoring_code_tag_object: str
    scoring_code_tag_commit: str
    code_commit: str
    remote: str
    remote_repository: str
    remote_branch_ref: str
    remote_branch_commit: str
    frozen_blob_ids: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("execution_seal_id", "protocol_sha256"):
            if _SHA256_PATTERN.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 string")
        for name in (
            "freeze_tag_object",
            "freeze_tag_commit",
            "scoring_code_tag_object",
            "scoring_code_tag_commit",
            "code_commit",
            "remote_branch_commit",
        ):
            if _GIT_OID_PATTERN.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase Git object ID")
        if self.freeze_tag != _LOCAL_SUPPORT_FREEZE_TAG:
            raise ValueError("authorization uses the wrong local-support freeze tag")
        if self.freeze_tag_object != _LOCAL_SUPPORT_FREEZE_TAG_OBJECT:
            raise ValueError("authorization freeze tag object differs from preregistration")
        if self.freeze_tag_commit != _LOCAL_SUPPORT_FREEZE_COMMIT:
            raise ValueError("authorization freeze commit differs from preregistration")
        if self.scoring_code_tag != _LOCAL_SUPPORT_SCORING_CODE_TAG:
            raise ValueError("authorization uses the wrong scoring-code tag")
        if self.scoring_code_tag_commit != self.code_commit:
            raise ValueError("scoring-code tag must point exactly at the scoring commit")
        if self.remote_branch_commit != self.code_commit:
            raise ValueError("remote branch must point exactly at the scoring commit")
        if self.protocol_sha256 != _LOCAL_SUPPORT_PROTOCOL_SHA256:
            raise ValueError("authorization protocol fingerprint differs from frozen 0.2.1")
        if self.remote_repository != _LOCAL_SUPPORT_REMOTE_REPOSITORY:
            raise ValueError("authorization uses another public repository identity")
        expected_blobs = tuple((path, blob) for path, blob, _ in _LOCAL_SUPPORT_FROZEN_FILES)
        if self.frozen_blob_ids != expected_blobs:
            raise ValueError("authorization frozen blob identities are incomplete or changed")
        for name in ("remote", "remote_branch_ref"):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string")

    @property
    def authorization_id(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "execution_seal_id": self.execution_seal_id,
                    "protocol_sha256": self.protocol_sha256,
                    "freeze_tag": self.freeze_tag,
                    "freeze_tag_object": self.freeze_tag_object,
                    "freeze_tag_commit": self.freeze_tag_commit,
                    "scoring_code_tag": self.scoring_code_tag,
                    "scoring_code_tag_object": self.scoring_code_tag_object,
                    "scoring_code_tag_commit": self.scoring_code_tag_commit,
                    "code_commit": self.code_commit,
                    "remote": self.remote,
                    "remote_repository": self.remote_repository,
                    "remote_branch_ref": self.remote_branch_ref,
                    "remote_branch_commit": self.remote_branch_commit,
                    "frozen_blob_ids": dict(self.frozen_blob_ids),
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthorizedExecution:
    """An execution seal and its matching, non-transferable 0.2.1 capability."""

    execution_seal: ExecutionSeal
    scoring_authorization: BackgroundScoringAuthorization

    def __post_init__(self) -> None:
        authorization = self.scoring_authorization
        if authorization.execution_seal_id != self.execution_seal.seal_id:
            raise ValueError("scoring authorization is bound to another execution seal")
        if authorization.protocol_sha256 != self.execution_seal.protocol_sha256:
            raise ValueError("scoring authorization is bound to another protocol")
        if authorization.code_commit != self.execution_seal.repository.code_commit:
            raise ValueError("scoring authorization is bound to another code commit")

    @property
    def authorization_id(self) -> str:
        return self.scoring_authorization.authorization_id


def create_authorized_execution(
    project_root: Path,
    config: BackgroundConfig,
    execution_seal: ExecutionSeal,
    *,
    runner: GitCommandRunner = subprocess_git_runner,
) -> AuthorizedExecution:
    """Build the sole 0.2.1 scoring capability after all identities are proven."""

    if str(config.protocol_version) != _LOCAL_SUPPORT_PROTOCOL_VERSION:
        raise BackgroundScoringNotAuthorizedError(
            "repository-bound authorization is defined only for protocol 0.2.1"
        )
    require_background_scoring_protocol_eligible(config)
    protocol_sha256 = hashlib.sha256(
        canonical_json_bytes(config.model_dump(mode="python"))
    ).hexdigest()
    if protocol_sha256 != _LOCAL_SUPPORT_PROTOCOL_SHA256:
        raise BackgroundScoringNotAuthorizedError(
            "local-support protocol fingerprint differs from the frozen preregistration"
        )
    if execution_seal.protocol_sha256 != protocol_sha256:
        raise BackgroundScoringNotAuthorizedError(
            "execution seal is bound to another protocol fingerprint"
        )
    repository = execution_seal.repository
    if not repository.ready:
        raise BackgroundScoringNotAuthorizedError(
            "execution seal repository is not clean and pushed"
        )
    if repository.freeze_tag != _LOCAL_SUPPORT_FREEZE_TAG:
        raise BackgroundScoringNotAuthorizedError("execution seal uses another protocol freeze tag")
    if repository.freeze_tag_commit != _LOCAL_SUPPORT_FREEZE_COMMIT:
        raise BackgroundScoringNotAuthorizedError(
            "execution seal freeze commit differs from the preregistration"
        )
    if repository.code_commit == _LOCAL_SUPPORT_FREEZE_COMMIT:
        raise BackgroundScoringNotAuthorizedError(
            "score-free preregistration commit cannot execute model scoring"
        )

    root = Path(project_root).resolve()
    try:
        require_execution_seal_unchanged(
            root,
            config,
            execution_seal,
            runner=runner,
        )
    except Exception as exc:
        raise BackgroundScoringNotAuthorizedError(
            "execution seal changed before scoring authorization"
        ) from exc
    input_hashes = execution_seal.input_hash_mapping()
    expected_inputs = {
        "issue_manifest": "d7ae5266c9143ed0a67a9954da52039b2753f108698fb05477466a6d5b934e38",
        "support_manifest": "632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d",
    }
    for name, expected in expected_inputs.items():
        if input_hashes.get(name) != expected:
            raise BackgroundScoringNotAuthorizedError(f"execution seal changed frozen {name} bytes")

    frozen_blob_ids: list[tuple[str, str]] = []
    for relative_path, expected_blob, expected_sha256 in _LOCAL_SUPPORT_FROZEN_FILES:
        current_path = root.joinpath(*relative_path.split("/"))
        if not current_path.is_file() or sha256_file(current_path) != expected_sha256:
            raise BackgroundScoringNotAuthorizedError(
                f"frozen scoring input bytes changed: {relative_path}"
            )
        freeze_blob = _oid(
            _git(
                runner,
                root,
                "rev-parse",
                "--verify",
                f"{_LOCAL_SUPPORT_FREEZE_TAG}:{relative_path}",
                description=f"resolve frozen blob {relative_path}",
            ),
            name=f"frozen blob {relative_path}",
        )
        head_blob = _oid(
            _git(
                runner,
                root,
                "rev-parse",
                "--verify",
                f"HEAD:{relative_path}",
                description=f"resolve current blob {relative_path}",
            ),
            name=f"current blob {relative_path}",
        )
        if freeze_blob != expected_blob or head_blob != expected_blob:
            raise BackgroundScoringNotAuthorizedError(
                f"frozen Git blob identity changed: {relative_path}"
            )
        frozen_blob_ids.append((relative_path, expected_blob))

    freeze_tag_object = _oid(
        _git(
            runner,
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{_LOCAL_SUPPORT_FREEZE_TAG}^{{tag}}",
            description="resolve the annotated local-support freeze tag",
        ),
        name="local-support freeze tag object",
    )
    if freeze_tag_object != _LOCAL_SUPPORT_FREEZE_TAG_OBJECT:
        raise BackgroundScoringNotAuthorizedError(
            "local freeze tag object differs from the published preregistration"
        )
    scoring_tag_object = _oid(
        _git(
            runner,
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{_LOCAL_SUPPORT_SCORING_CODE_TAG}^{{tag}}",
            description="resolve the annotated local-support scoring-code tag",
        ),
        name="local-support scoring-code tag object",
    )
    scoring_tag_commit = _oid(
        _git(
            runner,
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{_LOCAL_SUPPORT_SCORING_CODE_TAG}^{{commit}}",
            description="resolve the local-support scoring-code commit",
        ),
        name="local-support scoring-code commit",
    )
    if scoring_tag_commit != repository.code_commit:
        raise BackgroundScoringNotAuthorizedError(
            "scoring-code tag must point exactly at the pushed execution commit"
        )

    if "/" not in repository.upstream:
        raise BackgroundScoringNotAuthorizedError(
            "execution upstream must name a remote and branch"
        )
    remote, branch = repository.upstream.split("/", maxsplit=1)
    remote_repository = _canonical_remote_repository(
        _git(
            runner,
            root,
            "remote",
            "get-url",
            remote,
            description="resolve the scoring remote repository URL",
        )
    )
    branch_ref = f"refs/heads/{branch}"
    freeze_ref = f"refs/tags/{_LOCAL_SUPPORT_FREEZE_TAG}"
    scoring_ref = f"refs/tags/{_LOCAL_SUPPORT_SCORING_CODE_TAG}"
    remote_refs = _remote_refs(
        _git(
            runner,
            root,
            "ls-remote",
            "--refs",
            remote,
            branch_ref,
            freeze_ref,
            scoring_ref,
            description="verify live remote branch and tag identities",
        )
    )
    expected_remote_refs = {
        branch_ref: repository.code_commit,
        freeze_ref: freeze_tag_object,
        scoring_ref: scoring_tag_object,
    }
    if remote_refs != expected_remote_refs:
        raise BackgroundScoringNotAuthorizedError(
            "live remote branch or tag identities differ from the local execution"
        )

    authorization = BackgroundScoringAuthorization(
        execution_seal_id=execution_seal.seal_id,
        protocol_sha256=protocol_sha256,
        freeze_tag=_LOCAL_SUPPORT_FREEZE_TAG,
        freeze_tag_object=freeze_tag_object,
        freeze_tag_commit=repository.freeze_tag_commit,
        scoring_code_tag=_LOCAL_SUPPORT_SCORING_CODE_TAG,
        scoring_code_tag_object=scoring_tag_object,
        scoring_code_tag_commit=scoring_tag_commit,
        code_commit=repository.code_commit,
        remote=remote,
        remote_repository=remote_repository,
        remote_branch_ref=branch_ref,
        remote_branch_commit=remote_refs[branch_ref],
        frozen_blob_ids=tuple(frozen_blob_ids),
    )
    return AuthorizedExecution(
        execution_seal=execution_seal,
        scoring_authorization=authorization,
    )


def require_background_scoring_authorized(
    config: BackgroundConfig,
    authorized_execution: AuthorizedExecution | None = None,
) -> None:
    """Reject 0.2.1 unless the exact repository-bound capability is supplied."""

    protocol_version = str(config.protocol_version)
    if protocol_version == "0.2.0":
        if authorized_execution is not None:
            raise BackgroundScoringNotAuthorizedError(
                "protocol 0.2.0 must not use a 0.2.1 scoring authorization"
            )
        return
    if protocol_version == _LOCAL_SUPPORT_PROTOCOL_VERSION:
        if not isinstance(authorized_execution, AuthorizedExecution):
            raise BackgroundScoringNotAuthorizedError(
                "background protocol 0.2.1 was frozen as the score-free stage-2R-0 "
                "preregistration and now requires a repository-bound stage-2R-1 "
                "AuthorizedExecution"
            )
        protocol_sha256 = hashlib.sha256(
            canonical_json_bytes(config.model_dump(mode="python"))
        ).hexdigest()
        if authorized_execution.execution_seal.protocol_sha256 != protocol_sha256:
            raise BackgroundScoringNotAuthorizedError(
                "authorized execution is bound to another protocol fingerprint"
            )
        return
    raise BackgroundScoringNotAuthorizedError(
        f"background scoring is not authorized for protocol version {protocol_version!r}"
    )


__all__ = [
    "AuthorizedExecution",
    "BackgroundScoringAuthorization",
    "BackgroundScoringNotAuthorizedError",
    "create_authorized_execution",
    "require_background_scoring_authorized",
    "require_background_scoring_protocol_eligible",
]
