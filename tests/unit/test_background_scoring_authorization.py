from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import seismoflux.background.runner as runner_module
import seismoflux.background.scoring_authorization as authorization_module
from seismoflux.background.config import load_background_protocol
from seismoflux.background.deliverables import (
    build_background_deliverables,
    publish_background_deliverables,
)
from seismoflux.background.execution import CommandResult, ExecutionSeal, RepositoryIdentity
from seismoflux.background.horizons import run_issue_horizon_backtests
from seismoflux.background.pipeline import run_background_pipeline
from seismoflux.background.pipeline_etas import run_etas_pipeline
from seismoflux.background.pipeline_poisson import run_poisson_kde_pipeline
from seismoflux.background.publication import (
    build_background_registry,
    publish_backtest_bundle,
    publish_experiment_bundle,
    publish_model_bundle,
    publish_registry_and_report,
    publish_registry_and_report_sealed,
)
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    BackgroundScoringNotAuthorizedError,
    create_authorized_execution,
    require_background_scoring_authorized,
)

LOCAL_SUPPORT_CONFIG = Path("configs/background_local_support.yaml")
LOCAL_SUPPORT_PROJECT_CONFIG = Path("configs/base_local_support.yaml")
ORIGINAL_CONFIG = Path("configs/background.yaml")


def test_original_v020_scoring_remains_authorized() -> None:
    require_background_scoring_authorized(load_background_protocol(ORIGINAL_CONFIG))


def test_score_free_v021_is_rejected_by_every_public_scoring_pipeline() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)

    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_background_pipeline(
            config,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            production_fixture_path=Path("unused.json"),
        )
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_poisson_kde_pipeline(config, unavailable, unavailable, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_etas_pipeline(config, unavailable, unavailable, unavailable, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        run_issue_horizon_backtests(
            config,
            unavailable,
            unavailable,
            unavailable,
            selected_mc=4.0,
            uniform_model=unavailable,
            spatial_model=unavailable,
            etas_parameters=unavailable,
            etas_spec=unavailable,
            etas_parameter_snapshot_id="0" * 64,
            etas_model_variant_id="unused",
        )


def test_production_runner_rejects_v021_before_sealing_or_loading_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("score-free preflight must precede bound inputs and sealing")

    for attribute in (
        "load_project_background_config",
        "create_execution_seal",
        "load_study_area",
        "load_earthquake_catalog",
    ):
        monkeypatch.setattr(runner_module, attribute, forbidden)

    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        runner_module.run_background_stage2(LOCAL_SUPPORT_PROJECT_CONFIG)


def test_score_free_v021_is_rejected_before_score_bearing_deliverable_inputs() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)
    expected = pytest.raises(
        BackgroundScoringNotAuthorizedError,
        match="score-free stage-2R-0",
    )

    with expected:
        build_background_deliverables(config, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publish_background_deliverables(
            Path("unused"),
            config,
            unavailable,
            unavailable,
        )


@pytest.mark.parametrize(
    "publisher",
    [publish_model_bundle, publish_backtest_bundle, publish_experiment_bundle],
)
def test_score_free_v021_is_rejected_before_score_bearing_bundle_publication(
    publisher: Any,
) -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)

    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publisher(
            Path("unused"),
            config,
            unavailable,
            unavailable,
            uv_lock_sha256="unused",
        )


def test_score_free_v021_is_rejected_before_registry_inputs_or_filesystem_access() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    unavailable = cast(Any, None)

    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        build_background_registry(
            config,
            unavailable,
            unavailable,
            unavailable,
            scientific_summary=unavailable,
            g1=unavailable,
            selection=unavailable,
            stage3_allowed=False,
            uv_lock_sha256="unused",
        )
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publish_registry_and_report(Path("unused"), config, unavailable)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="score-free stage-2R-0"):
        publish_registry_and_report_sealed(
            Path("unused"),
            config,
            unavailable,
            unavailable,
        )


_FREEZE_TAG = "v0.2.1-background-local-support-protocol"
_FREEZE_TAG_OBJECT = "06136e22bb8c6e2606a9debd5e00d53b500f758d"
_FREEZE_COMMIT = "966fb4e84c36aba373d90b81fe9e1350ffe349b6"
_SCORING_TAG = "v0.2.1-background-local-support-scoring-code"
_SCORING_TAG_OBJECT = "3" * 40
_CODE_COMMIT = "2" * 40
_PROTOCOL_SHA256 = "c7d6488bd97f0017867573c8b99230d79091412322652af25badfe732606e76a"
_FROZEN_BLOBS = {
    "configs/background_local_support.yaml": "d12bf40de8f5814e3e33b988f106ed8621538487",
    "data/manifests/background_local_support_fold_manifest.json": (
        "0ff9be3c5b4b330569fcf549616745910c734ecf"
    ),
    "data/manifests/background_local_support_manifest.json": (
        "1e93b6a0e76825bea0482bc25540c3782202b2aa"
    ),
}


def _local_support_execution_seal() -> ExecutionSeal:
    return ExecutionSeal(
        repository=RepositoryIdentity(
            code_commit=_CODE_COMMIT,
            branch="codex/stage2-local-support",
            upstream="origin/codex/stage2-local-support",
            upstream_commit=_CODE_COMMIT,
            freeze_tag=_FREEZE_TAG,
            freeze_tag_commit=_FREEZE_COMMIT,
            git_available=True,
            worktree_clean=True,
            tag_is_ancestor=True,
            upstream_matches_head=True,
        ),
        protocol_sha256=_PROTOCOL_SHA256,
        input_hashes=(
            (
                "data_catalog",
                "0bd12b428d6395f5623ce343e0f52bd2f5edfa0529bc406c72a297430a136a50",
            ),
            (
                "earthquake_dataset",
                "2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347",
            ),
            (
                "environment_lock",
                "34188ff1d0aa38996233412d36ef65eb5076205c376634368ddd66582097efa5",
            ),
            (
                "issue_manifest",
                "d7ae5266c9143ed0a67a9954da52039b2753f108698fb05477466a6d5b934e38",
            ),
            (
                "oracle_metadata",
                "28f66bdbc192be37923108b063036104d73c8ef32142a58e57e633a60be2e924",
            ),
            (
                "production_fixture",
                "c6db5e4079f583f0ff12d24136459368614ab389765081ef8945594762cc7a04",
            ),
            (
                "study_area",
                "5e5dcf012e080882161c95bf592a1ee39a0f0fdad7114bcff58d645aeb30bb02",
            ),
            (
                "support_manifest",
                "632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d",
            ),
        ),
    )


def _authorization_git_runner(
    *,
    remote_branch_commit: str = _CODE_COMMIT,
    scoring_tag_exists: bool = True,
    remote_url: str = "https://github.com/Justin-147/SeismoFlux.git",
    worktree_clean: bool = True,
) -> Any:
    def run(command: tuple[str, ...], _: Path) -> CommandResult:
        if command == ("git", "rev-parse", "--is-inside-work-tree"):
            return CommandResult(0, "true\n")
        if command == (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return CommandResult(0, "" if worktree_clean else " M changed.py\n")
        if command == ("git", "rev-parse", "--verify", "HEAD^{commit}"):
            return CommandResult(0, f"{_CODE_COMMIT}\n")
        if command == (
            "git",
            "rev-parse",
            "--verify",
            f"refs/tags/{_FREEZE_TAG}^{{commit}}",
        ):
            return CommandResult(0, f"{_FREEZE_COMMIT}\n")
        if command == (
            "git",
            "merge-base",
            "--is-ancestor",
            _FREEZE_COMMIT,
            _CODE_COMMIT,
        ):
            return CommandResult(0)
        if command == ("git", "symbolic-ref", "--quiet", "--short", "HEAD"):
            return CommandResult(0, "codex/stage2-local-support\n")
        if command == (
            "git",
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return CommandResult(0, "origin/codex/stage2-local-support\n")
        if command == ("git", "rev-parse", "--verify", "@{upstream}^{commit}"):
            return CommandResult(0, f"{_CODE_COMMIT}\n")
        if command == ("git", "remote", "get-url", "origin"):
            return CommandResult(0, f"{remote_url}\n")
        if command[:3] == ("git", "rev-parse", "--verify"):
            revision = command[3]
            for path, object_id in _FROZEN_BLOBS.items():
                if revision in (f"{_FREEZE_TAG}:{path}", f"HEAD:{path}"):
                    return CommandResult(0, f"{object_id}\n")
            if revision == f"refs/tags/{_FREEZE_TAG}^{{tag}}":
                return CommandResult(0, f"{_FREEZE_TAG_OBJECT}\n")
            if revision == f"refs/tags/{_SCORING_TAG}^{{tag}}":
                if not scoring_tag_exists:
                    return CommandResult(1, stderr="missing scoring tag")
                return CommandResult(0, f"{_SCORING_TAG_OBJECT}\n")
            if revision == f"refs/tags/{_SCORING_TAG}^{{commit}}":
                return CommandResult(0, f"{_CODE_COMMIT}\n")
        if command[:3] == ("git", "ls-remote", "--refs"):
            branch_ref = "refs/heads/codex/stage2-local-support"
            return CommandResult(
                0,
                "\n".join(
                    (
                        f"{remote_branch_commit}\t{branch_ref}",
                        f"{_FREEZE_TAG_OBJECT}\trefs/tags/{_FREEZE_TAG}",
                        f"{_SCORING_TAG_OBJECT}\trefs/tags/{_SCORING_TAG}",
                    )
                )
                + "\n",
            )
        return CommandResult(1, stderr=f"unexpected command: {command!r}")

    return run


def _accept_synthetic_execution_seal(monkeypatch: pytest.MonkeyPatch) -> None:
    def unchanged(
        _: Path,
        __: object,
        expected: ExecutionSeal,
        **___: object,
    ) -> ExecutionSeal:
        return expected

    monkeypatch.setattr(
        authorization_module,
        "require_execution_seal_unchanged",
        unchanged,
    )


def test_v021_authorization_binds_frozen_blobs_scoring_tag_remote_and_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_synthetic_execution_seal(monkeypatch)
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    seal = _local_support_execution_seal()
    authorized = create_authorized_execution(
        Path.cwd(),
        config,
        seal,
        runner=_authorization_git_runner(),
    )

    assert isinstance(authorized, AuthorizedExecution)
    assert authorized.execution_seal is seal
    assert len(authorized.authorization_id) == 64
    require_background_scoring_authorized(config, authorized)


def test_v021_authorization_rejects_missing_scoring_tag_before_rows_are_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_synthetic_execution_seal(monkeypatch)
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="scoring-code tag"):
        create_authorized_execution(
            Path.cwd(),
            config,
            _local_support_execution_seal(),
            runner=_authorization_git_runner(scoring_tag_exists=False),
        )


def test_v021_authorization_rejects_stale_live_remote_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_synthetic_execution_seal(monkeypatch)
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="live remote"):
        create_authorized_execution(
            Path.cwd(),
            config,
            _local_support_execution_seal(),
            runner=_authorization_git_runner(remote_branch_commit="4" * 40),
        )


def test_v021_authorization_rejects_another_public_remote_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_synthetic_execution_seal(monkeypatch)
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="Justin-147/SeismoFlux"):
        create_authorized_execution(
            Path.cwd(),
            config,
            _local_support_execution_seal(),
            runner=_authorization_git_runner(
                remote_url="https://github.com/someone-else/SeismoFlux.git"
            ),
        )


def test_v021_authorization_rechecks_seal_before_granting_capability() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    with pytest.raises(BackgroundScoringNotAuthorizedError, match="seal changed"):
        create_authorized_execution(
            Path.cwd(),
            config,
            _local_support_execution_seal(),
            runner=_authorization_git_runner(worktree_clean=False),
        )


def test_authorization_cannot_be_rebound_to_another_execution_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_synthetic_execution_seal(monkeypatch)
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    authorized = create_authorized_execution(
        Path.cwd(),
        config,
        _local_support_execution_seal(),
        runner=_authorization_git_runner(),
    )
    changed = ExecutionSeal(
        repository=authorized.execution_seal.repository,
        protocol_sha256=authorized.execution_seal.protocol_sha256,
        input_hashes=tuple(
            (name, "0" * 64 if name == "data_catalog" else value)
            for name, value in authorized.execution_seal.input_hashes
        ),
    )
    with pytest.raises(ValueError, match="another execution seal"):
        AuthorizedExecution(
            execution_seal=changed,
            scoring_authorization=authorized.scoring_authorization,
        )
