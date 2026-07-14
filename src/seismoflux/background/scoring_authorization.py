"""Explicit protocol-version gate for background-model fitting and scoring."""

from __future__ import annotations

from seismoflux.background.config import BackgroundConfig


class BackgroundScoringNotAuthorizedError(RuntimeError):
    """Raised when a score-free preregistration is passed to a model pipeline."""


def require_background_scoring_authorized(config: BackgroundConfig) -> None:
    """Allow v0.2.0 only; v0.2.1 remains score-free until the 2R-0 tag exists."""

    protocol_version = str(config.protocol_version)
    if protocol_version == "0.2.0":
        return
    if protocol_version == "0.2.1":
        raise BackgroundScoringNotAuthorizedError(
            "background protocol 0.2.1 is the score-free stage-2R-0 preregistration; "
            "model fitting and scoring require the frozen "
            "v0.2.1-background-local-support-protocol tag and stage-2R-1 implementation"
        )
    raise BackgroundScoringNotAuthorizedError(
        f"background scoring is not authorized for protocol version {protocol_version!r}"
    )


__all__ = [
    "BackgroundScoringNotAuthorizedError",
    "require_background_scoring_authorized",
]
