"""Closed public entry points for Stage 2S execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seismoflux.stage2s.records import (
        Stage2SSyntheticAcceptanceRecord,
        Stage2SWholeRunRecord,
    )


def run_stage2s_synthetic_acceptance(
    *,
    repository_root: Path,
    scratch_root: Path,
) -> Stage2SSyntheticAcceptanceRecord:
    """Run the target-free same-path acceptance without opening real inputs."""

    from seismoflux.stage2s.protocol import load_protocol_bundle
    from seismoflux.stage2s.synthetic import run_synthetic_acceptance

    protocol = load_protocol_bundle(repository_root)
    return run_synthetic_acceptance(
        protocol=protocol,
        scratch_root=scratch_root,
    )


def run_stage2s_once(
    *,
    repository_root: Path,
    progress: Callable[[str], None] | None = print,
) -> Stage2SWholeRunRecord:
    """Run the single registered historical development attempt from the code tag.

    ``progress`` receives phase names only.  No assessment identity or metric is
    exposed before the master prediction seal.
    """

    from seismoflux.stage2s.execution_environment import (
        require_prepared_formal_execution_environment,
    )

    require_prepared_formal_execution_environment()

    # These imports must remain below the prepared-environment check.  The
    # formal launcher therefore constrains BLAS/OpenMP before NumPy/PyArrow and
    # the production module can initialize.
    from seismoflux.stage2s.production import (
        default_execution_services,
        execute_formal_once,
    )
    from seismoflux.stage2s.protocol import load_protocol_bundle

    protocol = load_protocol_bundle(repository_root)
    return execute_formal_once(
        protocol,
        services=default_execution_services(),
        progress=progress,
    )


__all__ = [
    "run_stage2s_once",
    "run_stage2s_synthetic_acceptance",
]
