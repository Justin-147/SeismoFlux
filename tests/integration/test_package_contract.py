from __future__ import annotations

from seismoflux import __version__
from seismoflux.cli import COMMAND_SPECS, build_parser


def test_package_version_and_console_contract() -> None:
    parser = build_parser()

    assert __version__ == "0.1.0"
    assert parser.prog == "seismoflux"
    assert len(COMMAND_SPECS) == 13
    assert {command for command, spec in COMMAND_SPECS.items() if spec.implemented} == {
        "build-anomaly-history",
        "build-background",
        "inventory",
        "ingest",
        "validate-data",
    }
    assert COMMAND_SPECS["forecast"].stage == 9
    assert COMMAND_SPECS["mature"].stage == 9
    assert all(
        not spec.implemented
        for command, spec in COMMAND_SPECS.items()
        if command
        not in {
            "build-anomaly-history",
            "build-background",
            "inventory",
            "ingest",
            "validate-data",
        }
    )
