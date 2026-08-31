from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROSPECTIVE_PATH = "outputs/prospective/p1_b0_r30/issue-001/raw/snapshot.bin"


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "-c",
            "core.safecrlf=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def test_prospective_runtime_artifact_is_stored_byte_for_byte_in_git_index(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    artifact = repository / PROSPECTIVE_PATH
    artifact.parent.mkdir(parents=True)
    original = (
        b"request-line\r\nheader: preserved\r\n\r\nbinary:\x00\xff\xfe\x80\r\ntrailing-crlf\r\n"
    )
    artifact.write_bytes(original)

    attribute = _git(repository, "check-attr", "text", "--", PROSPECTIVE_PATH).stdout
    assert attribute.decode("utf-8").strip().endswith(": text: unset")

    _git(repository, "add", "--", ".gitattributes", PROSPECTIVE_PATH)
    indexed = _git(repository, "cat-file", "blob", f":{PROSPECTIVE_PATH}").stdout

    assert indexed == original
    assert artifact.read_bytes() == original
