"""Clear exporter identity from an OOXML file while preserving its content."""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
}

IDENTITY_REPLACEMENTS = {
    b"Walnut Exporter": b"Microsoft Office",
    b"ChatGPT": b"SeismoFlux",
    b"OpenAI": b"",
    b"Codex": b"",
}


def _scrub_identity_tokens(payload: bytes) -> bytes:
    for source, replacement in IDENTITY_REPLACEMENTS.items():
        payload = payload.replace(source, replacement)
    return payload


def scrub(path: Path, *, title: str) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_metadata_",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(path, "r") as source:
            core = ET.fromstring(source.read("docProps/core.xml"))
            for name in ("dc:creator", "cp:lastModifiedBy"):
                element = core.find(name, NS)
                if element is not None:
                    element.text = ""
            title_element = core.find("dc:title", NS)
            if title_element is None:
                title_element = ET.SubElement(core, f"{{{NS['dc']}}}title")
            title_element.text = title
            core_bytes = _scrub_identity_tokens(
                ET.tostring(core, encoding="utf-8", xml_declaration=True)
            )

            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as destination:
                for info in source.infolist():
                    if info.filename == "docProps/core.xml":
                        destination.writestr(info, core_bytes)
                    else:
                        payload = source.read(info.filename)
                        if info.filename.endswith(".xml"):
                            payload = _scrub_identity_tokens(payload)
                        destination.writestr(info, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    scrub(args.path, title=args.title)


if __name__ == "__main__":
    main()
