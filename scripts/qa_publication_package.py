"""Validate and render the SeismoFlux science-communication package.

This is a read-only scientific-consistency and publication-layout check.  It
does not import or modify the frozen forecasting runtime.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfReader


PACKAGE = Path("outputs/publication/seismoflux_b0_r30_v1")
BANNED_PROVENANCE_TOKENS = ("Walnut Exporter", "OpenAI", "ChatGPT", "Codex")


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _render_pdf(pdf_path: Path, output_dir: Path, *, target_long_edge: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_page in output_dir.glob("page-*.png"):
        stale_page.unlink()
    document = pdfium.PdfDocument(pdf_path)
    for index in range(len(document)):
        page = document[index]
        width, height = page.get_size()
        scale = target_long_edge / max(width, height)
        bitmap = page.render(scale=scale, rev_byteorder=True)
        image = bitmap.to_pil().convert("RGB")
        image.save(output_dir / f"page-{index + 1:02d}.png", optimize=True)
        page.close()
    document.close()
    return index + 1 if "index" in locals() else 0


def _pdf_summary(path: Path) -> dict[str, object]:
    reader = PdfReader(path)
    text_lengths: list[int] = []
    page_sizes_points: list[list[float]] = []
    fonts: set[str] = set()
    for page in reader.pages:
        text_lengths.append(len((page.extract_text() or "").strip()))
        page_sizes_points.append(
            [round(float(page.mediabox.width), 2), round(float(page.mediabox.height), 2)]
        )
        resources = page.get("/Resources")
        if resources is None:
            continue
        resource_object = resources.get_object()
        font_dict = resource_object.get("/Font")
        if font_dict is None:
            continue
        for font_ref in font_dict.get_object().values():
            font = font_ref.get_object()
            base_font = font.get("/BaseFont")
            if base_font is not None:
                fonts.add(str(base_font).lstrip("/"))
    return {
        "pages": len(reader.pages),
        "text_lengths": text_lengths,
        "page_sizes_points": page_sizes_points,
        "fonts": sorted(fonts),
        "metadata": {str(key): str(value) for key, value in (reader.metadata or {}).items()},
    }


def _zip_slide_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return len(
            [
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ]
        )


def _ooxml_creator(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        core = archive.read("docProps/core.xml").decode("utf-8")
    match = re.search(r"<dc:creator[^>]*>(.*?)</dc:creator>", core, flags=re.DOTALL)
    return "" if match is None else match.group(1).strip()


def _ooxml_xml_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return "\n".join(
            archive.read(name).decode("utf-8", errors="replace")
            for name in archive.namelist()
            if name.endswith(".xml")
        )


def main() -> int:
    failures: list[str] = []
    report: dict[str, object] = {}

    payload_path = PACKAGE / "build/science_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    _check(metrics["hits"]["B0"][2] == 5, "B0 primary hits changed", failures)
    _check(metrics["hits"]["B0_R30"][2] == 9, "B0_R30 primary hits changed", failures)
    _check(len(payload["clusters"]) == 21, "primary cluster count is not 21", failures)
    _check(len(payload["cases"]) == 4, "illustrative case count is not four", failures)
    outcomes = Counter(item["outcome"] for item in payload["clusters"])
    _check(outcomes == {"both_hit": 5, "gain": 4, "both_miss": 12}, "paired outcomes changed", failures)

    with (PACKAGE / "figure_data/primary_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        primary_rows = list(csv.DictReader(handle))
    _check(len(primary_rows) >= 2, "primary metric table is empty", failures)
    primary_lookup = {
        (int(row["horizon_days"]), int(float(row["area_budget_km2"])), row["model_id"]): row
        for row in primary_rows
    }
    b0_primary = primary_lookup.get((30, 600_000, "B0"), {})
    r30_primary = primary_lookup.get((30, 600_000, "B0_R30"), {})
    _check(b0_primary.get("hit_count") == "5", "primary CSV B0 hits changed", failures)
    _check(r30_primary.get("hit_count") == "9", "primary CSV B0_R30 hits changed", failures)
    _check(r30_primary.get("bootstrap_lower_95") == "0.04761905", "primary CSV bootstrap lower bound changed", failures)
    _check(r30_primary.get("bootstrap_upper_95") == "0.38095238", "primary CSV bootstrap upper bound changed", failures)
    _check(
        r30_primary.get("bootstrap_positive_gain_replicate_proportion") == "0.99050000",
        "primary CSV positive-gain replicate proportion changed",
        failures,
    )

    figures: dict[str, dict[str, object]] = {}
    for index in range(1, 6):
        stem = f"figure_{index:02d}_"
        pngs = sorted((PACKAGE / "figures").glob(f"{stem}*.png"))
        svgs = sorted((PACKAGE / "figures").glob(f"{stem}*.svg"))
        _check(len(pngs) == 1 and len(svgs) == 1, f"figure {index} formats incomplete", failures)
        if pngs:
            with Image.open(pngs[0]) as image:
                figures[pngs[0].name] = {"width": image.width, "height": image.height}
                _check(image.width >= 1800 and image.height >= 1000, f"figure {index} PNG too small", failures)
    report["figures"] = figures

    html_path = PACKAGE / "interactive/seismoflux_science_explorer.html"
    html = html_path.read_text(encoding="utf-8")
    scrubbed_html = html.replace("http://www.w3.org/2000/svg", "")
    _check("<script src=" not in html.lower(), "interactive page has external script", failures)
    _check("<link " not in html.lower(), "interactive page has external stylesheet", failures)
    _check("fetch(" not in html.lower(), "interactive page performs a network fetch", failures)
    _check("http://" not in scrubbed_html and "https://" not in scrubbed_html, "interactive page has an external URL", failures)
    for anchor in ("performance", "clusters", "cases", "scope"):
        _check(f'id="{anchor}"' in html, f"interactive panel missing: {anchor}", failures)

    manuscript_docx = PACKAGE / "manuscript/seismoflux_manuscript_zh.docx"
    presentation_pptx = PACKAGE / "presentation/seismoflux_science_presentation_zh.pptx"
    poster_pptx = PACKAGE / "poster/seismoflux_scientific_poster_a0_landscape.pptx"
    for label, path in {
        "DOCX": manuscript_docx,
        "presentation PPTX": presentation_pptx,
        "poster PPTX": poster_pptx,
    }.items():
        _check(_ooxml_creator(path) == "", f"{label} creator metadata was not scrubbed", failures)
        xml_text = _ooxml_xml_text(path)
        for token in BANNED_PROVENANCE_TOKENS:
            _check(token not in xml_text, f"{label} contains unwanted provenance token: {token}", failures)
    _check(_zip_slide_count(presentation_pptx) == 12, "presentation does not contain 12 slides", failures)
    _check(_zip_slide_count(poster_pptx) == 1, "poster source does not contain one slide", failures)

    pdf_specs = {
        "manuscript": (
            PACKAGE / "manuscript/seismoflux_manuscript_zh.pdf",
            PACKAGE / "qa/manuscript_pdf_pages_v3",
            1600,
            None,
        ),
        "presentation": (
            PACKAGE / "presentation/seismoflux_science_presentation_zh.pdf",
            PACKAGE / "qa/presentation_pdf_pages_v3",
            1600,
            12,
        ),
        "poster": (
            PACKAGE / "poster/seismoflux_scientific_poster_a0_landscape.pdf",
            PACKAGE / "qa/poster_pdf_pages_v3",
            2400,
            1,
        ),
    }
    pdf_report: dict[str, object] = {}
    for label, (pdf_path, output_dir, target_edge, expected_pages) in pdf_specs.items():
        summary = _pdf_summary(pdf_path)
        rendered_pages = _render_pdf(pdf_path, output_dir, target_long_edge=target_edge)
        summary["rendered_pages"] = rendered_pages
        pdf_report[label] = summary
        _check(rendered_pages == summary["pages"], f"{label} PDF render count mismatch", failures)
        if expected_pages is not None:
            _check(summary["pages"] == expected_pages, f"{label} PDF page count changed", failures)
        if label == "manuscript":
            _check(summary["pages"] >= 10, "manuscript is unexpectedly short", failures)
            _check(
                min(summary["text_lengths"][1:-1]) >= 150,
                "manuscript contains a near-blank internal page",
                failures,
            )
        if label == "poster":
            width_points, height_points = summary["page_sizes_points"][0]
            _check(
                3300 <= width_points <= 3450 and 2350 <= height_points <= 2450,
                "poster PDF is not A0 landscape size",
                failures,
            )
        _check(sum(summary["text_lengths"]) >= 500, f"{label} PDF has too little editable/searchable text", failures)
        _check(bool(summary["fonts"]), f"{label} PDF exposes no text fonts", failures)
        metadata_text = "\n".join(str(value) for value in summary["metadata"].values())
        for token in BANNED_PROVENANCE_TOKENS:
            _check(token not in metadata_text, f"{label} PDF metadata contains: {token}", failures)
    report["pdfs"] = pdf_report

    report["science"] = {
        "primary_hits": {"B0": 5, "B0_R30": 9},
        "paired_outcomes": dict(outcomes),
        "cluster_count": len(payload["clusters"]),
        "case_count": len(payload["cases"]),
    }
    report["status"] = "passed" if not failures else "failed"
    report["failures"] = failures
    destination = PACKAGE / "qa/publication_qa_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
