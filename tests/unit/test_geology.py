from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from openpyxl import Workbook  # type: ignore[import-untyped]
from shapely.geometry import shape

from seismoflux.data.common import SourceFile
from seismoflux.data.geology import (
    FAULT_ATTRIBUTE_HEADERS,
    FAULT_COORDINATE_HEADERS,
    LONG_TERM_HAZARD_HEADERS,
    GeologyParseResult,
    parse_geology_sources,
)

AVAILABLE_AT = datetime(2026, 7, 13, tzinfo=UTC)


def _write_workbook(path: Path, headers: tuple[str, ...], rows: list[list[object]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet.append(list(headers))
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    workbook.close()


def _source(path: Path, relative_path: str, source_id: str) -> SourceFile:
    payload = path.read_bytes()
    return SourceFile(
        source_id=source_id,
        relative_path=relative_path,
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        modified_at_utc="2026-07-13T00:00:00Z",
    )


def _write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.write_bytes(content.encode(encoding))


def _make_sources(
    root: Path,
    *,
    reverse_hazard_rows: bool = False,
    invalid_hazard_score: bool = False,
    missing_last_event: bool = False,
) -> dict[str, SourceFile]:
    root.mkdir(parents=True)
    coordinate_rows: list[list[object]] = [
        [101, 10101, 1, 100.0, 30.0, 0, "甲断裂", "甲段"],
        [101, 10101, 2, 101.0, 30.0, 0, "甲断裂", "甲段"],
        [101, 10101, 3, 101.0, 30.0, 0, "甲断裂", "甲段"],
        [101, 10101, 4, 102.0, 30.0, 0, "甲断裂", "甲段"],
        [102, 10201, 1, 110.0, 40.0, 1, "乙断层", "乙段"],
        [102, 10201, 2, 111.0, 40.0, 1, "乙断层", "乙段"],
    ]
    attribute_rows: list[list[object]] = [
        [
            101,
            "甲断裂",
            10101,
            "甲段",
            90,
            60,
            0,
            2.0,
            None,
            None,
            2.1,
            2.0,
            0.2,
            0.1,
            0.2,
            100,
            1,
            1900,
            1,
        ],
        [
            102,
            "乙断层",
            10201,
            "乙段",
            45,
            70,
            90,
            None,
            None,
            None,
            1.1,
            1.0,
            0.1,
            0.1,
            0.1,
            None,
            2,
            -500,
            2,
        ],
    ]
    if missing_last_event:
        attribute_rows[0][17] = None
        attribute_rows[0][18] = None
    first_score = 6.5 if invalid_hazard_score else 5.5
    hazard_rows: list[list[object]] = [
        [101, "甲断裂", 10101, "甲段", 3.0, 1.5, 1.0, 0.0, first_score],
        [102, "乙断层", 10201, "乙段", 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    if reverse_hazard_rows:
        hazard_rows.reverse()

    coordinate_path = root / "FaultCord.xlsx"
    attribute_path = root / "FaultAttri_ALLV4_修改.xlsx"
    hazard_path = root / "weight_analysis-修改newV4_修改.xlsx"
    _write_workbook(coordinate_path, FAULT_COORDINATE_HEADERS, coordinate_rows)
    _write_workbook(attribute_path, FAULT_ATTRIBUTE_HEADERS, attribute_rows)
    _write_workbook(hazard_path, LONG_TERM_HAZARD_HEADERS, hazard_rows)

    plotbd = root / "PlotBD"
    plotbd.mkdir()
    _write_text(
        plotbd / "FaultX.dat",
        "\n".join(
            [
                "> 甲断裂",
                "100 30",
                "102 30",
                "> 甲断裂",
                "101 31",
                "101 31",
                "> 乙断裂",
                "110 40",
                "111 40",
                "> 冲突名",
                "100 30",
                "102 30",
            ]
        )
        + "\n",
        encoding="gb18030",
    )
    _write_text(
        plotbd / "CN-border-L1.gmt",
        "\n".join(
            [
                ">",
                "100 20",
                "101 20",
                "101 21",
                "100 21",
                "100 20",
                ">",
                "90 20",
                "90 45",
                "120 45",
                "120 20",
                "90 20",
            ]
        )
        + "\n",
    )
    simple_map = ">\n100 30\n101 31\n"
    _write_text(plotbd / "CN-block-L1-deduced.gmt", simple_map)
    _write_text(plotbd / "CN-block-L1.gmt", simple_map)
    _write_text(plotbd / "CN-block-L2.gmt", simple_map)
    _write_text(
        plotbd / "CN-plate-neighbor.dat",
        "# provenance\n>\n100 30\n101 31\n",
    )

    paths = [
        coordinate_path,
        attribute_path,
        hazard_path,
        plotbd / "CN-block-L1-deduced.gmt",
        plotbd / "CN-block-L1.gmt",
        plotbd / "CN-block-L2.gmt",
        plotbd / "CN-border-L1.gmt",
        plotbd / "CN-plate-neighbor.dat",
        plotbd / "FaultX.dat",
    ]
    result: dict[str, SourceFile] = {}
    for path in paths:
        relative_path = path.name if path.parent == root else f"PlotBD/{path.name}"
        source_id = "geology" if path.parent == root else "plotbd"
        result[relative_path] = _source(path, relative_path, source_id)
    return result


def _parse(sources: dict[str, SourceFile]) -> GeologyParseResult:
    return parse_geology_sources(sources, "EPSG:6933", AVAILABLE_AT)


def test_key_join_is_invariant_to_hazard_row_order(tmp_path: Path) -> None:
    ordered = _parse(_make_sources(tmp_path / "ordered"))
    shuffled = _parse(_make_sources(tmp_path / "shuffled", reverse_hazard_rows=True))

    assert ordered.fault_segments == shuffled.fault_segments
    scores = {
        row["segment_number"]: row["long_term_hazard_score"] for row in shuffled.fault_segments
    }
    assert scores == {10101: 5.5, 10201: 0.0}
    assert ordered.quality["fault_key_join_rule"] == (
        "fault_id_and_segment_number_strict_one_to_one"
    )


def test_zero_length_trace_is_retained_but_quarantined(tmp_path: Path) -> None:
    result = _parse(_make_sources(tmp_path / "sources"))

    zero_length = [row for row in result.fault_traces if not row["usable_for_geometry"]]
    assert len(zero_length) == 1
    assert zero_length[0]["source_segment_number"] == 2
    assert zero_length[0]["geometry_wkb"] is None
    assert "zero_length_geometry" in zero_length[0]["quality_flags"]
    assert result.quality["fault_trace_zero_length_count"] == 1
    assert result.quality["fault_trace_usable_count"] == 3


def test_study_area_uses_largest_valid_closed_ring_and_orients_ccw(
    tmp_path: Path,
) -> None:
    result = _parse(_make_sources(tmp_path / "sources"))

    properties = result.study_area["properties"]
    polygon = shape(result.study_area["geometry"])
    assert properties["source_segment_number"] == 2
    assert properties["selection_rule"] == (
        "largest_valid_closed_ring_by_equal_area_then_lowest_segment_number"
    )
    assert properties["island_policy"] == (
        "continuous_mainland_excludes_hainan_taiwan_and_other_islands"
    )
    assert len(properties["source_sha256"]) == 64
    assert properties["geodesic_area_km2"] > 0
    assert properties["target_independent"] is True
    assert properties["predictor_feature"] is False
    assert polygon.is_valid
    assert polygon.exterior.is_ccw
    assert list(polygon.bounds) == [90.0, 20.0, 120.0, 45.0]
    assert result.fault_segments[0]["source_available_at"] == AVAILABLE_AT
    assert result.fault_segments[0]["historical_model_eligible"] is False
    assert result.fault_segments[0]["elapsed_ratio_at_snapshot"] == pytest.approx(1.26)
    assert result.fault_segments[0]["true_trace_id"] is None
    assert result.fault_segments[0]["true_trace_mapping_status"] == "unreviewed"
    assert result.fault_segments[0]["true_trace_match_confidence"] == "unassessed"


def test_hazard_score_must_equal_component_sum(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources", invalid_hazard_score=True)

    with pytest.raises(ValueError, match="does not equal component sum"):
        _parse(sources)


def test_missing_last_strong_earthquake_is_explicit_not_zero_filled(tmp_path: Path) -> None:
    result = _parse(_make_sources(tmp_path / "sources", missing_last_event=True))
    segment = result.fault_segments[0]

    assert segment["last_strong_earthquake_year_raw"] is None
    assert segment["last_strong_earthquake_reliability"] is None
    assert segment["elapsed_ratio_at_snapshot"] is None
    assert "last_strong_earthquake_year_raw" in segment["missing_fields"]
    assert "last_strong_earthquake_reliability" in segment["missing_fields"]
