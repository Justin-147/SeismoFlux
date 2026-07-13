from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from openpyxl import Workbook  # type: ignore[import-untyped]
from shapely.geometry import Polygon

from seismoflux.data.common import SourceFile, sha256_bytes
from seismoflux.data.earthquake import parse_earthquake_catalogs

M5_HEADERS = ("年", "月", "日", "时", "分", "秒", "纬度", "经度", "震级", "深度", "地名")


def _source(path: Path, source_id: str) -> SourceFile:
    return SourceFile(
        source_id=source_id,
        relative_path=path.name,
        path=path,
        sha256=sha256_bytes(path.read_bytes()),
        modified_at_utc="2026-07-13T00:00:00Z",
    )


def _write_m3(path: Path, lines: list[str]) -> SourceFile:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return _source(path, "earthquake_catalog_m3_plus")


def _write_m5(path: Path, rows: list[tuple[object, ...]]) -> SourceFile:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "表单1"
    sheet.append(M5_HEADERS)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()
    return _source(path, "earthquake_catalog_m5_plus")


def _study_area() -> Polygon:
    return Polygon([(80.0, 20.0), (130.0, 20.0), (130.0, 50.0), (80.0, 50.0)])


def test_fixed_utc_plus_08_second_60_mojibake_negative_depth_and_covers(
    tmp_path: Path,
) -> None:
    repaired_place = "新疆塔城地区乌苏市"
    mojibake = repaired_place.encode("gb18030").decode("latin-1")
    m3 = _write_m3(
        tmp_path / "catalog.eqt",
        [f" 19711101 133000.0 44.000000 84.900000 5.00 -1.000000 000 {mojibake}"],
    )
    m5 = _write_m5(
        tmp_path / "catalog.xlsx",
        [
            (1900, 1, 1, 0, 0, 0, 44.5, 85.5, 5.0, 10, "历史事件"),
            (1971, 11, 1, 13, 29, 60, 44.0, 84.9, 5.0, 10, repaired_place),
        ],
    )
    study_area = Polygon([(84.9, 43.0), (86.0, 43.0), (86.0, 45.0), (84.9, 45.0)])

    result = parse_earthquake_catalogs(m3, m5, study_area)

    assert len(result.events) == 2
    assert len(result.source_records) == 3
    assert len(result.dedup_candidates) == 1
    candidate = result.dedup_candidates[0]
    assert candidate.exact_match
    assert candidate.auto_eligible
    assert candidate.decision == "auto_merged"
    assert candidate.merged_event_id is not None

    repaired = next(record for record in result.source_records if record.source_id == m3.source_id)
    assert repaired.place_raw == mojibake
    assert repaired.place == repaired_place
    assert repaired.depth_km is None
    assert "place_repaired_latin1_gb18030" in repaired.normalization_flags
    assert "negative_depth_sentinel" in repaired.normalization_flags
    assert repaired.inside_study_area  # Exactly on the polygon boundary: covers, not contains.

    carried = next(
        record
        for record in result.source_records
        if record.source_id == m5.source_id and record.origin_time_local.year == 1971
    )
    assert carried.origin_time_local == datetime(
        1971, 11, 1, 13, 30, tzinfo=timezone(timedelta(hours=8))
    )
    assert carried.origin_time_utc == datetime(1971, 11, 1, 5, 30, tzinfo=UTC)
    assert carried.available_at == carried.origin_time_utc
    assert "publication_time_assumed_origin_time" in carried.normalization_flags
    assert "second_60_carried" in carried.normalization_flags

    historical = next(event for event in result.events if event.origin_time_local.year == 1900)
    assert historical.origin_time_local.utcoffset() == timedelta(hours=8)
    assert historical.origin_time_utc == datetime(1899, 12, 31, 16, 0, tzinfo=UTC)
    assert historical.origin_time_utc != datetime(1899, 12, 31, 15, 54, 17, tzinfo=UTC)
    assert result.quality["m5_second_60_records"] == 1
    assert result == parse_earthquake_catalogs(m3, m5, study_area)


def test_same_source_only_collapses_all_field_exact_duplicates(tmp_path: Path) -> None:
    duplicate = "20200101 000000.0 30.000000 100.000000 3.00 -1.000000 000 四川"
    m3 = _write_m3(
        tmp_path / "catalog.eqt",
        [
            duplicate,
            duplicate,
            "20200101 000030.0 30.000000 100.000000 3.10 5.000000 000 四川",
        ],
    )
    m5 = _write_m5(
        tmp_path / "catalog.xlsx",
        [(1900, 1, 1, 0, 0, 0, 40.0, 110.0, 5.0, 10, "历史事件")],
    )

    result = parse_earthquake_catalogs(m3, m5, _study_area())

    assert len(result.source_records) == 4
    assert len(result.events) == 3
    assert result.dedup_candidates == ()
    exact = next(event for event in result.events if len(event.catalog_sources) == 2)
    assert exact.dedup_confidence == "exact"
    assert "same_source_exact_duplicate_collapsed" in exact.quality_flags
    assert result.quality["same_source_duplicate_groups"] == 1
    assert result.quality["same_source_duplicate_records_collapsed"] == 1


def test_auto_threshold_ambiguity_never_merges(tmp_path: Path) -> None:
    m3 = _write_m3(
        tmp_path / "catalog.eqt",
        [
            "20200101 000000.0 30.000000 100.000000 5.00 10.000000 000 四川",
            "20200101 000004.0 30.000000 100.000000 5.00 10.000000 000 四川",
        ],
    )
    m5 = _write_m5(
        tmp_path / "catalog.xlsx",
        [(2020, 1, 1, 0, 0, 2, 30.0, 100.0, 5.0, 10, "四川")],
    )

    result = parse_earthquake_catalogs(m3, m5, _study_area())

    assert len(result.events) == 3
    assert len(result.dedup_candidates) == 2
    assert {candidate.decision for candidate in result.dedup_candidates} == {"ambiguous_unmerged"}
    assert all(candidate.auto_eligible for candidate in result.dedup_candidates)
    assert all(candidate.m3_auto_degree == 1 for candidate in result.dedup_candidates)
    assert all(candidate.m5_auto_degree == 2 for candidate in result.dedup_candidates)
    assert all(candidate.merged_event_id is None for candidate in result.dedup_candidates)
    assert {event.dedup_confidence for event in result.events} == {"ambiguous_unmerged"}
    assert result.quality["cross_auto_merged_candidates"] == 0
    assert result.quality["cross_auto_ambiguous_edges"] == 2


def test_review_threshold_candidate_is_retained_without_merge(tmp_path: Path) -> None:
    m3 = _write_m3(
        tmp_path / "catalog.eqt",
        ["20200101 000000.0 30.000000 100.000000 5.00 10.000000 000 四川"],
    )
    m5 = _write_m5(
        tmp_path / "catalog.xlsx",
        [(2020, 1, 1, 0, 2, 0, 30.0, 100.0, 5.0, 10, "四川")],
    )

    result = parse_earthquake_catalogs(m3, m5, _study_area())

    assert len(result.events) == 2
    assert len(result.dedup_candidates) == 1
    candidate = result.dedup_candidates[0]
    assert candidate.time_delta_seconds == 120.0
    assert not candidate.auto_eligible
    assert candidate.decision == "review_unmerged"
    assert candidate.merged_event_id is None
    assert {event.dedup_confidence for event in result.events} == {"review_unmerged"}
