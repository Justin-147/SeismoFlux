from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import xlrd

from seismoflux.data import anomaly as anomaly_module
from seismoflux.data.anomaly import parse_anomaly_files
from seismoflux.data.common import SourceFile, fixed_local_midnight


class _FakeSheet:
    def __init__(self, name: str, rows: list[list[object]]) -> None:
        self.name = name
        self._rows = rows
        self.nrows = len(rows)
        self.ncols = len(rows[0])
        self.visibility = 0
        self.merged_cells: tuple[object, ...] = ()

    def row_values(self, row: int, start: int, end: int) -> list[object]:
        return self._rows[row][start:end]


class _FakeWorkbook:
    def __init__(self, sheets: list[_FakeSheet]) -> None:
        self._sheets = sheets
        self.released = False

    def sheet_names(self) -> list[str]:
        return [sheet.name for sheet in self._sheets]

    def sheets(self) -> list[_FakeSheet]:
        return self._sheets

    def release_resources(self) -> None:
        self.released = True


WorkbookFactory = Callable[[], _FakeWorkbook]


def _source(tmp_path: Path, filename: str, payload: bytes) -> SourceFile:
    path = tmp_path / filename
    path.write_bytes(payload)
    return SourceFile(
        source_id="anomaly",
        relative_path=f"anomaly/{filename}",
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        modified_at_utc="2026-07-13T00:00:00Z",
    )


def _install_workbooks(
    monkeypatch: pytest.MonkeyPatch,
    factories: dict[bytes, WorkbookFactory],
) -> None:
    def fake_open_workbook(
        *, file_contents: bytes, on_demand: bool, formatting_info: bool
    ) -> _FakeWorkbook:
        assert on_demand is True
        assert formatting_info is True
        return factories[file_contents]()

    monkeypatch.setattr(xlrd, "open_workbook", fake_open_workbook)


def _common_row(
    serial: int,
    *,
    discipline: str = "形变",
    row_report_date: str = "2024-08-28",
    station: str = "测试台",
    longitude: object = 105.1,
    latitude: object = 35.2,
    measurement: str = "钻孔应变观测北南分量",
    start_time: str = "2024-01-01",
    end_time: str = "",
    report_state: str = "持续",
) -> list[object]:
    return [
        serial,
        "周报",
        row_report_date,
        "测试地震局",
        station,
        longitude,
        latitude,
        measurement,
        "1",
        discipline,
        "B",
        "",
        "是" if end_time else "否",
        start_time,
        end_time,
        "1" if end_time else "",
        "趋势",
        "测试异常",
        "否",
        "0.1",
        "0.0",
        "",
        "中期",
        report_state,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]


def _cross_fault_row(serial: int) -> list[object]:
    return [
        serial,
        "周报",
        "2024-08-28",
        "测试地震局",
        "测试场地",
        105.1,
        35.2,
        "(A←B)基线",
        "基线",
        "1月",
        "B",
        "1月",
        "否",
        "2024-01-01",
        "",
        "",
        "测试异常",
        "0.2",
        "mm",
        "否",
        "0.1",
        "0.0",
        "中期",
        "持续",
        "2",
        "人工核实报告",
        "人工分析意见",
        "12个月",
        "5级",
        "B",
        "震例",
        "人工说明",
    ]


def _workbook_factory(rows_by_sheet: dict[str, list[list[object]]]) -> WorkbookFactory:
    def factory() -> _FakeWorkbook:
        sheets: list[_FakeSheet] = []
        for name in anomaly_module._SHEET_ORDER:
            headers = (
                anomaly_module._COMMON_HEADERS
                if name in anomaly_module._COMMON_SHEETS
                else anomaly_module._CROSS_FAULT_HEADERS
            )
            sheet_rows: list[list[object]] = [list(headers)]
            sheet_rows.extend(list(row) for row in rows_by_sheet.get(name, []))
            sheets.append(_FakeSheet(name, sheet_rows))
        return _FakeWorkbook(sheets)

    return factory


def test_report_date_uses_filename_and_available_at_uses_latest_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"date-mismatch"
    source = _source(
        tmp_path,
        "异常零数据All202435期周报-修改-20240828.XLS",
        payload,
    )
    older = _common_row(1, row_report_date="2024-08-21")
    future = _common_row(
        2,
        row_report_date="2024-09-01",
        station="另一个台站",
        measurement="静水位",
    )
    _install_workbooks(monkeypatch, {payload: _workbook_factory({"形变": [older, future]})})

    result = parse_anomaly_files([source])

    first, second = result.observations
    assert first["report_date"] == date(2024, 8, 28)
    assert first["raw_row_report_date"] == date(2024, 8, 21)
    assert first["available_at"] == datetime(2024, 8, 28, 16, tzinfo=UTC)
    assert "row_report_date_before_source" in first["reliability_flags"]
    assert second["available_at"] == datetime(2024, 9, 1, 16, tzinfo=UTC)
    assert "available_at_conservatively_delayed" in second["reliability_flags"]
    assert result.report_periods[0]["row_report_date_mismatch_count"] == 2


def test_later_end_time_does_not_backfill_earlier_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_payload = b"right-censored"
    second_payload = b"ended"
    first_source = _source(
        tmp_path,
        "异常零数据All202435期周报-修改-20240828.XLS",
        first_payload,
    )
    second_source = _source(
        tmp_path,
        "异常零数据All202437期周报-修改-20240911.XLS",
        second_payload,
    )
    _install_workbooks(
        monkeypatch,
        {
            first_payload: _workbook_factory({"形变": [_common_row(1)]}),
            second_payload: _workbook_factory(
                {"形变": [_common_row(1, row_report_date="2024-09-11", end_time="2024-09-01")]}
            ),
        },
    )

    result = parse_anomaly_files([second_source, first_source])

    earlier, later = result.observations
    assert earlier["anomaly_id"] == later["anomaly_id"]
    assert earlier["reported_end_time"] is None
    assert earlier["right_censored"] is True
    assert later["reported_end_time"] == fixed_local_midnight(date(2024, 9, 1))
    assert later["start_time"] == fixed_local_midnight(date(2024, 1, 1))
    assert "anomaly_time_date_only_assumed_fixed_utc_plus_08_midnight" in later["reliability_flags"]
    assert later["right_censored"] is False
    assert result.quality["missing_report_periods"] == (
        {
            "report_year": 2024,
            "report_period": 36,
            "expected_report_date": "2024-09-04",
        },
    )


def test_incomplete_identity_uses_source_scoped_provisional_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_payload = b"missing-identity-1"
    second_payload = b"missing-identity-2"
    first_source = _source(
        tmp_path,
        "异常零数据All202435期周报-修改-20240828.XLS",
        first_payload,
    )
    second_source = _source(
        tmp_path,
        "异常零数据All202437期周报-修改-20240911.XLS",
        second_payload,
    )
    missing = _common_row(1, station="", longitude="", latitude="", measurement="")
    later_missing = _common_row(
        1,
        row_report_date="2024-09-11",
        station="",
        longitude="",
        latitude="",
        measurement="",
    )
    _install_workbooks(
        monkeypatch,
        {
            first_payload: _workbook_factory({"形变": [missing]}),
            second_payload: _workbook_factory({"形变": [later_missing]}),
        },
    )

    result = parse_anomaly_files([first_source, second_source])

    first, second = result.observations
    assert first["identity_complete"] is False
    assert first["station_id"] != second["station_id"]
    assert first["anomaly_id"] != second["anomaly_id"]
    assert {
        "missing_station_name",
        "missing_longitude",
        "missing_latitude",
        "missing_measurement",
        "identity_incomplete",
        "entity_unresolved",
    }.issubset(first["reliability_flags"])
    incomplete_audits = [
        audit for audit in result.entity_audit if audit["audit_type"] == "identity_incomplete"
    ]
    assert len(incomplete_audits) == 2


def test_exact_duplicates_are_preserved_with_distinct_observation_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"duplicates"
    source = _source(
        tmp_path,
        "异常零数据All202435期周报-修改-20240828.XLS",
        payload,
    )
    first = _common_row(1)
    second = _common_row(2)
    _install_workbooks(monkeypatch, {payload: _workbook_factory({"形变": [first, second]})})

    result = parse_anomaly_files([source])

    assert len(result.observations) == 2
    assert result.observations[0]["observation_id"] != result.observations[1]["observation_id"]
    assert all(
        "exact_duplicate_in_report" in observation["reliability_flags"]
        for observation in result.observations
    )
    assert result.quality["exact_duplicate_group_count"] == 1
    assert result.quality["exact_duplicate_excess_row_count"] == 1
    assert result.quality["natural_key_collision_group_count"] == 1


def test_cross_fault_period_is_not_parsed_as_a_date_and_manual_fields_are_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"cross-fault"
    source = _source(
        tmp_path,
        "异常零数据All202435期周报-修改-20240828.XLS",
        payload,
    )
    _install_workbooks(
        monkeypatch,
        {payload: _workbook_factory({"跨断层": [_cross_fault_row(1)]})},
    )

    result = parse_anomaly_files([source])

    observation = result.observations[0]
    assert observation["discipline"] == "跨断层"
    assert observation["observation_period"] == "1月"
    assert observation["trend_time"] is None
    assert "cross_fault_trend_field_is_period" in observation["reliability_flags"]
    assert "manual_prediction_fields_excluded" in observation["reliability_flags"]
    forbidden = {
        "duration",
        "predicted_region",
        "predicted_place",
        "predicted_magnitude",
        "predicted_time",
        "analysis_opinion",
        "prediction_strength",
        "confidence",
        "earthquake_examples",
    }
    assert forbidden.isdisjoint(observation)


def test_parsing_is_deterministic_for_reversed_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_payload = b"deterministic-1"
    second_payload = b"deterministic-2"
    first_source = _source(
        tmp_path,
        "异常零数据All202435期周报-修改-20240828.XLS",
        first_payload,
    )
    second_source = _source(
        tmp_path,
        "异常零数据All202437期周报-修改-20240911.XLS",
        second_payload,
    )
    factories = {
        first_payload: _workbook_factory({"形变": [_common_row(1)]}),
        second_payload: _workbook_factory({"形变": [_common_row(1, row_report_date="2024-09-11")]}),
    }
    _install_workbooks(monkeypatch, factories)

    forward = parse_anomaly_files([first_source, second_source])
    reverse = parse_anomaly_files([second_source, first_source])

    assert forward == reverse
