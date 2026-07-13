"""Deterministic ingestion of the legacy weekly anomaly workbooks."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TypedDict

import xlrd

from seismoflux.data.common import (
    SourceFile,
    canonical_decimal,
    conservative_available_at,
    fixed_local_midnight,
    normalize_text,
    read_stable_bytes,
    stable_uuid,
)

_FILE_NAME = re.compile(
    r"^异常零数据All(?P<year>\d{4})(?P<period>\d{2})期周报-修改-"
    r"(?P<report_date>\d{8})\.XLS$",
    re.IGNORECASE,
)

_COMMON_HEADERS = (
    "序号",
    "报告类型",
    "报告日期",
    "单位",
    "台站",
    "经度",
    "纬度",
    "分量名",
    "测点编号",
    "学科",
    "预报效能",
    "趋势性异常时间",
    "异常是否结束",
    "异常起始时间",
    "异常结束时间",
    "异常持续时间",
    "异常类型",
    "异常特征描述",
    "年度异常",
    "R值",
    "R0值",
    "震例信息",
    "中长期_短期",
    "新增_持续_取消",
    "预测区域",
    "预测地点",
    "预测起始震级",
    "预测结束震级",
    "预测起始时间",
    "预测结束时间",
    "其它说明",
)

_CROSS_FAULT_HEADERS = (
    "序号",
    "报告类型",
    "报告日期",
    "单位",
    "场地",
    "经度",
    "纬度",
    "分量名",
    "手段",
    "观测周期",
    "预报效能",
    "趋势性异常时间",
    "异常是否结束",
    "异常起始时间",
    "异常结束时间",
    "异常持续时间",
    "异常特征描述",
    "异常幅度_百分比或幅度值",
    "异常幅度值单位",
    "年度异常",
    "R值",
    "R0值",
    "中长期_短期",
    "新增_持续_取消",
    "现场异常核实次数",
    "现场核实报告名称",
    "分析意见",
    "预测时间_月",
    "预测强度",
    "异常信度",
    "震例信息",
    "其它说明",
)

_SHEET_ORDER = ("形变", "流体", "电磁", "跨断层")
_SHEET_INDEX = {name: index for index, name in enumerate(_SHEET_ORDER)}
_COMMON_SHEETS = frozenset(_SHEET_ORDER[:3])
_END_FLAGS = frozenset({"是", "否"})
_REPORT_STATES = frozenset({"新增", "持续", "取消"})


class AnomalyObservation(TypedDict):
    observation_id: str
    anomaly_id: str
    station_id: str
    report_date: date
    raw_row_report_date: date
    available_at: datetime
    unit: str
    station_name: str | None
    longitude: float | None
    latitude: float | None
    discipline: str
    measurement: str | None
    instrument_or_method: str | None
    observation_period: str | None
    forecast_efficacy: str | None
    trend_time: date | None
    start_time: datetime
    reported_end_time: datetime | None
    end_flag: str
    report_state: str
    anomaly_type: str | None
    anomaly_description: str | None
    annual_anomaly: str | None
    r_value: str | None
    r0_value: str | None
    term_class: str | None
    is_listed: bool
    right_censored: bool
    identity_complete: bool
    reliability_flags: tuple[str, ...]
    source_id: str
    source_file: str
    source_sha256: str
    source_sheet: str
    source_row: int
    source_serial_number: int
    source_period_year: int
    source_period: int


class EntityAuditRecord(TypedDict):
    audit_id: str
    audit_type: str
    status: str
    source_file: str | None
    source_sheet: str | None
    report_date: date | None
    anomaly_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    reliability_flags: tuple[str, ...]
    previous_reported_end_time: datetime | None
    current_reported_end_time: datetime | None


class ReportPeriodRecord(TypedDict):
    report_id: str
    source_id: str
    source_file: str
    source_sha256: str
    report_year: int
    report_period: int
    report_date: date
    available_at: datetime
    row_count: int
    row_report_date_mismatch_count: int
    row_report_date_before_count: int
    row_report_date_after_count: int
    deformation_row_count: int
    fluid_row_count: int
    electromagnetic_row_count: int
    cross_fault_row_count: int


@dataclass(frozen=True, slots=True)
class AnomalyParseResult:
    observations: tuple[AnomalyObservation, ...]
    entity_audit: tuple[EntityAuditRecord, ...]
    report_periods: tuple[ReportPeriodRecord, ...]
    quality: dict[str, object]


@dataclass(frozen=True, slots=True)
class _FileMetadata:
    source: SourceFile
    report_year: int
    report_period: int
    report_date: date


@dataclass(slots=True)
class _ParsedRow:
    observation: AnomalyObservation
    flags: set[str]
    exact_content_key: tuple[str, ...]
    natural_key: tuple[str, ...]


def _parse_source_metadata(source: SourceFile) -> _FileMetadata:
    filename = Path(source.relative_path).name
    match = _FILE_NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"unexpected anomaly filename: {source.relative_path}")
    try:
        report_date = datetime.strptime(match.group("report_date"), "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(
            f"invalid anomaly report date in filename: {source.relative_path}"
        ) from exc
    return _FileMetadata(
        source=source,
        report_year=int(match.group("year")),
        report_period=int(match.group("period")),
        report_date=report_date,
    )


def _is_blank(value: object | None) -> bool:
    return normalize_text(value) is None


def _required_text(value: object | None, *, field: str, location: str) -> str:
    text = normalize_text(value)
    if text is None:
        raise ValueError(f"missing {field}: {location}")
    return text


def _parse_date(
    value: object | None,
    *,
    field: str,
    location: str,
    allow_none: bool = False,
) -> date | None:
    text = normalize_text(value)
    if text is None:
        if allow_none:
            return None
        raise ValueError(f"missing {field}: {location}")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid {field} {text!r}: {location}") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"non-canonical {field} {text!r}: {location}")
    return parsed


def _parse_coordinate(
    value: object | None,
    *,
    field: str,
    lower: Decimal,
    upper: Decimal,
    location: str,
) -> float | None:
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"invalid {field} {value!r}: {location}")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {field} {value!r}: {location}") from exc
    if not parsed.is_finite() or not lower <= parsed <= upper:
        raise ValueError(f"out-of-range {field} {value!r}: {location}")
    return float(parsed)


def _parse_serial(value: object | None, *, expected: int, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid serial number {value!r}: {location}")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid serial number {value!r}: {location}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"invalid serial number {value!r}: {location}")
    serial = int(parsed)
    if serial != expected:
        raise ValueError(f"non-contiguous serial number {serial}, expected {expected}: {location}")
    return serial


def _key_value(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | Decimal):
        return canonical_decimal(value)
    return normalize_text(value) or ""


def _sheet_rows(sheet: Any) -> list[list[object]]:
    return [list(sheet.row_values(row_index, 0, sheet.ncols)) for row_index in range(sheet.nrows)]


def _record_audit(
    *,
    audit_type: str,
    source_file: str | None,
    source_sheet: str | None,
    report_date: date | None,
    anomaly_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    reliability_flags: tuple[str, ...],
    previous_end: datetime | None = None,
    current_end: datetime | None = None,
) -> EntityAuditRecord:
    audit_id = stable_uuid(
        "anomaly_entity_audit",
        audit_type,
        source_file,
        source_sheet,
        report_date,
        *observation_ids,
        previous_end,
        current_end,
    )
    return {
        "audit_id": audit_id,
        "audit_type": audit_type,
        "status": "unreviewed",
        "source_file": source_file,
        "source_sheet": source_sheet,
        "report_date": report_date,
        "anomaly_ids": anomaly_ids,
        "observation_ids": observation_ids,
        "reliability_flags": reliability_flags,
        "previous_reported_end_time": previous_end,
        "current_reported_end_time": current_end,
    }


def _missing_periods(periods: list[ReportPeriodRecord]) -> tuple[dict[str, object], ...]:
    by_year: dict[int, dict[int, date]] = defaultdict(dict)
    for period in periods:
        by_year[period["report_year"]][period["report_period"]] = period["report_date"]

    missing: list[dict[str, object]] = []
    for year, dated_periods in sorted(by_year.items()):
        if not dated_periods:
            continue
        first = min(dated_periods)
        last = max(dated_periods)
        for absent in range(first, last + 1):
            if absent in dated_periods:
                continue
            anchor = min(dated_periods, key=lambda value: (abs(value - absent), value))
            expected = dated_periods[anchor] + timedelta(days=7 * (absent - anchor))
            missing.append(
                {
                    "report_year": year,
                    "report_period": absent,
                    "expected_report_date": expected.isoformat(),
                }
            )
    return tuple(missing)


def parse_anomaly_files(files: list[SourceFile] | tuple[SourceFile, ...]) -> AnomalyParseResult:
    """Parse immutable XLS inputs without carrying future or human forecast information."""

    metadata = [_parse_source_metadata(source) for source in files]
    metadata.sort(
        key=lambda item: (
            item.report_date,
            item.report_year,
            item.report_period,
            item.source.relative_path,
            item.source.sha256,
        )
    )
    relative_paths = [item.source.relative_path for item in metadata]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("duplicate anomaly source paths are forbidden")
    period_keys = [(item.report_year, item.report_period) for item in metadata]
    if len(period_keys) != len(set(period_keys)):
        raise ValueError("duplicate anomaly report periods are forbidden")

    parsed_rows: list[_ParsedRow] = []
    report_periods: list[ReportPeriodRecord] = []
    entity_audit: list[EntityAuditRecord] = []
    rows_by_discipline: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    end_flags: Counter[str] = Counter()
    report_states: Counter[str] = Counter()
    report_date_offsets: Counter[int] = Counter()
    blank_row_count = 0
    merged_cell_count = 0

    for item in metadata:
        payload = read_stable_bytes(item.source)
        try:
            workbook = xlrd.open_workbook(
                file_contents=payload,
                on_demand=True,
                formatting_info=True,
            )
        except Exception as exc:
            raise ValueError(f"cannot parse anomaly workbook: {item.source.relative_path}") from exc
        try:
            if tuple(workbook.sheet_names()) != _SHEET_ORDER:
                raise ValueError(
                    f"unexpected anomaly worksheets in {item.source.relative_path}: "
                    f"{workbook.sheet_names()!r}"
                )

            period_rows: Counter[str] = Counter()
            mismatch_count = 0
            before_count = 0
            after_count = 0

            for sheet in workbook.sheets():
                if getattr(sheet, "visibility", 0) != 0:
                    raise ValueError(
                        f"hidden anomaly worksheet is forbidden: "
                        f"{item.source.relative_path}/{sheet.name}"
                    )
                merged_cells = tuple(getattr(sheet, "merged_cells", ()))
                merged_cell_count += len(merged_cells)
                if merged_cells:
                    raise ValueError(
                        f"merged anomaly cells are forbidden: "
                        f"{item.source.relative_path}/{sheet.name}"
                    )
                sheet_rows = _sheet_rows(sheet)
                if not sheet_rows:
                    raise ValueError(
                        f"empty anomaly worksheet: {item.source.relative_path}/{sheet.name}"
                    )
                expected_headers = (
                    _COMMON_HEADERS if sheet.name in _COMMON_SHEETS else _CROSS_FAULT_HEADERS
                )
                headers = tuple(normalize_text(value) for value in sheet_rows[0])
                if headers != expected_headers:
                    raise ValueError(
                        f"unexpected anomaly headers: {item.source.relative_path}/{sheet.name}"
                    )

                expected_serial = 1
                for zero_based_row, values in enumerate(sheet_rows[1:], start=1):
                    source_row = zero_based_row + 1
                    location = f"{item.source.relative_path}/{sheet.name}!row={source_row}"
                    if all(_is_blank(value) for value in values):
                        blank_row_count += 1
                        continue
                    serial = _parse_serial(values[0], expected=expected_serial, location=location)
                    expected_serial += 1

                    report_type = _required_text(values[1], field="report_type", location=location)
                    if report_type != "周报":
                        raise ValueError(f"unexpected report type {report_type!r}: {location}")
                    raw_report_date_value = _parse_date(
                        values[2], field="raw_row_report_date", location=location
                    )
                    assert raw_report_date_value is not None
                    raw_report_date = raw_report_date_value
                    unit = _required_text(values[3], field="unit", location=location)
                    station_name = normalize_text(values[4])
                    longitude = _parse_coordinate(
                        values[5],
                        field="longitude",
                        lower=Decimal("-180"),
                        upper=Decimal("180"),
                        location=location,
                    )
                    latitude = _parse_coordinate(
                        values[6],
                        field="latitude",
                        lower=Decimal("-90"),
                        upper=Decimal("90"),
                        location=location,
                    )
                    measurement = normalize_text(values[7])
                    instrument_or_method = normalize_text(values[8])
                    if sheet.name in _COMMON_SHEETS:
                        discipline = _required_text(
                            values[9], field="discipline", location=location
                        )
                        if discipline != sheet.name:
                            raise ValueError(
                                f"discipline differs from worksheet {discipline!r}: {location}"
                            )
                        observation_period = None
                        trend_time = _parse_date(
                            values[11],
                            field="trend_time",
                            location=location,
                            allow_none=True,
                        )
                        anomaly_type = normalize_text(values[16])
                        anomaly_description = normalize_text(values[17])
                        annual_anomaly = normalize_text(values[18])
                        r_value = normalize_text(values[19])
                        r0_value = normalize_text(values[20])
                        term_class = normalize_text(values[22])
                        manual_indices = (21, 24, 25, 26, 27, 28, 29, 30)
                    else:
                        discipline = sheet.name
                        observation_period = normalize_text(values[11])
                        trend_time = None
                        anomaly_type = None
                        anomaly_description = normalize_text(values[16])
                        annual_anomaly = normalize_text(values[19])
                        r_value = normalize_text(values[20])
                        r0_value = normalize_text(values[21])
                        term_class = normalize_text(values[22])
                        manual_indices = (24, 25, 26, 27, 28, 29, 30, 31)

                    end_flag = _required_text(values[12], field="end_flag", location=location)
                    if end_flag not in _END_FLAGS:
                        raise ValueError(f"unexpected end flag {end_flag!r}: {location}")
                    start_value = _parse_date(values[13], field="start_time", location=location)
                    assert start_value is not None
                    start_date = start_value
                    reported_end_date = _parse_date(
                        values[14],
                        field="reported_end_time",
                        location=location,
                        allow_none=True,
                    )
                    report_state = _required_text(
                        values[23], field="report_state", location=location
                    )
                    if report_state not in _REPORT_STATES:
                        raise ValueError(f"unexpected report state {report_state!r}: {location}")

                    flags: set[str] = set()
                    delta_days = (raw_report_date - item.report_date).days
                    report_date_offsets[delta_days] += 1
                    if delta_days != 0:
                        mismatch_count += 1
                        flags.add("row_report_date_mismatch_source")
                        if delta_days < 0:
                            before_count += 1
                            flags.add("row_report_date_before_source")
                        else:
                            after_count += 1
                            flags.add("row_report_date_after_source")
                            flags.add("available_at_conservatively_delayed")

                    if station_name is None:
                        missing_fields["station_name"] += 1
                        flags.add("missing_station_name")
                    if longitude is None:
                        missing_fields["longitude"] += 1
                        flags.add("missing_longitude")
                    if latitude is None:
                        missing_fields["latitude"] += 1
                        flags.add("missing_latitude")
                    if measurement is None:
                        missing_fields["measurement"] += 1
                        flags.add("missing_measurement")

                    station_complete = (
                        station_name is not None and longitude is not None and latitude is not None
                    )
                    observation_id = stable_uuid(
                        "anomaly_observation",
                        item.source.sha256,
                        sheet.name,
                        source_row,
                    )
                    if station_complete:
                        station_id = stable_uuid(
                            "station",
                            unit,
                            station_name,
                            canonical_decimal(longitude, "0.00001"),
                            canonical_decimal(latitude, "0.00001"),
                        )
                    else:
                        station_id = stable_uuid("station_provisional", observation_id)

                    identity_complete = station_complete and measurement is not None
                    if identity_complete:
                        anomaly_id = stable_uuid(
                            "anomaly",
                            station_id,
                            discipline,
                            measurement,
                            instrument_or_method,
                            start_date.isoformat(),
                        )
                    else:
                        anomaly_id = stable_uuid("anomaly_provisional", observation_id)
                        flags.update({"identity_incomplete", "entity_unresolved"})

                    knowledge_date = max(item.report_date, raw_report_date)
                    right_censored = reported_end_date is None or reported_end_date > knowledge_date
                    if reported_end_date is not None:
                        if reported_end_date > item.report_date:
                            flags.add("future_reported_end_time")
                        if reported_end_date > raw_report_date:
                            flags.add("end_after_raw_row_report_date")
                        if reported_end_date < start_date:
                            flags.add("end_before_start")
                        if reported_end_date < item.report_date:
                            flags.add("listed_after_reported_end")
                    if start_date > knowledge_date:
                        flags.add("start_after_report_date")
                    if (end_flag == "是") != (reported_end_date is not None):
                        flags.add("end_flag_inconsistent")
                    if report_state == "取消" and reported_end_date is None:
                        flags.add("cancel_without_end_time")
                    elif report_state == "持续" and reported_end_date is not None:
                        flags.add("continued_with_end_time")
                    elif report_state == "新增" and reported_end_date is not None:
                        flags.add("new_with_end_time")
                    if sheet.name not in _COMMON_SHEETS:
                        flags.add("cross_fault_trend_field_is_period")
                    if not _is_blank(values[15]):
                        flags.add("source_duration_ignored")
                    flags.update(
                        {
                            "anomaly_time_date_only_assumed_fixed_utc_plus_08_midnight",
                            "manual_prediction_fields_excluded",
                        }
                    )
                    if any(not _is_blank(values[index]) for index in manual_indices):
                        flags.add("manual_prediction_fields_present")

                    start_time = fixed_local_midnight(start_date)
                    reported_end_time = (
                        None
                        if reported_end_date is None
                        else fixed_local_midnight(reported_end_date)
                    )

                    available_at = conservative_available_at(item.report_date, raw_report_date)
                    if available_at.tzinfo is None or available_at.utcoffset() is None:
                        raise AssertionError("available_at must be timezone-aware")
                    available_at = available_at.astimezone(UTC)

                    observation: AnomalyObservation = {
                        "observation_id": observation_id,
                        "anomaly_id": anomaly_id,
                        "station_id": station_id,
                        "report_date": item.report_date,
                        "raw_row_report_date": raw_report_date,
                        "available_at": available_at,
                        "unit": unit,
                        "station_name": station_name,
                        "longitude": longitude,
                        "latitude": latitude,
                        "discipline": discipline,
                        "measurement": measurement,
                        "instrument_or_method": instrument_or_method,
                        "observation_period": observation_period,
                        "forecast_efficacy": normalize_text(values[10]),
                        "trend_time": trend_time,
                        "start_time": start_time,
                        "reported_end_time": reported_end_time,
                        "end_flag": end_flag,
                        "report_state": report_state,
                        "anomaly_type": anomaly_type,
                        "anomaly_description": anomaly_description,
                        "annual_anomaly": annual_anomaly,
                        "r_value": r_value,
                        "r0_value": r0_value,
                        "term_class": term_class,
                        "is_listed": True,
                        "right_censored": right_censored,
                        "identity_complete": identity_complete,
                        "reliability_flags": (),
                        "source_id": item.source.source_id,
                        "source_file": item.source.relative_path,
                        "source_sha256": item.source.sha256,
                        "source_sheet": sheet.name,
                        "source_row": source_row,
                        "source_serial_number": serial,
                        "source_period_year": item.report_year,
                        "source_period": item.report_period,
                    }
                    exact_content_key = tuple(_key_value(value) for value in values[1:])
                    natural_key = (
                        item.report_date.isoformat(),
                        station_name or "",
                        "" if longitude is None else canonical_decimal(longitude, "0.00001"),
                        "" if latitude is None else canonical_decimal(latitude, "0.00001"),
                        measurement or "",
                        instrument_or_method or "",
                        discipline,
                        start_date.isoformat(),
                    )
                    parsed_rows.append(
                        _ParsedRow(
                            observation=observation,
                            flags=flags,
                            exact_content_key=exact_content_key,
                            natural_key=natural_key,
                        )
                    )
                    if not identity_complete:
                        identity_flags = tuple(
                            sorted(
                                flag
                                for flag in flags
                                if flag.startswith("missing_")
                                or flag in {"identity_incomplete", "entity_unresolved"}
                            )
                        )
                        entity_audit.append(
                            _record_audit(
                                audit_type="identity_incomplete",
                                source_file=item.source.relative_path,
                                source_sheet=sheet.name,
                                report_date=item.report_date,
                                anomaly_ids=(anomaly_id,),
                                observation_ids=(observation_id,),
                                reliability_flags=identity_flags,
                            )
                        )

                    rows_by_discipline[discipline] += 1
                    period_rows[discipline] += 1
                    end_flags[end_flag] += 1
                    report_states[report_state] += 1

            period_row_count = sum(period_rows.values())
            report_periods.append(
                {
                    "report_id": stable_uuid(
                        "anomaly_report_period",
                        item.source.sha256,
                        item.report_year,
                        item.report_period,
                        item.report_date.isoformat(),
                    ),
                    "source_id": item.source.source_id,
                    "source_file": item.source.relative_path,
                    "source_sha256": item.source.sha256,
                    "report_year": item.report_year,
                    "report_period": item.report_period,
                    "report_date": item.report_date,
                    "available_at": conservative_available_at(item.report_date),
                    "row_count": period_row_count,
                    "row_report_date_mismatch_count": mismatch_count,
                    "row_report_date_before_count": before_count,
                    "row_report_date_after_count": after_count,
                    "deformation_row_count": period_rows["形变"],
                    "fluid_row_count": period_rows["流体"],
                    "electromagnetic_row_count": period_rows["电磁"],
                    "cross_fault_row_count": period_rows["跨断层"],
                }
            )
        finally:
            workbook.release_resources()

    by_report_sheet: dict[tuple[str, str], list[_ParsedRow]] = defaultdict(list)
    for row in parsed_rows:
        observation = row.observation
        by_report_sheet[(observation["source_file"], observation["source_sheet"])].append(row)

    exact_duplicate_groups = 0
    exact_duplicate_excess = 0
    natural_collision_groups = 0
    natural_collision_excess = 0
    for grouped_rows in by_report_sheet.values():
        exact_groups: dict[tuple[str, ...], list[_ParsedRow]] = defaultdict(list)
        natural_groups: dict[tuple[str, ...], list[_ParsedRow]] = defaultdict(list)
        for row in grouped_rows:
            exact_groups[row.exact_content_key].append(row)
            natural_groups[row.natural_key].append(row)
        for duplicate_rows in exact_groups.values():
            if len(duplicate_rows) < 2:
                continue
            exact_duplicate_groups += 1
            exact_duplicate_excess += len(duplicate_rows) - 1
            for row in duplicate_rows:
                row.flags.add("exact_duplicate_in_report")
            observation_ids = tuple(row.observation["observation_id"] for row in duplicate_rows)
            anomalies = tuple(sorted({row.observation["anomaly_id"] for row in duplicate_rows}))
            first = duplicate_rows[0].observation
            entity_audit.append(
                _record_audit(
                    audit_type="exact_duplicate_in_report",
                    source_file=first["source_file"],
                    source_sheet=first["source_sheet"],
                    report_date=first["report_date"],
                    anomaly_ids=anomalies,
                    observation_ids=observation_ids,
                    reliability_flags=("exact_duplicate_in_report",),
                )
            )
        for collision_rows in natural_groups.values():
            if len(collision_rows) < 2:
                continue
            natural_collision_groups += 1
            natural_collision_excess += len(collision_rows) - 1
            for row in collision_rows:
                row.flags.add("natural_key_collision")
            observation_ids = tuple(row.observation["observation_id"] for row in collision_rows)
            anomalies = tuple(sorted({row.observation["anomaly_id"] for row in collision_rows}))
            first = collision_rows[0].observation
            entity_audit.append(
                _record_audit(
                    audit_type="natural_key_collision",
                    source_file=first["source_file"],
                    source_sheet=first["source_sheet"],
                    report_date=first["report_date"],
                    anomaly_ids=anomalies,
                    observation_ids=observation_ids,
                    reliability_flags=("natural_key_collision",),
                )
            )

    complete_entities: dict[str, list[_ParsedRow]] = defaultdict(list)
    for row in parsed_rows:
        if row.observation["identity_complete"]:
            complete_entities[row.observation["anomaly_id"]].append(row)
    for anomaly_id, entity_rows in complete_entities.items():
        entity_rows.sort(
            key=lambda row: (
                row.observation["report_date"],
                row.observation["source_file"],
                _SHEET_INDEX[row.observation["source_sheet"]],
                row.observation["source_row"],
            )
        )
        previous_observation_end: datetime | None = None
        last_reported_end: datetime | None = None
        for index, row in enumerate(entity_rows):
            current_end = row.observation["reported_end_time"]
            if index > 0 and previous_observation_end is not None and current_end is None:
                row.flags.add("end_time_retracted")
                entity_audit.append(
                    _record_audit(
                        audit_type="end_time_retracted",
                        source_file=row.observation["source_file"],
                        source_sheet=row.observation["source_sheet"],
                        report_date=row.observation["report_date"],
                        anomaly_ids=(anomaly_id,),
                        observation_ids=(row.observation["observation_id"],),
                        reliability_flags=("end_time_retracted",),
                        previous_end=previous_observation_end,
                    )
                )
            if (
                current_end is not None
                and last_reported_end is not None
                and current_end != last_reported_end
            ):
                row.flags.add("end_time_revised")
                entity_audit.append(
                    _record_audit(
                        audit_type="end_time_revised",
                        source_file=row.observation["source_file"],
                        source_sheet=row.observation["source_sheet"],
                        report_date=row.observation["report_date"],
                        anomaly_ids=(anomaly_id,),
                        observation_ids=(row.observation["observation_id"],),
                        reliability_flags=("end_time_revised",),
                        previous_end=last_reported_end,
                        current_end=current_end,
                    )
                )
            if current_end is not None:
                last_reported_end = current_end
            previous_observation_end = current_end

    for row in parsed_rows:
        row.observation["reliability_flags"] = tuple(sorted(row.flags))

    parsed_rows.sort(
        key=lambda row: (
            row.observation["report_date"],
            _SHEET_INDEX[row.observation["source_sheet"]],
            row.observation["source_row"],
            row.observation["source_file"],
            row.observation["observation_id"],
        )
    )
    report_periods.sort(
        key=lambda period: (
            period["report_year"],
            period["report_period"],
            period["report_date"],
            period["source_file"],
        )
    )
    entity_audit.sort(
        key=lambda audit: (
            audit["audit_type"],
            audit["source_file"] or "",
            audit["source_sheet"] or "",
            audit["report_date"] or date.min,
            audit["audit_id"],
        )
    )

    observations = tuple(row.observation for row in parsed_rows)
    unique_raw_report_dates = {row["raw_row_report_date"] for row in observations}
    longitudes = [row["longitude"] for row in observations if row["longitude"] is not None]
    latitudes = [row["latitude"] for row in observations if row["latitude"] is not None]
    reported_end_times = [
        row["reported_end_time"] for row in observations if row["reported_end_time"] is not None
    ]
    quality: dict[str, object] = {
        "file_count": len(metadata),
        "worksheet_count": len(metadata) * len(_SHEET_ORDER),
        "observation_count": len(observations),
        "rows_by_discipline": dict(sorted(rows_by_discipline.items())),
        "blank_row_count": blank_row_count,
        "merged_cell_count": merged_cell_count,
        "unique_file_report_date_count": len({item.report_date for item in metadata}),
        "unique_raw_row_report_date_count": len(unique_raw_report_dates),
        "report_date_min": min(row["report_date"] for row in observations).isoformat(),
        "report_date_max": max(row["report_date"] for row in observations).isoformat(),
        "raw_row_report_date_min": min(unique_raw_report_dates).isoformat(),
        "raw_row_report_date_max": max(unique_raw_report_dates).isoformat(),
        "available_at_min_utc": min(row["available_at"] for row in observations).isoformat(),
        "available_at_max_utc": max(row["available_at"] for row in observations).isoformat(),
        "start_time_min": min(row["start_time"] for row in observations).isoformat(),
        "start_time_max": max(row["start_time"] for row in observations).isoformat(),
        "reported_end_time_min": (
            min(reported_end_times).isoformat() if reported_end_times else None
        ),
        "reported_end_time_max": (
            max(reported_end_times).isoformat() if reported_end_times else None
        ),
        "spatial_bbox_wgs84": (
            {
                "min_longitude": min(longitudes),
                "min_latitude": min(latitudes),
                "max_longitude": max(longitudes),
                "max_latitude": max(latitudes),
            }
            if longitudes and latitudes
            else None
        ),
        "row_report_date_mismatch_count": sum(
            count for offset, count in report_date_offsets.items() if offset != 0
        ),
        "row_report_date_offset_days": {
            str(offset): count for offset, count in sorted(report_date_offsets.items())
        },
        "missing_field_counts": dict(sorted(missing_fields.items())),
        "exact_duplicate_group_count": exact_duplicate_groups,
        "exact_duplicate_excess_row_count": exact_duplicate_excess,
        "natural_key_collision_group_count": natural_collision_groups,
        "natural_key_collision_excess_row_count": natural_collision_excess,
        "end_flag_counts": dict(sorted(end_flags.items())),
        "report_state_counts": dict(sorted(report_states.items())),
        "right_censored_count": sum(row["right_censored"] for row in observations),
        "reported_end_time_count": sum(
            row["reported_end_time"] is not None for row in observations
        ),
        "missing_report_periods": _missing_periods(report_periods),
    }
    return AnomalyParseResult(
        observations=observations,
        entity_audit=tuple(entity_audit),
        report_periods=tuple(report_periods),
        quality=quality,
    )


__all__ = [
    "AnomalyObservation",
    "AnomalyParseResult",
    "EntityAuditRecord",
    "ReportPeriodRecord",
    "parse_anomaly_files",
]
