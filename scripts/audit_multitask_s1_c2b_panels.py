"""Count the three fixed C2B training panels without fitting or scoring models."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from seismoflux.multitask_s0 import verify_authoritative_catalog_identity  # noqa: E402

YEARS = (1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020)
PANELS = (
    ("D0_CANONICAL_M4_1970", 1970, 4.0, None),
    ("D1_M3SOURCE_M4_1980", 1980, 4.0, "earthquake_catalog_m3_plus"),
    ("D2_M5SOURCE_M5_1950", 1950, 5.0, "earthquake_catalog_m5_plus"),
)
SOURCE_SHA256 = "7c48cd5f4db5e85c6c10c7a3fdfc4d84ac3e5de3dde1e59f985f36bb47770556"
STAGE1_PATH = Path("processed/stage1/debc98054172a4a1")
SEMANTICS_PATH = Path("data/manifests/catalog_magnitude_semantics_2026-09-02.json")
OUTPUT_PATH = Path("outputs/multitask_s1/s1c2b_protocol_v1/panel_ledger.json")
EVENT_COLUMNS = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "magnitude",
    "inside_study_area",
    "catalog_sources",
)
SOURCE_COLUMNS = ("source_record_id", "source_id", "origin_time_utc", "available_at")


def local_year_start(year: int) -> pd.Timestamp:
    """Use the existing +08:00 catalog convention, without integer epoch casts."""

    return pd.Timestamp(f"{year}-01-01T00:00:00+08:00").tz_convert("UTC")


def training_cutoff(year: int) -> pd.Timestamp:
    return local_year_start(year) - pd.Timedelta(hours=24)


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _visible(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame["origin_time_utc"] <= cutoff) & (frame["available_at"] <= cutoff)]


def build_panel_rows(events: pd.DataFrame, sources: pd.DataFrame) -> list[dict[str, object]]:
    """Keep canonical values, use any causally visible source member, never the anchor."""

    if events["event_id"].duplicated().any() or sources["source_record_id"].duplicated().any():
        raise ValueError("duplicate canonical event or source-record identity")
    for frame in (events, sources):
        for column in ("origin_time_utc", "available_at"):
            if not isinstance(frame[column].dtype, pd.DatetimeTZDtype):
                raise ValueError("training timestamps must be timezone aware")
            if frame[column].isna().any():
                raise ValueError("training timestamps must not be missing")
        if (frame["available_at"] < frame["origin_time_utc"]).any():
            raise ValueError("availability cannot precede origin time")
    rows = []
    for year in YEARS:
        cutoff = training_cutoff(year)
        visible_events = _visible(events, cutoff)
        visible_sources = _visible(sources, cutoff)
        for panel_id, start_year, magnitude_minimum, source_id in PANELS:
            start = local_year_start(start_year)
            keep = (
                visible_events["inside_study_area"]
                & (visible_events["origin_time_utc"] >= start)
                & (visible_events["magnitude"] >= magnitude_minimum)
            )
            if source_id is not None:
                source_members = set(
                    visible_sources.loc[
                        visible_sources["source_id"] == source_id, "source_record_id"
                    ]
                )
                keep &= visible_events["catalog_sources"].map(
                    lambda members, ids=source_members: bool(ids.intersection(members))
                )
            selected = visible_events.loc[keep]
            count = len(selected)
            rows.append(
                {
                    "cutoff_year_local": year,
                    "issue_time_utc": _iso(local_year_start(year)),
                    "training_cutoff_inclusive_utc": _iso(cutoff),
                    "panel_id": panel_id,
                    "training_start_inclusive_utc": _iso(start),
                    "magnitude_minimum_inclusive": magnitude_minimum,
                    "required_any_source_id": source_id,
                    "training_event_count": count,
                    "observed_origin_min_utc": _iso(selected["origin_time_utc"].min())
                    if count
                    else None,
                    "observed_origin_max_utc": _iso(selected["origin_time_utc"].max())
                    if count
                    else None,
                    "observed_available_max_utc": _iso(selected["available_at"].max())
                    if count
                    else None,
                    "effective_magnitude_type": "Ms",
                }
            )
    return rows


def run_audit(project_root: Path, data_root: Path) -> Path:
    """Read only training fields before 2020, and save an aggregate-only ledger."""

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    event_path = data_root / STAGE1_PATH / "earthquake_event.parquet"
    source_path = data_root / STAGE1_PATH / "earthquake_source_record.parquet"
    event_identity = verify_authoritative_catalog_identity(event_path)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_count = pq.ParquetFile(source_path).metadata.num_rows
    if source_sha != SOURCE_SHA256 or source_count != 43_785:
        raise ValueError("authoritative source-record identity changed")
    semantics_path = project_root / SEMANTICS_PATH
    semantics_bytes = semantics_path.read_bytes()
    semantics = json.loads(semantics_bytes)
    expected_sources = {spec[3] for spec in PANELS if spec[3] is not None}
    if (
        {item["source_id"] for item in semantics["scope"]} != expected_sources
        or any(item["effective_magnitude_type"] != "Ms" for item in semantics["scope"])
        or semantics["numeric_magnitude_conversion"] != "none"
        or semantics["applies_to_external_catalogs_including_comcat"] is not False
    ):
        raise ValueError("the source-specific user Ms clarification changed")
    maximum_cutoff = training_cutoff(max(YEARS)).to_pydatetime()
    filters = [("origin_time_utc", "<=", maximum_cutoff), ("available_at", "<=", maximum_cutoff)]
    events = pq.read_table(
        event_path, columns=list(EVENT_COLUMNS), filters=filters, use_threads=False
    ).to_pandas()
    sources = pq.read_table(
        source_path, columns=list(SOURCE_COLUMNS), filters=filters, use_threads=False
    ).to_pandas()
    ledger = {
        "schema_version": 1,
        "ledger_id": "s1c2b_score_blind_training_panels_v1",
        "scientific_role": "training_coverage_inventory_not_prediction_or_target_counts",
        "inputs": {
            "canonical_event": {
                "path_relative_to_data_root": (STAGE1_PATH / event_path.name).as_posix(),
                "file_sha256": event_identity["file_sha256"],
                "full_file_rows_from_metadata": event_identity["row_count"],
                "columns_read": list(EVENT_COLUMNS),
            },
            "source_record": {
                "path_relative_to_data_root": (STAGE1_PATH / source_path.name).as_posix(),
                "file_sha256": source_sha,
                "full_file_rows_from_metadata": source_count,
                "columns_read": list(SOURCE_COLUMNS),
            },
            "maximum_row_read_cutoff_inclusive_utc": _iso(training_cutoff(max(YEARS))),
            "magnitude_semantics": {
                "path_relative_to_project_root": SEMANTICS_PATH.as_posix(),
                "file_sha256": hashlib.sha256(semantics_bytes).hexdigest(),
                "basis": "explicit_user_clarification_two_local_sources_only",
                "stored_type_nulls_preserved": True,
                "numeric_conversion": "none",
                "effective_type": "Ms",
                "applies_to_external_catalogs": False,
            },
        },
        "rules": {
            "source_membership": "any_visible_catalog_sources_record_not_canonical_anchor",
            "event_values": "unchanged_canonical_magnitude_origin_availability_and_domain_flag",
            "visibility": "both_origin_and_available_at<=local_Jan1_minus_24h_for_event_and_member",
            "study_domain": "canonical_inside_study_area_true_national_domain",
            "start_and_magnitude_boundaries": "inclusive",
            "time_comparison": "timezone_aware_datetime_no_implicit_ns_us_integer_cast",
            "national_support_percentage_gate": "none",
        },
        "rows": build_panel_rows(events, sources),
        "safety": {
            "raw_input_rewritten": False,
            "model_fit": False,
            "predictions_created": False,
            "scores_read": False,
            "holdout_or_audit_target_rows_read": False,
            "locked_test_run": False,
            "numerical_threads": 1,
            "pyarrow_cpu_threads": 1,
            "pyarrow_io_threads": 1,
        },
        "interpretation_zh": (
            "这是训练资料水位，不是独立目标样本数或模型成绩。D1同时改变来源和年代，"
            "只能称面板整体差异；D2同时改变来源、起始年代和震级门槛，不能单独归因。"
            "Ms说明不代表各年代各地完整度一致，也不适用于ComCat。"
        ),
    }
    output = project_root / OUTPUT_PATH
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != ledger:
            raise FileExistsError("existing panel ledger differs; preserve it for explicit review")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(ledger, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    print(run_audit(args.project_root.resolve(), args.data_root.resolve()))


if __name__ == "__main__":
    main()
