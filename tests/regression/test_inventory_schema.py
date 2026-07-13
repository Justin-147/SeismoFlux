from __future__ import annotations

import csv
import re
from pathlib import Path

from seismoflux.inventory import INVENTORY_COLUMNS


def test_inventory_column_order_is_frozen() -> None:
    assert INVENTORY_COLUMNS == (
        "source_id",
        "source_category",
        "relative_path",
        "file_extension",
        "size_bytes",
        "modified_at_utc",
        "sha256",
        "license_status",
    )


def test_committed_source_inventory_is_complete_and_contains_no_legacy_png() -> None:
    with Path("data/manifests/source_inventory.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 216
    assert sum(int(row["size_bytes"]) for row in rows) == 43_610_994
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in rows)
    assert all(not Path(row["relative_path"]).is_absolute() for row in rows)
    assert all(row["file_extension"] != ".png" for row in rows)
    assert all(row["license_status"] == "unknown_no_redistribution" for row in rows)
