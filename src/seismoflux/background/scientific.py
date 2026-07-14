"""Strict conversion of immutable scientific results to publication-safe JSON values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, date, datetime
from pathlib import PurePath
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel

ScientificJson: TypeAlias = (
    None | bool | int | float | str | list["ScientificJson"] | dict[str, "ScientificJson"]
)


def _datetime_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scientific datetimes must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def scientific_json(value: object, *, location: str = "$") -> ScientificJson:
    """Recursively snapshot supported result objects without lossy string fallbacks."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite scientific float at {location}")
        return number
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return _datetime_text(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in {"b", "i", "u", "f", "U", "S"}:
            raise TypeError(f"unsupported NumPy dtype at {location}: {value.dtype}")
        return scientific_json(value.tolist(), location=location)
    if isinstance(value, BaseModel):
        return scientific_json(value.model_dump(mode="python"), location=location)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: scientific_json(
                getattr(value, item.name),
                location=f"{location}.{item.name}",
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        mapped: dict[str, ScientificJson] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"scientific mapping keys must be strings at {location}")
            if key in mapped:
                raise ValueError(f"duplicate scientific mapping key at {location}: {key}")
            mapped[key] = scientific_json(item, location=f"{location}.{key}")
        return {key: mapped[key] for key in sorted(mapped)}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | memoryview):
        return [
            scientific_json(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, PurePath):
        raise TypeError(
            f"scientific paths require an explicit project-relative representation at {location}"
        )
    if isinstance(value, bytes | bytearray | memoryview):
        raise TypeError(f"binary scientific values require BundleBinary at {location}")
    raise TypeError(f"unsupported scientific result at {location}: {type(value).__name__}")


def scientific_mapping(value: object, *, location: str = "$") -> dict[str, ScientificJson]:
    """Convert and require a JSON mapping root for :class:`BundleDocument`."""

    converted = scientific_json(value, location=location)
    if not isinstance(converted, dict):
        raise TypeError("scientific publication root must be a mapping")
    return converted


__all__ = ["ScientificJson", "scientific_json", "scientific_mapping"]
