from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from seismoflux.background.scientific import scientific_json, scientific_mapping


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    value: float


@dataclass(frozen=True, slots=True)
class _Result:
    identifier: str
    values: np.ndarray
    cutoff: datetime
    model: _FrozenModel


def test_dataclass_numpy_datetime_and_pydantic_are_snapshotted() -> None:
    source = _Result(
        identifier="result-1",
        values=np.asarray([1.0, 2.5], dtype=np.float64),
        cutoff=datetime(2025, 1, 1, tzinfo=UTC),
        model=_FrozenModel(name="etas", value=0.25),
    )

    converted = scientific_mapping(source)

    assert converted == {
        "identifier": "result-1",
        "values": [1.0, 2.5],
        "cutoff": "2025-01-01T00:00:00.000000Z",
        "model": {"name": "etas", "value": 0.25},
    }
    source.values[0] = 99.0
    assert converted["values"] == [1.0, 2.5]


def test_mapping_keys_are_sorted_and_numpy_scalars_become_python_scalars() -> None:
    converted = scientific_mapping(
        {
            "z": np.int64(3),
            "a": np.bool_(True),
            "m": np.float64(1.5),
        }
    )

    assert tuple(converted) == ("a", "m", "z")
    assert converted == {"a": True, "m": 1.5, "z": 3}


@pytest.mark.parametrize(
    "value, message",
    (
        (float("nan"), "non-finite"),
        (datetime(2025, 1, 1), "timezone-aware"),
        (Path("result.json"), "project-relative"),
        (b"payload", "BundleBinary"),
        ({1: "invalid"}, "keys must be strings"),
    ),
)
def test_unsupported_or_ambiguous_values_are_rejected(value: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        scientific_json(value)


def test_publication_root_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="root must be a mapping"):
        scientific_mapping([1, 2, 3])
