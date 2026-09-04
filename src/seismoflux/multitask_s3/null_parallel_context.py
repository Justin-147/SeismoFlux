"""Local, read-only shared arrays for the authorized S3 process scheduler.

Only read context files produced by this runner in its own local run directory.
The pickle is an internal process transport, never a user-provided input format.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def write_parallel_context(path: Path, context: dict[str, Any]) -> None:
    """Save large numeric arrays once; spawned workers map the same files read-only."""
    path.parent.mkdir(parents=True, exist_ok=False)
    arrays: dict[int, tuple[np.ndarray, str]] = {}

    class ArrayPickler(pickle.Pickler):
        def persistent_id(self, value):
            if (
                isinstance(value, np.ndarray)
                and not value.dtype.hasobject
                and value.nbytes >= 65536
            ):
                key = id(value)
                if key not in arrays:
                    name = f"array_{len(arrays):05d}.npy"
                    np.save(path.parent / name, value, allow_pickle=False)
                    # Keep a strong reference: temporary arrays from object reducers
                    # must not be collected and have their id reused during pickling.
                    arrays[key] = (value, name)
                return ("readonly_numpy_v1", arrays[key][1])
            return None

    with path.open("xb") as handle:
        ArrayPickler(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(context)


def read_parallel_context(path: Path) -> dict[str, Any]:
    """Load an internally generated context; large arrays cannot be written."""
    arrays: dict[str, np.ndarray] = {}

    class ArrayUnpickler(pickle.Unpickler):
        def persistent_load(self, token):
            if (
                not isinstance(token, tuple)
                or len(token) != 2
                or token[0] != "readonly_numpy_v1"
                or not isinstance(token[1], str)
                or Path(token[1]).name != token[1]
            ):
                raise pickle.UnpicklingError("invalid internal shared-array token")
            name = token[1]
            if name not in arrays:
                arrays[name] = np.load(path.parent / name, mmap_mode="r", allow_pickle=False)
            return arrays[name]

    with path.open("rb") as handle:
        context = ArrayUnpickler(handle).load()
    if not isinstance(context, dict):
        raise ValueError("internal parallel context must be a dictionary")
    return context
