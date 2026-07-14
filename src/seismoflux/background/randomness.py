"""Deterministic SHA-256 namespaced random-number streams for stage 2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedContext:
    """The complete, worker-count-independent identity of one RNG stream."""

    root_seed: int
    protocol_version: str
    namespace: str
    model_id: str
    issue_id: str | None
    replicate_index: int

    def fields(self) -> tuple[str, ...]:
        """Return fields in the exact preregistered order."""

        if (
            not isinstance(self.root_seed, int)
            or isinstance(self.root_seed, bool)
            or self.root_seed < 0
        ):
            raise ValueError("root_seed must be a non-negative integer")
        if not self.protocol_version:
            raise ValueError("protocol_version must not be empty")
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.issue_id is not None and not self.issue_id:
            raise ValueError("issue_id must be non-empty or None")
        if (
            not isinstance(self.replicate_index, int)
            or isinstance(self.replicate_index, bool)
            or not 0 <= self.replicate_index <= 99_999_999
        ):
            raise ValueError("replicate_index must fit the frozen eight-decimal-digit field")
        fields = (
            "seismoflux",
            str(self.root_seed),
            self.protocol_version,
            self.namespace,
            self.model_id,
            self.issue_id if self.issue_id is not None else "-",
            f"{self.replicate_index:08d}",
        )
        if any("\x00" in field for field in fields):
            raise ValueError("seed context fields must not contain the NUL separator")
        return fields

    def digest(self) -> bytes:
        """Return the SHA-256 digest of NUL-separated UTF-8 context fields."""

        payload = b"\x00".join(field.encode("utf-8") for field in self.fields())
        return hashlib.sha256(payload).digest()

    def entropy(self) -> int:
        """Return the first 16 digest bytes as an unsigned big-endian integer."""

        return int.from_bytes(self.digest()[:16], "big")

    def generator(self) -> np.random.Generator:
        """Construct a fresh PCG64 generator without touching global RNG state."""

        return np.random.Generator(np.random.PCG64(self.entropy()))
