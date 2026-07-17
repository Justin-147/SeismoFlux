from __future__ import annotations

import struct

import pyarrow as pa
import pytest

from seismoflux.anomaly_increment.grid_features import (
    SELECTED_TABLE_LOGICAL_IDENTITY_DOMAIN_R1,
    SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1,
    assert_selected_columns_logically_exact_r1,
    selected_table_identity_sha256,
    selected_table_logical_identity_sha256_r1,
)


def _table(array: pa.Array | pa.ChunkedArray, *, field: pa.Field | None = None) -> pa.Table:
    selected_field = field or pa.field("value", array.type)
    return pa.Table.from_arrays([array], schema=pa.schema([selected_field]))


def _fixed_width_array(
    data_type: pa.DataType,
    validity: bytes,
    payload: bytes,
    *,
    length: int,
) -> pa.Array:
    return pa.Array.from_buffers(
        data_type,
        length,
        [pa.py_buffer(validity), pa.py_buffer(payload)],
        null_count=-1,
    )


def _string_array(validity: bytes, offsets: tuple[int, ...], data: bytes) -> pa.Array:
    offset_bytes = struct.pack(f"<{len(offsets)}i", *offsets)
    return pa.Array.from_buffers(
        pa.string(),
        len(offsets) - 1,
        [pa.py_buffer(validity), pa.py_buffer(offset_bytes), pa.py_buffer(data)],
        null_count=-1,
    )


def test_selected_table_logical_identity_r1_covers_frozen_canonicalization_contract() -> None:
    assert SELECTED_TABLE_LOGICAL_IDENTITY_METHOD_R1 == (
        "arrow_ipc_selected_table_logical_identity_r1"
    )
    assert SELECTED_TABLE_LOGICAL_IDENTITY_DOMAIN_R1 == (
        b"seismoflux.selected-table-logical-identity.r1\x00"
    )

    base = pa.table(
        {"value": pa.array([1.0, None, -0.0], type=pa.float64())},
        metadata={b"document": b"accepted"},
    )
    changed_top_metadata = base.replace_schema_metadata({b"document": b"rebuilt"})
    assert selected_table_identity_sha256(base, ("value",)) != selected_table_identity_sha256(
        changed_top_metadata, ("value",)
    )
    assert selected_table_logical_identity_sha256_r1(
        base, ("value",)
    ) == selected_table_logical_identity_sha256_r1(changed_top_metadata, ("value",))
    assert selected_table_logical_identity_sha256_r1(base, ("value",)) == (
        "dc082e0ad507d663a08f05235e0b321c38e11a1eb63e2594718548b96dff7a31"
    )
    assert_selected_columns_logically_exact_r1(
        base,
        changed_top_metadata,
        columns=("value",),
    )

    sliced = pa.table({"value": pa.array([99.0, 1.0, None, -0.0, 88.0])}).slice(1, 3)
    chunked = _table(
        pa.chunked_array(
            [
                pa.array([1.0]),
                pa.array([None, -0.0], type=pa.float64()),
            ],
            type=pa.float64(),
        )
    )
    expected = selected_table_logical_identity_sha256_r1(base, ("value",))
    assert selected_table_logical_identity_sha256_r1(sliced, ("value",)) == expected
    assert selected_table_logical_identity_sha256_r1(chunked, ("value",)) == expected

    explicit_all_valid = _table(
        pa.Array.from_buffers(
            pa.int16(),
            3,
            [pa.py_buffer(b"\xff"), pa.py_buffer(struct.pack("<hhh", 1, 2, 3))],
            null_count=0,
        )
    )
    absent_validity = _table(pa.array([1, 2, 3], type=pa.int16()))
    assert selected_table_logical_identity_sha256_r1(
        explicit_all_valid, ("value",)
    ) == selected_table_logical_identity_sha256_r1(absent_validity, ("value",))


@pytest.mark.parametrize(
    ("data_type", "zero_payload", "hidden_payload"),
    [
        (pa.int8(), b"\x01\x00\x03", b"\x01\x7f\x03"),
        (pa.uint32(), struct.pack("<III", 1, 0, 3), struct.pack("<III", 1, 0xFFFFFFFF, 3)),
        (
            pa.float64(),
            bytes.fromhex("000000000000f03f00000000000000000000000000000840"),
            bytes.fromhex("000000000000f03f010000000000f87f0000000000000840"),
        ),
        (
            pa.timestamp("ns", tz="UTC"),
            struct.pack("<qqq", 1, 0, 3),
            struct.pack("<qqq", 1, -(2**63), 3),
        ),
    ],
    ids=("signed-int8", "unsigned-int32", "float64", "timestamp-ns-utc"),
)
def test_r1_identity_zeroes_only_null_fixed_width_payload(
    data_type: pa.DataType,
    zero_payload: bytes,
    hidden_payload: bytes,
) -> None:
    left = _table(_fixed_width_array(data_type, b"\x05", zero_payload, length=3))
    right = _table(_fixed_width_array(data_type, b"\xfd", hidden_payload, length=3))
    assert selected_table_logical_identity_sha256_r1(
        left, ("value",)
    ) == selected_table_logical_identity_sha256_r1(right, ("value",))
    assert_selected_columns_logically_exact_r1(left, right, columns=("value",))

    changed_valid = _table(_fixed_width_array(data_type, b"\x07", zero_payload, length=3))
    assert selected_table_logical_identity_sha256_r1(
        left, ("value",)
    ) != selected_table_logical_identity_sha256_r1(changed_valid, ("value",))

    changed_valid_payload = bytearray(zero_payload)
    changed_valid_payload[0] ^= 1
    changed_payload = _table(
        _fixed_width_array(data_type, b"\x05", bytes(changed_valid_payload), length=3)
    )
    assert selected_table_logical_identity_sha256_r1(
        left, ("value",)
    ) != selected_table_logical_identity_sha256_r1(changed_payload, ("value",))


def test_r1_identity_canonicalizes_boolean_null_payload_and_padding() -> None:
    left = _table(
        pa.Array.from_buffers(
            pa.bool_(),
            3,
            [pa.py_buffer(b"\x05"), pa.py_buffer(b"\x05")],
            null_count=1,
        )
    )
    right = _table(
        pa.Array.from_buffers(
            pa.bool_(),
            3,
            [pa.py_buffer(b"\xfd"), pa.py_buffer(b"\xff")],
            null_count=1,
        )
    )
    assert selected_table_logical_identity_sha256_r1(
        left, ("value",)
    ) == selected_table_logical_identity_sha256_r1(right, ("value",))

    changed_valid_value = _table(
        pa.Array.from_buffers(
            pa.bool_(),
            3,
            [pa.py_buffer(b"\x05"), pa.py_buffer(b"\x04")],
            null_count=1,
        )
    )
    assert selected_table_logical_identity_sha256_r1(
        left, ("value",)
    ) != selected_table_logical_identity_sha256_r1(changed_valid_value, ("value",))


def test_r1_identity_canonicalizes_utf8_null_payload_but_preserves_valid_bytes() -> None:
    compact = _table(_string_array(b"\x05", (0, 1, 1, 2), b"AB"))
    hidden = _table(_string_array(b"\xfd", (0, 1, 4, 5), b"AXYZB"))
    assert selected_table_logical_identity_sha256_r1(
        compact, ("value",)
    ) == selected_table_logical_identity_sha256_r1(hidden, ("value",))

    changed = _table(_string_array(b"\x05", (0, 1, 1, 2), b"AC"))
    assert selected_table_logical_identity_sha256_r1(
        compact, ("value",)
    ) != selected_table_logical_identity_sha256_r1(changed, ("value",))

    sliced = _table(pa.array(["ignore", "A", None, "B", "ignore"]).slice(1, 3))
    chunked = _table(
        pa.chunked_array(
            [pa.array(["A"]), pa.array([None, "B"], type=pa.string())],
            type=pa.string(),
        )
    )
    expected = selected_table_logical_identity_sha256_r1(compact, ("value",))
    assert selected_table_logical_identity_sha256_r1(sliced, ("value",)) == expected
    assert selected_table_logical_identity_sha256_r1(chunked, ("value",)) == expected


def test_r1_identity_preserves_valid_float_payload_bits() -> None:
    positive_zero = _table(pa.Array.from_buffers(pa.float64(), 1, [None, pa.py_buffer(bytes(8))]))
    negative_zero = _table(
        pa.Array.from_buffers(
            pa.float64(), 1, [None, pa.py_buffer(bytes.fromhex("0000000000000080"))]
        )
    )
    nan_0 = _table(
        pa.Array.from_buffers(
            pa.float64(), 1, [None, pa.py_buffer(bytes.fromhex("000000000000f87f"))]
        )
    )
    nan_1 = _table(
        pa.Array.from_buffers(
            pa.float64(), 1, [None, pa.py_buffer(bytes.fromhex("010000000000f87f"))]
        )
    )
    identities = {
        selected_table_logical_identity_sha256_r1(table, ("value",))
        for table in (positive_zero, negative_zero, nan_0, nan_1)
    }
    assert len(identities) == 4


def test_r1_identity_excludes_top_metadata_but_preserves_fields_exactly() -> None:
    ordered_a = pa.field(
        "value",
        pa.int32(),
        nullable=True,
        metadata={b"zeta": b"last", b"alpha": b"first"},
    )
    ordered_b = pa.field(
        "value",
        pa.int32(),
        nullable=True,
        metadata={b"alpha": b"first", b"zeta": b"last"},
    )
    array = pa.array([1, 2], type=pa.int32())
    left = _table(array, field=ordered_a).replace_schema_metadata({b"left": b"1"})
    reordered_metadata = _table(array, field=ordered_b).replace_schema_metadata({b"right": b"2"})
    assert selected_table_logical_identity_sha256_r1(
        left, ("value",)
    ) == selected_table_logical_identity_sha256_r1(reordered_metadata, ("value",))

    changed_metadata = _table(
        array,
        field=pa.field("value", pa.int32(), metadata={b"alpha": b"changed"}),
    )
    nonnullable = _table(array, field=pa.field("value", pa.int32(), nullable=False))
    changed_type = _table(pa.array([1, 2], type=pa.int64()))
    for changed in (changed_metadata, nonnullable, changed_type):
        assert selected_table_logical_identity_sha256_r1(
            left, ("value",)
        ) != selected_table_logical_identity_sha256_r1(changed, ("value",))
        with pytest.raises(ValueError, match="fields|validity"):
            assert_selected_columns_logically_exact_r1(left, changed, columns=("value",))

    ordered_columns = pa.table({"left": [1], "right": [2]})
    assert selected_table_logical_identity_sha256_r1(
        ordered_columns, ("left", "right")
    ) != selected_table_logical_identity_sha256_r1(ordered_columns, ("right", "left"))


class _TestExtensionType(pa.ExtensionType):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__(pa.int32(), "seismoflux.test.logical_identity")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls,
        storage_type: pa.DataType,
        serialized: bytes,
    ) -> _TestExtensionType:
        del storage_type, serialized
        return cls()


@pytest.mark.parametrize(
    "array",
    [
        pa.array([[1], [2]], type=pa.list_(pa.int32())),
        pa.array([b"a", b"b"], type=pa.binary()),
        pa.array([1, 2], type=pa.date32()),
        pa.DictionaryArray.from_arrays(pa.array([0, 1]), pa.array(["a", "b"])),
        pa.ExtensionArray.from_storage(_TestExtensionType(), pa.array([1, 2], type=pa.int32())),
    ],
    ids=("nested-list", "binary", "date32", "dictionary", "extension"),
)
def test_r1_identity_fails_closed_for_unsupported_arrow_types(
    array: pa.Array,
) -> None:
    with pytest.raises(TypeError, match="does not support Arrow type"):
        selected_table_logical_identity_sha256_r1(_table(array), ("value",))
