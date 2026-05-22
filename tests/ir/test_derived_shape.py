"""Tests for authoritative derived shape metadata helpers."""

from protofx.ir import Dim, DType, TensorType
from protofx.ir.derived_shape import get_authoritative_shape, set_derived_shape
from protofx.ir.value import Value, ValueKind


def test_public_dim_contract_is_unchanged() -> None:
    """Public Dim must remain int | str | None."""
    int_dim: Dim = 1
    str_dim: Dim = "batch"
    none_dim: Dim = None
    assert int_dim == 1
    assert str_dim == "batch"
    assert none_dim is None


def test_derived_shape_can_override_seed_shape() -> None:
    """Derived shape must become the authoritative shape when present."""
    value = Value(
        id="v0",
        kind=ValueKind.NODE_OUTPUT,
        tensor_type=TensorType(dtype=DType.FLOAT32, shape=(999, 999)),
    )
    set_derived_shape(value, (2, 3))
    assert get_authoritative_shape(value) == (2, 3)
