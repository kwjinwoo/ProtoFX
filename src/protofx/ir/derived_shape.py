"""Internal authoritative derived-shape metadata helpers."""

from __future__ import annotations

from protofx.ir.dtype import DType
from protofx.ir.shape import Shape
from protofx.ir.tensor_type import TensorType
from protofx.ir.value import Value


def get_derived_tensor_type(value: Value) -> TensorType | None:
    """Return the internal derived tensor type, if present.

    Args:
        value: IR value to inspect.

    Returns:
        Derived tensor metadata or ``None`` when unset.
    """
    return value._derived_tensor_type


def set_derived_tensor_type(value: Value, tensor_type: TensorType | None) -> None:
    """Set or clear internal derived tensor metadata.

    Args:
        value: IR value to mutate.
        tensor_type: Derived metadata to store, or ``None`` to clear.
    """
    value._derived_tensor_type = tensor_type


def get_authoritative_tensor_type(value: Value) -> TensorType:
    """Return the authoritative tensor metadata for a value.

    Args:
        value: IR value to inspect.

    Returns:
        Derived tensor metadata when present, otherwise seed tensor metadata.
    """
    derived = get_derived_tensor_type(value)
    return value.tensor_type if derived is None else derived


def get_authoritative_shape(value: Value) -> Shape:
    """Return the authoritative shape metadata for a value.

    Args:
        value: IR value to inspect.

    Returns:
        Authoritative shape metadata.
    """
    return get_authoritative_tensor_type(value).shape


def get_authoritative_dtype(value: Value) -> DType | None:
    """Return the authoritative dtype metadata for a value.

    Args:
        value: IR value to inspect.

    Returns:
        Authoritative dtype metadata.
    """
    return get_authoritative_tensor_type(value).dtype


def set_derived_shape(value: Value, shape: Shape) -> None:
    """Set derived shape while preserving currently authoritative dtype.

    Args:
        value: IR value to mutate.
        shape: Derived shape metadata to store.
    """
    dtype = get_authoritative_tensor_type(value).dtype
    set_derived_tensor_type(value, TensorType(dtype=dtype, shape=shape))
