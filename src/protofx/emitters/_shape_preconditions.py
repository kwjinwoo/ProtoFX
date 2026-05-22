"""Shared shape-precondition helpers for emitter-side op handlers."""

from __future__ import annotations

from protofx.ir.derived_shape import get_authoritative_tensor_type
from protofx.ir.shape import Shape
from protofx.ir.value import Value


def authoritative_shape(value: Value) -> Shape:
    """Return authoritative shape metadata for a value.

    Args:
        value: IR value to inspect.

    Returns:
        Authoritative shape metadata.
    """
    return get_authoritative_tensor_type(value).shape


def require_authoritative_shape(value: Value, *, op_name: str, input_index: int) -> tuple[int | str | None, ...]:
    """Return authoritative shape or raise for unavailable metadata.

    Args:
        value: IR value to inspect.
        op_name: ONNX op name for diagnostics.
        input_index: Input index for diagnostics.

    Returns:
        Authoritative shape tuple.

    Raises:
        NotImplementedError: If authoritative shape metadata is unavailable.
    """
    shape = authoritative_shape(value)
    if shape is None:
        msg = f"{op_name}: input {input_index} has no authoritative shape metadata"
        raise NotImplementedError(msg)
    return shape
