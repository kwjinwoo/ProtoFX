"""ONNX ↔ IR ↔ PyTorch data type mapping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ir.dtype import DType

if TYPE_CHECKING:
    import torch

# The set of ONNX TensorProto.DataType int values that DType covers.
_SUPPORTED_ONNX_DTYPES: frozenset[int] = frozenset(m.value for m in DType)


def onnx_dtype_to_ir(elem_type: int) -> DType | None:
    """Convert an ONNX ``TensorProto.DataType`` integer to an IR ``DType``.

    Args:
        elem_type: An integer from ``onnx.TensorProto.DataType``.

    Returns:
        The corresponding ``DType`` member, or ``None`` for unsupported or
        undefined element types (e.g. ``UNDEFINED``, ``INT4``, ``UINT4``).
    """
    if elem_type in _SUPPORTED_ONNX_DTYPES:
        return DType(elem_type)
    return None


def ir_dtype_to_torch(dtype: DType | None) -> torch.dtype | None:
    """Convert an IR ``DType`` to a ``torch.dtype``.

    ``torch`` is imported lazily to keep import time fast for modules that
    only need the IR layer.

    Args:
        dtype: An IR data type, or ``None``.

    Returns:
        The corresponding ``torch.dtype``, or ``None`` when *dtype* is
        ``None`` or has no PyTorch equivalent (e.g. ``DType.STRING``).
    """
    if dtype is None:
        return None

    import torch

    match dtype:
        case DType.FLOAT32:
            return torch.float32
        case DType.FLOAT64:
            return torch.float64
        case DType.FLOAT16:
            return torch.float16
        case DType.BFLOAT16:
            return torch.bfloat16
        case DType.INT8:
            return torch.int8
        case DType.INT16:
            return torch.int16
        case DType.INT32:
            return torch.int32
        case DType.INT64:
            return torch.int64
        case DType.UINT8:
            return torch.uint8
        case DType.UINT16:
            return torch.uint16
        case DType.UINT32:
            return torch.uint32
        case DType.UINT64:
            return torch.uint64
        case DType.BOOL:
            return torch.bool
        case DType.COMPLEX64:
            return torch.complex64
        case DType.COMPLEX128:
            return torch.complex128
        case DType.FLOAT8E4M3FN:
            return torch.float8_e4m3fn
        case DType.FLOAT8E5M2:
            return torch.float8_e5m2
        case _:
            return None
