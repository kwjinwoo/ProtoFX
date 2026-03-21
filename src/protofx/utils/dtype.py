"""ONNX ↔ IR ↔ PyTorch data type mapping utilities."""

from protofx.ir.dtype import DType

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
