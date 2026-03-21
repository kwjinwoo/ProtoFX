"""Backend-neutral data type enumeration for ProtoFX IR.

Values are aligned with ONNX TensorProto.DataType for convenient mapping,
but this module has no dependency on the ``onnx`` package.
"""

import enum


class DType(enum.Enum):
    """Element data types supported by the ProtoFX IR.

    Integer values mirror ``onnx.TensorProto.DataType`` so that the importer
    can convert with a simple ``DType(elem_type)`` call.  The IR itself does
    not import or depend on the ``onnx`` package.
    """

    FLOAT32 = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    FLOAT64 = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E5M2 = 19
