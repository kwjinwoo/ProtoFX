"""Tests for protofx.utils.dtype mapping functions."""

import onnx

from protofx.ir import DType
from protofx.utils.dtype import onnx_dtype_to_ir


class TestOnnxDtypeToIr:
    """Verify ONNX TensorProto.DataType → ir.DType conversion."""

    def test_float32(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.FLOAT) is DType.FLOAT32

    def test_uint8(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.UINT8) is DType.UINT8

    def test_int8(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.INT8) is DType.INT8

    def test_uint16(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.UINT16) is DType.UINT16

    def test_int16(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.INT16) is DType.INT16

    def test_int32(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.INT32) is DType.INT32

    def test_int64(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.INT64) is DType.INT64

    def test_string(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.STRING) is DType.STRING

    def test_bool(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.BOOL) is DType.BOOL

    def test_float16(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.FLOAT16) is DType.FLOAT16

    def test_float64(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.DOUBLE) is DType.FLOAT64

    def test_uint32(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.UINT32) is DType.UINT32

    def test_uint64(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.UINT64) is DType.UINT64

    def test_complex64(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.COMPLEX64) is DType.COMPLEX64

    def test_complex128(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.COMPLEX128) is DType.COMPLEX128

    def test_bfloat16(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.BFLOAT16) is DType.BFLOAT16

    def test_float8e4m3fn(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.FLOAT8E4M3FN) is DType.FLOAT8E4M3FN

    def test_float8e5m2(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.FLOAT8E5M2) is DType.FLOAT8E5M2

    def test_undefined_returns_none(self) -> None:
        assert onnx_dtype_to_ir(onnx.TensorProto.UNDEFINED) is None

    def test_unknown_value_returns_none(self) -> None:
        assert onnx_dtype_to_ir(9999) is None
