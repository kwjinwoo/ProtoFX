"""torch.export round-trip smoke tests for representative synthetic ProtoFX-emitted graphs.

Verifies that small emitted ``GraphModule`` objects survive the
``torch.export.export`` → ``.module()`` round-trip and produce numerically
close outputs compared to eager execution.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from tests.downstream.conftest import assert_export_roundtrip

pytestmark = pytest.mark.downstream_validation


# ---------------------------------------------------------------------------
# ONNX model builders for representative op coverage
# ---------------------------------------------------------------------------


def _make_relu_model() -> helper.ModelProto:
    """Build a minimal ONNX model: X → Relu → Y."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "relu_graph", [X], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_add_relu_model() -> helper.ModelProto:
    """Build ONNX model: (A, B) → Add → Relu → Y."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    sum_vi = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 4])
    add_node = helper.make_node("Add", ["A", "B"], ["sum"])
    relu_node = helper.make_node("Relu", ["sum"], ["Y"])
    graph = helper.make_graph([add_node, relu_node], "add_relu_graph", [A, B], [Y], value_info=[sum_vi])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_matmul_model() -> helper.ModelProto:
    """Build ONNX model: (A, B) → MatMul → Y."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "matmul_graph", [A, B], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_conv_model() -> helper.ModelProto:
    """Build ONNX model: (X, W) -> Conv -> Y (3x3 kernel, stride 1, no padding)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3])
    node = helper.make_node(
        "Conv",
        ["X", "W"],
        ["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([node], "conv_graph", [X, W], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_layernorm_model() -> helper.ModelProto:
    """Build ONNX model: (X, scale, bias) → LayerNormalization → Y."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("LayerNormalization", ["X", "scale", "bias"], ["Y"], axis=-1)
    graph = helper.make_graph([node], "layernorm_graph", [X, scale, bias], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_multi_op_model() -> helper.ModelProto:
    """Build a multi-op ONNX model: X → Relu → Sigmoid → Y."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    mid_vi = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [2, 4])
    relu_node = helper.make_node("Relu", ["X"], ["mid"])
    sigmoid_node = helper.make_node("Sigmoid", ["mid"], ["Y"])
    graph = helper.make_graph([relu_node, sigmoid_node], "multi_op_graph", [X], [Y], value_info=[mid_vi])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportSmokeRelu:
    """torch.export round-trip parity for a minimal Relu graph."""

    def test_export_roundtrip(self) -> None:
        """Exported Relu graph must match eager output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        assert_export_roundtrip(_make_relu_model(), {"X": x})


class TestExportSmokeAddRelu:
    """torch.export round-trip parity for Add → Relu graph."""

    def test_export_roundtrip(self) -> None:
        """Exported Add+Relu graph must match eager output."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((2, 4)).astype(np.float32)
        b = rng.standard_normal((2, 4)).astype(np.float32)
        assert_export_roundtrip(_make_add_relu_model(), {"A": a, "B": b})


class TestExportSmokeMatMul:
    """torch.export round-trip parity for MatMul graph."""

    def test_export_roundtrip(self) -> None:
        """Exported MatMul graph must match eager output."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((2, 3)).astype(np.float32)
        b = rng.standard_normal((3, 4)).astype(np.float32)
        assert_export_roundtrip(_make_matmul_model(), {"A": a, "B": b})


class TestExportSmokeConv:
    """torch.export round-trip parity for Conv graph."""

    def test_export_roundtrip(self) -> None:
        """Exported Conv graph must match eager output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, 1, 5, 5)).astype(np.float32)
        w = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
        assert_export_roundtrip(_make_conv_model(), {"X": x, "W": w})


class TestExportSmokeLayerNorm:
    """torch.export round-trip parity for LayerNormalization graph."""

    def test_export_roundtrip(self) -> None:
        """Exported LayerNorm graph must match eager output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        scale = np.ones(4, dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        assert_export_roundtrip(
            _make_layernorm_model(),
            {"X": x, "scale": scale, "bias": bias},
        )


class TestExportSmokeMultiOp:
    """torch.export round-trip parity for a multi-op (Relu → Sigmoid) graph."""

    def test_export_roundtrip(self) -> None:
        """Exported multi-op graph must match eager output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        assert_export_roundtrip(_make_multi_op_model(), {"X": x})
