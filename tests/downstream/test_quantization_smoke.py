"""PT2E quantization smoke tests for representative synthetic ProtoFX-emitted graphs.

Verifies that small emitted ``GraphModule`` objects survive the torchao PT2E
quantization pipeline (``torch.export`` → ``prepare_pt2e`` → calibration →
``convert_pt2e`` → execute) without exceptions and produce outputs with the
expected shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from tests.downstream.conftest import assert_quantize_survives_pt2e

pytestmark = pytest.mark.downstream_validation


# ---------------------------------------------------------------------------
# ONNX model builders for representative op coverage
# ---------------------------------------------------------------------------


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


def _make_matmul_model() -> helper.ModelProto:
    """Build ONNX model: (A, B) → MatMul → Y."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "matmul_graph", [A, B], [Y])
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQuantizeSmokeConv:
    """PT2E quantization survival for a Conv graph."""

    def test_quantize_survives(self) -> None:
        """Conv graph must survive the PT2E quantization pipeline."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, 1, 5, 5)).astype(np.float32)
        w = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
        assert_quantize_survives_pt2e(_make_conv_model(), {"X": x, "W": w})


class TestQuantizeSmokeMatMul:
    """PT2E quantization survival for a MatMul graph."""

    def test_quantize_survives(self) -> None:
        """MatMul graph must survive the PT2E quantization pipeline."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((2, 3)).astype(np.float32)
        b = rng.standard_normal((3, 4)).astype(np.float32)
        assert_quantize_survives_pt2e(_make_matmul_model(), {"A": a, "B": b})


class TestQuantizeSmokeAddRelu:
    """PT2E quantization survival for an Add → Relu graph."""

    def test_quantize_survives(self) -> None:
        """Add+Relu graph must survive the PT2E quantization pipeline."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((2, 4)).astype(np.float32)
        b = rng.standard_normal((2, 4)).astype(np.float32)
        assert_quantize_survives_pt2e(_make_add_relu_model(), {"A": a, "B": b})
