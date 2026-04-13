"""Custom FX pass smoke tests for representative synthetic ProtoFX-emitted graphs.

Verifies that small emitted ``GraphModule`` objects survive standard library
FX passes (``ShapeProp``) and a custom node-replacement pass, producing
outputs with unchanged shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from onnx import TensorProto, helper

from tests.downstream.conftest import assert_fx_pass_survives

if TYPE_CHECKING:
    from onnx import ModelProto

pytestmark = pytest.mark.downstream_validation


# ---------------------------------------------------------------------------
# ONNX model builders
# ---------------------------------------------------------------------------


def _make_relu_model() -> ModelProto:
    """Build a minimal ONNX model: X → Relu → Y."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "relu_graph", [X], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_add_relu_model() -> ModelProto:
    """Build ONNX model: (A, B) → Add → Relu → Y."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    sum_vi = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 4])
    add_node = helper.make_node("Add", ["A", "B"], ["sum"])
    relu_node = helper.make_node("Relu", ["sum"], ["Y"])
    graph = helper.make_graph([add_node, relu_node], "add_relu_graph", [A, B], [Y], value_info=[sum_vi])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_conv_model() -> ModelProto:
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


def _make_matmul_model() -> ModelProto:
    """Build ONNX model: (A, B) → MatMul → Y."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "matmul_graph", [A, B], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


# ---------------------------------------------------------------------------
# FX pass functions
# ---------------------------------------------------------------------------


def _shape_prop_pass(gm: torch.fx.GraphModule, sample_inputs: list[torch.Tensor]) -> torch.fx.GraphModule:
    """Apply ``ShapeProp`` to annotate tensor metadata on graph nodes.

    Args:
        gm: The ``GraphModule`` to transform.
        sample_inputs: Representative inputs for shape propagation.

    Returns:
        The same ``GraphModule`` with shape metadata populated.
    """
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(*sample_inputs)
    return gm


def _relu_to_leaky_relu_pass(gm: torch.fx.GraphModule, sample_inputs: list[torch.Tensor]) -> torch.fx.GraphModule:
    """Replace ``torch.relu`` calls with ``torch.nn.functional.leaky_relu`` in the FX graph.

    Args:
        gm: The ``GraphModule`` to transform.
        sample_inputs: Not used by this pass but kept for uniform signature.

    Returns:
        The transformed ``GraphModule`` with Relu nodes replaced by LeakyRelu.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.relu:
            node.target = torch.nn.functional.leaky_relu
    gm.graph.lint()
    gm.recompile()
    return gm


# ---------------------------------------------------------------------------
# Tests — ShapeProp (stdlib FX pass)
# ---------------------------------------------------------------------------


class TestFxPassShapePropRelu:
    """ShapeProp pass on a minimal Relu graph."""

    def test_survives(self) -> None:
        """ShapeProp must complete and forward pass must preserve output shapes."""
        rng = np.random.default_rng(42)
        inputs = {"X": rng.standard_normal((2, 4)).astype(np.float32)}
        assert_fx_pass_survives(_make_relu_model(), inputs, _shape_prop_pass)


class TestFxPassShapePropAddRelu:
    """ShapeProp pass on an Add → Relu graph."""

    def test_survives(self) -> None:
        """ShapeProp must complete and forward pass must preserve output shapes."""
        rng = np.random.default_rng(42)
        inputs = {
            "A": rng.standard_normal((2, 4)).astype(np.float32),
            "B": rng.standard_normal((2, 4)).astype(np.float32),
        }
        assert_fx_pass_survives(_make_add_relu_model(), inputs, _shape_prop_pass)


class TestFxPassShapePropConv:
    """ShapeProp pass on a Conv graph."""

    def test_survives(self) -> None:
        """ShapeProp must complete and forward pass must preserve output shapes."""
        rng = np.random.default_rng(42)
        inputs = {
            "X": rng.standard_normal((1, 1, 5, 5)).astype(np.float32),
            "W": rng.standard_normal((1, 1, 3, 3)).astype(np.float32),
        }
        assert_fx_pass_survives(_make_conv_model(), inputs, _shape_prop_pass)


class TestFxPassShapePropMatMul:
    """ShapeProp pass on a MatMul graph."""

    def test_survives(self) -> None:
        """ShapeProp must complete and forward pass must preserve output shapes."""
        rng = np.random.default_rng(42)
        inputs = {
            "A": rng.standard_normal((2, 3)).astype(np.float32),
            "B": rng.standard_normal((3, 4)).astype(np.float32),
        }
        assert_fx_pass_survives(_make_matmul_model(), inputs, _shape_prop_pass)


# ---------------------------------------------------------------------------
# Tests — Custom node-replacement pass (Relu → LeakyRelu)
# ---------------------------------------------------------------------------


class TestFxPassReluToLeakyReluSingle:
    """Custom Relu→LeakyRelu replacement pass on a minimal Relu graph."""

    def test_survives(self) -> None:
        """Node replacement must complete and forward pass must preserve output shapes."""
        rng = np.random.default_rng(42)
        inputs = {"X": rng.standard_normal((2, 4)).astype(np.float32)}
        assert_fx_pass_survives(_make_relu_model(), inputs, _relu_to_leaky_relu_pass)


class TestFxPassReluToLeakyReluAddRelu:
    """Custom Relu→LeakyRelu replacement pass on an Add → Relu graph."""

    def test_survives(self) -> None:
        """Node replacement must complete and forward pass must preserve output shapes."""
        rng = np.random.default_rng(42)
        inputs = {
            "A": rng.standard_normal((2, 4)).astype(np.float32),
            "B": rng.standard_normal((2, 4)).astype(np.float32),
        }
        assert_fx_pass_survives(_make_add_relu_model(), inputs, _relu_to_leaky_relu_pass)
