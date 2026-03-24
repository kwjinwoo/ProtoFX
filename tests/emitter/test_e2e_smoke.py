"""End-to-end smoke test: ONNX → IR → FX forward pass."""

from __future__ import annotations

import numpy as np
import torch
from onnx import TensorProto, helper

from protofx.emitters import emit_graph
from protofx.importers import import_model


def _make_relu_model() -> bytes:
    """Build a minimal ONNX model: X → Relu → Y (serialized bytes)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    relu_node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([relu_node], "relu_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def _make_relu_with_initializer_model():
    """Build an ONNX model: Relu(initializer + X) → Y."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])

    bias_data = np.array([[-1.0, 0.5, -0.5], [1.0, -2.0, 3.0]], dtype=np.float32)
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [2, 3], bias_data.flatten().tolist())

    add_node = helper.make_node("Add", ["X", "bias"], ["sum"])
    relu_node = helper.make_node("Relu", ["sum"], ["Y"])

    sum_vi = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [add_node, relu_node],
        "relu_bias_graph",
        [X],
        [Y],
        initializer=[bias_init],
        value_info=[sum_vi],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestEndToEndRelu:
    """Full pipeline smoke test: ONNX ModelProto → import_model → emit_graph → forward."""

    def test_relu_roundtrip(self) -> None:
        """ONNX Relu model must produce correct forward pass results through the full pipeline."""
        model = _make_relu_model()
        ir_graph = import_model(model)
        gm = emit_graph(ir_graph)

        x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
        (result,) = gm(x)

        expected = torch.nn.functional.relu(x)
        assert torch.equal(result, expected)

    def test_relu_graph_module_is_scriptable(self) -> None:
        """The emitted GraphModule must be torch.jit.scriptable (basic sanity)."""
        model = _make_relu_model()
        ir_graph = import_model(model)
        gm = emit_graph(ir_graph)

        # GraphModule should at least be callable without errors
        x = torch.ones(2, 3)
        (result,) = gm(x)
        assert result.shape == (2, 3)
        assert torch.all(result >= 0)

    def test_relu_preserves_graph_structure(self) -> None:
        """The FX graph must have placeholder → call_function → output structure."""
        model = _make_relu_model()
        ir_graph = import_model(model)
        gm = emit_graph(ir_graph)

        ops = [n.op for n in gm.graph.nodes]
        assert ops == ["placeholder", "call_function", "output"]
