"""End-to-end smoke test: ONNX → IR → FX forward pass."""

from __future__ import annotations

import operator

import numpy as np
import torch
from onnx import TensorProto, helper

from protofx.emitters import emit_graph
from protofx.importers import import_model
from protofx.ir import DType, Graph, TensorType
from protofx.ir.derived_shape import get_authoritative_shape


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


class TestChildGraphEmissionSmoke:
    """Smoke-test child-graph helper integration with a temporary test op."""

    def test_nested_child_graph_roundtrip(self) -> None:
        """Nested child graph callables should run through emit_graph forward."""
        from protofx.ops._registry import register_op

        @register_op("_EmitterE2EChild")
        def _handler(node, args, fx_graph, module):
            helper = getattr(module, "_protofx_child_graph_emitter", None)
            if helper is None:
                raise ValueError("missing internal child graph emitter helper")

            child_graph = node.subgraphs.get("then_branch")
            if not isinstance(child_graph, Graph):
                raise ValueError("missing child graph for _EmitterE2EChild")

            branch, _ = helper.make_callable(owner_node=node, branch_name="then_branch", child_graph=child_graph)
            packed = fx_graph.call_function(branch, args=(args[0],))
            return [fx_graph.call_function(operator.getitem, args=(packed, 0))]

        graph = Graph(name="parent")
        x = graph.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="x")

        child = Graph(name="child", parent=graph)
        child_x = child.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="x")
        child_relu = child.make_node(
            op_type="Relu",
            inputs=[child_x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
        )
        child.set_graph_outputs([child_relu.outputs[0]])

        node = graph.make_node(
            op_type="_EmitterE2EChild",
            inputs=[x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
            subgraphs={"then_branch": child},
            name="cf",
        )
        graph.set_graph_outputs([node.outputs[0]])

        gm = emit_graph(graph)
        (result,) = gm(torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]]))
        expected = torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 4.0]])
        assert torch.equal(result, expected)


def test_e2e_derived_shape_beats_seed_metadata() -> None:
    """Importer propagation must make derived shape authoritative for emission."""
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [99])

    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["then_out"])],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["else_out"])],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )
    if_node = helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph)
    model = helper.make_model(
        helper.make_graph([if_node], "if_derived_truth", [cond, x], [y]),
        opset_imports=[helper.make_opsetid("", 17)],
    )

    graph = import_model(model)
    gm = emit_graph(graph)
    (result,) = gm(torch.tensor(True), torch.tensor([1.0, -2.0]))

    assert graph.outputs[0].tensor_type.shape == (99,)
    assert get_authoritative_shape(graph.outputs[0]) == (2,)
    assert result.shape == (2,)
