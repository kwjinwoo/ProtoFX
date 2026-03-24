"""Failing tests for the Relu op handler."""

from __future__ import annotations

import torch
import torch.fx

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType


class TestReluHandler:
    """Verify that the Relu op handler emits a call_function node."""

    def _make_relu_graph(self) -> Graph:
        """Build a minimal IR graph: X → Relu → Y."""
        g = Graph(name="relu_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
        relu_node = g.make_node(
            op_type="Relu",
            inputs=[x],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
            output_names=["Y"],
        )
        g.set_graph_outputs(list(relu_node.outputs))
        return g

    def test_relu_emits_call_function(self) -> None:
        """Relu must emit a call_function FX node."""
        g = self._make_relu_graph()
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_relu_call_function_target_is_relu(self) -> None:
        """The call_function target must be torch.nn.functional.relu."""
        g = self._make_relu_graph()
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.nn.functional.relu

    def test_relu_single_output(self) -> None:
        """Relu handler must return exactly one FX output node."""
        g = self._make_relu_graph()
        gm = emit_graph(g)
        # output node args[0] should be a tuple with 1 element
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_relu_forward_correctness(self) -> None:
        """The emitted GraphModule must produce correct Relu results."""
        g = self._make_relu_graph()
        gm = emit_graph(g)
        x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
        (result,) = gm(x)
        expected = torch.nn.functional.relu(x)
        assert torch.equal(result, expected)
