"""Tests for linear algebra op handlers (MatMul, Gemm)."""

from __future__ import annotations

import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# MatMul
# ---------------------------------------------------------------------------


def _make_matmul_graph(shape_a: tuple[int, ...], shape_b: tuple[int, ...], shape_y: tuple[int, ...]) -> Graph:
    """Build a minimal IR graph: (A, B) → MatMul → Y."""
    g = Graph(name="MatMul_test")
    a = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape_a), name="A")
    b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape_b), name="B")
    node = g.make_node(
        op_type="MatMul",
        inputs=[a, b],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=shape_y)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestMatMulHandler:
    """Verify that the MatMul op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """MatMul must emit a call_function FX node."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_call_function_target(self) -> None:
        """The call_function target must be torch.matmul."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.matmul

    def test_single_output(self) -> None:
        """MatMul handler must return exactly one FX output node."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_2d(self) -> None:
        """The emitted GraphModule must produce correct 2D MatMul results."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        (result,) = gm(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected)

    def test_forward_correctness_batched(self) -> None:
        """The emitted GraphModule must produce correct batched MatMul results."""
        g = _make_matmul_graph((2, 3, 4), (2, 4, 5), (2, 3, 5))
        gm = emit_graph(g)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        (result,) = gm(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected)
