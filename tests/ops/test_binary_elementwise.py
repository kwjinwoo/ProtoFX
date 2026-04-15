"""Failing tests for binary elementwise op handlers (Add, Sub, Mul, Div, Pow)."""

from __future__ import annotations

import operator

import pytest
import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

_BINARY_OPS: list[tuple[str, object, object]] = [
    ("Add", operator.add, torch.add),
    ("Sub", operator.sub, torch.sub),
    ("Mul", operator.mul, torch.mul),
    ("Div", operator.truediv, torch.div),
    ("Pow", operator.pow, torch.pow),
]


def _make_binary_graph(op_type: str) -> Graph:
    """Build a minimal IR graph: (A, B) → Op → Y."""
    g = Graph(name=f"{op_type}_test")
    a = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="A")
    b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="B")
    node = g.make_node(
        op_type=op_type,
        inputs=[a, b],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestBinaryElementwiseHandler:
    """Verify that binary elementwise op handlers emit correct FX nodes."""

    @pytest.mark.parametrize(("op_type", "_py_op", "torch_fn"), _BINARY_OPS, ids=[o[0] for o in _BINARY_OPS])
    def test_emits_call_function(self, op_type: str, _py_op: object, torch_fn: object) -> None:
        """Binary op must emit a call_function FX node."""
        g = _make_binary_graph(op_type)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    @pytest.mark.parametrize(("op_type", "_py_op", "torch_fn"), _BINARY_OPS, ids=[o[0] for o in _BINARY_OPS])
    def test_call_function_target(self, op_type: str, _py_op: object, torch_fn: object) -> None:
        """The call_function target must be the correct torch function."""
        g = _make_binary_graph(op_type)
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch_fn

    @pytest.mark.parametrize(("op_type", "_py_op", "torch_fn"), _BINARY_OPS, ids=[o[0] for o in _BINARY_OPS])
    def test_single_output(self, op_type: str, _py_op: object, torch_fn: object) -> None:
        """Binary op handler must return exactly one FX output node."""
        g = _make_binary_graph(op_type)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    @pytest.mark.parametrize(("op_type", "_py_op", "torch_fn"), _BINARY_OPS, ids=[o[0] for o in _BINARY_OPS])
    def test_forward_correctness(self, op_type: str, _py_op: object, torch_fn: object) -> None:
        """The emitted GraphModule must produce correct numerical results."""
        g = _make_binary_graph(op_type)
        gm = emit_graph(g)
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([[2.0, 3.0, 1.0], [1.0, 2.0, 3.0]])
        (result,) = gm(a, b)
        expected = torch_fn(a, b)
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Equal
# ---------------------------------------------------------------------------


class TestEqualHandler:
    """Verify that the Equal op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Equal must emit a call_function FX node."""
        g = _make_binary_graph("Equal")
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_forward_correctness(self) -> None:
        """Equal must produce correct boolean results."""
        g = _make_binary_graph("Equal")
        gm = emit_graph(g)
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([[1.0, 3.0, 3.0], [4.0, 0.0, 6.0]])
        (result,) = gm(a, b)
        expected = torch.eq(a, b)
        assert torch.equal(result, expected)
