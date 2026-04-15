"""Failing tests for unary elementwise op handlers (Sigmoid, Tanh, Abs, Neg, Exp, Log, Sqrt)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

_UNARY_OPS: list[tuple[str, Callable[..., torch.Tensor]]] = [
    ("Sigmoid", torch.sigmoid),
    ("Tanh", torch.tanh),
    ("Abs", torch.abs),
    ("Neg", torch.neg),
    ("Exp", torch.exp),
    ("Log", torch.log),
    ("Sqrt", torch.sqrt),
    ("Erf", torch.erf),
]


def _make_unary_graph(op_type: str) -> Graph:
    """Build a minimal IR graph: X → Op → Y."""
    g = Graph(name=f"{op_type}_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
    node = g.make_node(
        op_type=op_type,
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestUnaryElementwiseHandler:
    """Verify that unary elementwise op handlers emit correct FX nodes."""

    @pytest.mark.parametrize(("op_type", "torch_fn"), _UNARY_OPS, ids=[o[0] for o in _UNARY_OPS])
    def test_emits_call_function(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Unary op must emit a call_function FX node."""
        g = _make_unary_graph(op_type)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    @pytest.mark.parametrize(("op_type", "torch_fn"), _UNARY_OPS, ids=[o[0] for o in _UNARY_OPS])
    def test_call_function_target(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """The call_function target must be the correct torch function."""
        g = _make_unary_graph(op_type)
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch_fn

    @pytest.mark.parametrize(("op_type", "torch_fn"), _UNARY_OPS, ids=[o[0] for o in _UNARY_OPS])
    def test_single_output(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Unary op handler must return exactly one FX output node."""
        g = _make_unary_graph(op_type)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    @pytest.mark.parametrize(("op_type", "torch_fn"), _UNARY_OPS, ids=[o[0] for o in _UNARY_OPS])
    def test_forward_correctness(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """The emitted GraphModule must produce correct numerical results."""
        g = _make_unary_graph(op_type)
        gm = emit_graph(g)
        # Use positive values so Log and Sqrt are valid
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = torch_fn(x)
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Not
# ---------------------------------------------------------------------------


class TestNotHandler:
    """Verify that the Not op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Not must emit a call_function FX node."""
        g = Graph(name="Not_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.BOOL, shape=(2, 3)), name="X")
        node = g.make_node(
            op_type="Not",
            inputs=[x],
            output_types=[TensorType(dtype=DType.BOOL, shape=(2, 3))],
            output_names=["Y"],
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_forward_correctness(self) -> None:
        """Not must produce correct boolean negation."""
        g = Graph(name="Not_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.BOOL, shape=(4,)), name="X")
        node = g.make_node(
            op_type="Not",
            inputs=[x],
            output_types=[TensorType(dtype=DType.BOOL, shape=(4,))],
            output_names=["Y"],
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        x_data = torch.tensor([True, False, True, False])
        (result,) = gm(x_data)
        expected = torch.logical_not(x_data)
        assert torch.equal(result, expected)
