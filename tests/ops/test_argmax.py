"""Tests for ArgMax op handler."""

from __future__ import annotations

import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _make_argmax_graph(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    axis: int = 0,
    keepdims: int = 1,
    select_last_index: int = 0,
) -> Graph:
    """Build an ArgMax IR graph.

    Args:
        input_shape: Shape of the input tensor.
        output_shape: Expected shape of the output tensor.
        axis: Axis along which to compute argmax.
        keepdims: Whether to keep the reduced dimension.
        select_last_index: Whether to select the last index on ties.

    Returns:
        An IR ``Graph`` with a single ArgMax node.
    """
    g = Graph(name="ArgMax_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    node = g.make_node(
        op_type="ArgMax",
        inputs=[x],
        output_types=[TensorType(dtype=DType.INT64, shape=output_shape)],
        output_names=["Y"],
        attributes={"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestArgMaxStructure:
    """Verify that ArgMax emits correct FX graph structure."""

    def test_emits_call_function(self) -> None:
        """ArgMax must emit a call_function FX node."""
        g = _make_argmax_graph((2, 3, 4), (1, 3, 4), axis=0, keepdims=1)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """ArgMax handler must return exactly one FX output node."""
        g = _make_argmax_graph((2, 3, 4), (1, 3, 4), axis=0, keepdims=1)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1


# ---------------------------------------------------------------------------
# Forward correctness tests
# ---------------------------------------------------------------------------


class TestArgMaxForwardCorrectness:
    """Verify numerical correctness for ArgMax."""

    def test_axis0_keepdims(self) -> None:
        """ArgMax along axis=0 with keepdims=1."""
        g = _make_argmax_graph((2, 3, 4), (1, 3, 4), axis=0, keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch.argmax(x, dim=0, keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_axis1_keepdims(self) -> None:
        """ArgMax along axis=1 with keepdims=1."""
        g = _make_argmax_graph((2, 3, 4), (2, 1, 4), axis=1, keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch.argmax(x, dim=1, keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_axis_negative(self) -> None:
        """ArgMax along axis=-1 with keepdims=1."""
        g = _make_argmax_graph((2, 3, 4), (2, 3, 1), axis=-1, keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch.argmax(x, dim=-1, keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_no_keepdims(self) -> None:
        """ArgMax along axis=1 with keepdims=0."""
        g = _make_argmax_graph((2, 3, 4), (2, 4), axis=1, keepdims=0)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch.argmax(x, dim=1, keepdim=False)
        torch.testing.assert_close(result, expected)
