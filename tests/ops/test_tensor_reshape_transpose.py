"""Failing tests for Reshape and Transpose tensor op handlers."""

from __future__ import annotations

import numpy as np
import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


def _make_reshape_graph(input_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> Graph:
    """Build a minimal IR graph: X, shape_init → Reshape → Y.

    The *target_shape* is provided as an int64 initializer (static extraction).
    """
    g = Graph(name="Reshape_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    shape_data = np.array(target_shape, dtype=np.int64)
    shape_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(target_shape),)),
        data=shape_data,
        name="shape",
    )
    node = g.make_node(
        op_type="Reshape",
        inputs=[x, shape_val],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=target_shape)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestReshapeHandler:
    """Verify that the Reshape op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Reshape must emit a call_function FX node."""
        g = _make_reshape_graph((2, 3), (3, 2))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_call_function_target(self) -> None:
        """The call_function target must be torch.reshape."""
        g = _make_reshape_graph((2, 3), (3, 2))
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.reshape

    def test_single_output(self) -> None:
        """Reshape handler must return exactly one FX output node."""
        g = _make_reshape_graph((2, 3), (3, 2))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness(self) -> None:
        """The emitted GraphModule must produce correct Reshape results."""
        g = _make_reshape_graph((2, 6), (3, 4))
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        (result,) = gm(x)
        expected = torch.reshape(x, (3, 4))
        assert torch.equal(result, expected)

    def test_forward_correctness_with_neg_one(self) -> None:
        """Reshape with -1 dimension must infer the correct size."""
        g = _make_reshape_graph((2, 3), (6, -1))
        # Output shape in IR uses -1 placeholder; actual shape computed at runtime
        gm = emit_graph(g)
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        (result,) = gm(x)
        expected = torch.reshape(x, (6, -1))
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def _make_transpose_graph(input_shape: tuple[int, ...], perm: list[int]) -> Graph:
    """Build a minimal IR graph: X → Transpose(perm) → Y."""
    output_shape = tuple(input_shape[i] for i in perm)
    g = Graph(name="Transpose_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    node = g.make_node(
        op_type="Transpose",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
        attributes={"perm": perm},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestTransposeHandler:
    """Verify that the Transpose op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Transpose must emit a call_function FX node."""
        g = _make_transpose_graph((2, 3), [1, 0])
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_call_function_target(self) -> None:
        """The call_function target must be torch.permute."""
        g = _make_transpose_graph((2, 3), [1, 0])
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.permute

    def test_single_output(self) -> None:
        """Transpose handler must return exactly one FX output node."""
        g = _make_transpose_graph((2, 3), [1, 0])
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_2d(self) -> None:
        """The emitted GraphModule must produce correct 2D Transpose results."""
        g = _make_transpose_graph((2, 3), [1, 0])
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = torch.permute(x, (1, 0))
        assert torch.equal(result, expected)

    def test_forward_correctness_3d(self) -> None:
        """The emitted GraphModule must produce correct 3D Transpose results."""
        g = _make_transpose_graph((2, 3, 4), [0, 2, 1])
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        (result,) = gm(x)
        expected = torch.permute(x, (0, 2, 1))
        assert torch.equal(result, expected)
