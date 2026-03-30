"""Failing tests for Flatten, Squeeze, and Unsqueeze tensor op handlers."""

from __future__ import annotations

import numpy as np
import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


def _make_flatten_graph(input_shape: tuple[int, ...], axis: int) -> Graph:
    """Build a minimal IR graph: X → Flatten(axis) → Y."""
    # Compute expected output shape
    left = 1
    for d in input_shape[:axis]:
        left *= d
    right = 1
    for d in input_shape[axis:]:
        right *= d
    output_shape = (left, right)

    g = Graph(name="Flatten_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    node = g.make_node(
        op_type="Flatten",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
        attributes={"axis": axis},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestFlattenHandler:
    """Verify that the Flatten op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Flatten must emit a call_function FX node."""
        g = _make_flatten_graph((2, 3, 4), axis=1)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_call_function_target(self) -> None:
        """The call_function target must be torch.reshape."""
        g = _make_flatten_graph((2, 3, 4), axis=1)
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.reshape

    def test_single_output(self) -> None:
        """Flatten handler must return exactly one FX output node."""
        g = _make_flatten_graph((2, 3, 4), axis=1)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_axis_1(self) -> None:
        """Flatten with axis=1 must produce (batch, features) shape."""
        g = _make_flatten_graph((2, 3, 4), axis=1)
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        (result,) = gm(x)
        expected = x.reshape(2, 12)
        assert torch.equal(result, expected)

    def test_forward_correctness_axis_0(self) -> None:
        """Flatten with axis=0 must produce (1, N) shape."""
        g = _make_flatten_graph((2, 3, 4), axis=0)
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        (result,) = gm(x)
        expected = x.reshape(1, 24)
        assert torch.equal(result, expected)

    def test_forward_correctness_axis_2(self) -> None:
        """Flatten with axis=2 on a 3D tensor must produce (6, 4)."""
        g = _make_flatten_graph((2, 3, 4), axis=2)
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        (result,) = gm(x)
        expected = x.reshape(6, 4)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Squeeze (opset 13+: axes as input tensor)
# ---------------------------------------------------------------------------


def _make_squeeze_graph(input_shape: tuple[int, ...], axes: tuple[int, ...]) -> Graph:
    """Build a minimal IR graph: X, axes_init → Squeeze → Y.

    Opset 13+ passes axes as a second input tensor (initializer).
    """
    output_shape = tuple(d for i, d in enumerate(input_shape) if i not in axes)
    g = Graph(name="Squeeze_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    axes_data = np.array(axes, dtype=np.int64)
    axes_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(axes),)),
        data=axes_data,
        name="axes",
    )
    node = g.make_node(
        op_type="Squeeze",
        inputs=[x, axes_val],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


def _make_squeeze_no_axes_graph(input_shape: tuple[int, ...]) -> Graph:
    """Build a Squeeze graph without axes input (squeeze all 1-dims)."""
    output_shape = tuple(d for d in input_shape if d != 1)
    g = Graph(name="Squeeze_no_axes_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    node = g.make_node(
        op_type="Squeeze",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestSqueezeHandler:
    """Verify that the Squeeze op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Squeeze must emit a call_function FX node."""
        g = _make_squeeze_graph((1, 3, 1, 4), axes=(0,))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """Squeeze handler must return exactly one FX output node."""
        g = _make_squeeze_graph((1, 3, 1, 4), axes=(0,))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_single_axis(self) -> None:
        """Squeeze with single axis must remove that dim."""
        g = _make_squeeze_graph((1, 3, 4), axes=(0,))
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        (result,) = gm(x)
        expected = torch.squeeze(x, 0)
        assert torch.equal(result, expected)

    def test_forward_correctness_multiple_axes(self) -> None:
        """Squeeze with multiple axes must remove all specified dims."""
        g = _make_squeeze_graph((1, 3, 1, 4), axes=(0, 2))
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
        (result,) = gm(x)
        expected = torch.squeeze(torch.squeeze(x, 0), 1)  # after removing dim 0, dim 2 becomes dim 1
        assert torch.equal(result, expected)

    def test_forward_correctness_no_axes(self) -> None:
        """Squeeze without axes must remove all dims of size 1."""
        g = _make_squeeze_no_axes_graph((1, 3, 1, 4))
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
        (result,) = gm(x)
        expected = torch.squeeze(x)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Unsqueeze (opset 13+: axes as input tensor)
# ---------------------------------------------------------------------------


def _make_unsqueeze_graph(input_shape: tuple[int, ...], axes: tuple[int, ...]) -> Graph:
    """Build a minimal IR graph: X, axes_init → Unsqueeze → Y.

    Opset 13+ passes axes as a second input tensor (initializer).
    """
    # Compute output shape by inserting 1s at the specified axes
    ndim_out = len(input_shape) + len(axes)
    sorted_axes = sorted(a if a >= 0 else a + ndim_out for a in axes)
    out = list(input_shape)
    for _i, ax in enumerate(sorted_axes):
        out.insert(ax, 1)
    output_shape = tuple(out)

    g = Graph(name="Unsqueeze_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    axes_data = np.array(axes, dtype=np.int64)
    axes_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(axes),)),
        data=axes_data,
        name="axes",
    )
    node = g.make_node(
        op_type="Unsqueeze",
        inputs=[x, axes_val],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestUnsqueezeHandler:
    """Verify that the Unsqueeze op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Unsqueeze must emit a call_function FX node."""
        g = _make_unsqueeze_graph((3, 4), axes=(0,))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """Unsqueeze handler must return exactly one FX output node."""
        g = _make_unsqueeze_graph((3, 4), axes=(0,))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_single_axis(self) -> None:
        """Unsqueeze with single axis must insert a dim."""
        g = _make_unsqueeze_graph((3, 4), axes=(0,))
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        (result,) = gm(x)
        expected = torch.unsqueeze(x, 0)
        assert torch.equal(result, expected)

    def test_forward_correctness_multiple_axes(self) -> None:
        """Unsqueeze with multiple axes must insert all specified dims."""
        g = _make_unsqueeze_graph((3, 4), axes=(0, 2))
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        (result,) = gm(x)
        expected = torch.unsqueeze(torch.unsqueeze(x, 0), 2)
        assert torch.equal(result, expected)

    def test_forward_correctness_negative_axis(self) -> None:
        """Unsqueeze with negative axis must insert at the correct position."""
        g = _make_unsqueeze_graph((3, 4), axes=(-1,))
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        (result,) = gm(x)
        expected = torch.unsqueeze(x, -1)
        assert torch.equal(result, expected)
