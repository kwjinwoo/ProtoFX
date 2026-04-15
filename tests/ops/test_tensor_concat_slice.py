"""Failing tests for Concat and Slice tensor op handlers."""

from __future__ import annotations

import numpy as np
import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


def _make_concat_graph(
    shapes: list[tuple[int, ...]],
    axis: int,
) -> Graph:
    """Build a minimal IR graph: (A, B, ...) → Concat(axis) → Y."""
    out_shape = list(shapes[0])
    for s in shapes[1:]:
        out_shape[axis] += s[axis]

    g = Graph(name="Concat_test")
    inputs = []
    for i, shape in enumerate(shapes):
        v = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape), name=f"input_{i}")
        inputs.append(v)
    node = g.make_node(
        op_type="Concat",
        inputs=inputs,
        output_types=[TensorType(dtype=DType.FLOAT32, shape=tuple(out_shape))],
        output_names=["Y"],
        attributes={"axis": axis},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestConcatHandler:
    """Verify that the Concat op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Concat must emit a call_function FX node."""
        g = _make_concat_graph([(2, 3), (2, 4)], axis=1)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_call_function_target(self) -> None:
        """The call_function target must be torch.cat."""
        g = _make_concat_graph([(2, 3), (2, 4)], axis=1)
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.cat

    def test_single_output(self) -> None:
        """Concat handler must return exactly one FX output node."""
        g = _make_concat_graph([(2, 3), (2, 4)], axis=1)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_axis_0(self) -> None:
        """Concat along axis=0 must produce correct results."""
        g = _make_concat_graph([(2, 3), (3, 3)], axis=0)
        gm = emit_graph(g)
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        (result,) = gm(a, b)
        expected = torch.cat([a, b], dim=0)
        assert torch.equal(result, expected)

    def test_forward_correctness_axis_1(self) -> None:
        """Concat along axis=1 must produce correct results."""
        g = _make_concat_graph([(2, 3), (2, 4)], axis=1)
        gm = emit_graph(g)
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        (result,) = gm(a, b)
        expected = torch.cat([a, b], dim=1)
        assert torch.equal(result, expected)

    def test_forward_correctness_three_inputs(self) -> None:
        """Concat with three inputs must produce correct results."""
        g = _make_concat_graph([(2, 1), (2, 2), (2, 3)], axis=1)
        gm = emit_graph(g)
        a = torch.ones(2, 1)
        b = torch.ones(2, 2) * 2
        c = torch.ones(2, 3) * 3
        (result,) = gm(a, b, c)
        expected = torch.cat([a, b, c], dim=1)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


def _make_slice_graph(
    input_shape: tuple[int, ...],
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    axes: tuple[int, ...] | None = None,
    steps: tuple[int, ...] | None = None,
) -> Graph:
    """Build a minimal IR graph: X, starts, ends[, axes, steps] → Slice → Y.

    All slice parameters are provided as initializers (static extraction).
    The output shape is computed from the slice parameters.
    """
    actual_axes = axes if axes is not None else tuple(range(len(starts)))
    actual_steps = steps if steps is not None else (1,) * len(starts)

    # Compute output shape
    out_shape = list(input_shape)
    for a, s, e, st in zip(actual_axes, starts, ends, actual_steps, strict=False):
        dim_size = input_shape[a]
        # Clamp start/end to valid range
        clamped_s = max(0, min(s, dim_size)) if s >= 0 else max(0, dim_size + s)
        clamped_e = max(0, min(e, dim_size)) if e >= 0 else max(0, dim_size + e)
        if e > 2**30:  # INT_MAX-like sentinel
            clamped_e = dim_size
        out_shape[a] = max(0, (clamped_e - clamped_s + st - 1) // st)

    g = Graph(name="Slice_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")

    starts_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(starts),)),
        data=np.array(starts, dtype=np.int64),
        name="starts",
    )
    ends_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(ends),)),
        data=np.array(ends, dtype=np.int64),
        name="ends",
    )
    inputs = [x, starts_val, ends_val]

    if axes is not None:
        axes_val = g.add_initializer(
            tensor_type=TensorType(dtype=DType.INT64, shape=(len(axes),)),
            data=np.array(axes, dtype=np.int64),
            name="axes",
        )
        inputs.append(axes_val)
    if steps is not None:
        if axes is None:
            # Must provide axes sentinel before steps
            axes_sentinel = g.add_sentinel()
            inputs.append(axes_sentinel)
        steps_val = g.add_initializer(
            tensor_type=TensorType(dtype=DType.INT64, shape=(len(steps),)),
            data=np.array(steps, dtype=np.int64),
            name="steps",
        )
        inputs.append(steps_val)

    node = g.make_node(
        op_type="Slice",
        inputs=inputs,
        output_types=[TensorType(dtype=DType.FLOAT32, shape=tuple(out_shape))],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestSliceHandler:
    """Verify that the Slice op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Slice must emit a call_function FX node."""
        g = _make_slice_graph((4, 6), starts=(1,), ends=(3,), axes=(0,))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """Slice handler must return exactly one FX output node."""
        g = _make_slice_graph((4, 6), starts=(1,), ends=(3,), axes=(0,))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_basic(self) -> None:
        """Slice on a single axis must produce correct results."""
        g = _make_slice_graph((4, 6), starts=(1,), ends=(3,), axes=(0,))
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        (result,) = gm(x)
        expected = x[1:3, :]
        assert torch.equal(result, expected)

    def test_forward_correctness_multi_axis(self) -> None:
        """Slice on multiple axes must produce correct results."""
        g = _make_slice_graph((4, 6), starts=(1, 2), ends=(3, 5), axes=(0, 1))
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        (result,) = gm(x)
        expected = x[1:3, 2:5]
        assert torch.equal(result, expected)

    def test_forward_correctness_with_step(self) -> None:
        """Slice with step > 1 must produce correct results."""
        g = _make_slice_graph((6, 4), starts=(0,), ends=(6,), axes=(0,), steps=(2,))
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        (result,) = gm(x)
        expected = x[0:6:2, :]
        assert torch.equal(result, expected)

    def test_forward_correctness_default_axes(self) -> None:
        """Slice without explicit axes must default to axis 0, 1, ...."""
        g = _make_slice_graph((4, 6), starts=(1,), ends=(3,))
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        (result,) = gm(x)
        expected = x[1:3]
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def _make_split_graph(
    input_shape: tuple[int, ...],
    split_sizes: tuple[int, ...],
    axis: int = 0,
    *,
    num_outputs: int | None = None,
) -> Graph:
    """Build a minimal IR graph: X, split → Split(axis) → Y0, Y1, ...

    The split parameter is provided as an initializer (static extraction).

    Args:
        input_shape: Shape of the input tensor.
        split_sizes: Sizes of each split chunk along *axis*.
        axis: The axis to split along.
        num_outputs: Number of outputs. Defaults to ``len(split_sizes)``.
    """
    if num_outputs is None:
        num_outputs = len(split_sizes)

    g = Graph(name="Split_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")

    split_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(split_sizes),)),
        data=np.array(split_sizes, dtype=np.int64),
        name="split",
    )

    # Compute output shapes
    out_types = []
    for size in split_sizes:
        out_shape = list(input_shape)
        out_shape[axis] = size
        out_types.append(TensorType(dtype=DType.FLOAT32, shape=tuple(out_shape)))

    node = g.make_node(
        op_type="Split",
        inputs=[x, split_val],
        output_types=out_types,
        output_names=[f"Y_{i}" for i in range(num_outputs)],
        attributes={"axis": axis},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


def _make_split_graph_no_split_input(
    input_shape: tuple[int, ...],
    num_outputs: int,
    axis: int = 0,
) -> Graph:
    """Build a Split IR graph without a split sizes input (equal split).

    Args:
        input_shape: Shape of the input tensor.
        num_outputs: Number of equal output chunks.
        axis: The axis to split along.
    """
    chunk_size = input_shape[axis] // num_outputs
    g = Graph(name="Split_equal_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")

    out_types = []
    for _ in range(num_outputs):
        out_shape = list(input_shape)
        out_shape[axis] = chunk_size
        out_types.append(TensorType(dtype=DType.FLOAT32, shape=tuple(out_shape)))

    node = g.make_node(
        op_type="Split",
        inputs=[x],
        output_types=out_types,
        output_names=[f"Y_{i}" for i in range(num_outputs)],
        attributes={"axis": axis, "num_outputs": num_outputs},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestSplitHandler:
    """Verify that the Split op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Split must emit a call_function FX node."""
        g = _make_split_graph((6, 4), split_sizes=(3, 3), axis=0)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_multiple_outputs(self) -> None:
        """Split handler must return multiple FX output nodes."""
        g = _make_split_graph((6, 4), split_sizes=(2, 4), axis=0)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 2

    def test_forward_correctness_equal_split(self) -> None:
        """Equal split along axis=0 must produce correct results."""
        g = _make_split_graph((6, 4), split_sizes=(3, 3), axis=0)
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        y0, y1 = gm(x)
        expected = torch.split(x, [3, 3], dim=0)
        assert torch.equal(y0, expected[0])
        assert torch.equal(y1, expected[1])

    def test_forward_correctness_unequal_split(self) -> None:
        """Unequal split along axis=1 must produce correct results."""
        g = _make_split_graph((3, 7), split_sizes=(2, 3, 2), axis=1)
        gm = emit_graph(g)
        x = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        y0, y1, y2 = gm(x)
        expected = torch.split(x, [2, 3, 2], dim=1)
        assert torch.equal(y0, expected[0])
        assert torch.equal(y1, expected[1])
        assert torch.equal(y2, expected[2])

    def test_forward_correctness_no_split_input(self) -> None:
        """Split without explicit split sizes must split equally."""
        g = _make_split_graph_no_split_input((6, 4), num_outputs=3, axis=0)
        gm = emit_graph(g)
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        y0, y1, y2 = gm(x)
        expected = torch.split(x, 2, dim=0)
        assert torch.equal(y0, expected[0])
        assert torch.equal(y1, expected[1])
        assert torch.equal(y2, expected[2])

    def test_forward_correctness_three_outputs(self) -> None:
        """Split into three chunks along axis=0 must produce correct results."""
        g = _make_split_graph((9, 2), split_sizes=(2, 3, 4), axis=0)
        gm = emit_graph(g)
        x = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        y0, y1, y2 = gm(x)
        expected = torch.split(x, [2, 3, 4], dim=0)
        assert torch.equal(y0, expected[0])
        assert torch.equal(y1, expected[1])
        assert torch.equal(y2, expected[2])
