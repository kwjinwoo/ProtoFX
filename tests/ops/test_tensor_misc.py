"""Failing tests for Identity, Cast, Expand, and Gather tensor op handlers."""

from __future__ import annotations

import numpy as np
import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _make_identity_graph(shape: tuple[int, ...]) -> Graph:
    """Build a minimal IR graph: X → Identity → Y."""
    g = Graph(name="Identity_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape), name="X")
    node = g.make_node(
        op_type="Identity",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=shape)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestIdentityHandler:
    """Verify that the Identity op handler passes through values."""

    def test_single_output(self) -> None:
        """Identity handler must return exactly one FX output node."""
        g = _make_identity_graph((2, 3))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness(self) -> None:
        """The emitted GraphModule must pass through the input unchanged."""
        g = _make_identity_graph((2, 3))
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        assert torch.equal(result, x)


# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------


def _make_cast_graph(
    input_shape: tuple[int, ...],
    from_dtype: DType,
    to_dtype: DType,
    to_onnx_dtype_int: int,
) -> Graph:
    """Build a minimal IR graph: X → Cast(to=to_onnx_dtype_int) → Y."""
    g = Graph(name="Cast_test")
    x = g.add_input(tensor_type=TensorType(dtype=from_dtype, shape=input_shape), name="X")
    node = g.make_node(
        op_type="Cast",
        inputs=[x],
        output_types=[TensorType(dtype=to_dtype, shape=input_shape)],
        output_names=["Y"],
        attributes={"to": to_onnx_dtype_int},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestCastHandler:
    """Verify that the Cast op handler emits correct dtype conversion."""

    def test_emits_call_method(self) -> None:
        """Cast must emit a call_method FX node."""
        g = _make_cast_graph((2, 3), DType.FLOAT32, DType.INT64, DType.INT64.value)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_method" in ops

    def test_single_output(self) -> None:
        """Cast handler must return exactly one FX output node."""
        g = _make_cast_graph((2, 3), DType.FLOAT32, DType.INT64, DType.INT64.value)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_float_to_int(self) -> None:
        """Cast from float32 to int64 must produce correct results."""
        g = _make_cast_graph((2, 3), DType.FLOAT32, DType.INT64, DType.INT64.value)
        gm = emit_graph(g)
        x = torch.tensor([[1.5, 2.7, 3.1], [4.0, 5.9, 6.0]])
        (result,) = gm(x)
        expected = x.to(torch.int64)
        assert torch.equal(result, expected)

    def test_forward_correctness_float_to_half(self) -> None:
        """Cast from float32 to float16 must produce correct results."""
        g = _make_cast_graph((2, 3), DType.FLOAT32, DType.FLOAT16, DType.FLOAT16.value)
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = x.to(torch.float16)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Expand
# ---------------------------------------------------------------------------


def _make_expand_graph(
    input_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> Graph:
    """Build a minimal IR graph: X, shape_init → Expand → Y."""
    g = Graph(name="Expand_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    shape_data = np.array(target_shape, dtype=np.int64)
    shape_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(target_shape),)),
        data=shape_data,
        name="shape",
    )
    node = g.make_node(
        op_type="Expand",
        inputs=[x, shape_val],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=target_shape)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestExpandHandler:
    """Verify that the Expand op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Expand must emit a call_function FX node."""
        g = _make_expand_graph((1, 3), (2, 3))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """Expand handler must return exactly one FX output node."""
        g = _make_expand_graph((1, 3), (2, 3))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_broadcast(self) -> None:
        """Expand must broadcast input to target shape."""
        g = _make_expand_graph((1, 3), (4, 3))
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        (result,) = gm(x)
        expected = x.expand(4, 3)
        assert torch.equal(result, expected)

    def test_forward_correctness_add_dim(self) -> None:
        """Expand must handle adding leading dimensions."""
        g = _make_expand_graph((3,), (2, 3))
        gm = emit_graph(g)
        x = torch.tensor([1.0, 2.0, 3.0])
        (result,) = gm(x)
        expected = x.expand(2, 3)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


def _make_gather_graph(
    data_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    indices_data: list[int],
    axis: int = 0,
) -> Graph:
    """Build a minimal IR graph: X, indices_init → Gather(axis) → Y."""
    g = Graph(name="Gather_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=data_shape), name="X")
    idx_data = np.array(indices_data, dtype=np.int64).reshape(indices_shape)
    idx_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=indices_shape),
        data=idx_data,
        name="indices",
    )
    node = g.make_node(
        op_type="Gather",
        inputs=[x, idx_val],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
        attributes={"axis": axis},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestGatherHandler:
    """Verify that the Gather op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Gather must emit a call_function FX node."""
        g = _make_gather_graph((4, 3), (2,), (2, 3), [0, 2], axis=0)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """Gather handler must return exactly one FX output node."""
        g = _make_gather_graph((4, 3), (2,), (2, 3), [0, 2], axis=0)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_axis_0(self) -> None:
        """Gather along axis=0 must select correct rows."""
        g = _make_gather_graph((4, 3), (2,), (2, 3), [0, 2], axis=0)
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        (result,) = gm(x)
        expected = torch.index_select(x, 0, torch.tensor([0, 2]))
        assert torch.equal(result, expected)

    def test_forward_correctness_axis_1(self) -> None:
        """Gather along axis=1 must select correct columns."""
        g = _make_gather_graph((3, 4), (2,), (3, 2), [1, 3], axis=1)
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        (result,) = gm(x)
        expected = torch.index_select(x, 1, torch.tensor([1, 3]))
        assert torch.equal(result, expected)

    def test_forward_correctness_scalar_index(self) -> None:
        """Gather with a scalar index must reduce the gathered dimension."""
        g = _make_gather_graph((4, 3), (), (3,), [2], axis=0)
        gm = emit_graph(g)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        (result,) = gm(x)
        expected = torch.index_select(x, 0, torch.tensor([2])).squeeze(0)
        assert torch.equal(result, expected)
