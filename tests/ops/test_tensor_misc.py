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

    def test_forward_correctness_dynamic_indices(self) -> None:
        """Gather with dynamic indices (graph input) must work correctly."""
        g = Graph(name="Gather_dynamic_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(5, 3)), name="X")
        idx = g.add_input(tensor_type=TensorType(dtype=DType.INT64, shape=(2,)), name="indices")
        node = g.make_node(
            op_type="Gather",
            inputs=[x, idx],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
            output_names=["Y"],
            attributes={"axis": 0},
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        x_data = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        idx_data = torch.tensor([1, 3], dtype=torch.long)
        (result,) = gm(x_data, idx_data)
        expected = torch.index_select(x_data, 0, idx_data)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# GatherND
# ---------------------------------------------------------------------------


class TestGatherNDHandler:
    """Verify that the GatherND op handler emits correct FX nodes."""

    def test_forward_correctness_batch_dims_0(self) -> None:
        """GatherND with batch_dims=0 on a 2D tensor must gather correct elements."""
        g = Graph(name="GatherND_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(3, 4)), name="X")
        idx = g.add_input(tensor_type=TensorType(dtype=DType.INT64, shape=(2, 1)), name="indices")
        node = g.make_node(
            op_type="GatherND",
            inputs=[x, idx],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 4))],
            output_names=["Y"],
            attributes={"batch_dims": 0},
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        x_data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        idx_data = torch.tensor([[0], [2]], dtype=torch.long)
        (result,) = gm(x_data, idx_data)
        expected = x_data[torch.tensor([0, 2])]
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------


class TestWhereHandler:
    """Verify that the Where op handler emits correct FX nodes."""

    def test_forward_correctness(self) -> None:
        """Where must select elements based on condition."""
        g = Graph(name="Where_test")
        cond = g.add_input(tensor_type=TensorType(dtype=DType.BOOL, shape=(2, 3)), name="condition")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
        y = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="Y_in")
        node = g.make_node(
            op_type="Where",
            inputs=[cond, x, y],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
            output_names=["Y_out"],
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        c = torch.tensor([[True, False, True], [False, True, False]])
        x_data = torch.ones(2, 3)
        y_data = torch.zeros(2, 3)
        (result,) = gm(c, x_data, y_data)
        expected = torch.where(c, x_data, y_data)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# And
# ---------------------------------------------------------------------------


class TestAndHandler:
    """Verify that the And op handler emits correct FX nodes."""

    def test_forward_correctness(self) -> None:
        """And must compute element-wise logical AND."""
        g = Graph(name="And_test")
        a = g.add_input(tensor_type=TensorType(dtype=DType.BOOL, shape=(2, 3)), name="A")
        b = g.add_input(tensor_type=TensorType(dtype=DType.BOOL, shape=(2, 3)), name="B")
        node = g.make_node(
            op_type="And",
            inputs=[a, b],
            output_types=[TensorType(dtype=DType.BOOL, shape=(2, 3))],
            output_names=["C"],
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        a_data = torch.tensor([[True, False, True], [False, True, False]])
        b_data = torch.tensor([[True, True, False], [False, True, True]])
        (result,) = gm(a_data, b_data)
        expected = torch.logical_and(a_data, b_data)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# IsNaN
# ---------------------------------------------------------------------------


class TestIsNaNHandler:
    """Verify that the IsNaN op handler emits correct FX nodes."""

    def test_forward_correctness(self) -> None:
        """IsNaN must detect NaN elements."""
        g = Graph(name="IsNaN_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(4,)), name="X")
        node = g.make_node(
            op_type="IsNaN",
            inputs=[x],
            output_types=[TensorType(dtype=DType.BOOL, shape=(4,))],
            output_names=["Y"],
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        x_data = torch.tensor([1.0, float("nan"), 3.0, float("nan")])
        (result,) = gm(x_data)
        expected = torch.isnan(x_data)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# GatherElements
# ---------------------------------------------------------------------------


class TestGatherElementsHandler:
    """Verify that the GatherElements op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """GatherElements must emit a call_function FX node."""
        g = Graph(name="GatherElements_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(3, 3)), name="X")
        idx = g.add_input(tensor_type=TensorType(dtype=DType.INT64, shape=(2, 3)), name="indices")
        node = g.make_node(
            op_type="GatherElements",
            inputs=[x, idx],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
            output_names=["Y"],
            attributes={"axis": 0},
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_forward_correctness(self) -> None:
        """GatherElements must produce correct results."""
        g = Graph(name="GatherElements_test")
        x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(3, 3)), name="X")
        idx = g.add_input(tensor_type=TensorType(dtype=DType.INT64, shape=(2, 3)), name="indices")
        node = g.make_node(
            op_type="GatherElements",
            inputs=[x, idx],
            output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
            output_names=["Y"],
            attributes={"axis": 0},
        )
        g.set_graph_outputs(list(node.outputs))
        gm = emit_graph(g)
        x_data = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        idx_data = torch.tensor([[2, 0, 1], [0, 2, 1]], dtype=torch.long)
        (result,) = gm(x_data, idx_data)
        expected = torch.gather(x_data, 0, idx_data)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Trilu
# ---------------------------------------------------------------------------


def _make_trilu_graph(
    input_shape: tuple[int, ...],
    upper: int = 1,
    k: int | None = None,
) -> Graph:
    """Build a minimal IR graph: X [, k] → Trilu(upper) → Y.

    Args:
        input_shape: Shape of the input tensor.
        upper: 1 for upper triangular, 0 for lower triangular.
        k: Optional diagonal offset. Positive shifts up, negative shifts down.
    """
    g = Graph(name="Trilu_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")

    inputs = [x]
    if k is not None:
        k_val = g.add_initializer(
            tensor_type=TensorType(dtype=DType.INT64, shape=()),
            data=np.array(k, dtype=np.int64),
            name="k",
        )
        inputs.append(k_val)

    node = g.make_node(
        op_type="Trilu",
        inputs=inputs,
        output_types=[TensorType(dtype=DType.FLOAT32, shape=input_shape)],
        output_names=["Y"],
        attributes={"upper": upper},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestTriluHandler:
    """Verify that the Trilu op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """Trilu must emit a call_function FX node."""
        g = _make_trilu_graph((3, 4), upper=1)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """Trilu handler must return exactly one FX output node."""
        g = _make_trilu_graph((3, 4), upper=1)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_upper(self) -> None:
        """Trilu with upper=1 must produce upper triangular matrix."""
        g = _make_trilu_graph((4, 4), upper=1)
        gm = emit_graph(g)
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        (result,) = gm(x)
        expected = torch.triu(x)
        assert torch.equal(result, expected)

    def test_forward_correctness_lower(self) -> None:
        """Trilu with upper=0 must produce lower triangular matrix."""
        g = _make_trilu_graph((4, 4), upper=0)
        gm = emit_graph(g)
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        (result,) = gm(x)
        expected = torch.tril(x)
        assert torch.equal(result, expected)

    def test_forward_correctness_upper_with_k(self) -> None:
        """Trilu upper with k=1 must shift the diagonal up by 1."""
        g = _make_trilu_graph((4, 4), upper=1, k=1)
        gm = emit_graph(g)
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        (result,) = gm(x)
        expected = torch.triu(x, diagonal=1)
        assert torch.equal(result, expected)

    def test_forward_correctness_lower_with_negative_k(self) -> None:
        """Trilu lower with k=-1 must shift the diagonal down by 1."""
        g = _make_trilu_graph((4, 4), upper=0, k=-1)
        gm = emit_graph(g)
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        (result,) = gm(x)
        expected = torch.tril(x, diagonal=-1)
        assert torch.equal(result, expected)

    def test_forward_correctness_non_square(self) -> None:
        """Trilu must work on non-square matrices."""
        g = _make_trilu_graph((3, 5), upper=1)
        gm = emit_graph(g)
        x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        (result,) = gm(x)
        expected = torch.triu(x)
        assert torch.equal(result, expected)
