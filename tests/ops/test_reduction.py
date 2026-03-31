"""Tests for reduction op handlers (ReduceMean, ReduceSum, ReduceMax, ReduceMin)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _make_reduce_graph_attr_axes(
    op_type: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    axes: list[int],
    keepdims: int = 1,
) -> Graph:
    """Build a reduce IR graph with axes as an attribute (opset < 18 style)."""
    g = Graph(name=f"{op_type}_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    node = g.make_node(
        op_type=op_type,
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
        attributes={"axes": axes, "keepdims": keepdims},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


def _make_reduce_graph_input_axes(
    op_type: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    axes: list[int],
    keepdims: int = 1,
) -> Graph:
    """Build a reduce IR graph with axes as a second input tensor (opset 18 style)."""
    g = Graph(name=f"{op_type}_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    axes_data = np.array(axes, dtype=np.int64)
    axes_val = g.add_initializer(
        tensor_type=TensorType(dtype=DType.INT64, shape=(len(axes),)),
        data=axes_data,
        name="axes",
    )
    node = g.make_node(
        op_type=op_type,
        inputs=[x, axes_val],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
        attributes={"keepdims": keepdims},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


def _make_reduce_graph_no_axes(
    op_type: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    keepdims: int = 1,
) -> Graph:
    """Build a reduce IR graph with no axes (reduce all dimensions)."""
    g = Graph(name=f"{op_type}_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    node = g.make_node(
        op_type=op_type,
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=output_shape)],
        output_names=["Y"],
        attributes={"keepdims": keepdims},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


def _make_reduce_graph_noop_empty(
    op_type: str,
    input_shape: tuple[int, ...],
) -> Graph:
    """Build a reduce IR graph with noop_with_empty_axes=1 and no axes."""
    g = Graph(name=f"{op_type}_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=input_shape), name="X")
    node = g.make_node(
        op_type=op_type,
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=input_shape)],
        output_names=["Y"],
        attributes={"keepdims": 1, "noop_with_empty_axes": 1},
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# Op table: (onnx_name, torch_fn, reduce_all_fn)
# ---------------------------------------------------------------------------

_REDUCE_OPS: list[tuple[str, Callable[..., torch.Tensor]]] = [
    ("ReduceMean", torch.mean),
    ("ReduceSum", torch.sum),
    ("ReduceMax", torch.amax),
    ("ReduceMin", torch.amin),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReduceOpsStructure:
    """Verify that ReduceMean/Sum/Max/Min emit correct FX structure."""

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_emits_call_function(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce op must emit a call_function FX node."""
        g = _make_reduce_graph_attr_axes(op_type, (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_call_function_target(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """The call_function target must be the correct torch function."""
        g = _make_reduce_graph_attr_axes(op_type, (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch_fn

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_single_output(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce op handler must return exactly one FX output node."""
        g = _make_reduce_graph_attr_axes(op_type, (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1


class TestReduceOpsForwardCorrectness:
    """Verify numerical correctness for ReduceMean/Sum/Max/Min."""

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_reduce_single_axis_keepdims(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce along a single axis with keepdims=1."""
        g = _make_reduce_graph_attr_axes(op_type, (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch_fn(x, dim=1, keepdim=True)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_reduce_single_axis_no_keepdims(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce along a single axis with keepdims=0."""
        g = _make_reduce_graph_attr_axes(op_type, (2, 3, 4), (2, 4), axes=[1], keepdims=0)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch_fn(x, dim=1, keepdim=False)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_reduce_multiple_axes(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce along multiple axes with keepdims=1."""
        g = _make_reduce_graph_attr_axes(op_type, (2, 3, 4), (1, 1, 4), axes=[0, 1], keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch_fn(x, dim=(0, 1), keepdim=True)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_reduce_negative_axis(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce along a negative axis."""
        g = _make_reduce_graph_attr_axes(op_type, (2, 3, 4), (2, 3, 1), axes=[-1], keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch_fn(x, dim=-1, keepdim=True)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_reduce_all_dims_no_axes(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce all dimensions when no axes specified."""
        g = _make_reduce_graph_no_axes(op_type, (2, 3), (1, 1), keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3)
        (result,) = gm(x)
        # torch.mean/sum require explicit dim for keepdim; pass all dims
        expected = torch_fn(x, dim=(0, 1), keepdim=True)
        assert torch.allclose(result, expected)


class TestReduceOpsOpset18:
    """Verify axes-as-input-tensor path (opset 18 style)."""

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_input_axes_forward_correctness(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Reduce with axes provided as a second input tensor."""
        g = _make_reduce_graph_input_axes(op_type, (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch_fn(x, dim=1, keepdim=True)
        assert torch.allclose(result, expected)


class TestReduceOpsNoopEmptyAxes:
    """Verify noop_with_empty_axes=1 returns input unchanged."""

    @pytest.mark.parametrize(("op_type", "torch_fn"), _REDUCE_OPS, ids=[o[0] for o in _REDUCE_OPS])
    def test_noop_passes_through(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """With noop_with_empty_axes=1 and no axes, output must equal input."""
        g = _make_reduce_graph_noop_empty(op_type, (2, 3))
        gm = emit_graph(g)
        x = torch.randn(2, 3)
        (result,) = gm(x)
        assert torch.allclose(result, x)


# ===========================================================================
# ReduceLogSumExp
# ===========================================================================


class TestReduceLogSumExpStructure:
    """Verify that ReduceLogSumExp emits correct FX structure."""

    def test_emits_call_function(self) -> None:
        """ReduceLogSumExp must emit a call_function FX node."""
        g = _make_reduce_graph_attr_axes("ReduceLogSumExp", (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_call_function_target(self) -> None:
        """The call_function target must be torch.logsumexp."""
        g = _make_reduce_graph_attr_axes("ReduceLogSumExp", (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.logsumexp

    def test_single_output(self) -> None:
        """ReduceLogSumExp handler must return exactly one FX output node."""
        g = _make_reduce_graph_attr_axes("ReduceLogSumExp", (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1


class TestReduceLogSumExpForwardCorrectness:
    """Verify numerical correctness for ReduceLogSumExp."""

    def test_single_axis_keepdims(self) -> None:
        """ReduceLogSumExp along a single axis with keepdims=1."""
        g = _make_reduce_graph_attr_axes("ReduceLogSumExp", (2, 3, 4), (2, 1, 4), axes=[1], keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch.logsumexp(x, dim=1, keepdim=True)
        assert torch.allclose(result, expected)

    def test_single_axis_no_keepdims(self) -> None:
        """ReduceLogSumExp along a single axis with keepdims=0."""
        g = _make_reduce_graph_attr_axes("ReduceLogSumExp", (2, 3, 4), (2, 4), axes=[1], keepdims=0)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch.logsumexp(x, dim=1, keepdim=False)
        assert torch.allclose(result, expected)

    def test_multiple_axes(self) -> None:
        """ReduceLogSumExp along multiple axes."""
        g = _make_reduce_graph_attr_axes("ReduceLogSumExp", (2, 3, 4), (1, 1, 4), axes=[0, 1], keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3, 4)
        (result,) = gm(x)
        expected = torch.logsumexp(x, dim=(0, 1), keepdim=True)
        assert torch.allclose(result, expected)

    def test_reduce_all_dims(self) -> None:
        """ReduceLogSumExp over all dimensions."""
        g = _make_reduce_graph_no_axes("ReduceLogSumExp", (2, 3), (1, 1), keepdims=1)
        gm = emit_graph(g)
        x = torch.randn(2, 3)
        (result,) = gm(x)
        expected = torch.logsumexp(x, dim=(0, 1), keepdim=True)
        assert torch.allclose(result, expected)

    def test_noop_empty_axes(self) -> None:
        """ReduceLogSumExp with noop_with_empty_axes=1 passes through."""
        g = _make_reduce_graph_noop_empty("ReduceLogSumExp", (2, 3))
        gm = emit_graph(g)
        x = torch.randn(2, 3)
        (result,) = gm(x)
        assert torch.allclose(result, x)
