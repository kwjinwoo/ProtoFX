"""ORT numerical parity tests for reduction op handlers.

Covers: ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceLogSumExp,
ReduceProd, ReduceL1, ReduceL2, ReduceLogSum, ReduceSumSquare.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.parity.conftest import assert_parity

_OPSET = helper.make_opsetid("", 18)


def _make_reduce_model_opset18(
    op_type: str,
    input_shape: list[int],
    output_shape: list[int],
    axes: list[int] | None = None,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> helper.ModelProto:
    """Build a minimal ONNX reduce model (opset 18 style: axes as input tensor)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)

    inputs = ["X"]
    initializers = []
    if axes is not None:
        axes_arr = np.array(axes, dtype=np.int64)
        axes_init = numpy_helper.from_array(axes_arr, name="axes")
        initializers.append(axes_init)
        inputs.append("axes")

    node = helper.make_node(
        op_type,
        inputs,
        ["Y"],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )
    graph = helper.make_graph([node], f"{op_type}_test", [X], [Y], initializer=initializers)
    return helper.make_model(graph, opset_imports=[_OPSET])


# ---------------------------------------------------------------------------
# Simple reductions: ReduceMean, ReduceSum, ReduceMax, ReduceMin
# ---------------------------------------------------------------------------

_SIMPLE_REDUCE_OPS = ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"]


class TestSimpleReduceParity:
    """ORT parity for simple reduction ops."""

    @pytest.mark.parametrize("op_type", _SIMPLE_REDUCE_OPS)
    def test_single_axis_keepdims(self, op_type: str) -> None:
        """Reduce along a single axis with keepdims=1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4, 5)).astype(np.float32)
        model = _make_reduce_model_opset18(op_type, [3, 4, 5], [3, 1, 5], axes=[1], keepdims=1)
        assert_parity(model, {"X": x})

    @pytest.mark.parametrize("op_type", _SIMPLE_REDUCE_OPS)
    def test_single_axis_no_keepdims(self, op_type: str) -> None:
        """Reduce along a single axis with keepdims=0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4, 5)).astype(np.float32)
        model = _make_reduce_model_opset18(op_type, [3, 4, 5], [3, 5], axes=[1], keepdims=0)
        assert_parity(model, {"X": x})

    @pytest.mark.parametrize("op_type", _SIMPLE_REDUCE_OPS)
    def test_multi_axis(self, op_type: str) -> None:
        """Reduce along multiple axes."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4, 5)).astype(np.float32)
        model = _make_reduce_model_opset18(op_type, [3, 4, 5], [1, 1, 5], axes=[0, 1], keepdims=1)
        assert_parity(model, {"X": x})

    @pytest.mark.parametrize("op_type", _SIMPLE_REDUCE_OPS)
    def test_all_axes(self, op_type: str) -> None:
        """Reduce along all axes (no explicit axes, noop_with_empty_axes=0)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        model = _make_reduce_model_opset18(op_type, [3, 4], [1, 1], keepdims=1, noop_with_empty_axes=0)
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# ReduceLogSumExp
# ---------------------------------------------------------------------------


class TestReduceLogSumExpParity:
    """ORT parity for ReduceLogSumExp."""

    def test_parity(self) -> None:
        """ReduceLogSumExp must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        model = _make_reduce_model_opset18("ReduceLogSumExp", [3, 4], [1, 4], axes=[0], keepdims=1)
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# ReduceProd
# ---------------------------------------------------------------------------


class TestReduceProdParity:
    """ORT parity for ReduceProd."""

    def test_single_axis(self) -> None:
        """ReduceProd along a single axis must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        model = _make_reduce_model_opset18("ReduceProd", [3, 4], [1, 4], axes=[0], keepdims=1)
        assert_parity(model, {"X": x})

    def test_multi_axis(self) -> None:
        """ReduceProd along multiple axes must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 3, 4)).astype(np.float32)
        model = _make_reduce_model_opset18("ReduceProd", [2, 3, 4], [1, 1, 4], axes=[0, 1], keepdims=1)
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Compound reductions: ReduceL1, ReduceL2, ReduceLogSum, ReduceSumSquare
# ---------------------------------------------------------------------------


class TestReduceL1Parity:
    """ORT parity for ReduceL1."""

    def test_parity(self) -> None:
        """ReduceL1 must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        model = _make_reduce_model_opset18("ReduceL1", [3, 4], [1, 4], axes=[0], keepdims=1)
        assert_parity(model, {"X": x})


class TestReduceL2Parity:
    """ORT parity for ReduceL2."""

    def test_parity(self) -> None:
        """ReduceL2 must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        model = _make_reduce_model_opset18("ReduceL2", [3, 4], [1, 4], axes=[0], keepdims=1)
        assert_parity(model, {"X": x})


class TestReduceLogSumParity:
    """ORT parity for ReduceLogSum."""

    def test_parity(self) -> None:
        """ReduceLogSum must match ORT (positive inputs to keep log valid)."""
        rng = np.random.default_rng(42)
        x = (np.abs(rng.standard_normal((3, 4))) + 0.01).astype(np.float32)
        model = _make_reduce_model_opset18("ReduceLogSum", [3, 4], [1, 4], axes=[0], keepdims=1)
        assert_parity(model, {"X": x})


class TestReduceSumSquareParity:
    """ORT parity for ReduceSumSquare."""

    def test_parity(self) -> None:
        """ReduceSumSquare must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        model = _make_reduce_model_opset18("ReduceSumSquare", [3, 4], [1, 4], axes=[0], keepdims=1)
        assert_parity(model, {"X": x})
