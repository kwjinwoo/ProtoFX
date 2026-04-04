"""ORT numerical parity tests for tensor manipulation op handlers.

Covers: Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Slice,
Identity, Cast, Expand, Gather.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.parity.conftest import assert_parity

_OPSET = helper.make_opsetid("", 17)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


class TestReshapeParity:
    """ORT parity for Reshape op."""

    def test_parity(self) -> None:
        """Reshape must match ORT output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 3, 4)).astype(np.float32)
        shape = np.array([2, 12], dtype=np.int64)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 12])
        shape_init = numpy_helper.from_array(shape, name="shape")
        node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
        graph = helper.make_graph([node], "reshape_test", [X], [Y], initializer=[shape_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestTransposeParity:
    """ORT parity for Transpose op."""

    @pytest.mark.parametrize("perm", [[1, 0, 2], [2, 0, 1]])
    def test_parity(self, perm: list[int]) -> None:
        """Transpose must match ORT at different permutations."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 3, 4)).astype(np.float32)
        out_shape = [x.shape[p] for p in perm]

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
        node = helper.make_node("Transpose", ["X"], ["Y"], perm=perm)
        graph = helper.make_graph([node], "transpose_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


class TestFlattenParity:
    """ORT parity for Flatten op."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_parity(self, axis: int) -> None:
        """Flatten must match ORT at different axis values."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 3, 4)).astype(np.float32)
        # Compute expected flat shape
        shape = x.shape
        d0 = int(np.prod(shape[:axis])) if axis > 0 else 1
        d1 = int(np.prod(shape[axis:]))

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [d0, d1])
        node = helper.make_node("Flatten", ["X"], ["Y"], axis=axis)
        graph = helper.make_graph([node], "flatten_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Squeeze
# ---------------------------------------------------------------------------


class TestSqueezeParity:
    """ORT parity for Squeeze op."""

    def test_squeeze_specific_axes(self) -> None:
        """Squeeze with specific axes must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, 3, 1, 4)).astype(np.float32)
        axes = np.array([0, 2], dtype=np.int64)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
        axes_init = numpy_helper.from_array(axes, name="axes")
        node = helper.make_node("Squeeze", ["X", "axes"], ["Y"])
        graph = helper.make_graph([node], "squeeze_test", [X], [Y], initializer=[axes_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_squeeze_no_axes(self) -> None:
        """Squeeze without axes must remove all dim-1 axes like ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, 3, 1)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
        node = helper.make_node("Squeeze", ["X"], ["Y"])
        graph = helper.make_graph([node], "squeeze_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Unsqueeze
# ---------------------------------------------------------------------------


class TestUnsqueezeParity:
    """ORT parity for Unsqueeze op."""

    def test_parity(self) -> None:
        """Unsqueeze must match ORT output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        axes = np.array([0, 3], dtype=np.int64)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 1])
        axes_init = numpy_helper.from_array(axes, name="axes")
        node = helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])
        graph = helper.make_graph([node], "unsqueeze_test", [X], [Y], initializer=[axes_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


class TestConcatParity:
    """ORT parity for Concat op."""

    @pytest.mark.parametrize("axis", [0, 1])
    def test_parity(self, axis: int) -> None:
        """Concat must match ORT along different axes."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((2, 3)).astype(np.float32)
        b = rng.standard_normal((2, 3)).astype(np.float32)
        out_shape = list(a.shape)
        out_shape[axis] *= 2

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
        node = helper.make_node("Concat", ["A", "B"], ["Y"], axis=axis)
        graph = helper.make_graph([node], "concat_test", [A, B], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


class TestSliceParity:
    """ORT parity for Slice op."""

    def test_basic_slice(self) -> None:
        """Slice with starts/ends must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((4, 6)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 6])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        starts_init = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="starts")
        ends_init = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="ends")
        node = helper.make_node("Slice", ["X", "starts", "ends"], ["Y"])
        graph = helper.make_graph([node], "slice_test", [X], [Y], initializer=[starts_init, ends_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_slice_with_axes_and_steps(self) -> None:
        """Slice with axes and steps must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((6, 8)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [6, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
        starts = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="starts")
        ends = numpy_helper.from_array(np.array([6, 8], dtype=np.int64), name="ends")
        axes = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="axes")
        steps = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="steps")
        node = helper.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"])
        graph = helper.make_graph([node], "slice_test", [X], [Y], initializer=[starts, ends, axes, steps])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestIdentityParity:
    """ORT parity for Identity op."""

    def test_parity(self) -> None:
        """Identity must pass through data exactly like ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 3)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "identity_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------


class TestCastParity:
    """ORT parity for Cast op."""

    def test_float_to_double(self) -> None:
        """Cast float32 → float64 must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 3)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [2, 3])
        node = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.DOUBLE)
        graph = helper.make_graph([node], "cast_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_float_to_int32(self) -> None:
        """Cast float32 → int32 must match ORT."""
        rng = np.random.default_rng(42)
        x = (rng.standard_normal((2, 3)) * 10).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [2, 3])
        node = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.INT32)
        graph = helper.make_graph([node], "cast_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Expand
# ---------------------------------------------------------------------------


class TestExpandParity:
    """ORT parity for Expand op."""

    def test_parity(self) -> None:
        """Expand must match ORT output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, 3)).astype(np.float32)
        target_shape = np.array([4, 3], dtype=np.int64)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3])
        shape_init = numpy_helper.from_array(target_shape, name="shape")
        node = helper.make_node("Expand", ["X", "shape"], ["Y"])
        graph = helper.make_graph([node], "expand_test", [X], [Y], initializer=[shape_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


class TestGatherParity:
    """ORT parity for Gather op."""

    def test_1d_indices(self) -> None:
        """Gather with 1D indices must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((5, 3)).astype(np.float32)
        indices = np.array([0, 2, 4], dtype=np.int64)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])
        idx_init = numpy_helper.from_array(indices, name="indices")
        node = helper.make_node("Gather", ["X", "indices"], ["Y"], axis=0)
        graph = helper.make_graph([node], "gather_test", [X], [Y], initializer=[idx_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_scalar_index(self) -> None:
        """Gather with scalar index must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((5, 3)).astype(np.float32)
        indices = np.array(1, dtype=np.int64)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
        idx_init = numpy_helper.from_array(indices, name="indices")
        node = helper.make_node("Gather", ["X", "indices"], ["Y"], axis=0)
        graph = helper.make_graph([node], "gather_test", [X], [Y], initializer=[idx_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})
