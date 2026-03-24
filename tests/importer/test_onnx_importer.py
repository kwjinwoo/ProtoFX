"""Tests for ONNX importer — graph inputs and initializer import."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from protofx.importers import import_model
from protofx.ir import DType, ValueKind


def _make_model(
    inputs: list[onnx.ValueInfoProto],
    *,
    outputs: list[onnx.ValueInfoProto] | None = None,
    initializers: list[onnx.TensorProto] | None = None,
    nodes: list[onnx.NodeProto] | None = None,
    opset: int = 17,
    name: str = "test",
) -> onnx.ModelProto:
    """Build a minimal ONNX ModelProto for testing."""
    graph = helper.make_graph(
        nodes or [],
        name,
        inputs,
        outputs or [],
        initializer=initializers or [],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


# ── Graph Inputs ──────────────────────────────────────────────────────


class TestImportInputs:
    """Verify that ONNX graph inputs are correctly imported into ir.Graph."""

    def test_single_input_dtype(self) -> None:
        """A single FLOAT input should map to DType.FLOAT32."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        model = _make_model([X])

        g = import_model(model)

        assert g.inputs[0].tensor_type.dtype == DType.FLOAT32

    def test_single_input_shape(self) -> None:
        """Static dimensions should roundtrip as integers in Shape."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
        model = _make_model([X])

        g = import_model(model)

        assert g.inputs[0].tensor_type.shape == (1, 3, 224, 224)

    def test_single_input_name(self) -> None:
        """ONNX input name should be preserved on the Value."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        model = _make_model([X])

        g = import_model(model)

        assert g.inputs[0].name == "X"

    def test_single_input_kind(self) -> None:
        """Imported graph inputs must have GRAPH_INPUT kind."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        model = _make_model([X])

        g = import_model(model)

        assert g.inputs[0].kind == ValueKind.GRAPH_INPUT

    def test_multiple_inputs(self) -> None:
        """Multiple inputs should appear in order with correct dtypes."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [2])
        model = _make_model([X, Y])

        g = import_model(model)

        assert len(g.inputs) == 2
        assert g.inputs[0].tensor_type.dtype == DType.FLOAT32
        assert g.inputs[1].tensor_type.dtype == DType.INT64
        assert g.inputs[0].name == "X"
        assert g.inputs[1].name == "Y"

    def test_dynamic_dim_is_none(self) -> None:
        """Unknown dimensions in ONNX should map to None in Shape."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 3])
        model = _make_model([X])

        g = import_model(model)

        shape = g.inputs[0].tensor_type.shape
        assert shape is not None
        assert shape[0] is None
        assert shape[1] == 3

    def test_symbolic_dim_is_string(self) -> None:
        """Symbolic dim_param values should map to str in Shape."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])
        model = _make_model([X])

        g = import_model(model)

        shape = g.inputs[0].tensor_type.shape
        assert shape is not None
        assert shape[0] == "batch"
        assert shape[1] == 3

    def test_scalar_input_shape(self) -> None:
        """A scalar ONNX input (rank 0) should produce an empty shape tuple."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [])
        model = _make_model([X])

        g = import_model(model)

        assert g.inputs[0].tensor_type.shape == ()

    def test_input_producer_is_none(self) -> None:
        """Graph inputs must have no producer."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        model = _make_model([X])

        g = import_model(model)

        assert g.inputs[0].producer is None


# ── Initializers ──────────────────────────────────────────────────────


class TestImportInitializers:
    """Verify that ONNX initializers are correctly imported."""

    def test_initializer_count(self) -> None:
        """One ONNX initializer should produce one ir.Value in graph.initializers."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        W = onnx.numpy_helper.from_array(np.ones((3, 4), dtype=np.float32), name="W")
        model = _make_model([X], initializers=[W])

        g = import_model(model)

        assert len(g.initializers) == 1

    def test_initializer_kind(self) -> None:
        """Imported initializers must have INITIALIZER kind."""
        W = onnx.numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="W")
        model = _make_model([], initializers=[W])

        g = import_model(model)

        assert g.initializers[0].kind == ValueKind.INITIALIZER

    def test_initializer_dtype(self) -> None:
        """Initializer element type should map to the correct IR DType."""
        W = onnx.numpy_helper.from_array(np.zeros((2, 3), dtype=np.int64), name="W")
        model = _make_model([], initializers=[W])

        g = import_model(model)

        assert g.initializers[0].tensor_type.dtype == DType.INT64

    def test_initializer_shape(self) -> None:
        """Initializer shape should match the numpy array shape."""
        W = onnx.numpy_helper.from_array(np.zeros((5, 10), dtype=np.float32), name="W")
        model = _make_model([], initializers=[W])

        g = import_model(model)

        assert g.initializers[0].tensor_type.shape == (5, 10)

    def test_initializer_name(self) -> None:
        """ONNX initializer name should be preserved on the Value."""
        W = onnx.numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="bias")
        model = _make_model([], initializers=[W])

        g = import_model(model)

        assert g.initializers[0].name == "bias"

    def test_initializer_data(self) -> None:
        """Initializer numpy data should be preserved on the Value."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        W = onnx.numpy_helper.from_array(data, name="W")
        model = _make_model([], initializers=[W])

        g = import_model(model)

        np.testing.assert_array_equal(g.initializers[0].data, data)

    def test_initializer_producer_is_none(self) -> None:
        """Initializers must have no producer."""
        W = onnx.numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="W")
        model = _make_model([], initializers=[W])

        g = import_model(model)

        assert g.initializers[0].producer is None


# ── Initializer / Input Deduplication ─────────────────────────────────


class TestInitializerInputDedup:
    """Verify initializer names overlapping with graph.input are filtered."""

    def test_initializer_not_duplicated_as_input(self) -> None:
        """When an initializer name appears in graph.input, only keep it as an initializer."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        W_vi = helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 4])
        W_data = np.zeros((3, 4), dtype=np.float32)
        W = onnx.numpy_helper.from_array(W_data, name="W")
        # W appears in both input and initializer (opset < 9 pattern)
        model = _make_model([X, W_vi], initializers=[W])

        g = import_model(model)

        # Only X is a graph input; W is an initializer
        assert len(g.inputs) == 1
        assert g.inputs[0].name == "X"
        assert len(g.initializers) == 1
        assert g.initializers[0].name == "W"

    def test_multiple_initializers_dedup(self) -> None:
        """Multiple initializers overlapping with inputs should all be filtered."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        W_vi = helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 4])
        B_vi = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4])
        W_data = np.zeros((3, 4), dtype=np.float32)
        B_data = np.zeros((4,), dtype=np.float32)
        W = onnx.numpy_helper.from_array(W_data, name="W")
        B = onnx.numpy_helper.from_array(B_data, name="B")
        model = _make_model([X, W_vi, B_vi], initializers=[W, B])

        g = import_model(model)

        assert len(g.inputs) == 1
        assert g.inputs[0].name == "X"
        assert len(g.initializers) == 2
        names = {v.name for v in g.initializers}
        assert names == {"W", "B"}
