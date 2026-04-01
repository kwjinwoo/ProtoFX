"""Tests for ONNX importer — graph inputs and initializer import."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from protofx.importers import import_model
from protofx.importers._onnx import _normalize_attribute
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


# ── Attribute Normalization ───────────────────────────────────────────


class TestNormalizeAttributeInt:
    """Verify INT attribute normalization."""

    def test_int_value(self) -> None:
        """ONNX INT attribute should produce a Python int."""
        attr = helper.make_attribute("axis", 1)
        assert _normalize_attribute(attr) == 1

    def test_int_type(self) -> None:
        """Result must be a plain int."""
        attr = helper.make_attribute("axis", 2)
        assert isinstance(_normalize_attribute(attr), int)

    def test_negative_int(self) -> None:
        """Negative ints should be preserved."""
        attr = helper.make_attribute("axis", -1)
        assert _normalize_attribute(attr) == -1


class TestNormalizeAttributeFloat:
    """Verify FLOAT attribute normalization."""

    def test_float_value(self) -> None:
        """ONNX FLOAT attribute should produce a Python float."""
        attr = helper.make_attribute("alpha", 0.5)
        assert _normalize_attribute(attr) == 0.5

    def test_float_type(self) -> None:
        """Result must be a plain float."""
        attr = helper.make_attribute("alpha", 3.14)
        assert isinstance(_normalize_attribute(attr), float)


class TestNormalizeAttributeString:
    """Verify STRING attribute normalization to bytes."""

    def test_string_value(self) -> None:
        """ONNX STRING attribute should produce Python bytes."""
        attr = helper.make_attribute("mode", b"constant")
        assert _normalize_attribute(attr) == b"constant"

    def test_string_type(self) -> None:
        """Result must be bytes."""
        attr = helper.make_attribute("mode", b"reflect")
        assert isinstance(_normalize_attribute(attr), bytes)

    def test_binary_payload(self) -> None:
        """Raw binary content should be preserved as bytes."""
        attr = helper.make_attribute("raw", b"\x00\x01\xff")
        assert _normalize_attribute(attr) == b"\x00\x01\xff"


class TestNormalizeAttributeInts:
    """Verify INTS attribute normalization."""

    def test_ints_value(self) -> None:
        """ONNX INTS attribute should produce a Python list[int]."""
        attr = helper.make_attribute("pads", [1, 2, 3, 4])
        assert _normalize_attribute(attr) == [1, 2, 3, 4]

    def test_ints_type(self) -> None:
        """Each element must be int."""
        attr = helper.make_attribute("pads", [0, 1])
        result = _normalize_attribute(attr)
        assert isinstance(result, list)
        assert all(isinstance(v, int) for v in result)


class TestNormalizeAttributeFloats:
    """Verify FLOATS attribute normalization."""

    def test_floats_value(self) -> None:
        """ONNX FLOATS attribute should produce a Python list[float]."""
        attr = helper.make_attribute("scales", [1.0, 2.0, 3.0])
        assert _normalize_attribute(attr) == [1.0, 2.0, 3.0]

    def test_floats_type(self) -> None:
        """Each element must be float."""
        attr = helper.make_attribute("scales", [0.5, 1.5])
        result = _normalize_attribute(attr)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)


class TestNormalizeAttributeStrings:
    """Verify STRINGS attribute normalization to list[bytes]."""

    def test_strings_value(self) -> None:
        """ONNX STRINGS attribute should produce a Python list[bytes]."""
        attr = helper.make_attribute("names", [b"alpha", b"beta"])
        assert _normalize_attribute(attr) == [b"alpha", b"beta"]

    def test_strings_type(self) -> None:
        """Each element must be bytes."""
        attr = helper.make_attribute("names", [b"x", b"y"])
        result = _normalize_attribute(attr)
        assert isinstance(result, list)
        assert all(isinstance(v, bytes) for v in result)


class TestNormalizeAttributeUnsupported:
    """Verify unsupported attribute types raise NotImplementedError."""

    def test_tensor_raises(self) -> None:
        """TENSOR attributes are not supported in this slice."""
        tensor = onnx.numpy_helper.from_array(np.zeros((2,), dtype=np.float32))
        attr = helper.make_attribute("value", tensor)
        try:
            _normalize_attribute(attr)
            raise AssertionError("expected NotImplementedError")
        except NotImplementedError:
            pass


# ── Node Import and Output Type Resolution ────────────────────────────


class TestImportNodes:
    """Verify that ONNX nodes are correctly imported into ir.Graph."""

    def test_single_relu_op_type(self) -> None:
        """A single Relu node should have op_type 'Relu'."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert len(g.nodes) == 1
        assert g.nodes[0].op_type == "Relu"

    def test_single_relu_inputs_connected(self) -> None:
        """Relu node inputs should reference the graph input Value."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        node = g.nodes[0]
        assert len(node.inputs) == 1
        assert node.inputs[0] is g.inputs[0]

    def test_single_relu_outputs_count(self) -> None:
        """Relu should produce one output Value."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert len(g.nodes[0].outputs) == 1

    def test_node_output_name_preserved(self) -> None:
        """ONNX output name should be preserved on the Value."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert g.nodes[0].outputs[0].name == "Y"

    def test_two_node_chain(self) -> None:
        """Two chained nodes should share the intermediate Value."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        sigmoid = helper.make_node("Sigmoid", ["Y"], ["Z"])
        model = _make_model([X], outputs=[Z], nodes=[relu, sigmoid])

        g = import_model(model)

        assert len(g.nodes) == 2
        # sigmoid's input should be relu's output
        assert g.nodes[1].inputs[0] is g.nodes[0].outputs[0]

    def test_node_with_initializer_input(self) -> None:
        """Nodes should be able to reference initializer Values as inputs."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        W_data = np.ones((3, 4), dtype=np.float32)
        W = onnx.numpy_helper.from_array(W_data, name="W")
        matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])
        model = _make_model([X], outputs=[Y], initializers=[W], nodes=[matmul])

        g = import_model(model)

        node = g.nodes[0]
        assert len(node.inputs) == 2
        assert node.inputs[0] is g.inputs[0]
        assert node.inputs[1] is g.initializers[0]

    def test_node_attributes_normalized(self) -> None:
        """Node attributes should be Python-native values."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])
        pad = helper.make_node("Pad", ["X"], ["Y"], mode="constant")
        model = _make_model([X], outputs=[Y], nodes=[pad])

        g = import_model(model)

        assert "mode" in g.nodes[0].attributes
        assert g.nodes[0].attributes["mode"] == b"constant"

    def test_node_domain_default(self) -> None:
        """Default domain should be empty string."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert g.nodes[0].domain == ""

    def test_node_onnx_name_preserved(self) -> None:
        """ONNX node name should be preserved."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"], name="my_relu")
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert g.nodes[0].name == "my_relu"


class TestOutputTypeResolution:
    """Verify output type resolution from value_info and graph outputs."""

    def test_output_dtype_from_value_info(self) -> None:
        """Output dtype should come from ONNX value_info when available."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert g.nodes[0].outputs[0].tensor_type.dtype == DType.FLOAT32

    def test_output_shape_from_value_info(self) -> None:
        """Output shape should come from ONNX value_info when available."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert g.nodes[0].outputs[0].tensor_type.shape == (2, 3)

    def test_unknown_output_type_when_no_info(self) -> None:
        """Outputs without value_info are filled by shape inference when possible."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        # Intermediate value "Y" has no value_info, only graph output "Z" does
        Y_out = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        sigmoid = helper.make_node("Sigmoid", ["Y"], ["Z"])
        model = _make_model([X], outputs=[Y_out], nodes=[relu, sigmoid])

        g = import_model(model)

        # Shape inference fills intermediate "Y" with inferred type
        relu_out = g.nodes[0].outputs[0]
        assert relu_out.tensor_type.dtype == DType.FLOAT32
        assert relu_out.tensor_type.shape == (2, 3)

    def test_graph_outputs_set(self) -> None:
        """Graph outputs should be set after node import."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        assert len(g.outputs) == 1
        assert g.outputs[0] is g.nodes[0].outputs[0]

    def test_empty_input_becomes_sentinel(self) -> None:
        """An empty input name in ONNX should become a SENTINEL value."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        # Pad has optional "constant_value" input; use "" to omit it
        pad_node = helper.make_node("Pad", ["X", "", ""], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[pad_node])

        g = import_model(model)

        node = g.nodes[0]
        assert len(node.inputs) == 3
        assert node.inputs[0] is g.inputs[0]
        assert node.inputs[1].kind == ValueKind.SENTINEL
        assert node.inputs[2].kind == ValueKind.SENTINEL


# ── Constant Op Inlining ─────────────────────────────────────────────


class TestConstantOpInlining:
    """Verify that ONNX Constant ops are inlined as CONSTANT values."""

    def test_constant_op_not_in_nodes(self) -> None:
        """Constant ops should not appear as ir.Node in the graph."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        const_tensor = onnx.numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="C")
        const_node = helper.make_node("Constant", [], ["C"], value=const_tensor)
        add_node = helper.make_node("Add", ["X", "C"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[const_node, add_node])

        g = import_model(model)

        # Only the Add node should be in the graph, not Constant
        assert len(g.nodes) == 1
        assert g.nodes[0].op_type == "Add"

    def test_constant_value_kind(self) -> None:
        """Inlined constant should have CONSTANT kind."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        const_tensor = onnx.numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="C")
        const_node = helper.make_node("Constant", [], ["C"], value=const_tensor)
        add_node = helper.make_node("Add", ["X", "C"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[const_node, add_node])

        g = import_model(model)

        # Add's second input should be CONSTANT
        const_input = g.nodes[0].inputs[1]
        assert const_input.kind == ValueKind.CONSTANT

    def test_constant_data_preserved(self) -> None:
        """Inlined constant should carry the numpy data."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        expected = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        const_tensor = onnx.numpy_helper.from_array(expected, name="C")
        const_node = helper.make_node("Constant", [], ["C"], value=const_tensor)
        add_node = helper.make_node("Add", ["X", "C"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[const_node, add_node])

        g = import_model(model)

        const_input = g.nodes[0].inputs[1]
        np.testing.assert_array_equal(const_input.data, expected)

    def test_constant_producer_is_none(self) -> None:
        """Inlined constant should have no producer."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        const_tensor = onnx.numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="C")
        const_node = helper.make_node("Constant", [], ["C"], value=const_tensor)
        add_node = helper.make_node("Add", ["X", "C"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[const_node, add_node])

        g = import_model(model)

        const_input = g.nodes[0].inputs[1]
        assert const_input.producer is None

    def test_constant_dtype(self) -> None:
        """Inlined constant should have correct dtype."""
        X = helper.make_tensor_value_info("X", TensorProto.INT64, [2])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [2])
        const_tensor = onnx.numpy_helper.from_array(np.array([5, 10], dtype=np.int64), name="C")
        const_node = helper.make_node("Constant", [], ["C"], value=const_tensor)
        add_node = helper.make_node("Add", ["X", "C"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[const_node, add_node])

        g = import_model(model)

        const_input = g.nodes[0].inputs[1]
        assert const_input.tensor_type.dtype == DType.INT64

    def test_constant_shape(self) -> None:
        """Inlined constant should have correct shape."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        const_tensor = onnx.numpy_helper.from_array(np.zeros((2, 3), dtype=np.float32), name="C")
        const_node = helper.make_node("Constant", [], ["C"], value=const_tensor)
        add_node = helper.make_node("Add", ["X", "C"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[const_node, add_node])

        g = import_model(model)

        const_input = g.nodes[0].inputs[1]
        assert const_input.tensor_type.shape == (2, 3)


# ── import_model Entry Point ─────────────────────────────────────────


class TestImportModelEntryPoint:
    """Verify import_model top-level behavior: name, opset, shape inference, validate."""

    def test_graph_name_preserved(self) -> None:
        """Graph name should come from the ONNX ModelProto graph name."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu], name="my_graph")

        g = import_model(model)

        assert g.name == "my_graph"

    def test_opset_version_on_nodes(self) -> None:
        """Nodes should carry the default domain opset version from the model."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu], opset=17)

        g = import_model(model)

        assert g.nodes[0].opset_version == 17

    def test_validate_passes_after_import(self) -> None:
        """A well-formed imported graph should pass ir.Graph.validate()."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        relu = helper.make_node("Relu", ["X"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[relu])

        g = import_model(model)

        g.validate()  # should not raise

    def test_validate_with_initializers(self) -> None:
        """Imported graph with initializers should also pass validate()."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        W_data = np.ones((3, 4), dtype=np.float32)
        W = onnx.numpy_helper.from_array(W_data, name="W")
        matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])
        model = _make_model([X], outputs=[Y], initializers=[W], nodes=[matmul])

        g = import_model(model)

        g.validate()  # should not raise

    def test_validate_with_constant_inlining(self) -> None:
        """Graph with inlined constants should pass validate()."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        const_tensor = onnx.numpy_helper.from_array(np.ones((2, 3), dtype=np.float32), name="C")
        const_node = helper.make_node("Constant", [], ["C"], value=const_tensor)
        add_node = helper.make_node("Add", ["X", "C"], ["Y"])
        model = _make_model([X], outputs=[Y], nodes=[const_node, add_node])

        g = import_model(model)

        g.validate()  # should not raise

    def test_shape_inference_fills_intermediate_types(self) -> None:
        """ONNX shape inference should fill missing intermediate value_info."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)
        # Two-node chain: Relu(X)->Y, Sigmoid(Y)->Z
        # Only X and Z have explicit type info; Y should be inferred
        relu = helper.make_node("Relu", ["X"], ["Y"])
        sigmoid = helper.make_node("Sigmoid", ["Y"], ["Z"])
        model = _make_model([X], outputs=[Z], nodes=[relu, sigmoid])

        g = import_model(model)

        # After shape inference, intermediate "Y" should have dtype/shape filled
        relu_out = g.nodes[0].outputs[0]
        assert relu_out.tensor_type.dtype == DType.FLOAT32
        assert relu_out.tensor_type.shape == (2, 3)

    def test_shape_inference_fallback_to_none(self) -> None:
        """When shape inference cannot determine type, dtype/shape should be None."""
        # Build a model where an intermediate value cannot be inferred
        # Using a custom domain op that shape inference doesn't know about
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        custom_node = helper.make_node("MyCustomOp", ["X"], ["Y"], domain="custom.domain")
        relu = helper.make_node("Relu", ["Y"], ["Z"])
        graph = helper.make_graph(
            [custom_node, relu],
            "test",
            [X],
            [Z],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 17),
                helper.make_opsetid("custom.domain", 1),
            ],
        )

        g = import_model(model)

        # MyCustomOp output "Y" is not inferable => None
        custom_out = g.nodes[0].outputs[0]
        assert custom_out.tensor_type.dtype is None
        assert custom_out.tensor_type.shape is None


# ── Validate Contract ─────────────────────────────────────────────────


class TestImportModelValidateContract:
    """Verify import_model calls graph.validate() before returning (contracts.md fail-fast)."""

    def test_validate_is_called(self) -> None:
        """import_model must call graph.validate() before returning."""
        from unittest.mock import patch

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        model = _make_model([X])

        with patch("protofx.ir.graph.Graph.validate") as mock_validate:
            import_model(model)
            mock_validate.assert_called_once()

    def test_validate_error_propagates(self) -> None:
        """If graph.validate() raises ValueError, import_model must propagate it."""
        from unittest.mock import patch

        import pytest

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        model = _make_model([X])

        with patch("protofx.ir.graph.Graph.validate", side_effect=ValueError("invariant violated")):
            with pytest.raises(ValueError, match="invariant violated"):
                import_model(model)


# ── auto_pad normalization ────────────────────────────────────────────


class TestAutoPadNormalization:
    """Verify that Conv/ConvTranspose auto_pad is normalized to explicit pads during import."""

    def _make_conv_model(
        self,
        *,
        auto_pad: str,
        x_shape: list[int],
        w_shape: list[int],
        y_shape: list[int],
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        op_type: str = "Conv",
    ) -> onnx.ModelProto:
        """Build an ONNX model with a Conv/ConvTranspose node using auto_pad."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)
        W_init = onnx.numpy_helper.from_array(np.zeros(w_shape, dtype=np.float32), name="W")
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y_shape)

        attrs: dict[str, object] = {"auto_pad": auto_pad}
        if strides is not None:
            attrs["strides"] = strides
        if dilations is not None:
            attrs["dilations"] = dilations

        conv_node = helper.make_node(
            op_type,
            inputs=["X", "W"],
            outputs=["Y"],
            **attrs,
        )
        return _make_model([X], outputs=[Y], initializers=[W_init], nodes=[conv_node])

    def test_valid_produces_zero_pads(self) -> None:
        """auto_pad=VALID must be normalized to all-zero pads."""
        model = self._make_conv_model(
            auto_pad="VALID",
            x_shape=[1, 1, 5, 5],
            w_shape=[1, 1, 3, 3],
            y_shape=[1, 1, 3, 3],
        )
        g = import_model(model)
        conv_node = next(n for n in g.nodes if n.op_type == "Conv")
        assert conv_node.attributes.get("pads") == [0, 0, 0, 0]
        assert conv_node.attributes.get("auto_pad", b"NOTSET") == b"NOTSET"

    def test_same_upper_produces_explicit_pads(self) -> None:
        """auto_pad=SAME_UPPER must be normalized to explicit symmetric-ish pads."""
        # X: (1,1,5,5), W: (1,1,3,3), strides=[1,1] -> SAME_UPPER -> Y: (1,1,5,5)
        model = self._make_conv_model(
            auto_pad="SAME_UPPER",
            x_shape=[1, 1, 5, 5],
            w_shape=[1, 1, 3, 3],
            y_shape=[1, 1, 5, 5],
            strides=[1, 1],
        )
        g = import_model(model)
        conv_node = next(n for n in g.nodes if n.op_type == "Conv")
        # For 5x5 input, 3x3 kernel, stride 1: total_pad = 2 per axis
        # SAME_UPPER: pad_begin = total_pad // 2 = 1, pad_end = total_pad - pad_begin = 1
        assert conv_node.attributes.get("pads") == [1, 1, 1, 1]
        assert conv_node.attributes.get("auto_pad", b"NOTSET") == b"NOTSET"

    def test_same_lower_produces_explicit_pads(self) -> None:
        """auto_pad=SAME_LOWER must be normalized to explicit pads (more at begin)."""
        # X: (1,1,4,4), W: (1,1,3,3), strides=[1,1] -> SAME_LOWER -> Y: (1,1,4,4)
        model = self._make_conv_model(
            auto_pad="SAME_LOWER",
            x_shape=[1, 1, 4, 4],
            w_shape=[1, 1, 3, 3],
            y_shape=[1, 1, 4, 4],
            strides=[1, 1],
        )
        g = import_model(model)
        conv_node = next(n for n in g.nodes if n.op_type == "Conv")
        # For 4x4 input, 3x3 kernel, stride 1: total_pad = 2 per axis
        # SAME_LOWER: pad_end = total_pad // 2 = 1, pad_begin = total_pad - pad_end = 1
        assert conv_node.attributes.get("pads") == [1, 1, 1, 1]
        assert conv_node.attributes.get("auto_pad", b"NOTSET") == b"NOTSET"

    def test_same_upper_asymmetric(self) -> None:
        """auto_pad=SAME_UPPER with even input and stride=2 produces asymmetric pads."""
        # X: (1,1,6,6), W: (1,1,3,3), strides=[2,2] -> SAME_UPPER -> Y: (1,1,3,3)
        # effective_kernel = 3, output_size = ceil(6/2) = 3
        # total_pad = max(0, (3-1)*2 + 3 - 6) = max(0, 1) = 1
        # SAME_UPPER: pad_begin = 0, pad_end = 1
        model = self._make_conv_model(
            auto_pad="SAME_UPPER",
            x_shape=[1, 1, 6, 6],
            w_shape=[1, 1, 3, 3],
            y_shape=[1, 1, 3, 3],
            strides=[2, 2],
        )
        g = import_model(model)
        conv_node = next(n for n in g.nodes if n.op_type == "Conv")
        assert conv_node.attributes.get("pads") == [0, 0, 1, 1]
        assert conv_node.attributes.get("auto_pad", b"NOTSET") == b"NOTSET"

    def test_same_upper_with_dilation(self) -> None:
        """auto_pad=SAME_UPPER with dilation must account for effective kernel size."""
        # X: (1,1,7,7), W: (1,1,3,3), strides=[1,1], dilation=[2,2]
        # effective_kernel = 3 + (3-1)*(2-1) = 5
        # output_size = ceil(7/1) = 7
        # total_pad = max(0, (7-1)*1 + 5 - 7) = max(0, 4) = 4
        # SAME_UPPER: pad_begin = 2, pad_end = 2
        model = self._make_conv_model(
            auto_pad="SAME_UPPER",
            x_shape=[1, 1, 7, 7],
            w_shape=[1, 1, 3, 3],
            y_shape=[1, 1, 7, 7],
            strides=[1, 1],
            dilations=[2, 2],
        )
        g = import_model(model)
        conv_node = next(n for n in g.nodes if n.op_type == "Conv")
        assert conv_node.attributes.get("pads") == [2, 2, 2, 2]
        assert conv_node.attributes.get("auto_pad", b"NOTSET") == b"NOTSET"

    def test_conv_transpose_valid(self) -> None:
        """auto_pad=VALID on ConvTranspose must be normalized to all-zero pads."""
        model = self._make_conv_model(
            auto_pad="VALID",
            x_shape=[1, 1, 3, 3],
            w_shape=[1, 1, 3, 3],
            y_shape=[1, 1, 5, 5],
            op_type="ConvTranspose",
        )
        g = import_model(model)
        conv_node = next(n for n in g.nodes if n.op_type == "ConvTranspose")
        assert conv_node.attributes.get("pads") == [0, 0, 0, 0]
        assert conv_node.attributes.get("auto_pad", b"NOTSET") == b"NOTSET"

    def test_notset_is_passthrough(self) -> None:
        """auto_pad=NOTSET should not alter existing pads."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
        W_init = onnx.numpy_helper.from_array(np.zeros([1, 1, 3, 3], dtype=np.float32), name="W")
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "W"],
            outputs=["Y"],
            auto_pad="NOTSET",
            pads=[1, 1, 1, 1],
        )
        model = _make_model([X], outputs=[Y], initializers=[W_init], nodes=[conv_node])
        g = import_model(model)
        node = next(n for n in g.nodes if n.op_type == "Conv")
        assert node.attributes.get("pads") == [1, 1, 1, 1]

    def test_1d_same_upper(self) -> None:
        """auto_pad=SAME_UPPER on 1D Conv must produce correct 1D pads."""
        # X: (1,1,5), W: (1,1,3), strides=[1] -> Y: (1,1,5)
        # total_pad = 2, pad_begin = 1, pad_end = 1
        model = self._make_conv_model(
            auto_pad="SAME_UPPER",
            x_shape=[1, 1, 5],
            w_shape=[1, 1, 3],
            y_shape=[1, 1, 5],
            strides=[1],
        )
        g = import_model(model)
        conv_node = next(n for n in g.nodes if n.op_type == "Conv")
        assert conv_node.attributes.get("pads") == [1, 1]
        assert conv_node.attributes.get("auto_pad", b"NOTSET") == b"NOTSET"
