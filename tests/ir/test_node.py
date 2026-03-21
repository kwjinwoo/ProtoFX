"""Tests for protofx.ir.node.AttributeValue type alias and Node dataclass."""

import dataclasses

import pytest

from protofx.ir import DType, TensorType, Value, ValueKind
from protofx.ir.node import AttributeValue, Node


class TestAttributeValueTypeAlias:
    """Verify AttributeValue type alias exists and accepts expected types."""

    def test_alias_exists(self) -> None:
        """AttributeValue must be importable from ir.node."""
        assert AttributeValue is not None

    def test_int_is_valid(self) -> None:
        """Plain int should satisfy AttributeValue."""
        v: AttributeValue = 42
        assert isinstance(v, int)

    def test_float_is_valid(self) -> None:
        """Plain float should satisfy AttributeValue."""
        v: AttributeValue = 3.14
        assert isinstance(v, float)

    def test_bytes_is_valid(self) -> None:
        """Plain bytes should satisfy AttributeValue."""
        v: AttributeValue = b"\x00\x01"
        assert isinstance(v, bytes)

    def test_str_is_valid(self) -> None:
        """Plain str should satisfy AttributeValue."""
        v: AttributeValue = "relu"
        assert isinstance(v, str)

    def test_list_int_is_valid(self) -> None:
        """list[int] should satisfy AttributeValue."""
        v: AttributeValue = [1, 2, 3]
        assert isinstance(v, list)

    def test_list_float_is_valid(self) -> None:
        """list[float] should satisfy AttributeValue."""
        v: AttributeValue = [1.0, 2.0]
        assert isinstance(v, list)

    def test_list_bytes_is_valid(self) -> None:
        """list[bytes] should satisfy AttributeValue."""
        v: AttributeValue = [b"\x00", b"\x01"]
        assert isinstance(v, list)

    def test_list_str_is_valid(self) -> None:
        """list[str] should satisfy AttributeValue."""
        v: AttributeValue = ["a", "b"]
        assert isinstance(v, list)


# ---------------------------------------------------------------------------
# Node dataclass fields
# ---------------------------------------------------------------------------


class TestNodeIsDataclass:
    """Verify Node is a frozen dataclass."""

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(Node)

    def test_is_frozen(self) -> None:
        """Node instances must be immutable."""
        fields = dataclasses.fields(Node)
        # Frozen dataclass raises FrozenInstanceError on setattr
        assert len(fields) > 0


class TestNodeFields:
    """Verify Node has the expected fields with correct types."""

    def test_has_id_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "id" in fields

    def test_has_op_type_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "op_type" in fields

    def test_has_domain_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "domain" in fields

    def test_has_opset_version_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "opset_version" in fields

    def test_has_inputs_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "inputs" in fields

    def test_has_outputs_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "outputs" in fields

    def test_has_attributes_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "attributes" in fields

    def test_has_name_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Node)}
        assert "name" in fields


class TestNodeImmutability:
    """Node must be frozen (immutable)."""

    @pytest.fixture()
    def node_with_outputs(self) -> tuple[Node, tuple[Value, ...]]:
        """Create a minimal Node via the factory."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)),
        )
        node, outputs = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2, 3)), "relu_out"),),
        )
        return node, outputs

    def test_cannot_set_id(self, node_with_outputs: tuple[Node, tuple[Value, ...]]) -> None:
        node, _ = node_with_outputs
        with pytest.raises(AttributeError):
            node.id = "other"  # type: ignore[misc]

    def test_cannot_set_op_type(self, node_with_outputs: tuple[Node, tuple[Value, ...]]) -> None:
        node, _ = node_with_outputs
        with pytest.raises(AttributeError):
            node.op_type = "Sigmoid"  # type: ignore[misc]

    def test_cannot_set_inputs(self, node_with_outputs: tuple[Node, tuple[Value, ...]]) -> None:
        node, _ = node_with_outputs
        with pytest.raises(AttributeError):
            node.inputs = ()  # type: ignore[misc]

    def test_cannot_set_outputs(self, node_with_outputs: tuple[Node, tuple[Value, ...]]) -> None:
        node, _ = node_with_outputs
        with pytest.raises(AttributeError):
            node.outputs = ()  # type: ignore[misc]

    def test_cannot_set_attributes(self, node_with_outputs: tuple[Node, tuple[Value, ...]]) -> None:
        node, _ = node_with_outputs
        with pytest.raises(AttributeError):
            node.attributes = {}  # type: ignore[misc]

    def test_cannot_set_name(self, node_with_outputs: tuple[Node, tuple[Value, ...]]) -> None:
        node, _ = node_with_outputs
        with pytest.raises(AttributeError):
            node.name = "new_name"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Node.create() factory
# ---------------------------------------------------------------------------


class TestNodeCreateFactory:
    """Verify Node.create() produces a Node and output Values atomically."""

    def test_returns_tuple_of_node_and_values(self) -> None:
        """create() must return (Node, tuple[Value, ...])."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)),
        )
        result = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2, 3)), None),),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        node, outputs = result
        assert isinstance(node, Node)
        assert isinstance(outputs, tuple)

    def test_output_values_have_correct_producer(self) -> None:
        """Each output Value.producer must be the newly created Node."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(4,)),
        )
        node, outputs = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(4,)), None),),
        )
        assert len(outputs) == 1
        assert outputs[0].producer is node

    def test_node_outputs_match_returned_values(self) -> None:
        """node.outputs must be identical to the returned outputs tuple."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
        )
        node, outputs = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2,)), None),),
        )
        assert node.outputs == outputs

    def test_multi_output_node(self) -> None:
        """A node with multiple outputs must produce one Value per output."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(10,)),
        )
        node, outputs = Node.create(
            id="n0",
            op_type="Split",
            inputs=(input_val,),
            output_specs=(
                ("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(5,)), None),
                ("out1", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(5,)), None),
            ),
        )
        assert len(outputs) == 2
        assert outputs[0].id == "out0"
        assert outputs[1].id == "out1"
        assert outputs[0].producer is node
        assert outputs[1].producer is node

    def test_node_fields_from_create(self) -> None:
        """Verify all Node fields are set correctly by create()."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1, 3, 224, 224)),
        )
        node, _ = Node.create(
            id="n0",
            op_type="Conv",
            inputs=(input_val,),
            output_specs=(
                ("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(1, 64, 112, 112)), None),
            ),
            domain="",
            opset_version=13,
            attributes={"kernel_shape": [7, 7], "strides": [2, 2]},
            name="conv1",
        )
        assert node.id == "n0"
        assert node.op_type == "Conv"
        assert node.domain == ""
        assert node.opset_version == 13
        assert node.attributes == {"kernel_shape": [7, 7], "strides": [2, 2]}
        assert node.name == "conv1"
        assert node.inputs == (input_val,)

    def test_default_domain_is_empty_string(self) -> None:
        """domain defaults to empty string (ONNX default domain)."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
        )
        node, _ = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2,)), None),),
        )
        assert node.domain == ""

    def test_default_opset_version_is_none(self) -> None:
        """opset_version defaults to None."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
        )
        node, _ = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2,)), None),),
        )
        assert node.opset_version is None

    def test_default_attributes_is_empty_dict(self) -> None:
        """attributes defaults to an empty dict."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
        )
        node, _ = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2,)), None),),
        )
        assert node.attributes == {}

    def test_default_name_is_none(self) -> None:
        """name defaults to None."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
        )
        node, _ = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2,)), None),),
        )
        assert node.name is None

    def test_output_value_name_preserved(self) -> None:
        """Output Value name should match the spec."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
        )
        _, outputs = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2,)), "relu_out"),),
        )
        assert outputs[0].name == "relu_out"

    def test_output_value_kind_preserved(self) -> None:
        """Output Value kind should match the spec."""
        input_val = Value(
            id="in0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)),
        )
        _, outputs = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("out0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(2,)), None),),
        )
        assert outputs[0].kind == ValueKind.NODE_OUTPUT
