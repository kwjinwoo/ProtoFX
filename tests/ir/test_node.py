"""Tests for protofx.ir.node -- mutable Node and AttributeValue."""

import dataclasses

import pytest

from protofx.ir import DType, TensorType
from protofx.ir.node import AttributeValue, Node
from protofx.ir.value import Value, ValueKind


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
# Node is a mutable dataclass
# ---------------------------------------------------------------------------


class TestNodeIsDataclass:
    """Verify Node is a dataclass (not frozen)."""

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(Node)

    def test_is_not_frozen(self) -> None:
        """Node must NOT be frozen -- it is mutable, owned by Graph."""
        node = Node(id="n0", op_type="Relu")
        node.op_type = "Sigmoid"
        assert node.op_type == "Sigmoid"


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


# ---------------------------------------------------------------------------
# Node mutability -- Graph is the owner, but Node allows assignment
# ---------------------------------------------------------------------------


class TestNodeMutability:
    """Node must be mutable -- all field assignments succeed."""

    @pytest.fixture()
    def node(self) -> Node:
        """Create a minimal mutable Node."""
        return Node(id="n0", op_type="Relu")

    def test_can_set_id(self, node: Node) -> None:
        node.id = "n99"
        assert node.id == "n99"

    def test_can_set_op_type(self, node: Node) -> None:
        node.op_type = "Sigmoid"
        assert node.op_type == "Sigmoid"

    def test_can_set_inputs(self, node: Node) -> None:
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)))
        node.inputs = [v]
        assert node.inputs == [v]

    def test_can_set_outputs(self, node: Node) -> None:
        v = Value(id="v1", kind=ValueKind.NODE_OUTPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)))
        node.outputs = [v]
        assert node.outputs == [v]

    def test_can_set_attributes(self, node: Node) -> None:
        node.attributes = {"kernel_shape": [3, 3]}
        assert node.attributes == {"kernel_shape": [3, 3]}

    def test_can_set_name(self, node: Node) -> None:
        node.name = "my_relu"
        assert node.name == "my_relu"

    def test_can_set_domain(self, node: Node) -> None:
        node.domain = "com.custom"
        assert node.domain == "com.custom"

    def test_can_set_opset_version(self, node: Node) -> None:
        node.opset_version = 13
        assert node.opset_version == 13


# ---------------------------------------------------------------------------
# Node defaults
# ---------------------------------------------------------------------------


class TestNodeDefaults:
    """Verify default values for optional Node fields."""

    def test_inputs_default_empty_list(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert node.inputs == []

    def test_outputs_default_empty_list(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert node.outputs == []

    def test_domain_default_empty_string(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert node.domain == ""

    def test_opset_version_default_none(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert node.opset_version is None

    def test_attributes_default_empty_dict(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert node.attributes == {}

    def test_name_default_none(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert node.name is None


# ---------------------------------------------------------------------------
# Node.create() must not exist
# ---------------------------------------------------------------------------


class TestNodeCreateRemoved:
    """Node.create() factory must be removed -- Graph owns construction."""

    def test_no_create_classmethod(self) -> None:
        assert not hasattr(Node, "create"), "Node.create() should be removed; Graph owns node construction"


# ---------------------------------------------------------------------------
# Node inputs/outputs are lists (not tuples)
# ---------------------------------------------------------------------------


class TestNodeListInterfaces:
    """Node.inputs and Node.outputs must be list, not tuple."""

    def test_inputs_is_list(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert isinstance(node.inputs, list)

    def test_outputs_is_list(self) -> None:
        node = Node(id="n0", op_type="Relu")
        assert isinstance(node.outputs, list)

    def test_inputs_appendable(self) -> None:
        """Inputs list supports append (mutable)."""
        node = Node(id="n0", op_type="Relu")
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        node.inputs.append(v)
        assert len(node.inputs) == 1

    def test_outputs_appendable(self) -> None:
        """Outputs list supports append (mutable)."""
        node = Node(id="n0", op_type="Relu")
        v = Value(id="v0", kind=ValueKind.NODE_OUTPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        node.outputs.append(v)
        assert len(node.outputs) == 1
