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
