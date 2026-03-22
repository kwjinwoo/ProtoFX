"""Tests for protofx.ir.Value and protofx.ir.ValueKind -- mutable Value."""

import enum

import pytest

from protofx.ir import DType, TensorType
from protofx.ir.node import Node
from protofx.ir.value import Value, ValueKind


class TestValueKindEnum:
    """Verify ValueKind is an Enum with the expected members."""

    def test_is_enum(self) -> None:
        assert issubclass(ValueKind, enum.Enum)

    def test_has_graph_input(self) -> None:
        assert hasattr(ValueKind, "GRAPH_INPUT")

    def test_has_node_output(self) -> None:
        assert hasattr(ValueKind, "NODE_OUTPUT")

    def test_has_sentinel(self) -> None:
        assert hasattr(ValueKind, "SENTINEL")

    def test_has_constant(self) -> None:
        assert hasattr(ValueKind, "CONSTANT")

    def test_has_initializer(self) -> None:
        assert hasattr(ValueKind, "INITIALIZER")

    def test_member_count(self) -> None:
        assert len(ValueKind) == 5


class TestValueKindComparison:
    """Callers compare kinds directly via == against ValueKind members."""

    def test_same_kind_equal(self) -> None:
        assert ValueKind.GRAPH_INPUT == ValueKind.GRAPH_INPUT

    def test_different_kinds_not_equal(self) -> None:
        assert ValueKind.GRAPH_INPUT != ValueKind.SENTINEL

    def test_all_members_distinct(self) -> None:
        members = list(ValueKind)
        assert len(members) == len(set(members))


# ---------------------------------------------------------------------------
# Value construction
# ---------------------------------------------------------------------------


class TestValueConstruction:
    """Verify Value construction and field access."""

    def test_graph_input_defaults(self) -> None:
        tt = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        v = Value(id="x", kind=ValueKind.GRAPH_INPUT, tensor_type=tt)
        assert v.id == "x"
        assert v.kind == ValueKind.GRAPH_INPUT
        assert v.tensor_type is tt
        assert v.name is None
        assert v.producer is None
        assert v.users == []

    def test_node_output_with_producer(self) -> None:
        """Value can be constructed with a producer Node."""
        node = Node(id="n0", op_type="Identity")
        v = Value(
            id="y",
            kind=ValueKind.NODE_OUTPUT,
            tensor_type=TensorType(dtype=DType.INT64, shape=(4,)),
            producer=node,
        )
        assert v.producer is node

    def test_sentinel_value(self) -> None:
        v = Value(id="s0", kind=ValueKind.SENTINEL, tensor_type=TensorType(dtype=None, shape=None))
        assert v.kind == ValueKind.SENTINEL

    def test_constant_value(self) -> None:
        tt = TensorType(dtype=DType.FLOAT32, shape=())
        v = Value(id="c0", kind=ValueKind.CONSTANT, tensor_type=tt)
        assert v.kind == ValueKind.CONSTANT

    def test_initializer_value(self) -> None:
        tt = TensorType(dtype=DType.FLOAT32, shape=(64, 3, 7, 7))
        v = Value(id="w0", kind=ValueKind.INITIALIZER, tensor_type=tt)
        assert v.kind == ValueKind.INITIALIZER

    def test_name_preserved(self) -> None:
        tt = TensorType(dtype=DType.FLOAT32, shape=(1,))
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=tt, name="input_0")
        assert v.name == "input_0"

    def test_users_default_empty(self) -> None:
        tt = TensorType(dtype=DType.FLOAT32, shape=(2,))
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=tt)
        assert v.users == []
        assert isinstance(v.users, list)


# ---------------------------------------------------------------------------
# Value mutability
# ---------------------------------------------------------------------------


class TestValueMutability:
    """Value must be mutable -- Graph owns structural mutations."""

    @pytest.fixture()
    def value(self) -> Value:
        """A simple Value for mutation tests."""
        return Value(
            id="v0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)),
        )

    def test_can_set_id(self, value: Value) -> None:
        value.id = "v99"
        assert value.id == "v99"

    def test_can_set_kind(self, value: Value) -> None:
        value.kind = ValueKind.SENTINEL
        assert value.kind == ValueKind.SENTINEL

    def test_can_set_tensor_type(self, value: Value) -> None:
        new_tt = TensorType(dtype=None, shape=None)
        value.tensor_type = new_tt
        assert value.tensor_type is new_tt

    def test_can_set_name(self, value: Value) -> None:
        value.name = "new"
        assert value.name == "new"

    def test_can_set_producer(self, value: Value) -> None:
        node = Node(id="n0", op_type="Relu")
        value.producer = node
        assert value.producer is node

    def test_can_append_user(self, value: Value) -> None:
        node = Node(id="n0", op_type="Relu")
        value.users.append((node, 0))
        assert len(value.users) == 1
        assert value.users[0] == (node, 0)


# ---------------------------------------------------------------------------
# Value users field
# ---------------------------------------------------------------------------


class TestValueUsersField:
    """Verify the users field tracks consumer nodes."""

    def test_users_is_list(self) -> None:
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        assert isinstance(v.users, list)

    def test_users_stores_node_and_slot(self) -> None:
        """Each user entry is (Node, input_slot_index)."""
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        n1 = Node(id="n0", op_type="Relu")
        n2 = Node(id="n1", op_type="Sigmoid")
        v.users.append((n1, 0))
        v.users.append((n2, 0))
        assert len(v.users) == 2
        assert v.users[0] == (n1, 0)
        assert v.users[1] == (n2, 0)

    def test_users_independent_per_value(self) -> None:
        """Each Value has its own users list (no shared default)."""
        v1 = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        v2 = Value(id="v1", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        n = Node(id="n0", op_type="Relu")
        v1.users.append((n, 0))
        assert len(v1.users) == 1
        assert len(v2.users) == 0
