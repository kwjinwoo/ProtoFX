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
        assert v.users == ()

    def test_node_output_with_producer_via_graph(self) -> None:
        """Value produced by a node via Graph.make_node has correct producer."""
        from protofx.ir.graph import Graph

        g = Graph()
        inp = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(4,)))
        node = g.make_node(
            op_type="Identity",
            inputs=[inp],
            output_types=[TensorType(dtype=DType.INT64, shape=(4,))],
        )
        assert node.outputs[0].producer is node

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
        assert v.users == ()
        assert isinstance(v.users, tuple)


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

    def test_cannot_set_producer(self, value: Value) -> None:
        """producer is read-only; assignment must raise AttributeError."""
        node = Node(id="n0", op_type="Relu")
        with pytest.raises(AttributeError):
            value.producer = node  # type: ignore[misc]

    def test_cannot_append_user(self, value: Value) -> None:
        """users returns a tuple; append must raise AttributeError."""
        with pytest.raises(AttributeError):
            value.users.append((Node(id="n0", op_type="Relu"), 0))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Value users field
# ---------------------------------------------------------------------------


class TestValueUsersField:
    """Verify the users property returns a read-only tuple via Graph."""

    def test_users_is_tuple(self) -> None:
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        assert isinstance(v.users, tuple)

    def test_users_tracks_consumers_via_graph(self) -> None:
        """Each user entry is (Node, input_slot_index), wired by Graph."""
        from protofx.ir.graph import Graph

        g = Graph()
        v = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        n1 = g.make_node(op_type="Relu", inputs=[v], output_types=[TensorType(dtype=DType.FLOAT32, shape=(1,))])
        n2 = g.make_node(
            op_type="Sigmoid", inputs=[n1.outputs[0]], output_types=[TensorType(dtype=DType.FLOAT32, shape=(1,))]
        )
        assert len(v.users) == 1
        assert v.users[0] == (n1, 0)
        assert len(n1.outputs[0].users) == 1
        assert n1.outputs[0].users[0] == (n2, 0)

    def test_users_independent_per_value(self) -> None:
        """Each Value has its own users tuple (no shared default)."""
        from protofx.ir.graph import Graph

        g = Graph()
        v1 = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        v2 = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        g.make_node(op_type="Relu", inputs=[v1], output_types=[TensorType(dtype=DType.FLOAT32, shape=(1,))])
        assert len(v1.users) == 1
        assert len(v2.users) == 0


# ---------------------------------------------------------------------------
# Value read-only producer/users enforcement
# ---------------------------------------------------------------------------


class TestValueProducerReadOnly:
    """Value.producer must be a read-only property — assignment raises AttributeError."""

    def test_producer_assignment_raises(self) -> None:
        """Directly assigning to value.producer must raise AttributeError."""
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2,)))
        node = Node(id="n0", op_type="Relu")
        with pytest.raises(AttributeError):
            v.producer = node  # type: ignore[misc]

    def test_producer_default_is_none(self) -> None:
        """A freshly constructed Value has producer == None."""
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        assert v.producer is None


class TestValueUsersReadOnly:
    """Value.users must be a read-only tuple — mutation is forbidden."""

    def test_users_returns_tuple(self) -> None:
        """value.users must return a tuple, not a list."""
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        assert isinstance(v.users, tuple)

    def test_users_default_is_empty_tuple(self) -> None:
        """A freshly constructed Value has users == ()."""
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        assert v.users == ()

    def test_users_assignment_raises(self) -> None:
        """Directly assigning to value.users must raise AttributeError."""
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        with pytest.raises(AttributeError):
            v.users = []  # type: ignore[misc]
