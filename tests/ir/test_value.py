"""Tests for protofx.ir.Value and protofx.ir.ValueKind."""

import dataclasses
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

    def test_node_output_with_producer(self) -> None:
        input_val = Value(id="in0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.INT64, shape=(4,)))
        node, outputs = Node.create(
            id="n0",
            op_type="Identity",
            inputs=(input_val,),
            output_specs=(("y", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.INT64, shape=(4,)), None),),
        )
        assert outputs[0].producer is node

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


# ---------------------------------------------------------------------------
# Value immutability
# ---------------------------------------------------------------------------


class TestValueImmutability:
    """Value must be frozen (immutable)."""

    @pytest.fixture()
    def value(self) -> Value:
        """A simple Value for mutation tests."""
        return Value(
            id="v0",
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)),
        )

    def test_cannot_set_id(self, value: Value) -> None:
        with pytest.raises(AttributeError):
            value.id = "other"  # type: ignore[misc]

    def test_cannot_set_kind(self, value: Value) -> None:
        with pytest.raises(AttributeError):
            value.kind = ValueKind.SENTINEL  # type: ignore[misc]

    def test_cannot_set_tensor_type(self, value: Value) -> None:
        with pytest.raises(AttributeError):
            value.tensor_type = TensorType(dtype=None, shape=None)  # type: ignore[misc]

    def test_cannot_set_name(self, value: Value) -> None:
        with pytest.raises(AttributeError):
            value.name = "new"  # type: ignore[misc]

    def test_cannot_set_producer(self, value: Value) -> None:
        input_val = Value(id="in0", kind=ValueKind.GRAPH_INPUT, tensor_type=TensorType(dtype=DType.FLOAT32, shape=(1,)))
        node, _ = Node.create(
            id="n0",
            op_type="Relu",
            inputs=(input_val,),
            output_specs=(("o0", ValueKind.NODE_OUTPUT, TensorType(dtype=DType.FLOAT32, shape=(1,)), None),),
        )
        with pytest.raises(AttributeError):
            value.producer = node  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Value replacement (dataclasses.replace)
# ---------------------------------------------------------------------------


class TestValueReplacement:
    """Verify dataclasses.replace produces a new Value without mutating the original."""

    def test_replace_tensor_type(self) -> None:
        original_tt = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=original_tt)
        new_tt = TensorType(dtype=DType.FLOAT16, shape=(2, 3))
        v2 = dataclasses.replace(v, tensor_type=new_tt)
        assert v2.tensor_type is new_tt
        assert v.tensor_type is original_tt  # original unchanged

    def test_replace_name(self) -> None:
        tt = TensorType(dtype=DType.FLOAT32, shape=(1,))
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=tt)
        v2 = dataclasses.replace(v, name="renamed")
        assert v2.name == "renamed"
        assert v.name is None

    def test_replace_returns_new_instance(self) -> None:
        tt = TensorType(dtype=DType.FLOAT32, shape=(2,))
        v = Value(id="v0", kind=ValueKind.GRAPH_INPUT, tensor_type=tt)
        v2 = dataclasses.replace(v, name="new")
        assert v is not v2

    def test_replace_preserves_other_fields(self) -> None:
        tt = TensorType(dtype=DType.INT64, shape=(4,))
        input_val = Value(id="in0", kind=ValueKind.GRAPH_INPUT, tensor_type=tt)
        node, outputs = Node.create(
            id="n0",
            op_type="Identity",
            inputs=(input_val,),
            output_specs=(("v0", ValueKind.NODE_OUTPUT, tt, "out"),),
        )
        v = outputs[0]
        v2 = dataclasses.replace(v, name="renamed")
        assert v2.id == "v0"
        assert v2.kind == ValueKind.NODE_OUTPUT
        assert v2.tensor_type is tt
        assert v2.producer is node
