"""Tests for protofx.ir.Value and protofx.ir.ValueKind."""

import enum

from protofx.ir.value import ValueKind


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
