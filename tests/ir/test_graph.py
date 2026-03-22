"""Tests for protofx.ir.Graph — graph-owned mutable IR."""

from protofx.ir.graph import Graph


class TestGraphConstruction:
    """Verify Graph can be constructed with expected fields."""

    def test_empty_graph(self) -> None:
        """An empty Graph has no inputs, outputs, or nodes."""
        g = Graph()
        assert g.inputs == []
        assert g.outputs == []
        assert g.nodes == []

    def test_name_default_none(self) -> None:
        """Graph name defaults to None."""
        g = Graph()
        assert g.name is None

    def test_name_set(self) -> None:
        """Graph name can be explicitly set."""
        g = Graph(name="main")
        assert g.name == "main"

    def test_parent_default_none(self) -> None:
        """Parent defaults to None for top-level graphs."""
        g = Graph()
        assert g.parent is None

    def test_parent_can_be_set(self) -> None:
        """Parent can be set to another Graph for subgraph support."""
        parent = Graph(name="outer")
        child = Graph(name="inner", parent=parent)
        assert child.parent is parent

    def test_values_registry_empty(self) -> None:
        """Internal value registry starts empty."""
        g = Graph()
        assert g.value_count == 0

    def test_node_count_empty(self) -> None:
        """Internal node count starts at zero."""
        g = Graph()
        assert g.node_count == 0
