"""Graph for ProtoFX IR — the structural owner of nodes and values.

``Graph`` owns node membership, value registration, topological order,
and use-def consistency. All mutations to ``Node`` and ``Value``
relationships must go through ``Graph`` methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protofx.ir.node import Node
    from protofx.ir.value import Value


class Graph:
    """Graph-owned mutable IR container.

    ``Graph`` is the single owner of all ``Node`` and ``Value`` instances in
    the IR. Construction, mutation, and deletion of graph elements must go
    through methods on this class to maintain use-def consistency.

    Attributes:
        name: Optional graph name for diagnostics.
        parent: Optional parent ``Graph`` reference for future subgraph support.
        inputs: Ordered list of graph input ``Value`` instances.
        outputs: Ordered list of graph output ``Value`` instances.
        nodes: Ordered list of ``Node`` instances in insertion order.
    """

    def __init__(self, *, name: str | None = None, parent: Graph | None = None) -> None:
        """Initialize an empty graph.

        Args:
            name: Optional graph name.
            parent: Optional parent graph for subgraph support.
        """
        self.name = name
        self.parent = parent
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []
        self.nodes: list[Node] = []

        # Internal registries
        self._values: dict[str, Value] = {}
        self._nodes: dict[str, Node] = {}

        # Auto-ID counters
        self._next_value_id: int = 0
        self._next_node_id: int = 0

    @property
    def value_count(self) -> int:
        """Return the number of registered values."""
        return len(self._values)

    @property
    def node_count(self) -> int:
        """Return the number of registered nodes."""
        return len(self._nodes)
