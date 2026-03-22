"""Graph for ProtoFX IR — the structural owner of nodes and values.

``Graph`` owns node membership, value registration, topological order,
and use-def consistency. All mutations to ``Node`` and ``Value``
relationships must go through ``Graph`` methods.
"""

from __future__ import annotations

from protofx.ir.node import AttributeValue, Node
from protofx.ir.tensor_type import TensorType
from protofx.ir.value import Value, ValueKind


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

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _new_value_id(self) -> str:
        """Generate a unique value id."""
        vid = f"v{self._next_value_id}"
        self._next_value_id += 1
        return vid

    def _new_node_id(self) -> str:
        """Generate a unique node id."""
        nid = f"n{self._next_node_id}"
        self._next_node_id += 1
        return nid

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register_value(self, value: Value) -> None:
        """Register a value in the internal registry.

        Args:
            value: The Value to register.
        """
        self._values[value.id] = value

    def _register_node(self, node: Node) -> None:
        """Register a node in the internal registry.

        Args:
            node: The Node to register.
        """
        self._nodes[node.id] = node

    # ------------------------------------------------------------------
    # Construction APIs
    # ------------------------------------------------------------------

    def add_input(self, *, tensor_type: TensorType, name: str | None = None) -> Value:
        """Create and register a graph input ``Value``.

        Args:
            tensor_type: Tensor metadata for the input.
            name: Optional ONNX source name.

        Returns:
            The newly created ``GRAPH_INPUT`` Value.
        """
        value = Value(
            id=self._new_value_id(),
            kind=ValueKind.GRAPH_INPUT,
            tensor_type=tensor_type,
            name=name,
        )
        self._register_value(value)
        self.inputs.append(value)
        return value

    def make_node(
        self,
        *,
        op_type: str,
        inputs: list[Value],
        output_types: list[TensorType],
        domain: str = "",
        opset_version: int | None = None,
        attributes: dict[str, AttributeValue] | None = None,
        name: str | None = None,
        output_names: list[str | None] | None = None,
    ) -> Node:
        """Create a ``Node`` with output ``Value`` instances and wire use-def links.

        Args:
            op_type: ONNX operator type string.
            inputs: Ordered input Values.
            output_types: One ``TensorType`` per output to create.
            domain: ONNX operator domain. Defaults to ``""``.
            opset_version: ONNX opset version, or ``None``.
            attributes: Normalized attribute dict, or ``None`` for empty.
            name: Optional ONNX node name.
            output_names: Optional list of ONNX names for output Values.

        Returns:
            The newly created ``Node`` with outputs populated.
        """
        node = Node(
            id=self._new_node_id(),
            op_type=op_type,
            inputs=list(inputs),
            domain=domain,
            opset_version=opset_version,
            attributes=attributes if attributes is not None else {},
            name=name,
        )

        # Create output Values
        names = output_names or [None] * len(output_types)
        for tt, oname in zip(output_types, names, strict=True):
            out_value = Value(
                id=self._new_value_id(),
                kind=ValueKind.NODE_OUTPUT,
                tensor_type=tt,
                name=oname,
                producer=node,
            )
            node.outputs.append(out_value)
            self._register_value(out_value)

        # Wire input users
        for slot, inp_value in enumerate(node.inputs):
            inp_value.users.append((node, slot))

        self._register_node(node)
        self.nodes.append(node)
        return node
