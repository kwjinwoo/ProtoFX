"""Graph for ProtoFX IR — the structural owner of nodes and values.

``Graph`` owns node membership, value registration, topological order,
and use-def consistency. All mutations to ``Node`` and ``Value``
relationships must go through ``Graph`` methods.
"""

from __future__ import annotations

import numpy as np

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
        initializers: Ordered list of ``INITIALIZER`` ``Value`` instances.
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
        self.initializers: list[Value] = []
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

    def add_sentinel(self) -> Value:
        """Create and register a ``SENTINEL`` value for an omitted optional input.

        Each call creates a new distinct sentinel instance.

        Returns:
            The newly created ``SENTINEL`` Value.
        """
        value = Value(
            id=self._new_value_id(),
            kind=ValueKind.SENTINEL,
            tensor_type=TensorType(dtype=None, shape=None),
        )
        self._register_value(value)
        return value

    def add_constant(
        self,
        *,
        tensor_type: TensorType,
        data: np.ndarray,
        name: str | None = None,
    ) -> Value:
        """Create and register a ``CONSTANT`` value with a data payload.

        Constants represent values produced by inlined ``Constant`` ops during
        import. They are not appended to ``graph.inputs`` or
        ``graph.initializers``.

        Args:
            tensor_type: Tensor metadata for the constant.
            data: The constant tensor data as a numpy array.
            name: Optional ONNX source name.

        Returns:
            The newly created ``CONSTANT`` Value.
        """
        value = Value(
            id=self._new_value_id(),
            kind=ValueKind.CONSTANT,
            tensor_type=tensor_type,
            name=name,
            data=data,
        )
        self._register_value(value)
        return value

    def add_initializer(
        self,
        *,
        tensor_type: TensorType,
        data: np.ndarray,
        name: str | None = None,
    ) -> Value:
        """Create and register an ``INITIALIZER`` value with a data payload.

        Initializers represent pretrained weights and biases. They are appended
        to ``graph.initializers`` but not to ``graph.inputs``.

        Args:
            tensor_type: Tensor metadata for the initializer.
            data: The initializer tensor data as a numpy array.
            name: Optional ONNX source name.

        Returns:
            The newly created ``INITIALIZER`` Value.
        """
        value = Value(
            id=self._new_value_id(),
            kind=ValueKind.INITIALIZER,
            tensor_type=tensor_type,
            name=name,
            data=data,
        )
        self._register_value(value)
        self.initializers.append(value)
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
            domain=domain,
            opset_version=opset_version,
            attributes=attributes if attributes is not None else {},
            name=name,
        )
        node._inputs = list(inputs)

        # Create output Values
        names = output_names or [None] * len(output_types)
        for tt, oname in zip(output_types, names, strict=True):
            out_value = Value(
                id=self._new_value_id(),
                kind=ValueKind.NODE_OUTPUT,
                tensor_type=tt,
                name=oname,
            )
            out_value._producer = node
            node._outputs.append(out_value)
            self._register_value(out_value)

        # Wire input users
        for slot, inp_value in enumerate(node._inputs):
            inp_value._users.append((node, slot))

        self._register_node(node)
        self.nodes.append(node)
        return node

    # ------------------------------------------------------------------
    # Mutation APIs
    # ------------------------------------------------------------------

    def set_node_inputs(self, node: Node, new_inputs: list[Value]) -> None:
        """Atomically rewire a node's inputs, maintaining use-def consistency.

        Atomic replacement order:
        1. Remove this node's entries from old input Value.users.
        2. Replace node.inputs with *new_inputs*.
        3. Add ``(node, slot)`` entries to each new input Value.users.

        Args:
            node: The node whose inputs are being replaced.
            new_inputs: New ordered input Values.
        """
        # Step 1: remove old user entries
        for slot, old_value in enumerate(node._inputs):
            old_value._users.remove((node, slot))

        # Step 2: replace
        node._inputs = list(new_inputs)

        # Step 3: add new user entries
        for slot, new_value in enumerate(node._inputs):
            new_value._users.append((node, slot))

    def set_value_type(self, value: Value, tensor_type: TensorType) -> None:
        """Update the tensor metadata on a ``Value``.

        Args:
            value: The value to update.
            tensor_type: New tensor type to assign.
        """
        value.tensor_type = tensor_type

    def set_graph_outputs(self, outputs: list[Value]) -> None:
        """Set the graph output values, replacing any previous outputs.

        Args:
            outputs: Ordered list of output Values.
        """
        self.outputs = list(outputs)

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graph.

        Raises ``ValueError`` if any of the node's output Values still have
        consumers (fast-fail policy).

        Cleanup steps:
        1. Check that all output Values have no users.
        2. Remove this node's entries from input Value.users.
        3. Unregister output Values from the value registry.
        4. Remove the node from the node list and registry.

        Args:
            node: The node to remove.

        Raises:
            ValueError: If any output Value is still in use.
        """
        # Step 1: fast-fail check
        for out_value in node._outputs:
            if out_value._users:
                msg = f"cannot remove node {node.id!r}: output {out_value.id!r} is still in use"
                raise ValueError(msg)

        # Step 2: clean input users
        for slot, inp_value in enumerate(node._inputs):
            inp_value._users.remove((node, slot))

        # Step 3: unregister output values
        for out_value in node._outputs:
            del self._values[out_value.id]

        # Step 4: remove node
        self.nodes.remove(node)
        del self._nodes[node.id]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[Node]:
        """Return nodes in topological order using Kahn's algorithm.

        A node is "ready" when every ``Value`` in its ``inputs`` list that
        is produced by another ``Node`` (i.e. ``ValueKind.NODE_OUTPUT``)
        has already been emitted.  Graph inputs and other external values
        carry no inter-node dependency.

        Returns:
            A new list of ``Node`` instances in topological order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        from collections import deque

        # Build in-degree map: for each node, count how many producer-nodes
        # must come before it.
        in_degree: dict[str, int] = {}
        for node in self.nodes:
            count = 0
            seen_producers: set[str] = set()
            for inp_value in node.inputs:
                if inp_value.kind == ValueKind.NODE_OUTPUT and inp_value.producer is not None:
                    pid = inp_value.producer.id
                    if pid not in seen_producers:
                        seen_producers.add(pid)
                        count += 1
            in_degree[node.id] = count

        # Seed the queue with nodes that have zero in-degree.
        queue: deque[Node] = deque()
        for node in self.nodes:
            if in_degree[node.id] == 0:
                queue.append(node)

        result: list[Node] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            # For each output value, decrement in-degree of consumer nodes.
            for out_value in node.outputs:
                seen_consumers: set[str] = set()
                for consumer, _slot in out_value.users:
                    if consumer.id not in seen_consumers:
                        seen_consumers.add(consumer.id)
                        in_degree[consumer.id] -= 1
                        if in_degree[consumer.id] == 0:
                            queue.append(consumer)

        if len(result) != len(self.nodes):
            msg = "graph contains a cycle"
            raise ValueError(msg)

        return result

    def validate(self) -> None:
        """Check all IR invariants and raise ``ValueError`` on any violation.

        Invariants checked:
        1. Every node output has a ``producer`` pointing back to the owning node.
        2. Every node input is a value registered in this graph.
        3. Use-def consistency: each ``(consumer, slot)`` in ``value.users``
           corresponds to ``consumer.inputs[slot] is value``.
        4. The graph is acyclic (via ``topological_sort``).

        Raises:
            ValueError: If any invariant is violated.
        """
        # 1. Producer back-references
        for node in self.nodes:
            for out_value in node.outputs:
                if out_value.producer is not node:
                    msg = f"node {node.id!r} output {out_value.id!r}: producer mismatch"
                    raise ValueError(msg)

        # 2. All node inputs must be registered values
        for node in self.nodes:
            for inp_value in node.inputs:
                if inp_value.id not in self._values:
                    msg = f"node {node.id!r} input {inp_value.id!r}: not registered in graph"
                    raise ValueError(msg)

        # 3. Use-def consistency
        for value in self._values.values():
            for consumer, slot in value.users:
                if consumer.id not in self._nodes:
                    msg = f"value {value.id!r}: user node {consumer.id!r} not registered in graph"
                    raise ValueError(msg)
                if slot >= len(consumer._inputs) or consumer._inputs[slot] is not value:
                    msg = f"value {value.id!r}: user ({consumer.id!r}, slot={slot}) back-reference mismatch"
                    raise ValueError(msg)

        # 4. Cycle check
        self.topological_sort()
