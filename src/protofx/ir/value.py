"""Value and ValueKind for ProtoFX IR data-flow representation.

``Value`` is the primary data-flow unit in the IR graph. Every graph input,
node output, constant, initializer, and omitted optional input is represented
as a ``Value`` instance.

``ValueKind`` classifies the origin of each ``Value``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protofx.ir.node import Node

from protofx.ir.tensor_type import TensorType


class ValueKind(enum.Enum):
    """Classification of a ``Value``'s origin in the IR graph.

    Callers should compare kinds directly::

        if value.kind == ValueKind.SENTINEL:
            ...

    Members:
        GRAPH_INPUT: A runtime graph input.
        NODE_OUTPUT: An output produced by an IR node.
        SENTINEL: A placeholder for an omitted optional ONNX input.
        CONSTANT: A constant produced by a ``Constant`` op during import.
        INITIALIZER: A graph-level initializer (pretrained weight, etc.).
    """

    GRAPH_INPUT = enum.auto()
    NODE_OUTPUT = enum.auto()
    SENTINEL = enum.auto()
    CONSTANT = enum.auto()
    INITIALIZER = enum.auto()


@dataclass
class Value:
    """Mutable data-flow unit in the IR graph.

    All data-flow — graph inputs, node outputs, constants, initializers, and
    omitted optional inputs — is represented as a ``Value``.

    ``Value`` is mutable and its lifecycle is owned by ``ir.Graph``.
    All structural mutations (producer/user rewiring) must go through
    ``Graph`` methods to maintain use-def consistency.

    The ``producer`` and ``users`` relationships are graph-managed internals.
    They are exposed as read-only properties; only ``Graph`` may modify them
    via the private ``_producer`` and ``_users`` attributes.

    Attributes:
        id: Stable internal identifier. Assigned by the graph owner.
        kind: Classification of this value's origin.
        tensor_type: Tensor metadata (dtype and shape).
        name: Original ONNX name preserved as source metadata, or ``None``.
        producer: (read-only) The IR ``Node`` that produces this value, or
            ``None`` for graph inputs and sentinel values.
        users: (read-only) Tuple of ``(node, input_slot_index)`` pairs
            tracking every consumer of this value.
    """

    id: str
    kind: ValueKind
    tensor_type: TensorType
    name: str | None = None
    _producer: Node | None = field(default=None, init=False, repr=False)
    _users: list[tuple[Node, int]] = field(default_factory=list, init=False, repr=False)

    @property
    def producer(self) -> Node | None:
        """Return the producer node, or ``None`` for graph inputs / sentinels."""
        return self._producer

    @property
    def users(self) -> tuple[tuple[Node, int], ...]:
        """Return an immutable snapshot of this value's consumers."""
        return tuple(self._users)
