"""Value and ValueKind for ProtoFX IR data-flow representation.

``Value`` is the primary data-flow unit in the IR graph. Every graph input,
node output, constant, initializer, and omitted optional input is represented
as a ``Value`` instance.

``ValueKind`` classifies the origin of each ``Value``.
"""

import enum
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Value:
    """Immutable data-flow unit in the IR graph.

    All data-flow — graph inputs, node outputs, constants, initializers, and
    omitted optional inputs — is represented as a ``Value``.

    ``Value`` is frozen. To update metadata such as ``tensor_type`` or
    ``name``, use ``dataclasses.replace()`` to produce a new instance.

    Attributes:
        id: Stable internal identifier. Assigned externally; uniqueness is
            enforced by the graph owner, not by ``Value`` itself.
        kind: Classification of this value's origin.
        tensor_type: Tensor metadata (dtype and shape).
        name: Original ONNX name preserved as source metadata, or ``None``.
        producer: The IR ``Node`` that produces this value, or ``None`` for
            graph inputs and sentinel values.
    """

    id: str
    kind: ValueKind
    tensor_type: TensorType
    name: str | None = None
    producer: Node | None = None
