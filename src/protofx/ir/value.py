"""Value and ValueKind for ProtoFX IR data-flow representation.

``Value`` is the primary data-flow unit in the IR graph. Every graph input,
node output, constant, initializer, and omitted optional input is represented
as a ``Value`` instance.

``ValueKind`` classifies the origin of each ``Value``.
"""

import enum


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
