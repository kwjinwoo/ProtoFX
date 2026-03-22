"""Node and AttributeValue for ProtoFX IR operation representation.

``Node`` represents one normalized ONNX operation in the IR graph. It is
a mutable dataclass whose lifecycle is managed by ``ir.Graph``.

``AttributeValue`` is a type alias for normalized ONNX attribute values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protofx.ir.value import Value

type AttributeValue = int | float | str | bytes | list[int] | list[float] | list[str] | list[bytes]
"""Normalized ONNX attribute value type.

All node attributes are converted to Python-native forms during import.
The emitter must not depend on raw ``onnx.AttributeProto`` structures.
"""


@dataclass
class Node:
    """Mutable IR node representing one normalized ONNX operation.

    ``Node`` is mutable and its lifecycle is owned by ``ir.Graph``.
    All structural mutations (input/output rewiring, creation, deletion)
    must go through ``Graph`` methods to maintain use-def consistency.

    The ``inputs`` and ``outputs`` relationships are graph-managed internals.
    They are exposed as read-only properties returning tuple snapshots; only
    ``Graph`` may modify them via the private ``_inputs`` and ``_outputs``
    attributes.

    Attributes:
        id: Stable internal identifier assigned by the graph owner.
        op_type: ONNX operator type (e.g. ``"Relu"``, ``"Conv"``).
        inputs: (read-only) Ordered input ``Value`` references preserving ONNX positional order.
        outputs: (read-only) Ordered output ``Value`` references, one per operator output.
        domain: ONNX operator domain. Defaults to ``""`` (default domain).
        opset_version: ONNX opset version, or ``None`` if unspecified.
        attributes: Normalized Python-native attributes. Defaults to empty dict.
        name: Original ONNX node name for diagnostics, or ``None``.
    """

    id: str
    op_type: str
    _inputs: list[Value] = field(default_factory=list, init=False, repr=False)
    _outputs: list[Value] = field(default_factory=list, init=False, repr=False)
    domain: str = ""
    opset_version: int | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    name: str | None = None

    @property
    def inputs(self) -> tuple[Value, ...]:
        """Return an immutable snapshot of this node's input values."""
        return tuple(self._inputs)

    @property
    def outputs(self) -> tuple[Value, ...]:
        """Return an immutable snapshot of this node's output values."""
        return tuple(self._outputs)
