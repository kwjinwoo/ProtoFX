"""Node and AttributeValue for ProtoFX IR operation representation.

``Node`` represents one normalized ONNX operation in the IR graph. It is
a frozen dataclass whose output ``Value`` instances are constructed
atomically via the ``Node.create()`` factory classmethod.

``AttributeValue`` is a type alias for normalized ONNX attribute values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protofx.ir.tensor_type import TensorType
    from protofx.ir.value import Value, ValueKind

type AttributeValue = int | float | str | bytes | list[int] | list[float] | list[str] | list[bytes]
"""Normalized ONNX attribute value type.

All node attributes are converted to Python-native forms during import.
The emitter must not depend on raw ``onnx.AttributeProto`` structures.
"""


@dataclass(frozen=True)
class Node:
    """Immutable IR node representing one normalized ONNX operation.

    ``Node`` is frozen. Construction requires the ``create()`` classmethod
    which atomically builds the node and its output ``Value`` instances,
    resolving the circular ``Node ↔ Value`` reference.

    Attributes:
        id: Stable internal identifier assigned by the graph owner.
        op_type: ONNX operator type (e.g. ``"Relu"``, ``"Conv"``).
        inputs: Ordered input ``Value`` references preserving ONNX positional order.
        outputs: Ordered output ``Value`` references, one per operator output.
        domain: ONNX operator domain. Defaults to ``""`` (default domain).
        opset_version: ONNX opset version, or ``None`` if unspecified.
        attributes: Normalized Python-native attributes. Defaults to empty dict.
        name: Original ONNX node name for diagnostics, or ``None``.
    """

    id: str
    op_type: str
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...] = field(default=())
    domain: str = ""
    opset_version: int | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    name: str | None = None

    @classmethod
    def create(
        cls,
        *,
        id: str,
        op_type: str,
        inputs: tuple[Value, ...],
        output_specs: tuple[tuple[str, ValueKind, TensorType, str | None], ...],
        domain: str = "",
        opset_version: int | None = None,
        attributes: dict[str, AttributeValue] | None = None,
        name: str | None = None,
    ) -> tuple[Node, tuple[Value, ...]]:
        """Atomically create a ``Node`` and its output ``Value`` instances.

        This factory resolves the circular reference between ``Node.outputs``
        and ``Value.producer`` by using ``object.__setattr__`` to set the
        ``outputs`` field on the frozen node after initial construction.

        Args:
            id: Stable internal identifier for the node.
            op_type: ONNX operator type string.
            inputs: Ordered tuple of input ``Value`` references.
            output_specs: Tuple of ``(id, kind, tensor_type, name)`` specs,
                one per output. Each spec is used to construct a ``Value``
                whose ``producer`` points back to this node.
            domain: ONNX operator domain. Defaults to ``""``.
            opset_version: ONNX opset version, or ``None``.
            attributes: Normalized attribute dict, or ``None`` for empty.
            name: Original ONNX node name, or ``None``.

        Returns:
            A ``(node, outputs)`` tuple where each output ``Value`` has
            ``producer`` set to the returned node.
        """
        from protofx.ir.value import Value

        node = cls(
            id=id,
            op_type=op_type,
            inputs=inputs,
            domain=domain,
            opset_version=opset_version,
            attributes=attributes if attributes is not None else {},
            name=name,
        )

        outputs = tuple(
            Value(id=spec[0], kind=spec[1], tensor_type=spec[2], name=spec[3], producer=node) for spec in output_specs
        )

        # Bypass frozen restriction to set outputs after Value construction.
        object.__setattr__(node, "outputs", outputs)

        return node, outputs
