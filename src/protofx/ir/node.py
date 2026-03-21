"""Stub for ProtoFX IR Node.

This module provides a minimal ``Node`` class used as a type reference by
``Value.producer``. The full ``Node`` implementation will be added in a
later milestone.
"""

type AttributeValue = int | float | str | bytes | list[int] | list[float] | list[str] | list[bytes]
"""Normalized ONNX attribute value type.

All node attributes are converted to Python-native forms during import.
The emitter must not depend on raw ``onnx.AttributeProto`` structures.
"""


class Node:
    """Placeholder IR node.

    This stub exists so that ``Value`` can reference ``Node`` as its producer
    type without a forward-reference string. The real implementation will
    replace this class.
    """
