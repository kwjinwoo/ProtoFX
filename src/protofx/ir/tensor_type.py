"""TensorType for ProtoFX IR tensor metadata.

``TensorType`` holds the element data type and shape of a tensor value.
It is backend-neutral — it depends only on ``ir.DType`` and ``ir.Shape``,
not on ``torch`` or ``onnx``.
"""

from dataclasses import dataclass

from protofx.ir.dtype import DType
from protofx.ir.shape import Shape


@dataclass(frozen=True)
class TensorType:
    """Immutable tensor metadata attached to an IR ``Value``.

    Attributes:
        dtype: Element data type, or ``None`` if unknown.
        shape: Tensor shape, or ``None`` if entirely unknown.
    """

    dtype: DType | None
    shape: Shape
