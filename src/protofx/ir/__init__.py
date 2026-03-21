"""ProtoFX intermediate representation."""

from protofx.ir.dim import Dim
from protofx.ir.dtype import DType
from protofx.ir.shape import Shape
from protofx.ir.tensor_type import TensorType

__all__ = ["DType", "Dim", "Shape", "TensorType"]
