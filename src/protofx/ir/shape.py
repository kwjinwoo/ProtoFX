"""Shape type for ProtoFX IR tensor metadata.

A shape can be:
- ``None``                — completely unknown shape
- ``()``                  — scalar (0-d tensor)
- ``(2, 3)``              — fully known shape
- ``(None, 3)``           — partially known (unknown first dim)
- ``("batch", 3, 224)``   — symbolic + concrete mix
"""

import enum

from protofx.ir.dim import Dim, is_static_dim

type Shape = tuple[Dim, ...] | None
"""Type alias representing a tensor shape, or ``None`` when entirely unknown."""


class ShapeCompatibility(enum.Enum):
    """Tri-state compatibility result for two shape metadata values."""

    COMPATIBLE = enum.auto()
    INCOMPATIBLE = enum.auto()
    UNKNOWN = enum.auto()


def rank(shape: Shape) -> int | None:
    """Return the number of dimensions, or ``None`` if *shape* is unknown.

    Args:
        shape: A shape value.

    Returns:
        The rank as an ``int``, or ``None`` when *shape* is ``None``.
    """
    if shape is None:
        return None
    return len(shape)


def is_fully_known(shape: Shape) -> bool:
    """Return ``True`` if every dimension in *shape* is a concrete integer.

    A scalar shape ``()`` is considered fully known.
    An unknown shape (``None``) is not fully known.

    Args:
        shape: A shape value.

    Returns:
        ``True`` when all dimensions are static integers.
    """
    if shape is None:
        return False
    return all(is_static_dim(d) for d in shape)


def compare_shapes(lhs: Shape, rhs: Shape) -> ShapeCompatibility:
    """Compare two shapes with tri-state compatibility semantics.

    Args:
        lhs: First shape metadata.
        rhs: Second shape metadata.

    Returns:
        ``ShapeCompatibility.INCOMPATIBLE`` when mismatch is provable from known metadata,
        ``ShapeCompatibility.UNKNOWN`` when compatibility cannot be proven either way,
        otherwise ``ShapeCompatibility.COMPATIBLE``.
    """
    if lhs is None or rhs is None:
        return ShapeCompatibility.UNKNOWN
    if len(lhs) != len(rhs):
        return ShapeCompatibility.INCOMPATIBLE

    has_unknown = False
    for left_dim, right_dim in zip(lhs, rhs, strict=True):
        if left_dim is None or right_dim is None:
            has_unknown = True
            continue
        if isinstance(left_dim, str) or isinstance(right_dim, str):
            has_unknown = True
            continue
        if left_dim != right_dim:
            return ShapeCompatibility.INCOMPATIBLE

    if has_unknown:
        return ShapeCompatibility.UNKNOWN
    return ShapeCompatibility.COMPATIBLE
