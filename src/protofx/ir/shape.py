"""Shape type for ProtoFX IR tensor metadata.

A shape can be:
- ``None``                — completely unknown shape
- ``()``                  — scalar (0-d tensor)
- ``(2, 3)``              — fully known shape
- ``(None, 3)``           — partially known (unknown first dim)
- ``("batch", 3, 224)``   — symbolic + concrete mix
"""

from protofx.ir.dim import Dim, is_static_dim

type Shape = tuple[Dim, ...] | None
"""Type alias representing a tensor shape, or ``None`` when entirely unknown."""


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
