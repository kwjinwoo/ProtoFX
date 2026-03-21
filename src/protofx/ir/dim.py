"""Dimension type for ProtoFX IR shapes.

A dimension can be:
- ``int``  — a concrete static size (e.g. ``32``)
- ``str``  — a symbolic named dimension (e.g. ``"batch"``)
- ``None`` — an unknown dimension
"""

type Dim = int | str | None
"""Type alias representing a single tensor dimension."""


def is_static_dim(dim: Dim) -> bool:
    """Return ``True`` if *dim* is a concrete integer size.

    Args:
        dim: A dimension value.

    Returns:
        ``True`` when *dim* is an ``int``, ``False`` otherwise.
    """
    return isinstance(dim, int)
