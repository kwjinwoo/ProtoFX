"""Tests for protofx.ir.Shape type alias and helpers."""

from protofx.ir import Shape
from protofx.ir.shape import is_fully_known, rank


class TestShapeFiveCases:
    """IR invariant 6: TensorType must distinguish these five shape cases."""

    def test_unknown_shape(self) -> None:
        """None represents a completely unknown shape."""
        s: Shape = None
        assert s is None

    def test_scalar_shape(self) -> None:
        """Empty tuple represents a scalar (0-d tensor)."""
        s: Shape = ()
        assert s == ()

    def test_empty_dimension(self) -> None:
        """A dimension of size 0 is a valid concrete shape."""
        s: Shape = (0,)
        assert s == (0,)

    def test_partially_known(self) -> None:
        """Mix of unknown (None) and concrete dims."""
        s: Shape = (None, 3)
        assert s == (None, 3)

    def test_fully_known(self) -> None:
        """All dimensions are concrete integers."""
        s: Shape = (2, 3)
        assert s == (2, 3)

    def test_symbolic_dimension(self) -> None:
        """A symbolic string dimension."""
        s: Shape = ("batch", 3, 224, 224)
        assert s[0] == "batch"


class TestRank:
    """Verify rank() helper."""

    def test_unknown_shape_rank_is_none(self) -> None:
        assert rank(None) is None

    def test_scalar_rank_is_zero(self) -> None:
        assert rank(()) == 0

    def test_known_shape_rank(self) -> None:
        assert rank((2, 3, 4)) == 3

    def test_partial_shape_rank(self) -> None:
        assert rank((None, 3)) == 2


class TestIsFullyKnown:
    """Verify is_fully_known() helper."""

    def test_unknown_shape(self) -> None:
        assert is_fully_known(None) is False

    def test_scalar_is_fully_known(self) -> None:
        assert is_fully_known(()) is True

    def test_fully_known(self) -> None:
        assert is_fully_known((2, 3, 4)) is True

    def test_partially_known_is_not_fully_known(self) -> None:
        assert is_fully_known((None, 3)) is False

    def test_symbolic_is_not_fully_known(self) -> None:
        assert is_fully_known(("batch", 3)) is False
