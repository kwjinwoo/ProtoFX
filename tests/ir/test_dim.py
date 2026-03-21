"""Tests for protofx.ir.Dim type alias and helpers."""

from protofx.ir import Dim


class TestDimTypeAlias:
    """Verify Dim accepts the three expected kinds."""

    def test_int_dim(self) -> None:
        """A concrete integer dimension."""
        d: Dim = 32
        assert d == 32

    def test_symbolic_dim(self) -> None:
        """A symbolic string dimension."""
        d: Dim = "batch"
        assert d == "batch"

    def test_unknown_dim(self) -> None:
        """None represents an unknown dimension."""
        d: Dim = None
        assert d is None


class TestDimIsStatic:
    """Verify the is_static_dim helper."""

    def test_int_is_static(self) -> None:
        from protofx.ir.dim import is_static_dim

        assert is_static_dim(32) is True

    def test_symbolic_is_not_static(self) -> None:
        from protofx.ir.dim import is_static_dim

        assert is_static_dim("batch") is False

    def test_none_is_not_static(self) -> None:
        from protofx.ir.dim import is_static_dim

        assert is_static_dim(None) is False
