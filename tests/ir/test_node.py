"""Tests for protofx.ir.node.AttributeValue type alias."""

from protofx.ir.node import AttributeValue


class TestAttributeValueTypeAlias:
    """Verify AttributeValue type alias exists and accepts expected types."""

    def test_alias_exists(self) -> None:
        """AttributeValue must be importable from ir.node."""
        assert AttributeValue is not None

    def test_int_is_valid(self) -> None:
        """Plain int should satisfy AttributeValue."""
        v: AttributeValue = 42
        assert isinstance(v, int)

    def test_float_is_valid(self) -> None:
        """Plain float should satisfy AttributeValue."""
        v: AttributeValue = 3.14
        assert isinstance(v, float)

    def test_bytes_is_valid(self) -> None:
        """Plain bytes should satisfy AttributeValue."""
        v: AttributeValue = b"\x00\x01"
        assert isinstance(v, bytes)

    def test_str_is_valid(self) -> None:
        """Plain str should satisfy AttributeValue."""
        v: AttributeValue = "relu"
        assert isinstance(v, str)

    def test_list_int_is_valid(self) -> None:
        """list[int] should satisfy AttributeValue."""
        v: AttributeValue = [1, 2, 3]
        assert isinstance(v, list)

    def test_list_float_is_valid(self) -> None:
        """list[float] should satisfy AttributeValue."""
        v: AttributeValue = [1.0, 2.0]
        assert isinstance(v, list)

    def test_list_bytes_is_valid(self) -> None:
        """list[bytes] should satisfy AttributeValue."""
        v: AttributeValue = [b"\x00", b"\x01"]
        assert isinstance(v, list)

    def test_list_str_is_valid(self) -> None:
        """list[str] should satisfy AttributeValue."""
        v: AttributeValue = ["a", "b"]
        assert isinstance(v, list)
