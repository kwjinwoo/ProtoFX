"""Tests for protofx.ir.DType enum."""

from protofx.ir import DType


class TestDTypeMembers:
    """Verify all expected DType enum members exist."""

    def test_float16(self) -> None:
        assert DType.FLOAT16.value == 10

    def test_float32(self) -> None:
        assert DType.FLOAT32.value == 1

    def test_float64(self) -> None:
        assert DType.FLOAT64.value == 11

    def test_bfloat16(self) -> None:
        assert DType.BFLOAT16.value == 16

    def test_int8(self) -> None:
        assert DType.INT8.value == 3

    def test_int16(self) -> None:
        assert DType.INT16.value == 5

    def test_int32(self) -> None:
        assert DType.INT32.value == 6

    def test_int64(self) -> None:
        assert DType.INT64.value == 7

    def test_uint8(self) -> None:
        assert DType.UINT8.value == 2

    def test_uint16(self) -> None:
        assert DType.UINT16.value == 4

    def test_uint32(self) -> None:
        assert DType.UINT32.value == 12

    def test_uint64(self) -> None:
        assert DType.UINT64.value == 13

    def test_bool(self) -> None:
        assert DType.BOOL.value == 9

    def test_complex64(self) -> None:
        assert DType.COMPLEX64.value == 14

    def test_complex128(self) -> None:
        assert DType.COMPLEX128.value == 15

    def test_float8e4m3fn(self) -> None:
        assert DType.FLOAT8E4M3FN.value == 17

    def test_float8e5m2(self) -> None:
        assert DType.FLOAT8E5M2.value == 19

    def test_string(self) -> None:
        assert DType.STRING.value == 8


class TestDTypeBehavior:
    """Verify DType enum behavior."""

    def test_member_count(self) -> None:
        """Ensure enum has exactly the expected number of members."""
        assert len(DType) == 18

    def test_members_are_unique(self) -> None:
        """Ensure all enum values are unique."""
        values = [m.value for m in DType]
        assert len(values) == len(set(values))

    def test_repr(self) -> None:
        assert "FLOAT32" in repr(DType.FLOAT32)

    def test_from_value(self) -> None:
        """Ensure DType can be constructed from int value."""
        assert DType(1) is DType.FLOAT32

    def test_invalid_value_raises(self) -> None:
        """Ensure invalid int raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            DType(9999)
