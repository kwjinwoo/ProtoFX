"""Tests for protofx.ir.TensorType."""

import pytest

from protofx.ir import DType, TensorType


class TestTensorTypeCreation:
    """Verify TensorType construction."""

    def test_fully_specified(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        assert t.dtype is DType.FLOAT32
        assert t.shape == (2, 3)

    def test_unknown_dtype(self) -> None:
        t = TensorType(dtype=None, shape=(2, 3))
        assert t.dtype is None

    def test_unknown_shape(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=None)
        assert t.shape is None

    def test_both_unknown(self) -> None:
        t = TensorType(dtype=None, shape=None)
        assert t.dtype is None
        assert t.shape is None

    def test_scalar(self) -> None:
        t = TensorType(dtype=DType.INT64, shape=())
        assert t.shape == ()

    def test_symbolic_shape(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=("batch", 3, 224, 224))
        assert t.shape == ("batch", 3, 224, 224)


class TestTensorTypeImmutability:
    """TensorType must be frozen (immutable)."""

    def test_cannot_set_dtype(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        with pytest.raises(AttributeError):
            t.dtype = DType.INT32  # type: ignore[misc]

    def test_cannot_set_shape(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        with pytest.raises(AttributeError):
            t.shape = (4, 5)  # type: ignore[misc]


class TestTensorTypeEquality:
    """Verify equality semantics."""

    def test_equal(self) -> None:
        a = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        b = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        assert a == b

    def test_not_equal_dtype(self) -> None:
        a = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        b = TensorType(dtype=DType.INT32, shape=(2, 3))
        assert a != b

    def test_not_equal_shape(self) -> None:
        a = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        b = TensorType(dtype=DType.FLOAT32, shape=(3, 2))
        assert a != b

    def test_hashable(self) -> None:
        """Frozen dataclass should be hashable."""
        t = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        assert hash(t) == hash(TensorType(dtype=DType.FLOAT32, shape=(2, 3)))


class TestTensorTypeRepr:
    """Verify repr output."""

    def test_repr_contains_dtype(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        assert "FLOAT32" in repr(t)

    def test_repr_contains_shape(self) -> None:
        t = TensorType(dtype=DType.FLOAT32, shape=(2, 3))
        assert "(2, 3)" in repr(t)

    def test_repr_unknown_dtype(self) -> None:
        t = TensorType(dtype=None, shape=(2, 3))
        assert "None" in repr(t)
