"""ORT numerical parity tests for elementwise op handlers.

Covers: Add, Sub, Mul, Div, Pow, Sigmoid, Tanh, Abs, Neg, Exp, Log, Sqrt.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from tests.parity.conftest import assert_parity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPSET = helper.make_opsetid("", 17)


def _make_binary_model(op_type: str) -> helper.ModelProto:
    """Build a minimal ONNX model with one binary elementwise op: (A, B) → Y."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_type, ["A", "B"], ["Y"])
    graph = helper.make_graph([node], f"{op_type}_test", [A, B], [Y])
    return helper.make_model(graph, opset_imports=[_OPSET])


def _make_unary_model(op_type: str) -> helper.ModelProto:
    """Build a minimal ONNX model with one unary elementwise op: X → Y."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_type, ["X"], ["Y"])
    graph = helper.make_graph([node], f"{op_type}_test", [X], [Y])
    return helper.make_model(graph, opset_imports=[_OPSET])


# ---------------------------------------------------------------------------
# Binary elementwise ops
# ---------------------------------------------------------------------------

_BINARY_OPS = ["Add", "Sub", "Mul", "Div"]


class TestBinaryElementwiseParity:
    """ORT parity for binary elementwise ops (Add, Sub, Mul, Div)."""

    @pytest.mark.parametrize("op_type", _BINARY_OPS)
    def test_parity(self, op_type: str) -> None:
        """Binary elementwise op must match ORT output."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((2, 3)).astype(np.float32)
        b = rng.standard_normal((2, 3)).astype(np.float32)
        # Avoid division by zero for Div
        if op_type == "Div":
            b = np.where(np.abs(b) < 1e-6, 1.0, b).astype(np.float32)
        model = _make_binary_model(op_type)
        assert_parity(model, {"A": a, "B": b})


class TestPowParity:
    """ORT parity for Pow op."""

    def test_parity(self) -> None:
        """Pow op must match ORT output."""
        rng = np.random.default_rng(42)
        a = np.abs(rng.standard_normal((2, 3))).astype(np.float32) + 0.1
        b = rng.standard_normal((2, 3)).astype(np.float32)
        model = _make_binary_model("Pow")
        assert_parity(model, {"A": a, "B": b})


# ---------------------------------------------------------------------------
# Unary elementwise ops
# ---------------------------------------------------------------------------

_UNARY_OPS_GENERAL = ["Sigmoid", "Tanh", "Abs", "Neg"]


class TestUnaryElementwiseParity:
    """ORT parity for unary elementwise ops that accept general float inputs."""

    @pytest.mark.parametrize("op_type", _UNARY_OPS_GENERAL)
    def test_parity(self, op_type: str) -> None:
        """Unary elementwise op must match ORT output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 3)).astype(np.float32)
        model = _make_unary_model(op_type)
        assert_parity(model, {"X": x})


class TestExpParity:
    """ORT parity for Exp op."""

    def test_parity(self) -> None:
        """Exp op must match ORT output."""
        rng = np.random.default_rng(42)
        # Keep inputs small to avoid overflow
        x = (rng.standard_normal((2, 3)) * 2).astype(np.float32)
        model = _make_unary_model("Exp")
        assert_parity(model, {"X": x})


class TestLogParity:
    """ORT parity for Log op."""

    def test_parity(self) -> None:
        """Log op must match ORT output (positive inputs only)."""
        rng = np.random.default_rng(42)
        x = (np.abs(rng.standard_normal((2, 3))) + 0.01).astype(np.float32)
        model = _make_unary_model("Log")
        assert_parity(model, {"X": x})


class TestSqrtParity:
    """ORT parity for Sqrt op."""

    def test_parity(self) -> None:
        """Sqrt op must match ORT output (non-negative inputs only)."""
        rng = np.random.default_rng(42)
        x = np.abs(rng.standard_normal((2, 3))).astype(np.float32)
        model = _make_unary_model("Sqrt")
        assert_parity(model, {"X": x})
