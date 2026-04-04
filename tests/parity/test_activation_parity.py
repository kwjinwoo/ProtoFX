"""ORT numerical parity tests for activation op handlers.

Covers: Relu, Softmax, LogSoftmax, Gelu, Elu, LeakyRelu, Selu, Celu, PRelu,
HardSigmoid, HardSwish, Mish, Softplus, Softsign, ThresholdedRelu.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from tests.parity.conftest import assert_parity

_OPSET = helper.make_opsetid("", 20)


def _make_unary_model(op_type: str, opset: int = 20, **attrs: int | float | str) -> helper.ModelProto:
    """Build a minimal ONNX model: X → Op(attrs) → Y."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(op_type, ["X"], ["Y"], **attrs)
    graph = helper.make_graph([node], f"{op_type}_test", [X], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


# ---------------------------------------------------------------------------
# Simple unary activations (no special attributes)
# ---------------------------------------------------------------------------

_SIMPLE_ACTIVATIONS = ["Relu", "Selu", "HardSwish", "Mish", "Softplus"]


class TestSimpleActivationParity:
    """ORT parity for simple activation ops without special attributes."""

    @pytest.mark.parametrize("op_type", _SIMPLE_ACTIVATIONS)
    def test_parity(self, op_type: str) -> None:
        """Simple activation must match ORT output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model(op_type)
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Activations with axis attribute
# ---------------------------------------------------------------------------


class TestSoftmaxParity:
    """ORT parity for Softmax op."""

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_parity(self, axis: int) -> None:
        """Softmax must match ORT at different axes."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("Softmax", axis=axis)
        assert_parity(model, {"X": x})


class TestLogSoftmaxParity:
    """ORT parity for LogSoftmax op."""

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_parity(self, axis: int) -> None:
        """LogSoftmax must match ORT at different axes."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("LogSoftmax", axis=axis)
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Gelu variants
# ---------------------------------------------------------------------------


class TestGeluParity:
    """ORT parity for Gelu op."""

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_parity(self, approximate: str) -> None:
        """Gelu must match ORT with exact and tanh approximation."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("Gelu", approximate=approximate)
        assert_parity(model, {"X": x}, atol=1e-5)


# ---------------------------------------------------------------------------
# Activations with alpha/slope attributes
# ---------------------------------------------------------------------------


class TestEluParity:
    """ORT parity for Elu op."""

    @pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
    def test_parity(self, alpha: float) -> None:
        """Elu must match ORT with various alpha values."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("Elu", alpha=alpha)
        assert_parity(model, {"X": x})


class TestLeakyReluParity:
    """ORT parity for LeakyRelu op."""

    @pytest.mark.parametrize("alpha", [0.01, 0.1, 0.3])
    def test_parity(self, alpha: float) -> None:
        """LeakyRelu must match ORT with various alpha values."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("LeakyRelu", alpha=alpha)
        assert_parity(model, {"X": x})


class TestCeluParity:
    """ORT parity for Celu op."""

    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_parity(self, alpha: float) -> None:
        """Celu must match ORT with various alpha values."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("Celu", opset=12, alpha=alpha)
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# PRelu (two inputs)
# ---------------------------------------------------------------------------


class TestPReluParity:
    """ORT parity for PRelu op."""

    def test_parity(self) -> None:
        """PRelu must match ORT output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        slope = rng.uniform(0.01, 0.5, size=(1, 4)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        S = helper.make_tensor_value_info("slope", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        node = helper.make_node("PRelu", ["X", "slope"], ["Y"])
        graph = helper.make_graph([node], "prelu_test", [X, S], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x, "slope": slope})


# ---------------------------------------------------------------------------
# HardSigmoid
# ---------------------------------------------------------------------------


class TestHardSigmoidParity:
    """ORT parity for HardSigmoid op."""

    def test_default_params(self) -> None:
        """HardSigmoid with default alpha/beta must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("HardSigmoid")
        assert_parity(model, {"X": x})

    def test_custom_params(self) -> None:
        """HardSigmoid with custom alpha/beta must match ORT."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("HardSigmoid", alpha=0.1, beta=0.3)
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# Softsign
# ---------------------------------------------------------------------------


class TestSoftsignParity:
    """ORT parity for Softsign op."""

    def test_parity(self) -> None:
        """Softsign must match ORT output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("Softsign")
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# ThresholdedRelu
# ---------------------------------------------------------------------------


class TestThresholdedReluParity:
    """ORT parity for ThresholdedRelu op."""

    @pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
    def test_parity(self, alpha: float) -> None:
        """ThresholdedRelu must match ORT with various alpha values."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        model = _make_unary_model("ThresholdedRelu", opset=10, alpha=alpha)
        assert_parity(model, {"X": x})
