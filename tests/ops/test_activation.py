"""Tests for activation op handlers.

Covers Relu, Softmax, LogSoftmax, Gelu, Elu, LeakyRelu, Selu, Celu, PRelu,
HardSigmoid, HardSwish, Mish, Softplus, Softsign, and ThresholdedRelu.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Helper: build a simple unary activation graph (no attributes)
# ---------------------------------------------------------------------------


def _make_unary_graph(op_type: str) -> Graph:
    """Build a minimal IR graph: X → Op → Y."""
    g = Graph(name=f"{op_type}_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
    node = g.make_node(
        op_type=op_type,
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


def _make_attr_graph(op_type: str, attributes: dict[str, int | float | str]) -> Graph:
    """Build a minimal IR graph: X → Op(attributes) → Y."""
    g = Graph(name=f"{op_type}_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
    node = g.make_node(
        op_type=op_type,
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
        output_names=["Y"],
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# Simple unary activations (no attributes, direct torch function mapping)
# ---------------------------------------------------------------------------

_SIMPLE_ACTIVATIONS: list[tuple[str, Callable[..., torch.Tensor]]] = [
    ("Relu", F.relu),
    ("Mish", F.mish),
    ("Softplus", F.softplus),
]


class TestSimpleActivations:
    """Verify simple activation ops that need no attributes."""

    @pytest.mark.parametrize(("op_type", "torch_fn"), _SIMPLE_ACTIVATIONS, ids=[o[0] for o in _SIMPLE_ACTIVATIONS])
    def test_emits_call_function(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Activation op must emit a call_function FX node."""
        g = _make_unary_graph(op_type)
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    @pytest.mark.parametrize(("op_type", "torch_fn"), _SIMPLE_ACTIVATIONS, ids=[o[0] for o in _SIMPLE_ACTIVATIONS])
    def test_single_output(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """Activation handler must return exactly one FX output node."""
        g = _make_unary_graph(op_type)
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    @pytest.mark.parametrize(("op_type", "torch_fn"), _SIMPLE_ACTIVATIONS, ids=[o[0] for o in _SIMPLE_ACTIVATIONS])
    def test_forward_correctness(self, op_type: str, torch_fn: Callable[..., torch.Tensor]) -> None:
        """The emitted GraphModule must produce correct numerical results."""
        g = _make_unary_graph(op_type)
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = torch_fn(x)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Softmax / LogSoftmax — axis attribute, negative axis support
# ---------------------------------------------------------------------------


class TestSoftmax:
    """Verify Softmax handler with axis attribute."""

    def test_softmax_default_axis(self) -> None:
        """Softmax with default axis=1 must produce correct results."""
        g = _make_unary_graph("Softmax")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = F.softmax(x, dim=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_softmax_axis_0(self) -> None:
        """Softmax with axis=0 must produce correct results."""
        g = _make_attr_graph("Softmax", {"axis": 0})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = F.softmax(x, dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_softmax_negative_axis(self) -> None:
        """Softmax with negative axis must produce correct results."""
        g = _make_attr_graph("Softmax", {"axis": -1})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = F.softmax(x, dim=-1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_softmax_emits_call_function(self) -> None:
        """Softmax must emit a call_function FX node."""
        g = _make_unary_graph("Softmax")
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops


class TestLogSoftmax:
    """Verify LogSoftmax handler with axis attribute."""

    def test_logsoftmax_default_axis(self) -> None:
        """LogSoftmax with default axis=1 must produce correct results."""
        g = _make_unary_graph("LogSoftmax")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = F.log_softmax(x, dim=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_logsoftmax_negative_axis(self) -> None:
        """LogSoftmax with negative axis must produce correct results."""
        g = _make_attr_graph("LogSoftmax", {"axis": -1})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (result,) = gm(x)
        expected = F.log_softmax(x, dim=-1)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gelu — approximate attribute (opset 20)
# ---------------------------------------------------------------------------


class TestGelu:
    """Verify Gelu handler with approximate attribute."""

    def test_gelu_default(self) -> None:
        """Gelu with default approximate='none' must produce correct results."""
        g = _make_unary_graph("Gelu")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.gelu(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_gelu_approximate_tanh(self) -> None:
        """Gelu with approximate='tanh' must produce correct results."""
        g = _make_attr_graph("Gelu", {"approximate": "tanh"})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.gelu(x, approximate="tanh")
        assert torch.allclose(result, expected, atol=1e-6)

    def test_gelu_approximate_none_explicit(self) -> None:
        """Gelu with explicit approximate='none' must produce correct results."""
        g = _make_attr_graph("Gelu", {"approximate": "none"})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.gelu(x, approximate="none")
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Elu — alpha attribute
# ---------------------------------------------------------------------------


class TestElu:
    """Verify Elu handler with alpha attribute."""

    def test_elu_default_alpha(self) -> None:
        """Elu with default alpha=1.0 must produce correct results."""
        g = _make_unary_graph("Elu")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.elu(x, alpha=1.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_elu_custom_alpha(self) -> None:
        """Elu with custom alpha must produce correct results."""
        g = _make_attr_graph("Elu", {"alpha": 0.5})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.elu(x, alpha=0.5)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# LeakyRelu — alpha attribute
# ---------------------------------------------------------------------------


class TestLeakyRelu:
    """Verify LeakyRelu handler with alpha attribute."""

    def test_leakyrelu_default_alpha(self) -> None:
        """LeakyRelu with default alpha=0.01 must produce correct results."""
        g = _make_unary_graph("LeakyRelu")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.leaky_relu(x, negative_slope=0.01)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_leakyrelu_custom_alpha(self) -> None:
        """LeakyRelu with custom alpha must produce correct results."""
        g = _make_attr_graph("LeakyRelu", {"alpha": 0.2})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.leaky_relu(x, negative_slope=0.2)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Selu
# ---------------------------------------------------------------------------


class TestSelu:
    """Verify Selu handler (fixed alpha and gamma per ONNX spec)."""

    def test_selu_correctness(self) -> None:
        """Selu must produce correct results matching F.selu."""
        g = _make_unary_graph("Selu")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.selu(x)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Celu — alpha attribute
# ---------------------------------------------------------------------------


class TestCelu:
    """Verify Celu handler with alpha attribute."""

    def test_celu_default_alpha(self) -> None:
        """Celu with default alpha=1.0 must produce correct results."""
        g = _make_unary_graph("Celu")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.celu(x, alpha=1.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_celu_custom_alpha(self) -> None:
        """Celu with custom alpha must produce correct results."""
        g = _make_attr_graph("Celu", {"alpha": 2.0})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.celu(x, alpha=2.0)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# PRelu — slope as second input
# ---------------------------------------------------------------------------


def _make_prelu_graph() -> Graph:
    """Build a minimal IR graph: (X, slope) → PRelu → Y."""
    g = Graph(name="PRelu_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(2, 3)), name="X")
    slope = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(3,)), name="slope")
    node = g.make_node(
        op_type="PRelu",
        inputs=[x, slope],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=(2, 3))],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestPRelu:
    """Verify PRelu handler with slope as second input."""

    def test_prelu_correctness(self) -> None:
        """PRelu must produce correct results matching F.prelu."""
        g = _make_prelu_graph()
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        slope = torch.tensor([0.1, 0.2, 0.3])
        (result,) = gm(x, slope)
        expected = F.prelu(x, slope)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_prelu_emits_call_function(self) -> None:
        """PRelu must emit a call_function FX node."""
        g = _make_prelu_graph()
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops


# ---------------------------------------------------------------------------
# HardSigmoid — alpha, beta attributes (manual clamp formula)
# ---------------------------------------------------------------------------


class TestHardSigmoid:
    """Verify HardSigmoid handler with alpha/beta attributes."""

    def test_hardsigmoid_default(self) -> None:
        """HardSigmoid with default alpha=0.2, beta=0.5 must produce correct results."""
        g = _make_unary_graph("HardSigmoid")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -3.0, 2.0], [-1.0, 0.5, 5.0]])
        (result,) = gm(x)
        expected = torch.clamp(0.2 * x + 0.5, 0.0, 1.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_hardsigmoid_custom_params(self) -> None:
        """HardSigmoid with custom alpha/beta must produce correct results."""
        g = _make_attr_graph("HardSigmoid", {"alpha": 0.166667, "beta": 0.5})
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -3.0, 2.0], [-1.0, 0.5, 5.0]])
        (result,) = gm(x)
        expected = torch.clamp(0.166667 * x + 0.5, 0.0, 1.0)
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# HardSwish
# ---------------------------------------------------------------------------


class TestHardSwish:
    """Verify HardSwish handler."""

    def test_hardswish_correctness(self) -> None:
        """HardSwish must produce correct results matching F.hardswish."""
        g = _make_unary_graph("HardSwish")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -3.0, 2.0], [-1.0, 0.5, 5.0]])
        (result,) = gm(x)
        expected = F.hardswish(x)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Mish
# ---------------------------------------------------------------------------


class TestMish:
    """Verify Mish handler."""

    def test_mish_correctness(self) -> None:
        """Mish must produce correct results matching F.mish."""
        g = _make_unary_graph("Mish")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.mish(x)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Softplus
# ---------------------------------------------------------------------------


class TestSoftplus:
    """Verify Softplus handler."""

    def test_softplus_correctness(self) -> None:
        """Softplus must produce correct results matching F.softplus."""
        g = _make_unary_graph("Softplus")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = F.softplus(x)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Softsign — manual formula: x / (1 + abs(x))
# ---------------------------------------------------------------------------


class TestSoftsign:
    """Verify Softsign handler (manual formula)."""

    def test_softsign_correctness(self) -> None:
        """Softsign must produce correct results: x / (1 + abs(x))."""
        g = _make_unary_graph("Softsign")
        gm = emit_graph(g)
        x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = x / (1.0 + torch.abs(x))
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# ThresholdedRelu — alpha attribute
# ---------------------------------------------------------------------------


class TestThresholdedRelu:
    """Verify ThresholdedRelu handler with alpha attribute."""

    def test_thresholdedrelu_default(self) -> None:
        """ThresholdedRelu with default alpha=1.0 must produce correct results."""
        g = _make_unary_graph("ThresholdedRelu")
        gm = emit_graph(g)
        x = torch.tensor([[0.5, 1.5, 2.0], [-1.0, 1.0, 3.0]])
        (result,) = gm(x)
        # ThresholdedRelu: y = x if x > alpha, else 0
        expected = torch.where(x > 1.0, x, torch.zeros_like(x))
        assert torch.allclose(result, expected, atol=1e-6)

    def test_thresholdedrelu_custom_alpha(self) -> None:
        """ThresholdedRelu with custom alpha must produce correct results."""
        g = _make_attr_graph("ThresholdedRelu", {"alpha": 0.5})
        gm = emit_graph(g)
        x = torch.tensor([[0.3, 0.6, 2.0], [-1.0, 0.5, 3.0]])
        (result,) = gm(x)
        expected = torch.where(x > 0.5, x, torch.zeros_like(x))
        assert torch.allclose(result, expected, atol=1e-6)
