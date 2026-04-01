"""Tests for normalization op handlers (BatchNormalization, LayerNormalization)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# BatchNormalization helpers
# ---------------------------------------------------------------------------


def _make_bn_graph(
    *,
    x_shape: tuple[int, ...],
    num_features: int,
    epsilon: float | None = None,
    training_mode: int | None = None,
) -> Graph:
    """Build a minimal IR graph: (X, scale, B, mean, var) -> BatchNormalization -> Y.

    Args:
        x_shape: Shape of the input tensor X.
        num_features: Number of channels (C dimension).
        epsilon: Optional epsilon attribute. Uses ONNX default (1e-5) when ``None``.
        training_mode: Optional training_mode attribute. ``0`` for inference (default).
    """
    g = Graph(name="BatchNormalization_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=x_shape), name="X")
    scale = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(num_features,)), name="scale")
    b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(num_features,)), name="B")
    mean = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(num_features,)), name="input_mean")
    var = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=(num_features,)), name="input_var")

    attributes: dict[str, int | float] = {}
    if epsilon is not None:
        attributes["epsilon"] = epsilon
    if training_mode is not None:
        attributes["training_mode"] = training_mode

    y_shape = x_shape
    node = g.make_node(
        op_type="BatchNormalization",
        inputs=[x, scale, b, mean, var],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=y_shape)],
        output_names=["Y"],
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# BatchNormalization tests
# ---------------------------------------------------------------------------


class TestBatchNormalizationHandler:
    """Verify that the BatchNormalization op handler emits correct FX nodes."""

    def test_bn_basic(self) -> None:
        """BatchNormalization with default epsilon must produce correct results."""
        g = _make_bn_graph(x_shape=(1, 3, 4, 4), num_features=3)
        gm = emit_graph(g)
        x = torch.randn(1, 3, 4, 4)
        scale = torch.randn(3)
        bias = torch.randn(3)
        mean = torch.randn(3)
        var = torch.rand(3).abs() + 0.01
        (result,) = gm(x, scale, bias, mean, var)
        expected = F.batch_norm(x, mean, var, weight=scale, bias=bias, training=False, eps=1e-5)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_bn_custom_epsilon(self) -> None:
        """BatchNormalization with custom epsilon must produce correct results."""
        eps = 1e-3
        g = _make_bn_graph(x_shape=(2, 4, 8, 8), num_features=4, epsilon=eps)
        gm = emit_graph(g)
        x = torch.randn(2, 4, 8, 8)
        scale = torch.randn(4)
        bias = torch.randn(4)
        mean = torch.randn(4)
        var = torch.rand(4).abs() + 0.01
        (result,) = gm(x, scale, bias, mean, var)
        expected = F.batch_norm(x, mean, var, weight=scale, bias=bias, training=False, eps=eps)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_bn_1d(self) -> None:
        """BatchNormalization on 1D input (N, C, L) must produce correct results."""
        g = _make_bn_graph(x_shape=(1, 2, 10), num_features=2)
        gm = emit_graph(g)
        x = torch.randn(1, 2, 10)
        scale = torch.randn(2)
        bias = torch.randn(2)
        mean = torch.randn(2)
        var = torch.rand(2).abs() + 0.01
        (result,) = gm(x, scale, bias, mean, var)
        expected = F.batch_norm(x, mean, var, weight=scale, bias=bias, training=False, eps=1e-5)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_bn_training_mode_raises(self) -> None:
        """BatchNormalization with training_mode=1 must raise NotImplementedError."""
        g = _make_bn_graph(x_shape=(1, 3, 4, 4), num_features=3, training_mode=1)
        with pytest.raises(NotImplementedError, match="BatchNormalization"):
            emit_graph(g)
