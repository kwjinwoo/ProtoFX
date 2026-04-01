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


# ---------------------------------------------------------------------------
# LayerNormalization helpers
# ---------------------------------------------------------------------------


def _make_ln_graph(
    *,
    x_shape: tuple[int, ...],
    axis: int = -1,
    epsilon: float | None = None,
    has_bias: bool = True,
) -> Graph:
    """Build a minimal IR graph: (X, scale [, B]) -> LayerNormalization -> Y.

    Args:
        x_shape: Shape of the input tensor X.
        axis: The axis attribute. Defaults to ``-1``.
        epsilon: Optional epsilon attribute. Uses ONNX default (1e-5) when ``None``.
        has_bias: Whether to include the bias input. When ``False``, a sentinel is used.
    """
    g = Graph(name="LayerNormalization_test")

    # Compute normalized_shape from axis
    ndim = len(x_shape)
    resolved_axis = axis if axis >= 0 else ndim + axis
    norm_shape = x_shape[resolved_axis:]

    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=x_shape), name="X")
    scale = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=norm_shape), name="scale")

    if has_bias:
        b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=norm_shape), name="B")
    else:
        b = g.add_sentinel()

    attributes: dict[str, int | float] = {"axis": axis}
    if epsilon is not None:
        attributes["epsilon"] = epsilon

    node = g.make_node(
        op_type="LayerNormalization",
        inputs=[x, scale, b],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=x_shape)],
        output_names=["Y"],
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# LayerNormalization tests
# ---------------------------------------------------------------------------


class TestLayerNormalizationHandler:
    """Verify that the LayerNormalization op handler emits correct FX nodes."""

    def test_ln_axis_neg1(self) -> None:
        """LayerNormalization with axis=-1 (default) must produce correct results."""
        x_shape = (2, 3, 4)
        g = _make_ln_graph(x_shape=x_shape, axis=-1)
        gm = emit_graph(g)
        x = torch.randn(*x_shape)
        scale = torch.randn(x_shape[-1])
        bias = torch.randn(x_shape[-1])
        (result,) = gm(x, scale, bias)
        expected = F.layer_norm(x, [x_shape[-1]], weight=scale, bias=bias, eps=1e-5)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_ln_axis_1(self) -> None:
        """LayerNormalization with axis=1 must normalize over last N-1 dims."""
        x_shape = (2, 3, 4)
        g = _make_ln_graph(x_shape=x_shape, axis=1)
        gm = emit_graph(g)
        x = torch.randn(*x_shape)
        norm_shape = x_shape[1:]  # (3, 4)
        scale = torch.randn(*norm_shape)
        bias = torch.randn(*norm_shape)
        (result,) = gm(x, scale, bias)
        expected = F.layer_norm(x, list(norm_shape), weight=scale, bias=bias, eps=1e-5)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_ln_custom_epsilon(self) -> None:
        """LayerNormalization with custom epsilon must produce correct results."""
        eps = 1e-3
        x_shape = (1, 4, 8)
        g = _make_ln_graph(x_shape=x_shape, axis=-1, epsilon=eps)
        gm = emit_graph(g)
        x = torch.randn(*x_shape)
        scale = torch.randn(x_shape[-1])
        bias = torch.randn(x_shape[-1])
        (result,) = gm(x, scale, bias)
        expected = F.layer_norm(x, [x_shape[-1]], weight=scale, bias=bias, eps=eps)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_ln_no_bias(self) -> None:
        """LayerNormalization without bias must produce correct results."""
        x_shape = (2, 3, 4)
        g = _make_ln_graph(x_shape=x_shape, axis=-1, has_bias=False)
        gm = emit_graph(g)
        x = torch.randn(*x_shape)
        scale = torch.randn(x_shape[-1])
        (result,) = gm(x, scale)
        expected = F.layer_norm(x, [x_shape[-1]], weight=scale, bias=None, eps=1e-5)
        assert torch.allclose(result, expected, atol=1e-6)
