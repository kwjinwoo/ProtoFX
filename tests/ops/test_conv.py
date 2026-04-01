"""Tests for convolution op handlers (Conv, ConvTranspose)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# Conv helpers
# ---------------------------------------------------------------------------


def _make_conv_graph(
    *,
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    bias_shape: tuple[int, ...] | None = None,
    strides: list[int] | None = None,
    pads: list[int] | None = None,
    dilations: list[int] | None = None,
    group: int = 1,
) -> Graph:
    """Build a minimal IR graph: (X, W [, B]) -> Conv -> Y.

    When *bias_shape* is ``None`` the third input is a sentinel (omitted optional).
    """
    g = Graph(name="Conv_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=x_shape), name="X")
    w = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=w_shape), name="W")

    if bias_shape is not None:
        b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=bias_shape), name="B")
    else:
        b = g.add_sentinel()

    attributes: dict[str, int | float | list[int]] = {}
    if strides is not None:
        attributes["strides"] = strides
    if pads is not None:
        attributes["pads"] = pads
    if dilations is not None:
        attributes["dilations"] = dilations
    if group != 1:
        attributes["group"] = group

    node = g.make_node(
        op_type="Conv",
        inputs=[x, w, b],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=y_shape)],
        output_names=["Y"],
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# Conv tests
# ---------------------------------------------------------------------------


class TestConvHandler:
    """Verify that the Conv op handler emits correct FX nodes."""

    # -- 1D --

    def test_conv1d_basic(self) -> None:
        """Conv1d with default attributes must produce correct results."""
        # X: (N=1, C=1, L=5), W: (OC=1, IC=1, K=3) -> Y: (1, 1, 3)
        g = _make_conv_graph(x_shape=(1, 1, 5), w_shape=(1, 1, 3), y_shape=(1, 1, 3))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 5)
        w = torch.randn(1, 1, 3)
        (result,) = gm(x, w)
        expected = F.conv1d(x, w)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv1d_with_bias(self) -> None:
        """Conv1d with bias must produce correct results."""
        g = _make_conv_graph(x_shape=(1, 1, 5), w_shape=(1, 1, 3), y_shape=(1, 1, 3), bias_shape=(1,))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 5)
        w = torch.randn(1, 1, 3)
        b = torch.randn(1)
        (result,) = gm(x, w, b)
        expected = F.conv1d(x, w, b)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- 2D --

    def test_conv2d_basic(self) -> None:
        """Conv2d with default attributes must produce correct results."""
        # X: (1,1,5,5), W: (1,1,3,3) -> Y: (1,1,3,3)
        g = _make_conv_graph(x_shape=(1, 1, 5, 5), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 3, 3))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 5, 5)
        w = torch.randn(1, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv2d(x, w)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv2d_with_bias(self) -> None:
        """Conv2d with bias must produce correct results."""
        g = _make_conv_graph(x_shape=(1, 1, 5, 5), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 3, 3), bias_shape=(1,))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 5, 5)
        w = torch.randn(1, 1, 3, 3)
        b = torch.randn(1)
        (result,) = gm(x, w, b)
        expected = F.conv2d(x, w, b)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv2d_stride(self) -> None:
        """Conv2d with stride=2 must produce correct results."""
        g = _make_conv_graph(x_shape=(1, 1, 6, 6), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 2, 2), strides=[2, 2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        w = torch.randn(1, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv2d(x, w, stride=2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv2d_padding(self) -> None:
        """Conv2d with explicit padding must produce correct results."""
        g = _make_conv_graph(x_shape=(1, 1, 5, 5), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 5, 5), pads=[1, 1, 1, 1])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 5, 5)
        w = torch.randn(1, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv2d(x, w, padding=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv2d_dilation(self) -> None:
        """Conv2d with dilation must produce correct results."""
        g = _make_conv_graph(x_shape=(1, 1, 7, 7), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 3, 3), dilations=[2, 2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 7, 7)
        w = torch.randn(1, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv2d(x, w, dilation=2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv2d_groups(self) -> None:
        """Conv2d with group > 1 must produce correct results."""
        # Depthwise-like: IC=OC=2, group=2
        g = _make_conv_graph(x_shape=(1, 2, 5, 5), w_shape=(2, 1, 3, 3), y_shape=(1, 2, 3, 3), group=2)
        gm = emit_graph(g)
        x = torch.randn(1, 2, 5, 5)
        w = torch.randn(2, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv2d(x, w, groups=2)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- 3D --

    def test_conv3d_basic(self) -> None:
        """Conv3d with default attributes must produce correct results."""
        # X: (1,1,5,5,5), W: (1,1,3,3,3) -> Y: (1,1,3,3,3)
        g = _make_conv_graph(x_shape=(1, 1, 5, 5, 5), w_shape=(1, 1, 3, 3, 3), y_shape=(1, 1, 3, 3, 3))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 5, 5, 5)
        w = torch.randn(1, 1, 3, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv3d(x, w)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv3d_with_bias(self) -> None:
        """Conv3d with bias must produce correct results."""
        g = _make_conv_graph(x_shape=(1, 1, 5, 5, 5), w_shape=(1, 1, 3, 3, 3), y_shape=(1, 1, 3, 3, 3), bias_shape=(1,))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 5, 5, 5)
        w = torch.randn(1, 1, 3, 3, 3)
        b = torch.randn(1)
        (result,) = gm(x, w, b)
        expected = F.conv3d(x, w, b)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- FX structure checks --

    def test_emits_call_function(self) -> None:
        """Conv must emit a call_function FX node."""
        g = _make_conv_graph(x_shape=(1, 1, 5, 5), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 3, 3))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """Conv handler must return exactly one FX output node."""
        g = _make_conv_graph(x_shape=(1, 1, 5, 5), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 3, 3))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1


# ---------------------------------------------------------------------------
# ConvTranspose helpers
# ---------------------------------------------------------------------------


def _make_conv_transpose_graph(
    *,
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    bias_shape: tuple[int, ...] | None = None,
    strides: list[int] | None = None,
    pads: list[int] | None = None,
    dilations: list[int] | None = None,
    output_padding: list[int] | None = None,
    group: int = 1,
) -> Graph:
    """Build a minimal IR graph: (X, W [, B]) -> ConvTranspose -> Y.

    When *bias_shape* is ``None`` the third input is a sentinel (omitted optional).
    """
    g = Graph(name="ConvTranspose_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=x_shape), name="X")
    w = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=w_shape), name="W")

    if bias_shape is not None:
        b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=bias_shape), name="B")
    else:
        b = g.add_sentinel()

    attributes: dict[str, int | float | list[int]] = {}
    if strides is not None:
        attributes["strides"] = strides
    if pads is not None:
        attributes["pads"] = pads
    if dilations is not None:
        attributes["dilations"] = dilations
    if output_padding is not None:
        attributes["output_padding"] = output_padding
    if group != 1:
        attributes["group"] = group

    node = g.make_node(
        op_type="ConvTranspose",
        inputs=[x, w, b],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=y_shape)],
        output_names=["Y"],
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# ConvTranspose tests
# ---------------------------------------------------------------------------


class TestConvTransposeHandler:
    """Verify that the ConvTranspose op handler emits correct FX nodes."""

    # -- 1D --

    def test_conv_transpose1d_basic(self) -> None:
        """ConvTranspose1d with default attributes must produce correct results."""
        # X: (1,1,3), W: (IC=1,OC=1,K=3) -> Y: (1,1,5)
        g = _make_conv_transpose_graph(x_shape=(1, 1, 3), w_shape=(1, 1, 3), y_shape=(1, 1, 5))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3)
        w = torch.randn(1, 1, 3)
        (result,) = gm(x, w)
        expected = F.conv_transpose1d(x, w)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv_transpose1d_with_bias(self) -> None:
        """ConvTranspose1d with bias must produce correct results."""
        g = _make_conv_transpose_graph(x_shape=(1, 1, 3), w_shape=(1, 1, 3), y_shape=(1, 1, 5), bias_shape=(1,))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3)
        w = torch.randn(1, 1, 3)
        b = torch.randn(1)
        (result,) = gm(x, w, b)
        expected = F.conv_transpose1d(x, w, b)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- 2D --

    def test_conv_transpose2d_basic(self) -> None:
        """ConvTranspose2d with default attributes must produce correct results."""
        # X: (1,1,3,3), W: (IC=1,OC=1,K=3,3) -> Y: (1,1,5,5)
        g = _make_conv_transpose_graph(x_shape=(1, 1, 3, 3), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 5, 5))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3, 3)
        w = torch.randn(1, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv_transpose2d(x, w)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv_transpose2d_with_bias(self) -> None:
        """ConvTranspose2d with bias must produce correct results."""
        g = _make_conv_transpose_graph(
            x_shape=(1, 1, 3, 3), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 5, 5), bias_shape=(1,)
        )
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3, 3)
        w = torch.randn(1, 1, 3, 3)
        b = torch.randn(1)
        (result,) = gm(x, w, b)
        expected = F.conv_transpose2d(x, w, b)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv_transpose2d_stride(self) -> None:
        """ConvTranspose2d with stride=2 must produce correct results."""
        # X: (1,1,3,3), W: (1,1,3,3), stride=2 -> Y: (1,1,7,7)
        g = _make_conv_transpose_graph(x_shape=(1, 1, 3, 3), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 7, 7), strides=[2, 2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3, 3)
        w = torch.randn(1, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv_transpose2d(x, w, stride=2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv_transpose2d_output_padding(self) -> None:
        """ConvTranspose2d with output_padding must produce correct results."""
        # stride=2, output_padding=1 -> Y: (1,1,8,8)
        g = _make_conv_transpose_graph(
            x_shape=(1, 1, 3, 3),
            w_shape=(1, 1, 3, 3),
            y_shape=(1, 1, 8, 8),
            strides=[2, 2],
            output_padding=[1, 1],
        )
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3, 3)
        w = torch.randn(1, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv_transpose2d(x, w, stride=2, output_padding=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv_transpose2d_groups(self) -> None:
        """ConvTranspose2d with group > 1 must produce correct results."""
        g = _make_conv_transpose_graph(x_shape=(1, 2, 3, 3), w_shape=(2, 1, 3, 3), y_shape=(1, 2, 5, 5), group=2)
        gm = emit_graph(g)
        x = torch.randn(1, 2, 3, 3)
        w = torch.randn(2, 1, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv_transpose2d(x, w, groups=2)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- 3D --

    def test_conv_transpose3d_basic(self) -> None:
        """ConvTranspose3d with default attributes must produce correct results."""
        # X: (1,1,3,3,3), W: (1,1,3,3,3) -> Y: (1,1,5,5,5)
        g = _make_conv_transpose_graph(x_shape=(1, 1, 3, 3, 3), w_shape=(1, 1, 3, 3, 3), y_shape=(1, 1, 5, 5, 5))
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3, 3, 3)
        w = torch.randn(1, 1, 3, 3, 3)
        (result,) = gm(x, w)
        expected = F.conv_transpose3d(x, w)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_conv_transpose3d_with_bias(self) -> None:
        """ConvTranspose3d with bias must produce correct results."""
        g = _make_conv_transpose_graph(
            x_shape=(1, 1, 3, 3, 3), w_shape=(1, 1, 3, 3, 3), y_shape=(1, 1, 5, 5, 5), bias_shape=(1,)
        )
        gm = emit_graph(g)
        x = torch.randn(1, 1, 3, 3, 3)
        w = torch.randn(1, 1, 3, 3, 3)
        b = torch.randn(1)
        (result,) = gm(x, w, b)
        expected = F.conv_transpose3d(x, w, b)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- FX structure checks --

    def test_emits_call_function(self) -> None:
        """ConvTranspose must emit a call_function FX node."""
        g = _make_conv_transpose_graph(x_shape=(1, 1, 3, 3), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 5, 5))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_single_output(self) -> None:
        """ConvTranspose handler must return exactly one FX output node."""
        g = _make_conv_transpose_graph(x_shape=(1, 1, 3, 3), w_shape=(1, 1, 3, 3), y_shape=(1, 1, 5, 5))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1
