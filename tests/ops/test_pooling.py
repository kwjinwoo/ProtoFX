"""Tests for pooling op handlers (MaxPool, AveragePool, GlobalAveragePool)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# MaxPool helpers
# ---------------------------------------------------------------------------


def _make_maxpool_graph(
    *,
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    kernel_shape: list[int],
    strides: list[int] | None = None,
    pads: list[int] | None = None,
    dilations: list[int] | None = None,
    ceil_mode: int = 0,
    auto_pad: str | None = None,
    num_outputs: int = 1,
) -> Graph:
    """Build a minimal IR graph: X -> MaxPool -> Y [, Indices].

    Args:
        x_shape: Shape of the input tensor X.
        y_shape: Shape of the output tensor Y.
        kernel_shape: Pooling kernel shape attribute.
        strides: Optional strides attribute.
        pads: Optional pads attribute (ONNX format).
        dilations: Optional dilations attribute.
        ceil_mode: 0 for floor (default), 1 for ceil.
        auto_pad: Optional auto_pad attribute.
        num_outputs: Number of outputs (1 for Y only, 2 for Y + Indices).
    """
    g = Graph(name="MaxPool_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=x_shape), name="X")

    attributes: dict[str, int | float | list[int] | str] = {"kernel_shape": kernel_shape}
    if strides is not None:
        attributes["strides"] = strides
    if pads is not None:
        attributes["pads"] = pads
    if dilations is not None:
        attributes["dilations"] = dilations
    if ceil_mode != 0:
        attributes["ceil_mode"] = ceil_mode
    if auto_pad is not None:
        attributes["auto_pad"] = auto_pad

    output_types = [TensorType(dtype=DType.FLOAT32, shape=y_shape)]
    output_names: list[str | None] = ["Y"]
    if num_outputs == 2:
        output_types.append(TensorType(dtype=DType.INT64, shape=y_shape))
        output_names.append("Indices")

    node = g.make_node(
        op_type="MaxPool",
        inputs=[x],
        output_types=output_types,
        output_names=output_names,
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# MaxPool tests
# ---------------------------------------------------------------------------


class TestMaxPoolHandler:
    """Verify that the MaxPool op handler emits correct FX nodes."""

    # -- 1D --

    def test_maxpool1d_basic(self) -> None:
        """MaxPool1d with default attributes must produce correct results."""
        # X: (1, 1, 6), kernel_shape=[3] -> Y: (1, 1, 4) (ONNX default stride=1)
        g = _make_maxpool_graph(x_shape=(1, 1, 6), y_shape=(1, 1, 4), kernel_shape=[3])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6)
        (result,) = gm(x)
        expected = F.max_pool1d(x, kernel_size=3, stride=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_maxpool1d_stride(self) -> None:
        """MaxPool1d with stride must produce correct results."""
        # X: (1, 1, 6), kernel_shape=[3], strides=[2] -> Y: (1, 1, 2)
        g = _make_maxpool_graph(x_shape=(1, 1, 6), y_shape=(1, 1, 2), kernel_shape=[3], strides=[2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6)
        (result,) = gm(x)
        expected = F.max_pool1d(x, kernel_size=3, stride=2)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- 2D --

    def test_maxpool2d_basic(self) -> None:
        """MaxPool2d with default attributes must produce correct results."""
        # X: (1, 1, 6, 6), kernel_shape=[3, 3] -> Y: (1, 1, 4, 4) (ONNX default stride=1)
        g = _make_maxpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 4, 4), kernel_shape=[3, 3])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        (result,) = gm(x)
        expected = F.max_pool2d(x, kernel_size=3, stride=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_maxpool2d_stride(self) -> None:
        """MaxPool2d with stride must produce correct results."""
        # X: (1, 1, 6, 6), kernel_shape=[3, 3], strides=[2, 2] -> Y: (1, 1, 2, 2)
        g = _make_maxpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 2, 2), kernel_shape=[3, 3], strides=[2, 2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        (result,) = gm(x)
        expected = F.max_pool2d(x, kernel_size=3, stride=2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_maxpool2d_padding(self) -> None:
        """MaxPool2d with symmetric padding must produce correct results."""
        # X: (1, 1, 6, 6), kernel_shape=[3, 3], pads=[1,1,1,1] -> Y: (1, 1, 6, 6) (ONNX default stride=1)
        g = _make_maxpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 6, 6), kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        (result,) = gm(x)
        expected = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_maxpool2d_ceil_mode(self) -> None:
        """MaxPool2d with ceil_mode=1 must produce correct results."""
        # X: (1, 1, 7, 7), kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1 -> Y: (1, 1, 3, 3)
        g = _make_maxpool_graph(
            x_shape=(1, 1, 7, 7), y_shape=(1, 1, 3, 3), kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1
        )
        gm = emit_graph(g)
        x = torch.randn(1, 1, 7, 7)
        (result,) = gm(x)
        expected = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_maxpool2d_dilations(self) -> None:
        """MaxPool2d with dilations must produce correct results."""
        # X: (1, 1, 8, 8), kernel_shape=[3, 3], dilations=[2, 2] -> Y: (1, 1, 4, 4) (ONNX default stride=1)
        g = _make_maxpool_graph(x_shape=(1, 1, 8, 8), y_shape=(1, 1, 4, 4), kernel_shape=[3, 3], dilations=[2, 2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 8, 8)
        (result,) = gm(x)
        expected = F.max_pool2d(x, kernel_size=3, stride=1, dilation=2)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- Error cases --

    def test_maxpool_auto_pad_raises(self) -> None:
        """MaxPool with auto_pad other than NOTSET must raise NotImplementedError."""
        g = _make_maxpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 6, 6), kernel_shape=[3, 3], auto_pad="SAME_UPPER")
        with pytest.raises(NotImplementedError, match="auto_pad"):
            emit_graph(g)

    def test_maxpool_indices_output_raises(self) -> None:
        """MaxPool with Indices output must raise NotImplementedError."""
        g = _make_maxpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 4, 4), kernel_shape=[3, 3], num_outputs=2)
        with pytest.raises(NotImplementedError, match="Indices"):
            emit_graph(g)


# ---------------------------------------------------------------------------
# AveragePool helpers
# ---------------------------------------------------------------------------


def _make_avgpool_graph(
    *,
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    kernel_shape: list[int],
    strides: list[int] | None = None,
    pads: list[int] | None = None,
    count_include_pad: int = 0,
    ceil_mode: int = 0,
    auto_pad: str | None = None,
) -> Graph:
    """Build a minimal IR graph: X -> AveragePool -> Y.

    Args:
        x_shape: Shape of the input tensor X.
        y_shape: Shape of the output tensor Y.
        kernel_shape: Pooling kernel shape attribute.
        strides: Optional strides attribute.
        pads: Optional pads attribute (ONNX format).
        count_include_pad: Whether to include zero-padding in averaging.
        ceil_mode: 0 for floor (default), 1 for ceil.
        auto_pad: Optional auto_pad attribute.
    """
    g = Graph(name="AveragePool_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=x_shape), name="X")

    attributes: dict[str, int | float | list[int] | str] = {"kernel_shape": kernel_shape}
    if strides is not None:
        attributes["strides"] = strides
    if pads is not None:
        attributes["pads"] = pads
    if count_include_pad != 0:
        attributes["count_include_pad"] = count_include_pad
    if ceil_mode != 0:
        attributes["ceil_mode"] = ceil_mode
    if auto_pad is not None:
        attributes["auto_pad"] = auto_pad

    node = g.make_node(
        op_type="AveragePool",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=y_shape)],
        output_names=["Y"],
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# AveragePool tests
# ---------------------------------------------------------------------------


class TestAveragePoolHandler:
    """Verify that the AveragePool op handler emits correct FX nodes."""

    # -- 1D --

    def test_avgpool1d_basic(self) -> None:
        """AveragePool1d with default attributes must produce correct results."""
        # X: (1, 1, 6), kernel_shape=[3] -> Y: (1, 1, 4) (ONNX default stride=1)
        g = _make_avgpool_graph(x_shape=(1, 1, 6), y_shape=(1, 1, 4), kernel_shape=[3])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6)
        (result,) = gm(x)
        expected = F.avg_pool1d(x, kernel_size=3, stride=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_avgpool1d_stride(self) -> None:
        """AveragePool1d with stride must produce correct results."""
        g = _make_avgpool_graph(x_shape=(1, 1, 6), y_shape=(1, 1, 2), kernel_shape=[3], strides=[2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6)
        (result,) = gm(x)
        expected = F.avg_pool1d(x, kernel_size=3, stride=2)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- 2D --

    def test_avgpool2d_basic(self) -> None:
        """AveragePool2d with default attributes must produce correct results."""
        # X: (1, 1, 6, 6), kernel_shape=[3, 3] -> Y: (1, 1, 4, 4) (ONNX default stride=1)
        g = _make_avgpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 4, 4), kernel_shape=[3, 3])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        (result,) = gm(x)
        expected = F.avg_pool2d(x, kernel_size=3, stride=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_avgpool2d_stride(self) -> None:
        """AveragePool2d with stride must produce correct results."""
        g = _make_avgpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 2, 2), kernel_shape=[3, 3], strides=[2, 2])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        (result,) = gm(x)
        expected = F.avg_pool2d(x, kernel_size=3, stride=2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_avgpool2d_padding(self) -> None:
        """AveragePool2d with symmetric padding must produce correct results."""
        g = _make_avgpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 6, 6), kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        (result,) = gm(x)
        # ONNX default count_include_pad=0 maps to PyTorch count_include_pad=False
        expected = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_avgpool2d_count_include_pad(self) -> None:
        """AveragePool2d with count_include_pad=1 must produce correct results."""
        g = _make_avgpool_graph(
            x_shape=(1, 1, 6, 6),
            y_shape=(1, 1, 6, 6),
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            count_include_pad=1,
        )
        gm = emit_graph(g)
        x = torch.randn(1, 1, 6, 6)
        (result,) = gm(x)
        expected = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=True)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_avgpool2d_ceil_mode(self) -> None:
        """AveragePool2d with ceil_mode=1 must produce correct results."""
        g = _make_avgpool_graph(
            x_shape=(1, 1, 7, 7), y_shape=(1, 1, 3, 3), kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1
        )
        gm = emit_graph(g)
        x = torch.randn(1, 1, 7, 7)
        (result,) = gm(x)
        expected = F.avg_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        assert torch.allclose(result, expected, atol=1e-6)

    # -- Error cases --

    def test_avgpool_auto_pad_raises(self) -> None:
        """AveragePool with auto_pad other than NOTSET must raise NotImplementedError."""
        g = _make_avgpool_graph(x_shape=(1, 1, 6, 6), y_shape=(1, 1, 6, 6), kernel_shape=[3, 3], auto_pad="SAME_UPPER")
        with pytest.raises(NotImplementedError, match="auto_pad"):
            emit_graph(g)


# ---------------------------------------------------------------------------
# GlobalAveragePool helpers
# ---------------------------------------------------------------------------


def _make_global_avgpool_graph(
    *,
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
) -> Graph:
    """Build a minimal IR graph: X -> GlobalAveragePool -> Y.

    Args:
        x_shape: Shape of the input tensor X.
        y_shape: Shape of the output tensor Y (spatial dims are 1).
    """
    g = Graph(name="GlobalAveragePool_test")
    x = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=x_shape), name="X")

    node = g.make_node(
        op_type="GlobalAveragePool",
        inputs=[x],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=y_shape)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


# ---------------------------------------------------------------------------
# GlobalAveragePool tests
# ---------------------------------------------------------------------------


class TestGlobalAveragePoolHandler:
    """Verify that the GlobalAveragePool op handler emits correct FX nodes."""

    def test_global_avgpool_1d(self) -> None:
        """GlobalAveragePool on 1D spatial input must produce correct results."""
        # X: (1, 3, 8) -> Y: (1, 3, 1)
        g = _make_global_avgpool_graph(x_shape=(1, 3, 8), y_shape=(1, 3, 1))
        gm = emit_graph(g)
        x = torch.randn(1, 3, 8)
        (result,) = gm(x)
        expected = F.adaptive_avg_pool1d(x, 1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_global_avgpool_2d(self) -> None:
        """GlobalAveragePool on 2D spatial input must produce correct results."""
        # X: (1, 3, 8, 8) -> Y: (1, 3, 1, 1)
        g = _make_global_avgpool_graph(x_shape=(1, 3, 8, 8), y_shape=(1, 3, 1, 1))
        gm = emit_graph(g)
        x = torch.randn(1, 3, 8, 8)
        (result,) = gm(x)
        expected = F.adaptive_avg_pool2d(x, 1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_global_avgpool_3d(self) -> None:
        """GlobalAveragePool on 3D spatial input must produce correct results."""
        # X: (1, 3, 4, 4, 4) -> Y: (1, 3, 1, 1, 1)
        g = _make_global_avgpool_graph(x_shape=(1, 3, 4, 4, 4), y_shape=(1, 3, 1, 1, 1))
        gm = emit_graph(g)
        x = torch.randn(1, 3, 4, 4, 4)
        (result,) = gm(x)
        expected = F.adaptive_avg_pool3d(x, 1)
        assert torch.allclose(result, expected, atol=1e-6)
