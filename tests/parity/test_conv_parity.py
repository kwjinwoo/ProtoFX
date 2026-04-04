"""ORT numerical parity tests for convolution op handlers.

Covers: Conv (1D/2D, padding, stride, dilation, groups), ConvTranspose.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from tests.parity.conftest import assert_parity

_OPSET = helper.make_opsetid("", 17)


def _make_conv2d_model(
    in_channels: int = 3,
    out_channels: int = 8,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    *,
    with_bias: bool = True,
    seed: int = 42,
) -> tuple[helper.ModelProto, dict[str, np.ndarray]]:
    """Build a Conv2D ONNX model with random weights and return (model, inputs).

    Each test should use a unique *seed* to avoid ORT's process-level
    weight-content caching that can return stale fused-bias results when
    the same weight tensor appears in models with different bias configs.
    """
    rng = np.random.default_rng(seed)
    h, w = 8, 8
    x = rng.standard_normal((1, in_channels, h, w)).astype(np.float32)
    weight = rng.standard_normal((out_channels, in_channels // groups, kernel_size, kernel_size)).astype(np.float32)

    pads = [padding, padding, padding, padding]
    strides = [stride, stride]
    dilations = [dilation, dilation]
    oh = (h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    ow = (w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_channels, h, w])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_channels, oh, ow])
    W_init = numpy_helper.from_array(weight, name="W")

    inputs_list = ["X", "W"]
    initializers = [W_init]
    if with_bias:
        bias = rng.standard_normal((out_channels,)).astype(np.float32)
        B_init = numpy_helper.from_array(bias, name="B")
        inputs_list.append("B")
        initializers.append(B_init)

    node = helper.make_node(
        "Conv",
        inputs_list,
        ["Y"],
        kernel_shape=[kernel_size, kernel_size],
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=groups,
    )
    graph = helper.make_graph([node], "conv_test", [X], [Y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[_OPSET])
    return model, {"X": x}


class TestConv2dParity:
    """ORT parity for Conv op (2D)."""

    def test_basic(self) -> None:
        """Basic Conv2D with bias must match ORT."""
        model, inputs = _make_conv2d_model(seed=100)
        assert_parity(model, inputs)

    def test_no_bias(self) -> None:
        """Conv2D without bias must match ORT."""
        model, inputs = _make_conv2d_model(with_bias=False, seed=101)
        assert_parity(model, inputs)

    def test_padding(self) -> None:
        """Conv2D with padding must match ORT."""
        model, inputs = _make_conv2d_model(padding=1, seed=102)
        assert_parity(model, inputs)

    def test_stride(self) -> None:
        """Conv2D with stride must match ORT."""
        model, inputs = _make_conv2d_model(stride=2, seed=103)
        assert_parity(model, inputs)

    def test_dilation(self) -> None:
        """Conv2D with dilation must match ORT."""
        model, inputs = _make_conv2d_model(dilation=2, kernel_size=3, seed=104)
        assert_parity(model, inputs)

    def test_groups(self) -> None:
        """Conv2D with groups (depthwise) must match ORT."""
        model, inputs = _make_conv2d_model(in_channels=4, out_channels=4, groups=4, seed=105)
        assert_parity(model, inputs)


# ---------------------------------------------------------------------------
# Conv1D
# ---------------------------------------------------------------------------


class TestConv1dParity:
    """ORT parity for Conv op (1D)."""

    def test_basic(self) -> None:
        """Basic Conv1D must match ORT."""
        rng = np.random.default_rng(200)
        x = rng.standard_normal((1, 3, 16)).astype(np.float32)
        weight = rng.standard_normal((8, 3, 3)).astype(np.float32)
        bias = rng.standard_normal((8,)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 14])
        W_init = numpy_helper.from_array(weight, name="W")
        B_init = numpy_helper.from_array(bias, name="B")
        node = helper.make_node("Conv", ["X", "W", "B"], ["Y"], kernel_shape=[3], strides=[1], pads=[0, 0])
        graph = helper.make_graph([node], "conv1d_test", [X], [Y], initializer=[W_init, B_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


# ---------------------------------------------------------------------------
# ConvTranspose
# ---------------------------------------------------------------------------


class TestConvTransposeParity:
    """ORT parity for ConvTranspose op."""

    def test_basic_2d(self) -> None:
        """Basic ConvTranspose2D must match ORT."""
        rng = np.random.default_rng(300)
        x = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
        weight = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)
        bias = rng.standard_normal((3,)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 6, 6])
        W_init = numpy_helper.from_array(weight, name="W")
        B_init = numpy_helper.from_array(bias, name="B")
        node = helper.make_node(
            "ConvTranspose",
            ["X", "W", "B"],
            ["Y"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )
        graph = helper.make_graph([node], "conv_t_test", [X], [Y], initializer=[W_init, B_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_stride_2d(self) -> None:
        """ConvTranspose2D with stride must match ORT."""
        rng = np.random.default_rng(301)
        x = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
        weight = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 9, 9])
        W_init = numpy_helper.from_array(weight, name="W")
        node = helper.make_node(
            "ConvTranspose",
            ["X", "W"],
            ["Y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
        )
        graph = helper.make_graph([node], "conv_t_test", [X], [Y], initializer=[W_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})
