"""ORT numerical parity tests for pooling op handlers.

Covers: MaxPool, AveragePool, GlobalAveragePool (2D).
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from tests.parity.conftest import assert_parity

_OPSET = helper.make_opsetid("", 17)


class TestMaxPoolParity:
    """ORT parity for MaxPool op."""

    def test_basic_2d(self) -> None:
        """MaxPool 2D with default settings must match ORT."""
        rng = np.random.default_rng(600)
        x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 6, 6])
        node = helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[3, 3])
        graph = helper.make_graph([node], "maxpool_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_stride(self) -> None:
        """MaxPool 2D with stride must match ORT."""
        rng = np.random.default_rng(601)
        x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 3, 3])
        node = helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2])
        graph = helper.make_graph([node], "maxpool_stride_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_padding(self) -> None:
        """MaxPool 2D with padding must match ORT."""
        rng = np.random.default_rng(602)
        x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])
        node = helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        graph = helper.make_graph([node], "maxpool_pad_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_ceil_mode(self) -> None:
        """MaxPool 2D with ceil_mode must match ORT."""
        rng = np.random.default_rng(603)
        x = rng.standard_normal((1, 3, 7, 7)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 7, 7])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
        node = helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
        graph = helper.make_graph([node], "maxpool_ceil_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


class TestAveragePoolParity:
    """ORT parity for AveragePool op."""

    def test_basic_2d(self) -> None:
        """AveragePool 2D with default settings must match ORT."""
        rng = np.random.default_rng(700)
        x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 6, 6])
        node = helper.make_node("AveragePool", ["X"], ["Y"], kernel_shape=[3, 3])
        graph = helper.make_graph([node], "avgpool_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_stride(self) -> None:
        """AveragePool 2D with stride must match ORT."""
        rng = np.random.default_rng(701)
        x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 3, 3])
        node = helper.make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2])
        graph = helper.make_graph([node], "avgpool_stride_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_padding_count_include(self) -> None:
        """AveragePool 2D with padding and count_include_pad must match ORT."""
        rng = np.random.default_rng(702)
        x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])
        node = helper.make_node(
            "AveragePool", ["X"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], count_include_pad=1
        )
        graph = helper.make_graph([node], "avgpool_pad_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


class TestGlobalAveragePoolParity:
    """ORT parity for GlobalAveragePool op."""

    def test_basic_2d(self) -> None:
        """GlobalAveragePool 2D must match ORT."""
        rng = np.random.default_rng(800)
        x = rng.standard_normal((2, 4, 6, 6)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 6, 6])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 1, 1])
        node = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
        graph = helper.make_graph([node], "gap_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_single_spatial(self) -> None:
        """GlobalAveragePool 2D with 1x1 spatial must match ORT."""
        rng = np.random.default_rng(801)
        x = rng.standard_normal((1, 8, 1, 1)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 1, 1])
        node = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
        graph = helper.make_graph([node], "gap_1x1_test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})
