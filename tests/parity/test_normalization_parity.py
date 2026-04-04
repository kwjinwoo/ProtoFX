"""ORT numerical parity tests for normalization op handlers.

Covers: BatchNormalization (inference), LayerNormalization (with/without bias).
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from tests.parity.conftest import assert_parity

_OPSET = helper.make_opsetid("", 17)


class TestBatchNormalizationParity:
    """ORT parity for BatchNormalization op (inference mode)."""

    def test_basic(self) -> None:
        """BatchNormalization with default epsilon must match ORT."""
        rng = np.random.default_rng(400)
        channels = 4
        x = rng.standard_normal((2, channels, 6, 6)).astype(np.float32)
        scale = rng.standard_normal((channels,)).astype(np.float32)
        bias = rng.standard_normal((channels,)).astype(np.float32)
        mean = rng.standard_normal((channels,)).astype(np.float32)
        var = np.abs(rng.standard_normal((channels,))).astype(np.float32) + 0.1

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, channels, 6, 6])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, channels, 6, 6])
        node = helper.make_node(
            "BatchNormalization",
            ["X", "scale", "B", "mean", "var"],
            ["Y"],
        )
        graph = helper.make_graph(
            [node],
            "bn_test",
            [X],
            [Y],
            initializer=[
                numpy_helper.from_array(scale, "scale"),
                numpy_helper.from_array(bias, "B"),
                numpy_helper.from_array(mean, "mean"),
                numpy_helper.from_array(var, "var"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_custom_epsilon(self) -> None:
        """BatchNormalization with custom epsilon must match ORT."""
        rng = np.random.default_rng(401)
        channels = 8
        x = rng.standard_normal((1, channels, 4, 4)).astype(np.float32)
        scale = rng.standard_normal((channels,)).astype(np.float32)
        bias = rng.standard_normal((channels,)).astype(np.float32)
        mean = rng.standard_normal((channels,)).astype(np.float32)
        var = np.abs(rng.standard_normal((channels,))).astype(np.float32) + 0.01

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, channels, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, channels, 4, 4])
        node = helper.make_node(
            "BatchNormalization",
            ["X", "scale", "B", "mean", "var"],
            ["Y"],
            epsilon=1e-3,
        )
        graph = helper.make_graph(
            [node],
            "bn_eps_test",
            [X],
            [Y],
            initializer=[
                numpy_helper.from_array(scale, "scale"),
                numpy_helper.from_array(bias, "B"),
                numpy_helper.from_array(mean, "mean"),
                numpy_helper.from_array(var, "var"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})


class TestLayerNormalizationParity:
    """ORT parity for LayerNormalization op."""

    def test_basic(self) -> None:
        """LayerNormalization with bias over last axis must match ORT."""
        rng = np.random.default_rng(500)
        x = rng.standard_normal((2, 3, 8)).astype(np.float32)
        scale = rng.standard_normal((8,)).astype(np.float32)
        bias = rng.standard_normal((8,)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 8])
        node = helper.make_node("LayerNormalization", ["X", "scale", "B"], ["Y"], axis=-1)
        graph = helper.make_graph(
            [node],
            "ln_test",
            [X],
            [Y],
            initializer=[numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(bias, "B")],
        )
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_no_bias(self) -> None:
        """LayerNormalization without bias must match ORT."""
        rng = np.random.default_rng(501)
        x = rng.standard_normal((2, 3, 8)).astype(np.float32)
        scale = rng.standard_normal((8,)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 8])
        node = helper.make_node("LayerNormalization", ["X", "scale"], ["Y"], axis=-1)
        graph = helper.make_graph(
            [node], "ln_nobias_test", [X], [Y], initializer=[numpy_helper.from_array(scale, "scale")]
        )
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_axis_1(self) -> None:
        """LayerNormalization normalizing from axis=1 must match ORT."""
        rng = np.random.default_rng(502)
        x = rng.standard_normal((2, 4, 6)).astype(np.float32)
        scale = rng.standard_normal((4, 6)).astype(np.float32)
        bias = rng.standard_normal((4, 6)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 6])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 6])
        node = helper.make_node("LayerNormalization", ["X", "scale", "B"], ["Y"], axis=1)
        graph = helper.make_graph(
            [node],
            "ln_axis1_test",
            [X],
            [Y],
            initializer=[numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(bias, "B")],
        )
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})

    def test_custom_epsilon(self) -> None:
        """LayerNormalization with custom epsilon must match ORT."""
        rng = np.random.default_rng(503)
        x = rng.standard_normal((1, 4, 16)).astype(np.float32)
        scale = rng.standard_normal((16,)).astype(np.float32)
        bias = rng.standard_normal((16,)).astype(np.float32)

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 16])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 16])
        node = helper.make_node("LayerNormalization", ["X", "scale", "B"], ["Y"], axis=-1, epsilon=1e-3)
        graph = helper.make_graph(
            [node],
            "ln_eps_test",
            [X],
            [Y],
            initializer=[numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(bias, "B")],
        )
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"X": x})
