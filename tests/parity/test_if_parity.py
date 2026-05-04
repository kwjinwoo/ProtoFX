"""ORT numerical parity tests for ONNX If."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from tests.parity.conftest import assert_parity


def _make_if_model() -> helper.ModelProto:
    """Build an ONNX If model with one captured tensor input."""
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])

    then_node = helper.make_node("Identity", ["x"], ["then_out"])
    then_graph = helper.make_graph(
        [then_node],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )

    else_node = helper.make_node("Neg", ["x"], ["else_out"])
    else_graph = helper.make_graph(
        [else_node],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )

    if_node = helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph)
    graph = helper.make_graph([if_node], "if_graph", [cond, x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestIfParity:
    """ORT parity for If branch selection."""

    @pytest.mark.parametrize("condition", [True, False])
    def test_if_parity(self, condition: bool) -> None:
        """If outputs must match ONNX Runtime for both predicate values."""
        model = _make_if_model()
        inputs = {
            "cond": np.asarray(condition, dtype=np.bool_),
            "x": np.array([1.0, -2.0], dtype=np.float32),
        }
        assert_parity(model, inputs)
