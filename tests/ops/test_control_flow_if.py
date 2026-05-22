"""Tests for ONNX If control-flow handler behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from onnx import TensorProto, helper

from protofx.emitters import emit_graph
from protofx.importers import import_model
from protofx.ir.derived_shape import get_authoritative_shape
from protofx.ir.shape_propagation import propagate_shapes


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


class TestIfHandler:
    """Validate end-to-end If branch selection and validation failures."""

    @pytest.mark.parametrize(
        ("condition", "expected"),
        [
            (True, np.array([1.0, -2.0], dtype=np.float32)),
            (False, np.array([-1.0, 2.0], dtype=np.float32)),
        ],
    )
    def test_if_selects_correct_branch(self, condition: bool, expected: np.ndarray) -> None:
        """If must select the then/else branch based on the predicate."""
        model = _make_if_model()
        gm = emit_graph(import_model(model))
        (result,) = gm(torch.tensor(condition), torch.tensor([1.0, -2.0]))
        torch.testing.assert_close(result, torch.from_numpy(expected))

    def test_if_raises_for_branch_output_arity_mismatch(self) -> None:
        """Import must fail when then/else branches expose different output arity."""
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
        else_node = helper.make_node("Identity", ["x"], ["else_out"])
        else_graph = helper.make_graph([else_node], "else_branch", [], [])

        if_node = helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph)
        model = helper.make_model(
            helper.make_graph([if_node], "if_graph", [cond, x], [y]),
            opset_imports=[helper.make_opsetid("", 17)],
        )

        with pytest.raises(ValueError, match="arity"):
            import_model(model)

    def test_if_propagation_merges_branch_shape(self) -> None:
        """If output should use propagated branch shape, not stale seed metadata."""
        model = _make_if_model()
        graph = import_model(model)
        if_node = graph.nodes[0]
        if_node.outputs[0].tensor_type = if_node.outputs[0].tensor_type.__class__(
            dtype=if_node.outputs[0].tensor_type.dtype, shape=(9, 9)
        )

        propagate_shapes(graph)

        assert get_authoritative_shape(if_node.outputs[0]) == (2,)
