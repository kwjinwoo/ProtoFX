"""Tests for ONNX Loop control-flow handler behavior."""

from __future__ import annotations

import pytest
import torch
from onnx import TensorProto, helper

from protofx.emitters import emit_graph
from protofx.importers import import_model


def _make_loop_model(
    *,
    loop_inputs: list[str],
    cond_mode: str,
) -> helper.ModelProto:
    """Build an ONNX Loop model with one loop-carried state and one explicit capture.

    Args:
        loop_inputs: Positional ONNX Loop inputs ``[M, cond, state_init]`` using ``""`` for omitted optionals.
        cond_mode: Condition update mode for the body graph.

    Returns:
        A minimal Loop model for handler-level lowering tests.
    """
    m_input = helper.make_tensor_value_info("M", TensorProto.INT64, [])
    cond_input = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    state_init = helper.make_tensor_value_info("state_init", TensorProto.FLOAT, [2])
    capture = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])

    state_node = helper.make_node("Add", ["state_in", "x"], ["state_out"])
    match cond_mode:
        case "identity":
            cond_node = helper.make_node("Identity", ["cond_in"], ["cond_out"])
        case "toggle":
            cond_node = helper.make_node("Not", ["cond_in"], ["cond_out"])
        case _:
            msg = f"unsupported cond_mode: {cond_mode}"
            raise ValueError(msg)

    body_graph = helper.make_graph(
        [cond_node, state_node],
        "body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [2]),
        ],
    )
    loop_node = helper.make_node("Loop", loop_inputs, ["y"], body=body_graph)
    graph = helper.make_graph([loop_node], "loop_graph", [m_input, cond_input, state_init, capture], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestLoopHandler:
    """Validate handler-owned Loop lowering and loop-state contracts."""

    def test_loop_exposes_only_final_carried_outputs(self) -> None:
        """Loop outputs must expose only final loop-carried values."""
        model = _make_loop_model(loop_inputs=["M", "cond", "state_init"], cond_mode="identity")
        gm = emit_graph(import_model(model))

        (result,) = gm(
            torch.tensor(3, dtype=torch.int64),
            torch.tensor(True),
            torch.tensor([1.0, -1.0]),
            torch.tensor([2.0, 0.5]),
        )
        torch.testing.assert_close(result, torch.tensor([7.0, 0.5]))

    def test_loop_treats_captures_as_closed_over_operands(self) -> None:
        """Loop explicit captures must not be part of ``torch.while_loop`` state."""
        model = _make_loop_model(loop_inputs=["M", "cond", "state_init"], cond_mode="identity")
        gm = emit_graph(import_model(model))

        while_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "__name__", "") in {"_call_torch_while_loop", "while_loop"}
        )
        loop_state = while_node.args[2]
        assert isinstance(loop_state, tuple)
        assert len(loop_state) == 3

    @pytest.mark.parametrize(
        "loop_inputs",
        [
            ["", "cond", "state_init"],
            ["M", "", "state_init"],
            ["", "", "state_init"],
        ],
    )
    def test_loop_accepts_omitted_m_and_cond_sentinel_combinations(self, loop_inputs: list[str]) -> None:
        """Loop lowering must accept omitted M/cond sentinel combinations."""
        model = _make_loop_model(loop_inputs=loop_inputs, cond_mode="toggle")
        gm = emit_graph(import_model(model))

        (result,) = gm(
            torch.tensor(5, dtype=torch.int64),
            torch.tensor(True),
            torch.tensor([1.0, -1.0]),
            torch.tensor([2.0, 0.5]),
        )
        torch.testing.assert_close(result, torch.tensor([3.0, -0.5]))
