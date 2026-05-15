"""Tests for ONNX Scan control-flow handler behavior."""

from __future__ import annotations

from onnx import TensorProto, helper

from protofx.emitters import emit_graph
from protofx.importers import import_model


def _make_scan_model() -> helper.ModelProto:
    """Build an ONNX Scan model with one state, one scan input, and one explicit capture."""
    state_init = helper.make_tensor_value_info("state_init", TensorProto.FLOAT, [])
    scan_in = helper.make_tensor_value_info("scan_in", TensorProto.FLOAT, [3, 2])
    capture = helper.make_tensor_value_info("x", TensorProto.FLOAT, [])
    final_state = helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [])
    scanned = helper.make_tensor_value_info("scanned", TensorProto.FLOAT, [3, 2])

    body_graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["scan_step"], ["scan_reduced"], keepdims=0),
            helper.make_node("Add", ["state_in", "scan_reduced"], ["state_mid"]),
            helper.make_node("Add", ["state_mid", "x"], ["state_out"]),
            helper.make_node("Identity", ["scan_step"], ["scan_out"]),
        ],
        "body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_step", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [2]),
        ],
    )
    scan_node = helper.make_node(
        "Scan",
        ["state_init", "scan_in"],
        ["final_state", "scanned"],
        num_scan_inputs=1,
        body=body_graph,
    )
    graph = helper.make_graph([scan_node], "scan_graph", [state_init, scan_in, capture], [final_state, scanned])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_scan_independent_output_model() -> helper.ModelProto:
    """Build an ONNX Scan model where scan output metadata is independent from scan input metadata."""
    state_init = helper.make_tensor_value_info("state_init", TensorProto.FLOAT, [1])
    scan_in = helper.make_tensor_value_info("scan_in", TensorProto.FLOAT, [3, 2])
    capture = helper.make_tensor_value_info("x", TensorProto.FLOAT, [])
    final_state = helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [1])
    scanned_identity = helper.make_tensor_value_info("scanned_identity", TensorProto.FLOAT, [3, 2])
    scanned_reduced = helper.make_tensor_value_info("scanned_reduced", TensorProto.FLOAT, [3])

    body_graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["scan_step"], ["scan_reduced_vec"], keepdims=1),
            helper.make_node("ReduceSum", ["scan_step"], ["scan_reduced_scalar"], keepdims=0),
            helper.make_node("Add", ["state_in", "scan_reduced_vec"], ["state_mid"]),
            helper.make_node("Add", ["state_mid", "x"], ["state_out"]),
            helper.make_node("Identity", ["scan_step"], ["scan_out0"]),
            helper.make_node("Identity", ["scan_reduced_scalar"], ["scan_out1"]),
        ],
        "body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("scan_step", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("scan_out0", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("scan_out1", TensorProto.FLOAT, []),
        ],
    )
    scan_node = helper.make_node(
        "Scan",
        ["state_init", "scan_in"],
        ["final_state", "scanned_identity", "scanned_reduced"],
        num_scan_inputs=1,
        body=body_graph,
    )
    graph = helper.make_graph(
        [scan_node], "scan_graph", [state_init, scan_in, capture], [final_state, scanned_identity, scanned_reduced]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestScanHandler:
    """Validate handler-owned Scan lowering and output contracts."""

    def test_scan_exposes_final_state_then_scanned_outputs(self) -> None:
        """Scan must return final state outputs before scanned outputs."""
        model = _make_scan_model()
        gm = emit_graph(import_model(model))

        output_node = next(node for node in gm.graph.nodes if node.op == "output")
        returned = output_node.args[0]
        assert isinstance(returned, tuple)
        assert len(returned) == 2
        assert returned[0].op == "call_function"
        assert getattr(returned[0].target, "__name__", "") == "getitem"
        assert returned[0].args[1] == 1
        assert returned[1].args[1] == 2

    def test_scan_treats_captures_as_closed_over_operands(self) -> None:
        """Scan captures must stay outside the ``torch.while_loop`` state tuple."""
        model = _make_scan_model()
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

    def test_scan_supports_independent_scan_output_family_metadata(self) -> None:
        """Scan lowering must allow scan output family metadata independent from scan input metadata."""
        model = _make_scan_independent_output_model()
        gm = emit_graph(import_model(model))

        output_node = next(node for node in gm.graph.nodes if node.op == "output")
        returned = output_node.args[0]
        assert isinstance(returned, tuple)
        assert len(returned) == 3
        assert returned[0].args[1] == 1
        assert returned[1].args[1] == 2
        assert returned[2].args[1] == 3
