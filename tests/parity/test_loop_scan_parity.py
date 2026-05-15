"""ORT numerical parity tests for representative Loop and Scan coverage."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from tests.parity.conftest import assert_parity


def _make_loop_model() -> helper.ModelProto:
    """Build a representative ONNX Loop model with one carried state and one capture."""
    m_input = helper.make_tensor_value_info("M", TensorProto.INT64, [])
    cond_input = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    state_init = helper.make_tensor_value_info("state_init", TensorProto.FLOAT, [2])
    capture = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])

    body_graph = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Add", ["state_in", "x"], ["state_out"]),
        ],
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
    loop_node = helper.make_node("Loop", ["M", "cond", "state_init"], ["y"], body=body_graph)
    graph = helper.make_graph([loop_node], "loop_graph", [m_input, cond_input, state_init, capture], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_scan_model() -> helper.ModelProto:
    """Build a representative ONNX Scan model with one state, one scan input, and one capture."""
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
    """Build a representative ONNX Scan model with scan outputs independent from scan-input metadata."""
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


class TestLoopScanParity:
    """Representative ORT parity for Loop and Scan within the MVP boundary."""

    def test_loop_parity(self) -> None:
        """Loop carried-state output must match ORT on a representative case."""
        assert_parity(
            _make_loop_model(),
            {
                "M": np.asarray(3, dtype=np.int64),
                "cond": np.asarray(True, dtype=np.bool_),
                "state_init": np.asarray([1.0, -1.0], dtype=np.float32),
                "x": np.asarray([2.0, 0.5], dtype=np.float32),
            },
        )

    def test_scan_parity(self) -> None:
        """Scan final-state and scanned outputs must match ORT on a representative case."""
        assert_parity(
            _make_scan_model(),
            {
                "state_init": np.asarray(0.0, dtype=np.float32),
                "scan_in": np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                "x": np.asarray(1.0, dtype=np.float32),
            },
        )

    def test_scan_parity_with_independent_output_metadata(self) -> None:
        """Scan parity must hold when scan outputs are independent from scan-input metadata."""
        assert_parity(
            _make_scan_independent_output_model(),
            {
                "state_init": np.asarray([0.0], dtype=np.float32),
                "scan_in": np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                "x": np.asarray(1.0, dtype=np.float32),
            },
        )
