"""ORT numerical parity tests for linear algebra op handlers.

Covers: MatMul, Gemm (with transA/transB/alpha/beta combinations).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.parity.conftest import assert_parity

_OPSET = helper.make_opsetid("", 17)


# ---------------------------------------------------------------------------
# MatMul
# ---------------------------------------------------------------------------


class TestMatMulParity:
    """ORT parity for MatMul op."""

    def test_2d(self) -> None:
        """MatMul with 2D inputs must match ORT."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((3, 4)).astype(np.float32)
        b = rng.standard_normal((4, 5)).astype(np.float32)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 5])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
        node = helper.make_node("MatMul", ["A", "B"], ["Y"])
        graph = helper.make_graph([node], "matmul_test", [A, B], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})

    def test_batched(self) -> None:
        """MatMul with batched 3D inputs must match ORT."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((2, 3, 4)).astype(np.float32)
        b = rng.standard_normal((2, 4, 5)).astype(np.float32)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 4, 5])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 5])
        node = helper.make_node("MatMul", ["A", "B"], ["Y"])
        graph = helper.make_graph([node], "matmul_test", [A, B], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})


# ---------------------------------------------------------------------------
# Gemm
# ---------------------------------------------------------------------------


class TestGemmParity:
    """ORT parity for Gemm op (with various transA/transB/alpha/beta)."""

    def test_default_params(self) -> None:
        """Gemm with default parameters must match ORT."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((3, 4)).astype(np.float32)
        b = rng.standard_normal((4, 5)).astype(np.float32)
        c = rng.standard_normal((5,)).astype(np.float32)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 5])
        C_init = numpy_helper.from_array(c, name="C")
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
        node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"])
        graph = helper.make_graph([node], "gemm_test", [A, B], [Y], initializer=[C_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})

    def test_transA(self) -> None:
        """Gemm with transA=1 must match ORT."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 3)).astype(np.float32)
        b = rng.standard_normal((4, 5)).astype(np.float32)
        c = rng.standard_normal((5,)).astype(np.float32)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 3])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 5])
        C_init = numpy_helper.from_array(c, name="C")
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
        node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], transA=1)
        graph = helper.make_graph([node], "gemm_test", [A, B], [Y], initializer=[C_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})

    def test_transB(self) -> None:
        """Gemm with transB=1 must match ORT."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((3, 4)).astype(np.float32)
        b = rng.standard_normal((5, 4)).astype(np.float32)
        c = rng.standard_normal((5,)).astype(np.float32)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [5, 4])
        C_init = numpy_helper.from_array(c, name="C")
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
        node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], transB=1)
        graph = helper.make_graph([node], "gemm_test", [A, B], [Y], initializer=[C_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})

    @pytest.mark.parametrize(("alpha", "beta"), [(0.5, 0.5), (2.0, 0.0), (1.0, 1.0)])
    def test_alpha_beta(self, alpha: float, beta: float) -> None:
        """Gemm with various alpha/beta must match ORT."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((3, 4)).astype(np.float32)
        b = rng.standard_normal((4, 5)).astype(np.float32)
        c = rng.standard_normal((5,)).astype(np.float32)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 5])
        C_init = numpy_helper.from_array(c, name="C")
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
        node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=beta)
        graph = helper.make_graph([node], "gemm_test", [A, B], [Y], initializer=[C_init])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})

    def test_no_bias(self) -> None:
        """Gemm without bias input must match ORT."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((3, 4)).astype(np.float32)
        b = rng.standard_normal((4, 5)).astype(np.float32)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 5])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
        node = helper.make_node("Gemm", ["A", "B"], ["Y"])
        graph = helper.make_graph([node], "gemm_test", [A, B], [Y])
        model = helper.make_model(graph, opset_imports=[_OPSET])
        assert_parity(model, {"A": a, "B": b})
