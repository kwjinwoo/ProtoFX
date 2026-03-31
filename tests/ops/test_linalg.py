"""Tests for linear algebra op handlers (MatMul, Gemm)."""

from __future__ import annotations

import torch

from protofx.emitters import emit_graph
from protofx.ir import DType, Graph, TensorType

# ---------------------------------------------------------------------------
# MatMul
# ---------------------------------------------------------------------------


def _make_matmul_graph(shape_a: tuple[int, ...], shape_b: tuple[int, ...], shape_y: tuple[int, ...]) -> Graph:
    """Build a minimal IR graph: (A, B) → MatMul → Y."""
    g = Graph(name="MatMul_test")
    a = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape_a), name="A")
    b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape_b), name="B")
    node = g.make_node(
        op_type="MatMul",
        inputs=[a, b],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=shape_y)],
        output_names=["Y"],
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestMatMulHandler:
    """Verify that the MatMul op handler emits correct FX nodes."""

    def test_emits_call_function(self) -> None:
        """MatMul must emit a call_function FX node."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        ops = [n.op for n in gm.graph.nodes]
        assert "call_function" in ops

    def test_call_function_target(self) -> None:
        """The call_function target must be torch.matmul."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.matmul

    def test_single_output(self) -> None:
        """MatMul handler must return exactly one FX output node."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    def test_forward_correctness_2d(self) -> None:
        """The emitted GraphModule must produce correct 2D MatMul results."""
        g = _make_matmul_graph((2, 3), (3, 4), (2, 4))
        gm = emit_graph(g)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        (result,) = gm(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected)

    def test_forward_correctness_batched(self) -> None:
        """The emitted GraphModule must produce correct batched MatMul results."""
        g = _make_matmul_graph((2, 3, 4), (2, 4, 5), (2, 3, 5))
        gm = emit_graph(g)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        (result,) = gm(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Gemm
# ---------------------------------------------------------------------------


def _make_gemm_graph(
    *,
    shape_a: tuple[int, ...] = (2, 3),
    shape_b: tuple[int, ...] = (3, 4),
    shape_y: tuple[int, ...] = (2, 4),
    shape_c: tuple[int, ...] | None = None,
    trans_a: int = 0,
    trans_b: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Graph:
    """Build a minimal IR graph: (A, B [, C]) → Gemm → Y.

    When *shape_c* is ``None`` the third input is a sentinel (omitted optional).
    """
    g = Graph(name="Gemm_test")
    a = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape_a), name="A")
    b = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape_b), name="B")

    if shape_c is not None:
        c = g.add_input(tensor_type=TensorType(dtype=DType.FLOAT32, shape=shape_c), name="C")
    else:
        c = g.add_sentinel()

    attributes: dict[str, int | float] = {}
    if trans_a:
        attributes["transA"] = trans_a
    if trans_b:
        attributes["transB"] = trans_b
    if alpha != 1.0:
        attributes["alpha"] = alpha
    if beta != 1.0:
        attributes["beta"] = beta

    node = g.make_node(
        op_type="Gemm",
        inputs=[a, b, c],
        output_types=[TensorType(dtype=DType.FLOAT32, shape=shape_y)],
        output_names=["Y"],
        attributes=attributes,
    )
    g.set_graph_outputs(list(node.outputs))
    return g


class TestGemmHandler:
    """Verify that the Gemm op handler emits correct FX nodes."""

    # Case 1: basic Gemm without bias (Y = A @ B)
    def test_basic_no_bias(self) -> None:
        """Gemm without C must produce correct A @ B results."""
        g = _make_gemm_graph(shape_a=(2, 3), shape_b=(3, 4), shape_y=(2, 4))
        gm = emit_graph(g)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        (result,) = gm(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-6)

    # Case 2: Gemm with bias (Y = A @ B + C)
    def test_with_bias(self) -> None:
        """Gemm with C must produce correct A @ B + C results."""
        g = _make_gemm_graph(shape_a=(2, 3), shape_b=(3, 4), shape_y=(2, 4), shape_c=(4,))
        gm = emit_graph(g)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        c = torch.randn(4)
        (result,) = gm(a, b, c)
        expected = torch.matmul(a, b) + c
        assert torch.allclose(result, expected, atol=1e-6)

    # Case 3: transA=1, transB=1 (Y = A^T @ B^T)
    def test_transpose(self) -> None:
        """Gemm with transA=1 and transB=1 must transpose before matmul."""
        g = _make_gemm_graph(
            shape_a=(3, 2),
            shape_b=(4, 3),
            shape_y=(2, 4),
            trans_a=1,
            trans_b=1,
        )
        gm = emit_graph(g)
        a = torch.randn(3, 2)
        b = torch.randn(4, 3)
        (result,) = gm(a, b)
        expected = torch.matmul(a.T, b.T)
        assert torch.allclose(result, expected, atol=1e-6)

    # Case 4: alpha/beta scaling (Y = alpha * A @ B + beta * C)
    def test_alpha_beta(self) -> None:
        """Gemm with alpha and beta must scale matmul and bias correctly."""
        g = _make_gemm_graph(
            shape_a=(2, 3),
            shape_b=(3, 4),
            shape_y=(2, 4),
            shape_c=(4,),
            alpha=0.5,
            beta=2.0,
        )
        gm = emit_graph(g)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        c = torch.randn(4)
        (result,) = gm(a, b, c)
        expected = 0.5 * torch.matmul(a, b) + 2.0 * c
        assert torch.allclose(result, expected, atol=1e-6)

    def test_single_output(self) -> None:
        """Gemm handler must return exactly one FX output node."""
        g = _make_gemm_graph(shape_a=(2, 3), shape_b=(3, 4), shape_y=(2, 4))
        gm = emit_graph(g)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        assert len(output_node.args[0]) == 1

    # Optimization check: alpha=1.0 should not emit extra mul nodes
    def test_default_alpha_no_extra_mul(self) -> None:
        """Gemm with alpha=1.0 must not emit an unnecessary mul node for alpha."""
        g = _make_gemm_graph(shape_a=(2, 3), shape_b=(3, 4), shape_y=(2, 4))
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        # Only matmul, no mul for alpha
        assert len(call_nodes) == 1
        assert call_nodes[0].target is torch.matmul

    # Optimization check: beta=1.0 should not emit extra mul nodes for C
    def test_default_beta_no_extra_mul(self) -> None:
        """Gemm with beta=1.0 and C present must not emit a mul node for beta."""
        g = _make_gemm_graph(shape_a=(2, 3), shape_b=(3, 4), shape_y=(2, 4), shape_c=(4,))
        gm = emit_graph(g)
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        targets = [n.target for n in call_nodes]
        # matmul + add, but no mul
        assert torch.matmul in targets
        assert torch.add in targets
        assert torch.mul not in targets
