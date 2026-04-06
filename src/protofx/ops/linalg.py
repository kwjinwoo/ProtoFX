"""Linear algebra ONNX op handlers (MatMul, Gemm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


@register_op("MatMul", opset_range=(11, 21))
def _matmul(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.matmul`` for the ONNX MatMul op.

    Args:
        node: The IR MatMul node.
        args: Two-element list containing input FX nodes (A, B).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for MatMul).

    Returns:
        A single-element list containing the matmul FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.matmul, args=(args[0], args[1]))]


@register_op("Gemm", opset_range=(11, 21))
def _gemm(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit FX nodes for the ONNX Gemm op.

    Computes ``Y = alpha * A' @ B' + beta * C`` where ``A'`` and ``B'`` are
    optionally transposed. When ``alpha`` or ``beta`` equal ``1.0``, the
    corresponding ``torch.mul`` node is elided.

    Args:
        node: The IR Gemm node.
        args: Two- or three-element list ``[A, B, C]``. ``C`` is ``None``
            when the optional bias input is omitted (sentinel).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Gemm).

    Returns:
        A single-element list containing the final result FX node.
    """
    import torch

    trans_a = node.attributes.get("transA", 0)
    trans_b = node.attributes.get("transB", 0)
    alpha = node.attributes.get("alpha", 1.0)
    beta = node.attributes.get("beta", 1.0)

    a: torch.fx.Node = args[0]  # type: ignore[assignment]
    b: torch.fx.Node = args[1]  # type: ignore[assignment]

    # Optional transpose nodes
    if trans_a:
        a = fx_graph.call_function(torch.transpose, args=(a, 0, 1))
    if trans_b:
        b = fx_graph.call_function(torch.transpose, args=(b, 0, 1))

    # Core matmul
    y = fx_graph.call_function(torch.matmul, args=(a, b))

    # Scale by alpha (elide when 1.0)
    if alpha != 1.0:
        y = fx_graph.call_function(torch.mul, args=(y, alpha))

    # Add bias C scaled by beta (elide mul when beta == 1.0)
    if len(args) > 2 and args[2] is not None:
        c: torch.fx.Node = args[2]  # type: ignore[assignment]
        if beta != 1.0:
            c = fx_graph.call_function(torch.mul, args=(c, beta))
        y = fx_graph.call_function(torch.add, args=(y, c))

    return [y]
