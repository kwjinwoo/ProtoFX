"""Linear algebra ONNX op handlers (MatMul, Gemm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


@register_op("MatMul")
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
