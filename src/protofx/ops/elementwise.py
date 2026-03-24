"""Elementwise ONNX op handlers (Relu, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


@register_op("Relu")
def _relu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.relu`` for the ONNX Relu op.

    Args:
        node: The IR Relu node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Relu).

    Returns:
        A single-element list containing the relu FX call_function node.
    """
    import torch.nn.functional

    return [fx_graph.call_function(torch.nn.functional.relu, args=(args[0],))]
