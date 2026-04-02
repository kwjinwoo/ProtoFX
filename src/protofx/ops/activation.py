"""Activation ONNX op handlers (Relu, Softmax, Gelu, etc.)."""

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


@register_op("Softmax")
def _softmax(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.softmax`` for the ONNX Softmax op (opset ≥ 13).

    Supports the ``axis`` attribute (default ``1``). Negative axis values are
    passed through directly to ``F.softmax`` which handles them natively.

    Args:
        node: The IR Softmax node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Softmax).

    Returns:
        A single-element list containing the softmax FX call_function node.
    """
    import torch.nn.functional as F

    axis = node.attributes.get("axis", 1)
    return [fx_graph.call_function(F.softmax, args=(args[0],), kwargs={"dim": int(axis)})]


@register_op("LogSoftmax")
def _log_softmax(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.log_softmax`` for the ONNX LogSoftmax op (opset ≥ 13).

    Supports the ``axis`` attribute (default ``1``). Negative axis values are
    passed through directly to ``F.log_softmax`` which handles them natively.

    Args:
        node: The IR LogSoftmax node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for LogSoftmax).

    Returns:
        A single-element list containing the log_softmax FX call_function node.
    """
    import torch.nn.functional as F

    axis = node.attributes.get("axis", 1)
    return [fx_graph.call_function(F.log_softmax, args=(args[0],), kwargs={"dim": int(axis)})]
