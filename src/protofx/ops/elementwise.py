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


# ---------------------------------------------------------------------------
# Binary elementwise ops
# ---------------------------------------------------------------------------


@register_op("Add")
def _add(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.add`` for the ONNX Add op.

    Args:
        node: The IR Add node.
        args: Two-element list containing input FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Add).

    Returns:
        A single-element list containing the add FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.add, args=(args[0], args[1]))]


@register_op("Sub")
def _sub(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.sub`` for the ONNX Sub op.

    Args:
        node: The IR Sub node.
        args: Two-element list containing input FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Sub).

    Returns:
        A single-element list containing the sub FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.sub, args=(args[0], args[1]))]


@register_op("Mul")
def _mul(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.mul`` for the ONNX Mul op.

    Args:
        node: The IR Mul node.
        args: Two-element list containing input FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Mul).

    Returns:
        A single-element list containing the mul FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.mul, args=(args[0], args[1]))]


@register_op("Div")
def _div(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.div`` for the ONNX Div op.

    Args:
        node: The IR Div node.
        args: Two-element list containing input FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Div).

    Returns:
        A single-element list containing the div FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.div, args=(args[0], args[1]))]


@register_op("Pow")
def _pow(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.pow`` for the ONNX Pow op.

    Args:
        node: The IR Pow node.
        args: Two-element list containing input FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Pow).

    Returns:
        A single-element list containing the pow FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.pow, args=(args[0], args[1]))]
