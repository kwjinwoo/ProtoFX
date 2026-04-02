"""Elementwise ONNX op handlers (Add, Sub, Mul, Div, Pow, Sigmoid, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


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


# ---------------------------------------------------------------------------
# Unary elementwise ops
# ---------------------------------------------------------------------------


@register_op("Sigmoid")
def _sigmoid(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.sigmoid`` for the ONNX Sigmoid op.

    Args:
        node: The IR Sigmoid node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Sigmoid).

    Returns:
        A single-element list containing the sigmoid FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.sigmoid, args=(args[0],))]


@register_op("Tanh")
def _tanh(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.tanh`` for the ONNX Tanh op.

    Args:
        node: The IR Tanh node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Tanh).

    Returns:
        A single-element list containing the tanh FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.tanh, args=(args[0],))]


@register_op("Abs")
def _abs(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.abs`` for the ONNX Abs op.

    Args:
        node: The IR Abs node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Abs).

    Returns:
        A single-element list containing the abs FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.abs, args=(args[0],))]


@register_op("Neg")
def _neg(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.neg`` for the ONNX Neg op.

    Args:
        node: The IR Neg node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Neg).

    Returns:
        A single-element list containing the neg FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.neg, args=(args[0],))]


@register_op("Exp")
def _exp(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.exp`` for the ONNX Exp op.

    Args:
        node: The IR Exp node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Exp).

    Returns:
        A single-element list containing the exp FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.exp, args=(args[0],))]


@register_op("Log")
def _log(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.log`` for the ONNX Log op.

    Args:
        node: The IR Log node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Log).

    Returns:
        A single-element list containing the log FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.log, args=(args[0],))]


@register_op("Sqrt")
def _sqrt(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.sqrt`` for the ONNX Sqrt op.

    Args:
        node: The IR Sqrt node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Sqrt).

    Returns:
        A single-element list containing the sqrt FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.sqrt, args=(args[0],))]
