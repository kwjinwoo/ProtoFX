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


@register_op("Add", opset_range=(11, 21))
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


@register_op("Sub", opset_range=(11, 21))
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


@register_op("Mul", opset_range=(11, 21))
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


@register_op("Div", opset_range=(11, 21))
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


@register_op("Pow", opset_range=(11, 21))
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


@register_op("Sigmoid", opset_range=(11, 21))
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


@register_op("Tanh", opset_range=(11, 21))
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


@register_op("Abs", opset_range=(11, 21))
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


@register_op("Neg", opset_range=(11, 21))
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


@register_op("Exp", opset_range=(11, 21))
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


@register_op("Log", opset_range=(11, 21))
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


@register_op("Sqrt", opset_range=(11, 21))
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


@register_op("Erf", opset_range=(11, 21))
def _erf(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.erf`` for the ONNX Erf op.

    Args:
        node: The IR Erf node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Erf).

    Returns:
        A single-element list containing the erf FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.erf, args=(args[0],))]


# ---------------------------------------------------------------------------
# Comparison / logical ops
# ---------------------------------------------------------------------------


@register_op("Where", opset_range=(11, 21))
def _where(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.where`` for the ONNX Where op.

    Args:
        node: The IR Where node.
        args: Three-element list ``[condition, X, Y]``.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Where).

    Returns:
        A single-element list containing the where FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.where, args=(args[0], args[1], args[2]))]


@register_op("And", opset_range=(11, 21))
def _and(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.logical_and`` for the ONNX And op.

    Args:
        node: The IR And node.
        args: Two-element list containing the input FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for And).

    Returns:
        A single-element list containing the logical_and FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.logical_and, args=(args[0], args[1]))]


@register_op("IsNaN", opset_range=(13, 21))
def _isnan(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.isnan`` for the ONNX IsNaN op.

    Args:
        node: The IR IsNaN node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for IsNaN).

    Returns:
        A single-element list containing the isnan FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.isnan, args=(args[0],))]


@register_op("Equal", opset_range=(11, 21))
def _equal(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.eq`` for the ONNX Equal op.

    Args:
        node: The IR Equal node.
        args: Two-element list containing the input FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Equal).

    Returns:
        A single-element list containing the eq FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.eq, args=(args[0], args[1]))]


@register_op("Not", opset_range=(11, 21))
def _not(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.logical_not`` for the ONNX Not op.

    Args:
        node: The IR Not node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Not).

    Returns:
        A single-element list containing the logical_not FX call_function node.
    """
    import torch

    return [fx_graph.call_function(torch.logical_not, args=(args[0],))]
