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


@register_op("Gelu")
def _gelu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.gelu`` for the ONNX Gelu op (opset 20).

    Supports the ``approximate`` attribute: ``"none"`` (default) for the exact
    formulation, ``"tanh"`` for the fast tanh approximation.

    Args:
        node: The IR Gelu node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Gelu).

    Returns:
        A single-element list containing the gelu FX call_function node.
    """
    import torch.nn.functional as F

    approximate = str(node.attributes.get("approximate", "none"))
    return [fx_graph.call_function(F.gelu, args=(args[0],), kwargs={"approximate": approximate})]


@register_op("Elu")
def _elu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.elu`` for the ONNX Elu op.

    Supports the ``alpha`` attribute (default ``1.0``).

    Args:
        node: The IR Elu node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Elu).

    Returns:
        A single-element list containing the elu FX call_function node.
    """
    import torch.nn.functional as F

    alpha = float(node.attributes.get("alpha", 1.0))
    return [fx_graph.call_function(F.elu, args=(args[0],), kwargs={"alpha": alpha})]


@register_op("LeakyRelu")
def _leaky_relu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.leaky_relu`` for the ONNX LeakyRelu op.

    Supports the ``alpha`` attribute (default ``0.01``), mapped to the PyTorch
    ``negative_slope`` parameter.

    Args:
        node: The IR LeakyRelu node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for LeakyRelu).

    Returns:
        A single-element list containing the leaky_relu FX call_function node.
    """
    import torch.nn.functional as F

    alpha = float(node.attributes.get("alpha", 0.01))
    return [fx_graph.call_function(F.leaky_relu, args=(args[0],), kwargs={"negative_slope": alpha})]


@register_op("Selu")
def _selu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.selu`` for the ONNX Selu op.

    The ONNX spec defines fixed ``alpha`` and ``gamma`` constants that match
    the PyTorch ``F.selu`` implementation exactly.

    Args:
        node: The IR Selu node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Selu).

    Returns:
        A single-element list containing the selu FX call_function node.
    """
    import torch.nn.functional as F

    return [fx_graph.call_function(F.selu, args=(args[0],))]


@register_op("Celu")
def _celu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.celu`` for the ONNX Celu op.

    Supports the ``alpha`` attribute (default ``1.0``).

    Args:
        node: The IR Celu node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Celu).

    Returns:
        A single-element list containing the celu FX call_function node.
    """
    import torch.nn.functional as F

    alpha = float(node.attributes.get("alpha", 1.0))
    return [fx_graph.call_function(F.celu, args=(args[0],), kwargs={"alpha": alpha})]


@register_op("PRelu")
def _prelu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.prelu`` for the ONNX PRelu op.

    ONNX inputs: ``[X, slope]``. The slope tensor is passed as the ``weight``
    argument to ``F.prelu``.

    Args:
        node: The IR PRelu node.
        args: Two-element list ``[X, slope]`` containing FX nodes.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for PRelu).

    Returns:
        A single-element list containing the prelu FX call_function node.
    """
    import torch.nn.functional as F

    return [fx_graph.call_function(F.prelu, args=(args[0], args[1]))]
