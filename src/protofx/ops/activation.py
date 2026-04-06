"""Activation ONNX op handlers (Relu, Softmax, Gelu, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


@register_op("Relu", opset_range=(11, 21))
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


@register_op("Softmax", opset_range=(13, 21))
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


@register_op("LogSoftmax", opset_range=(13, 21))
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


@register_op("Gelu", opset_range=(20, 21))
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

    raw = node.attributes.get("approximate", "none")
    approximate = raw.decode() if isinstance(raw, bytes) else str(raw)
    return [fx_graph.call_function(F.gelu, args=(args[0],), kwargs={"approximate": approximate})]


@register_op("Elu", opset_range=(11, 21))
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


@register_op("LeakyRelu", opset_range=(11, 21))
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


@register_op("Selu", opset_range=(11, 21))
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


@register_op("Celu", opset_range=(12, 21))
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


@register_op("PRelu", opset_range=(11, 21))
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

    slope_node = args[1]
    # torch.prelu requires weight to be scalar or 1D; flatten if needed.
    slope_value = node.inputs[1]
    if slope_value.tensor_type.shape is not None and len(slope_value.tensor_type.shape) > 1:
        import torch

        slope_node = fx_graph.call_function(torch.flatten, args=(slope_node,))

    return [fx_graph.call_function(F.prelu, args=(args[0], slope_node))]


@register_op("HardSigmoid", opset_range=(11, 21))
def _hard_sigmoid(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``clamp(alpha * x + beta, 0, 1)`` for the ONNX HardSigmoid op.

    The ONNX spec defines ``HardSigmoid(x) = max(0, min(1, alpha * x + beta))``
    with ``alpha=0.2`` and ``beta=0.5`` defaults. This does **not** match PyTorch's
    ``F.hardsigmoid`` formula, so we emit the arithmetic manually.

    Args:
        node: The IR HardSigmoid node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for HardSigmoid).

    Returns:
        A single-element list containing the clamped FX node.
    """
    import torch

    alpha = float(node.attributes.get("alpha", 0.2))
    beta = float(node.attributes.get("beta", 0.5))

    mul_node = fx_graph.call_function(torch.mul, args=(args[0], alpha))
    add_node = fx_graph.call_function(torch.add, args=(mul_node, beta))
    return [fx_graph.call_function(torch.clamp, args=(add_node,), kwargs={"min": 0.0, "max": 1.0})]


@register_op("HardSwish", opset_range=(14, 21))
def _hard_swish(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.hardswish`` for the ONNX HardSwish op (opset 14).

    Args:
        node: The IR HardSwish node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for HardSwish).

    Returns:
        A single-element list containing the hardswish FX call_function node.
    """
    import torch.nn.functional as F

    return [fx_graph.call_function(F.hardswish, args=(args[0],))]


@register_op("Mish", opset_range=(18, 21))
def _mish(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.mish`` for the ONNX Mish op (opset 18).

    Args:
        node: The IR Mish node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Mish).

    Returns:
        A single-element list containing the mish FX call_function node.
    """
    import torch.nn.functional as F

    return [fx_graph.call_function(F.mish, args=(args[0],))]


@register_op("Softplus", opset_range=(11, 21))
def _softplus(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.softplus`` for the ONNX Softplus op.

    Args:
        node: The IR Softplus node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Softplus).

    Returns:
        A single-element list containing the softplus FX call_function node.
    """
    import torch.nn.functional as F

    return [fx_graph.call_function(F.softplus, args=(args[0],))]


@register_op("Softsign", opset_range=(11, 21))
def _softsign(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``x / (1 + abs(x))`` for the ONNX Softsign op.

    PyTorch does not provide a built-in ``F.softsign``, so the formula is
    emitted manually using primitive ops.

    Args:
        node: The IR Softsign node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Softsign).

    Returns:
        A single-element list containing the softsign FX node.
    """
    import torch

    abs_node = fx_graph.call_function(torch.abs, args=(args[0],))
    denom = fx_graph.call_function(torch.add, args=(abs_node, 1.0))
    return [fx_graph.call_function(torch.div, args=(args[0], denom))]


@register_op("ThresholdedRelu", opset_range=(10, 21))
def _thresholded_relu(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.where(x > alpha, x, 0)`` for the ONNX ThresholdedRelu op.

    Supports the ``alpha`` attribute (default ``1.0``).

    Args:
        node: The IR ThresholdedRelu node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ThresholdedRelu).

    Returns:
        A single-element list containing the thresholded relu FX node.
    """
    import torch

    alpha = float(node.attributes.get("alpha", 1.0))

    gt_node = fx_graph.call_function(torch.gt, args=(args[0], alpha))
    zeros_node = fx_graph.call_function(torch.zeros_like, args=(args[0],))
    return [fx_graph.call_function(torch.where, args=(gt_node, args[0], zeros_node))]
