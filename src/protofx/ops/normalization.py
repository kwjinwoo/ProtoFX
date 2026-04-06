"""Normalization ONNX op handlers (BatchNormalization, LayerNormalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


@register_op("BatchNormalization", opset_range=(15, 21))
def _batch_normalization(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.batch_norm`` for the ONNX BatchNormalization op (opset 15).

    Only inference mode (``training_mode=0``) is supported. Training mode
    raises ``NotImplementedError``.

    ONNX inputs: ``[X, scale, B, input_mean, input_var]``
    Maps to: ``F.batch_norm(X, input_mean, input_var, weight=scale, bias=B, training=False, eps=epsilon)``

    Args:
        node: The IR BatchNormalization node.
        args: Five-element list ``[X, scale, B, input_mean, input_var]``.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for BatchNormalization).

    Returns:
        A single-element list containing the batch_norm FX call_function node.

    Raises:
        NotImplementedError: If ``training_mode`` is ``1``.
    """
    import torch.nn.functional as F

    training_mode = node.attributes.get("training_mode", 0)
    if training_mode != 0:
        msg = "BatchNormalization: training_mode=1 is not supported"
        raise NotImplementedError(msg)

    epsilon = node.attributes.get("epsilon", 1e-5)

    x_node = args[0]
    scale_node = args[1]
    b_node = args[2]
    mean_node = args[3]
    var_node = args[4]

    return [
        fx_graph.call_function(
            F.batch_norm,
            args=(x_node, mean_node, var_node),
            kwargs={
                "weight": scale_node,
                "bias": b_node,
                "training": False,
                "eps": float(epsilon),  # type: ignore[arg-type]
            },
        )
    ]


@register_op("LayerNormalization", opset_range=(17, 21))
def _layer_normalization(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.layer_norm`` for the ONNX LayerNormalization op (opset 17).

    The ``axis`` attribute determines the ``normalized_shape`` by slicing
    ``X.shape[axis:]``. Negative axis values are supported.

    ONNX inputs: ``[X, scale, B]`` where ``B`` is optional (sentinel when omitted).
    Maps to: ``F.layer_norm(X, normalized_shape, weight=scale, bias=B, eps=epsilon)``

    Args:
        node: The IR LayerNormalization node.
        args: Three-element list ``[X, scale, B]``. ``B`` is ``None``
            when the optional bias input is omitted (sentinel).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for LayerNormalization).

    Returns:
        A single-element list containing the layer_norm FX call_function node.

    Raises:
        NotImplementedError: If the input shape is unavailable and ``normalized_shape``
            cannot be determined.
    """
    import torch.nn.functional as F

    axis = node.attributes.get("axis", -1)
    epsilon = node.attributes.get("epsilon", 1e-5)

    x_value = node.inputs[0]
    if x_value.tensor_type.shape is None:
        msg = "LayerNormalization: cannot determine normalized_shape (input shape unknown)"
        raise NotImplementedError(msg)

    x_shape = x_value.tensor_type.shape
    ndim = len(x_shape)
    resolved_axis = int(axis) if int(axis) >= 0 else ndim + int(axis)  # type: ignore[arg-type]
    normalized_shape = list(x_shape[resolved_axis:])

    x_node = args[0]
    scale_node = args[1]
    b_node = args[2] if len(args) > 2 else None  # None if sentinel or omitted

    return [
        fx_graph.call_function(
            F.layer_norm,
            args=(x_node, normalized_shape),
            kwargs={
                "weight": scale_node,
                "bias": b_node,
                "eps": float(epsilon),  # type: ignore[arg-type]
            },
        )
    ]
