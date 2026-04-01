"""Normalization ONNX op handlers (BatchNormalization, LayerNormalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


@register_op("BatchNormalization")
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
