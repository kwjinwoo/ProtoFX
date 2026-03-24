"""Op handler registry for ONNX-to-FX operator lowering.

Each ONNX op handler is registered with the ``@register_op`` decorator
and dispatched during emission via ``dispatch_op``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node

type OpHandler = Callable[
    ["Node", list["torch.fx.Node | None"], "torch.fx.Graph", "torch.nn.Module"],
    list["torch.fx.Node"],
]
"""Signature for an op handler function.

Args:
    node: The IR node being lowered.
    args: FX node references (or None for sentinels) matching the IR node's inputs.
    fx_graph: The ``torch.fx.Graph`` being constructed.
    module: The root ``torch.nn.Module`` for buffer/parameter registration.

Returns:
    A list of ``torch.fx.Node`` instances, one per IR node output.
"""

_REGISTRY: dict[str, OpHandler] = {}


def register_op(op_type: str) -> Callable[[OpHandler], OpHandler]:
    """Decorator that registers an op handler for the given ONNX op type.

    Args:
        op_type: The ONNX operator type string (e.g. ``"Relu"``).

    Returns:
        A decorator that registers and returns the handler function.

    Raises:
        ValueError: If a handler for *op_type* is already registered.
    """

    def decorator(fn: OpHandler) -> OpHandler:
        if op_type in _REGISTRY:
            msg = f"op handler already registered: {op_type}"
            raise ValueError(msg)
        _REGISTRY[op_type] = fn
        return fn

    return decorator


def dispatch_op(op_type: str) -> OpHandler:
    """Look up the registered handler for an ONNX op type.

    Args:
        op_type: The ONNX operator type string.

    Returns:
        The registered handler function.

    Raises:
        NotImplementedError: If no handler is registered for *op_type*.
    """
    handler = _REGISTRY.get(op_type)
    if handler is None:
        msg = f"no op handler registered for: {op_type}"
        raise NotImplementedError(msg)
    return handler
