"""Op handler registry for ONNX-to-FX operator lowering.

Each ONNX op handler is registered with the ``@register_op`` decorator
and dispatched during emission via ``dispatch_op``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _OpEntry:
    """Registry entry pairing an op handler with its supported opset range.

    Attributes:
        handler: The op handler function.
        opset_range: Inclusive ``(min_opset, max_opset)`` range, or ``None``
            if the handler does not declare version constraints.
    """

    handler: OpHandler
    opset_range: tuple[int, int] | None


_REGISTRY: dict[str, _OpEntry] = {}


def register_op(
    op_type: str,
    *,
    opset_range: tuple[int, int] | None = None,
) -> Callable[[OpHandler], OpHandler]:
    """Decorator that registers an op handler for the given ONNX op type.

    Args:
        op_type: The ONNX operator type string (e.g. ``"Relu"``).
        opset_range: Inclusive ``(min_opset, max_opset)`` of supported
            ONNX opset versions. ``None`` means no version constraint.

    Returns:
        A decorator that registers and returns the handler function.

    Raises:
        ValueError: If a handler for *op_type* is already registered.
    """

    def decorator(fn: OpHandler) -> OpHandler:
        if op_type in _REGISTRY:
            msg = f"op handler already registered: {op_type}"
            raise ValueError(msg)
        _REGISTRY[op_type] = _OpEntry(handler=fn, opset_range=opset_range)
        return fn

    return decorator


def dispatch_op(op_type: str, opset_version: int | None = None) -> OpHandler:
    """Look up the registered handler for an ONNX op type.

    When *opset_version* is provided and the handler declares an
    ``opset_range``, the version is validated against the range.

    Args:
        op_type: The ONNX operator type string.
        opset_version: The model's opset version, or ``None`` to skip
            version checking.

    Returns:
        The registered handler function.

    Raises:
        NotImplementedError: If no handler is registered for *op_type*,
            or if *opset_version* falls outside the handler's declared range.
    """
    entry = _REGISTRY.get(op_type)
    if entry is None:
        msg = f"no op handler registered for: {op_type}"
        raise NotImplementedError(msg)

    if opset_version is not None and entry.opset_range is not None:
        lo, hi = entry.opset_range
        if not (lo <= opset_version <= hi):
            msg = f"opset version {opset_version} is not supported for {op_type} (supported: {lo}-{hi})"
            raise NotImplementedError(msg)

    return entry.handler


def list_registry() -> dict[str, tuple[int, int] | None]:
    """Return a snapshot of all registered ops and their opset ranges.

    Returns:
        A dict mapping op type strings to their ``(min_opset, max_opset)``
        range, or ``None`` if no range is declared.
    """
    return {op_type: entry.opset_range for op_type, entry in _REGISTRY.items()}
