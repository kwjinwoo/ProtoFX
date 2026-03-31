"""Reduction ONNX op handlers (ReduceMean, ReduceSum, ReduceMax, ReduceMin, etc.)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_axes(node: Node) -> list[int] | None:
    """Read reduction axes from either an attribute or a second input tensor.

    Supports both opset < 18 (axes as attribute) and opset >= 18 (axes as
    second input tensor) styles.

    Args:
        node: The IR reduction node.

    Returns:
        A list of axis ints, or ``None`` if no axes are specified.
    """
    # opset < 18: axes stored as attribute
    attr_axes = node.attributes.get("axes")
    if attr_axes is not None:
        return [int(a) for a in attr_axes]

    # opset >= 18: axes as second input tensor
    if len(node.inputs) >= 2:
        value = node.inputs[1]
        if value.data is not None:
            return [int(v) for v in value.data.flat]

    return None


def _read_keepdims(node: Node) -> bool:
    """Read the keepdims attribute from a reduction node.

    Args:
        node: The IR reduction node.

    Returns:
        ``True`` if keepdims is 1 (default), ``False`` if 0.
    """
    return bool(node.attributes.get("keepdims", 1))


def _read_noop_with_empty_axes(node: Node) -> bool:
    """Read the noop_with_empty_axes attribute from a reduction node.

    Args:
        node: The IR reduction node.

    Returns:
        ``True`` if noop_with_empty_axes is 1, ``False`` if 0 (default).
    """
    return bool(node.attributes.get("noop_with_empty_axes", 0))


def _make_simple_reduce_handler(
    op_type: str,
    torch_fn_path: str,
) -> Callable[..., list[torch.fx.Node]]:
    """Create a reduce handler for ops that map directly to a single torch function.

    The generated handler reads axes/keepdims/noop_with_empty_axes and calls
    the specified torch function with ``dim`` and ``keepdim`` kwargs.

    Args:
        op_type: The ONNX op type string for registration.
        torch_fn_path: Dotted attribute path on the ``torch`` module (e.g. ``"mean"``).

    Returns:
        A registered op handler function.
    """

    @register_op(op_type)
    def _handler(
        node: Node,
        args: list[torch.fx.Node | None],
        fx_graph: torch.fx.Graph,
        module: torch.nn.Module,
    ) -> list[torch.fx.Node]:
        """Emit a torch reduction function for the ONNX reduce op.

        Args:
            node: The IR reduction node.
            args: Input FX nodes; first element is the data tensor.
            fx_graph: The FX graph being constructed.
            module: The root module (unused for reduction).

        Returns:
            A single-element list containing the reduction FX call_function node.
        """
        import torch as _torch

        torch_fn = getattr(_torch, torch_fn_path)
        axes = _read_axes(node)
        keepdims = _read_keepdims(node)
        noop = _read_noop_with_empty_axes(node)

        # noop_with_empty_axes=1 and no axes: pass through input unchanged
        if axes is None and noop:
            return [args[0]]

        kwargs: dict[str, object] = {"keepdim": keepdims}
        if axes is not None:
            dim = axes[0] if len(axes) == 1 else axes
            kwargs["dim"] = dim
        else:
            # torch.mean/sum require explicit dim when keepdim is used;
            # pass all dims to reduce over the entire tensor.
            ndim = len(node.inputs[0].tensor_type.shape)
            kwargs["dim"] = list(range(ndim))

        return [fx_graph.call_function(torch_fn, args=(args[0],), kwargs=kwargs)]

    return _handler


# ---------------------------------------------------------------------------
# Register simple reduction ops
# ---------------------------------------------------------------------------

_reduce_mean = _make_simple_reduce_handler("ReduceMean", "mean")
_reduce_sum = _make_simple_reduce_handler("ReduceSum", "sum")
_reduce_max = _make_simple_reduce_handler("ReduceMax", "amax")
_reduce_min = _make_simple_reduce_handler("ReduceMin", "amin")
_reduce_logsumexp = _make_simple_reduce_handler("ReduceLogSumExp", "logsumexp")
