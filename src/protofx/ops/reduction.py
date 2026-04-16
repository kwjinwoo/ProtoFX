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
    *,
    opset_range: tuple[int, int] | None = None,
) -> Callable[..., list[torch.fx.Node]]:
    """Create a reduce handler for ops that map directly to a single torch function.

    The generated handler reads axes/keepdims/noop_with_empty_axes and calls
    the specified torch function with ``dim`` and ``keepdim`` kwargs.

    Args:
        op_type: The ONNX op type string for registration.
        torch_fn_path: Dotted attribute path on the ``torch`` module (e.g. ``"mean"``).
        opset_range: Inclusive ``(min_opset, max_opset)`` of supported
            ONNX opset versions. ``None`` means no version constraint.

    Returns:
        A registered op handler function.
    """

    @register_op(op_type, opset_range=opset_range)
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

_reduce_mean = _make_simple_reduce_handler("ReduceMean", "mean", opset_range=(11, 21))
_reduce_sum = _make_simple_reduce_handler("ReduceSum", "sum", opset_range=(11, 21))
_reduce_max = _make_simple_reduce_handler("ReduceMax", "amax", opset_range=(11, 21))
_reduce_min = _make_simple_reduce_handler("ReduceMin", "amin", opset_range=(11, 21))
_reduce_logsumexp = _make_simple_reduce_handler("ReduceLogSumExp", "logsumexp", opset_range=(11, 21))


# ---------------------------------------------------------------------------
# ReduceProd — requires iterative single-dim reduction
# ---------------------------------------------------------------------------


@register_op("ReduceProd", opset_range=(11, 21))
def _reduce_prod(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.prod`` for the ONNX ReduceProd op.

    ``torch.prod`` only supports a single dim, so multi-axis reduction is
    handled by applying ``torch.prod`` iteratively in descending axis order
    to keep indices stable.

    Args:
        node: The IR ReduceProd node.
        args: Input FX nodes; first element is the data tensor.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ReduceProd).

    Returns:
        A single-element list containing the prod FX call_function node.
    """
    import torch as _torch

    axes = _read_axes(node)
    keepdims = _read_keepdims(node)
    noop = _read_noop_with_empty_axes(node)

    if axes is None and noop:
        return [args[0]]

    if axes is None:
        axes = list(range(len(node.inputs[0].tensor_type.shape)))

    result = args[0]
    for ax in sorted(axes, reverse=True):
        result = fx_graph.call_function(_torch.prod, args=(result,), kwargs={"dim": ax, "keepdim": keepdims})
    return [result]


# ---------------------------------------------------------------------------
# Compound reduction helpers
# ---------------------------------------------------------------------------


def _resolve_reduce_kwargs(node: Node) -> tuple[list[int], bool, bool]:
    """Resolve axes, keepdims, and noop for compound reduce handlers.

    Args:
        node: The IR reduction node.

    Returns:
        A tuple of (axes, keepdims, noop).
    """
    axes = _read_axes(node)
    keepdims = _read_keepdims(node)
    noop = _read_noop_with_empty_axes(node)
    if axes is None and not noop:
        axes = list(range(len(node.inputs[0].tensor_type.shape)))
    return axes, keepdims, noop


def _sum_kwargs(axes: list[int], keepdims: bool) -> dict[str, object]:
    """Build kwargs dict for a ``torch.sum`` call inside a compound reduce.

    Args:
        axes: Reduction axes.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        kwargs dict with ``dim`` and ``keepdim``.
    """
    dim: int | list[int] = axes[0] if len(axes) == 1 else axes
    return {"dim": dim, "keepdim": keepdims}


# ---------------------------------------------------------------------------
# ReduceL1 — abs → sum (compile-friendly, avoids torch.norm)
# ---------------------------------------------------------------------------


@register_op("ReduceL1", opset_range=(11, 21))
def _reduce_l1(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.abs`` + ``torch.sum`` for the ONNX ReduceL1 op.

    Uses primitive ops instead of ``torch.norm`` for better ``torch.compile``
    fusion opportunities.

    Args:
        node: The IR ReduceL1 node.
        args: Input FX nodes; first element is the data tensor.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ReduceL1).

    Returns:
        A single-element list containing the final FX node.
    """
    import torch as _torch

    axes, keepdims, noop = _resolve_reduce_kwargs(node)
    if axes is None and noop:
        return [args[0]]

    abs_node = fx_graph.call_function(_torch.abs, args=(args[0],))
    return [fx_graph.call_function(_torch.sum, args=(abs_node,), kwargs=_sum_kwargs(axes, keepdims))]


# ---------------------------------------------------------------------------
# ReduceL2 — square → sum → sqrt (compile-friendly, avoids torch.norm)
# ---------------------------------------------------------------------------


@register_op("ReduceL2", opset_range=(11, 21))
def _reduce_l2(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.square`` + ``torch.sum`` + ``torch.sqrt`` for ReduceL2.

    Uses primitive ops instead of ``torch.norm`` for better ``torch.compile``
    fusion opportunities.

    Args:
        node: The IR ReduceL2 node.
        args: Input FX nodes; first element is the data tensor.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ReduceL2).

    Returns:
        A single-element list containing the final FX node.
    """
    import torch as _torch

    axes, keepdims, noop = _resolve_reduce_kwargs(node)
    if axes is None and noop:
        return [args[0]]

    sq_node = fx_graph.call_function(_torch.square, args=(args[0],))
    sum_node = fx_graph.call_function(_torch.sum, args=(sq_node,), kwargs=_sum_kwargs(axes, keepdims))
    return [fx_graph.call_function(_torch.sqrt, args=(sum_node,))]


# ---------------------------------------------------------------------------
# ReduceLogSum — sum → log
# ---------------------------------------------------------------------------


@register_op("ReduceLogSum", opset_range=(11, 21))
def _reduce_logsum(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.sum`` + ``torch.log`` for the ONNX ReduceLogSum op.

    Args:
        node: The IR ReduceLogSum node.
        args: Input FX nodes; first element is the data tensor.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ReduceLogSum).

    Returns:
        A single-element list containing the final FX node.
    """
    import torch as _torch

    axes, keepdims, noop = _resolve_reduce_kwargs(node)
    if axes is None and noop:
        return [args[0]]

    sum_node = fx_graph.call_function(_torch.sum, args=(args[0],), kwargs=_sum_kwargs(axes, keepdims))
    return [fx_graph.call_function(_torch.log, args=(sum_node,))]


# ---------------------------------------------------------------------------
# ReduceSumSquare — square → sum
# ---------------------------------------------------------------------------


@register_op("ReduceSumSquare", opset_range=(11, 21))
def _reduce_sumsquare(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.square`` + ``torch.sum`` for the ONNX ReduceSumSquare op.

    Args:
        node: The IR ReduceSumSquare node.
        args: Input FX nodes; first element is the data tensor.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ReduceSumSquare).

    Returns:
        A single-element list containing the final FX node.
    """
    import torch as _torch

    axes, keepdims, noop = _resolve_reduce_kwargs(node)
    if axes is None and noop:
        return [args[0]]

    sq_node = fx_graph.call_function(_torch.square, args=(args[0],))
    return [fx_graph.call_function(_torch.sum, args=(sq_node,), kwargs=_sum_kwargs(axes, keepdims))]


# ---------------------------------------------------------------------------
# CumSum
# ---------------------------------------------------------------------------


def _cumsum_impl(x: torch.Tensor, dim: int, exclusive: bool, reverse: bool) -> torch.Tensor:
    """Pure-Python CumSum implementation matching ONNX semantics.

    Args:
        x: The input tensor.
        dim: The axis along which to compute cumulative sum.
        exclusive: If ``True``, exclude the current element.
        reverse: If ``True``, compute in reverse direction.

    Returns:
        Cumulative sum tensor with the same shape as *x*.
    """
    import torch

    if reverse:
        x = torch.flip(x, dims=[dim])

    result = torch.cumsum(x, dim=dim)

    if exclusive:
        result = torch.roll(result, shifts=1, dims=dim)
        result = result.index_fill(dim, torch.tensor([0], device=x.device), 0.0)

    if reverse:
        result = torch.flip(result, dims=[dim])

    return result


@register_op("CumSum", opset_range=(11, 21))
def _cumsum(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit a CumSum call for the ONNX CumSum op.

    Supports ``exclusive`` and ``reverse`` attributes. The axis is statically
    extracted from the second input.

    Args:
        node: The IR CumSum node.
        args: Two-element list; first is the data FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for CumSum).

    Returns:
        A single-element list containing the CumSum FX call_function node.
    """
    axis_value = node.inputs[1]
    if axis_value.data is None:
        msg = "CumSum: axis input has no static data"
        raise NotImplementedError(msg)
    dim = int(axis_value.data.flat[0])

    exclusive = bool(node.attributes.get("exclusive", 0))
    reverse = bool(node.attributes.get("reverse", 0))

    return [fx_graph.call_function(_cumsum_impl, args=(args[0], dim, exclusive, reverse))]


# ---------------------------------------------------------------------------
# ArgMax
# ---------------------------------------------------------------------------


@register_op("ArgMax", opset_range=(11, 21))
def _argmax(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.argmax`` for the ONNX ArgMax op.

    Reads ``axis`` (default 0), ``keepdims`` (default 1), and
    ``select_last_index`` (default 0) attributes from the IR node.

    ``select_last_index=1`` is not supported because ``torch.argmax``
    always returns the first occurrence.

    Args:
        node: The IR ArgMax node.
        args: Input FX nodes; first element is the data tensor.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ArgMax).

    Returns:
        A single-element list containing the ArgMax FX call_function node.

    Raises:
        NotImplementedError: If ``select_last_index=1`` is requested.
    """
    import torch as _torch

    axis = int(node.attributes.get("axis", 0))
    keepdims = bool(node.attributes.get("keepdims", 1))
    select_last_index = int(node.attributes.get("select_last_index", 0))

    if select_last_index:
        msg = "ArgMax: select_last_index=1 is not supported"
        raise NotImplementedError(msg)

    return [fx_graph.call_function(_torch.argmax, args=(args[0],), kwargs={"dim": axis, "keepdim": keepdims})]
