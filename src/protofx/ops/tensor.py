"""Tensor manipulation ONNX op handlers (Reshape, Transpose, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


def _extract_static_int_data(node: Node, input_index: int) -> tuple[int, ...]:
    """Extract a static int tuple from an IR node's input Value data.

    Used for shape/axes inputs that are stored as initializers or constants.

    Args:
        node: The IR node.
        input_index: Positional index of the input Value to read.

    Returns:
        A tuple of ints extracted from the Value's numpy data.

    Raises:
        NotImplementedError: If the input Value has no static data.
    """
    value = node.inputs[input_index]
    if value.data is None:
        msg = f"{node.op_type}: input {input_index} ({value.name or value.id}) has no static data"
        raise NotImplementedError(msg)
    return tuple(int(v) for v in value.data.flat)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


@register_op("Reshape")
def _reshape(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.reshape`` for the ONNX Reshape op.

    The target shape is statically extracted from the second input Value's
    data (initializer or constant). Dynamic shapes are not yet supported.

    Args:
        node: The IR Reshape node.
        args: Two-element list; first is the data FX node, second is the shape FX node (unused).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Reshape).

    Returns:
        A single-element list containing the reshape FX call_function node.
    """
    import torch

    target_shape = _extract_static_int_data(node, 1)
    return [fx_graph.call_function(torch.reshape, args=(args[0], target_shape))]


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


@register_op("Transpose")
def _transpose(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.permute`` for the ONNX Transpose op.

    The permutation order is read from the ``perm`` attribute on the IR node.

    Args:
        node: The IR Transpose node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Transpose).

    Returns:
        A single-element list containing the permute FX call_function node.
    """
    import torch

    perm = node.attributes.get("perm")
    if perm is None:
        msg = "Transpose: missing required 'perm' attribute"
        raise NotImplementedError(msg)
    return [fx_graph.call_function(torch.permute, args=(args[0], list(perm)))]


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


@register_op("Flatten")
def _flatten(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.reshape`` for the ONNX Flatten op.

    ONNX Flatten always produces a 2D output by splitting at *axis*:
    ``(product(d[:axis]), product(d[axis:]))``. The target shape is read
    from the IR node's output tensor type.

    Args:
        node: The IR Flatten node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Flatten).

    Returns:
        A single-element list containing the reshape FX call_function node.
    """
    import torch

    output_shape = node.outputs[0].tensor_type.shape
    if output_shape is None:
        msg = "Flatten: output tensor type has no static shape"
        raise NotImplementedError(msg)
    return [fx_graph.call_function(torch.reshape, args=(args[0], tuple(int(d) for d in output_shape)))]


# ---------------------------------------------------------------------------
# Squeeze (opset 13+: axes as optional input tensor)
# ---------------------------------------------------------------------------


@register_op("Squeeze")
def _squeeze(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.squeeze`` for the ONNX Squeeze op (opset 13+).

    If an axes input is provided (second input), axes are statically extracted
    and applied in descending order. If no axes input is present, all dims
    of size 1 are squeezed.

    Args:
        node: The IR Squeeze node.
        args: One or two element list; first is the data FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Squeeze).

    Returns:
        A single-element list containing the squeeze FX call_function node.
    """
    import torch

    if len(node.inputs) < 2:
        # No axes specified — squeeze all dims of size 1
        return [fx_graph.call_function(torch.squeeze, args=(args[0],))]

    axes = _extract_static_int_data(node, 1)
    # Apply squeezes in descending axis order to keep indices stable
    result = args[0]
    for ax in sorted(axes, reverse=True):
        result = fx_graph.call_function(torch.squeeze, args=(result, int(ax)))
    return [result]


# ---------------------------------------------------------------------------
# Unsqueeze (opset 13+: axes as input tensor)
# ---------------------------------------------------------------------------


@register_op("Unsqueeze")
def _unsqueeze(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.unsqueeze`` for the ONNX Unsqueeze op (opset 13+).

    Axes are statically extracted from the second input and applied
    in ascending order (after normalizing negatives) to keep indices stable.

    Args:
        node: The IR Unsqueeze node.
        args: Two-element list; first is the data FX node, second is axes (unused).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Unsqueeze).

    Returns:
        A single-element list containing the unsqueeze FX call_function node.
    """
    import torch

    axes = _extract_static_int_data(node, 1)
    ndim_out = len(node.inputs[0].tensor_type.shape) + len(axes)
    sorted_axes = sorted(a if a >= 0 else a + ndim_out for a in axes)
    result = args[0]
    for ax in sorted_axes:
        result = fx_graph.call_function(torch.unsqueeze, args=(result, ax))
    return [result]


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


@register_op("Concat")
def _concat(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.cat`` for the ONNX Concat op.

    All inputs are concatenated along the ``axis`` attribute.

    Args:
        node: The IR Concat node.
        args: List of FX nodes to concatenate.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Concat).

    Returns:
        A single-element list containing the cat FX call_function node.
    """
    import torch

    axis = node.attributes.get("axis", 0)
    return [fx_graph.call_function(torch.cat, args=(list(args), int(axis)))]


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


def _extract_optional_int_data(node: Node, input_index: int) -> tuple[int, ...] | None:
    """Extract a static int tuple from an optional IR input, or return ``None``.

    Returns ``None`` if the input index is out of range or the input is a sentinel.

    Args:
        node: The IR node.
        input_index: Positional index of the input Value to read.

    Returns:
        A tuple of ints, or ``None`` if the input is missing or sentinel.
    """
    from protofx.ir.value import ValueKind

    if input_index >= len(node.inputs):
        return None
    value = node.inputs[input_index]
    if value.kind == ValueKind.SENTINEL:
        return None
    if value.data is None:
        msg = f"{node.op_type}: input {input_index} ({value.name or value.id}) has no static data"
        raise NotImplementedError(msg)
    return tuple(int(v) for v in value.data.flat)


@register_op("Slice")
def _slice(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``operator.getitem`` with slice objects for the ONNX Slice op.

    All slice parameters (starts, ends, axes, steps) are statically extracted.
    The result is emitted as ``operator.getitem`` with constructed slice objects.

    Args:
        node: The IR Slice node.
        args: FX node list; first element is the data tensor.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Slice).

    Returns:
        A single-element list containing the sliced FX node.
    """
    import operator

    starts = _extract_static_int_data(node, 1)
    ends = _extract_static_int_data(node, 2)
    axes = _extract_optional_int_data(node, 3)
    steps = _extract_optional_int_data(node, 4)

    num_slices = len(starts)
    if axes is None:
        axes = tuple(range(num_slices))
    if steps is None:
        steps = (1,) * num_slices

    # Determine the total number of dims from the input shape
    input_shape = node.inputs[0].tensor_type.shape
    ndim = len(input_shape) if input_shape is not None else max(axes) + 1

    # Build a full-dim slice tuple
    slices: list[slice] = [slice(None)] * ndim
    for a, s, e, st in zip(axes, starts, ends, steps, strict=False):
        slices[a] = slice(s, e, st)

    return [fx_graph.call_function(operator.getitem, args=(args[0], tuple(slices)))]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@register_op("Identity")
def _identity(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Pass through the input FX node unchanged for the ONNX Identity op.

    Args:
        node: The IR Identity node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Identity).

    Returns:
        A single-element list containing the same input FX node.
    """
    return [args[0]]


# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------


@register_op("Cast")
def _cast(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``Tensor.to(dtype)`` for the ONNX Cast op.

    The target dtype is read from the ``to`` attribute (ONNX dtype int),
    converted to an IR ``DType``, then mapped to a ``torch.dtype``.

    Args:
        node: The IR Cast node.
        args: Single-element list containing the input FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Cast).

    Returns:
        A single-element list containing the cast FX call_method node.
    """
    from protofx.ir.dtype import DType
    from protofx.utils.dtype import ir_dtype_to_torch

    to_attr = node.attributes.get("to")
    if to_attr is None:
        msg = "Cast: missing required 'to' attribute"
        raise NotImplementedError(msg)
    ir_dtype = DType(int(to_attr))
    torch_dtype = ir_dtype_to_torch(ir_dtype)
    if torch_dtype is None:
        msg = f"Cast: unsupported target dtype {ir_dtype.name}"
        raise NotImplementedError(msg)
    return [fx_graph.call_method("to", args=(args[0], torch_dtype))]


# ---------------------------------------------------------------------------
# Expand
# ---------------------------------------------------------------------------


@register_op("Expand")
def _expand(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.broadcast_to`` for the ONNX Expand op.

    The target shape is statically extracted from the second input Value.

    Args:
        node: The IR Expand node.
        args: Two-element list; first is the data FX node.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Expand).

    Returns:
        A single-element list containing the broadcast FX call_function node.
    """
    import torch

    target_shape = _extract_static_int_data(node, 1)
    return [fx_graph.call_function(torch.broadcast_to, args=(args[0], target_shape))]


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


@register_op("Gather")
def _gather(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.index_select`` for the ONNX Gather op.

    Indices are statically extracted from the second input. For scalar indices,
    the gathered dimension is squeezed to match ONNX semantics.

    Args:
        node: The IR Gather node.
        args: Two-element list; first is the data FX node, second is indices (unused).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Gather).

    Returns:
        A single-element list containing the gather FX call_function node.
    """
    import torch

    axis = int(node.attributes.get("axis", 0))
    indices_value = node.inputs[1]
    if indices_value.data is None:
        msg = f"Gather: indices input ({indices_value.name or indices_value.id}) has no static data"
        raise NotImplementedError(msg)

    indices_np = indices_value.data
    is_scalar = indices_np.ndim == 0

    # Flatten indices to 1D for torch.index_select
    flat_indices = indices_np.flatten().tolist()
    idx_tensor = torch.tensor(flat_indices, dtype=torch.long)
    # Register as buffer for FX graph
    attr_name = f"_gather_indices_{node.id}"
    module.register_buffer(attr_name, idx_tensor)
    idx_node = fx_graph.get_attr(attr_name)

    result = fx_graph.call_function(torch.index_select, args=(args[0], axis, idx_node))

    # Squeeze the axis dim for scalar indices to match ONNX Gather output shape
    if is_scalar:
        result = fx_graph.call_function(torch.squeeze, args=(result, axis))

    return [result]
