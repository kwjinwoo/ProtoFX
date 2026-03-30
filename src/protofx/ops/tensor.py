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
