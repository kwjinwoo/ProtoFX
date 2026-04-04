"""Convolution ONNX op handlers (Conv, ConvTranspose)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


def _get_spatial_rank(node: Node) -> int:
    """Infer the spatial rank from the weight input shape.

    The weight tensor has shape ``(OC, IC/g, *kernel_shape)`` for Conv
    or ``(IC, OC/g, *kernel_shape)`` for ConvTranspose.

    Args:
        node: The IR Conv or ConvTranspose node.

    Returns:
        The number of spatial dimensions (1, 2, or 3).

    Raises:
        NotImplementedError: If the weight shape is unavailable or the
            spatial rank is unsupported.
    """
    w_value = node.inputs[1]
    if w_value.tensor_type.shape is not None:
        spatial_rank = len(w_value.tensor_type.shape) - 2
    else:
        kernel_shape = node.attributes.get("kernel_shape")
        if kernel_shape is not None:
            spatial_rank = len(kernel_shape)  # type: ignore[arg-type]
        else:
            msg = f"{node.op_type}: cannot determine spatial rank (no weight shape or kernel_shape)"
            raise NotImplementedError(msg)

    if spatial_rank not in (1, 2, 3):
        msg = f"{node.op_type}: unsupported spatial rank {spatial_rank}"
        raise NotImplementedError(msg)

    return spatial_rank


def _onnx_pads_to_torch(pads: list[int]) -> tuple[int, ...]:
    """Convert ONNX pads format to PyTorch padding format.

    ONNX format: ``[begin_d0, begin_d1, ..., end_d0, end_d1, ...]``
    PyTorch format: ``(pad_d0, pad_d1, ...)`` where each value applies to both sides.

    When pads are symmetric, returns a simplified tuple. When asymmetric,
    returns the full ONNX-style tuple (requires manual F.pad before conv).

    Args:
        pads: ONNX-format padding list.

    Returns:
        Tuple of per-dimension padding values for PyTorch.
    """
    n = len(pads) // 2
    begins = pads[:n]
    ends = pads[n:]
    return tuple(begins[i] for i in range(n) if begins[i] == ends[i]) if begins == ends else tuple(pads)


@register_op("Conv")
def _conv(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.conv{N}d`` for the ONNX Conv op.

    Supports 1D, 2D, and 3D convolution. The spatial rank is inferred
    from the weight tensor shape. Attributes ``strides``, ``pads``,
    ``dilations``, and ``group`` are mapped to PyTorch kwargs.

    Args:
        node: The IR Conv node.
        args: Three-element list ``[X, W, B]``. ``B`` is ``None``
            when the optional bias input is omitted (sentinel).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for Conv).

    Returns:
        A single-element list containing the conv FX call_function node.
    """
    import torch.nn.functional as F

    spatial_rank = _get_spatial_rank(node)
    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial_rank]

    x_node = args[0]
    w_node = args[1]
    b_node = args[2] if len(args) > 2 else None  # None if sentinel or omitted

    strides = node.attributes.get("strides", [1] * spatial_rank)
    pads_raw = node.attributes.get("pads", [0] * (2 * spatial_rank))
    dilations = node.attributes.get("dilations", [1] * spatial_rank)
    group = node.attributes.get("group", 1)

    padding = _onnx_pads_to_torch(pads_raw)  # type: ignore[arg-type]

    # If padding is asymmetric, we need F.pad first
    n = len(pads_raw) // 2  # type: ignore[arg-type]
    begins = pads_raw[:n]  # type: ignore[index]
    ends = pads_raw[n:]  # type: ignore[index]
    if begins != ends:
        import torch

        # F.pad expects reversed order: (pad_last_dim_begin, pad_last_dim_end, ..., pad_first_dim_begin, ...)
        pad_args: list[int] = []
        for i in range(n - 1, -1, -1):
            pad_args.extend([begins[i], ends[i]])
        x_node = fx_graph.call_function(torch.nn.functional.pad, args=(x_node, tuple(pad_args)))
        padding = tuple([0] * n)

    return [
        fx_graph.call_function(
            conv_fn,
            args=(x_node, w_node, b_node),
            kwargs={
                "stride": tuple(strides),  # type: ignore[arg-type]
                "padding": padding,
                "dilation": tuple(dilations),  # type: ignore[arg-type]
                "groups": int(group),
            },
        )
    ]


@register_op("ConvTranspose")
def _conv_transpose(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.conv_transpose{N}d`` for the ONNX ConvTranspose op.

    Supports 1D, 2D, and 3D transposed convolution. The spatial rank is
    inferred from the weight tensor shape. Attributes ``strides``, ``pads``,
    ``dilations``, ``output_padding``, and ``group`` are mapped to PyTorch kwargs.

    Args:
        node: The IR ConvTranspose node.
        args: Three-element list ``[X, W, B]``. ``B`` is ``None``
            when the optional bias input is omitted (sentinel).
        fx_graph: The FX graph being constructed.
        module: The root module (unused for ConvTranspose).

    Returns:
        A single-element list containing the conv_transpose FX call_function node.
    """
    import torch.nn.functional as F

    spatial_rank = _get_spatial_rank(node)
    conv_t_fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[spatial_rank]

    x_node = args[0]
    w_node = args[1]
    b_node = args[2] if len(args) > 2 else None  # None if sentinel or omitted

    strides = node.attributes.get("strides", [1] * spatial_rank)
    pads_raw = node.attributes.get("pads", [0] * (2 * spatial_rank))
    dilations = node.attributes.get("dilations", [1] * spatial_rank)
    output_pad = node.attributes.get("output_padding", [0] * spatial_rank)
    group = node.attributes.get("group", 1)

    # PyTorch conv_transpose padding = ONNX pads begin values (symmetric expected after normalization)
    padding = _onnx_pads_to_torch(pads_raw)  # type: ignore[arg-type]

    return [
        fx_graph.call_function(
            conv_t_fn,
            args=(x_node, w_node, b_node),
            kwargs={
                "stride": tuple(strides),  # type: ignore[arg-type]
                "padding": padding,
                "output_padding": tuple(output_pad),  # type: ignore[arg-type]
                "groups": int(group),
                "dilation": tuple(dilations),  # type: ignore[arg-type]
            },
        )
    ]
