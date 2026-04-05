"""Pooling ONNX op handlers (MaxPool, AveragePool, GlobalAveragePool)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from protofx.ops._registry import register_op
from protofx.ops.conv import _onnx_pads_to_torch

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


def _get_pool_spatial_rank(node: Node) -> int:
    """Infer the spatial rank from the ``kernel_shape`` attribute.

    Args:
        node: The IR MaxPool or AveragePool node.

    Returns:
        The number of spatial dimensions (1, 2, or 3).

    Raises:
        NotImplementedError: If ``kernel_shape`` is missing or the
            spatial rank is unsupported.
    """
    kernel_shape = node.attributes.get("kernel_shape")
    if kernel_shape is None:
        msg = f"{node.op_type}: kernel_shape attribute is required"
        raise NotImplementedError(msg)

    spatial_rank = len(kernel_shape)  # type: ignore[arg-type]
    if spatial_rank not in (1, 2, 3):
        msg = f"{node.op_type}: unsupported spatial rank {spatial_rank}"
        raise NotImplementedError(msg)

    return spatial_rank


def _check_auto_pad(node: Node) -> None:
    """Raise ``NotImplementedError`` when ``auto_pad`` is not ``NOTSET``.

    Args:
        node: The IR pooling node.

    Raises:
        NotImplementedError: If ``auto_pad`` is set to a value other than ``NOTSET``.
    """
    auto_pad_raw = node.attributes.get("auto_pad", "NOTSET")
    auto_pad = auto_pad_raw.decode() if isinstance(auto_pad_raw, bytes) else str(auto_pad_raw)
    if auto_pad != "NOTSET":
        msg = f"{node.op_type}: auto_pad='{auto_pad}' is not supported"
        raise NotImplementedError(msg)


@register_op("MaxPool")
def _max_pool(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.max_pool{N}d`` for the ONNX MaxPool op.

    Supports 1D, 2D, and 3D max pooling. The spatial rank is inferred
    from the ``kernel_shape`` attribute. Attributes ``strides``, ``pads``,
    ``dilations``, and ``ceil_mode`` are mapped to PyTorch kwargs.

    The optional Indices output and ``auto_pad`` modes other than ``NOTSET``
    raise ``NotImplementedError``.

    Args:
        node: The IR MaxPool node.
        args: Single-element list ``[X]``.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for MaxPool).

    Returns:
        A single-element list containing the max_pool FX call_function node.

    Raises:
        NotImplementedError: If ``auto_pad`` is not ``NOTSET`` or
            the Indices output is requested.
    """
    import torch.nn.functional as F

    _check_auto_pad(node)

    if len(node.outputs) > 1:
        msg = "MaxPool: Indices output is not supported"
        raise NotImplementedError(msg)

    spatial_rank = _get_pool_spatial_rank(node)
    pool_fn = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[spatial_rank]

    kernel_shape = node.attributes["kernel_shape"]
    strides = node.attributes.get("strides", [1] * spatial_rank)
    pads_raw = node.attributes.get("pads", [0] * (2 * spatial_rank))
    dilations = node.attributes.get("dilations", [1] * spatial_rank)
    ceil_mode = node.attributes.get("ceil_mode", 0)

    padding = _onnx_pads_to_torch(pads_raw)  # type: ignore[arg-type]

    # Handle asymmetric padding via F.pad
    n = len(pads_raw) // 2  # type: ignore[arg-type]
    begins = pads_raw[:n]  # type: ignore[index]
    ends = pads_raw[n:]  # type: ignore[index]
    x_node = args[0]
    if begins != ends:
        import torch

        pad_args: list[int] = []
        for i in range(n - 1, -1, -1):
            pad_args.extend([begins[i], ends[i]])
        x_node = fx_graph.call_function(torch.nn.functional.pad, args=(x_node, tuple(pad_args)))
        padding = tuple([0] * n)

    return [
        fx_graph.call_function(
            pool_fn,
            args=(x_node,),
            kwargs={
                "kernel_size": tuple(kernel_shape),  # type: ignore[arg-type]
                "stride": tuple(strides),  # type: ignore[arg-type]
                "padding": padding,
                "dilation": tuple(dilations),  # type: ignore[arg-type]
                "ceil_mode": bool(ceil_mode),
            },
        )
    ]


@register_op("AveragePool")
def _average_pool(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.avg_pool{N}d`` for the ONNX AveragePool op.

    Supports 1D, 2D, and 3D average pooling. The spatial rank is inferred
    from the ``kernel_shape`` attribute. Attributes ``strides``, ``pads``,
    ``count_include_pad``, and ``ceil_mode`` are mapped to PyTorch kwargs.

    ``auto_pad`` modes other than ``NOTSET`` raise ``NotImplementedError``.

    Args:
        node: The IR AveragePool node.
        args: Single-element list ``[X]``.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for AveragePool).

    Returns:
        A single-element list containing the avg_pool FX call_function node.

    Raises:
        NotImplementedError: If ``auto_pad`` is not ``NOTSET``.
    """
    import torch.nn.functional as F

    _check_auto_pad(node)

    spatial_rank = _get_pool_spatial_rank(node)
    pool_fn = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[spatial_rank]

    kernel_shape = node.attributes["kernel_shape"]
    strides = node.attributes.get("strides", [1] * spatial_rank)
    pads_raw = node.attributes.get("pads", [0] * (2 * spatial_rank))
    count_include_pad = node.attributes.get("count_include_pad", 0)
    ceil_mode = node.attributes.get("ceil_mode", 0)

    padding = _onnx_pads_to_torch(pads_raw)  # type: ignore[arg-type]

    # Handle asymmetric padding via F.pad
    n = len(pads_raw) // 2  # type: ignore[arg-type]
    begins = pads_raw[:n]  # type: ignore[index]
    ends = pads_raw[n:]  # type: ignore[index]
    x_node = args[0]
    if begins != ends:
        import torch

        pad_args: list[int] = []
        for i in range(n - 1, -1, -1):
            pad_args.extend([begins[i], ends[i]])
        x_node = fx_graph.call_function(torch.nn.functional.pad, args=(x_node, tuple(pad_args)))
        padding = tuple([0] * n)

    return [
        fx_graph.call_function(
            pool_fn,
            args=(x_node,),
            kwargs={
                "kernel_size": tuple(kernel_shape),  # type: ignore[arg-type]
                "stride": tuple(strides),  # type: ignore[arg-type]
                "padding": padding,
                "count_include_pad": bool(count_include_pad),
                "ceil_mode": bool(ceil_mode),
            },
        )
    ]


@register_op("GlobalAveragePool")
def _global_average_pool(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.nn.functional.adaptive_avg_pool{N}d`` for the ONNX GlobalAveragePool op.

    The spatial rank is inferred from the input tensor shape (total dims minus 2
    for batch and channel). Output spatial dimensions are all 1.

    Args:
        node: The IR GlobalAveragePool node.
        args: Single-element list ``[X]``.
        fx_graph: The FX graph being constructed.
        module: The root module (unused for GlobalAveragePool).

    Returns:
        A single-element list containing the adaptive_avg_pool FX call_function node.

    Raises:
        NotImplementedError: If the input shape is unavailable or
            the spatial rank is unsupported.
    """
    import torch.nn.functional as F

    x_value = node.inputs[0]
    if x_value.tensor_type.shape is None:
        msg = "GlobalAveragePool: cannot determine spatial rank (no input shape)"
        raise NotImplementedError(msg)

    spatial_rank = len(x_value.tensor_type.shape) - 2
    if spatial_rank not in (1, 2, 3):
        msg = f"GlobalAveragePool: unsupported spatial rank {spatial_rank}"
        raise NotImplementedError(msg)

    pool_fn = {1: F.adaptive_avg_pool1d, 2: F.adaptive_avg_pool2d, 3: F.adaptive_avg_pool3d}[spatial_rank]

    return [
        fx_graph.call_function(
            pool_fn,
            args=(args[0], 1),
        )
    ]
