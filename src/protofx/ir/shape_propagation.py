"""IR-level symbolic shape propagation."""

from __future__ import annotations

import math

from protofx.ir.derived_shape import get_authoritative_tensor_type, set_derived_tensor_type
from protofx.ir.graph import Graph
from protofx.ir.node import Node
from protofx.ir.shape import Shape
from protofx.ir.tensor_type import TensorType
from protofx.ir.value import Value

_PASSTHROUGH_OPS = {
    "Abs",
    "Cast",
    "Clip",
    "Erf",
    "Exp",
    "Gelu",
    "Identity",
    "Neg",
    "Relu",
    "Sigmoid",
    "Tanh",
}

_ELEMENTWISE_BROADCAST_OPS = {
    "Add",
    "Div",
    "Max",
    "Min",
    "Mul",
    "Sub",
}

_SIMPLE_SHAPE_TRANSFORM_OPS = {
    "Concat",
    "Flatten",
    "Reshape",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
}

_REDUCTION_OPS = {
    "ArgMax",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
}

_LINALG_OPS = {
    "Gemm",
    "MatMul",
}

_SPATIAL_OPS = {
    "AveragePool",
    "Conv",
    "ConvTranspose",
    "GlobalAveragePool",
    "MaxPool",
}


def _iter_graph_values(graph: Graph) -> list[Value]:
    """Return all values reachable from one graph.

    Args:
        graph: IR graph to scan.

    Returns:
        Deduplicated value list.
    """
    ordered: list[Value] = []
    seen: set[int] = set()

    for value in graph.inputs + graph.initializers + graph.outputs:
        if id(value) not in seen:
            seen.add(id(value))
            ordered.append(value)
    for node in graph.nodes:
        for value in node.inputs + node.outputs:
            if id(value) not in seen:
                seen.add(id(value))
                ordered.append(value)
    return ordered


def _seed_authoritative_tensor_types(graph: Graph) -> None:
    """Seed authoritative metadata from imported tensor metadata.

    Args:
        graph: IR graph to initialize.
    """
    for value in _iter_graph_values(graph):
        set_derived_tensor_type(value, value.tensor_type)
    for node in graph.nodes:
        for subgraph in node.subgraphs.values():
            if isinstance(subgraph, Graph):
                _seed_authoritative_tensor_types(subgraph)
            else:
                for child in subgraph:
                    _seed_authoritative_tensor_types(child)


def _broadcast_shapes(lhs: Shape, rhs: Shape) -> Shape:
    """Derive broadcast output shape for two operand shapes.

    Args:
        lhs: Left operand shape.
        rhs: Right operand shape.

    Returns:
        Broadcasted shape when derivable, otherwise ``None``.
    """
    if lhs is None or rhs is None:
        return None

    result_reversed: list[int | str | None] = []
    max_rank = max(len(lhs), len(rhs))
    for idx in range(1, max_rank + 1):
        left_dim = lhs[-idx] if idx <= len(lhs) else 1
        right_dim = rhs[-idx] if idx <= len(rhs) else 1
        if left_dim == 1:
            result_reversed.append(right_dim)
            continue
        if right_dim == 1:
            result_reversed.append(left_dim)
            continue
        if left_dim is None or right_dim is None:
            result_reversed.append(None)
            continue
        if isinstance(left_dim, str) or isinstance(right_dim, str):
            result_reversed.append(None)
            continue
        if left_dim != right_dim:
            return None
        result_reversed.append(left_dim)
    return tuple(reversed(result_reversed))


def _merge_if_shapes(then_shape: Shape, else_shape: Shape) -> Shape:
    """Merge two If branch output shapes.

    Args:
        then_shape: Shape from then branch output.
        else_shape: Shape from else branch output.

    Returns:
        Merged shape metadata.
    """
    if then_shape is None or else_shape is None:
        return None
    if len(then_shape) != len(else_shape):
        return None

    merged: list[int | str | None] = []
    for then_dim, else_dim in zip(then_shape, else_shape, strict=True):
        if then_dim == else_dim:
            merged.append(then_dim)
            continue
        if then_dim is None or else_dim is None:
            merged.append(None)
            continue
        if isinstance(then_dim, str) or isinstance(else_dim, str):
            merged.append(None)
            continue
        return None
    return tuple(merged)


def _set_output_shape(value: Value, shape: Shape) -> None:
    """Set one output's authoritative shape while preserving dtype.

    Args:
        value: Output value to mutate.
        shape: Derived shape.
    """
    dtype = get_authoritative_tensor_type(value).dtype
    set_derived_tensor_type(value, TensorType(dtype=dtype, shape=shape))


def _bind_if_captures(node: Node, branch: Graph) -> None:
    """Bind If capture-derived metadata onto child-graph inputs.

    Args:
        node: Parent If node.
        branch: Child branch graph.
    """
    for slot, capture in enumerate(node.inputs[1:]):
        if slot >= len(branch.inputs):
            return
        set_derived_tensor_type(branch.inputs[slot], get_authoritative_tensor_type(capture))


def _normalize_axis(axis: int, rank: int) -> int | None:
    """Normalize one axis into ``[0, rank)``.

    Args:
        axis: Axis to normalize.
        rank: Tensor rank.

    Returns:
        Normalized axis, or ``None`` for invalid input.
    """
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        return None
    return normalized


def _normalize_unsqueeze_axis(axis: int, rank: int) -> int | None:
    """Normalize one Unsqueeze axis into ``[0, rank]``.

    Args:
        axis: Axis to normalize.
        rank: Output tensor rank.

    Returns:
        Normalized axis, or ``None`` for invalid input.
    """
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized > rank:
        return None
    return normalized


def _product_dims(dims: tuple[int | str | None, ...]) -> int | None:
    """Return the integer product of dims when fully static.

    Args:
        dims: Dims to multiply.

    Returns:
        Integer product for static dims, otherwise ``None``.
    """
    product = 1
    for dim in dims:
        if not isinstance(dim, int):
            return None
        product *= dim
    return product


def _get_static_ints_from_input(node: Node, input_index: int) -> list[int] | None:
    """Return integer data from one node input when statically available.

    Args:
        node: Node whose input tensor is inspected.
        input_index: Input index to read.

    Returns:
        Integer list, or ``None`` when dynamic/unavailable.
    """
    if input_index >= len(node.inputs):
        return None
    data = node.inputs[input_index].data
    if data is None:
        return None
    return [int(v) for v in data.flat]


def _get_reduction_axes(node: Node, rank: int) -> list[int] | None:
    """Resolve reduction axes from attribute or optional axes input.

    Args:
        node: Reduction node.
        rank: Input rank.

    Returns:
        Normalized sorted axis list, or ``None`` when invalid/unavailable.
    """
    raw_axes: list[int] | None
    attr_axes = node.attributes.get("axes")
    if isinstance(attr_axes, int):
        raw_axes = [int(attr_axes)]
    elif isinstance(attr_axes, list):
        raw_axes = [int(v) for v in attr_axes]
    else:
        raw_axes = _get_static_ints_from_input(node, 1)

    if raw_axes is None:
        return None

    normalized: set[int] = set()
    for axis in raw_axes:
        norm_axis = _normalize_axis(axis, rank)
        if norm_axis is None:
            return None
        normalized.add(norm_axis)
    return sorted(normalized)


def _derive_flatten_shape(node: Node) -> Shape:
    """Derive Flatten output shape.

    Args:
        node: Flatten node.

    Returns:
        Derived shape metadata.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    if input_shape is None:
        return None

    axis = int(node.attributes.get("axis", 1))
    norm_axis = _normalize_axis(axis, len(input_shape))
    if norm_axis is None:
        if axis == len(input_shape):
            norm_axis = len(input_shape)
        else:
            return None

    left = _product_dims(input_shape[:norm_axis])
    right = _product_dims(input_shape[norm_axis:])
    return (left, right)


def _derive_transpose_shape(node: Node) -> Shape:
    """Derive Transpose output shape.

    Args:
        node: Transpose node.

    Returns:
        Derived shape metadata.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    if input_shape is None:
        return None

    perm_attr = node.attributes.get("perm")
    if perm_attr is None:
        perm = list(reversed(range(len(input_shape))))
    elif isinstance(perm_attr, list):
        perm = [int(v) for v in perm_attr]
    else:
        return None

    if len(perm) != len(input_shape):
        return None
    return tuple(input_shape[idx] for idx in perm)


def _derive_reshape_shape(node: Node) -> Shape:
    """Derive Reshape output shape from static target-shape input.

    Args:
        node: Reshape node.

    Returns:
        Derived shape metadata.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    target_dims = _get_static_ints_from_input(node, 1)
    if target_dims is None:
        return None

    allowzero = int(node.attributes.get("allowzero", 0))
    output_rank = len(target_dims)
    known_product = _product_dims(input_shape) if input_shape is not None else None
    neg_one_index: int | None = None
    resolved: list[int | str | None] = []

    for idx, dim in enumerate(target_dims):
        if dim == -1:
            if neg_one_index is not None:
                return None
            neg_one_index = idx
            resolved.append(None)
            continue
        if dim == 0 and allowzero == 0:
            if input_shape is None or idx >= len(input_shape):
                return None
            resolved.append(input_shape[idx])
            continue
        resolved.append(int(dim))

    if neg_one_index is None:
        return tuple(resolved)

    known_target = _product_dims(tuple(d for d in resolved if isinstance(d, int)))
    has_unknown = any((d is None or isinstance(d, str)) for d in resolved if d != resolved[neg_one_index])
    if known_product is None or known_target is None or known_target == 0 or has_unknown:
        resolved[neg_one_index] = None
        return tuple(resolved[:output_rank])
    if known_product % known_target != 0:
        resolved[neg_one_index] = None
        return tuple(resolved[:output_rank])

    resolved[neg_one_index] = known_product // known_target
    return tuple(resolved[:output_rank])


def _derive_squeeze_shape(node: Node) -> Shape:
    """Derive Squeeze output shape.

    Args:
        node: Squeeze node.

    Returns:
        Derived shape metadata.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    if input_shape is None:
        return None

    axes = _get_static_ints_from_input(node, 1)
    if axes is None:
        attr_axes = node.attributes.get("axes")
        if isinstance(attr_axes, int):
            axes = [int(attr_axes)]
        elif isinstance(attr_axes, list):
            axes = [int(v) for v in attr_axes]

    if axes is None:
        return tuple(dim for dim in input_shape if dim != 1)

    normalized_axes: set[int] = set()
    for axis in axes:
        norm_axis = _normalize_axis(axis, len(input_shape))
        if norm_axis is None:
            return None
        normalized_axes.add(norm_axis)

    result: list[int | str | None] = []
    for idx, dim in enumerate(input_shape):
        if idx not in normalized_axes:
            result.append(dim)
            continue
        if dim == 1:
            continue
        return None
    return tuple(result)


def _derive_unsqueeze_shape(node: Node) -> Shape:
    """Derive Unsqueeze output shape.

    Args:
        node: Unsqueeze node.

    Returns:
        Derived shape metadata.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    if input_shape is None:
        return None

    axes = _get_static_ints_from_input(node, 1)
    if axes is None:
        attr_axes = node.attributes.get("axes")
        if isinstance(attr_axes, int):
            axes = [int(attr_axes)]
        elif isinstance(attr_axes, list):
            axes = [int(v) for v in attr_axes]
    if axes is None:
        return None

    output_rank = len(input_shape) + len(axes)
    normalized_axes: list[int] = []
    seen: set[int] = set()
    for axis in axes:
        norm_axis = _normalize_unsqueeze_axis(axis, output_rank)
        if norm_axis is None or norm_axis in seen:
            return None
        seen.add(norm_axis)
        normalized_axes.append(norm_axis)
    normalized_axes.sort()

    result: list[int | str | None] = list(input_shape)
    for axis in normalized_axes:
        result.insert(axis, 1)
    return tuple(result)


def _derive_concat_shape(node: Node) -> Shape:
    """Derive Concat output shape.

    Args:
        node: Concat node.

    Returns:
        Derived shape metadata.
    """
    if not node.inputs:
        return None
    input_shapes = [get_authoritative_tensor_type(value).shape for value in node.inputs]
    if any(shape is None for shape in input_shapes):
        return None
    first_shape = input_shapes[0]
    assert first_shape is not None

    axis = int(node.attributes.get("axis", 0))
    norm_axis = _normalize_axis(axis, len(first_shape))
    if norm_axis is None:
        return None

    output_shape: list[int | str | None] = list(first_shape)
    concat_total: int | None = 0
    for shape in input_shapes:
        assert shape is not None
        if len(shape) != len(first_shape):
            return None
        for idx, dim in enumerate(shape):
            if idx == norm_axis:
                if concat_total is None or not isinstance(dim, int):
                    concat_total = None
                else:
                    concat_total += dim
                continue
            if output_shape[idx] == dim:
                continue
            if output_shape[idx] is None or dim is None:
                output_shape[idx] = None
                continue
            if isinstance(output_shape[idx], str) or isinstance(dim, str):
                output_shape[idx] = None
                continue
            return None
    output_shape[norm_axis] = concat_total
    return tuple(output_shape)


def _derive_simple_transform_shape(node: Node) -> Shape:
    """Derive shape for simple shape-transform families.

    Args:
        node: Node to process.

    Returns:
        Derived output shape.
    """
    match node.op_type:
        case "Flatten":
            return _derive_flatten_shape(node)
        case "Reshape":
            return _derive_reshape_shape(node)
        case "Transpose":
            return _derive_transpose_shape(node)
        case "Squeeze":
            return _derive_squeeze_shape(node)
        case "Unsqueeze":
            return _derive_unsqueeze_shape(node)
        case "Concat":
            return _derive_concat_shape(node)
        case _:
            return None


def _derive_reduction_shape(node: Node) -> Shape:
    """Derive shape for reduction-family operators.

    Args:
        node: Reduction node.

    Returns:
        Derived output shape.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    if input_shape is None:
        return None

    keepdims = bool(node.attributes.get("keepdims", 1))
    noop_with_empty_axes = bool(node.attributes.get("noop_with_empty_axes", 0))
    axes = _get_reduction_axes(node, len(input_shape))
    if axes is None and node.op_type == "ArgMax":
        axis = int(node.attributes.get("axis", 0))
        norm_axis = _normalize_axis(axis, len(input_shape))
        if norm_axis is None:
            return None
        axes = [norm_axis]
    if axes is None:
        if noop_with_empty_axes:
            return input_shape
        axes = list(range(len(input_shape)))
    if not axes and noop_with_empty_axes:
        return input_shape

    if keepdims:
        reduced = list(input_shape)
        for axis in axes:
            reduced[axis] = 1
        return tuple(reduced)
    return tuple(dim for idx, dim in enumerate(input_shape) if idx not in set(axes))


def _dims_provably_mismatch(lhs: int | str | None, rhs: int | str | None) -> bool:
    """Return whether two dims are provably incompatible.

    Args:
        lhs: Left dimension metadata.
        rhs: Right dimension metadata.

    Returns:
        ``True`` when mismatch is provable.
    """
    return isinstance(lhs, int) and isinstance(rhs, int) and lhs != rhs


def _derive_matmul_shape(node: Node) -> Shape:
    """Derive MatMul output shape.

    Args:
        node: MatMul node.

    Returns:
        Derived output shape.
    """
    lhs = get_authoritative_tensor_type(node.inputs[0]).shape
    rhs = get_authoritative_tensor_type(node.inputs[1]).shape
    if lhs is None or rhs is None or not lhs or not rhs:
        return None

    lhs_promoted = (1, lhs[0]) if len(lhs) == 1 else lhs
    rhs_promoted = (rhs[0], 1) if len(rhs) == 1 else rhs
    if len(lhs_promoted) < 2 or len(rhs_promoted) < 2:
        return None

    batch = _broadcast_shapes(lhs_promoted[:-2], rhs_promoted[:-2])
    if batch is None:
        return None

    lhs_k = lhs_promoted[-1]
    rhs_k = rhs_promoted[-2]
    if _dims_provably_mismatch(lhs_k, rhs_k):
        return None

    output: tuple[int | str | None, ...] = (*batch, lhs_promoted[-2], rhs_promoted[-1])
    if len(lhs) == 1 and len(rhs) == 1:
        return ()
    if len(lhs) == 1:
        return output[:-2] + (output[-1],)
    if len(rhs) == 1:
        return output[:-1]
    return output


def _derive_gemm_shape(node: Node) -> Shape:
    """Derive Gemm output shape.

    Args:
        node: Gemm node.

    Returns:
        Derived output shape.
    """
    lhs = get_authoritative_tensor_type(node.inputs[0]).shape
    rhs = get_authoritative_tensor_type(node.inputs[1]).shape
    if lhs is None or rhs is None or len(lhs) != 2 or len(rhs) != 2:
        return None

    trans_a = bool(node.attributes.get("transA", 0))
    trans_b = bool(node.attributes.get("transB", 0))
    lhs_shape = (lhs[1], lhs[0]) if trans_a else lhs
    rhs_shape = (rhs[1], rhs[0]) if trans_b else rhs

    if _dims_provably_mismatch(lhs_shape[1], rhs_shape[0]):
        return None
    return (lhs_shape[0], rhs_shape[1])


def _derive_linalg_shape(node: Node) -> Shape:
    """Derive shape for linalg-family operators.

    Args:
        node: Linalg node.

    Returns:
        Derived output shape.
    """
    match node.op_type:
        case "MatMul":
            return _derive_matmul_shape(node)
        case "Gemm":
            return _derive_gemm_shape(node)
        case _:
            return None


def _conv_dim(
    input_dim: int | str | None,
    kernel_dim: int | str | None,
    stride: int,
    dilation: int,
    pad_begin: int,
    pad_end: int,
    *,
    ceil_mode: bool = False,
) -> int | None:
    """Compute one conv/pool output dimension when statically derivable.

    Args:
        input_dim: Input spatial dim metadata.
        kernel_dim: Kernel spatial dim metadata.
        stride: Stride value.
        dilation: Dilation value.
        pad_begin: Padding before.
        pad_end: Padding after.
        ceil_mode: Whether to use ceil-mode output sizing.

    Returns:
        Derived dim, or ``None`` when not statically derivable.
    """
    if not isinstance(input_dim, int) or not isinstance(kernel_dim, int):
        return None
    effective_kernel = dilation * (kernel_dim - 1) + 1
    numerator = input_dim + pad_begin + pad_end - effective_kernel
    value = numerator / stride + 1
    return int(math.ceil(value) if ceil_mode else math.floor(value))


def _derive_conv_shape(node: Node) -> Shape:
    """Derive Conv output shape.

    Args:
        node: Conv node.

    Returns:
        Derived output shape.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    weight_shape = get_authoritative_tensor_type(node.inputs[1]).shape
    if input_shape is None or weight_shape is None or len(input_shape) < 3 or len(weight_shape) < 3:
        return None

    spatial_rank = len(input_shape) - 2
    if len(weight_shape) != spatial_rank + 2:
        return None

    pads = node.attributes.get("pads", [0] * (2 * spatial_rank))
    strides = node.attributes.get("strides", [1] * spatial_rank)
    dilations = node.attributes.get("dilations", [1] * spatial_rank)
    if not isinstance(pads, list) or not isinstance(strides, list) or not isinstance(dilations, list):
        return None
    if len(pads) != 2 * spatial_rank:
        return None

    spatial: list[int | None] = []
    for idx in range(spatial_rank):
        spatial.append(
            _conv_dim(
                input_dim=input_shape[2 + idx],
                kernel_dim=weight_shape[2 + idx],
                stride=int(strides[idx]),
                dilation=int(dilations[idx]),
                pad_begin=int(pads[idx]),
                pad_end=int(pads[idx + spatial_rank]),
            )
        )
    return (input_shape[0], weight_shape[0], *spatial)


def _derive_conv_transpose_shape(node: Node) -> Shape:
    """Derive ConvTranspose output shape.

    Args:
        node: ConvTranspose node.

    Returns:
        Derived output shape.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    weight_shape = get_authoritative_tensor_type(node.inputs[1]).shape
    if input_shape is None or weight_shape is None or len(input_shape) < 3 or len(weight_shape) < 3:
        return None

    spatial_rank = len(input_shape) - 2
    if len(weight_shape) != spatial_rank + 2:
        return None

    pads = node.attributes.get("pads", [0] * (2 * spatial_rank))
    strides = node.attributes.get("strides", [1] * spatial_rank)
    dilations = node.attributes.get("dilations", [1] * spatial_rank)
    output_padding = node.attributes.get("output_padding", [0] * spatial_rank)
    group = int(node.attributes.get("group", 1))
    if not isinstance(pads, list) or not isinstance(strides, list) or not isinstance(dilations, list):
        return None
    if not isinstance(output_padding, list) or len(pads) != 2 * spatial_rank:
        return None

    output_channels = None
    if isinstance(weight_shape[1], int):
        output_channels = weight_shape[1] * group

    spatial: list[int | None] = []
    for idx in range(spatial_rank):
        in_dim = input_shape[2 + idx]
        kernel = weight_shape[2 + idx]
        if not isinstance(in_dim, int) or not isinstance(kernel, int):
            spatial.append(None)
            continue
        derived = (
            (in_dim - 1) * int(strides[idx])
            - int(pads[idx])
            - int(pads[idx + spatial_rank])
            + int(dilations[idx]) * (kernel - 1)
            + int(output_padding[idx])
            + 1
        )
        spatial.append(derived)
    return (input_shape[0], output_channels, *spatial)


def _derive_pool_shape(node: Node) -> Shape:
    """Derive MaxPool/AveragePool output shape.

    Args:
        node: Pooling node.

    Returns:
        Derived output shape.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    if input_shape is None or len(input_shape) < 3:
        return None

    spatial_rank = len(input_shape) - 2
    kernel_shape = node.attributes.get("kernel_shape")
    if not isinstance(kernel_shape, list) or len(kernel_shape) != spatial_rank:
        return None

    pads = node.attributes.get("pads", [0] * (2 * spatial_rank))
    strides = node.attributes.get("strides", [1] * spatial_rank)
    dilations = node.attributes.get("dilations", [1] * spatial_rank)
    ceil_mode = bool(node.attributes.get("ceil_mode", 0))
    if not isinstance(pads, list) or not isinstance(strides, list) or not isinstance(dilations, list):
        return None
    if len(pads) != 2 * spatial_rank:
        return None

    spatial: list[int | None] = []
    for idx in range(spatial_rank):
        spatial.append(
            _conv_dim(
                input_dim=input_shape[2 + idx],
                kernel_dim=int(kernel_shape[idx]),
                stride=int(strides[idx]),
                dilation=int(dilations[idx]),
                pad_begin=int(pads[idx]),
                pad_end=int(pads[idx + spatial_rank]),
                ceil_mode=ceil_mode,
            )
        )
    return (input_shape[0], input_shape[1], *spatial)


def _derive_global_average_pool_shape(node: Node) -> Shape:
    """Derive GlobalAveragePool output shape.

    Args:
        node: GlobalAveragePool node.

    Returns:
        Derived output shape.
    """
    input_shape = get_authoritative_tensor_type(node.inputs[0]).shape
    if input_shape is None or len(input_shape) < 3:
        return None
    return input_shape[:2] + tuple(1 for _ in input_shape[2:])


def _derive_spatial_shape(node: Node) -> Shape:
    """Derive shape for spatial-family operators.

    Args:
        node: Spatial node.

    Returns:
        Derived output shape.
    """
    match node.op_type:
        case "Conv":
            return _derive_conv_shape(node)
        case "ConvTranspose":
            return _derive_conv_transpose_shape(node)
        case "MaxPool" | "AveragePool":
            return _derive_pool_shape(node)
        case "GlobalAveragePool":
            return _derive_global_average_pool_shape(node)
        case _:
            return None


def _propagate_graph(graph: Graph) -> None:
    """Run one propagation pass for a graph and child graphs.

    Args:
        graph: Graph to process.
    """
    for node in graph.topological_sort():
        match node.op_type:
            case op if op in _PASSTHROUGH_OPS:
                for output in node.outputs:
                    _set_output_shape(output, get_authoritative_tensor_type(node.inputs[0]).shape)
            case op if op in _ELEMENTWISE_BROADCAST_OPS:
                if len(node.inputs) < 2:
                    continue
                shape = _broadcast_shapes(
                    get_authoritative_tensor_type(node.inputs[0]).shape,
                    get_authoritative_tensor_type(node.inputs[1]).shape,
                )
                for output in node.outputs:
                    _set_output_shape(output, shape)
            case op if op in _SIMPLE_SHAPE_TRANSFORM_OPS:
                shape = _derive_simple_transform_shape(node)
                for output in node.outputs:
                    _set_output_shape(output, shape)
            case op if op in _REDUCTION_OPS:
                shape = _derive_reduction_shape(node)
                for output in node.outputs:
                    _set_output_shape(output, shape)
            case op if op in _LINALG_OPS:
                shape = _derive_linalg_shape(node)
                for output in node.outputs:
                    _set_output_shape(output, shape)
            case op if op in _SPATIAL_OPS:
                shape = _derive_spatial_shape(node)
                for output in node.outputs:
                    _set_output_shape(output, shape)
            case "If":
                then_branch = node.subgraphs.get("then_branch")
                else_branch = node.subgraphs.get("else_branch")
                if not isinstance(then_branch, Graph) or not isinstance(else_branch, Graph):
                    continue
                _bind_if_captures(node, then_branch)
                _bind_if_captures(node, else_branch)
                _propagate_graph(then_branch)
                _propagate_graph(else_branch)
                for then_out, else_out, node_out in zip(
                    then_branch.outputs, else_branch.outputs, node.outputs, strict=False
                ):
                    merged_shape = _merge_if_shapes(
                        get_authoritative_tensor_type(then_out).shape,
                        get_authoritative_tensor_type(else_out).shape,
                    )
                    _set_output_shape(node_out, merged_shape)
            case _:
                continue


def propagate_shapes(graph: Graph) -> None:
    """Propagate authoritative shape metadata across one graph tree.

    Args:
        graph: Graph to process in-place.
    """
    _seed_authoritative_tensor_types(graph)
    _propagate_graph(graph)
