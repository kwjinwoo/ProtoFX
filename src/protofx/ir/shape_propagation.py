"""IR-level symbolic shape propagation."""

from __future__ import annotations

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
                    then_branch.outputs, else_branch.outputs, node.outputs, strict=True
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
