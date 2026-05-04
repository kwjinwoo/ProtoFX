"""Control-flow ONNX op handlers."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from protofx.ir.graph import Graph
from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


@register_op("If", opset_range=(11, 21))
def _if(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.cond`` for the ONNX ``If`` op.

    Args:
        node: The IR If node.
        args: Input FX nodes where ``args[0]`` is the predicate and the remaining
            entries are explicit capture operands.
        fx_graph: The FX graph being constructed.
        module: The root module carrying the internal child-graph helper.

    Returns:
        A list of FX nodes, one per If output.
    """
    import torch

    predicate = args[0] if args else None
    if predicate is None:
        msg = "If: missing predicate input"
        raise ValueError(msg)

    helper = getattr(module, "_protofx_child_graph_emitter", None)
    if helper is None:
        msg = "If: internal child graph emitter helper is unavailable"
        raise ValueError(msg)

    then_branch = node.subgraphs.get("then_branch")
    else_branch = node.subgraphs.get("else_branch")
    if not isinstance(then_branch, Graph) or not isinstance(else_branch, Graph):
        msg = "If: missing then_branch/else_branch child graphs"
        raise ValueError(msg)

    then_attr, then_arity = helper.make_callable_attr(
        owner_node=node, branch_name="then_branch", child_graph=then_branch
    )
    else_attr, else_arity = helper.make_callable_attr(
        owner_node=node, branch_name="else_branch", child_graph=else_branch
    )

    expected_arity = len(node.outputs)
    if then_arity != expected_arity or else_arity != expected_arity:
        msg = (
            f"If: branch output arity mismatch before emission "
            f"(then={then_arity}, else={else_arity}, node={expected_arity})"
        )
        raise ValueError(msg)

    captures = tuple(arg for arg in args[1:] if arg is not None)
    then_fn = fx_graph.get_attr(then_attr)
    else_fn = fx_graph.get_attr(else_attr)
    packed = fx_graph.call_function(torch.cond, args=(predicate, then_fn, else_fn, captures))
    if expected_arity == 1:
        return [fx_graph.call_function(operator.getitem, args=(packed, 0))]
    return [fx_graph.call_function(operator.getitem, args=(packed, idx)) for idx in range(expected_arity)]
