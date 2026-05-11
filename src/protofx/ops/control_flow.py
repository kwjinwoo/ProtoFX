"""Control-flow ONNX op handlers."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

from protofx.ir.graph import Graph
from protofx.ops._registry import register_op

if TYPE_CHECKING:
    import torch
    import torch.fx

    from protofx.ir.node import Node


def _make_loop_cond_fn(max_trip_count: torch.Tensor | None) -> Callable[..., torch.Tensor]:
    """Build a ``torch.while_loop`` condition callable.

    Args:
        max_trip_count: Optional max-trip-count tensor from Loop input ``M``.

    Returns:
        A callable that consumes ``(iteration, current_condition, *carried)`` and returns the next loop predicate.
    """
    import torch

    def _cond_fn(iteration: torch.Tensor, current_condition: torch.Tensor, *carried: torch.Tensor) -> torch.Tensor:
        del carried
        if max_trip_count is None:
            return current_condition.clone()
        return torch.logical_and(current_condition, torch.lt(iteration, max_trip_count))

    return _cond_fn


def _make_loop_body_fn(
    body_callable: Callable[..., tuple[torch.Tensor, ...]],
    captures: tuple[torch.Tensor, ...],
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Build a ``torch.while_loop`` body callable for one Loop node.

    Args:
        body_callable: Lowered child-graph callable for the Loop body.
        captures: Explicit capture operands to close over.

    Returns:
        A callable that consumes ``(iteration, current_condition, *carried)`` and returns the updated loop state.
    """

    def _body_fn(
        iteration: torch.Tensor, current_condition: torch.Tensor, *carried: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        body_outputs = body_callable(iteration, current_condition, *carried, *captures)
        next_iteration = iteration + 1
        next_condition = body_outputs[0]
        next_carried = body_outputs[1:]
        return (next_iteration, next_condition, *next_carried)

    return _body_fn


def _call_torch_while_loop(
    cond_fn: Callable[..., torch.Tensor],
    body_fn: Callable[..., tuple[torch.Tensor, ...]],
    loop_state: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Invoke ``torch.while_loop`` through a stable callable target.

    Args:
        cond_fn: Loop condition callable.
        body_fn: Loop body callable.
        loop_state: Initial loop state tuple.

    Returns:
        Final loop state tuple returned by ``torch.while_loop``.
    """
    import torch

    return torch.while_loop(cond_fn, body_fn, loop_state)


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


@register_op("Loop", opset_range=(11, 21))
def _loop(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.while_loop`` for the ONNX ``Loop`` op.

    Args:
        node: The IR Loop node.
        args: Input FX nodes in normalized order ``(M, cond, carried..., captures...)``.
        fx_graph: The FX graph being constructed.
        module: The root module carrying the internal child-graph helper.

    Returns:
        A list of FX nodes corresponding to final loop-carried outputs.
    """
    import torch

    helper = getattr(module, "_protofx_child_graph_emitter", None)
    if helper is None:
        msg = "Loop: internal child graph emitter helper is unavailable"
        raise ValueError(msg)

    body_graph = node.subgraphs.get("body")
    if not isinstance(body_graph, Graph):
        msg = "Loop: missing body child graph"
        raise ValueError(msg)

    carried_count = len(node.outputs)
    expected_min_inputs = 2 + carried_count
    if len(args) < expected_min_inputs:
        msg = f"Loop: expected at least {expected_min_inputs} inputs, got {len(args)}"
        raise ValueError(msg)

    max_trip_count = args[0]
    initial_condition = args[1]
    carried_inputs = args[2 : 2 + carried_count]
    for slot, carried_input in enumerate(carried_inputs):
        if carried_input is None:
            msg = f"Loop: missing loop-carried input at slot {slot}"
            raise ValueError(msg)
    captures = args[2 + carried_count :]
    if any(capture is None for capture in captures):
        msg = "Loop: explicit captures cannot be omitted"
        raise ValueError(msg)

    body_attr, body_arity = helper.make_callable_attr(owner_node=node, branch_name="body", child_graph=body_graph)
    expected_body_arity = 1 + carried_count
    if body_arity != expected_body_arity:
        msg = f"Loop: body output arity mismatch before emission (body={body_arity}, expected={expected_body_arity})"
        raise ValueError(msg)

    body_callable = fx_graph.get_attr(body_attr)
    cond_fn = fx_graph.call_function(_make_loop_cond_fn, args=(max_trip_count,))
    body_fn = fx_graph.call_function(_make_loop_body_fn, args=(body_callable, tuple(captures)))
    iteration0 = fx_graph.call_function(torch.tensor, args=(0,), kwargs={"dtype": torch.int64})
    cond0 = (
        initial_condition
        if initial_condition is not None
        else fx_graph.call_function(torch.tensor, args=(True,), kwargs={"dtype": torch.bool})
    )
    loop_state = (iteration0, cond0, *carried_inputs)
    final_state = fx_graph.call_function(_call_torch_while_loop, args=(cond_fn, body_fn, loop_state))
    return [fx_graph.call_function(operator.getitem, args=(final_state, slot + 2)) for slot in range(carried_count)]
