"""Control-flow ONNX op handlers."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

from protofx.ir.graph import Graph
from protofx.ops._registry import register_op
from protofx.utils.dtype import ir_dtype_to_torch

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


def _scan_trip_count(scan_input: torch.Tensor) -> int:
    """Return the Scan iteration count from one normalized scan input.

    Args:
        scan_input: One normalized Scan sequence input.

    Returns:
        The leading-axis iteration count for Scan execution.
    """
    return int(scan_input.shape[0])


def _make_scan_cond_fn(trip_count: int) -> Callable[..., torch.Tensor]:
    """Build a ``torch.while_loop`` condition callable for one Scan node.

    Args:
        trip_count: Total Scan iteration count.

    Returns:
        A callable that consumes ``(iteration, *state_and_accum)`` and returns
        whether another iteration should run.
    """
    import torch

    def _cond_fn(iteration: torch.Tensor, *state_and_accum: torch.Tensor) -> torch.Tensor:
        del state_and_accum
        return torch.lt(iteration, trip_count)

    return _cond_fn


def _make_scan_initial_accumulator(
    trip_count: int,
    per_step_shape: tuple[int, ...],
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """Create an empty accumulator for one Scan output family.

    Args:
        trip_count: Total Scan iteration count.
        per_step_shape: Per-step shape for one Scan output family.
        dtype: Optional ``torch.dtype`` for the accumulator tensor.

    Returns:
        A zero-initialized accumulator with shape ``(trip_count, *per_step_shape)``.
    """
    import torch

    return torch.zeros((trip_count, *per_step_shape), dtype=dtype)


def _scan_output_accumulator_spec(scan_output: Node | object) -> tuple[tuple[int, ...], torch.dtype | None]:
    """Infer Scan accumulator metadata from one normalized parent scan output.

    Args:
        scan_output: Normalized parent Scan output value.

    Returns:
        A tuple of ``(per_step_shape, dtype)`` for accumulator initialization.

    Raises:
        ValueError: If scan-output metadata is missing or non-concrete for
            accumulator initialization.
    """
    output_shape = scan_output.tensor_type.shape
    if output_shape is None or len(output_shape) < 1:
        msg = "Scan: scanned output shape metadata is required before emission"
        raise ValueError(msg)
    per_step_shape: list[int] = []
    for dim in output_shape[1:]:
        if not isinstance(dim, int):
            msg = "Scan: scanned output per-step shape must be concrete before emission"
            raise ValueError(msg)
        per_step_shape.append(dim)
    output_dtype = scan_output.tensor_type.dtype
    torch_dtype = ir_dtype_to_torch(output_dtype)
    if output_dtype is not None and torch_dtype is None:
        msg = "Scan: scanned output dtype is unsupported for torch accumulation"
        raise ValueError(msg)
    return tuple(per_step_shape), torch_dtype


def _make_scan_body_fn(
    body_callable: Callable[..., tuple[torch.Tensor, ...]],
    scan_inputs: tuple[torch.Tensor, ...],
    captures: tuple[torch.Tensor, ...],
    state_count: int,
    scan_output_count: int,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Build a ``torch.while_loop`` body callable for one Scan node.

    Args:
        body_callable: Lowered child-graph callable for the Scan body.
        scan_inputs: Normalized Scan sequence inputs.
        captures: Explicit capture operands to close over.
        state_count: Number of Scan state families.
        scan_output_count: Number of Scan output families.

    Returns:
        A callable that consumes ``(iteration, *state_and_accum)`` and returns
        the updated loop state tuple.
    """
    import torch

    def _body_fn(iteration: torch.Tensor, *state_and_accum: torch.Tensor) -> tuple[torch.Tensor, ...]:
        current_states = state_and_accum[:state_count]
        current_accums = state_and_accum[state_count:]
        scan_slices = tuple(
            torch.squeeze(torch.index_select(scan_input, 0, torch.unsqueeze(iteration, 0)), dim=0)
            for scan_input in scan_inputs
        )
        body_outputs = body_callable(*current_states, *scan_slices, *captures)
        next_iteration = iteration + 1
        next_states = body_outputs[:state_count]
        scan_steps = body_outputs[state_count : state_count + scan_output_count]
        next_accums = []
        for scan_slot in range(scan_output_count):
            next_accums.append(
                torch.index_copy(
                    current_accums[scan_slot],
                    0,
                    torch.unsqueeze(iteration, 0),
                    torch.unsqueeze(scan_steps[scan_slot], 0),
                )
            )
        return (next_iteration, *next_states, *next_accums)

    return _body_fn


def _scan_sequence_matches_slice(
    sequence_value: Node | object,
    slice_value: Node | object,
) -> bool:
    """Return whether Scan sequence metadata matches per-step slice metadata.

    Args:
        sequence_value: Sequence value metadata from a parent Scan interface.
        slice_value: Per-step slice metadata from a Scan body interface.

    Returns:
        ``True`` when known metadata is compatible, else ``False``.
    """
    sequence_type = sequence_value.tensor_type
    slice_type = slice_value.tensor_type
    if sequence_type.dtype is not None and slice_type.dtype is not None and sequence_type.dtype != slice_type.dtype:
        return False
    sequence_shape = sequence_type.shape
    if sequence_shape is None:
        return True
    if len(sequence_shape) == 0:
        return False
    if slice_type.shape is None:
        return True
    if len(sequence_shape) - 1 != len(slice_type.shape):
        return False
    for seq_dim, slice_dim in zip(sequence_shape[1:], slice_type.shape, strict=True):
        if seq_dim is None or slice_dim is None:
            continue
        if isinstance(seq_dim, str) or isinstance(slice_dim, str):
            continue
        if seq_dim != slice_dim:
            return False
    return True


def _infer_scan_state_count(node: Node, body_graph: Graph, num_scan_inputs: int) -> int:
    """Infer the Scan state-family arity from normalized node/body metadata.

    Args:
        node: Normalized parent Scan node.
        body_graph: Normalized Scan body subgraph.
        num_scan_inputs: Number of Scan sequence-input families.

    Returns:
        The inferred number of Scan state families.

    Raises:
        ValueError: If the node/body contract is ambiguous or invalid.
    """
    if len(body_graph.inputs) != len(node.inputs) or len(body_graph.outputs) != len(node.outputs):
        msg = "Scan: body input/output arity mismatch before emission"
        raise ValueError(msg)

    candidate_state_counts: list[int] = []
    max_state_count = len(node.inputs) - num_scan_inputs
    for state_count in range(max_state_count + 1):
        if state_count > len(node.outputs):
            continue
        scan_output_count = len(node.outputs) - state_count
        matches = True
        for slot in range(state_count):
            parent_state_in = node.inputs[slot]
            body_state_in = body_graph.inputs[slot]
            body_state_out = body_graph.outputs[slot]
            parent_state_out = node.outputs[slot]
            known_dtypes = [
                dtype
                for dtype in (
                    parent_state_in.tensor_type.dtype,
                    body_state_in.tensor_type.dtype,
                    body_state_out.tensor_type.dtype,
                    parent_state_out.tensor_type.dtype,
                )
                if dtype is not None
            ]
            if known_dtypes and any(dtype != known_dtypes[0] for dtype in known_dtypes[1:]):
                matches = False
                break
        if not matches:
            continue
        for slot in range(num_scan_inputs):
            if not _scan_sequence_matches_slice(node.inputs[state_count + slot], body_graph.inputs[state_count + slot]):
                matches = False
                break
        if not matches:
            continue
        for slot in range(scan_output_count):
            if not _scan_sequence_matches_slice(
                node.outputs[state_count + slot], body_graph.outputs[state_count + slot]
            ):
                matches = False
                break
        if matches:
            candidate_state_counts.append(state_count)

    if len(candidate_state_counts) != 1:
        msg = "Scan: ambiguous or invalid state/scan family split before emission"
        raise ValueError(msg)
    return candidate_state_counts[0]


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
    if initial_condition is None:
        msg = "Loop: missing cond input"
        raise ValueError(msg)
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
    cond0 = initial_condition
    loop_state = (iteration0, cond0, *carried_inputs)
    final_state = fx_graph.call_function(_call_torch_while_loop, args=(cond_fn, body_fn, loop_state))
    return [fx_graph.call_function(operator.getitem, args=(final_state, slot + 2)) for slot in range(carried_count)]


@register_op("Scan", opset_range=(11, 21))
def _scan(
    node: Node,
    args: list[torch.fx.Node | None],
    fx_graph: torch.fx.Graph,
    module: torch.nn.Module,
) -> list[torch.fx.Node]:
    """Emit ``torch.while_loop`` for the ONNX ``Scan`` op.

    Args:
        node: The IR Scan node.
        args: Input FX nodes in normalized order ``(state..., scan_inputs..., captures...)``.
        fx_graph: The FX graph being constructed.
        module: The root module carrying the internal child-graph helper.

    Returns:
        A list of FX nodes ordered as final state outputs then scanned outputs.

    Raises:
        ValueError: If normalized inputs or child-graph metadata are invalid.
    """
    import torch

    helper = getattr(module, "_protofx_child_graph_emitter", None)
    if helper is None:
        msg = "Scan: internal child graph emitter helper is unavailable"
        raise ValueError(msg)

    body_graph = node.subgraphs.get("body")
    if not isinstance(body_graph, Graph):
        msg = "Scan: missing body child graph"
        raise ValueError(msg)

    num_scan_inputs_attr = node.attributes.get("num_scan_inputs", 1)
    if not isinstance(num_scan_inputs_attr, int):
        msg = "Scan: num_scan_inputs must be an int"
        raise ValueError(msg)
    num_scan_inputs = int(num_scan_inputs_attr)
    if num_scan_inputs < 1:
        msg = "Scan: num_scan_inputs must be >= 1"
        raise ValueError(msg)

    state_count = _infer_scan_state_count(node, body_graph, num_scan_inputs)
    scan_output_count = len(node.outputs) - state_count

    expected_min_inputs = state_count + num_scan_inputs
    if len(args) < expected_min_inputs:
        msg = f"Scan: expected at least {expected_min_inputs} inputs, got {len(args)}"
        raise ValueError(msg)

    state_inputs = args[:state_count]
    scan_inputs = args[state_count : state_count + num_scan_inputs]
    captures = args[state_count + num_scan_inputs :]
    if any(state is None for state in state_inputs):
        msg = "Scan: missing state input"
        raise ValueError(msg)
    if any(scan_input is None for scan_input in scan_inputs):
        msg = "Scan: missing scan input"
        raise ValueError(msg)
    if any(capture is None for capture in captures):
        msg = "Scan: explicit captures cannot be omitted"
        raise ValueError(msg)

    body_attr, body_arity = helper.make_callable_attr(owner_node=node, branch_name="body", child_graph=body_graph)
    expected_body_arity = state_count + scan_output_count
    if body_arity != expected_body_arity:
        msg = f"Scan: body output arity mismatch before emission (body={body_arity}, expected={expected_body_arity})"
        raise ValueError(msg)

    body_callable = fx_graph.get_attr(body_attr)
    trip_count = fx_graph.call_function(_scan_trip_count, args=(scan_inputs[0],))
    cond_fn = fx_graph.call_function(_make_scan_cond_fn, args=(trip_count,))
    body_fn = fx_graph.call_function(
        _make_scan_body_fn,
        args=(body_callable, tuple(scan_inputs), tuple(captures), state_count, scan_output_count),
    )
    iteration0 = fx_graph.call_function(torch.tensor, args=(0,), kwargs={"dtype": torch.int64})
    scan_accums: list[torch.fx.Node] = []
    for scan_slot in range(scan_output_count):
        per_step_shape, scan_dtype = _scan_output_accumulator_spec(node.outputs[state_count + scan_slot])
        scan_accums.append(
            fx_graph.call_function(_make_scan_initial_accumulator, args=(trip_count, per_step_shape, scan_dtype))
        )
    loop_state = (iteration0, *state_inputs, *scan_accums)
    final_state = fx_graph.call_function(_call_torch_while_loop, args=(cond_fn, body_fn, loop_state))
    state_outputs = [
        fx_graph.call_function(operator.getitem, args=(final_state, 1 + state_slot))
        for state_slot in range(state_count)
    ]
    scan_outputs = [
        fx_graph.call_function(operator.getitem, args=(final_state, 1 + state_count + scan_slot))
        for scan_slot in range(scan_output_count)
    ]
    return [*state_outputs, *scan_outputs]
