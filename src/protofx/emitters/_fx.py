"""Internal IR-to-FX emission logic.

This module implements the pipeline from ``ir.Graph`` to
``torch.fx.GraphModule``. All FX-aware lowering happens here so the
importer and IR core never depend on ``torch.fx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.fx

from protofx.ir.graph import Graph
from protofx.ir.value import Value, ValueKind
from protofx.ops import dispatch_op


def _sanitize_name(name: str | None, fallback: str) -> str:
    """Return a valid Python identifier for use as an FX node or attribute name.

    Args:
        name: Original name (may contain dots, slashes, etc.).
        fallback: Fallback name when *name* is ``None`` or empty.

    Returns:
        A cleaned identifier safe for ``torch.fx`` attribute names.
    """
    raw = name or fallback
    cleaned = raw.replace(".", "_").replace("/", "_").replace("::", "_").replace("-", "_")
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def _emit_data_value(
    value: Value,
    root: torch.nn.Module,
    fx_graph: torch.fx.Graph,
    value_map: dict[str, torch.fx.Node],
    attr_counter: dict[str, int],
) -> None:
    """Emit a CONSTANT or INITIALIZER value as a buffer + get_attr node.

    Args:
        value: An IR value with ``data`` payload.
        root: The root ``torch.nn.Module`` for buffer registration.
        fx_graph: The ``torch.fx.Graph`` being constructed.
        value_map: Mutable map from IR value id to FX node.
        attr_counter: Mutable counter dict to avoid attribute name collisions.
    """
    import torch

    base = _sanitize_name(value.name, value.id)
    count = attr_counter.get(base, 0)
    attr_name = base if count == 0 else f"{base}_{count}"
    attr_counter[base] = count + 1

    tensor = torch.from_numpy(value.data.copy())
    root.register_buffer(attr_name, tensor)
    fx_node = fx_graph.get_attr(attr_name)
    value_map[value.id] = fx_node


def emit_graph(graph: Graph) -> torch.fx.GraphModule:
    """Convert a normalized ``ir.Graph`` into a ``torch.fx.GraphModule``.

    This is the main emitter entry point. It walks the IR graph in
    topological order, emits FX nodes for each IR value and operation,
    and returns a fully constructed ``GraphModule``.

    Args:
        graph: A validated, normalized IR graph produced by the importer.

    Returns:
        A ``torch.fx.GraphModule`` equivalent to the IR graph.
    """
    import torch
    import torch.fx

    fx_graph = torch.fx.Graph()
    root = torch.nn.Module()
    value_map: dict[str, torch.fx.Node] = {}
    attr_counter: dict[str, int] = {}

    # Phase 1: emit placeholders for graph inputs
    for inp_value in graph.inputs:
        name = _sanitize_name(inp_value.name, inp_value.id)
        fx_node = fx_graph.placeholder(name)
        value_map[inp_value.id] = fx_node

    # Phase 2: emit get_attr nodes for initializers
    for init_value in graph.initializers:
        _emit_data_value(init_value, root, fx_graph, value_map, attr_counter)

    # Phase 3: walk nodes in topological order and dispatch op handlers
    for ir_node in graph.topological_sort():
        # Resolve FX args from IR inputs — SENTINEL → None, CONSTANT emitted lazily
        args: list[torch.fx.Node | None] = []
        for inp_value in ir_node.inputs:
            if inp_value.kind == ValueKind.SENTINEL:
                args.append(None)
            elif inp_value.id in value_map:
                args.append(value_map[inp_value.id])
            elif inp_value.kind in (ValueKind.CONSTANT, ValueKind.INITIALIZER) and inp_value.data is not None:
                # Lazily emit constant/initializer data values not yet in value_map
                _emit_data_value(inp_value, root, fx_graph, value_map, attr_counter)
                args.append(value_map[inp_value.id])
            else:
                msg = f"unresolved input value {inp_value.id!r} (kind={inp_value.kind.name})"
                raise ValueError(msg)

        handler = dispatch_op(ir_node.op_type, ir_node.opset_version)
        fx_outputs = handler(ir_node, args, fx_graph, root)

        # Map IR outputs to FX outputs by slot position
        ir_outputs = ir_node.outputs
        if len(fx_outputs) != len(ir_outputs):
            msg = f"op handler for {ir_node.op_type!r} returned {len(fx_outputs)} outputs, expected {len(ir_outputs)}"
            raise ValueError(msg)
        for ir_out, fx_out in zip(ir_outputs, fx_outputs, strict=True):
            value_map[ir_out.id] = fx_out

    # Phase 4: emit output node
    output_refs: list[torch.fx.Node | None] = []
    for out_value in graph.outputs:
        if out_value.id in value_map:
            output_refs.append(value_map[out_value.id])
        elif out_value.kind in (ValueKind.CONSTANT, ValueKind.INITIALIZER) and out_value.data is not None:
            _emit_data_value(out_value, root, fx_graph, value_map, attr_counter)
            output_refs.append(value_map[out_value.id])
        else:
            msg = f"unresolved output value {out_value.id!r}"
            raise ValueError(msg)

    fx_graph.output(tuple(output_refs))

    return torch.fx.GraphModule(root, fx_graph)
